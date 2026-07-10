#!/usr/bin/env python3
"""Fetch the first N ``agent_*.zip`` of a run into its local sweep dir, cheaply.

Two sources, tried in order:

1. HF ``results/<run>/agents.tar`` -- streamed with ``tarfile`` in ``r|`` mode and
   the connection dropped once N zips are out. The tars store members in agent
   order, so this reads ~1 MB instead of the tar's 120-300 MB.
2. torch ``/scratch/se2161/picbreeder-vlm/sweep_logs/sweep/<run>/agents/`` -- a
   targeted rsync of just the N filenames (needs the ControlMaster socket up).

Also records the run's *true* agent count (HF ``data_manifest.json`` ``n_agents``,
else a remote ``ls | wc -l``) into ``agents/.total_agents`` so the transcript
builder can label "12 of 1000" rather than "12 of 12".

    python -m tools.fetch_agent_zips --runs runs.txt --n 12
"""
from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path

import requests
from huggingface_hub import get_token, hf_hub_download, hf_hub_url

REPO = Path(__file__).resolve().parent.parent
SWEEP = REPO / "sweep_logs" / "sweep"
DATASET = "smearle/picbreeder-vlm-archive"
TORCH_SWEEP = "/scratch/se2161/picbreeder-vlm/sweep_logs/sweep"


def _auth() -> dict:
    tok = get_token()
    return {"Authorization": f"Bearer {tok}"} if tok else {}


def total_from_manifest(run: str) -> int | None:
    try:
        p = hf_hub_download(DATASET, f"results/{run}/data_manifest.json", repo_type="dataset")
    except Exception:
        return None
    n = json.loads(Path(p).read_text()).get("n_agents")
    return int(n) if n else None


def total_from_torch(run: str) -> int | None:
    cmd = f"ls {TORCH_SWEEP}/{run}/agents/agent_*.zip 2>/dev/null | wc -l"
    out = subprocess.run(["ssh", "-o", "BatchMode=yes", "torch", cmd],
                         capture_output=True, text=True, timeout=120)
    n = int(out.stdout.strip() or 0)
    return n or None


def from_hf_tar(run: str, dest: Path, n: int) -> int:
    """Stream results/<run>/agents.tar, write the first n agent zips, abort early."""
    url = hf_hub_url(DATASET, f"results/{run}/agents.tar", repo_type="dataset")
    r = requests.get(url, headers=_auth(), stream=True, timeout=60)
    r.raise_for_status()
    got = 0
    try:
        with tarfile.open(fileobj=r.raw, mode="r|") as tf:
            for m in tf:
                name = Path(m.name).name
                if not (m.isfile() and name.startswith("agent_") and name.endswith(".zip")):
                    continue
                fh = tf.extractfile(m)
                if fh is None:
                    continue
                (dest / name).write_bytes(fh.read())
                got += 1
                if got >= n:
                    break
    finally:
        r.close()   # drop the rest of the tar on the floor
    return got


def from_torch(run: str, dest: Path, n: int) -> int:
    names = [f"agent_{i:03d}.zip" for i in range(n)]
    src = f"torch:{TORCH_SWEEP}/{run}/agents/"
    cmd = ["rsync", "-e", "ssh -o BatchMode=yes", "--ignore-missing-args",
           *[src + nm for nm in names], str(dest) + "/"]
    subprocess.run(cmd, check=True, timeout=600, capture_output=True)
    return len(list(dest.glob("agent_*.zip")))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, required=True, help="file with one run name per line")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--source", choices=("hf", "torch"), required=True)
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    runs = [r for r in args.runs.read_text().split() if r]
    for i, run in enumerate(runs, 1):
        dest = SWEEP / run / "agents"
        dest.mkdir(parents=True, exist_ok=True)
        have = len(list(dest.glob("agent_*.zip")))
        if args.skip_existing and have >= args.n:
            print(f"[{i}/{len(runs)}] {run}: have {have}, skip", flush=True)
            continue
        try:
            got = from_hf_tar(run, dest, args.n) if args.source == "hf" else from_torch(run, dest, args.n)
        except Exception as e:                                  # noqa: BLE001
            print(f"[{i}/{len(runs)}] {run}: FAIL {type(e).__name__}: {e}", flush=True)
            continue
        total = total_from_manifest(run) if args.source == "hf" else total_from_torch(run)
        if total:
            (dest / ".total_agents").write_text(str(total))
        print(f"[{i}/{len(runs)}] {run}: {got} zips, total={total}", flush=True)


if __name__ == "__main__":
    main()
