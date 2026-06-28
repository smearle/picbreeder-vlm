#!/usr/bin/env python3
"""Batch-build transcript bundles for every local sweep run with agent data, into a
staging dir, capped per run. Resumable (--skip-existing per agent). One subprocess
per run (fresh process keeps memory flat). Push with tools/push_transcripts.py.

    .venv/bin/python tools/build_all_transcripts.py --stage _transcripts_stage \
        --cap 120 --jobs 16
"""
from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))
from hf_archive_push import parse_config, canonical_arc  # noqa: E402


def target_runs(min_agents: int):
    runs = []
    for d in sorted(glob.glob(str(REPO / "sweep_logs" / "sweep" / "*") + "/")):
        d = Path(d)
        if not (d / "agents").is_dir() or not (d / "archive" / "images").is_dir():
            continue
        n = len(list((d / "agents").glob("agent_*.zip")))
        if n < min_agents:
            continue
        cfg = parse_config(d.name)
        runs.append((canonical_arc(d.name, cfg), cfg.get("model"), cfg.get("seed"), n, d))
    runs.sort(key=lambda r: (str(r[0]), str(r[1]), str(r[2])))
    return runs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=Path, default=REPO / "_transcripts_stage")
    ap.add_argument("--cap", type=int, default=120, help="max agents per run")
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--min-agents", type=int, default=20)
    ap.add_argument("--demo-run",
                    default="th1_ag20_model-gemini-2.5-pro_tb-1_scheme-toggle_nopersonalities_fixed-sesh_s3",
                    help="skip this run (kept local/committed under the blog)")
    args = ap.parse_args()

    runs = target_runs(args.min_agents)
    runs = [r for r in runs if r[4].name != args.demo_run]
    args.stage.mkdir(parents=True, exist_ok=True)
    marker = args.stage / ".current"   # run currently being written (don't push it)
    done_flag = args.stage / ".done"
    if done_flag.exists():
        done_flag.unlink()
    print(f"{len(runs)} runs to build (cap {args.cap}, jobs {args.jobs}) -> {args.stage}\n", flush=True)

    for i, (arc, model, seed, n, d) in enumerate(runs, 1):
        out = args.stage / d.name
        marker.write_text(d.name)
        print(f"=== [{i}/{len(runs)}] {d.name}  {arc} {model} s{seed}  "
              f"({n} agents, build {min(args.cap,n)}) ===", flush=True)
        cmd = [sys.executable, str(REPO / "tools" / "build_transcript_data.py"),
               "--run", str(d), "--agent", "all", "--max-agents", str(args.cap),
               "--jobs", str(args.jobs), "--skip-existing", "--out", str(out)]
        r = subprocess.run(cmd)
        if r.returncode != 0:
            print(f"!!! {d.name} exited {r.returncode}", flush=True)
    if marker.exists():
        marker.unlink()
    done_flag.write_text("ok")
    print("\nBATCH DONE", flush=True)


if __name__ == "__main__":
    main()
