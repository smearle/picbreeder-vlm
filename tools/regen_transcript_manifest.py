#!/usr/bin/env python3
"""Rebuild the blog's transcripts ``manifest.json`` from *every* run on HF.

``push_transcripts.py --stage X`` regenerates the manifest from X's runs only, so
pushing a partial stage silently drops the runs built in earlier batches. This
tool instead enumerates ``transcripts/<run>/index.json`` on the dataset, so the
manifest always reflects the full set. The demo run stays local (no ``hf`` flag).

    python -m tools.regen_transcript_manifest
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.push_transcripts import BLOG_TX, REPO, label_for, run_index  # noqa: E402

DEMO_RUN = "th1_ag20_model-gemini-2.5-pro_tb-1_scheme-toggle_nopersonalities_fixed-sesh_s3"


def hf_runs(repo: str) -> list[str]:
    files = HfApi().list_repo_files(repo, repo_type="dataset")
    return sorted({m.group(1) for m in
                   (re.match(r"transcripts/([^/]+)/index\.json$", f) for f in files) if m})


def fetch_index(repo: str, run: str) -> tuple[str, dict | None]:
    try:
        p = hf_hub_download(repo, f"transcripts/{run}/index.json", repo_type="dataset")
        return run, json.loads(Path(p).read_text())
    except Exception:                                          # noqa: BLE001
        return run, None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--demo-run", default=DEMO_RUN)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    runs: list[dict] = []
    didx = run_index(BLOG_TX / args.demo_run)
    if didx is not None:
        runs.append({"run": args.demo_run, "label": label_for(args.demo_run, didx) + " — demo",
                     "arc": didx.get("arc"), "model": didx.get("model"), "seed": didx.get("seed")})
    else:
        print(f"[warn] demo run not committed under {BLOG_TX}", file=sys.stderr)

    names = [r for r in hf_runs(args.repo) if r != args.demo_run]
    with ThreadPoolExecutor(max_workers=16) as ex:
        for run, idx in ex.map(lambda r: fetch_index(args.repo, r), names):
            if idx is None:
                print(f"[warn] no index.json for {run}", file=sys.stderr)
                continue
            runs.append({"run": run, "label": label_for(run, idx), "hf": True,
                         "arc": idx.get("arc"), "model": idx.get("model"), "seed": idx.get("seed")})

    print(f"{len(runs)} runs ({sum(1 for r in runs if r.get('hf'))} on HF)")
    if args.dry_run:
        return
    (BLOG_TX / "manifest.json").write_text(json.dumps({"runs": runs}, indent=1))
    print(f"wrote {BLOG_TX/'manifest.json'}")


if __name__ == "__main__":
    main()
