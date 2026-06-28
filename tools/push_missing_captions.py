#!/usr/bin/env python
"""Upload per-run VLM caption files (archive/captions_*.json) to the HF archive so
the /breed site can display them.

Captions are generated post-hoc by a fixed captioner (gemini-2.5-pro for every run,
plus a qwen3-vl-8b file for a handful) and historically were only pushed for some
runs. The breed site (pbsite.js loadCaptions) fetches results/<run>/captions_<model>.json
and merges captions into run items, so every run whose metadata is on HF needs its
caption file there too.

This pushes any LOCAL captions_*.json whose run already has archive_metadata.json on
HF but is missing that caption file. Idempotent: already-present files are skipped.
"""
import argparse, glob, os
from pathlib import Path
from huggingface_hub import HfApi

REPO = "picbreeder-vlm/picbreeder-vlm-archive"
SWEEP = Path("/home/jupyter-smearle/picbreeder-vlm/sweep_logs/sweep")


def local_caption_files():
    """run_name -> [local caption file paths]."""
    out = {}
    for m in glob.glob(str(SWEEP / "*" / "archive" / "archive_metadata.json")):
        d = os.path.dirname(m)                       # .../<run>/archive
        run = os.path.basename(os.path.dirname(d))   # <run>
        caps = sorted(glob.glob(os.path.join(d, "captions_*.json")))
        if caps:
            out[run] = caps
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="list what would be pushed, upload nothing")
    args = ap.parse_args()

    api = HfApi()
    hf = set(api.list_repo_files(REPO, repo_type="dataset"))
    hf_meta_runs = {f.split("/")[1] for f in hf if f.endswith("/archive_metadata.json")}

    local = local_caption_files()
    to_push = []   # (local_path, repo_path)
    skip_no_meta = []
    for run, caps in sorted(local.items()):
        if run not in hf_meta_runs:
            skip_no_meta.append(run)          # run not exposed on HF; nothing to view there
            continue
        for c in caps:
            repo_path = f"results/{run}/{os.path.basename(c)}"
            if repo_path not in hf:
                to_push.append((c, repo_path))

    print(f"local runs with captions: {len(local)}")
    print(f"  (skipped, metadata not on HF: {len(skip_no_meta)})")
    print(f"caption files to upload: {len(to_push)}")
    for _, rp in to_push:
        print("   +", rp)
    if args.dry_run or not to_push:
        return

    for lp, rp in to_push:
        api.upload_file(path_or_fileobj=lp, path_in_repo=rp, repo_id=REPO,
                        repo_type="dataset",
                        commit_message=f"Add caption file {rp} for /breed display")
        print("uploaded", rp)
    print(f"\nDone: uploaded {len(to_push)} caption files.")


if __name__ == "__main__":
    main()
