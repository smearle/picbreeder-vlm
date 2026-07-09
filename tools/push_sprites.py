#!/usr/bin/env python
"""Push the per-run sprite sets (layout.json + sprites.json + sheets/) staged in the local
archive mirror to the HF dataset, under site/<run>/sprite/ (web assets the gallery fetches).

One commit for all sprite dirs. The top-level runs index is built separately by
tools/build_hf_index.py. Idempotent at the commit level (re-uploading identical files is a no-op).
"""
import json
import sys
from pathlib import Path
from huggingface_hub import HfApi

REPO = "picbreeder-vlm/picbreeder-vlm-archive"
SITE = Path("/home/jupyter-smearle/picbreeder-vlm/archive_animations/_archive_mirror/site")


def main():
    idx_local = json.load(open(SITE.parent / "index.json"))
    runs = idx_local["runs"]
    api = HfApi()
    allow = [f"{r['run']}/sprite/**" for r in runs]
    if "--dry-run" in sys.argv:
        n = sum(1 for r in runs for _ in (SITE / r["run"] / "sprite").rglob("*") if _.is_file())
        print(f"would upload {n} files across {len(runs)} sprite sets -> site/<run>/sprite/")
        return
    api.upload_folder(repo_id=REPO, repo_type="dataset", folder_path=str(SITE),
                      path_in_repo="site", allow_patterns=allow,
                      commit_message=f"Add sprite sets for {len(runs)} table runs")
    print(f"[sprites] uploaded {len(runs)} run sprite sets -> site/")
    print("[index] run `python tools/build_hf_index.py` to refresh runs/index.json")


if __name__ == "__main__":
    main()
