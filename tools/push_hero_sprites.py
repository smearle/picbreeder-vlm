#!/usr/bin/env python
"""Push the heavy hero-banner sprite SHEETS (img_*.jpg, the per-lineage rewind
morphs, ~45 MB total at 128px) to the HF dataset under hero_sprites/, so the blog
git repo doesn't carry them. The banner's above-the-fold assets stay local and
tiny: thumbs_atlas.jpg (initial paint), manifest.json, and the per-cell
img_*_thumb.jpg fallbacks. hero-grid.js loads the sheets from HF (see heroBase).

  python tools/push_hero_sprites.py            # upload the 43 sheets
  python tools/push_hero_sprites.py --dry-run  # list what would upload

Idempotent at the commit level (re-uploading identical files is a no-op).
"""
import json
import sys
from pathlib import Path

from huggingface_hub import HfApi

REPO = "picbreeder-vlm/picbreeder-vlm-archive"
HERO = Path("/home/jupyter-smearle/smearle.github.io/picbreeder-vlm-06b0d76d/assets/hero_sprites")


def sheets():
    """Only the sheets referenced by the manifest (skip orphaned archive-id sheets
    left over from older builds)."""
    man = json.loads((HERO / "manifest.json").read_text())
    names = [Path(e["sprite"]).name for e in man]
    return [HERO / n for n in names]


def main():
    files = sheets()
    total_mb = sum(p.stat().st_size for p in files) / 1e6
    print(f"{len(files)} hero sprite sheets, {total_mb:.1f} MB -> {REPO}:hero_sprites/")
    if "--dry-run" in sys.argv:
        for p in files:
            print("   ", p.name)
        return
    HfApi().upload_folder(
        repo_id=REPO, repo_type="dataset",
        folder_path=str(HERO), path_in_repo="hero_sprites",
        allow_patterns=[p.name for p in files],
        commit_message=f"Add hero-banner sprite sheets ({len(files)} lineages, 128px)",
    )
    print(f"[hero] uploaded {len(files)} sheets -> hero_sprites/")


if __name__ == "__main__":
    main()
