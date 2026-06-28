#!/usr/bin/env python3
"""Add lineage-derived orderings to already-built sprite layout.json files in place.

The sprite sheets and the chronological/siglip layouts are unchanged; we only
compute the new orderings from each run's archive_metadata.json and merge them
into site/<run>/sprite/layout.json. This avoids re-running UMAP / re-packing
sheets just to add a column ordering. Re-push with tools/push_sprites.py after.

  python tools/add_lineage_layouts.py                 # all runs, grid orderings
  python tools/add_lineage_layouts.py --with-phylogeny # also the scatter layout
  python tools/add_lineage_layouts.py default_s4       # one run dir
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

REPO = Path("/home/jupyter-smearle/picbreeder-vlm")
SWEEP = REPO / "sweep_logs" / "sweep"
SITE = REPO / "archive_animations" / "_archive_mirror" / "site"

# import the generators from build_archive_image_lib without running its main()
_spec = importlib.util.spec_from_file_location(
    "balib", REPO / "tools" / "build_archive_image_lib.py")
_balib = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_balib)


def update_run(run_dir: Path, with_phylogeny: bool) -> str:
    layout_path = run_dir / "sprite" / "layout.json"
    if not layout_path.is_file():
        return f"[skip] {run_dir.name}: no sprite/layout.json"
    meta = SWEEP / run_dir.name / "archive" / "archive_metadata.json"
    if not meta.is_file():
        return f"[skip] {run_dir.name}: no archive_metadata.json (e.g. human)"

    layout = json.loads(layout_path.read_text())
    n = int(layout["n"])
    cols = int(layout["layouts"]["chronological"]["dims"][1])
    entries = _balib._aligned_entries(meta, n)

    layout["layouts"]["branched"] = _balib.branched_layout(entries, n, cols)
    layout["layouts"]["ratings"] = _balib.ratings_layout(entries, n, cols)
    extra = "branched,ratings"
    if with_phylogeny:
        layout["layouts"]["phylogeny"] = _balib.phylogeny_layout(entries, n)
        extra += ",phylogeny"

    layout_path.write_text(json.dumps(layout) + "\n")
    kb = layout_path.stat().st_size // 1024
    return f"[ok]   {run_dir.name}: +{extra}  (layout.json {kb} KB, n={n})"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*", help="run dir names (default: all in site/)")
    ap.add_argument("--with-phylogeny", action="store_true",
                    help="also emit the kind:scatter phylogeny layout")
    args = ap.parse_args()

    if args.runs:
        dirs = [SITE / r for r in args.runs]
    else:
        dirs = sorted(d for d in SITE.iterdir() if (d / "sprite").is_dir())
    for d in dirs:
        print(update_run(d, args.with_phylogeny))
    return 0


if __name__ == "__main__":
    sys.exit(main())
