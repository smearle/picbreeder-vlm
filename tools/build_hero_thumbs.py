#!/usr/bin/env python3
"""
Build the hero-banner thumbnails: one tiny per-lineage image plus a single atlas,
so the banner paints from one ~150 KB request while the per-cell CPPN genomes
(used for the live rewind morph) stream in behind it.

Each thumbnail is a lineage's TIP -- the published archive image the cell rests on.
We read those pixels straight from the original archive PNG, located via
archive_animations/teaser_provenance.json (hero id -> sweep archive PNG).

Reads:  <deploy>/assets/hero_sprites/manifest.json   (cell order + ids)
        <repo>/archive_animations/teaser_provenance.json
Writes: <id>_thumb.jpg per cell (fallback used when the atlas 404s),
        rewrites the manifest in place with a `thumb` field per entry,
        thumbs_atlas.jpg  (all tips in a grid, cell i = manifest index i),
        thumbs_atlas.json (atlas metadata).

ATLAS_COLS / ATLAS_CELL below MUST match the loader constants in assets/page/hero-grid.js.

Manifest ORDER is the source of truth for atlas slots and is a committed artifact.
This script never reorders or adds entries; it only refreshes pixels and `thumb`.

This replaces an earlier version that cropped the last frame out of each sprite
sheet. The sheets were retired when the hero moved to live GPU morphs; the archive
PNGs are the same images, without the extra JPEG generation.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image

ATLAS_COLS = 8
ATLAS_CELL = 128

REPO = Path(__file__).resolve().parents[1]
TEASER = REPO / "archive_animations" / "teaser_provenance.json"


def find_root() -> Path:
    if len(sys.argv) > 1:
        return Path(sys.argv[1])
    base = Path("/home/jupyter-smearle/smearle.github.io")
    # prefer an exact dir, else the newest hash-suffixed picbreeder-vlm* deploy
    candidates = sorted(base.glob("picbreeder-vlm*"), key=lambda p: p.stat().st_mtime, reverse=True)
    for c in candidates:
        if (c / "assets" / "hero_sprites" / "manifest.json").is_file():
            return c
    raise SystemExit("no picbreeder-vlm deploy dir with hero_sprites/manifest.json found")


def tip_image(teaser: dict, hero_id: str) -> Image.Image | None:
    """A lineage's published tip, as a 128px RGB square."""
    src = teaser.get(hero_id + ".png")
    if not src:
        return None
    png = REPO / src
    if not png.is_file():
        return None
    img = Image.open(png)
    img.load()
    if img.mode != "RGB":
        img = img.convert("RGB")          # some lineages are greyscale ("L")
    if img.size != (ATLAS_CELL, ATLAS_CELL):
        img = img.resize((ATLAS_CELL, ATLAS_CELL), Image.LANCZOS)
    return img


def main() -> int:
    root = find_root()
    sprites = root / "assets" / "hero_sprites"
    manifest = sprites / "manifest.json"
    print(f"deploy: {root}")

    teaser = json.loads(TEASER.read_text())
    entries = json.loads(manifest.read_text())
    thumbs: list[Image.Image] = []      # in manifest order, for the atlas
    misses = 0
    for e in entries:
        thumb = tip_image(teaser, e["id"])
        if thumb is None:
            print(f"  MISS tip PNG for {e['id']}")
            misses += 1
            thumbs.append(Image.new("RGB", (ATLAS_CELL, ATLAS_CELL), (242, 242, 242)))
            continue

        rel = f"assets/hero_sprites/{e['id']}_thumb.jpg"
        thumb.save(root / rel, "JPEG", quality=88, optimize=True, progressive=True)
        e["thumb"] = rel
        thumbs.append(thumb)

    manifest.write_text(json.dumps(entries, indent=1) + "\n")
    print(f"Updated {manifest} with thumb fields for {len(entries)} entries ({misses} misses).")

    # Pack the atlas (cell i = manifest index i; the loader relies on this order).
    rows = (len(thumbs) + ATLAS_COLS - 1) // ATLAS_COLS
    atlas = Image.new("RGB", (ATLAS_COLS * ATLAS_CELL, rows * ATLAS_CELL), (242, 242, 242))
    for i, t in enumerate(thumbs):
        atlas.paste(t, ((i % ATLAS_COLS) * ATLAS_CELL, (i // ATLAS_COLS) * ATLAS_CELL))
    atlas_path = sprites / "thumbs_atlas.jpg"
    atlas.save(atlas_path, "JPEG", quality=86, optimize=True, progressive=True)
    (sprites / "thumbs_atlas.json").write_text(json.dumps({
        "file": "assets/hero_sprites/thumbs_atlas.jpg",
        "cell": ATLAS_CELL, "cols": ATLAS_COLS, "count": len(thumbs),
        "ids": [e["id"] for e in entries],
    }) + "\n")
    print(f"Wrote {atlas_path.name} ({atlas_path.stat().st_size//1024} KB, "
          f"{ATLAS_COLS}x{rows}) for {len(thumbs)} thumbs.")
    return 1 if misses else 0


if __name__ == "__main__":
    raise SystemExit(main())
