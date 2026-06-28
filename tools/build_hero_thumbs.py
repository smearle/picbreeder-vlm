#!/usr/bin/env python3
"""
Extract the final (published) frame from each hero-grid sprite sheet as a tiny
per-lineage thumbnail, then pack all of them into ONE atlas image, so the banner
paints from a single ~80 KB request while the full sprite sheets (used for the
rewind morph) load in the background.

Reads:  <deploy>/assets/hero_sprites/manifest.json  (+ the sprite sheets)
Writes: <id>_thumb.jpg next to each sprite (per-cell fallback),
        rewrites the manifest in place with a `thumb` field per entry,
        thumbs_atlas.jpg  (all final frames in a grid, cell i = manifest index i),
        thumbs_atlas.json (atlas metadata).

ATLAS_COLS / ATLAS_CELL below MUST match the loader constants in index.html.

The deploy dir is hash-suffixed (e.g. picbreeder-vlm-06b0d76d), so we locate it
rather than hard-coding it; pass an explicit path as argv[1] to override.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image

ATLAS_COLS = 8
ATLAS_CELL = 88


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


def main() -> int:
    root = find_root()
    sprites = root / "assets" / "hero_sprites"
    manifest = sprites / "manifest.json"
    print(f"deploy: {root}")

    entries = json.loads(manifest.read_text())
    thumbs: list[Image.Image] = []      # in manifest order, for the atlas
    for e in entries:
        sprite = root / e["sprite"]
        if not sprite.is_file():
            print(f"  MISS {sprite}")
            thumbs.append(Image.new("RGB", (ATLAS_CELL, ATLAS_CELL), (242, 242, 242)))
            continue

        n, cols, fw = e["n"], e["cols"], e["fw"]
        last = n - 1
        c, r = last % cols, last // cols
        box = (c * fw, r * fw, (c + 1) * fw, (r + 1) * fw)

        img = Image.open(sprite); img.load()
        if img.mode != "RGB":
            img = img.convert("RGB")
        thumb = img.crop(box)

        rel = f"assets/hero_sprites/{e['id']}_thumb.jpg"
        out = root / rel
        thumb.save(out, "JPEG", quality=88, optimize=True, progressive=True)
        e["thumb"] = rel
        thumbs.append(thumb.resize((ATLAS_CELL, ATLAS_CELL)))

    manifest.write_text(json.dumps(entries, indent=2) + "\n")
    print(f"Updated {manifest} with thumb fields for {len(entries)} entries.")

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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
