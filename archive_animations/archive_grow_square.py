#!/usr/bin/env python3
"""Square archive-growth animation.

The archive is always shown as a filled square. Images are laid out in
publication order along concentric square rings from the centre outward, so each
step's newest images land *around the border* of the square. The viewport zooms
out smoothly (continuously) as the square grows, always framing the whole thing.

Per the brief: always a square; ~20 new children around the border each step;
smooth zoom-out. Edges are omitted (in a fully packed square they'd be hidden
behind thumbnails anyway); the newest border images are outlined in orange.

Usage:
    python archive_animations/archive_grow_square.py \
        --run sweep_logs/sweep/ag20_..._s3 \
        --out archive_animations/out/archive_square.mp4 \
        --frame 1000 --per-step 20 --cell-px 26 --fps 24
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ORANGE = (255, 108, 0)


def load_order(run: Path, max_images):
    meta = json.loads((run / "archive" / "archive_metadata.json").read_text())
    entries = meta["entries"] if isinstance(meta, dict) else meta

    def ts(e):
        try:
            return datetime.fromisoformat(e["added_at"])
        except Exception:
            return datetime.min

    entries = sorted(entries, key=ts)
    if max_images:
        entries = entries[:max_images]
    img_dir = run / "archive" / "images"
    return [e["id"] for e in entries if (img_dir / f"{e['id']}.png").exists()], img_dir


def square_spiral(n):
    """Concentric square-ring coordinates (centre first); each ring clockwise."""
    pts = [(0, 0)]
    r = 1
    while len(pts) < n:
        for x in range(-r, r + 1):        pts.append((x, -r))   # top edge L->R
        for y in range(-r + 1, r + 1):    pts.append((r, y))    # right edge T->B
        for x in range(r - 1, -r - 1, -1):pts.append((x, r))    # bottom edge R->L
        for y in range(r - 1, -r, -1):    pts.append((-r, y))   # left edge B->T
        r += 1
    return pts[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--frame", type=int, default=1000, help="output square size (px)")
    ap.add_argument("--per-step", type=int, default=20, help="new images per frame")
    ap.add_argument("--cell-px", type=int, default=26, help="thumbnail size on the world canvas")
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--margin-cells", type=float, default=1.5, help="zoom padding beyond the filled square")
    ap.add_argument("--min-half-cells", type=float, default=3.0, help="cap how far the first frames zoom in")
    ap.add_argument("--hold", type=int, default=48)
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--no-outline", action="store_true")
    args = ap.parse_args()

    order, img_dir = load_order(args.run, args.max_images)
    n = len(order)
    if n == 0:
        raise SystemExit("no images found")
    pts = square_spiral(n)
    cell = args.cell_px
    R = max(max(abs(x), abs(y)) for x, y in pts)            # outer ring radius
    half_world = (R + math.ceil(args.margin_cells) + 1) * cell
    world = 2 * half_world
    cx = cy = half_world
    print(f"{n} images -> square radius {R} rings, world {world}px")

    # world position (px, centre of thumbnail) for each node
    wpos = [(cx + x * cell, cy + y * cell) for (x, y) in pts]

    canvas = np.full((world, world, 3), 255, np.uint8)
    half = cell // 2

    def paste(i):
        im = Image.open(img_dir / f"{order[i]}.png").convert("RGB").resize((cell, cell), Image.LANCZOS)
        x, y = wpos[i]
        canvas[y - half:y - half + cell, x - half:x - half + cell] = np.asarray(im)

    F = args.frame
    args.out.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{F}x{F}", "-r", str(args.fps), "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(args.out)],
        stdin=subprocess.PIPE,
    )

    def emit(revealed, new_idx):
        # smooth zoom: half-side grows continuously with sqrt(count)
        half_cells = max(args.min_half_cells, math.sqrt(revealed) / 2 + args.margin_cells)
        hs = int(round(half_cells * cell))
        hs = min(hs, half_world)
        x0, y0 = cx - hs, cy - hs
        side = 2 * hs
        crop = canvas[y0:y0 + side, x0:x0 + side]
        f = cv2.resize(crop, (F, F), interpolation=cv2.INTER_AREA)
        if not args.no_outline and new_idx:
            sc = F / side
            r = max(2, int(half * 1.3 * sc))
            for i in new_idx:
                px, py = wpos[i]
                cv2.circle(f, (int((px - x0) * sc), int((py - y0) * sc)), r, ORANGE, 1, cv2.LINE_AA)
        proc.stdin.write(np.ascontiguousarray(f).tobytes())

    revealed = 0
    nframes = 0
    while revealed < n:
        batch = list(range(revealed, min(revealed + args.per_step, n)))
        for i in batch:
            paste(i)
        revealed += len(batch)
        emit(revealed, batch)
        nframes += 1

    for _ in range(args.hold):
        emit(revealed, [])
        nframes += 1
    proc.stdin.close()
    proc.wait()
    print(f"wrote {args.out} ({nframes} frames, {nframes/args.fps:.1f}s)")


if __name__ == "__main__":
    main()
