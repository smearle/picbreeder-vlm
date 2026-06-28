#!/usr/bin/env python3
"""Animate the growth of a Picbreeder-VLM archive over publication order.

Reads a run's ``archive/archive_metadata.json`` and the pre-rendered images in
``archive/images/{id}.png``, then renders an MP4 in which a grid fills in one
publication at a time (in ``added_at`` order). The most-recently-added cells are
briefly outlined so the eye can follow the growth front.

No NEAT / CPPN evaluation is needed here -- we composite the images the run
already rendered, which keeps this fast and dependency-light.

Usage:
    python archive_animations/archive_growth.py \
        --run sweep_logs/sweep/th0_ag20_model-gemini-2.5-pro_tb-1_scheme-toggle_nopersonalities_fixed-sesh_s4 \
        --out archive_animations/out/archive_growth.mp4 \
        --max-images 3427 --cell 22 --per-frame 16 --fps 24
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image


def load_entries(run: Path, max_images: int | None):
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
    return entries


def grid_shape(n: int, aspect: float = 16 / 9) -> tuple[int, int]:
    """cols x rows closest to ``aspect`` that holds n cells."""
    cols = max(1, round(math.sqrt(n * aspect)))
    rows = math.ceil(n / cols)
    return cols, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--cell", type=int, default=22, help="thumbnail px per side")
    ap.add_argument("--per-frame", type=int, default=16, help="new images per frame")
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--gap", type=int, default=1, help="px gap between cells")
    ap.add_argument("--hold", type=int, default=24, help="frames to hold on full grid")
    ap.add_argument("--bg", type=int, default=255)
    args = ap.parse_args()

    img_dir = args.run / "archive" / "images"
    entries = load_entries(args.run, args.max_images)
    # keep only entries whose image exists locally
    items = [(e["id"], img_dir / f"{e['id']}.png") for e in entries]
    items = [(i, p) for i, p in items if p.exists()]
    n = len(items)
    if n == 0:
        raise SystemExit(f"no images found under {img_dir}")

    cols, rows = grid_shape(n)
    cell, gap = args.cell, args.gap
    W = cols * cell + (cols + 1) * gap
    H = rows * cell + (rows + 1) * gap
    # ffmpeg/libx264 wants even dimensions
    W += W % 2
    H += H % 2
    print(f"{n} images -> {cols}x{rows} grid, frame {W}x{H}")

    def cell_xy(idx):
        r, c = divmod(idx, cols)
        return gap + c * (cell + gap), gap + r * (cell + gap)

    # preload + resize thumbnails (RGB)
    thumbs = []
    for _id, p in items:
        im = Image.open(p).convert("RGB").resize((cell, cell), Image.LANCZOS)
        thumbs.append(np.asarray(im))

    canvas = np.full((H, W, 3), args.bg, dtype=np.uint8)

    # ffmpeg pipe (raw rgb24 -> H.264, web-friendly)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
        "-r", str(args.fps), "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(args.out),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    HL = np.array([255, 108, 0], dtype=np.uint8)  # Sakana orange for the growth front

    placed = 0
    frame_idx = 0
    while placed < n:
        batch = range(placed, min(placed + args.per_frame, n))
        for idx in batch:
            x, y = cell_xy(idx)
            canvas[y:y + cell, x:x + cell] = thumbs[idx]
        # outline the cells added this frame
        frame = canvas.copy()
        for idx in batch:
            x, y = cell_xy(idx)
            frame[y - 1:y + cell + 1, x - 1:x + 1] = HL
            frame[y - 1:y + cell + 1, x + cell - 1:x + cell + 1] = HL
            frame[y - 1:y + 1, x - 1:x + cell + 1] = HL
            frame[y + cell - 1:y + cell + 1, x - 1:x + cell + 1] = HL
        proc.stdin.write(frame.tobytes())
        frame_idx += 1
        placed = batch.stop

    # hold on the finished, un-outlined grid
    for _ in range(args.hold):
        proc.stdin.write(canvas.tobytes())
        frame_idx += 1

    proc.stdin.close()
    proc.wait()
    print(f"wrote {args.out} ({frame_idx} frames, {frame_idx/args.fps:.1f}s)")


if __name__ == "__main__":
    main()
