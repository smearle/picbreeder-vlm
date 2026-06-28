#!/usr/bin/env python3
"""Compact, zooming archive-growth animation.

Lays the published images on an integer lattice, placed in publication order:
each new image is dropped into the free cell *nearest its parent and furthest
from the current centroid*, so the cluster stays compact while keeping freshly
spawned leaves on the outer surface (where their own children can later appear).
Roots (random-init sessions) seed near the centre.

The result is fed to the shared reveal renderer with zoom on, so the view always
frames exactly the current archive as a compact square and zooms out as it grows.
Lineage edges are drawn behind the thumbnails.

Usage:
    python archive_animations/archive_grow_compact.py \
        --run sweep_logs/sweep/th0_..._s4 \
        --out archive_animations/out/archive_compact_s4.mp4 \
        --frame 1000 --per-frame 8 --fps 24
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

from anim_render import render_reveal, lineage_colors, VIRTUAL_ROOT


def load_forest(run: Path, max_images):
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
    ids = {e["id"] for e in entries}
    parent = {}
    for e in entries:
        src = e.get("source_entry_ids") or []
        parent[e["id"]] = src[0] if (src and src[0] in ids) else VIRTUAL_ROOT
    return [e["id"] for e in entries], parent


def _ring(c, radius):
    """Chebyshev ring of integer cells at given radius around c."""
    if radius == 0:
        yield c
        return
    ci, cj = c
    for di in range(-radius, radius + 1):
        for dj in range(-radius, radius + 1):
            if max(abs(di), abs(dj)) == radius:
                yield (ci + di, cj + dj)


def place_lattice(order, parent, mode="compact"):
    """Return {id: (i, j)} integer cells.

    Each node goes to the free cell nearest its seed (parent cell, or centroid for
    roots). Within the nearest non-empty ring we tie-break by distance to the
    centroid: ``compact`` pulls inward (dense, square-ish blob -> new nodes land on
    the growing perimeter once the interior fills); ``outward`` pushes to the rim.
    """
    cell = {}
    occ = {}
    sums = [0.0, 0.0]
    cnt = 0

    def centroid():
        return (sums[0] / cnt, sums[1] / cnt) if cnt else (0.0, 0.0)

    def commit(nid, c):
        nonlocal cnt
        occ[c] = nid
        cell[nid] = c
        sums[0] += c[0]
        sums[1] += c[1]
        cnt += 1

    def nearest_free(seed):
        cx, cy = centroid()
        for radius in range(0, 100000):
            free = [c for c in _ring(seed, radius) if c not in occ]
            if free:
                d2 = lambda c: (c[0] - cx) ** 2 + (c[1] - cy) ** 2
                return (min if mode == "compact" else max)(free, key=d2)
        raise RuntimeError("no free cell")  # unreachable

    for nid in order:
        par = parent.get(nid, VIRTUAL_ROOT)
        if par == VIRTUAL_ROOT or par not in cell:
            if cnt == 0:
                commit(nid, (0, 0))
            else:
                cx, cy = centroid()
                commit(nid, nearest_free((int(round(cx)), int(round(cy)))))
        else:
            commit(nid, nearest_free(cell[par]))
    return cell


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--frame", type=int, default=1000)
    ap.add_argument("--per-frame", type=int, default=8, help="new images revealed per step")
    ap.add_argument("--frames-per-step", type=int, default=2,
                    help="frames to dwell on each step (higher = slower, same images/step)")
    ap.add_argument("--plain-edges", action="store_true", help="grey edges instead of per-lineage colour")
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--thumb-px", type=int, default=26)
    ap.add_argument("--world", type=int, default=2400)
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--placement", choices=["compact", "outward"], default="compact")
    ap.add_argument("--node-size", choices=["uniform", "children"], default="uniform",
                    help="uniform thumbnails, or scale each by its #published children (hubs grow)")
    ap.add_argument("--size-min", type=float, default=0.62, help="min thumbnail scale (x thumb-px) when node-size=children")
    ap.add_argument("--size-max", type=float, default=2.6, help="max thumbnail scale (x thumb-px) when node-size=children")
    ap.add_argument("--no-zoom", action="store_true")
    args = ap.parse_args()

    img_dir = args.run / "archive" / "images"
    order, parent = load_forest(args.run, args.max_images)
    order = [n for n in order if (img_dir / f"{n}.png").exists()]
    present = set(order)
    parent = {n: (parent[n] if parent[n] in present or parent[n] == VIRTUAL_ROOT else VIRTUAL_ROOT)
              for n in order}
    if not order:
        raise SystemExit("no images found")

    cell = place_lattice(order, parent, mode=args.placement)
    print(f"{len(order)} nodes placed on lattice ({args.placement})")

    th = args.thumb_px
    # per-node display size (px): uniform, or scaled by #published children
    if args.node_size == "children":
        from collections import Counter
        n_children = Counter(parent[n] for n in order if parent[n] != VIRTUAL_ROOT)
        cmax = max(n_children.values(), default=0)
        def node_px(nid):
            frac = (n_children.get(nid, 0) / cmax) ** 0.5 if cmax else 0.0
            return max(2, int(round(th * (args.size_min + (args.size_max - args.size_min) * frac))))
        print(f"node-size=children: max #children={cmax} -> "
              f"{node_px(max(n_children, key=n_children.get)) if cmax else th}px, leaf={node_px('__none__')}px")
    else:
        def node_px(nid):
            return th

    thumbs = {}
    for nid in order:
        s = node_px(nid)
        im = Image.open(img_dir / f"{nid}.png").convert("RGB").resize((s, s), Image.LANCZOS)
        thumbs[nid] = np.asarray(im)

    positions = {nid: (float(c[0]), float(c[1])) for nid, c in cell.items()}
    edge_color = None if args.plain_edges else lineage_colors(order, parent)
    render_reveal(
        args.out, positions, parent, order, thumbs,
        frame=args.frame, fps=args.fps, per_frame=args.per_frame,
        frames_per_step=args.frames_per_step,
        zoom=not args.no_zoom, world=args.world, thumb_px=th,
        edge_color=edge_color,
    )


if __name__ == "__main__":
    main()
