#!/usr/bin/env python3
"""Render 36 walker traversals over the phylogeny graph as a GIF (or MP4).

Reuses the graphviz `sfdp` layout from `archive_tree.py` to place all nodes,
draws the full forest faintly in the background, then animates each walker as a
colored trail expanding along the directed edges of its assigned Euler-tour
slice. One frame per Euler-tour step; all 36 walkers advance in lockstep, so
their walks are all the same length on screen too.

Usage:
    .venv/bin/python archive_animations/walker_trails.py \\
        --run sweep_logs/sweep/th0_..._s4 \\
        --out archive_animations/out/walker_trails_s4.gif \\
        -k 36 --size 900 --fps 24 --stride 2
"""
from __future__ import annotations

import argparse
import colorsys
import math
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw

from archive_tree import graphviz_layout, radial_layout
from walker_partition import VIRTUAL_ROOT, best_first_walks, build_walks, edge_coverage


def walker_colors(k: int) -> List[Tuple[int, int, int]]:
    """k visually-distinct RGB colors, guaranteed no duplicates.

    Hues are the k evenly-spaced points on the wheel, but emitted in a permuted
    order (stride by an integer coprime with k near the golden ratio) so that
    consecutive walker indices land far apart on the wheel. Three saturation/
    value tiers are layered on top for extra separability between similar hues.
    Tuned for a dark background (high value, medium saturation -> luminous).
    """
    cols = []
    stride = max(1, int(round(k * 0.6180339887))) | 1   # near golden, force odd
    while math.gcd(stride, k) != 1:
        stride += 2
    tiers = [(0.78, 1.00), (0.58, 0.93), (0.95, 0.82)]   # (sat, val)
    for i in range(k):
        hue = ((i * stride) % k) / k
        sat, val = tiers[i % len(tiers)]
        r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
        cols.append((int(r * 255), int(g * 255), int(b * 255)))
    return cols


# (background, base-edge, base-node) RGB per theme
THEMES = {
    "light": {"bg": (255, 255, 255), "edge": (235, 235, 235), "node": (170, 170, 170),
              "head_ring": (30, 30, 30), "glow": False},
    "dark":  {"bg": (16, 16, 22), "edge": (54, 54, 66), "node": (90, 90, 104),
              "head_ring": (245, 245, 245), "glow": True},
}


def render_background(
    pos: Dict[str, Tuple[float, float]],
    parent: Dict[str, str],
    order: List[str],
    size: int,
    bg_color: Tuple[int, int, int] = (255, 255, 255),
    edge_color: Tuple[int, int, int] = (235, 235, 235),
    node_color: Tuple[int, int, int] = (170, 170, 170),
    node_radius: int = 1,
) -> Image.Image:
    """Faint static rendering of every real edge + every node, as a base layer."""
    img = Image.new("RGB", (size, size), bg_color)
    draw = ImageDraw.Draw(img)
    for n in order:
        p = parent.get(n, VIRTUAL_ROOT)
        if p == VIRTUAL_ROOT or p not in pos or n not in pos:
            continue
        x1, y1 = pos[n]
        x0, y0 = pos[p]
        draw.line([(x0, y0), (x1, y1)], fill=edge_color, width=1)
    for n, (x, y) in pos.items():
        if n == VIRTUAL_ROOT:
            continue
        r = node_radius
        draw.ellipse([(x - r, y - r), (x + r, y + r)], fill=node_color)
    return img


def _bloom(frame: np.ndarray, strength: float = 0.7, sigma: float = 3.5) -> np.ndarray:
    """Additive gaussian bloom -- makes bright trails glow on a dark background."""
    blur = cv2.GaussianBlur(frame, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return cv2.addWeighted(frame, 1.0, blur, strength, 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path,
                    help="Output path; suffix .gif | .mp4 picks the writer.")
    ap.add_argument("-k", type=int, default=36, help="number of walkers")
    ap.add_argument("--method", choices=["euler-slice", "best-first"], default="best-first",
                    help="euler-slice: global rating-ordered DFS tour split into "
                         "k equal-length contiguous segments (tiles all edges, but "
                         "cross-root-tree transitions JUMP). best-first: k "
                         "continuous walkers, no jumps, retread penalty.")
    ap.add_argument("-K", type=int, default=200, help="(best-first) edges per walker")
    ap.add_argument("--max-n", type=int, default=None,
                    help="cap phylogeny to first N publications (matches walker_grid)")
    ap.add_argument("--require-genome", action="store_true",
                    help="restrict to nodes with on-disk genome pickles (matches walker_grid)")
    ap.add_argument("--alpha", type=float, default=0.5, help="(best-first) rating weight")
    ap.add_argument("--beta", type=float, default=2.0, help="(best-first) unvisited-edge bonus")
    ap.add_argument("--gamma", type=float, default=1.0, help="(best-first) per-walker retread penalty")
    ap.add_argument("--engine", default="sfdp",
                    choices=["sfdp", "fdp", "neato", "twopi", "radial"])
    ap.add_argument("--size", type=int, default=900)
    ap.add_argument("--margin", type=int, default=24)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--stride", type=int, default=1,
                    help="Edges advanced per frame (when frames-per-edge=1). "
                         ">1 speeds up via discrete jumps.")
    ap.add_argument("--frames-per-edge", type=int, default=1,
                    help="If >1, walker head sub-frame-interpolates along each edge "
                         "over this many frames (gives smooth pacing for syncing to "
                         "walker_grid). Overrides --stride.")
    ap.add_argument("--hold", type=int, default=0,
                    help="Extra still frames at the end (final state).")
    ap.add_argument("--trail-width", type=int, default=2)
    ap.add_argument("--head-radius", type=int, default=4)
    ap.add_argument("--theme", choices=list(THEMES), default="dark",
                    help="dark = luminous trails on near-black with bloom (default); "
                         "light = original white background.")
    ap.add_argument("--glow", dest="glow", action="store_true", default=None,
                    help="force bloom on (default follows theme)")
    ap.add_argument("--no-glow", dest="glow", action="store_false",
                    help="force bloom off")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="If >0, cap total frame count (useful for debugging).")
    args = ap.parse_args()
    theme = THEMES[args.theme]
    glow = theme["glow"] if args.glow is None else args.glow
    head_ring = theme["head_ring"]

    if args.method == "euler-slice":
        walks, parent, score, order = build_walks(args.run, args.k)
    else:
        walks, parent, score, order, _starts = best_first_walks(
            args.run, args.k, args.K,
            alpha=args.alpha, beta=args.beta, gamma=args.gamma,
            max_n=args.max_n, require_genome=args.require_genome,
        )
    n_steps = max(len(w) for w in walks)
    if args.frames_per_edge > 1:
        n_frames = n_steps * args.frames_per_edge + args.hold
    else:
        n_frames = math.ceil(n_steps / args.stride) + args.hold
    if args.max_frames:
        n_frames = min(n_frames, args.max_frames)
    cov, tot = edge_coverage(walks, parent)
    print(f"forest: {len(order)} nodes, "
          f"{sum(1 for c,p in parent.items() if p != VIRTUAL_ROOT)} real edges; "
          f"k={args.k} walkers, max walk len={n_steps}, frames={n_frames}; "
          f"method={args.method}, coverage={cov}/{tot} ({100*cov/tot:.1f}%)")

    # Layout
    if args.engine == "radial":
        pos, _depth, _max_d = radial_layout(order, parent, args.size, args.margin)
    else:
        pos = graphviz_layout(order, parent, args.engine, args.size, args.margin)

    # Static background (faint full forest) — composited once.
    bg = render_background(pos, parent, order, args.size,
                           bg_color=theme["bg"], edge_color=theme["edge"],
                           node_color=theme["node"])
    bg_arr = np.asarray(bg).copy()

    colors = walker_colors(args.k)

    # Pre-resolve each walker's edge endpoints in pixel space.
    edge_px: List[List[Tuple[Tuple[int, int], Tuple[int, int]]]] = []
    for w in walks:
        ws = []
        for a, b in w:
            if a not in pos or b not in pos:
                ws.append(None)
                continue
            ax, ay = pos[a]
            bx, by = pos[b]
            ws.append(((int(ax), int(ay)), (int(bx), int(by))))
        edge_px.append(ws)

    # Layer that accumulates trails (each walker's drawn edges so far).
    # Use a single RGB image and overdraw — last-in-frame win is fine since
    # we want the most-recent walker's color visible where walks overlap.
    trail = bg_arr.copy()

    suffix = args.out.suffix.lower()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    # Two writers: imageio for .gif, ffmpeg pipe for .mp4
    if suffix == ".gif":
        import imageio
        writer = imageio.get_writer(str(args.out), mode="I",
                                    duration=1.0 / args.fps, loop=0)
    elif suffix == ".mp4":
        import subprocess
        writer_proc = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{args.size}x{args.size}", "-r", str(args.fps), "-i", "-",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
             str(args.out)],
            stdin=__import__("subprocess").PIPE,
        )
        class _MP4:
            def append_data(self, im): writer_proc.stdin.write(np.ascontiguousarray(im).tobytes())
            def close(self):
                writer_proc.stdin.close(); writer_proc.wait()
        writer = _MP4()
    else:
        raise SystemExit(f"unknown output suffix {suffix!r} (use .gif or .mp4)")

    try:
        if args.frames_per_edge > 1:
            # Smooth per-edge sub-frame interpolation: walker head advances
            # along the current edge over `frames_per_edge` sub-frames; once
            # the edge is fully traversed it gets committed to the trail layer.
            fpe = args.frames_per_edge
            last_committed = [-1] * len(edge_px)  # per-walker
            walk_frames = n_steps * fpe
            for frame_i in range(n_frames):
                if frame_i < walk_frames:
                    edge_idx = frame_i // fpe
                    sub = frame_i % fpe
                else:
                    edge_idx = n_steps - 1
                    sub = fpe - 1
                # Commit any edges that are now fully done (sub == fpe-1 of
                # their slot) into the trail layer, once each.
                for w_i, edges in enumerate(edge_px):
                    col = colors[w_i]
                    # commit edges up through (edge_idx - 1) plus edge_idx if sub == fpe-1
                    final = edge_idx - 1 if sub < fpe - 1 else edge_idx
                    while last_committed[w_i] < final:
                        last_committed[w_i] += 1
                        ei = last_committed[w_i]
                        if ei < len(edges) and edges[ei] is not None:
                            p0, p1 = edges[ei]
                            cv2.line(trail, p0, p1, col, args.trail_width, cv2.LINE_AA)

                # per-frame overlay: partial current edge + head dot
                frame = trail.copy()
                for w_i, edges in enumerate(edge_px):
                    col = colors[w_i]
                    if edge_idx >= len(edges) or edges[edge_idx] is None:
                        continue
                    p0, p1 = edges[edge_idx]
                    frac = (sub + 1) / fpe   # so head ends at p1 when sub==fpe-1
                    hx = int(p0[0] + frac * (p1[0] - p0[0]))
                    hy = int(p0[1] + frac * (p1[1] - p0[1]))
                    if sub < fpe - 1:
                        cv2.line(frame, p0, (hx, hy), col, args.trail_width, cv2.LINE_AA)
                    cv2.circle(frame, (hx, hy), args.head_radius, col, -1, cv2.LINE_AA)
                    cv2.circle(frame, (hx, hy), args.head_radius + 1, head_ring, 1, cv2.LINE_AA)

                if glow:
                    frame = _bloom(frame)
                writer.append_data(frame)
                if (frame_i + 1) % max(1, n_frames // 12) == 0 or frame_i == n_frames - 1:
                    print(f"  frame {frame_i + 1}/{n_frames}")
        else:
            # Original discrete-stride mode.
            step = 0
            for frame_i in range(n_frames):
                head_pts: List[Tuple[Tuple[int, int], Tuple[int, int, int]]] = []
                for w_i, edges in enumerate(edge_px):
                    col = colors[w_i]
                    for s in range(args.stride):
                        idx = step + s
                        if idx >= len(edges):
                            break
                        e = edges[idx]
                        if e is None:
                            continue
                        p0, p1 = e
                        cv2.line(trail, p0, p1, col, args.trail_width, cv2.LINE_AA)
                    last_idx = min(step + args.stride - 1, len(edges) - 1)
                    if 0 <= last_idx < len(edges) and edges[last_idx] is not None:
                        head_pts.append((edges[last_idx][1], col))
                step += args.stride

                frame = trail.copy()
                for (hx, hy), col in head_pts:
                    cv2.circle(frame, (hx, hy), args.head_radius, col, -1, cv2.LINE_AA)
                    cv2.circle(frame, (hx, hy), args.head_radius + 1, head_ring, 1, cv2.LINE_AA)

                if glow:
                    frame = _bloom(frame)
                writer.append_data(frame)
                if (frame_i + 1) % 20 == 0 or frame_i == n_frames - 1:
                    print(f"  frame {frame_i + 1}/{n_frames}")
    finally:
        writer.close()
    print(f"wrote {args.out} ({n_frames} frames, {n_frames / args.fps:.1f}s)")


if __name__ == "__main__":
    main()
