#!/usr/bin/env python3
"""Animate the *branching structure* of a Picbreeder-VLM archive as a radial
phylogenetic tree that grows from the center outward.

Every published image is a node; an edge connects it to the archive image it was
branched from (``source_entry_ids[0]``). Images that started from a random
population are roots, attached to a virtual center. We lay the forest out
radially (depth -> radius, sub-tree leaf-count -> angular wedge) so siblings fan
out without overlapping, then reveal nodes in publication order (``added_at``):
the archive blooms from the middle, each parent spawning offspring toward the rim.

Unlike a row-major fill, this exposes the tree's shape -- the hubs that get
re-branched dozens of times, and the long timid chains.

Usage:
    python archive_animations/archive_tree.py \
        --run sweep_logs/sweep/th0_..._s4 \
        --out archive_animations/out/archive_tree_s4.mp4 \
        --size 1080 --thumb 13 --per-frame 22 --fps 24
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

from anim_render import render_reveal, lineage_colors, VIRTUAL_ROOT, ORANGE


def load_forest(run: Path, max_images: int | None):
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
        p = src[0] if src and src[0] in ids else VIRTUAL_ROOT
        parent[e["id"]] = p
    order = [e["id"] for e in entries]
    return order, parent


def radial_layout(order, parent, size, margin):
    children = defaultdict(list)
    for node, par in parent.items():
        children[par].append(node)
    # publication order within siblings -> stable, time-consistent fan
    pos_in_order = {n: i for i, n in enumerate(order)}
    for par in children:
        children[par].sort(key=lambda n: pos_in_order.get(n, 0))

    # depth via BFS from virtual root
    depth = {VIRTUAL_ROOT: 0}
    dq = deque([VIRTUAL_ROOT])
    while dq:
        u = dq.popleft()
        for v in children.get(u, []):
            depth[v] = depth[u] + 1
            dq.append(v)
    max_depth = max((d for n, d in depth.items() if n != VIRTUAL_ROOT), default=1)

    # leaf counts via iterative post-order
    leaf = {}
    stack = [(VIRTUAL_ROOT, False)]
    while stack:
        node, processed = stack.pop()
        kids = children.get(node, [])
        if not kids:
            leaf[node] = 1
            continue
        if not processed:
            stack.append((node, True))
            for k in kids:
                stack.append((k, False))
        else:
            leaf[node] = sum(leaf[k] for k in kids)

    # assign angular spans (pre-order)
    ang0, ang1 = {VIRTUAL_ROOT: 0.0}, {VIRTUAL_ROOT: 2 * math.pi}
    dq = deque([VIRTUAL_ROOT])
    while dq:
        u = dq.popleft()
        kids = children.get(u, [])
        if not kids:
            continue
        total = sum(leaf[k] for k in kids)
        a = ang0[u]
        span = ang1[u] - ang0[u]
        for k in kids:
            frac = leaf[k] / total
            ang0[k], ang1[k] = a, a + span * frac
            a = ang1[k]
            dq.append(k)

    cx = cy = size / 2
    R = size / 2 - margin
    pos = {}
    for n, d in depth.items():
        if n == VIRTUAL_ROOT:
            pos[n] = (cx, cy)
            continue
        # sqrt radial spacing -> inner rings less cramped, outer rings breathe
        r = R * math.sqrt(d / max_depth)
        ang = 0.5 * (ang0[n] + ang1[n])
        pos[n] = (cx + r * math.cos(ang), cy + r * math.sin(ang))
    return pos, depth, max_depth


def graphviz_layout(order, parent, engine, size, margin):
    """Compute node positions with a graphviz engine (twopi=radial, sfdp/fdp/
    neato=force-directed). Returns {id: (x,y)} scaled into the frame."""
    import subprocess
    present = set(order)
    lines = ["graph G {", "  node [shape=point, width=0.02, height=0.02];", "  edge [len=0.4];"]
    if engine == "twopi":
        lines.append(f'  "{VIRTUAL_ROOT}";')
    # declare every node so even childless roots get coordinates
    for nid in order:
        lines.append(f'  "{nid}";')
    for nid in order:
        par = parent[nid]
        if par == VIRTUAL_ROOT and engine != "twopi":
            continue  # don't anchor roots to a virtual hub for force layouts
        if par in present or par == VIRTUAL_ROOT:
            lines.append(f'  "{par}" -- "{nid}";')
    lines.append("}")
    dot = "\n".join(lines).encode()

    cmd = [engine, "-Tplain"]
    if engine == "twopi":
        cmd += [f"-Groot={VIRTUAL_ROOT}", "-Goverlap=false", "-Granksep=1.2"]
    else:
        cmd += ["-Goverlap=prism", "-Gne=0"]
    out = subprocess.run(cmd, input=dot, capture_output=True).stdout.decode()

    pos = {}
    for line in out.splitlines():
        if line.startswith("node "):
            parts = line.split()
            # node "name" x y w h label style shape color fillcolor
            name = parts[1].strip('"')
            pos[name] = (float(parts[2]), float(parts[3]))
    if not pos:
        raise SystemExit(f"{engine} produced no coordinates")
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    minx, maxx, miny, maxy = min(xs), max(xs), min(ys), max(ys)
    span = max(maxx - minx, maxy - miny) or 1.0
    scale = (size - 2 * margin) / span
    ox = (size - (maxx - minx) * scale) / 2
    oy = (size - (maxy - miny) * scale) / 2
    return {n: (ox + (x - minx) * scale, size - (oy + (y - miny) * scale)) for n, (x, y) in pos.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--engine", default="radial",
                    choices=["radial", "twopi", "sfdp", "fdp", "neato"],
                    help="radial = built-in; others use graphviz")
    ap.add_argument("--size", type=int, default=1080)
    ap.add_argument("--frame-w", type=int, default=None,
                    help="output width for a non-square (portrait/landscape) frame; "
                         "the layout is stretched to FILL it. Default: square --size.")
    ap.add_argument("--frame-h", type=int, default=None, help="output height (see --frame-w)")
    ap.add_argument("--thumb", type=int, default=14)
    ap.add_argument("--per-frame", type=int, default=16, help="new nodes revealed per step")
    ap.add_argument("--frames-per-step", type=int, default=1, help="frames to dwell on each step")
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--zoom", action="store_true", help="zoom the viewport to the revealed sub-archive")
    ap.add_argument("--plain-edges", action="store_true", help="grey edges instead of per-lineage colour")
    ap.add_argument("--margin", type=int, default=40)
    ap.add_argument("--board-timeline", type=Path, default=None,
                    help="JSON board-membership timeline from leaderboard.py "
                         "(--emit-board-timeline). When given, whichever images are on "
                         "the leaderboard at each step are enlarged in place, sized by "
                         "their current branch count; nodes shrink back when they drop off.")
    ap.add_argument("--hub-scale-min", type=float, default=2.0,
                    help="thumbnail size multiplier for an image as it enters the board")
    ap.add_argument("--hub-scale-max", type=float, default=3.0,
                    help="thumbnail size multiplier for the most-branched leader")
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--hold", type=int, default=36)
    ap.add_argument("--edge-shade", type=int, default=205, help="grey level for edges")
    args = ap.parse_args()

    img_dir = args.run / "archive" / "images"
    order, parent = load_forest(args.run, args.max_images)
    order = [n for n in order if (img_dir / f"{n}.png").exists()]
    present = set(order)
    # drop parents that were filtered out
    parent = {n: (parent[n] if parent[n] in present or parent[n] == VIRTUAL_ROOT else VIRTUAL_ROOT)
              for n in order}
    n = len(order)
    if n == 0:
        raise SystemExit("no images found")

    if args.engine == "radial":
        pos, depth, max_depth = radial_layout(order, parent, args.size, args.margin)
        print(f"{n} nodes, max tree depth {max_depth} (radial)")
    else:
        pos = graphviz_layout(order, parent, args.engine, args.size, args.margin)
        print(f"{n} nodes laid out with {args.engine}")

    th = args.thumb
    thumbs = {}
    for nid in order:
        im = Image.open(img_dir / f"{nid}.png").convert("RGB").resize((th, th), Image.LANCZOS)
        thumbs[nid] = np.asarray(im)

    # Dynamic emphasis: enlarge whichever images are on the Most-Branched
    # leaderboard at each step, driven by a board-membership timeline emitted by
    # leaderboard.py (so both panels agree exactly). A node grows when it joins
    # the board (sized by its current branch count) and snaps back when it drops.
    hi_thumbs = None
    emphasis_seq = None
    if args.board_timeline:
        hi = max(args.thumb * 6, 128)
        hi_thumbs = {nid: np.asarray(
            Image.open(img_dir / f"{nid}.png").convert("RGB").resize((hi, hi), Image.LANCZOS))
            for nid in order}

        def emph_size(cnt):  # current branch count -> thumbnail size in WORLD px
            frac = min(1.0, cnt / 40.0)
            return th * (args.hub_scale_min + (args.hub_scale_max - args.hub_scale_min) * frac)

        tl = json.loads(Path(args.board_timeline).read_text())["timeline"]
        # step-function over publication count -> {id: world-px size}
        emphasis_seq, ptr, cur_board = [], 0, {}
        for k in range(len(order)):
            count = k + 1
            while ptr < len(tl) and tl[ptr]["count"] <= count:
                cur_board = {eid: emph_size(c) for eid, c in tl[ptr]["board"]}
                ptr += 1
            emphasis_seq.append(cur_board)
        distinct = len(set().union(*[set(d) for d in emphasis_seq])) if emphasis_seq else 0
        print(f"board timeline: {len(tl)} changes; {distinct} distinct images emphasised over the run")

    # Hand off to the shared renderer: lineage edges behind thumbnails, optional
    # zoom-out viewport, half-speed reveal.
    edge_color = None if args.plain_edges else lineage_colors(order, parent)
    frame = (args.frame_w, args.frame_h) if (args.frame_w and args.frame_h) else args.size
    render_reveal(
        args.out, pos, parent, order, thumbs,
        frame=frame, fps=args.fps, per_frame=args.per_frame,
        frames_per_step=args.frames_per_step, hold=args.hold,
        zoom=args.zoom, thumb_px=th, edge_shade=args.edge_shade, edge_color=edge_color,
        hi_thumbs=hi_thumbs, emphasis_seq=emphasis_seq,
    )


if __name__ == "__main__":
    main()
