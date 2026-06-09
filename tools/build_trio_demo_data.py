#!/usr/bin/env python3
"""Export the data that drives the *pure-HTML* trio animation (trio_demo.html):
a phylogeny-growth panel, a running leaderboard, and the archive reel, all
scrubbed from one clock -- the HTML counterpart of the rendered ``trio_branched.mp4``.

For the chosen run we emit, under ``assets/trio_demo/<run-key>/``:

  atlas.png        a sprite sheet of every published image (publication order,
                   one ATLAS_CELL-px cell per node). Phylogeny nodes and the
                   leaderboard fullness-bar tiles are blitted from this one image
                   so the page makes a single thumbnail request, not 3,000.
  phylo.json       { n, cell, cols, nodes:[{x,y,p,c,col}] } in publication order.
                   x,y are the radial layout normalised to [0,1] (the page scales
                   them to the panel); p = parent node index (-1 for a founder);
                   c = final direct-child count; col = lineage [r,g,b].
  leaderboard.json { board, K, n, eval_stride,
                     items:[{t,r,nr,c}],         per node index: title, mean
                                                 rating, #ratings, final children
                     timeline:[{count, board:[idx,...]}] }  step-function of the
                     top-K membership over publications (idx into items/nodes).

The phylogeny layout reuses ``archive_tree.radial_layout`` and the per-lineage
hues from ``anim_render.lineage_colors``; the leaderboard step-function
replicates ``leaderboard.py``'s Most-Branched ranking (live child counts, ties
broken by mean rating then rating count) so the HTML panels agree with the video.

Usage:
    python tools/build_trio_demo_data.py \
        --run sweep_logs/sweep/th10_..._s5 --board branched --k 10 --eval-stride 4
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "archive_animations"))
from anim_render import lineage_colors, VIRTUAL_ROOT  # noqa: E402
from archive_tree import radial_layout, graphviz_layout  # noqa: E402

BLOG = Path.home() / "smearle.github.io" / "picbreeder-vlm-06b0d76d"
ATLAS_CELL = 40   # px per thumbnail in the shared sprite sheet


def load_entries(run: Path):
    meta = json.loads((run / "archive" / "archive_metadata.json").read_text())
    entries = meta["entries"] if isinstance(meta, dict) else meta

    def ts(e):
        try:
            return datetime.fromisoformat(e["added_at"])
        except Exception:
            return datetime.min

    return sorted(entries, key=ts)


def build_atlas(order, img_dir: Path, cell: int):
    n = len(order)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    atlas = Image.new("RGB", (cols * cell, rows * cell), (240, 240, 240))
    for i, eid in enumerate(order):
        try:
            im = Image.open(img_dir / f"{eid}.png").convert("RGB").resize((cell, cell), Image.LANCZOS)
        except FileNotFoundError:
            im = Image.new("RGB", (cell, cell), (235, 235, 235))
        atlas.paste(im, ((i % cols) * cell, (i // cols) * cell))
    return atlas, cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--engine", default="sfdp", choices=["sfdp", "radial", "twopi", "fdp", "neato"],
                    help="phylogeny layout: sfdp/fdp/neato/twopi via graphviz (matches the trio video's "
                         "sfdp force-directed tree), radial = the built-in fan")
    ap.add_argument("--frame-w", type=int, default=554, help="layout aspect width the video stretched into (portrait)")
    ap.add_argument("--frame-h", type=int, default=1028, help="layout aspect height the video stretched into")
    ap.add_argument("--board", choices=["rated", "branched"], default="branched")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--eval-stride", type=int, default=4,
                    help="re-rank the board every N publications (branched: raise to tame churn)")
    ap.add_argument("--min-ratings", type=int, default=1, help="rated board: min ratings to be eligible")
    ap.add_argument("--cell", type=int, default=ATLAS_CELL, help="atlas thumbnail px")
    ap.add_argument("--out", type=Path, default=None,
                    help="output dir (default: <blog>/assets/trio_demo/<run-key>)")
    args = ap.parse_args()

    run = args.run
    img_dir = run / "archive" / "images"
    entries = load_entries(run)
    # keep only entries whose image exists, preserving publication order
    entries = [e for e in entries if (img_dir / f"{e['id']}.png").exists()]
    order = [e["id"] for e in entries]
    n = len(order)
    if n == 0:
        raise SystemExit("no images found")
    idx_of = {eid: i for i, eid in enumerate(order)}
    by_id = {e["id"]: e for e in entries}
    present = set(order)

    # parent forest: source_entry_ids[0] if it's a published image we kept, else founder
    parent = {}
    for e in entries:
        src = e.get("source_entry_ids") or []
        p = src[0] if src and src[0] in present else VIRTUAL_ROOT
        parent[e["id"]] = p

    # ---- phylogeny layout (reuse the trio video's), normalised to [0,1] ----
    # The blog trio's left panel was 3,123 nodes laid out with `sfdp` (force-
    # directed graphviz), stretched into a 554x1028 portrait frame. We recompute
    # the same layout here; per-axis normalisation below + a portrait panel in the
    # page reproduces that anisotropic stretch.
    SIZE = 1000
    if args.engine == "radial":
        pos, _depth, _md = radial_layout(order, parent, SIZE, margin=30)
    else:
        pos = graphviz_layout(order, parent, args.engine, SIZE, margin=30)
    xs = [pos[i][0] for i in order]
    ys = [pos[i][1] for i in order]
    minx, maxx, miny, maxy = min(xs), max(xs), min(ys), max(ys)
    spanx = (maxx - minx) or 1.0
    spany = (maxy - miny) or 1.0

    colors = lineage_colors(order, parent)

    # ---- final direct-child counts ----
    children = {eid: 0 for eid in order}
    for e in entries:
        p = parent[e["id"]]
        if p != VIRTUAL_ROOT:
            children[p] += 1

    nodes = []
    for eid in order:
        x, y = pos[eid]
        p = parent[eid]
        nodes.append({
            "x": round((x - minx) / spanx, 5),
            "y": round((y - miny) / spany, 5),
            "p": idx_of[p] if p != VIRTUAL_ROOT else -1,
            "c": children[eid],
            "col": list(colors[eid]),
        })

    # ---- leaderboard step-function (Most-Branched, faithful to leaderboard.py) ----
    def final_avg(eid):
        r = by_id[eid].get("vlm_ratings") or []
        return (sum(r) / len(r)) if r else float("-inf")

    def n_ratings(eid):
        return len(by_id[eid].get("vlm_ratings") or [])

    BRANCHED = args.board == "branched"
    K = args.k
    children_now = {eid: 0 for eid in order}
    valmap: dict[str, float] = {}          # eid -> current ranking value
    cur: list[str] = []
    timeline = []
    since_eval = 0

    # rated board: each image's score = its FINAL mean from the moment it is published
    def rated_val(eid):
        r = by_id[eid].get("vlm_ratings") or []
        if len(r) < args.min_ratings:
            return None
        return sum(r) / len(r)

    def topk():
        if BRANCHED:
            cand = [eid for eid, c in valmap.items() if c > 0]
            cand.sort(key=lambda eid: (valmap[eid], final_avg(eid), n_ratings(eid)), reverse=True)
        else:
            cand = list(valmap.keys())
            cand.sort(key=lambda eid: (valmap[eid], n_ratings(eid)), reverse=True)
        return cand[:K]

    for i, eid in enumerate(order):
        if BRANCHED:
            p = parent[eid]
            if p != VIRTUAL_ROOT:
                children_now[p] += 1
                valmap[p] = children_now[p]
        else:
            v = rated_val(eid)
            if v is not None:
                valmap[eid] = v
        since_eval += 1
        if since_eval >= args.eval_stride or i == n - 1:
            since_eval = 0
            new = topk()
            if new != cur:
                cur = new
                timeline.append({"count": i + 1, "board": [idx_of[x] for x in new]})

    items = [{
        "t": (by_id[eid].get("title") or eid),
        "r": round(final_avg(eid), 2) if n_ratings(eid) else None,
        "nr": n_ratings(eid),
        "c": children[eid],
    } for eid in order]

    # ---- write ----
    run_key = run.name
    out = args.out or (BLOG / "assets" / "trio_demo" / run_key)
    out.mkdir(parents=True, exist_ok=True)

    atlas, cols = build_atlas(order, img_dir, args.cell)
    # JPEG (opaque tiles, no alpha needed) keeps the single thumbnail request light
    atlas.save(out / "atlas.jpg", quality=88, optimize=True)

    (out / "phylo.json").write_text(json.dumps({
        "n": n, "cell": args.cell, "cols": cols, "nodes": nodes,
        "engine": args.engine, "fw": args.frame_w, "fh": args.frame_h,
    }))
    (out / "leaderboard.json").write_text(json.dumps({
        "board": args.board, "K": K, "n": n, "eval_stride": args.eval_stride,
        "cell": args.cell, "cols": cols, "items": items, "timeline": timeline,
    }))

    print(f"run         {run_key}")
    print(f"nodes       {n}  (atlas {cols}x{int(np.ceil(n/cols))} @ {args.cell}px = {atlas.size[0]}x{atlas.size[1]}, "
          f"{(out/'atlas.jpg').stat().st_size//1024} KiB jpg)")
    print(f"board       {args.board} K={K} stride={args.eval_stride}: {len(timeline)} changes")
    print(f"layout      {args.engine}  (stretched to {args.frame_w}x{args.frame_h} portrait)")
    print(f"founders    {sum(1 for nd in nodes if nd['p'] < 0)}")
    print(f"wrote       {out}/  (atlas.jpg, phylo.json, leaderboard.json)")


if __name__ == "__main__":
    main()
