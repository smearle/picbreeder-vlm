#!/usr/bin/env python3
"""Scrolling-feed archive growth: fixed zoom, no history kept.

A fixed grid of `--cols` columns scrolls upward at a constant rate. Each new row
of published images enters at the BOTTOM and, as it rises, CPPN-morphs from each
image's parent into the image's own final form (real weight/activation
interpolation between the parent's archive genome and the child's). Rows scroll
off the top -- history isn't retained, so it reads as an endless feed of the
archive being born, one branched form at a time.

Morphs need genomes for each image AND its parent; this run saves genomes for the
first ~1000 publications, so keep `--n` within that.

Precompute (CPPN eval) is parallelised across processes; compositing/scroll is
cheap.

Usage:
    python archive_animations/archive_scroll.py \
        --run sweep_logs/sweep/th0_..._s4 \
        --out archive_animations/out/archive_scroll.mp4 \
        --n 200 --cols 9 --cell 64 --morph-steps 10 --jobs 10
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import pickle
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from render_lineage_animation import build_superset, render_frames, pick_variant  # noqa: E402
from rendering import render_genome_image  # noqa: E402

ORANGE = (255, 108, 0)

# --- worker globals (one config per process) ---
_CFG = None
_GDIR = None
_RUN = None
_CELL = 64
_STEPS = 10


def _init(gdir, run, cell, steps):
    global _CFG, _GDIR, _RUN, _CELL, _STEPS
    from archive_animations.cppn_interp import build_config
    _CFG = build_config()
    _GDIR, _RUN, _CELL, _STEPS = gdir, run, cell, steps


def _load(eid):
    with open(os.path.join(_GDIR, eid + ".pkl"), "rb") as f:
        return pickle.load(f)


def _gen0_genome(agent_id, child):
    """The session's random-initial genome (gen-0 selection), from the agent's
    saved population -- read from the agent zip, or from an unpacked agent dir
    when there's no zip. If the agent crossed over two gen-0 picks, choose the
    one sharing the most connection genes with the published genome (its true
    ancestor). Returns None when no gen-0 population was saved (e.g. founder
    agents, whose populations were never persisted)."""
    inner = f"{agent_id}/populations/selected_gen_000.pkl.gz"
    raw = None
    zp = os.path.join(_RUN, "agents", agent_id + ".zip")
    if os.path.exists(zp):
        try:
            with zipfile.ZipFile(zp) as z, z.open(inner) as fh:
                raw = fh.read()
        except (KeyError, FileNotFoundError, OSError):
            raw = None
    if raw is None:
        dp = os.path.join(_RUN, "agents", agent_id, "populations", "selected_gen_000.pkl.gz")
        if os.path.exists(dp):
            with open(dp, "rb") as fh:
                raw = fh.read()
    if raw is None:
        return None
    sel = pickle.loads(gzip.decompress(raw)).get("selected") or []
    if not sel:
        return None
    pub = set(child.connections.keys())
    return max(sel, key=lambda s: len(set(s["genome"].connections.keys()) & pub))["genome"]


def _synth_seed_genome(child_id):
    """A fresh random-initial CPPN of the run's config -- the kind of network a
    session starts from before interactive evolution. Used for root publications
    whose true gen-0 genome was never saved (founder agents), so they still morph
    out of a random initial network rather than appearing static. Seeded per
    child id so the synthesized ancestor is stable across renders."""
    import random
    g = _CFG.genome_type(0)
    state = random.getstate()
    try:
        random.seed("seed:" + child_id)
        g.configure_new(_CFG.genome_config)
    finally:
        random.setstate(state)
    return g


def _static_stack(cg, color):
    gray, col = render_genome_image(cg, _CFG, _CELL, _CELL)
    return [np.asarray(pick_variant(gray, col, "auto", color).convert("RGB"))]


def _morph_stack(spec):
    """Return list of cell×cell×3 uint8 frames morphing source -> child.
    spec = (child_id, kind, ref, color); kind in {archive, root, none}.
    A root with no saved gen-0 genome (founder agent) morphs from a synthesized
    random-initial CPPN, so it still grows out of a starting network. Only kind
    'none' (or a failed morph) renders static."""
    child_id, kind, ref, color = spec
    cg = _load(child_id)
    src = None
    if kind == "archive":
        src = _load(ref)
    elif kind == "root":
        src = _gen0_genome(ref, cg) or _synth_seed_genome(child_id)
    if src is None:
        return _static_stack(cg, color)
    try:
        ss, nodp, conp, outs = build_superset(src, cg)
        frames = render_frames(ss, nodp, conp, _CFG, steps=_STEPS, width=_CELL, height=_CELL,
                               variant_mode="auto", color_enabled=color, output_activation_stats=outs)
        return [np.asarray(fr.convert("RGB")) for fr in frames]
    except Exception:
        return _static_stack(cg, color)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n", type=int, default=200, help="number of publications to feed (<= genomes saved)")
    ap.add_argument("--cols", type=int, default=9)
    ap.add_argument("--cell", type=int, default=88, help="tile size px (also CPPN render res)")
    ap.add_argument("--gap", type=int, default=6)
    ap.add_argument("--rows-visible", type=int, default=7)
    ap.add_argument("--morph-steps", type=int, default=14, help="rendered CPPN frames per morph")
    ap.add_argument("--morph-frames", type=int, default=26, help="display frames a morph plays over")
    ap.add_argument("--row-frames", type=int, default=26, help="(unused in sequential mode; see --scroll-frames)")
    ap.add_argument("--scroll-frames", type=int, default=18, help="frames to scroll up one row between cycles")
    ap.add_argument("--descent-frames", type=int, default=0, help="base descent frames (sped up 1.5x; 0 = 30)")
    ap.add_argument("--hold", type=int, default=30, help="frames to hold on the final frame")
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--outline", action="store_true", help="orange ring on tiles still mid-morph")
    args = ap.parse_args()

    img_dir = args.run / "archive" / "images"
    gdir = str(args.run / "archive" / "genomes")
    ents = sorted(json.loads((args.run / "archive" / "archive_metadata.json").read_text())["entries"],
                  key=lambda e: e.get("added_at", ""))
    # keep first N that have a genome on disk
    feed = []
    for e in ents:
        if os.path.exists(os.path.join(gdir, e["id"] + ".pkl")):
            feed.append(e)
        if len(feed) >= args.n:
            break
    n = len(feed)
    byid = {e["id"]: e for e in ents}

    def spec_of(e):
        s = e.get("source_entry_ids") or []
        p = s[0] if s else None
        if p and os.path.exists(os.path.join(gdir, p + ".pkl")):
            return (e["id"], "archive", p, bool(e.get("color_enabled")))     # branched: morph from parent image
        # root: founder evolved from a random-init CPPN -- morph from its gen-0
        # genome, or a synthesized random-init seed when gen-0 wasn't saved.
        return (e["id"], "root", e["agent_id"], bool(e.get("color_enabled")))

    specs = [spec_of(e) for e in feed]
    nbr = sum(1 for s in specs if s[1] == "archive")
    nrt = sum(1 for s in specs if s[1] == "root")
    print(f"feeding {n} publications: {nbr} branched-morphs, {nrt} root(random-init)-morphs, "
          f"{n-nbr-nrt} static; rendering...")

    from multiprocessing import Pool
    with Pool(args.jobs, initializer=_init,
              initargs=(gdir, str(args.run), args.cell, args.morph_steps)) as pool:
        stacks = pool.map(_morph_stack, specs)
    print("morphs rendered; compositing scroll...")

    C, d, g = args.cols, args.cell, args.gap
    V = args.rows_visible
    step = d + g
    W = C * step + g
    H = V * step + g
    W += W % 2
    H += H % 2
    total_rows = math.ceil(n / C)
    DESC = max(1, round((args.descent_frames if args.descent_frames else 30) / 1.5))  # descent sped up 1.5x
    MORPH = args.morph_frames
    SCROLL = args.scroll_frames
    col = [i % C for i in range(n)]
    row = [i // C for i in range(n)]
    id2idx = {feed[i]["id"]: i for i in range(n)}
    psrc = [id2idx.get(specs[i][2]) if specs[i][1] == "archive" else None for i in range(n)]

    def wx(c): return g + c * step

    # ---- sequential schedule -------------------------------------------------
    # Fill phase (rows 0..V-1): each row lands at its own screen row, TOP-to-BOTTOM,
    #   cycle = DESC (descend from parents above) -> MORPH (in place). No scrolling.
    # Steady phase (rows >= V): cycle = SCROLL (whole grid up one row, top exits,
    #   bottom clears) -> DESC (into the bottom row) -> MORPH (in place).
    # Within a row, descent (motion) and morph (stationary) never overlap; scrolling
    # is its own phase between rows, so it only happens once the frame is full.
    desc_s = [0] * total_rows
    land = [0] * total_rows          # descent done; morph starts (stationary)
    morph_e = [0] * total_rows
    scroll_seg = []                  # (start, end, from_offset, to_offset)
    f = 0
    offset = 0
    for r in range(total_rows):
        if r >= V:
            scroll_seg.append((f, f + SCROLL, offset, offset + 1))
            offset += 1
            f += SCROLL
        desc_s[r] = f
        land[r] = f + DESC
        morph_e[r] = land[r] + MORPH
        f = morph_e[r]
    total_frames = f + args.hold

    def scroll_offset(fr):
        o = 0.0
        for (s, e, a, b) in scroll_seg:
            if fr >= e:
                o = b
            elif fr > s:
                tt = (fr - s) / (e - s)
                o = a + (b - a) * (tt * tt * (3 - 2 * tt))
            else:
                break
        return o

    def rest_y(r, fr):  # resting screen-y of row r's slot (screen-row = r - offset)
        return g + (r - scroll_offset(fr)) * step

    import cv2

    def blit(canvas, tile, x, y, outline=False):
        h, w = tile.shape[:2]
        ya, yb = max(0, y), min(H, y + h)
        xa, xb = max(0, x), min(W, x + w)
        if ya >= yb or xa >= xb:
            return
        canvas[ya:yb, xa:xb] = tile[ya - y:yb - y, xa - x:xb - x]
        if outline:
            cv2.rectangle(canvas, (xa, ya), (xb - 1, yb - 1), ORANGE, 1)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{W}x{H}", "-r", str(args.fps), "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(args.out)],
        stdin=subprocess.PIPE,
    )
    for f in range(total_frames):
        canvas = np.full((H, W, 3), 255, np.uint8)
        descenders = []   # drawn last, on top
        for i in range(n):
            r = row[i]
            if f < desc_s[r]:                 # not yet its turn
                continue
            cx = wx(col[i])
            ry = rest_y(r, f)                 # this slot's resting screen-y at frame f
            if f < land[r]:                   # DESCENDING (motion), showing the source clone
                pj = psrc[i]
                if pj is None:                # root: random-init appears in its slot (no parent)
                    if -d < ry < H:
                        descenders.append((cx, int(round(ry)), stacks[i][0]))
                    continue
                t = (f - desc_s[r]) / DESC
                t = t * t * (3 - 2 * t)
                px = wx(col[pj]) + (cx - wx(col[pj])) * t
                py = rest_y(row[pj], f) + (ry - rest_y(row[pj], f)) * t
                descenders.append((int(round(px)), int(round(py)), stacks[i][0]))
                continue
            # MORPH (stationary, in place) then SETTLED; both ride the scroll afterwards
            sy = int(round(ry))
            if sy <= -d or sy >= H:
                continue
            prog = (f - land[r]) / MORPH
            prog = 0.0 if prog < 0 else (1.0 if prog > 1 else prog)
            idx = int(round(prog * (len(stacks[i]) - 1)))
            blit(canvas, stacks[i][idx], cx, sy, outline=args.outline and prog < 1.0)
        for bx, by, tile in descenders:
            blit(canvas, tile, bx, by, outline=args.outline)
        proc.stdin.write(canvas.tobytes())
    proc.stdin.close()
    proc.wait()
    print(f"wrote {args.out} ({total_frames} frames, {total_frames/args.fps:.1f}s)")


if __name__ == "__main__":
    main()
