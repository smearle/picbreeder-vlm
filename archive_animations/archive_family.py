#!/usr/bin/env python3
"""Reorganizing "family grid": a fixed grid that continually refreshes so that a
parent is ALWAYS on-screen the moment its child is branched.

Children are processed in publication order, in waves of `--cols`. Each new child
flies in from its parent's tile (guaranteed present) to the nearest **free or
"done"** slot -- "done" = a tile that has no remaining children to spawn -- and
morphs from a clone of its source into its own form. A tile is never evicted while
it still has children to come (and is re-introduced if it ever was), so branching
always originates from a visible parent. Roots appear in place (random-init) and
morph. Settled tiles don't move; the grid reorganizes by *what* fills each slot.

Reuses the morph precompute + config from `archive_scroll`.

Usage:
    python archive_animations/archive_family.py --run sweep_logs/sweep/th0_..._s4 \
        --out archive_animations/out/archive_family.mp4 \
        --n 200 --cols 9 --rows 7 --cell 88 --jobs 12
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import archive_animations.archive_scroll as scr  # _init / _morph_stack workers  # noqa: E402

ORANGE = (255, 108, 0)


def load_feed(run: Path, n):
    gdir = run / "archive" / "genomes"
    ents = sorted(json.loads((run / "archive" / "archive_metadata.json").read_text())["entries"],
                  key=lambda e: e.get("added_at", ""))
    feed = [e for e in ents if (gdir / f"{e['id']}.pkl").exists()][:n]
    return feed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--cols", type=int, default=9)
    ap.add_argument("--rows", type=int, default=7)
    ap.add_argument("--cell", type=int, default=88)
    ap.add_argument("--gap", type=int, default=6)
    ap.add_argument("--morph-steps", type=int, default=14)
    ap.add_argument("--morph-frames", type=int, default=30)
    ap.add_argument("--descent-frames", type=int, default=20, help="fly-in frames per wave")
    ap.add_argument("--pause-frames", type=int, default=6, help="settle pause after each wave")
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--hold", type=int, default=36)
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--outline", action="store_true")
    args = ap.parse_args()

    gdir = str(args.run / "archive" / "genomes")
    feed = load_feed(args.run, args.n)
    n = len(feed)
    idx = {e["id"]: i for i, e in enumerate(feed)}

    def spec_of(e):
        s = e.get("source_entry_ids") or []
        p = s[0] if s else None
        if p and os.path.exists(os.path.join(gdir, p + ".pkl")):
            return (e["id"], "archive", p, bool(e.get("color_enabled")))
        if os.path.exists(os.path.join(str(args.run), "agents", e["agent_id"] + ".zip")):
            return (e["id"], "root", e["agent_id"], bool(e.get("color_enabled")))
        return (e["id"], "none", None, bool(e.get("color_enabled")))

    specs = [spec_of(e) for e in feed]
    parent_idx = [idx.get((feed[i].get("source_entry_ids") or [None])[0]) for i in range(n)]
    children = [[] for _ in range(n)]
    for i in range(n):
        if parent_idx[i] is not None:
            children[parent_idx[i]].append(i)
    spawned = [0] * n
    remaining = [len(children[i]) for i in range(n)]
    print(f"family grid: {n} tiles, {sum(1 for p in parent_idx if p is not None)} branched")

    from multiprocessing import Pool
    with Pool(args.jobs, initializer=scr._init,
              initargs=(gdir, str(args.run), args.cell, args.morph_steps)) as pool:
        stacks = pool.map(scr._morph_stack, specs)
    print("morphs rendered; simulating reorganizing grid...")

    C, Rr, d, g = args.cols, args.rows, args.cell, args.gap
    step = d + g
    S = C * Rr
    W = C * step + g
    H = Rr * step + g
    W += W % 2; H += H % 2
    center = (S // 2)

    def sxy(slot):
        r, c = divmod(slot, C)
        return g + c * step, g + r * step

    def dist2(a, b):
        ax, ay = sxy(a); bx, by = sxy(b)
        return (ax - bx) ** 2 + (ay - by) ** 2

    occ = {}                       # slot -> tile idx
    placed = {}                    # tile idx -> slot
    waves = []                     # (bg: slot->tile, descenders: [(tile, src_slot_or_None, tgt_slot)])

    def next_child_idx(t):
        return children[t][spawned[t]] if spawned[t] < len(children[t]) else 10**9

    def pick_target(src_slot, protected):
        free = [s for s in range(S) if s not in occ]
        anchor = src_slot if src_slot is not None else center
        if free:
            return min(free, key=lambda s: dist2(s, anchor))
        done = [s for s in occ if remaining[occ[s]] == 0 and occ[s] not in protected]
        if done:
            return min(done, key=lambda s: dist2(s, anchor))
        cand = [s for s in occ if occ[s] not in protected]               # forced: evict least-imminent
        return min(cand, key=lambda s: (-next_child_idx(occ[s]), dist2(s, anchor)))

    for w0 in range(0, n, C):
        wave = list(range(w0, min(w0 + C, n)))
        protected = set(p for p in (parent_idx[i] for i in wave) if p is not None)
        # 1) re-introduce any needed parent that isn't currently placed (keeps it visible)
        for p in list(protected):
            if p not in placed:
                tgt = pick_target(None, protected)
                if tgt in occ:
                    del placed[occ[tgt]]
                occ[tgt] = p; placed[p] = tgt
        bg = dict(occ)             # background for this wave (incl. soon-to-be-evicted tiles)
        descenders = []
        for i in wave:
            src = placed[parent_idx[i]] if parent_idx[i] is not None else None
            tgt = pick_target(src, protected)
            if tgt in occ:
                del placed[occ[tgt]]
            occ[tgt] = i; placed[i] = tgt
            protected.add(i)        # don't let a later sibling evict me this wave
            if parent_idx[i] is not None:
                spawned[parent_idx[i]] += 1
                remaining[parent_idx[i]] -= 1
            descenders.append((i, src, tgt))
        waves.append((bg, descenders))

    # ---- render -------------------------------------------------------------
    DESC, MORPH, PAUSE = args.descent_frames, args.morph_frames, args.pause_frames
    args.out.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{W}x{H}", "-r", str(args.fps), "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(args.out)],
        stdin=subprocess.PIPE,
    )

    def blit(canvas, tile, x, y, outline=False):
        h, w = tile.shape[:2]
        canvas[y:y + h, x:x + w] = tile
        if outline:
            cv2.rectangle(canvas, (x, y), (x + w - 1, y + h - 1), ORANGE, 1)

    desc_set = set()
    for bg, descenders in waves:
        moving = {tile for tile, _, _ in descenders}
        for ph in range(DESC + MORPH + PAUSE):
            canvas = np.full((H, W, 3), 255, np.uint8)
            # background: settled tiles not currently flying in (final form)
            for slot, t in bg.items():
                if t in moving:
                    continue
                x, y = sxy(slot)
                blit(canvas, stacks[t][-1], x, y)
            # descenders
            for tile, src, tgt in descenders:
                tx, ty = sxy(tgt)
                if ph < DESC and src is not None:                 # flying in (clone of source)
                    t = ph / DESC; t = t * t * (3 - 2 * t)
                    sx, sy = sxy(src)
                    x = int(round(sx + (tx - sx) * t)); y = int(round(sy + (ty - sy) * t))
                    blit(canvas, stacks[tile][0], x, y, outline=args.outline)
                else:                                             # arrived: morph in place
                    prog = (ph - DESC) / MORPH
                    prog = 0.0 if prog < 0 else (1.0 if prog > 1 else prog)
                    k = int(round(prog * (len(stacks[tile]) - 1)))
                    blit(canvas, stacks[tile][k], tx, ty, outline=args.outline and prog < 1.0)
            proc.stdin.write(canvas.tobytes())
    # hold final
    canvas = np.full((H, W, 3), 255, np.uint8)
    last_occ = waves[-1][0].copy()
    for tile, src, tgt in waves[-1][1]:
        last_occ[tgt] = tile
    for slot, t in last_occ.items():
        x, y = sxy(slot); blit(canvas, stacks[t][-1], x, y)
    for _ in range(args.hold):
        proc.stdin.write(canvas.tobytes())
    proc.stdin.close()
    proc.wait()
    nf = sum(DESC + MORPH + PAUSE for _ in waves) + args.hold
    print(f"wrote {args.out} ({nf} frames, {nf/args.fps:.1f}s)")


if __name__ == "__main__":
    main()
