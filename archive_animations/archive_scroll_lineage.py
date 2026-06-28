#!/usr/bin/env python3
"""Descent scrolling feed with the lineage version's continuous-scroll timing.

Images are laid out in strict feed order (``col = i % C``, ``row = i // C``) and
scroll upward. Parents are **never reorganized** -- they simply scroll up and off
the top. A branched child launches from wherever its parent currently sits on
screen and **morphs out of it as it falls**.

Steady-phase per row N: one event of ``DESC`` frames.

  - **Continuous scroll** of 1 row over the whole event. Every pre-existing tile
    interpolates linearly from its pre-event row R to R - 1.
  - **Descent + morph (combined)** -- the new clones descend from their source
    into the bottom row while CPPN-morphing from stack[0] (the source/parent
    form) to stack[-1] (their published form), arriving fully morphed exactly
    as the scroll completes one full row. Morph and motion share one window, so
    a row is "born" in a single motion rather than falling first and morphing
    afterwards.

A branched child launches from its parent's on-screen position. Once the parent
has **scrolled out of view**, the child launches from the **invisible row just
above the top** (screen-row -1) at the parent's column, rather than tracing a
long diagonal back to the parent's distant off-screen row. Roots (no archive
parent / session founders) have nothing to morph out of -- morphing them from a
random-initial CPPN just reads as growing out of a flat colour -- so they simply
**appear at their published form** (rising one row from below the bottom in the
steady feed), and only branched children morph.

Fill phase (rows 0..V-1): sequential per row, no scroll; branched tiles descend
from their visible parent while morphing, roots appear at their published form.
The all-root opening row is placed instantly (no dwell), so the feed starts
right as the grid clones in and the first branched row begins morphing.

``--reveal`` picks how new rows are born:
  * ``morph`` (default) -- branched children descend from their parent and morph
    in (the behaviour described above).
  * ``pop``           -- every tile behaves like a root: it simply rises one row
    from below the bottom in its published form, no descent and no morph. The
    smooth continuous scroll (one row per ``--descent-frames``) is unchanged --
    only the per-tile birth is instant. Use this when the feed scrolls so fast
    that the morph is an illegible blur. Both paths are kept on purpose; ``pop``
    also skips the (expensive) CPPN morph precompute entirely.

Reuses morph precompute (workers + config) from ``archive_scroll``.

Usage:
    python archive_animations/archive_scroll_lineage.py \
        --run sweep_logs/sweep/th0_..._s4 \
        --out archive_animations/out/archive_scroll_lineage.mp4 \
        --n 200 --cols 9 --rows-visible 7 --cell 88 --morph-steps 14 --jobs 10 --reveal pop
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import archive_animations.archive_scroll as scr  # noqa: E402


def smoothstep(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n", type=int, default=200, help="publications to feed (<= genomes saved)")
    ap.add_argument("--cols", type=int, default=9)
    ap.add_argument("--cell", type=int, default=88, help="tile size px (also CPPN render res)")
    ap.add_argument("--gap", type=int, default=6)
    ap.add_argument("--rows-visible", type=int, default=7)
    ap.add_argument("--morph-steps", type=int, default=14)
    ap.add_argument("--morph-frames", type=int, default=22,
                    help="DEPRECATED / unused: the morph now rides the descent over --descent-frames "
                         "(kept so existing commands still parse)")
    ap.add_argument("--descent-frames", type=int, default=22,
                    help="frames per row event: the new clones descend AND morph over these frames, "
                         "and the whole grid scrolls up one row in the same window (they're synced so "
                         "clones land fully morphed as the bottom row clears). Bump it if the combined "
                         "descent+morph reads too fast")
    ap.add_argument("--hold", type=int, default=36)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--reveal", choices=["morph", "pop"], default="morph",
                    help="morph = branched tiles descend+morph from parent (cinematic); "
                         "pop = every tile rises from below in published form, no descent/morph "
                         "(for fast feeds). Scroll cadence is identical either way.")
    args = ap.parse_args()
    POP = args.reveal == "pop"

    gdir = str(args.run / "archive" / "genomes")
    idir_feed = str(args.run / "archive" / "images")
    ents = sorted(json.loads((args.run / "archive" / "archive_metadata.json").read_text())["entries"],
                  key=lambda e: e.get("added_at", ""))
    feed = []
    for e in ents:
        # pop mode renders from the published images and never touches genomes,
        # so gate the feed on image presence; morph modes need the genome on disk.
        have = (os.path.exists(os.path.join(idir_feed, e["id"] + ".png")) if POP
                else os.path.exists(os.path.join(gdir, e["id"] + ".pkl")))
        if have:
            feed.append(e)
        if len(feed) >= args.n:
            break
    n = len(feed)

    def spec_of(e):
        # branched: morph from the parent archive image when its genome is on disk.
        # otherwise static: a session founder/root has no archive parent to morph
        # out of -- morphing it from a random-initial CPPN just reads as growing
        # out of a flat colour, so roots simply appear at their published form.
        s = e.get("source_entry_ids") or []
        p = s[0] if s else None
        if p and os.path.exists(os.path.join(gdir, p + ".pkl")):
            return (e["id"], "archive", p, bool(e.get("color_enabled")))
        return (e["id"], "static", None, bool(e.get("color_enabled")))

    specs = [spec_of(e) for e in feed]
    id2idx = {feed[i]["id"]: i for i in range(n)}
    parent_idx = [id2idx.get(specs[i][2]) if specs[i][1] == "archive" else None for i in range(n)]

    nbr = sum(1 for s in specs if s[1] == "archive")
    nrt = n - nbr

    if POP:
        # pop mode never shows a morph -- only each tile's final published form --
        # so skip the CPPN morph precompute and load the already-rendered archive
        # image for every publication. stacks[i] is a 1-frame list so the shared
        # render code (which reads stacks[tile][-1]) is unchanged.
        from PIL import Image as _Im
        idir = args.run / "archive" / "images"
        def _final(eid):
            im = _Im.open(idir / f"{eid}.png").convert("RGB").resize(
                (args.cell, args.cell), _Im.LANCZOS)
            return np.asarray(im)
        print(f"feeding {n} publications: pop mode (no morph), loading final images...")
        stacks = [[_final(e["id"])] for e in feed]
        print("final images loaded; planning...")
    else:
        print(f"feeding {n} publications: {nbr} branched-morphs, {nrt} roots (appear, no morph); rendering...")
        from multiprocessing import Pool
        with Pool(args.jobs, initializer=scr._init,
                  initargs=(gdir, str(args.run), args.cell, args.morph_steps)) as pool:
            stacks = pool.map(scr._morph_stack, specs)
        print("morphs rendered; planning...")

    C, d, g = args.cols, args.cell, args.gap
    V = args.rows_visible
    step = d + g
    W = C * step + g
    H = V * step + g
    W += W % 2; H += H % 2
    DESC = args.descent_frames

    # ---- planning ----
    # Strict feed-order layout: tiles fill left-to-right, top-to-bottom, C per
    # row (col = i % C, row = i // C). No reorganization -- parents just scroll
    # up and off the top. A child's descent source is its parent index; the
    # renderer descends from the parent's current on-screen position if it's
    # still visible, else from just below the bottom row (root-like).
    def row_slots(tiles: list[int]) -> list[dict]:
        return [{"tile": t, "src": parent_idx[t], "col": c} for c, t in enumerate(tiles)]

    events: list[tuple] = []
    idx = 0
    for r in range(V):
        if idx >= n:
            break
        tiles = list(range(idx, min(idx + C, n)))
        idx += C
        events.append(("fill", r, row_slots(tiles)))
    while idx < n:
        tiles = list(range(idx, min(idx + C, n)))
        idx += C
        events.append(("steady", row_slots(tiles)))

    n_fill = sum(1 for ev in events if ev[0] == "fill")
    n_steady = sum(1 for ev in events if ev[0] == "steady")

    # ---- rendering ----
    tile_screen_pos: dict[int, tuple[float, float]] = {}
    overlays: dict[int, dict] = {}

    def col_x(c):
        return g + c * step

    def row_y(r):
        return g + r * step

    def blit(canvas, tile, x, y):
        h, w = tile.shape[:2]
        ya, yb = max(0, y), min(H, y + h)
        xa, xb = max(0, x), min(W, x + w)
        if ya >= yb or xa >= xb:
            return
        canvas[ya:yb, xa:xb] = tile[ya - y:yb - y, xa - x:xb - x]

    def render_frame():
        canvas = np.full((H, W, 3), 255, np.uint8)
        for tile, (fr, fc) in tile_screen_pos.items():
            if tile in overlays:
                continue
            blit(canvas, stacks[tile][-1], int(round(col_x(fc))), int(round(row_y(fr))))
        for tile, info in overlays.items():
            blit(canvas, stacks[tile][info['stack_idx']],
                 int(round(info['x'])), int(round(info['y'])))
        return canvas

    args.out.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{W}x{H}", "-r", str(args.fps), "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(args.out)],
        stdin=subprocess.PIPE,
    )
    emitted = 0

    def emit():
        nonlocal emitted
        proc.stdin.write(render_frame().tobytes())
        emitted += 1

    def emit_fill_row(r_target, slots):
        """Fill a row over DESC frames: branched tiles descend from their visible
        parent while morphing from stack[0] -> stack[-1]; root tiles just appear
        at their published form (no morph). No scroll during fill.

        A row with nothing to animate -- the all-root opening row -- is placed
        instantly with no dwell, so the feed starts right as the grid clones in
        and the first branched row begins morphing rather than holding on a
        static opening."""
        if POP:
            # pop: the row simply appears at its published form in place. Spend DESC
            # frames (a brief dwell, no scroll in the fill phase) so the fill cadence
            # matches the morph version.
            for s in slots:
                tile_screen_pos[s["tile"]] = (float(r_target), float(s["col"]))
            for _ in range(DESC):
                emit()
            return
        pre_state = dict(tile_screen_pos)
        animating = any(s["src"] is not None and s["src"] in pre_state for s in slots)
        if not animating:
            for s in slots:
                tile_screen_pos[s["tile"]] = (float(r_target), float(s["col"]))
            return
        for f in range(DESC):
            t = (f + 1) / DESC
            td = smoothstep(t)
            overlays.clear()
            for s in slots:
                tile = s["tile"]; src = s["src"]; dst_c = s["col"]
                final = len(stacks[tile]) - 1
                if src is not None and src in pre_state:
                    si = min(final, int(round(t * final)))
                    sr0, sc0 = pre_state[src]
                    cr = sr0 + (r_target - sr0) * td
                    cc = sc0 + (dst_c - sc0) * td
                    overlays[tile] = {'x': col_x(cc), 'y': row_y(cr), 'stack_idx': si}
                else:
                    overlays[tile] = {'x': col_x(dst_c), 'y': row_y(r_target), 'stack_idx': final}
            emit()
        for s in slots:
            tile_screen_pos[s["tile"]] = (float(r_target), float(s["col"]))
        overlays.clear()

    def emit_steady_event(slots, scrolled):
        """Single continuous event over DESC frames: the whole grid scrolls up by
        one row while the new clones descend into the bottom row AND morph from
        stack[0] (their source/parent form) to stack[-1] (their published form),
        landing fully morphed exactly as the scroll completes one full row.

        `scrolled` is the number of rows committed off the top before this event.
        A branched child launches from its parent's on-screen row (screen-row =
        parent_abs_row - scrolled); once the parent has scrolled out of view that
        row is negative, so the launch is clamped to screen-row -1 -- the invisible
        row just above the top -- at the parent's column, giving a short fall from
        above the top rather than a long diagonal back to the distant parent.
        Roots (src None) rise from just below the bottom row."""
        total = DESC
        if total <= 0:
            return

        pre_state = dict(tile_screen_pos)

        # Per-tile final position at end of event: every tile scrolls up by 1.
        targets: dict[int, tuple[float, float]] = {}
        for tile, (fr0, fc0) in pre_state.items():
            targets[tile] = (fr0 - 1.0, fc0)

        desc_sources: dict[int, tuple[float, float]] = {}
        for s in slots:
            tile = s["tile"]; src = s["src"]; dst_c = s["col"]
            if src is not None and not POP:
                src_row = (src // C) - scrolled            # parent's screen row at event start
                if src_row < 0:                            # parent out of view -> launch from just above the top
                    src_row = -1.0
                desc_sources[tile] = (float(src_row), float(src % C))
            else:
                # root, or pop mode: rise one row from just below the bottom in
                # published form (no parent descent, no morph).
                desc_sources[tile] = (float(V), float(dst_c))

        for f in range(total):
            t = (f + 1) / total
            for tile, (fr0, fc0) in pre_state.items():
                tr, tc = targets[tile]
                tile_screen_pos[tile] = (fr0 + (tr - fr0) * t, fc0 + (tc - fc0) * t)
            overlays.clear()
            t_d = smoothstep(t)
            for s in slots:
                tile = s["tile"]; dst_c = s["col"]
                src_r, src_c = desc_sources[tile]
                # A straight one-row rise (roots / pop tiles, sourced from just
                # below the bottom in the same column) must track the reel exactly:
                # move it with the SAME linear cadence as the scroll. Easing it
                # (smoothstep) makes the incoming bottom row accelerate/decelerate
                # against the linearly-scrolling rows above, which reads as the
                # bottom tiles jiggling. Only the diagonal morph descent (a branched
                # child falling in from a distant parent) keeps the cinematic ease.
                straight = (src_r >= V - 1e-6) and (abs(src_c - dst_c) < 1e-6)
                te = t if straight else t_d
                cr = src_r + (V - 1 - src_r) * te
                cc = src_c + (dst_c - src_c) * te
                si = min(len(stacks[tile]) - 1, int(round(t * (len(stacks[tile]) - 1))))
                overlays[tile] = {'x': col_x(cc), 'y': row_y(cr), 'stack_idx': si}
            emit()

        # commit
        for tile, target in targets.items():
            tile_screen_pos[tile] = target
        for tile in list(tile_screen_pos.keys()):
            if tile_screen_pos[tile][0] < -0.5:
                del tile_screen_pos[tile]
        for s in slots:
            tile_screen_pos[s["tile"]] = (float(V - 1), float(s["col"]))
        overlays.clear()

    scrolled = 0
    for ev in events:
        if ev[0] == "fill":
            _, r_target, slots = ev
            emit_fill_row(r_target, slots)
        else:
            _, slots = ev
            emit_steady_event(slots, scrolled)
            scrolled += 1

    for _ in range(args.hold):
        emit()
    proc.stdin.close()
    proc.wait()
    print(f"planned {len(events)} events ({n_fill} fill, {n_steady} steady)")
    print(f"wrote {args.out} ({emitted} frames, {emitted/args.fps:.1f}s, {W}x{H})")


if __name__ == "__main__":
    main()
