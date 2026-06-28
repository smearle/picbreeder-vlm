#!/usr/bin/env python3
"""Running leaderboard of the archive's standout published images.

A fullness bar across the top fills as publications come in (added_at order),
each newly-published image dropping a tiny thumbnail onto a growing stack along
the bar (newest at the leading edge). Below, a top-K board holds the standout
images so far. When a newcomer cracks the board it slides into rank order,
bumping the loser off the end.

Two boards (``--board``), each reproducing the exact ranking the agents' own
sampler used in ``archive_manager.sample_branching_entries``:

  rated     the "Top Rated" shelf: rank by mean of ``vlm_ratings`` (descending),
            ties broken by the number of ratings. This is the *plain* running
            average the sampler used -- NOT a smoothed/Bayesian one (pass
            ``--score bayes`` to shrink toward the global mean with a prior
            count ``C``, which de-ranks one-off 5/5s, purely for inspection).
  branched  the "Most Branched" shelf: rank by the number of direct archive
            children an image has accrued *so far* (descending), ties broken by
            mean rating then rating count. Children are counted live as each
            child is published (``source_entry_ids[0]`` is the unique branch
            parent), so this board is faithful as a time series.

Rated board, two fidelities:
  default      per-rating timestamps are not stored in the archive metadata, so
               each image's score uses its *final* accumulated ``vlm_ratings``
               from the moment it is published. Board membership changes as new
               images appear, not as old ones slowly accrue ratings.
  --rating-timeline FILE
               TRUE chronological means. The agents logged a branching snapshot
               each session (``agent_*/logs/branching_snapshot.json`` inside the
               per-session zips, on torch scratch) recording the (avg, count)
               every sampled image carried *at that timestamp*. Extract those to
               a JSONL of ``{ts, obs:[[id, avg, count], ...]}`` and pass it here;
               each image's score then steps through exactly the values the
               sampler used, keyed by the archive size at each snapshot's time.
The branched board needs no timeline -- child publications are timestamped, so
its live child counts are exact.

Two layouts:
  --layout list  vertical ranked rows (thumbnail + title + rating/branch bar)
  --layout grid  packed square mosaic, reading-order (top-left = best), with the
                 image title set underneath each cell -- styled after the
                 system-overview tikz figure (Latin Modern serif, gold #DAA520).

Usage:
    python archive_animations/leaderboard.py \
        --run sweep_logs/sweep/th10_..._s8 \
        --out archive_animations/out/leaderboard_rated_th10s8.mp4 \
        --board rated --layout list --n 400 --k 10
    python archive_animations/leaderboard.py \
        --run sweep_logs/sweep/th10_..._s8 \
        --out archive_animations/out/leaderboard_branched_th10s8.mp4 \
        --board branched --layout list --k 10 --eval-stride 4 --quiet-stride 4
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ORANGE = (255, 108, 0)
INK = (28, 28, 28)
MUTED = (140, 140, 140)
BG = (255, 255, 255)
ROW_ALT = (246, 246, 246)
BAR_BG = (232, 232, 232)
TRACK = (228, 228, 228)
GOLD = (218, 165, 32)  # #DAA520, matches system_overview ratingGold

# Latin Modern (the system_overview tikz default serif), staged locally.
FONT_DIR = Path(__file__).resolve().parent / "assets" / "fonts"
LM_REG = str(FONT_DIR / "lmroman10-regular.otf")
LM_BOLD = str(FONT_DIR / "lmroman10-bold.otf")
LM_ITAL = str(FONT_DIR / "lmroman10-italic.otf")


def smoothstep(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)


def lerp(c0, c1, t):
    """Blend colour c0->c1 by t (used to fade tiles in/out over white)."""
    return tuple(int(round(a + (b - a) * t)) for a, b in zip(c0, c1))


def star_points(cx, cy, r):
    pts = []
    for k in range(10):
        ang = -math.pi / 2 + k * math.pi / 5
        rad = r if k % 2 == 0 else r * 0.42
        pts.append((cx + rad * math.cos(ang), cy + rad * math.sin(ang)))
    return pts


STAR_GREY = (201, 201, 201)


def draw_star_rating(img, x, y, w, h, frac, n=5):
    """A row of ``n`` stars filled gold left-to-right to fraction ``frac``
    (partial last star), grey beyond -- in the style of the system_overview
    tikz figure's gold/grey star ratings."""
    w, h = int(w), int(h)
    d = ImageDraw.Draw(img)
    sw = w / n
    r = min(sw, h) / 2 * 0.94
    for i in range(n):
        d.polygon(star_points(x + sw * (i + 0.5), y + h / 2, r), fill=STAR_GREY)
    cut = int(round(max(0.0, min(1.0, frac)) * w))
    if cut <= 0:
        return
    gold = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    gd = ImageDraw.Draw(gold)
    for i in range(n):
        gd.polygon(star_points(sw * (i + 0.5), h / 2, r), fill=GOLD + (255,))
    crop = gold.crop((0, 0, cut, h))
    img.paste(crop, (int(x), int(y)), crop)


def draw_count_bar(img, x, y, w, h, frac):
    """A rounded gold progress bar filled to fraction ``frac`` -- the branched
    board's analogue of the star row (branch count relative to the leader)."""
    d = ImageDraw.Draw(img)
    rad = h // 2
    d.rounded_rectangle([x, y, x + w, y + h], radius=rad, fill=TRACK)
    fw = int(round(max(0.0, min(1.0, frac)) * w))
    if fw >= h:
        d.rounded_rectangle([x, y, x + fw, y + h], radius=rad, fill=GOLD)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--board", choices=["rated", "branched"], default="rated",
                    help="rated = Top-Rated shelf (mean vlm_rating); branched = Most-Branched shelf (live child count)")
    ap.add_argument("--score", choices=["mean", "bayes"], default="mean",
                    help="rated board only: 'mean' = plain running average the sampler used; 'bayes' = shrink toward global mean (inspection)")
    ap.add_argument("--n", type=int, default=None, help="publications to feed (default: 400 for rated, full archive for branched)")
    ap.add_argument("--layout", choices=["list", "grid"], default="list")
    ap.add_argument("--k", type=int, default=10, help="leaderboard slots (grid: total cells)")
    ap.add_argument("--cols", type=int, default=5, help="grid columns (grid layout only)")
    ap.add_argument("--cell", type=int, default=80, help="thumbnail px")
    ap.add_argument("--gap", type=int, default=4, help="grid gap px (grid layout only)")
    ap.add_argument("--prior", type=float, default=10.0, help="Bayesian prior count C (--score bayes only)")
    ap.add_argument("--min-ratings", type=int, default=1, help="min ratings to be eligible (rated board)")
    ap.add_argument("--rating-timeline", type=Path, default=None,
                    help="rated board: JSONL(.gz) of agent branching snapshots ({ts, obs:[[id,avg,cnt],...]}) "
                         "to drive TRUE chronological means (what agents actually saw) instead of final ratings")
    ap.add_argument("--enter-frames", type=int, default=13, help="frames per board transition")
    ap.add_argument("--tick-frames", type=int, default=1, help="frames per quiet publication")
    ap.add_argument("--quiet-stride", type=int, default=1, help="publications advanced per quiet tick frame")
    ap.add_argument("--eval-stride", type=int, default=1, help="re-evaluate the board every N publications (branched: raise to tame churn)")
    ap.add_argument("--warmup-stride", type=int, default=8,
                    help="while the board is still EMPTY (rated board: no ratings until the archive reaches 100), advance this many publications per emitted frame to zip through the pre-rating prefix")
    ap.add_argument("--hold", type=int, default=44)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--emit-board-timeline", type=Path, default=None,
                    help="also write a JSON step-function of board membership "
                         "({count, board:[[id,value],...]} per change) so a companion "
                         "animation (e.g. archive_tree.py) can emphasise the same images")
    args = ap.parse_args()

    BRANCHED = args.board == "branched"

    gdir = args.run / "archive" / "images"
    ents = sorted(json.loads((args.run / "archive" / "archive_metadata.json").read_text())["entries"],
                  key=lambda e: e.get("added_at", ""))
    N = len(ents) if args.n is None else min(args.n, len(ents))
    feed = ents[:N]

    allv = [v for e in ents if e.get("vlm_ratings") for v in e["vlm_ratings"]]
    m = (sum(allv) / len(allv)) if allv else 0.0
    C = args.prior

    by_id = {e["id"]: e for e in ents}

    # Optional TRUE chronological ratings: replay the (avg, count) each image
    # carried in the agents' logged branching snapshots, keyed by the archive
    # size at each snapshot's timestamp -- so the board evolves exactly as the
    # ratings the sampler used evolved, not from each image's final mean.
    TIMELINE = (not BRANCHED) and args.rating_timeline is not None
    events: list[tuple] = []   # (archive_size_at_snapshot, eid, avg, count), in time order within a size
    if TIMELINE:
        import bisect
        import gzip as _gz
        added_keys = [e.get("added_at", "") for e in ents]            # ascending (ents sorted by added_at)
        opn = _gz.open if str(args.rating_timeline).endswith(".gz") else open
        rows = [json.loads(line) for line in opn(args.rating_timeline, "rt")]
        rows.sort(key=lambda r: r["ts"])
        for r in rows:
            s = bisect.bisect_right(added_keys, r["ts"])              # publications that exist at snapshot time
            for ob in r.get("obs", []):
                eid, avg = ob[0], ob[1]
                if avg is None or not eid:
                    continue
                events.append((s, eid, float(avg), int(ob[2] or 0)))
        events.sort(key=lambda x: x[0])                               # stable -> preserves time order within a size

    def final_avg(e):
        r = e.get("vlm_ratings") or []
        return (sum(r) / len(r)) if r else float("-inf")

    def rated_val(e):
        """The number the Top-Rated sampler ranked on (plain mean), or a shrunk
        mean under --score bayes. None if too few ratings to be eligible."""
        r = e.get("vlm_ratings") or []
        if len(r) < args.min_ratings:
            return None
        if args.score == "bayes":
            return (C * m + sum(r)) / (C + len(r))
        return sum(r) / len(r)

    GRID = args.layout == "grid"
    K = args.k
    cell = args.cell
    margin = 18

    # fullness bar header
    bar_h = 30
    header_h = bar_h + 22

    frank = ImageFont.truetype(LM_BOLD, 30)
    ftitle = ImageFont.truetype(LM_BOLD, 19)
    fscore = ImageFont.truetype(LM_BOLD, 20)
    fsmall = ImageFont.truetype(LM_REG, 13)
    fbadge = ImageFont.truetype(LM_BOLD, max(13, cell // 6))
    cap_size = max(11, cell // 9)
    fcap = ImageFont.truetype(LM_ITAL, cap_size)
    cap_h = cap_size + 8  # caption strip under each grid thumbnail

    if GRID:
        cols = args.cols
        rows = math.ceil(K / cols)
        gap = args.gap
        block_h = cell + 3 + cap_h            # thumbnail + title underneath
        W = margin * 2 + cols * (cell + gap) - gap
        H = header_h + rows * (block_h + gap) - gap + margin
    else:
        row_h = cell + 14
        W = margin + 46 + cell + 14 + 372 + margin
        H = header_h + K * row_h + margin
    W += W % 2; H += H % 2

    thumbs: dict[tuple, Image.Image] = {}

    def thumb(eid, size):
        key = (eid, size)
        if key not in thumbs:
            try:
                im = Image.open(gdir / f"{eid}.png").convert("RGB").resize((size, size), Image.LANCZOS)
            except FileNotFoundError:
                im = Image.new("RGB", (size, size), (240, 240, 240))
            thumbs[key] = im
        return thumbs[key]

    feed_ids = [e["id"] for e in feed]

    def slot_xy(rank):
        if GRID:
            c, r = rank % args.cols, rank // args.cols
            return (margin + c * (cell + args.gap), header_h + r * ((cell + 3 + cap_h) + args.gap))
        return (margin, header_h + rank * (cell + 14))

    def trunc(d, text, font, maxw):
        if d.textlength(text, font=font) <= maxw:
            return text
        while text and d.textlength(text + "…", font=font) > maxw:
            text = text[:-1]
        return (text.rstrip() + "…") if text else ""

    def draw_header_bar(d, canvas, count):
        """Fullness bar: a growing stack of tiny published-image thumbnails along
        a track, newest at the leading edge."""
        bx0, bx1 = margin, W - margin
        by0 = 12
        bw = bx1 - bx0
        d.rounded_rectangle([bx0, by0, bx1, by0 + bar_h], radius=4, fill=TRACK)
        if count <= 0:
            return
        card = bar_h
        span = max(1, bw - card)
        last = min(count, N)
        for i in range(last):
            x = bx0 + int(round(span * (i / max(1, N - 1))))
            canvas.paste(thumb(feed_ids[i], card), (x, by0))
        xl = bx0 + int(round(span * ((last - 1) / max(1, N - 1))))
        d.rectangle([xl, by0, xl + card - 1, by0 + card - 1], outline=GOLD, width=2)

    def draw_tile(d, canvas, eid, x, y, rank, val, ref):
        e = by_id[eid]
        if GRID:
            canvas.paste(thumb(eid, cell), (int(x), int(y)))
            d.rectangle([x, y, x + cell - 1, y + cell - 1],
                        outline=GOLD if rank == 0 else (222, 222, 222), width=3 if rank == 0 else 1)
            bw = cell // 4
            d.rectangle([x, y, x + bw, y + bw], fill=(GOLD if rank == 0 else (0, 0, 0)))
            rn = str(rank + 1)
            tw = d.textlength(rn, font=fbadge)
            d.text((x + bw / 2 - tw / 2, y + bw / 2 - fbadge.size * 0.62), rn, font=fbadge, fill=(255, 255, 255))
            title = trunc(d, e.get("title") or e["id"], fcap, cell - 2)
            tw = d.textlength(title, font=fcap)
            d.text((x + cell / 2 - tw / 2, y + cell + 3), title, font=fcap, fill=INK)
            return
        # list row
        row_h = cell + 14
        band = (255, 248, 232) if rank == 0 else (ROW_ALT if rank % 2 else BG)
        d.rectangle([margin, y, W - margin, y + row_h - 2], fill=band)
        rnum = str(rank + 1)
        rw = d.textlength(rnum, font=frank)
        d.text((margin + 23 - rw / 2, y + row_h / 2 - 18), rnum, font=frank, fill=GOLD if rank == 0 else MUTED)
        tx = margin + 46
        ty = y + (row_h - cell) // 2
        canvas.paste(thumb(eid, cell), (int(tx), int(ty)))
        d.rectangle([tx, ty, tx + cell - 1, ty + cell - 1], outline=(220, 220, 220))
        txt = margin + 46 + cell + 14
        title = trunc(d, e.get("title") or e["id"], ftitle, 372 - 8)
        d.text((txt, ty + 4), title, font=ftitle, fill=INK)
        meter_w, meter_h = 130, 22
        sy = ty + 32
        if BRANCHED:
            children = int(val)
            draw_count_bar(canvas, txt, sy, meter_w, meter_h, (children / ref) if ref else 0.0)
            d.text((txt + meter_w + 12, sy - 1), str(children), font=fscore, fill=INK)
        else:
            draw_star_rating(canvas, txt, sy, meter_w, meter_h, val / 5.0)
            d.text((txt + meter_w + 12, sy + 1), f"{val:.2f}", font=fscore, fill=INK)
            n = cntmap.get(eid, len(e.get("vlm_ratings") or []))
            d.text((txt, sy + meter_h + 5), f"{n} ratings", font=fsmall, fill=MUTED)

    def render(placed, count, ref):
        img = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(img)
        draw_header_bar(d, img, count)
        for eid, x, y, rank, val in placed:
            if y > H or y < header_h - cell - cap_h - 16:
                continue
            draw_tile(d, img, eid, x, y, max(0, min(K - 1, rank)), val, ref)
        return np.asarray(img)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{W}x{H}", "-r", str(args.fps), "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(args.out)],
        stdin=subprocess.PIPE,
    )
    emitted = 0

    def emit(frame):
        nonlocal emitted
        proc.stdin.write(frame.tobytes())
        emitted += 1

    # --- live state -------------------------------------------------------
    children_now: dict[str, int] = {}   # running direct-child count (branched board)
    valmap: dict[str, float] = {}       # current ranking value per eligible eid
    cntmap: dict[str, int] = {}         # current rating count per eid (display + tiebreak)
    cur: list[str] = []
    n_changes = 0
    board_timeline: list[dict] = []   # step-function of board membership for companion anims
    ingested = 0
    ev_ptr = 0

    def ingest_upto(j):
        """Account for every publication with index <= j (so child counts and
        the rated pool stay correct even when we skip frames during quiet runs)."""
        nonlocal ingested, ev_ptr
        while ingested <= j:
            e = feed[ingested]
            if BRANCHED:
                src = e.get("source_entry_ids") or []
                if src:
                    p = src[0]
                    children_now[p] = children_now.get(p, 0) + 1
                    if p in by_id:
                        valmap[p] = children_now[p]
            elif not TIMELINE:
                v = rated_val(e)
                if v is not None:
                    valmap[e["id"]] = v
                    cntmap[e["id"]] = len(e.get("vlm_ratings") or [])
            ingested += 1
        if TIMELINE:
            count = j + 1
            while ev_ptr < len(events) and events[ev_ptr][0] <= count:
                _, eid, avg, cnt = events[ev_ptr]
                if cnt >= args.min_ratings:
                    valmap[eid] = avg
                    cntmap[eid] = cnt
                ev_ptr += 1

    def topk():
        if BRANCHED:
            cand = [eid for eid, c in valmap.items() if c > 0]
            cand.sort(key=lambda eid: (valmap[eid], final_avg(by_id[eid]),
                                       len(by_id[eid].get("vlm_ratings") or [])), reverse=True)
        else:
            cand = list(valmap.keys())
            cand.sort(key=lambda eid: (valmap[eid], cntmap.get(eid, 0)), reverse=True)
        return cand[:K]

    def ref_for(order):
        """Bar normaliser for the branched board: the leader's child count."""
        if not BRANCHED or not order:
            return 1.0
        return max(1.0, max(valmap.get(eid, 0) for eid in order))

    def placed_static(order, count):
        ref = ref_for(order)
        return render([(eid, *slot_xy(r), r, valmap.get(eid, 0.0)) for r, eid in enumerate(order)], count, ref)

    # Ingest every publication (i advances by 1, so child counts / the rated pool
    # stay exact), but only RE-RANK the board every ``eval-stride`` publications
    # and only emit a quiet "bar advancing" frame every ``quiet-stride`` -- the two
    # cadences are independent of i's parity, so neither can skip the other.
    i = 0
    since_eval = 0
    qstride = max(1, args.quiet_stride)
    while i < N:
        ingest_upto(i)
        count = i + 1
        since_eval += 1
        do_eval = since_eval >= args.eval_stride or i == N - 1
        new = topk() if do_eval else cur
        if do_eval:
            since_eval = 0
        if new != cur:
            old_rank = {eid: r for r, eid in enumerate(cur)}
            new_rank = {eid: r for r, eid in enumerate(new)}
            union = list(dict.fromkeys(cur + new))
            ref = ref_for(new)
            for f in range(args.enter_frames):
                t = smoothstep((f + 1) / args.enter_frames)
                placed = []
                for eid in union:
                    r0 = old_rank.get(eid, K)
                    r1 = new_rank.get(eid, K)
                    x0, y0 = slot_xy(r0); x1, y1 = slot_xy(r1)
                    placed.append((eid, x0 + (x1 - x0) * t, y0 + (y1 - y0) * t,
                                   r1 if eid in new_rank else r0, valmap.get(eid, 0.0)))
                placed.sort(key=lambda o: -o[3])
                emit(render(placed, count, ref))
            cur = new
            n_changes += 1
            board_timeline.append({"count": count,
                                   "board": [[eid, int(valmap.get(eid, 0))] for eid in new]})
        else:
            # Zip through the pre-rating prefix (board still empty) at warmup-stride,
            # then settle to quiet-stride once the leaderboard has populated.
            stride = max(qstride, args.warmup_stride) if not cur else qstride
            if (i % stride == 0) or i == N - 1:
                for _ in range(args.tick_frames):
                    emit(placed_static(cur, count))
        i += 1

    final = placed_static(cur, N)
    for _ in range(args.hold):
        emit(final)
    proc.stdin.close()
    proc.wait()
    print(f"leaderboard[{args.board}/{args.layout}]: {n_changes} board changes over {N} publications; "
          f"global mean rating={m:.2f}")
    print(f"wrote {args.out} ({emitted} frames, {emitted/args.fps:.1f}s, {W}x{H})")
    if args.emit_board_timeline:
        args.emit_board_timeline.parent.mkdir(parents=True, exist_ok=True)
        args.emit_board_timeline.write_text(json.dumps(
            {"board": args.board, "K": K, "n": N, "timeline": board_timeline}))
        print(f"wrote board timeline -> {args.emit_board_timeline} ({len(board_timeline)} changes)")


if __name__ == "__main__":
    main()
