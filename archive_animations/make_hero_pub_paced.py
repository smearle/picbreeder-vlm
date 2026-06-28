#!/usr/bin/env python3
"""EXPERIMENTAL hero pacing: a CONSTANT number of frames between publications.

Instead of distance-pacing the whole morph (uniform visual change/sec, which made
titles flit through visually-dense stretches), we give every publication ->
publication segment the SAME number of frames -- and the random-init run-up to the
first publication is one more such segment, carrying no title. Consequences:

  * titles sit at their TRUE publication frames (image stays aligned to the name);
  * because each segment is the same length, consecutive titles are evenly spaced
    in time at one global FPS;
  * morph DURATION now scales with the number of publications, not visual distance.

Within each segment frames are distance-paced (from the existing clip), so it is
as smooth as the clip allows. NOTE: this RESAMPLES the existing MP4s; segments
that are visually tiny just hold (few real frames stretched), and very large ones
are coarser. If the look is right we can re-render with a true per-segment budget.

Writes sprites + manifest.json + titles.json into the hero_sprites dir, overwriting
the distance-paced versions (re-run make_hero_sprites.py + make_hero_titles.py to
revert).
"""
from __future__ import annotations
import argparse, glob, json, math, re
from pathlib import Path

import cv2
import numpy as np

REPO = Path("/home/jupyter-smearle/picbreeder-vlm")
HOLD_END = 40


def archive_branch(entries, tid):
    par = {e["id"]: (e.get("source_entry_ids") or [None])[0] for e in entries}
    chain, cur, seen = [tid], tid, {tid}
    while par.get(cur) and par[cur] not in seen:
        cur = par[cur]; chain.append(cur); seen.add(cur)
    return chain[::-1]


def _title(e):
    t = (e.get("title") or "").strip()
    if t and t.lower() != "none":
        return t
    for x in (e.get("vlm_reported_titles") or []):
        if x and x.strip().lower() != "none":
            return x
    return ""


def _gray(im, s):
    return cv2.cvtColor(cv2.resize(im, (s, s), interpolation=cv2.INTER_AREA),
                        cv2.COLOR_BGR2GRAY).astype(np.float32)


def _decode(mp4):
    cap = cv2.VideoCapture(str(mp4)); a = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        a.append(fr)
    cap.release()
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provenance", type=Path, default=REPO / "archive_animations/teaser_provenance.json")
    ap.add_argument("--clips", type=Path, default=REPO / "archive_animations/out/teaser_lineages")
    ap.add_argument("--out", type=Path,
                    default=Path("/home/jupyter-smearle/smearle.github.io/picbreeder-vlm-06b0d76d/assets/hero_sprites"))
    ap.add_argument("--order", type=Path, default=Path("/tmp/hero_order.json"))
    ap.add_argument("--seg-frames", type=int, default=40,
                    help="frames per publication->publication segment (and for the root run-up)")
    ap.add_argument("--fw", type=int, default=88)
    ap.add_argument("--quality", type=int, default=80)
    ap.add_argument("--match-size", type=int, default=64)
    args = ap.parse_args()
    M, S = args.seg_frames, args.match_size

    prov = json.loads(args.provenance.read_text())
    od = json.loads(args.order.read_text())
    order, fig2id = od["order"], od["fig2id"]
    mp4s = {Path(f).name.split("__")[0]: f for f in glob.glob(str(args.clips / "*.mp4"))}

    md_cache: dict[str, list] = {}
    manifest, titles, seen = [], {}, set()
    for fig in order:
        rel = prov.get(fig + ".png")
        if not rel:
            print("  miss prov", fig); continue
        run = re.search(r"sweep/([^/]+)/archive", rel).group(1)
        tid = Path(rel).name[:-4]
        mp = REPO / "sweep_logs/sweep" / run / "archive/archive_metadata.json"
        if str(mp) not in md_cache:
            md_cache[str(mp)] = json.loads(mp.read_text())["entries"]
        by = {e["id"]: e for e in md_cache[str(mp)]}
        branch = archive_branch(md_cache[str(mp)], tid)

        sid = fig2id.get(fig, fig)
        mp4 = mp4s.get(sid) or mp4s.get(fig)
        if not mp4:
            print("  miss mp4", fig); continue
        allf = _decode(Path(mp4))
        if len(allf) < 3:
            print("  short", fig); continue
        morph = len(allf) - HOLD_END if len(allf) > HOLD_END + 2 else len(allf)

        def s48(fr):
            return cv2.cvtColor(cv2.resize(fr, (48, 48), interpolation=cv2.INTER_AREA),
                                cv2.COLOR_BGR2GRAY).astype(np.int16)
        cum = np.zeros(morph); prev = s48(allf[0])
        for j in range(1, morph):
            cur = s48(allf[j]); cum[j] = cum[j - 1] + float(np.abs(cur - prev).mean()); prev = cur

        # match publications -> morph frames (monotonic), distinct consecutive titles
        mg = np.stack([_gray(allf[j], S) for j in range(morph)])
        imgdir = REPO / "sweep_logs/sweep" / run / "archive/images"
        pubs = []  # (morph_frame, title)
        lo, last = 0, None
        for iid in branch:
            png = imgdir / f"{iid}.png"
            if not png.exists():
                continue
            q = _gray(cv2.imread(str(png)), S); seg = mg[lo:]
            mf = lo + int(np.argmin(np.abs(seg - q).mean(axis=(1, 2)))) if len(seg) else morph - 1
            lo = mf
            t = _title(by.get(iid, {}))
            if not t or t == last:
                continue
            pubs.append((mf, t)); last = t
        if not pubs:
            print("  no titles", fig); continue
        pubs[-1] = (morph - 1, pubs[-1][1])               # published tip = last frame

        bounds = [0] + [min(p[0], morph - 1) for p in pubs]   # root + K publications
        bounds[-1] = morph - 1                                 # tip pinned to last frame
        for i in range(len(bounds) - 2, -1, -1):               # separate from the tip backward
            if bounds[i] >= bounds[i + 1]:
                bounds[i] = bounds[i + 1] - 1
        bounds = [max(0, b) for b in bounds]

        # sample M frames per segment (distance-paced within the segment)
        idx, pub_sprite = [], []
        for si in range(len(bounds) - 1):
            a, b = bounds[si], bounds[si + 1]
            if cum[b] - cum[a] < 1e-6:
                lv = np.linspace(a, b, M)
            else:
                lv = np.interp(np.linspace(cum[a], cum[b], M), cum[a:b + 1], np.arange(a, b + 1))
            seg = [int(round(x)) for x in lv]
            if si > 0:
                seg = seg[1:]                              # drop shared boundary frame
            idx += seg
            pub_sprite.append(len(idx) - 1)                # sprite index of this segment's end publication
        n = len(idx)

        cols = math.ceil(math.sqrt(n)); rows = (n + cols - 1) // cols
        sheet = np.full((rows * args.fw, cols * args.fw, 3), 255, np.uint8)
        for k, j in enumerate(idx):
            r, c = divmod(k, cols)
            sheet[r * args.fw:(r + 1) * args.fw, c * args.fw:(c + 1) * args.fw] = \
                cv2.resize(allf[j], (args.fw, args.fw), interpolation=cv2.INTER_AREA)
        if sid not in seen:
            cv2.imwrite(str(args.out / f"{sid}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, args.quality])
            seen.add(sid)

        manifest.append({"fig": fig, "id": sid, "sprite": f"assets/hero_sprites/{sid}.jpg",
                         "n": n, "cols": cols, "rows": rows, "fw": args.fw, "dist": round(float(cum[-1]), 1)})
        frames = [{"f": 0, "title": "", "id": branch[0]}]
        for (mf, t), sf in zip(pubs, pub_sprite):
            frames.append({"f": int(sf), "title": t})
        dd = [frames[0]]
        for fr in frames[1:]:
            if fr["f"] == dd[-1]["f"]:
                dd[-1] = fr
            else:
                dd.append(fr)
        titles[fig] = {"n": n, "frames": dd}

    (args.out / "manifest.json").write_text(json.dumps(manifest))
    (args.out / "titles.json").write_text(json.dumps(titles, indent=2))
    ns = [m["n"] for m in manifest]
    print(f"wrote {len(seen)} sprites, {len(manifest)} cells; seg-frames={M} "
          f"(=> {M-1} frames / {(M-1)/20:.2f}s between titles @20fps); n range [{min(ns)}, {max(ns)}]")


if __name__ == "__main__":
    main()
