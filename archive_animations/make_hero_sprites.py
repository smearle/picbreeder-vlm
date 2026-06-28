#!/usr/bin/env python3
"""Turn the teaser-lineage clips into sprite sheets for the interactive hero.

For each hero image (in the paper's teaser order) we sample N evenly-spaced
frames from its lineage morph (root -> current, excluding the trailing hold) and
tile them into one JPG. The blog draws these on a <canvas>, advancing the frame
index forward/backward for fully reversible playback.

Outputs into the site repo: <site>/picbreeder-vlm/assets/hero_sprites/{id}.jpg
plus manifest.json (hero order + sprite layout).
"""
from __future__ import annotations
import argparse, glob, json, math, os
from pathlib import Path
import cv2
import numpy as np


def _decode(mp4: Path):
    cap = cv2.VideoCapture(str(mp4))
    allf = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        allf.append(fr)
    cap.release()
    if not allf:
        raise RuntimeError(f"no frames read from {mp4}")
    return allf


def _morph_len(total, hold_end):
    return total - hold_end if total > hold_end + 2 else total   # drop static tail


def clip_dist(mp4: Path, hold_end: int) -> float:
    """Total visual distance of a clip's morph (independent of frame budget)."""
    allf = _decode(mp4)
    _, dist = _frame_indices(allf, 2, _morph_len(len(allf), hold_end), "distance")
    return dist


def _frame_indices(allf, n, morph, pace, dist_size=48):
    """Pick n frame indices in [0, morph-1] and report the morph's total visual
    distance (sum of per-frame pixel change).

    pace='time'     -> equal spacing in time (linear interpolation parameter).
    pace='distance' -> equal spacing in cumulative *pixel distance*, so each
                       displayed step carries ~the same visual change. At a
                       constant playback fps this makes the morph LOOK uniform:
                       it dwells on fast (high-change) stretches and skims slow ones.

    The returned total distance lets the banner pace cells *relative to each
    other* (speed proportional to 1/distance), so visual change-per-second is
    ~uniform across the grid rather than every cell taking the same wall time.
    """
    def small(fr):
        g = cv2.cvtColor(cv2.resize(fr, (dist_size, dist_size), interpolation=cv2.INTER_AREA),
                         cv2.COLOR_BGR2GRAY)
        return g.astype(np.int16)

    cum = np.zeros(morph)
    prev = small(allf[0])
    for j in range(1, morph):
        cur = small(allf[j])
        cum[j] = cum[j - 1] + float(np.abs(cur - prev).mean())
        prev = cur
    total_dist = float(cum[-1])

    if pace == "time" or morph < 3 or total_dist < 1e-6:   # static -> fall back to time
        idx = [int(round(x)) for x in np.linspace(0, morph - 1, n)]
    else:
        levels = np.linspace(0, total_dist, n)
        idx = [int(round(x)) for x in np.interp(levels, cum, np.arange(morph))]
    return idx, total_dist


def make_sprite(mp4: Path, n: int, fw: int, hold_end: int, pace: str = "distance"):
    """Tile n distance-paced frames into a square-ish sheet. n is capped at the
    clip's available morph frames so we never pad with duplicate frames."""
    allf = _decode(mp4)
    morph = _morph_len(len(allf), hold_end)
    n = max(2, min(n, morph))                      # don't invent frames the clip lacks
    cols = math.ceil(math.sqrt(n))                 # keep each sheet ~square regardless of n
    idx, dist = _frame_indices(allf, n, morph, pace)
    seq = [cv2.resize(allf[j], (fw, fw), interpolation=cv2.INTER_AREA) for j in idx]
    rows = (n + cols - 1) // cols
    sheet = np.full((rows * fw, cols * fw, 3), 255, np.uint8)
    for k, fr in enumerate(seq):
        r, c = divmod(k, cols)
        sheet[r * fw:(r + 1) * fw, c * fw:(c + 1) * fw] = fr
    return sheet, n, cols, rows, fw, dist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", type=Path, default=Path("archive_animations/out/teaser_lineages"))
    ap.add_argument("--out", type=Path,
                    default=Path("/home/jupyter-smearle/smearle.github.io/picbreeder-vlm-06b0d76d/assets/hero_sprites"))
    ap.add_argument("--order", type=Path, default=Path("/tmp/hero_order.json"))
    ap.add_argument("--grain-ref-n", type=int, default=224,
                    help="frame count a MEDIAN-distance clip gets; sets the global visual grain. Every clip "
                         "is sized n = dist/grain so all morphs share the same per-frame visual change, and "
                         "(at one playback FPS) the same smoothness -- busier lineages just get more frames / run longer.")
    ap.add_argument("--n-min", type=int, default=72, help="floor on frames (caps shortest morph)")
    ap.add_argument("--n-max", type=int, default=720, help="ceiling on frames (caps sheet size / longest morph)")
    ap.add_argument("--fw", type=int, default=88)
    ap.add_argument("--hold-end", type=int, default=40)
    ap.add_argument("--quality", type=int, default=80)
    ap.add_argument("--pace", choices=["distance", "time"], default="distance",
                    help="distance = equal pixel-change per frame (uniform-looking morph); time = equal-t sampling")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    od = json.loads(args.order.read_text())
    order, fig2id = od["order"], od["fig2id"]
    mp4s = {os.path.basename(f).split("__")[0]: f for f in glob.glob(str(args.clips / "*.mp4"))}

    # Phase 1: total visual distance per unique clip -> a single global grain g.
    dist_by_sid = {}
    for fig in order:
        sid = fig2id.get(fig, fig)
        if sid in dist_by_sid:
            continue
        mp4 = mp4s.get(sid) or mp4s.get(fig)
        if not mp4:
            print("  MISSING mp4 for", fig); continue
        try:
            dist_by_sid[sid] = clip_dist(Path(mp4), args.hold_end)
        except Exception as e:
            print(f"  SKIP dist {fig} ({sid}): {e}")
    if not dist_by_sid:
        raise SystemExit("no clips found under " + str(args.clips))
    med = sorted(dist_by_sid.values())[len(dist_by_sid) // 2]
    g = med / max(1, args.grain_ref_n - 1)          # visual change per frame, shared by ALL cells

    def n_for(d):
        return int(max(args.n_min, min(args.n_max, round(d / g))))

    # Phase 2: tile each clip at its own n = dist/g (clamped) so per-frame change is uniform.
    manifest = []
    seen = {}
    total_bytes = 0
    for fig in order:
        sid = fig2id.get(fig, fig)
        mp4 = mp4s.get(sid) or mp4s.get(fig)
        if not mp4:
            continue
        if sid not in seen:
            try:
                sheet, n, cols, rows, fw, dist = make_sprite(Path(mp4), n_for(dist_by_sid[sid]), args.fw, args.hold_end, args.pace)
            except Exception as e:
                print(f"  SKIP {fig} ({sid}): {e}"); continue
            sp = args.out / f"{sid}.jpg"
            cv2.imwrite(str(sp), sheet, [cv2.IMWRITE_JPEG_QUALITY, args.quality])
            seen[sid] = (n, cols, rows, fw, dist)
            total_bytes += sp.stat().st_size
        n, cols, rows, fw, dist = seen[sid]
        manifest.append({"fig": fig, "id": sid, "sprite": f"assets/hero_sprites/{sid}.jpg",
                         "n": n, "cols": cols, "rows": rows, "fw": fw, "dist": round(dist, 1)})
    (args.out / "manifest.json").write_text(json.dumps(manifest))
    ns = [s[0] for s in seen.values()]
    print(f"wrote {len(seen)} sprites ({total_bytes//1024} KB total) + manifest for {len(manifest)} hero cells -> {args.out}")
    print(f"grain g={g:.2f} units/frame (median dist={med:.0f}); n range [{min(ns)}, {max(ns)}]")


if __name__ == "__main__":
    main()
