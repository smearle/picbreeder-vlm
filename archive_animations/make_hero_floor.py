#!/usr/bin/env python3
"""Tile the distance-floor morphs into hero sprites + manifest + titles.

Pairs with ``teaser_lineages.py --pace distance-floor``, which renders each clip
at a uniform visual grain with a per-publication-segment frame floor, and emits a
sidecar of publication frame positions + titles. Here we tile the morph frames 1:1
into a sprite sheet (so the on-screen image IS the rendered morph -- uniform grain,
played at one global FPS -> uniform smoothness) and place each title exactly on its
publication frame. Because the render floored every publication-segment at
``--min-gap`` frames, consecutive titles are already >= that gap apart, with genuine
interpolations between them. The random-init run-up before the first publication
carries no title.
"""
from __future__ import annotations
import argparse, glob, json, math
from pathlib import Path

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", type=Path, default=Path("archive_animations/out/teaser_lineages_floor"))
    ap.add_argument("--sidecar-dir", type=Path, default=Path("/tmp/pf"))
    ap.add_argument("--order", type=Path, default=Path("/tmp/hero_order.json"))
    ap.add_argument("--out", type=Path,
                    default=Path("/home/jupyter-smearle/smearle.github.io/picbreeder-vlm-06b0d76d/assets/hero_sprites"))
    ap.add_argument("--fw", type=int, default=88)
    ap.add_argument("--quality", type=int, default=80)
    ap.add_argument("--n-max", type=int, default=1000, help="cap on sprite frames (downsample if a morph is longer)")
    ap.add_argument("--min-gap", type=int, default=24,
                    help="safety floor on frames between titles; only bites on genome-revisit clusters "
                         "(distinct publications are already >= this from the render's per-segment floor)")
    args = ap.parse_args()
    min_gap = max(1, args.min_gap)
    args.out.mkdir(parents=True, exist_ok=True)

    order = json.loads(args.order.read_text())["order"]
    sc: dict = {}
    for f in glob.glob(str(args.sidecar_dir / "*.json")):
        try:
            sc.update(json.loads(Path(f).read_text()))
        except Exception as e:
            print("  bad sidecar", f, e)

    # Identity is the UNIQUE fig name (img_001896 vs img_001896_2 are distinct teaser
    # slots from different runs that share an archive id). The sidecar -- keyed by fig --
    # tells us each slot's own clip, so they never collide.
    manifest, titles, seen, misses = [], {}, set(), 0
    for fig in order:
        entry = sc.get(fig)
        clip = (args.clips / entry["clip"]) if entry else None
        if not entry or not clip.exists():
            print("  miss", fig); misses += 1; continue
        cap = cv2.VideoCapture(str(clip)); allf = []
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            allf.append(fr)
        cap.release()
        morph = min(entry["n_total"] - entry["hold_end"], len(allf))
        if morph < 3:
            print("  short", fig); misses += 1; continue

        if morph > args.n_max:
            idx = [int(round(x)) for x in np.linspace(0, morph - 1, args.n_max)]
            scale = lambda f: int(round(f * (args.n_max - 1) / (morph - 1)))
        else:
            idx = list(range(morph)); scale = lambda f: int(f)
        n = len(idx)

        cols = math.ceil(math.sqrt(n)); rows = (n + cols - 1) // cols
        sheet = np.full((rows * args.fw, cols * args.fw, 3), 255, np.uint8)
        for k, j in enumerate(idx):
            r, c = divmod(k, cols)
            sheet[r * args.fw:(r + 1) * args.fw, c * args.fw:(c + 1) * args.fw] = \
                cv2.resize(allf[j], (args.fw, args.fw), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(args.out / f"{fig}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, args.quality])
        seen.add(fig)
        manifest.append({"fig": fig, "id": fig, "sprite": f"assets/hero_sprites/{fig}.jpg",
                         "n": n, "cols": cols, "rows": rows, "fw": args.fw})

        # titles: each publication at its true (scaled) frame; collapse consecutive
        # same-title runs. Distinct publications are already >= min_gap apart from the
        # render's per-segment floor; the forward/backward min_gap passes below only
        # nudge genome-REVISIT clusters (one genome published under several names lands
        # on a single frame) so those don't rapid-fire. Tip pinned to n-1, root blank.
        root_id = entry["pubs"][0]["id"] if entry["pubs"] else ""
        kept, last = [], None
        for p in entry["pubs"]:
            t = p["title"]
            if not t or t == last:
                continue
            kept.append([max(1, scale(p["f"])), t]); last = t
        if not kept:
            titles[fig] = {"n": n, "frames": [{"f": 0, "title": "", "id": root_id}]}
            continue
        for i in range(1, len(kept)):
            if kept[i][0] < kept[i - 1][0] + min_gap:
                kept[i][0] = kept[i - 1][0] + min_gap
        kept[-1][0] = n - 1
        for i in range(len(kept) - 2, -1, -1):
            if kept[i][0] > kept[i + 1][0] - min_gap:
                kept[i][0] = kept[i + 1][0] - min_gap
        frames = [{"f": 0, "title": "", "id": root_id}]
        frames += [{"f": int(max(1, sf)), "title": t} for sf, t in kept]
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
    print(f"tiled {len(seen)} sprites, {len(manifest)} cells ({misses} misses); n range [{min(ns)}, {max(ns)}]")


if __name__ == "__main__":
    main()
