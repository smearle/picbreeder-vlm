#!/usr/bin/env python3
"""Emit hero_sprites/titles.json: per-hero lineage of VLM titles, each placed at
the EXACT morph frame where its published image actually appears.

The hero morph (teaser_lineages) plays the full genome lineage:

    random-init root -> unnamed intra-session evolution -> published image

across the cross-session phylogeny. Only the *published* archive images carry
VLM titles, and they are a sparse subset of the morph -- the first one does NOT
sit at frame 0 (a random-init "dot" with no name does). So we:

  1. trace the published-image chain (``source_entry_ids[0]``, root -> teaser);
  2. locate each published image in the clip by matching its archive PNG against
     the clip's frames (monotonic argmin on a small grayscale thumbnail);
  3. project that morph frame onto the sprite's (distance-paced) frame axis via
     the cumulative-distance curve -- the same axis make_hero_sprites samples on,
     so titles land in lockstep with the image;
  4. emit a title keyframe there, collapsing runs of identical consecutive
     titles, and PREPEND an empty keyframe at frame 0 so the random-init root and
     all pre-publication evolution carry no name.

Output shape:
    { "img_000234": { "n": 224, "frames": [
        {"f": 0,   "title": "",            "id": "<root>"},     # unnamed root prefix
        {"f": 27,  "title": "Ghostly Vase", "id": "img_000019"},
        ...
        {"f": 223, "title": "Chrome Owl",   "id": "img_000234"} # published tip
    ] } }
"""
from __future__ import annotations
import argparse, glob, json, re
from pathlib import Path

import cv2
import numpy as np

REPO = Path("/home/jupyter-smearle/picbreeder-vlm")
HOLD_END = 40


def archive_branch(entries, teaser_id):
    """Chain of archive ids root -> teaser via source_entry_ids[0]."""
    parent = {e["id"]: (e.get("source_entry_ids") or [None])[0] for e in entries}
    chain, cur, seen = [teaser_id], teaser_id, {teaser_id}
    while parent.get(cur) and parent[cur] not in seen:
        cur = parent[cur]; chain.append(cur); seen.add(cur)
    return chain[::-1]


def _title(entry):
    t = (entry.get("title") or "").strip()
    if t and t.lower() != "none":
        return t
    rt = entry.get("vlm_reported_titles") or []
    return next((x for x in rt if x and x.strip().lower() != "none"), "")


def _gray(im, s):
    return cv2.cvtColor(cv2.resize(im, (s, s), interpolation=cv2.INTER_AREA),
                        cv2.COLOR_BGR2GRAY).astype(np.float32)


def _decode(mp4):
    cap = cv2.VideoCapture(str(mp4)); allf = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        allf.append(fr)
    cap.release()
    return allf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provenance", type=Path, default=REPO / "archive_animations/teaser_provenance.json")
    ap.add_argument("--clips", type=Path, default=REPO / "archive_animations/out/teaser_lineages")
    ap.add_argument("--manifest", type=Path,
                    default=Path("/home/jupyter-smearle/smearle.github.io/picbreeder-vlm-06b0d76d/assets/hero_sprites/manifest.json"))
    ap.add_argument("--out", type=Path,
                    default=Path("/home/jupyter-smearle/smearle.github.io/picbreeder-vlm-06b0d76d/assets/hero_sprites/titles.json"))
    ap.add_argument("--match-size", type=int, default=64)
    ap.add_argument("--min-title-gap", type=int, default=24,
                    help="minimum sprite-frames between consecutive titles (~1.2s at 20fps). Titles keep "
                         "their true publication frames where they're already spaced; clustered ones are "
                         "nudged apart so dense lineages (e.g. The Watcher) don't rapid-fire.")
    args = ap.parse_args()
    min_gap = max(1, args.min_title_gap)

    prov = json.loads(args.provenance.read_text())
    manifest = json.loads(args.manifest.read_text())
    mp4s = {Path(f).name.split("__")[0]: f for f in glob.glob(str(args.clips / "*.mp4"))}

    md_cache: dict[str, list] = {}
    out: dict[str, dict] = {}
    misses = 0
    S = args.match_size
    for m in manifest:
        fig, sid, n = m["fig"], m["id"], m["n"]
        rel = prov.get(fig + ".png")
        if not rel:
            print("  miss provenance:", fig); misses += 1; continue
        run = re.search(r"sweep/([^/]+)/archive", rel).group(1)
        tid = Path(rel).name[:-4]
        md_path = REPO / "sweep_logs/sweep" / run / "archive/archive_metadata.json"
        if str(md_path) not in md_cache:
            try:
                md_cache[str(md_path)] = json.loads(md_path.read_text())["entries"]
            except Exception as e:
                print(f"  miss metadata {fig}: {e}"); misses += 1; continue
        by = {e["id"]: e for e in md_cache[str(md_path)]}
        branch = archive_branch(md_cache[str(md_path)], tid)

        mp4 = mp4s.get(sid) or mp4s.get(fig)
        if not mp4:
            print("  miss mp4:", fig); misses += 1; continue
        allf = _decode(Path(mp4))
        if len(allf) < 3:
            print("  short mp4:", fig); misses += 1; continue
        total = len(allf)
        morph = total - HOLD_END if total > HOLD_END + 2 else total

        # cumulative visual distance over the morph (same 48px gray metric the
        # sprite pacing uses), so morph-frame -> sprite-frame is the exact inverse
        # of make_hero_sprites' distance sampling.
        def small48(fr):
            return cv2.cvtColor(cv2.resize(fr, (48, 48), interpolation=cv2.INTER_AREA),
                                cv2.COLOR_BGR2GRAY).astype(np.int16)
        cum = np.zeros(morph); prev = small48(allf[0])
        for j in range(1, morph):
            cur = small48(allf[j]); cum[j] = cum[j - 1] + float(np.abs(cur - prev).mean()); prev = cur
        total_dist = float(cum[-1]) if cum[-1] > 1e-6 else 1.0

        morph_gray = np.stack([_gray(allf[j], S) for j in range(morph)])  # (morph,S,S)
        imgdir = REPO / "sweep_logs/sweep" / run / "archive/images"

        # Match each published image to a morph frame, constrained to be monotonic
        # (each later image must appear at/after the previous one).
        kept: list[list] = []          # [sprite_f, title]
        last_title, lo = None, 0
        for iid in branch:
            png = imgdir / f"{iid}.png"
            if not png.exists():
                continue
            q = _gray(cv2.imread(str(png)), S)
            seg = morph_gray[lo:]
            mf = lo + int(np.argmin(np.abs(seg - q).mean(axis=(1, 2)))) if len(seg) else morph - 1
            lo = mf
            t = _title(by.get(iid, {}))
            if not t or t == last_title:
                continue               # untitled, or a run of the same title -> keep first
            kept.append([int(round(cum[mf] / total_dist * (n - 1))), t])
            last_title = t
        if not kept:
            print(f"  no matched titled entries {fig}"); misses += 1; continue

        # titles sit at their true matched publication frames, but keep a MINIMUM
        # gap between them: forward pass pushes clustered titles apart, the tip is
        # pinned to the last frame, then a backward pass pulls any overshoot back
        # (compressing toward even spacing only when the lineage is too title-dense
        # to honor the gap, e.g. The Watcher's rapid-fire tail).
        for i in range(1, len(kept)):
            if kept[i][0] < kept[i - 1][0] + min_gap:
                kept[i][0] = kept[i - 1][0] + min_gap
        kept[-1][0] = n - 1
        for i in range(len(kept) - 2, -1, -1):
            if kept[i][0] > kept[i + 1][0] - min_gap:
                kept[i][0] = kept[i + 1][0] - min_gap

        # Prepend the unnamed root: random-init + pre-publication evolution.
        frames = [{"f": 0, "title": "", "id": branch[0]}]
        for sf, t in kept:
            frames.append({"f": int(max(1, sf)), "title": t})
        dd = [frames[0]]                           # dedupe identical adjacent f
        for fr in frames[1:]:
            if fr["f"] == dd[-1]["f"]:
                dd[-1] = fr
            else:
                dd.append(fr)
        out[fig] = {"n": n, "frames": dd}

    args.out.write_text(json.dumps(out, indent=2))
    tot = sum(len(v["frames"]) for v in out.values())
    print(f"wrote {len(out)} heroes, {tot} title-keyframes ({misses} misses) -> {args.out}")


if __name__ == "__main__":
    main()
