#!/usr/bin/env python3
"""Export the human + VLM "car" lineage morphs as a *captionless* frame atlas +
per-frame keyframe metadata, so the blog can play a single morph tile that RIDES
ALONG the static lineage strip -- gliding from one published thumbnail to the
next and pausing briefly on each publication.

This reuses the exact frame generation of ``lineage_morph.py`` (same genomes,
same per-segment frame budget, same publication dwell), but instead of baking a
looping GIF with captions, it writes:

* ``<grp>_ride.webp`` -- a grayscale grid atlas of every morph frame at ``tile``
  px (the morph never renders larger than it is shown, so it stays crisp).
* ``<grp>_ride.json`` -- ``{fps, tile, cols, nframes, frameKf, pubKf, titles}``
  where ``frameKf[i]`` is the fractional canon-genome coordinate of frame i and
  ``pubKf`` is the sorted list of canon indices that are *published* (one per
  thumbnail in the strip). The page brackets each frame's ``frameKf`` between two
  ``pubKf`` stations and lerps the two matching thumbnails' screen positions, so
  the tile physically travels the strip; the dwell frames at integer ``pubKf``
  coordinates produce the pause at each publication.

Frames are trimmed to span the first..last publication, so the ride begins
parked on thumbnail 0 (the first published image) and ends parked on the last.

Usage:
    .venv/bin/python archive_animations/lineage_ride.py \
        --out-dir /home/jupyter-smearle/smearle.github.io/picbreeder-vlm-06b0d76d/assets/lineages
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "archive_animations"))

import cppn_interp as ci  # noqa: E402
import lineage_morph as lm  # noqa: E402


def trim_to_publications(frames, frame_kf, pub_idx):
    """Keep only frames whose keyframe coordinate lies within the published span
    [pub_idx[0], pub_idx[-1]] -- the strip shows only published thumbnails, so
    the ride should start parked on the first publication and end on the last."""
    lo, hi = pub_idx[0], pub_idx[-1]
    keep = [(f, kf) for f, kf in zip(frames, frame_kf) if lo - 1e-6 <= kf <= hi + 1e-6]
    return [f for f, _ in keep], [kf for _, kf in keep]


def build_atlas(frames, tile, cols):
    """Lay every frame into a `cols`-wide grayscale grid, each cell `tile` px."""
    rows = math.ceil(len(frames) / cols)
    atlas = Image.new("L", (cols * tile, rows * tile), 0)
    for i, fr in enumerate(frames):
        im = fr.convert("L")
        if im.size != (tile, tile):
            im = im.resize((tile, tile), Image.LANCZOS)
        atlas.paste(im, ((i % cols) * tile, (i // cols) * tile))
    return atlas, rows


def export(grp, frames, frame_kf, pub_idx, titles, out_dir, tile, cols, fps):
    frames, frame_kf = trim_to_publications(frames, frame_kf, pub_idx)
    atlas, rows = build_atlas(frames, tile, cols)
    out_dir.mkdir(parents=True, exist_ok=True)
    webp = out_dir / f"{grp}_ride.webp"
    atlas.save(webp, quality=90, method=6)
    meta = {
        "fps": fps,
        "tile": tile,
        "cols": cols,
        "rows": rows,
        "nframes": len(frames),
        "frameKf": [round(kf, 4) for kf in frame_kf],
        "pubKf": list(pub_idx),  # canon indices of the published thumbnails, in order
        "titles": titles,        # per-thumbnail title ("" for the human strip)
    }
    (out_dir / f"{grp}_ride.json").write_text(json.dumps(meta))
    kb = webp.stat().st_size / 1024
    print(f"  {grp}: {len(frames)} frames, {len(pub_idx)} stations, "
          f"atlas {atlas.width}x{atlas.height} ({kb:.0f} KB)")


def write_thumbs(grp, out_dir, res):
    """Re-render the strip thumbnails from the SAME station genomes the morph
    parks on, so the riding tile matches each publication pixel-for-pixel. The
    human founder's pre-rendered archive thumbnail (extract_lineage_assets.py)
    disagrees with its rep-genome render -- a founder data quirk -- which would
    otherwise show through when the tile parks on it; rendering from the genome
    keeps strip and morph in lockstep."""
    dest = out_dir / grp
    dest.mkdir(parents=True, exist_ok=True)
    if grp == "vlm":
        config = ci.build_config()
        genomes, meta = lm.vlm_full_chain()
        pub_idx = [i for i, (_, _, p) in enumerate(meta) if p]
        for j, idx in enumerate(pub_idx):
            ci.canon_frame(genomes[idx], config, res, "gray", False).convert("RGB").save(dest / f"{j:02d}.png")
    else:
        chain = lm.human_chain()
        pub_idx = [i for i, (_, _, p) in enumerate(chain) if p]
        for j, idx in enumerate(pub_idx):
            lm.render_pbcppn(lm.load_pbcppn_from_chain(chain[idx][0]), res).convert("RGB").save(dest / f"{j:02d}.png")
    print(f"  {grp}: rewrote {len(pub_idx)} strip thumbnails at {res}px")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path,
                    default=Path("/home/jupyter-smearle/smearle.github.io/picbreeder-vlm-06b0d76d/assets/lineages"))
    ap.add_argument("--tile", type=int, default=160, help="atlas cell size (px)")
    ap.add_argument("--cols", type=int, default=24, help="atlas columns")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--vlm-budget", type=int, default=200)
    ap.add_argument("--vlm-min-seg", type=int, default=2)
    ap.add_argument("--vlm-max-seg", type=int, default=16)
    ap.add_argument("--human-budget", type=int, default=240)
    ap.add_argument("--human-min-seg", type=int, default=2)
    ap.add_argument("--human-max-seg", type=int, default=14)
    ap.add_argument("--hold-pub", type=int, default=12, help="frames to dwell on each published station")
    ap.add_argument("--which", choices=["both", "vlm", "human", "none"], default="both")
    ap.add_argument("--write-thumbs", choices=["none", "both", "vlm", "human"], default="none",
                    help="also re-render the strip thumbnails from the station genomes (seamless parking)")
    ap.add_argument("--thumb-res", type=int, default=256)
    args = ap.parse_args()

    if args.write_thumbs != "none":
        print("Re-rendering strip thumbnails ...")
        for grp in (["vlm", "human"] if args.write_thumbs == "both" else [args.write_thumbs]):
            write_thumbs(grp, args.out_dir, args.thumb_res)

    if args.which in ("vlm", "both"):
        print("VLM lineage ride ...")
        config = ci.build_config()
        frames, frame_kf, meta = lm.render_vlm_frames(config, args.tile, args.vlm_budget,
                                                       args.vlm_min_seg, args.vlm_max_seg, args.hold_pub)
        pub_idx = [i for i, (_, _, pub) in enumerate(meta) if pub]
        titles = [meta[i][1] for i in pub_idx]
        export("vlm", frames, frame_kf, pub_idx, titles, args.out_dir, args.tile, args.cols, args.fps)

    if args.which in ("human", "both"):
        print("Human lineage ride ...")
        frames, frame_kf, meta = lm.render_human_frames(args.tile, args.human_budget,
                                                         args.human_min_seg, args.human_max_seg, args.hold_pub)
        pub_idx = [i for i, (_, pub) in enumerate(meta) if pub]
        titles = ["" for _ in pub_idx]
        export("human", frames, frame_kf, pub_idx, titles, args.out_dir, args.tile, args.cols, args.fps)


if __name__ == "__main__":
    main()
