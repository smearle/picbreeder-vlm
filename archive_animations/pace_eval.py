#!/usr/bin/env python3
"""Bake pacing into the CPPN sampling itself, and save comparison GIFs.

For a teaser image's full random-root lineage we render a *fine* uniform-t
sequence per segment (one segment = one parent->child step), then derive two
re-timed outputs from the same fine frames:

  * AGE-UNIFORM   -- equal screen-time per evolutionary step (uniform in the
    interpolation parameter / generation). Big visual jumps whip by; tiny
    refinements linger.
  * VISUAL-UNIFORM -- frames chosen at equal cumulative *pixel distance*, so the
    perceived rate of change is constant. This is the "regularized" pacing,
    applied to the genome sampling rather than post-hoc to a finished clip.

Saves <id>_age.gif, <id>_visual.gif and a labelled side-by-side <id>_compare.gif
for visual evaluation.

Usage:
    python archive_animations/pace_eval.py                 # default sample
    python archive_animations/pace_eval.py --fig img_000093.png --fine 18
    python archive_animations/pace_eval.py --all
"""
from __future__ import annotations
import argparse, json, os, sys, tempfile
from pathlib import Path
import numpy as np
import cv2
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cppn_interp as ci
import teaser_lineages as tl
from render_lineage_animation import build_superset, render_frames

DEFAULT_SAMPLE = ["img_000093.png", "img_000063.png", "img_000043.png",
                  "img_000125.png", "img_000234.png"]


def render_fine(chain, config, img_res, fine, variant, color_enabled):
    """Fine uniform-t render of the whole lineage. Returns (frames, age) where
    age[i] is the evolutionary position (0..n_segments) of frame i."""
    genomes = [g for _, g in chain]
    frames = [ci.canon_frame(genomes[0], config, img_res, variant, color_enabled)]
    age = [0.0]
    for i in range(len(genomes) - 1):
        ss, npr, cpr, outs = build_superset(genomes[i], genomes[i + 1])
        segf = render_frames(ss, npr, cpr, config, steps=fine, width=img_res, height=img_res,
                             variant_mode=variant, color_enabled=color_enabled,
                             output_activation_stats=outs)
        for k in range(1, fine):                      # skip seg start (== prev seg end)
            frames.append(segf[k]); age.append(i + k / (fine - 1))
    return frames, np.asarray(age)


def cum_pixel_dist(frames, ds=64):
    g = [cv2.cvtColor(cv2.resize(np.asarray(f), (ds, ds)), cv2.COLOR_RGB2GRAY).astype(np.int16) for f in frames]
    cum = np.zeros(len(frames))
    for i in range(1, len(frames)):
        cum[i] = cum[i - 1] + float(np.abs(g[i] - g[i - 1]).mean())
    return cum


def resample(frames, param, T):
    """Pick T frames so they are equally spaced along monotonic `param`."""
    levels = np.linspace(param[0], param[-1], T)
    idx = np.round(np.interp(levels, param, np.arange(len(frames)))).astype(int)
    return [frames[j] for j in idx]


def save_gif(frames, path, fps, hold=12):
    seq = list(frames) + [frames[-1]] * hold
    seq[0].save(path, save_all=True, append_images=seq[1:],
                duration=int(round(1000 / fps)), loop=0, optimize=True)


def label(img, text):
    out = Image.new("RGB", (img.width, img.height + 18), (255, 255, 255))
    out.paste(img, (0, 18))
    ImageDraw.Draw(out).text((4, 3), text, fill=(0, 0, 0))
    return out


def compare_gif(age_frames, vis_frames, path, fps, hold=12):
    seq = []
    for a, v in zip(age_frames, vis_frames):
        la, lv = label(a, "age-uniform"), label(v, "visually uniform")
        canvas = Image.new("RGB", (la.width + lv.width + 6, la.height), (255, 255, 255))
        canvas.paste(la, (0, 0)); canvas.paste(lv, (la.width + 6, 0))
        seq.append(canvas)
    save_gif(seq, path, fps, hold)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provenance", type=Path, default=Path("archive_animations/teaser_provenance.json"))
    ap.add_argument("--fig", action="append", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=Path("archive_animations/out/pace_eval"))
    ap.add_argument("--img-res", type=int, default=132)
    ap.add_argument("--fine", type=int, default=16, help="fine uniform-t frames rendered per segment")
    ap.add_argument("--out-frames", type=int, default=130, help="frames in each output GIF")
    ap.add_argument("--fps", type=int, default=20)
    args = ap.parse_args()

    prov = json.loads(args.provenance.read_text())
    figs = (sorted(prov) if args.all else (args.fig or DEFAULT_SAMPLE))
    config = ci.build_config()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for fig in figs:
        if fig not in prov:
            print(f"[skip] {fig}: not in provenance"); continue
        run, tid = tl.run_of(prov[fig]), tl.id_of(prov[fig])
        run_dir = Path("sweep_logs/sweep") / run
        wd = Path(tempfile.mkdtemp())
        try:
            chain, info = tl.build_full_lineage(run_dir, tid, wd)
        except Exception as e:
            print(f"[err] {fig}: {e}"); continue
        if info["missing"]:
            print(f"[skip] {fig}: missing agent zips {info['missing'][:3]}"); continue
        if len(chain) < 2:
            print(f"[skip] {fig}: short chain"); continue
        color = info["color_enabled"]
        variant = "color" if color else "gray"
        frames, age = render_fine(chain, config, args.img_res, args.fine, variant, color)
        dist = cum_pixel_dist(frames)
        T = args.out_frames
        age_frames = resample(frames, age, T)
        vis_frames = resample(frames, dist, T)
        save_gif(age_frames, args.out_dir / f"{tid}_age.gif", args.fps)
        save_gif(vis_frames, args.out_dir / f"{tid}_visual.gif", args.fps)
        compare_gif(age_frames, vis_frames, args.out_dir / f"{tid}_compare.gif", args.fps)
        # report uniformity of each output
        def cv(seq):
            d = cum_pixel_dist(seq); step = np.diff(d)
            return step.std() / step.mean() if step.mean() > 0 else 0.0
        print(f"[ok] {fig}: depth={info['depth']} keyframes={len(chain)} fine={len(frames)} "
              f"| CV age={cv(age_frames):.2f} visual={cv(vis_frames):.2f} -> {args.out_dir}")


if __name__ == "__main__":
    main()
