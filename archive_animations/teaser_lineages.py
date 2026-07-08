#!/usr/bin/env python3
"""For each hero/teaser image, animate its CPPN morphing all the way back from a
random-init root to the published image -- tracing the full *cross-session*
archive phylogeny.

Each teaser image was published in some run. We:
  1. follow ``source_entry_ids`` from the image up to its root (an image whose
     session began from a random population), giving the chain of archive images;
  2. for each archive image in that chain, reconstruct its session's true
     parent->child genome lineage (``cppn_interp.load_lineage_chain`` on the
     agent that published it);
  3. concatenate root-session -> ... -> teaser-session and dedupe, yielding one
     genome sequence from a random CPPN to the teaser;
  4. morph along it with a fixed total **frame budget** (so a depth-2 and a
     depth-20 lineage are similar lengths -- deep ones just move faster).

Provenance (teaser fig -> source archive image path) comes from a JSON map
(default ``/tmp/teaser_provenance.json``: {"img_000093.png": ".../archive/images/img_000093.png"}).

Usage:
    python archive_animations/teaser_lineages.py --fig img_000093.png
    python archive_animations/teaser_lineages.py --all --budget 520 --panel 256
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cppn_interp as ci
from picbreeder_vlm.viz.render_lineage_animation import build_superset, render_frames


def run_of(path: str) -> str:
    return re.search(r"sweep/([^/]+)/archive", path).group(1)


def id_of(path: str) -> str:
    return os.path.basename(path)[:-4]


def _title(entry):
    t = (entry.get("title") or "").strip()
    if t and t.lower() != "none":
        return t
    for x in (entry.get("vlm_reported_titles") or []):
        if x and x.strip().lower() != "none":
            return x
    return ""


def archive_branch(meta_entries, teaser_id):
    """Chain of archive ids from root -> teaser via source_entry_ids[0]."""
    parent = {e["id"]: (e.get("source_entry_ids") or [None])[0] for e in meta_entries}
    chain, cur, seen = [teaser_id], teaser_id, {teaser_id}
    while parent.get(cur) and parent[cur] not in seen:
        cur = parent[cur]
        chain.append(cur)
        seen.add(cur)
    return chain[::-1], parent


def build_full_lineage(run_dir: Path, teaser_id: str, workdir: Path):
    """Return ([(key, genome), ...] root->teaser, info dict). Raises if a needed
    agent zip is missing (incomplete provenance)."""
    meta = json.loads((run_dir / "archive" / "archive_metadata.json").read_text())["entries"]
    agent = {e["id"]: e.get("agent_id") for e in meta}
    color_of = {e["id"]: bool(e.get("color_enabled", False)) for e in meta}
    title_of = {e["id"]: _title(e) for e in meta}
    branch, _ = archive_branch(meta, teaser_id)
    full, missing, pub_full_pos, pub_ids = [], [], [], []
    for iid in branch:
        aid = agent.get(iid)
        z = run_dir / "agents" / f"{aid}.zip"
        if not z.exists():
            missing.append(aid)
            continue
        with zipfile.ZipFile(z) as zf:
            zf.extractall(workdir)
        seg = ci.load_lineage_chain(workdir / aid)
        if not seg:
            continue
        full += seg
        pub_full_pos.append(len(full) - 1)        # this session's published genome = its last
        pub_ids.append(iid)                        # ...and which archive image it published
    # Dedupe consecutive identical genomes, tracking full->canon index so we can flag
    # which canon keyframes are publications (needed for per-publication frame budgets).
    canon, full2canon, last_sig = [], [], None
    for gen, g in full:
        sig = ci.genome_signature(g)
        if sig != last_sig:
            canon.append((gen, g)); last_sig = sig
        full2canon.append(len(canon) - 1)
    pub_canon_ordered = [full2canon[p] for p in pub_full_pos]   # one per branch member, in order
    pub_canon_idx = sorted(set(pub_canon_ordered)) if full else []
    # Title for each unique publication canon index (first archive image landing there).
    canon_to_id = {}
    for ci_idx, iid in zip(pub_canon_ordered, pub_ids):
        canon_to_id.setdefault(ci_idx, iid)
    pub_titles = [title_of.get(canon_to_id.get(ci_idx), "") for ci_idx in pub_canon_idx]
    info = dict(branch=branch, depth=len(branch), missing=missing,
                color_enabled=color_of.get(teaser_id, False),
                pub_canon_idx=pub_canon_idx, pub_canon_ordered=pub_canon_ordered,
                pub_titles=pub_titles)
    return canon, info


def allocate_steps(canon_imgs, budget, min_seg, max_seg):
    dists = [ci.visual_distance(canon_imgs[i], canon_imgs[i + 1]) for i in range(len(canon_imgs) - 1)]
    total = sum(dists) or 1.0
    steps = []
    for d in dists:
        s = round(min_seg + (budget * d / total))
        steps.append(int(min(max_seg, max(min_seg, s))))
    return steps


def allocate_steps_pub(canon_imgs, pub_idx, seg_budget, min_seg):
    """Give each publication->publication segment ~seg_budget interpolation frames,
    distributed within the segment in proportion to visual distance. The run-up to
    the first publication is its own segment. Result: a roughly constant number of
    genuine CPPN interpolations between consecutive publications (no global max cap,
    so even a single-mutation segment still gets the full budget)."""
    n = len(canon_imgs)
    dists = [ci.visual_distance(canon_imgs[i], canon_imgs[i + 1]) for i in range(n - 1)]
    steps = [min_seg] * (n - 1)
    bnds = sorted({0, n - 1} | {p for p in pub_idx if 0 < p < n})
    for s in range(len(bnds) - 1):
        a, b = bnds[s], bnds[s + 1]
        tot = sum(dists[a:b]) or 1.0
        for i in range(a, b):
            steps[i] = max(min_seg, int(round(seg_budget * dists[i] / tot)))
    return steps


def allocate_steps_distfloor(canon_imgs, pub_idx, grain, min_gap, min_seg):
    """Distance-paced (uniform grain) WITH a per-publication-segment floor.

    Each publication->publication segment gets frames ~ seg_dist/grain (so the
    whole banner shares one visual grain and one change-per-second), but never
    fewer than `min_gap` -- so visually-tiny clusters of publications still get
    enough genuine interpolation frames that their titles, pinned to those frames,
    stay >= min_gap apart. Frames within a segment are split in proportion to
    distance. `grain` is in the same units as ci.visual_distance (per frame)."""
    n = len(canon_imgs)
    dists = [ci.visual_distance(canon_imgs[i], canon_imgs[i + 1]) for i in range(n - 1)]
    steps = [min_seg] * (n - 1)
    bnds = sorted({0, n - 1} | {p for p in pub_idx if 0 < p < n})
    for s in range(len(bnds) - 1):
        a, b = bnds[s], bnds[s + 1]
        tot = sum(dists[a:b]) or 1.0
        nsub = b - a
        net = max(min_gap, int(round(tot / grain)))   # net morph frames this segment should contribute
        target = net + nsub                            # render_chain nets ~steps-1 per sub-segment
        for i in range(a, b):
            steps[i] = max(min_seg, int(round(target * dists[i] / tot)))
    return steps


def render_chain(chain, out: Path, config, *, panel, img_res, fps, budget,
                 min_seg, max_seg, hold_keyframe, hold_end, variant, color_enabled,
                 final_png=None, pub_idx=None, seg_budget=0, pace="distance", grain=20.0, min_gap=24):
    genomes = [g for _, g in chain]
    canon = [ci.canon_frame(g, config, img_res, variant, color_enabled) for g in genomes]
    if pace == "distance-floor" and pub_idx:
        steps = allocate_steps_distfloor(canon, pub_idx, grain, min_gap, min_seg)
    elif pace == "publications" and pub_idx:
        steps = allocate_steps_pub(canon, pub_idx, seg_budget, min_seg)
    else:
        steps = allocate_steps(canon, budget, min_seg, max_seg)

    out.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{panel}x{panel}", "-r", str(fps), "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)],
        stdin=subprocess.PIPE,
    )

    def emit(img, repeat=1):
        import cv2
        a = cv2.resize(np.asarray(img.convert("RGB")), (panel, panel), interpolation=cv2.INTER_NEAREST)
        buf = np.ascontiguousarray(a).tobytes()
        for _ in range(repeat):
            proc.stdin.write(buf)

    total = 0
    canon_pos = [0] * len(genomes)               # morph-frame index where each canon genome lands
    emit(canon[0], hold_keyframe); total += hold_keyframe
    canon_pos[0] = max(0, total - 1)
    for i in range(len(genomes) - 1):
        ss, npairs, cpairs, outs = build_superset(genomes[i], genomes[i + 1])
        frames = render_frames(ss, npairs, cpairs, config, steps=max(2, steps[i]),
                               width=img_res, height=img_res, variant_mode=variant,
                               color_enabled=color_enabled, output_activation_stats=outs)
        for fr in frames[1:-1]:
            emit(fr); total += 1
        emit(canon[i + 1], 1); total += 1
        canon_pos[i + 1] = total - 1             # before the trailing hold, so within the morph region
    emit(canon[-1], hold_end); total += hold_end
    proc.stdin.close()
    proc.wait()
    return total, canon_pos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provenance", type=Path, default=Path("/tmp/teaser_provenance.json"))
    ap.add_argument("--fig", type=str, default=None, help="single teaser fig basename, e.g. img_000093.png")
    ap.add_argument("--all", action="store_true", help="render every teaser with complete data")
    ap.add_argument("--out-dir", type=Path, default=Path("archive_animations/out/teaser_lineages"))
    ap.add_argument("--budget", type=int, default=520, help="target interior frames per clip")
    ap.add_argument("--min-seg", type=int, default=2)
    ap.add_argument("--max-seg", type=int, default=40)
    ap.add_argument("--panel", type=int, default=256)
    ap.add_argument("--img-res", type=int, default=160)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--hold-keyframe", type=int, default=0)
    ap.add_argument("--hold-end", type=int, default=40)
    ap.add_argument("--pace", choices=["distance", "publications", "distance-floor"], default="distance",
                    help="distance = frames ∝ visual distance over the whole morph; "
                         "publications = constant frames per publication-segment; "
                         "distance-floor = distance-paced (uniform grain) but with a floor of --min-gap "
                         "frames per publication-segment so titles never rapid-fire")
    ap.add_argument("--seg-budget", type=int, default=90,
                    help="interpolation frames per publication-segment when --pace publications")
    ap.add_argument("--grain", type=float, default=20.0,
                    help="visual-distance units per frame for --pace distance-floor (uniform grain across banner)")
    ap.add_argument("--min-gap", type=int, default=24,
                    help="floor on frames per publication-segment for --pace distance-floor (=> min frames between titles)")
    ap.add_argument("--sidecar", type=Path, default=None,
                    help="merge per-fig publication frame positions + titles into this JSON (for the tiler)")
    ap.add_argument("--allow-partial", action="store_true",
                    help="render even if some agent zips are missing (lineage starts mid-branch)")
    args = ap.parse_args()

    prov = json.loads(args.provenance.read_text())
    figs = [args.fig] if args.fig else sorted(prov.keys())
    config = ci.build_config()

    done, skipped = [], []
    for fig in figs:
        if fig not in prov:
            print(f"[skip] {fig}: not in provenance"); continue
        path = prov[fig]
        fig_base = fig[:-4] if fig.endswith(".png") else fig   # UNIQUE per teaser slot (img_001896 vs img_001896_2)
        run, tid = run_of(path), id_of(path)
        run_dir = Path("sweep_logs/sweep") / run
        workdir = Path(tempfile.mkdtemp())
        try:
            chain, info = build_full_lineage(run_dir, tid, workdir)
        except Exception as e:
            print(f"[err]  {fig}: {e}"); skipped.append((fig, "error")); continue
        if info["missing"] and not args.allow_partial:
            print(f"[skip] {fig}: missing {len(info['missing'])}/{info['depth']} agent zips ({info['missing'][:3]}...)")
            skipped.append((fig, "missing-zips")); continue
        if len(chain) < 2:
            print(f"[skip] {fig}: <2 keyframes"); skipped.append((fig, "short")); continue
        color = bool(info.get("color_enabled", False))
        out = args.out_dir / f"{fig_base}__{run[:24]}.mp4"   # name by fig, not archive id, to keep slots distinct
        use_pub = args.pace in ("publications", "distance-floor")
        n, canon_pos = render_chain(chain, out, config, panel=args.panel, img_res=args.img_res, fps=args.fps,
                         budget=args.budget, min_seg=args.min_seg, max_seg=args.max_seg,
                         hold_keyframe=args.hold_keyframe, hold_end=args.hold_end,
                         variant=("color" if color else "gray"), color_enabled=color,
                         pub_idx=(info["pub_canon_idx"] if use_pub else None),
                         seg_budget=args.seg_budget, pace=args.pace, grain=args.grain, min_gap=args.min_gap)
        if args.sidecar:
            meta = json.loads((run_dir / "archive/archive_metadata.json").read_text())["entries"]
            by = {e["id"]: e for e in meta}
            pubs = [{"f": int(canon_pos[ci] if ci < len(canon_pos) else n - 1),
                     "title": _title(by.get(info["branch"][k], {})), "id": info["branch"][k]}
                    for k, ci in enumerate(info["pub_canon_ordered"])]
            sc = {}
            if args.sidecar.exists():
                try: sc = json.loads(args.sidecar.read_text())
                except Exception: sc = {}
            sc[fig_base] = {"clip": out.name, "n_total": n, "hold_end": args.hold_end, "pubs": pubs}
            args.sidecar.write_text(json.dumps(sc, indent=2))
        print(f"[ok]   {fig}: depth={info['depth']} keyframes={len(chain)} "
              f"pubs={len(info['pub_canon_idx'])} frames={n} -> {out}")
        done.append(fig)

    print(f"\nDONE {len(done)}; skipped {len(skipped)}")
    for f, why in skipped:
        print(f"  skip {f}: {why}")


if __name__ == "__main__":
    main()
