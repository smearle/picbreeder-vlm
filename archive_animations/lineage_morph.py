#!/usr/bin/env python3
"""Render the human and VLM "car" lineages as looping CPPN-morph GIFs, to sit
ABOVE the matching reflowing lineage strips in the blog post.

Two independent CPPN substrates, two code paths:

* VLM  -- NEAT ``PicbreederGenome`` pickles (the 9 published archive genomes of
  the s8 gemini-2.5-pro run, traced root->tip via ``source_entry_ids[0]``,
  exactly the strip shown in the post). Morphed with the topology-matching
  interpolation in ``render_lineage_animation`` (same as cppn_interp). Each
  published keyframe carries the title the model gave it; titles are baked into
  the GIF in Latin Modern italic and crossfade between keyframes, matching the
  banner's hover-title style.

* Human -- original Picbreeder genomes (the PID-464 ancestry, root->car via
  ``branchFrom``). These are a different CPPN format (historical-marking graphs
  loaded by ``fer.src.save_lineage_figures.load_pbcppn``); we interpolate them
  in the union of their nodes/links (markings are stable across the lineage, so
  a shared link lerps its weight and a not-yet-born link fades in from 0). No
  titles -- the human archive images are untitled.

Frames per segment are allocated in proportion to visual change so a big
conceptual leap gets more frames than a tiny tweak (uniform change-per-second).

Usage:
    .venv/bin/python archive_animations/lineage_morph.py \
        --out-dir /home/jupyter-smearle/smearle.github.io/picbreeder-vlm-06b0d76d/assets/lineages
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "archive_animations"))

import cppn_interp as ci  # noqa: E402
from render_lineage_animation import build_superset, render_frames  # noqa: E402

# ---- the two lineages (identical provenance to extract_lineage_assets.py) ----
VLM_GENOME_DIR = REPO / "archive_animations/_lineage_genomes/vlm"
VLM_CHAIN = [  # root -> tip, with the model's published titles
    ("img_000002", "Chrome Bird"),
    ("img_000016", "Driver's Seat"),
    ("img_000123", "Dashboard View"),
    ("img_000148", "Vintage Chrome"),
    ("img_000166", "Chrome Insignia"),
    ("img_000194", "Vintage Hood Ornament"),
    ("img_000782", "Chrome Roadster"),
    ("img_000803", "Chrome Speedster"),
    ("img_000810", "Chrome Streamliner"),
]
HUMAN_ROOT = REPO / "fer/spaghetti/pbRender/genomeAll"
HUMAN_CHAIN = ["57", "86", "167", "372", "375", "464"]  # root -> car (PID 464)

FONT_PATH = Path(  # italic -- reserved for image titles (the VLM clip)
    "/home/jupyter-smearle/smearle.github.io/picbreeder-vlm-06b0d76d/assets/fonts/lmroman10-italic.otf"
)
REGULAR_FONT_PATH = REPO / "archive_animations/assets/fonts/lmroman10-regular.otf"  # upright -- metadata (Age / Published)


# --------------------------------------------------------------------------- #
# frame-budget allocation (frames per segment proportional to visual distance)
# --------------------------------------------------------------------------- #
def allocate_steps(canon_imgs, budget, min_seg, max_seg):
    dists = [ci.visual_distance(canon_imgs[i], canon_imgs[i + 1]) for i in range(len(canon_imgs) - 1)]
    total = sum(dists) or 1.0
    return [int(min(max_seg, max(min_seg, round(min_seg + budget * d / total)))) for d in dists]


def smoothstep(t: float) -> float:
    t = min(1.0, max(0.0, t))
    return t * t * (3 - 2 * t)


# --------------------------------------------------------------------------- #
# VLM: NEAT genomes
# --------------------------------------------------------------------------- #
VLM_RUN = REPO / "sweep_logs/sweep/th1_ag20_model-gemini-2.5-pro_tb-1_scheme-toggle_nopersonalities_fixed-sesh_s8"
VLM_AGENT_DIR = REPO / "archive_animations/_lineage_genomes/vlm_agents"


def vlm_full_chain():
    """Full per-mutation VLM lineage root->car, the same way teaser_lineages does:
    follow the published image's ``source_entry_ids[0]`` chain, and for each
    published image concatenate its session's true parent->child genome lineage
    (``cppn_interp.load_lineage_chain``). Dedupe consecutive identical genomes.

    Returns (genomes, meta) root->tip, where meta[i] = (age, title, published):
    age = 1-based step along the lineage, title = the model's published title (only
    on the published keyframes; "" elsewhere), published = is a published archive
    image. So the morph ticks an Age on every frame and surfaces title+Published at
    the 9 publication milestones -- mirroring the human Age/Published treatment."""
    import json
    import tempfile
    import zipfile

    meta_entries = json.loads((VLM_RUN / "archive/archive_metadata.json").read_text())["entries"]
    by = {e["id"]: e for e in meta_entries}
    chain_ids, cur, seen = [], VLM_CHAIN[-1][0], set()
    while cur and cur in by and cur not in seen:
        seen.add(cur); chain_ids.append(cur)
        src = by[cur].get("source_entry_ids") or []
        cur = str(src[0]) if src else None
    chain_ids.reverse()
    title_of = {iid: t for (iid, t) in VLM_CHAIN}

    wd = Path(tempfile.mkdtemp())
    full, pub_full_pos = [], []
    for iid in chain_ids:
        aid = by[iid]["agent_id"]
        with zipfile.ZipFile(VLM_AGENT_DIR / f"{aid}.zip") as zf:
            zf.extractall(wd)
        seg = ci.load_lineage_chain(wd / aid)
        full += seg
        pub_full_pos.append((len(full) - 1, iid))  # this session's published genome = its last

    canon, full2canon, last_sig = [], [], None
    for _, g in full:
        sig = ci.genome_signature(g)
        if sig != last_sig:
            canon.append(g); last_sig = sig
        full2canon.append(len(canon) - 1)
    pub_canon = {full2canon[p]: iid for p, iid in pub_full_pos}

    meta = []
    for idx in range(len(canon)):
        iid = pub_canon.get(idx)
        meta.append((idx + 1, title_of.get(iid, "") if iid else "", iid is not None))
    return canon, meta


def render_vlm_frames(config, img_res, budget, min_seg, max_seg, hold_pub):
    genomes, meta = vlm_full_chain()
    canon = [ci.canon_frame(g, config, img_res, "gray", False) for g in genomes]
    steps = allocate_steps(canon, budget, min_seg, max_seg)

    frames, frame_kf = [canon[0]], [0.0]
    for i in range(len(genomes) - 1):
        ss, npairs, cpairs, outs = build_superset(genomes[i], genomes[i + 1])
        seg = render_frames(ss, npairs, cpairs, config, steps=max(2, steps[i]),
                            width=img_res, height=img_res, variant_mode="gray",
                            color_enabled=False, output_activation_stats=outs)
        for k, fr in enumerate(seg[1:-1], start=1):
            frames.append(fr); frame_kf.append(i + k / (max(2, steps[i]) - 1))
        frames.append(canon[i + 1]); frame_kf.append(float(i + 1))
        for _ in range(hold_pub if meta[i + 1][2] else 0):  # dwell on publications
            frames.append(canon[i + 1]); frame_kf.append(float(i + 1))
    return frames, frame_kf, meta


# --------------------------------------------------------------------------- #
# Human: original Picbreeder pbcppn genomes (union-of-markings interpolation)
# --------------------------------------------------------------------------- #
def _ident_key(ident):
    return (ident["@branch"], ident["@id"]) if isinstance(ident, dict) else None


def human_chain():
    """Full per-age ancestral chain root->car (PID 464), one genome per age.

    We pool every genome stored across the branchFrom lineage, then trace the
    final published rep back through its single ``parents`` pointer -- giving a
    contiguous one-mutation-per-step chain (ages 1..106), NOT just the 6
    published reps. Each entry is flagged ``published`` when its identifier is a
    PID's published representative (``rep.zip``), which is also what lives in the
    human Picbreeder published archive. Returns [(genome, age, published), ...]."""
    from fer.src.picbreeder_util import load_zip_xml_as_dict
    from fer.src import lineage_utils as lu

    pool = {}
    for pid in HUMAN_CHAIN:
        for arch in lu.iter_generation_archives(HUMAN_ROOT, pid):
            root = load_zip_xml_as_dict(str(arch))["genome"]
            gen = root["generation"] if "storage" not in root else root["storage"]["generation"]
            for g in lu.recursive_parse_all_genomes(gen):
                k = _ident_key(g.get("identifier"))
                if k:
                    pool.setdefault(k, g)
    rep_keys = set()
    tip = None
    for pid in HUMAN_CHAIN:
        g = load_zip_xml_as_dict(str(HUMAN_ROOT / pid / "rep.zip"))["genome"]
        k = _ident_key(g.get("identifier"))
        pool.setdefault(k, g)
        rep_keys.add(k)
        tip = k  # last PID in the chain (464) = the car

    chain, cur, seen = [], tip, set()
    while cur and cur in pool and cur not in seen:
        seen.add(cur)
        chain.append(pool[cur])
        par = pool[cur].get("parents")
        cur = _ident_key(par.get("identifier")) if isinstance(par, dict) else None
    chain.reverse()
    return [(g, int(g.get("@age", 0)), _ident_key(g.get("identifier")) in rep_keys) for g in chain]


def load_human_nn(pid: str):
    from fer.src.picbreeder_util import load_zip_xml_as_dict
    from fer.src.save_lineage_figures import load_pbcppn
    genome = load_zip_xml_as_dict(str(HUMAN_ROOT / pid / "rep.zip"))["genome"]
    return load_pbcppn(genome)


def render_pbcppn(nn, res):
    from fer.src.save_lineage_figures import do_forward_pass
    rgb = np.asarray(do_forward_pass(nn, res=res)["rgb"], np.float32)
    return Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8), "RGB")


def interp_pbcppn(nn_a, nn_b, t):
    """Interpolate two pbcppn graphs in the union of their nodes/links.

    A connection is keyed by its (source, target) node pair -- the link marking
    ``@id`` is NOT unique within a genome (it repeats across branches), whereas a
    CPPN has at most one edge per ordered node pair. A connection in both
    endpoints lerps its weight; one in only a single endpoint fades to/from 0
    (its node sits disconnected until the connection ramps in). Node ids
    (``branch_id``) are globally unique, so nodes union cleanly."""
    nodes = {}
    for nn in (nn_a, nn_b):
        for n in nn["nodes"]:
            nodes.setdefault(n["id"], n)

    def lk(l):
        return (l["source"], l["target"])

    wa = {lk(l): l["weight"] for l in nn_a["links"]}
    wb = {lk(l): l["weight"] for l in nn_b["links"]}
    links_by_pair = {}
    for nn in (nn_a, nn_b):
        for l in nn["links"]:
            links_by_pair.setdefault(lk(l), l)
    links = []
    for key, l in links_by_pair.items():
        w = (1 - t) * wa.get(key, 0.0) + t * wb.get(key, 0.0)
        links.append({**l, "weight": w})
    return {"nodes": list(nodes.values()), "links": links,
            "special_nodes": nn_b["special_nodes"]}


def render_human_frames(img_res, budget, min_seg, max_seg, hold_pub):
    """Morph through the full per-age chain; hold on published keyframes so the
    "Published" tag is readable. Returns (frames, frame_kf, meta) where frame_kf
    is the fractional keyframe coordinate of each frame and meta=[(age, pub)]."""
    chain = human_chain()
    nns = [load_pbcppn_from_chain(g) for g, _, _ in chain]
    canon = [render_pbcppn(nn, img_res) for nn in nns]
    steps = allocate_steps(canon, budget, min_seg, max_seg)
    meta = [(age, pub) for _, age, pub in chain]

    frames, frame_kf = [canon[0]], [0.0]
    for _ in range(hold_pub if meta[0][1] else 0):
        frames.append(canon[0]); frame_kf.append(0.0)
    for i in range(len(nns) - 1):
        n = max(2, steps[i])
        for k in range(1, n - 1):  # interior frames only; endpoints are canon
            # LINEAR t -- smoothstep here would race through the middle of a
            # segment, spiking the per-frame change on big single-mutation jumps.
            frames.append(render_pbcppn(interp_pbcppn(nns[i], nns[i + 1], k / (n - 1)), img_res))
            frame_kf.append(i + k / (n - 1))
        frames.append(canon[i + 1]); frame_kf.append(float(i + 1))
        for _ in range(hold_pub if meta[i + 1][1] else 0):  # dwell on published milestones
            frames.append(canon[i + 1]); frame_kf.append(float(i + 1))
    return frames, frame_kf, meta


def load_pbcppn_from_chain(genome):
    from fer.src.save_lineage_figures import load_pbcppn
    return load_pbcppn(genome)


# --------------------------------------------------------------------------- #
# compositing + GIF output
# --------------------------------------------------------------------------- #
def caption_blend(kf_coord, n_kf):
    """(i, x): blend caption strips cap[i] and cap[i+1] by x. Each caption is
    held across the first 30% of its segment, then crossfades through a steep
    smoothstep -- the banner's S-curve, which avoids lingering mid-fade. On a
    held keyframe (integer coord) x=0, so that keyframe's caption shows cleanly."""
    if n_kf <= 1:
        return 0, 0.0
    i = int(min(max(0, int(kf_coord)), n_kf - 2))
    frac = kf_coord - i
    return i, smoothstep((frac - 0.3) / 0.4)


def caption_block_height(fonts, panel):
    gap = max(2, int(panel * 0.014))
    return sum(sum(f.getmetrics()) for f in fonts) + gap * (len(fonts) - 1)


def render_caption(panel, title_h, lines):
    """White-background RGB caption strip. `lines` is a list of (text, font,
    shade) placed at FIXED vertical slots (one per entry, sized by font metrics)
    -- an empty text still reserves its slot, so e.g. the "Age" line stays put
    whether or not a title/"Published" line is present. Crossfading two such
    white-bg strips by alpha reproduces the banner's opacity crossfade."""
    img = Image.new("RGB", (panel, title_h), "white")
    draw = ImageDraw.Draw(img)
    gap = max(2, int(panel * 0.014))
    line_h = [sum(f.getmetrics()) for _, f, _ in lines]
    y = (title_h - (sum(line_h) + gap * (len(lines) - 1))) / 2
    for (t, f, shade), h in zip(lines, line_h):
        if t:
            b = draw.textbbox((0, 0), t, font=f)
            draw.text(((panel - (b[2] - b[0])) / 2 - b[0], y), t, font=f, fill=(shade, shade, shade))
        y += h + gap
    return img


def compose(frames, panel, *, frame_kf=None, caps=None, title_h=0):
    """Place each morph frame (rendered at `panel` px, so never upscaled) onto a
    white panel; optionally bake a crossfading caption strip underneath, driven
    by each frame's fractional keyframe coordinate."""
    out = []
    H = panel + (title_h if caps else 0)
    cap_arr = [np.asarray(c, np.float32) for c in caps] if caps else None
    for idx, fr in enumerate(frames):
        img = fr.convert("RGB")
        if img.size != (panel, panel):
            img = img.resize((panel, panel), Image.LANCZOS)
        if not caps or title_h <= 0:
            out.append(img)
            continue
        canvas = Image.new("RGB", (panel, H), "white")
        canvas.paste(img, (0, 0))
        i, x = caption_blend(frame_kf[idx], len(cap_arr))
        a = cap_arr[i]
        b = cap_arr[i + 1] if i + 1 < len(cap_arr) else a
        strip = np.clip(a * (1 - x) + b * x, 0, 255).astype(np.uint8)
        canvas.paste(Image.fromarray(strip, "RGB"), (0, panel))
        out.append(canvas)
    return out


def save_gif(frames, path: Path, fps, hold_end, colors):
    path.parent.mkdir(parents=True, exist_ok=True)
    seq = list(frames) + [frames[-1]] * hold_end
    # One shared palette for the whole clip (the morphs are grayscale, so a small
    # color count posterizes cleanly and compresses far better than per-frame
    # adaptive palettes + dithering). No dither -> no high-entropy speckle.
    sample = Image.new("RGB", (seq[0].width, seq[0].height * len(seq)))
    for i, f in enumerate(seq):
        sample.paste(f, (0, f.height * i))
    pal = sample.quantize(colors=colors, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    pframes = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in seq]
    dur = int(round(1000 / fps))
    pframes[0].save(path, save_all=True, append_images=pframes[1:], duration=dur,
                    loop=0, optimize=True, disposal=2)
    kb = path.stat().st_size / 1024
    print(f"  wrote {path}  ({len(seq)} frames, {kb:.0f} KB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path,
                    default=Path("/home/jupyter-smearle/smearle.github.io/picbreeder-vlm-06b0d76d/assets/lineages"))
    ap.add_argument("--panel", type=int, default=220)
    ap.add_argument("--img-res", type=int, default=0,
                    help="CPPN render resolution; 0 = panel (never upscale the morph)")
    ap.add_argument("--fps", type=int, default=20)
    # both lineages are now full per-mutation chains (VLM ~64 steps, human ~106),
    # so each gets a modest per-segment budget: big visual jumps earn extra
    # interpolation frames, tiny tweaks cut straight to the next genome.
    ap.add_argument("--vlm-budget", type=int, default=200)
    ap.add_argument("--vlm-min-seg", type=int, default=2)
    ap.add_argument("--vlm-max-seg", type=int, default=16)
    ap.add_argument("--human-budget", type=int, default=240)
    ap.add_argument("--human-min-seg", type=int, default=2)
    ap.add_argument("--human-max-seg", type=int, default=14)
    ap.add_argument("--hold-pub", type=int, default=10, help="frames to dwell on each published keyframe")
    ap.add_argument("--hold-end", type=int, default=20)
    ap.add_argument("--colors", type=int, default=64, help="GIF palette size (grayscale posterization)")
    ap.add_argument("--which", choices=["both", "vlm", "human"], default="both")
    args = ap.parse_args()
    img_res = args.img_res or args.panel

    # italic = image titles (reserved for titles); upright = metadata (Age / Published)
    title_font = ImageFont.truetype(str(FONT_PATH), int(args.panel * 0.085))
    meta_font = ImageFont.truetype(str(REGULAR_FONT_PATH), int(args.panel * 0.078))
    meta_sub_font = ImageFont.truetype(str(REGULAR_FONT_PATH), int(args.panel * 0.054))

    if args.which in ("vlm", "both"):
        print("VLM lineage morph ...")
        config = ci.build_config()
        frames, frame_kf, meta = render_vlm_frames(config, img_res, args.vlm_budget,
                                                    args.vlm_min_seg, args.vlm_max_seg, args.hold_pub)
        print(f"  {len(meta)} steps, {sum(p for _, _, p in meta)} published, {len(frames)} frames")
        # 3 fixed slots: italic title / Age / Published -- title+Published only on
        # publication milestones, Age ticks on every step.
        title_h = caption_block_height([title_font, meta_font, meta_sub_font], args.panel) + int(args.panel * 0.07)
        caps = [render_caption(args.panel, title_h,
                               [(title, title_font, 34), (f"Age {age}", meta_font, 34),
                                ("Published" if pub else "", meta_sub_font, 120)])
                for age, title, pub in meta]
        comp = compose(frames, args.panel, frame_kf=frame_kf, caps=caps, title_h=title_h)
        save_gif(comp, args.out_dir / "vlm_morph.gif", args.fps, args.hold_end, args.colors)

    if args.which in ("human", "both"):
        print("Human lineage morph ...")
        frames, frame_kf, meta = render_human_frames(img_res, args.human_budget,
                                                      args.human_min_seg, args.human_max_seg, args.hold_pub)
        print(f"  {len(meta)} ages ({meta[0][0]}..{meta[-1][0]}), "
              f"{sum(p for _, p in meta)} published, {len(frames)} frames")
        # 2 fixed slots: Age / Published (human archive images have no titles)
        title_h = caption_block_height([meta_font, meta_sub_font], args.panel) + int(args.panel * 0.07)
        caps = [render_caption(args.panel, title_h,
                               [(f"Age {age}", meta_font, 34), ("Published" if pub else "", meta_sub_font, 120)])
                for age, pub in meta]
        comp = compose(frames, args.panel, frame_kf=frame_kf, caps=caps, title_h=title_h)
        save_gif(comp, args.out_dir / "human_morph.gif", args.fps, args.hold_end, args.colors)


if __name__ == "__main__":
    main()
