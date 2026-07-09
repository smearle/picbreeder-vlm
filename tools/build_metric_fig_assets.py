#!/usr/bin/env python3
"""Build the image/scatter/meta assets for the three eval-metric pipeline figures
(figures/metric_figs/{semantic_recall,visual_novelty,semantic_novelty}.tex), in the
visual style of the system-overview figure.

All assets are derived from ONE clean default run (gemini-2.5-pro, CL=1, no noise,
no personalities, 2000 published images) so that all three figures depict the same
archive. Everything is computed from precomputed caches already on disk -- no model
is loaded here.

Outputs (into --out-dir, default figures/metric_figs):

    archive_grid.png        white-padded mosaic of ~100 archive images (shared source)

    recall_match_{0..N}.png best-matching archive image for each curated THINGS noun
    _recall_meta.tex        \def\recallCells{i/noun/sim, ...} + \def\recallScore

    visual_scatter.png      UMAP-2D of SigLIP2-alignet image embeddings, with k FPS
                            centers + equal-radius covering circles (illustrative k;
                            the reported k=100 radius goes in the score box)

    cap_thumb_{0..2}.png    a few archive thumbnails for the caption examples
    _semantic_meta.tex      \def\captionCells{i/caption, ...} + \def\semanticScore +
                            \def\visualScore
    semantic_scatter.png    UMAP-2D of gemini caption embeddings, same FPS treatment

Run:
    .venv/bin/python tools/build_metric_fig_assets.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.offsetbox import OffsetImage, AnnotationBbox, TextArea, VPacker
from matplotlib.font_manager import FontProperties

REPO = Path(__file__).resolve().parent.parent

DEFAULT_RUN = (
    REPO / "sweep_logs/sweep/"
    "th1_ag20_model-gemini-2.5-pro_tb-1_scheme-toggle_nopersonalities_fixed-sesh_s6"
)

# --- recall config ---------------------------------------------------------
# Curated, recognizable THINGS nouns; the builder keeps the N with the highest
# image-similarity that map to *distinct* archive images (the "rediscovery" story).
RECALL_CANDIDATE_NOUNS = [
    "eye", "owl", "duck", "dolphin", "teapot", "seal", "butterfly",
    "parrot", "alligator", "cockatoo", "kettle", "puffin", "fish", "flower",
]
N_RECALL = 6

# --- GLOBAL recall grid config ---------------------------------------------
# The blog's Semantic Recall panel (grid_recall.png) is not drawn from a single
# run: it shows the 9 best noun->image matches across EVERY archive -- all 100+
# VLM sweep runs plus the full human Picbreeder archive -- each cell labelled
# with the run (hyper-params, or "human") that produced it. We take the REAL
# top-9 over the whole 1.8k-noun THINGS vocabulary (highest-similarity, distinct
# image per cell), the same ranking the easter-egg scroller reveals -- so the
# panel and the scroller's first nine match. (A curated "recognizable nouns"
# pool used to gate this; dropped so both views agree on the real top matches.)
N_GLOBAL_RECALL = 9

# Webli image-embedding cache filename shared by every archive (the joint
# SigLIP2 text-image space the Semantic Recall metric lives in).
WEBLI_CACHE = "image_embeddings_cache_ViT-SO400M-14-SigLIP2_webli.npy"
SWEEP_DIR = REPO / "sweep_logs/sweep"
# Human archive embeddings WITH filenames (built by tools/embed_human_webli.py;
# the plain webli cache omits names and lags the on-disk image set).
HUMAN_WEBLI_NPZ = REPO / "fer/src/archive_res-128/webli_emb_named.npz"
HUMAN_IMAGES = REPO / "fer/src/archive_res-128/images"

# Pretty model names for the per-cell condition label.
RECALL_MODEL_NAMES = {
    "gemini-2.5-pro": "Gemini 2.5 Pro",
    "gemini-2.5-flash": "Gemini 2.5 Flash",
    "gemini-2.5-flash-lite": "Gemini 2.5 Flash-Lite",
    "gemini-3-pro-preview": "Gemini 3 Pro",
    "gemini-3-pro": "Gemini 3 Pro",
    "qwen3-vl-30b-fp8": "Qwen3-VL 30B",
    "qwen3-vl-8b": "Qwen3-VL 8B",
}

# --- caption examples ------------------------------------------------------
# Curated, visually distinct + recognizable caption examples spread across the
# run (deliberately disjoint from the recall nouns). Falls back to an automatic
# spread if any are missing.
CAPTION_PICKS = ["img_000641.png", "img_000721.png", "img_001761.png"]
N_CAPTIONS = 3

# --- scatter / FPS config --------------------------------------------------
# Illustrative number of equal-radius covering spheres to draw (the reported
# metric uses k=100; a smaller k keeps the "grow spheres until covered" idea
# legible). UMAP params are tuned per embedding space for an evenly-tiled,
# connected layout; a tiny fraction of extreme outliers may be trimmed so the
# picture isn't dominated by stray points (the score box still reports the real
# full-archive k=100 radius). dict: k, n_neighbors, min_dist, trim, seed.
SCATTER_CFG = {
    "visual":   dict(k=8,  nn=15, min_dist=0.12, trim=0.0,  seed=0),
    "semantic": dict(k=15, nn=50, min_dist=0.5,  trim=0.01, seed=1),
    "recall":   dict(nn=15, min_dist=0.12, seed=0),  # image cloud only
}
SCATTER_DPI = 200

# Isometric-plane rendering (for the stacked-spaces figure). Each space is drawn
# as a slab sheared right + foreshortened vertically: iso(x,y) = (x+S*y, V*y).
ISO_SHEAR = 0.55
ISO_SQUASH = 0.5
# Every plane is drawn into the SAME fixed unit box and saved on the SAME fixed
# canvas, so all three slabs come out identical in size and angle (the stack
# figure then gets even vertical whitespace from equal anchor spacing). The cloud
# is fitted into the inner region; recall reserves extra left room for noun labels.
ISO_BOX_W = 1.0          # slab width  (pre-shear)
ISO_BOX_H = 0.82         # slab height (pre-shear)
ISO_PAD = 0.05           # padding around the slab within the canvas
ISO_LABEL_MARGIN = 0.34  # extra left room (canvas units) for recall noun labels


# ---------------------------------------------------------------------------- helpers


def _ordered_filenames(run: Path) -> List[str]:
    """Publication-ordered archive filenames (from the alignet npz, which was built
    in archive order and matches the recall .npy cache row-for-row)."""
    npz = np.load(run / "embeddings_openclip_SigLIP2-B-alignet.npz")
    return [str(x) for x in npz["filenames"]]


def _load_img(run: Path, fname: str, size: int) -> Image.Image:
    im = Image.open(run / "archive" / "images" / fname).convert("RGB")
    if im.size != (size, size):
        im = im.resize((size, size), Image.LANCZOS)
    return im


def build_archive_grid(run: Path, out: Path, n_side: int = 10, cell: int = 80,
                       pad: int = 3) -> None:
    fnames = _ordered_filenames(run)
    n = n_side * n_side
    idx = np.linspace(0, len(fnames) - 1, n).round().astype(int)
    canvas = Image.new("RGB", (n_side * cell + pad * (n_side + 1),) * 2, "white")
    for k, i in enumerate(idx):
        r, c = divmod(k, n_side)
        thumb = _load_img(run, fnames[i], cell)
        canvas.paste(thumb, (pad + c * (cell + pad), pad + r * (cell + pad)))
    canvas.save(out / "archive_grid.png")
    print(f"wrote {out/'archive_grid.png'}  ({n} imgs)")


# ---------------------------------------------------------------------------- recall


def _recall_chosen(run: Path, n: int = N_RECALL):
    """Return (filenames, image embeddings, noun embeddings, n_nouns, chosen).
    chosen = [(noun, noun_index, image_index, cosine_sim), ...] -- the curated,
    highest-sim, distinct-image set used by the recall figure and the iso stack."""
    fnames = _ordered_filenames(run)
    img_emb = np.load(run / "image_embeddings_cache_ViT-SO400M-14-SigLIP2_webli.npy")
    assert img_emb.shape[0] == len(fnames), (img_emb.shape, len(fnames))

    nouns = [ln.strip().replace("_", " ")
             for ln in (REPO / "noun_lists/things_deduped.txt").read_text().splitlines()
             if ln.strip()]
    noun_emb = np.load(
        REPO / "noun_lists/embeddings/"
        "noun_embeddings_things_deduped_ViT-SO400M-14-SigLIP2_webli.npy")
    assert noun_emb.shape[0] == len(nouns), (noun_emb.shape, len(nouns))
    noun_idx = {n: i for i, n in enumerate(nouns)}

    # similarity (embeddings are L2-normalized -> dot == cosine); per noun best image
    cands = []
    for noun in RECALL_CANDIDATE_NOUNS:
        if noun not in noun_idx:
            print(f"  [warn] noun not in list: {noun}")
            continue
        ni = noun_idx[noun]
        sims = img_emb @ noun_emb[ni]
        best = int(np.argmax(sims))
        cands.append((noun, ni, best, float(sims[best])))
    cands.sort(key=lambda x: -x[3])  # keep highest-sim with distinct images
    chosen = []
    used = set()
    for noun, ni, best, sim in cands:
        if best in used:
            continue
        used.add(best)
        chosen.append((noun, ni, best, sim))
        if len(chosen) == n:
            break
    return fnames, img_emb, noun_emb, len(nouns), chosen


def build_recall(run: Path, out: Path) -> None:
    fnames, img_emb, noun_emb, n_nouns, chosen = _recall_chosen(run)
    cells = []
    for i, (noun, _ni, best, sim) in enumerate(chosen):
        _load_img(run, fnames[best], 256).save(out / f"recall_match_{i}.png")
        cells.append(f"{i}/{noun}/{sim:.3f}")
        print(f"  recall[{i}] {noun:10s} sim={sim:.3f}  -> {fnames[best]}")

    # embedding-space node: joint text-image cloud with each curated noun linked
    # to its nearest image by a broken edge (the per-noun distance the metric sums).
    build_recall_scatter(img_emb, chosen, out / "recall_scatter.png")

    score = json.load(open(run / "noun_similarity_metrics.json"))["mean_max_similarity"]
    num_nouns_tex = f"{n_nouns:,}".replace(",", "{,}")  # LaTeX thousands sep
    meta = "".join([
        "% auto-generated by tools/build_metric_fig_assets.py\n",
        "\\def\\recallCells{" + ", ".join(cells) + "}\n",
        f"\\def\\recallScore{{{score:.3f}}}\n",
        f"\\def\\recallNumNouns{{{num_nouns_tex}}}\n",
    ])
    (out / "_recall_meta.tex").write_text(meta)
    print(f"wrote {out/'_recall_meta.tex'}  (score={score:.3f})")


# ---------------------------------------------------------------------------- FPS scatter


def _greedy_kcenter_2d(pts: np.ndarray, k: int) -> Tuple[np.ndarray, float]:
    """Hand-rolled greedy farthest-point sampling in 2D (mirrors the high-dim
    k-center used by compute_visual_coverage). Returns center indices and the
    covering radius (max over points of distance to nearest center)."""
    n = pts.shape[0]
    centroid = pts.mean(0)
    first = int(np.argmax(((pts - centroid) ** 2).sum(1)))
    centers = [first]
    mind = np.sqrt(((pts - pts[first]) ** 2).sum(1))
    for _ in range(1, k):
        nxt = int(np.argmax(mind))
        centers.append(nxt)
        d = np.sqrt(((pts - pts[nxt]) ** 2).sum(1))
        mind = np.minimum(mind, d)
    return np.array(centers), float(mind.max())


def _greedy_kcenter_highdim(emb: np.ndarray, k: int) -> Tuple[np.ndarray, float]:
    """Greedy farthest-point sampling in the FULL embedding space -- this is the
    same construction compute_k_covering uses for the actual Visual/Semantic
    Coverage metric. Returns center indices and the covering radius (max over
    points of distance to nearest center) in high-dim. `emb` is assumed
    L2-normalized; distances are Euclidean (monotone in cosine for unit vectors).
    Seeded from the point farthest from the centroid for determinism."""
    centroid = emb.mean(0)
    first = int(np.argmax(((emb - centroid) ** 2).sum(1)))
    centers = [first]
    mind = np.sqrt(((emb - emb[first]) ** 2).sum(1))
    for _ in range(1, k):
        nxt = int(np.argmax(mind))
        centers.append(nxt)
        d = np.sqrt(((emb - emb[nxt]) ** 2).sum(1))
        mind = np.minimum(mind, d)
    return np.array(centers), float(mind.max())


def _umap2d(emb: np.ndarray, nn: int, min_dist: float, seed: int) -> np.ndarray:
    import umap
    xy = umap.UMAP(n_components=2, random_state=seed,
                   n_neighbors=nn, min_dist=min_dist).fit_transform(emb)
    xy = np.asarray(xy, dtype=float)
    # scale to unit-ish box for stable circle radii
    xy -= xy.mean(0)
    xy /= (xy.std(0).mean() + 1e-9)
    return xy


def build_scatter(emb: np.ndarray, out_path: Path, cfg: dict, dot_color: str,
                  center_color: str, circle_color: str) -> None:
    # L2-normalize (defensive; caches already normalized)
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    xy = _umap2d(emb, cfg["nn"], cfg["min_dist"], cfg["seed"])
    if cfg["trim"] > 0:  # drop a few extreme outliers so the picture tiles cleanly
        rad = np.sqrt((xy ** 2).sum(1))
        xy = xy[rad <= np.quantile(rad, 1.0 - cfg["trim"])]
    centers, radius = _greedy_kcenter_2d(xy, cfg["k"])

    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    ax.scatter(xy[:, 0], xy[:, 1], s=6, c=dot_color, alpha=0.30,
               linewidths=0, zorder=1)
    for ci in centers:
        ax.add_patch(Circle((xy[ci, 0], xy[ci, 1]), radius, facecolor=circle_color,
                            edgecolor=circle_color, lw=1.1, alpha=0.13, zorder=2))
        ax.add_patch(Circle((xy[ci, 0], xy[ci, 1]), radius, facecolor="none",
                            edgecolor=circle_color, lw=1.0, alpha=0.55, zorder=3))
    ax.scatter(xy[centers, 0], xy[centers, 1], s=46, c=center_color,
               edgecolors="white", linewidths=0.8, zorder=4)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    pad = radius * 1.05
    ax.set_xlim(xy[:, 0].min() - pad, xy[:, 0].max() + pad)
    ax.set_ylim(xy[:, 1].min() - pad, xy[:, 1].max() + pad)
    fig.tight_layout(pad=0.1)
    fig.savefig(out_path, dpi=SCATTER_DPI, transparent=True,
                bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"wrote {out_path}  (k={cfg['k']}, 2D radius={radius:.3f}, n={xy.shape[0]})")


def _dotted_edge(ax, p0, p1, color: str, lw: float = 1.0) -> None:
    """A plain dotted line from p0 to p1 -- the noun -> nearest-image distance."""
    p0 = np.asarray(p0, float); p1 = np.asarray(p1, float)
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=color, lw=lw, alpha=0.75,
            zorder=3, linestyle=(0, (1, 1.6)), solid_capstyle="round")


def _recall_noun_xy(img_xy: np.ndarray, chosen):
    """SCHEMATIC noun placement. In the real joint SigLIP2 space the text modality
    forms its OWN island, far from every image (cross-modal cosine ~0.04 -- nearly
    orthogonal), so a literal projection collapses all nouns onto one point. We
    instead fake a small noun island just left of the image cloud and place each
    hero noun there as a star, then draw a line across to its nearest IMAGE point
    (the per-noun recall distance the metric sums). Positions are illustrative,
    spread for legibility; stars are staggered (not a rigid column) and ordered by
    their match's height to minimise edge crossings. Returns noun_xy aligned to
    `chosen`."""
    xmn, xmx = img_xy[:, 0].min(), img_xy[:, 0].max()
    ymn, ymx = img_xy[:, 1].min(), img_xy[:, 1].max()
    w, h = xmx - xmn, ymx - ymn
    cx = xmn - 0.20 * w                       # island centre: left of the cloud
    cy = 0.5 * (ymn + ymx)                     # column centred on the cloud
    n = len(chosen)
    order = sorted(range(n), key=lambda k: -img_xy[chosen[k][2], 1])
    # Spread the stars over a vertical span tied to the cloud's WIDTH, not its
    # height: the image cloud is wide + short, so the aspect-preserving fit is
    # width-driven and a height-based column collapses to an unreadable band.
    half = 0.40 * w
    top, bot = cy + half, cy - half
    noun_xy = np.zeros((n, 2))
    for slot, k in enumerate(order):
        frac = (slot + 0.5) / n
        y = top - frac * (top - bot)
        x = cx + (0.025 * w if slot % 2 else -0.025 * w)  # slight stagger, not a line
        noun_xy[k] = (x, y)
    return noun_xy


def _draw_recall(ax, img_xy: np.ndarray, noun_xy: np.ndarray, chosen, iso: bool) -> None:
    """Shared recall overlay: image cloud (dots) + the noun set, each star placed
    next to the image it is recalled by and linked to it by a dotted line (length =
    relative distance). iso shears everything onto the slab. Coordinates are passed
    in already-laid-out. Labels are pushed OUTWARD (away from the cloud centroid)
    so they don't sit on top of the image dots."""
    T = _iso if iso else (lambda z: np.asarray(z, float))
    P = T(img_xy)
    cen = P.mean(0)
    ax.scatter(P[:, 0], P[:, 1], s=5, c="#7a52c0", alpha=0.20, linewidths=0, zorder=1)
    for k, (noun, _ni, best, _sim) in enumerate(chosen):
        npt = T(noun_xy[k:k + 1])[0]
        ipt = T(img_xy[best:best + 1])[0]
        _dotted_edge(ax, npt, ipt, "#5a2ca0")
        # nearest image = a dot; noun = a star
        ax.scatter([ipt[0]], [ipt[1]], s=26, c="#5a2ca0", marker="o",
                   edgecolors="white", linewidths=0.6, zorder=5)
        ax.scatter([npt[0]], [npt[1]], s=95, c="#3a1a70", marker="*",
                   edgecolors="white", linewidths=0.7, zorder=5)
        d = npt - cen
        nrm = float(np.linalg.norm(d))
        u = d / nrm if nrm > 1e-9 else np.array([1.0, 0.0])
        ha = "left" if u[0] >= 0 else "right"
        ax.annotate(noun, (npt[0], npt[1]), xytext=(8 * u[0], 8 * u[1]),
                    textcoords="offset points", fontsize=10, fontstyle="italic",
                    color="#3a1a70", ha=ha, va="center", zorder=6)


def build_recall_scatter(img_emb: np.ndarray, chosen, out_path: Path) -> None:
    """Upright joint text-image space node for the standalone recall figure."""
    img_xy = _proj_xy(img_emb, SCATTER_CFG["recall"])
    noun_xy = _recall_noun_xy(img_xy, chosen)
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    _draw_recall(ax, img_xy, noun_xy, chosen, iso=False)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    allp = np.vstack([img_xy, noun_xy])
    px = (allp[:, 0].max() - allp[:, 0].min()) * 0.16
    py = (allp[:, 1].max() - allp[:, 1].min()) * 0.16
    ax.set_xlim(allp[:, 0].min() - px, allp[:, 0].max() + px)
    ax.set_ylim(allp[:, 1].min() - py, allp[:, 1].max() + py)
    fig.tight_layout(pad=0.1)
    fig.savefig(out_path, dpi=SCATTER_DPI, transparent=True,
                bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"wrote {out_path}  (recall: {len(chosen)} nouns + broken edges)")


def build_visual(run: Path, out: Path) -> None:
    npz = np.load(run / "embeddings_openclip_SigLIP2-B-alignet.npz")
    build_scatter(npz["embeddings"].astype(float), out / "visual_scatter.png",
                  SCATTER_CFG["visual"], dot_color="#4060a0",
                  center_color="#c01515", circle_color="#c01515")


# ---------------------------------------------------------------------------- semantic


def build_semantic(run: Path, out: Path) -> None:
    fnames = _ordered_filenames(run)
    caps = json.load(open(run / "archive/captions_gemini-2.5-pro.json"))
    cap_emb_d = json.load(open(run / "archive/embeddings_gemini-embedding-001_.json"))

    # caption embedding matrix in publication order
    keys = [f for f in fnames if f in cap_emb_d]
    emb = np.array([cap_emb_d[f] for f in keys], dtype=float)
    build_scatter(emb, out / "semantic_scatter.png", SCATTER_CFG["semantic"],
                  dot_color="#2e8b57", center_color="#b8860b",
                  circle_color="#b8860b")

    # caption examples: prefer the curated picks; fall back to an automatic
    # spread of short, distinct captions across the archive.
    chosen = [(f, caps[f]) for f in CAPTION_PICKS if f in caps]
    if len(chosen) < N_CAPTIONS:
        seen_words = {c.split()[1].lower() for _, c in chosen if len(c.split()) > 1}
        for f in fnames[::40]:
            c = caps.get(f, "")
            if not (25 <= len(c) <= 75):
                continue
            head = c.split()[1].lower() if len(c.split()) > 1 else c
            if head in seen_words:
                continue
            seen_words.add(head)
            chosen.append((f, c))
            if len(chosen) == N_CAPTIONS:
                break
    chosen = chosen[:N_CAPTIONS]

    cells = []
    for i, (f, c) in enumerate(chosen):
        _load_img(run, f, 256).save(out / f"cap_thumb_{i}.png")
        c_esc = c.replace("&", "\\&").replace("%", "\\%").replace("_", "\\_")
        cells.append(f"{i}/{{{c_esc}}}")
        print(f"  caption[{i}] {f}  {c!r}")

    vis_r = json.load(open(
        run / "embedding_mean_pairwise_distance_over_time_SigLIP2-B-alignet.json"))
    vis_last = vis_r[sorted(vis_r, key=lambda x: int(x))[-1]]["k_covering_radii"]["100"]
    sem_r = json.load(open(
        run / "archive/metrics_gemini-2.5-pro_gemini-embedding-001.json"))
    sem_last = sem_r[sorted(sem_r, key=lambda x: int(x))[-1]]["k_covering_radii"]["100"]

    meta = (
        "% auto-generated by tools/build_metric_fig_assets.py\n"
        f"\\def\\captionCells{{{', '.join(cells)}}}\n"
        f"\\def\\visualScore{{{vis_last:.3f}}}\n"
        f"\\def\\semanticScore{{{sem_last:.3f}}}\n"
    )
    (out / "_semantic_meta.tex").write_text(meta)
    print(f"wrote {out/'_semantic_meta.tex'}  (visual={vis_last:.3f}, semantic={sem_last:.3f})")


# ---------------------------------------------------------------------------- iso planes


def _iso(xy: np.ndarray) -> np.ndarray:
    """Shear right + foreshorten vertically: a square -> a leaning slab."""
    return np.column_stack([xy[:, 0] + ISO_SHEAR * xy[:, 1], ISO_SQUASH * xy[:, 1]])


def _iso_circle(cx: float, cy: float, r: float, m: int = 64) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, m)
    return _iso(np.column_stack([cx + r * np.cos(t), cy + r * np.sin(t)]))


def _proj_xy(emb: np.ndarray, cfg: dict) -> np.ndarray:
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    xy = _umap2d(emb, cfg["nn"], cfg["min_dist"], cfg["seed"])
    if cfg.get("trim", 0) > 0:
        rad = np.sqrt((xy ** 2).sum(1))
        xy = xy[rad <= np.quantile(rad, 1.0 - cfg["trim"])]
    return xy


def _iso_slab_corners() -> np.ndarray:
    """The four corners of the fixed slab in iso (display) coords -- identical for
    every plane, which is what makes all three slabs the same size and angle."""
    c = np.array([[0, 0], [ISO_BOX_W, 0], [ISO_BOX_W, ISO_BOX_H], [0, ISO_BOX_H]],
                 dtype=float)
    return _iso(c)


def _iso_fixed_limits():
    ic = _iso_slab_corners()
    return ((ic[:, 0].min() - ISO_LABEL_MARGIN, ic[:, 0].max() + ISO_PAD),
            (ic[:, 1].min() - ISO_PAD, ic[:, 1].max() + ISO_PAD))


def _fit_to_box(xy: np.ndarray, x0, x1, y0, y1):
    """Uniform-scale xy (aspect preserved) to sit centred in the rectangle.
    Returns (fitted_xy, scale)."""
    xy = np.asarray(xy, float)
    mn = xy.min(0); span = np.maximum(xy.max(0) - mn, 1e-9)
    s = min((x1 - x0) / span[0], (y1 - y0) / span[1])
    out = (xy - mn) * s
    out[:, 0] += x0 + ((x1 - x0) - span[0] * s) / 2
    out[:, 1] += y0 + ((y1 - y0) - span[1] * s) / 2
    return out, s


def _iso_fig_fixed():
    """A figure on the fixed canvas: same limits + same figsize for every plane,
    so the saved PNGs are pixel-identical in framing."""
    (x0, x1), (y0, y1) = _iso_fixed_limits()
    h = 1.9
    fig, ax = plt.subplots(figsize=(h * (x1 - x0) / (y1 - y0), h))
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
    return fig, ax


def _iso_border_fixed(ax, color: str):
    """Draw the fixed slab outline/fill (identical geometry on every plane)."""
    ic = _iso_slab_corners()
    ax.fill(ic[:, 0], ic[:, 1], facecolor="white", edgecolor="none",
            alpha=0.82, zorder=0)
    ax.fill(ic[:, 0], ic[:, 1], facecolor=color, edgecolor=color,
            lw=1.3, alpha=0.07, zorder=0.5)
    ax.plot(np.append(ic[:, 0], ic[0, 0]), np.append(ic[:, 1], ic[0, 1]),
            color=color, lw=1.3, alpha=0.85, zorder=6)


def _iso_inner_box(label_room: bool = False):
    """Inner region (pre-iso) that the point cloud is fitted into. Recall reserves
    extra left room so its noun labels stay inside the canvas."""
    x0 = 0.30 if label_room else 0.06
    return x0, ISO_BOX_W - 0.06, 0.07, ISO_BOX_H - 0.07


def _save_iso(fig, out_path: Path):
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(out_path, dpi=SCATTER_DPI, transparent=True)
    plt.close(fig)
    print(f"wrote {out_path}")


def render_iso_cover(xy: np.ndarray, centers: np.ndarray, radius: float,
                     out_path: Path, dot_color: str, center_color: str,
                     circle_color: str) -> None:
    xyf, s = _fit_to_box(xy, *_iso_inner_box())
    ixy = _iso(xyf)
    fig, ax = _iso_fig_fixed()
    _iso_border_fixed(ax, circle_color)
    ax.scatter(ixy[:, 0], ixy[:, 1], s=5, c=dot_color, alpha=0.30,
               linewidths=0, zorder=2)
    for ci in centers:
        poly = _iso_circle(xyf[ci, 0], xyf[ci, 1], radius * s)
        ax.fill(poly[:, 0], poly[:, 1], facecolor=circle_color, edgecolor=circle_color,
                lw=1.0, alpha=0.12, zorder=3)
        ax.plot(poly[:, 0], poly[:, 1], color=circle_color, lw=1.0, alpha=0.55, zorder=4)
    ax.scatter(ixy[centers, 0], ixy[centers, 1], s=34, c=center_color,
               edgecolors="white", linewidths=0.7, zorder=5)
    _save_iso(fig, out_path)


def render_iso_recall(img_xy: np.ndarray, chosen, out_path: Path) -> None:
    """Bottom-of-stack recall plane: image cloud (dots) + the noun set down the
    left (stars), each noun linked to its nearest image by a dotted line. Fitted
    into the SAME fixed slab as the other two planes (extra left room for labels)."""
    noun_xy = _recall_noun_xy(img_xy, chosen)
    comb, _s = _fit_to_box(np.vstack([img_xy, noun_xy]), *_iso_inner_box(label_room=True))
    img_f, noun_f = comb[:len(img_xy)], comb[len(img_xy):]
    fig, ax = _iso_fig_fixed()
    _iso_border_fixed(ax, "#7a52c0")
    _draw_recall(ax, img_f, noun_f, chosen, iso=True)
    _save_iso(fig, out_path)


def build_iso(run: Path, out: Path) -> None:
    # visual: SigLIP2-alignet image embeddings
    cfg = SCATTER_CFG["visual"]
    emb = np.load(run / "embeddings_openclip_SigLIP2-B-alignet.npz")["embeddings"].astype(float)
    xy = _proj_xy(emb, cfg)
    centers, _r = _greedy_kcenter_2d(xy, cfg["k"])
    render_iso_cover(xy, centers, _r, out / "visual_iso.png",
                     dot_color="#4060a0", center_color="#c01515", circle_color="#c01515")

    # semantic: gemini caption embeddings
    cfg = SCATTER_CFG["semantic"]
    fnames = _ordered_filenames(run)
    cap_emb_d = json.load(open(run / "archive/embeddings_gemini-embedding-001_.json"))
    emb = np.array([cap_emb_d[f] for f in fnames if f in cap_emb_d], dtype=float)
    xy = _proj_xy(emb, cfg)
    centers, _r = _greedy_kcenter_2d(xy, cfg["k"])
    render_iso_cover(xy, centers, _r, out / "semantic_iso.png",
                     dot_color="#2e8b57", center_color="#b8860b", circle_color="#b8860b")

    # recall: joint text-image space, nouns linked to nearest image by broken edges
    _fn, img_emb, _ne, _nn, chosen = _recall_chosen(run)
    img_xy = _proj_xy(img_emb, SCATTER_CFG["recall"])
    render_iso_recall(img_xy, chosen, out / "recall_iso.png")


# --------------------------------------------------------------- metric result grids
# Three panels of REAL archive results, one illustrating each metric (a row beneath
# the eval-metrics figure on the blog):
#   grid_visual.png    -- 2D atlas of all images, key-area representatives
#                         (Visual Coverage; see build_coverage_atlas)
#   grid_recall.png    -- the best noun->image match across EVERY archive, per cell
#                         labelled with its winning run (Semantic Recall)
#   grid_semantic.png  -- 2D atlas of all caption embeddings, key-area
#                         representatives + captions (Semantic Coverage)

GRID_SIDE = 3   # recall grid is still a 3x3

# Recall grid (grid_recall.png) page geometry, in inches. Hoisted to module level so
# the coverage atlases can match the recall panel's output aspect ratio (and thus its
# displayed height) — see build_global_recall_grid and build_coverage_atlas_aggregate.
RECALL_GRID_M = 0.06        # outer margin
RECALL_GRID_G = 0.07        # gutter between cells
RECALL_GRID_LB = 0.74       # per-cell label band (noun + 2 condition lines)
RECALL_GRID_W = 4.6         # figure width
_recall_cell = (RECALL_GRID_W - 2 * RECALL_GRID_M
                - (GRID_SIDE - 1) * RECALL_GRID_G) / GRID_SIDE
RECALL_GRID_H = (2 * RECALL_GRID_M + GRID_SIDE * _recall_cell
                 + GRID_SIDE * RECALL_GRID_LB + (GRID_SIDE - 1) * RECALL_GRID_G)
RECALL_GRID_ASPECT = RECALL_GRID_H / RECALL_GRID_W   # h/w ~= 1.483


# ---------------------------------------------------------- global recall grid


def _recall_condition_parts(source: str) -> List[str]:
    """Concise label lines for the run that won a recall cell: line 1 is the model
    (or "human"); line 2, when present, is the one knob that differs from the clean
    default (noise / memory / agents). Returned as 1-2 short lines so the caption
    wraps cleanly instead of breaking mid-phrase."""
    if source == "human":
        return ["human"]
    from hf_archive_push import parse_config  # tools/ is on sys.path
    cfg = parse_config(source)
    if cfg["model"] == "gemini-random":
        return ["random baseline"]
    model = RECALL_MODEL_NAMES.get(cfg["model"], cfg["model"])
    if cfg["personalities"]:
        return [model, f"{cfg['personalities']} agents"]
    if cfg["noise_eps"] and cfg["noise_eps"] > 0:
        return [model, f"noise {cfg['noise_eps']:g}"]
    cl = cfg["memory_cl"]
    knob = {0: "no memory", 2: "memory 2", 10: "memory 10", -1: "full memory"}.get(cl)
    return [model, knob] if knob else [model]  # th1/no-noise/no-traits == default


def _recall_image_path(source: str, fname: str) -> Path:
    if source == "human":
        return HUMAN_IMAGES / fname
    return SWEEP_DIR / source / "archive" / "images" / fname


def _iter_recall_sources():
    """Yield (source_key, image_embeddings, filenames) for every archive that has
    the joint webli cache: the human Picbreeder archive plus all VLM sweep runs."""
    if HUMAN_WEBLI_NPZ.exists():
        h = np.load(HUMAN_WEBLI_NPZ, allow_pickle=True)
        yield "human", h["embeddings"].astype(np.float32), [str(x) for x in h["filenames"]]
    else:
        print(f"  [warn] human webli cache missing ({HUMAN_WEBLI_NPZ}); "
              "run tools/embed_human_webli.py -- skipping human archive")
    for d in sorted(SWEEP_DIR.iterdir()):
        cache = d / WEBLI_CACHE
        if not cache.exists():
            continue
        emb = np.load(cache).astype(np.float32)
        fns = _ordered_filenames(d)
        n = min(emb.shape[0], len(fns))  # caches occasionally lag the npz by a row
        yield d.name, emb[:n], fns[:n]


def _global_recall_picks(n: int = N_GLOBAL_RECALL):
    """The real top-n noun->image matches across ALL archives: the n highest-
    similarity, distinct-image matches over the WHOLE THINGS vocabulary (not the
    old curated pool), returned as [(noun, source_key, filename, sim), ...]. This
    is the same ranking the easter-egg scroller shows (recall-egg.js), so the
    panel's nine thumbnails and the scroller's first nine agree."""
    nouns = [ln.strip().replace("_", " ")
             for ln in (REPO / "noun_lists/things_deduped.txt").read_text().splitlines()
             if ln.strip()]
    noun_emb = np.load(
        REPO / "noun_lists/embeddings/"
        "noun_embeddings_things_deduped_ViT-SO400M-14-SigLIP2_webli.npy").astype(np.float32)
    assert noun_emb.shape[0] == len(nouns), (noun_emb.shape, len(nouns))

    n_noun = len(nouns)
    best_sim = np.full(n_noun, -1.0, np.float32)
    best_src = [None] * n_noun
    best_fn = [None] * n_noun
    n_arch = 0
    for source, emb, fns in _iter_recall_sources():
        n_arch += 1
        sims = emb @ noun_emb.T              # (n_img, n_noun); embeddings L2-normalized
        mi = sims.argmax(0)
        mv = sims[mi, np.arange(n_noun)]
        upd = mv > best_sim
        for p in np.where(upd)[0]:
            best_sim[p] = mv[p]
            best_src[p] = source
            best_fn[p] = fns[mi[p]]
    print(f"  searched {n_arch} archives over {n_noun} nouns")

    ranked = sorted(range(n_noun), key=lambda p: -best_sim[p])
    picks, used = [], set()
    for p in ranked:
        key = (best_src[p], best_fn[p])
        if key in used:                       # distinct image per cell
            continue
        used.add(key)
        picks.append((nouns[p], best_src[p], best_fn[p], float(best_sim[p])))
        if len(picks) == n:
            break
    return picks


def build_global_recall_grid(out: Path) -> None:
    """Render grid_recall.png: the 9 best noun->image matches across every archive
    (all VLM runs + the human Picbreeder archive), each cell captioned with the
    noun (bold) and the winning run's condition. A square GRID_SIDE x GRID_SIDE of
    thumbnails, sized to sit beside the grid_visual / grid_semantic atlases."""
    picks = _global_recall_picks()
    for noun, src, fn, sim in picks:
        print(f"  recall  {noun:10s} sim={sim:.3f}  "
              f"{' / '.join(_recall_condition_parts(src))}  [{src}/{fn}]")

    side = GRID_SIDE
    M, G, lb = RECALL_GRID_M, RECALL_GRID_G, RECALL_GRID_LB
    fig_w, cell, fig_h = RECALL_GRID_W, _recall_cell, RECALL_GRID_H
    fig = plt.figure(figsize=(fig_w, fig_h))
    for k, (noun, src, fn, _sim) in enumerate(picks):
        r, c = divmod(k, side)
        x0 = (M + c * (cell + G)) / fig_w
        top_in = M + r * (cell + lb + G)
        y0 = 1 - (top_in + cell) / fig_h
        ax = fig.add_axes([x0, y0, cell / fig_w, cell / fig_h])
        ax.imshow(Image.open(_recall_image_path(src, fn)).convert("RGB"))
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        cx = x0 + cell / fig_w / 2
        fig.text(cx, y0 - 0.006, noun, ha="center", va="top",
                 fontsize=9.6, color="#3a1a70", fontstyle="italic", fontweight="bold")
        # model + knob on ONE line right under the noun (like the coverage panels)
        cond = " · ".join(_recall_condition_parts(src))
        fig.text(cx, y0 - 0.032, cond, ha="center", va="top",
                 fontsize=7.4, color="#666")
    fig.savefig(out / "grid_recall.png", dpi=200, facecolor="white")
    plt.close(fig)
    print(f"wrote {out/'grid_recall.png'}  ({len(picks)} cross-archive cells)")


# ------------------------------------------------------- coverage atlas panels
# Visual / Semantic Coverage are shown as a 2D projection of ALL archive items
# (UMAP), softly partitioned into k regions (nearest-representative Voronoi cells
# = "key areas"), with one representative thumbnail per region placed at the
# region's heart. The semantic panel projects caption-text embeddings and prints
# each representative's VLM caption beneath its thumbnail.

ATLAS_K = 6                       # number of key-area representatives
# Soft qualitative palette (one hue per region); thumbnails get the matching border.
ATLAS_PALETTE = ["#5B8FB0", "#C9745A", "#6FA86B", "#9A6FB0", "#C7A14A", "#4FA3A0"]


def _atlas_layout(emb: np.ndarray, nn: int, min_dist: float, seed: int, k: int,
                  trim: float = 0.0):
    """UMAP all points to 2D for DISPLAY ONLY, drop a few extreme outliers (so a
    detached satellite cluster doesn't dominate the frame -- illustrative, like
    build_scatter's trim), then choose the k representatives and the k regions by
    greedy k-center IN THE FULL EMBEDDING SPACE -- the same construction the actual
    Visual/Semantic Coverage metric (compute_k_covering) uses. Each point is
    assigned to its nearest representative in high-dim, so the coloured regions are
    the true covering cells; UMAP only places them on the page. (As a result, a
    region's points need not form a contiguous 2D blob -- that scatter is the
    honest signature of UMAP distorting high-dim distances.)
    Returns (xy, fnames_keep_idx, assign, reps): xy/assign cover the KEPT points;
    keep_idx maps kept-row -> original index for thumbnails."""
    embn = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    xy = _umap2d(embn, nn, min_dist, seed)
    keep = np.arange(xy.shape[0])
    if trim > 0:
        r = np.sqrt(((xy - np.median(xy, 0)) ** 2).sum(1))
        keep = np.where(r <= np.quantile(r, 1.0 - trim))[0]
        xy = xy[keep]
    embk = embn[keep]
    seeds, _r = _greedy_kcenter_highdim(embk, k)         # reps in TRUE metric space
    d2 = ((embk[:, None, :] - embk[seeds][None, :, :]) ** 2).sum(2)   # (N, k) high-dim
    assign = d2.argmin(1)                                 # high-dim Voronoi cells
    return xy, keep, assign, [int(s) for s in seeds]


def _wrap_caption_lines(c: str, width: int = 22, max_lines: int = 3) -> str:
    import textwrap
    lines = textwrap.wrap(c.strip().rstrip("."), width=width)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1].rstrip(".,") + "…"
    return "\n".join(lines)


def build_coverage_atlas(run: Path, out: Path, space: str) -> None:
    """Render grid_visual.png / grid_semantic.png as an embedding atlas: a 2D
    projection of every archive item, partitioned into key areas, with a
    representative thumbnail (+ caption, for semantic) per area."""
    fnames = _ordered_filenames(run)
    if space == "visual":
        emb = np.load(run / "embeddings_openclip_SigLIP2-B-alignet.npz")["embeddings"].astype(float)
        caps = None
        cfg = dict(nn=80, min_dist=0.9, seed=0, trim=0.03)  # space-filling atlas
        thumb_px, zoom = 150, 0.30
        out_name = "grid_visual.png"
    elif space == "semantic":
        caps_d = json.load(open(run / "archive/captions_gemini-2.5-pro.json"))
        cap_emb_d = json.load(open(run / "archive/embeddings_gemini-embedding-001_.json"))
        keys = [f for f in fnames if f in cap_emb_d]
        emb = np.array([cap_emb_d[f] for f in keys], dtype=float)
        fnames, caps = keys, caps_d
        cfg = dict(nn=50, min_dist=0.5, seed=1, trim=0.012)
        thumb_px, zoom = 124, 0.28
        out_name = "grid_semantic.png"
    else:
        raise ValueError(space)

    xy, keep, assign, reps = _atlas_layout(emb, k=ATLAS_K, **cfg)

    # Axes hug the full cloud so every point fills the frame; the representative
    # thumbnails sit in the figure MARGIN at fixed perimeter slots (so they never
    # overlap the cloud), each linked to its key area by a thin leader. Reps are
    # matched to the slot nearest their real direction to minimise crossings.
    cen = np.median(xy, 0)
    # 6 perimeter slots in axes-fraction coords (top, upper sides, lower sides, bottom)
    SLOTS = [(0.5, 1.22), (1.22, 0.74), (1.22, 0.24),
             (0.5, -0.22), (-0.22, 0.24), (-0.22, 0.74)]
    slot_ang = [np.arctan2(sy - 0.5, sx - 0.5) for sx, sy in SLOTS]
    rep_ang = [np.arctan2(xy[i, 1] - cen[1], xy[i, 0] - cen[0]) for i in reps]
    # assign reps to slots greedily by angular proximity (stable, no overlaps)
    rep_order = sorted(range(len(reps)), key=lambda r: rep_ang[r])
    slot_order = sorted(range(len(SLOTS)), key=lambda s: slot_ang[s])
    rep_slot = {rep_order[i]: slot_order[i] for i in range(len(reps))}

    fig_w = 4.6
    fig, ax = plt.subplots(figsize=(fig_w, fig_w))
    dot_c = np.array([ATLAS_PALETTE[a] for a in assign])
    ax.scatter(xy[:, 0], xy[:, 1], s=12, c=dot_c, alpha=0.50, linewidths=0, zorder=1)

    cap_font = FontProperties(size=6.4, style="italic")
    for r, idx in enumerate(reps):
        color = ATLAS_PALETTE[r]
        ax.scatter([xy[idx, 0]], [xy[idx, 1]], s=24, c=color,
                   edgecolors="white", linewidths=0.9, zorder=4)
        fname = fnames[keep[idx]]                       # kept-row -> original index
        thumb = _load_img(run, fname, thumb_px)
        img_box = OffsetImage(np.asarray(thumb), zoom=zoom)
        img_box.image.axes = ax
        if space == "semantic":
            txt = TextArea(_wrap_caption_lines(caps.get(fname, "")),
                           textprops=dict(fontproperties=cap_font, color="#555",
                                          ha="center", multialignment="center"))
            packed = VPacker(children=[img_box, txt], align="center", pad=1, sep=2)
        else:
            packed = img_box
        ab = AnnotationBbox(
            packed, (xy[idx, 0], xy[idx, 1]), xybox=SLOTS[rep_slot[r]],
            xycoords="data", boxcoords="axes fraction", frameon=True, pad=0.16,
            zorder=5, box_alignment=(0.5, 0.5), annotation_clip=False,
            bboxprops=dict(edgecolor=color, facecolor="white", linewidth=1.2),
            arrowprops=dict(arrowstyle="-", color=color, lw=0.9, alpha=0.6))
        ax.add_artist(ab)

    # 'auto' aspect: stretch each axis independently so the cloud fills a SQUARE
    # panel (UMAP distances are already non-metric, and all three blog panels must
    # share one row height). Thumbnails are in axes-fraction space, so unaffected.
    ax.set_aspect("auto")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    padx = np.ptp(xy[:, 0]) * 0.04
    pady = np.ptp(xy[:, 1]) * 0.04
    ax.set_xlim(xy[:, 0].min() - padx, xy[:, 0].max() + padx)
    ax.set_ylim(xy[:, 1].min() - pady, xy[:, 1].max() + pady)
    fig.savefig(out / out_name, dpi=200, facecolor="white",
                bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print(f"wrote {out/out_name}  ({space} atlas, {len(xy)} pts, {ATLAS_K} regions)")


# ------------------------------------------------ AGGREGATE coverage atlases
# Like build_coverage_atlas, but over the SAME aggregate set as the recall panel:
# every VLM sweep run + the full human Picbreeder archive, not one clean run. Each
# of the 9 region representatives is labelled with its source experiment (hyper-
# params, or "human"); the semantic panel adds the representative's caption.
#
# Parsing the caption embeddings (82 x ~80MB JSON) is slow, so we cache the
# subsampled aggregate to npz once (build_aggregate_caches) and re-render from it.

AGG_K = 9
N_AGG_SAMPLE = 24000            # points kept for the displayed cloud + UMAP + k-center
AGG_VISUAL_NPZ = "_agg_visual_webli.npz"
AGG_SEMANTIC_NPZ = "_agg_semantic_caption.npz"
# 9-hue qualitative palette (one per region).
ATLAS_PALETTE_9 = ["#5B8FB0", "#C9745A", "#6FA86B", "#9A6FB0", "#C7A14A",
                   "#4FA3A0", "#B0689A", "#7E8AC0", "#9C8A5A"]
HUMAN_CAPTION_EMB = REPO / "fer/src/archive_res-128/embeddings_gemini-embedding-001_.json"
HUMAN_CAPTIONS = REPO / "human_baseline/res-128/captions_human.json"


def _iter_caption_sources():
    """Yield (source_key, caption embeddings (n x 3072), filenames, captions) for
    every archive with cached caption embeddings: the human archive + all VLM runs
    that were captioned and caption-embedded."""
    if HUMAN_CAPTION_EMB.exists():
        ce = json.load(open(HUMAN_CAPTION_EMB))
        caps = json.load(open(HUMAN_CAPTIONS)) if HUMAN_CAPTIONS.exists() else {}
        keys = list(ce)
        emb = np.array([ce[k] for k in keys], dtype=np.float32)
        yield "human", emb, keys, [caps.get(k, "") for k in keys]
    for d in sorted(SWEEP_DIR.iterdir()):
        ce_path = d / "archive/embeddings_gemini-embedding-001_.json"
        if not ce_path.exists():
            continue
        cap_path = d / "archive/captions_gemini-2.5-pro.json"
        if not cap_path.exists():
            globbed = sorted(d.glob("archive/captions_*.json"))
            cap_path = globbed[0] if globbed else None
        ce = json.load(open(ce_path))
        caps = json.load(open(cap_path)) if cap_path else {}
        keys = list(ce)
        emb = np.array([ce[k] for k in keys], dtype=np.float32)
        yield d.name, emb, keys, [caps.get(k, "") for k in keys]


def build_aggregate_caches(out: Path) -> None:
    """Subsample the aggregate (all experiments) in each metric space and cache to
    npz, so the atlas render is fast and reproducible."""
    rng = np.random.default_rng(0)

    # visual: webli image space (same sources as the recall panel)
    embs, srcs, fns = [], [], []
    for source, emb, fnames in _iter_recall_sources():
        embs.append(emb); srcs += [source] * len(fnames); fns += list(fnames)
    E = np.vstack(embs).astype(np.float32)
    sel = rng.choice(E.shape[0], size=min(N_AGG_SAMPLE, E.shape[0]), replace=False)
    sel.sort()
    np.savez(out / AGG_VISUAL_NPZ, emb=E[sel],
             source=np.array(srcs)[sel], fname=np.array(fns)[sel])
    print(f"wrote {out/AGG_VISUAL_NPZ}  ({len(sel)}/{E.shape[0]} imgs, "
          f"{len(set(np.array(srcs)[sel]))} experiments)")
    del E, embs

    # semantic: gemini caption-embedding space
    embs, srcs, fns, caps = [], [], [], []
    for source, emb, fnames, captions in _iter_caption_sources():
        embs.append(emb); srcs += [source] * len(fnames)
        fns += list(fnames); caps += list(captions)
    E = np.vstack(embs).astype(np.float32)
    sel = rng.choice(E.shape[0], size=min(N_AGG_SAMPLE, E.shape[0]), replace=False)
    sel.sort()
    np.savez(out / AGG_SEMANTIC_NPZ, emb=E[sel], source=np.array(srcs)[sel],
             fname=np.array(fns)[sel], caption=np.array(caps, dtype=object)[sel])
    print(f"wrote {out/AGG_SEMANTIC_NPZ}  ({len(sel)}/{E.shape[0]} captions, "
          f"{len(set(np.array(srcs)[sel]))} experiments)")


def build_coverage_atlas_aggregate(out: Path, space: str) -> None:
    """Render grid_visual.png / grid_semantic.png as an aggregate embedding atlas:
    a 2D projection of a uniform sample of EVERY experiment's items, partitioned
    into 9 key areas, each with a representative thumbnail labelled by its source
    experiment (+ caption, for semantic)."""
    if space == "visual":
        npz = np.load(out / AGG_VISUAL_NPZ, allow_pickle=True)
        captions = None
        cfg = dict(nn=80, min_dist=0.9, seed=0, trim=0.03)
        thumb_px, zoom = 150, 0.50
        out_name = "grid_visual.png"
    elif space == "semantic":
        npz = np.load(out / AGG_SEMANTIC_NPZ, allow_pickle=True)
        captions = [str(c) for c in npz["caption"]]
        cfg = dict(nn=50, min_dist=0.5, seed=1, trim=0.012)
        thumb_px, zoom = 124, 0.60
        out_name = "grid_semantic.png"
    else:
        raise ValueError(space)
    emb = npz["emb"].astype(float)
    sources = [str(s) for s in npz["source"]]
    fnames = [str(f) for f in npz["fname"]]

    xy, keep, assign, reps = _atlas_layout(emb, k=AGG_K, **cfg)

    cen = np.median(xy, 0)
    # 9 evenly-spaced slots on a circle in axes-fraction space, pulled in close to
    # the cloud (overlap is fine -- it kills the dead margin the perimeter created).
    base = np.pi / 2 + 0.15
    SLOTS = [(0.5 + 0.60 * np.cos(base + 2 * np.pi * i / AGG_K),
              0.5 + 0.60 * np.sin(base + 2 * np.pi * i / AGG_K)) for i in range(AGG_K)]
    slot_ang = [np.arctan2(sy - 0.5, sx - 0.5) for sx, sy in SLOTS]
    rep_ang = [np.arctan2(xy[i, 1] - cen[1], xy[i, 0] - cen[0]) for i in reps]
    rep_order = sorted(range(len(reps)), key=lambda r: rep_ang[r])
    slot_order = sorted(range(len(SLOTS)), key=lambda s: slot_ang[s])
    rep_slot = {rep_order[i]: slot_order[i] for i in range(len(reps))}

    # Match the height of the middle recall panel (grid_recall.png) so all three
    # aggregate panels render at the same height when laid out at equal widths in
    # the blog flexbox. The recall grid's output aspect (h/w) is fixed by its
    # geometry in build_global_recall_grid (see RECALL_GRID_ASPECT); we render the
    # cloud into a portrait canvas of the same aspect (aspect='auto' restretches
    # the point cloud and the axes-fraction label ring to fill it).
    fig_w = 5.2
    fig, ax = plt.subplots(figsize=(fig_w, fig_w * RECALL_GRID_ASPECT))
    dot_c = np.array([ATLAS_PALETTE_9[a % len(ATLAS_PALETTE_9)] for a in assign])
    ax.scatter(xy[:, 0], xy[:, 1], s=6, c=dot_c, alpha=0.42, linewidths=0, zorder=1)

    cap_font = FontProperties(size=6.2, style="italic")
    lab_font = FontProperties(size=6.6)
    for r, idx in enumerate(reps):
        color = ATLAS_PALETTE_9[r % len(ATLAS_PALETTE_9)]
        oi = keep[idx]
        ax.scatter([xy[idx, 0]], [xy[idx, 1]], s=22, c=color,
                   edgecolors="white", linewidths=0.8, zorder=4)
        thumb = Image.open(_recall_image_path(sources[oi], fnames[oi])).convert("RGB")
        if thumb.size != (thumb_px, thumb_px):
            thumb = thumb.resize((thumb_px, thumb_px), Image.LANCZOS)
        children = [OffsetImage(np.asarray(thumb), zoom=zoom)]
        children[0].image.axes = ax
        if space == "semantic":
            children.append(TextArea(
                _wrap_caption_lines(captions[oi]),
                textprops=dict(fontproperties=cap_font, color="#555",
                               ha="center", multialignment="center")))
        children.append(TextArea(
            " · ".join(_recall_condition_parts(sources[oi])),
            textprops=dict(fontproperties=lab_font, color=color,
                           ha="center", multialignment="center")))
        packed = VPacker(children=children, align="center", pad=1, sep=2)
        ab = AnnotationBbox(
            packed, (xy[idx, 0], xy[idx, 1]), xybox=SLOTS[rep_slot[r]],
            xycoords="data", boxcoords="axes fraction", frameon=True, pad=0.16,
            zorder=5, box_alignment=(0.5, 0.5), annotation_clip=False,
            bboxprops=dict(edgecolor="none", facecolor="white", linewidth=0),
            arrowprops=dict(arrowstyle="-", color=color, lw=0.8, alpha=0.55))
        ax.add_artist(ab)

    ax.set_aspect("auto")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    padx = np.ptp(xy[:, 0]) * 0.04
    pady = np.ptp(xy[:, 1]) * 0.04
    ax.set_xlim(xy[:, 0].min() - padx, xy[:, 0].max() + padx)
    ax.set_ylim(xy[:, 1].min() - pady, xy[:, 1].max() + pady)
    fig.savefig(out / out_name, dpi=210, facecolor="white",
                bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)

    # The portrait figsize gets us close, but bbox_inches="tight" trims to the label
    # ring (whose physical size is fixed), so the final aspect lands just short of the
    # recall grid's. Pad the canvas (white, centered) to exactly RECALL_GRID_ASPECT so
    # the three panels render at identical heights in the equal-width blog flexbox.
    im = Image.open(out / out_name).convert("RGB")
    target_h = max(im.height, int(round(im.width * RECALL_GRID_ASPECT)))
    if target_h > im.height:
        padded = Image.new("RGB", (im.width, target_h), (255, 255, 255))
        padded.paste(im, (0, (target_h - im.height) // 2))
        padded.save(out / out_name)

    n_exp = len({sources[keep[idx]] for idx in reps})
    print(f"wrote {out/out_name}  ({space} aggregate atlas, {len(xy)} pts, "
          f"{AGG_K} reps from {n_exp} experiments)")


def build_grids(run: Path, out: Path) -> None:
    # 1) Visual Coverage: aggregate 2D atlas of images across ALL experiments.
    build_coverage_atlas_aggregate(out, "visual")

    # 2) Best noun-matches: CROSS-ARCHIVE -- the 9 strongest noun->image matches over
    #    every VLM run + the human archive (see build_global_recall_grid).
    build_global_recall_grid(out)

    # 3) Semantic Coverage: aggregate 2D atlas of caption embeddings across ALL experiments.
    build_coverage_atlas_aggregate(out, "semantic")


# ---------------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    ap.add_argument("--out-dir", type=Path, default=REPO / "figures/metric_figs")
    ap.add_argument("--only", nargs="*", default=None,
                    choices=["grid", "recall", "visual", "semantic", "iso", "grids",
                             "globalrecall", "atlas", "aggcache", "aggatlas"])
    args = ap.parse_args()

    run = args.run_dir.resolve()
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    print(f"run: {run}\nout: {out}\n")

    todo = args.only or ["grid", "recall", "visual", "semantic", "iso", "grids"]
    if "grid" in todo:
        build_archive_grid(run, out)
    if "recall" in todo:
        build_recall(run, out)
    if "visual" in todo:
        build_visual(run, out)
    if "semantic" in todo:
        build_semantic(run, out)
    if "iso" in todo:
        build_iso(run, out)
    if "grids" in todo:
        build_grids(run, out)
    if "globalrecall" in todo:
        build_global_recall_grid(out)
    if "atlas" in todo:
        build_coverage_atlas(run, out, "visual")
        build_coverage_atlas(run, out, "semantic")
    if "aggcache" in todo:
        build_aggregate_caches(out)
    if "aggatlas" in todo:
        build_coverage_atlas_aggregate(out, "visual")
        build_coverage_atlas_aggregate(out, "semantic")


if __name__ == "__main__":
    main()
