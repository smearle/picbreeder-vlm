"""Build the hover-reveal data for the blog's Visual/Semantic Coverage clouds.

The two coverage panels (grid_visual.png / grid_semantic.png, drawn by
build_coverage_atlas_aggregate in build_metric_fig_assets) are static UMAP
scatters of ~24k pooled archive items. This tool re-runs the SAME layout, then
exports — per space — a representative sample of those items as a thumbnail atlas
plus a manifest of their positions in the final PNG's pixel frame, so the
frontend (assets/page/coverage-cloud.js) can bloom real individuals in under the
cursor as you hover the cloud.

To guarantee the manifest lines up with the picture the reader sees, we RE-RENDER
each panel here from one shared UMAP layout and write the PNG next to the atlas —
PNG and point positions therefore come from the identical `xy`. The render mirrors
build_coverage_atlas_aggregate exactly (keep it in sync if that changes).

Outputs (under <blog>/assets/coverage_clouds/<space>/):
    grid_<space>.png   the re-rendered coverage panel (replaces assets/grid_<space>.png)
    atlas.jpg          `cell`x`cell` thumbnails of the sampled individuals, atlas order
    manifest.json      {space,w,h,cell,cols,count,atlas,pts:[[nx,ny],...]}
                       pts are normalized [0,1] positions in the PNG frame (y down),
                       atlas slot i -> (col=i%cols,row=i//cols).

Usage:
    python tools/build_coverage_clouds.py                    # both spaces -> blog
    python tools/build_coverage_clouds.py --only visual
    python tools/build_coverage_clouds.py --sample 2200 --cell 56
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_metric_fig_assets as B  # noqa: E402  (REPO, layout + image helpers)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.font_manager import FontProperties  # noqa: E402
from matplotlib.offsetbox import (AnnotationBbox, OffsetImage,  # noqa: E402
                                  TextArea, VPacker)

DEFAULT_OUT = Path.home() / "smearle.github.io/picbreeder-vlm-06b0d76d/assets/coverage_clouds"
# Also refresh the panel the page actually <img>s, so PNG == manifest frame.
BLOG_ASSETS = Path.home() / "smearle.github.io/picbreeder-vlm-06b0d76d/assets"

SPACE_CFG = {
    "visual":   dict(npz=B.AGG_VISUAL_NPZ, nn=80, min_dist=0.9, seed=0, trim=0.03,
                     thumb_px=150, zoom=0.50, caption=False),
    "semantic": dict(npz=B.AGG_SEMANTIC_NPZ, nn=50, min_dist=0.5, seed=1, trim=0.012,
                     thumb_px=124, zoom=0.60, caption=True),
}


def _render_panel(space, cfg, npz):
    """Mirror build_coverage_atlas_aggregate: return (fig, ax, xy, keep, assign,
    reps, sources, fnames, captions). The panel is fully drawn but not yet saved,
    so the caller can read ax.transData for the point export."""
    emb = npz["emb"].astype(float)
    sources = [str(s) for s in npz["source"]]
    fnames = [str(f) for f in npz["fname"]]
    captions = [str(c) for c in npz["caption"]] if cfg["caption"] else None

    lay = dict(nn=cfg["nn"], min_dist=cfg["min_dist"], seed=cfg["seed"], trim=cfg["trim"])
    xy, keep, assign, reps = B._atlas_layout(emb, k=B.AGG_K, **lay)

    cen = np.median(xy, 0)
    base = np.pi / 2 + 0.15
    SLOTS = [(0.5 + 0.60 * np.cos(base + 2 * np.pi * i / B.AGG_K),
              0.5 + 0.60 * np.sin(base + 2 * np.pi * i / B.AGG_K)) for i in range(B.AGG_K)]
    slot_ang = [np.arctan2(sy - 0.5, sx - 0.5) for sx, sy in SLOTS]
    rep_ang = [np.arctan2(xy[i, 1] - cen[1], xy[i, 0] - cen[0]) for i in reps]
    rep_order = sorted(range(len(reps)), key=lambda r: rep_ang[r])
    slot_order = sorted(range(len(SLOTS)), key=lambda s: slot_ang[s])
    rep_slot = {rep_order[i]: slot_order[i] for i in range(len(reps))}

    fig_w = 5.2
    fig, ax = plt.subplots(figsize=(fig_w, fig_w * B.RECALL_GRID_ASPECT))
    dot_c = np.array([B.ATLAS_PALETTE_9[a % len(B.ATLAS_PALETTE_9)] for a in assign])
    ax.scatter(xy[:, 0], xy[:, 1], s=6, c=dot_c, alpha=0.42, linewidths=0, zorder=1)

    cap_font = FontProperties(size=6.2, style="italic")
    lab_font = FontProperties(size=6.6)
    for r, idx in enumerate(reps):
        color = B.ATLAS_PALETTE_9[r % len(B.ATLAS_PALETTE_9)]
        oi = keep[idx]
        ax.scatter([xy[idx, 0]], [xy[idx, 1]], s=22, c=color,
                   edgecolors="white", linewidths=0.8, zorder=4)
        thumb = Image.open(B._recall_image_path(sources[oi], fnames[oi])).convert("RGB")
        if thumb.size != (cfg["thumb_px"], cfg["thumb_px"]):
            thumb = thumb.resize((cfg["thumb_px"], cfg["thumb_px"]), Image.LANCZOS)
        children = [OffsetImage(np.asarray(thumb), zoom=cfg["zoom"])]
        children[0].image.axes = ax
        if cfg["caption"]:
            children.append(TextArea(
                B._wrap_caption_lines(captions[oi]),
                textprops=dict(fontproperties=cap_font, color="#555",
                               ha="center", multialignment="center")))
        children.append(TextArea(
            " · ".join(B._recall_condition_parts(sources[oi])),
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
    return fig, ax, xy, keep, assign, reps, sources, fnames, captions


def _save_and_map(fig, ax, xy, out_png):
    """Save the panel exactly as build_coverage_atlas_aggregate does (tight bbox +
    white pad to RECALL_GRID_ASPECT) and return a function mapping data (x,y) ->
    normalized [0,1] position (y down) in the FINAL saved PNG."""
    dpi, pad_in = 210, 0.06
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    # Tight bbox (inches, figure frame, y up) — includes the annotation callouts,
    # exactly what bbox_inches="tight" crops to.
    tb = fig.get_tightbbox(renderer)
    cx0, cy0 = tb.x0 - pad_in, tb.y0 - pad_in           # cropped-image origin (inches)
    cw, ch = tb.width + 2 * pad_in, tb.height + 2 * pad_in

    fig.savefig(out_png, dpi=dpi, facecolor="white",
                bbox_inches="tight", pad_inches=pad_in)
    plt.close(fig)

    im = Image.open(out_png).convert("RGB")
    Wc, Hc = im.size                                    # cropped (pre-pad) size
    target_h = max(Hc, int(round(Wc * B.RECALL_GRID_ASPECT)))
    y_off = 0
    if target_h > Hc:
        padded = Image.new("RGB", (Wc, target_h), (255, 255, 255))
        y_off = (target_h - Hc) // 2
        padded.paste(im, (0, y_off))
        padded.save(out_png)
        H = target_h
    else:
        H = Hc
    W = Wc

    def data_to_norm(pts):
        # ax.transData -> display px at fig.dpi; /fig.dpi -> inches (fig frame, y up)
        disp = ax.transData.transform(pts)
        inch = disp / fig.dpi
        fx = (inch[:, 0] - cx0) / cw                    # 0..1 across cropped width
        fy_up = (inch[:, 1] - cy0) / ch                 # 0..1 up cropped height
        px = fx * Wc
        py = (1.0 - fy_up) * Hc + y_off                 # image y is down; add pad
        return np.stack([px / W, py / H], 1)

    return data_to_norm, W, H


def build(space, out_root, sample, cell, quality, refresh_blog=True):
    cfg = SPACE_CFG[space]
    npz = np.load(B.REPO / "figures/metric_figs" / cfg["npz"], allow_pickle=True)
    out = out_root / space
    out.mkdir(parents=True, exist_ok=True)

    fig, ax, xy, keep, assign, reps, sources, fnames, captions = _render_panel(space, cfg, npz)
    out_png = out / f"grid_{space}.png"
    data_to_norm, W, H = _save_and_map(fig, ax, xy, out_png)
    print(f"  rendered {out_png.name}  {W}x{H}")

    # Representative sample of the KEPT points, spread evenly across the 2D cloud by
    # farthest-point sampling, so the bloom has individuals to reveal everywhere.
    n_keep = xy.shape[0]
    k = min(sample, n_keep)
    sel_local, _r = B._greedy_kcenter_2d(xy, k)         # indices into kept rows
    sel_local = np.asarray(sel_local, int)
    norm = data_to_norm(xy[sel_local])                  # positions in PNG frame

    cols = int(np.ceil(np.sqrt(k)))
    rows = int(np.ceil(k / cols))
    atlas = Image.new("RGB", (cols * cell, rows * cell), (255, 255, 255))
    pts, miss = [], 0
    for i, li in enumerate(sel_local):
        oi = keep[li]
        try:
            im = Image.open(B._recall_image_path(sources[oi], fnames[oi])).convert("RGB")
            if im.size != (cell, cell):
                im = im.resize((cell, cell), Image.LANCZOS)
        except FileNotFoundError:
            miss += 1
            im = Image.new("RGB", (cell, cell), (235, 235, 235))
        atlas.paste(im, ((i % cols) * cell, (i // cols) * cell))
        pts.append([round(float(norm[i, 0]), 5), round(float(norm[i, 1]), 5)])

    atlas.save(out / "atlas.jpg", quality=quality, optimize=True)
    manifest = {"space": space, "w": W, "h": H, "cell": cell, "cols": cols,
                "count": k, "atlas": "atlas.jpg", "pts": pts}
    (out / "manifest.json").write_text(json.dumps(manifest, separators=(",", ":")))
    sz = (out / "atlas.jpg").stat().st_size
    print(f"  wrote {out/'atlas.jpg'}  ({sz/1e6:.2f} MB, {k} pts, {miss} missing) "
          f"+ manifest.json")

    # Refresh the panel the blog page <img>s, so the reader sees the SAME frame the
    # manifest was computed in.
    if refresh_blog:
        dest = BLOG_ASSETS / f"grid_{space}.png"
        Image.open(out_png).save(dest)
        print(f"  refreshed {dest}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--only", nargs="*", choices=["visual", "semantic"], default=None)
    ap.add_argument("--sample", type=int, default=2200)
    ap.add_argument("--cell", type=int, default=56)
    ap.add_argument("--quality", type=int, default=82)
    ap.add_argument("--no-blog", action="store_true",
                    help="don't overwrite the blog's grid_<space>.png (for verification)")
    args = ap.parse_args()
    spaces = args.only or ["visual", "semantic"]
    for sp in spaces:
        print(f"[{sp}]")
        build(sp, args.out_dir.resolve(), args.sample, args.cell, args.quality,
              refresh_blog=not args.no_blog)


if __name__ == "__main__":
    main()
