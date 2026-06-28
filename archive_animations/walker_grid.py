#!/usr/bin/env python3
"""Render the 36-walker phylogeny traversal as a 6x6 grid of CPPN morphs.

Each cell of the grid plays one walker's continuous walk over the neutral-root-
stitched phylogeny: at every step the walker morphs from the current node's
CPPN to the next node's CPPN along its assigned edge. Walks revisit edges
heavily, so CPPN frames are rendered once per undirected edge and reused (the
reverse direction is the forward frame list reversed).

Cross-lineage edges (those touching the synthetic NEUTRAL_ROOT_ID node)
render as a smooth fade through solid grey -- the neutral root's CPPN is the
hand-built genome from ``neutral_root.py``.

Usage:
    .venv/bin/python archive_animations/walker_grid.py \\
        --run sweep_logs/sweep/th0_..._s4 \\
        --out archive_animations/out/walker_grid_s4.mp4 \\
        --max-n 300 -k 36 -K 30 --cell 64 --frames-per-edge 8 \\
        --jobs 8 --fps 24
"""
from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import neat

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # repo root

from neat_components import (
    PicbreederGenome, apply_picbreeder_config_defaults, InteractiveStagnation,
)
from picbreeder_reproduction import PicbreederReproduction

from walker_partition import (
    best_first_walks, NEUTRAL_ROOT_ID, edge_coverage,
)
from neutral_root import (
    make_neutral_root_genome, render_neutral_to_child_frames,
)


def build_config(config_path: str = "picture2d/interactive_config_color") -> neat.Config:
    config = neat.Config(
        PicbreederGenome, PicbreederReproduction, neat.DefaultSpeciesSet,
        InteractiveStagnation, config_path,
    )
    apply_picbreeder_config_defaults(
        config, enable_output_activations=True, enable_input_activations=False,
        enable_crossover=True,
    )
    return config


def _load_genome(run: Path, nid: str, config: neat.Config):
    if nid == NEUTRAL_ROOT_ID:
        return make_neutral_root_genome(config)
    return pickle.load(open(run / "archive" / "genomes" / f"{nid}.pkl", "rb"))


# Worker: must be top-level for multiprocessing pickling.
def _render_edge_worker(args):
    run_str, a, b, cell, frames_per_edge, config_path, color_a, color_b = args
    run = Path(run_str)
    config = build_config(config_path)
    g_a = _load_genome(run, a, config)
    g_b = _load_genome(run, b, config)
    # If EITHER endpoint was published as color, render the morph as color
    # (the morph reaches that endpoint at one end). Otherwise grayscale, which
    # is what 98% of this archive uses and what archive PNGs actually show.
    use_color = bool(color_a or color_b)
    frames = render_neutral_to_child_frames(
        g_a, g_b, config,
        steps=frames_per_edge, width=cell, height=cell,
        variant_mode=("color" if use_color else "gray"),
        color_enabled=use_color,
    )
    return [np.asarray(f, dtype=np.uint8) for f in frames]


def load_color_map(run: Path) -> Dict[str, bool]:
    """Map each archive entry id -> color_enabled flag (default False)."""
    meta = json.loads((run / "archive" / "archive_metadata.json").read_text())
    entries = meta["entries"] if isinstance(meta, dict) else meta
    return {e["id"]: bool(e.get("color_enabled")) for e in entries}


def precompute_morphs(
    run: Path,
    walks: List[List[Tuple[str, str]]],
    cell: int,
    frames_per_edge: int,
    jobs: int,
    config_path: str,
    color_map: Dict[str, bool],
) -> Dict[Tuple[str, str], List[np.ndarray]]:
    """Render one frame-list per *undirected* edge in the walks; cache both directions.
    Reverse direction is the forward frames in reverse order (interpolation is symmetric).
    """
    undirected = set()
    for w in walks:
        for a, b in w:
            undirected.add(frozenset((a, b)))
    canonical: List[Tuple[str, str]] = []
    for ue in undirected:
        a, b = sorted(ue, key=str)
        canonical.append((a, b))
    n_color = sum(1 for a, b in canonical
                  if color_map.get(a, False) or color_map.get(b, False))
    print(f"  rendering {len(canonical)} unique edges at {cell}x{cell} "
          f"({frames_per_edge} frames each) with {jobs} workers... "
          f"({n_color} color, {len(canonical)-n_color} grayscale)")
    t0 = time.time()
    args_list = [
        (str(run), a, b, cell, frames_per_edge, config_path,
         color_map.get(a, False), color_map.get(b, False))
        for a, b in canonical
    ]
    cache: Dict[Tuple[str, str], List[np.ndarray]] = {}
    done = 0
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        for (a, b), frames in zip(canonical, pool.map(_render_edge_worker, args_list, chunksize=1)):
            cache[(a, b)] = frames
            cache[(b, a)] = frames[::-1]
            done += 1
            if done % max(1, len(canonical) // 10) == 0 or done == len(canonical):
                dt = time.time() - t0
                rate = done / dt if dt > 0 else 0
                eta = (len(canonical) - done) / rate if rate > 0 else 0
                print(f"    {done}/{len(canonical)} edges in {dt:.1f}s "
                      f"({rate:.1f}/s, ETA {eta:.0f}s)")
    return cache


def composite_grid(
    walker_frames: List[np.ndarray | None],
    cols: int,
    rows: int,
    cell: int,
    border_px: int = 1,
) -> np.ndarray:
    """Compose 36 cells into a single grid frame.

    Cells with frame=None are filled with neutral grey (128). A thin dark
    border separates each cell.
    """
    H = rows * cell + (rows + 1) * border_px
    W = cols * cell + (cols + 1) * border_px
    canvas = np.full((H, W, 3), 30, dtype=np.uint8)
    for i, f in enumerate(walker_frames):
        r, c = i // cols, i % cols
        y = border_px + r * (cell + border_px)
        x = border_px + c * (cell + border_px)
        if f is None:
            canvas[y:y + cell, x:x + cell] = 128
        else:
            canvas[y:y + cell, x:x + cell] = f
    return canvas


def siglip_slot_order(
    run: Path,
    walks: List[List[Tuple[str, str]]],
    cols: int,
    rows: int,
    method: str = "umap",
    siglip_model: str = "ViT-B-16-SigLIP2",
    siglip_pretrained: str = "webli",
) -> List[int]:
    """Reorder grid cells so visually-similar walkers sit near each other.

    Each walker is summarized by the *mean SigLIP embedding of the archive
    images along its whole path* (its net visual identity over the morph
    sequence). Those k mean-vectors are reduced to 2D and snapped to a
    cols x rows lattice with RasterFairy, giving each walker a fixed slot.
    Cells never move afterwards -- only morph.

    Returns ``slot_to_walker``: a list of length cols*rows where entry ``s`` is
    the index (into ``walks``) of the walker shown in slot ``s`` (row-major).
    """
    import numpy as _np
    import torch
    import rasterfairy

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from model_loader import load_model_by_name, embed_images
    from embed_and_visualize import reduce_embeddings, _ensure_rasterfairy_ready

    # node sequence each walker traverses (start node + every edge target)
    node_seqs: List[List[str]] = []
    for w in walks:
        node_seqs.append([w[0][0]] + [b for (_a, b) in w] if w else [])

    img_dir = run / "archive" / "images"
    uniq = sorted({n for seq in node_seqs for n in seq
                   if n != NEUTRAL_ROOT_ID and (img_dir / f"{n}.png").exists()})

    # embeddings cached per run+model so re-renders are cheap
    cache_path = run / "archive" / f"_walkergrid_siglip_{siglip_model}.npz"
    id2emb: Dict[str, _np.ndarray] = {}
    if cache_path.exists():
        z = _np.load(cache_path, allow_pickle=True)
        id2emb = {k: v for k, v in zip(z["ids"], z["emb"])}
    missing = [n for n in uniq if n not in id2emb]
    if missing:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  [siglip] embedding {len(missing)} node images "
              f"({len(uniq)} unique, {len(id2emb)} cached) with {siglip_model}...")
        model, preprocess, _ = load_model_by_name(siglip_model, siglip_pretrained, device)
        _, embs = embed_images(
            model, preprocess, [img_dir / f"{n}.png" for n in missing], device, batch_size=64)
        for n, e in zip(missing, embs):
            id2emb[n] = e
        ids = list(id2emb)
        _np.savez_compressed(cache_path, ids=_np.array(ids),
                             emb=_np.vstack([id2emb[i] for i in ids]))

    dim = len(next(iter(id2emb.values())))
    walker_vecs = _np.zeros((len(walks), dim), dtype=_np.float32)
    for i, seq in enumerate(node_seqs):
        vs = [id2emb[n] for n in seq if n in id2emb]
        if vs:
            walker_vecs[i] = _np.mean(vs, axis=0)

    print(f"  [siglip] reducing {len(walks)} walker vectors via {method} "
          f"and snapping to {cols}x{rows} with RasterFairy...")
    coords = reduce_embeddings(walker_vecs, method=method)
    _ensure_rasterfairy_ready()
    grid_points, dims = rasterfairy.transformPointCloud2D(
        _np.asarray(coords, dtype=float), target=(cols, rows))
    grid_points = _np.rint(_np.asarray(grid_points, dtype=float)).astype(int)

    slot_to_walker: List[int] = [-1] * (cols * rows)
    for w, (c, r) in enumerate(grid_points):
        slot_to_walker[int(r) * cols + int(c)] = w
    if any(s < 0 for s in slot_to_walker):
        raise RuntimeError("RasterFairy did not fill every grid slot; "
                           f"got dims={tuple(dims)} for {cols}x{rows}")
    return slot_to_walker


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--max-n", type=int, default=300,
                    help="cap phylogeny to first N publications (genomes must be on disk)")
    ap.add_argument("-k", type=int, default=36)
    ap.add_argument("-K", type=int, default=30, help="edges per walker")
    ap.add_argument("--cols", type=int, default=6)
    ap.add_argument("--rows", type=int, default=6)
    ap.add_argument("--cell", type=int, default=64, help="cell side in pixels")
    ap.add_argument("--frames-per-edge", type=int, default=8)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--beta", type=float, default=2.0)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--config", default="picture2d/interactive_config_color")
    ap.add_argument("--hold", type=int, default=24, help="extra frames at end")
    ap.add_argument("--sort", choices=["none", "siglip"], default="none",
                    help="cell ordering: 'none' = by walker start; 'siglip' = "
                         "RasterFairy layout of each walker's mean SigLIP embedding "
                         "over its whole path (similar lineages sit together)")
    ap.add_argument("--sort-method", choices=["umap", "tsne", "pca"], default="umap",
                    help="2D reduction used before RasterFairy when --sort siglip")
    args = ap.parse_args()

    if args.cols * args.rows != args.k:
        raise SystemExit(f"k={args.k} must equal cols*rows = {args.cols*args.rows}")

    # ------------- build walks -------------
    walks, parent, score, order, starts = best_first_walks(
        args.run, args.k, args.K,
        alpha=args.alpha, beta=args.beta, gamma=args.gamma,
        max_n=args.max_n, require_genome=True, synthetic_root=True,
    )
    cov, tot = edge_coverage(walks, parent)
    print(f"phylogeny (capped): {len(order)} nodes, {tot} edges, "
          f"k={args.k}, K={args.K}; coverage {cov}/{tot} ({100*cov/tot:.1f}%)")

    # ------------- order cells (optional SigLIP self-sort) -------------
    if args.sort == "siglip":
        slot_to_walker = siglip_slot_order(
            args.run, walks, args.cols, args.rows, method=args.sort_method)
        walks = [walks[w] for w in slot_to_walker]

    # ------------- precompute morphs -------------
    color_map = load_color_map(args.run)
    cache = precompute_morphs(
        args.run, walks, args.cell, args.frames_per_edge, args.jobs, args.config,
        color_map,
    )

    # ------------- composite frames -------------
    total_steps = args.K
    total_frames = total_steps * args.frames_per_edge + args.hold
    print(f"  compositing {total_frames} grid frames "
          f"({total_steps} edges x {args.frames_per_edge} frames + {args.hold} hold)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    border = 1
    H = args.rows * args.cell + (args.rows + 1) * border
    W = args.cols * args.cell + (args.cols + 1) * border
    # libx264 requires even dimensions
    pad_h = (H + 1) // 2 * 2 - H
    pad_w = (W + 1) // 2 * 2 - W
    H_out, W_out = H + pad_h, W + pad_w
    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{W_out}x{H_out}", "-r", str(args.fps), "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
         str(args.out)],
        stdin=subprocess.PIPE,
    )

    t0 = time.time()
    for t in range(total_frames):
        edge_idx = min(t // args.frames_per_edge, total_steps - 1)
        sub_t = t % args.frames_per_edge
        if t >= total_steps * args.frames_per_edge:
            edge_idx = total_steps - 1
            sub_t = args.frames_per_edge - 1
        cells: List[np.ndarray | None] = []
        for w in walks:
            if edge_idx < len(w):
                e = w[edge_idx]
                cells.append(cache[e][sub_t])
            elif w:
                # walker ran out of edges -- hold on its final node
                last = w[-1]
                cells.append(cache[last][-1])
            else:
                cells.append(None)
        grid = composite_grid(cells, args.cols, args.rows, args.cell, border)
        if pad_h or pad_w:
            padded = np.full((H_out, W_out, 3), 30, dtype=np.uint8)
            padded[:H, :W] = grid
            grid = padded
        proc.stdin.write(np.ascontiguousarray(grid).tobytes())
        if (t + 1) % max(1, total_frames // 10) == 0 or t == total_frames - 1:
            dt = time.time() - t0
            print(f"    grid frame {t + 1}/{total_frames} ({dt:.1f}s)")
    proc.stdin.close()
    proc.wait()
    print(f"wrote {args.out} ({total_frames} frames, {total_frames / args.fps:.1f}s @ {args.fps}fps)")


if __name__ == "__main__":
    main()
