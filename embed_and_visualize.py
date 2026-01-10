#!/usr/bin/env python3
"""Embed and visualize images from an experiment archive using OpenCLIP + UMAP/TSNE.

This script expects PNG images under: <experiment_dir>/archive/images/*.png

Example:
    python scripts/embed_and_visualize.py --experiment-dir logs_collaborative/exp123

Outputs saved to the experiment directory:
    - embeddings_openclip.npz        (filenames + embeddings)
    - embed_viz_<method>.png         (scatter visualization)
    - embed_grid_<method>.png        (organic grid layout approximation)
    - embed_grid_rect_<method>.png   (rasterfairy rectangular grid)

"""
from collections import deque
from dataclasses import dataclass, field
import importlib
import json
import math
from pathlib import Path
from typing import Optional

from PIL import Image
import numpy as np
import torch
from tqdm import tqdm

import hydra
from hydra.conf import HelpConf, HydraConf
from hydra.core.config_store import ConfigStore
from hydra.utils import get_original_cwd

import open_clip
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
import umap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import rasterfairy

from config import PicbreederConfig, ensure_valid_config



def _validate_embed_options(cfg: "EmbedVisualizeConfig") -> None:
    if cfg.method not in VALID_METHODS:
        raise ValueError(f"method must be one of {VALID_METHODS}")
    if cfg.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if cfg.thumbs_limit <= 0:
        raise ValueError("thumbs_limit must be positive")
    if cfg.archive_limit is not None and cfg.archive_limit <= 0:
        raise ValueError("archive_limit must be a positive integer when provided")
    if cfg.pairwise_sample_limit < 2:
        raise ValueError("pairwise_sample_limit must be at least 2")


VALID_METHODS = ("umap", "tsne", "pca")
_RASTERFAIRY_READY = False


@dataclass
class EmbedVisualizeConfig(PicbreederConfig):
    embedding_model: str = "ViT-B-32"
    pretrained: str = "openai"
    method: str = "umap"
    batch_size: int = 64
    thumbs_limit: int = 200
    archive_limit: Optional[int] = None
    device: Optional[str] = None
    pairwise_sample_limit: int = 2000
    k_center_values: str = "1,5,10,20"
    hydra: HydraConf = field(
        default_factory=lambda: HydraConf(
            help=HelpConf(
                app_name="embed_and_visualize",
                header=(
                    "Hydra entry point for OpenCLIP embedding + visualization.\n"
                    "\n"
                    "Common overrides:\n"
                    "  experiment_dir      Point directly at an existing run.\n"
                    "  goal/scheme/seed    Combine with ensure_valid_config to infer a run directory.\n"
                    "  method              Choose between umap/tsne/pca.\n"
                    "  k_center_values     Comma-separated radii to measure diversity.\n"
                ),
                footer="Override with +option=value (e.g. method=pca k_center_values=5,10).",
            )
        )
    )


ConfigStore.instance().store(name="embed_visualize_base", node=EmbedVisualizeConfig)


def prepare_openclip_components(cfg: EmbedVisualizeConfig, device: torch.device):
    """Create the OpenCLIP model + preprocess transform for this config."""

    model, _, preprocess = open_clip.create_model_and_transforms(
        cfg.embedding_model,
        pretrained=cfg.pretrained,
    )
    model.to(device)
    model.eval()
    return model, preprocess


def load_image_paths(experiment_dir: Path):
    images_dir = experiment_dir / "archive" / "images"
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    imgs = sorted(images_dir.glob("*.png"))
    return imgs


def batch(iterable, n=32):
    l = len(iterable)
    for i in range(0, l, n):
        yield iterable[i : i + n]


def embed_images(model, preprocess, image_paths, device, batch_size=64):
    model.eval()
    embeddings = []
    filenames = []
    with torch.no_grad():
        for chunk in batch(image_paths, batch_size):
            tensors = []
            for p in chunk:
                img = Image.open(p).convert("RGB")
                t = preprocess(img)
                tensors.append(t)
                filenames.append(str(p.name))
            x = torch.stack(tensors, dim=0).to(device)
            emb = model.encode_image(x)
            emb = emb.cpu().numpy()
            # L2 normalize
            norm = np.linalg.norm(emb, axis=1, keepdims=True)
            norm[norm == 0] = 1.0
            emb = emb / norm
            embeddings.append(emb)
    embeddings = np.vstack(embeddings)
    return filenames, embeddings


def reduce_embeddings(embeddings, method="umap"):
    if method == "umap":
        if umap is None:
            raise RuntimeError("umap-learn not installed. Install `umap-learn` or choose --method tsne")
        reducer = umap.UMAP(n_components=2, random_state=0)
        coords = reducer.fit_transform(embeddings)
    elif method == "tsne":
        reducer = TSNE(n_components=2, init="pca", random_state=0, learning_rate="auto")
        coords = reducer.fit_transform(embeddings)
    elif method == "pca":
        # PCA is fast and deterministic; useful as a simple linear baseline
        reducer = PCA(n_components=2)
        coords = reducer.fit_transform(embeddings)
    else:
        raise ValueError("method must be 'umap', 'tsne', or 'pca'")
    return coords


def plot_coords(coords, image_paths, outpath: Path, thumbs_limit=200):
    fig, ax = plt.subplots(figsize=(10, 10))
    x = coords[:, 0]
    y = coords[:, 1]
    sc = ax.scatter(x, y, s=10, cmap="tab10", alpha=0.7)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Image embeddings (2D)")

    # place thumbnails for a subset to avoid clutter
    n = len(image_paths)
    stride = 1
    if n > thumbs_limit:
        stride = math.ceil(n / thumbs_limit)

    for i in range(0, n, stride):
        try:
            img = Image.open(image_paths[i]).convert("RGB")
            arr = np.array(img)
            # small thumbnail
            im = OffsetImage(arr, zoom=0.2)
            ab = AnnotationBbox(im, (x[i], y[i]), frameon=False)
            ax.add_artist(ab)
        except Exception:
            # skip any images that fail to load
            continue

    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


# def _direction_priority(vec):
#     """Return the cardinal direction order that best aligns with the embedding offset."""
#     direction_vectors = {
#         (0, 1): np.array([1.0, 0.0]),
#         (0, -1): np.array([-1.0, 0.0]),
#         (-1, 0): np.array([0.0, 1.0]),
#         (1, 0): np.array([0.0, -1.0]),
#     }
#     base_order = [(0, 1), (0, -1), (-1, 0), (1, 0)]
#     norm = np.linalg.norm(vec)
#     if norm < 1e-9:
#         return base_order
#     vec_unit = vec / norm
#     return sorted(base_order, key=lambda d: -float(np.dot(vec_unit, direction_vectors[d])))


# def _find_grid_position(neighbor_pos, direction_order, occupied):
#     """Breadth-first search for the nearest free grid cell, honoring a direction priority."""
#     visited = set()
#     queue = deque()
#     visited.add(neighbor_pos)
#     for direction in direction_order:
#         target = (neighbor_pos[0] + direction[0], neighbor_pos[1] + direction[1])
#         queue.append(target)

#     default_dirs = direction_order
#     safety_cap = 100000  # guard against unexpected infinite loops
#     iterations = 0

#     while queue:
#         iterations += 1
#         if iterations > safety_cap:
#             raise RuntimeError("Grid placement search exceeded safety cap; layout may be stuck.")

#         pos = queue.popleft()
#         if pos in visited:
#             continue
#         visited.add(pos)

#         if pos not in occupied:
#             return pos

#         for direction in default_dirs:
#             next_pos = (pos[0] + direction[0], pos[1] + direction[1])
#             if next_pos not in visited:
#                 queue.append(next_pos)

#     raise RuntimeError("Failed to find an empty grid position for image placement.")


# def layout_embeddings_to_grid(coords: np.ndarray):
#     """Assign each embedding to an integer grid coordinate while preserving locality."""
#     coords = np.asarray(coords, dtype=float)
#     n = coords.shape[0]
#     if n == 0:
#         return {}, (0, 0, 0, 0)

#     centroid = coords.mean(axis=0)
#     center_idx = int(np.argmin(np.linalg.norm(coords - centroid, axis=1)))

#     assignments = {center_idx: (0, 0)}
#     occupied = {(0, 0)}

#     unassigned = [idx for idx in range(n) if idx != center_idx]
#     if not unassigned:
#         return assignments, (0, 0, 0, 0)

#     unassigned_arr = np.array(unassigned, dtype=int)
#     best_dist = np.abs(coords[unassigned_arr] - coords[center_idx]).sum(axis=1)
#     best_neighbor = [center_idx for _ in unassigned]

#     while unassigned:
#         min_pos = int(np.argmin(best_dist))
#         idx = unassigned.pop(min_pos)
#         neighbor_idx = best_neighbor.pop(min_pos)
#         best_dist = np.delete(best_dist, min_pos)

#         neighbor_pos = assignments[neighbor_idx]
#         vec = coords[idx] - coords[neighbor_idx]
#         direction_order = _direction_priority(vec)
#         new_pos = _find_grid_position(neighbor_pos, direction_order, occupied)

#         assignments[idx] = new_pos
#         occupied.add(new_pos)

#         if unassigned:
#             unassigned_arr = np.array(unassigned, dtype=int)
#             new_dists = np.abs(coords[unassigned_arr] - coords[idx]).sum(axis=1)
#             mask = new_dists < best_dist
#             best_dist[mask] = new_dists[mask]
#             for i, use_new in enumerate(mask.tolist()):
#                 if use_new:
#                     best_neighbor[i] = idx

#     rows = [pos[0] for pos in assignments.values()]
#     cols = [pos[1] for pos in assignments.values()]
#     bounds = (min(rows), max(rows), min(cols), max(cols))
#     return assignments, bounds


# def render_grid(assignments, bounds, image_paths, outpath: Path):
#     """Render the grid-aligned images to disk."""
#     if not assignments:
#         return

#     min_row, max_row, min_col, max_col = bounds
#     rows = max_row - min_row + 1
#     cols = max_col - min_col + 1

#     fig_w = min(30.0, max(4.0, cols * 1.6))
#     fig_h = min(30.0, max(4.0, rows * 1.6))
#     fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h))

#     if rows == 1 and cols == 1:
#         axes = np.array([[axes]])
#     elif rows == 1:
#         axes = axes.reshape(1, cols)
#     elif cols == 1:
#         axes = axes.reshape(rows, 1)

#     for r in range(rows):
#         for c in range(cols):
#             ax = axes[r, c]
#             ax.set_xticks([])
#             ax.set_yticks([])
#             ax.axis("off")

#     for idx, (row, col) in assignments.items():
#         rr = row - min_row
#         cc = col - min_col
#         ax = axes[rr, cc]
#         try:
#             with Image.open(image_paths[idx]) as img:
#                 img_rgb = img.convert("RGB")
#             ax.imshow(img_rgb)
#         except Exception:
#             continue

#     fig.tight_layout()
#     fig.savefig(outpath, dpi=150)
#     plt.close(fig)


def _ensure_rasterfairy_ready():
    """Prepare rasterfairy for use on modern NumPy versions."""
    global _RASTERFAIRY_READY
    if _RASTERFAIRY_READY:
        return
    if rasterfairy is None:
            raise RuntimeError("rasterfairy not installed. Run `pip install rasterfairy`.")

    try:
        rf_module = importlib.import_module("rasterfairy.rasterfairy")
        prime_module = importlib.import_module("rasterfairy.prime")
        rf_module.prime = prime_module
    except Exception as exc:  # pragma: no cover - import-time defensive guard
        raise RuntimeError("Failed to initialize rasterfairy internals") from exc

    for alias_name, builtin_type in (("float", float), ("int", int), ("bool", bool)):
        if not hasattr(np, alias_name):
            setattr(np, alias_name, builtin_type)

    _RASTERFAIRY_READY = True


def render_rectangular_grid(coords: np.ndarray, image_paths, outpath: Path):
    """Render a strictly rectangular grid using rasterfairy."""
    coords = np.asarray(coords, dtype=float)
    if coords.size == 0:
        return

    try:
        _ensure_rasterfairy_ready()
    except RuntimeError as exc:
        print(f"Skipping rectangular grid visualization: {exc}")
        return

    grid_points, dims = rasterfairy.transformPointCloud2D(coords)
    if not isinstance(dims, (tuple, list)) or len(dims) != 2:
        print("Skipping rectangular grid visualization: rasterfairy returned invalid dimensions")
        return

    cols, rows = (int(dims[0]), int(dims[1]))
    if cols <= 0 or rows <= 0:
        print("Skipping rectangular grid visualization: rasterfairy produced non-positive grid size")
        return

    grid_positions = np.rint(np.asarray(grid_points, dtype=float)).astype(int)

    # Use width=cols and height=rows so each subplot is approximately square.
    fig, axes = plt.subplots(rows, cols, figsize=(cols, rows))

    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, cols)
    elif cols == 1:
        axes = axes.reshape(rows, 1)

    for r in range(rows):
        for c in range(cols):
            ax = axes[r, c]
            ax.set_xticks([])
            ax.set_yticks([])
            ax.axis("off")

    for idx, (col, row) in enumerate(grid_positions):
        rr = int(row)
        cc = int(col)
        if rr < 0 or rr >= rows or cc < 0 or cc >= cols:
            continue
        ax = axes[rr, cc]
        try:
            with Image.open(image_paths[idx]) as img:
                img_rgb = img.convert("RGB")
            ax.imshow(img_rgb)
            ax.set_aspect("equal")
        except Exception:
            continue

    fig.subplots_adjust(
        left=0.0, right=1.0, bottom=0.0, top=1.0,
        wspace=0.02, hspace=0.02,  # shrink or set to 0.0 for no gutters
    )

    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def _compute_pairwise_distance_stats(
    embeddings: np.ndarray, max_points: int = 2000, random_state: int = 0
):
    """Compute statistics on the distribution of pairwise distances.
    
    Returns mean, std, min, max, and percentiles of pairwise distances.
    This gives a fuller picture of how spread out the embeddings are.
    """
    n = embeddings.shape[0]
    if n < 2:
        return {"computed_on": n, "sampled": False}

    if n > max_points:
        rng = np.random.default_rng(random_state)
        sample_idx = rng.choice(n, size=max_points, replace=False)
        sample = embeddings[sample_idx]
        sampled = True
        sample_size = max_points
    else:
        sample = embeddings
        sampled = False
        sample_size = n

    dists = pairwise_distances(sample, metric="euclidean")
    iu = np.triu_indices(sample_size, k=1)
    
    if iu[0].size == 0:
        return {"computed_on": sample_size, "sampled": sampled}
    
    pairwise_dists = dists[iu]
    
    return {
        "mean": float(pairwise_dists.mean()),
        "std": float(pairwise_dists.std()),
        "min": float(pairwise_dists.min()),
        "max": float(pairwise_dists.max()),
        "median": float(np.median(pairwise_dists)),
        "p10": float(np.percentile(pairwise_dists, 10)),
        "p25": float(np.percentile(pairwise_dists, 25)),
        "p75": float(np.percentile(pairwise_dists, 75)),
        "p90": float(np.percentile(pairwise_dists, 90)),
        "computed_on": sample_size,
        "sampled": sampled,
    }


def _compute_nearest_neighbor_stats(embeddings: np.ndarray):
    """Compute statistics on nearest neighbor distances.
    
    For each point, find the distance to its nearest neighbor.
    This measures local density/clustering - if points are evenly spread,
    NN distances should be relatively uniform. High variance indicates clustering.
    """
    n = embeddings.shape[0]
    if n < 2:
        return {"computed_on": n}
    
    dists = pairwise_distances(embeddings, metric="euclidean")
    # Set diagonal to inf so we don't pick self as nearest neighbor
    np.fill_diagonal(dists, np.inf)
    nn_dists = dists.min(axis=1)
    
    return {
        "mean": float(nn_dists.mean()),
        "std": float(nn_dists.std()),
        "min": float(nn_dists.min()),
        "max": float(nn_dists.max()),
        "median": float(np.median(nn_dists)),
        "coefficient_of_variation": float(nn_dists.std() / nn_dists.mean()) if nn_dists.mean() > 0 else None,
        "computed_on": n,
    }


def _greedy_k_center(embeddings: np.ndarray, k: int):
    """Run greedy k-center algorithm and return centers + final distances.
    
    The greedy k-center algorithm:
    1. Start with point furthest from centroid
    2. Iteratively add the point furthest from any existing center
    3. Return the center indices and the distance of each point to its nearest center
    
    This is a 2-approximation algorithm for the k-center problem.
    """
    n = embeddings.shape[0]
    if n == 0 or k <= 0:
        return None, None
    k = min(k, n)

    mean_vec = embeddings.mean(axis=0)
    start = int(np.argmax(np.linalg.norm(embeddings - mean_vec, axis=1)))
    centers = [start]
    dist_to_centers = np.linalg.norm(embeddings - embeddings[start], axis=1)

    for _ in range(1, k):
        next_center = int(np.argmax(dist_to_centers))
        centers.append(next_center)
        new_dist = np.linalg.norm(embeddings - embeddings[next_center], axis=1)
        dist_to_centers = np.minimum(dist_to_centers, new_dist)

    return centers, dist_to_centers


def _compute_k_center_metrics(embeddings: np.ndarray, k: int):
    """Compute comprehensive k-center coverage metrics.
    
    Returns:
    - covering_radius: max distance to nearest center (the classic k-center objective)
    - mean_distance: average distance to nearest center
    - std_distance: std dev of distances (low = even coverage)
    - median_distance: median distance to nearest center
    - coverage_uniformity: 1 - (std/mean), higher is more uniform (0-1 scale, can be negative)
    """
    n = embeddings.shape[0]
    if n == 0 or k <= 0:
        return None
    
    centers, dist_to_centers = _greedy_k_center(embeddings, k)
    if centers is None:
        return None
    
    effective_k = len(centers)
    
    # Basic statistics on distances to nearest center
    covering_radius = float(dist_to_centers.max())
    mean_dist = float(dist_to_centers.mean())
    std_dist = float(dist_to_centers.std())
    median_dist = float(np.median(dist_to_centers))
    
    # Coverage uniformity: coefficient of variation inverted
    # Low CV means distances are uniform (good even coverage)
    cv = std_dist / mean_dist if mean_dist > 0 else 0.0
    
    # Count how many points each center covers (assigned to nearest center)
    # High variance in cluster sizes indicates uneven coverage
    if effective_k > 1:
        # Compute full distance matrix to centers
        center_embeddings = embeddings[centers]
        all_dists = pairwise_distances(embeddings, center_embeddings, metric="euclidean")
        assignments = all_dists.argmin(axis=1)
        cluster_sizes = np.bincount(assignments, minlength=effective_k)
        cluster_size_std = float(cluster_sizes.std())
        cluster_size_cv = cluster_size_std / cluster_sizes.mean() if cluster_sizes.mean() > 0 else 0.0
    else:
        cluster_size_std = 0.0
        cluster_size_cv = 0.0
    
    return {
        "effective_k": effective_k,
        "covering_radius": covering_radius,
        "mean_distance_to_center": mean_dist,
        "std_distance_to_center": std_dist,
        "median_distance_to_center": median_dist,
        "distance_coefficient_of_variation": cv,
        "cluster_size_coefficient_of_variation": cluster_size_cv,
    }


def compute_embedding_metrics(
    embeddings: np.ndarray,
    *,
    random_state: int = 0,
    pairwise_sample_limit: int = 2000,
    k_values=None,
):
    """Compute comprehensive metrics for embedding space coverage.
    
    Metrics computed:
    1. Basic info (num_embeddings, embedding_dim)
    2. Pairwise distance distribution (mean, std, percentiles) - overall spread
    3. Nearest neighbor statistics - local density/clustering  
    4. K-center metrics for various k - coverage with k balls
    
    For understanding "evenness" of coverage:
    - Low coefficient of variation in NN distances = even local density
    - Low coefficient of variation in k-center distances = even global coverage
    - Low cluster size CV = centers cover roughly equal numbers of points
    """
    if k_values is None:
        k_values = [1, 5, 10, 20]

    n = embeddings.shape[0]
    metrics = {
        "num_embeddings": n,
        "embedding_dim": int(embeddings.shape[1]) if embeddings.size else 0,
    }

    # Pairwise distance distribution - tells us about overall spread
    metrics["pairwise_distances"] = _compute_pairwise_distance_stats(
        embeddings, max_points=pairwise_sample_limit, random_state=random_state
    )

    # Nearest neighbor statistics - tells us about local density/clustering
    if n <= pairwise_sample_limit:
        # Only compute if dataset is small enough (O(n^2) memory)
        metrics["nearest_neighbor"] = _compute_nearest_neighbor_stats(embeddings)
    else:
        # Sample for large datasets
        rng = np.random.default_rng(random_state)
        sample_idx = rng.choice(n, size=pairwise_sample_limit, replace=False)
        sample = embeddings[sample_idx]
        nn_stats = _compute_nearest_neighbor_stats(sample)
        nn_stats["sampled"] = True
        nn_stats["computed_on"] = pairwise_sample_limit
        metrics["nearest_neighbor"] = nn_stats

    # K-center metrics for various k values
    k_center_metrics = {}
    for k in k_values:
        result = _compute_k_center_metrics(embeddings, k)
        if result is not None:
            k_center_metrics[str(k)] = result
    metrics["k_center"] = k_center_metrics

    return metrics
def _parse_k_center_values(raw: str) -> list[int]:
    try:
        values = [int(value.strip()) for value in raw.split(",") if value.strip()]
    except ValueError as exc:  # pragma: no cover - validation guard
        raise ValueError("k_center_values must be a comma-separated list of integers") from exc
    if not values or any(value <= 0 for value in values):
        raise ValueError("k_center_values must contain positive integers")
    return values


@hydra.main(version_base="1.3", config_path=None, config_name="embed_visualize_base")
def main(
    cfg: EmbedVisualizeConfig,
    *,
    model=None,
    preprocess=None,
) -> None:
    original_cwd = Path(get_original_cwd())
    validated_cfg = ensure_valid_config(cfg, original_cwd=original_cwd)
    _validate_embed_options(validated_cfg)

    exp_dir = Path(validated_cfg.experiment_dir).resolve()
    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment directory does not exist: {exp_dir}")

    image_paths = load_image_paths(exp_dir)
    if validated_cfg.archive_limit is not None:
        image_paths = image_paths[: validated_cfg.archive_limit]
    if not image_paths:
        raise SystemExit("No PNG images found in archive/images")

    if validated_cfg.device:
        device = torch.device(validated_cfg.device)
    else:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    if (model is None) ^ (preprocess is None):
        raise ValueError("Provide both model and preprocess, or neither.")

    if model is None:
        print(f"Loading OpenCLIP model {validated_cfg.embedding_model} ({validated_cfg.pretrained})...")
        model, preprocess = prepare_openclip_components(validated_cfg, device)
    else:
        model.to(device)
        model.eval()

    print(f"Embedding {len(image_paths)} images (batch_size={validated_cfg.batch_size})...")
    filenames, embeddings = embed_images(
        model,
        preprocess,
        image_paths,
        device,
        batch_size=validated_cfg.batch_size,
    )

    emb_out = exp_dir / "embeddings_openclip.npz"
    np.savez_compressed(emb_out, filenames=np.array(filenames), embeddings=embeddings)
    print(f"Saved embeddings to {emb_out}")

    k_values = _parse_k_center_values(validated_cfg.k_center_values)

    metrics = compute_embedding_metrics(
        embeddings,
        random_state=0,
        pairwise_sample_limit=validated_cfg.pairwise_sample_limit,
        k_values=k_values,
    )
    metrics_out = exp_dir / "embedding_metrics.json"
    with metrics_out.open("w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved embedding coverage metrics to {metrics_out}")

    print(f"Reducing to 2D using {validated_cfg.method}...")
    coords = reduce_embeddings(embeddings, method=validated_cfg.method)

    viz_out = exp_dir / f"embed_viz_{validated_cfg.method}.pdf"
    print(f"Creating visualization (thumbnails limit={validated_cfg.thumbs_limit}) -> {viz_out}")
    plot_coords(coords, list(image_paths), viz_out, thumbs_limit=validated_cfg.thumbs_limit)

    # grid_assignments, grid_bounds = layout_embeddings_to_grid(coords)
    # grid_out = exp_dir / f"embed_grid_{validated_cfg.method}.pdf"
    # print(f"Rendering grid approximation -> {grid_out}")
    # render_grid(grid_assignments, grid_bounds, image_paths, grid_out)

    rect_grid_out = exp_dir / f"embed_grid_rect_{validated_cfg.method}.pdf"
    print(f"Rendering rectangular rasterfairy grid -> {rect_grid_out}")
    render_rectangular_grid(coords, image_paths, rect_grid_out)

    print("Done.")


if __name__ == "__main__":
    main()
