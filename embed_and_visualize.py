#!/usr/bin/env python3
"""Embed and visualize images from an experiment archive using OpenCLIP + UMAP/TSNE.

This script expects PNG images under: <experiment_dir>/archive/images/*.png

Example:
  python scripts/embed_and_visualize.py --experiment-dir logs_collaborative/exp123

Outputs saved to the experiment directory:
  - embeddings_openclip.npz    (filenames + embeddings)
    - embed_viz_<method>.png     (scatter visualization)
    - embed_grid_<method>.png    (grid layout approximation)

"""
from pathlib import Path
import argparse
import json
import sys
import math
from collections import deque

from PIL import Image
import numpy as np
import torch
from tqdm import tqdm

try:
    import open_clip
except Exception as e:
    print("open_clip import failed. Make sure `open_clip_torch` is installed.")
    raise

try:
    import umap
except Exception:
    umap = None

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


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


def _direction_priority(vec):
    """Return the cardinal direction order that best aligns with the embedding offset."""
    direction_vectors = {
        (0, 1): np.array([1.0, 0.0]),
        (0, -1): np.array([-1.0, 0.0]),
        (-1, 0): np.array([0.0, 1.0]),
        (1, 0): np.array([0.0, -1.0]),
    }
    base_order = [(0, 1), (0, -1), (-1, 0), (1, 0)]
    norm = np.linalg.norm(vec)
    if norm < 1e-9:
        return base_order
    vec_unit = vec / norm
    return sorted(base_order, key=lambda d: -float(np.dot(vec_unit, direction_vectors[d])))


def _find_grid_position(neighbor_pos, direction_order, occupied):
    """Breadth-first search for the nearest free grid cell, honoring a direction priority."""
    visited = set()
    queue = deque()
    visited.add(neighbor_pos)
    for direction in direction_order:
        target = (neighbor_pos[0] + direction[0], neighbor_pos[1] + direction[1])
        queue.append(target)

    default_dirs = direction_order
    safety_cap = 100000  # guard against unexpected infinite loops
    iterations = 0

    while queue:
        iterations += 1
        if iterations > safety_cap:
            raise RuntimeError("Grid placement search exceeded safety cap; layout may be stuck.")

        pos = queue.popleft()
        if pos in visited:
            continue
        visited.add(pos)

        if pos not in occupied:
            return pos

        for direction in default_dirs:
            next_pos = (pos[0] + direction[0], pos[1] + direction[1])
            if next_pos not in visited:
                queue.append(next_pos)

    raise RuntimeError("Failed to find an empty grid position for image placement.")


def layout_embeddings_to_grid(coords: np.ndarray):
    """Assign each embedding to an integer grid coordinate while preserving locality."""
    coords = np.asarray(coords, dtype=float)
    n = coords.shape[0]
    if n == 0:
        return {}, (0, 0, 0, 0)

    centroid = coords.mean(axis=0)
    center_idx = int(np.argmin(np.linalg.norm(coords - centroid, axis=1)))

    assignments = {center_idx: (0, 0)}
    occupied = {(0, 0)}

    unassigned = [idx for idx in range(n) if idx != center_idx]
    if not unassigned:
        return assignments, (0, 0, 0, 0)

    unassigned_arr = np.array(unassigned, dtype=int)
    best_dist = np.abs(coords[unassigned_arr] - coords[center_idx]).sum(axis=1)
    best_neighbor = [center_idx for _ in unassigned]

    while unassigned:
        min_pos = int(np.argmin(best_dist))
        idx = unassigned.pop(min_pos)
        neighbor_idx = best_neighbor.pop(min_pos)
        best_dist = np.delete(best_dist, min_pos)

        neighbor_pos = assignments[neighbor_idx]
        vec = coords[idx] - coords[neighbor_idx]
        direction_order = _direction_priority(vec)
        new_pos = _find_grid_position(neighbor_pos, direction_order, occupied)

        assignments[idx] = new_pos
        occupied.add(new_pos)

        if unassigned:
            unassigned_arr = np.array(unassigned, dtype=int)
            new_dists = np.abs(coords[unassigned_arr] - coords[idx]).sum(axis=1)
            mask = new_dists < best_dist
            best_dist[mask] = new_dists[mask]
            for i, use_new in enumerate(mask.tolist()):
                if use_new:
                    best_neighbor[i] = idx

    rows = [pos[0] for pos in assignments.values()]
    cols = [pos[1] for pos in assignments.values()]
    bounds = (min(rows), max(rows), min(cols), max(cols))
    return assignments, bounds


def render_grid(assignments, bounds, image_paths, outpath: Path):
    """Render the grid-aligned images to disk."""
    if not assignments:
        return

    min_row, max_row, min_col, max_col = bounds
    rows = max_row - min_row + 1
    cols = max_col - min_col + 1

    fig_w = min(30.0, max(4.0, cols * 1.6))
    fig_h = min(30.0, max(4.0, rows * 1.6))
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h))

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

    for idx, (row, col) in assignments.items():
        rr = row - min_row
        cc = col - min_col
        ax = axes[rr, cc]
        try:
            with Image.open(image_paths[idx]) as img:
                img_rgb = img.convert("RGB")
            ax.imshow(img_rgb)
        except Exception:
            continue

    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def _mean_pairwise_distance(embeddings: np.ndarray, max_points: int = 2000, random_state: int = 0):
    n = embeddings.shape[0]
    if n < 2:
        return {"value": None, "computed_on": n, "sampled": False}

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
        mean_dist = 0.0
    else:
        mean_dist = float(dists[iu].mean())
    return {"value": mean_dist, "computed_on": sample_size, "sampled": sampled}


def _greedy_k_center_radius(embeddings: np.ndarray, k: int):
    n = embeddings.shape[0]
    if n == 0 or k <= 0:
        return None
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

    return float(dist_to_centers.max())


def compute_embedding_metrics(
    embeddings: np.ndarray,
    *,
    random_state: int = 0,
    pairwise_sample_limit: int = 2000,
    k_values=None,
):
    if k_values is None:
        k_values = [1, 5, 10, 20]

    metrics = {
        "num_embeddings": int(embeddings.shape[0]),
        "embedding_dim": int(embeddings.shape[1]) if embeddings.size else 0,
    }

    mpd = _mean_pairwise_distance(embeddings, max_points=pairwise_sample_limit, random_state=random_state)
    metrics["mean_pairwise_distance"] = mpd

    n = metrics["num_embeddings"]
    k_radius = {}
    for k in k_values:
        radius = _greedy_k_center_radius(embeddings, k)
        if radius is not None:
            k_radius[str(k)] = {
                "radius": radius,
                "effective_k": min(k, n),
            }
    metrics["k_center_radius"] = k_radius

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Embed and visualize images from an experiment archive using OpenCLIP + UMAP/TSNE")
    parser.add_argument("--experiment-dir", required=True, help="Path to experiment directory containing archive/images/*.png")
    parser.add_argument("--model", default="ViT-B-32", help="OpenCLIP model name (default: ViT-B-32)")
    parser.add_argument("--pretrained", default="openai", help="Pretrained weights key for open_clip.create_model_and_transforms (default: openai)")
    parser.add_argument("--method", choices=["umap", "tsne", "pca"], default="umap", help="Dimensionality reduction method")
    parser.add_argument("--batch-size", type=int, default=64, help="Image embedding batch size")
    parser.add_argument("--thumbs-limit", type=int, default=200, help="Max number of thumbnails to draw on the plot")
    parser.add_argument(
        "--archive-limit",
        type=int,
        default=None,
        help="Only consider the first N archive images before embedding (default: all).",
    )
    parser.add_argument("--device", default=None, help="torch device string (auto-detect if not provided)")
    parser.add_argument(
        "--pairwise-sample-limit",
        type=int,
        default=2000,
        help="Max number of embeddings used when computing mean pairwise distance (default: 2000)",
    )
    parser.add_argument(
        "--k-center-values",
        type=str,
        default="1,5,10,20",
        help="Comma-separated list of k values for k-center radius computation (default: 1,5,10,20)",
    )
    args = parser.parse_args()

    exp_dir = Path(args.experiment_dir)
    if not exp_dir.exists():
        print(f"Experiment directory does not exist: {exp_dir}")
        sys.exit(1)

    image_paths = load_image_paths(exp_dir)
    if args.archive_limit is not None:
        if args.archive_limit <= 0:
            print("--archive-limit must be a positive integer.")
            sys.exit(1)
        image_paths = image_paths[: args.archive_limit]
    if len(image_paths) == 0:
        print("No PNG images found in archive/images")
        sys.exit(1)

    device = torch.device(args.device) if args.device else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    print(f"Using device: {device}")

    print(f"Loading OpenCLIP model {args.model} ({args.pretrained})...")
    model, _, preprocess = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained)
    model.to(device)

    print(f"Embedding {len(image_paths)} images (batch_size={args.batch_size})...")
    filenames, embeddings = embed_images(model, preprocess, image_paths, device, batch_size=args.batch_size)

    # save embeddings
    emb_out = exp_dir / "embeddings_openclip.npz"
    np.savez_compressed(emb_out, filenames=np.array(filenames), embeddings=embeddings)
    print(f"Saved embeddings to {emb_out}")

    try:
        k_values = [int(v) for v in args.k_center_values.split(",") if v.strip()]
        if len(k_values) == 0:
            raise ValueError
    except ValueError:
        print("Invalid --k-center-values: provide a comma-separated list of integers.")
        sys.exit(1)

    metrics = compute_embedding_metrics(
        embeddings,
        random_state=0,
        pairwise_sample_limit=max(2, args.pairwise_sample_limit),
        k_values=k_values,
    )
    metrics_out = exp_dir / "embedding_metrics.json"
    with metrics_out.open("w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved embedding coverage metrics to {metrics_out}")

    print(f"Reducing to 2D using {args.method}...")
    coords = reduce_embeddings(embeddings, method=args.method)

    viz_out = exp_dir / f"embed_viz_{args.method}.png"
    print(f"Creating visualization (thumbnails limit={args.thumbs_limit}) -> {viz_out}")
    plot_coords(coords, [p for p in image_paths], viz_out, thumbs_limit=args.thumbs_limit)

    grid_assignments, grid_bounds = layout_embeddings_to_grid(coords)
    grid_out = exp_dir / f"embed_grid_{args.method}.png"
    print(f"Rendering grid approximation -> {grid_out}")
    render_grid(grid_assignments, grid_bounds, image_paths, grid_out)

    print("Done.")


if __name__ == "__main__":
    main()
