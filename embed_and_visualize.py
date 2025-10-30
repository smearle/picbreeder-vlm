#!/usr/bin/env python3
"""Embed and visualize images from an experiment archive using OpenCLIP + UMAP/TSNE.

This script expects PNG images under: <experiment_dir>/archive/images/*.png

Example:
  python scripts/embed_and_visualize.py --experiment-dir logs_collaborative/exp123

Outputs saved to the experiment directory:
  - embeddings_openclip.npz    (filenames + embeddings)
  - embed_viz_umap.png         (scatter visualization)

"""
from pathlib import Path
import argparse
import sys
import math
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


def main():
    parser = argparse.ArgumentParser(description="Embed and visualize images from an experiment archive using OpenCLIP + UMAP/TSNE")
    parser.add_argument("--experiment-dir", required=True, help="Path to experiment directory containing archive/images/*.png")
    parser.add_argument("--model", default="ViT-B-32", help="OpenCLIP model name (default: ViT-B-32)")
    parser.add_argument("--pretrained", default="openai", help="Pretrained weights key for open_clip.create_model_and_transforms (default: openai)")
    parser.add_argument("--method", choices=["umap", "tsne", "pca"], default="umap", help="Dimensionality reduction method")
    parser.add_argument("--batch-size", type=int, default=64, help="Image embedding batch size")
    parser.add_argument("--thumbs-limit", type=int, default=200, help="Max number of thumbnails to draw on the plot")
    parser.add_argument("--device", default=None, help="torch device string (auto-detect if not provided)")
    args = parser.parse_args()

    exp_dir = Path(args.experiment_dir)
    if not exp_dir.exists():
        print(f"Experiment directory does not exist: {exp_dir}")
        sys.exit(1)

    image_paths = load_image_paths(exp_dir)
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

    print(f"Reducing to 2D using {args.method}...")
    coords = reduce_embeddings(embeddings, method=args.method)

    viz_out = exp_dir / f"embed_viz_{args.method}.png"
    print(f"Creating visualization (thumbnails limit={args.thumbs_limit}) -> {viz_out}")
    plot_coords(coords, [p for p in image_paths], viz_out, thumbs_limit=args.thumbs_limit)

    print("Done.")


if __name__ == "__main__":
    main()
