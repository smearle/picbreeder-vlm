#!/usr/bin/env python3
"""Compute CLIP noun similarity metrics for an archive of images.

This utility mirrors the archive layout expected by embed_and_visualize.py:
  <experiment_dir>/archive/images/*.png

It embeds every archive image, embeds every noun in a provided noun list, and
then measures the maximum cosine similarity between each noun and any image.
The resulting statistics are saved to JSON for further analysis.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image
import torch
from tqdm import tqdm

try:
    import open_clip
except Exception as exc:  # pragma: no cover - import guard
    raise RuntimeError("open_clip import failed. Is `open_clip_torch` installed?") from exc


def load_image_paths(experiment_dir: Path) -> List[Path]:
    """Return sorted PNG image paths from <experiment_dir>/archive/images."""
    images_dir = experiment_dir / "archive" / "images"
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    return sorted(images_dir.glob("*.png"))


def load_nouns(noun_file: Path) -> List[str]:
    """Read newline-delimited nouns, ignoring blank lines."""
    if not noun_file.exists():
        raise FileNotFoundError(f"Noun file not found: {noun_file}")
    nouns: List[str] = []
    for line in noun_file.read_text().splitlines():
        noun = line.strip()
        if noun:
            nouns.append(noun)
    if not nouns:
        raise ValueError("No nouns loaded from the provided noun list.")
    return nouns


def batch(iterable: Sequence, n: int) -> Iterable[Sequence]:
    """Yield slices of size n from the sequence."""
    length = len(iterable)
    for i in range(0, length, n):
        yield iterable[i : i + n]


def embed_images(
    model: torch.nn.Module,
    preprocess,
    image_paths: Sequence[Path],
    device: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    """Return L2-normalized CLIP embeddings for a list of image paths."""
    model.eval()
    embeddings = []
    total = math.ceil(len(image_paths) / batch_size)
    with torch.no_grad():
        for chunk in tqdm(
            batch(image_paths, batch_size),
            total=total,
            desc="Embedding images",
        ):
            tensors = []
            for image_path in chunk:
                img = Image.open(image_path).convert("RGB")
                tensors.append(preprocess(img))
            x = torch.stack(tensors, dim=0).to(device)
            emb = model.encode_image(x)
            emb = emb.cpu().numpy()
            norm = np.linalg.norm(emb, axis=1, keepdims=True)
            norm[norm == 0] = 1.0
            embeddings.append(emb / norm)
    return np.vstack(embeddings)


def embed_texts(
    model: torch.nn.Module,
    tokenizer,
    texts: Sequence[str],
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    """Return L2-normalized CLIP text embeddings for the provided prompts."""
    model.eval()
    embeddings = []
    total = math.ceil(len(texts) / batch_size)
    with torch.no_grad():
        for chunk in tqdm(
            batch(texts, batch_size),
            total=total,
            desc="Embedding nouns",
        ):
            tokens = tokenizer(list(chunk)).to(device)
            emb = model.encode_text(tokens)
            emb = emb.cpu().numpy()
            norm = np.linalg.norm(emb, axis=1, keepdims=True)
            norm[norm == 0] = 1.0
            embeddings.append(emb / norm)
    return np.vstack(embeddings)


def format_prompts(nouns: Sequence[str], template: str) -> List[str]:
    """Format nouns using the provided template containing {label}."""
    if "{label}" not in template:
        raise ValueError("--label-template must contain '{label}' placeholder.")
    prompts = []
    for noun in nouns:
        prompts.append(template.format(label=noun))
    return prompts


def compute_max_similarities(
    image_embeddings: np.ndarray,
    noun_embeddings: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """Return per-noun max cosine similarity and their mean."""
    if image_embeddings.size == 0:
        raise ValueError("No image embeddings available for similarity calculation.")
    sims = image_embeddings @ noun_embeddings.T
    max_per_noun = np.max(sims, axis=0)
    return max_per_noun, float(max_per_noun.mean())


def parse_args():
    parser = argparse.ArgumentParser(description="Compute CLIP noun coverage metrics for an archive.")
    parser.add_argument("--experiment-dir", required=True, help="Path to experiment directory (contains archive/images).")
    parser.add_argument("--noun-file", default="nounlist.txt", help="Path to newline-delimited noun list.")
    parser.add_argument("--model", default="ViT-H-14", help="OpenCLIP model name (default: ViT-H-14).")
    parser.add_argument("--pretrained", default="laion2b_s32b_b79k", help="Pretrained weights identifier for OpenCLIP.")
    parser.add_argument("--batch-size", type=int, default=64, help="Image embedding batch size (default: 64).")
    parser.add_argument("--noun-batch-size", type=int, default=512, help="Text embedding batch size (default: 512).")
    parser.add_argument("--device", default=None, help="Torch device (default: auto-detect).")
    parser.add_argument(
        "--archive-limit",
        type=int,
        default=None,
        help="Only consider the first N archive images (default: all images).",
    )
    parser.add_argument(
        "--label-template",
        default="{label}",
        help="Python format string applied to each noun (default: '{label}').",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional output path for metrics JSON (default: <experiment_dir>/noun_similarity_metrics.json).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    exp_dir = Path(args.experiment_dir)
    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment directory does not exist: {exp_dir}")

    image_paths = load_image_paths(exp_dir)
    if args.archive_limit is not None:
        if args.archive_limit <= 0:
            raise ValueError("--archive-limit must be a positive integer.")
        image_paths = image_paths[: args.archive_limit]
    if not image_paths:
        raise RuntimeError("No PNG images found in archive/images.")

    noun_file = Path(args.noun_file)
    nouns = load_nouns(noun_file)
    prompts = format_prompts(nouns, args.label_template)

    device = torch.device(args.device) if args.device else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    print(f"Using device: {device}")

    print(f"Loading OpenCLIP model {args.model} ({args.pretrained})...")
    model, _, preprocess = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained)
    model.to(device)
    tokenizer = open_clip.get_tokenizer(args.model)

    print(f"Embedding {len(image_paths)} images...")
    image_embeddings = embed_images(model, preprocess, image_paths, device, batch_size=args.batch_size)

    print(f"Embedding {len(prompts)} noun prompts...")
    noun_embeddings = embed_texts(model, tokenizer, prompts, device, batch_size=args.noun_batch_size)

    max_per_noun, mean_similarity = compute_max_similarities(image_embeddings, noun_embeddings)

    metrics = {
        "experiment_dir": str(exp_dir),
        "num_images": len(image_paths),
        "num_nouns": len(nouns),
        "model": args.model,
        "pretrained": args.pretrained,
        "label_template": args.label_template,
        "mean_max_similarity": mean_similarity,
        "max_similarity_per_noun": {noun: float(score) for noun, score in zip(nouns, max_per_noun)},
    }
    output_path = Path(args.output_json) if args.output_json else exp_dir / "noun_similarity_metrics.json"
    with output_path.open("w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved noun similarity metrics to {output_path}")
    print(f"Mean of per-noun max cosine similarity: {mean_similarity:.4f}")


if __name__ == "__main__":
    main()
