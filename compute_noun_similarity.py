#!/usr/bin/env python3
"""Compute CLIP noun similarity metrics for an archive of images.

This utility mirrors the archive layout expected by embed_and_visualize.py:
  <experiment_dir>/archive/images/*.png

It embeds every archive image, embeds every noun in a provided noun list, and
then measures the maximum cosine similarity between each noun and any image.
The resulting statistics are saved to JSON for further analysis.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
import torch
from tqdm import tqdm

import hydra
from hydra.conf import HelpConf, HydraConf
from hydra.core.config_store import ConfigStore
from hydra.utils import get_original_cwd

try:
    import open_clip
except Exception as exc:  # pragma: no cover - import guard
    raise RuntimeError("open_clip import failed. Is `open_clip_torch` installed?") from exc

from config import CollaborativeConfig, ensure_valid_config
from utils import _ensure_absolute


@dataclass
class NounSimilarityConfig(CollaborativeConfig):
    noun_file: Path = Path("nounlist.txt")
    model: str = "ViT-H-14"
    pretrained: str = "laion2b_s32b_b79k"
    batch_size: int = 64
    noun_batch_size: int = 512
    device: Optional[str] = None
    archive_limit: Optional[int] = None
    label_template: str = "{label}"
    output_json: Optional[Path] = None
    hydra: HydraConf = field(
        default_factory=lambda: HydraConf(
            help=HelpConf(
                app_name="compute_noun_similarity",
                header=(
                    "Hydra entry point for noun coverage metrics.\n"
                    "\n"
                    "Common overrides:\n"
                    "  noun_file          Path to noun list (defaults to nounlist.txt).\n"
                    "  label_template    Format each noun (must include {label}).\n"
                    "  experiment_dir    Override to target a specific archive directory.\n"
                ),
                footer="Override with +option=value (e.g. noun_file=data/nouns.txt).",
            )
        )
    )


ConfigStore.instance().store(name="noun_similarity_base", node=NounSimilarityConfig)


def _validate_noun_similarity_options(cfg: NounSimilarityConfig) -> None:
    if cfg.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if cfg.noun_batch_size <= 0:
        raise ValueError("noun_batch_size must be positive")
    if cfg.archive_limit is not None and cfg.archive_limit <= 0:
        raise ValueError("archive_limit must be a positive integer when provided")
    if "{label}" not in cfg.label_template:
        raise ValueError("label_template must contain '{label}' placeholder")


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


def _resolve_optional_path(value: Optional[Path], base: Path) -> Optional[Path]:
    if value is None:
        return None
    return _ensure_absolute(Path(value), base)


@hydra.main(version_base="1.3", config_path=None, config_name="noun_similarity_base")
def main(cfg: NounSimilarityConfig) -> None:
    original_cwd = Path(get_original_cwd())
    validated_cfg = ensure_valid_config(cfg, original_cwd=original_cwd)
    _validate_noun_similarity_options(validated_cfg)

    exp_dir = Path(validated_cfg.experiment_dir).resolve()
    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment directory does not exist: {exp_dir}")

    image_paths = load_image_paths(exp_dir)
    if validated_cfg.archive_limit is not None:
        image_paths = image_paths[: validated_cfg.archive_limit]
    if not image_paths:
        raise RuntimeError("No PNG images found in archive/images.")

    noun_file = _ensure_absolute(Path(validated_cfg.noun_file), original_cwd)
    nouns = load_nouns(noun_file)
    prompts = format_prompts(nouns, validated_cfg.label_template)

    if validated_cfg.device:
        device = torch.device(validated_cfg.device)
    else:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    print(f"Loading OpenCLIP model {validated_cfg.model} ({validated_cfg.pretrained})...")
    model, _, preprocess = open_clip.create_model_and_transforms(validated_cfg.model, pretrained=validated_cfg.pretrained)
    model.to(device)
    tokenizer = open_clip.get_tokenizer(validated_cfg.model)

    print(f"Embedding {len(image_paths)} images...")
    image_embeddings = embed_images(
        model,
        preprocess,
        image_paths,
        device,
        batch_size=validated_cfg.batch_size,
    )

    print(f"Embedding {len(prompts)} noun prompts...")
    noun_embeddings = embed_texts(
        model,
        tokenizer,
        prompts,
        device,
        batch_size=validated_cfg.noun_batch_size,
    )

    max_per_noun, mean_similarity = compute_max_similarities(image_embeddings, noun_embeddings)

    metrics = {
        "experiment_dir": str(exp_dir),
        "num_images": len(image_paths),
        "num_nouns": len(nouns),
        "model": validated_cfg.model,
        "pretrained": validated_cfg.pretrained,
        "label_template": validated_cfg.label_template,
        "mean_max_similarity": mean_similarity,
        "max_similarity_per_noun": {noun: float(score) for noun, score in zip(nouns, max_per_noun)},
    }

    output_path = _resolve_optional_path(validated_cfg.output_json, original_cwd)
    if output_path is None:
        output_path = exp_dir / "noun_similarity_metrics.json"
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved noun similarity metrics to {output_path}")
    print(f"Mean of per-noun max cosine similarity: {mean_similarity:.4f}")


if __name__ == "__main__":
    main()
