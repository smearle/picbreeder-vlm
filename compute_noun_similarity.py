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
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw
import torch
from tqdm import tqdm

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import hydra
from hydra.conf import HelpConf, HydraConf
from hydra.core.config_store import ConfigStore
from hydra.utils import get_original_cwd

try:
    import open_clip
except Exception as exc:  # pragma: no cover - import guard
    raise RuntimeError("open_clip import failed. Is `open_clip_torch` installed?") from exc

from config import PicbreederConfig, ensure_valid_config
from rendering import try_load_font, create_captioned_grid
from utils import _ensure_absolute, resolve_nounlist
from model_loader import prepare_model


@dataclass
class NounSimilarityConfig(PicbreederConfig):
    embedding_model: str = "ViT-SO400M-14-SigLIP2"
    pretrained: str = "webli"
    batch_size: int = 64
    noun_batch_size: int = 512
    device: Optional[str] = None
    archive_limit: Optional[int] = None
    label_template: str = "{label}"
    output_json: Optional[Path] = None
    output_trajectory_json: Optional[Path] = None
    output_trajectory_plot: Optional[Path] = None
    render_grid: bool = False
    output_grid: Optional[Path] = None
    grid_thumb_size: int = 192
    grid_margin: int = 12
    grid_font_size: int = 10
    grid_top_k: Optional[int] = 900
    hydra: HydraConf = field(
        default_factory=lambda: HydraConf(
            help=HelpConf(
                app_name="compute_noun_similarity",
                header=(
                    "Hydra entry point for noun coverage metrics.\n"
                    "\n"
                    "Common overrides:\n"
                    "  nounlist           Noun list name (e.g. imagenet_leaves) or path.\n"
                    "  label_template    Format each noun (must include {label}).\n"
                    "  render_grid       Emit a noun grid sorted by max similarity.\n"
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
    if cfg.render_grid:
        if cfg.grid_thumb_size <= 0:
            raise ValueError("grid_thumb_size must be positive when render_grid is enabled")
        if cfg.grid_margin < 0:
            raise ValueError("grid_margin must be non-negative when render_grid is enabled")
        if cfg.grid_font_size <= 0:
            raise ValueError("grid_font_size must be positive when render_grid is enabled")
        if cfg.grid_top_k is not None and cfg.grid_top_k <= 0:
            raise ValueError("grid_top_k must be positive when provided")


def prepare_openclip_components(
    cfg: NounSimilarityConfig,
    device: torch.device,
):
    """Create the OpenCLIP model + preprocess + tokenizer for this config."""
    return prepare_model(cfg, device)


def prepare_noun_text_embeddings(
    cfg: NounSimilarityConfig,
    *,
    original_cwd: Path,
    device: torch.device,
    model=None,
    tokenizer=None,
    nouns: Optional[Sequence[str]] = None,
    prompts: Optional[Sequence[str]] = None,
) -> tuple[list[str], list[str], np.ndarray]:
    """Load/format nouns and embed them once.

    Intended for sweep callers to compute noun embeddings once and pass them into main.
    If model/tokenizer are not provided, they will be created.
    """

    validated_cfg = ensure_valid_config(cfg, original_cwd=original_cwd)
    _validate_noun_similarity_options(validated_cfg)

    if nouns is None:
        noun_file = resolve_nounlist(validated_cfg.nounlist, original_cwd)
        nouns_list = load_nouns(noun_file)
    else:
        nouns_list = [str(noun) for noun in nouns]
        if not nouns_list:
            raise ValueError("Provided nouns list is empty")

    if prompts is None:
        prompts_list = format_prompts(nouns_list, validated_cfg.label_template)
    else:
        prompts_list = [str(prompt) for prompt in prompts]
        if len(prompts_list) != len(nouns_list):
            raise ValueError("prompts must have the same length as nouns")

    if tokenizer is None or model is None:
        model, _, tokenizer = prepare_openclip_components(validated_cfg, device)

    noun_embeddings = embed_texts(
        model,
        tokenizer,
        prompts_list,
        device,
        batch_size=validated_cfg.noun_batch_size,
    )
    return nouns_list, prompts_list, noun_embeddings


def load_image_paths(experiment_dir: Path) -> List[Path]:
    """Return sorted PNG image paths from <experiment_dir>/archive/images."""
    images_dir = experiment_dir / "archive" / "images"
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    return sorted(images_dir.glob("*.png"))


def _numeric_suffix(path: Path) -> int:
    stem = path.stem
    parts = stem.split("_")
    try:
        return int(parts[-1])
    except ValueError:
        return 0


def infer_archive_order(experiment_dir: Path) -> List[Path]:
    """Infer archive insertion order, mirroring plot_novelty_over_time.py behavior.

    Priority:
      1) <experiment_dir>/archive/archive_metadata.json if present
      2) Otherwise, fall back to filename ordering (numeric suffix if present)
    """
    archive_dir = experiment_dir / "archive"
    images_dir = archive_dir / "images"
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    ordered: List[Path] = []
    seen = set()
    metadata_path = archive_dir / "archive_metadata.json"

    if metadata_path.exists():
        try:
            data = json.loads(metadata_path.read_text(encoding="utf-8"))
            for entry in data.get("entries", []):
                raw_path = entry.get("image_path")
                img_id = str(entry.get("id", "")).strip()

                candidate = None
                if raw_path:
                    candidate = Path(raw_path)
                    if not candidate.is_absolute():
                        candidate = images_dir / candidate
                if candidate is None or not candidate.exists():
                    if img_id:
                        filename = f"{img_id}.png" if not img_id.endswith(".png") else img_id
                        candidate = images_dir / filename

                if candidate and candidate.exists() and candidate.name not in seen:
                    ordered.append(candidate)
                    seen.add(candidate.name)
        except Exception as exc:
            print(f"Warning: failed to parse archive metadata ({exc}); falling back to filename ordering.")

    remaining = sorted(images_dir.glob("*.png"), key=_numeric_suffix)
    for path in remaining:
        if path.name in seen:
            continue
        ordered.append(path)
        seen.add(path.name)

    if not ordered:
        raise RuntimeError(f"No PNG images found under {images_dir}")

    return ordered


def load_nouns(noun_file: Path) -> List[str]:
    """Read newline-delimited nouns, ignoring blank lines."""
    if not noun_file.exists():
        raise FileNotFoundError(f"Noun file not found: {noun_file}")
    nouns: List[str] = []
    for line in noun_file.read_text().splitlines():
        noun = line.strip().replace("_", " ")
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
) -> Tuple[np.ndarray, float, np.ndarray]:
    """Return per-noun max cosine similarity, their mean, and argmax image indices."""
    if image_embeddings.size == 0:
        raise ValueError("No image embeddings available for similarity calculation.")
    sims = image_embeddings @ noun_embeddings.T
    max_per_noun = np.max(sims, axis=0)
    best_image_indices = np.argmax(sims, axis=0)
    return max_per_noun, float(max_per_noun.mean()), best_image_indices


def compute_mean_max_similarity_trajectory(
    image_embeddings: np.ndarray,
    noun_embeddings: np.ndarray,
    image_paths: Sequence[Path],
) -> List[Dict[str, object]]:
    """Compute running mean of per-noun max cosine similarity as images are added."""
    if image_embeddings.size == 0:
        raise ValueError("No image embeddings available for trajectory calculation.")
    if noun_embeddings.size == 0:
        raise ValueError("No noun embeddings available for trajectory calculation.")
    if len(image_paths) != image_embeddings.shape[0]:
        raise ValueError("image_paths length must match number of image embeddings")

    max_per_noun = np.full((noun_embeddings.shape[0],), -np.inf, dtype=np.float32)
    results: List[Dict[str, object]] = []

    for idx in range(image_embeddings.shape[0]):
        sims = image_embeddings[idx] @ noun_embeddings.T
        max_per_noun = np.maximum(max_per_noun, sims)
        results.append(
            {
                "index": int(idx + 1),
                "image": image_paths[idx].name,
                "mean_max_similarity": float(max_per_noun.mean()),
            }
        )

    return results


def plot_mean_max_similarity_trajectory(results: Sequence[Dict[str, object]], outpath: Path) -> None:
    steps = [int(row["index"]) for row in results]
    vals = [float(row["mean_max_similarity"]) for row in results]

    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    ax.plot(steps, vals, color="#1f77b4", linewidth=2)
    ax.set_title("Noun coverage (mean per-noun max similarity) over archive growth")
    ax.set_xlabel("Archive insertion order")
    ax.set_ylabel("Mean of per-noun max cosine similarity")
    ax.grid(True, which="major", alpha=0.3)

    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=150)
    plt.close(fig)





def save_trajectory_json(results: Sequence[Dict[str, object]], outpath: Path) -> None:
    serializable = [
        {
            "index": int(row["index"]),
            "image": str(row["image"]),
            "mean_max_similarity": float(row["mean_max_similarity"]),
        }
        for row in results
    ]
    outpath.parent.mkdir(parents=True, exist_ok=True)
    outpath.write_text(json.dumps(serializable, indent=2), encoding="utf-8")


def render_noun_similarity_grid(
    nouns: Sequence[str],
    max_per_noun: np.ndarray,
    best_image_indices: np.ndarray,
    image_paths: Sequence[Path],
    output_path: Path,
    thumb_size: int,
    margin: int,
    font_size: int,
    top_k: Optional[int] = None,
) -> None:
    order = np.argsort(-max_per_noun)
    if top_k is not None:
        order = order[:top_k]

    images: List[Image.Image] = []
    captions: List[str] = []
    cache: Dict[Path, Image.Image] = {}

    for noun_idx in order:
        image_idx = int(best_image_indices[noun_idx])
        image_path = image_paths[image_idx]
        cached = cache.get(image_path)
        if cached is None:
            cached = Image.open(image_path).convert("RGB")
            cache[image_path] = cached
        images.append(cached)
        distance = 1.0 - float(max_per_noun[noun_idx])
        captions.append(f"{nouns[noun_idx]} (dist {distance:.3f})")

    grid = create_captioned_grid(images, captions, thumb_size, margin, font_size)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output_path, format="PNG")


def _resolve_optional_path(value: Optional[Path], base: Path) -> Optional[Path]:
    if value is None:
        return None
    return _ensure_absolute(Path(value), base)


@hydra.main(version_base="1.3", config_path=None, config_name="noun_similarity_base")
def main(
    cfg: NounSimilarityConfig,
    *,
    model=None,
    preprocess=None,
    tokenizer=None,
    nouns: Optional[Sequence[str]] = None,
    prompts: Optional[Sequence[str]] = None,
    noun_embeddings: Optional[np.ndarray] = None,
    original_cwd_override: Optional[Path] = None,
) -> None:
    if original_cwd_override:
        original_cwd = original_cwd_override
    else:
        try:
            original_cwd = Path(get_original_cwd())
        except ValueError:
            # Fallback if hydra is not initialized (e.g. called directly)
            original_cwd = Path.cwd()
            
    validated_cfg = ensure_valid_config(cfg, original_cwd=original_cwd)
    _validate_noun_similarity_options(validated_cfg)

    exp_dir = Path(validated_cfg.experiment_dir).resolve()
    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment directory does not exist: {exp_dir}")

    image_paths = infer_archive_order(exp_dir)
    if validated_cfg.archive_limit is not None:
        image_paths = image_paths[: validated_cfg.archive_limit]
    if not image_paths:
        raise RuntimeError("No PNG images found in archive/images.")

    if nouns is None:
        noun_file = resolve_nounlist(validated_cfg.nounlist, original_cwd)
        nouns_list = load_nouns(noun_file)
    else:
        nouns_list = [str(noun) for noun in nouns]
        if not nouns_list:
            raise ValueError("Provided nouns list is empty")

    if prompts is None:
        prompts_list = format_prompts(nouns_list, validated_cfg.label_template)
    else:
        prompts_list = [str(prompt) for prompt in prompts]
        if len(prompts_list) != len(nouns_list):
            raise ValueError("prompts must have the same length as nouns")

    if validated_cfg.device:
        device = torch.device(validated_cfg.device)
    else:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    if (model is None) ^ (preprocess is None):
        raise ValueError("Provide both model and preprocess, or neither.")
    if (tokenizer is None) and (model is not None):
        raise ValueError("Provide tokenizer when providing a pre-built model.")

    if noun_embeddings is not None and nouns is None:
        # We need the base noun labels to write per-noun metrics.
        raise ValueError("Provide nouns when providing noun_embeddings")

    if model is None:
        print(f"Loading OpenCLIP model {validated_cfg.embedding_model} ({validated_cfg.pretrained})...")
        model, preprocess, tokenizer = prepare_openclip_components(validated_cfg, device)
    else:
        model.to(device)
        model.eval()

    print(f"Embedding {len(image_paths)} images...")
    image_embeddings = embed_images(
        model,
        preprocess,
        image_paths,
        device,
        batch_size=validated_cfg.batch_size,
    )

    if noun_embeddings is None:
        print(f"Embedding {len(prompts_list)} noun prompts...")
        noun_embeddings = embed_texts(
            model,
            tokenizer,
            prompts_list,
            device,
            batch_size=validated_cfg.noun_batch_size,
        )
    else:
        noun_embeddings = np.asarray(noun_embeddings)
        if noun_embeddings.ndim != 2 or noun_embeddings.shape[0] != len(prompts_list):
            raise ValueError(
                "noun_embeddings must be a 2D array with shape (num_nouns, embed_dim) matching prompts"
            )

    max_per_noun, mean_similarity, best_image_indices = compute_max_similarities(
        image_embeddings,
        noun_embeddings,
    )

    trajectory = compute_mean_max_similarity_trajectory(image_embeddings, noun_embeddings, image_paths)

    metrics = {
        "experiment_dir": str(exp_dir),
        "num_images": len(image_paths),
        "num_nouns": len(nouns_list),
        "model": validated_cfg.embedding_model,
        "pretrained": validated_cfg.pretrained,
        "label_template": validated_cfg.label_template,
        "mean_max_similarity": mean_similarity,
        "max_similarity_per_noun": {noun: float(score) for noun, score in zip(nouns_list, max_per_noun)},
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

    nounlist_name = Path(validated_cfg.nounlist).stem
    model_name_sanitized = validated_cfg.embedding_model.replace("/", "-")
    trajectory_json = _resolve_optional_path(validated_cfg.output_trajectory_json, original_cwd)
    if trajectory_json is None:
        trajectory_json = exp_dir / f"noun_similarity_over_time_{nounlist_name}_{model_name_sanitized}.json"
    trajectory_plot = _resolve_optional_path(validated_cfg.output_trajectory_plot, original_cwd)
    if trajectory_plot is None:
        trajectory_plot = exp_dir / f"noun_similarity_over_time_{nounlist_name}_{model_name_sanitized}.png"

    save_trajectory_json(trajectory, trajectory_json)
    plot_mean_max_similarity_trajectory(trajectory, trajectory_plot)
    print(f"Saved noun similarity trajectory JSON to {trajectory_json}")
    print(f"Saved noun similarity trajectory plot to {trajectory_plot}")

    if validated_cfg.render_grid:
        grid_output = _resolve_optional_path(validated_cfg.output_grid, original_cwd)
        if grid_output is None:
            suffix = f"_top{validated_cfg.grid_top_k}" if validated_cfg.grid_top_k else ""
            grid_output = exp_dir / f"noun_similarity_grid_{nounlist_name}_{model_name_sanitized}{suffix}.png"
        render_noun_similarity_grid(
            nouns_list,
            max_per_noun,
            best_image_indices,
            image_paths,
            grid_output,
            validated_cfg.grid_thumb_size,
            validated_cfg.grid_margin,
            validated_cfg.grid_font_size,
            top_k=validated_cfg.grid_top_k,
        )
        print(f"Saved noun similarity grid to {grid_output}")


if __name__ == "__main__":
    main()
