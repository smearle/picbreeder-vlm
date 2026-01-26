#!/usr/bin/env python3
"""Compute CLIP noun similarity metrics for an archive of images.

This utility mirrors the archive layout expected by embed_and_visualize.py:
  <experiment_dir>/archive/images/*.png

It embeds every archive image, embeds every noun in a provided noun list, and
then measures the maximum cosine similarity between each noun and any image.
The resulting statistics are saved to JSON for further analysis.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
import pickle
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from scipy.optimize import linear_sum_assignment
import numpy as np
from PIL import Image, ImageDraw
import torch
from tqdm import tqdm

import matplotlib

from constants import NOUN_LISTS_DIR

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
from rendering import try_load_font, create_captioned_grid, save_captioned_grid_pdf
from utils import _ensure_absolute, resolve_nounlist
from model_loader import prepare_model, embed_images


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
    grid_font_size: int = 20
    grid_top_k: Optional[int] = 16
    negative_anchors: str = "negative_anchors"
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


def load_or_compute_noun_embeddings(
    cfg: NounSimilarityConfig,
    original_cwd: Path,
    device: torch.device,
    prompts_list: List[str],
    model: torch.nn.Module,
    tokenizer,
) -> np.ndarray:
    noun_embed_cache_dir = original_cwd / "noun_lists" / "embeddings"
    noun_embed_cache_dir.mkdir(parents=True, exist_ok=True)
    
    model_name_sanitized = cfg.embedding_model.replace("/", "-")
    pretrained_sanitized = str(cfg.pretrained).replace("/", "-")
    nounlist_stem = Path(cfg.nounlist).stem
    
    cache_path = noun_embed_cache_dir / f"noun_embeddings_{nounlist_stem}_{model_name_sanitized}_{pretrained_sanitized}.npy"
    
    if cache_path.exists():
        try:
            print(f"Loading cached noun embeddings from {cache_path}...")
            emb = np.load(cache_path)
            if emb.shape[0] == len(prompts_list):
                return emb
            print(f"Cached noun embeddings shape mismatch. Recomputing.")
        except Exception as e:
            print(f"Failed to load cached noun embeddings: {e}")
            
    print(f"Embedding {len(prompts_list)} noun prompts...")
    emb = embed_texts(model, tokenizer, prompts_list, device, batch_size=cfg.noun_batch_size)
    
    print(f"Saving noun embeddings to {cache_path}")
    try:
        np.save(cache_path, emb)
    except Exception as e:
        print(f"Failed to save noun embeddings cache: {e}")
        
    return emb


def load_or_compute_negative_embeddings(
    cfg: NounSimilarityConfig,
    original_cwd: Path,
    device: torch.device,
    model: torch.nn.Module,
    tokenizer,
) -> tuple[Optional[np.ndarray], List[str]]:
    if not cfg.negative_anchors:
        return None, []
        
    anchors_name = cfg.negative_anchors
    anchors_path = original_cwd / "noun_lists" / f"{anchors_name}.txt"
    
    neg_anchors_list = []
    if anchors_path.exists():
        neg_anchors_list = load_nouns(anchors_path)
    else:
        print(f"Warning: Negative anchors file {anchors_path} not found.")
        return None, []
        
    noun_embed_cache_dir = original_cwd / "noun_lists" / "embeddings"
    noun_embed_cache_dir.mkdir(parents=True, exist_ok=True)
    
    model_name_sanitized = cfg.embedding_model.replace("/", "-")
    pretrained_sanitized = str(cfg.pretrained).replace("/", "-")
    
    cache_path = noun_embed_cache_dir / f"negative_anchors_{anchors_name}_{model_name_sanitized}_{pretrained_sanitized}.npy"
    
    if cache_path.exists():
        try:
            print(f"Loading cached negative embeddings from {cache_path}...")
            emb = np.load(cache_path)
            if emb.shape[0] == len(neg_anchors_list):
                return emb, neg_anchors_list
            print(f"Cached negative embeddings size mismatch. Recomputing.")
        except Exception as e:
            print(f"Failed to load cached negative embeddings: {e}")

    print(f"Embedding {len(neg_anchors_list)} negative anchors...")
    emb = embed_texts(model, tokenizer, neg_anchors_list, device, batch_size=cfg.noun_batch_size)
    
    print(f"Saving negative embeddings to {cache_path}")
    try:
        np.save(cache_path, emb)
    except Exception as e:
        print(f"Failed to save negative embeddings cache: {e}")
        
    return emb, neg_anchors_list


def prepare_noun_text_embeddings(
    cfg: NounSimilarityConfig,
    original_cwd: Path,
    device: torch.device,
    model=None,
    tokenizer=None,
    nouns: Optional[Sequence[str]] = None,
    prompts: Optional[Sequence[str]] = None,
) -> tuple[list[str], list[str], np.ndarray, Optional[np.ndarray]]:
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

    noun_embeddings = load_or_compute_noun_embeddings(
        validated_cfg,
        original_cwd,
        device,
        prompts_list,
        model,
        tokenizer,
    )

    neg_embeddings, _ = load_or_compute_negative_embeddings(
        validated_cfg,
        original_cwd,
        device,
        model,
        tokenizer,
    )

    return nouns_list, prompts_list, noun_embeddings, neg_embeddings


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
    neg_embeddings: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float, np.ndarray, Optional[np.ndarray], Optional[float]]:
    """Return per-noun max cosine similarity, mean, best indices, and optionally contrastive metrics."""
    if image_embeddings.size == 0:
        raise ValueError("No image embeddings available for similarity calculation.")
    
    sims = image_embeddings @ noun_embeddings.T
    max_per_noun = np.max(sims, axis=0)
    best_image_indices = np.argmax(sims, axis=0)
    mean_sim = float(max_per_noun.mean())

    max_contrastive_per_noun = None
    mean_contrastive = None

    if neg_embeddings is not None and neg_embeddings.size > 0:
        # Compute max similarity to any negative anchor for each image
        # shape: (N_images,)
        neg_sims = image_embeddings @ neg_embeddings.T
        max_neg_sim_per_image = np.max(neg_sims, axis=1)

        # Contrastive score: sim(im, noun) - max(sim(im, neg_set))
        # shape: (N_images, N_nouns)
        contrastive_sims = sims - max_neg_sim_per_image[:, None]
        
        max_contrastive_per_noun = np.max(contrastive_sims, axis=0)
        mean_contrastive = float(max_contrastive_per_noun.mean())

    return max_per_noun, mean_sim, best_image_indices, max_contrastive_per_noun, mean_contrastive


def compute_trajectory_metrics(
    image_embeddings: np.ndarray,
    noun_embeddings: np.ndarray,
    image_paths: Sequence[Path],
    neg_embeddings: Optional[np.ndarray],
    contrastive_key: Optional[str] = None,
) -> List[Dict[str, object]]:
    """Compute running mean of metrics as images are added."""
    if image_embeddings.size == 0:
        raise ValueError("No image embeddings available for trajectory calculation.")
    if noun_embeddings.size == 0:
        raise ValueError("No noun embeddings available for trajectory calculation.")
    if len(image_paths) != image_embeddings.shape[0]:
        raise ValueError("image_paths length must match number of image embeddings")

    num_nouns = noun_embeddings.shape[0]
    max_per_noun = np.full((num_nouns,), -np.inf, dtype=np.float32)
    
    if neg_embeddings is not None:
        if contrastive_key is None:
            raise ValueError("contrastive_key must be provided when neg_embeddings is not None")
        max_contrastive_per_noun = np.full((num_nouns,), -np.inf, dtype=np.float32)
    else:
        max_contrastive_per_noun = None

    results: List[Dict[str, object]] = []

    for idx in range(image_embeddings.shape[0]):
        # Current image embedding: (1, D)
        img_emb = image_embeddings[idx : idx + 1]
        
        # Similarities to nouns: (1, N_nouns) -> (N_nouns,)
        sims = (img_emb @ noun_embeddings.T).flatten()
        max_per_noun = np.maximum(max_per_noun, sims)

        # Similarity to negatives: (1, N_neg)
        neg_sims = img_emb @ neg_embeddings.T
        # Max neg sim for this image
        max_neg = float(np.max(neg_sims))
        
        # Contrastive: sim(im, noun) - max_neg
        contrastive = sims - max_neg
        max_contrastive_per_noun = np.maximum(max_contrastive_per_noun, contrastive)
        
        entry = {
            "index": int(idx + 1),
            "image": image_paths[idx].name,
            "mean_max_similarity": float(max_per_noun.mean()),
            contrastive_key: float(max_contrastive_per_noun.mean()),
        }

        results.append(entry)

    return results


def plot_mean_max_similarity_trajectory(results: Sequence[Dict[str, object]], outpath: Path) -> None:
    if not results:
        return
    steps = [int(row["index"]) for row in results]
    vals = [float(row["mean_max_similarity"]) for row in results]

    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    ax.plot(steps, vals, label="Noun Similarity", color="#1f77b4", linewidth=2)
    
    # Identify all contrastive keys present in the data
    all_keys = set()
    for row in results:
        for k in row.keys():
            if k.startswith("mean_max_contrastive"):
                all_keys.add(k)
    
    sorted_c_keys = sorted(list(all_keys))
    # Use tab10 colormap skipping the first blue (used for main metric)
    cmap = plt.get_cmap("tab10")
    
    for i, c_key in enumerate(sorted_c_keys):
        c_vals = [float(row.get(c_key, float('nan'))) for row in results]
        
        label = "Contrastive"
        suffix = c_key.replace('mean_max_contrastive', '')
        if suffix.startswith('_'):
            suffix = suffix[1:]
        if suffix:
            label += f" ({suffix})"
        
        # i+1 to skip blue
        color = cmap((i + 1) % 10)
        ax.plot(steps, c_vals, label=label, color=color, linewidth=2, linestyle="--")

    ax.set_title("Noun coverage metrics over archive growth")
    ax.set_xlabel("Archive insertion order")
    ax.set_ylabel("Metric Value")
    ax.legend()
    ax.grid(True, which="major", alpha=0.3)

    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def save_trajectory_json(results: Sequence[Dict[str, object]], outpath: Path) -> List[Dict[str, object]]:
    final_data = []
    existing_map = {}

    if outpath.exists():
        try:
            existing_data = json.loads(outpath.read_text(encoding="utf-8"))
            for item in existing_data:
                idx = item.get("index")
                if idx is not None:
                    existing_map[idx] = item
        except Exception as e:
            print(f"Could not read existing trajectory file {outpath}: {e}")

    # Process new results
    for row in results:
        idx = row["index"]
        item = existing_map.get(idx, {})
        
        # Update core fields
        item["index"] = int(row["index"])
        item["image"] = str(row["image"])
        item["mean_max_similarity"] = float(row["mean_max_similarity"])
        
        # Merge contrastive keys
        for k, v in row.items():
            if k.startswith("mean_max_contrastive"):
                item[k] = float(v)
        
        existing_map[idx] = item

    all_indices = sorted(existing_map.keys())
    for idx in all_indices:
        final_data.append(existing_map[idx])

    outpath.parent.mkdir(parents=True, exist_ok=True)
    outpath.write_text(json.dumps(final_data, indent=2), encoding="utf-8")
    
    return final_data



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
        # distance = 1.0 - float(max_per_noun[noun_idx])
        # captions.append(f"{nouns[noun_idx]} (dist {distance:.3f})")
        captions.append(nouns[noun_idx])

    if output_path.suffix.lower() == ".pdf":
        save_captioned_grid_pdf(images, captions, output_path, font_size=font_size)
    else:
        grid = create_captioned_grid(images, captions, thumb_size, margin, font_size)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        grid.save(output_path, format="PNG")


def _resolve_optional_path(value: Optional[Path], base: Path) -> Optional[Path]:
    if value is None:
        return None
    return _ensure_absolute(Path(value), base)


def process_noun_similarity(
    validated_cfg: NounSimilarityConfig,
    image_paths: List[Path],
    nouns_list: List[str],
    prompts_list: List[str],
    original_cwd: Path,
    output_dir: Path,
    cache_dir: Path,
    model=None,
    preprocess=None,
    tokenizer=None,
    noun_embeddings: Optional[np.ndarray] = None,
    neg_embeddings: Optional[np.ndarray] = None,
    device: Optional[torch.device] = None,
) -> None:
    if device is None:
        if validated_cfg.device:
            device = torch.device(validated_cfg.device)
        else:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    # Setup Embedding Cache
    # We use a shared directory for noun/negative embeddings to avoid recomputing them
    # across different runs/experiments.
    noun_embed_cache_dir = original_cwd / "noun_lists" / "embeddings"
    noun_embed_cache_dir.mkdir(parents=True, exist_ok=True)
    
    model_name_sanitized = validated_cfg.embedding_model.replace("/", "-")
    pretrained_sanitized = str(validated_cfg.pretrained).replace("/", "-")

    # 1. Check for cached Noun Embeddings
    noun_cached_exists = False
    if noun_embeddings is None:
        nounlist_stem = Path(validated_cfg.nounlist).stem
        noun_cache_path = noun_embed_cache_dir / f"noun_embeddings_{nounlist_stem}_{model_name_sanitized}_{pretrained_sanitized}.npy"
        if noun_cache_path.exists():
            noun_cached_exists = True

    # 2. Check for cached Negative Embeddings
    neg_anchors_list = []
    neg_cached_exists = False
    if neg_embeddings is None and validated_cfg.negative_anchors:
        anchors_name = validated_cfg.negative_anchors
        # Resolve anchors file for later
        anchors_path = original_cwd / "noun_lists" / f"{anchors_name}.txt"
        if anchors_path.exists():
            neg_anchors_list = load_nouns(anchors_path)
            neg_cache_path = noun_embed_cache_dir / f"negative_anchors_{anchors_name}_{model_name_sanitized}_{pretrained_sanitized}.npy"
            if neg_cache_path.exists():
                neg_cached_exists = True
        else:
            print(f"Warning: Negative anchors file {anchors_path} not found. Skipping negative anchors.")

    if (model is None) ^ (preprocess is None):
        raise ValueError("Provide both model and preprocess, or neither.")
    if (tokenizer is None) and (model is not None):
        raise ValueError("Provide tokenizer when providing a pre-built model.")

    if noun_embeddings is not None and len(nouns_list) == 0:
        raise ValueError("Provide nouns when providing noun_embeddings")

    # Check for cached image embeddings
    embeddings_cache_path = cache_dir / f"image_embeddings_cache_{model_name_sanitized}_{pretrained_sanitized}.npy"

    cached_image_embeddings = None
    existing_count = 0
    if embeddings_cache_path.exists():
        try:
            print(f"Loading cached embeddings from {embeddings_cache_path}...")
            cached_image_embeddings = np.load(embeddings_cache_path)
            existing_count = cached_image_embeddings.shape[0]
            print(f"Loaded {existing_count} cached embeddings.")
        except Exception as e:
            print(f"Failed to load cached embeddings: {e}")
            cached_image_embeddings = None
            existing_count = 0

    images_fully_cached = (existing_count >= len(image_paths))
    
    # Determine if we need the model
    # We need model if:
    # - Images are not fully cached
    # - OR Noun embeddings need to be computed (no cache, or passed None)
    # - OR Negative embeddings need to be computed (no cache, and we have anchors)
    # Note: load_or_compute_... needs model if cache is missing.
    
    need_model = False
    if not images_fully_cached:
        need_model = True
    elif (noun_embeddings is None) and (not noun_cached_exists):
        need_model = True
    elif (neg_embeddings is None) and neg_anchors_list and (not neg_cached_exists):
        need_model = True

    if need_model:
        if model is None:
            print(f"Loading OpenCLIP model {validated_cfg.embedding_model} ({validated_cfg.pretrained})...")
            model, preprocess, tokenizer = prepare_openclip_components(validated_cfg, device)
        else:
            model.to(device)
            model.eval()

    # Load/Compute Noun Embeddings
    if noun_embeddings is None:
        noun_embeddings = load_or_compute_noun_embeddings(
            validated_cfg,
            original_cwd,
            device,
            prompts_list,
            model,
            tokenizer,
        )

    # Load/Compute Negative Embeddings
    if neg_embeddings is None and neg_anchors_list:
        neg_embeddings, _ = load_or_compute_negative_embeddings(
            validated_cfg,
            original_cwd,
            device,
            model,
            tokenizer,
        )

    # Process Images
    if images_fully_cached:
        print(f"Using {len(image_paths)} cached image embeddings.")
        image_embeddings = cached_image_embeddings[:len(image_paths)]
    else:
        paths_to_compute = image_paths[existing_count:]
        print(f"Embedding {len(paths_to_compute)} new images (found {existing_count} cached)...")
        
        _, new_embeddings = embed_images(
            model,
            preprocess,
            paths_to_compute,
            device,
            batch_size=validated_cfg.batch_size,
        )
        
        if existing_count > 0:
            image_embeddings = np.vstack([cached_image_embeddings, new_embeddings])
        else:
            image_embeddings = new_embeddings

        print(f"Saving {image_embeddings.shape[0]} embeddings to {embeddings_cache_path}")
        np.save(embeddings_cache_path, image_embeddings)
    
    # Validation
    if noun_embeddings is None:
         # Should verify handled above, but technically if prompts_list was empty or something...
         # But prompts_list comes from main.
         # If prompts_list was not empty, noun_embeddings should be set.
         pass
    else:
        # Check type
        if isinstance(noun_embeddings, list):
             noun_embeddings = np.asarray(noun_embeddings)

    max_per_noun, mean_similarity, best_image_indices, max_contrastive, mean_contrastive = compute_max_similarities(
        image_embeddings,
        noun_embeddings,
        neg_embeddings=neg_embeddings,
    )

    contrastive_key = None
    per_noun_key = None
    if validated_cfg.negative_anchors:
        anchors_str = f"{validated_cfg.negative_anchors.replace(" ", "-")}"
        contrastive_key = f"mean_max_contrastive_{anchors_str}"
        per_noun_key = f"max_contrastive_per_noun_{anchors_str}"

    trajectory = compute_trajectory_metrics(
        image_embeddings,
        noun_embeddings,
        image_paths,
        neg_embeddings=neg_embeddings,
        contrastive_key=contrastive_key
    )

    metrics = {
        "experiment_dir": str(output_dir),
        "num_images": len(image_paths),
        "num_nouns": len(nouns_list),
        "model": validated_cfg.embedding_model,
        "pretrained": validated_cfg.pretrained,
        "label_template": validated_cfg.label_template,
        "negative_anchors": validated_cfg.negative_anchors,
        "mean_max_similarity": mean_similarity,
        "max_similarity_per_noun": {noun: float(score) for noun, score in zip(nouns_list, max_per_noun)},
    }
    
    if contrastive_key:
        metrics[contrastive_key] = mean_contrastive
        metrics[per_noun_key] = {noun: float(score) for noun, score in zip(nouns_list, max_contrastive)}

    output_path = _resolve_optional_path(validated_cfg.output_json, original_cwd)
    if output_path is None:
        output_path = output_dir / "noun_similarity_metrics.json"
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        try:
            existing_metrics = json.loads(output_path.read_text(encoding="utf-8"))
            existing_metrics.update(metrics)
            metrics = existing_metrics
        except Exception as e:
            print(f"Warning: Could not merge with existing metrics file {output_path}: {e}")

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved noun similarity metrics to {output_path}")
    print(f"Mean of per-noun max cosine similarity: {mean_similarity:.4f}")

    nounlist_name = Path(validated_cfg.nounlist).stem
    model_name_sanitized = validated_cfg.embedding_model.replace("/", "-")
    trajectory_json = _resolve_optional_path(validated_cfg.output_trajectory_json, original_cwd)
    if trajectory_json is None:
        trajectory_json = output_dir / f"noun_similarity_over_time_{nounlist_name}_{model_name_sanitized}.json"
    trajectory_plot = _resolve_optional_path(validated_cfg.output_trajectory_plot, original_cwd)
    if trajectory_plot is None:
        trajectory_plot = output_dir / f"noun_similarity_over_time_{nounlist_name}_{model_name_sanitized}.png"

    merged_trajectory = save_trajectory_json(trajectory, trajectory_json)
    plot_mean_max_similarity_trajectory(merged_trajectory, trajectory_plot)
    print(f"Saved noun similarity trajectory JSON to {trajectory_json}")
    print(f"Saved noun similarity trajectory plot to {trajectory_plot}")

    if validated_cfg.render_grid:
        # Override max_per_noun and best_image_indices using linear assignment for better visualization
        # We want to assign each image to at most one noun (and each noun to at most one image)
        print(f"Solving linear assignment for grid rendering ({image_embeddings.shape[0]} images, {noun_embeddings.shape[0]} nouns)...")
        sims = image_embeddings @ noun_embeddings.T
        # linear_sum_assignment solves min cost, so we pass negative similarity (maximize=True works in newer scipy)
        row_ind, col_ind = linear_sum_assignment(sims, maximize=True)
        
        # Re-initialize for grid usage: unmatched nouns get -1.0 so they appear at end
        max_per_noun_grid = np.full(len(nouns_list), -1.0, dtype=np.float32)
        best_image_indices_grid = np.zeros(len(nouns_list), dtype=np.int64)
        
        max_per_noun_grid[col_ind] = sims[row_ind, col_ind]
        best_image_indices_grid[col_ind] = row_ind
        
        grid_output = _resolve_optional_path(validated_cfg.output_grid, original_cwd)
        if grid_output is None:
            suffix = f"_top{validated_cfg.grid_top_k}" if validated_cfg.grid_top_k else ""
            grid_output = output_dir / f"noun_similarity_grid_{nounlist_name}_{model_name_sanitized}{suffix}.pdf"
        render_noun_similarity_grid(
            nouns_list,
            max_per_noun_grid,
            best_image_indices_grid,
            image_paths,
            grid_output,
            validated_cfg.grid_thumb_size,
            validated_cfg.grid_margin,
            validated_cfg.grid_font_size,
            top_k=validated_cfg.grid_top_k,
        )
        print(f"Saved noun similarity grid to {grid_output}")


@hydra.main(version_base="1.3", config_path=None, config_name="noun_similarity_base")
def main(
    cfg: NounSimilarityConfig,
    model=None,
    preprocess=None,
    tokenizer=None,
    nouns: Optional[Sequence[str]] = None,
    prompts: Optional[Sequence[str]] = None,
    noun_embeddings: Optional[np.ndarray] = None,
    neg_embeddings: Optional[np.ndarray] = None,
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

    process_noun_similarity(
        validated_cfg,
        image_paths,
        nouns_list,
        prompts_list,
        original_cwd,
        exp_dir, # output_dir
        exp_dir, # cache_dir
        model=model,
        preprocess=preprocess,
        tokenizer=tokenizer,
        noun_embeddings=noun_embeddings,
        neg_embeddings=neg_embeddings,
    )


if __name__ == "__main__":
    main()
