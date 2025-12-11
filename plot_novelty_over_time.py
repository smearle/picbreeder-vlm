#!/usr/bin/env python3
"""Plot how the archive's mean pairwise distance shifts as each image is added, plus nearest-neighbor distance stats."""

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import hydra
from hydra.conf import HelpConf, HydraConf
from hydra.core.config_store import ConfigStore
from hydra.utils import get_original_cwd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open_clip
import torch

from config import PicbreederConfig, ensure_valid_config
from embed_and_visualize import embed_images


@dataclass
class PairwiseDistanceConfig(PicbreederConfig):
    embedding_model: str = "ViT-B-32"
    pretrained: str = "openai"
    batch_size: int = 64
    archive_limit: Optional[int] = None
    recompute_embeddings: bool = False
    device: Optional[str] = None
    hydra: HydraConf = field(
        default_factory=lambda: HydraConf(
            help=HelpConf(
                app_name="plot_mean_pairwise_distance_over_time",
                header=(
                    "Plots how mean pairwise embedding distance evolves as images enter the archive.\n"
                    "Provide experiment_dir or use goal/scheme/seed to resolve one."
                ),
                footer="Common overrides: experiment_dir=/path/to/run recompute_embeddings=true",
            )
        )
    )


ConfigStore.instance().store(name="pairwise_distance_base", node=PairwiseDistanceConfig)


def _numeric_suffix(path: Path) -> int:
    stem = path.stem
    parts = stem.split("_")
    try:
        return int(parts[-1])
    except ValueError:
        return 0


def infer_archive_order(exp_dir: Path) -> List[Path]:
    archive_dir = exp_dir / "archive"
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

    remaining = sorted(images_dir.glob("img_*.png"), key=_numeric_suffix)
    for path in remaining:
        if path.name in seen:
            continue
        ordered.append(path)
        seen.add(path.name)

    if not ordered:
        raise SystemExit(f"No archive images found under {images_dir}")

    return ordered


def _prepare_model(cfg: PairwiseDistanceConfig, device: torch.device):
    model, _, preprocess = open_clip.create_model_and_transforms(
        cfg.embedding_model,
        pretrained=cfg.pretrained,
    )
    model.to(device)
    model.eval()
    return model, preprocess


def load_embeddings_in_order(
    image_paths: Sequence[Path],
    exp_dir: Path,
    cfg: PairwiseDistanceConfig,
    device: torch.device,
) -> np.ndarray:
    cache_path = exp_dir / "embeddings_openclip.npz"
    embeddings_map: Dict[str, np.ndarray] = {}
    missing: List[Path] = []

    if cache_path.exists() and not cfg.recompute_embeddings:
        data = np.load(cache_path)
        filenames = [str(name) for name in data["filenames"]]
        cached_embeddings = np.asarray(data["embeddings"])
        embeddings_map.update({name: cached_embeddings[i] for i, name in enumerate(filenames)})
        missing = [p for p in image_paths if p.name not in embeddings_map]
        if missing:
            print(f"{len(missing)} images missing from cache; embedding the remainder.")
    else:
        missing = list(image_paths)

    if missing:
        model, preprocess = _prepare_model(cfg, device)
        missing_names, missing_embeddings = embed_images(
            model,
            preprocess,
            missing,
            device,
            batch_size=cfg.batch_size,
        )
        for name, emb in zip(missing_names, missing_embeddings):
            embeddings_map[name] = emb

    ordered_embeddings = []
    for path in image_paths:
        emb = embeddings_map.get(path.name)
        if emb is None:
            raise ValueError(f"No embedding available for {path.name}; try recompute_embeddings=true")
        ordered_embeddings.append(emb)

    return np.vstack(ordered_embeddings)


def compute_mpd_trajectory(embeddings: np.ndarray, image_paths: Sequence[Path]):
    results = []
    cumulative_sum = 0.0
    cumulative_min_sum = 0.0
    min_count = 0

    for idx, emb in enumerate(embeddings):
        n = idx + 1
        incremental_sum = None
        mpd = None
        delta = None
        min_dist = None
        mean_min_distance = None
        sum_min_distance = None

        if idx > 0:
            prev = embeddings[:idx]
            dists = np.linalg.norm(prev - emb, axis=1)
            incremental_sum = float(dists.sum())
            min_dist = float(dists.min())
            cumulative_sum += incremental_sum
            total_pairs = n * (n - 1) / 2
            mpd = cumulative_sum / total_pairs
            prev_mpd = results[-1]["mean_pairwise_distance"]
            if prev_mpd is not None and mpd is not None:
                delta = mpd - prev_mpd
            cumulative_min_sum += min_dist
            min_count += 1
            mean_min_distance = cumulative_min_sum / min_count
            sum_min_distance = cumulative_min_sum
        else:
            cumulative_sum = 0.0

        results.append(
            {
                "index": n,
                "image": image_paths[idx].name,
                "mean_pairwise_distance": mpd,
                "delta_from_previous": delta,
                "distance_added": incremental_sum,
                "nearest_neighbor_distance": min_dist,
                "mean_min_distance": mean_min_distance,
                "sum_min_distance": sum_min_distance,
            }
        )

    return results


def plot_mpd_trajectory(results, outpath: Path):
    steps = [item["index"] for item in results]
    mpd_vals = [item["mean_pairwise_distance"] for item in results]
    min_dists = [item["nearest_neighbor_distance"] for item in results]
    mean_min_vals = [item["mean_min_distance"] for item in results]
    sum_min_vals = [item["sum_min_distance"] for item in results]

    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

    axes[0].plot(steps, mpd_vals, color="#1f77b4", linewidth=2)
    axes[0].set_ylabel("Mean pairwise distance")
    axes[0].set_title("Embedding diversity as archive grows")
    axes[0].grid(True, which="major", alpha=0.3)

    nn_steps = [s for s, d in zip(steps, min_dists) if d is not None]
    nn_vals = [d for d in min_dists if d is not None]
    mean_steps = [s for s, d in zip(steps, mean_min_vals) if d is not None]
    mean_vals = [d for d in mean_min_vals if d is not None]
    sum_steps = [s for s, d in zip(steps, sum_min_vals) if d is not None]
    sum_vals = [d for d in sum_min_vals if d is not None]

    axes[1].plot(nn_steps, nn_vals, color="#17becf", linewidth=1.3, alpha=0.7, label="Nearest neighbor distance")
    axes[1].plot(mean_steps, mean_vals, color="#9467bd", linewidth=2, label="Running mean min distance")
    axes[1].set_xlabel("Archive insertion order")
    axes[1].set_ylabel("Min distance scales")
    axes[1].grid(True, which="major", alpha=0.3)

    sum_axis = axes[1].twinx()
    sum_axis.plot(sum_steps, sum_vals, color="#ff7f0e", linewidth=1.5, linestyle="--", label="Cumulative min distance")
    sum_axis.set_ylabel("Cumulative min distance")

    handles, labels = axes[1].get_legend_handles_labels()
    sum_handles, sum_labels = sum_axis.get_legend_handles_labels()
    axes[1].legend(handles + sum_handles, labels + sum_labels, loc="upper left")

    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def save_trajectory_json(results, outpath: Path):
    serializable = []
    for row in results:
        serializable.append(
            {
                "index": int(row["index"]),
                "image": row["image"],
                "mean_pairwise_distance": row["mean_pairwise_distance"],
                "delta_from_previous": row["delta_from_previous"],
                "distance_added": row["distance_added"],
                "nearest_neighbor_distance": row["nearest_neighbor_distance"],
                "mean_min_distance": row["mean_min_distance"],
                "sum_min_distance": row["sum_min_distance"],
            }
        )
    outpath.parent.mkdir(parents=True, exist_ok=True)
    outpath.write_text(json.dumps(serializable, indent=2), encoding="utf-8")


@hydra.main(version_base="1.3", config_path=None, config_name="pairwise_distance_base")
def main(cfg: PairwiseDistanceConfig) -> None:
    original_cwd = Path(get_original_cwd())
    validated_cfg = ensure_valid_config(cfg, original_cwd=original_cwd)

    exp_dir = Path(validated_cfg.experiment_dir).resolve()
    archive_dir = exp_dir / "archive"
    if not archive_dir.exists():
        raise FileNotFoundError(f"Archive directory not found: {archive_dir}")

    image_paths = infer_archive_order(exp_dir)
    if validated_cfg.archive_limit is not None:
        image_paths = image_paths[: validated_cfg.archive_limit]

    if validated_cfg.device:
        device = torch.device(validated_cfg.device)
    else:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    embeddings = load_embeddings_in_order(image_paths, exp_dir, validated_cfg, device)
    print(f"Loaded embeddings for {len(image_paths)} images.")

    results = compute_mpd_trajectory(embeddings, image_paths)

    plot_path = exp_dir / "embedding_mean_pairwise_distance_over_time.png"
    data_path = exp_dir / "embedding_mean_pairwise_distance_over_time.json"
    plot_mpd_trajectory(results, plot_path)
    save_trajectory_json(results, data_path)

    final_mpd = results[-1]["mean_pairwise_distance"] if results else None
    print(f"Saved plot to {plot_path}")
    print(f"Saved trajectory data to {data_path}")
    print(f"Final mean pairwise distance: {final_mpd}")


if __name__ == "__main__":
    main()
