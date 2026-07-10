#!/usr/bin/env python3
"""Compute Visual Coverage for an archive of images.

Embeds every archive image (SigLIP2 by default) and, after each insertion,
runs greedy k-center over the image embeddings. The headline metric is the
k-covering RADIUS (`compute_k_covering`): the distance from the worst-covered
image to its nearest of k representatives. Higher = the archive is more spread
out, since k representatives no longer cover it tightly. The paper reports k=100.

Mean pairwise distance and nearest-neighbour stats are also recorded, but they
are secondary diversity numbers, NOT Visual Coverage.

The on-disk artifact keeps its former name -- `embedding_mean_pairwise_distance
_over_time_<model>.json`, after the secondary stat -- because the paper's runs
are on disk and on the Hub under it. Coverage lives in its `k_covering_radii`
key. See picbreeder_vlm/analysis/__init__.py for the full name mapping.
"""

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
from sklearn.metrics import pairwise_distances

from picbreeder_vlm.core.config import PicbreederConfig, ensure_valid_config
from picbreeder_vlm.vlm.model_loader import prepare_model, embed_images


@dataclass
class VisualCoverageConfig(PicbreederConfig):
    embedding_model: str = "SigLIP2-B-alignet"
    pretrained: str = "laion2b_s32b_b79k"
    batch_size: int = 64
    archive_limit: Optional[int] = None
    k_covering_ks: str = "1,5,10,20,50,100"
    recompute_embeddings: bool = False
    device: Optional[str] = None
    hydra: HydraConf = field(
        default_factory=lambda: HydraConf(
            help=HelpConf(
                app_name="compute_visual_coverage",
                header=(
                    "Plots how mean pairwise embedding distance evolves as images enter the archive.\n"
                    "Provide experiment_dir or use goal/scheme/seed to resolve one."
                ),
                footer="Common overrides: experiment_dir=/path/to/run recompute_embeddings=true",
            )
        )
    )


ConfigStore.instance().store(name="visual_coverage_base", node=VisualCoverageConfig)


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


def _prepare_model(cfg: VisualCoverageConfig, device: torch.device):
    model, preprocess, _ = prepare_model(cfg, device)
    return model, preprocess


def prepare_openclip_components(
    cfg: VisualCoverageConfig,
    device: torch.device,
):
    """Create the OpenCLIP model + preprocess transform for this config."""

    return _prepare_model(cfg, device)


def load_embeddings_in_order(
    image_paths: Sequence[Path],
    exp_dir: Path,
    cfg: VisualCoverageConfig,
    device: torch.device,
    model=None,
    preprocess=None,
) -> np.ndarray:
    model_name_sanitized = cfg.embedding_model.replace("/", "-")
    cache_path = exp_dir / f"embeddings_openclip_{model_name_sanitized}.npz"
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
        if (model is None) ^ (preprocess is None):
            raise ValueError("Provide both model and preprocess, or neither.")
        if model is None:
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

        print(f"Updating cache at {cache_path} with {len(missing)} new embeddings.")
        cache_filenames = list(embeddings_map.keys())
        cache_embeddings = np.stack([embeddings_map[name] for name in cache_filenames])
        np.savez(cache_path, filenames=cache_filenames, embeddings=cache_embeddings)

    ordered_embeddings = []
    for path in image_paths:
        emb = embeddings_map.get(path.name)
        if emb is None:
            raise ValueError(f"No embedding available for {path.name}; try recompute_embeddings=true")
        ordered_embeddings.append(emb)

    return np.vstack(ordered_embeddings)


def compute_k_covering(vectors: np.ndarray, k_values: List[int]) -> Dict[str, float]:
    """Calculate k-covering radii for a set of vectors using greedy k-center."""
    N = vectors.shape[0]
    if N == 0:
        return {}
    
    max_k = max(k_values)
    eff_max_k = min(max_k, N)
    
    radii = {}
    centers = [0]
    # Distances from the first center to all points
    min_dists = pairwise_distances(vectors[centers], vectors).flatten()
    radii[1] = float(np.max(min_dists))
    
    for m in range(2, eff_max_k + 2):
        if m > N: 
            break 
        new_center = np.argmax(min_dists)
        centers.append(new_center)
        
        # Update min distances with the new center
        new_dists = pairwise_distances(vectors[[new_center]], vectors).flatten()
        min_dists = np.minimum(min_dists, new_dists)
        
        radius = float(np.max(min_dists))
        radii[m] = radius
        
    return {str(k): radii.get(k, 0.0) for k in k_values if k <= eff_max_k}


def compute_visual_coverage_trajectory(
        embeddings: np.ndarray, image_paths: Sequence[Path], k_values: List[int],
        data_path: Path, step_size: int) -> List[Dict]:
    results = []
    cumulative_sum = 0.0
    cumulative_min_sum = 0.0
    min_count = 0

    existing_data = {}
    if data_path.is_file():
        try:
            loaded_data = json.loads(data_path.read_text(encoding="utf-8"))
            if isinstance(loaded_data, dict):
                existing_data = loaded_data
        except Exception:
            pass

    for idx in range(0, len(embeddings), step_size):
        n = idx + 1

        if str(n) in existing_data:
            entry = existing_data[str(n)]
            results.append(entry)

            if idx > 0:
                mpd = entry.get("mean_pairwise_distance")
                if mpd is not None:
                    total_pairs = n * (n - 1) / 2
                    cumulative_sum = mpd * total_pairs

                sum_min = entry.get("sum_min_distance")
                if sum_min is not None:
                    cumulative_min_sum = sum_min

                min_count += 1
            else:
                cumulative_sum = 0.0
            continue

        emb = embeddings[idx]
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
        
        # Compute k-covering
        # Only compute if we have enough points for at least the smallest k
        k_covering = compute_k_covering(embeddings[:n], k_values)

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
                "k_covering_radii": k_covering,
            }
        )

    save_trajectory_json(results, data_path)

    return results


def plot_mpd_trajectory(results, outpath: Path):
    n_init_skip = 2
    steps = [item["index"] for item in results][n_init_skip:]
    mpd_vals = [item["mean_pairwise_distance"] for item in results][n_init_skip:]
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))

    ax.plot(steps, mpd_vals, color="#1f77b4", linewidth=2)
    ax.set_xlabel("Archive insertion order")
    ax.set_ylabel("Mean pairwise distance")
    ax.set_title("Embedding diversity as archive grows")
    ax.grid(True, which="major", alpha=0.3)

    # The nearest-neighbor / min-distance subplot was not useful for our analysis,
    # but we keep the code here (commented) in case we want it back later.
    #
    # min_dists = [item["nearest_neighbor_distance"] for item in results][n_init_skip:]
    # mean_min_vals = [item["mean_min_distance"] for item in results][n_init_skip:]
    # sum_min_vals = [item["sum_min_distance"] for item in results][n_init_skip:]
    #
    # fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    # axes[0].plot(steps, mpd_vals, color="#1f77b4", linewidth=2)
    # axes[0].set_ylabel("Mean pairwise distance")
    # axes[0].set_title("Embedding diversity as archive grows")
    # axes[0].grid(True, which="major", alpha=0.3)
    #
    # nn_steps = [s for s, d in zip(steps, min_dists) if d is not None]
    # nn_vals = [d for d in min_dists if d is not None]
    # mean_steps = [s for s, d in zip(steps, mean_min_vals) if d is not None]
    # mean_vals = [d for d in mean_min_vals if d is not None]
    # sum_steps = [s for s, d in zip(steps, sum_min_vals) if d is not None]
    # sum_vals = [d for d in sum_min_vals if d is not None]
    #
    # axes[1].plot(nn_steps, nn_vals, color="#17becf", linewidth=1.3, alpha=0.7, label="Nearest neighbor distance")
    # axes[1].plot(mean_steps, mean_vals, color="#9467bd", linewidth=2, label="Running mean min distance")
    # axes[1].set_xlabel("Archive insertion order")
    # axes[1].set_ylabel("Min distance scales")
    # axes[1].grid(True, which="major", alpha=0.3)
    #
    # sum_axis = axes[1].twinx()
    # sum_axis.plot(sum_steps, sum_vals, color="#ff7f0e", linewidth=1.5, linestyle="--", label="Cumulative min distance")
    # sum_axis.set_ylabel("Cumulative min distance")
    #
    # handles, labels = axes[1].get_legend_handles_labels()
    # sum_handles, sum_labels = sum_axis.get_legend_handles_labels()
    # axes[1].legend(handles + sum_handles, labels + sum_labels, loc="upper left")

    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def save_trajectory_json(results, outpath: Path):
    serializable = {}
    for row in results:
        serializable[str(row["index"])] = {
            "index": int(row["index"]),
            "image": row["image"],
            "mean_pairwise_distance": row["mean_pairwise_distance"],
            "delta_from_previous": row["delta_from_previous"],
            "distance_added": row["distance_added"],
            "nearest_neighbor_distance": row["nearest_neighbor_distance"],
            "mean_min_distance": row["mean_min_distance"],
            "sum_min_distance": row["sum_min_distance"],
            "k_covering_radii": row.get("k_covering_radii", {}),
        }
    outpath.parent.mkdir(parents=True, exist_ok=True)
    outpath.write_text(json.dumps(serializable, indent=2), encoding="utf-8")


@hydra.main(version_base="1.3", config_path=None, config_name="visual_coverage_base")
def main(
    cfg: VisualCoverageConfig,
    model=None,
    preprocess=None,
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

    if (model is None) ^ (preprocess is None):
        raise ValueError("Provide both model and preprocess, or neither.")

    if model is None:
        model, preprocess = prepare_openclip_components(validated_cfg, device)
    else:
        # Ensure the injected model is ready for inference on the chosen device.
        model.to(device)
        model.eval()

    embeddings = load_embeddings_in_order(
        image_paths,
        exp_dir,
        validated_cfg,
        device,
        model=model,
        preprocess=preprocess,
    )
    print(f"Loaded embeddings for {len(image_paths)} images.")

    model_name_sanitized = validated_cfg.embedding_model.replace("/", "-")
    plot_path = exp_dir / f"embedding_mean_pairwise_distance_over_time_{model_name_sanitized}.png"
    data_path = exp_dir / f"embedding_mean_pairwise_distance_over_time_{model_name_sanitized}.json"

    k_values = [int(k.strip()) for k in validated_cfg.k_covering_ks.split(",") if k.strip()]
    results = compute_visual_coverage_trajectory(embeddings, image_paths, k_values, data_path=data_path, step_size=50)

    plot_mpd_trajectory(results, plot_path)

    final_mpd = results[-1]["mean_pairwise_distance"] if results else None
    print(f"Saved plot to {plot_path}")
    print(f"Saved trajectory data to {data_path}")
    print(f"Final mean pairwise distance: {final_mpd}")


if __name__ == "__main__":
    main()
