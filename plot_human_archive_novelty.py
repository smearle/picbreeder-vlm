#!/usr/bin/env python3
"""
Plot mean pairwise distance (novelty) over time for human archives in fer/src/archive_res-{size}.
"""
from dataclasses import dataclass, field
from pathlib import Path
import sys
from typing import List

import hydra
from hydra.conf import HelpConf, HydraConf
from hydra.core.config_store import ConfigStore
from hydra.utils import get_original_cwd

import matplotlib
import matplotlib.pyplot as plt
import torch
import numpy as np

# Set backend before importing other plotting modules
matplotlib.use("Agg")

try:
    from embed_and_visualize import embed_images
except ImportError:
    # Fallback if embed_and_visualize is not easily importable or if we want to use the one from compute_noun_similarity
    from compute_noun_similarity import embed_images

from plot_novelty_over_time import (
    compute_mpd_trajectory,
    plot_mpd_trajectory,
    save_trajectory_json,
    prepare_openclip_components,
    PairwiseDistanceConfig
)
from utils import load_human_archive_images
from config import ensure_valid_config

@dataclass
class HumanPairwiseDistanceConfig(PairwiseDistanceConfig):
    render_size: int = 128
    hydra: HydraConf = field(
        default_factory=lambda: HydraConf(
            help=HelpConf(
                app_name="plot_human_archive_novelty",
                header=(
                    "Hydra entry point for human archive novelty metrics.\n"
                    "Plots mean pairwise distance over time for human archives.\n"
                ),
                footer="Override with +option=value (e.g. render_size=512).",
            )
        )
    )

ConfigStore.instance().store(name="human_pairwise_distance_base", node=HumanPairwiseDistanceConfig)

@hydra.main(version_base="1.3", config_path=None, config_name="human_pairwise_distance_base")
def main(
    cfg: HumanPairwiseDistanceConfig,
) -> None:
    original_cwd = Path(get_original_cwd())
    validated_cfg = ensure_valid_config(cfg, original_cwd=original_cwd)
    
    # Construct archive dir based on render_size
    root_dir = original_cwd
    archive_dir = root_dir / "fer/src" / f"archive_res-{validated_cfg.render_size}"
    
    print(f"Looking for images in {archive_dir}")
    try:
        image_paths = load_human_archive_images(archive_dir)
    except FileNotFoundError:
        print(f"Directory not found: {archive_dir}")
        return

    print(f"Found {len(image_paths)} images.")
    
    if not image_paths:
        print("No images found. Exiting.")
        return

    # Setup device
    if validated_cfg.device:
        device = torch.device(validated_cfg.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    print(f"Loading OpenCLIP model {validated_cfg.embedding_model} ({validated_cfg.pretrained})...")
    
    # prepare_openclip_components in plot_novelty_over_time returns (model, preprocess)
    model, preprocess = prepare_openclip_components(validated_cfg, device)

    print(f"Embedding {len(image_paths)} images...")
    
    # We use embed_images from compute_noun_similarity or embed_and_visualize.
    # We should stick to one. In previous step I used the one from compute_noun_similarity.
    # Let's ensure consistency.
    from compute_noun_similarity import embed_images as embed_images_simple
    
    embeddings = embed_images_simple(
        model,
        preprocess,
        image_paths,
        device,
        batch_size=validated_cfg.batch_size,
    )
    
    # compute_mpd_trajectory expects (embeddings, image_paths)
    results = compute_mpd_trajectory(embeddings, image_paths)
    
    # Define output paths with resolution and model name
    model_name = validated_cfg.embedding_model.replace("/", "-")
    filename_base = f"novelty_res{validated_cfg.render_size}_{model_name}"
    
    output_dir = root_dir / "human_baseline"
    output_dir.mkdir(exist_ok=True, parents=True)

    output_base = output_dir / filename_base
    plot_path = output_base.with_suffix(".png")
    data_path = output_base.with_suffix(".json")
    
    plot_mpd_trajectory(results, plot_path)
    save_trajectory_json(results, data_path)
    
    final_mpd = results[-1]["mean_pairwise_distance"] if results else None
    print(f"Saved plot to {plot_path}")
    print(f"Saved trajectory data to {data_path}")
    print(f"Final mean pairwise distance: {final_mpd}")

if __name__ == "__main__":
    main()
