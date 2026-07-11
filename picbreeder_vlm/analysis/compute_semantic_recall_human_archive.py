#!/usr/bin/env python3
"""
Plot noun similarity score over time for human archives in fer/src/archive_res-{size}.
"""
from dataclasses import dataclass, field
from pathlib import Path

import hydra
from hydra.conf import HelpConf, HydraConf
from hydra.core.config_store import ConfigStore
from hydra.utils import get_original_cwd

import matplotlib
import torch
import numpy as np

# Set backend before importing other plotting modules
matplotlib.use("Agg")

from typing import Optional, Sequence
from picbreeder_vlm.analysis.compute_semantic_recall import (
    SemanticRecallConfig,
    process_semantic_recall,
    format_prompts,
    load_nouns,
)
from picbreeder_vlm.core.utils import _ensure_absolute, load_human_archive_images, resolve_nounlist
from picbreeder_vlm.core.config import ensure_valid_config
from picbreeder_vlm.core.constants import HUMAN_BASELINE_DIR
from picbreeder_vlm._paths import FER_ROOT


@dataclass
class HumanSemanticRecallConfig(SemanticRecallConfig):
    render_size: int = 128
    hydra: HydraConf = field(
        default_factory=lambda: HydraConf(
            help=HelpConf(
                app_name="plot_human_archive_semantic_recall",
                header=(
                    "Hydra entry point for human archive noun metrics.\n"
                    "Plots noun similarity score over time for human archives.\n"
                ),
                footer="Override with +option=value (e.g. render_size=512).",
            )
        )
    )

ConfigStore.instance().store(name="human_semantic_recall_base", node=HumanSemanticRecallConfig)

@hydra.main(version_base="1.3", config_path=None, config_name="human_semantic_recall_base")
def main(
    cfg: HumanSemanticRecallConfig,
) -> None:
    original_cwd = Path(get_original_cwd())
    validated_cfg = ensure_valid_config(cfg, original_cwd=original_cwd)
    
    # Construct archive dir based on render_size
    root_dir = original_cwd
    archive_dir = FER_ROOT / "src" / f"archive_res-{validated_cfg.render_size}" / "images"
    noun_file = resolve_nounlist(validated_cfg.nounlist, original_cwd)
    
    print(f"Looking for images in {archive_dir}")
    try:
        image_paths = load_human_archive_images(archive_dir)
    except FileNotFoundError:
        print(f"Directory not found: {archive_dir}")
        return

    print(f"Found {len(image_paths)} images.")
    
    if validated_cfg.archive_limit is not None:
        image_paths = image_paths[: validated_cfg.archive_limit]
        print(f"Limited to {len(image_paths)} images.")
    
    if not image_paths:
        print("No images found. Exiting.")
        return

    # Load nouns and format prompts
    nouns_list = load_nouns(noun_file)
    prompts_list = format_prompts(nouns_list, validated_cfg.label_template)

    # Setup output paths
    nounlist_name = noun_file.stem
    model_name = validated_cfg.embedding_model.replace("/", "-")
    
    output_dir = root_dir / HUMAN_BASELINE_DIR
    output_dir.mkdir(exist_ok=True, parents=True)
    
    filename_base = f"noun_similarity_res{validated_cfg.render_size}_{model_name}_{nounlist_name}"
    
    # Configure output paths
    validated_cfg.output_trajectory_json = output_dir / f"{filename_base}.json"
    validated_cfg.output_trajectory_plot = output_dir / f"{filename_base}.png"
    validated_cfg.output_json = output_dir / f"{filename_base}_metrics.json"

    # Cache dir - use the parent directory of the images (archive_res-{size})
    cache_dir = archive_dir.parent

    print(f"Saving results to {output_dir}")
    print(f"Using cache dir: {cache_dir}")

    process_semantic_recall(
        validated_cfg,
        image_paths,
        nouns_list,
        prompts_list,
        original_cwd,
        output_dir=output_dir,
        cache_dir=cache_dir,
    )

if __name__ == "__main__":
    main()
