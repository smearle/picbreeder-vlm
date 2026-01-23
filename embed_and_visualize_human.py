#!/usr/bin/env python3
"""Embed and visualize human archive images using OpenCLIP + UMAP/TSNE."""

from dataclasses import dataclass, field
import json
from pathlib import Path
import sys

import hydra
from hydra.conf import HelpConf, HydraConf
from hydra.core.config_store import ConfigStore
from hydra.utils import get_original_cwd

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")

from config import ensure_valid_config
from utils import load_human_archive_images
from embed_and_visualize import (
    EmbedVisualizeConfig,
    _greedy_k_center,
    prepare_openclip_components,
    embed_images,
    reduce_embeddings,
    plot_coords,
    render_rectangular_grid,
    render_simple_grid,
    compute_embedding_metrics,
    _validate_embed_options,
)

@dataclass
class HumanEmbedVisualizeConfig(EmbedVisualizeConfig):
    render_size: int = 128
    hydra: HydraConf = field(
        default_factory=lambda: HydraConf(
            help=HelpConf(
                app_name="embed_and_visualize_human",
                header=(
                    "Embed and visualize human archive images.\n"
                ),
                footer="Override with +option=value (e.g. render_size=512 method=tsne).",
            )
        )
    )

ConfigStore.instance().store(name="human_embed_visualize_base", node=HumanEmbedVisualizeConfig)

@hydra.main(version_base="1.3", config_path=None, config_name="human_embed_visualize_base")
def main(cfg: HumanEmbedVisualizeConfig) -> None:
    original_cwd = Path(get_original_cwd())
    validated_cfg = ensure_valid_config(cfg, original_cwd=original_cwd)
    _validate_embed_options(validated_cfg)

    # Input directory
    root_dir = original_cwd
    archive_dir = root_dir / "fer/src" / f"archive_res-{validated_cfg.render_size}"
    
    if not archive_dir.exists():
        print(f"Archive directory not found: {archive_dir}")
        return

    print(f"Loading images from {archive_dir}...")
    image_paths = load_human_archive_images(archive_dir)
    
    if validated_cfg.archive_limit is not None:
        image_paths = image_paths[: validated_cfg.archive_limit]
    
    if not image_paths:
        print("No images found.")
        return

    # Output directory
    output_dir = root_dir / "human_baseline"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # We will use output_dir as the "experiment_dir" equivalent for saving files
    # We use a subdirectory for the resolution
    run_dir = output_dir / f"res-{validated_cfg.render_size}"
    if validated_cfg.archive_limit is not None:
        run_dir = run_dir / f"limit-{validated_cfg.archive_limit}"
    run_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"Output directory: {run_dir}")

    # Device
    if validated_cfg.device:
        device = torch.device(validated_cfg.device)
    else:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    # Model
    print(f"Loading OpenCLIP model {validated_cfg.embedding_model} ({validated_cfg.pretrained})...")
    model, preprocess = prepare_openclip_components(validated_cfg, device)

    # Embed
    print(f"Embedding {len(image_paths)} images (batch_size={validated_cfg.batch_size})...")
    filenames, embeddings = embed_images(
        model,
        preprocess,
        image_paths,
        device,
        batch_size=validated_cfg.batch_size,
    )

    model_name_sanitized = validated_cfg.embedding_model.replace("/", "-")
    emb_out = run_dir / f"embeddings_openclip_{model_name_sanitized}.npz"
    np.savez_compressed(emb_out, filenames=np.array(filenames), embeddings=embeddings)
    print(f"Saved embeddings to {emb_out}")

    # Reduce
    print(f"Reducing to 2D using {validated_cfg.method}...")
    coords = reduce_embeddings(embeddings, method=validated_cfg.method)

    # Visualize
    viz_out = run_dir / f"embed_viz_{model_name_sanitized}_{validated_cfg.method}.pdf"
    print(f"Creating visualization (thumbnails limit={validated_cfg.thumbs_limit}) -> {viz_out}")
    plot_coords(coords, list(image_paths), viz_out, thumbs_limit=validated_cfg.thumbs_limit)

    # Rectangular Grid
    rect_grid_out = run_dir / f"embed_grid_rect_{model_name_sanitized}_{validated_cfg.method}.pdf"
    print(f"Rendering rectangular rasterfairy grid -> {rect_grid_out}")
    render_rectangular_grid(coords, image_paths, rect_grid_out)

    # Representatives
    if validated_cfg.representative_k > 0:
        print(f"Selecting {validated_cfg.representative_k} representative images via FPS...")
        rep_indices, _ = _greedy_k_center(embeddings, validated_cfg.representative_k)
        if rep_indices is not None and len(rep_indices) > 0:
            rep_coords = coords[rep_indices]
            rep_paths = [image_paths[i] for i in rep_indices]

            rep_grid_out = run_dir / f"embed_grid_representative_{model_name_sanitized}_{validated_cfg.method}.pdf"
            print(f"Rendering representative grid -> {rep_grid_out}")
            render_rectangular_grid(rep_coords, rep_paths, rep_grid_out)

            rep_simple_out = run_dir / f"embed_grid_representative_simple_{model_name_sanitized}_{validated_cfg.method}.pdf"
            print(f"Rendering simple representative grid -> {rep_simple_out}")
            render_simple_grid(rep_paths, rep_simple_out)
            
            # Uniform sample (even intervals)
            n_total = len(image_paths)
            k = validated_cfg.representative_k
            if n_total > 0:
                indices = np.linspace(0, n_total - 1, min(k, n_total), dtype=int)
                interval_paths = [image_paths[i] for i in indices]
                interval_out = run_dir / f"embed_grid_uniform_interval_{model_name_sanitized}_{validated_cfg.method}.pdf"
                print(f"Rendering uniform interval grid -> {interval_out}")
                render_simple_grid(interval_paths, interval_out)

            # Uniform random sample
            if n_total > 0:
                rng = np.random.default_rng(0) 
                if n_total <= k:
                    rand_indices = np.arange(n_total)
                else:
                    rand_indices = rng.choice(n_total, size=k, replace=False)
                rand_indices.sort()
                rand_paths = [image_paths[i] for i in rand_indices]
                rand_out = run_dir / f"embed_grid_uniform_random_{model_name_sanitized}_{validated_cfg.method}.pdf"
                print(f"Rendering uniform random grid -> {rand_out}")
                render_simple_grid(rand_paths, rand_out)

    print("Done.")

if __name__ == "__main__":
    main()
