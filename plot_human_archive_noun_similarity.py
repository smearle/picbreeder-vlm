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

# Set backend before importing other plotting modules
matplotlib.use("Agg")

from compute_noun_similarity import (
    prepare_openclip_components,
    load_nouns,
    format_prompts,
    embed_images,
    embed_texts,
    compute_mean_max_similarity_trajectory,
    plot_mean_max_similarity_trajectory,
    save_trajectory_json,
    NounSimilarityConfig,
)
from utils import _ensure_absolute, load_human_archive_images, resolve_nounlist
from config import ensure_valid_config


@dataclass
class HumanNounSimilarityConfig(NounSimilarityConfig):
    render_size: int = 128
    hydra: HydraConf = field(
        default_factory=lambda: HydraConf(
            help=HelpConf(
                app_name="plot_human_archive_noun_similarity",
                header=(
                    "Hydra entry point for human archive noun metrics.\n"
                    "Plots noun similarity score over time for human archives.\n"
                ),
                footer="Override with +option=value (e.g. render_size=512).",
            )
        )
    )

ConfigStore.instance().store(name="human_noun_similarity_base", node=HumanNounSimilarityConfig)

@hydra.main(version_base="1.3", config_path=None, config_name="human_noun_similarity_base")
def main(
    cfg: HumanNounSimilarityConfig,
) -> None:
    original_cwd = Path(get_original_cwd())
    validated_cfg = ensure_valid_config(cfg, original_cwd=original_cwd)
    
    # Construct archive dir based on render_size
    root_dir = original_cwd
    archive_dir = root_dir / "fer/src" / f"archive_res-{validated_cfg.render_size}"
    noun_file = resolve_nounlist(validated_cfg.nounlist, original_cwd)
    
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

    # Load nouns
    nouns_list = load_nouns(noun_file)
    prompts_list = format_prompts(nouns_list, validated_cfg.label_template)
    print(f"Loaded {len(nouns_list)} nouns.")

    # Setup device
    if validated_cfg.device:
        device = torch.device(validated_cfg.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    print(f"Loading OpenCLIP model {validated_cfg.embedding_model} ({validated_cfg.pretrained})...")
    model, preprocess, tokenizer = prepare_openclip_components(validated_cfg, device)

    # Embed images
    print("Embedding images...")
    image_embeddings = embed_images(
        model,
        preprocess,
        image_paths,
        device,
        batch_size=validated_cfg.batch_size,
    )

    # Embed nouns
    print("Embedding nouns...")
    noun_embeddings = embed_texts(
        model,
        tokenizer,
        prompts_list,
        device,
        batch_size=validated_cfg.noun_batch_size,
    )

    # Compute trajectory
    print("Computing trajectory...")
    trajectory = compute_mean_max_similarity_trajectory(image_embeddings, noun_embeddings, image_paths)

    # Output filenames
    # Include resolution and CLIP model in filename
    nounlist_name = noun_file.stem
    model_name = validated_cfg.embedding_model.replace("/", "-")
    
    filename_base = f"noun_similarity_res{validated_cfg.render_size}_{model_name}_{nounlist_name}"

    output_dir = root_dir / "human_baseline"
    output_dir.mkdir(exist_ok=True, parents=True)

    output_base = output_dir / filename_base
    json_path = output_base.with_suffix(".json")
    plot_path = output_base.with_suffix(".png")

    print(f"Saving results to {json_path} and {plot_path}")
    save_trajectory_json(trajectory, json_path)
    plot_mean_max_similarity_trajectory(trajectory, plot_path)
    
    print("Done.")

if __name__ == "__main__":
    main()
