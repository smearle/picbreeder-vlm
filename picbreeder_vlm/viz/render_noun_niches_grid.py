#!/usr/bin/env python3
"""Render a captioned grid of current niche elites from the latest clip_noun_niche_es state."""
from __future__ import annotations

import json
import math
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt
from hydra import main as hydra_main
from hydra.core.config_store import ConfigStore
from hydra.utils import get_original_cwd
import numpy as np
from PIL import Image, ImageDraw

from picbreeder_vlm.niches.clip_noun_niche_config import ClipNounNicheConfig
from picbreeder_vlm.niches.clip_noun_niche_es import (
    NicheElite,
    STATE_FILENAME,
    build_config,
    load_nouns,
    render_population,
    sanitize_noun,
)
from picbreeder_vlm.niches.clip_noun_niche_shared import build_run_name, resolve_path, decompress_run_images
from picbreeder_vlm.core.neat_components import PicbreederGenome
from picbreeder_vlm.core.rendering import try_load_font, create_captioned_grid
from picbreeder_vlm.core.utils import resolve_nounlist


@dataclass
class RenderNounGridConfig(ClipNounNicheConfig):
    run_dir: Path | None = None
    output: Path | None = None
    thumb_size: int | None = None
    margin: int = 12
    font_size: int = 10
    include_score: bool = True
    start_generation: int = 10


cs = ConfigStore.instance()
cs.store(name="render_noun_grid", node=RenderNounGridConfig)

def load_state(path: Path) -> dict:
    with path.open("rb") as handle:
        return pickle.load(handle)


def format_caption(noun: str, score: float | None, include_score: bool) -> str:
    if include_score and score is not None and np.isfinite(score):
        distance = 1.0 - score
        return f"{noun} (dist {distance:.3f})"
    return noun


def run_render(cfg: RenderNounGridConfig, original_cwd: Path) -> None:
    nounlist_path = resolve_nounlist(cfg.nounlist, original_cwd)
    config_path = resolve_path(cfg.config, original_cwd)
    output_root = resolve_path(cfg.output_dir, original_cwd)
    run_dir = resolve_path(cfg.run_dir, original_cwd) if cfg.run_dir else output_root / build_run_name(cfg)

    # Decompress images if they are zipped (to access elites), but only if zip is newer than elites
    zip_path = run_dir / "images.zip"
    elites_dir = run_dir / "elites"
    should_decompress = False

    if zip_path.exists():
        should_decompress = True
        if elites_dir.exists():
            # Check if elites dir has any files
            try:
                # Quick check for any file
                if any(elites_dir.iterdir()):
                     # Get max mtime of elites (files only)
                     elite_mtimes = (p.stat().st_mtime for p in elites_dir.rglob("*") if p.is_file())
                     # Use a default low value if iterator is empty (though any check passed)
                     elites_mtime = max(elite_mtimes, default=0)
                     zip_mtime = zip_path.stat().st_mtime
                     
                     if elites_mtime > zip_mtime:
                         print(f"Skipping decompression: 'elites' directory is newer than 'images.zip'")
                         should_decompress = False
            except (OSError, ValueError):
                pass

    if should_decompress:
        decompress_run_images(run_dir, remove_zip=True)

    state_path = run_dir / STATE_FILENAME
    if not state_path.exists():
        raise FileNotFoundError(f"State file not found: {state_path}")

    payload = load_state(state_path)
    signature = payload.get("args_signature") or {}

    render_size = int(signature.get("render_size", cfg.render_size))
    thumb_size = cfg.thumb_size or render_size
    mutation_strength = float(signature.get("mutation_strength", cfg.mutation_strength))
    crossover_strength = float(signature.get("crossover_strength", cfg.crossover_strength))
    batch_size = int(signature.get("batch_size", cfg.batch_size))

    nouns = load_nouns(nounlist_path)
    niche_elites: List[NicheElite | None] = payload["niche_elites"]
    best_scores = payload.get("best_scores")

    config = build_config(config_path, batch_size, mutation_strength, crossover_strength)
    
    # Identify active niches
    active_indices = [i for i, elite in enumerate(niche_elites) if elite is not None]
    if not active_indices:
        raise RuntimeError(f"No elites found in state {state_path}")

    # Sort active_indices by score descending
    def get_score(idx):
        s = None
        if best_scores is not None:
             s = best_scores[idx]
        elif niche_elites[idx] is not None:
             s = niche_elites[idx].score
        
        if s is None or not np.isfinite(s):
            return -float('inf')
        return s

    active_indices.sort(key=get_score, reverse=True)

    # Prepare captions
    captions = [
        format_caption(
            nouns[idx],
            niche_elites[idx].score if best_scores is None else best_scores[idx],
            cfg.include_score,
        )
        for idx in active_indices
    ]

    # Load images from disk, falling back to rendering if necessary
    images: List[Image.Image] = []
    genomes_to_render: List[PicbreederGenome] = []
    render_indices: List[int] = []

    print(f"Loading images for {len(active_indices)} active niches...")
    for idx in active_indices:
        noun = nouns[idx]
        noun_slug = sanitize_noun(noun)
        img_path = run_dir / "elites" / f"{noun_slug}.png"
        
        loaded = False
        if img_path.exists():
            try:
                img = Image.open(img_path).convert("RGB")
                # We do not resize here if we want to support any render size, 
                # but create_captioned_grid will resize to thumb_size anyway.
                # However, if render_population returns images of render_size,
                # and loaded images are render_size, it's consistent.
                images.append(img)
                loaded = True
            except Exception as e:
                print(f"Warning: Failed to load {img_path}: {e}")
        
        if not loaded:
            # Placeholder to keep index alignment, will replace later
            images.append(None) 
            genomes_to_render.append(niche_elites[idx].genome)
            render_indices.append(len(images) - 1)

    if genomes_to_render:
        print(f"Rendering {len(genomes_to_render)} missing images...")
        # Note: render_population returns images of size config.render_size (which comes from signature)
        # We should use that size or thumb_size?
        # render_population uses 'render_size' argument for width/height.
        # We pass 'thumb_size' to render_population in original code?
        # Original: images = render_population(genomes, config, thumb_size, ...)
        # So we should pass thumb_size if we want them to be thumb_size.
        # But if we load from disk, they are 'render_size' (from simulation).
        # If thumb_size != render_size, we might have mixed sizes.
        # create_captioned_grid resizes everything to thumb_size.
        # So it's safe to render at render_size or thumb_size.
        # Let's render at thumb_size to be fast, or render_size to be accurate?
        # Original used thumb_size.
        rendered_imgs = render_population(genomes_to_render, config, thumb_size)
        for list_idx, img in zip(render_indices, rendered_imgs):
            images[list_idx] = img
 
    generation = payload.get("generation", 0)
    default_output = run_dir / f"grid_gen{int(generation):04d}.png"
    output_path = resolve_path(cfg.output, original_cwd) if cfg.output else default_output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    grid = create_captioned_grid(images, captions, thumb_size, cfg.margin, cfg.font_size)

    grid.save(output_path, format="PNG")

    print(f"Loaded/Rendered {len(images)} niches from {state_path}")
    print(f"Generation: {generation} | Run dir: {run_dir}")
    print(f"Grid saved to: {output_path}")

    try:
        plot_metrics(run_dir, output_path.parent / "metrics.png", start_gen=cfg.start_generation)
    except Exception as e:
        print(f"Error plotting metrics: {e}")


def plot_metrics(run_dir: Path, output_path: Path, start_gen: int = 0) -> None:
    metrics_file = run_dir / "metrics.jsonl"
    if not metrics_file.exists():
        print(f"Metrics file not found at {metrics_file}, skipping plot.")
        return

    data = []
    try:
        with metrics_file.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        if record.get("generation", -1) >= start_gen:
                            data.append(record)
                    except json.JSONDecodeError:
                        continue
    except Exception as e:
        print(f"Error reading metrics file: {e}")
        return
    
    if not data:
        print(f"No metrics data found (start_gen={start_gen}).")
        return

    generations = [d.get("generation", 0) for d in data]
    replacements = [d.get("replacements", 0) for d in data]
    qd_scores = [d.get("qd_score", 0) for d in data]
    max_scores = [d.get("max_best_score", 0) for d in data]
    mean_scores = [d.get("mean_best_score", 0) for d in data]
    std_scores = [d.get("std_best_score", 0) for d in data]

    # Use a style if available, else default
    try:
        plt.style.use('ggplot')
    except:
        pass

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Evolution Metrics: {run_dir.name}\n(starting from gen {start_gen})", fontsize=16)

    def plot_ax(ax, x, y, title, ylabel, color, std=None):
        ax.plot(x, y, marker='.', linestyle='-', color=color, alpha=0.7)
        if std is not None:
            upper = [yi + si for yi, si in zip(y, std)]
            lower = [yi - si for yi, si in zip(y, std)]
            ax.fill_between(x, lower, upper, color=color, alpha=0.2)
        ax.set_title(title)
        ax.set_xlabel("Generation")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    plot_ax(axs[0, 0], generations, replacements, "Replacements", "Count", "tab:blue")
    plot_ax(axs[0, 1], generations, qd_scores, "QD Score", "Score", "tab:orange")
    plot_ax(axs[1, 0], generations, max_scores, "Max Fitness", "Score", "tab:red")
    
    has_std = any(s > 0 for s in std_scores)
    plot_ax(axs[1, 1], generations, mean_scores, "Mean Fitness", "Score", "tab:purple", std=std_scores if has_std else None)

    plt.tight_layout()
    try:
        plt.savefig(output_path)
        print(f"Metrics plot saved to {output_path}")
    except Exception as e:
        print(f"Failed to save metrics plot: {e}")
    finally:
        plt.close(fig)


@hydra_main(version_base="1.3", config_path=None, config_name="render_noun_grid")
def main(cfg: RenderNounGridConfig) -> None:
    original_cwd = Path(get_original_cwd())
    run_render(cfg, original_cwd)


if __name__ == "__main__":
    main()
