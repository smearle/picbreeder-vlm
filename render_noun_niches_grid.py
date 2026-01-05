#!/usr/bin/env python3
"""Render a captioned grid of current niche elites from the latest clip_noun_niche_es state."""
from __future__ import annotations

import math
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

from hydra import main as hydra_main
from hydra.core.config_store import ConfigStore
from hydra.utils import get_original_cwd
import numpy as np
from PIL import Image, ImageDraw

from clip_noun_niche_config import ClipNounNicheConfig
from clip_noun_niche_es import (
    NicheElite,
    STATE_FILENAME,
    build_config,
    load_nouns,
    render_population,
)
from clip_noun_niche_shared import build_run_name, resolve_path
from rendering import try_load_font


@dataclass
class RenderNounGridConfig(ClipNounNicheConfig):
    run_dir: Path | None = None
    output: Path | None = None
    thumb_size: int | None = None
    margin: int = 12
    font_size: int = 18
    include_score: bool = False


cs = ConfigStore.instance()
cs.store(name="render_noun_grid", node=RenderNounGridConfig)

def load_state(path: Path) -> dict:
    with path.open("rb") as handle:
        return pickle.load(handle)


def format_caption(noun: str, score: float | None, include_score: bool) -> str:
    if include_score and score is not None and np.isfinite(score):
        return f"{noun} ({score:.3f})"
    return noun


def text_size(text: str, font) -> Tuple[int, int]:
    bbox = font.getbbox(text)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    return width, height


def create_captioned_grid(
    images: Sequence[Image.Image],
    captions: Sequence[str],
    thumb_size: int,
    margin: int,
    font_size: int,
) -> Image.Image:
    font = try_load_font(font_size)
    if not images:
        raise ValueError("No images to render in grid.")

    cols = max(1, int(math.ceil(math.sqrt(len(images)))))
    rows = int(math.ceil(len(images) / cols))
    label_height = max(text_size(caption, font)[1] for caption in captions)

    cell_height = thumb_size + label_height + 6
    width = (cols * thumb_size) + ((cols + 1) * margin)
    height = (rows * cell_height) + ((rows + 1) * margin)
    canvas = Image.new("RGB", (width, height), (16, 16, 20))
    draw = ImageDraw.Draw(canvas)

    for i, (img, caption) in enumerate(zip(images, captions)):
        row = i // cols
        col = i % cols
        x = margin + col * (thumb_size + margin)
        y = margin + row * (cell_height + margin)

        if img.size != (thumb_size, thumb_size):
            img = img.resize((thumb_size, thumb_size), resample=Image.BICUBIC)
        canvas.paste(img, (x, y))

        text_w, text_h = text_size(caption, font)
        text_x = x + max(0, (thumb_size - text_w) // 2)
        text_y = y + thumb_size + 2
        draw.text((text_x, text_y), caption, font=font, fill=(255, 255, 0))

    return canvas


def run_render(cfg: RenderNounGridConfig, original_cwd: Path) -> None:
    nounlist_path = resolve_path(cfg.nounlist, original_cwd)
    config_path = resolve_path(cfg.config, original_cwd)
    output_root = resolve_path(cfg.output_dir, original_cwd)
    run_dir = resolve_path(cfg.run_dir, original_cwd) if cfg.run_dir else output_root / build_run_name(cfg)

    state_path = run_dir / STATE_FILENAME
    if not state_path.exists():
        raise FileNotFoundError(f"State file not found: {state_path}")

    payload = load_state(state_path)
    signature = payload.get("args_signature") or {}

    render_size = int(signature.get("render_size", cfg.render_size))
    thumb_size = cfg.thumb_size or render_size
    mutation_strength = float(signature.get("mutation_strength", cfg.mutation_strength))
    mu = int(signature.get("mu", cfg.mu))

    nouns = load_nouns(nounlist_path)
    niche_elites: List[NicheElite | None] = payload["niche_elites"]
    best_scores = payload.get("best_scores")

    config = build_config(config_path, mu, mutation_strength)
    genomes = [elite.genome for elite in niche_elites if elite is not None]
    captions = [
        format_caption(
            nouns[idx],
            niche_elites[idx].score if best_scores is None else best_scores[idx],
            cfg.include_score,
        )
        for idx, elite in enumerate(niche_elites)
        if elite is not None
    ]

    if not genomes:
        raise RuntimeError(f"No elites found in state {state_path}")

    generation = payload.get("generation", 0)
    default_output = run_dir / f"grid_gen{int(generation):04d}"
    output_path = resolve_path(cfg.output, original_cwd) if cfg.output else default_output
    image_paths = [Path(os.path.join(run_dir, f"gen{int(generation):04d}.png", noun)) for noun in nouns if noun is not None]
    images = render_population(genomes, config, thumb_size, image_paths=image_paths)
    grid = create_captioned_grid(images, captions, thumb_size, cfg.margin, cfg.font_size)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output_path, format="PNG")

    print(f"Loaded {len(genomes)} niches from {state_path}")
    print(f"Generation: {generation} | Run dir: {run_dir}")
    print(f"Grid saved to: {output_path}")


@hydra_main(version_base="1.3", config_path=None, config_name="render_noun_grid")
def main(cfg: RenderNounGridConfig) -> None:
    original_cwd = Path(get_original_cwd())
    run_render(cfg, original_cwd)


if __name__ == "__main__":
    main()
