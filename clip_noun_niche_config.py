from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from constants import DEFAULT_NOUNLIST_PATH


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "picture2d" / "interactive_config_color"


@dataclass
class ClipNounNicheConfig:
    generations: int = 1_000
    mu: int = 12
    lambda_offspring: int = 36
    stage_length: int = 5
    render_size: int = 224
    batch_size: int = 8
    mutation_strength: float = 0.5
    new_random_prob: float = 0.05
    save_offspring_grids: bool = False
    save_images: bool = False
    nounlist: Path = DEFAULT_NOUNLIST_PATH
    config: Path = DEFAULT_CONFIG_PATH
    output_dir: Path = Path("outputs") / "clip_noun_niche_es"
    device: str | None = None
    seed: int | None = None
    clip_model: str = "ViT-H-14"
    clip_pretrained: str = "laion2b_s32b_b79k"
