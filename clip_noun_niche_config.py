from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "picture2d" / "interactive_config_color"


@dataclass
class ClipNounNicheConfig:
    generations: int = 10_000
    batch_size: int = 15
    stage_length: int = 5
    render_size: int = 128
    num_proc: int = 15
    mutation_strength: float = 0.5
    new_random_prob: float = 0.05
    crossover_strength: float = 0.2
    save_offspring_grids: bool = False
    save_images: bool = False
    nounlist: str = "things_deduped"
    config: Path = DEFAULT_CONFIG_PATH
    output_dir: Path = "clip_noun_niche_es_logs"
    device: str | None = None
    seed: int | None = None
    clip_model: str = "ViT-SO400M-14-SigLIP2"
    clip_pretrained: str = "webli"
