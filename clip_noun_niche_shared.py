from __future__ import annotations

import re
from pathlib import Path

from clip_noun_niche_config import ClipNounNicheConfig


def sanitize_tag(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip())
    return cleaned or "run"


def build_run_name(cfg: ClipNounNicheConfig) -> str:
    noun_tag = sanitize_tag(Path(cfg.nounlist).stem)
    config_tag = sanitize_tag(Path(cfg.config).stem)
    clip_tag = f"{sanitize_tag(cfg.clip_model)}-{sanitize_tag(cfg.clip_pretrained)}"
    parts = [
        f"nouns-{noun_tag}",
        f"mu{cfg.mu}",
        f"lam{cfg.lambda_offspring}",
        f"stage{cfg.stage_length}",
        f"render{cfg.render_size}",
        f"batch{cfg.batch_size}",
        f"mut{cfg.mutation_strength}",
        f"rand{cfg.new_random_prob}",
        f"clip-{clip_tag}",
        f"cfg-{config_tag}",
    ]
    if cfg.seed is not None:
        parts.append(f"seed{cfg.seed}")
    if cfg.save_images:
        parts.append("imgs")
        if cfg.save_offspring_grids:
            parts.append("grids")
    if cfg.save_offspring_grids and not cfg.save_images:
        parts.append("grids_flag_noimg")
    return "_".join(parts)


def resolve_path(path: Path, base: Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else base / p
