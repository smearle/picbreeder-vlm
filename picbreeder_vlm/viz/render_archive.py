#!/usr/bin/env python3
"""Render high-resolution archive images and grids from saved genomes."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import neat
from PIL import Image

from picbreeder_vlm.core.neat_components import (
    InteractiveStagnation,
    PicbreederGenome,
    apply_picbreeder_config_defaults,
)
from picbreeder_vlm.core.picbreeder_reproduction import PicbreederReproduction
from picbreeder_vlm.core.rendering import render_genome_image
from picbreeder_vlm.core.config import PicbreederConfig, ensure_valid_config
from picbreeder_vlm.core.utils import _ensure_absolute
import hydra
from hydra.conf import HelpConf, HydraConf
from hydra.core.config_store import ConfigStore
from hydra.utils import get_original_cwd


try:
    RESAMPLING = Image.Resampling  # Pillow >= 10
except AttributeError:  # pragma: no cover - Pillow < 10
    RESAMPLING = Image
LANCZOS = RESAMPLING.LANCZOS


VALID_VARIANTS = ("auto", "color", "gray")


@dataclass
class ArchiveHighresConfig(PicbreederConfig):
    output_dir: Optional[Path] = None
    grid_output: Optional[Path] = None
    image_size: int = 128
    grid_thumb_size: int = 128
    grid_margin: int = 24
    variant: str = "auto"
    limit: Optional[int] = None
    overwrite: bool = False
    subset_count: int = 100
    hydra: HydraConf = field(
        default_factory=lambda: HydraConf(
            help=HelpConf(
                app_name="render_archive",
                header=(
                    "Hydra entry point for rendering high-resolution archive images and grids.\n"
                    "\n"
                    "Common overrides:\n"
                    "  experiment_dir      Point directly at an existing run.\n"
                    "  goal/scheme/seed    Combine with ensure_valid_config to infer a run directory.\n"
                    "  output_dir          Custom directory for per-image renders.\n"
                    "  grid_output         Override the archive grid destination.\n"
                ),
                footer="Override with +option=value (e.g. variant=color image_size=256 subset_count=200).",
            )
        )
    )


ConfigStore.instance().store(name="archive_highres_base", node=ArchiveHighresConfig)


def _validate_render_options(cfg: ArchiveHighresConfig) -> None:
    if cfg.image_size <= 0:
        raise ValueError("image_size must be positive")
    if cfg.grid_thumb_size <= 0:
        raise ValueError("grid_thumb_size must be positive")
    if cfg.grid_margin < 0:
        raise ValueError("grid_margin must be non-negative")
    if cfg.limit is not None and cfg.limit <= 0:
        raise ValueError("limit must be positive when provided")
    if cfg.subset_count < -1 or cfg.subset_count == 0:
        raise ValueError("subset_count must be positive or -1")
    if cfg.variant not in VALID_VARIANTS:
        raise ValueError(f"variant must be one of {VALID_VARIANTS}")


def _resolve_output_paths(
    cfg: ArchiveHighresConfig,
    archive_dir: Path,
    original_cwd: Path,
) -> tuple[Path, Path]:
    if cfg.output_dir is None:
        output_dir = archive_dir / "highres_images"
    else:
        output_dir = _ensure_absolute(Path(cfg.output_dir), original_cwd)

    if cfg.grid_output is None:
        grid_output = archive_dir / "archive_grid_highres.png"
    else:
        grid_output = _ensure_absolute(Path(cfg.grid_output), original_cwd)

    return output_dir.resolve(), grid_output.resolve()


def load_config(config_path: Path) -> neat.Config:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    config = neat.Config(
        PicbreederGenome,
        PicbreederReproduction,
        neat.DefaultSpeciesSet,
        InteractiveStagnation,
        str(config_path),
    )
    apply_picbreeder_config_defaults(config)
    return config


def load_archive_metadata(archive_dir: Path) -> Dict[str, Any]:
    metadata_path = archive_dir / "archive_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"archive_metadata.json not found under {archive_dir}")
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def resolve_entry_path(entry_value: Optional[str], fallback_dir: Path) -> Path:
    if entry_value:
        candidate = Path(entry_value)
        if candidate.exists():
            return candidate
        fallback = fallback_dir / candidate.name
        if fallback.exists():
            return fallback
    if entry_value:
        return fallback_dir / Path(entry_value).name
    raise FileNotFoundError(f"No valid path supplied and fallback directory ({fallback_dir}) missing target")


def pick_variant(
    gray_image: Image.Image,
    color_image: Image.Image,
    entry: Dict[str, Any],
    mode: str,
) -> Image.Image:
    if mode == "color":
        return color_image
    if mode == "gray":
        return gray_image
    color_enabled = bool(entry.get("color_enabled"))
    return color_image if color_enabled else gray_image


def render_entry_image(
    entry: Dict[str, Any],
    config: neat.Config,
    image_size: int,
    archive_dir: Path,
    output_dir: Path,
    variant_mode: str,
    overwrite: bool,
) -> Optional[Path]:
    entry_id = entry.get("id", "unknown")
    output_path = output_dir / f"{entry_id}.png"
    if output_path.exists() and not overwrite:
        return output_path

    genome_path = resolve_entry_path(entry.get("genome_path"), archive_dir / "genomes")
    if not genome_path.exists():
        print(f"[warn] Genome for {entry_id} missing ({genome_path}). Skipping.", file=sys.stderr)
        return None

    with genome_path.open("rb") as handle:
        genome = pickle.load(handle)

    gray_image, color_image = render_genome_image(genome, config, image_size, image_size)
    variant = pick_variant(gray_image, color_image, entry, variant_mode)
    variant.save(output_path, format="PNG")
    return output_path


def build_label_free_grid(
    image_paths: Sequence[Path],
    thumb_size: int,
    margin: int,
    output_path: Path,
) -> Optional[Path]:
    if not image_paths:
        return None

    tiles: List[Image.Image] = []
    for path in image_paths:
        if not path.exists():
            continue
        with Image.open(path) as img:
            tile = img.convert("RGB")
            if thumb_size > 0 and (tile.width != thumb_size or tile.height != thumb_size):
                tile = tile.resize((thumb_size, thumb_size), LANCZOS)
            tiles.append(tile.copy())

    if not tiles:
        return None

    width, height = tiles[0].size
    columns = max(1, math.ceil(math.sqrt(len(tiles))))
    rows = math.ceil(len(tiles) / columns)
    canvas_width = columns * width + (columns + 1) * margin
    canvas_height = rows * height + (rows + 1) * margin
    canvas = Image.new("RGB", (canvas_width, canvas_height), (18, 18, 22))

    for idx, tile in enumerate(tiles):
        col = idx % columns
        row = idx // columns
        x = margin + col * (width + margin)
        y = margin + row * (height + margin)
        canvas.paste(tile, (x, y))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="PNG")
    return output_path


def iter_entries(metadata: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    entries = metadata.get("entries", [])
    try:
        return sorted(entries, key=lambda item: int(item.get("id", 0)))
    except ValueError:
        return sorted(entries, key=lambda item: item.get("id", ""))


@hydra.main(version_base="1.3", config_path=None, config_name="archive_highres_base")
def main(cfg: ArchiveHighresConfig) -> None:
    original_cwd = Path(get_original_cwd())
    validated_cfg = ensure_valid_config(cfg, original_cwd=original_cwd)
    _validate_render_options(validated_cfg)

    experiment_dir = Path(validated_cfg.experiment_dir).resolve()
    archive_dir = experiment_dir / "archive"
    output_dir, grid_output = _resolve_output_paths(validated_cfg, archive_dir, original_cwd)

    output_dir.mkdir(parents=True, exist_ok=True)

    neat_config_path = Path(validated_cfg.neat_config_path)
    config = load_config(neat_config_path)
    metadata = load_archive_metadata(archive_dir)
    entries = list(iter_entries(metadata))

    if validated_cfg.subset_count != -1 and len(entries) > validated_cfg.subset_count:
        if validated_cfg.subset_count == 1:
            entries = [entries[0]]
        elif validated_cfg.subset_count > 0:
            indices = [
                int(i * (len(entries) - 1) / (validated_cfg.subset_count - 1))
                for i in range(validated_cfg.subset_count)
            ]
            entries = [entries[i] for i in indices]

    if validated_cfg.limit is not None:
        entries = entries[: validated_cfg.limit]

    rendered_paths: List[Path] = []
    for index, entry in enumerate(entries, start=1):
        path = render_entry_image(
            entry,
            config=config,
            image_size=validated_cfg.image_size,
            archive_dir=archive_dir,
            output_dir=output_dir,
            variant_mode=validated_cfg.variant,
            overwrite=validated_cfg.overwrite,
        )
        if path is not None:
            rendered_paths.append(path)
            print(f"[{index}/{len(entries)}] Rendered {entry.get('id')} -> {path}")
        else:
            print(f"[{index}/{len(entries)}] Skipped {entry.get('id')} (missing genome or render failure).")

    if not rendered_paths:
        print("No images rendered; skipping grid assembly.", file=sys.stderr)
        return

    grid_path = build_label_free_grid(
        rendered_paths,
        thumb_size=validated_cfg.grid_thumb_size,
        margin=validated_cfg.grid_margin,
        output_path=grid_output,
    )
    if grid_path:
        print(f"High-res archive grid saved to {grid_path}")
    else:
        print("Grid rendering skipped (no valid images).", file=sys.stderr)


if __name__ == "__main__":
    main()
