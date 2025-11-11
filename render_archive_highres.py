#!/usr/bin/env python3
"""Render high-resolution archive images and grids from saved genomes."""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import neat
from PIL import Image

from neat_components import (
    InteractiveStagnation,
    PicbreederGenome,
    apply_picbreeder_config_defaults,
)
from picbreeder_reproduction import PicbreederReproduction
from rendering import render_genome_image


try:
    RESAMPLING = Image.Resampling  # Pillow >= 10
except AttributeError:  # pragma: no cover - Pillow < 10
    RESAMPLING = Image
LANCZOS = RESAMPLING.LANCZOS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        required=True,
        help="Path to the experiment directory that contains (the archive subdirectory with) archive_metadata.json.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to the NEAT config file used for the experiment "
        "(e.g. picture2d/interactive_config_color).",
        default="picture2d/interactive_config_color",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Destination for per-entry renders (default: <archive-dir>/highres_images).",
    )
    parser.add_argument(
        "--grid-output",
        type=Path,
        default=None,
        help="Path for the label-free archive grid (default: <archive-dir>/archive_grid_highres.png).",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=1024,
        help="Width/height (in px) used when rendering individual genomes.",
    )
    parser.add_argument(
        "--grid-thumb-size",
        type=int,
        default=512,
        help="Tile size (in px) used when composing the grid.",
    )
    parser.add_argument(
        "--grid-margin",
        type=int,
        default=24,
        help="Margin (in px) used between tiles on the grid.",
    )
    parser.add_argument(
        "--variant",
        choices=("auto", "color", "gray"),
        default="auto",
        help="Color channel to export per image. 'auto' follows each archive entry's color flag.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on the number of archive entries to render (useful for testing).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-render entries even if their high-res image already exists.",
    )
    return parser.parse_args()


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
    return sorted(entries, key=lambda item: item.get("id", ""))


def main() -> None:
    args = parse_args()
    archive_dir = args.experiment_dir.resolve() / "archive"
    output_dir = (args.output_dir or (archive_dir / "highres_images")).resolve()
    grid_output = (args.grid_output or (archive_dir / "archive_grid_highres.png")).resolve()

    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config.resolve())
    metadata = load_archive_metadata(archive_dir)
    entries = list(iter_entries(metadata))
    if args.limit is not None:
        entries = entries[: args.limit]

    rendered_paths: List[Path] = []
    for index, entry in enumerate(entries, start=1):
        path = render_entry_image(
            entry,
            config=config,
            image_size=args.image_size,
            archive_dir=archive_dir,
            output_dir=output_dir,
            variant_mode=args.variant,
            overwrite=args.overwrite,
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
        thumb_size=args.grid_thumb_size,
        margin=args.grid_margin,
        output_path=grid_output,
    )
    if grid_path:
        print(f"High-res archive grid saved to {grid_path}")
    else:
        print("Grid rendering skipped (no valid images).", file=sys.stderr)


if __name__ == "__main__":
    main()
