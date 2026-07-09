import base64
import json
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import PIL
import neat

from picbreeder_vlm.core.picture2d import eval_genome_as_grayscale_and_color
from picbreeder_vlm.core.rendering import decode_image, render_genome_diagram, render_genome_image


def render_genome_images(
    genomes: List[Tuple[int, neat.DefaultGenome]],
    config: neat.Config,
    thumb_size: int,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[int, bytes]]]:
    """Build per-variant generation states and PNG caches.

    Returns a tuple of:
    - states: {"gray"|"color": {generation, rows, cols, thumbSize, images[], variant}}
    - png_caches: {"gray"|"color": {index -> png_bytes}}
    """

    gray_images = []
    color_images = []

    for genome_id, genome in genomes:
        gray_image, color_image = render_genome_image(
            genome, config, thumb_size, thumb_size
        )
        gray_images.append(gray_image)
        color_images.append(color_image)

    return gray_images, color_images


def save_neat_genome_diagrams(
    genomes: Sequence[Tuple[int, neat.DefaultGenome]],
    config: neat.Config,
    output_dir: Path,
    generation: int,
    fmt: str = "svg",
) -> List[Path]:
    generation_dir = output_dir / "diagrams" / f"gen_{generation:03d}"
    generation_dir.mkdir(parents=True, exist_ok=True)

    node_names = _infer_io_node_labels(config)

    artifacts: List[Path] = []
    for index, (genome_id, genome) in enumerate(genomes):
        stem = generation_dir / f"idx_{index:02d}_id_{genome_id}"
        rendered = render_genome_diagram(
            genome,
            config,
            stem,
            show_disabled=True,
            prune_unused=False,
            fmt=fmt,
            node_names=node_names if node_names else None,
        )
        if rendered is not None:
            artifacts.append(rendered)

    return artifacts


def _infer_io_node_labels(config: neat.Config) -> Dict[int, str]:
    """Infer friendly labels for known input/output nodes based on config ordering."""

    labels: Dict[int, str] = {}

    input_keys = list(getattr(config.genome_config, "input_keys", ()))
    output_keys = list(getattr(config.genome_config, "output_keys", ()))

    input_aliases = ("X", "Y", "R", "B")
    output_aliases = ("H", "S", "B")

    for key, alias in zip(input_keys, input_aliases):
        labels[int(key)] = alias

    for key, alias in zip(output_keys, output_aliases):
        labels[int(key)] = alias

    return labels
