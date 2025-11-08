import base64
import json
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import neat

from picture2d.common import eval_genome_as_grayscale_and_color
from rendering import decode_image, render_genome_diagram, render_genome_image


def build_generation_state(
    genomes: List[Tuple[int, neat.DefaultGenome]],
    config: neat.Config,
    generation: int,
    rows: int,
    cols: int,
    thumb_size: int,
    variant: str = "gray",  # "gray" | "color" | "both"
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[int, bytes]]]:
    """Build per-variant generation states and PNG caches.

    Returns a tuple of:
    - states: {"gray"|"color": {generation, rows, cols, thumbSize, images[], variant}}
    - png_caches: {"gray"|"color": {index -> png_bytes}}
    """

    # Determine which variants to produce
    variants = ["gray", "color"] if variant == "both" else [variant]

    # Prepare containers per variant
    images_by_variant: Dict[str, List[Dict[str, Any]]] = {v: [] for v in variants}
    png_caches: Dict[str, Dict[int, bytes]] = {v: {} for v in variants}

    # Render each genome once, then fill all requested variants
    for idx, (genome_id, genome) in enumerate(genomes):
        gray_image, color_image = render_genome_image(
            genome, config, thumb_size, thumb_size
        )

        row = idx // cols
        col = idx % cols

        for v in variants:
            image = gray_image if v == "gray" else color_image

            buffer = BytesIO()
            image.save(buffer, format="PNG")
            png_bytes = buffer.getvalue()
            png_caches[v][idx] = png_bytes

            encoded = base64.b64encode(png_bytes).decode("ascii")

            images_by_variant[v].append(
                {
                    "index": idx,
                    "row": row,
                    "col": col,
                    "width": image.width,
                    "height": image.height,
                    "data": encoded,
                    "genomeId": genome_id,
                    "encoding": "png",
                    "mode": image.mode,
                }
            )

    # Assemble final state objects per variant
    states: Dict[str, Dict[str, Any]] = {}
    for v in variants:
        states[v] = {
            "generation": generation,
            "rows": rows,
            "cols": cols,
            "thumbSize": thumb_size,
            "images": images_by_variant[v],
            "variant": v,
        }

    return states, png_caches


def save_neat_population(
    state: Dict[str, Any],
    output_dir: Path,
    generation: int,
    image_cache: Optional[Dict[int, bytes]] = None,
    decode_image_fn: Callable[[Dict[str, Any]], Any] = decode_image,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    snapshot = {
        key: state[key]
        for key in ("generation", "rows", "cols", "thumbSize", "scheme")
        if key in state
    }
    snapshot["images"] = []
    for entry in state["images"]:
        snapshot_entry = {
            key: value
            for key, value in entry.items()
            if key != "data"
        }
        snapshot["images"].append(snapshot_entry)

    state_path = output_dir / f"gen_{generation:03d}_state.json"
    state_path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    return state_path


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
