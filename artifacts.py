import base64
import json
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import neat

from rendering import decode_image, render_genome_diagram, render_genome_image


def build_generation_state(
    genomes: List[Tuple[int, neat.DefaultGenome]],
    config: neat.Config,
    generation: int,
    rows: int,
    cols: int,
    thumb_size: int,
    scheme: str,
    palette: str,
    render_image_fn: Callable[..., Any] = render_genome_image,
) -> Tuple[Dict[str, Any], Dict[int, bytes]]:
    images: List[Dict[str, Any]] = []
    png_cache: Dict[int, bytes] = {}
    for idx, (genome_id, genome) in enumerate(genomes):
        image = render_image_fn(genome, config, thumb_size, thumb_size, scheme, palette)
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        png_bytes = buffer.getvalue()
        png_cache[idx] = png_bytes
        encoded = base64.b64encode(png_bytes).decode("ascii")
        row = idx // cols
        col = idx % cols
        images.append(
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

    state = {
        "generation": generation,
        "rows": rows,
        "cols": cols,
        "thumbSize": thumb_size,
        "scheme": scheme,
        "palette": palette,
        "images": images,
    }
    return state, png_cache


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
        for key in ("generation", "rows", "cols", "thumbSize", "scheme", "palette")
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
