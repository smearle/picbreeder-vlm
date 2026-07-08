"""
Utilities for turning saved population metrics into plots.
"""

from __future__ import annotations

import gzip
import json
import math
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from picbreeder_vlm.core.neat_components import CHECKPOINT_SUFFIX
from picbreeder_vlm.core.rendering import render_genome_image


METRICS_FILENAME = "population_structure.jsonl"
PLOT_FILENAME = "population_structure.png"
SELECTION_GRID_FILENAME = "first_selection_grid.png"


def load_population_metrics(metrics_path: Path) -> List[Dict[str, Any]]:
    """
    Load population metrics from a JSONL file.

    Entries are expected to contain a `generation` integer key alongside nested
    statistics for `node_count` and `depth`, and the `parents_selected` count.
    """
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")

    metrics: List[Dict[str, Any]] = []
    with metrics_path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            metrics.append(data)

    return sorted(metrics, key=lambda item: item.get("generation", 0))


def render_population_structure_plots(
    metrics: Iterable[Dict[str, Any]],
    output_dir: Path,
    filename: str = PLOT_FILENAME,
) -> Path:
    """
    Save plots that describe the evolution of population structure metrics.

    Returns the path to the written figure.
    """
    metrics_list = sorted(metrics, key=lambda item: item.get("generation", 0))
    if not metrics_list:
        raise ValueError("No metrics provided for plotting.")

    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "matplotlib is required to render population structure plots."
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)

    generations = [int(entry.get("generation", 0)) for entry in metrics_list]

    def _series(key: str, sub_key: str) -> List[float]:
        values: List[float] = []
        for entry in metrics_list:
            group = entry.get(key, {})
            value = group.get(sub_key, 0.0) if isinstance(group, dict) else 0.0
            values.append(float(value))
        return values

    node_avg = _series("node_count", "avg")
    node_min = _series("node_count", "min")
    node_max = _series("node_count", "max")
    node_std = _series("node_count", "std")

    depth_avg = _series("depth", "avg")
    depth_min = _series("depth", "min")
    depth_max = _series("depth", "max")
    depth_std = _series("depth", "std")

    parents_selected = [
        float(entry.get("parents_selected", 0)) for entry in metrics_list
    ]

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 12))

    ax_nodes = axes[0]
    ax_nodes.plot(generations, node_avg, label="average", color="#1f77b4")
    ax_nodes.plot(generations, node_min, label="min", color="#2ca02c", linestyle="--")
    ax_nodes.plot(generations, node_max, label="max", color="#d62728", linestyle="--")
    node_lower = [avg - std for avg, std in zip(node_avg, node_std)]
    node_upper = [avg + std for avg, std in zip(node_avg, node_std)]
    ax_nodes.fill_between(
        generations,
        node_lower,
        node_upper,
        color="#1f77b4",
        alpha=0.15,
        label="±1 std",
    )
    ax_nodes.set_ylabel("Nodes")
    ax_nodes.set_title("Population node counts")
    ax_nodes.legend(loc="best")
    ax_nodes.grid(True, which="major", alpha=0.3)

    ax_depth = axes[1]
    ax_depth.plot(generations, depth_avg, label="average", color="#ff7f0e")
    ax_depth.plot(generations, depth_min, label="min", color="#2ca02c", linestyle="--")
    ax_depth.plot(generations, depth_max, label="max", color="#d62728", linestyle="--")
    depth_lower = [avg - std for avg, std in zip(depth_avg, depth_std)]
    depth_upper = [avg + std for avg, std in zip(depth_avg, depth_std)]
    ax_depth.fill_between(
        generations,
        depth_lower,
        depth_upper,
        color="#ff7f0e",
        alpha=0.15,
        label="±1 std",
    )
    ax_depth.set_ylabel("Depth")
    ax_depth.set_title("Population depths")
    ax_depth.legend(loc="best")
    ax_depth.grid(True, which="major", alpha=0.3)

    ax_parents = axes[2]
    ax_parents.plot(
        generations,
        parents_selected,
        label="parents selected",
        color="#9467bd",
        marker="o",
    )
    ax_parents.set_xlabel("Generation")
    ax_parents.set_ylabel("Parents")
    ax_parents.set_title("Parents selected per generation")
    ax_parents.legend(loc="best")
    ax_parents.grid(True, which="major", alpha=0.3)

    # Compute mean and std dev for parents selected.
    if parents_selected:
        mean_parents = sum(parents_selected) / len(parents_selected)
        variance = sum(
            (x - mean_parents) ** 2 for x in parents_selected
        ) / len(parents_selected)
        stddev_parents = variance**0.5
        ax_parents.axhline(
            mean_parents,
            color="#9467bd",
            linestyle="--",
            label="mean",
        )
        ax_parents.fill_between(
            generations,
            [mean_parents - stddev_parents] * len(generations),
            [mean_parents + stddev_parents] * len(generations),
            color="#9467bd",
            alpha=0.15,
            label="±1 std",
        )
        ax_parents.legend(loc="best")
        # Also save these stats to disk in json
        stats_path = output_dir / "parents_selected_stats.json"
        stats = {
            "mean": mean_parents,
            "stddev": stddev_parents,
        }
        stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    fig.tight_layout()

    output_path = output_dir / filename
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return output_path


def render_population_structure_plots_from_experiment(
    experiment_dir: Path,
    metrics_filename: str = METRICS_FILENAME,
) -> Path:
    """
    Convenience wrapper that reads metrics for an experiment and writes plots
    alongside the metrics directory.
    """
    metrics_path = experiment_dir / "metrics" / metrics_filename
    metrics = load_population_metrics(metrics_path)
    return render_population_structure_plots(metrics, metrics_path.parent)


def _parse_generation_index(prefix: str) -> int:
    try:
        _, suffix = prefix.split("_", 1)
        return int(suffix)
    except (ValueError, IndexError) as exc:
        raise ValueError(f"Unable to parse generation number from '{prefix}'.") from exc


def _load_first_selection(selection_path: Path) -> Optional[int]:
    with selection_path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    selected = data.get("selected") or []
    if not isinstance(selected, list) or not selected:
        return None
    first = selected[0]
    if not isinstance(first, int):
        raise ValueError(
            f"Expected integer for first selection in {selection_path}, got {first!r}"
        )
    return first


def _find_first_selection_image(
    populations_dir: Path,
    generation_prefix: str,
    selected_index: int,
) -> Optional[Path]:
    index_variants = (
        f"{selected_index:02d}",
        f"{selected_index:03d}",
        str(selected_index),
    )
    for index_str in index_variants:
        candidate = populations_dir / f"{generation_prefix}_idx_{index_str}.png"
        if candidate.exists():
            return candidate
    return None


def _infer_grid_dimensions(count: int) -> Tuple[int, int]:
    if count <= 0:
        raise ValueError("Grid size must be based on a positive image count.")
    cols = max(1, math.ceil(math.sqrt(count)))
    rows = math.ceil(count / cols)
    while (rows - 1) * cols >= count:
        rows -= 1
    return rows, cols


def _load_checkpoint_config_and_population(checkpoint_path: Path) -> Tuple[Any, Dict[int, Any]]:
    try:
        import neat  # type: ignore  # noqa: F401
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "The 'neat' package is required to regenerate missing selection images."
        ) from exc

    with gzip.open(checkpoint_path, "rb") as handle:
        data = pickle.load(handle)

    if not isinstance(data, tuple) or len(data) < 3:
        raise ValueError(f"Unexpected checkpoint format in {checkpoint_path}")

    _, config, population = data[:3]
    if not isinstance(population, dict):
        raise ValueError(f"Checkpoint {checkpoint_path} does not contain a population dictionary.")
    return config, population


def _resolve_genome_from_checkpoints(
    populations_dir: Path,
    generation_index: int,
    genome_id: int,
) -> Tuple[Any, Any]:
    preferred: List[Path] = []

    primary = populations_dir / f"gen_{generation_index:03d}{CHECKPOINT_SUFFIX}"
    if primary.exists():
        preferred.append(primary)

    secondary = populations_dir / f"gen_{generation_index + 1:03d}{CHECKPOINT_SUFFIX}"
    if secondary.exists() and secondary not in preferred:
        preferred.append(secondary)

    if not preferred:
        preferred = sorted(populations_dir.glob(f"gen_*{CHECKPOINT_SUFFIX}"))

    checked: set[Path] = set()
    for checkpoint_path in preferred:
        if checkpoint_path in checked or not checkpoint_path.exists():
            continue
        checked.add(checkpoint_path)
        config, population = _load_checkpoint_config_and_population(checkpoint_path)
        genome = population.get(genome_id)
        if genome is not None:
            return config, genome

    # Fall back to scanning remaining checkpoints if preferred set did not contain the genome.
    for checkpoint_path in sorted(populations_dir.glob(f"gen_*{CHECKPOINT_SUFFIX}")):
        if checkpoint_path in checked:
            continue
        config, population = _load_checkpoint_config_and_population(checkpoint_path)
        genome = population.get(genome_id)
        if genome is not None:
            return config, genome

    raise ValueError(
        f"Genome id {genome_id} could not be located in checkpoints under {populations_dir}"
    )


def _regenerate_first_selection_image(
    experiment_dir: Path,
    generation_prefix: str,
    generation_index: int,
    selected_index: int,
) -> Optional[Path]:
    populations_dir = experiment_dir / "populations"
    state_path = populations_dir / f"{generation_prefix}_state.json"
    if not state_path.exists():
        print(
            f"[WARN] Missing state snapshot for {generation_prefix} in {experiment_dir}",
            file=sys.stderr,
        )
        return None

    state = json.loads(state_path.read_text(encoding="utf-8"))
    images = state.get("images") or []
    target_entry: Optional[Dict[str, Any]] = None
    for entry in images:
        try:
            if int(entry.get("index")) == selected_index:
                target_entry = entry
                break
        except (TypeError, ValueError):
            continue

    if target_entry is None:
        print(
            f"[WARN] Selection index {selected_index} not found in {state_path}",
            file=sys.stderr,
        )
        return None

    genome_id = target_entry.get("genomeId")
    if genome_id is None:
        print(
            f"[WARN] Missing genomeId for index {selected_index} in {state_path}",
            file=sys.stderr,
        )
        return None

    try:
        genome_id_int = int(genome_id)
    except (TypeError, ValueError):
        print(
            f"[WARN] Invalid genomeId {genome_id!r} for index {selected_index} in {state_path}",
            file=sys.stderr,
        )
        return None

    scheme = str(state.get("scheme", "color"))

    width = target_entry.get("width") or state.get("thumbSize")
    height = target_entry.get("height") or state.get("thumbSize")
    try:
        width_int = int(width)
        height_int = int(height)
    except (TypeError, ValueError):
        print(
            f"[WARN] Invalid dimensions for index {selected_index} in {state_path}",
            file=sys.stderr,
        )
        return None

    try:
        config, genome = _resolve_genome_from_checkpoints(
            populations_dir,
            generation_index,
            genome_id_int,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(
            f"[WARN] Unable to regenerate image for {generation_prefix} index {selected_index}: {exc}",
            file=sys.stderr,
        )
        return None

    image = render_genome_image(
        genome,
        config,
        width_int,
        height_int,
        scheme,
    )

    primary_path = populations_dir / f"{generation_prefix}_idx_{selected_index:03d}.png"
    primary_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(primary_path)

    legacy_path: Optional[Path] = None
    if selected_index < 100:
        legacy_path = populations_dir / f"{generation_prefix}_idx_{selected_index:02d}.png"
        if legacy_path != primary_path:
            image.save(legacy_path)

    image.close()
    return legacy_path if legacy_path is not None else primary_path


def render_first_selection_grid(
    experiment_dir: Path,
    output_dir: Optional[Path] = None,
    filename: str = SELECTION_GRID_FILENAME,
    background_color: Tuple[int, int, int] = (0, 0, 0),
) -> Path:
    """
    Render a grid showing the first selected individual from each generation.
    """
    try:
        from PIL import Image  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Pillow is required to render the first-selection grid."
        ) from exc

    queries_dir = experiment_dir / "queries"
    if not queries_dir.exists():
        raise FileNotFoundError(f"Missing queries directory in {experiment_dir}")

    metadata_dir = queries_dir / "metadata"
    if metadata_dir.exists():
        selection_files = list(metadata_dir.glob("gen_*_selection.json"))
    else:
        selection_files = list(queries_dir.glob("gen_*_selection.json"))

    if not selection_files:
        raise FileNotFoundError(
            f"No selection metadata found in {queries_dir} (or its metadata subdirectory)."
        )

    populations_dir = experiment_dir / "populations"
    if not populations_dir.exists():
        raise FileNotFoundError(f"Missing populations directory in {experiment_dir}")

    ordered_entries: List[Tuple[int, Path]] = []
    for selection_file in selection_files:
        generation_prefix = selection_file.stem.replace("_selection", "")
        try:
            generation_index = _parse_generation_index(generation_prefix)
        except ValueError:
            print(
                f"[WARN] Skipping unparseable selection file: {selection_file}",
                file=sys.stderr,
            )
            continue

        selected_index = _load_first_selection(selection_file)
        if selected_index is None:
            continue

        image_path = _find_first_selection_image(
            populations_dir, generation_prefix, selected_index
        )
        if image_path is None:
            image_path = _regenerate_first_selection_image(
                experiment_dir,
                generation_prefix,
                generation_index,
                selected_index,
            )
        if image_path is None:
            print(
                f"[WARN] Could not locate image for {generation_prefix} index {selected_index} in {experiment_dir}",
                file=sys.stderr,
            )
            continue

        ordered_entries.append((generation_index, image_path))

    ordered_entries.sort(key=lambda item: item[0])
    image_paths = [entry[1] for entry in ordered_entries]

    if not image_paths:
        raise ValueError("No first-selection images were found to render.")

    loaded_images: List[Image.Image] = []
    for image_path in image_paths:
        with Image.open(image_path) as img:
            loaded_images.append(img.convert("RGB"))

    rows, cols = _infer_grid_dimensions(len(loaded_images))
    column_widths = [0] * cols
    row_heights = [0] * rows

    for index, image in enumerate(loaded_images):
        row = index // cols
        col = index % cols
        column_widths[col] = max(column_widths[col], image.width)
        row_heights[row] = max(row_heights[row], image.height)

    total_width = sum(column_widths)
    total_height = sum(row_heights)

    x_offsets = []
    offset = 0
    for width in column_widths:
        x_offsets.append(offset)
        offset += width

    y_offsets = []
    offset = 0
    for height in row_heights:
        y_offsets.append(offset)
        offset += height

    canvas = Image.new("RGB", (total_width, total_height), color=background_color)

    for index, image in enumerate(loaded_images):
        row = index // cols
        col = index % cols
        canvas.paste(image, (x_offsets[col], y_offsets[row]))
        image.close()

    if output_dir is None:
        output_dir = experiment_dir / "metrics"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / filename
    canvas.save(output_path)
    canvas.close()

    return output_path
