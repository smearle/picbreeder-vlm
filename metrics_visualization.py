"""
Utilities for turning persisted population metrics into plots.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


METRICS_FILENAME = "population_structure.jsonl"
PLOT_FILENAME = "population_structure.png"


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
