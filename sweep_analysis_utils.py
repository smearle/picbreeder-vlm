from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Use Agg backend for non-interactive plotting
matplotlib.use("Agg")

SCRIPT_ROOT = Path(__file__).resolve().parent

_TAG_SANITIZE_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")

def sanitize_filename_tag(tag: str) -> str:
    cleaned = _TAG_SANITIZE_PATTERN.sub("_", str(tag)).strip("_")
    return cleaned or "sweep"

# Centralized display-name mapping for cross-eval/visualization labels.
HYPERPARAM_PRETTY_LABELS: Dict[str, str] = {
    "chat_history_turns": "turns in context",
    "rand_select_prob": "rand. selection prob.",
}

def pretty_hyperparam_name(name: str) -> str:
    return HYPERPARAM_PRETTY_LABELS.get(name, name)

def pretty_hyperparam_value(name: str, value: Any, values: Dict[str, Any]) -> str:
    """Pretty-print a config value for plotting/table labels."""

    if name == "chat_history_turns":
        # -1 means: keep all turns; for plotting, map to agent_generations.
        try:
            turns = int(value)
        except Exception:
            return str(value)
        if turns == -1:
            gens = values.get("agent_generations")
            try:
                return str(int(gens)) if gens is not None else str(turns)
            except Exception:
                return str(turns)
        return str(turns)

    if isinstance(value, float):
        return f"{value:g}"
    return str(value)

def effective_numeric_for_sort(name: str, value: Any, values: Dict[str, Any]) -> Optional[float]:
    """Return a numeric sort key for the swept hyperparameter when possible."""

    if name == "chat_history_turns":
        try:
            turns = int(value)
        except Exception:
            return None
        if turns == -1:
            gens = values.get("agent_generations")
            try:
                return float(int(gens)) if gens is not None else float(turns)
            except Exception:
                return float(turns)
        return float(turns)

    try:
        return float(value)
    except Exception:
        return None

def normalize_group_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        # Keep floats stable-ish in labels/keys.
        return float(f"{value:.6g}")
    if isinstance(value, list):
         return tuple(value)
    return value

def compute_varying_fields(group_keys: Sequence[Tuple[Tuple[str, Any], ...]]) -> List[str]:
    values_by_field: Dict[str, set] = {}
    for key in group_keys:
        for name, value in key:
            values_by_field.setdefault(name, set()).add(value)
    varying = [name for name, values in values_by_field.items() if len(values) > 1]
    # Deterministic ordering.
    varying.sort()
    return varying

def format_group_label(group_key: Tuple[Tuple[str, Any], ...], varying_fields: Sequence[str]) -> str:
    values = dict(group_key)
    
    if values.get("rand_select_mode") == "all":
        prob = values.get("rand_select_prob")
        if prob is not None:
            try:
                if np.isclose(float(prob), 2.0):
                    return "Random Baseline"
            except (ValueError, TypeError):
                pass

    parts = []
    for name in varying_fields:
        if name in values:
            pretty_name = pretty_hyperparam_name(name)
            pretty_value = pretty_hyperparam_value(name, values[name], values)
            parts.append(f"{pretty_name} = {pretty_value}")
    if not parts:
        return "default"
    return " ".join(parts)

def load_human_baseline(
    metric: str,
    render_size: int,
    model: str,
    nounlist: Optional[str] = None
) -> Optional[Dict[int, float]]:
    """Load human baseline trajectory for the given metric."""
    baseline_dir = SCRIPT_ROOT / "human_baseline"
    if not baseline_dir.exists():
        print(f"Human baseline directory not found: {baseline_dir}")
        return None
        
    model_name = model.replace("/", "-")
    
    if metric == "novelty":
        filename = f"novelty_res{render_size}_{model_name}.json"
        key = "mean_pairwise_distance"
    elif metric == "noun":
        if not nounlist:
            print("Nounlist name must be provided to load noun similarity baseline.")
            return None
        filename = f"noun_similarity_res{render_size}_{model_name}_{nounlist}.json"
        key = "mean_max_similarity"
    else:
        print(f"Unknown metric for human baseline: {metric}")
        return None
        
    path = baseline_dir / filename
    if not path.exists():
        print(f"Human baseline file not found: {path}")
        return None
        
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        trajectory = {}
        for row in data:
            idx = row.get("index")
            val = row.get(key)
            if idx is not None and val is not None:
                trajectory[int(idx)] = float(val)
        return trajectory
    except Exception as e:
        print(f"Failed to load human baseline from {path}: {e}")
        return None

def write_aggregate_plot(
    *,
    grouped_runs: Dict[Tuple[Tuple[str, Any], ...], List[Dict[int, float]]],
    outpath: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    baselines: Optional[List[Tuple[str, Dict[int, float]]]] = None,
) -> None:
    if not grouped_runs:
        return

    group_keys = list(grouped_runs.keys())
    varying_fields = compute_varying_fields(group_keys)

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, which="major", alpha=0.3)

    plotted = 0
    max_x = 0

    for group_key in group_keys:
        runs = grouped_runs[group_key]
        if not runs:
            continue

        # Use intersection so mean/std correspond to the same x positions across seeds.
        index_sets = [set(run.keys()) for run in runs]
        common = set.intersection(*index_sets) if len(index_sets) > 1 else index_sets[0]
        if not common:
            continue
        indices = sorted(common)
        if indices:
            max_x = max(max_x, indices[-1])
        values = np.array([[run[i] for i in indices] for run in runs], dtype=float)
        mean = values.mean(axis=0)
        std = values.std(axis=0)

        label = format_group_label(group_key, varying_fields)
        (line,) = ax.plot(indices, mean, linewidth=2, label=label)
        ax.fill_between(indices, mean - std, mean + std, alpha=0.2, color=line.get_color())
        plotted += 1

    if baselines and max_x > 0:
        # Use black if there's only one baseline, otherwise cycle colors
        use_black = (len(baselines) == 1)

        for label, trajectory in baselines:
            indices = sorted([i for i in trajectory.keys() if i <= max_x])
            if indices:
                values = [trajectory[i] for i in indices]
                
                kwargs = {
                    "linestyle": "--",
                    "linewidth": 2,
                    "label": label,
                    "alpha": 0.7,
                }
                if use_black:
                    kwargs["color"] = "black"
                else:
                    # Explicit coloring for known baselines
                    if "human" in label.lower():
                        kwargs["color"] = "black"
                    elif "random" in label.lower():
                        kwargs["color"] = "red"
                    # Else let it cycle or pick default

                ax.plot(indices, values, **kwargs)
                plotted += 1

    if plotted == 0:
        plt.close(fig)
        return

    ax.legend(loc="best", fontsize=9)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    print(f"Wrote aggregate plot to {outpath}")
    plt.close(fig)

def write_scalar_bar_plot(
    *,
    grouped_values: Dict[Tuple[Tuple[str, Any], ...], List[float]],
    outpath: Path,
    title: str,
    ylabel: str,
    baselines: Optional[List[Tuple[str, float]]] = None,
) -> None:
    if not grouped_values:
        return

    group_keys = list(grouped_values.keys())
    varying_fields = compute_varying_fields(group_keys)
    sort_field: Optional[str] = varying_fields[0] if len(varying_fields) == 1 else None

    records: List[Tuple[Optional[float], str, float, float]] = []

    for group_key in group_keys:
        vals = grouped_values.get(group_key, [])
        if not vals:
            continue
        arr = np.asarray(vals, dtype=float)
        mean = float(arr.mean())
        std = float(arr.std())

        values = dict(group_key)

        label = format_group_label(group_key, varying_fields)

        sort_key: Optional[float] = None
        if sort_field is not None:
            raw = values.get(sort_field)
            if raw is not None:
                sort_key = effective_numeric_for_sort(sort_field, raw, values)

        records.append((sort_key, label, mean, std))

    if not records:
        return

    # Order bars by the swept axis (when numeric), otherwise lexically.
    numeric = [r for r in records if r[0] is not None]
    non_numeric = [r for r in records if r[0] is None]
    numeric.sort(key=lambda r: (r[0], r[1]))
    non_numeric.sort(key=lambda r: r[1])
    ordered = numeric + non_numeric

    labels = [r[1] for r in ordered]
    means = [r[2] for r in ordered]
    stds = [r[3] for r in ordered]

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    x = np.arange(len(labels), dtype=float)

    # Distinct colors per bar.
    cmap = plt.get_cmap("tab20")
    colors = [cmap(i % cmap.N) for i in range(len(labels))]

    ax.bar(x, means, yerr=stds, capsize=6, color=colors)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.grid(True, axis="y", alpha=0.3)

    # Plot baselines if provided
    baseline_values = []
    if baselines:
        use_black = (len(baselines) == 1)
        for label, val in baselines:
            baseline_values.append(val)
            kwargs = {
                "linestyle": "--",
                "linewidth": 2,
                "label": label,
                "alpha": 0.7,
            }
            if use_black:
                kwargs["color"] = "black"
            else:
                # Explicit coloring for known baselines
                if "human" in label.lower():
                    kwargs["color"] = "black"
                elif "random" in label.lower():
                    kwargs["color"] = "red"
                # Else let it cycle or pick default
            
            ax.axhline(y=val, **kwargs)

    # Make small differences easier to see by starting slightly below
    # the minimum value touched by an error bar or baseline.
    min_vals = [(m - s) for m, s in zip(means, stds)]
    max_vals = [(m + s) for m, s in zip(means, stds)]
    
    if baseline_values:
        min_vals.extend(baseline_values)
        max_vals.extend(baseline_values)

    lower = min(min_vals)
    upper = max(max_vals)
    span = max(upper - lower, 1e-6)
    pad = 0.05 * span
    ax.set_ylim(bottom=lower - pad)

    if baselines:
        ax.legend(loc="best", fontsize=9)

    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
