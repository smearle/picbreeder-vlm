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
    "chat_history_turns": "Context length",
    "rand_select_prob": "$\epsilon$",
    "n_personality_traits": "Num. agents",
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
                    return "Random Baseline", None, None
            except (ValueError, TypeError):
                pass

    parts = []
    pretty_names = []
    pretty_values = []
    for name in varying_fields:
        if name in values:
            pretty_name = pretty_hyperparam_name(name)
            pretty_value = pretty_hyperparam_value(name, values[name], values)
            pretty_names.append(pretty_name)
            pretty_values.append(pretty_value)
            parts.append(f"{pretty_name} = {pretty_value}")
    if not parts:
        label = "default"
        hyper_name = ""
        hyper_value = ""
    else:
        label = " ".join(parts)
        hyper_name = ", ".join(pretty_names)
        hyper_value = ", ".join(pretty_values)
    return label, hyper_name, hyper_value

def load_human_baseline(
    metric: str,
    render_size: int,
    model: str,
    nounlist: Optional[str] = None,
    k: Optional[int] = None,
    strict: bool = True,
    negative_anchors: Optional[str] = None,
) -> Optional[Dict[int, float]]:
    """Load human baseline trajectory for the given metric."""
    baseline_dir = SCRIPT_ROOT / "human_baseline"
    if not baseline_dir.exists():
        msg = f"Human baseline directory not found: {baseline_dir}"
        if strict:
            raise FileNotFoundError(msg)
        print(msg)
        return None
        
    model_name = model.replace("/", "-")
    neg_suffix = f"_{Path(negative_anchors).stem}" if negative_anchors else ""
    
    key_extractor = None

    if metric == "novelty":
        filename = f"novelty_res{render_size}_{model_name}.json"
        key = "mean_pairwise_distance"
    elif metric == "noun":
        if not nounlist:
            msg = "Nounlist name must be provided to load noun similarity baseline."
            if strict:
                raise ValueError(msg)
            print(msg)
            return None
        filename = f"noun_similarity_res{render_size}_{model_name}_{nounlist}.json"
        key = "mean_max_similarity"
    elif metric == "noun_contrastive":
        if not nounlist:
            msg = "Nounlist name must be provided to load noun contrastive baseline."
            if strict:
                raise ValueError(msg)
            print(msg)
            return None
        filename = f"noun_similarity_res{render_size}_{model_name}_{nounlist}.json"
        key = f"mean_max_contrastive{neg_suffix}"
    elif metric == "visual_k_covering":
        if k is None:
            msg = "k must be provided for visual_k_covering baseline."
            if strict:
                raise ValueError(msg)
            print(msg)
            return None
        filename = f"novelty_res{render_size}_{model_name}.json"
        def key_extractor(row):
            radii = row.get("k_covering_radii")
            if isinstance(radii, dict):
                return radii.get(str(k))
            return None
        key = f"k_covering_radii[k={k}]"
    elif metric == "caption_diversity":
        filename = f"caption_metrics_res{render_size}_{model_name}.json"
        key = "mean_pairwise_distance"
    elif metric == "caption_k_covering":
        if k is None:
            msg = "k must be provided for caption_k_covering baseline."
            if strict:
                raise ValueError(msg)
            print(msg)
            return None
        filename = f"caption_metrics_res{render_size}_{model_name}.json"
        def key_extractor(row):
            radii = row.get("k_covering_radii")
            if isinstance(radii, dict):
                return radii.get(str(k))
            return None
        key = f"k_covering_radii[k={k}]"
    else:
        msg = f"Unknown metric for human baseline: {metric}"
        if strict:
            raise ValueError(msg)
        print(msg)
        return None
        
    path = baseline_dir / filename
    if not path.exists():
        msg = f"Human baseline file not found: {path}"
        if strict:
            raise FileNotFoundError(msg)
        print(msg)
        return None
        
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        trajectory = {}
        
        # Handle both list-of-dicts and dict-of-dicts
        if isinstance(data, dict):
            for k_idx, row in data.items():
                idx = row.get("index")
                if idx is None:
                    # Attempt to use key as index
                    try:
                        idx = int(k_idx)
                    except ValueError:
                        pass

                if key_extractor:
                    val = key_extractor(row)
                else:
                    val = row.get(key)
                
                if idx is not None and val is not None:
                    trajectory[int(idx)] = float(val)
        else:
            for row in data:
                idx = row.get("index")
                if key_extractor:
                    val = key_extractor(row)
                else:
                    val = row.get(key)
                
                if idx is not None and val is not None:
                    trajectory[int(idx)] = float(val)
        
        if not trajectory and strict:
            raise ValueError(f"No valid data found for key '{key}' in {path}")

        return trajectory
    except Exception as e:
        msg = f"Failed to load human baseline from {path}: {e}"
        if strict:
            raise RuntimeError(msg) from e
        print(msg)
        return None


def write_aggregate_plot(
    *,
    grouped_runs: Dict[Tuple[Tuple[str, Any], ...], List[Dict[int, float]]],
    outpath: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    baselines: Optional[List[Tuple[str, Dict[int, float]]]] = None,
    ylim: Optional[Tuple[float, float]] = None,
) -> None:
    if not grouped_runs:
        return

    group_keys = list(grouped_runs.keys())
    varying_fields = compute_varying_fields(group_keys)

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(ylim)
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
        sem = values.std(axis=0) / np.sqrt(values.shape[0])

        label, _, _ = format_group_label(group_key, varying_fields)
        (line,) = ax.plot(indices, mean, linewidth=2, label=label)
        ax.fill_between(indices, mean - sem, mean + sem, alpha=0.2, color=line.get_color())
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
        std = float(arr.std()) / np.sqrt(len(arr))

        values = dict(group_key)

        label, hyper_name, hyper_value = format_group_label(group_key, varying_fields)

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

    labels = [r[0] for r in ordered]
    means = [r[2] for r in ordered]
    stds = [r[3] for r in ordered]

    fig, ax = plt.subplots(1, 1, figsize=(4, 6))
    x = np.arange(len(labels), dtype=float)

    # Distinct colors per bar.
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i % cmap.N) for i in range(len(labels))]

    ax.bar(x, means, yerr=stds, capsize=6, color=colors)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    # ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_xticklabels(labels)
    ax.set_xlabel(" | ".join([pretty_hyperparam_name(f) for f in varying_fields]))
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
        # put legend in top right corner
        ax.legend(loc="upper right", fontsize=9)

    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def write_aggregate_bar_chart(
    grouped_runs: Dict[Tuple[Tuple[str, Any], ...], List[Dict[int, float]]],
    outpath: Path,
    title: str,
    ylabel: str,
    baselines: Optional[List[Tuple[str, Dict[int, float]]]] = None,
    ylim: Optional[Tuple[float, float]] = None,
) -> None:
    # Convert grouped_runs to grouped_values (taking the final value)
    grouped_values = {}
    for key, runs in grouped_runs.items():
        vals = []
        for run in runs:
            if run:
                # Assuming max key is the final step
                vals.append(run[max(run.keys())])
        if vals:
            grouped_values[key] = vals
            
    # Convert baselines
    scalar_baselines = []
    if baselines:
        for label, traj in baselines:
            if traj:
                scalar_baselines.append((label, traj[max(traj.keys())]))

    write_scalar_bar_plot(
        grouped_values=grouped_values,
        outpath=outpath,
        title=title,
        ylabel=ylabel,
        baselines=scalar_baselines,
    )



def write_combined_plot_and_bar(
    grouped_runs: Dict[Tuple[Tuple[str, Any], ...], List[Dict[int, float]]],
    grouped_values: Dict[Tuple[Tuple[str, Any], ...], List[float]],
    outpath: Path,
    title: str,
    xlabel: str,
    ylabel_line: str,
    ylabel_bar: str,
    baselines_line: Optional[List[Tuple[str, Dict[int, float]]]] = None,
    baselines_bar: Optional[List[Tuple[str, float]]] = None,
    ylim: Optional[Tuple[float, float]] = None,
) -> None:
    if not grouped_runs and not grouped_values:
        return

    # Create figure with 2 subplots, width ratio 2:1
    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[2, 1])
    ax_line = fig.add_subplot(gs[0])
    ax_bar = fig.add_subplot(gs[1])

    # --- Plot Line Chart on ax_line ---
    group_keys = list(grouped_runs.keys())
    varying_fields = compute_varying_fields(group_keys)

    sorted_items = _get_sorted_group_keys(group_keys, varying_fields)

    fig.suptitle(title)
    ax_line.set_title("Score over time")
    ax_line.set_xlabel(xlabel)
    ax_line.set_ylabel(ylabel_line)
    if ylim is not None:
        ax_line.set_ylim(ylim)
    ax_line.grid(True, which="major", alpha=0.3)

    plotted_line = 0
    max_x = 0
    
    # distinct colors
    cmap = plt.get_cmap("tab10")
    # We want consistent colors between line and bar if possible, 
    # based on the sorted order.
    # Note: If grouped_runs and grouped_values have the same keys (which they should),
    # then indices match.
    
    colors = [cmap(i % cmap.N) for i in range(len(sorted_items))]

    for i, (group_key, label, _, _) in enumerate(sorted_items):
        runs = grouped_runs[group_key]
        if not runs:
            continue
        
        index_sets = [set(run.keys()) for run in runs]
        common = set.intersection(*index_sets) if len(index_sets) > 1 else index_sets[0]
        if not common:
            continue
        indices = sorted(common)
        if indices:
            max_x = max(max_x, indices[-1])
        values = np.array([[run[i] for i in indices] for run in runs], dtype=float)
        mean = values.mean(axis=0)
        sem = values.std(axis=0) / np.sqrt(values.shape[0])

        color = colors[i]
        (line,) = ax_line.plot(indices, mean, linewidth=2, label=label, color=color)
        ax_line.fill_between(indices, mean - sem, mean + sem, alpha=0.2, color=color)
        plotted_line += 1

    if baselines_line and max_x > 0:
        use_black = (len(baselines_line) == 1)
        for label, trajectory in baselines_line:
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
                    if "human" in label.lower():
                        kwargs["color"] = "black"
                    elif "random" in label.lower():
                        kwargs["color"] = "red"
                ax_line.plot(indices, values, **kwargs)
                plotted_line += 1
    
    if plotted_line > 0:
        ax_line.legend(loc="best", fontsize=9)

    # --- Plot Bar Chart on ax_bar ---
    group_keys_bar = list(grouped_values.keys())
    varying_fields_bar = compute_varying_fields(group_keys_bar)
    sorted_items_bar = _get_sorted_group_keys(group_keys_bar, varying_fields_bar)

    means = []
    stds = []
    labels = []
    x_tick_labels = []
    x_axis_label = ""
    
    for group_key, label, h_name, h_val in sorted_items_bar:
        vals = grouped_values.get(group_key, [])
        if not vals:
            # Handle empty values to keep alignment? 
            # If we skip, colors might shift if we rely on index.
            # But line plot skipped empty runs too.
            # Ideally we assume consistency.
            continue
            
        arr = np.asarray(vals, dtype=float)
        mean = float(arr.mean())
        std = float(arr.std()) / np.sqrt(len(arr))

        means.append(mean)
        stds.append(std)
        labels.append(label)
        x_tick_labels.append(h_val)
        x_axis_label = h_name

    if means:
        x = np.arange(len(labels), dtype=float)
        
        # Let's filter colors to match plotted items
        final_colors = []
        for i, (group_key, _, _, _) in enumerate(sorted_items_bar):
             if grouped_values.get(group_key):
                 final_colors.append(colors[i])

        ax_bar.bar(x, means, yerr=stds, capsize=6, color=final_colors)
        ax_bar.set_title("Final score")
        # ax_bar.set_ylabel(ylabel_bar)
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(x_tick_labels)
        ax_bar.set_xlabel(x_axis_label)
        ax_bar.grid(True, axis="y", alpha=0.3)

        baseline_values = []
        if baselines_bar:
            use_black = (len(baselines_bar) == 1)
            for label, val in baselines_bar:
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
                    if "human" in label.lower():
                        kwargs["color"] = "black"
                    elif "random" in label.lower():
                        kwargs["color"] = "red"
                ax_bar.axhline(y=val, **kwargs)

        min_vals = [(m - s) for m, s in zip(means, stds)]
        max_vals = [(m + s) for m, s in zip(means, stds)]
        if baseline_values:
            min_vals.extend(baseline_values)
            max_vals.extend(baseline_values)
        
        if min_vals and max_vals:
            lower = min(min_vals)
            upper = max(max_vals)
            span = max(upper - lower, 1e-6)
            pad = 0.05 * span
            ax_bar.set_ylim(bottom=lower - pad)

    if baselines_bar:
        ax_bar.legend(loc="best", fontsize=9)

    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    print(f"Wrote combined plot to {outpath}")
    plt.close(fig)

def _get_sorted_group_keys(
    group_keys: List[Tuple[Tuple[str, Any], ...]],
    varying_fields: List[str]
) -> List[Tuple[Tuple[Tuple[str, Any], ...], str, str, str]]:
    """Return sorted list of (group_key, label, hyper_name, hyper_value)."""
    
    sort_field: Optional[str] = varying_fields[0] if len(varying_fields) == 1 else None
    
    records = []
    
    for group_key in group_keys:
        values = dict(group_key)
        label, hyper_name, hyper_value = format_group_label(group_key, varying_fields)
        
        sort_key: Optional[float] = None
        if sort_field is not None:
            raw = values.get(sort_field)
            if raw is not None:
                sort_key = effective_numeric_for_sort(sort_field, raw, values)
                
        records.append((sort_key, group_key, label, hyper_name, hyper_value))
        
    numeric = [r for r in records if r[0] is not None]
    non_numeric = [r for r in records if r[0] is None]
    numeric.sort(key=lambda r: (r[0], r[2]))
    non_numeric.sort(key=lambda r: r[2])
    ordered = numeric + non_numeric
    
    return [(r[1], r[2], r[3], r[4]) for r in ordered]
