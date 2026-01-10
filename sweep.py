#!/usr/bin/env python3
"""Launch collaborative_multi_agent sweeps locally or via Submitit."""

from __future__ import annotations

import os
import re
import sys
import json
from dataclasses import dataclass, field, replace, fields, asdict
from itertools import product
from pathlib import Path
from typing import Dict, List, Sequence, Optional, Tuple, Any, Union

import hydra
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_original_cwd
from hydra.conf import HelpConf, HydraConf
import omegaconf
import submitit  # Do not remove this.

from collaborative_multi_agent import run as run_collaborative
from config import PicbreederConfig, ensure_valid_config


SCRIPT_ROOT = Path(__file__).resolve().parent


_TAG_SANITIZE_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")


def _sanitize_filename_tag(tag: str) -> str:
    cleaned = _TAG_SANITIZE_PATTERN.sub("_", str(tag)).strip("_")
    return cleaned or "sweep"


class CollaborativeRun:
    """Submitit-compatible callable that executes a configured run."""

    def __init__(self, sweep_cfg: SweepConfig, run_cfg: PicbreederConfig, mode='train'):
        self.run_cfg = run_cfg

    def __call__(self) -> int:
        print(_format_run_prefix(self.run_cfg, "[submitit]"))
        run_collaborative(self.run_cfg)
        return 0

    def checkpoint(self) -> "submitit.helpers.DelayedSubmission":
        refreshed = replace(self.cfg)
        return submitit.helpers.DelayedSubmission(self.__class__(refreshed))


def _execute_job(job: CollaborativeRun) -> int:
    return job()


@dataclass
class SweepConfig(PicbreederConfig):
    seed: List[int] = field(default_factory=lambda: [0])  # Random seeds swept over collaborative runs
    chat_history_turns: List[int] = field(default_factory=lambda: [1])  # Chat history lengths to evaluate
    rand_select_prob: List[float] = field(default_factory=lambda: [0.0])  # Probability of random parent selection
    temperature: List[Union[int, float, str]] = field(default_factory=lambda: [1.0])  # Sampling temperature values to evaluate
    goal: List[str] = field(default_factory=lambda: [  # Goals to sweep over
        "familiar_objects",
        # "fun",
        # "lizards", 
        # "fish", 
        # "skulls", 
        # "butterflies"
    ])
    model: List[str] = field(default_factory=lambda: [  # VLM models to evaluate
        # "gemini-3-pro-preview",
        "gemini-2.5-pro",
        # "gemini-2.5-flash",
        # "gemini-2.5-flash-lite",
    ])
    sweep_name: str = "sweep"  # Base directory for experiment outputs
    log_dir: str = "sweep_logs"
    submitit_log_dir: str = "submitit_logs"
    slurm: bool = True  # Enable SLURM submission via Submitit
    partition: str = "cpu"  # SLURM partition name
    # account: Optional[str] = None  # Optional SLURM account override
    account: Optional[str] = "pr_174_tandon_advanced"  # Optional SLURM account override
    timeout_hours: int = 24  # Wall-time limit in hours
    mem_gb: int = 30  # Memory requested per task (GB)
    num_proc: int = 10  # Number of parallel processes per task
    evaluate: bool = False  # If true, run evaluation instead of training
    visualize: bool = False  # If true, run phylogeny visualization instead of training
    plot: bool = False  # If true, run plotting/analysis scripts instead of training
    cross_eval: bool = False  # If true, summarize embedding metrics from the configured runs
    archive_limit: Optional[int] = None  # Limit the number of archive images passed to analysis scripts
    hydra: HydraConf = field(
        default_factory=lambda: HydraConf(
            help=HelpConf(
                app_name="sweep",
                header=(
                    "Submitit/Hydra sweep launcher for collaborative_multi_agent.\n"
                    "\n"
                    "Common overrides:\n"
                    "  seeds                 List of random seeds to evaluate.\n"
                    "  chat_history_turns    Values swept for chat context length (-1 keeps all turns).\n"
                    "  sweep_name            Named sweep preset (also used as output directory name).\n"
                    "  slurm                 true to submit jobs to a SLURM cluster.\n"
                    "  partition / account   SLURM resource parameters appended to submissions.\n"
                    "  cross_eval            true to summarize embedding metrics for the configured runs.\n"
                ),
                footer="Hydra overrides (e.g. +option=value) are supported. Use --cfg=job to inspect merged configs.",
            )
        )
    )


@dataclass
class SweepBasePreset(SweepConfig):
    """No-op preset: preserves whatever list-valued axes you pass explicitly."""


@dataclass
class ChatHistoryTurnsSweep(SweepConfig):
    chat_history_turns: List[int] = field(default_factory=lambda: [-1, 10, 2, 1, 0])
    temperature: List[Union[int, float, str]] = field(default_factory=lambda: [1.0])
    rand_select_prob: List[float] = field(default_factory=lambda: [0.0])
    goal: List[str] = field(default_factory=lambda: ["familiar_objects"])
    model: List[str] = field(default_factory=lambda: ["gemini-2.5-pro"])
    seed: List[int] = field(default_factory=lambda: [3, 4, 5])
    num_agents: int = 200


@dataclass
class TemperatureSweep(SweepConfig):
    temperature: List[Union[int, float, str]] = field(default_factory=lambda: [0.0, 1.0, 2.0, "random"])
    chat_history_turns: List[int] = field(default_factory=lambda: [1])
    rand_select_prob: List[float] = field(default_factory=lambda: [0.0])
    goal: List[str] = field(default_factory=lambda: ["familiar_objects"])
    model: List[str] = field(default_factory=lambda: ["gemini-2.5-pro"])
    seed: List[int] = field(default_factory=lambda: [3, 4, 5])
    num_agents: int = 200


@dataclass
class RandSelectProbSweep(SweepConfig):
    rand_select_prob: List[float] = field(default_factory=lambda: [0.0, 0.25, 0.5, 0.75])
    chat_history_turns: List[int] = field(default_factory=lambda: [1])
    temperature: List[Union[int, float, str]] = field(default_factory=lambda: [1.0])
    goal: List[str] = field(default_factory=lambda: ["familiar_objects"])
    model: List[str] = field(default_factory=lambda: ["gemini-2.5-pro"])
    # These were run with a small bug where agents could publish twice during a generation 
    # (even with fixed session length), and random selections could happen at the last generation,
    # leading to random publications.
    # seed: List[int] = field(default_factory=lambda: [0, 1, 2])
    seed: List[int] = field(default_factory=lambda: [3, 4, 5])
    num_agents: int = 200

@dataclass
class ModelSweep(SweepConfig):
    model: List[str] = field(default_factory=lambda: ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-3-pro-preview"])
    chat_history_turns: List[int] = field(default_factory=lambda: [1])
    seed: List[int] = field(default_factory=lambda: [3, 4, 5])
    num_agents: int = 500


_NAMED_SWEEPS: Dict[str, type[SweepConfig]] = {
    "sweep": SweepBasePreset,
    "chat_history_turns": ChatHistoryTurnsSweep,
    "temperature": TemperatureSweep,
    "rand_select_prob": RandSelectProbSweep,
    "model": ModelSweep,
}


def _extract_list_axes_from_preset(preset: SweepConfig) -> Dict[str, Any]:
    """Return list-valued overrides from a preset.

    We only apply list-valued fields so we don't clobber non-sweep runtime
    settings (e.g. slurm params, log dirs).
    """

    payload = asdict(preset)
    axes: Dict[str, Any] = {}
    for key, value in payload.items():
        if key == "hydra":
            continue
        if isinstance(value, (list, tuple)):
            axes[key] = list(value)
    return axes


def _apply_named_sweep(cfg: SweepConfig) -> SweepConfig:
    """Apply list-valued sweep axes based on cfg.sweep_name.

    Named sweeps are *defaults* applied after Hydra composition.
    If a key was explicitly overridden on the CLI, we leave it unchanged.
    """

    sweep_name = str(getattr(cfg, "sweep_name", "sweep"))
    preset_cls = _NAMED_SWEEPS.get(sweep_name)
    if preset_cls is None:
        known = ", ".join(sorted(_NAMED_SWEEPS.keys()))
        raise ValueError(f"Unknown sweep_name={sweep_name!r}. Known: {known}")

    preset = preset_cls()
    updates = _extract_list_axes_from_preset(preset)
    if not updates:
        return cfg

    overridden_root_keys: set[str] = set()
    try:
        overrides = HydraConfig.get().overrides.task
    except Exception:
        overrides = []

    for override in overrides:
        if "=" not in override:
            continue
        key = override.split("=", 1)[0].lstrip("+")
        overridden_root_keys.add(key.split(".", 1)[0])

    filtered_updates = {
        key: value
        for key, value in updates.items()
        if key not in overridden_root_keys
    }
    if not filtered_updates:
        return cfg

    # Merge into the (DictConfig-backed) cfg so list values become ListConfig.
    return omegaconf.OmegaConf.merge(cfg, omegaconf.OmegaConf.create(filtered_updates))


def _ensure_absolute(path: Path, base: Path) -> Path:
    expanded = Path(path).expanduser()
    if expanded.is_absolute():
        return expanded.resolve()
    return (base / expanded).resolve()


def _call_hydra_wrapped_main(main_func, cfg_obj, **kwargs) -> None:
    """Call a Hydra-decorated main(cfg) without spawning a subprocess.

    Hydra's @hydra.main wraps the original function; we call the underlying
    implementation directly via __wrapped__.
    """

    wrapped = getattr(main_func, "__wrapped__", None)
    if wrapped is None:
        raise RuntimeError(
            f"Expected {main_func!r} to be Hydra-decorated and expose __wrapped__."
        )
    wrapped(cfg_obj, **kwargs)


_AGGREGATE_EXCLUDE_FIELDS: Tuple[str, ...] = (
    "hydra",
    "seed",
    "experiment_dir",
    "resume",
    "resume_agent_id",
)


# Centralized display-name mapping for cross-eval/visualization labels.
# Keep this small and extend as needed.
HYPERPARAM_PRETTY_LABELS: Dict[str, str] = {
    "chat_history_turns": "turns in context",
    "rand_select_prob": "rand. selection prob.",
}


def _pretty_hyperparam_name(name: str) -> str:
    return HYPERPARAM_PRETTY_LABELS.get(name, name)


def _pretty_hyperparam_value(name: str, value: Any, values: Dict[str, Any]) -> str:
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


def _effective_numeric_for_sort(name: str, value: Any, values: Dict[str, Any]) -> Optional[float]:
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


def _normalize_group_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        # Keep floats stable-ish in labels/keys.
        return float(f"{value:.6g}")
    return value


def _group_key_for_aggregate(cfg: PicbreederConfig) -> Tuple[Tuple[str, Any], ...]:
    items: List[Tuple[str, Any]] = []
    for field_def in fields(PicbreederConfig):
        name = field_def.name
        if name in _AGGREGATE_EXCLUDE_FIELDS:
            continue
        items.append((name, _normalize_group_value(getattr(cfg, name))))
    return tuple(items)


def _compute_varying_fields(group_keys: Sequence[Tuple[Tuple[str, Any], ...]]) -> List[str]:
    values_by_field: Dict[str, set] = {}
    for key in group_keys:
        for name, value in key:
            values_by_field.setdefault(name, set()).add(value)
    varying = [name for name, values in values_by_field.items() if len(values) > 1]
    # Deterministic ordering.
    varying.sort()
    return varying


def _format_group_label(group_key: Tuple[Tuple[str, Any], ...], varying_fields: Sequence[str]) -> str:
    values = dict(group_key)
    parts = []
    for name in varying_fields:
        if name in values:
            pretty_name = _pretty_hyperparam_name(name)
            pretty_value = _pretty_hyperparam_value(name, values[name], values)
            parts.append(f"{pretty_name} = {pretty_value}")
    if not parts:
        return "default"
    return " ".join(parts)


def _load_trajectory_metric(path: Path, metric_key: str) -> Dict[int, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    result: Dict[int, float] = {}
    for row in data:
        idx = row.get("index")
        value = row.get(metric_key)
        if idx is None or value is None:
            continue
        try:
            idx_int = int(idx)
            value_float = float(value)
        except (TypeError, ValueError):
            continue
        result[idx_int] = value_float
    return result


def _load_noun_similarity_scalar(exp_dir: Path) -> Optional[float]:
    """Load a single noun similarity scalar for an experiment.

    Prefers noun_similarity_metrics.json (written by compute_noun_similarity.py).
    Falls back to the final value in noun_similarity_over_time.json.
    """

    metrics_path = exp_dir / "noun_similarity_metrics.json"
    if metrics_path.exists():
        try:
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            value = payload.get("mean_max_similarity")
            if value is None:
                return None
            return float(value)
        except Exception:
            return None

    trajectory_path = exp_dir / "noun_similarity_over_time.json"
    if not trajectory_path.exists():
        return None
    try:
        traj = json.loads(trajectory_path.read_text(encoding="utf-8"))
        if not isinstance(traj, list) or not traj:
            return None
        last = traj[-1]
        if not isinstance(last, dict):
            return None
        value = last.get("mean_max_similarity")
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _load_embedding_mean_pairwise_distance_scalar(exp_dir: Path) -> Optional[float]:
    """Load mean pairwise distance scalar for an experiment.

    Prefers embedding_metrics.json (produced by embed_and_visualize.py) using either:
      - mean_pairwise_distance.value (legacy schema)
      - pairwise_distances.mean (newer schema)

    Falls back to the final value in embedding_mean_pairwise_distance_over_time.json.
    """

    metrics_path = exp_dir / "embedding_metrics.json"
    if metrics_path.exists():
        try:
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            mpd = payload.get("mean_pairwise_distance")
            if isinstance(mpd, dict):
                value = mpd.get("value")
                if value is not None:
                    return float(value)

            pairwise = payload.get("pairwise_distances")
            if isinstance(pairwise, dict):
                value = pairwise.get("mean")
                if value is not None:
                    return float(value)
        except Exception:
            return None

    traj_path = exp_dir / "embedding_mean_pairwise_distance_over_time.json"
    if not traj_path.exists():
        return None
    try:
        traj = json.loads(traj_path.read_text(encoding="utf-8"))
        if not isinstance(traj, list) or not traj:
            return None
        last = traj[-1]
        if not isinstance(last, dict):
            return None
        value = last.get("mean_pairwise_distance")
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _write_scalar_bar_plot(
    *,
    grouped_values: Dict[Tuple[Tuple[str, Any], ...], List[float]],
    outpath: Path,
    title: str,
    ylabel: str,
) -> None:
    import numpy as np

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not grouped_values:
        return

    group_keys = list(grouped_values.keys())
    varying_fields = _compute_varying_fields(group_keys)
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

        label = _format_group_label(group_key, varying_fields)

        sort_key: Optional[float] = None
        if sort_field is not None:
            raw = values.get(sort_field)
            if raw is not None:
                sort_key = _effective_numeric_for_sort(sort_field, raw, values)

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

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
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

    # Make small differences easier to see by starting slightly below
    # the minimum value touched by an error bar.
    lower = min((m - s) for m, s in zip(means, stds))
    upper = max((m + s) for m, s in zip(means, stds))
    span = max(upper - lower, 1e-6)
    pad = 0.05 * span
    ax.set_ylim(bottom=lower - pad)

    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def _write_aggregate_plot(
    *,
    grouped_runs: Dict[Tuple[Tuple[str, Any], ...], List[Dict[int, float]]],
    outpath: Path,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    import numpy as np

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not grouped_runs:
        return

    group_keys = list(grouped_runs.keys())
    varying_fields = _compute_varying_fields(group_keys)

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, which="major", alpha=0.3)

    plotted = 0

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
        values = np.array([[run[i] for i in indices] for run in runs], dtype=float)
        mean = values.mean(axis=0)
        std = values.std(axis=0)

        label = _format_group_label(group_key, varying_fields)
        (line,) = ax.plot(indices, mean, linewidth=2, label=label)
        ax.fill_between(indices, mean - std, mean + std, alpha=0.2, color=line.get_color())
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        return

    ax.legend(loc="best", fontsize=9)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def _plot_seed_aggregates(
    *,
    run_configs: Sequence[PicbreederConfig],
    output_dir: Path,
    filename_tag: str,
) -> None:
    novelty_grouped: Dict[Tuple[Tuple[str, Any], ...], List[Dict[int, float]]] = {}
    noun_grouped: Dict[Tuple[Tuple[str, Any], ...], List[Dict[int, float]]] = {}
    noun_scalar_grouped: Dict[Tuple[Tuple[str, Any], ...], List[float]] = {}
    mpd_scalar_grouped: Dict[Tuple[Tuple[str, Any], ...], List[float]] = {}

    for run_cfg in run_configs:
        group_key = _group_key_for_aggregate(run_cfg)
        exp_dir = Path(run_cfg.experiment_dir)

        novelty_path = exp_dir / "embedding_mean_pairwise_distance_over_time.json"
        if novelty_path.exists():
            novelty = _load_trajectory_metric(novelty_path, "mean_pairwise_distance")
            novelty_grouped.setdefault(group_key, []).append(novelty)

        noun_path = exp_dir / "noun_similarity_over_time.json"
        if noun_path.exists():
            noun = _load_trajectory_metric(noun_path, "mean_max_similarity")
            noun_grouped.setdefault(group_key, []).append(noun)

        noun_scalar = _load_noun_similarity_scalar(exp_dir)
        if noun_scalar is not None:
            noun_scalar_grouped.setdefault(group_key, []).append(noun_scalar)

        mpd_scalar = _load_embedding_mean_pairwise_distance_scalar(exp_dir)
        if mpd_scalar is not None:
            mpd_scalar_grouped.setdefault(group_key, []).append(mpd_scalar)

    agg_plot_distance_path = output_dir / f"aggregate_embedding_mean_pairwise_distance_over_time_{filename_tag}.png"
    _write_aggregate_plot(
        grouped_runs=novelty_grouped,
        outpath=agg_plot_distance_path,
        title="Embedding diversity over time (mean±std across seeds)",
        xlabel="Archive insertion order",
        ylabel="Mean pairwise distance",
    )
    agg_plot_noun_path = output_dir / f"aggregate_noun_similarity_over_time_{filename_tag}.png"
    _write_aggregate_plot(
        grouped_runs=noun_grouped,
        outpath=agg_plot_noun_path,
        title="Noun similarity over time (mean±std across seeds)",
        xlabel="Archive insertion order",
        ylabel="Mean max cosine similarity",
    )

    _write_scalar_bar_plot(
        grouped_values=noun_scalar_grouped,
        outpath=output_dir / f"aggregate_noun_similarity_mean_bar_{filename_tag}.png",
        title="Mean max noun similarity (mean±std across seeds)",
        ylabel="Mean of per-noun max cosine similarity",
    )

    _write_scalar_bar_plot(
        grouped_values=mpd_scalar_grouped,
        outpath=output_dir / f"aggregate_mean_pairwise_distance_mean_bar_{filename_tag}.png",
        title="Mean pairwise distance (mean±std across seeds)",
        ylabel="Mean pairwise distance (euclidean)",
    )


def _expand_sweep_configs(cfg: SweepConfig) -> List[SweepConfig]:
    """Produce one config per cartesian product of list-valued fields."""
    pic_fields = {field_def.name for field_def in fields(PicbreederConfig) if field_def.name != "hydra"}
    sweep_axes: List[Tuple[str, Sequence[Any]]] = []

    for field_def in fields(SweepConfig):
        name = field_def.name
        if name == "hydra" or name not in pic_fields:
            continue
        value = getattr(cfg, name)

        if isinstance(value, (omegaconf.listconfig.ListConfig, list, tuple)):
            if len(value) == 0:
                sweep_axes.append((name, value))
            else:
                sweep_axes.append((name, value))

    if not sweep_axes:
        return [cfg]
    if any(len(values) == 0 for _, values in sweep_axes):
        return []

    configs: List[SweepConfig] = []
    cfg = {k: v for k, v in cfg.items() if hasattr(PicbreederConfig, k)}
    for combo in product(*(values for _, values in sweep_axes)):
        updates = {name: value for (name, _), value in zip(sweep_axes, combo)}
        configs.append(replace(SweepConfig(**cfg), **updates))
    return configs


def _build_run_config(cfg: SweepConfig, original_cwd: Path) -> PicbreederConfig:
    """Create a per-run config, letting collaborative_multi_agent name directories."""
    base_kwargs = {field_def.name: getattr(cfg, field_def.name) for field_def in fields(PicbreederConfig) if field_def.name != "hydra"}
    base_cfg = PicbreederConfig(**base_kwargs)
    per_run_cfg = replace(base_cfg, experiment_dir=None, resume=False)
    validated_cfg = ensure_valid_config(per_run_cfg, original_cwd=original_cwd)
    exp_name = Path(validated_cfg.experiment_dir).name
    exp_dir = _ensure_absolute(os.path.join(cfg.log_dir, cfg.sweep_name), original_cwd) / exp_name
    validated_cfg = replace(validated_cfg, experiment_dir=exp_dir)
    resume = exp_dir.exists()
    if resume != validated_cfg.resume:
        validated_cfg = replace(validated_cfg, resume=resume)
    return validated_cfg


def _format_run_prefix(cfg: PicbreederConfig, prefix: str) -> str:
    return (
        f"{prefix} seed={cfg.seed} chat={cfg.chat_history_turns} "
        f"goal={cfg.goal} scheme={cfg.scheme} resume={cfg.resume} -> {cfg.experiment_dir}"
    )

def launch_locally(configs: Sequence[PicbreederConfig]) -> None:
    for run_cfg in configs:
        print(_format_run_prefix(run_cfg, "[local]"))
        run_collaborative(run_cfg)

def launch_slurm(cfg: SweepConfig, log_dir: Path, configs: Sequence[PicbreederConfig]) -> None:
    try:
        import submitit
    except ImportError as exc:
        raise RuntimeError("submitit is required when using --slurm") from exc

    executor = submitit.AutoExecutor(folder=cfg.submitit_log_dir)
    executor.update_parameters(
        timeout_min=cfg.timeout_hours * 60,
        mem_gb=cfg.mem_gb,
        cpus_per_task=cfg.num_proc,
        # slurm_partition=cfg.partition,
        slurm_account=cfg.account,
        name="picbreeder-vlm",
    )
    jobs = [CollaborativeRun(cfg, run_cfg) for run_cfg in configs]
    futures = executor.map_array(_execute_job, jobs)
    for run_cfg, future in zip(configs, futures):
        print(_format_run_prefix(run_cfg, "[slurm]") + f" submitted as job {future.job_id}")


cs = ConfigStore.instance()
cs.store(name="sweep_base", node=SweepConfig)


@hydra.main(version_base=None, config_path=None, config_name="sweep_base")
def main(cfg: SweepConfig) -> None:
    original_cwd = Path(get_original_cwd())
    log_dir = _ensure_absolute(cfg.log_dir, original_cwd)
    log_dir.mkdir(parents=True, exist_ok=True)

    cfg = _apply_named_sweep(cfg)

    base_configs = _expand_sweep_configs(cfg)
    run_configs = [_build_run_config(run_cfg, original_cwd) for run_cfg in base_configs]

    if not run_configs:
        print("No runs scheduled (empty sweep axes).")
        return

    experiment_prefix = Path(run_configs[0].experiment_dir).parent
    cross_eval_dir = Path(os.path.join("cross_eval", cfg.sweep_name))
    filename_tag = _sanitize_filename_tag(cfg.sweep_name)

    if cfg.cross_eval:
        from render_embedding_metrics_table import DEFAULT_METRICS_FILENAME, render_tables

        cross_eval_dir.mkdir(parents=True, exist_ok=True)

        _plot_seed_aggregates(
            run_configs=run_configs,
            output_dir=cross_eval_dir,
            filename_tag=filename_tag,
        )

        metrics_name = DEFAULT_METRICS_FILENAME
        existing_dirs = []
        existing_configs = []
        missing_metrics = []

        for run_cfg in run_configs:
            metrics_path = Path(run_cfg.experiment_dir) / metrics_name
            if metrics_path.exists():
                existing_dirs.append(Path(run_cfg.experiment_dir))
                existing_configs.append(run_cfg)
            else:
                missing_metrics.append(
                    (run_cfg.seed, run_cfg.chat_history_turns, run_cfg.goal, metrics_path)
                )

        if not existing_dirs:
            print("Cross evaluation aborted: no embedding metrics found for the configured runs.")
            if missing_metrics:
                print("Missing metrics files:")
                for seed, chat_turns, goal, missing_path in missing_metrics:
                    print(f"  seed={seed} chat={chat_turns} goal={goal} -> {missing_path}")
            return

        # Build group labels from config fields (excluding seed) for proper averaging
        group_keys = [_group_key_for_aggregate(run_cfg) for run_cfg in existing_configs]
        varying_fields = _compute_varying_fields(group_keys)
        group_labels: Dict[str, str] = {}
        for run_cfg, group_key in zip(existing_configs, group_keys):
            exp_name = Path(run_cfg.experiment_dir).name
            label = _format_group_label(group_key, varying_fields)
            group_labels[exp_name] = label

        try:
            per_table, agg_table, per_csv, agg_csv, latex_path = render_tables(
                experiment_prefix,
                output_dir=cross_eval_dir,
                metrics_name=metrics_name,
                experiment_dirs=existing_dirs,
                filename_tag=filename_tag,
                group_labels=group_labels,
            )
        except ValueError as exc:
            print(f"Cross evaluation failed: {exc}", file=sys.stderr)
            return

        print("Per-experiment metrics:\n")
        print(per_table)
        print(f"\nWrote CSV: {per_csv}")

        print("\nAggregated across seeds:\n")
        print(agg_table)
        print(f"\nWrote CSV: {agg_csv}")
        print(f"Wrote LaTeX: {latex_path}")

        if missing_metrics:
            print("\nMissing metrics files:")
            for seed, chat_turns, goal, missing_path in missing_metrics:
                print(f"  seed={seed} chat={chat_turns} goal={goal} -> {missing_path}")

        return

    if cfg.plot:
        # Match the launch_locally pattern: instantiate the relevant Config objects
        # and call their main functions directly (no subprocess).
        from dataclasses import fields as dataclass_fields

        previous_cwd = Path.cwd()
        os.chdir(original_cwd)
        try:
            import torch

            from plot_novelty_over_time import (
                PairwiseDistanceConfig,
                main as novelty_main,
                prepare_openclip_components as prepare_novelty_clip,
            )
            from compute_noun_similarity import (
                NounSimilarityConfig,
                main as noun_main,
                prepare_openclip_components as prepare_noun_clip,
                prepare_noun_text_embeddings,
            )

            # Initialize heavy CLIP components once for the entire sweep.
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            print(f"[local-plot] Using device: {device}")

            base_kwargs0 = {
                field_def.name: getattr(run_configs[0], field_def.name)
                for field_def in dataclass_fields(PicbreederConfig)
                if field_def.name != "hydra"
            }
            novelty_cfg0 = PairwiseDistanceConfig(**base_kwargs0)
            noun_cfg0 = NounSimilarityConfig(**base_kwargs0)

            novelty_model, novelty_preprocess = prepare_novelty_clip(novelty_cfg0, device)
            noun_model, noun_preprocess, noun_tokenizer = prepare_noun_clip(noun_cfg0, device)
            nouns_list, prompts_list, noun_text_embeddings = prepare_noun_text_embeddings(
                noun_cfg0,
                original_cwd=original_cwd,
                device=device,
                model=noun_model,
                tokenizer=noun_tokenizer,
            )

            for run_cfg in run_configs:
                base_kwargs = {
                    field_def.name: getattr(run_cfg, field_def.name)
                    for field_def in dataclass_fields(PicbreederConfig)
                    if field_def.name != "hydra"
                }

                novelty_cfg = PairwiseDistanceConfig(**base_kwargs)
                novelty_cfg = replace(novelty_cfg, archive_limit=cfg.archive_limit)
                desc = _format_run_prefix(run_cfg, "[local-plot]")
                extra = (
                    f" archive_limit={cfg.archive_limit}"
                    if cfg.archive_limit is not None
                    else ""
                )
                print(f"{desc} -> plot_novelty_over_time{extra}")
                _call_hydra_wrapped_main(
                    novelty_main,
                    novelty_cfg,
                    model=novelty_model,
                    preprocess=novelty_preprocess,
                )

                noun_cfg = NounSimilarityConfig(**base_kwargs)
                noun_cfg = replace(noun_cfg, archive_limit=cfg.archive_limit)
                print(f"{desc} -> compute_noun_similarity{extra}")
                _call_hydra_wrapped_main(
                    noun_main,
                    noun_cfg,
                    model=noun_model,
                    preprocess=noun_preprocess,
                    tokenizer=noun_tokenizer,
                    nouns=nouns_list,
                    prompts=prompts_list,
                    noun_embeddings=noun_text_embeddings,
                )

            # Plot the result across seeds here as well, for convenience.
            _plot_seed_aggregates(
                run_configs=run_configs,
                output_dir=cross_eval_dir,
                filename_tag=filename_tag,
            )

        finally:
            os.chdir(previous_cwd)

        return

    if cfg.evaluate or cfg.visualize:
        # Match the launch_locally pattern: instantiate the relevant Config objects
        # and call their main functions directly (no subprocess).
        from dataclasses import fields as dataclass_fields

        previous_cwd = Path.cwd()
        os.chdir(original_cwd)
        try:
            if cfg.evaluate:
                import torch

                from embed_and_visualize import (
                    EmbedVisualizeConfig,
                    main as embed_main,
                    prepare_openclip_components as prepare_eval_clip,
                )

                device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
                print(f"[local-eval] Using device: {device}")

                base_kwargs0 = {
                    field_def.name: getattr(run_configs[0], field_def.name)
                    for field_def in dataclass_fields(PicbreederConfig)
                    if field_def.name != "hydra"
                }
                eval_cfg0 = EmbedVisualizeConfig(**base_kwargs0)
                eval_model, eval_preprocess = prepare_eval_clip(eval_cfg0, device)

                for run_cfg in run_configs:
                    base_kwargs = {
                        field_def.name: getattr(run_cfg, field_def.name)
                        for field_def in dataclass_fields(PicbreederConfig)
                        if field_def.name != "hydra"
                    }
                    eval_cfg = EmbedVisualizeConfig(**base_kwargs)
                    eval_cfg = replace(eval_cfg, archive_limit=cfg.archive_limit)
                    desc = _format_run_prefix(run_cfg, "[local-eval]")
                    extra = (
                        f" archive_limit={cfg.archive_limit}"
                        if cfg.archive_limit is not None
                        else ""
                    )
                    print(f"{desc} -> embed_and_visualize{extra}")
                    _call_hydra_wrapped_main(
                        embed_main,
                        eval_cfg,
                        model=eval_model,
                        preprocess=eval_preprocess,
                    )
            else:
                from visualize_archive_phylogeny import ArchivePhylogenyConfig, main as viz_main

                for run_cfg in run_configs:
                    base_kwargs = {
                        field_def.name: getattr(run_cfg, field_def.name)
                        for field_def in dataclass_fields(PicbreederConfig)
                        if field_def.name != "hydra"
                    }
                    viz_cfg = ArchivePhylogenyConfig(**base_kwargs)
                    viz_cfg = replace(viz_cfg, archive_limit=cfg.archive_limit)
                    desc = _format_run_prefix(run_cfg, "[local-viz]")
                    extra = (
                        f" archive_limit={cfg.archive_limit}"
                        if cfg.archive_limit is not None
                        else ""
                    )
                    print(f"{desc} -> visualize_archive_phylogeny{extra}")
                    _call_hydra_wrapped_main(viz_main, viz_cfg)
        finally:
            os.chdir(previous_cwd)
    elif cfg.slurm:
        launch_slurm(cfg, log_dir, run_configs)
    else:
        launch_locally(run_configs)


if __name__ == "__main__":
    main()
