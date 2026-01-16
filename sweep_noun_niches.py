#!/usr/bin/env python3
"""Launch clip_noun_niche_es sweeps locally or via Submitit."""

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
import submitit
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from scipy.spatial.distance import pdist

from clip_noun_niche_es import (
    run_es, 
    ClipNounNicheConfig, 
    load_clip_components, 
    embed_images,
    sanitize_noun
)
from render_noun_niches_grid import run_render, RenderNounGridConfig
from clip_noun_niche_shared import build_run_name

# Helper to load/validate configs
def ensure_valid_config(cfg: ClipNounNicheConfig, original_cwd: Path) -> ClipNounNicheConfig:
    # Basic validation if needed
    return cfg

_TAG_SANITIZE_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")

def _sanitize_filename_tag(tag: str) -> str:
    cleaned = _TAG_SANITIZE_PATTERN.sub("_", str(tag)).strip("_")
    return cleaned or "sweep"

class NicheRun:
    """Submitit-compatible callable that executes a configured run."""

    def __init__(self, cfg: ClipNounNicheConfig, original_cwd: Path):
        self.cfg = cfg
        self.original_cwd = original_cwd

    def __call__(self) -> int:
        print(f"[submitit] Running ES: {build_run_name(self.cfg)}")
        # When running in submitit, we might be in a different dir, but we want to reference files relative to repo root
        # if provided as relative paths.
        run_es(self.cfg, self.original_cwd)
        return 0

    def checkpoint(self) -> "submitit.helpers.DelayedSubmission":
        return submitit.helpers.DelayedSubmission(self.__class__(self.cfg, self.original_cwd))

class RenderRun:
    """Submitit-compatible callable that executes a render."""
    def __init__(self, cfg: RenderNounGridConfig, original_cwd: Path):
        self.cfg = cfg
        self.original_cwd = original_cwd

    def __call__(self) -> int:
        print(f"[submitit] Rendering: {build_run_name(self.cfg)}")
        run_render(self.cfg, self.original_cwd)
        return 0

@dataclass
class SweepNicheConfig(RenderNounGridConfig):
    # Sweepable parameters
    seed: List[int] = field(default_factory=lambda: [0])
    mutation_strength: List[float] = field(default_factory=lambda: [0.5])
    new_random_prob: List[float] = field(default_factory=lambda: [0.05])
    # ... add other sweepable params as needed
    
    sweep_name: str = "mutation"
    log_dir: str = "clip_noun_niche_es_logs" # Default output dir
    submitit_log_dir: str = "submitit_logs"
    slurm: bool = True
    partition: str = "cpu"
    account: Optional[str] = "pr_174_tandon_advanced"
    timeout_hours: int = 24
    mem_gb: int = 30
    num_proc: int = 15
    gpu: bool = True # ES needs GPU for CLIP usually
    
    plot: bool = False # If true, run rendering and analysis locally
    compare: bool = False # If true, run comparison logic
    cross_eval: bool = False # If true, summarize metrics from the configured runs
    collaborative_logs: Optional[str] = None # Path to collaborative logs for comparison
    
    hydra: HydraConf = field(
        default_factory=lambda: HydraConf(
            help=HelpConf(
                app_name="sweep_niche",
                header="Submitit/Hydra sweep launcher for clip_noun_niche_es.",
            )
        )
    )

@dataclass
class SweepBasePreset(SweepNicheConfig):
    pass

@dataclass
class MutationSweep(SweepNicheConfig):
    mutation_strength: List[float] = field(default_factory=lambda: [0.1, 0.3, 0.5, 0.7])
    seed: List[int] = field(default_factory=lambda: [0, 1, 2])
    generations: int = 10_000
    render_size: int = 224

_NAMED_SWEEPS: Dict[str, type[SweepNicheConfig]] = {
    "sweep": SweepBasePreset,
    "mutation": MutationSweep,
}

def _extract_overrides_from_preset(preset: SweepNicheConfig) -> Dict[str, Any]:
    payload = asdict(preset)
    axes: Dict[str, Any] = {}
    for key, value in payload.items():
        if key == "hydra":
            continue
        if isinstance(value, (list, tuple)):
            axes[key] = list(value)
        else:
            axes[key] = value
    return axes

def _apply_named_sweep(cfg: SweepNicheConfig) -> SweepNicheConfig:
    sweep_name = str(getattr(cfg, "sweep_name", "sweep"))
    preset_cls = _NAMED_SWEEPS.get(sweep_name)
    if preset_cls is None:
        raise ValueError(f"Unknown sweep_name={sweep_name!r}")

    preset = preset_cls()
    updates = _extract_overrides_from_preset(preset)
    
    # Simple merge, respecting command line overrides logic is harder without Hydra internals access here
    # But we can try to merge. 
    # Logic from sweep.py is safer.
    
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
    
    return omegaconf.OmegaConf.merge(cfg, omegaconf.OmegaConf.create(filtered_updates))

def _ensure_absolute(path: Path, base: Path) -> Path:
    expanded = Path(path).expanduser()
    if expanded.is_absolute():
        return expanded.resolve()
    return (base / expanded).resolve()

def _expand_sweep_configs(cfg: Union[SweepNicheConfig, omegaconf.DictConfig]) -> List[SweepNicheConfig]:
    # Convert DictConfig to dict if necessary
    if isinstance(cfg, omegaconf.DictConfig):
        cfg_dict = omegaconf.OmegaConf.to_container(cfg, resolve=True)
    else:
        cfg_dict = asdict(cfg)

    # Identify list fields
    sweep_axes: List[Tuple[str, Sequence[Any]]] = []
    # Base config fields (all fields in SweepNicheConfig which includes inherited ones)
    base_fields = {f.name for f in fields(SweepNicheConfig)}
    
    for name, value in cfg_dict.items():
        if name == "hydra" or name not in base_fields:
            continue
        
        # In Hydra/OmegaConf, lists are typically what we sweep over
        # But we need to distinguish between a list that is a value (e.g. layers) vs a sweep list
        # Here we assume any list in SweepNicheConfig is a sweep axis if defined as such in the dataclass default factories
        # However, the user might provide a single list as a value. 
        # For simplicity, we assume lists in top-level sweep config are axes.
        if isinstance(value, (list, tuple, omegaconf.listconfig.ListConfig)):
             sweep_axes.append((name, value))

    valid_keys = {f.name for f in fields(SweepNicheConfig)}
    clean_cfg_dict = {k: v for k, v in cfg_dict.items() if k in valid_keys}

    if not sweep_axes:
        return [SweepNicheConfig(**clean_cfg_dict)]
        
    configs: List[SweepNicheConfig] = []
    
    for combo in product(*(values for _, values in sweep_axes)):
        updates = {name: value for (name, _), value in zip(sweep_axes, combo)}
        
        # Create base object then replace
        # Note: We must ensure lists that are NOT sweep axes are handled correctly?
        # If we updated sweep_axes, we take one element.
        # Construct new config from merged dict
        merged = clean_cfg_dict.copy()
        merged.update(updates)
        configs.append(SweepNicheConfig(**merged))
        
    return configs

def load_metrics(run_dir: Path) -> List[Dict]:
    metrics_file = run_dir / "metrics.jsonl"
    data = []
    if metrics_file.exists():
        with metrics_file.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    data.append(json.loads(line))
                except: pass
    return data

_AGGREGATE_EXCLUDE_FIELDS = (
    "hydra", "seed", "output_dir", "submitit_log_dir", "log_dir", 
    "slurm", "partition", "account", "timeout_hours", "mem_gb", "num_proc", 
    "gpu", "render", "compare", "cross_eval", "collaborative_logs", "sweep_name"
)

def _normalize_group_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        # Keep floats stable-ish in labels/keys.
        return float(f"{value:.6g}")
    if isinstance(value, list):
         return tuple(value)
    return value

def _group_key_for_aggregate(cfg: Any) -> Tuple[Tuple[str, Any], ...]:
    items: List[Tuple[str, Any]] = []
    for field_def in fields(cfg):
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
            val = values[name]
            parts.append(f"{name}={val}")
    if not parts:
        return "default"
    return " ".join(parts)

def _effective_numeric_for_sort(name: str, value: Any, values: Dict[str, Any]) -> Optional[float]:
    try:
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
    # Use local import to avoid display issues if backend not configured
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

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
        if not index_sets:
            continue
            
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
    print(f"Wrote aggregate plot to {outpath}")
    plt.close(fig)

def _compute_diversity_for_run(rcfg: ClipNounNicheConfig, original_cwd: Path, device: torch.device, clip_model, clip_preprocess) -> None:
    """Compute mean pairwise distance of the niche population over time and final state."""
    run_name = build_run_name(rcfg)
    output_root = _ensure_absolute(Path(rcfg.output_dir), original_cwd)
    run_dir = output_root / run_name
    niches_dir = run_dir / "images" / "niches"
    elites_dir = run_dir / "elites"
    
    traj_file = run_dir / "embedding_mean_pairwise_distance_over_time.json"
    scalar_file = run_dir / "embedding_metrics.json"

    # 1. Trajectory from history (niches_dir)
    if niches_dir.exists() and not traj_file.exists():
        print(f"Computing diversity trajectory for {run_name}...")
        
        # Parse history from filenames
        updates_by_gen: Dict[int, List[Tuple[str, Path]]] = {}
        for img_path in sorted(niches_dir.glob("*.png")):
            # gen_XXXX_mode-YYYY_NOUN_SLUG_score_ZZZZ.png
            match = re.search(r"gen_(\d+)_mode-[^_]+_(.+)_score_", img_path.stem)
            if match:
                gen = int(match.group(1))
                noun_slug = match.group(2)
                updates_by_gen.setdefault(gen, []).append((noun_slug, img_path))
        
        if updates_by_gen:
            max_gen = max(updates_by_gen.keys())
            current_population: Dict[str, Path] = {}
            path_cache: Dict[Path, torch.Tensor] = {}
            history: List[Dict] = []
            
            for gen in range(max_gen + 1):
                updates = updates_by_gen.get(gen, [])
                for noun, path in updates:
                    current_population[noun] = path
                
                if not current_population:
                    continue
                    
                embeddings_list = []
                for path in current_population.values():
                    if path not in path_cache:
                        try:
                            img = Image.open(path)
                            # Pass dummy output_dim=512, not used if list not empty
                            emb = embed_images(clip_model, clip_preprocess, device, [img], batch_size=1, output_dim=512)
                            path_cache[path] = emb.cpu()
                        except Exception as e:
                            print(f"Error loading {path}: {e}")
                            continue
                    embeddings_list.append(path_cache[path])
                
                if len(embeddings_list) > 1:
                    all_embs = torch.cat(embeddings_list, dim=0).numpy()
                    dists = pdist(all_embs, metric='euclidean')
                    mean_dist = float(np.mean(dists))
                else:
                    mean_dist = 0.0
                    
                history.append({
                    "index": gen,
                    "generation": gen,
                    "mean_pairwise_distance": mean_dist,
                    "population_size": len(current_population)
                })
                
                if gen % 100 == 0:
                    print(f"  Gen {gen}: dist={mean_dist:.4f}")
            
            with traj_file.open("w") as f:
                json.dump(history, f)
            print(f"Saved diversity trajectory to {traj_file}")

    # 2. Final Scalar from elites_dir
    if elites_dir.exists() and not scalar_file.exists():
        print(f"Computing final diversity scalar for {run_name}...")
        image_paths = sorted(elites_dir.glob("*.png"))
        if image_paths:
            embeddings_list = []
            for path in image_paths:
                try:
                    img = Image.open(path)
                    emb = embed_images(clip_model, clip_preprocess, device, [img], batch_size=1, output_dim=512)
                    embeddings_list.append(emb.cpu())
                except Exception as e:
                    print(f"Error loading {path}: {e}")
            
            if len(embeddings_list) > 1:
                all_embs = torch.cat(embeddings_list, dim=0).numpy()
                dists = pdist(all_embs, metric='euclidean')
                mean_dist = float(np.mean(dists))
                std_dist = float(np.std(dists))
            else:
                mean_dist = 0.0
                std_dist = 0.0
            
            payload = {
                "pairwise_distances": {
                    "mean": mean_dist,
                    "std": std_dist,
                    "n": len(image_paths)
                }
            }
            with scalar_file.open("w") as f:
                json.dump(payload, f, indent=2)
            print(f"Saved final diversity metrics to {scalar_file}")

def perform_cross_eval(cfg: SweepNicheConfig, run_configs: Sequence[ClipNounNicheConfig], original_cwd: Path) -> None:
    output_dir = _ensure_absolute(Path("cross_eval_clip_niches"), original_cwd)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Organize data
    mean_best_grouped: Dict[Tuple, List[Dict[int, float]]] = {}
    mpd_grouped: Dict[Tuple, List[Dict[int, float]]] = {}
    
    # Scalars for bar plots
    noun_scalar_grouped: Dict[Tuple, List[float]] = {}
    mpd_scalar_grouped: Dict[Tuple, List[float]] = {}
    
    print("Gathering metrics for cross-eval...")
    
    for rcfg in run_configs:
        group_key = _group_key_for_aggregate(rcfg)
        
        run_name = build_run_name(rcfg)
        output_root = _ensure_absolute(Path(rcfg.output_dir), original_cwd)
        run_dir = output_root / run_name
        
        # Load standard metrics
        metrics = load_metrics(run_dir)
        mean_best_traj = {}
        for m in metrics:
            gen = m.get("generation")
            if gen is not None and "mean_best_score" in m:
                mean_best_traj[gen] = float(m["mean_best_score"])
        
        if mean_best_traj:
            mean_best_grouped.setdefault(group_key, []).append(mean_best_traj)
            # Final scalar from trajectory
            last_gen = max(mean_best_traj.keys())
            noun_scalar_grouped.setdefault(group_key, []).append(mean_best_traj[last_gen])
            
        # Load diversity metrics
        traj_file = run_dir / "embedding_mean_pairwise_distance_over_time.json"
        scalar_file = run_dir / "embedding_metrics.json"
        
        mpd_traj = {}
        if traj_file.exists():
            try:
                with traj_file.open("r") as f:
                    data = json.load(f)
                    for entry in data:
                        idx = entry.get("generation")
                        val = entry.get("mean_pairwise_distance")
                        if idx is not None and val is not None:
                            mpd_traj[idx] = float(val)
            except: pass
        
        if mpd_traj:
            mpd_grouped.setdefault(group_key, []).append(mpd_traj)
        
        # Load final scalar for MPD
        mpd_val = None
        if scalar_file.exists():
            try:
                with scalar_file.open("r") as f:
                    data = json.load(f)
                    # Support both schemas
                    if "pairwise_distances" in data:
                        mpd_val = data["pairwise_distances"].get("mean")
                    elif "mean_pairwise_distance" in data:
                        mpd_val = data["mean_pairwise_distance"] # might be dict or val
                        if isinstance(mpd_val, dict):
                            mpd_val = mpd_val.get("value")
            except: pass
        
        # Fallback to last point of trajectory if scalar file missing
        if mpd_val is None and mpd_traj:
             mpd_val = mpd_traj[max(mpd_traj.keys())]
             
        if mpd_val is not None:
             mpd_scalar_grouped.setdefault(group_key, []).append(float(mpd_val))

    sweep_name = _sanitize_filename_tag(getattr(cfg, "sweep_name", "sweep"))
    
    # 1. Trajectory Plots
    if mean_best_grouped:
        _write_aggregate_plot(
            grouped_runs=mean_best_grouped,
            outpath=output_dir / f"aggregate_noun_similarity_over_time_{sweep_name}.png",
            title="Noun similarity over time (mean±std across seeds)",
            xlabel="Generation",
            ylabel="Mean max cosine similarity",
        )
    
    if mpd_grouped:
        _write_aggregate_plot(
            grouped_runs=mpd_grouped,
            outpath=output_dir / f"aggregate_embedding_mean_pairwise_distance_over_time_{sweep_name}.png",
            title="Embedding diversity over time (mean±std across seeds)",
            xlabel="Generation",
            ylabel="Mean pairwise distance",
        )
        
    # 2. Bar Plots
    if noun_scalar_grouped:
        _write_scalar_bar_plot(
            grouped_values=noun_scalar_grouped,
            outpath=output_dir / f"aggregate_noun_similarity_mean_bar_{sweep_name}.png",
            title="Mean max noun similarity (mean±std across seeds)",
            ylabel="Mean of per-noun max cosine similarity",
        )

    if mpd_scalar_grouped:
        _write_scalar_bar_plot(
            grouped_values=mpd_scalar_grouped,
            outpath=output_dir / f"aggregate_mean_pairwise_distance_mean_bar_{sweep_name}.png",
            title="Mean pairwise distance (mean±std across seeds)",
            ylabel="Mean pairwise distance (euclidean)",
        )

def launch_locally(run_configs: Sequence[ClipNounNicheConfig], original_cwd: Path, mode: str) -> None:
    for rcfg in run_configs:
        name = build_run_name(rcfg)
        print(f"[local] {mode}: {name}")
        if mode == "es":
            run_es(rcfg, original_cwd)
        elif mode == "render":
            run_render(rcfg, original_cwd)

def compare_runs(cfg: SweepNicheConfig, run_configs: Sequence[ClipNounNicheConfig], original_cwd: Path):
    # Gather niche ES runs
    niche_data = {}
    
    for rcfg in run_configs:
        # Determine run directory
        output_root = _ensure_absolute(Path(rcfg.output_dir), original_cwd)
        run_name = build_run_name(rcfg)
        run_dir = output_root / run_name
        
        metrics = load_metrics(run_dir)
        if metrics:
            niche_data[run_name] = metrics

    # Gather collaborative data if provided
    collab_data = {}
    if cfg.collaborative_logs:
        collab_root = _ensure_absolute(Path(cfg.collaborative_logs), original_cwd)
        # Assuming typical structure sweep_logs/sweep_name/experiment_dir
        # We might need to walk directories
        for exp_dir in collab_root.iterdir():
            if exp_dir.is_dir():
                # Look for noun_similarity_over_time.json
                # Collaborative runs might have different noun list files. 
                # We check for any file matching pattern or specific name
                # Usually: noun_similarity_over_time_{nounlist}.json or just noun_similarity_over_time.json
                
                # Check for standard one first
                ns_file = exp_dir / "noun_similarity_over_time.json"
                if not ns_file.exists():
                     # Check for others
                     for f in exp_dir.glob("noun_similarity_over_time_*.json"):
                         ns_file = f
                         break
                
                if ns_file and ns_file.exists():
                     try:
                         with ns_file.open("r") as f:
                             data = json.load(f)
                             if isinstance(data, list):
                                 collab_data[exp_dir.name] = data
                     except: pass

    # Plotting
    plt.figure(figsize=(12, 8))
    
    # Plot Niche Runs
    for name, metrics in niche_data.items():
        gens = [m['generation'] for m in metrics]
        # Calculate Mean Max Similarity
        # qd_score = sum(best_scores)
        # nouns = count
        means = []
        for m in metrics:
            if 'mean_best_score' in m:
                means.append(m['mean_best_score'])
            elif 'qd_score' in m and 'nouns' in m and m['nouns'] > 0:
                means.append(m['qd_score'] / m['nouns'])
            else:
                means.append(0)
        
        plt.plot(gens, means, label=f"Niche: {name}")

    # Plot Collaborative Runs
    for name, data in collab_data.items():
        # Collaborative data from noun_similarity is list of {index, mean_max_similarity, ...}
        # index is typically archive size or generation
        
        indices = [d.get('index', i) for i, d in enumerate(data)]
        vals = [d.get('mean_max_similarity', 0) for d in data]
        
        plt.plot(indices, vals, label=f"Collab: {name}", linestyle='--')
        
    plt.xlabel("Generation / Archive Size")
    plt.ylabel("Mean Max Similarity (QD / Nouns)")
    plt.legend()
    plt.title("Evolution Comparison: Mean Max Similarity")
    plt.grid(True, alpha=0.3)
    out_path = Path("qd_comparison.png")
    plt.savefig(out_path)
    print(f"Saved comparison plot to {out_path}")

def launch_slurm(cfg: SweepNicheConfig, run_configs: Sequence[ClipNounNicheConfig], original_cwd: Path, mode: str) -> None:
    executor = submitit.AutoExecutor(folder=cfg.submitit_log_dir)
    executor.update_parameters(
        timeout_min=cfg.timeout_hours * 60,
        mem_gb=cfg.mem_gb,
        cpus_per_task=cfg.num_proc,
        slurm_account=cfg.account,
        name="picbreeder-niche",
    )
    if cfg.gpu:
        executor.update_parameters(slurm_gres='gpu:1')

    if mode == "es":
        jobs = [NicheRun(replace(rcfg, output_dir=Path(rcfg.output_dir)), original_cwd) for rcfg in run_configs]
    elif mode == "render":
        jobs = [RenderRun(replace(rcfg, output_dir=Path(rcfg.output_dir)), original_cwd) for rcfg in run_configs]
    
    futures = executor.map_array(lambda j: j(), jobs)
    for rcfg, future in zip(run_configs, futures):
        print(f"Submitted {build_run_name(rcfg)} as job {future.job_id}")

cs = ConfigStore.instance()
cs.store(name="sweep_niche_base", node=SweepNicheConfig)

@hydra.main(version_base=None, config_path=None, config_name="sweep_niche_base")
def main(cfg: SweepNicheConfig) -> None:
    original_cwd = Path(get_original_cwd())
    log_dir = _ensure_absolute(Path(cfg.log_dir), original_cwd)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    cfg = _apply_named_sweep(cfg)
    
    # Expand configs
    configs = _expand_sweep_configs(cfg)
    
    # Convert to specific config objects for the runners
    # Since we used SweepNicheConfig which inherits RenderNounGridConfig (and thus ClipNounNicheConfig),
    # we can just pass them.
    
    if cfg.compare:
        compare_runs(cfg, configs, original_cwd)
        return

    mode = "render" if cfg.plot else "es"
    
    if cfg.slurm:
        if cfg.plot:
             print("Warning: plot=True with slurm=True is not supported. Running plot locally.")
             # Fall through to local plot logic
        else:
            launch_slurm(cfg, configs, original_cwd, mode)
            return
    
    if cfg.plot:
        # Local plotting logic
        # 1. Initialize CLIP once
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading CLIP for diversity calculation on {device}...")
        # Use first config to determine model type (assuming all same in sweep)
        ref_cfg = configs[0]
        # We need preprocess but not text embeddings for diversity
        # Reuse load_clip_components but dummy nouns
        clip = load_clip_components(ref_cfg.clip_model, ref_cfg.clip_pretrained, str(device), ["dummy"])
        
        for rcfg in configs:
            # Run rendering (if needed)
            print(f"[plot] Rendering: {build_run_name(rcfg)}")
            run_render(rcfg, original_cwd)
            
            # Run diversity calc
            _compute_diversity_for_run(rcfg, original_cwd, device, clip.model, clip.preprocess)
            
    if cfg.plot or cfg.cross_eval:
        # Run aggregation
        perform_cross_eval(cfg, configs, original_cwd)
    else:
        # Local ES
        launch_locally(configs, original_cwd, mode)

if __name__ == "__main__":
    main()
