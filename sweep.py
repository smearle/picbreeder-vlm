#!/usr/bin/env python3
"""Launch Picbreeder-VLM collaborative_multi_agent sweeps locally or via Submitit."""

from __future__ import annotations

import os
import re
import sys
import json
import multiprocessing
import numpy as np
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
from utils import resolve_nounlist
from sweep_analysis_utils import (
    sanitize_filename_tag,
    normalize_group_value,
    compute_varying_fields,
    format_group_label,
    load_human_baseline,
    write_aggregate_plot,
    write_scalar_bar_plot,
)


SCRIPT_ROOT = Path(__file__).resolve().parent


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
    thumb_size: List[int] = field(default_factory=lambda: [128])  # Thumbnail sizes to evaluate
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
    n_personality_traits: List[int] = field(default_factory=lambda: [0])  # Number of personality traits to use
    image_embedding_model: str = "SigLIP2-B-alignet"
    image_pretrained: str = "laion2b_s32b_b79k"
    text_image_embedding_model: str = "ViT-SO400M-14-SigLIP2"
    text_image_pretrained: str = "webli"
    sweep_name: str = "rand_select_prob"  # Base directory for experiment outputs
    log_dir: str = "sweep_logs"
    submitit_log_dir: str = "submitit_logs"
    slurm: bool = True  # Enable SLURM submission via Submitit
    partition: str = "cpu"  # SLURM partition name
    gpu: bool = False
    # account: Optional[str] = None  # Optional SLURM account override
    account: Optional[str] = "pr_174_tandon_advanced"  # Optional SLURM account override
    timeout_hours: int = 24  # Wall-time limit in hours
    mem_gb: int = 30  # Memory requested per task (GB)
    num_proc: int = 10  # Number of parallel processes per task
    render_archive: bool = False  # If true, run evaluation instead of training
    render_tree: bool = False  # If true, run phylogeny visualization instead of training
    eval: bool = False  # If true, run plotting/analysis scripts instead of training
    overwrite_evals: bool = True  # If false, skip evaluation if output files already exist
    cross_eval: bool = False  # If true, summarize embedding metrics from the configured runs
    archive_limit: Optional[int] = None  # Limit the number of archive images passed to analysis scripts
    nounlist: List[str] = field(default_factory=lambda: ["imagenet21k"])  # Noun list(s) to evaluate
    hydra: HydraConf = field(
        default_factory=lambda: HydraConf(
            help=HelpConf(                app_name="sweep",
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
    num_agents: int = 750


@dataclass
class ChatHistoryTurnsQwenSweep(SweepConfig):
    chat_history_turns: List[int] = field(default_factory=lambda: [0, 1, 2, 3])
    temperature: List[Union[int, float, str]] = field(default_factory=lambda: [1.0])
    rand_select_prob: List[float] = field(default_factory=lambda: [0.0])
    goal: List[str] = field(default_factory=lambda: ["familiar_objects"])
    model: List[str] = field(default_factory=lambda: ["qwen3-vl-8b"])
    seed: List[int] = field(default_factory=lambda: [3, 4, 5])
    num_agents: int = 500
    num_proc: int = 1
    gpu: bool = True


@dataclass
class TemperatureSweep(SweepConfig):
    temperature: List[Union[int, float, str]] = field(default_factory=lambda: [0.0, 1.0, 2.0, "random"])
    chat_history_turns: List[int] = field(default_factory=lambda: [1])
    rand_select_prob: List[float] = field(default_factory=lambda: [0.0])
    goal: List[str] = field(default_factory=lambda: ["familiar_objects"])
    model: List[str] = field(default_factory=lambda: ["gemini-2.5-pro"])
    seed: List[int] = field(default_factory=lambda: [3, 4, 5])
    num_agents: int = 500


@dataclass
class RandSelectProbSweep(SweepConfig):
    rand_select_prob: List[float] = field(default_factory=lambda: [0.0, 0.25, 0.5, 0.75, 1.0])
    chat_history_turns: List[int] = field(default_factory=lambda: [1])
    temperature: List[Union[int, float, str]] = field(default_factory=lambda: [1.0])
    goal: List[str] = field(default_factory=lambda: ["familiar_objects"])
    model: List[str] = field(default_factory=lambda: ["gemini-2.5-pro"])
    seed: List[int] = field(default_factory=lambda: [3, 4, 5])
    # thumb_size: List[int] = field(default_factory=lambda: [128, 224])
    thumb_size: List[int] = field(default_factory=lambda: [128,])
    num_agents: int = 500

@dataclass
class FullRandSelectProbSweep(SweepConfig):
    rand_select_prob: List[float] = field(default_factory=lambda: [0.0, 0.25, 0.5, 0.75, 1.0])
    rand_select_mode: str = 'all'
    chat_history_turns: List[int] = field(default_factory=lambda: [1])
    temperature: List[Union[int, float, str]] = field(default_factory=lambda: [1.0])
    goal: List[str] = field(default_factory=lambda: ["familiar_objects"])
    model: List[str] = field(default_factory=lambda: ["gemini-2.5-pro"])
    seed: List[int] = field(default_factory=lambda: [3, 4, 5])
    thumb_size: List[int] = field(default_factory=lambda: [128,])
    num_agents: int = 1_000

@dataclass
class RandBaselineSweep(SweepConfig):
    rand_select_prob: List[float] = field(default_factory=lambda: [2.0])
    rand_select_mode: str = 'all'
    chat_history_turns: List[int] = field(default_factory=lambda: [1])
    temperature: List[Union[int, float, str]] = field(default_factory=lambda: [1.0])
    goal: List[str] = field(default_factory=lambda: ["familiar_objects"])
    model: List[str] = field(default_factory=lambda: ["gemini-2.5-pro"])
    seed: List[int] = field(default_factory=lambda: [3, 4, 5])
    thumb_size: List[int] = field(default_factory=lambda: [128,])
    num_agents: int = 9_586

@dataclass
class ModelSweep(SweepConfig):
    model: List[str] = field(default_factory=lambda: [
        "gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-3-pro-preview",
        "gemini-random",
        # "qwen3-vl-8b",
    ])
    chat_history_turns: List[int] = field(default_factory=lambda: [1])
    seed: List[int] = field(default_factory=lambda: [3, 4, 5])
    num_agents: int = 1_000

@dataclass
class TraitsSweep(SweepConfig):
    n_personality_traits: List[int] = field(default_factory=lambda: [
        0,
        10, 100, 1_000
    ])
    model: List[str] = field(default_factory=lambda: ["gemini-2.5-pro"])
    chat_history_turns: List[int] = field(default_factory=lambda: [1])
    seed: List[int] = field(default_factory=lambda: [3, 4, 5])
    num_agents: int = 500

@dataclass
class LongSweep(SweepConfig):
    rand_select_prob: List[float] = field(default_factory=lambda: [0.0])
    chat_history_turns: List[int] = field(default_factory=lambda: [1])
    temperature: List[Union[int, float, str]] = field(default_factory=lambda: [1.0])
    goal: List[str] = field(default_factory=lambda: ["familiar_objects"])
    model: List[str] = field(default_factory=lambda: ["gemini-2.5-pro"])
    seed: List[int] = field(default_factory=lambda: [3])
    num_agents: int = 9_586

@dataclass
class LongSweep2(SweepConfig):
    rand_select_prob: List[float] = field(default_factory=lambda: [0.25])
    rand_select_mode: str = 'all'
    chat_history_turns: List[int] = field(default_factory=lambda: [1])
    temperature: List[Union[int, float, str]] = field(default_factory=lambda: [1.0])
    goal: List[str] = field(default_factory=lambda: ["familiar_objects"])
    model: List[str] = field(default_factory=lambda: ["gemini-2.5-pro"])
    seed: List[int] = field(default_factory=lambda: [5])
    num_agents: int = 9_586


_NAMED_SWEEPS: Dict[str, type[SweepConfig]] = {
    "sweep": SweepBasePreset,
    "chat_history_turns": ChatHistoryTurnsSweep,
    "chat_history_turns_qwen": ChatHistoryTurnsQwenSweep,
    "temperature": TemperatureSweep,
    "rand_select_prob": RandSelectProbSweep,
    "full_rand_select_prob": FullRandSelectProbSweep,
    "rand_baseline": RandBaselineSweep,
    "model": ModelSweep,
    "traits": TraitsSweep,
    "long_sweep": LongSweep,
    "long_sweep_2": LongSweep2,
}


def _extract_overrides_from_preset(preset: SweepConfig) -> Dict[str, Any]:
    """Return overrides from a preset.

    We apply all fields from the preset so that scalar overrides (like num_agents)
    take effect. We rely on _apply_named_sweep to protect explicit CLI overrides.
    """

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
    updates = _extract_overrides_from_preset(preset)
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


def _run_eval_phase_1(cfg: SweepConfig, run_configs: Sequence[PicbreederConfig], original_cwd: Path, device_str: str):
    import torch
    import os
    from pathlib import Path
    from dataclasses import fields as dataclass_fields, replace
    from plot_novelty_over_time import (
        PairwiseDistanceConfig,
        main as novelty_main,
        prepare_openclip_components as prepare_novelty_clip,
    )
    from config import PicbreederConfig

    # Attempt to set TF memory growth
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(f"Error setting TF memory growth: {e}")
    except ImportError:
        pass

    os.chdir(original_cwd)
    device = torch.device(device_str)
    print(f"[Phase 1 Worker] Using device: {device}")

    base_kwargs0 = {
        field_def.name: getattr(run_configs[0], field_def.name)
        for field_def in dataclass_fields(PicbreederConfig)
        if field_def.name != "hydra"
    }
    
    print("\n[Phase 1] Evaluating visual novelty...")
    novelty_cfg0 = PairwiseDistanceConfig(**base_kwargs0, embedding_model=cfg.image_embedding_model, pretrained=cfg.image_pretrained)
    novelty_model, novelty_preprocess = prepare_novelty_clip(novelty_cfg0, device)

    for run_cfg in run_configs:
        # Check if output exists
        model_name_sanitized = cfg.image_embedding_model.replace("/", "-")
        exp_dir = Path(run_cfg.experiment_dir)
        output_file = exp_dir / f"embedding_mean_pairwise_distance_over_time_{model_name_sanitized}.json"
        
        if not cfg.overwrite_evals and output_file.exists():
            print(f"Skipping novelty eval for {exp_dir} (already exists)")
            continue

        base_kwargs = {
            field_def.name: getattr(run_cfg, field_def.name)
            for field_def in dataclass_fields(PicbreederConfig)
            if field_def.name != "hydra"
        }

        novelty_cfg = PairwiseDistanceConfig(**base_kwargs, embedding_model=cfg.image_embedding_model, pretrained=cfg.image_pretrained)
        novelty_cfg = replace(novelty_cfg, archive_limit=cfg.archive_limit)
        desc = _format_run_prefix(run_cfg, "[local-eval]")
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
            original_cwd_override=original_cwd,
        )


def _run_eval_phase_2(cfg: SweepConfig, run_configs: Sequence[PicbreederConfig], original_cwd: Path, device_str: str):
    import torch
    import os
    from pathlib import Path
    from collections import defaultdict
    from dataclasses import fields as dataclass_fields, replace
    from compute_noun_similarity import (
        NounSimilarityConfig,
        main as noun_main,
        prepare_openclip_components as prepare_noun_clip,
        prepare_noun_text_embeddings,
    )
    from config import PicbreederConfig

    # Attempt to set TF memory growth (just in case)
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(f"Error setting TF memory growth: {e}")
    except ImportError:
        pass

    os.chdir(original_cwd)
    device = torch.device(device_str)
    print(f"[Phase 2 Worker] Using device: {device}")

    base_kwargs0 = {
        field_def.name: getattr(run_configs[0], field_def.name)
        for field_def in dataclass_fields(PicbreederConfig)
        if field_def.name != "hydra"
    }

    print("\n[Phase 2] Evaluating noun similarity...")
    # Initialize model once (assuming embedding model is constant across sweep)
    noun_cfg_template = NounSimilarityConfig(**base_kwargs0, embedding_model=cfg.text_image_embedding_model, pretrained=cfg.text_image_pretrained)
    noun_model, noun_preprocess, noun_tokenizer = prepare_noun_clip(noun_cfg_template, device)

    # Group runs by nounlist to avoid reloading/re-embedding nouns unnecessarily
    runs_by_nounlist = defaultdict(list)
    for rc in run_configs:
        runs_by_nounlist[rc.nounlist].append(rc)

    for nounlist, group_configs in runs_by_nounlist.items():
        print(f"  Processing group with nounlist: {nounlist}")
        
        # Create a config for this group to prepare embeddings
        group_ref_cfg = group_configs[0]
        base_kwargs_group = {
             field_def.name: getattr(group_ref_cfg, field_def.name)
             for field_def in dataclass_fields(PicbreederConfig)
             if field_def.name != "hydra"
        }
        noun_cfg_group = NounSimilarityConfig(**base_kwargs_group, embedding_model=cfg.text_image_embedding_model, pretrained=cfg.text_image_pretrained)
        noun_cfg_group.nounlist = nounlist

        nouns_list, prompts_list, noun_text_embeddings = prepare_noun_text_embeddings(
            noun_cfg_group,
            original_cwd=original_cwd,
            device=device,
            model=noun_model,
            tokenizer=noun_tokenizer,
        )

        for run_cfg in group_configs:
            # Check if output exists
            nounlist_name = Path(run_cfg.nounlist).stem
            model_name_sanitized = cfg.text_image_embedding_model.replace("/", "-")
            exp_dir = Path(run_cfg.experiment_dir)
            output_file = exp_dir / f"noun_similarity_over_time_{nounlist_name}_{model_name_sanitized}.json"

            if not cfg.overwrite_evals and output_file.exists():
                print(f"Skipping noun similarity eval for {exp_dir} (already exists)")
                continue

            base_kwargs = {
                field_def.name: getattr(run_cfg, field_def.name)
                for field_def in dataclass_fields(PicbreederConfig)
                if field_def.name != "hydra"
            }
            noun_cfg = NounSimilarityConfig(**base_kwargs, render_grid=True, embedding_model=cfg.text_image_embedding_model, pretrained=cfg.text_image_pretrained)
            noun_cfg = replace(noun_cfg, archive_limit=cfg.archive_limit)
            
            noun_cfg.nounlist = nounlist

            desc = _format_run_prefix(run_cfg, "[local-eval]")
            extra = (
                f" archive_limit={cfg.archive_limit}"
                if cfg.archive_limit is not None
                else ""
            )
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
                original_cwd_override=original_cwd,
            )


_AGGREGATE_EXCLUDE_FIELDS: Tuple[str, ...] = (
    "hydra",
    "seed",
    "experiment_dir",
    "resume",
    "resume_agent_id",
)











def _group_key_for_aggregate(cfg: PicbreederConfig) -> Tuple[Tuple[str, Any], ...]:
    items: List[Tuple[str, Any]] = []
    for field_def in fields(PicbreederConfig):
        name = field_def.name
        if name in _AGGREGATE_EXCLUDE_FIELDS:
            continue
        items.append((name, normalize_group_value(getattr(cfg, name))))
    return tuple(items)





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


def _load_noun_similarity_scalar(exp_dir: Path, model_name: Optional[str] = None, nounlist_name: Optional[str] = None) -> Optional[float]:
    """Load a single noun similarity scalar for an experiment.

    Prefers noun_similarity_metrics.json (written by compute_noun_similarity.py).
    Falls back to the final value in noun_similarity_over_time.json.
    """
    model_suffix = ""
    if model_name:
        sanitized = model_name.replace("/", "-")
        model_suffix = f"_{sanitized}"

    # Try model-specific metrics first
    if model_name:
        metrics_path = exp_dir / f"noun_similarity_metrics{model_suffix}.json"
    else:
        metrics_path = exp_dir / "noun_similarity_metrics.json"

    if metrics_path.exists():
        try:
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            value = payload.get("mean_max_similarity")
            if value is None:
                return None
            return float(value)
        except Exception:
            pass

    # Try trajectory files
    if nounlist_name is None:
        raise ValueError("nounlist_name must be provided to load from trajectory files.")

    candidates = []
    if model_name:
        candidates.append(exp_dir / f"noun_similarity_over_time_{nounlist_name}{model_suffix}.json")
    else:
        candidates.append(exp_dir / f"noun_similarity_over_time_{nounlist_name}.json")
    
    for trajectory_path in candidates:
        if not trajectory_path.exists():
            continue
        try:
            traj = json.loads(trajectory_path.read_text(encoding="utf-8"))
            if not isinstance(traj, list) or not traj:
                continue
            last = traj[-1]
            if not isinstance(last, dict):
                continue
            value = last.get("mean_max_similarity")
            if value is not None:
                return float(value)
        except Exception:
            continue
            
    return None


def _load_embedding_mean_pairwise_distance_scalar(exp_dir: Path, model_name: Optional[str] = None) -> Optional[float]:
    """Load mean pairwise distance scalar for an experiment.

    Prefers embedding_metrics.json (produced by embed_and_visualize.py) using either:
      - mean_pairwise_distance.value (legacy schema)
      - pairwise_distances.mean (newer schema)

    Falls back to the final value in embedding_mean_pairwise_distance_over_time.json.
    """
    model_suffix = ""
    if model_name:
        sanitized = model_name.replace("/", "-")
        model_suffix = f"_{sanitized}"

    # Try model-specific metrics first
    candidates_metrics = []
    if model_name:
         candidates_metrics.append(exp_dir / f"embedding_metrics{model_suffix}.json")
    else:
         candidates_metrics.append(exp_dir / "embedding_metrics.json")

    for metrics_path in candidates_metrics:
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
                continue

    # Try trajectory files
    candidates_traj = []
    if model_name:
        candidates_traj.append(exp_dir / f"embedding_mean_pairwise_distance_over_time{model_suffix}.json")
    else:
        candidates_traj.append(exp_dir / "embedding_mean_pairwise_distance_over_time.json")

    for traj_path in candidates_traj:
        if not traj_path.exists():
            continue
        try:
            traj = json.loads(traj_path.read_text(encoding="utf-8"))
            if not isinstance(traj, list) or not traj:
                continue
            last = traj[-1]
            if not isinstance(last, dict):
                continue
            value = last.get("mean_pairwise_distance")
            if value is not None:
                return float(value)
        except Exception:
            continue
            
    return None


def _get_random_baseline_configs(
    num_agents: int,
    thumb_size: int,
    nounlist: str,
    original_cwd: Path
) -> List[PicbreederConfig]:
    """Generate run configs for the Random Baseline (rand_select_prob=2.0)."""
    base = RandBaselineSweep()
    base.num_agents = num_agents
    base.thumb_size = [thumb_size]
    base.nounlist = [nounlist]
    
    configs = _expand_sweep_configs(base)
    
    existing = []
    for cfg in configs:
        run_cfg = _build_run_config(cfg, original_cwd)
        if Path(run_cfg.experiment_dir).exists():
            existing.append(run_cfg)
            
    return existing

def _compute_mean_trajectory(
    run_configs: List[PicbreederConfig],
    metric_type: str,
    model_name: Optional[str] = None,
    nounlist_name: Optional[str] = None,
) -> Optional[Dict[int, float]]:
    trajectories = []
    for run_cfg in run_configs:
        exp_dir = Path(run_cfg.experiment_dir)
        limit = run_cfg.num_agents
        
        if metric_type == "novelty":
            suffix = ""
            if model_name:
                sanitized = model_name.replace("/", "-")
                suffix = f"_{sanitized}"
            
            path = exp_dir / f"embedding_mean_pairwise_distance_over_time{suffix}.json"
            if path.exists():
                traj = _load_trajectory_metric(path, "mean_pairwise_distance")
                traj = {k: v for k, v in traj.items() if k <= limit}
                if traj:
                    trajectories.append(traj)
                    
        elif metric_type == "noun":
            if not nounlist_name:
                continue
            suffix = ""
            if model_name:
                sanitized = model_name.replace("/", "-")
                suffix = f"_{sanitized}"
            
            path = exp_dir / f"noun_similarity_over_time_{nounlist_name}{suffix}.json"
            if path.exists():
                traj = _load_trajectory_metric(path, "mean_max_similarity")
                traj = {k: v for k, v in traj.items() if k <= limit}
                if traj:
                    trajectories.append(traj)

    if not trajectories:
        return None

    # Compute mean per step
    all_steps = set()
    for t in trajectories:
        all_steps.update(t.keys())
    
    mean_traj = {}
    sorted_steps = sorted(all_steps)
    for step in sorted_steps:
        values = [t[step] for t in trajectories if step in t]
        if values:
            mean_traj[step] = float(np.mean(values))
            
    return mean_traj

def _compute_mean_scalar(
    run_configs: List[PicbreederConfig],
    metric_type: str,
    model_name: Optional[str] = None,
    nounlist_name: Optional[str] = None,
) -> Optional[float]:
    values = []
    for run_cfg in run_configs:
        exp_dir = Path(run_cfg.experiment_dir)
        val = None
        
        if metric_type == "novelty":
            val = _load_embedding_mean_pairwise_distance_scalar(exp_dir, model_name)
        elif metric_type == "noun":
            val = _load_noun_similarity_scalar(exp_dir, model_name, nounlist_name)
        elif metric_type in ("sackin", "colless", "depth"):
             metrics_path = exp_dir / "archive" / "phylogeny_metrics.json"
             if metrics_path.exists():
                 try:
                    m = json.loads(metrics_path.read_text(encoding="utf-8"))
                    if metric_type == "sackin":
                        val = m.get("sackin_index")
                    elif metric_type == "colless":
                        val = m.get("colless_index")
                    elif metric_type == "depth":
                        val = m.get("max_depth")
                 except Exception:
                     pass
        
        if val is not None:
            values.append(float(val))
            
    if not values:
        return None
        
    return float(np.mean(values))


def _plot_seed_aggregates(
    *,
    run_configs: Sequence[PicbreederConfig],
    output_dir: Path,
    filename_tag: str,
    image_embedding_model: Optional[str] = None,
    text_image_embedding_model: Optional[str] = None,
) -> None:
    novelty_grouped: Dict[Tuple[Tuple[str, Any], ...], List[Dict[int, float]]] = {}
    noun_grouped: Dict[Tuple[Tuple[str, Any], ...], List[Dict[int, float]]] = {}
    noun_scalar_grouped: Dict[Tuple[Tuple[str, Any], ...], List[float]] = {}
    mpd_scalar_grouped: Dict[Tuple[Tuple[str, Any], ...], List[float]] = {}

    image_model_suffix = ""
    if image_embedding_model:
        sanitized = image_embedding_model.replace("/", "-")
        image_model_suffix = f"_{sanitized}"

    text_image_model_suffix = ""
    if text_image_embedding_model:
        sanitized = text_image_embedding_model.replace("/", "-")
        text_image_model_suffix = f"_{sanitized}"

    # We might have different nounlists across run_configs
    # We collect all used nounlists to load baselines for them
    used_nounlists = set()
    unique_sizes = set()

    if run_configs:
        unique_sizes = sorted(list({cfg.thumb_size for cfg in run_configs}))
        for cfg in run_configs:
            used_nounlists.add(Path(cfg.nounlist).stem)

    baselines_novelty: List[Tuple[str, Dict[int, float]]] = []
    baselines_noun: List[Tuple[str, Dict[int, float]]] = []
    
    # Store random baseline scalars separately to merge later
    extra_scalars_novelty: List[Tuple[str, float]] = []
    extra_scalars_noun: List[Tuple[str, float]] = []
    
    if run_configs:
        baseline_model_novelty = image_embedding_model if image_embedding_model else "ViT-B-32"
        baseline_model_noun = text_image_embedding_model if text_image_embedding_model else "ViT-H-14"
        
        for size in unique_sizes:
            label_suffix = f" ({size}px)" if len(unique_sizes) > 1 else ""
            
            bn = load_human_baseline("novelty", size, baseline_model_novelty)
            if bn:
                baselines_novelty.append((f"Human Baseline{label_suffix}", bn))
            
            for nl_name in used_nounlists:
                 nl_suffix = f" {nl_name}" if len(used_nounlists) > 1 else ""
                 bnn = load_human_baseline("noun", size, baseline_model_noun, nounlist=nl_name)
                 if bnn:
                     baselines_noun.append((f"Human Baseline{label_suffix}{nl_suffix}", bnn))
            
            # Random Baseline
            for nl_name in used_nounlists:
                nl_suffix = f" {nl_name}" if len(used_nounlists) > 1 else ""
                
                # Fetch random baseline runs
                # Using run_configs[0].num_agents assuming constant limit
                limit = run_configs[0].num_agents if run_configs else 0
                if limit > 0:
                     rb_configs = _get_random_baseline_configs(limit, size, nl_name, Path(get_original_cwd()))
                     if rb_configs:
                         # Novelty
                         traj_nov = _compute_mean_trajectory(rb_configs, "novelty", image_embedding_model)
                         if traj_nov:
                             baselines_novelty.append((f"Random Baseline{label_suffix}", traj_nov))
                             scalar_nov = _compute_mean_scalar(rb_configs, "novelty", image_embedding_model)
                             if scalar_nov is not None:
                                 extra_scalars_novelty.append((f"Random Baseline{label_suffix}", scalar_nov))

                         # Noun
                         traj_noun = _compute_mean_trajectory(rb_configs, "noun", text_image_embedding_model, nl_name)
                         if traj_noun:
                             baselines_noun.append((f"Random Baseline{label_suffix}{nl_suffix}", traj_noun))
                             scalar_noun = _compute_mean_scalar(rb_configs, "noun", text_image_embedding_model, nl_name)
                             if scalar_noun is not None:
                                 extra_scalars_noun.append((f"Random Baseline{label_suffix}{nl_suffix}", scalar_noun))
                         
                         # Avoid adding novelty baseline multiple times if we loop over nounlists 
                         # (though directories might differ, novelty is same concept).
                         # But since we look up runs by nounlist, we might find different runs.
                         # If multiple nounlists are used in current sweep, we might have multiple random baselines?
                         # Let's just keep them all for now, labeled by nounlist if necessary.

    for run_cfg in run_configs:
        group_key = _group_key_for_aggregate(run_cfg)
        exp_dir = Path(run_cfg.experiment_dir)
        limit = run_cfg.num_agents
        
        current_nounlist_name = Path(run_cfg.nounlist).stem

        # Try specific model file first
        if image_embedding_model:
            novelty_path = exp_dir / f"embedding_mean_pairwise_distance_over_time{image_model_suffix}.json"
        else:
            novelty_path = exp_dir / "embedding_mean_pairwise_distance_over_time.json"
            
        if novelty_path.exists():
            novelty = _load_trajectory_metric(novelty_path, "mean_pairwise_distance")
            # Truncate at num_agents
            novelty = {k: v for k, v in novelty.items() if k <= limit}
            if novelty:
                novelty_grouped.setdefault(group_key, []).append(novelty)

        # Try specific model file first
        if text_image_embedding_model:
            noun_path = exp_dir / f"noun_similarity_over_time_{current_nounlist_name}{text_image_model_suffix}.json"
        else:
            noun_path = exp_dir / f"noun_similarity_over_time_{current_nounlist_name}.json"
            
        if noun_path.exists():
            noun = _load_trajectory_metric(noun_path, "mean_max_similarity")
            # Truncate at num_agents
            noun = {k: v for k, v in noun.items() if k <= limit}
            if noun:
                noun_grouped.setdefault(group_key, []).append(noun)

    # Derive scalars strictly from the truncated trajectories
    for group_key, runs in novelty_grouped.items():
        scalars = []
        for run in runs:
            if run:
                scalars.append(run[max(run.keys())])
        if scalars:
            mpd_scalar_grouped[group_key] = scalars

    for group_key, runs in noun_grouped.items():
        scalars = []
        for run in runs:
            if run:
                scalars.append(run[max(run.keys())])
        if scalars:
            noun_scalar_grouped[group_key] = scalars

    # Use image model suffix for novelty plots, text-image model suffix for noun plots
    # If mixed, we append both or stick to one convention. 
    # Let's append suffixes specific to the metric.

    def _compute_max_x(grouped_runs: Dict[Any, List[Dict[int, float]]]) -> int:
        max_x = 0
        for runs in grouped_runs.values():
            if not runs:
                continue
            index_sets = [set(run.keys()) for run in runs]
            common = set.intersection(*index_sets) if len(index_sets) > 1 else index_sets[0]
            if common:
                max_x = max(max_x, max(common))
        return max_x

    def _extract_baseline_scalars(baselines: List[Tuple[str, Dict[int, float]]], limit_x: int) -> List[Tuple[str, float]]:
        scalars = []
        for label, traj in baselines:
            valid_indices = [i for i in traj.keys() if i <= limit_x]
            if valid_indices:
                idx = max(valid_indices)
                scalars.append((label, traj[idx]))
        return scalars
    
    max_x_novelty = _compute_max_x(novelty_grouped)
    baseline_scalars_novelty = _extract_baseline_scalars(baselines_novelty, max_x_novelty) if max_x_novelty > 0 else []
    baseline_scalars_novelty.extend(extra_scalars_novelty)

    max_x_noun = _compute_max_x(noun_grouped)
    baseline_scalars_noun = _extract_baseline_scalars(baselines_noun, max_x_noun) if max_x_noun > 0 else []
    baseline_scalars_noun.extend(extra_scalars_noun)
    
    if len(used_nounlists) == 1:
        nounlist_name_for_file = list(used_nounlists)[0]
    else:
        nounlist_name_for_file = "mixed"

    agg_plot_distance_path = output_dir / f"aggregate_embedding_mean_pairwise_distance_over_time_{filename_tag}{image_model_suffix}.png"
    write_aggregate_plot(
        grouped_runs=novelty_grouped,
        outpath=agg_plot_distance_path,
        title="Embedding diversity over time (mean±std across seeds)",
        xlabel="Archive insertion order",
        ylabel="Mean pairwise distance",
        baselines=baselines_novelty,
    )
    agg_plot_noun_path = output_dir / f"aggregate_noun_similarity_over_time_{nounlist_name_for_file}_{filename_tag}{text_image_model_suffix}.png"
    write_aggregate_plot(
        grouped_runs=noun_grouped,
        outpath=agg_plot_noun_path,
        title="Noun similarity over time (mean±std across seeds)",
        xlabel="Archive insertion order",
        ylabel="Mean max cosine similarity",
        baselines=baselines_noun,
    )

    write_scalar_bar_plot(
        grouped_values=noun_scalar_grouped,
        outpath=output_dir / f"aggregate_noun_similarity_mean_bar_{nounlist_name_for_file}_{filename_tag}{text_image_model_suffix}.png",
        title="Mean max noun similarity (mean±std across seeds)",
        ylabel="Mean of per-noun max cosine similarity",
        baselines=baseline_scalars_noun,
    )

    write_scalar_bar_plot(
        grouped_values=mpd_scalar_grouped,
        outpath=output_dir / f"aggregate_mean_pairwise_distance_mean_bar_{filename_tag}{image_model_suffix}.png",
        title="Mean pairwise distance (mean±std across seeds)",
        ylabel="Mean pairwise distance (euclidean)",
        baselines=baseline_scalars_novelty,
    )


def _plot_tree_metrics_aggregates(
    *,
    run_configs: Sequence[PicbreederConfig],
    output_dir: Path,
    filename_tag: str,
) -> None:
    sackin_grouped: Dict[Tuple[Tuple[str, Any], ...], List[float]] = {}
    colless_grouped: Dict[Tuple[Tuple[str, Any], ...], List[float]] = {}
    depth_grouped: Dict[Tuple[Tuple[str, Any], ...], List[float]] = {}

    missing_metrics = []

    for run_cfg in run_configs:
        group_key = _group_key_for_aggregate(run_cfg)
        metrics_path = Path(run_cfg.experiment_dir) / "archive" / "phylogeny_metrics.json"
        
        if metrics_path.exists():
            try:
                m = json.loads(metrics_path.read_text(encoding="utf-8"))
                if m.get("sackin_index") is not None:
                    sackin_grouped.setdefault(group_key, []).append(float(m["sackin_index"]))
                if m.get("colless_index") is not None:
                    colless_grouped.setdefault(group_key, []).append(float(m["colless_index"]))
                if m.get("max_depth") is not None:
                    depth_grouped.setdefault(group_key, []).append(float(m["max_depth"]))
            except Exception as e:
                print(f"Error reading metrics from {metrics_path}: {e}")
        else:
            missing_metrics.append(metrics_path)

    if missing_metrics:
        print(f"Warning: Missing tree metrics for {len(missing_metrics)} runs (out of {len(run_configs)}).")
        # Only print specific paths if just a few are missing, to avoid spam
        if len(missing_metrics) < 5:
            for p in missing_metrics:
                print(f"  Missing: {p}")

    # Load human baseline if available
    human_metrics_path = Path("figures/lineages/lineage_phylogeny_metrics.json")
    human_baselines_sackin = []
    human_baselines_colless = []
    human_baselines_depth = []

    if run_configs:
         # Random Baseline
         # Check unique thumb sizes and nounlists
         unique_sizes = sorted(list({cfg.thumb_size for cfg in run_configs}))
         used_nounlists = sorted(list({Path(cfg.nounlist).stem for cfg in run_configs}))
         limit = run_configs[0].num_agents
         
         for size in unique_sizes:
             label_suffix = f" ({size}px)" if len(unique_sizes) > 1 else ""
             
             for nl_name in used_nounlists:
                 nl_suffix = f" {nl_name}" if len(used_nounlists) > 1 else ""
                 
                 rb_configs = _get_random_baseline_configs(limit, size, nl_name, Path(get_original_cwd()))
                 if rb_configs:
                     # Tree Metrics
                     val = _compute_mean_scalar(rb_configs, "sackin")
                     if val is not None:
                         human_baselines_sackin.append((f"Random Baseline{label_suffix}", val))
                     
                     val = _compute_mean_scalar(rb_configs, "colless")
                     if val is not None:
                         human_baselines_colless.append((f"Random Baseline{label_suffix}", val))
                         
                     val = _compute_mean_scalar(rb_configs, "depth")
                     if val is not None:
                         human_baselines_depth.append((f"Random Baseline{label_suffix}", val))

    if human_metrics_path.exists() and run_configs:
         try:
            hm = json.loads(human_metrics_path.read_text(encoding="utf-8"))
            # hm is keyed by limit string "500", "750", etc.
            target_limit = str(run_configs[0].num_agents)
            
            if target_limit in hm:
                m = hm[target_limit]
                if m.get("sackin_index") is not None:
                    human_baselines_sackin.append(("Human Baseline", float(m["sackin_index"])))
                if m.get("colless_index") is not None:
                    human_baselines_colless.append(("Human Baseline", float(m["colless_index"])))
                if m.get("max_depth") is not None:
                    human_baselines_depth.append(("Human Baseline", float(m["max_depth"])))
         except Exception as e:
             print(f"Error reading human baseline metrics: {e}")

    # Plot aggregates
    if any(len(v) > 0 for v in sackin_grouped.values()):
        write_scalar_bar_plot(
            grouped_values=sackin_grouped,
            outpath=output_dir / f"aggregate_sackin_index_{filename_tag}.png",
            title="Sackin Index (Tree Balance) (mean±std across seeds)",
            ylabel="Sackin Index (lower is more balanced)",
            baselines=human_baselines_sackin,
        )
    
    if any(len(v) > 0 for v in colless_grouped.values()):
        write_scalar_bar_plot(
            grouped_values=colless_grouped,
            outpath=output_dir / f"aggregate_colless_index_{filename_tag}.png",
            title="Colless Index (Tree Balance) (mean±std across seeds)",
            ylabel="Colless Index (lower is more balanced)",
            baselines=human_baselines_colless,
        )
        
    if any(len(v) > 0 for v in depth_grouped.values()):
        write_scalar_bar_plot(
            grouped_values=depth_grouped,
            outpath=output_dir / f"aggregate_tree_depth_{filename_tag}.png",
            title="Max Tree Depth (mean±std across seeds)",
            ylabel="Max Depth",
            baselines=human_baselines_depth,
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
    
    if hasattr(cfg, "items"):
        cfg_dict = {k: v for k, v in cfg.items() if hasattr(PicbreederConfig, k)}
    else:
        cfg_dict = {k: v for k, v in asdict(cfg).items() if hasattr(PicbreederConfig, k)}

    for combo in product(*(values for _, values in sweep_axes)):
        updates = {name: value for (name, _), value in zip(sweep_axes, combo)}
        configs.append(replace(SweepConfig(**cfg_dict), **updates))
    return configs


def _build_run_config(cfg: SweepConfig, original_cwd: Path) -> PicbreederConfig:
    """Create a per-run config, letting collaborative_multi_agent name directories."""
    base_kwargs = {field_def.name: getattr(cfg, field_def.name) for field_def in fields(PicbreederConfig) if field_def.name != "hydra"}
    base_cfg = PicbreederConfig(**base_kwargs)
    per_run_cfg = replace(base_cfg, experiment_dir=None, resume=False)
    validated_cfg = ensure_valid_config(per_run_cfg, original_cwd=original_cwd)
    exp_name = Path(validated_cfg.experiment_dir).name
    exp_dir = _ensure_absolute(os.path.join(cfg.log_dir, 'sweep'), original_cwd) / exp_name
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
    if cfg.gpu:
        # Check if any config uses a Qwen model, which requires specific GPUs (rtx8000)
        # use_qwen = any("qwen" in run_cfg.model.lower() for run_cfg in configs)
        gres_params = {'slurm_gres': 'gpu:1'}
        # if use_qwen:
        #     gres_params['slurm_constraint'] = 'rtx8000'

        executor.update_parameters(**gres_params)

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
    filename_tag = sanitize_filename_tag(cfg.sweep_name)

    if cfg.cross_eval:
        from render_embedding_metrics_table import DEFAULT_METRICS_FILENAME, render_tables

        cross_eval_dir.mkdir(parents=True, exist_ok=True)

        _plot_seed_aggregates(
            run_configs=run_configs,
            output_dir=cross_eval_dir,
            filename_tag=filename_tag,
            image_embedding_model=cfg.image_embedding_model,
            text_image_embedding_model=cfg.text_image_embedding_model,
        )

        _plot_tree_metrics_aggregates(
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
        varying_fields = compute_varying_fields(group_keys)
        group_labels: Dict[str, str] = {}
        for run_cfg, group_key in zip(existing_configs, group_keys):
            exp_name = Path(run_cfg.experiment_dir).name
            label = format_group_label(group_key, varying_fields)
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

    if cfg.eval:
        # We run the plotting phases in separate processes to ensure clean memory (especially VRAM/TF) usage.
        import torch
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        ctx = multiprocessing.get_context("spawn")
        
        # Phase 1: Novelty
        p1 = ctx.Process(
            target=_run_eval_phase_1,
            args=(cfg, run_configs, original_cwd, device_str)
        )
        p1.start()
        p1.join()
        
        if p1.exitcode != 0:
            print(f"Phase 1 failed with exit code {p1.exitcode}")
            return

        # Phase 2: Noun Similarity
        p2 = ctx.Process(
            target=_run_eval_phase_2,
            args=(cfg, run_configs, original_cwd, device_str)
        )
        p2.start()
        p2.join()

        if p2.exitcode != 0:
            print(f"Phase 2 failed with exit code {p2.exitcode}")
            return

        # Plot the result across seeds here as well, for convenience.
        _plot_seed_aggregates(
            run_configs=run_configs,
            output_dir=cross_eval_dir,
            filename_tag=filename_tag,
            image_embedding_model=cfg.image_embedding_model,
            text_image_embedding_model=cfg.text_image_embedding_model,
        )

        return

    if cfg.render_archive or cfg.render_tree:
        # Match the launch_locally pattern: instantiate the relevant Config objects
        # and call their main functions directly (no subprocess).
        from dataclasses import fields as dataclass_fields
        import shutil

        # Helper to compute group label for a run (used for organization)
        group_keys = [_group_key_for_aggregate(rc) for rc in run_configs]
        varying_fields = compute_varying_fields(group_keys)

        previous_cwd = Path.cwd()
        os.chdir(original_cwd)
        try:
            if cfg.render_archive:
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
                eval_cfg0 = EmbedVisualizeConfig(**base_kwargs0, embedding_model=cfg.image_embedding_model, pretrained=cfg.image_pretrained)
                eval_model, eval_preprocess = prepare_eval_clip(eval_cfg0, device)

                for run_cfg, group_key in zip(run_configs, group_keys):
                    base_kwargs = {
                        field_def.name: getattr(run_cfg, field_def.name)
                        for field_def in dataclass_fields(PicbreederConfig)
                        if field_def.name != "hydra"
                    }
                    eval_cfg = EmbedVisualizeConfig(**base_kwargs, embedding_model=cfg.image_embedding_model, pretrained=cfg.image_pretrained)
                    
                    # Enforce archive limit = num_agents
                    eval_cfg = replace(eval_cfg, archive_limit=run_cfg.num_agents)
                    
                    desc = _format_run_prefix(run_cfg, "[local-eval]")
                    print(f"{desc} -> embed_and_visualize limit={eval_cfg.archive_limit}")
                    
                    _call_hydra_wrapped_main(
                        embed_main,
                        eval_cfg,
                        model=eval_model,
                        preprocess=eval_preprocess,
                    )

                    # Copy results to cross_eval
                    label = format_group_label(group_key, varying_fields)
                    # Sanitize label for directory usage
                    sanitized_label = label.replace(" ", "_").replace("=", "_").replace(":", "_").replace("/", "-")
                    dest_dir = cross_eval_dir / sanitized_label
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    
                    model_name_sanitized = cfg.image_embedding_model.replace("/", "-")
                    
                    # Files to copy
                    src_files = [
                        Path(run_cfg.experiment_dir) / f"embed_viz_{model_name_sanitized}_{eval_cfg.method}.pdf",
                        Path(run_cfg.experiment_dir) / f"embed_grid_rect_{model_name_sanitized}_{eval_cfg.method}.pdf",
                        Path(run_cfg.experiment_dir) / f"embed_grid_representative_{model_name_sanitized}_{eval_cfg.method}.pdf",
                        Path(run_cfg.experiment_dir) / f"embed_grid_representative_simple_{model_name_sanitized}_{eval_cfg.method}.pdf",
                        Path(run_cfg.experiment_dir) / f"embed_grid_uniform_interval_{model_name_sanitized}_{eval_cfg.method}.pdf",
                        Path(run_cfg.experiment_dir) / f"embed_grid_uniform_random_{model_name_sanitized}_{eval_cfg.method}.pdf",
                    ]
                    
                    for src in src_files:
                        if src.exists():
                            # Append seed to filename
                            dest_name = f"{src.stem}_seed{run_cfg.seed}{src.suffix}"
                            shutil.copy2(src, dest_dir / dest_name)
                            print(f"Copied {src.name} -> {dest_dir / dest_name}")

            else:
                from visualize_archive_phylogeny import ArchivePhylogenyConfig, main as viz_main

                for run_cfg, group_key in zip(run_configs, group_keys):
                    base_kwargs = {
                        field_def.name: getattr(run_cfg, field_def.name)
                        for field_def in dataclass_fields(PicbreederConfig)
                        if field_def.name != "hydra"
                    }
                    viz_cfg = ArchivePhylogenyConfig(**base_kwargs)
                    
                    # Enforce archive limit = num_agents
                    viz_cfg = replace(viz_cfg, archive_limit=run_cfg.num_agents)
                    
                    desc = _format_run_prefix(run_cfg, "[local-viz]")
                    print(f"{desc} -> visualize_archive_phylogeny limit={viz_cfg.archive_limit}")
                    
                    _call_hydra_wrapped_main(viz_main, viz_cfg)

                    # Copy results to cross_eval
                    label = format_group_label(group_key, varying_fields)
                    # Sanitize label for directory usage
                    sanitized_label = label.replace(" ", "_").replace("=", "_").replace(":", "_").replace("/", "-")
                    dest_dir = cross_eval_dir / sanitized_label
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Files to copy (default format is pdf)
                    src_file = Path(run_cfg.experiment_dir) / "archive" / f"archive_phylogeny.{viz_cfg.format}"
                    
                    if src_file.exists():
                        dest_name = f"archive_phylogeny_seed{run_cfg.seed}.{viz_cfg.format}"
                        shutil.copy2(src_file, dest_dir / dest_name)
                        print(f"Copied {src_file.name} -> {dest_dir / dest_name}")
                    else:
                        print(f"Warning: Expected output {src_file} not found.")

                # Plot aggregates
                _plot_tree_metrics_aggregates(
                    run_configs=run_configs,
                    output_dir=cross_eval_dir,
                    filename_tag=filename_tag,
                )

        finally:
            os.chdir(previous_cwd)
    elif cfg.slurm:
        launch_slurm(cfg, log_dir, run_configs)
    else:
        launch_locally(run_configs)


if __name__ == "__main__":
    main()
