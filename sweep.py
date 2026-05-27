#!/usr/bin/env python3
"""Launch Picbreeder-VLM collaborative_multi_agent sweeps locally or via Submitit."""

from __future__ import annotations

import os
import re
import sys
import json
import signal
import multiprocessing
import numpy as np
from dataclasses import replace, fields, asdict
from itertools import product
from pathlib import Path
import sys
from typing import Dict, List, Sequence, Optional, Tuple, Any, Union

import hydra
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_original_cwd
import omegaconf
import submitit  # Do not remove this.

from collaborative_multi_agent import run as run_collaborative
from config import PicbreederConfig, ensure_valid_config
from sweep_configs import _NAMED_SWEEPS, RandBaselineSweep, SweepConfig
from utils import resolve_nounlist
from sweep_analysis_utils import (
    sanitize_filename_tag,
    normalize_group_value,
    compute_varying_fields,
    format_group_label,
    load_human_baseline,
    write_aggregate_plot,
    write_aggregate_bar_chart,
    write_scalar_bar_plot,
    write_combined_plot_and_bar,
)


SCRIPT_ROOT = Path(__file__).resolve().parent


def _format_process_exit(eval_name: str, exitcode: Optional[int]) -> str:
    """Build a more actionable error message for subprocess failures."""
    if exitcode is None:
        return f"{eval_name} did not report an exit code."
    if exitcode == 0:
        return f"{eval_name} completed successfully."
    if exitcode > 0:
        return f"{eval_name} failed with exit status {exitcode}."

    sig_num = -exitcode
    try:
        sig_name = signal.Signals(sig_num).name
    except ValueError:
        sig_name = "UNKNOWN"

    msg = f"{eval_name} was terminated by signal {sig_num} ({sig_name})."
    if sig_num == signal.SIGKILL:
        msg += " This is commonly caused by the OS/scheduler killing the process (often out-of-memory)."
    elif sig_num == signal.SIGTERM:
        msg += " This often indicates an external timeout, preemption, or manual termination."
    return msg


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


def _run_eval_visual_coverage(cfg: SweepConfig, run_configs: Sequence[PicbreederConfig], original_cwd: Path, device_str: str):
    import torch
    import os
    from pathlib import Path
    from dataclasses import fields as dataclass_fields, replace
    from compute_visual_coverage import (
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
    print(f"[Visual Coverage Worker] Using device: {device}")

    base_kwargs0 = {
        field_def.name: getattr(run_configs[0], field_def.name)
        for field_def in dataclass_fields(PicbreederConfig)
        if field_def.name != "hydra"
    }
    
    print("\n[Visual Coverage] Evaluating visual novelty...")
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


def _run_eval_noun_coverage(cfg: SweepConfig, run_configs: Sequence[PicbreederConfig], original_cwd: Path, device_str: str, cross_eval_dir: Path):
    import torch
    import os
    import shutil
    from pathlib import Path
    from collections import defaultdict
    from dataclasses import fields as dataclass_fields, replace
    from compute_noun_coverage import (
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
    print(f"[Noun Coverage Worker] Using device: {device}")

    # Pre-compute varying fields for labeling
    all_group_keys = [_group_key_for_aggregate(rc) for rc in run_configs]
    varying_fields = compute_varying_fields(all_group_keys)

    base_kwargs0 = {
        field_def.name: getattr(run_configs[0], field_def.name)
        for field_def in dataclass_fields(PicbreederConfig)
        if field_def.name != "hydra"
    }

    print("\n[Noun Coverage] Evaluating noun similarity...")
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

        nouns_list, prompts_list, noun_text_embeddings, neg_embeddings = prepare_noun_text_embeddings(
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
                neg_embeddings=neg_embeddings,
                original_cwd_override=original_cwd,
            )

            if cfg.render_noun_grids:
                # Copy rendered grid to cross_eval
                group_key = _group_key_for_aggregate(run_cfg)
                label, _, _ = format_group_label(group_key, varying_fields)
                sanitized_label = label.replace(" ", "_").replace("=", "_").replace(":", "_").replace("/", "-")
                dest_dir = cross_eval_dir / sanitized_label
                dest_dir.mkdir(parents=True, exist_ok=True)
                
                pattern = f"noun_similarity_grid_{nounlist_name}_{model_name_sanitized}*.pdf"
                src_files = list(exp_dir.glob(pattern))
                
                for src in src_files:
                    dest_name = f"{src.stem}_seed{run_cfg.seed}{src.suffix}"
                    shutil.copy2(src, dest_dir / dest_name)
                    print(f"Copied {src.name} -> {dest_dir / dest_name}")


def _run_eval_captions(cfg: SweepConfig, run_configs: Sequence[PicbreederConfig], original_cwd: Path):
    import os
    import gc
    import torch
    from pathlib import Path
    from dataclasses import fields as dataclass_fields
    from caption_and_embed_archive import (
        CaptionEmbedConfig,
        run_captioning_phase,
        run_embedding_phase,
        load_embedding_model,
    )
    from vlm_backends import create_vlm_backend
    from config import PicbreederConfig

    os.chdir(original_cwd)
    print("\n[Phase 3] Captioning and embedding archives...")
    
    # 1. Captioning Phase
    print(f"Initializing Caption Model ({cfg.caption_model})...")
    # Note: create_vlm_backend usually wraps API clients, but for local models (e.g. Qwen) it might load weights.
    # If it loads weights, doing it once here is beneficial.
    # However, create_vlm_backend returns a VLMBackend instance.
    caption_backend = create_vlm_backend(cfg.caption_model, max_model_len=10_000)
    
    for run_cfg in run_configs:
        exp_dir = Path(run_cfg.experiment_dir)
        archive_path = exp_dir / "archive"
        
        # Build config
        base_kwargs = {
            field_def.name: getattr(run_cfg, field_def.name)
            for field_def in dataclass_fields(PicbreederConfig)
            if field_def.name != "hydra"
        }
        
        caption_cfg = CaptionEmbedConfig(
            **base_kwargs,
            archive_path=str(archive_path),
            caption_model=cfg.caption_model,
            embedding_model=cfg.caption_embedding_model,
            embedding_pretrained=cfg.caption_embedding_pretrained,
            max_images=cfg.archive_limit if cfg.archive_limit is not None else run_cfg.num_agents,
            render_grid=True,
            grid_thumb_size=128
        )
        
        desc = _format_run_prefix(run_cfg, "[captioning]")
        print(f"{desc}")
        try:
            run_captioning_phase(caption_cfg, backend=caption_backend)
        except Exception as e:
            print(f"Error in captioning phase for {exp_dir}: {e}")

    # Unload caption model
    del caption_backend
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 2. Embedding Phase
    print(f"Initializing Embedding Model ({cfg.caption_embedding_model})...")
    try:
        embed_model_dict = load_embedding_model(cfg.caption_embedding_model, cfg.caption_embedding_pretrained)
    except Exception as e:
        print(f"Failed to load embedding model: {e}")
        return

    for run_cfg in run_configs:
        exp_dir = Path(run_cfg.experiment_dir)
        archive_path = exp_dir / "archive"
        
        base_kwargs = {
            field_def.name: getattr(run_cfg, field_def.name)
            for field_def in dataclass_fields(PicbreederConfig)
            if field_def.name != "hydra"
        }
        
        caption_cfg = CaptionEmbedConfig(
            **base_kwargs,
            archive_path=str(archive_path),
            caption_model=cfg.caption_model,
            embedding_model=cfg.caption_embedding_model,
            embedding_pretrained=cfg.caption_embedding_pretrained,
            max_images=cfg.archive_limit if cfg.archive_limit is not None else run_cfg.num_agents,
        )
        
        desc = _format_run_prefix(run_cfg, "[embedding]")
        print(f"{desc}")
        run_embedding_phase(caption_cfg, embed_model=embed_model_dict)


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
    
    iterator = data.values() if isinstance(data, dict) else data
    
    for row in iterator:
        if not isinstance(row, dict):
            continue
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
            last = None
            if isinstance(traj, list) and traj:
                 last = traj[-1]
            elif isinstance(traj, dict) and traj:
                 max_k = -1
                 for k, v in traj.items():
                     try:
                        ki = int(k)
                        if ki > max_k:
                            max_k = ki
                            last = v
                     except ValueError:
                        continue
            
            if last is None or not isinstance(last, dict):
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
    k: Optional[int] = None,
    caption_model_name: Optional[str] = None,
    caption_embedding_model: Optional[str] = None,
    negative_anchors_name: Optional[str] = None,
) -> Optional[Dict[int, float]]:
    trajectories = []
    for run_cfg in run_configs:
        exp_dir = Path(run_cfg.experiment_dir)
        limit = run_cfg.num_agents
        
        if metric_type == "novelty" or metric_type == "visual_k_covering":
            suffix = ""
            if model_name:
                sanitized = model_name.replace("/", "-")
                suffix = f"_{sanitized}"
            
            path = exp_dir / f"embedding_mean_pairwise_distance_over_time{suffix}.json"
            if path.exists():
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    iterator = data.values() if isinstance(data, dict) else data
                    traj = {}
                    for row in iterator:
                        if not isinstance(row, dict): continue
                        idx = row.get("index")
                        if idx is None: continue
                        try:
                            idx_int = int(idx)
                        except ValueError: continue
                        
                        if idx_int > limit: continue
                        
                        if metric_type == "novelty":
                            val = row.get("mean_pairwise_distance")
                            if val is not None:
                                traj[idx_int] = float(val)
                        elif metric_type == "visual_k_covering":
                            if k is None: continue
                            radii = row.get("k_covering_radii")
                            if isinstance(radii, dict):
                                val = radii.get(str(k))
                                if val is not None:
                                    traj[idx_int] = float(val)
                    if traj:
                        trajectories.append(traj)
                except Exception:
                    pass

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
                traj = {key: val for key, val in traj.items() if key <= limit}
                if traj:
                    trajectories.append(traj)

        elif metric_type == "noun_contrastive":
            suffix = ""
            sanitized = model_name.replace("/", "-")
            suffix = f"_{sanitized}"
            
            path = exp_dir / f"noun_similarity_over_time_{nounlist_name}{suffix}.json"
            traj = _load_trajectory_metric(path, f"mean_max_contrastive_{negative_anchors_name}")
            traj = {key: val for key, val in traj.items() if key <= limit}
            if traj:
                trajectories.append(traj)

        elif metric_type == "noun_per_image":
            if not nounlist_name:
                continue
            suffix = ""
            if model_name:
                sanitized = model_name.replace("/", "-")
                suffix = f"_{sanitized}"
            
            path = exp_dir / f"noun_similarity_over_time_{nounlist_name}{suffix}.json"
            if path.exists():
                traj = _load_trajectory_metric(path, "mean_max_per_image_similarity")
                traj = {key: val for key, val in traj.items() if key <= limit}
                if traj:
                    trajectories.append(traj)

        elif metric_type == "caption_diversity":
             if not caption_model_name or not caption_embedding_model:
                 continue
             
             embed_sanitized = caption_embedding_model.replace("/", "-")
             caption_suffix = f"_{caption_model_name}_{embed_sanitized}"
             path = exp_dir / "archive" / f"metrics{caption_suffix}.json"
             
             if path.exists():
                 try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    traj = {}
                    for idx_str, info in data.items():
                        try:
                            idx_int = int(idx_str)
                        except ValueError: continue
                        if idx_int > limit: continue
                        
                        val = info.get("mean_pairwise_distance")
                        if val is not None:
                             traj[idx_int] = float(val)
                    if traj:
                        trajectories.append(traj)
                 except Exception:
                     pass
        
        elif metric_type == "caption_k_covering":
             if k is None or not caption_model_name or not caption_embedding_model:
                 continue
             
             embed_sanitized = caption_embedding_model.replace("/", "-")
             caption_suffix = f"_{caption_model_name}_{embed_sanitized}"
             path = exp_dir / "archive" / f"metrics{caption_suffix}.json"
             
             if path.exists():
                 try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    # data format: { "1": { "mean_pairwise_distance": ..., "k_covering_radii": { "10": ... } }, ... }
                    traj = {}
                    for idx_str, info in data.items():
                        try:
                            idx_int = int(idx_str)
                        except ValueError: continue
                        if idx_int > limit: continue
                        
                        radii = info.get("k_covering_radii")
                        if isinstance(radii, dict):
                            val = radii.get(str(k))
                            if val is not None:
                                traj[idx_int] = float(val)
                    if traj:
                        trajectories.append(traj)
                 except Exception:
                     pass

    if not trajectories:
        return None

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
        elif metric_type in ("sackin", "colless", "depth", "j1"):
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
                    elif metric_type == "j1":
                        val = m.get("j1_index")
                 except Exception:
                     pass
        
        if val is not None:
            values.append(float(val))
            
    if not values:
        return None
        
    return float(np.mean(values))


def _plot_seed_aggregates(
    run_configs: Sequence[PicbreederConfig],
    output_dir: Path,
    filename_tag: str,
    image_embedding_model: str,
    text_image_embedding_model: str,
    caption_model: str,
    caption_embedding_model: str,
    negative_anchors: str,
    novelty_ylim: Optional[List[float]] = None,
    noun_ylim: Optional[List[float]] = None,
) -> None:
    novelty_grouped: Dict[Tuple[Tuple[str, Any], ...], List[Dict[int, float]]] = {}
    noun_grouped: Dict[Tuple[Tuple[str, Any], ...], List[Dict[int, float]]] = {}
    noun_per_image_grouped: Dict[Tuple[Tuple[str, Any], ...], List[Dict[int, float]]] = {}
    noun_contrastive_grouped: Dict[Tuple[Tuple[str, Any], ...], List[Dict[int, float]]] = {}
    caption_grouped: Dict[Tuple[Tuple[str, Any], ...], List[Dict[int, float]]] = {}
    caption_k_covering_grouped: Dict[str, Dict[Tuple[Tuple[str, Any], ...], List[Dict[int, float]]]] = {}
    visual_k_covering_grouped: Dict[str, Dict[Tuple[Tuple[str, Any], ...], List[Dict[int, float]]]] = {}
    noun_scalar_grouped: Dict[Tuple[Tuple[str, Any], ...], List[float]] = {}
    noun_contrastive_scalar_grouped: Dict[Tuple[Tuple[str, Any], ...], List[float]] = {}
    noun_per_image_scalar_grouped: Dict[Tuple[Tuple[str, Any], ...], List[float]] = {}
    mpd_scalar_grouped: Dict[Tuple[Tuple[str, Any], ...], List[float]] = {}

    final_novelty_scores: Dict[Tuple[Tuple[str, Any], ...], List[Tuple[int, float]]] = {}
    final_noun_scores: Dict[Tuple[Tuple[str, Any], ...], List[Tuple[int, float]]] = {}
    final_caption_scores: Dict[Tuple[Tuple[str, Any], ...], List[Tuple[int, float]]] = {}
    
    final_visual_k_scores: Dict[int, Dict[Tuple[Tuple[str, Any], ...], List[Tuple[int, float]]]] = {}
    final_caption_k_scores: Dict[int, Dict[Tuple[Tuple[str, Any], ...], List[Tuple[int, float]]]] = {}

    image_model_suffix = ""
    if image_embedding_model:
        sanitized = image_embedding_model.replace("/", "-")
        image_model_suffix = f"_{sanitized}"

    text_image_model_suffix = ""
    if text_image_embedding_model:
        sanitized = text_image_embedding_model.replace("/", "-")
        text_image_model_suffix = f"_{sanitized}"
        
    caption_suffix = ""
    if caption_model and caption_embedding_model:
        embed_sanitized = caption_embedding_model.replace("/", "-")
        caption_suffix = f"_{caption_model}_{embed_sanitized}"

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
    baselines_noun_per_image: List[Tuple[str, Dict[int, float]]] = []
    baselines_noun_contrastive: List[Tuple[str, Dict[int, float]]] = []
    baselines_caption_diversity: List[Tuple[str, Dict[int, float]]] = []
    
    # Store random baseline scalars separately to merge later
    extra_scalars_novelty: List[Tuple[str, float]] = []
    extra_scalars_noun: List[Tuple[str, float]] = []
    
    if run_configs:
        baseline_model_novelty = image_embedding_model if image_embedding_model else "ViT-B-32"
        baseline_model_noun = text_image_embedding_model if text_image_embedding_model else "ViT-H-14"
        
        neg_anchors = negative_anchors

        for size in unique_sizes:
            label_suffix = f" ({size}px)" if len(unique_sizes) > 1 else ""
            
            bn = load_human_baseline("novelty", size, baseline_model_novelty)
            if bn:
                baselines_novelty.append((f"Human Baseline{label_suffix}", bn))
            
            for nl_name in used_nounlists:
                 nl_suffix = f" {nl_name}" if len(used_nounlists) > 1 else ""
                 bnn = load_human_baseline("noun", size, baseline_model_noun, nounlist=nl_name, negative_anchors=neg_anchors)
                 if bnn:
                     baselines_noun.append((f"Human Baseline{label_suffix}{nl_suffix}", bnn))
                 
                 bnnc = load_human_baseline("noun_contrastive", size, baseline_model_noun, nounlist=nl_name, strict=False, negative_anchors=neg_anchors)
                 if bnnc:
                     baselines_noun_contrastive.append((f"Human Baseline{label_suffix}{nl_suffix}", bnnc))

                 bnnpi = load_human_baseline("noun_per_image", size, baseline_model_noun, nounlist=nl_name, strict=False, negative_anchors=neg_anchors)
                 if bnnpi:
                     baselines_noun_per_image.append((f"Human Baseline{label_suffix}{nl_suffix}", bnnpi))
            
            # Human Baseline for Caption Diversity
            if caption_model:
                 hbc = load_human_baseline("caption_diversity", size, caption_model, strict=False)
                 if hbc:
                     baselines_caption_diversity.append((f"Human Baseline{label_suffix}", hbc))

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
                        baselines_novelty.append((f"Random Baseline{label_suffix}", traj_nov))
                        # else:
                        #     scalar_nov = _compute_mean_scalar(rb_configs, "novelty", image_embedding_model)
                        #     if scalar_nov is not None:
                        #         extra_scalars_novelty.append((f"Random Baseline{label_suffix}", scalar_nov))

                        # Caption Diversity
                        if caption_model and caption_embedding_model:
                             traj_cd = _compute_mean_trajectory(
                                 rb_configs, 
                                 "caption_diversity", 
                                 caption_model_name=caption_model, 
                                 caption_embedding_model=caption_embedding_model
                             )
                             if traj_cd:
                                 baselines_caption_diversity.append((f"Random Baseline{label_suffix}", traj_cd))

                        # Noun
                        traj_noun = _compute_mean_trajectory(rb_configs, "noun", text_image_embedding_model, nl_name, negative_anchors_name=negative_anchors)
                        if traj_noun:
                            baselines_noun.append((f"Random Baseline{label_suffix}{nl_suffix}", traj_noun))

                        # Noun Contrastive
                        neg_stem = Path(neg_anchors).stem if neg_anchors else None
                        if neg_stem:
                            traj_noun_cont = _compute_mean_trajectory(rb_configs, "noun_contrastive", text_image_embedding_model, nl_name, negative_anchors_name=neg_stem)
                            if traj_noun_cont:
                                baselines_noun_contrastive.append((f"Random Baseline{label_suffix}{nl_suffix}", traj_noun_cont))
                        
                        traj_noun_per_image = _compute_mean_trajectory(rb_configs, "noun_per_image", text_image_embedding_model, nl_name, negative_anchors_name=negative_anchors)
                        if traj_noun_per_image:
                            baselines_noun_per_image.append((f"Random Baseline{label_suffix}{nl_suffix}", traj_noun_per_image))
                        
                        # else:
                        #     scalar_noun = _compute_mean_scalar(rb_configs, "noun", text_image_embedding_model, nl_name)
                        #     if scalar_noun is not None:
                        #         extra_scalars_noun.append((f"Random Baseline{label_suffix}{nl_suffix}", scalar_noun))
                        
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
            try:
                raw_data = json.loads(novelty_path.read_text(encoding="utf-8"))
                iterator = raw_data.values() if isinstance(raw_data, dict) else raw_data
                
                mpd_traj = {}
                k_covering_trajs = {} # k -> {index -> radius}

                for row in iterator:
                    if not isinstance(row, dict): continue
                    idx_str = row.get("index")
                    if idx_str is None: continue
                    try:
                        idx = int(idx_str)
                    except ValueError: continue
                    
                    if idx > limit: continue

                    # MPD
                    mpd = row.get("mean_pairwise_distance")
                    if mpd is not None:
                        mpd_traj[idx] = float(mpd)
                    
                    # K-Covering
                    radii = row.get("k_covering_radii")
                    if isinstance(radii, dict):
                        for k, r in radii.items():
                            try:
                                k_int = int(k)
                                if k_int not in k_covering_trajs:
                                    k_covering_trajs[k_int] = {}
                                k_covering_trajs[k_int][idx] = float(r)
                            except ValueError: pass

                if mpd_traj:
                    novelty_grouped.setdefault(group_key, []).append(mpd_traj)
                    last_idx = max(mpd_traj.keys())
                    final_novelty_scores.setdefault(group_key, []).append((run_cfg.seed, mpd_traj[last_idx]))
                
                for k, traj in k_covering_trajs.items():
                    k_str = str(k)
                    if k_str not in visual_k_covering_grouped:
                        visual_k_covering_grouped[k_str] = {}
                    visual_k_covering_grouped[k_str].setdefault(group_key, []).append(traj)
                    
                    last_idx = max(traj.keys())
                    if k not in final_visual_k_scores:
                        final_visual_k_scores[k] = {}
                    final_visual_k_scores[k].setdefault(group_key, []).append((run_cfg.seed, traj[last_idx]))

            except Exception as e:
                print(f"Error reading/parsing {novelty_path}: {e}")

        # Try specific model file first
        neg_suffix = ""
        if hasattr(run_cfg, "negative_anchors") and run_cfg.negative_anchors:
             neg_suffix = f"_{Path(run_cfg.negative_anchors).stem}"
             
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
                last_idx = max(noun.keys())
                final_noun_scores.setdefault(group_key, []).append((run_cfg.seed, noun[last_idx]))
            
            noun_cont = _load_trajectory_metric(noun_path, f"mean_max_contrastive{neg_suffix}")
            noun_cont = {k: v for k, v in noun_cont.items() if k <= limit}
            if noun_cont:
                noun_contrastive_grouped.setdefault(group_key, []).append(noun_cont)
            elif noun:
                raise ValueError(f"Missing contrastive noun similarity with key {f'mean_max_contrastive{neg_suffix}'} in {noun_path} (eval_noun_coverage was likely run with an old version or incomplete data)")

            noun_per_image = _load_trajectory_metric(noun_path, "mean_max_per_image_similarity")
            noun_per_image = {k: v for k, v in noun_per_image.items() if k <= limit}
            if noun_per_image:
                 noun_per_image_grouped.setdefault(group_key, []).append(noun_per_image)


        # Try caption metrics
        if caption_model and caption_embedding_model:
            cap_metrics_path = exp_dir / "archive" / f"metrics{caption_suffix}.json"
            if cap_metrics_path.exists():
                trajectory = json.loads(cap_metrics_path.read_text(encoding="utf-8"))
                traj_points_mpd = {}
                traj_points_k = {} # Map k -> {n -> val}
                
                for n in trajectory:
                    if int(n) <= limit:
                        n_int = int(n)
                        # MPD
                        val = trajectory[n].get("mean_pairwise_distance")

                        if val is not None:
                            traj_points_mpd[n_int] = float(val)
                            
                        # K-Covering
                        k_radii = trajectory[n].get("k_covering_radii", {})
                        for k, radius in k_radii.items():
                            k_key = int(k)

                            if k_key not in traj_points_k:
                                traj_points_k[k_key] = {}
                            traj_points_k[k_key][n_int] = float(radius)
                            
                if traj_points_mpd:
                        caption_grouped.setdefault(group_key, []).append(traj_points_mpd)
                        last_idx = max(traj_points_mpd.keys())
                        final_caption_scores.setdefault(group_key, []).append((run_cfg.seed, traj_points_mpd[last_idx]))
                        
                for k, traj in traj_points_k.items():
                    if k not in caption_k_covering_grouped:
                        caption_k_covering_grouped[k] = {}
                    caption_k_covering_grouped[k].setdefault(group_key, []).append(traj)
                    
                    last_idx = max(traj.keys())
                    if k not in final_caption_k_scores:
                        final_caption_k_scores[k] = {}
                    final_caption_k_scores[k].setdefault(group_key, []).append((run_cfg.seed, traj[last_idx]))
                            
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

    for group_key, runs in noun_contrastive_grouped.items():
        scalars = []
        for run in runs:
            if run:
                scalars.append(run[max(run.keys())])
        if scalars:
            noun_contrastive_scalar_grouped[group_key] = scalars

    for group_key, runs in noun_per_image_grouped.items():
        scalars = []
        for run in runs:
            if run:
                scalars.append(run[max(run.keys())])
        if scalars:
            noun_per_image_scalar_grouped[group_key] = scalars

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
    
    max_x_noun_cont = _compute_max_x(noun_contrastive_grouped)
    baseline_scalars_noun_contrastive = _extract_baseline_scalars(baselines_noun_contrastive, max_x_noun_cont) if max_x_noun_cont > 0 else []
    
    max_x_noun_per_image = _compute_max_x(noun_per_image_grouped)
    baseline_scalars_noun_per_image = _extract_baseline_scalars(baselines_noun_per_image, max_x_noun_per_image) if max_x_noun_per_image > 0 else []
    
    if len(used_nounlists) == 1:
        nounlist_name_for_file = list(used_nounlists)[0]
    else:
        nounlist_name_for_file = "mixed"

    neg_anchors_for_file = ""
    # We need to collect used negative anchors from run_configs if available
    used_neg_anchors = set()
    if run_configs:
         for rc in run_configs:
             if hasattr(rc, "negative_anchors") and rc.negative_anchors:
                 used_neg_anchors.add(Path(rc.negative_anchors).stem)
    
    if len(used_neg_anchors) == 1:
        neg_anchors_for_file = f"_{list(used_neg_anchors)[0]}"
    elif len(used_neg_anchors) > 1:
        neg_anchors_for_file = "_mixed_neg"

    agg_plot_distance_path = output_dir / f"aggregate_embedding_mean_pairwise_distance_over_time_{filename_tag}{image_model_suffix}.png"
    write_aggregate_plot(
        grouped_runs=novelty_grouped,
        outpath=agg_plot_distance_path,
        title="Embedding diversity over time (mean±sem across seeds)",
        xlabel="Archive insertion order",
        ylabel="Mean pairwise distance",
        baselines=baselines_novelty,
        ylim=tuple(novelty_ylim) if novelty_ylim else None,
    )
    agg_plot_noun_combined_path = output_dir / f"aggregate_noun_similarity_combined_{nounlist_name_for_file}_{filename_tag}{text_image_model_suffix}.png"
    write_combined_plot_and_bar(
        grouped_runs=noun_grouped,
        grouped_values=noun_scalar_grouped,
        outpath=agg_plot_noun_combined_path,
        title="Semantic Recall (mean±sem across seeds)",
        xlabel="Archive insertion order",
        ylabel_line="Mean max cosine similarity",
        ylabel_bar="Mean max cosine similarity",
        baselines_line=baselines_noun,
        baselines_bar=baseline_scalars_noun,
        ylim=tuple(noun_ylim) if noun_ylim else None,
    )

    agg_plot_noun_cont_path = output_dir / f"aggregate_noun_contrastive_over_time_{nounlist_name_for_file}{neg_anchors_for_file}_{filename_tag}{text_image_model_suffix}.png"
    write_aggregate_plot(
        grouped_runs=noun_contrastive_grouped,
        outpath=agg_plot_noun_cont_path,
        title="Contrastive noun similarity over time (mean±sem across seeds)",
        xlabel="Archive insertion order",
        ylabel="Mean max contrastive similarity",
        baselines=baselines_noun_contrastive,
        ylim=None,
    )

    write_scalar_bar_plot(
        grouped_values=noun_contrastive_scalar_grouped,
        outpath=output_dir / f"aggregate_noun_contrastive_mean_bar_{nounlist_name_for_file}{neg_anchors_for_file}_{filename_tag}{text_image_model_suffix}.png",
        title="Mean max contrastive noun similarity (mean±sem across seeds)",
        ylabel="Mean of per-noun max contrastive similarity",
        baselines=baseline_scalars_noun_contrastive,
    )

    agg_plot_noun_per_image_path = output_dir / f"aggregate_noun_per_image_combined_{nounlist_name_for_file}_{filename_tag}{text_image_model_suffix}.png"
    write_combined_plot_and_bar(
        grouped_runs=noun_per_image_grouped,
        grouped_values=noun_per_image_scalar_grouped,
        outpath=agg_plot_noun_per_image_path,
        title="Mean Max Per-Image Noun Similarity (mean±sem across seeds)",
        xlabel="Archive insertion order",
        ylabel_line="Mean max per-image similarity",
        ylabel_bar="Mean max per-image similarity",
        baselines_line=baselines_noun_per_image,
        baselines_bar=baseline_scalars_noun_per_image,
    )

    write_scalar_bar_plot(
        grouped_values=mpd_scalar_grouped,
        outpath=output_dir / f"aggregate_mean_pairwise_distance_mean_bar_{filename_tag}{image_model_suffix}.png",
        title="Mean pairwise distance (mean±sem across seeds)",
        ylabel="Mean pairwise distance (euclidean)",
        baselines=baseline_scalars_novelty,
    )

    for k, grouped_data in visual_k_covering_grouped.items():
        k_int = int(k)
        current_baselines = []
        
        if run_configs:
            baseline_model_novelty = image_embedding_model if image_embedding_model else "ViT-B-32"
            
            # Human Baselines
            for size in unique_sizes:
                label_suffix = f" ({size}px)" if len(unique_sizes) > 1 else ""
                # Strict=True will raise Exception if baseline missing
                hb = load_human_baseline("visual_k_covering", size, baseline_model_novelty, k=k_int, strict=True)
                if hb:
                    current_baselines.append((f"Human Baseline{label_suffix}", hb))
            
            # Random Baselines
            limit = run_configs[0].num_agents
            for size in unique_sizes:
                label_suffix = f" ({size}px)" if len(unique_sizes) > 1 else ""
                # We iterate used_nounlists to find the correct random baseline run
                # (Assuming random baseline was run with one of these nounlists)
                for nl_name in used_nounlists:
                    rb_configs = _get_random_baseline_configs(limit, size, nl_name, Path(get_original_cwd()))
                    if not rb_configs:
                        raise ValueError(f"Random Baseline experiments not found for size={size}, nounlist={nl_name}")
                    
                    traj = _compute_mean_trajectory(rb_configs, "visual_k_covering", image_embedding_model, k=k_int)
                    if not traj:
                        raise ValueError(f"Missing visual_k_covering stats (k={k}) for Random Baseline (size={size})")
                    
                    current_baselines.append((f"Random Baseline{label_suffix}", traj))
                    # Only add one random baseline per size (assuming nounlist doesn't affect visual random baseline significantly)
                    break

        agg_plot_k_path = output_dir / f"aggregate_visual_k_covering_k{k}_over_time_{filename_tag}{image_model_suffix}.png"
        write_aggregate_plot(
            grouped_runs=grouped_data,
            outpath=agg_plot_k_path,
            title=f"Visual K-Covering Radius (k={k}) over time",
            xlabel="Archive insertion order",
            ylabel=f"Covering Radius (k={k})",
            baselines=current_baselines,
            ylim=None,
        )

        agg_bar_k_path = output_dir / f"aggregate_visual_k_covering_k{k}_final_{filename_tag}{image_model_suffix}.png"
        write_aggregate_bar_chart(
            grouped_runs=grouped_data,
            outpath=agg_bar_k_path,
            title=f"Final Visual K-Covering Radius (k={k})",
            ylabel=f"Covering Radius (k={k})",
            baselines=current_baselines,
            ylim=None,
        )

    if caption_grouped:
        agg_plot_caption_path = output_dir / f"aggregate_caption_diversity_over_time_{filename_tag}{caption_suffix}.png"
        write_aggregate_plot(
            grouped_runs=caption_grouped,
            outpath=agg_plot_caption_path,
            title=f"Caption Diversity (MPD) over time ({caption_model}/{caption_embedding_model})",
            xlabel="Archive insertion order",
            ylabel="Mean pairwise distance",
            baselines=baselines_caption_diversity,
            ylim=None,
        )

    for k, grouped_data in caption_k_covering_grouped.items():
        k_int = int(k)
        current_baselines = []
        
        if run_configs and caption_model:
            # Human Baselines
            for size in unique_sizes:
                label_suffix = f" ({size}px)" if len(unique_sizes) > 1 else ""
                # Strict=True will raise Exception if baseline missing
                hb = load_human_baseline("caption_k_covering", size, caption_model, k=k_int, strict=True)
                if hb:
                    current_baselines.append((f"Human Baseline{label_suffix}", hb))
            
            # Random Baselines
            limit = run_configs[0].num_agents
            for size in unique_sizes:
                label_suffix = f" ({size}px)" if len(unique_sizes) > 1 else ""
                for nl_name in used_nounlists:
                    rb_configs = _get_random_baseline_configs(limit, size, nl_name, Path(get_original_cwd()))
                    if not rb_configs:
                        raise ValueError(f"Random Baseline experiments not found for size={size}, nounlist={nl_name}")
                    
                    traj = _compute_mean_trajectory(
                        rb_configs, 
                        "caption_k_covering", 
                        k=k_int, 
                        caption_model_name=caption_model,
                        caption_embedding_model=caption_embedding_model
                    )
                    
                    if not traj:
                         raise ValueError(f"Missing caption_k_covering stats (k={k}) for Random Baseline")
                    
                    current_baselines.append((f"Random Baseline{label_suffix}", traj))
                    break

        agg_plot_k_path = output_dir / f"aggregate_caption_k_covering_k{k}_over_time_{filename_tag}{caption_suffix}.png"
        write_aggregate_plot(
            grouped_runs=grouped_data,
            outpath=agg_plot_k_path,
            title=f"Caption K-Covering Radius (k={k}) over time",
            xlabel="Archive insertion order",
            ylabel=f"Covering Radius (k={k})",
            baselines=current_baselines,
            ylim=None,
        )

        agg_bar_k_path = output_dir / f"aggregate_caption_k_covering_k{k}_final_{filename_tag}{caption_suffix}.png"
        write_aggregate_bar_chart(
            grouped_runs=grouped_data,
            outpath=agg_bar_k_path,
            title=f"Final Caption K-Covering Radius (k={k})",
            ylabel=f"Covering Radius (k={k})",
            baselines=current_baselines,
            ylim=None,
        )

    # Write best seeds JSON analysis
    all_group_keys = set(novelty_grouped.keys()) | set(noun_grouped.keys()) | set(caption_grouped.keys())
    if run_configs:
        group_keys_list = [_group_key_for_aggregate(rc) for rc in run_configs]
        varying_fields = compute_varying_fields(group_keys_list)
        
        for group_key in all_group_keys:
            label, _, _ = format_group_label(group_key, varying_fields)
            sanitized_label = label.replace(" ", "_").replace("=", "_").replace(":", "_").replace("/", "-")
            group_dir = output_dir / sanitized_label
            group_dir.mkdir(parents=True, exist_ok=True)
            
            best_seeds_data = {}
            
            if group_key in final_novelty_scores:
                seeds_vals = final_novelty_scores[group_key]
                if seeds_vals:
                    best_seed, best_val = max(seeds_vals, key=lambda x: x[1])
                    best_seeds_data["visual_coverage"] = {
                        "best_seed": best_seed,
                        "value": best_val,
                        "metric": "mean_pairwise_distance"
                    }

            if group_key in final_noun_scores:
                seeds_vals = final_noun_scores[group_key]
                if seeds_vals:
                    best_seed, best_val = max(seeds_vals, key=lambda x: x[1])
                    best_seeds_data["noun_coverage"] = {
                        "best_seed": best_seed,
                        "value": best_val,
                        "metric": "mean_max_similarity"
                    }
                    
            if group_key in final_caption_scores:
                seeds_vals = final_caption_scores[group_key]
                if seeds_vals:
                    best_seed, best_val = max(seeds_vals, key=lambda x: x[1])
                    best_seeds_data["semantic_coverage"] = {
                        "best_seed": best_seed,
                        "value": best_val,
                        "metric": "mean_pairwise_distance"
                    }
            
            # Add K-covering metrics
            for k, group_scores in final_visual_k_scores.items():
                if group_key in group_scores:
                    seeds_vals = group_scores[group_key]
                    if seeds_vals:
                        best_seed, best_val = max(seeds_vals, key=lambda x: x[1])
                        best_seeds_data[f"visual_k_covering_k{k}"] = {
                            "best_seed": best_seed,
                            "value": best_val,
                            "metric": f"k_covering_radius_k{k}"
                        }

            for k, group_scores in final_caption_k_scores.items():
                if group_key in group_scores:
                    seeds_vals = group_scores[group_key]
                    if seeds_vals:
                        best_seed, best_val = max(seeds_vals, key=lambda x: x[1])
                        best_seeds_data[f"semantic_k_covering_k{k}"] = {
                            "best_seed": best_seed,
                            "value": best_val,
                            "metric": f"k_covering_radius_k{k}"
                        }
                    
            if best_seeds_data:
                json_path = group_dir / "best_seeds.json"
                json_path.write_text(json.dumps(best_seeds_data, indent=2))
                print(f"Wrote best seeds to {json_path}")


def _plot_tree_metrics_aggregates(
    *,
    run_configs: Sequence[PicbreederConfig],
    output_dir: Path,
    filename_tag: str,
) -> None:
    sackin_grouped: Dict[Tuple[Tuple[str, Any], ...], List[float]] = {}
    colless_grouped: Dict[Tuple[Tuple[str, Any], ...], List[float]] = {}
    depth_grouped: Dict[Tuple[Tuple[str, Any], ...], List[float]] = {}
    j1_grouped: Dict[Tuple[Tuple[str, Any], ...], List[float]] = {}

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
                if m.get("j1_index") is not None:
                    j1_grouped.setdefault(group_key, []).append(float(m["j1_index"]))
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
    human_baselines_j1 = []

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

                     val = _compute_mean_scalar(rb_configs, "j1")
                     if val is not None:
                         human_baselines_j1.append((f"Random Baseline{label_suffix}", val))

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
                if m.get("j1_index") is not None:
                    human_baselines_j1.append(("Human Baseline", float(m["j1_index"])))
         except Exception as e:
             print(f"Error reading human baseline metrics: {e}")

    # Plot aggregates
    if any(len(v) > 0 for v in sackin_grouped.values()):
        write_scalar_bar_plot(
            grouped_values=sackin_grouped,
            outpath=output_dir / f"aggregate_sackin_index_{filename_tag}.png",
            title="Sackin Index (Tree Balance) (mean±sem across seeds)",
            ylabel="Sackin Index (lower is more balanced)",
            baselines=human_baselines_sackin,
        )
    
    if any(len(v) > 0 for v in colless_grouped.values()):
        write_scalar_bar_plot(
            grouped_values=colless_grouped,
            outpath=output_dir / f"aggregate_colless_index_{filename_tag}.png",
            title="Colless Index (Tree Balance) (mean±sem across seeds)",
            ylabel="Colless Index (lower is more balanced)",
            baselines=human_baselines_colless,
        )
        
    if any(len(v) > 0 for v in depth_grouped.values()):
        write_scalar_bar_plot(
            grouped_values=depth_grouped,
            outpath=output_dir / f"aggregate_tree_depth_{filename_tag}.png",
            title="Max Tree Depth (mean±sem across seeds)",
            ylabel="Max Depth",
            baselines=human_baselines_depth,
        )

    if any(len(v) > 0 for v in j1_grouped.values()):
        write_scalar_bar_plot(
            grouped_values=j1_grouped,
            outpath=output_dir / f"aggregate_j1_index_{filename_tag}.png",
            title="J1 Index (Tree Balance)",
            ylabel="J1 Index (higher is more balanced)",
            baselines=human_baselines_j1,
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

def _run_render_archive(cfg: SweepConfig, run_configs: Sequence[PicbreederConfig], original_cwd: Path, cross_eval_dir: Path):
    import torch
    import shutil
    import os
    from pathlib import Path
    from dataclasses import fields as dataclass_fields, replace
    from embed_and_visualize import (
        EmbedVisualizeConfig,
        main as embed_main,
        prepare_openclip_components as prepare_eval_clip,
    )
    from config import PicbreederConfig

    # Helper to compute group label for a run (used for organization)
    group_keys = [_group_key_for_aggregate(rc) for rc in run_configs]
    varying_fields = compute_varying_fields(group_keys)

    previous_cwd = Path.cwd()
    os.chdir(original_cwd)
    try:
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
            label, _, _ = format_group_label(group_key, varying_fields)
            # Sanitize label for directory usage
            sanitized_label = label.replace(" ", "_").replace("=", "_").replace(":", "_").replace("/", "-")
            dest_dir = cross_eval_dir / sanitized_label
            dest_dir.mkdir(parents=True, exist_ok=True)
            
            model_name_sanitized = cfg.image_embedding_model.replace("/", "-")
            
            # Files to copy
            src_files = [
                Path(run_cfg.experiment_dir) / f"embed_viz_{model_name_sanitized}_{eval_cfg.method}.png",
                Path(run_cfg.experiment_dir) / f"embed_grid_rect_{model_name_sanitized}_{eval_cfg.method}.png",
                Path(run_cfg.experiment_dir) / f"embed_grid_representative_{model_name_sanitized}_{eval_cfg.method}.png",
                Path(run_cfg.experiment_dir) / f"embed_grid_representative_simple_{model_name_sanitized}_{eval_cfg.method}.png",
                Path(run_cfg.experiment_dir) / f"embed_grid_uniform_interval_{model_name_sanitized}_{eval_cfg.method}.png",
                Path(run_cfg.experiment_dir) / f"embed_grid_uniform_random_{model_name_sanitized}_{eval_cfg.method}.png",
            ]
            
            for src in src_files:
                if src.exists():
                    # Append seed to filename
                    dest_name = f"{src.stem}_seed{run_cfg.seed}{src.suffix}"
                    shutil.copy2(src, dest_dir / dest_name)
                    print(f"Copied {src.name} -> {dest_dir / dest_name}")
    finally:
        os.chdir(previous_cwd)


def _run_eval_tree(cfg: SweepConfig, run_configs: Sequence[PicbreederConfig], original_cwd: Path, cross_eval_dir: Path, filename_tag: str):
    import shutil
    import os
    from pathlib import Path
    from dataclasses import fields as dataclass_fields, replace
    from visualize_archive_phylogeny import ArchivePhylogenyConfig, main as viz_main
    from config import PicbreederConfig

    # Helper to compute group label for a run (used for organization)
    group_keys = [_group_key_for_aggregate(rc) for rc in run_configs]
    varying_fields = compute_varying_fields(group_keys)

    previous_cwd = Path.cwd()
    os.chdir(original_cwd)
    try:
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
            label, _, _ = format_group_label(group_key, varying_fields)
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


def _run_render_aggregate_noun_grid(cfg: SweepConfig, run_configs: Sequence[PicbreederConfig], original_cwd: Path, cross_eval_dir: Path):
    import torch
    import os
    import numpy as np
    from pathlib import Path
    from dataclasses import fields as dataclass_fields
    from collections import defaultdict
    from scipy.optimize import linear_sum_assignment
    from compute_noun_coverage import (
        NounSimilarityConfig,
        prepare_openclip_components,
        prepare_noun_text_embeddings,
        infer_archive_order,
        embed_images,
        compute_max_similarities,
        render_noun_similarity_grid,
        load_nouns,
    )
    from config import PicbreederConfig

    # Helper to compute group label for a run (used for organization)
    group_keys = [_group_key_for_aggregate(rc) for rc in run_configs]
    varying_fields = compute_varying_fields(group_keys)
    
    # Group runs
    grouped_runs = defaultdict(list)
    grouped_labels = {}
    
    for rc, gk in zip(run_configs, group_keys):
        label, _, _ = format_group_label(gk, varying_fields)
        # Sanitize label for directory usage
        sanitized_label = label.replace(" ", "_").replace("=", "_").replace(":", "_").replace("/", "-")
        grouped_runs[sanitized_label].append(rc)
        grouped_labels[sanitized_label] = label

    previous_cwd = Path.cwd()
    os.chdir(original_cwd)
    try:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        print(f"[Aggregate Noun Grid] Using device: {device}")

        # Initialize model once (assuming embedding model is constant across sweep)
        base_kwargs0 = {
            field_def.name: getattr(run_configs[0], field_def.name)
            for field_def in dataclass_fields(PicbreederConfig)
            if field_def.name != "hydra"
        }
        noun_cfg_template = NounSimilarityConfig(
            **base_kwargs0, embedding_model=cfg.text_image_embedding_model, pretrained=cfg.text_image_pretrained,
            render_grid=cfg.render_noun_grids)
        noun_model, noun_preprocess, noun_tokenizer = prepare_openclip_components(noun_cfg_template, device)

        for label, group_configs in grouped_runs.items():
            print(f"Processing aggregate grid for group: {label}")
            
            # Use the first config in the group as reference
            ref_cfg = group_configs[0]
            base_kwargs = {
                field_def.name: getattr(ref_cfg, field_def.name)
                for field_def in dataclass_fields(PicbreederConfig)
                if field_def.name != "hydra"
            }
            noun_cfg = NounSimilarityConfig(**base_kwargs, embedding_model=cfg.text_image_embedding_model, pretrained=cfg.text_image_pretrained)
            
            # Load and deduplicate nouns
            noun_file = resolve_nounlist(noun_cfg.nounlist, original_cwd)
            raw_nouns = load_nouns(noun_file)
            dedup_nouns = []
            seen = set()
            for n in raw_nouns:
                if n not in seen:
                    dedup_nouns.append(n)
                    seen.add(n)

            # Prepare nouns
            nouns_list, prompts_list, noun_embeddings, _ = prepare_noun_text_embeddings(
                noun_cfg,
                original_cwd=original_cwd,
                device=device,
                model=noun_model,
                tokenizer=noun_tokenizer,
                nouns=dedup_nouns,
            )

            all_image_paths = []
            all_image_embeddings = []

            for run_cfg in group_configs:
                exp_dir = Path(run_cfg.experiment_dir)
                try:
                    run_paths = infer_archive_order(exp_dir)
                    if cfg.archive_limit is not None:
                        run_paths = run_paths[:cfg.archive_limit]
                    
                    if not run_paths:
                        continue

                    model_name_sanitized = cfg.text_image_embedding_model.replace("/", "-")
                    pretrained_sanitized = str(cfg.text_image_pretrained).replace("/", "-")
                    cache_path = exp_dir / f"image_embeddings_cache_{model_name_sanitized}_{pretrained_sanitized}.npy"
                    
                    run_embeddings = None
                    if cache_path.exists():
                        try:
                            cached = np.load(cache_path)
                            if len(cached) >= len(run_paths):
                                run_embeddings = cached[:len(run_paths)]
                            else:
                                print(f"Cache partial for {exp_dir} ({len(cached)} vs {len(run_paths)}), re-embedding...")
                        except Exception:
                            print(f"Cache load failed for {exp_dir}, re-embedding...")
                    
                    if run_embeddings is None:
                         _, run_embeddings = embed_images(
                            noun_model,
                            noun_preprocess,
                            run_paths,
                            device,
                            batch_size=noun_cfg.batch_size
                        )
                    
                    all_image_paths.extend(run_paths)
                    all_image_embeddings.append(run_embeddings)

                except Exception as e:
                    print(f"Skipping run {exp_dir}: {e}")

            if not all_image_embeddings:
                print(f"No valid images/embeddings for group {label}")
                continue

            full_embeddings = np.vstack(all_image_embeddings)
            
            # Compute similarities
            # We want to assign each image to at most one noun (and each noun to at most one image)
            # such that total similarity is maximized.
            print(f"Solving linear assignment for {full_embeddings.shape[0]} images and {noun_embeddings.shape[0]} nouns...")
            sims = full_embeddings @ noun_embeddings.T
            
            # linear_sum_assignment solves min cost, so we pass negative similarity (maximize=True works in newer scipy)
            # row_ind are image indices, col_ind are noun indices
            row_ind, col_ind = linear_sum_assignment(sims, maximize=True)
            
            # Construct max_per_noun and best_image_indices for rendering
            # Initialize with -1.0 so unmatched nouns appear at the end
            max_per_noun = np.full(len(nouns_list), -1.0, dtype=np.float32)
            best_image_indices = np.zeros(len(nouns_list), dtype=np.int64)
            
            # Fill in the matched pairs
            max_per_noun[col_ind] = sims[row_ind, col_ind]
            best_image_indices[col_ind] = row_ind
            
            num_matches = len(row_ind)
            print(f"Matched {num_matches} unique image-noun pairs.")
            
            # Determine effective top_k to avoid showing unmatched nouns
            effective_top_k = num_matches
            if noun_cfg.grid_top_k is not None:
                effective_top_k = min(effective_top_k, noun_cfg.grid_top_k)
            
            # Render Grid
            dest_dir = cross_eval_dir / label
            dest_dir.mkdir(parents=True, exist_ok=True)
            nounlist_name = Path(noun_cfg.nounlist).stem
            model_name_sanitized = cfg.text_image_embedding_model.replace("/", "-")
            output_path = dest_dir / f"aggregate_noun_grid_{nounlist_name}_{model_name_sanitized}.pdf"
            
            print(f"Rendering aggregate grid to {output_path}")
            render_noun_similarity_grid(
                nouns_list,
                max_per_noun,
                best_image_indices,
                all_image_paths,
                output_path,
                thumb_size=noun_cfg.grid_thumb_size,
                margin=noun_cfg.grid_margin,
                font_size=noun_cfg.grid_font_size,
                top_k=effective_top_k
            )

    finally:
        os.chdir(previous_cwd)


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
    base_params = dict(
        timeout_min=cfg.timeout_hours * 60,
        mem_gb=cfg.mem_gb,
        cpus_per_task=cfg.num_proc,
        slurm_account=cfg.account,
        name=cfg.sweep_name,
        # Auto-requeue preempted jobs so the sweep is fairshare-resilient; combined
        # with the orchestrator's resume=True (auto when exp_dir exists) the run
        # continues from its last saved agents.
        slurm_requeue=True,
    )
    # Optional QOS (e.g. "gpu168" for >48h 4-GPU/user jobs on torch).
    qos = getattr(cfg, "qos", None)
    if qos:
        base_params["slurm_qos"] = qos
    # Throttle concurrent array tasks (max_concurrent=0 = no throttle). Useful for
    # one-at-a-time sequential sweeps so fairshare doesn't tank.
    max_concurrent = int(getattr(cfg, "max_concurrent", 0) or 0)
    if max_concurrent > 0:
        base_params["slurm_array_parallelism"] = max_concurrent
    executor.update_parameters(**base_params)
    if cfg.gpu:
        gpus_per_task = int(getattr(cfg, "gpus_per_task", 1) or 1)
        gres_params = {"slurm_gres": f"gpu:{gpus_per_task}"}
        # Pass an explicit GPU partition when one is configured (the default "cpu" is
        # only meaningful for CPU sweeps; leaving it unset falls back to the cluster default).
        if cfg.partition and cfg.partition != "cpu":
            gres_params["slurm_partition"] = cfg.partition
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
            caption_model=cfg.caption_model,
            caption_embedding_model=cfg.caption_embedding_model,
            novelty_ylim=cfg.novelty_ylim,
            noun_ylim=cfg.noun_ylim,
            negative_anchors=cfg.negative_anchors,
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
            label, _, _ = format_group_label(group_key, varying_fields)
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

    any_eval_or_render = (
        cfg.eval_visual_coverage or
        cfg.eval_noun_coverage or
        cfg.eval_captions or
        cfg.render_archive or
        cfg.eval_tree or
        cfg.render_aggregate_noun_grid
    )

    if any_eval_or_render:
        import multiprocessing
        import torch
        ctx = multiprocessing.get_context("spawn")
        device_str = "cuda" if torch.cuda.is_available() else "cpu"

        if cfg.render_aggregate_noun_grid:
            _run_render_aggregate_noun_grid(cfg, run_configs, original_cwd, cross_eval_dir)

        if cfg.eval_visual_coverage:
            p1 = ctx.Process(
                target=_run_eval_visual_coverage,
                args=(cfg, run_configs, original_cwd, device_str)
            )
            p1.start()
            p1.join()
            if p1.exitcode != 0:
                print(_format_process_exit("Visual coverage eval", p1.exitcode))
                if p1.exitcode == -signal.SIGKILL:
                    print(
                        "Hint: try a smaller embedding model or lower archive_limit to reduce memory usage."
                    )

        if cfg.eval_noun_coverage:
            p2 = ctx.Process(
                target=_run_eval_noun_coverage,
                args=(cfg, run_configs, original_cwd, device_str, cross_eval_dir)
            )
            p2.start()
            p2.join()
            if p2.exitcode != 0:
                print(_format_process_exit("Noun coverage eval", p2.exitcode))
                if p2.exitcode == -signal.SIGKILL:
                    print(
                        "Hint: this stage loads image/text embeddings together; reduce archive_limit or switch to a lighter model."
                    )

        if cfg.eval_captions:
            p3 = ctx.Process(
                target=_run_eval_captions,
                args=(cfg, run_configs, original_cwd)
            )
            p3.start()
            p3.join()
            if p3.exitcode != 0:
                print(_format_process_exit("Caption eval", p3.exitcode))
                if p3.exitcode == -signal.SIGKILL:
                    print("Hint: caption evaluation may have exceeded memory limits.")

        if cfg.render_archive:
            _run_render_archive(cfg, run_configs, original_cwd, cross_eval_dir)

        if cfg.eval_tree:
            _run_eval_tree(cfg, run_configs, original_cwd, cross_eval_dir, filename_tag)
        # Plot aggregates for whatever was run
        _plot_seed_aggregates(
            run_configs=run_configs,
            output_dir=cross_eval_dir,
            filename_tag=filename_tag,
            image_embedding_model=cfg.image_embedding_model,
            text_image_embedding_model=cfg.text_image_embedding_model,
            caption_model=cfg.caption_model,
            caption_embedding_model=cfg.caption_embedding_model,
            novelty_ylim=cfg.novelty_ylim,
            noun_ylim=cfg.noun_ylim,
            negative_anchors=cfg.negative_anchors,
        )
        return

    if cfg.slurm:
        launch_slurm(cfg, log_dir, run_configs)
    else:
        launch_locally(run_configs)


if __name__ == "__main__":
    main()
