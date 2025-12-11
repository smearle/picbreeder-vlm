#!/usr/bin/env python3
"""Launch collaborative_multi_agent sweeps locally or via Submitit."""

from __future__ import annotations

import ast
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass, field, replace, fields
from itertools import product
from pathlib import Path
from typing import List, Sequence, Optional

import hydra
from hydra.core.config_store import ConfigStore
from hydra.utils import get_original_cwd
from hydra.conf import HelpConf, HydraConf
import omegaconf
import submitit  # Do not remove this.

from collaborative_multi_agent import run as run_collaborative
from config import PicbreederConfig, ensure_valid_config


SCRIPT_ROOT = Path(__file__).resolve().parent


class CollaborativeRun:
    """Submitit-compatible callable that executes a configured run."""

    def __init__(self, cfg: PicbreederConfig):
        self.cfg = cfg

    def __call__(self) -> int:
        print(_format_run_prefix(self.cfg, "[submitit]"))
        run_collaborative(self.cfg)
        return 0

    def checkpoint(self) -> "submitit.helpers.DelayedSubmission":
        refreshed = replace(self.cfg)
        return submitit.helpers.DelayedSubmission(self.__class__(refreshed))


def _execute_job(job: CollaborativeRun) -> int:
    return job()


@dataclass
class SweepConfig(PicbreederConfig):
    seed: List[int] = field(default_factory=lambda: [0])  # Random seeds swept over collaborative runs
    # chat_history_turns: List[int] = field(default_factory=lambda: [-1, 15, 10, 5, 0])  # Chat history lengths to evaluate
    chat_history_turns: List[int] = field(default_factory=lambda: [-1])  # Chat history lengths to evaluate
    rand_select_prob: List[float] = field(default_factory=lambda: [0.0])  # Probability of random parent selection
    goal: List[str] = field(default_factory=lambda: [  # Goals to sweep over
        "familiar_objects",
        # "fun",
        # "lizards", 
        # "fish", 
        # "skulls", 
        # "butterflies"
    ])
    model: List[str] = field(default_factory=lambda: [  # VLM models to evaluate
        "gemini-3-pro-preview",
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
    ])
    sweep_name: str = "sweep"  # Base directory for experiment outputs
    log_dir: str = "sweep_logs"
    slurm: bool = True  # Enable SLURM submission via Submitit
    partition: str = "cpu"  # SLURM partition name
    account: Optional[str] = None  # Optional SLURM account override
    timeout_hours: int = 24  # Wall-time limit in hours
    mem_gb: int = 30  # Memory requested per task (GB)
    num_proc: int = 10  # Number of parallel processes per task
    evaluate: bool = False  # If true, run evaluation instead of training
    visualize: bool = False  # If true, run phylogeny visualization instead of training
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
                    "  sweep_name            Root directory for generated experiment logs.\n"
                    "  slurm                 true to submit jobs to a SLURM cluster.\n"
                    "  partition / account   SLURM resource parameters appended to submissions.\n"
                    "  cross_eval            true to summarize embedding metrics for the configured runs.\n"
                ),
                footer="Hydra overrides (e.g. +option=value) are supported. Use --cfg=job to inspect merged configs.",
            )
        )
    )


def _ensure_absolute(path: Path, base: Path) -> Path:
    expanded = Path(path).expanduser()
    if expanded.is_absolute():
        return expanded.resolve()
    return (base / expanded).resolve()


def _build_eval_command(
    cfg: PicbreederConfig, script: str, archive_limit: Optional[int],
) -> List[str]:
    cmd: List[str] = [
        sys.executable,
        str(SCRIPT_ROOT / script),
        f"--experiment-dir={cfg.experiment_dir}",
    ]
    if archive_limit is not None:
        cmd.append(f"--archive-limit={archive_limit}")
    return cmd


def _expand_sweep_configs(cfg: SweepConfig) -> List[SweepConfig]:
    """Produce one config per cartesian product of list-valued fields."""
    sweep_axes = []
    for field_def in fields(SweepConfig):
        name = field_def.name
        if name == "hydra":
            continue
        value = getattr(cfg, name)
        print(f"Sweeping over {name} with value {value} type {type(value)}")
        if isinstance(value, omegaconf.listconfig.ListConfig) and isinstance(value[0], type(getattr(PicbreederConfig, name))):
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

    executor = submitit.AutoExecutor(folder=log_dir)
    executor.update_parameters(
        timeout_min=cfg.timeout_hours * 60,
        mem_gb=cfg.mem_gb,
        cpus_per_task=cfg.num_proc,
        slurm_partition=cfg.partition,
        slurm_account=cfg.account,
        name="picbreeder-vlm",
    )
    jobs = [CollaborativeRun(run_cfg) for run_cfg in configs]
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

    base_configs = _expand_sweep_configs(cfg)
    run_configs = [_build_run_config(run_cfg, original_cwd) for run_cfg in base_configs]

    if not run_configs:
        print("No runs scheduled (empty sweep axes).")
        return

    experiment_prefix = Path(run_configs[0].experiment_dir).parent
    experiment_prefix.mkdir(parents=True, exist_ok=True)

    if cfg.cross_eval:
        from render_embedding_metrics_table import DEFAULT_METRICS_FILENAME, render_tables

        metrics_name = DEFAULT_METRICS_FILENAME
        existing_dirs = []
        missing_metrics = []

        for run_cfg in run_configs:
            metrics_path = Path(run_cfg.experiment_dir) / metrics_name
            if metrics_path.exists():
                existing_dirs.append(Path(run_cfg.experiment_dir))
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

        try:
            per_table, agg_table, per_csv, agg_csv = render_tables(
                experiment_prefix,
                output_dir=experiment_prefix,
                metrics_name=metrics_name,
                experiment_dirs=existing_dirs,
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

        if missing_metrics:
            print("\nMissing metrics files:")
            for seed, chat_turns, goal, missing_path in missing_metrics:
                print(f"  seed={seed} chat={chat_turns} goal={goal} -> {missing_path}")
        return

    if cfg.evaluate or cfg.visualize:
        script = "embed_and_visualize.py" if cfg.evaluate else "visualize_archive_phylogeny.py"
        commands = [
            _build_eval_command(run_cfg, script, cfg.archive_limit)
            for run_cfg in run_configs
        ]
        for cmd, run_cfg in zip(commands, run_configs):
            desc = _format_run_prefix(run_cfg, "[local-eval]")
            pretty_cmd = " ".join(cmd)
            print(f"{desc} -> {pretty_cmd}")
            subprocess.run(cmd, check=True, cwd=original_cwd)
    elif cfg.slurm:
        launch_slurm(cfg, log_dir, run_configs)
    else:
        launch_locally(run_configs)


if __name__ == "__main__":
    main()
