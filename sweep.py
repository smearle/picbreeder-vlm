#!/usr/bin/env python3
"""Launch collaborative_multi_agent sweeps locally or via Submitit."""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Sequence, Optional

import hydra
from hydra.core.config_store import ConfigStore
from hydra.utils import get_original_cwd
from hydra.conf import HelpConf, HydraConf
import submitit  # Do not remove this.

from collaborative_multi_agent import CollaborativeConfig


REPO_ROOT = Path(__file__).resolve().parent


@dataclass
class SweepCommand:
    """Container storing the command to execute and its working directory."""

    argv: Sequence[str]
    workdir: Path
    seed: int
    chat_history_turns: int
    experiment_dir: Path
    goal: str
    scheme: str


class CollaborativeRun:
    """Simple Submitit-compatible callable that executes a shell command."""

    def __init__(self, command: SweepCommand):
        self.command = list(command.argv)
        self.workdir = Path(command.workdir)
        self.seed = command.seed
        self.chat_history_turns = command.chat_history_turns
        self.experiment_dir = Path(command.experiment_dir)
        self.goal = command.goal
        self.scheme = command.scheme

    def __call__(self) -> int:
        pretty_cmd = " ".join(self.command)
        print(
            f"[submitit] seed={self.seed} chat={self.chat_history_turns} goal={self.goal} scheme={self.scheme} -> {pretty_cmd} (cwd={self.workdir})"
        )
        subprocess.run(self.command, check=True, cwd=self.workdir)
        return 0

    def checkpoint(self) -> "submitit.helpers.DelayedSubmission":

        refreshed = SweepCommand(
            argv=list(self.command),
            workdir=self.workdir,
            seed=self.seed,
            chat_history_turns=self.chat_history_turns,
            experiment_dir=self.experiment_dir,
            goal=self.goal,
            scheme=self.scheme,
        )
        return submitit.helpers.DelayedSubmission(self.__class__(refreshed))


def _execute_job(job: CollaborativeRun) -> int:
    return job()


@dataclass
class SweepConfig(CollaborativeConfig):
    seeds: List[int] = field(default_factory=lambda: [0])  # Random seeds swept over collaborative runs
    # chat_history_turns: List[int] = field(default_factory=lambda: [-1, 15, 10, 5, 0])  # Chat history lengths to evaluate
    chat_history_turns: List[int] = field(default_factory=lambda: [-1, 10, 0])  # Chat history lengths to evaluate
    goals: List[str] = field(default_factory=lambda: [  # Goals to sweep over
        "familiar_objects",
        # "fun",
        # "lizards", 
        # "fish", 
        # "skulls", 
        # "butterflies"
    ])
    scheme: str = "gray"
    extra_args: str = ""  # Additional Hydra overrides forwarded to collaborative_multi_agent
    experiment_prefix: Path = Path("logs_collaborative")  # Base directory for experiment outputs
    log_dir: Path = Path("submitit_logs")  # Submitit log directory
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
                    "  experiment_prefix     Root directory for generated experiment logs.\n"
                    "  slurm                 true to submit jobs to a SLURM cluster.\n"
                    "  partition / account   SLURM resource parameters appended to submissions.\n"
                    "  extra_args            Additional per-run overrides (e.g. \"+scheme=color\").\n"
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


def _experiment_directory(cfg: "SweepConfig", seed: int, chat_turns: int, goal: str, experiment_prefix: Path) -> Path:
    exp_dir = experiment_prefix / f"seed_{seed}_chat{chat_turns}"
    if goal != "familiar_objects":
        exp_dir = Path(f"{exp_dir}_goal-{goal}")
    if cfg.scheme != "gray":
        exp_dir = Path(f"{exp_dir}_scheme-{cfg.scheme}")
    if getattr(cfg, "warm_start_structure", 0) > 0:
        exp_dir = Path(f"{exp_dir}_warmstart{cfg.warm_start_structure}")
    return exp_dir


def build_command(cfg: SweepConfig, seed: int, chat_turns: int, goal: str, experiment_prefix: Path) -> SweepCommand:
    exp_dir = _experiment_directory(cfg, seed, chat_turns, goal, experiment_prefix)

    # If the experiment directory already exists, set to resume from it
    if os.path.exists(exp_dir):
        resume = "True"
    else:
        resume = "False"
        exp_dir.mkdir(parents=True, exist_ok=True)

    exp_dir_str = str(exp_dir)

    if cfg.evaluate or cfg.visualize:
        experiment_override = f'--experiment-dir="{exp_dir_str}"' if " " in exp_dir_str else f"--experiment-dir={exp_dir_str}"
        overrides: List[str] = [
            experiment_override,
        ]
        if cfg.archive_limit is not None:
            overrides.append(f"--archive-limit={cfg.archive_limit}")
        if cfg.evaluate:
            main_script = "embed_and_visualize.py"
        if cfg.visualize:
            main_script = "visualize_archive_phylogeny.py"
    else:
        main_script = "collaborative_multi_agent.py"
        experiment_override = f'experiment_dir="{exp_dir_str}"' if " " in exp_dir_str else f"experiment_dir={exp_dir_str}"
        overrides: List[str] = [
            f"seed={seed}",
            f"rows={cfg.rows}",
            f"cols={cfg.cols}",
            f"agent_generations={cfg.agent_generations}",
            f"num_agents={cfg.num_agents}",
            f"scheme={cfg.scheme}",
            f"chat_history_turns={chat_turns}",
            f"resume={resume}",
            f"goal={goal}",
            f"num_proc={cfg.num_proc}",
            f"personality_path={cfg.personality_path}",
            experiment_override,
        ]
        if getattr(cfg, "selection_baseline", "none") != "none":
            overrides.append(f"selection_baseline={cfg.selection_baseline}")
        if cfg.output_activations:
            overrides.append("output_activations=true")
    if cfg.extra_args:
        overrides.extend(shlex.split(cfg.extra_args))

    cmd: List[str] = [sys.executable, str(REPO_ROOT / main_script), *overrides]
    return SweepCommand(
        argv=cmd,
        workdir=REPO_ROOT,
        seed=seed,
        chat_history_turns=chat_turns,
        experiment_dir=exp_dir,
        goal=goal,
        scheme=cfg.scheme,
    )


def launch_locally(commands: Sequence[SweepCommand]) -> None:
    for command in commands:
        pretty_cmd = " ".join(command.argv)
        print(
            f"[local] seed={command.seed} chat={command.chat_history_turns} goal={command.goal} scheme={command.scheme} -> {pretty_cmd} (cwd={command.workdir})"
        )
        subprocess.run(command.argv, check=True, cwd=command.workdir)


def launch_slurm(cfg: SweepConfig, log_dir: Path, commands: Sequence[SweepCommand]) -> None:
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
    jobs = [CollaborativeRun(command) for command in commands]
    futures = executor.map_array(_execute_job, jobs)
    for command, future in zip(commands, futures):
        print(
            f"[slurm] seed={command.seed} chat={command.chat_history_turns} goal={command.goal} scheme={command.scheme} submitted as job {future.job_id}"
        )


cs = ConfigStore.instance()
cs.store(name="sweep_base", node=SweepConfig)


@hydra.main(version_base=None, config_path=None, config_name="sweep_base")
def main(cfg: SweepConfig) -> None:
    original_cwd = Path(get_original_cwd())
    experiment_prefix = _ensure_absolute(cfg.experiment_prefix, original_cwd)
    experiment_prefix.mkdir(parents=True, exist_ok=True)
    log_dir = _ensure_absolute(cfg.log_dir, original_cwd)
    log_dir.mkdir(parents=True, exist_ok=True)

    sweep_entries = [
        (seed, chat_turns, goal, _experiment_directory(cfg, seed, chat_turns, goal, experiment_prefix))
        for seed in cfg.seeds
        for chat_turns in cfg.chat_history_turns
        for goal in cfg.goals
    ]

    if not sweep_entries:
        print("No runs scheduled (empty seeds or chat_history_turns).")
        return

    if cfg.cross_eval:
        from render_embedding_metrics_table import DEFAULT_METRICS_FILENAME, render_tables

        metrics_name = DEFAULT_METRICS_FILENAME
        existing_dirs = []
        missing_metrics = []

        for seed, chat_turns, goal, exp_dir in sweep_entries:
            metrics_path = exp_dir / metrics_name
            if metrics_path.exists():
                existing_dirs.append(exp_dir)
            else:
                missing_metrics.append((seed, chat_turns, goal, metrics_path))

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

    commands = [
        build_command(cfg, seed, chat_turns, goal, experiment_prefix)
        for seed, chat_turns, goal, _ in sweep_entries
    ]

    if cfg.slurm:
        launch_slurm(cfg, log_dir, commands)
    else:
        launch_locally(commands)


if __name__ == "__main__":
    main()
