#!/usr/bin/env python3
"""Launch collaborative_multi_agent sweeps locally or via Submitit."""

from __future__ import annotations

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

from collaborative_multi_agent import CollaborativeConfig


REPO_ROOT = Path(__file__).resolve().parent


@dataclass
class SweepCommand:
    """Container storing the command to execute and its working directory."""

    argv: Sequence[str]
    workdir: Path
    seed: int
    chat_history_turns: int


class CollaborativeRun:
    """Simple Submitit-compatible callable that executes a shell command."""

    def __init__(self, command: SweepCommand):
        self.command = list(command.argv)
        self.workdir = Path(command.workdir)
        self.seed = command.seed
        self.chat_history_turns = command.chat_history_turns

    def __call__(self) -> int:
        pretty_cmd = " ".join(self.command)
        print(
            f"[submitit] seed={self.seed} chat={self.chat_history_turns} -> {pretty_cmd} (cwd={self.workdir})"
        )
        subprocess.run(self.command, check=True, cwd=self.workdir)
        return 0

    def checkpoint(self) -> "submitit.helpers.DelayedSubmission":
        import submitit

        refreshed = SweepCommand(
            argv=list(self.command),
            workdir=self.workdir,
            seed=self.seed,
            chat_history_turns=self.chat_history_turns,
        )
        return submitit.helpers.DelayedSubmission(self.__class__(refreshed))


def _execute_job(job: CollaborativeRun) -> int:
    return job()


@dataclass
class SweepConfig(CollaborativeConfig):
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2])  # Random seeds swept over collaborative runs
    chat_history_turns: List[int] = field(default_factory=lambda: [-1, 0])  # Chat history lengths to evaluate
    extra_args: str = ""  # Additional Hydra overrides forwarded to collaborative_multi_agent
    experiment_prefix: Path = Path("logs_collaborative/submitit_sweeps")  # Base directory for experiment outputs
    log_dir: Path = Path("log/submitit")  # Submitit log directory
    slurm: bool = False  # Enable SLURM submission via Submitit
    partition: str = "cpu"  # SLURM partition name
    account: Optional[str] = None  # Optional SLURM account override
    cpus_per_task: int = 4  # CPUs requested per task
    timeout_hours: int = 24  # Wall-time limit in hours
    mem_gb: int = 30  # Memory requested per task (GB)
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


def build_command(cfg: SweepConfig, seed: int, chat_turns: int, experiment_prefix: Path) -> SweepCommand:
    exp_dir = experiment_prefix / f"seed_{seed}_chat{chat_turns}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    exp_dir_str = str(exp_dir)
    experiment_override = f'experiment_dir="{exp_dir_str}"' if " " in exp_dir_str else f"experiment_dir={exp_dir_str}"

    overrides: List[str] = [
        f"seed={seed}",
        f"rows={cfg.rows}",
        f"cols={cfg.cols}",
        f"agent_generations={cfg.agent_generations}",
        f"num_agents={cfg.num_agents}",
        f"scheme={cfg.scheme}",
        f"color_palette={cfg.color_palette}",
        f"chat_history_turns={chat_turns}",
        experiment_override,
    ]
    if cfg.dry_run:
        overrides.append("dry_run=true")
    if cfg.output_activations:
        overrides.append("output_activations=true")
    if cfg.extra_args:
        overrides.extend(shlex.split(cfg.extra_args))

    cmd: List[str] = [sys.executable, str(REPO_ROOT / "collaborative_multi_agent.py"), *overrides]
    return SweepCommand(argv=cmd, workdir=REPO_ROOT, seed=seed, chat_history_turns=chat_turns)


def launch_locally(commands: Sequence[SweepCommand]) -> None:
    for command in commands:
        pretty_cmd = " ".join(command.argv)
        print(
            f"[local] seed={command.seed} chat={command.chat_history_turns} -> {pretty_cmd} (cwd={command.workdir})"
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
        cpus_per_task=cfg.cpus_per_task,
        slurm_partition=cfg.partition,
        slurm_account=cfg.account,
        name="collab-sweep",
    )
    jobs = [CollaborativeRun(command) for command in commands]
    futures = executor.map_array(_execute_job, jobs)
    for command, future in zip(commands, futures):
        print(
            f"[slurm] seed={command.seed} chat={command.chat_history_turns} submitted as job {future.job_id}"
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

    commands = [
        build_command(cfg, seed, chat_turns, experiment_prefix)
        for seed in cfg.seeds
        for chat_turns in cfg.chat_history_turns
    ]

    if not commands:
        print("No runs scheduled (empty seeds or chat_history_turns).")
        return

    if cfg.slurm:
        launch_slurm(cfg, log_dir, commands)
    else:
        launch_locally(commands)


if __name__ == "__main__":
    main()
