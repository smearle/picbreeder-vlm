#!/usr/bin/env python3
from dataclasses import dataclass, field
import os
from typing import List, Union, Optional, Dict

import dotenv
from hydra.conf import HelpConf, HydraConf

from config import PicbreederConfig

dotenv.load_dotenv()

@dataclass
class SweepConfig(PicbreederConfig):
    seed: List[int] = field(default_factory=lambda: [0])  # Random seeds swept over collaborative runs
    chat_history_turns: List[int] = field(default_factory=lambda: [1])  # Chat history lengths to evaluate
    always_include_branched_image: List[bool] = field(default_factory=lambda: [False])  # Preserve branched image reference after branch archive sample leaves chat history
    always_include_archive_sample: List[bool] = field(default_factory=lambda: [False])  # Preserve archive branching sample turn once it falls out of chat history
    allow_restart_from_publications: List[bool] = field(default_factory=lambda: [True])  # Allow restart branching from an agent's own prior publications
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
    account: Optional[str] = os.environ.get("SLURM_ACCOUNT")  # Optional SLURM account override
    timeout_hours: int = 120  # Wall-time limit in hours
    mem_gb: int = 30  # Memory requested per task (GB)
    num_proc: int = 10  # Number of parallel processes per task
    # SLURM controls for large/restricted jobs (e.g. torch's gpu168 QOS allows
    # 4 GPU/user for >48h jobs); leave qos=None to use the cluster default.
    qos: Optional[str] = None  # e.g. "gpu168" for 4-GPU/user >48h jobs
    gpus_per_task: int = 1  # GPUs per array task (e.g. 4 for TP=4 on a 235B model)
    max_concurrent: int = 0  # Cap concurrent array tasks (0 = unlimited); set 1 for sequential
    render_archive: bool = False  # If true, run evaluation instead of training
    eval_tree: bool = False  # If true, run phylogeny visualization instead of training
    eval_visual_coverage: bool = False  # If true, run visual coverage evaluation
    eval_noun_coverage: bool = False  # If true, run noun coverage evaluation
    render_noun_grids: bool = False  # If true, render noun similarity grids
    render_aggregate_noun_grid: bool = False # If true, render aggregate noun similarity grid
    overwrite_evals: bool = True  # If false, skip evaluation if output files already exist
    cross_eval: bool = False  # If true, summarize embedding metrics from the configured runs
    eval_captions: bool = False  # If true, run captioning and embedding analysis
    caption_model: str = "gemini-2.5-pro"  # Model used for captioning in eval_captions
    # caption_embedding_model: str = "tencent/KaLM-Embedding-Gemma3-12B-2511"  # Embedding model for captions
    caption_embedding_model: str = "gemini-embedding-001"  # Embedding model for captions
    caption_embedding_pretrained: str = ""
    archive_limit: Optional[int] = None  # Limit the number of archive images passed to analysis scripts
    nounlist: List[str] = field(default_factory=lambda: ["things_deduped"])  # Noun list(s) to evaluate
    novelty_ylim: Optional[List[float]] = None  # Optional Y-axis limits for novelty plots (e.g., [0.6, 1.0])
    noun_ylim: Optional[List[float]] = None
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
    chat_history_turns: List[int] = field(default_factory=lambda: [
                                                                -1, 
                                                                10, 
                                                                2, 
                                                                1, 
                                                                0
    ])
    temperature: List[Union[int, float, str]] = field(default_factory=lambda: [1.0])
    rand_select_prob: List[float] = field(default_factory=lambda: [0.0])
    goal: List[str] = field(default_factory=lambda: ["familiar_objects"])
    model: List[str] = field(default_factory=lambda: ["gemini-2.5-pro"])
    seed: List[int] = field(default_factory=lambda: [
        3, 4, 5,
        # 6, 7, 8
    ])
    num_agents: int = 10_000
    noun_ylim: Optional[List[float]] = field(default_factory=lambda: [0.06, 0.09])


@dataclass
class IncludeBranchedImageSweep(SweepConfig):
    chat_history_turns: List[int] = field(default_factory=lambda: [
                                                                1, 
    ])
    temperature: List[Union[int, float, str]] = field(default_factory=lambda: [1.0])
    rand_select_prob: List[float] = field(default_factory=lambda: [0.0])
    goal: List[str] = field(default_factory=lambda: ["familiar_objects"])
    model: List[str] = field(default_factory=lambda: ["gemini-2.5-pro"])
    seed: List[int] = field(default_factory=lambda: [3, 4, 5])
    num_agents: int = 2_000
    always_include_branched_image: List[bool] = field(default_factory=lambda: [
        True,
        False,
    ])


@dataclass
class FullRandSelectProbSweep(SweepConfig):
    rand_select_prob: List[float] = field(default_factory=lambda: [
        0.0, 
        0.05,
        0.25, 0.5, 0.75, 1.0
    ])
    rand_select_mode: str = 'all'
    chat_history_turns: List[int] = field(default_factory=lambda: [1])
    temperature: List[Union[int, float, str]] = field(default_factory=lambda: [1.0])
    goal: List[str] = field(default_factory=lambda: ["familiar_objects"])
    model: List[str] = field(default_factory=lambda: ["gemini-2.5-pro"])
    seed: List[int] = field(default_factory=lambda: [
        3, 4, 5, 
        6, 7, 8,
    ])
    thumb_size: List[int] = field(default_factory=lambda: [128,])
    num_agents: int = 2_000
    noun_ylim: Optional[List[float]] = field(default_factory=lambda: [0.06, 0.095])


@dataclass
class TraitsSweep(SweepConfig):
    n_personality_traits: List[int] = field(default_factory=lambda: [
        0,
        10, 100, 1_000
    ])
    model: List[str] = field(default_factory=lambda: ["gemini-2.5-pro"])
    chat_history_turns: List[int] = field(default_factory=lambda: [1])
    seed: List[int] = field(default_factory=lambda: [3, 4, 5, 6, 7, 8])
    num_agents: int = 2_000
    noun_ylim: Optional[List[float]] = field(default_factory=lambda: [0.06, 0.095])

@dataclass
class ChatHistoryTurnsGemini3Sweep(SweepConfig):
    chat_history_turns: List[int] = field(default_factory=lambda: [
                                                                2, 
                                                                1, 
                                                                0
    ])
    temperature: List[Union[int, float, str]] = field(default_factory=lambda: [1.0])
    rand_select_prob: List[float] = field(default_factory=lambda: [0.0])
    goal: List[str] = field(default_factory=lambda: ["familiar_objects"])
    model: List[str] = field(default_factory=lambda: ["gemini-3-pro-preview"])
    seed: List[int] = field(default_factory=lambda: [3, 4, 5])
    num_agents: int = 1_000

@dataclass
class ChatHistoryTurnsQwenSweep(SweepConfig):
    chat_history_turns: List[int] = field(default_factory=lambda: [0, 1, 2, 3])
    temperature: List[Union[int, float, str]] = field(default_factory=lambda: [1.0])
    rand_select_prob: List[float] = field(default_factory=lambda: [0.0])
    goal: List[str] = field(default_factory=lambda: ["familiar_objects"])
    model: List[str] = field(default_factory=lambda: ["qwen3-vl-8b"])
    seed: List[int] = field(default_factory=lambda: [3, 4, 5])
    num_agents: int = 1_000
    num_proc: int = 10
    gpu: bool = True
    novelty_ylim: Optional[List[float]] = field(default_factory=lambda: [0.6, 0.93])


@dataclass
class ChatHistoryTurnsQwen30BSweep(SweepConfig):
    chat_history_turns: List[int] = field(default_factory=lambda: [
        0, 1, 2, 10, -1])
    temperature: List[Union[int, float, str]] = field(default_factory=lambda: [1.0])
    rand_select_prob: List[float] = field(default_factory=lambda: [0.0])
    goal: List[str] = field(default_factory=lambda: ["familiar_objects"])
    # Registry name so each SLURM job auto-starts its own vLLM server (set VLLM_TP_SIZE
    # if the model needs >1 GPU). For the local shared server, override model=remote:Qwen/...
    model: List[str] = field(default_factory=lambda: ["qwen3-vl-30b-fp8"])
    seed: List[int] = field(default_factory=lambda: [3, 4, 5])
    num_agents: int = 1_000
    num_proc: int = 10
    gpu: bool = True
    partition: str = "h100_tandon"
    novelty_ylim: Optional[List[float]] = field(default_factory=lambda: [0.6, 0.93])
    noun_ylim: Optional[List[float]] = field(default_factory=lambda: [0.04, 0.080])


@dataclass
class FullRandSelectProbQwen30b(SweepConfig):
    chat_history_turns: List[int] = field(default_factory=lambda: [1])
    temperature: List[Union[int, float, str]] = field(default_factory=lambda: [1.0])
    rand_select_prob: List[float] = field(default_factory=lambda: [0.05, 0.1, 0.25, 0.5])
    rand_select_mode: str = 'all'
    goal: List[str] = field(default_factory=lambda: ["familiar_objects"])
    # Registry name so each SLURM job auto-starts its own vLLM server (set VLLM_TP_SIZE
    # if the model needs >1 GPU). For the local shared server, override model=remote:Qwen/...
    model: List[str] = field(default_factory=lambda: ["qwen3-vl-30b-fp8"])
    seed: List[int] = field(default_factory=lambda: [3, 4, 5])
    num_agents: int = 1_000
    num_proc: int = 10
    gpu: bool = True
    partition: str = "h100_tandon"
    novelty_ylim: Optional[List[float]] = field(default_factory=lambda: [0.6, 0.93])
    noun_ylim: Optional[List[float]] = field(default_factory=lambda: [0.04, 0.080])


@dataclass
class ChatHistoryTurnsQwenColorSweep(SweepConfig):
    """Probe Qwen color usage across chat-history lengths (nudge OFF by default).

    Each SLURM job auto-starts its own 1-GPU vLLM server (registry model names).
    Run nudge-off first; if no color appears, re-run with color_nudge=true (CLI override
    or set it here) — the experiment dir gets a `_colornudge` suffix so runs don't collide.
    """
    chat_history_turns: List[int] = field(default_factory=lambda: [0, 1, 2, 10, -1])
    temperature: List[Union[int, float, str]] = field(default_factory=lambda: [1.0])
    rand_select_prob: List[float] = field(default_factory=lambda: [0.0])
    goal: List[str] = field(default_factory=lambda: ["familiar_objects"])
    model: List[str] = field(default_factory=lambda: ["qwen3-vl-8b", "qwen3-vl-30b-fp8"])
    color_nudge: bool = False
    seed: List[int] = field(default_factory=lambda: [3, 4, 5])
    num_agents: int = 1_000
    num_proc: int = 10
    gpu: bool = True
    partition: str = "h100_tandon"
    novelty_ylim: Optional[List[float]] = field(default_factory=lambda: [0.6, 0.93])
    noun_ylim: Optional[List[float]] = field(default_factory=lambda: [0.04, 0.080])


@dataclass
class Qwen235BBf16Sweep(SweepConfig):
    """Longer-term sweep on Qwen3-VL-235B-A22B-Instruct (bf16) across history lengths.

    Sequential (max_concurrent=1) on h200_tandon / qos=gpu168 with 4 GPUs per task
    (TP=4) and --time=49h so each job stays under the gpu168 walltime band. Each job
    auto-resumes on preemption (slurm_requeue=True + collaborative_multi_agent's
    resume=True when dir exists). One seed for now; cross-eval natural via sweep_logs.
    """
    chat_history_turns: List[int] = field(default_factory=lambda: [1, -1, 10, 0, 2])
    temperature: List[Union[int, float, str]] = field(default_factory=lambda: [1.0])
    rand_select_prob: List[float] = field(default_factory=lambda: [0.0])
    goal: List[str] = field(default_factory=lambda: ["familiar_objects"])
    model: List[str] = field(default_factory=lambda: ["Qwen/Qwen3-VL-235B-A22B-Instruct"])
    seed: List[int] = field(default_factory=lambda: [0])
    num_agents: int = 2_000
    # NYU torch cancels GPU jobs averaging <60% util over 2h. num_proc=10 left util at
    # ~52% and num_proc=32 STILL got th2 killed at 58% (email 2026-05-30). Short-prompt
    # configs (th1/th2) finish each VLM call fast so CPU-side work between calls is a
    # bigger fraction → vLLM's 16 batch slots starve unless many agents are queued.
    # 64 gives generous slack (32+ agents waiting at any time over the 16-slot batch).
    num_proc: int = 64
    gpu: bool = True
    gpus_per_task: int = 4
    partition: str = "h200_tandon"
    qos: str = "gpu168"
    timeout_hours: int = 49  # >48h is required for gpu168
    mem_gb: int = 480
    max_concurrent: int = 1  # sequential -> fairshare-friendly + cross-eval natural
    log_dir: str = "sweep_logs_qwen_235b_bf16"
    novelty_ylim: Optional[List[float]] = field(default_factory=lambda: [0.6, 0.93])
    noun_ylim: Optional[List[float]] = field(default_factory=lambda: [0.04, 0.080])


@dataclass
class Qwen235BBf16DaemonSweep(SweepConfig):
    """CPU-only agent jobs that connect to a persistent vLLM daemon (daemon_vllm_235b.sbatch).

    Avoids both pitfalls of Qwen235BBf16Sweep on torch:
      - the 30-min vLLM cold-load each cycle (the daemon loads ONCE)
      - the GPU-util killer firing at ~2h17 (agents have no GPU; the daemon's util
        stays high because all 5 configs constantly feed it)

    Resumes the SAME exp dirs as Qwen235BBf16Sweep: config.py:_resolve_model strips
    the "remote:" prefix and resolves the alias so dir names match.

    Launch order:
      1. sbatch daemon_vllm_235b.sbatch ; wait for vllm_daemon.endpoint to exist
         AND for the URL inside it to respond on /v1/models.
      2. export VLLM_BASE_URL=$(cat vllm_daemon.endpoint) ; .venv/bin/python sweep.py
         sweep_name=qwen_235b_bf16_daemon slurm=true partition=cpu_short
    """
    chat_history_turns: List[int] = field(default_factory=lambda: [1, -1, 10, 0, 2])
    temperature: List[Union[int, float, str]] = field(default_factory=lambda: [1.0])
    rand_select_prob: List[float] = field(default_factory=lambda: [0.0])
    goal: List[str] = field(default_factory=lambda: ["familiar_objects"])
    # The "remote:" prefix routes vlm_backends to the OpenAI-compatible client; the
    # orchestrator's is_local_model returns False and skips _start_vlm_server.
    model: List[str] = field(default_factory=lambda: ["remote:Qwen/Qwen3-VL-235B-A22B-Instruct"])
    seed: List[int] = field(default_factory=lambda: [0])
    num_agents: int = 2_000
    # cpu_short QOS limits: MaxWall=6h, MaxCPUsPU=32. So 5 parallel configs × 6 cpus
    # each = 30 CPUs (under the 32 cap), and submitit auto-requeues at the 6h boundary
    # via slurm_additional_parameters={"requeue": True} + CollaborativeRun.checkpoint().
    # 5 configs × 6 agents = 30 total agents queuing on the daemon's 16-slot batch —
    # ~2x oversupply keeps the daemon saturated.
    num_proc: int = 6
    gpu: bool = False  # CPU-only agent jobs; daemon owns the GPUs.
    gpus_per_task: int = 0
    partition: str = "cpu_short"
    qos: Optional[str] = None  # default QOS for cpu_short (only one available to our account)
    timeout_hours: int = 6  # cpu_short MaxWall=6h
    mem_gb: int = 8  # CPU-only HTTP client + NEAT/CPPN render is small
    max_concurrent: int = 5  # all 5 configs in parallel (5 × 6 cpus = 30, under 32 cap)
    log_dir: str = "sweep_logs_qwen_235b_bf16"  # same as GPU sweep -> resume picks up
    novelty_ylim: Optional[List[float]] = field(default_factory=lambda: [0.6, 0.93])
    noun_ylim: Optional[List[float]] = field(default_factory=lambda: [0.04, 0.080])


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
    thumb_size: List[int] = field(default_factory=lambda: [224])
    # thumb_size: List[int] = field(default_factory=lambda: [128,])
    num_agents: int = 500


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
    num_agents: int = 10_000


@dataclass
class ObjectiveFreeSweep(SweepConfig):
    goal: List[str] = field(default_factory=lambda: [
        "none",
        "familiar_objects", 
    ])
    chat_history_turns: List[int] = field(default_factory=lambda: [1])
    temperature: List[Union[int, float, str]] = field(default_factory=lambda: [1.0])
    rand_select_prob: List[float] = field(default_factory=lambda: [0.0])
    model: List[str] = field(default_factory=lambda: ["gemini-2.5-pro"])
    seed: List[int] = field(default_factory=lambda: [3, 4, 5])
    num_agents: int = 1_000


@dataclass
class ModelSweep(SweepConfig):
    model: List[str] = field(default_factory=lambda: [
        "gemini-2.5-pro",
        "gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-3-pro-preview",
        "gemini-random",
        "qwen3-vl-8b",
        "qwen3-vl-30b-fp8",
    ])
    chat_history_turns: List[int] = field(default_factory=lambda: [1])
    seed: List[int] = field(default_factory=lambda: [3, 4, 5])
    num_agents: int = 1_000


@dataclass
class LongSweep(SweepConfig):
    rand_select_prob: List[float] = field(default_factory=lambda: [0.0])
    chat_history_turns: List[int] = field(default_factory=lambda: [1])
    temperature: List[Union[int, float, str]] = field(default_factory=lambda: [1.0])
    goal: List[str] = field(default_factory=lambda: ["familiar_objects"])
    model: List[str] = field(default_factory=lambda: ["gemini-2.5-pro"])
    seed: List[int] = field(default_factory=lambda: [3])
    num_agents: int = 9_377


@dataclass
class LongSweep2(SweepConfig):
    rand_select_prob: List[float] = field(default_factory=lambda: [0.25])
    rand_select_mode: str = 'all'
    chat_history_turns: List[int] = field(default_factory=lambda: [1])
    temperature: List[Union[int, float, str]] = field(default_factory=lambda: [1.0])
    goal: List[str] = field(default_factory=lambda: ["familiar_objects"])
    model: List[str] = field(default_factory=lambda: ["gemini-2.5-pro"])
    seed: List[int] = field(default_factory=lambda: [5])
    num_agents: int = 9_377


_NAMED_SWEEPS: Dict[str, type[SweepConfig]] = {
    "sweep": SweepBasePreset,
    "chat_history_turns": ChatHistoryTurnsSweep,
    "chat_history_turns_gemini3": ChatHistoryTurnsGemini3Sweep,
    "chat_history_turns_qwen": ChatHistoryTurnsQwenSweep,
    "chat_history_turns_qwen_30b": ChatHistoryTurnsQwen30BSweep,
    "chat_history_turns_qwen_color": ChatHistoryTurnsQwenColorSweep,
    "qwen_235b_bf16": Qwen235BBf16Sweep,
    "qwen_235b_bf16_daemon": Qwen235BBf16DaemonSweep,
    "temperature": TemperatureSweep,
    "rand_select_prob": RandSelectProbSweep,
    "full_rand_select_prob": FullRandSelectProbSweep,
    "full_rand_select_prob_qwen_30b": FullRandSelectProbQwen30b,
    "rand_baseline": RandBaselineSweep,
    "model": ModelSweep,
    "traits": TraitsSweep,
    "long_sweep": LongSweep,
    "long_sweep_2": LongSweep2,
    "objective_free": ObjectiveFreeSweep,
    "include_branched_image": IncludeBranchedImageSweep,
}
