#!/usr/bin/env python3
from dataclasses import dataclass, field
from typing import List, Union, Optional, Dict

from hydra.conf import HelpConf, HydraConf

from config import PicbreederConfig


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
    eval_tree: bool = False  # If true, run phylogeny visualization instead of training
    eval_visual_coverage: bool = False  # If true, run visual coverage evaluation
    eval_noun_coverage: bool = False  # If true, run noun coverage evaluation
    overwrite_evals: bool = True  # If false, skip evaluation if output files already exist
    cross_eval: bool = False  # If true, summarize embedding metrics from the configured runs
    eval_captions: bool = False  # If true, run captioning and embedding analysis
    caption_model: str = "gemini-2.5-pro"  # Model used for captioning in eval_captions
    # caption_embedding_model: str = "tencent/KaLM-Embedding-Gemma3-12B-2511"  # Embedding model for captions
    caption_embedding_model: str = "gemini-embedding-001"  # Embedding model for captions
    caption_embedding_pretrained: str = ""
    archive_limit: Optional[int] = None  # Limit the number of archive images passed to analysis scripts
    nounlist: List[str] = field(default_factory=lambda: ["things"])  # Noun list(s) to evaluate
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
    seed: List[int] = field(default_factory=lambda: [3, 4, 5, 6, 7, 8])
    num_agents: int = 1_000


@dataclass
class FullRandSelectProbSweep(SweepConfig):
    rand_select_prob: List[float] = field(default_factory=lambda: [
        0.0, 
        0.25, 0.5, 0.75, 1.0])
    rand_select_mode: str = 'all'
    chat_history_turns: List[int] = field(default_factory=lambda: [1])
    temperature: List[Union[int, float, str]] = field(default_factory=lambda: [1.0])
    goal: List[str] = field(default_factory=lambda: ["familiar_objects"])
    model: List[str] = field(default_factory=lambda: ["gemini-2.5-pro"])
    seed: List[int] = field(default_factory=lambda: [3, 4, 5, 6, 7, 8])
    thumb_size: List[int] = field(default_factory=lambda: [128,])
    num_agents: int = 2_000
    noun_ylim: Optional[List[float]] = field(default_factory=lambda: [0.05, 0.085])


@dataclass
class TraitsSweep(SweepConfig):
    n_personality_traits: List[int] = field(default_factory=lambda: [
        0,
        10, 100, 1_000
    ])
    model: List[str] = field(default_factory=lambda: ["gemini-2.5-pro"])
    chat_history_turns: List[int] = field(default_factory=lambda: [1])
    seed: List[int] = field(default_factory=lambda: [3, 4, 5, 6, 7, 8])
    num_agents: int = 1_000
    noun_ylim: Optional[List[float]] = field(default_factory=lambda: [0.05, 0.08])


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
    model: List[str] = field(default_factory=lambda: ["qwen3-vl-30b-fp8"])
    seed: List[int] = field(default_factory=lambda: [3, 4, 5])
    num_agents: int = 1_000
    num_proc: int = 10
    gpu: bool = True
    novelty_ylim: Optional[List[float]] = field(default_factory=lambda: [0.6, 0.93])
    noun_ylim: Optional[List[float]] = field(default_factory=lambda: [0.04, 0.080])


@dataclass
class FullRandSelectProbQwen30b(SweepConfig):
    chat_history_turns: List[int] = field(default_factory=lambda: [1])
    temperature: List[Union[int, float, str]] = field(default_factory=lambda: [1.0])
    rand_select_prob: List[float] = field(default_factory=lambda: [0.05, 0.1, 0.25, 0.5])
    rand_select_mode: str = 'all'
    goal: List[str] = field(default_factory=lambda: ["familiar_objects"])
    model: List[str] = field(default_factory=lambda: ["qwen3-vl-30b-fp8"])
    seed: List[int] = field(default_factory=lambda: [3, 4, 5])
    num_agents: int = 1_000
    num_proc: int = 10
    gpu: bool = True
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
    num_agents: int = 9_377


@dataclass
class ObjectiveFreeSweep(SweepConfig):
    goal: List[str] = field(default_factory=lambda: ["none"])
    chat_history_turns: List[int] = field(default_factory=lambda: [1])
    temperature: List[Union[int, float, str]] = field(default_factory=lambda: [1.0])
    rand_select_prob: List[float] = field(default_factory=lambda: [0.0])
    model: List[str] = field(default_factory=lambda: ["gemini-2.5-pro"])
    seed: List[int] = field(default_factory=lambda: [3, 4, 5])
    num_agents: int = 500


@dataclass
class ModelSweep(SweepConfig):
    model: List[str] = field(default_factory=lambda: [
        # "gemini-2.5-pro",
        "gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-3-pro-preview",
        "gemini-random",
        # "qwen3-vl-8b",
    ])
    chat_history_turns: List[int] = field(default_factory=lambda: [1])
    seed: List[int] = field(default_factory=lambda: [3, 4, 5])
    num_agents: int = 1_500


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
    "chat_history_turns_qwen": ChatHistoryTurnsQwenSweep,
    "chat_history_turns_qwen_30b": ChatHistoryTurnsQwen30BSweep,
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
}