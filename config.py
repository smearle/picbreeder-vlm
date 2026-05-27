import copy
import re
from dataclasses import dataclass, field, fields
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Union

from hydra.conf import HelpConf, HydraConf
from hydra.core.config_store import ConfigStore

from constants import REPO_ROOT, SELECTION_BASELINES
from utils import _ensure_absolute


# Short, path-safe aliases for known local model ids, used in experiment dir names
# so they match the existing scheme (e.g. .../model-qwen3-vl-30b-fp8_...).
_MODEL_DIR_ALIASES = {
    "Qwen/Qwen3-VL-30B-A3B-Instruct-FP8": "qwen3-vl-30b-fp8",
    "Qwen/Qwen3-VL-8B-Instruct": "qwen3-vl-8b",
    "Qwen/Qwen3-VL-4B-Instruct": "qwen3-vl-4b",
    "Qwen/Qwen3-VL-2B-Instruct": "qwen3-vl-2b",
    "Qwen/Qwen3-VL-32B-Instruct": "qwen3-vl-32b",
    "Qwen/Qwen3-VL-235B-A22B-Instruct": "qwen3-vl-235b-bf16",
    "Qwen/Qwen3-VL-235B-A22B-Instruct-FP8": "qwen3-vl-235b-fp8",
}


@dataclass
class PicbreederConfig:
    log_dir: str = "logs_collaborative"  # Base directory for experiment logs
    goal: str = "familiar_objects"
    rows: int = 3  # Rows in the CPPN grid (legacy Picbreeder default)
    cols: int = 5  # Columns in the CPPN grid
    thumb_size: int = 128  # Pixel size for rendered genome thumbnails
    chat_history_turns: int = 1  # How many prior turns each agent sees (-1 keeps all)
    always_include_branched_image: bool = False  # Keep selected branch image as reference once branch sample falls out of chat history
    always_include_archive_sample: bool = False  # Keep the branching archive-sample step in context once it falls out of chat history
    allow_restart_from_publications: bool = True  # Allow restart branching from this agent's own prior publications
    model: str = "remote:Qwen/Qwen3-VL-30B-A3B-Instruct-FP8"  # local vLLM server (serve_local_vlm.sh); was "gemini-2.5-pro"
    temperature: Union[int, float, str] = 1.0  # Sampling temperature for Gemini responses (or "random")
    thinking_budget: int = -1
    request_rationale: bool = True  # Request natural-language reasoning in agent selections
    keep_query_images: bool = False  # Preserve query images/logs for post-run inspection
    compress_completed_agents: bool = True  # Zip completed agent dirs to reduce inode count
    scheme: str = "toggle"  # Rendering scheme: color, gray, or mono
    color_nudge: bool = False  # Append a Qwen-specific addendum nudging the model to use color (off by default)
    select_k: Optional[int] = None  # Max parents per generation (clamped to grid size when provided)
    agent_generations: int = 200  # Generations executed for each agent
    negative_anchors: Optional[str] = "negative_anchors"  # Filename for negative anchors list (in noun_lists/)
    num_agents: int = 200  # How many agents run sequentially in this session
    neat_config_path: Optional[Path] = None  # Path to NEAT config file (uses default if None)
    num_proc: int = 1  # Number of parallel agent processes
    warm_start_structure: int = 0  # Number of initial agents restricted to structure-only mutation
    experiment_dir: Optional[Path] = None  # Output directory for logs and artefacts
    output_activations: bool = True  # Enable CPPN output activation mutations
    input_activations: bool = False
    enable_crossover: bool = True  # Toggle Picbreeder-style crossover (CrossoverCombiner)
    selection_baseline: str = "none"  # Parent-selection policy: none/random/max-depth/max-nodes/clip-nouns
    rand_select_prob: float = 0.0  # Probability of short-circuiting VLM selection with a random pick
    rand_select_mode: str = "select-only"
    generate_personalities: bool = False  # Generate persona prompts before agent runs
    personality_path: Optional[Path] = None  # Destination for generated personality JSON
    resume: bool = False  # Resume a previously interrupted experiment
    resume_agent_id: Optional[str] = None  # Specific agent identifier to resume (requires resume=true)
    test_mode: bool = False  # Shortened settings for quick validation runs
    seed: int = 0  # Random seed for deterministic behaviour
    render_genome_diagrams: bool = False  # Render genome structure diagrams per generation
    log_raw_responses: bool = False  # When true, dump raw VLM responses to timestamped text files
    fixed_session_lengths: bool = True  # When true, agents run for exactly 20 generations and must publish at the 20th generation
    nounlist: str = "things_deduped"  # Noun list name (in noun_lists/) or path
    n_personality_traits: int = 0  # Number of random personality traits to prepend to the system prompt
    hydra: HydraConf = field(
        default_factory=lambda: HydraConf(
            help=HelpConf(
                app_name="collaborative_multi_agent",
                header=(
                    "Collaborative multi-agent Picbreeder workflow with shared archive.\n"
                    "\n"
                    "Key options:\n"
                    "  rows / cols             Grid dimensions rendered per generation.\n"
                    "  chat_history_turns      Conversation context length (-1 keeps full history).\n"
                    "  agent_generations       Generations executed by each agent.\n"
                    "  num_agents              Sequential agents to schedule for this run.\n"
                    "  num_proc                Parallel worker processes to launch.\n"
                    "  temperature             Sampling temperature for Gemini responses (or 'random').\n"
                    "  experiment_dir          Destination for logs, grids, and archives.\n"
                    "  selection_baseline      Automated parent selection (none/random/max-depth/max-nodes/clip-nouns).\n"
                    "  resume / resume_agent_id Resume an interrupted run from disk records.\n"
                ),
                footer="Override with +option=value (e.g. +scheme=color) or pass --cfg=job to inspect the full config.",
            )
        )
    )


def resolve_neat_config_path(cfg: PicbreederConfig) -> Path:
    if cfg.neat_config_path is not None:
        return Path(cfg.neat_config_path)
    base = REPO_ROOT / "picture2d"
    config_name = "interactive_config_color"
    return base / config_name


def ensure_valid_config(cfg: PicbreederConfig, *, original_cwd: Path) -> PicbreederConfig:
    cfg = copy.copy(cfg)
    cfg.selection_baseline = str(cfg.selection_baseline).lower()
    try:
        cfg.rand_select_prob = float(cfg.rand_select_prob)
    except (TypeError, ValueError):
        raise ValueError("rand-select-prob must be a number") from None
    # Resolve OmegaConf UnionNode to primitive type
    temp_raw = cfg.temperature
    if hasattr(temp_raw, '_value'):
        temp_raw = temp_raw._value()
    # Need to do this twice because it's nested (Union)
    if hasattr(temp_raw, '_value'):
        temp_raw = temp_raw._value()
    cfg.temperature = temp_raw
    if not (isinstance(cfg.temperature, (int, float)) or cfg.temperature == "random"):
        raise ValueError("temperature must be a number or 'random'") from None
    if cfg.temperature != "random" and (cfg.temperature < 0 or cfg.temperature > 2):
        raise ValueError("temperature must be between 0 and 2")
    if cfg.resume_agent_id and not cfg.resume:
        raise ValueError("--resume-agent-id requires --resume")
    if cfg.rows < 1 or cfg.cols < 1:
        raise ValueError("rows and cols must be positive integers")
    if cfg.thumb_size < 8:
        raise ValueError("thumb-size must be at least 8")
    if cfg.agent_generations < 1:
        raise ValueError("agent-generations must be at least 1")
    if cfg.num_agents < 1:
        raise ValueError("num-agents must be at least 1")
    if cfg.num_proc < 1:
        raise ValueError("num-proc must be at least 1")
    if cfg.select_k is not None and cfg.select_k < 1:
        raise ValueError("select-k must be at least 1 when provided")
    if cfg.warm_start_structure < 0:
        raise ValueError("warm-start-structure must be non-negative")
    if cfg.selection_baseline not in SELECTION_BASELINES:
        raise ValueError(f"selection-baseline must be one of {sorted(SELECTION_BASELINES)}")
    if not (0 <= cfg.rand_select_prob <= 1 or cfg.rand_select_prob == 2):
        raise ValueError("rand-select-prob must be between 0 and 1, or exactly 2")
    if cfg.rand_select_mode not in {"select-only", "all"}:
        raise ValueError("rand-select-mode must be 'select-only' or 'all'")
    if cfg.always_include_branched_image and cfg.always_include_archive_sample:
        raise ValueError(
            "always_include_branched_image and always_include_archive_sample cannot both be true."
        )
    if cfg.chat_history_turns == -1 and (
        cfg.always_include_branched_image or cfg.always_include_archive_sample
    ):
        raise ValueError(
            "always_include_branched_image/always_include_archive_sample are incompatible with chat_history_turns=-1."
        )

    is_hack_mode = (cfg.rand_select_prob == 2 and cfg.rand_select_mode == "all")
    if is_hack_mode and cfg.model != "gemini-2.5-pro":
        raise ValueError(
            f"Hack mode (rand_select_prob=2, rand_select_mode='all') ignores the model, "
            f"but a non-default model '{cfg.model}' was specified. Please leave model as default."
        )
    if is_hack_mode and cfg.chat_history_turns != 1:
        raise ValueError(
            f"Hack mode (rand_select_prob=2, rand_select_mode='all') ignores chat history, "
            f"but chat_history_turns was set to {cfg.chat_history_turns}. Please set it to -1 (default)."
        )
    if is_hack_mode and cfg.temperature != 1.0:
        raise ValueError(
            f"Hack mode (rand_select_prob=2, rand_select_mode='all') ignores temperature, "
            f"but temperature was set to {cfg.temperature}. Please set it to 1.0 (default)."
        )

    # Enforce 20 generations when fixed_session_lengths is enabled
    if cfg.fixed_session_lengths:
        cfg.agent_generations = 20

    config_path = resolve_neat_config_path(cfg)
    cfg.neat_config_path = config_path
    config_path = _ensure_absolute(Path(config_path), original_cwd)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")

    if cfg.experiment_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        if is_hack_mode:
            experiment_name = f"ag{cfg.agent_generations}"
        else:
            # Build a path-safe model tag. Prefer the short registry alias for known
            # local models so dirs match the existing naming scheme
            # (remote:Qwen/Qwen3-VL-30B-A3B-Instruct-FP8 -> qwen3-vl-30b-fp8); otherwise
            # strip any "remote:" prefix and replace path-unsafe chars ("/" and ":").
            _model_for_tag = cfg.model[len("remote:"):] if cfg.model.startswith("remote:") else cfg.model
            model_tag = _MODEL_DIR_ALIASES.get(
                _model_for_tag, re.sub(r"[^A-Za-z0-9._-]", "-", _model_for_tag)
            )
            experiment_name = f"th{cfg.chat_history_turns}_ag{cfg.agent_generations}_model-{model_tag}"

        experiment_name += f"_tb{cfg.thinking_budget}"
        if cfg.goal != "familiar_objects":
            experiment_name += f"_goal-{cfg.goal}"
        experiment_name += f"_scheme-{cfg.scheme}"
        if cfg.thumb_size != 128:
            experiment_name += f"_ts{cfg.thumb_size}"
        if cfg.warm_start_structure > 0:
            experiment_name += f"_warmstart{cfg.warm_start_structure}"
        if cfg.selection_baseline != "none":
            experiment_name += f"_baseline-{cfg.selection_baseline}"
        if cfg.rand_select_prob > 0:
            experiment_name += f"_randp{cfg.rand_select_prob:g}"
            if cfg.rand_select_mode == "all":
                experiment_name += "_rmode-all"
        if cfg.temperature != 1.0 and not is_hack_mode:
            if cfg.temperature == "random":
                experiment_name += "_temp-random"
            else:
                experiment_name += f"_temp{cfg.temperature:g}"
        experiment_name += "_personalities" if cfg.generate_personalities else "_nopersonalities"
        if cfg.n_personality_traits > 0:
            experiment_name += f"_traits{cfg.n_personality_traits}"
        experiment_name += "_norationale" if not cfg.request_rationale else ""
        if cfg.fixed_session_lengths:
            experiment_name += "_fixed-sesh"
        if cfg.always_include_branched_image:
            experiment_name += "_include-branch-img"
        if cfg.always_include_archive_sample:
            experiment_name += "_include-archive-sample"
        if cfg.color_nudge:
            experiment_name += "_colornudge"
        # experiment_name += f"_{timestamp}"
        experiment_name += f"_s{cfg.seed}"
        relative = Path(cfg.log_dir) / experiment_name
        exp_dir = _ensure_absolute(relative, original_cwd)
    else:
        exp_dir = _ensure_absolute(Path(cfg.experiment_dir), original_cwd)
    cfg.experiment_dir = exp_dir

    if cfg.resume:
        exp_dir = _ensure_absolute(Path(cfg.experiment_dir), original_cwd)
        if not exp_dir.exists():
            raise FileNotFoundError(f"Experiment directory not found at {exp_dir}")

    if not cfg.generate_personalities:
        cfg.personality_path = None
    else:
        if cfg.personality_path is None:
            personality_path = Path("outputs") / "personalities.json"
        else:
            personality_path = Path(cfg.personality_path)
        cfg.personality_path = _ensure_absolute(personality_path, original_cwd)

    if cfg.select_k is not None:
        max_possible = cfg.rows * cfg.cols
        cfg.select_k = min(max_possible, cfg.select_k)

    if cfg.test_mode and not cfg.resume:
        cfg.agent_generations = min(3, cfg.agent_generations)
        cfg.num_agents = min(2, cfg.num_agents)

    return cfg


PATH_FIELDS = {"config_path", "experiment_dir", "personality_path"}

def _serialize_config_for_worker(cfg: PicbreederConfig) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    for field_def in fields(PicbreederConfig):
        if field_def.name == "hydra":
            continue
        value = getattr(cfg, field_def.name)
        if isinstance(value, Path):
            payload[field_def.name] = str(value)
        else:
            payload[field_def.name] = value
    return payload


def _deserialize_config_for_worker(payload: Dict[str, Any]) -> PicbreederConfig:
    kwargs: Dict[str, Any] = {}
    for field_def in fields(PicbreederConfig):
        if field_def.name == "hydra":
            continue
        value = payload.get(field_def.name)
        if field_def.name in PATH_FIELDS and value is not None:
            value = Path(value)
        kwargs[field_def.name] = value
    return PicbreederConfig(**kwargs)


cs = ConfigStore.instance()
cs.store(name="collaborative_base", node=PicbreederConfig)
