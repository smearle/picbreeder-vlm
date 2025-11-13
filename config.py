import copy
from dataclasses import dataclass, field, fields
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

from hydra.conf import HelpConf, HydraConf
from hydra.core.config_store import ConfigStore

from constants import DEFAULT_AGENT_GENERATIONS, DEFAULT_CHAT_HISTORY_TURNS, REPO_ROOT, SELECTION_BASELINES
from utils import _ensure_absolute


@dataclass
class CollaborativeConfig:
    goal: str = "familiar_objects"
    rows: int = 3  # Rows in the CPPN grid (legacy Picbreeder default)
    cols: int = 5  # Columns in the CPPN grid
    thumb_size: int = 200  # Pixel size for rendered genome thumbnails
    chat_history_turns: int = DEFAULT_CHAT_HISTORY_TURNS  # How many prior turns each agent sees (-1 keeps all)
    scheme: str = "toggle"  # Rendering scheme: color, gray, or mono
    config_path: Optional[Path] = None  # Optional override for the NEAT config file
    select_k: Optional[int] = None  # Max parents per generation (clamped to grid size when provided)
    agent_generations: int = DEFAULT_AGENT_GENERATIONS  # Generations executed for each agent
    num_agents: int = 400  # How many agents run sequentially in this session
    num_proc: int = 1  # Number of parallel agent processes
    warm_start_structure: int = 0  # Number of initial agents restricted to structure-only mutation
    experiment_dir: Optional[Path] = None  # Output directory for logs and artefacts
    output_activations: bool = True  # Enable CPPN output activation mutations
    input_activations: bool = False
    selection_baseline: str = "none"  # Parent-selection policy: none/random/max-depth/max-nodes
    generate_personalities: bool = False  # Generate persona prompts before agent runs
    personality_path: Optional[Path] = None  # Destination for generated personality JSON
    resume: bool = False  # Resume a previously interrupted experiment
    resume_agent_id: Optional[str] = None  # Specific agent identifier to resume (requires resume=true)
    test_mode: bool = False  # Shortened settings for quick validation runs
    seed: Optional[int] = None  # Random seed for deterministic behaviour
    render_genome_diagrams: bool = False  # Render genome structure diagrams per generation
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
                    "  experiment_dir          Destination for logs, grids, and archives.\n"
                    "  selection_baseline      Automated parent selection (none/random/max-depth/max-nodes).\n"
                    "  resume / resume_agent_id Resume an interrupted run from disk records.\n"
                ),
                footer="Override with +option=value (e.g. +scheme=color) or pass --cfg=job to inspect the full config.",
            )
        )
    )


def resolve_config_path(cfg: CollaborativeConfig) -> Path:
    if cfg.config_path is not None:
        return Path(cfg.config_path)
    base = REPO_ROOT / "picture2d"
    config_name = "interactive_config_color"
    return base / config_name


def ensure_valid_config(cfg: CollaborativeConfig, *, original_cwd: Path) -> CollaborativeConfig:
    cfg = copy.copy(cfg)
    cfg.selection_baseline = str(cfg.selection_baseline).lower()
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

    config_path = resolve_config_path(cfg)
    config_path = _ensure_absolute(Path(config_path), original_cwd)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")
    cfg.config_path = config_path

    if cfg.resume:
        if cfg.experiment_dir is None:
            raise ValueError("--resume requires --experiment-dir pointing to an existing directory")
        exp_dir = _ensure_absolute(Path(cfg.experiment_dir), original_cwd)
        if not exp_dir.exists():
            raise FileNotFoundError(f"Experiment directory not found at {exp_dir}")
    else:
        if cfg.experiment_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            experiment_name = f"th{cfg.chat_history_turns}_ag{cfg.agent_generations}_na{cfg.num_agents}"
            if cfg.goal != "familiar_objects":
                experiment_name += f"_goal-{cfg.goal}"
            experiment_name += f"_scheme-{cfg.scheme}"
            if cfg.warm_start_structure > 0:
                experiment_name += f"_warmstart{cfg.warm_start_structure}"
            if cfg.selection_baseline != "none":
                experiment_name += f"_baseline-{cfg.selection_baseline}"
            experiment_name += "_personalities" if cfg.generate_personalities else "_nopersonalities"
            experiment_name += f"_{timestamp}"
            relative = Path("logs_collaborative") / experiment_name
            exp_dir = _ensure_absolute(relative, original_cwd)
        else:
            exp_dir = _ensure_absolute(Path(cfg.experiment_dir), original_cwd)
        exp_dir.mkdir(parents=True, exist_ok=True)
    cfg.experiment_dir = exp_dir

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

def _serialize_config_for_worker(cfg: CollaborativeConfig) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    for field_def in fields(CollaborativeConfig):
        if field_def.name == "hydra":
            continue
        value = getattr(cfg, field_def.name)
        if isinstance(value, Path):
            payload[field_def.name] = str(value)
        else:
            payload[field_def.name] = value
    return payload


def _deserialize_config_for_worker(payload: Dict[str, Any]) -> CollaborativeConfig:
    kwargs: Dict[str, Any] = {}
    for field_def in fields(CollaborativeConfig):
        if field_def.name == "hydra":
            continue
        value = payload.get(field_def.name)
        if field_def.name in PATH_FIELDS and value is not None:
            value = Path(value)
        kwargs[field_def.name] = value
    return CollaborativeConfig(**kwargs)


cs = ConfigStore.instance()
cs.store(name="collaborative_base", node=CollaborativeConfig)
