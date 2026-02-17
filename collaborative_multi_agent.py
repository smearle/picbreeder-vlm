#!/usr/bin/env python3
"""Multi-agent collaborative Picbreeder workflow with shared archive.

This script orchestrates a chain of visual-language-model (VLM) driven agents
that collaboratively evolve CPPN image populations. Each agent can now run up to
200 generations (turns) with full conversational history. After completing its
run, it publishes a favourite image to a public archive. Future agents review
that archive before starting; they may branch from any previously published
favourites or begin from a fresh population. Each agent can optionally publish
its own favourite image back to the archive, enabling collaborative evolution
across sessions.

Key behaviours:
- Every agent reuses the NEAT-driven workflow from ``auto_evolve.py``.
- Agents record numbered grids and selection overlays per generation.
- Branching and favourite-selection prompts (and responses) are logged.
- The archive stores PNGs, pickled genomes, and rolling checkpoints whenever a
    new favourite is added.
"""

from __future__ import annotations

import copy
import gzip
import json
import multiprocessing
import os
import pickle
import random
import shutil
import traceback
import time
import sys
import socket
import subprocess
import requests
import atexit
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from multiprocessing.connection import Connection
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence, Set, Tuple
from itertools import count

import hydra
from hydra.utils import get_original_cwd

from agent_runner import AgentRunner, AgentQuitRequested, AgentRestartRequested
from config import PicbreederConfig, _deserialize_config_for_worker, _serialize_config_for_worker, ensure_valid_config
import personalities

import neat
from neat.population import CompleteExtinctionException

from chat import (
    ensure_vlm_credentials,
)
from archive_manager import (
    ArchiveEntry,
    ArchiveManager,
    atomic_write_json,
)
from neat_components import (
    CHECKPOINT_SUFFIX,
    LATEST_POPULATION_FILENAME,
    InteractiveStagnation,
    PicbreederGenome,
    apply_picbreeder_config_defaults,
    sync_population_output_activations,
)
from constants import (
    ARCHIVE_DIR_NAME,
    AGENT_DIR_PREFIX,
    MAX_MODEL_LEN,
    PERSONALITY_BATCH_SIZE,
    PERSONALITY_TOTAL,
    RATING_BATCH_SIZE,
    REPO_NAME,
    REPO_ROOT,
)
from picbreeder_reproduction import PicbreederReproduction
from prompts import GOAL_PROMPTS
from utils import relative_suffix_after_dir, apply_random_seed

TRAITS_FILE = "personality_traits.json"

from rate_archive_with_vlm import (
    ArchiveEntry as RatingArchiveEntry,
    RatingResult,
    build_rating_system_prompt,
    format_rating_entry_label,
    parse_rating_batch_response,
)
from vlm_backends import create_vlm_backend, VLMBackend, is_local_model, _resolve_qwen_model_name

# Cache for VLM backend used in rating
_RATING_BACKEND: Optional[VLMBackend] = None

GEMINI_RANDOM_MODELS = [
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-3-pro-preview",
]


def _get_rating_backend(model) -> VLMBackend:
    """Get or create a cached VLM backend for rating."""
    global _RATING_BACKEND
    if _RATING_BACKEND is None or _RATING_BACKEND.name != model:
        _RATING_BACKEND = create_vlm_backend(model)
    return _RATING_BACKEND


def find_latest_checkpoint(population_dir: Path) -> Optional[Path]:
    latest_path = population_dir / LATEST_POPULATION_FILENAME
    if latest_path.exists():
        return latest_path
    # TODO: Remove this. For backward compatibility only.
    pattern = f"gen_*{CHECKPOINT_SUFFIX}"
    candidates = sorted(population_dir.glob(pattern))
    if not candidates:
        return None
    return candidates[-1]


def restore_population_from_checkpoint(checkpoint_path: Path) -> neat.Population:
    with gzip.open(checkpoint_path, "rb") as handle:
        next_generation, config, population_data, species_set, random_state = pickle.load(handle)
    random.setstate(random_state)
    return neat.Population(config, (population_data, species_set, next_generation))


def _rehydrate_reproduction_state(population: neat.Population) -> None:
    reproduction = getattr(population, "reproduction", None)
    if not isinstance(reproduction, PicbreederReproduction):
        return

    population_keys: Set[int] = set(population.population.keys())
    species_set = getattr(population, "species", None)
    if species_set is not None:
        for species in species_set.species.values():
            population_keys.update(species.members.keys())

    if not population_keys:
        reproduction.genome_indexer = count(1)
        reproduction.ancestors = {}
        return

    reproduction.genome_indexer = count(max(population_keys) + 1)
    reproduction.ancestors = dict(getattr(reproduction, "ancestors", {}))
    for key in population_keys:
        reproduction.ancestors.setdefault(key, tuple())


def _safe_int(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: int, lower: int, upper: int) -> int:
    return max(lower, min(upper, value))


def _resolve_task_temperature(cfg: PicbreederConfig, task: AgentTask) -> float:
    temperature = cfg.temperature
    if isinstance(temperature, str) and temperature.lower() == "random":
        if task.temperature is not None:
            return float(task.temperature)
        seed = cfg.seed if cfg.seed is not None else 0
        rng = random.Random(f"{seed}:{task.agent_id}:{task.agent_index}")
        return rng.uniform(0.0, 2.0)
    return float(temperature)


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _wait_for_vlm_server(base_url: str, timeout: int = 600) -> bool:
    print(f"Waiting for vLLM server at {base_url}...")
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"{base_url}/models", timeout=5)
            if response.status_code == 200:
                print("vLLM server is ready.")
                return True
        except requests.RequestException:
            pass
        time.sleep(5)
    return False


def _start_vlm_server(model: str, port: int) -> subprocess.Popen:
    full_model_names = {
        'qwen3-vl-8b': 'qwen/qwen3-vl-8b-instruct'
    }
    full_model_name = full_model_names.get(model, model)
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", full_model_name,
        "--port", str(port),
        "--trust-remote-code",
        "--gpu-memory-utilization", "0.9",
        "--max-model-len", f"{MAX_MODEL_LEN}",
    ]
    print(f"Starting vLLM server: {' '.join(cmd)}")
    log_path = f"vllm_server_{port}.log"
    log_file = open(log_path, "w")
    process = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
    return process



# ---------------------------------------------------------------------------
# Agent directory compression/decompression helpers
# ---------------------------------------------------------------------------

AGENT_ARCHIVE_SUFFIX = ".zip"


def _compress_agent_directory(agent_dir: Path) -> Optional[Path]:
    """Compress a completed agent directory into a zip archive.

    The zip is created alongside the original directory (e.g. agent_000.zip)
    and the original directory is removed on success.

    Returns the path to the created archive, or None on failure.
    """
    if not agent_dir.is_dir():
        return None
    archive_path = agent_dir.with_suffix(AGENT_ARCHIVE_SUFFIX)
    try:
        # shutil.make_archive wants the base name without extension
        base_name = str(archive_path.with_suffix(""))
        created_path = shutil.make_archive(base_name, "zip", agent_dir.parent, agent_dir.name)
        shutil.rmtree(agent_dir, ignore_errors=True)
        return Path(created_path)
    except Exception as exc:
        print(f"Warning: failed to compress {agent_dir}: {exc}")
        # Clean up partial archive if it exists
        if archive_path.exists():
            try:
                archive_path.unlink()
            except OSError:
                pass
        return None


def _decompress_agent_directory(archive_path: Path) -> Optional[Path]:
    """Decompress an agent archive back into its original directory.

    Returns the path to the restored directory, or None on failure.
    """
    if not archive_path.is_file() or archive_path.suffix != AGENT_ARCHIVE_SUFFIX:
        return None
    agent_dir = archive_path.with_suffix("")
    try:
        shutil.unpack_archive(archive_path, agent_dir.parent)
        archive_path.unlink()
        return agent_dir
    except Exception as exc:
        print(f"Warning: failed to decompress {archive_path}: {exc}")
        return None


def _ensure_agent_directory_available(agent_dir: Path) -> Path:
    """Ensure an agent directory is available (decompress if needed).

    If the directory exists, returns it unchanged.
    If a .zip archive exists in its place, decompresses it.
    Otherwise returns the original path (caller must handle non-existence).
    """
    if agent_dir.is_dir():
        return agent_dir
    archive_path = agent_dir.with_suffix(AGENT_ARCHIVE_SUFFIX)
    if archive_path.is_file():
        restored = _decompress_agent_directory(archive_path)
        if restored is not None:
            return restored
    return agent_dir


@dataclass
class AgentTask:
    agent_id: str
    agent_index: int
    resume: bool
    warm_start_active: bool
    personality_prompt: Optional[str]
    personality_trait: Optional[str] = None
    temperature: Optional[float] = None
    branching_decision: Optional[Dict[str, Any]] = None
    favorite_selection: Optional[Dict[str, Any]] = None
    archive_entry: Optional[Dict[str, Any]] = None
    model: Optional[str] = None

    def to_message(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "agent_index": self.agent_index,
            "resume": self.resume,
            "warm_start_active": self.warm_start_active,
            "personality_prompt": self.personality_prompt,
            "personality_trait": self.personality_trait,
            "temperature": self.temperature,
            "branching_decision": self.branching_decision,
            "favorite_selection": self.favorite_selection,
            "archive_entry": self.archive_entry,
            "model": self.model,
        }


@dataclass
class AgentRecord:
    agent_id: str
    status: str = "in_progress"
    personality_trait: Optional[str] = None
    temperature: Optional[float] = None
    branching_decision: Optional[Dict[str, Any]] = None
    favorite_selection: Optional[Dict[str, Any]] = None
    archive_entry: Optional[Dict[str, Any]] = None
    created_at: Optional[str] = None
    resumed_at: Optional[str] = None
    completed_at: Optional[str] = None
    last_generation: int = 0
    extinct: bool = False
    quit_requested: bool = False
    quit_reason: Optional[str] = None
    model: Optional[str] = None


@dataclass
class BranchingDecisionMessage:
    agent_id: str
    decision: Dict[str, Any]
    restart: bool = False
    type: str = field(init=False, default="branching_decision")


@dataclass
class ProgressMessage:
    agent_id: str
    generation: int
    favorite: Optional[Dict[str, Any]]
    archive: Optional[Dict[str, Any]]
    type: str = field(init=False, default="progress")


@dataclass
class JobCompleteMessage:
    agent_id: str
    favorite: Optional[Dict[str, Any]]
    archive_entry: Optional[Dict[str, Any]]
    extinct: bool
    final_generation: int
    branching_decision: Optional[Dict[str, Any]]
    quit_requested: bool
    quit_reason: Optional[str]
    type: str = field(init=False, default="job_complete")


@dataclass
class WorkerState:
    index: int
    task_conn: Connection
    archive_conn: Connection
    process: multiprocessing.Process
    current_task: Optional[AgentTask] = None
    stopping: bool = False
    errored: bool = False


def build_neat_config(
    neat_config_path: Path,
    rows: int,
    cols: int,
    enable_output_activations: bool,
    enable_input_activations: bool,
    enable_crossover: bool,
) -> neat.Config:
    config = neat.Config(
        PicbreederGenome,
        PicbreederReproduction,
        neat.DefaultSpeciesSet,
        InteractiveStagnation,
        str(neat_config_path),
    )
    apply_picbreeder_config_defaults(
        config,
        enable_output_activations=enable_output_activations,
        enable_input_activations=enable_input_activations,
        enable_crossover=enable_crossover,
    )
    config.pop_size = rows * cols
    return config


class CollaborativeMultiAgentOrchestrator:
    """Coordinates sequential agent runs and archive management."""

    def __init__(
        self,
        config: PicbreederConfig,
        experiment_dir: Path,
        rows: int,
        cols: int,
        thumb_size: int,
        scheme: str,
        neat_config_path: Path,
        select_k: Optional[int],
        agent_generations: int,
        warm_start_structure: int,
        enable_output_activations: bool,
        selection_baseline: str,
        seed: Optional[int],
        chat_history_turns: int,
        render_genome_diagrams: bool = False,
        process_index: Optional[int] = None,
    ) -> None:
        self.config = config
        self.experiment_dir = experiment_dir
        self.chat_history_turns = chat_history_turns
        self.rows = rows
        self.cols = cols
        self.thumb_size = thumb_size
        self.scheme = scheme
        self.neat_config_path = neat_config_path
        self.select_k = select_k
        self.agent_generations = agent_generations
        self.warm_start_structure = warm_start_structure
        self.enable_output_activations = enable_output_activations
        self.enable_input_activations = config.input_activations
        self.enable_crossover = config.enable_crossover
        self.selection_baseline = selection_baseline
        self.seed = seed
        self.render_genome_diagrams = render_genome_diagrams
        self.process_index = process_index
        self.global_generation_count = 0
        self.start_time = 0.0
        self.keep_query_images = bool(getattr(self.config, "keep_query_images", False))
        self.compress_completed_agents = bool(getattr(self.config, "compress_completed_agents", True))
        self.archive_manager = ArchiveManager(
            self.experiment_dir / ARCHIVE_DIR_NAME,
            goal_prompt=GOAL_PROMPTS[self.config.goal],
            thumb_size=self.thumb_size,
        )
        self.agents_dir = self.experiment_dir / "agents"
        self.agents_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_path = self.experiment_dir / "agents_metadata.json"
        self._metadata = self._load_metadata()
        # Determine personality traits list (before verifying config, so it can be saved)
        self._personality_traits_list: List[str] = []
        
        # Check metadata first
        existing_run_config = self._metadata.get("run_config") or {}
        saved_traits = existing_run_config.get("sampled_personality_traits")
        
        if saved_traits and isinstance(saved_traits, list):
            self._personality_traits_list = saved_traits
            print(f"Loaded {len(self._personality_traits_list)} personality traits from metadata.")
        elif self.config.n_personality_traits > 0:
            if self._metadata.get("agents"):
                raise RuntimeError(
                    "Resuming an experiment that has existing agents, but 'sampled_personality_traits' "
                    "is missing from metadata. Cannot ensure consistency with a new random sample."
                )

            traits_path = self.experiment_dir.parent / TRAITS_FILE # Or just use repo root?
            # User said: "we'll hardcode the path as a constant for now".
            # The script saves to current directory (repo root).
            traits_path = Path(TRAITS_FILE)
            if not traits_path.exists():
                # Try finding it in repo root if experiment dir is nested
                traits_path = REPO_ROOT / TRAITS_FILE
            
            if traits_path.exists():
                try:
                    with open(traits_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        full_traits_list = data.get("traits", [])
                    
                    if full_traits_list and self.config.n_personality_traits > 0:
                        count = min(len(full_traits_list), self.config.n_personality_traits)
                        self._personality_traits_list = random.sample(full_traits_list, count)
                        print(f"Subsampled {len(self._personality_traits_list)} personality traits from {len(full_traits_list)} loaded traits.")
                    else:
                        self._personality_traits_list = []
                        
                except Exception as e:
                    print(f"Error loading personality traits from {traits_path}: {e}")
            else:
                 print(f"Warning: Personality traits file not found at {traits_path}, but n_personality_traits > 0.")

        self._ensure_run_config()
        self._parallel_workers_active = False

        self._personality_records: List[Dict[str, Any]] = []
        self._personality_prompts: List[str] = []
        if self.config.generate_personalities:
            self._personality_records = self._load_personality_records()
            self._personality_prompts = [
                self._format_personality_prompt(record) for record in self._personality_records
            ]
            if not self._personality_prompts:
                raise ValueError(
                    "Personality generation is enabled, but no personality prompts were loaded."
                )
            print(
                f"Loaded {len(self._personality_prompts)} personality prompts from {self.config.personality_path}"
            )

    def _load_metadata(self) -> Dict[str, Any]:
        if self.metadata_path.exists():
            metadata = self._read_metadata_file()
        else:
            metadata = self._default_metadata()
            self._write_metadata_file(metadata)
        # metadata, _ = self._ensure_metadata_defaults(metadata)
        # if changed:
        #     self._write_metadata_file(metadata)
        return metadata

    def _default_metadata(self) -> Dict[str, Any]:
        return {
            "created_at": datetime.now().isoformat(),
            "next_agent_number": 0,
            "agents": {},
            "seed": self.seed,
            "run_config": None,
        }

    def _read_metadata_file(self) -> Dict[str, Any]:
        with self.metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)

        # TODO: This is for backward compatibility only and can be removed
        if isinstance(metadata.get("agents"), list):
            agents_dict = {}
            for entry in metadata.get("agents", []):
                agents_dict[entry["agent_id"]] = entry
            metadata["agents"] = agents_dict

        # Backward compatibility for personality_traits -> personality_trait
        # TODO: Remove this block once old runs are successfully resumed
        raw_agents = metadata.get("agents", {})
        processed_agents = {}
        for agent_id, entry in raw_agents.items():
            if "personality_traits" in entry:
                entry["personality_trait"] = entry.pop("personality_traits")
            # If it was a list (legacy), take the first element or join?
            # Based on previous context, user said "can't we just use the record" and "assume each agent only gets one trait".
            # The old code stored it as a list. The new code expects a string.
            # So if we find a list, we should probably take the first item.
            trait = entry.get("personality_trait")
            if isinstance(trait, list):
                if trait:
                    entry["personality_trait"] = trait[0]
                else:
                    entry["personality_trait"] = None
            processed_agents[agent_id] = AgentRecord(**entry)
        
        metadata["agents"] = processed_agents
        # End backward compatibility block. Replace with this:
        # metadata["agents"] = {agent_id: AgentRecord(**entry) for agent_id, entry in metadata.get("agents", {}).items()}

        return metadata

    def _write_metadata_file(self, metadata: Dict[str, Any]) -> None:
        serializable = dict(metadata)
        agents = serializable.get("agents", {})
        serializable["agents"] = {agent_id: asdict(entry) if isinstance(entry, AgentRecord) else entry for agent_id, entry in agents.items()}
        atomic_write_json(self.metadata_path, serializable)

    def _reload_metadata(self) -> Dict[str, Any]:
        if self.metadata_path.exists():
            metadata = self._read_metadata_file()
        else:
            metadata = self._default_metadata()
        # metadata, _ = self._ensure_metadata_defaults(metadata)
        self._metadata = metadata
        return metadata

    def _mutate_metadata(self, mutator: Callable[[Dict[str, Any]], Any], read_write: bool = True) -> Any:
        if read_write:
            if self.metadata_path.exists():
                metadata = self._read_metadata_file()
            else:
                metadata = self._default_metadata()
        else:
            metadata = self._metadata
        # metadata, _ = self._ensure_metadata_defaults(metadata)
        result = mutator(metadata)
        if read_write:
            self._write_metadata_file(metadata)
        self._metadata = metadata
        # print(self._metadata['agents'])
        return result

    def _ensure_run_config(self) -> None:
        self._reload_metadata()
        if self.config.personality_path:
            self.config.personality_path = relative_suffix_after_dir(self.config.personality_path, dir_name=REPO_NAME)
        personality_path_str = str(self.config.personality_path) if self.config.personality_path else None
        run_config = {
            "rows": self.rows,
            "cols": self.cols,
            "thumb_size": self.thumb_size,
            "scheme": self.scheme,
            "select_k": self.select_k,
            "agent_generations": self.agent_generations,
            "enable_output_activations": self.enable_output_activations,
            "enable_input_activations": self.enable_input_activations,
            "enable_crossover": self.enable_crossover,
            "warm_start_structure": self.warm_start_structure,
            "selection_baseline": self.selection_baseline,
            "temperature": self.config.temperature,
            "generate_personalities": self.config.generate_personalities,
            "personality_path": personality_path_str,
            "request_rationale": self.config.request_rationale,
            "fixed_session_lengths": self.config.fixed_session_lengths,
            "n_personality_traits": self.config.n_personality_traits,
            "sampled_personality_traits": self._personality_traits_list,
        }
        existing = self._metadata.get("run_config")
        if existing is None:
            self._mutate_metadata(lambda data: data.__setitem__("run_config", run_config))
            return
        if existing['personality_path']:
            existing["personality_path"] = str(relative_suffix_after_dir(Path(existing.get("personality_path")), dir_name=REPO_NAME))
        missing = []
        if "warm_start_structure" not in existing:
            missing.append(("warm_start_structure", 0))
        if "selection_baseline" not in existing:
            missing.append(("selection_baseline", "none"))
        if "temperature" not in existing:
            missing.append(("temperature", self.config.temperature))
        if "generate_personalities" not in existing:
            missing.append(("generate_personalities", self.config.generate_personalities))
        if "personality_path" not in existing:
            missing.append(("personality_path", personality_path_str))
        if "request_rationale" not in existing:
            missing.append(("request_rationale", self.config.request_rationale))
        if "enable_input_activations" not in existing:
            missing.append(("enable_input_activations", self.enable_input_activations))
        if "enable_crossover" not in existing:
            missing.append(("enable_crossover", self.enable_crossover))
        if "fixed_session_lengths" not in existing:
            missing.append(("fixed_session_lengths", self.config.fixed_session_lengths))
        if "n_personality_traits" not in existing:
            missing.append(("n_personality_traits", self.config.n_personality_traits))
        if "sampled_personality_traits" not in existing:
            missing.append(("sampled_personality_traits", self._personality_traits_list))
        if missing:
            def mutator(data: Dict[str, Any]) -> None:
                details = data.setdefault("run_config", {})
                for key, value in missing:
                    details.setdefault(key, value)

            self._mutate_metadata(mutator, read_write=True)
            existing = self._metadata.get("run_config")
        mismatches = {
            key: (existing.get(key), value) for key, value in run_config.items() if existing.get(key) != value
        }
        if mismatches:
            details = ", ".join(
                f"{key}: expected {prev!r}, received {current!r}" for key, (prev, current) in mismatches.items()
            )
            raise ValueError(
                "Experiment configuration does not match existing metadata. "
                f"Use a fresh experiment directory or matching parameters ({details})."
            )

    def _load_personality_records(self) -> List[Dict[str, Any]]:
        path = self.config.personality_path
        if path is None:
            raise ValueError(
                "Personality generation is enabled but no personality path was configured."
            )
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Expected personality dataset at {path}, but the file does not exist."
            ) from exc
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Failed to parse personality dataset at {path}: {exc}"
            ) from exc
        if not isinstance(data, list):
            raise ValueError(
                f"Personality dataset at {path} must contain a JSON array of objects."
            )
        records = [entry for entry in data if isinstance(entry, dict)]
        if not records:
            raise ValueError(
                f"Personality dataset at {path} contained no valid personality objects."
            )
        if len(records) < PERSONALITY_TOTAL:
            print(
                f"Warning: generated {len(records)} personalities, expected {PERSONALITY_TOTAL}."
            )
        return records

    @staticmethod
    def _format_personality_prompt(record: Dict[str, Any]) -> str:
        name = str(record.get("name") or "Unnamed personality").strip()
        age_value = record.get("age")
        age_text: Optional[str]
        if isinstance(age_value, (int, float)):
            age_text = f"{int(age_value)}-year-old"
        else:
            try:
                age_text = f"{int(age_value)}-year-old"
            except (TypeError, ValueError):
                age_text = None
        profession = str(record.get("profession") or "").strip()
        identity_parts = [part for part in (age_text, profession) if part]
        identity_suffix = f" ({', '.join(identity_parts)})" if identity_parts else ""
        header = f"Persona Context: You are roleplaying as {name}{identity_suffix}."

        background = str(record.get("background") or "").strip()
        upbringing = str(record.get("upbringing") or "").strip()
        personality_desc = str(record.get("personality") or "").strip()
        short_bio = str(record.get("short_bio") or "").strip()

        lines = [header]
        if background:
            lines.append(f"Background: {background}")
        if upbringing:
            lines.append(f"Upbringing: {upbringing}")
        if personality_desc:
            lines.append(f"Personality: {personality_desc}")
        if profession:
            lines.append(f"Profession: {profession}")
        if short_bio:
            lines.append(f"Short Bio: {short_bio}")
        return "\n".join(lines)

    def _personality_prompt_for_agent(self, agent_id: str) -> Optional[str]:
        if not self._personality_prompts:
            return None
        return random.choice(self._personality_prompts)

    def _allocate_agent_id(self) -> str:
        def mutator(metadata: Dict[str, Any]) -> int:
            agent_number = metadata.get("next_agent_number", 0)
            metadata["next_agent_number"] = agent_number + 1
            return agent_number

        agent_number = self._mutate_metadata(mutator)
        return self._agent_id_from_index(agent_number)

    @staticmethod
    def _parse_agent_index(agent_id: str) -> Optional[int]:
        if not agent_id.startswith(AGENT_DIR_PREFIX):
            return None
        suffix = agent_id[len(AGENT_DIR_PREFIX) :]
        try:
            return int(suffix)
        except ValueError:
            return None

    def _agent_id_from_index(self, index: int) -> str:
        return f"{AGENT_DIR_PREFIX}{index:03d}"

    def _agent_dir(self, agent_id: str) -> Path:
        return self.agents_dir / agent_id

    def _ensure_agent_number_progress(self, agent_id: str, read_write: bool) -> None:
        index = self._parse_agent_index(agent_id)
        if index is None:
            raise ValueError(f"Invalid agent identifier '{agent_id}'.")

        def mutator(metadata: Dict[str, Any]) -> None:
            metadata["next_agent_number"] = max(metadata.get("next_agent_number", 0), index + 1)

        self._mutate_metadata(mutator, read_write=read_write)

    def _is_warm_start_agent(self, agent_id: str) -> bool:
        if self.warm_start_structure <= 0:
            return False
        index = self._parse_agent_index(agent_id)
        return index is not None and index < self.warm_start_structure

    def _find_agent_record(self, agent_id: str) -> Optional[AgentRecord]:
        agents = self._metadata.get("agents", {})
        record: AgentRecord = agents.get(agent_id, None)
        return record

    def _register_agent(self, agent_id: str, read_write: bool) -> AgentRecord:
        def mutator(metadata: Dict[str, Any]) -> AgentRecord:
            agents = metadata.setdefault("agents", {})
            for record in agents:
                if isinstance(record, AgentRecord) and record.agent_id == agent_id:
                    return record
            record = AgentRecord(
                agent_id=agent_id,
                status="in_progress",
                created_at=datetime.now().isoformat(),
                last_generation=0,
            )
            agents[agent_id] = record
            return record

        return self._mutate_metadata(mutator, read_write=read_write)

    def _resolve_agent_temperature(
        self,
        agent_id: str,
        agent_index: int,
        record: Optional[AgentRecord],
        read_write: bool,
    ) -> float:
        temperature = self.config.temperature
        if isinstance(temperature, str) and temperature.lower() == "random":
            if record is not None and record.temperature is not None:
                return float(record.temperature)
            seed = self.seed if self.seed is not None else 0
            rng = random.Random(f"{seed}:{agent_id}:{agent_index}")
            resolved = rng.uniform(0.0, 2.0)
            if record is not None:
                self._update_agent_record(agent_id, temperature=resolved, read_write=read_write)
            return resolved
        return float(temperature)

    def _update_agent_record(self, agent_id: str, read_write: bool = True, **updates: Any) -> None:
        def mutator(metadata: Dict[str, Any]) -> None:
            record = metadata["agents"][agent_id]
            if isinstance(record, AgentRecord):
                for key, value in updates.items():
                    setattr(record, key, value)
            elif isinstance(record, dict):
                record.update(updates)

        self._mutate_metadata(mutator, read_write=read_write)

    def _find_agent_to_resume(
        self,
        resume_agent_id: Optional[str],
        allowed_agent_ids: Optional[Set[str]] = None,
    ) -> Optional[AgentRecord]:
        self._reload_metadata()
        agents = self._metadata.get("agents", {})
        finished = {"complete", "extinct", "stopped"}
        if resume_agent_id:
            if allowed_agent_ids is not None and resume_agent_id not in allowed_agent_ids:
                return None
            record = next(
                (entry for agent_id, entry in agents.items() if isinstance(entry, AgentRecord) and entry.agent_id == resume_agent_id),
                None,
            )
            if record is None:
                raise ValueError(f"Agent '{resume_agent_id}' not found in experiment metadata.")
            if record.status in finished:
                raise ValueError(f"Agent '{resume_agent_id}' has already completed.")
            return record
        for record in reversed(agents):
            if not isinstance(record, AgentRecord):
                continue
            if allowed_agent_ids is not None and record.agent_id not in allowed_agent_ids:
                continue
            if record.status not in finished:
                return record
        return None

    def _hydrate_agent_record_from_disk(self, record: AgentRecord) -> None:
        agent_dir = self._agent_dir(record.agent_id)
        # Decompress agent directory if it was archived
        agent_dir = _ensure_agent_directory_available(agent_dir)
        logs_dir = agent_dir / "logs"
        updates: Dict[str, Any] = {}
        branch_path = logs_dir / "branching_selection.json"
        if record.branching_decision is None and branch_path.exists():
            try:
                branching = json.loads(branch_path.read_text(encoding="utf-8"))
                record.branching_decision = branching
                updates["branching_decision"] = branching
            except json.JSONDecodeError:
                pass
        favourite_path = logs_dir / "favorite_selection.json"
        if record.favorite_selection is None and favourite_path.exists():
            try:
                favorite = json.loads(favourite_path.read_text(encoding="utf-8"))
                record.favorite_selection = favorite
                updates["favorite_selection"] = favorite
            except json.JSONDecodeError:
                pass
        publication_path = logs_dir / "publication_history.jsonl"
        if record.archive_entry is None and publication_path.exists():
            last_payload: Optional[Dict[str, Any]] = None
            with publication_path.open("r", encoding="utf-8") as handle:
                for raw_line in handle:
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        last_payload = json.loads(line)
                    except json.JSONDecodeError:
                        continue
            if last_payload and last_payload.get("archive_entry_id"):
                entry = self.archive_manager.get_entry(last_payload["archive_entry_id"])
                if entry is not None:
                    record.archive_entry = entry.as_dict()
                    if record.favorite_selection is None:
                        record.favorite_selection = last_payload
                        updates.setdefault("favorite_selection", last_payload)
                    updates["archive_entry"] = record.archive_entry
        if updates:
            self._update_agent_record(record.agent_id, **updates)

    def _load_population_for_agent(self, agent_dir: Path) -> Tuple[Optional[neat.Population], Optional[Path]]:
        population_dir = agent_dir / "populations"
        if not population_dir.exists():
            return None, None
        checkpoint_path = find_latest_checkpoint(population_dir)
        if checkpoint_path is None:
            return None, None
        population = restore_population_from_checkpoint(checkpoint_path)
        apply_picbreeder_config_defaults(
            population.config,
            enable_output_activations=self.enable_output_activations,
            enable_input_activations=self.enable_input_activations,
            enable_crossover=self.enable_crossover,
        )
        sync_population_output_activations(population, population.config.genome_config)
        _rehydrate_reproduction_state(population)
        return population, checkpoint_path

    def _build_config(self) -> neat.Config:
        return build_neat_config(
            self.neat_config_path,
            self.rows,
            self.cols,
            self.enable_output_activations,
            self.enable_input_activations,
            self.enable_crossover,
        )

    def _build_runner(
        self,
        agent_id: str,
        agent_dir: Path,
        neat_config: neat.Config,
        population: Optional[neat.Population],
        resume: bool,
    ) -> AgentRunner:
        callback = lambda generation, favorite, archive: self._on_generation_progress(
            agent_id, generation, favorite, archive
        )
        personality_prompt = self._personality_prompt_for_agent(agent_id)
        record = self._find_agent_record(agent_id)
        agent_index = self._parse_agent_index(agent_id) or 0
        temperature = self._resolve_agent_temperature(
            agent_id,
            agent_index,
            record,
            read_write=False,
        )
        model = self._resolve_agent_model(
            agent_id,
            agent_index,
            record,
            read_write=False,
        )
        
        final_personality_prompt = personality_prompt
        if record and record.personality_trait:
            traits_str = record.personality_trait
            if final_personality_prompt:
                final_personality_prompt = f"{final_personality_prompt}\n\n{traits_str}"
            else:
                final_personality_prompt = traits_str

        config = copy.copy(self.config)
        config.temperature = temperature
        config.model = model
        
        try:
            return AgentRunner(
                agent_id,
                agent_dir,
                config=config,
                neat_config=neat_config,
                archive_manager=self.archive_manager,
                generations=self.agent_generations,
                rows=self.rows,
                cols=self.cols,
                thumb_size=self.thumb_size,
                scheme=self.scheme,
                select_k=self.select_k,
                chat_history_turns=self.chat_history_turns,
                selection_baseline=self.selection_baseline,
                population=population,
                progress_callback=callback,
                resume_mode=resume,
                warm_start_active=self._is_warm_start_agent(agent_id),
                render_genome_diagrams=self.render_genome_diagrams,
                process_index=self.process_index,
                personality_prompt=final_personality_prompt,
            )
        except ValueError as e:
            if "Could not restore image for chat history" in str(e):
                print(f"[{agent_id}] Encountered corruption during initialization: {e}. Cleaning up agent directory and restarting from scratch.")
                if agent_dir.exists():
                    shutil.rmtree(agent_dir)
                agent_dir.mkdir(parents=True, exist_ok=True)

                return AgentRunner(
                    agent_id,
                    agent_dir,
                    config=config,
                    neat_config=neat_config,
                    archive_manager=self.archive_manager,
                    generations=self.agent_generations,
                    rows=self.rows,
                    cols=self.cols,
                    thumb_size=self.thumb_size,
                    scheme=self.scheme,
                    select_k=self.select_k,
                    chat_history_turns=self.chat_history_turns,
                    selection_baseline=self.selection_baseline,
                    population=None,
                    progress_callback=callback,
                    resume_mode=False,
                    warm_start_active=self._is_warm_start_agent(agent_id),
                    render_genome_diagrams=self.render_genome_diagrams,
                    process_index=self.process_index,
                    personality_prompt=final_personality_prompt,
                )
            raise e

    def _on_generation_progress(
        self,
        agent_id: str,
        generation: int,
        favorite_payload: Optional[Dict[str, Any]],
        archive_payload: Optional[Dict[str, Any]],
    ) -> None:
        self.global_generation_count += 1
        if self.start_time > 0:
            elapsed = time.time() - self.start_time
            rate = self.global_generation_count / elapsed if elapsed > 0 else 0.0
            sys.stderr.write(f"\rGlobal Generations: {self.global_generation_count} ({rate:.2f} gen/s)")
            sys.stderr.flush()

        is_new_archive_entry = False
        if archive_payload:
            entry_id = archive_payload.get("id")
            if entry_id:
                record = self._find_agent_record(agent_id)
                previous_entry_id: Optional[str] = None
                if record:
                    previous_payload = record.archive_entry
                    if isinstance(previous_payload, dict):
                        previous_entry_id = previous_payload.get("id")
                is_new_archive_entry = entry_id != previous_entry_id
        updates: Dict[str, Any] = {"last_generation": generation}
        if favorite_payload is not None:
            updates["favorite_selection"] = favorite_payload
        if archive_payload is not None:
            updates["archive_entry"] = archive_payload
        self._update_agent_record(agent_id, **updates)
        if is_new_archive_entry:
            if not self._parallel_workers_active:
                self._maybe_run_auto_rating_serial()
            else:
                return is_new_archive_entry

    def _execute_runner(
        self,
        agent_id: str,
        runner: AgentRunner,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[ArchiveEntry], bool, int, bool, Optional[str]]:
        remaining = max(0, runner.generations - runner.population.generation)
        extinct = False
        quit_requested = False
        quit_reason: Optional[str] = None
        while remaining > 0:
            try:
                runner.population.run(runner.evaluate_generation, remaining)
                break
            except CompleteExtinctionException:
                extinct = True
                break
            except AgentQuitRequested as exc:
                quit_requested = True
                quit_reason = str(exc) or runner.quit_reason
                break
            except AgentRestartRequested:
                decision = runner.apply_pending_restart()
                if decision is not None:
                    self._update_agent_record(agent_id, branching_decision=decision)
                remaining = max(0, runner.generations - runner.population.generation)
                continue
        archive_entry = runner.favorite_archive_entry
        favorite = runner.favorite_decision if runner.favorite_decision else None
        final_generation = runner.population.generation
        if quit_reason is None:
            quit_reason = runner.quit_reason
        return favorite, archive_entry, extinct, final_generation, quit_requested, quit_reason

    def run_agents(
        self,
        total_agents: int,
        resume: bool,
        resume_agent_id: Optional[str],
        num_workers: int = 1,
    ) -> None:
        using_parallel = num_workers > 1
        previous_mode = self._parallel_workers_active
        self._parallel_workers_active = using_parallel
        self.global_generation_count = 0
        self.start_time = time.time()
        try:
            if not using_parallel:
                self._run_agents_serial(total_agents, resume, resume_agent_id)
            else:
                self._run_agents_parallel(total_agents, resume, resume_agent_id, num_workers)
        finally:
            sys.stderr.write("\n")
            self._parallel_workers_active = previous_mode

    def _run_agents_serial(
        self,
        total_agents: int,
        resume: bool,
        resume_agent_id: Optional[str],
    ) -> None:
        self._reload_metadata()
        target_indices = list(range(total_agents))
        if not target_indices:
            return
        target_ids = {self._agent_id_from_index(index) for index in target_indices}
        resume_request = resume_agent_id if (resume_agent_id and resume_agent_id in target_ids) else None
        resumed = False
        if resume:
            resumed = self._resume_agent(resume_request, allowed_agent_ids=target_ids)
            if resume_request and not resumed:
                print(f"No in-progress agent found for resume request '{resume_request}'.")
        for index in target_indices:
            agent_id = self._agent_id_from_index(index)
            record = self._find_agent_record(agent_id)
            if record is not None and record.status in {"complete", "extinct", "stopped"}:
                continue
            if record is not None:
                self._resume_agent(agent_id, allowed_agent_ids={agent_id})
                continue
            self._run_new_agent(agent_id=agent_id)

    def _run_agents_parallel(
        self,
        total_agents: int,
        resume: bool,
        resume_agent_id: Optional[str],
        num_workers: int,
    ) -> None:
        self._reload_metadata()
        pending_tasks = self._prepare_parallel_tasks(total_agents, resume, resume_agent_id)
        if not pending_tasks:
            return

        ctx = multiprocessing.get_context("spawn")
        config_payload = _serialize_config_for_worker(self.config)
        
        # Setup vLLM server if needed
        vllm_process = None
        if num_workers > 1 and is_local_model(self.config.model):
            try:
                # Resolve model name to ensure consistency between server and client
                resolved_model = _resolve_qwen_model_name(self.config.model)
                port = _find_free_port()
                vllm_process = _start_vlm_server(resolved_model, port)
                base_url = f"http://localhost:{port}/v1"
                if not _wait_for_vlm_server(base_url):
                    raise RuntimeError("Failed to start vLLM server.")
                
                # Wrap config payload
                # Note: We need to modify the config to use 'remote:' prefix for the model
                # But self.config is immutable (sort of).
                # We can modify the payload dict.
                # However, _deserialize_config_for_worker might complain if we change values?
                # Usually it just loads data.
                
                # We tell the worker to use the remote backend via the environment variable
                # AND by prefixing the model name so create_vlm_backend uses VLLMClientBackend.
                
                # Modify the serialized payload directly
                if "model" in config_payload:
                    config_payload["model"] = f"remote:{resolved_model}"
                
                # Wrap in the special structure we defined in _continual_agent_worker
                config_payload = {
                    "__vlm_config__": {"base_url": base_url},
                    "config": config_payload
                }
                
                atexit.register(vllm_process.kill)
            except Exception as e:
                print(f"Error starting vLLM server: {e}")
                if vllm_process:
                    vllm_process.kill()
                raise e

        worker_states: List[WorkerState] = []
        conn_to_worker: Dict[Connection, WorkerState] = {}
        errors: List[str] = []
        abort_requested = False

        try:
            for worker_index in range(num_workers):
                parent_task_conn, child_task_conn = ctx.Pipe()
                parent_archive_conn, child_archive_conn = ctx.Pipe()
                process = ctx.Process(
                    target=_continual_agent_worker,
                    args=(child_task_conn, child_archive_conn, config_payload, worker_index),
                )
                process.start()
                state = WorkerState(
                    index=worker_index,
                    task_conn=parent_task_conn,
                    archive_conn=parent_archive_conn,
                    process=process,
                )
                worker_states.append(state)
                conn_to_worker[parent_task_conn] = state
                conn_to_worker[parent_archive_conn] = state

            active_workers = len(worker_states)
            while active_workers > 0 and conn_to_worker:
                ready_conns = multiprocessing.connection.wait(list(conn_to_worker.keys()))
                for conn in ready_conns:
                    state = conn_to_worker.get(conn)
                    if state is None:
                        continue
                    try:
                        message = conn.recv()
                    except EOFError:
                        errors.append(f"Worker {state.index} exited unexpectedly.")
                        abort_requested = True
                        pending_tasks.clear()
                        active_workers -= 1
                        self._cleanup_worker(state, conn_to_worker)
                        continue

                    if conn is state.archive_conn:
                        self._handle_archive_rpc(state, message)
                        continue

                    msg_type = message.get("type")
                    if msg_type == "ready":
                        self._handle_worker_ready(state, pending_tasks, abort_requested)
                    elif msg_type == "branching_decision":
                        payload = {k: v for k, v in message.items() if k != "type"}
                        branching_msg = BranchingDecisionMessage(**payload)
                        if state.current_task and state.current_task.agent_id == branching_msg.agent_id:
                            state.current_task.branching_decision = branching_msg.decision
                        self._update_agent_record(branching_msg.agent_id, branching_decision=branching_msg.decision)
                    elif msg_type == "progress":
                        payload = {k: v for k, v in message.items() if k != "type"}
                        progress = ProgressMessage(**payload)
                        self._on_generation_progress(
                            progress.agent_id,
                            progress.generation,
                            progress.favorite,
                            progress.archive,
                        )
                    elif msg_type == "rating_result":
                        self._apply_rating_results_from_worker(message)
                    elif msg_type == "rating_failed":
                        trigger_entry_count = _safe_int(message.get("trigger_entry_count"), default=0)
                        error_text = message.get("error")
                        if error_text:
                            print(f"Auto-rating failed: {error_text}")
                        if trigger_entry_count:
                            self.archive_manager.mark_auto_rating_failed(trigger_entry_count)
                    elif msg_type == "job_complete":
                        payload = {k: v for k, v in message.items() if k != "type"}
                        job_complete = JobCompleteMessage(**payload)
                        self._handle_job_complete(state, job_complete)
                        state.current_task = None
                    elif msg_type == "task_failed":
                        agent_id = message.get("agent_id")
                        error_text = message.get("error") or "Unknown failure"
                        errors.append(
                            f"Worker {state.index} failed on {agent_id or 'unknown agent'}: {error_text}"
                        )
                        abort_requested = True
                        pending_tasks.clear()
                        state.current_task = None
                    elif msg_type == "worker_error":
                        error_text = message.get("error") or "Unknown worker error"
                        errors.append(f"Worker {state.index} error: {error_text}")
                        abort_requested = True
                        pending_tasks.clear()
                        state.errored = True
                    elif msg_type == "stopped":
                        active_workers -= 1
                        self._cleanup_worker(state, conn_to_worker)
                    else:
                        continue
        finally:
            if vllm_process:
                print("Stopping vLLM server...")
                vllm_process.terminate()
                try:
                    vllm_process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    vllm_process.kill()
                    
            for state in worker_states:
                try:
                    state.task_conn.close()
                except Exception:
                    pass
                try:
                    state.archive_conn.close()
                except Exception:
                    pass
                state.process.join()

        if errors:
            raise RuntimeError("\n".join(errors))

    def _handle_worker_ready(
        self,
        state: WorkerState,
        pending_tasks: Deque[AgentTask],
        abort: bool = False,
    ) -> None:
        if state.current_task is not None:
            return
        if abort:
            if not state.stopping:
                state.task_conn.send({"type": "stop"})
                state.stopping = True
            return
        if pending_tasks:
            task = pending_tasks.popleft()
            state.current_task = task
            payload = {
                "type": "run_agent",
                "task": task.to_message(),
            }
            state.task_conn.send(payload)
            state.stopping = False
            return
        if not state.stopping:
            state.task_conn.send({"type": "stop"})
            state.stopping = True

    def _cleanup_worker(self, state: WorkerState, conn_map: Dict[Connection, WorkerState]) -> None:
        conn_map.pop(state.task_conn, None)
        conn_map.pop(state.archive_conn, None)
        try:
            state.task_conn.close()
        except Exception:
            pass
        try:
            state.archive_conn.close()
        except Exception:
            pass

    def _handle_archive_rpc(self, state: WorkerState, message: Dict[str, Any]) -> None:
        if not isinstance(message, dict) or message.get("type") != "archive_call":
            return
        call_id = message.get("call_id")
        method_name = str(message.get("method") or "")
        args = message.get("args") or []
        kwargs = message.get("kwargs") or {}
        try:
            result = self._invoke_archive_method(method_name, args, kwargs)
            response = {"type": "archive_response", "call_id": call_id, "result": result}
        except Exception as exc:  # pylint: disable=broad-except
            traceback.print_exc()
            response = {"type": "archive_response", "call_id": call_id, "error": str(exc)}
        state.archive_conn.send(response)

    def _invoke_archive_method(self, method_name: str, args: Sequence[Any], kwargs: Dict[str, Any]) -> Any:
        if method_name == "get_entries":
            return self.archive_manager.entries
        archive_methods = {
            "create_archive_grid": self.archive_manager.create_archive_grid,
            "sample_branching_entries": self.archive_manager.sample_branching_entries,
            "get_elite_names": self.archive_manager.get_elite_names,
            "get_entry": self.archive_manager.get_entry,
            "load_genome": self.archive_manager.load_genome,
            "add_entry": self.archive_manager.add_entry,
            "remove_entry": self.archive_manager.remove_entry,
            "prepare_auto_rating_batch": self.archive_manager.prepare_auto_rating_batch,
        }
        method = archive_methods.get(method_name)
        if method is None:
            raise AttributeError(f"Archive method '{method_name}' is not available for workers.")
        return method(*args, **kwargs)

    def _apply_rating_results_from_worker(self, payload: Dict[str, Any]) -> None:
        trigger_entry_count = _safe_int(payload.get("trigger_entry_count"), default=0)
        raw_results = payload.get("results")
        if not isinstance(raw_results, dict) or not raw_results:
            if trigger_entry_count:
                self.archive_manager.mark_auto_rating_failed(trigger_entry_count)
            return
        parsed: Dict[str, RatingResult] = {}
        for entry_id, value in raw_results.items():
            if isinstance(value, RatingResult):
                parsed[entry_id] = value
                continue
            if not isinstance(value, dict):
                continue
            try:
                score = float(value.get("score", 0.0))
            except (TypeError, ValueError):
                continue
            parsed[entry_id] = RatingResult(
                score=score,
                justification=value.get("justification"),
                reported_title=value.get("reported_title"),
            )
        if parsed:
            applied = self.archive_manager.apply_rating_results(parsed)
            if trigger_entry_count and applied > 0:
                self.archive_manager.mark_auto_rating_complete(trigger_entry_count, applied)
            elif trigger_entry_count:
                self.archive_manager.mark_auto_rating_failed(trigger_entry_count)
        elif trigger_entry_count:
            self.archive_manager.mark_auto_rating_failed(trigger_entry_count)

    def _maybe_run_auto_rating_serial(self) -> None:
        try:
            targets, trigger_entry_count = self.archive_manager.prepare_auto_rating_batch(RATING_BATCH_SIZE)
        except Exception as exc:  # pylint: disable=broad-except
            print(f"Auto-rating scheduling failed: {exc}")
            targets = []
            trigger_entry_count = 0
        if not targets:
            return
        goal_prompt = self.archive_manager.goal_prompt
        archive_dir = self.archive_manager.archive_dir
        try:
            results = _perform_rating(
                targets,
                goal_prompt,
                archive_dir,
                model=self.config.model,
                rand_select_prob=self.config.rand_select_prob,
                rand_select_mode=getattr(self.config, "rand_select_mode", "select-only"),
            )
        except Exception as exc:  # pylint: disable=broad-except
            print(f"Auto-rating failed: {exc}")
            self.archive_manager.mark_auto_rating_failed(trigger_entry_count)
            return
        if not results:
            self.archive_manager.mark_auto_rating_failed(trigger_entry_count)
            return
        applied = self.archive_manager.apply_rating_results(results)
        if applied > 0:
            self.archive_manager.mark_auto_rating_complete(trigger_entry_count, applied)
        else:
            self.archive_manager.mark_auto_rating_failed(trigger_entry_count)

    def _handle_job_complete(self, state: WorkerState, payload: JobCompleteMessage) -> None:
        agent_id = payload.agent_id
        if not agent_id:
            return
        task = state.current_task
        agent_dir = self._agent_dir(agent_id)
        favorite_payload = payload.favorite
        archive_entry_data = payload.archive_entry
        archive_entry: Optional[ArchiveEntry]
        if isinstance(archive_entry_data, ArchiveEntry):
            archive_entry = archive_entry_data
        elif isinstance(archive_entry_data, dict):
            try:
                archive_entry = ArchiveEntry.from_dict(archive_entry_data)
            except Exception:  # pylint: disable=broad-except
                archive_entry = None
        else:
            archive_entry = None
        extinct = payload.extinct
        final_generation = payload.final_generation
        quit_requested = payload.quit_requested
        quit_reason = payload.quit_reason
        branching_decision = None
        if payload.branching_decision is not None:
            branching_decision = payload.branching_decision
        elif task is not None:
            branching_decision = task.branching_decision
        else:
            record = self._find_agent_record(agent_id)
            if record is not None:
                branching_decision = record.branching_decision
        self._finalize_agent(
            agent_id,
            agent_dir,
            branching_decision,
            favorite_payload,
            archive_entry,
            extinct=extinct,
            final_generation=final_generation,
            quit_requested=quit_requested,
            quit_reason=quit_reason,
        )

    def _resolve_agent_model(
        self,
        agent_id: str,
        agent_index: int,
        record: Optional[AgentRecord],
        read_write: bool,
    ) -> str:
        model = self.config.model
        if model == "gemini-random":
            if record is not None and record.model:
                return record.model
            seed = self.seed if self.seed is not None else 0
            rng = random.Random(f"{seed}:{agent_id}:{agent_index}:model")
            resolved = rng.choice(GEMINI_RANDOM_MODELS)
            if record is not None:
                self._update_agent_record(agent_id, model=resolved, read_write=read_write)
            return resolved
        return model

    def _build_new_agent_task(self, agent_id: str, agent_index: int) -> AgentTask:
        # print(f"[{agent_id}] Starting new agent...")
        self._ensure_agent_number_progress(agent_id, read_write=False)
        agent_dir = self._agent_dir(agent_id)
        record = self._register_agent(agent_id, read_write=False)
        if record.status != "in_progress":
            self._update_agent_record(agent_id, status="in_progress", last_generation=0, read_write=False)
        warm_start_active = self._is_warm_start_agent(agent_id)
        personality_prompt = self._personality_prompt_for_agent(agent_id)
        temperature = self._resolve_agent_temperature(
            agent_id,
            agent_index,
            record,
            read_write=False,
        )
        model = self._resolve_agent_model(
            agent_id,
            agent_index,
            record,
            read_write=False,
        )
        
        selected_trait = None
        if self._personality_traits_list and self.config.n_personality_traits > 0:
            # Sample exactly ONE trait from the subsampled list
            trait = random.choice(self._personality_traits_list)
            selected_trait = trait
            self._update_agent_record(agent_id, personality_trait=selected_trait, read_write=False)
            print(f"[{agent_id}] Assigned personality traits: {selected_trait}")

        return AgentTask(
            agent_id=agent_id,
            agent_index=agent_index,
            resume=False,
            warm_start_active=warm_start_active,
            personality_prompt=personality_prompt,
            personality_trait=selected_trait,
            temperature=temperature,
            model=model,
        )

    def _build_resume_agent_task(self, record: AgentRecord) -> AgentTask:
        # print(f"[{record.agent_id}] Resuming agent from previous state...")
        self._hydrate_agent_record_from_disk(record)
        agent_id = record.agent_id
        # agent_dir = self._agent_dir(agent_id)
        agent_index = self._parse_agent_index(agent_id) or 0
        warm_start_active = self._is_warm_start_agent(agent_id)
        personality_prompt = self._personality_prompt_for_agent(agent_id)
        temperature = self._resolve_agent_temperature(
            agent_id,
            agent_index,
            record,
            read_write=True,
        )
        model = self._resolve_agent_model(
            agent_id,
            agent_index,
            record,
            read_write=True,
        )
        # last_generation = record.last_generation
        # self._update_agent_record(
        #     agent_id,
        #     status="in_progress",
        #     resumed_at=datetime.now().isoformat(),
        #     last_generation=last_generation,
        # )
        return AgentTask(
            agent_id=agent_id,
            agent_index=agent_index,
            resume=True,
            warm_start_active=warm_start_active,
            personality_prompt=personality_prompt,
            personality_trait=record.personality_trait,
            temperature=temperature,
            branching_decision=record.branching_decision,
            favorite_selection=record.favorite_selection,
            archive_entry=record.archive_entry,
            model=model,
        )

    def _prepare_parallel_tasks(
        self,
        total_agents: int,
        resume: bool,
        resume_agent_id: Optional[str],
    ) -> Deque[AgentTask]:
        pending: Deque[AgentTask] = deque()
        target_indices = list(range(total_agents))
        if not target_indices:
            return pending
        target_ids = {self._agent_id_from_index(index) for index in target_indices}
        scheduled: Set[str] = set()
        if resume and resume_agent_id and resume_agent_id in target_ids:
            record = self._find_agent_record(resume_agent_id)
            if record and record.status not in {"complete", "extinct", "stopped"}:
                pending.append(self._build_resume_agent_task(record))
                scheduled.add(resume_agent_id)
        for index in target_indices:
            agent_id = self._agent_id_from_index(index)
            if agent_id in scheduled:
                continue
            record = self._find_agent_record(agent_id)
            if record is not None and record.status in {"complete", "extinct", "stopped"}:
                continue
            if record is not None:
                task = self._build_resume_agent_task(record)
            else:
                task = self._build_new_agent_task(agent_id, index)
            pending.append(task)
            scheduled.add(agent_id)
        self._write_metadata_file(self._metadata)
        return pending

    def _run_new_agent(self, agent_id: Optional[str] = None) -> None:
        if agent_id is None:
            agent_id = self._allocate_agent_id()
        else:
            self._ensure_agent_number_progress(agent_id, read_write=True)
        agent_dir = self._agent_dir(agent_id)
        self._register_agent(agent_id, read_write=True)
        
        if self._personality_traits_list and self.config.n_personality_traits > 0:
            # Sample exactly ONE trait from the subsampled list
            trait = random.choice(self._personality_traits_list)
            selected_trait = trait
            self._update_agent_record(agent_id, personality_trait=selected_trait, read_write=True)
            print(f"[{agent_id}] Assigned personality traits: {selected_trait}")

        config = self._build_config()
        runner = self._build_runner(agent_id, agent_dir, config, None, resume=False)
        decision = runner.select_starting_point()
        runner.initialise_population(decision)
        runner.branching_decision = decision
        self._update_agent_record(agent_id, branching_decision=decision, status="in_progress", last_generation=0)
        favorite, archive_entry, extinct, final_generation, quit_requested, quit_reason = self._execute_runner(agent_id, runner)
        self._finalize_agent(
            agent_id,
            agent_dir,
            decision,
            favorite,
            archive_entry,
            extinct=extinct,
            final_generation=final_generation,
            quit_requested=quit_requested,
            quit_reason=quit_reason,
        )
        self._maybe_run_auto_rating_serial()

    def _resume_agent(
        self,
        resume_agent_id: Optional[str],
        allowed_agent_ids: Optional[Set[str]] = None,
    ) -> bool:
        record = self._find_agent_to_resume(resume_agent_id, allowed_agent_ids=allowed_agent_ids)
        if record is None:
            return False
        self._hydrate_agent_record_from_disk(record)
        agent_id = record.agent_id
        if record.personality_trait:
            print(f"[{agent_id}] Resuming with personality traits: {record.personality_trait}")
        agent_dir = self._agent_dir(agent_id)
        # Ensure directory is available (may have been compressed)
        agent_dir = _ensure_agent_directory_available(agent_dir)
        population, checkpoint_path = self._load_population_for_agent(agent_dir)
        if population is not None and checkpoint_path is not None:
            print(f"[{agent_id}] Resuming from checkpoint {checkpoint_path.name}")
            config = population.config
        else:
            print(f"[{agent_id}] Resume requested without checkpoint; restarting agent workflow.")
            config = self._build_config()
        runner = self._build_runner(agent_id, agent_dir, config, population, resume=True)
        decision = record.branching_decision
        if decision is None:
            decision = runner.select_starting_point()
            runner.initialise_population(decision)
            runner.branching_decision = decision
            self._update_agent_record(agent_id, branching_decision=decision)
        else:
            runner.branching_decision = decision
            if population is None:
                runner.initialise_population(decision)
            else:
                runner.restore_reference_context(decision)
        favorite_selection = record.favorite_selection
        if favorite_selection is not None:
            runner.favorite_decision = favorite_selection
        archive_entry_dict = record.archive_entry
        if archive_entry_dict is not None:
            try:
                entry = ArchiveEntry.from_dict(archive_entry_dict)
            except Exception:
                entry = None
            if entry is not None:
                runner.favorite_archive_entry = entry
        self._update_agent_record(
            agent_id,
            status="in_progress",
            resumed_at=datetime.now().isoformat(),
            last_generation=runner.population.generation,
        )
        favorite, archive_entry, extinct, final_generation, quit_requested, quit_reason = self._execute_runner(agent_id, runner)
        self._finalize_agent(
            agent_id,
            agent_dir,
            decision,
            favorite,
            archive_entry,
            extinct=extinct,
            final_generation=final_generation,
            quit_requested=quit_requested,
            quit_reason=quit_reason,
        )
        self._maybe_run_auto_rating_serial()
        return True

    def _cleanup_agent_artifacts(self, agent_dir: Path, *, compress: bool = False) -> None:
        images_dir = agent_dir / "images"
        if images_dir.exists():
            shutil.rmtree(images_dir, ignore_errors=True)
        if not self.keep_query_images:
            queries_dir = agent_dir / "queries"
            if queries_dir.exists():
                shutil.rmtree(queries_dir, ignore_errors=True)
        latest_checkpoint = agent_dir / "populations" / LATEST_POPULATION_FILENAME
        try:
            latest_checkpoint.unlink(missing_ok=True)
        except OSError:
            pass
        # Compress the agent directory if requested (reduces inode count on cluster)
        if compress and self.compress_completed_agents and agent_dir.is_dir():
            _compress_agent_directory(agent_dir)

    def _finalize_agent(
        self,
        agent_id: str,
        agent_dir: Path,
        branching_decision: Optional[Dict[str, Any]],
        favourite: Optional[Dict[str, Any]],
        archive_entry: Optional[ArchiveEntry],
        *,
        extinct: bool,
        final_generation: int,
        quit_requested: bool = False,
        quit_reason: Optional[str] = None,
    ) -> None:
        if quit_requested:
            status = "stopped"
        else:
            status = "extinct" if extinct else "complete"
        updates: Dict[str, Any] = {
            "extinct": extinct,
            "status": status,
            "completed_at": datetime.now().isoformat(),
            "last_generation": final_generation,
            "quit_requested": quit_requested,
        }
        if quit_reason:
            updates["quit_reason"] = quit_reason
        if branching_decision is not None:
            updates["branching_decision"] = branching_decision
        if favourite is not None:
            updates["favorite_selection"] = favourite
        if archive_entry is not None:
            updates["archive_entry"] = archive_entry.as_dict()
        self._update_agent_record(agent_id, **updates)
        self._cleanup_agent_artifacts(agent_dir, compress=True)



def _build_orchestrator(
    cfg: PicbreederConfig,
    process_index: Optional[int] = None,
) -> CollaborativeMultiAgentOrchestrator:
    return CollaborativeMultiAgentOrchestrator(
        config=cfg,
        experiment_dir=cfg.experiment_dir,
        rows=cfg.rows,
        cols=cfg.cols,
        thumb_size=cfg.thumb_size,
        scheme=cfg.scheme,
        neat_config_path=cfg.neat_config_path,
        select_k=cfg.select_k,
        agent_generations=cfg.agent_generations,
        warm_start_structure=cfg.warm_start_structure,
        enable_output_activations=cfg.output_activations,
        selection_baseline=cfg.selection_baseline,
        seed=cfg.seed,
        chat_history_turns=cfg.chat_history_turns,
        render_genome_diagrams=cfg.render_genome_diagrams,
        process_index=process_index,
    )


def _generate_personalities_for_run(cfg: PicbreederConfig) -> None:
    if cfg.personality_path is None:
        raise ValueError(
            "Personality generation is enabled but no personality path was computed."
        )
    output_dir = cfg.personality_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"Generating {PERSONALITY_TOTAL} personalities at {cfg.personality_path}"
    )
    original_output_dir = personalities.OUTPUT_DIR
    try:
        personalities.OUTPUT_DIR = str(output_dir)
        os.makedirs(personalities.OUTPUT_DIR, exist_ok=True)
        personalities.main(
            total_personalities=PERSONALITY_TOTAL,
            batch_size=PERSONALITY_BATCH_SIZE,
        )
    finally:
        personalities.OUTPUT_DIR = original_output_dir
    if not cfg.personality_path.exists():
        raise FileNotFoundError(
            f"Personality generation did not produce the expected file at {cfg.personality_path}."
        )


def run(cfg: PicbreederConfig) -> None:
    if cfg.generate_personalities and not cfg.resume:
        _generate_personalities_for_run(cfg)
    apply_random_seed(cfg.seed)
    if cfg.selection_baseline == "none":
        ensure_vlm_credentials()
    orchestrator = _build_orchestrator(cfg)
    orchestrator.run_agents(
        cfg.num_agents,
        resume=cfg.resume,
        resume_agent_id=cfg.resume_agent_id,
        num_workers=max(1, cfg.num_proc),
    )
    orchestrator.archive_manager.create_archive_grid(cfg.thumb_size)


class RemoteArchiveClient:
    def __init__(self, conn: Connection) -> None:
        self._conn = conn
        self._ids = count(1)

    def _call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        call_id = next(self._ids)
        self._conn.send(
            {
                "type": "archive_call",
                "call_id": call_id,
                "method": method,
                "args": args,
                "kwargs": kwargs,
            }
        )
        while True:
            response = self._conn.recv()
            if not isinstance(response, dict):
                continue
            if response.get("type") != "archive_response":
                continue
            if response.get("call_id") != call_id:
                continue
            if "error" in response:
                raise RuntimeError(response["error"])
            return response.get("result")

    def create_archive_grid(self, *args: Any, **kwargs: Any) -> Any:
        return self._call("create_archive_grid", *args, **kwargs)

    def sample_branching_entries(self, *args: Any, **kwargs: Any) -> Any:
        return self._call("sample_branching_entries", *args, **kwargs)

    def get_elite_names(self, *args: Any, **kwargs: Any) -> Any:
        return self._call("get_elite_names", *args, **kwargs)

    def get_entry(self, entry_id: str) -> Any:
        return self._call("get_entry", entry_id)

    def load_genome(self, entry_id: str) -> Any:
        return self._call("load_genome", entry_id)

    def add_entry(self, *args: Any, **kwargs: Any) -> Any:
        return self._call("add_entry", *args, **kwargs)

    def remove_entry(self, entry_id: str) -> Any:
        return self._call("remove_entry", entry_id)

    def prepare_auto_rating_batch(self, limit: Optional[int] = None) -> Any:
        if limit is None:
            return self._call("prepare_auto_rating_batch")
        return self._call("prepare_auto_rating_batch", limit)

    @property
    def entries(self) -> Any:
        return self._call("get_entries")


def _deserialize_agent_task(payload: Dict[str, Any]) -> AgentTask:
    return AgentTask(
        agent_id=str(payload["agent_id"]),
        agent_index=int(payload.get("agent_index", 0)),
        resume=bool(payload.get("resume")),
        warm_start_active=bool(payload.get("warm_start_active")),
        personality_prompt=payload.get("personality_prompt"),
        personality_trait=payload.get("personality_trait"),
        temperature=payload.get("temperature"),
        branching_decision=payload.get("branching_decision"),
        favorite_selection=payload.get("favorite_selection"),
        archive_entry=payload.get("archive_entry"),
        model=payload.get("model"),
    )


def _load_population_for_worker(
    agent_dir: Path,
    enable_output_activations: bool,
    enable_input_activations: bool,
    enable_crossover: bool,
) -> Tuple[Optional[neat.Population], Optional[Path]]:
    population_dir = agent_dir / "populations"
    if not population_dir.exists():
        return None, None
    checkpoint_path = find_latest_checkpoint(population_dir)
    if checkpoint_path is None:
        return None, None
    population = restore_population_from_checkpoint(checkpoint_path)
    apply_picbreeder_config_defaults(
        population.config,
        enable_output_activations=enable_output_activations,
        enable_input_activations=enable_input_activations,
        enable_crossover=enable_crossover,
    )
    sync_population_output_activations(population, population.config.genome_config)
    _rehydrate_reproduction_state(population)
    return population, checkpoint_path


def _execute_runner_in_worker(
    agent_id: str,
    runner: AgentRunner,
    task_conn: Connection,
) -> Tuple[Optional[Dict[str, Any]], Optional[ArchiveEntry], bool, int, bool, Optional[str]]:
    remaining = max(0, runner.generations - runner.population.generation)
    extinct = False
    quit_requested = False
    quit_reason: Optional[str] = None
    while remaining > 0:
        try:
            runner.population.run(runner.evaluate_generation, remaining)
            break
        except CompleteExtinctionException:
            extinct = True
            break
        except AgentQuitRequested as exc:
            quit_requested = True
            quit_reason = str(exc) or runner.quit_reason
            break
        except AgentRestartRequested:
            decision = runner.apply_pending_restart()
            if decision is not None:
                task_conn.send(asdict(BranchingDecisionMessage(agent_id=agent_id, decision=decision, restart=True)))
            remaining = max(0, runner.generations - runner.population.generation)
            continue
    archive_entry = runner.favorite_archive_entry
    favorite = runner.favorite_decision if runner.favorite_decision else None
    final_generation = runner.population.generation
    if quit_reason is None:
        quit_reason = runner.quit_reason
    return favorite, archive_entry, extinct, final_generation, quit_requested, quit_reason


def _perform_rating(
    targets: Sequence[Dict[str, str]],
    goal_prompt: str,
    archive_dir: Path,
    model: str,
    rand_select_prob: float = 0.0,
    rand_select_mode: str = "select-only",
) -> Dict[str, RatingResult]:
    if model == "gemini-random":
        model = random.choice(GEMINI_RANDOM_MODELS)
    
    # print(f"Generating ratings with model: {model}")

    if not targets:
        return {}
    
    if rand_select_mode == "all" and rand_select_prob == 2.0:
        results: Dict[str, RatingResult] = {}
        print("Hack mode active: assigning random ratings to all targets.")
        for target in targets:
            image_id = str(target.get("id"))
            # Assign a random score between 0 and 5
            # We want integers for consistency with typical VLM outputs, or floats? 
            # The VLM usually outputs integers 0-5. Let's do integers.
            random_score = float(random.randint(0, 5))
            results[image_id] = RatingResult(
                score=random_score,
                justification="Random rating triggered by hack mode.",
                reported_title=None
            )
        return results

    entries: List[RatingArchiveEntry] = []
    for target in targets:
        image_path = Path(target.get("image_path", ""))
        if not image_path.exists():
            continue
        entries.append(
            RatingArchiveEntry(
                image_id=str(target.get("id")),
                title=str(target.get("title") or target.get("id") or ""),
                image_path=image_path,
            )
        )
    if not entries:
        return {}

    random.shuffle(entries)

    results: Dict[str, RatingResult] = {}
    prompt_dir = archive_dir / "vlm_ratings"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = prompt_dir / "system_prompt.txt"

    rating_batch_size = 100
    include_titles = True
    require_titles = False

    for start in range(0, len(entries), rating_batch_size):
        batch = entries[start : start + rating_batch_size]
        if not batch:
            continue
        system_prompt = build_rating_system_prompt(
            batch,
            require_titles=require_titles,
            goal_prompt=goal_prompt,
        )
        if start == 0:
            prompt_path.write_text(system_prompt)
        try:
            image_bytes_list = [entry.image_path.read_bytes() for entry in batch]
        except Exception as exc:  # pylint: disable=broad-except
            print(f"Auto-rating image read failed: {exc}")
            continue
        captions = [format_rating_entry_label(idx, entry, include_titles) for idx, entry in enumerate(batch)]
        try:
            backend = _get_rating_backend(model=model)
            response = backend.query_multiple(
                image_bytes_list,
                captions,
                prompt="",
                system_instruction=system_prompt,
            )
            response_text = response.text or ""
        except Exception as exc:  # pylint: disable=broad-except
            print(f"Auto-rating query failed: {exc}")
            continue
        parsed = parse_rating_batch_response(response_text, batch)
        for idx, rating in parsed.items():
            if 0 <= idx < len(batch):
                results[batch[idx].image_id] = rating
    # print(f"Auto-rated {len(results)} / {len(entries)} entries in batches of {rating_batch_size}.")
    return results


def _run_auto_rating_job(
    task_conn: Connection,
    rating_targets: Sequence[Dict[str, str]],
    trigger_entry_count: int,
    goal_prompt: str,
    archive_dir: Path,
    model:str,
    rand_select_prob: float = 0.0,
    rand_select_mode: str = "select-only",
) -> None:
    if not rating_targets:
        task_conn.send(
            {
                "type": "rating_failed",
                "trigger_entry_count": trigger_entry_count,
                "error": "No rating targets provided.",
            }
        )
        return
    try:
        rating_results = _perform_rating(
            rating_targets,
            goal_prompt,
            archive_dir,
            model=model,
            rand_select_prob=rand_select_prob,
            rand_select_mode=rand_select_mode,
        )
    except Exception as exc:  # pylint: disable=broad-except
        task_conn.send(
            {
                "type": "rating_failed",
                "trigger_entry_count": trigger_entry_count,
                "error": f"Rating failed: {exc}",
            }
        )
        return
    serializable_results = {
        entry_id: {
            "score": rating.score,
            "justification": rating.justification,
            "reported_title": rating.reported_title,
        }
        for entry_id, rating in rating_results.items()
    }
    task_conn.send(
        {
            "type": "rating_result",
            "results": serializable_results,
            "trigger_entry_count": trigger_entry_count,
        }
    )


def _maybe_trigger_auto_rating_after_publish(
    cfg: PicbreederConfig,
    archive_client: RemoteArchiveClient,
    task_conn: Connection,
) -> None:
    try:
        response = archive_client.prepare_auto_rating_batch()
    except Exception as exc:  # pylint: disable=broad-except
        print(f"Auto-rating check failed in worker: {exc}")
        return
    if not isinstance(response, (list, tuple)) or len(response) != 2:
        return
    rating_targets, trigger_entry_count = response
    trigger_entry_count = _safe_int(trigger_entry_count, default=0)
    if not rating_targets or trigger_entry_count <= 0:
        return
    goal_prompt = GOAL_PROMPTS[cfg.goal]
    archive_dir = Path(cfg.experiment_dir) / ARCHIVE_DIR_NAME
    _run_auto_rating_job(
        task_conn=task_conn,
        rating_targets=rating_targets,
        trigger_entry_count=trigger_entry_count,
        goal_prompt=goal_prompt,
        archive_dir=archive_dir,
        model=cfg.model,
        rand_select_prob=cfg.rand_select_prob,
        rand_select_mode=getattr(cfg, "rand_select_mode", "select-only"),
    )


def _execute_agent_task(
    task: AgentTask,
    cfg: PicbreederConfig,
    archive_client: RemoteArchiveClient,
    worker_index: int,
    task_conn: Connection,
) -> Dict[str, Any]:
    agent_dir = Path(cfg.experiment_dir) / "agents" / task.agent_id
    # Ensure directory is available (may have been compressed)
    agent_dir = _ensure_agent_directory_available(agent_dir)
    resolved_temperature = _resolve_task_temperature(cfg, task)
    agent_cfg = copy.copy(cfg)
    agent_cfg.temperature = resolved_temperature
    if task.model:
        # If the worker config uses a remote model and the task model matches (ignoring prefix),
        # preserve the remote prefix so the backend client is used.
        # We must verify against the resolved name since the parallel runner resolves it.
        resolved_task_model = _resolve_qwen_model_name(task.model) if is_local_model(task.model) else task.model
        remote_cfg_model = cfg.model[7:] if cfg.model.startswith("remote:") else cfg.model
        
        if cfg.model.startswith("remote:") and resolved_task_model == remote_cfg_model:
            agent_cfg.model = cfg.model
        else:
            agent_cfg.model = task.model
    elif agent_cfg.model == "gemini-random":
        # Fallback if model not in task (e.g. old data), though ideally task has it
        agent_cfg.model = random.choice(GEMINI_RANDOM_MODELS)
    
    # Print and save personality traits
    if task.personality_trait:
        action = "Resuming" if task.resume else "Starting"
        print(f"[{task.agent_id}] {action} with personality traits: {task.personality_trait}")

    population: Optional[neat.Population]
    population = None
    if task.resume:
        population, _ = _load_population_for_worker(
            agent_dir,
            enable_output_activations=cfg.output_activations,
            enable_input_activations=cfg.input_activations,
            enable_crossover=cfg.enable_crossover,
        )
        if population is not None:
            config = population.config
        else:
            config = build_neat_config(
                cfg.neat_config_path,
                cfg.rows,
                cfg.cols,
                cfg.output_activations,
                cfg.input_activations,
                cfg.enable_crossover,
            )
    else:
        config = build_neat_config(
            cfg.neat_config_path,
            cfg.rows,
            cfg.cols,
            cfg.output_activations,
            cfg.input_activations,
            cfg.enable_crossover,
        )

    last_archive_entry_id: Optional[str] = None
    if isinstance(task.archive_entry, dict):
        last_archive_entry_id = task.archive_entry.get("id")

    def progress_callback(
        generation: int,
        favorite_payload: Optional[Dict[str, Any]],
        archive_payload: Optional[Dict[str, Any]],
    ) -> None:
        nonlocal last_archive_entry_id
        task_conn.send(
            asdict(
                ProgressMessage(
                    agent_id=task.agent_id,
                    generation=generation,
                    favorite=favorite_payload,
                    archive=archive_payload,
                )
            )
        )
        entry_id: Optional[str] = None
        if archive_payload and isinstance(archive_payload, dict):
            entry_id = archive_payload.get("id")
        if entry_id and entry_id != last_archive_entry_id:
            last_archive_entry_id = entry_id
            _maybe_trigger_auto_rating_after_publish(cfg, archive_client, task_conn)

    final_personality_prompt = task.personality_prompt
    if task.personality_trait:
        traits_str = task.personality_trait
        if final_personality_prompt:
            final_personality_prompt = f"{final_personality_prompt}\n\n{traits_str}"
        else:
            final_personality_prompt = traits_str

    try:
        runner = AgentRunner(
            task.agent_id,
            agent_dir,
            config=agent_cfg,
            neat_config=config,
            archive_manager=archive_client,
            generations=agent_cfg.agent_generations,
            rows=agent_cfg.rows,
            cols=agent_cfg.cols,
            thumb_size=agent_cfg.thumb_size,
            scheme=agent_cfg.scheme,
            select_k=agent_cfg.select_k,
            chat_history_turns=agent_cfg.chat_history_turns,
            selection_baseline=agent_cfg.selection_baseline,
            population=population,
            progress_callback=progress_callback,
            resume_mode=task.resume,
            warm_start_active=task.warm_start_active,
            render_genome_diagrams=agent_cfg.render_genome_diagrams,
            process_index=worker_index,
            personality_prompt=final_personality_prompt,
        )
    except ValueError as e:
        if "Could not restore image for chat history" in str(e):
            print(f"[{task.agent_id}] Encountered corruption during initialization: {e}. Cleaning up agent directory and restarting from scratch.")
            if agent_dir.exists():
                shutil.rmtree(agent_dir)
            agent_dir.mkdir(parents=True, exist_ok=True)

            runner = AgentRunner(
                task.agent_id,
                agent_dir,
                config=agent_cfg,
                neat_config=config,
                archive_manager=archive_client,
                generations=agent_cfg.agent_generations,
                rows=agent_cfg.rows,
                cols=agent_cfg.cols,
                thumb_size=agent_cfg.thumb_size,
                scheme=agent_cfg.scheme,
                select_k=agent_cfg.select_k,
                chat_history_turns=agent_cfg.chat_history_turns,
                selection_baseline=agent_cfg.selection_baseline,
                population=None,
                progress_callback=progress_callback,
                resume_mode=False,
                warm_start_active=task.warm_start_active,
                render_genome_diagrams=agent_cfg.render_genome_diagrams,
                process_index=worker_index,
                personality_prompt=final_personality_prompt,
            )
        else:
            raise e

    if task.resume:
        decision = task.branching_decision
        if decision is None:
            decision = runner.select_starting_point()
            runner.initialise_population(decision)
            runner.branching_decision = decision
            task_conn.send(
                {
                    "type": "branching_decision",
                    "agent_id": task.agent_id,
                    "decision": decision,
                }
            )
        else:
            runner.branching_decision = decision
            if population is None:
                runner.initialise_population(decision)
            else:
                runner.restore_reference_context(decision)
        if task.favorite_selection is not None:
            runner.favorite_decision = task.favorite_selection
        if task.archive_entry is not None:
            try:
                favourite_entry = ArchiveEntry.from_dict(task.archive_entry)
            except Exception:  # pylint: disable=broad-except
                favourite_entry = None
            if favourite_entry is not None:
                runner.favorite_archive_entry = favourite_entry
    else:
        decision = runner.select_starting_point()
        runner.initialise_population(decision)
        runner.branching_decision = decision
        task_conn.send(asdict(BranchingDecisionMessage(agent_id=task.agent_id, decision=decision)))

    favorite, archive_entry, extinct, final_generation, quit_requested, quit_reason = _execute_runner_in_worker(
        task.agent_id,
        runner,
        task_conn,
    )

    archive_payload = archive_entry.as_dict() if isinstance(archive_entry, ArchiveEntry) else None

    return asdict(
        JobCompleteMessage(
            agent_id=task.agent_id,
            favorite=favorite,
            archive_entry=archive_payload,
            extinct=extinct,
            final_generation=final_generation,
            branching_decision=runner.branching_decision,
            quit_requested=quit_requested,
            quit_reason=quit_reason,
        )
    )


def _continual_agent_worker(
    task_conn: Connection,
    archive_conn: Connection,
    cfg_payload: Dict[str, Any],
    worker_index: int,
) -> None:
    # Handle wrapped payload for vLLM support
    if "__vlm_config__" in cfg_payload:
        vlm_config = cfg_payload["__vlm_config__"]
        base_url = vlm_config.get("base_url")
        if base_url:
            os.environ["VLLM_BASE_URL"] = base_url
        real_cfg_payload = cfg_payload.get("config", {})
    else:
        real_cfg_payload = cfg_payload

    cfg = _deserialize_config_for_worker(real_cfg_payload)
    worker_seed = None if cfg.seed is None else cfg.seed + worker_index
    print(f"[Worker {worker_index}] Applying random seed: {worker_seed}")
    apply_random_seed(worker_seed)
    if cfg.selection_baseline == "none":
        ensure_vlm_credentials()
    archive_client = RemoteArchiveClient(archive_conn)
    try:
        while True:
            task_conn.send({"type": "ready"})
            message = task_conn.recv()
            if not isinstance(message, dict):
                continue
            msg_type = message.get("type")
            if msg_type == "run_agent":
                task_payload = message.get("task") or {}
                task = _deserialize_agent_task(task_payload)
                try:
                    result_payload = _execute_agent_task(
                        task,
                        cfg,
                        archive_client,
                        worker_index,
                        task_conn,
                    )
                except Exception:  # pylint: disable=broad-except
                    task_conn.send(
                        {
                            "type": "task_failed",
                            "agent_id": task.agent_id,
                            "error": traceback.format_exc(),
                        }
                    )
                else:
                    task_conn.send(result_payload)
            elif msg_type == "run_rating":
                rating_targets = message.get("targets") or []
                trigger_entry_count = _safe_int(message.get("trigger_entry_count"), default=0)
                goal_prompt = message.get("goal_prompt") or GOAL_PROMPTS[cfg.goal]
                archive_dir_value = message.get("archive_dir")
                archive_dir = Path(archive_dir_value) if archive_dir_value else cfg.experiment_dir / ARCHIVE_DIR_NAME
                _run_auto_rating_job(
                    task_conn=task_conn,
                    rating_targets=rating_targets,
                    trigger_entry_count=trigger_entry_count,
                    goal_prompt=goal_prompt,
                    archive_dir=archive_dir,
                    model=cfg.model,
                    rand_select_prob=cfg.rand_select_prob,
                    rand_select_mode=getattr(cfg, "rand_select_mode", "select-only"),
                )
            elif msg_type == "stop":
                task_conn.send({"type": "stopped"})
                break
    finally:
        try:
            archive_conn.close()
        except Exception:
            pass
        try:
            task_conn.close()
        except Exception:
            pass

@hydra.main(version_base="1.3", config_path=None, config_name="collaborative_base")
def main(cfg: PicbreederConfig) -> None:
    original_cwd = Path(get_original_cwd())
    cfg = ensure_valid_config(cfg, original_cwd=original_cwd)
    os.makedirs(cfg.experiment_dir, exist_ok=True)
    print(f"Experiment directory: {cfg.experiment_dir}")
    run(cfg)


if __name__ == "__main__":
    main()
