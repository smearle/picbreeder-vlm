#!/usr/bin/env python3
"""Multi-agent collaborative Picbreeder workflow with shared archive.

This script orchestrates a chain of visual-language-model (VLM) driven agents
that collaboratively evolve CPPN image populations. The first agent runs a
20-generation session with full conversational history. After completing its
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

import gzip
import json
import multiprocessing
import os
import pickle
import random
import traceback
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from multiprocessing.connection import Connection
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence, Set, Tuple
from itertools import count

import hydra
from hydra.utils import get_original_cwd

from agent_runner import AgentRunner
from config import CollaborativeConfig, _deserialize_config_for_worker, _serialize_config_for_worker, ensure_valid_config
import personalities

import neat
from neat.population import CompleteExtinctionException
from PIL import Image, ImageDraw

from chat import (
    ensure_gemini_key,
)
from archive_manager import (
    ArchiveEntry,
    ArchiveManager,
    atomic_write_json,
    interprocess_lock,
)
from artifacts import build_generation_state, save_neat_genome_diagrams, save_neat_population
from neat_components import (
    CHECKPOINT_SUFFIX,
    InteractiveStagnation,
    PicbreederGenome,
    apply_picbreeder_config_defaults,
    sync_population_output_activations,
)
from constants import (
    ARCHIVE_DIR_NAME,
    AGENT_DIR_PREFIX,
    PERSONALITY_BATCH_SIZE,
    PERSONALITY_TOTAL,
    RATING_BATCH_SIZE,
)
from picbreeder_reproduction import PicbreederReproduction
from prompts import GOAL_PROMPTS
from utils import apply_random_seed

from rate_archive_with_vlm import (
    ArchiveEntry as RatingArchiveEntry,
    RatingResult,
    build_rating_system_prompt,
    format_rating_entry_label,
    parse_rating_batch_response,
)
from im_query import query_images_with_captions


def find_latest_checkpoint(population_dir: Path) -> Optional[Path]:
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



@dataclass
class AgentTask:
    agent_id: str
    agent_index: int
    agent_dir: Path
    resume: bool
    warm_start_active: bool
    personality_prompt: Optional[str]
    branching_decision: Optional[Dict[str, Any]] = None
    favorite_selection: Optional[Dict[str, Any]] = None
    archive_entry: Optional[Dict[str, Any]] = None

    def to_message(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "agent_index": self.agent_index,
            "agent_dir": str(self.agent_dir),
            "resume": self.resume,
            "warm_start_active": self.warm_start_active,
            "personality_prompt": self.personality_prompt,
            "branching_decision": self.branching_decision,
            "favorite_selection": self.favorite_selection,
            "archive_entry": self.archive_entry,
        }


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
    )
    config.pop_size = rows * cols
    return config


class CollaborativeMultiAgentOrchestrator:
    """Coordinates sequential agent runs and archive management."""

    def __init__(
        self,
        config: CollaborativeConfig,
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
        self.selection_baseline = selection_baseline
        self.seed = seed
        self.render_genome_diagrams = render_genome_diagrams
        self.process_index = process_index

        self.archive_manager = ArchiveManager(self.experiment_dir / ARCHIVE_DIR_NAME,
                                              goal_prompt=GOAL_PROMPTS[self.config.goal])
        self.agents_dir = self.experiment_dir / "agents"
        self.agents_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_path = self.experiment_dir / "agents_metadata.json"
        # self._metadata_lock_path = self.metadata_path.with_suffix(".lock")
        self._metadata = self._load_metadata()
        self._ensure_run_config()

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
    #     with interprocess_lock(self._metadata_lock_path):
        if self.metadata_path.exists():
            metadata = self._read_metadata_file()
        else:
            metadata = self._default_metadata()
            self._write_metadata_file(metadata)
        metadata, changed = self._ensure_metadata_defaults(metadata)
        if changed:
            self._write_metadata_file(metadata)
        return metadata

    def _default_metadata(self) -> Dict[str, Any]:
        return {
            "created_at": datetime.now().isoformat(),
            "next_agent_number": 0,
            "agents": [],
            "seed": self.seed,
            "run_config": None,
        }

    def _read_metadata_file(self) -> Dict[str, Any]:
        with self.metadata_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _write_metadata_file(self, metadata: Dict[str, Any]) -> None:
        atomic_write_json(self.metadata_path, metadata)

    def _ensure_metadata_defaults(self, metadata: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
        changed = False
        metadata.setdefault("agents", [])
        metadata.setdefault("next_agent_number", 0)
        metadata.setdefault("run_config", None)
        if self.seed is not None and metadata.get("seed") != self.seed:
            metadata["seed"] = self.seed
            changed = True
        elif self.seed is None and metadata.get("seed") is not None:
            self.seed = metadata.get("seed")
        for record in metadata["agents"]:
            if "status" not in record:
                record["status"] = "complete" if record.get("completed_at") else "in_progress"
                changed = True
            if "last_generation" not in record:
                record["last_generation"] = None
                changed = True
        return metadata, changed

    def _reload_metadata(self) -> Dict[str, Any]:
        if self.metadata_path.exists():
            metadata = self._read_metadata_file()
        else:
            metadata = self._default_metadata()
        metadata, _ = self._ensure_metadata_defaults(metadata)
        self._metadata = metadata
        return metadata

    def _mutate_metadata(self, mutator: Callable[[Dict[str, Any]], Any]) -> Any:
        if self.metadata_path.exists():
            metadata = self._read_metadata_file()
        else:
            metadata = self._default_metadata()
        metadata, _ = self._ensure_metadata_defaults(metadata)
        result = mutator(metadata)
        self._write_metadata_file(metadata)
        self._metadata = metadata
        return result

    def _ensure_run_config(self) -> None:
        self._reload_metadata()
        personality_path_str = str(self.config.personality_path) if self.config.personality_path else None
        run_config = {
            "rows": self.rows,
            "cols": self.cols,
            "thumb_size": self.thumb_size,
            "scheme": self.scheme,
            "select_k": self.select_k,
            "agent_generations": self.agent_generations,
            "enable_output_activations": self.enable_output_activations,
            "warm_start_structure": self.warm_start_structure,
            "selection_baseline": self.selection_baseline,
            "generate_personalities": self.config.generate_personalities,
            "personality_path": personality_path_str,
        }
        existing = self._metadata.get("run_config")
        if existing is None:
            self._mutate_metadata(lambda data: data.__setitem__("run_config", run_config))
            return
        missing = []
        if "warm_start_structure" not in existing:
            missing.append(("warm_start_structure", 0))
        if "selection_baseline" not in existing:
            missing.append(("selection_baseline", "none"))
        if "generate_personalities" not in existing:
            missing.append(("generate_personalities", self.config.generate_personalities))
        if "personality_path" not in existing:
            missing.append(("personality_path", personality_path_str))
        if missing:
            def mutator(data: Dict[str, Any]) -> None:
                details = data.setdefault("run_config", {})
                for key, value in missing:
                    details.setdefault(key, value)

            self._mutate_metadata(mutator)
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
        index = self._parse_agent_index(agent_id)
        if index is None:
            return None
        persona_index = index % PERSONALITY_TOTAL
        if persona_index >= len(self._personality_prompts):
            persona_index = index % len(self._personality_prompts)
        return self._personality_prompts[persona_index]

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

    def _ensure_agent_number_progress(self, agent_id: str) -> None:
        index = self._parse_agent_index(agent_id)
        if index is None:
            raise ValueError(f"Invalid agent identifier '{agent_id}'.")

        def mutator(metadata: Dict[str, Any]) -> None:
            metadata["next_agent_number"] = max(metadata.get("next_agent_number", 0), index + 1)

        self._mutate_metadata(mutator)

    def _is_warm_start_agent(self, agent_id: str) -> bool:
        if self.warm_start_structure <= 0:
            return False
        index = self._parse_agent_index(agent_id)
        return index is not None and index < self.warm_start_structure

    def _find_agent_record(self, agent_id: str) -> Optional[Dict[str, Any]]:
        for record in self._metadata.get("agents", []):
            if record.get("agent_id") == agent_id:
                return record
        return None

    def _register_agent(self, agent_id: str, agent_dir: Path) -> Dict[str, Any]:
        def mutator(metadata: Dict[str, Any]) -> Dict[str, Any]:
            agents = metadata.setdefault("agents", [])
            for record in agents:
                if record.get("agent_id") == agent_id:
                    return record
            record = {
                "agent_id": agent_id,
                "agent_dir": str(agent_dir),
                "branching_decision": None,
                "favorite_selection": None,
                "archive_entry": None,
                "status": "in_progress",
                "created_at": datetime.now().isoformat(),
                "last_generation": 0,
            }
            agents.append(record)
            return record

        return self._mutate_metadata(mutator)

    def _update_agent_record(self, agent_id: str, **updates: Any) -> None:
        def mutator(metadata: Dict[str, Any]) -> None:
            for record in metadata.get("agents", []):
                if record.get("agent_id") == agent_id:
                    record.update(updates)
                    break

        self._mutate_metadata(mutator)

    def _find_agent_to_resume(
        self,
        resume_agent_id: Optional[str],
        allowed_agent_ids: Optional[Set[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        self._reload_metadata()
        agents = self._metadata.get("agents", [])
        finished = {"complete", "extinct"}
        if resume_agent_id:
            if allowed_agent_ids is not None and resume_agent_id not in allowed_agent_ids:
                return None
            record = next((entry for entry in agents if entry.get("agent_id") == resume_agent_id), None)
            if record is None:
                raise ValueError(f"Agent '{resume_agent_id}' not found in experiment metadata.")
            if record.get("status") in finished:
                raise ValueError(f"Agent '{resume_agent_id}' has already completed.")
            return record
        for record in reversed(agents):
            if allowed_agent_ids is not None and record.get("agent_id") not in allowed_agent_ids:
                continue
            if record.get("status") not in finished:
                return record
        return None

    def _hydrate_agent_record_from_disk(self, record: Dict[str, Any]) -> None:
        agent_dir = Path(record["agent_dir"])
        logs_dir = agent_dir / "logs"
        updates: Dict[str, Any] = {}
        branch_path = logs_dir / "branching_selection.json"
        if record.get("branching_decision") is None and branch_path.exists():
            try:
                branching = json.loads(branch_path.read_text(encoding="utf-8"))
                record["branching_decision"] = branching
                updates["branching_decision"] = branching
            except json.JSONDecodeError:
                pass
        favourite_path = logs_dir / "favorite_selection.json"
        if record.get("favorite_selection") is None and favourite_path.exists():
            try:
                favorite = json.loads(favourite_path.read_text(encoding="utf-8"))
                record["favorite_selection"] = favorite
                updates["favorite_selection"] = favorite
            except json.JSONDecodeError:
                pass
        publication_path = logs_dir / "publication_history.jsonl"
        if record.get("archive_entry") is None and publication_path.exists():
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
                    record["archive_entry"] = entry.as_dict()
                    if record.get("favorite_selection") is None:
                        record["favorite_selection"] = last_payload
                        updates.setdefault("favorite_selection", last_payload)
                    updates["archive_entry"] = record["archive_entry"]
        if updates:
            self._update_agent_record(record["agent_id"], **updates)

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
            self.config.input_activations,
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
        return AgentRunner(
            agent_id,
            agent_dir,
            config=self.config,
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
            personality_prompt=personality_prompt,
        )

    def _on_generation_progress(
        self,
        agent_id: str,
        generation: int,
        favorite_payload: Optional[Dict[str, Any]],
        archive_payload: Optional[Dict[str, Any]],
    ) -> None:
        updates: Dict[str, Any] = {"last_generation": generation}
        if favorite_payload is not None:
            updates["favorite_selection"] = favorite_payload
        if archive_payload is not None:
            updates["archive_entry"] = archive_payload
        self._update_agent_record(agent_id, **updates)

    def _execute_runner(
        self,
        agent_id: str,
        runner: AgentRunner,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[ArchiveEntry], bool, int]:
        remaining = max(0, runner.generations - runner.population.generation)
        extinct = False
        if remaining > 0:
            try:
                runner.population.run(runner.evaluate_generation, remaining)
            except CompleteExtinctionException:
                extinct = True
        archive_entry = runner.commit_pending_publication()
        favorite = runner.favorite_decision if runner.favorite_decision else None
        final_generation = runner.population.generation
        return favorite, archive_entry, extinct, final_generation

    def run_agents(
        self,
        total_agents: int,
        resume: bool,
        resume_agent_id: Optional[str],
        num_workers: int = 1,
    ) -> None:
        if num_workers <= 1:
            self._run_agents_serial(total_agents, resume, resume_agent_id)
            return
        self._run_agents_parallel(total_agents, resume, resume_agent_id, num_workers)

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
            if record is not None and record.get("status") in {"complete", "extinct"}:
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
        worker_states: List[WorkerState] = []
        conn_to_worker: Dict[Connection, WorkerState] = {}
        errors: List[str] = []

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
                        active_workers -= 1
                        self._cleanup_worker(state, conn_to_worker)
                        continue

                    if conn is state.archive_conn:
                        self._handle_archive_rpc(state, message)
                        continue

                    msg_type = message.get("type")
                    if msg_type == "ready":
                        self._handle_worker_ready(state, pending_tasks)
                    elif msg_type == "branching_decision":
                        decision = message.get("decision")
                        agent_id = message.get("agent_id")
                        if agent_id and decision is not None:
                            if state.current_task and state.current_task.agent_id == agent_id:
                                state.current_task.branching_decision = decision
                            self._update_agent_record(agent_id, branching_decision=decision)
                    elif msg_type == "progress":
                        agent_id = message.get("agent_id")
                        if agent_id:
                            generation = _safe_int(message.get("generation"), default=0)
                            self._on_generation_progress(
                                agent_id,
                                generation,
                                message.get("favorite"),
                                message.get("archive"),
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
                        self._handle_job_complete(state, message)
                        state.current_task = None
                    elif msg_type == "task_failed":
                        agent_id = message.get("agent_id")
                        error_text = message.get("error") or "Unknown failure"
                        errors.append(
                            f"Worker {state.index} failed on {agent_id or 'unknown agent'}: {error_text}"
                        )
                        state.current_task = None
                    elif msg_type == "worker_error":
                        error_text = message.get("error") or "Unknown worker error"
                        errors.append(f"Worker {state.index} error: {error_text}")
                        state.errored = True
                    elif msg_type == "stopped":
                        active_workers -= 1
                        self._cleanup_worker(state, conn_to_worker)
                    else:
                        continue
        finally:
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

    def _handle_worker_ready(self, state: WorkerState, pending_tasks: Deque[AgentTask]) -> None:
        if state.current_task is not None:
            return
        try:
            rating_targets, trigger_entry_count = self.archive_manager.prepare_auto_rating_batch(
                RATING_BATCH_SIZE
            )
        except Exception as exc:  # pylint: disable=broad-except
            print(f"Auto-rating scheduling failed: {exc}")
            rating_targets = []
            trigger_entry_count = 0
        if rating_targets:
            payload = {
                "type": "run_rating",
                "targets": rating_targets,
                "trigger_entry_count": trigger_entry_count,
                "goal_prompt": self.archive_manager.goal_prompt,
                "archive_dir": str(self.archive_manager.archive_dir),
            }
            state.task_conn.send(payload)
            state.stopping = False
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
            results = _perform_rating(targets, goal_prompt, archive_dir)
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

    def _handle_job_complete(self, state: WorkerState, payload: Dict[str, Any]) -> None:
        agent_id = payload.get("agent_id")
        if not agent_id:
            return
        task = state.current_task
        agent_dir = task.agent_dir if task else (self.agents_dir / agent_id)
        favorite_payload = payload.get("favorite")
        archive_entry_data = payload.get("archive_entry")
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
        extinct = bool(payload.get("extinct"))
        final_generation = _safe_int(payload.get("final_generation"), default=0)
        branching_decision = None
        if payload.get("branching_decision") is not None:
            branching_decision = payload.get("branching_decision")
        elif task is not None:
            branching_decision = task.branching_decision
        else:
            record = self._find_agent_record(agent_id)
            if record is not None:
                branching_decision = record.get("branching_decision")
        self._finalize_agent(
            agent_id,
            agent_dir,
            branching_decision,
            favorite_payload,
            archive_entry,
            extinct=extinct,
            final_generation=final_generation,
        )

    def _build_new_agent_task(self, agent_id: str, agent_index: int) -> AgentTask:
        self._ensure_agent_number_progress(agent_id)
        agent_dir = self.agents_dir / agent_id
        # agent_dir.mkdir(parents=True, exist_ok=True)
        record = self._register_agent(agent_id, agent_dir)
        if record.get("status") != "in_progress":
            self._update_agent_record(agent_id, status="in_progress", last_generation=0)
        warm_start_active = self._is_warm_start_agent(agent_id)
        personality_prompt = self._personality_prompt_for_agent(agent_id)
        return AgentTask(
            agent_id=agent_id,
            agent_index=agent_index,
            agent_dir=agent_dir,
            resume=False,
            warm_start_active=warm_start_active,
            personality_prompt=personality_prompt,
        )

    def _build_resume_agent_task(self, record: Dict[str, Any]) -> AgentTask:
        self._hydrate_agent_record_from_disk(record)
        agent_id = record["agent_id"]
        agent_dir = Path(record["agent_dir"])
        agent_index = self._parse_agent_index(agent_id) or 0
        warm_start_active = self._is_warm_start_agent(agent_id)
        personality_prompt = self._personality_prompt_for_agent(agent_id)
        last_generation = _safe_int(record.get("last_generation"), default=0)
        self._update_agent_record(
            agent_id,
            status="in_progress",
            resumed_at=datetime.now().isoformat(),
            last_generation=last_generation,
        )
        return AgentTask(
            agent_id=agent_id,
            agent_index=agent_index,
            agent_dir=agent_dir,
            resume=True,
            warm_start_active=warm_start_active,
            personality_prompt=personality_prompt,
            branching_decision=record.get("branching_decision"),
            favorite_selection=record.get("favorite_selection"),
            archive_entry=record.get("archive_entry"),
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
            if record and record.get("status") not in {"complete", "extinct"}:
                pending.append(self._build_resume_agent_task(record))
                scheduled.add(resume_agent_id)
        for index in target_indices:
            agent_id = self._agent_id_from_index(index)
            if agent_id in scheduled:
                continue
            record = self._find_agent_record(agent_id)
            if record is not None and record.get("status") in {"complete", "extinct"}:
                continue
            if record is not None:
                task = self._build_resume_agent_task(record)
            else:
                task = self._build_new_agent_task(agent_id, index)
            pending.append(task)
            scheduled.add(agent_id)
        return pending

    def _run_new_agent(self, agent_id: Optional[str] = None) -> None:
        if agent_id is None:
            agent_id = self._allocate_agent_id()
        else:
            self._ensure_agent_number_progress(agent_id)
        agent_dir = self.agents_dir / agent_id
        agent_dir.mkdir(parents=True, exist_ok=True)
        self._register_agent(agent_id, agent_dir)
        config = self._build_config()
        runner = self._build_runner(agent_id, agent_dir, config, None, resume=False)
        decision = runner.select_starting_point()
        runner.initialise_population(decision)
        runner.branching_decision = decision
        self._update_agent_record(agent_id, branching_decision=decision, status="in_progress", last_generation=0)
        favorite, archive_entry, extinct, final_generation = self._execute_runner(agent_id, runner)
        self._finalize_agent(
            agent_id,
            agent_dir,
            decision,
            favorite,
            archive_entry,
            extinct=extinct,
            final_generation=final_generation,
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
        agent_id = record["agent_id"]
        agent_dir = Path(record["agent_dir"])
        population, checkpoint_path = self._load_population_for_agent(agent_dir)
        if population is not None and checkpoint_path is not None:
            print(f"[{agent_id}] Resuming from checkpoint {checkpoint_path.name}")
            config = population.config
        else:
            print(f"[{agent_id}] Resume requested without checkpoint; restarting agent workflow.")
            config = self._build_config()
        runner = self._build_runner(agent_id, agent_dir, config, population, resume=True)
        decision = record.get("branching_decision")
        if decision is None:
            decision = runner.select_starting_point()
            runner.initialise_population(decision)
            runner.branching_decision = decision
            self._update_agent_record(agent_id, branching_decision=decision)
        else:
            runner.branching_decision = decision
            if population is None:
                runner.initialise_population(decision)
        favorite_selection = record.get("favorite_selection")
        if favorite_selection is not None:
            runner.favorite_decision = favorite_selection
        archive_entry_dict = record.get("archive_entry")
        if archive_entry_dict is not None:
            try:
                entry = ArchiveEntry.from_dict(archive_entry_dict)
            except Exception:
                entry = None
            if entry is not None:
                runner.favorite_archive_entry = entry
                runner._current_publication_entry_id = entry.entry_id
        self._update_agent_record(
            agent_id,
            status="in_progress",
            resumed_at=datetime.now().isoformat(),
            last_generation=runner.population.generation,
        )
        favorite, archive_entry, extinct, final_generation = self._execute_runner(agent_id, runner)
        self._finalize_agent(
            agent_id,
            agent_dir,
            decision,
            favorite,
            archive_entry,
            extinct=extinct,
            final_generation=final_generation,
        )
        self._maybe_run_auto_rating_serial()
        return True

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
    ) -> None:
        updates: Dict[str, Any] = {
            "extinct": extinct,
            "status": "extinct" if extinct else "complete",
            "completed_at": datetime.now().isoformat(),
            "last_generation": final_generation,
        }
        if branching_decision is not None:
            updates["branching_decision"] = branching_decision
        if favourite is not None:
            updates["favorite_selection"] = favourite
        if archive_entry is not None:
            updates["archive_entry"] = archive_entry.as_dict()
        self._update_agent_record(agent_id, **updates)



def _build_orchestrator(
    cfg: CollaborativeConfig,
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


def _generate_personalities_for_run(cfg: CollaborativeConfig) -> None:
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


def run(cfg: CollaborativeConfig) -> None:
    if cfg.generate_personalities and not cfg.resume:
        _generate_personalities_for_run(cfg)
    apply_random_seed(cfg.seed)
    if cfg.selection_baseline == "none":
        ensure_gemini_key()
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

    @property
    def entries(self) -> Any:
        return self._call("get_entries")


def _deserialize_agent_task(payload: Dict[str, Any]) -> AgentTask:
    return AgentTask(
        agent_id=str(payload["agent_id"]),
        agent_index=int(payload.get("agent_index", 0)),
        agent_dir=Path(payload["agent_dir"]),
        resume=bool(payload.get("resume")),
        warm_start_active=bool(payload.get("warm_start_active")),
        personality_prompt=payload.get("personality_prompt"),
        branching_decision=payload.get("branching_decision"),
        favorite_selection=payload.get("favorite_selection"),
        archive_entry=payload.get("archive_entry"),
    )


def _load_population_for_worker(
    agent_dir: Path,
    enable_output_activations: bool,
    enable_input_activations: bool,
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
    )
    sync_population_output_activations(population, population.config.genome_config)
    _rehydrate_reproduction_state(population)
    return population, checkpoint_path


def _execute_runner_in_worker(
    agent_id: str,
    runner: AgentRunner,
) -> Tuple[Optional[Dict[str, Any]], Optional[ArchiveEntry], bool, int]:
    remaining = max(0, runner.generations - runner.population.generation)
    extinct = False
    if remaining > 0:
        try:
            runner.population.run(runner.evaluate_generation, remaining)
        except CompleteExtinctionException:
            extinct = True
    archive_entry = runner.commit_pending_publication()
    favorite = runner.favorite_decision if runner.favorite_decision else None
    final_generation = runner.population.generation
    return favorite, archive_entry, extinct, final_generation


def _perform_rating(
    targets: Sequence[Dict[str, str]],
    goal_prompt: str,
    archive_dir: Path,
) -> Dict[str, RatingResult]:
    if not targets:
        return {}
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
            response = query_images_with_captions(
                image_bytes_list,
                captions,
                prompt=None,
                system_instruction=system_prompt,
            )
            response_text = getattr(response, "text", "") or ""
        except Exception as exc:  # pylint: disable=broad-except
            print(f"Auto-rating query failed: {exc}")
            continue
        parsed = parse_rating_batch_response(response_text, batch)
        for idx, rating in parsed.items():
            if 0 <= idx < len(batch):
                results[batch[idx].image_id] = rating
    return results


def _execute_agent_task(
    task: AgentTask,
    cfg: CollaborativeConfig,
    archive_client: RemoteArchiveClient,
    worker_index: int,
    task_conn: Connection,
) -> Dict[str, Any]:
    agent_dir = task.agent_dir
    agent_dir.mkdir(parents=True, exist_ok=True)

    population: Optional[neat.Population]
    population = None
    if task.resume:
        population, _ = _load_population_for_worker(
            agent_dir,
            enable_output_activations=cfg.output_activations,
            enable_input_activations=cfg.input_activations,
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
            )
    else:
        config = build_neat_config(
            cfg.neat_config_path,
            cfg.rows,
            cfg.cols,
            cfg.output_activations,
            cfg.input_activations,
        )

    def progress_callback(
        generation: int,
        favorite_payload: Optional[Dict[str, Any]],
        archive_payload: Optional[Dict[str, Any]],
    ) -> None:
        task_conn.send(
            {
                "type": "progress",
                "agent_id": task.agent_id,
                "generation": generation,
                "favorite": favorite_payload,
                "archive": archive_payload,
            }
        )

    runner = AgentRunner(
        task.agent_id,
        agent_dir,
        config=cfg,
        neat_config=config,
        archive_manager=archive_client,
        generations=cfg.agent_generations,
        rows=cfg.rows,
        cols=cfg.cols,
        thumb_size=cfg.thumb_size,
        scheme=cfg.scheme,
        select_k=cfg.select_k,
        chat_history_turns=cfg.chat_history_turns,
        selection_baseline=cfg.selection_baseline,
        population=population,
        progress_callback=progress_callback,
        resume_mode=task.resume,
        warm_start_active=task.warm_start_active,
        render_genome_diagrams=cfg.render_genome_diagrams,
        process_index=worker_index,
        personality_prompt=task.personality_prompt,
    )

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
        if task.favorite_selection is not None:
            runner.favorite_decision = task.favorite_selection
        if task.archive_entry is not None:
            try:
                favourite_entry = ArchiveEntry.from_dict(task.archive_entry)
            except Exception:  # pylint: disable=broad-except
                favourite_entry = None
            if favourite_entry is not None:
                runner.favorite_archive_entry = favourite_entry
                runner._current_publication_entry_id = favourite_entry.entry_id
    else:
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

    favorite, archive_entry, extinct, final_generation = _execute_runner_in_worker(
        task.agent_id,
        runner,
    )

    archive_payload = archive_entry.as_dict() if isinstance(archive_entry, ArchiveEntry) else None

    return {
        "type": "job_complete",
        "agent_id": task.agent_id,
        "favorite": favorite,
        "archive_entry": archive_payload,
        "extinct": extinct,
        "final_generation": final_generation,
        "branching_decision": runner.branching_decision,
    }


def _continual_agent_worker(
    task_conn: Connection,
    archive_conn: Connection,
    cfg_payload: Dict[str, Any],
    worker_index: int,
) -> None:
    cfg = _deserialize_config_for_worker(cfg_payload)
    worker_seed = None if cfg.seed is None else cfg.seed + worker_index
    apply_random_seed(worker_seed)
    if cfg.selection_baseline == "none":
        ensure_gemini_key()
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
                if not rating_targets:
                    task_conn.send(
                        {
                            "type": "rating_failed",
                            "trigger_entry_count": trigger_entry_count,
                            "error": "No rating targets provided.",
                        }
                    )
                    continue
                try:
                    rating_results = _perform_rating(rating_targets, goal_prompt, archive_dir)
                except Exception as exc:  # pylint: disable=broad-except
                    task_conn.send(
                        {
                            "type": "rating_failed",
                            "trigger_entry_count": trigger_entry_count,
                            "error": f"Rating failed: {exc}",
                        }
                    )
                else:
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
def main(cfg: CollaborativeConfig) -> None:
    original_cwd = Path(get_original_cwd())
    cfg = ensure_valid_config(cfg, original_cwd=original_cwd)
    os.makedirs(cfg.experiment_dir, exist_ok=True)
    print(f"Experiment directory: {cfg.experiment_dir}")
    run(cfg)


if __name__ == "__main__":
    main()
