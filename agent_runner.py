import base64
import copy
from dataclasses import dataclass, field
from datetime import datetime
import gzip
import json
import math
import os
from pathlib import Path
import pickle
import random
import shutil
import tempfile
from typing import Optional, Callable, Dict, Any, List, Sequence, Set, Tuple, Iterable

import graphviz
import neat
from PIL import Image, ImageDraw

from archive_manager import ARCHIVE_GRID_MARGIN, ArchiveEntry, ArchiveManager
from artifacts import build_generation_state, save_neat_genome_diagrams, save_neat_population
from chat import extract_json_object, query_with_history, reset_chat_session, restore_chat_history_from_metadata, select_parents_from_grid, summarize_genome_structure
from config import CollaborativeConfig
from constants import DEFAULT_BASELINE_SELECTION_LIMIT
from neat_components import CHECKPOINT_SUFFIX, GenerationCheckpointer, seed_initial_population, sync_population_node_indexer, sync_population_output_activations
from prompts import ARCHIVE_BRANCHING_PROMPT, ARCHIVE_NOVELTY_PROMPT, COLOR_PROMPT, GOAL_PROMPTS, MUTATION_STRENGTH_PROMPT, PARENT_SELECTION_PROMPT, DEFAULT_SYSTEM_INSTRUCTION, gen_selection_prompt
from rendering import _draw_dotted_rectangle, create_numbered_grid
from utils import _ensure_int_list

BRANCH_TOP_RATED_LIMIT = 50
BRANCH_RANDOM_LIMIT = 50

@dataclass
class ImageVariantPaths:
    color: Path
    gray: Path

    def for_color_mode(self, color_enabled: bool) -> Path:
        return self.color if color_enabled else self.gray

@dataclass
class GenerationArtifacts:
    state_path: Path
    grid_path: Path
    selection_path: Path
    image_paths: Dict[int, ImageVariantPaths] = field(default_factory=dict)
    genome_snapshots: Dict[int, neat.DefaultGenome] = field(default_factory=dict)



class AgentRunner:
    """Encapsulates the per-agent evolution workflow."""

    def __init__(
        self,
        agent_id: str,
        agent_dir: Path,
        config: CollaborativeConfig,
        neat_config: neat.Config,
        archive_manager: ArchiveManager,
        generations: int,
        rows: int,
        cols: int,
        thumb_size: int,
        scheme: str,
        select_k: Optional[int],
        chat_history_turns: int,
        selection_baseline: str = "none",
        population: Optional[neat.Population] = None,
        progress_callback: Optional[
            Callable[[int, Optional[Dict[str, Any]], Optional[Dict[str, Any]]], None]
        ] = None,
        resume_mode: bool = False,
        warm_start_active: bool = False,
        render_genome_diagrams: bool = False,
        process_index: Optional[int] = None,
        personality_prompt: Optional[str] = None,
    ) -> None:
        self.agent_id = agent_id
        self.agent_dir = agent_dir
        self.config = config
        agents_dir = agent_dir.parent
        # Save running latest images (global + per-process)
        self.latest_img_paths: List[Path] = [agents_dir / "latest_image.png"]
        if process_index is not None:
            self.latest_img_paths.append(agents_dir / f"latest_image_proc_{process_index}.png")
        self.archive_manager = archive_manager
        self.generations = generations
        self.rows = rows
        self.cols = cols
        self.thumb_size = thumb_size
        self.scheme = scheme
        self.select_k = select_k
        self.chat_history_turns = chat_history_turns
        self.selection_baseline = selection_baseline
        self.neat_config = neat_config
        self.progress_callback = progress_callback
        self.warm_start_active = bool(warm_start_active)
        self.resume_mode = resume_mode or (population is not None)
        self.personality_prompt = personality_prompt.strip() if personality_prompt else None

        # Start each agent with a fresh conversation history before any VLM calls.
        reset_chat_session()

        # Directories
        self.population_dir = agent_dir / "populations"
        self.images_dir = agent_dir / "images"
        self.query_dir = agent_dir / "queries"
        self.logs_dir = agent_dir / "logs"
        for directory in (
            agent_dir,
            self.population_dir,
            self.images_dir,
            self.query_dir,
            self.logs_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

        # Reporter & population
        if population is None:
            self.population = neat.Population(self.neat_config)
        else:
            self.population = population
        sync_population_output_activations(self.population, self.neat_config.genome_config)
        self.population.add_reporter(GenerationCheckpointer(self.population_dir))

        initial_mode = "structure_only" if self.warm_start_active else getattr(
            self.neat_config.genome_config,
            "picbreeder_mutation_mode",
            "all",
        )
        self._mutation_mode = self._normalize_mutation_mode(initial_mode)
        self._apply_mutation_mode(self._mutation_mode)
        self._mutation_strength = 0.5
        initial_strength = getattr(
            self.neat_config.genome_config,
            "picbreeder_mutation_strength",
            self._mutation_strength,
        )
        self._mutation_strength = self._normalize_mutation_strength(initial_strength)
        self._apply_mutation_strength(self._mutation_strength)
        self._color_enabled = False

        self.prompt_template = PARENT_SELECTION_PROMPT
        if (self.scheme == "color" or self.scheme == "toggle"):
            color_prompt = COLOR_PROMPT
        else:
            color_prompt = ""

        if self.chat_history_turns is None:
            archive_novelty_prompt = ARCHIVE_NOVELTY_PROMPT
        else:
            archive_novelty_prompt = ""

        if self.scheme == "toggle":
            if self.warm_start_active:
                mutation_mode_prompt = (
                    "For the warm-start phase we are focusing on grayscale structure. "
                    "Keep the `mutation_mode` field set to `structure_only` so only the brightness channel mutates."
                )
            else:
                mutation_mode_prompt = (
                    "If \"color\" is on, then at each generation, you may choose to mutate only an isolated subnetwork of the CPPN affecting color or structure, "
                    "or to mutate the entire CPPN. Indicate your choice in a \"mutation_mode\" field in your JSON response, set to either \"color_only\", \"structure_only\", or \"all\". "
                )
        else:
            mutation_mode_prompt = ""

        mutation_strength_prompt = MUTATION_STRENGTH_PROMPT
        instruction_body = DEFAULT_SYSTEM_INSTRUCTION.format(
            goal_prompt=GOAL_PROMPTS[self.config.goal],
            selection_prompt=gen_selection_prompt(self.select_k),
            n_generations=self.generations,
            color_prompt=color_prompt,
            archive_novelty_prompt=archive_novelty_prompt,
            mutation_strength_prompt=mutation_strength_prompt,
            mutation_mode_prompt=mutation_mode_prompt,
        )
        if self.personality_prompt:
            self.system_instruction = f"{self.personality_prompt}\n\n{instruction_body}"
        else:
            self.system_instruction = instruction_body
        if self.agent_id == 'agent_000':
            with open(self.agent_dir / "system_instruction.txt", "w", encoding="utf-8") as fp:
                fp.write(self.system_instruction)

        self._restored_chat_turns = 0
        should_restore_chat = (self.selection_baseline == "none") and (self.chat_history_turns is None or self.chat_history_turns != 0)
        if should_restore_chat:
            restored = restore_chat_history_from_metadata(
                self.query_dir,
                chat_history_turns=self.chat_history_turns,
                prompt_template=self.prompt_template,
            )
            if restored:
                self._restored_chat_turns = restored
                print(f"[{self.agent_id}] Restored {restored} prior chat turn(s) from saved metadata.")

        self.branching_decision: Dict[str, Any] = {}
        self.favorite_decision: Dict[str, Any] = {}
        self.favorite_archive_entry: Optional[ArchiveEntry] = None
        self._pending_publication_request: Optional[Dict[str, Any]] = None
        self._generation_records: Dict[int, GenerationArtifacts] = {}
        self._selection_history_path = self.logs_dir / "selection_history.jsonl"
        self._selection_history_path.touch(exist_ok=True)
        self.publication_history_path = self.logs_dir / "publication_history.jsonl"
        self.publication_history_path.touch(exist_ok=True)
        self._current_publication_entry_id: Optional[str] = None
        self._lineage_log_path = self.logs_dir / "lineage.jsonl"
        self._lineage_log_path.touch(exist_ok=True)
        self._pending_publication_path = self.logs_dir / "pending_publication.json"
        self._load_pending_publication()
        self._archive_seed_map: Dict[int, Dict[str, Any]] = {}
        self._genome_lineage: Dict[int, Dict[str, Any]] = {}
        self.render_genome_diagrams = render_genome_diagrams
        self._diagram_warning_emitted = False
        if self.resume_mode:
            self._load_existing_publication_state()

    def _update_latest_image(self, source_path: Path) -> None:
        for target in self.latest_img_paths:
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(source_path, target)
            except OSError:
                continue

    def _load_pending_publication(self) -> None:
        if not self._pending_publication_path.exists():
            return
        try:
            payload = json.loads(self._pending_publication_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return
        if not isinstance(payload, dict):
            return
        favorite = payload.get("favorite")
        if not isinstance(favorite, dict):
            return
        self._pending_publication_request = payload
        self.favorite_decision = favorite

    def _validate_branch_indices(
        self,
        selected_indices: Sequence[int],
        archive_entries: Sequence[Dict[str, Any]],
    ) -> Tuple[List[int], List[int]]:
        valid: List[int] = []
        invalid: List[int] = []
        for idx in selected_indices:
            if 0 <= idx < len(archive_entries):
                path_value = archive_entries[idx].get("image_path")
                if path_value and Path(path_value).exists():
                    valid.append(idx)
                    continue
            invalid.append(idx)
        return valid, invalid

    def _has_publication(self) -> bool:
        return bool(self.favorite_archive_entry or self._pending_publication_request)

    def _normalize_mutation_mode(self, mode: Optional[str]) -> str:
        candidate = str(mode).lower() if mode is not None else ""
        valid = {"all", "color_only", "structure_only"}
        return candidate if candidate in valid else "all"

    def _apply_mutation_mode(self, mode: str) -> None:
        normalized = self._normalize_mutation_mode(mode)
        setattr(self.neat_config, "picbreeder_mutation_mode", normalized)
        setattr(self.neat_config.genome_config, "picbreeder_mutation_mode", normalized)
        setattr(self.population.config, "picbreeder_mutation_mode", normalized)
        setattr(self.population.config.genome_config, "picbreeder_mutation_mode", normalized)

    def _update_mutation_mode(self, requested_mode: Optional[str]) -> str:
        if self.warm_start_active or not self._color_enabled:
            target = "structure_only"
        else:
            target = self._normalize_mutation_mode(requested_mode)
        if target != self._mutation_mode:
            self._mutation_mode = target
            self._apply_mutation_mode(self._mutation_mode)
        else:
            # Keep config in sync even if mode unchanged (useful after resume).
            self._apply_mutation_mode(self._mutation_mode)
        return self._mutation_mode

    def _normalize_mutation_strength(self, value: Optional[Any]) -> float:
        if value is None:
            return getattr(self, "_mutation_strength", 0.5)
        if isinstance(value, (int, float)):
            candidate = float(value)
        elif isinstance(value, str):
            lowered = value.strip().lower()
            label_map = {
                "small": 0.1,
                "tiny": 0.0,
                "gentle": 0.25,
                "medium": 0.5,
                "balanced": 0.5,
                "default": 0.5,
                "large": 0.9,
                "big": 0.9,
                "aggressive": 0.85,
                "huge": 1.0,
                "max": 1.0,
                "min": 0.0,
            }
            if lowered in label_map:
                candidate = label_map[lowered]
            else:
                try:
                    candidate = float(lowered)
                except ValueError:
                    return getattr(self, "_mutation_strength", 0.5)
        else:
            return getattr(self, "_mutation_strength", 0.5)
        if math.isnan(candidate) or math.isinf(candidate):
            return getattr(self, "_mutation_strength", 0.5)
        return max(0.0, min(1.0, candidate))

    def _apply_mutation_strength(self, strength: float) -> None:
        setattr(self.neat_config, "picbreeder_mutation_strength", strength)
        setattr(self.neat_config.genome_config, "picbreeder_mutation_strength", strength)
        setattr(self.population.config, "picbreeder_mutation_strength", strength)
        setattr(self.population.config.genome_config, "picbreeder_mutation_strength", strength)

    def _update_mutation_strength(self, requested_strength: Optional[Any]) -> float:
        target = self._normalize_mutation_strength(requested_strength)
        if target != self._mutation_strength:
            self._mutation_strength = target
            self._apply_mutation_strength(self._mutation_strength)
        else:
            self._apply_mutation_strength(self._mutation_strength)
        return self._mutation_strength

    def _prompt_with_settings(self, base_template: str) -> str:
        status = (
            "Current settings: "
            f"color={'ON' if self._color_enabled else 'OFF'}, "
            f"mutation_mode={self._mutation_mode}, "
            f"mutation_strength={self._mutation_strength:.2f}."
        )
        return f"{base_template}\n{status}"

    def _resolve_query_path(self, path_value: Optional[str]) -> Optional[Path]:
        if not path_value:
            return None
        candidate = Path(path_value)
        if not candidate.is_absolute():
            candidate = (self.query_dir / candidate.name).resolve()
        return candidate

    @staticmethod
    def _coerce_bool(value: Any) -> Optional[bool]:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "1", "yes", "on"}:
                return True
            if lowered in {"false", "0", "no", "off"}:
                return False
        return None

    def _build_archive_query_parts(
        self,
        entries: Sequence[Dict[str, Any]],
    ) -> Tuple[List[Tuple[bytes, str]], List[Dict[str, Any]]]:
        parts: List[Tuple[bytes, str]] = []
        metadata: List[Dict[str, Any]] = []
        for index, entry in enumerate(entries):
            image_path_value = entry.get("image_path")
            image_path = Path(image_path_value)
            image_bytes = image_path.read_bytes()
            title_value = str(entry.get("title") or "").strip()
            subset_label = str(entry.get("branching_subset_label") or "").strip()
            caption_prefix = f"{subset_label}: " if subset_label else ""
            caption = f"{caption_prefix}Image {index}"
            parts.append((image_bytes, caption))
            metadata.append(
                {
                    "index": index,
                    "caption": caption,
                    "image_path": str(image_path),
                    "image_b64": base64.b64encode(image_bytes).decode("ascii"),
                    "title": title_value,
                    "subset_label": subset_label or None,
                    "subset": entry.get("branching_subset"),
                    "archive_index": entry.get("_archive_index"),
                    "average_rating": entry.get("_average_rating"),
                    "rating_count": entry.get("_rating_count"),
                }
            )
        return parts, metadata

    def _color_output_keys(self) -> Tuple[int, ...]:
        output_keys = list(getattr(self.neat_config.genome_config, "output_keys", ()))
        if len(output_keys) >= 3:
            return tuple(int(key) for key in output_keys[:2])
        if len(output_keys) >= 2:
            return tuple(int(key) for key in output_keys[:-1])
        return tuple()

    def _zero_color_weights(self, genomes: Iterable[neat.DefaultGenome]) -> None:
        if not self.warm_start_active:
            return
        color_keys = self._color_output_keys()
        if not color_keys:
            return
        for genome in genomes:
            for (src, dst), connection in genome.connections.items():
                if int(dst) in color_keys:
                    connection.weight = 0.0

    def _enforce_structure_only_population(self) -> None:
        if not self.warm_start_active:
            return
        self._zero_color_weights(self.population.population.values())

    def _write_generation_checkpoint(self, next_generation: int) -> Path:
        self.population_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = self.population_dir / f"gen_{next_generation:03d}{CHECKPOINT_SUFFIX}"
        fd, tmp_name = tempfile.mkstemp(
            prefix=checkpoint_path.name,
            suffix=".tmp",
            dir=str(self.population_dir),
        )
        os.close(fd)
        tmp_path = Path(tmp_name)
        try:
            with gzip.open(tmp_path, "wb", compresslevel=5) as handle:
                payload = (
                    next_generation,
                    self.population.config,
                    self.population.population,
                    self.population.species,
                    random.getstate(),
                )
                pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
            tmp_path.replace(checkpoint_path)
        finally:
            tmp_path.unlink(missing_ok=True)
        return checkpoint_path

    # ------------------------------------------------------------------
    # Population initialisation and branching
    # ------------------------------------------------------------------
    def select_starting_point(self) -> Dict[str, Any]:
        self.archive_manager.create_archive_grid(self.thumb_size)
        top_entries_raw, random_entries_raw = self.archive_manager.sample_branching_entries(
            BRANCH_TOP_RATED_LIMIT,
            BRANCH_RANDOM_LIMIT,
        )
        archive_entries: List[Dict[str, Any]] = []
        for entry in top_entries_raw:
            entry_copy = copy.deepcopy(entry)
            entry_copy["branching_subset"] = "top_rated"
            entry_copy["branching_subset_label"] = "Top Rated"
            archive_entries.append(entry_copy)
        for entry in random_entries_raw:
            entry_copy = copy.deepcopy(entry)
            entry_copy["branching_subset"] = "random"
            entry_copy["branching_subset_label"] = "Random"
            archive_entries.append(entry_copy)

        archive_grid: Optional[Path]
        if archive_entries:
            archive_grid = self.archive_manager.create_archive_grid(
                self.thumb_size,
                entries=archive_entries,
            )
        else:
            archive_grid = None
        elite_name_list = self.archive_manager.get_elite_names()
        if not archive_entries:
            rationale = "Archive empty; defaulting to fresh population."
            selected_images: List[int] = []
            choice = "fresh"
            decision = {
                "choice": choice,
                "selected_images": selected_images,
                "rationale": rationale,
                "raw_response": None,
                "timestamp": datetime.now().isoformat(),
                "archive_elite_names": elite_name_list,
            }

        elif self.selection_baseline != "none":
            rationale = "Dry-run mode; random decision."
            selected_images: List[int] = []
            choice = "fresh" if random.random() < 0.5 else "branch"
            decision = {
                "choice": choice,
                "selected_images": selected_images,
                "rationale": rationale,
                "raw_response": None,
                "timestamp": datetime.now().isoformat(),
                "archive_elite_names": elite_name_list,
            }
            if decision["choice"] == "branch":
                decision["selected_images"] = [random.randrange(len(archive_entries))]
        
        elif archive_grid is None:
            decision = {
                "choice": "fresh",
                "selected_images": [],
                "rationale": "Archive grid unavailable; falling back to fresh population.",
                "raw_response": None,
                "timestamp": datetime.now().isoformat(),
                "archive_elite_names": elite_name_list,
            }

        else:
            display_order = list(range(len(archive_entries)))
            shuffled_entries = [archive_entries[idx] for idx in display_order]

            prompt_lines = [ARCHIVE_BRANCHING_PROMPT]
            top_count = len(top_entries_raw)
            random_count = len(random_entries_raw)
            if top_count:
                top_range = "0" if top_count == 1 else f"0-{top_count - 1}"
                prompt_lines.append(f"Top Rated: images {top_range}.")
            if random_count:
                start = top_count
                end = start + random_count - 1
                random_range = f"{start}" if random_count == 1 else f"{start}-{end}"
                prompt_lines.append(f"Random: images {random_range}.")
            archive_prompt = "\n".join(prompt_lines)

            image_caption_pairs, input_parts_metadata = self._build_archive_query_parts(shuffled_entries)
            for display_index, archive_index in enumerate(display_order):
                input_parts_metadata[display_index]["archive_sample_index"] = archive_index
                entry_archive_index = shuffled_entries[display_index].get("_archive_index")
                if entry_archive_index is not None:
                    input_parts_metadata[display_index]["archive_index"] = entry_archive_index
            display_to_archive_index = {idx: archive_idx for idx, archive_idx in enumerate(display_order)}

            response = query_with_history(
                image_caption_pairs,
                prompt=archive_prompt,
                system_instruction=self.system_instruction,
                chat_history_turns=self.chat_history_turns,
            )

            response_text = getattr(response, "text", "") or ""
            try:
                parsed = extract_json_object(response_text)
            except Exception:
                parsed = {}

            selected_display_indices = parsed.get("selected", [])
            selected_display_indices = [] if selected_display_indices is None else selected_display_indices
            selected_display_indices = _ensure_int_list(selected_display_indices)
            selected_images = [
                display_to_archive_index[idx]
                for idx in selected_display_indices
                if idx in display_to_archive_index
            ][:1]
            rationale = str(parsed.get("rationale", ""))
            choice = "branch" if selected_images else "fresh"
            decision = {
                "choice": choice,
                "selected_images": selected_images,
                "rationale": rationale,
                "raw_response": response_text,
                "timestamp": datetime.now().isoformat(),
                "archive_grid_path": str(archive_grid) if archive_grid else None,
                "archive_elite_names": elite_name_list,
                "archive_display_order": list(display_order),
                "input_parts": input_parts_metadata,
                "selected_display_indices": selected_display_indices,
                "archive_subset_counts": {
                    "top_rated": top_count,
                    "random": random_count,
                },
            }
            if choice == "branch":
                preview_path = self._save_archive_branch_preview(decision, archive_entries)
                decision["branch_preview_path"] = str(preview_path)

        if archive_entries:
            selected_entry_ids = [
                archive_entries[idx]["id"]
                for idx in decision.get("selected_images", [])
                if 0 <= idx < len(archive_entries)
            ]
        else:
            selected_entry_ids = []
        decision["selected_entry_ids"] = selected_entry_ids

        self._write_branching_log(decision)
        print(
            f"[{self.agent_id}] Branching decision:\nChoice: {choice}\nSelected: {selected_images}\nRationale: {rationale}"
        )
        return decision

    def initialise_population(self, decision: Dict[str, Any]) -> None:
        self._archive_seed_map.clear()
        self._genome_lineage.clear()
        # Default to grayscale view unless overridden by branching metadata.
        self._color_enabled = False
        self._update_mutation_mode(self._mutation_mode)
        if decision.get("choice") != "branch" or not decision.get("selected_images"):
            seed_initial_population(self.population, self.neat_config.genome_config)
            sync_population_node_indexer(self.population)
            self._enforce_structure_only_population()
            self._write_generation_checkpoint(int(self.population.generation))
            return

        selected_records: List[Tuple[Dict[str, Any], neat.DefaultGenome]] = []
        selected_entry_ids = decision.get("selected_entry_ids") or []
        if selected_entry_ids:
            for entry_id in selected_entry_ids:
                archive_entry = self.archive_manager.get_entry(entry_id)
                if archive_entry is None:
                    continue
                genome = self.archive_manager.load_genome(entry_id)
                if genome is None:
                    continue
                selected_records.append((archive_entry.as_dict(), genome))
        else:
            archive_entries = self.archive_manager.entries
            selected_indices = decision.get("selected_images", [])
            for idx in selected_indices:
                if not (0 <= idx < len(archive_entries)):
                    continue
                entry = archive_entries[idx]
                genome = self.archive_manager.load_genome(entry["id"])
                if genome is None:
                    continue
                selected_records.append((entry, genome))

        if not selected_records:
            seed_initial_population(self.population, self.neat_config.genome_config)
            sync_population_node_indexer(self.population)
            self._enforce_structure_only_population()
            self._write_generation_checkpoint(int(self.population.generation))
            return

        population_keys = list(self.population.population.keys())
        random.shuffle(population_keys)
        self.population.population.clear()

        branch_color_pref: Optional[bool] = None
        for key, (entry_dict, genome) in zip(population_keys, selected_records):
            clone = copy.deepcopy(genome)
            clone.key = key
            clone.fitness = None
            self.population.population[key] = clone
            self._archive_seed_map[key] = {
                "entry_id": entry_dict.get("id"),
                "agent_id": entry_dict.get("agent_id"),
                "generation": entry_dict.get("generation"),
            }
            if branch_color_pref is None and isinstance(entry_dict, dict):
                branch_color_pref = bool(entry_dict.get("color_enabled", False))

        sync_population_node_indexer(self.population)
        sync_population_output_activations(self.population, self.neat_config.genome_config)
        self.population.species.speciate(
            self.neat_config,
            self.population.population,
            self.population.generation,
        )
        self.population.population = self.population.reproduction.reproduce(
            self.population.config,
            self.population.species,
            self.population.config.pop_size,
            self.population.generation,
        )

        self._enforce_structure_only_population()
        if branch_color_pref is not None:
            self._color_enabled = bool(branch_color_pref)
            self._update_mutation_mode(self._mutation_mode)
        self._write_branching_summary(decision, len(selected_records))
        self._write_generation_checkpoint(int(self.population.generation))

    # ------------------------------------------------------------------
    # Evolution loop
    # ------------------------------------------------------------------
    def evaluate_generation(
        self,
        genomes: List[Tuple[int, neat.DefaultGenome]],
        config: neat.Config,
    ) -> None:
        generation = int(self.population.generation)
        if len(genomes) != self.rows * self.cols:
            raise ValueError(
                f"Expected {self.rows * self.cols} genomes, received {len(genomes)}."
            )

        self._zero_color_weights(genome for _, genome in genomes)

        if self.render_genome_diagrams:
            diagram_paths = save_neat_genome_diagrams(genomes, config, self.population_dir, generation)
            if diagram_paths:
                diagram_dir = diagram_paths[0].parent
                print(f"Genome diagrams saved to {diagram_dir}")
            elif graphviz is None and not self._diagram_warning_emitted:
                print("Graphviz not available; skipping genome diagram export.")
                self._diagram_warning_emitted = True

        states, caches = build_generation_state(
            genomes,
            config,
            generation,
            self.rows,
            self.cols,
            self.thumb_size,
            variant="both",
        )
        color_state = states["color"]
        gray_state = states["gray"]
        color_cache = caches["color"]
        gray_cache = caches["gray"]

        state_path = save_neat_population(color_state, self.population_dir, generation, color_cache)

        system_instruction = self.system_instruction
        prompt_template = self.prompt_template
        require_selection = True
        if generation == self.generations - 1:
            require_selection = False
            require_publish = not self._has_publication()
            if require_publish:
                prompt_template += (
                    " You have not published any favorite yet and this is the final generation; "
                    "you must include a publish object selecting exactly one image to share. "
                    "If you omit it, one of your selected parents will be published automatically."
                )
            else:
                prompt_template += (
                    " This is the final generation and your last chance to publish. "
                    "You have already published during this session, "
                    "but consider if anything in this final grid would make a better contribution to the archive. "
                    "(Note that your parent selections from this generation will have no effect.)"
                )

        if generation == 0 and self.chat_history_turns == 0:
            decision = self.branching_decision or {}
            if decision.get("choice") == "branch":
                prompt_template += (
                    " You have already branched from the archive; treat this as the first grid of your branch and select parents from it."
                )
            else:
                prompt_template += (
                    " You are starting from a fresh population; treat this as the first grid of your session and select parents from it."
                )

        # If it's the first generation of the first agent, let it know that the first generation is not a branching step.
        if generation == 0 and self.agent_id.endswith("0"):
            prompt_template += (
                " You are the first agent. This is an initial random population, and you may select one or more parents for the next step of evolution. "
            )

        grid_path: Optional[Path] = None
        view_index = 0

        if self.selection_baseline == "none":
            selection_meta_raw: Dict[str, Any]
            while True:
                prompt_with_settings = self._prompt_with_settings(prompt_template)
                variant_key = "color" if self._color_enabled else "gray"
                state_variant = color_state if variant_key == "color" else gray_state
                selection_meta_candidate = select_parents_from_grid(
                    state_variant,
                    prompt_with_settings,
                    self.query_dir,
                    self.select_k,
                    system_instruction,
                    self.chat_history_turns,
                    require_selection=require_selection,
                    allow_color_toggle=True,
                    current_color=self._color_enabled,
                    view_index=view_index,
                )
                grid_path_candidate = self._resolve_query_path(selection_meta_candidate.get("grid_path"))
                if grid_path_candidate is not None and grid_path_candidate.exists():
                    self._update_latest_image(grid_path_candidate)
                requested_color = self._coerce_bool(selection_meta_candidate.get("color"))
                if requested_color is not None and requested_color != self._color_enabled:
                    self._color_enabled = requested_color
                    self._update_mutation_mode(self._mutation_mode)
                    view_index += 1
                    continue
                selection_meta_raw = selection_meta_candidate
                state = state_variant
                grid_path = grid_path_candidate
                break
        else:
            state = color_state
            grid_image = create_numbered_grid(state)
            grid_path = self.query_dir / f"gen_{generation:03d}_view_{view_index:02d}_grid.png"
            grid_image.save(grid_path, format="PNG")
            self._update_latest_image(grid_path)
            selection_meta_raw = self._select_parents_baseline(
                generation,
                genomes,
                config,
                state,
            )
            selection_meta_raw["grid_path"] = str(grid_path)
        selection_meta = dict(selection_meta_raw)
        if grid_path is None:
            resolved_grid_str = selection_meta.get("grid_path")
            if resolved_grid_str:
                grid_path = Path(resolved_grid_str)
            else:
                grid_path = self.query_dir / f"gen_{generation:03d}_view_{view_index:02d}_grid.png"
        selection_meta["grid_path"] = str(grid_path)
        selection_meta["color"] = self._color_enabled
        resolved_mode = self._update_mutation_mode(selection_meta.get("mutation_mode"))
        selection_meta["mutation_mode"] = resolved_mode
        resolved_strength = self._update_mutation_strength(selection_meta.get("mutation_strength"))
        selection_meta["mutation_strength"] = resolved_strength
        if self.warm_start_active:
            selection_meta["mutation_mode_forced"] = True
        selection_path_value = selection_meta.get("selection_path")
        selection_path = Path(selection_path_value) if selection_path_value else Path(grid_path)
        selected_indices: Sequence[int] = selection_meta["selected"]
        publish_details: Optional[Dict[str, Any]] = None
        for idx, (_, genome) in enumerate(genomes):
            genome.fitness = 1.0 if idx in selected_indices else 0.0

        images_dir = self.images_dir / f"gen_{generation:03d}"
        images_dir.mkdir(parents=True, exist_ok=True)
        image_paths: Dict[int, ImageVariantPaths] = {}
        genome_snapshots: Dict[int, neat.DefaultGenome] = {}
        for idx, (_, genome) in enumerate(genomes):
            color_bytes = color_cache[idx]
            gray_bytes = gray_cache[idx]
            color_path = images_dir / f"idx_{idx:02d}.png"
            gray_path = images_dir / f"idx_{idx:02d}_gray.png"
            color_path.write_bytes(color_bytes)
            gray_path.write_bytes(gray_bytes)
            image_paths[idx] = ImageVariantPaths(color=color_path, gray=gray_path)
            genome_snapshots[idx] = copy.deepcopy(genome)

        record = GenerationArtifacts(
            state_path=state_path,
            grid_path=grid_path,
            selection_path=selection_path,
            image_paths=image_paths,
            genome_snapshots=genome_snapshots,
        )
        print(f"Saved selection grid to {selection_path}")
        self._generation_records[generation] = record
        self._log_generation_lineage(generation, genomes)

        publish_payload = None
        if self.selection_baseline == "none":
            publish_payload = self._parse_publish_payload(selection_meta.get("response_text", ""))
        else:
            if random.random() < 0.25:
                publish_payload = {
                    "index": next(iter(record.image_paths.keys()), 0),
                    "reason": "Dry-run random publication",
                    "title": "Dry-run favorite",
                    "raw": None,
                }

        publication_scheduled = False
        publish_index_for_highlight: Optional[int] = None
        previous_entry_id = self._current_publication_entry_id
        if publish_payload is not None:
            publish_index = publish_payload.get("index")
            if publish_index in record.image_paths:
                favorite_reason = publish_payload.get("reason", "")
                favorite_title = publish_payload.get("title", "")
                favorite = {
                    "generation": generation,
                    "index": publish_index,
                    "reason": favorite_reason,
                    "title": favorite_title,
                }
                publish_details = {
                    "index": publish_index,
                    "reason": favorite_reason,
                    "title": favorite_title,
                }
                publish_index_for_highlight = publish_index
                scheduled = self._apply_publication(
                    favorite,
                    forced=False,
                    response_text=selection_meta.get("response_text"),
                    source="vlm",
                    replaced_entry_id=previous_entry_id,
                )
                publication_scheduled = scheduled
                selection_meta["publish"] = {
                    "index": publish_index,
                    "reason": favorite_reason,
                }
                if scheduled:
                    publish_index_for_highlight = publish_index
            else:
                selection_meta["publish_error"] = {
                    "index": publish_payload.get("index"),
                    "reason": "Index out of range",
                }
                selection_meta["publish"] = None
        else:
            selection_meta["publish"] = None

        if (
            not publication_scheduled
            and generation == self.generations - 1
            and not self._has_publication()
            and record.image_paths
        ):
            fallback_index = self._choose_forced_publication_index(selected_indices, record)
            forced_rationale = (
                "Forced publication from selected parents." if fallback_index in selected_indices else "Forced publication at final generation."
            )
            favorite = {
                "generation": generation,
                "index": fallback_index,
                "rationale": forced_rationale,
            }
            publish_details = {
                "index": fallback_index,
                "rationale": forced_rationale,
                "title": favorite.get("title"),
            }
            publish_index_for_highlight = fallback_index
            scheduled = self._apply_publication(
                favorite,
                forced=True,
                response_text=selection_meta.get("response_text"),
                source="forced",
                replaced_entry_id=previous_entry_id,
            )
            if scheduled:
                selection_meta["forced_publish"] = {
                    "index": fallback_index,
                    "rationale": forced_rationale,
                }
                selection_meta["publish"] = {
                    "index": fallback_index,
                    "rationale": forced_rationale,
                    "forced": True,
                }

        selection_path = Path(selection_meta.get("selection_path") or grid_path)
        self._render_selection_with_publication(
            state,
            selected_indices,
            publish_index_for_highlight,
            selection_path,
        )
        self._update_latest_image(selection_path)
        record.selection_path = selection_path

        self._print_selection_response(
            generation,
            selected_indices,
            selection_meta.get("rationale", ""),
            selection_meta.get("mutation_mode"),
            selection_meta.get("mutation_strength"),
            self._color_enabled,
            publish_details,
        )
        if self.progress_callback is not None:
            favorite_payload = (
                copy.deepcopy(self.favorite_decision) if self.favorite_decision else None
            )
            archive_payload = (
                self.favorite_archive_entry.as_dict()
                if self.favorite_archive_entry
                else None
            )
            self.progress_callback(generation, favorite_payload, archive_payload)
        self._append_selection_history(generation, selection_meta)

    def publish_to_archive(self, favorite: Dict[str, Any]) -> Optional[ArchiveEntry]:
        generation = favorite["generation"]
        index = favorite["index"]
        record = self._generation_records.get(generation)
        if record is None:
            return None

        image_variants = record.image_paths.get(index)
        genome = record.genome_snapshots.get(index)
        if image_variants is None or genome is None:
            return None

        prefer_color = bool(favorite.get("color_enabled", False))
        if isinstance(image_variants, ImageVariantPaths):
            image_path = image_variants.for_color_mode(prefer_color)
        else:
            image_path = image_variants
        if image_path is None or not image_path.exists():
            return None
        image_bytes = image_path.read_bytes()
        favourite_log_path = self.logs_dir / "favorite_selection.json"
        if favourite_log_path.exists():
            log_path = favourite_log_path
        else:
            log_path = None

        entry = self.archive_manager.add_entry(
            image_bytes=image_bytes,
            genome=genome,
            agent_id=self.agent_id,
            generation=generation,
            image_index=index,
            rationale=favorite.get("rationale", ""),
            title=favorite.get("title", ""),
            source_experiment=self.agent_dir,
            favorite_log_path=log_path,
            selection_grid_path=record.grid_path,
            genome_key=favorite.get("genome_key"),
            parent_genome_keys=favorite.get("parent_genome_keys"),
            source_entry_ids=favorite.get("source_archive_entry_ids"),
            ancestor_genome_keys=favorite.get("ancestor_genome_keys"),
            color_enabled=prefer_color,
        )
        return entry

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------
    def _log_generation_lineage(
        self,
        generation: int,
        genomes: List[Tuple[int, neat.DefaultGenome]],
    ) -> None:
        reproduction = getattr(self.population, "reproduction", None)
        if reproduction is not None and hasattr(reproduction, "ancestors"):
            ancestors_map = getattr(reproduction, "ancestors")
        else:
            ancestors_map = {}

        memo_sources: Dict[int, List[str]] = {}
        memo_ancestors: Dict[int, Set[int]] = {}
        log_entries: List[Dict[str, Any]] = []

        for idx, (genome_id, _) in enumerate(genomes):
            parents_raw = ancestors_map.get(genome_id, tuple())
            parents = [
                int(parent)
                for parent in parents_raw
                if isinstance(parent, int) and parent not in (-1,)
            ]
            source_entries = self._resolve_source_entries(genome_id, ancestors_map, memo_sources)
            ancestor_keys = sorted(
                self._collect_ancestor_genomes(genome_id, ancestors_map, memo_ancestors)
            )

            lineage_record = self._genome_lineage.get(genome_id, {})
            if "first_seen_generation" not in lineage_record:
                lineage_record["first_seen_generation"] = generation
            lineage_record["parents"] = parents
            lineage_record["source_entries"] = list(source_entries)
            lineage_record["ancestor_keys"] = ancestor_keys
            self._genome_lineage[genome_id] = lineage_record

            log_entries.append(
                {
                    "agent_id": self.agent_id,
                    "generation": generation,
                    "image_index": idx,
                    "genome_key": genome_id,
                    "parent_genome_keys": parents,
                    "source_entry_ids": source_entries,
                    "ancestor_genome_keys": ancestor_keys,
                }
            )

        if not log_entries:
            return

        with self._lineage_log_path.open("a", encoding="utf-8") as fp:
            for entry in log_entries:
                fp.write(json.dumps(entry))
                fp.write("\n")

    def _resolve_source_entries(
        self,
        genome_key: int,
        ancestors_map: Dict[int, Tuple[int, int]],
        memo: Dict[int, List[str]],
    ) -> List[str]:
        if genome_key in memo:
            return memo[genome_key]

        seed_info = self._archive_seed_map.get(genome_key)
        if seed_info is not None:
            entry_id = seed_info.get("entry_id")
            memo[genome_key] = [entry_id] if entry_id else []
            return memo[genome_key]

        lineage_record = self._genome_lineage.get(genome_key)
        if lineage_record and lineage_record.get("source_entries") is not None:
            memo[genome_key] = list(lineage_record["source_entries"])
            return memo[genome_key]

        parents_raw = ancestors_map.get(genome_key, tuple())
        parents = [
            int(parent)
            for parent in parents_raw
            if isinstance(parent, int) and parent not in (-1,)
        ]
        if not parents:
            memo[genome_key] = []
            return memo[genome_key]

        collected: Set[str] = set()
        for parent in parents:
            collected.update(self._resolve_source_entries(parent, ancestors_map, memo))

        memo[genome_key] = sorted(value for value in collected if value)
        return memo[genome_key]

    def _collect_ancestor_genomes(
        self,
        genome_key: int,
        ancestors_map: Dict[int, Tuple[int, int]],
        memo: Dict[int, Set[int]],
    ) -> Set[int]:
        if genome_key in memo:
            return memo[genome_key]

        parents_raw = ancestors_map.get(genome_key, tuple())
        parents = [
            int(parent)
            for parent in parents_raw
            if isinstance(parent, int) and parent not in (-1,)
        ]

        ancestors: Set[int] = set()
        for parent in parents:
            ancestors.add(parent)
            ancestors.update(self._collect_ancestor_genomes(parent, ancestors_map, memo))

        memo[genome_key] = ancestors
        return ancestors

    def _write_branching_log(self, payload: Dict[str, Any]) -> None:
        path = self.logs_dir / "branching_selection.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _write_branching_summary(self, decision: Dict[str, Any], seeded: int) -> None:
        summary = {
            "decision": decision,
            "seeded_parent_count": seeded,
            "timestamp": datetime.now().isoformat(),
        }
        path = self.logs_dir / "branching_summary.json"
        path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    def _write_favorite_log(self, payload: Dict[str, Any]) -> None:
        path = self.logs_dir / "favorite_selection.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _append_selection_history(self, generation: int, metadata: Dict[str, Any]) -> None:
        entry = dict(metadata)
        entry["generation"] = generation
        entry["timestamp"] = datetime.now().isoformat()
        with self._selection_history_path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(entry))
            fp.write("\n")

    def _parse_publish_payload(self, response_text: str) -> Optional[Dict[str, Any]]:
        if not response_text:
            return None
        try:
            parsed = extract_json_object(response_text)
        except Exception:
            return None
        payload = parsed.get("publish")
        if payload in (None, "", "none", "null"):
            return None
        index_value: Any
        rationale: str = ""
        if isinstance(payload, dict):
            index_value = payload.get("index")
            rationale = (
                payload.get("reason")
                or payload.get("rationale")
                or ""
            )
            title = payload.get("title")
        else:
            index_value = payload
            rationale = (
                parsed.get("publish_reason")
                or parsed.get("publish_rationale")
                or ""
            )
            title = parsed.get("publish_title") or ""
        try:
            index_int = int(index_value)
        except (TypeError, ValueError):
            return None
        return {
            "index": index_int,
            "reason": str(rationale).strip(),
            "raw": payload,
            "title": str(title).strip() if 'title' in locals() else "",
        }

    def _append_publication_history(self, payload: Dict[str, Any]) -> None:
        with self.publication_history_path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(payload))
            fp.write("\n")

    def _load_existing_publication_state(self) -> None:
        if not self.publication_history_path.exists():
            return
        last_payload: Optional[Dict[str, Any]] = None
        with self.publication_history_path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    last_payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
        if not last_payload:
            return
        self.favorite_decision = dict(last_payload)
        entry_id = last_payload.get("archive_entry_id")
        if entry_id:
            entry = self.archive_manager.get_entry(entry_id)
            if entry is not None:
                self.favorite_archive_entry = entry
                self._current_publication_entry_id = entry.entry_id

    def _apply_publication(
        self,
        favorite: Dict[str, Any],
        *,
        forced: bool,
        response_text: Optional[str],
        source: str,
        replaced_entry_id: Optional[str],
    ) -> bool:
        generation_raw = favorite.get("generation")
        index_raw = favorite.get("index")
        title = favorite.get("title", "")
        if generation_raw is None or index_raw is None:
            return None
        try:
            generation_int = int(generation_raw)
            index_int = int(index_raw)
        except (TypeError, ValueError):
            return None
        record = self._generation_records.get(generation_int)
        if record is None or index_int not in record.image_paths:
            return None

        payload = dict(favorite)
        payload["generation"] = generation_int
        payload["index"] = index_int
        payload.setdefault("rationale", "")
        payload["timestamp"] = datetime.now().isoformat()
        payload["forced"] = forced
        payload["source"] = source
        payload["color_enabled"] = bool(self._color_enabled)
        if response_text is not None:
            payload["response_text"] = response_text
        if replaced_entry_id is not None:
            payload["replaced_entry_id"] = replaced_entry_id

        genome_snapshot = record.genome_snapshots.get(index_int) if record else None
        reproduction = getattr(self.population, "reproduction", None)
        if genome_snapshot is not None and hasattr(genome_snapshot, "key"):
            genome_key = int(getattr(genome_snapshot, "key"))
        else:
            genome_key = None
        if reproduction is not None and hasattr(reproduction, "ancestors"):
            ancestors_map = getattr(reproduction, "ancestors")
        else:
            ancestors_map = {}
        if genome_key is not None:
            parent_keys_raw = ancestors_map.get(genome_key, tuple())
            parent_keys = [
                int(parent)
                for parent in parent_keys_raw
                if isinstance(parent, int) and parent not in (-1,)
            ]
            source_entry_ids = self._resolve_source_entries(genome_key, ancestors_map, {})
            ancestor_key_set = self._collect_ancestor_genomes(genome_key, ancestors_map, {})
            ancestor_keys = sorted(ancestor_key_set)
        else:
            parent_keys = []
            source_entry_ids = []
            ancestor_keys = []

        payload["genome_key"] = genome_key
        payload["parent_genome_keys"] = parent_keys
        payload["source_archive_entry_ids"] = source_entry_ids
        payload["ancestor_genome_keys"] = ancestor_keys

        self._write_favorite_log(payload)

        pending_request = {
            "favorite": payload,
            "forced": forced,
            "response_text": response_text,
            "source": source,
            "replaced_entry_id": replaced_entry_id,
        }
        self._pending_publication_request = pending_request
        try:
            self._pending_publication_path.write_text(
                json.dumps(pending_request, indent=2),
                encoding="utf-8",
            )
        except OSError:
            pass
        self.favorite_decision = payload
        print(
            f"[{self.agent_id}] Queued publication for generation={generation_int}, index={index_int}, title='{title}', forced={payload.get('forced')}."
        )
        return True

    def commit_pending_publication(self) -> Optional[ArchiveEntry]:
        if not self._pending_publication_request:
            return self.favorite_archive_entry
        pending = self._pending_publication_request
        favorite = pending.get("favorite", {})
        if not isinstance(favorite, dict):
            return self.favorite_archive_entry
        replaced_entry_id = pending.get("replaced_entry_id")
        entry = self.publish_to_archive(favorite)
        if entry is None:
            print(f"[{self.agent_id}] Pending publication failed; entry could not be created.")
            return self.favorite_archive_entry
        favorite["archive_entry_id"] = entry.entry_id
        genome_key = favorite.get("genome_key")
        if genome_key is not None:
            self._archive_seed_map[genome_key] = {
                "entry_id": entry.entry_id,
                "agent_id": self.agent_id,
                "generation": favorite.get("generation"),
            }
        self._append_publication_history(favorite)
        if replaced_entry_id:
            self.archive_manager.remove_entry(replaced_entry_id)
        self.favorite_archive_entry = entry
        self.favorite_decision = favorite
        self._current_publication_entry_id = entry.entry_id
        self._pending_publication_request = None
        try:
            self._pending_publication_path.unlink(missing_ok=True)
        except OSError:
            pass
        print(
            f"[{self.agent_id}] Publication committed: generation={favorite.get('generation')}, index={favorite.get('index')}, title='{favorite.get('title', '')}', forced={favorite.get('forced')}."
        )
        return entry

    def _choose_forced_publication_index(
        self,
        selected_indices: Sequence[int],
        record: GenerationArtifacts,
    ) -> int:
        for idx in selected_indices:
            if idx in record.image_paths:
                return idx
        if record.image_paths:
            return min(record.image_paths)
        return 0

    def _render_selection_with_publication(
        self,
        state: Dict[str, Any],
        selected_indices: Sequence[int],
        publish_index: Optional[int],
        selection_path: Path,
    ) -> None:
        image = create_numbered_grid(state, selected=selected_indices)
        if publish_index is not None:
            entry = next(
                (item for item in state.get("images", []) if int(item.get("index", -1)) == publish_index),
                None,
            )
            if entry is not None:
                margin = 12
                thumb = int(state.get("thumbSize", 0))
                col = int(entry.get("col", 0))
                row = int(entry.get("row", 0))
                x0 = margin + col * (thumb + margin)
                y0 = margin + row * (thumb + margin)
                x1 = x0 + thumb
                y1 = y0 + thumb
                draw = ImageDraw.Draw(image)
                _draw_dotted_rectangle(
                    draw,
                    (x0, y0, x1, y1),
                    color=(0, 255, 0),
                    width=4,
                    dash_length=10,
                    gap_length=6,
                )
        image.save(selection_path, format="PNG")

    def _save_archive_branch_preview(
        self,
        decision: Dict[str, Any],
        archive_entries: List[Dict[str, Any]],
    ) -> Optional[Path]:
        if decision.get("choice") != "branch":
            raise ValueError("Branch preview can only be generated for branching decisions.")
        selected = _ensure_int_list(decision.get("selected_images", []))
        if not selected:
            raise ValueError("No selected images in branching decision.")
        grid_path_str = decision.get("archive_grid_path")
        if not grid_path_str:
            raise ValueError("Missing archive grid path in branching decision.")
        grid_path = Path(grid_path_str)
        if not grid_path.exists():
            raise FileNotFoundError(f"Archive grid image not found: {grid_path}")

        total_entries = len(archive_entries)
        if total_entries == 0:
            raise ValueError("No archive entries available for preview generation.")

        metadata_path = grid_path.with_suffix(".json")
        index_to_bbox: Dict[int, Tuple[int, int, int, int]] = {}
        if metadata_path.exists():
            try:
                payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                payload = {}
            for entry in payload.get("entries", []):
                idx_value = entry.get("index")
                bbox_value = entry.get("bbox")
                if not isinstance(idx_value, int):
                    continue
                if (
                    isinstance(bbox_value, (list, tuple))
                    and len(bbox_value) == 4
                    and all(isinstance(coord, (int, float)) for coord in bbox_value)
                ):
                    index_to_bbox[idx_value] = tuple(int(round(coord)) for coord in bbox_value)

        fallback_layout_needed = not index_to_bbox

        if fallback_layout_needed:
            valid_indices: List[int] = []
            for entry_index, entry in enumerate(archive_entries):
                path_value = entry.get("image_path")
                if not path_value:
                    raise ValueError(f"Missing image path for archive entry at index {entry_index}.")
                if Path(path_value).exists():
                    valid_indices.append(entry_index)

            if not valid_indices:
                raise ValueError("No valid archive entries with existing images found.")

            columns = max(1, math.ceil(math.sqrt(len(valid_indices))))
            index_to_position = {entry_idx: pos for pos, entry_idx in enumerate(valid_indices)}
            tile_size = self.thumb_size
            margin = ARCHIVE_GRID_MARGIN

        output_path = self.query_dir / "archive_branch.png"

        with Image.open(grid_path) as img:
            preview = img.convert("RGB")
            draw = ImageDraw.Draw(preview)
            for idx in selected:
                if not (0 <= idx < total_entries):
                    raise ValueError(f"Selected index {idx} outside archive sample range.")
                if index_to_bbox:
                    bbox = index_to_bbox.get(idx)
                    if bbox is None:
                        raise ValueError(f"Bounding box not found for selected index {idx}.")
                    x0, y0, x1, y1 = bbox
                else:
                    position = index_to_position.get(idx)
                    if position is None:
                        raise ValueError(f"Selected index {idx} not found among valid archive entries.")
                    col = position % columns
                    row = position // columns
                    x0 = margin + col * (tile_size + margin)
                    y0 = margin + row * (tile_size + margin)
                    x1 = x0 + tile_size
                    y1 = y0 + tile_size
                _draw_dotted_rectangle(
                    draw,
                    (x0, y0, x1, y1),
                    color=(255, 0, 0),
                    width=5,
                )
            preview.save(output_path, format="PNG")
            self._update_latest_image(output_path)

        return output_path

    def _print_selection_response(
        self,
        generation: int,
        selected_indices: Sequence[int],
        rationale: str,
        mutation_mode: Optional[str],
        mutation_strength: Optional[float],
        color_enabled: bool,
        publish_details: Optional[Dict[str, Any]],
    ) -> None:
        publish_index = publish_details.get("index") if publish_details else None
        publish_title = publish_details.get("title") if publish_details else None
        publish_rationale = publish_details.get("reason") if publish_details else None
        log_str = (
            f"[{self.agent_id}] Gen {generation} selection:\n"
            f"Selected: {list(selected_indices)}\n"
            f"Rationale: {rationale}\n"
            f"Color: {'ON' if color_enabled else 'OFF'}"
        )
        if mutation_mode is not None:
            log_str += f"\nMutation Mode: '{mutation_mode}'"
        if mutation_strength is not None:
            log_str += f"\nMutation Strength: {mutation_strength:.2f}"
        if publish_index is not None:
            log_str += (
                f"\n\nPublish Index: {publish_index}"
                f"\nPublish Title: '{publish_title}'"
                f"\nPublish Rationale: {publish_rationale}\n"
            )
        print(log_str)

    def _select_parents_baseline(
        self,
        generation: int,
        genomes: List[Tuple[int, neat.DefaultGenome]],
        config: neat.Config,
        state: Dict[str, Any],
    ) -> Dict[str, Any]:
        genome_config = config.genome_config
        metrics: List[Dict[str, Any]] = []
        for idx, (genome_id, genome) in enumerate(genomes):
            summary = summarize_genome_structure(genome, genome_config)
            summary.update(
                {
                    "index": idx,
                    "genome_id": genome_id,
                }
            )
            metrics.append(summary)

        max_selection = self.select_k if self.select_k is not None else DEFAULT_BASELINE_SELECTION_LIMIT
        selection_count = max(1, min(max_selection, len(genomes)))
        selected_indices: List[int]
        rationale: str

        if self.selection_baseline == "random":
            selected_indices = random.sample(range(len(genomes)), k=selection_count)
            rationale = f"Random baseline selected {selected_indices}."
        else:
            if self.selection_baseline == "max-depth":
                scoring_key = "depth"
                baseline_label = "maximum depth"
            else:
                scoring_key = "hidden_node_count"
                baseline_label = "maximum hidden-node count"

            max_score = max(metric[scoring_key] for metric in metrics) if metrics else 0
            top_candidates = [metric for metric in metrics if metric[scoring_key] == max_score]
            random.shuffle(top_candidates)
            selected_entries = top_candidates[:selection_count]
            if len(selected_entries) < selection_count:
                remaining = [metric for metric in metrics if metric[scoring_key] < max_score]
                random.shuffle(remaining)
                selected_entries.extend(remaining[: selection_count - len(selected_entries)])
            selected_indices = [entry["index"] for entry in selected_entries]
            rationale = (
                f"Baseline favoring {baseline_label} selected {selected_indices} "
                f"(score={max_score})."
            )

        metadata: Dict[str, Any] = {
            "selected": selected_indices,
            "rationale": rationale,
            "baseline": self.selection_baseline,
            "metrics": metrics,
            "select_k": self.select_k,
            "selection_count": selection_count,
        }

        return self._write_baseline_artifacts(generation, state, metadata)

    def _write_baseline_artifacts(
        self,
        generation: int,
        state: Dict[str, Any],
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        self.query_dir.mkdir(parents=True, exist_ok=True)
        selected = metadata.get("selected", [])
        grid_image = create_numbered_grid(state)
        selection_image = create_numbered_grid(state, selected=selected)

        suffix = "_view_00"
        grid_path = self.query_dir / f"gen_{generation:03d}{suffix}_grid.png"
        selection_path = self.query_dir / f"gen_{generation:03d}{suffix}_selection.png"
        grid_image.save(grid_path, format="PNG")
        selection_image.save(selection_path, format="PNG")

        payload = dict(metadata)
        payload.update(
            {
                "generation": generation,
                "grid_path": str(grid_path),
                "selection_path": str(selection_path),
                "select_k": self.select_k,
                "chat_history_turns": self.chat_history_turns,
                "response_text": None,
                "view_index": 0,
                "color": bool(state.get("variant") == "color"),
                "color_toggle_only": False,
            }
        )

        metadata_dir = self.query_dir / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        meta_path = metadata_dir / f"gen_{generation:03d}{suffix}_selection.json"
        meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        payload["metadata_path"] = str(meta_path)

        return payload


