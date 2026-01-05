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

import numpy as np

import PIL
import graphviz
import neat
from PIL import Image, ImageChops, ImageDraw, ImageFont, ImageStat

from archive_manager import ARCHIVE_GRID_MARGIN, ArchiveEntry, ArchiveManager
from artifacts import render_genome_images, save_neat_genome_diagrams
from chat import GeminiPromptBlockedError, extract_json_object, query_with_history, reset_chat_session, restore_chat_history_from_metadata, select_parents_from_grid, summarize_genome_structure
from config import PicbreederConfig
from constants import DEFAULT_BASELINE_SELECTION_LIMIT, REPO_ROOT

# Module-level cache so CLIP noun embeddings are computed once per process
_CLIP_NOUN_CACHE: Dict[str, Any] = {}
from neat_components import GenerationCheckpointer, LATEST_POPULATION_FILENAME, seed_initial_population, sync_population_node_indexer, sync_population_output_activations
from prompts import ARCHIVE_BRANCHING_PROMPT, ARCHIVE_NOVELTY_PROMPT, COLOR_PROMPT, GOAL_PROMPTS, MUTATION_STRENGTH_PROMPT, PARENT_SELECTION_PROMPT, DEFAULT_SYSTEM_INSTRUCTION, FIXED_SESSION_SYSTEM_INSTRUCTION, gen_selection_prompt
from rendering import _draw_dotted_rectangle, create_numbered_grid
from utils import _ensure_int_list, relative_suffix_after_dir


BRANCH_TOP_RATED_LIMIT = 20
BRANCH_RANDOM_LIMIT = 20
BRANCH_BEST_NEW_LIMIT = 20
BRANCH_BEST_NEW_WINDOW_MULTIPLIER = 10
BRANCH_MOST_BRANCHED_LIMIT = 20
BRANCH_NEWEST_LIMIT = 20
DUPLICATE_PIXEL_EPSILON = 1e-3


class AgentQuitRequested(RuntimeError):
    """Raised when the VLM explicitly stops the current evolutionary run."""


class AgentRestartRequested(RuntimeError):
    """Raised when the VLM asks to abandon the current trajectory and restart."""

    def __init__(self, payload: Dict[str, Any]):
        super().__init__("Agent requested a restart")
        self.payload = payload


@dataclass
class ImageVariantPaths:
    color: Path
    gray: Path

    def for_color_mode(self, color_enabled: bool) -> Path:
        return self.color if color_enabled else self.gray


@dataclass
class GenerationArtifacts:
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
        config: PicbreederConfig,
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
        self.latest_img_paths: List[Path] = [agents_dir.parent / "latest_image.png"]
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
        self.request_rationale = bool(getattr(self.config, "request_rationale", True))
        self.log_raw_responses = bool(getattr(self.config, "log_raw_responses", False))
        self.fixed_session_lengths = bool(getattr(self.config, "fixed_session_lengths", True))
        self.rand_select_prob = max(
            0.0, min(1.0, float(getattr(self.config, "rand_select_prob", 0.0) or 0.0))
        )

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

        if self.chat_history_turns == -1 and self.request_rationale:
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
        selection_json_suffix = ', "rationale": "brief explanation"' if self.request_rationale else ""
        publish_reason_suffix = ', "reason": "Brief publication note."' if self.request_rationale else ""

        # Choose the appropriate system instruction template based on fixed session mode
        if self.fixed_session_lengths:
            instruction_template = FIXED_SESSION_SYSTEM_INSTRUCTION
            final_generation = self.generations - 1  # 0-indexed, so gen 19 for 20 generations
            instruction_body = instruction_template.format(
                goal_prompt=GOAL_PROMPTS[self.config.goal],
                selection_prompt=gen_selection_prompt(self.select_k, self.config.enable_crossover),
                n_generations=self.generations,
                final_generation=final_generation,
                color_prompt=color_prompt,
                archive_novelty_prompt=archive_novelty_prompt,
                mutation_strength_prompt=mutation_strength_prompt,
                mutation_mode_prompt=mutation_mode_prompt,
                selection_json_suffix=selection_json_suffix,
                publish_reason_suffix=publish_reason_suffix,
            )
        else:
            instruction_body = DEFAULT_SYSTEM_INSTRUCTION.format(
                goal_prompt=GOAL_PROMPTS[self.config.goal],
                selection_prompt=gen_selection_prompt(self.select_k, self.config.enable_crossover),
                n_generations=self.generations,
                color_prompt=color_prompt,
                archive_novelty_prompt=archive_novelty_prompt,
                mutation_strength_prompt=mutation_strength_prompt,
                mutation_mode_prompt=mutation_mode_prompt,
                selection_json_suffix=selection_json_suffix,
                publish_reason_suffix=publish_reason_suffix,
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
                self.config.model,
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
        self._generation_records: Dict[int, GenerationArtifacts] = {}
        self._selection_history_path = self.logs_dir / "selection_history.jsonl"
        self._selection_history_path.touch(exist_ok=True)
        self.publication_history_path = self.logs_dir / "publication_history.jsonl"
        self.publication_history_path.touch(exist_ok=True)
        self._lineage_log_path = self.logs_dir / "lineage.jsonl"
        self._lineage_log_path.touch(exist_ok=True)
        self._branching_snapshot_path = self.logs_dir / "branching_snapshot.json"
        self._restart_log_path = self.logs_dir / "restart_history.jsonl"
        self._restart_log_path.touch(exist_ok=True)
        self._initial_branch_archive_entries: List[Dict[str, Any]] = []
        self._initial_branch_subset_ranges: List[Dict[str, Any]] = []
        self._initial_branch_subset_counts: Dict[str, int] = {}
        self._initial_branch_elite_names: List[str] = []
        self._initial_branch_display_order: List[int] = []
        self._pending_restart_request: Optional[Dict[str, Any]] = None
        self._restart_history: List[Dict[str, Any]] = []
        self._restart_counter = 0
        self._pending_prompt_notes: List[str] = []
        self._last_publish_rejection: Optional[Dict[str, Any]] = None
        self._duplicate_publish_epsilon = max(
            0.0,
            float(
                getattr(self.config, "duplicate_publish_epsilon", DUPLICATE_PIXEL_EPSILON)
                or DUPLICATE_PIXEL_EPSILON
            ),
        )
        self._archive_seed_map: Dict[int, Dict[str, Any]] = {}
        self._genome_lineage: Dict[int, Dict[str, Any]] = {}
        self.render_genome_diagrams = render_genome_diagrams
        self._diagram_warning_emitted = False
        self._quit_requested = False
        self._quit_reason: Optional[str] = None
        self._quit_generation: Optional[int] = None
        self._load_branching_snapshot()
        if self.resume_mode:
            self._load_existing_publication_state()
            self._restore_selection_settings()

        # Lazily populated state for CLIP noun baseline (selection_baseline == "clip-nouns")
        self._clip_model = None
        self._clip_preprocess = None
        self._clip_tokenizer = None
        self._clip_device = None
        self._clip_text_embeddings: Optional[np.ndarray] = None
        self._clip_noun_labels: List[str] = []
        self._clip_torch = None

    def _update_latest_image(self, source_path: Path) -> None:
        for target in self.latest_img_paths:
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(source_path, target)
            except OSError:
                continue

    def _save_generation_images(
        self,
        generation: int,
        color_images: List[PIL.Image.Image],
        gray_images: List[PIL.Image.Image],
    ) -> Dict[int, ImageVariantPaths]:
        images_dir = self.images_dir / self._current_run_label() / f"gen_{generation:03d}"
        images_dir.mkdir(parents=True, exist_ok=True)
        image_paths: Dict[int, ImageVariantPaths] = {}
        if len(color_images) != len(gray_images):
            raise ValueError(
                f"Mismatched color/gray image list lengths: {len(color_images)} vs {len(gray_images)}"
            )
        for idx, (color_im, gray_im) in enumerate(zip(color_images, gray_images)):
            color_path = images_dir / f"idx_{idx:02d}.png"
            gray_path = images_dir / f"idx_{idx:02d}_gray.png"
            color_im.save(color_path, format="PNG")
            gray_im.save(gray_path, format="PNG")
            image_paths[idx] = ImageVariantPaths(color=color_path, gray=gray_path)
        return image_paths

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
        return self.favorite_archive_entry is not None

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

    def _queue_prompt_note(self, note: str) -> None:
        cleaned = str(note or "").strip()
        if not cleaned:
            return
        self._pending_prompt_notes.append(cleaned)

    def _apply_pending_prompt_notes(self, base_template: str) -> str:
        if not self._pending_prompt_notes:
            return base_template
        notes = " ".join(self._pending_prompt_notes)
        self._pending_prompt_notes.clear()
        return f"{base_template} {notes}"

    def _resolve_archive_image_path(self, path_value: Any) -> Optional[Path]:
        if not path_value:
            return None
        archive_root = Path(getattr(self.archive_manager, "archive_dir", self.agent_dir))
        raw_path = Path(path_value)
        candidates: List[Path] = [raw_path]
        if not raw_path.is_absolute():
            candidates.append((archive_root / raw_path).resolve())
        relative_suffix = relative_suffix_after_dir(raw_path, "archive")
        if relative_suffix is not None:
            candidates.append((archive_root / relative_suffix).resolve())
        seen: Set[Path] = set()
        for candidate in candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            if candidate.exists():
                return candidate
        return None

    def _normalized_pixel_distance(self, img_a: Image.Image, img_b: Image.Image) -> float:
        base = img_a.convert("RGB")
        other = img_b.convert("RGB")
        if base.size != other.size:
            other = other.resize(base.size, Image.Resampling.LANCZOS)
        diff = ImageChops.difference(base, other)
        stats = ImageStat.Stat(diff)
        if not stats.mean:
            return 0.0
        return sum(stats.mean) / len(stats.mean) / 255.0

    def _find_duplicate_archive_entry(
        self,
        candidate_path: Optional[Path],
    ) -> Tuple[Optional[ArchiveEntry], Optional[float]]:
        if candidate_path is None or not candidate_path.exists():
            return None, None
        try:
            with Image.open(candidate_path) as candidate_img:
                candidate_rgb = candidate_img.convert("RGB")
        except OSError:
            return None, None

        for raw_entry in self.archive_manager.entries:
            try:
                entry = ArchiveEntry.from_dict(raw_entry)
            except Exception:
                continue
            archive_path = self._resolve_archive_image_path(entry.image_path)
            if archive_path is None or not archive_path.exists():
                continue
            try:
                with Image.open(archive_path) as archived_img:
                    distance = self._normalized_pixel_distance(candidate_rgb, archived_img)
            except OSError:
                continue
            if distance <= self._duplicate_publish_epsilon:
                return entry, distance
        return None, None

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
        checkpoint_path = self.population_dir / LATEST_POPULATION_FILENAME
        fd, tmp_name = tempfile.mkstemp(
            prefix="latest_population_",
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

    def _save_selected_genomes(
        self,
        generation: int,
        genomes: List[Tuple[int, neat.DefaultGenome]],
        selected_indices: Sequence[int],
    ) -> None:
        if not selected_indices:
            return

        selected_payload: List[Dict[str, Any]] = []
        for idx in selected_indices:
            if not (0 <= idx < len(genomes)):
                continue
            genome_key, genome = genomes[idx]
            lineage_record = self._genome_lineage.get(genome_key, {})
            selected_payload.append(
                {
                    "grid_index": idx,
                    "genome_key": genome_key,
                    "genome": copy.deepcopy(genome),
                    "parents": lineage_record.get("parents"),
                    "source_entry_ids": lineage_record.get("source_entries"),
                    "ancestor_genome_keys": lineage_record.get("ancestor_keys"),
                }
            )

        if not selected_payload:
            return

        snapshot = {
            "generation": generation,
            "selected": selected_payload,
        }
        target_path = self.population_dir / f"selected_gen_{generation:03d}.pkl.gz"
        fd, tmp_name = tempfile.mkstemp(
            prefix=target_path.name,
            suffix=".tmp",
            dir=str(self.population_dir),
        )
        os.close(fd)
        tmp_path = Path(tmp_name)
        try:
            with gzip.open(tmp_path, "wb", compresslevel=5) as handle:
                pickle.dump(snapshot, handle, protocol=pickle.HIGHEST_PROTOCOL)
            tmp_path.replace(target_path)
        finally:
            tmp_path.unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # Population initialisation and branching
    # ------------------------------------------------------------------
    def select_starting_point(self) -> Dict[str, Any]:
        (
            top_entries_raw,
            best_new_entries_raw,
            most_branched_entries_raw,
            newest_entries_raw,
            random_entries_raw,
        ) = self.archive_manager.sample_branching_entries(
            BRANCH_TOP_RATED_LIMIT,
            BRANCH_RANDOM_LIMIT,
            best_new_count=BRANCH_BEST_NEW_LIMIT,
            best_new_window_multiplier=BRANCH_BEST_NEW_WINDOW_MULTIPLIER,
            most_branched_count=BRANCH_MOST_BRANCHED_LIMIT,
            newest_count=BRANCH_NEWEST_LIMIT,
        )

        subset_specs: List[Tuple[str, str, List[Dict[str, Any]]]] = [
            ("top_rated", "Top Rated", top_entries_raw),
            ("best_new", "Best New Images", best_new_entries_raw),
            ("most_branched", "Most Branched", most_branched_entries_raw),
            ("newest", "Newest", newest_entries_raw),
            ("random", "Random", random_entries_raw),
        ]

        archive_entries: List[Dict[str, Any]] = []
        subset_ranges: List[Dict[str, Any]] = []
        subset_counts: Dict[str, int] = {}
        running_index = 0
        for subset_key, subset_label, subset_entries in subset_specs:
            count = len(subset_entries)
            subset_counts[subset_key] = count
            if not count:
                continue
            start = running_index
            for entry in subset_entries:
                entry_copy = copy.deepcopy(entry)
                entry_copy["branching_subset"] = subset_key
                entry_copy["branching_subset_label"] = subset_label
                archive_entries.append(entry_copy)
            end = running_index + count - 1
            subset_ranges.append(
                {
                    "key": subset_key,
                    "label": subset_label,
                    "start": start,
                    "end": end,
                }
            )
            running_index += count

        elite_name_list = self.archive_manager.get_elite_names()
        decision = self._decide_branching_from_entries(
            archive_entries,
            subset_ranges,
            subset_counts,
            elite_name_list,
        )
        self._store_branching_snapshot(
            archive_entries,
            subset_ranges,
            subset_counts,
            elite_name_list,
            decision.get("archive_display_order"),
        )
        choice = decision.get("choice")
        selected_images = decision.get("selected_images", [])
        rationale = decision.get("rationale", "")
        self._write_branching_log(decision)
        print(
            f"[{self.agent_id}] Branching decision:\nChoice: {choice}\nSelected: {selected_images}\nRationale: {rationale}"
        )
        return decision

    def _decide_branching_from_entries(
        self,
        archive_entries: List[Dict[str, Any]],
        subset_ranges: List[Dict[str, Any]],
        subset_counts: Dict[str, int],
        elite_name_list: Sequence[str],
        *,
        prompt_note: Optional[str] = None,
        display_order: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        timestamp = datetime.now().isoformat()
        if not archive_entries:
            return {
                "choice": "fresh",
                "selected_images": [],
                "rationale": "Archive empty; defaulting to fresh population.",
                "raw_response": None,
                "timestamp": timestamp,
                "archive_elite_names": list(elite_name_list),
                "archive_subset_counts": dict(subset_counts),
                "selected_entry_ids": [],
            }

        if self.selection_baseline == "clip-nouns":
            return self._decide_branching_clip_nouns(
                archive_entries,
                subset_ranges,
                subset_counts,
                elite_name_list,
                timestamp=timestamp,
            )

        if self.selection_baseline != "none":
            rationale = "Dry-run mode; random decision."
            if prompt_note:
                rationale = f"{prompt_note} ({rationale})"
            choice = "fresh" if random.random() < 0.5 else "branch"
            selected_images: List[int] = []
            if choice == "branch":
                selected_images = [random.randrange(len(archive_entries))]
            decision = {
                "choice": choice,
                "selected_images": selected_images,
                "rationale": rationale,
                "raw_response": None,
                "timestamp": timestamp,
                "archive_elite_names": list(elite_name_list),
                "archive_subset_counts": dict(subset_counts),
            }
            decision["selected_entry_ids"] = [
                archive_entries[idx]["id"]
                for idx in selected_images
                if 0 <= idx < len(archive_entries)
            ]
            return decision

        working_display_order = list(display_order) if display_order is not None else list(range(len(archive_entries)))
        shuffled_entries = [archive_entries[idx] for idx in working_display_order]

        prompt_lines = [ARCHIVE_BRANCHING_PROMPT]
        if prompt_note:
            prompt_lines.append(prompt_note)
        for subset in subset_ranges:
            start = subset["start"]
            end = subset["end"]
            range_str = f"{start}" if start == end else f"{start}-{end}"
            prompt_lines.append(f"{subset['label']}: images {range_str}.")
        archive_prompt = "\n".join(prompt_lines)

        image_caption_pairs, input_parts_metadata = self._build_archive_query_parts(shuffled_entries)
        display_to_archive_index: Dict[int, int] = {}
        for display_index, archive_index in enumerate(working_display_order):
            display_to_archive_index[display_index] = archive_index
            input_parts_metadata[display_index]["archive_sample_index"] = archive_index
            entry_archive_index = shuffled_entries[display_index].get("_archive_index")
            if entry_archive_index is not None:
                input_parts_metadata[display_index]["archive_index"] = entry_archive_index

        response = query_with_history(
            self.config.model,
            image_caption_pairs,
            self.config.thinking_budget,
            prompt=archive_prompt,
            system_instruction=self.system_instruction,
            chat_history_turns=self.chat_history_turns,
            temperature=self.config.temperature,
        )

        response_text = getattr(response, "text", "") or ""
        try:
            parsed = extract_json_object(response_text)
        except Exception:
            parsed = {}
        if isinstance(parsed, ValueError) or not isinstance(parsed, dict):
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
            "timestamp": timestamp,
            "archive_elite_names": list(elite_name_list),
            "archive_display_order": list(working_display_order),
            "input_parts": input_parts_metadata,
            "selected_display_indices": selected_display_indices,
            "archive_subset_counts": dict(subset_counts),
        }
        decision["selected_entry_ids"] = [
            archive_entries[idx]["id"]
            for idx in selected_images
            if 0 <= idx < len(archive_entries)
        ]
        if choice == "branch":
            preview_path = self._save_archive_branch_preview(decision, archive_entries)
            decision["branch_preview_path"] = str(preview_path)
        return decision

    def _decide_branching_clip_nouns(
        self,
        archive_entries: List[Dict[str, Any]],
        subset_ranges: List[Dict[str, Any]],
        subset_counts: Dict[str, int],
        elite_name_list: Sequence[str],
        *,
        timestamp: Optional[str] = None,
    ) -> Dict[str, Any]:
        decision_timestamp = timestamp or datetime.now().isoformat()
        if not archive_entries:
            return {
                "choice": "fresh",
                "selected_images": [],
                "rationale": "Archive empty; defaulting to fresh population.",
                "raw_response": None,
                "timestamp": decision_timestamp,
                "archive_elite_names": list(elite_name_list),
                "archive_subset_counts": dict(subset_counts),
                "selected_entry_ids": [],
            }

        branch_probability = 0.9
        roll = random.random()
        if roll >= branch_probability:
            rationale = (
                f"CLIP noun baseline prefers branching (p={branch_probability:.2f}); "
                f"roll={roll:.3f} triggered a fresh start."
            )
            return {
                "choice": "fresh",
                "selected_images": [],
                "rationale": rationale,
                "raw_response": None,
                "timestamp": decision_timestamp,
                "archive_elite_names": list(elite_name_list),
                "archive_subset_counts": dict(subset_counts),
                "selected_entry_ids": [],
                "branch_probability": branch_probability,
                "branch_roll": roll,
            }

        valid_images: List[Image.Image] = []
        valid_indices: List[int] = []
        for idx, entry in enumerate(archive_entries):
            image_path_value = entry.get("image_path")
            if not image_path_value:
                continue
            image_path = Path(image_path_value)
            try:
                with Image.open(image_path) as img:
                    valid_images.append(img.convert("RGB"))
                    valid_indices.append(idx)
            except OSError:
                continue

        if not valid_images:
            rationale = (
                "CLIP noun baseline could not load any archive images; defaulting to fresh population."
            )
            return {
                "choice": "fresh",
                "selected_images": [],
                "rationale": rationale,
                "raw_response": None,
                "timestamp": decision_timestamp,
                "archive_elite_names": list(elite_name_list),
                "archive_subset_counts": dict(subset_counts),
                "selected_entry_ids": [],
                "branch_probability": branch_probability,
                "branch_roll": roll,
            }

        scores = self._clip_max_similarity_scores(valid_images)
        probabilities = self._normalize_similarity_scores(scores)
        if not probabilities:
            selected_local_idx = 0
        else:
            selected_local_idx = random.choices(range(len(valid_indices)), weights=probabilities, k=1)[0]

        archive_selected_idx = valid_indices[selected_local_idx]
        selected_entry_ids = [
            archive_entries[archive_selected_idx].get("id")
        ] if 0 <= archive_selected_idx < len(archive_entries) else []
        branch_scores = {
            valid_indices[i]: float(scores[i]) for i in range(len(scores))
        }
        branch_probabilities = {
            valid_indices[i]: float(probabilities[i]) for i in range(len(probabilities))
        }

        selected_score = scores[selected_local_idx] if selected_local_idx < len(scores) else float("-inf")
        selected_prob = probabilities[selected_local_idx] if selected_local_idx < len(probabilities) else 0.0
        rationale = (
            f"CLIP noun baseline branched with 90% preference (roll={roll:.3f}); "
            f"sampled archive index {archive_selected_idx} (max similarity={selected_score:.3f}, p={selected_prob:.3f})."
        )

        decision = {
            "choice": "branch",
            "selected_images": [archive_selected_idx],
            "rationale": rationale,
            "raw_response": None,
            "timestamp": decision_timestamp,
            "archive_elite_names": list(elite_name_list),
            "archive_subset_counts": dict(subset_counts),
            "archive_display_order": list(range(len(archive_entries))),
            "selected_entry_ids": [entry_id for entry_id in selected_entry_ids if entry_id],
            "branch_probability": branch_probability,
            "branch_roll": roll,
            "branch_scores": branch_scores,
            "branch_probabilities": branch_probabilities,
            "branch_subset_ranges": subset_ranges,
        }
        return decision

    def _store_branching_snapshot(
        self,
        archive_entries: List[Dict[str, Any]],
        subset_ranges: List[Dict[str, Any]],
        subset_counts: Dict[str, int],
        elite_name_list: Sequence[str],
        display_order: Optional[List[int]],
    ) -> None:
        self._initial_branch_archive_entries = copy.deepcopy(archive_entries)
        self._initial_branch_subset_ranges = copy.deepcopy(subset_ranges)
        self._initial_branch_subset_counts = dict(subset_counts)
        self._initial_branch_elite_names = list(elite_name_list)
        if display_order is None:
            self._initial_branch_display_order = list(range(len(archive_entries)))
        else:
            self._initial_branch_display_order = list(display_order)
        snapshot = {
            "entries": self._initial_branch_archive_entries,
            "subset_ranges": self._initial_branch_subset_ranges,
            "subset_counts": self._initial_branch_subset_counts,
            "elite_names": self._initial_branch_elite_names,
            "display_order": self._initial_branch_display_order,
            "timestamp": datetime.now().isoformat(),
        }
        try:
            self._branching_snapshot_path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
        except OSError:
            pass

    def _load_branching_snapshot(self) -> None:
        if not hasattr(self, "_branching_snapshot_path"):
            return
        if not self._branching_snapshot_path.exists():
            return
        try:
            data = json.loads(self._branching_snapshot_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        self._initial_branch_archive_entries = data.get("entries") or []
        self._initial_branch_subset_ranges = data.get("subset_ranges") or []
        stored_counts = data.get("subset_counts") or {}
        self._initial_branch_subset_counts = dict(stored_counts)
        self._initial_branch_elite_names = list(data.get("elite_names") or [])
        display_order = data.get("display_order") or list(range(len(self._initial_branch_archive_entries)))
        self._initial_branch_display_order = list(display_order)

    def _normalize_restart_request(self, payload: Any, generation: int) -> Optional[Dict[str, Any]]:
        if payload in (None, False, "", [], {}):
            return None
        mode: Optional[str] = None
        reason: Optional[str] = None
        selected_indices: List[int] = []
        if isinstance(payload, str):
            value = payload.strip().lower()
            if value in {"branch", "archive", "rebranch"}:
                mode = "branch"
            elif value in {"fresh", "random", "reset", "restart", "new"}:
                mode = "fresh"
        elif isinstance(payload, bool):
            mode = "fresh" if payload else None
        elif isinstance(payload, dict):
            raw_mode = payload.get("mode") or payload.get("choice") or payload.get("type")
            if isinstance(raw_mode, str):
                value = raw_mode.strip().lower()
                if value in {"branch", "archive", "rebranch"}:
                    mode = "branch"
                elif value in {"fresh", "random", "reset", "restart", "new"}:
                    mode = "fresh"
            reason_value = payload.get("reason") or payload.get("rationale") or payload.get("why")
            if reason_value is not None:
                text = str(reason_value).strip()
                if text:
                    reason = text
            selected_value = (
                payload.get("selected")
                or payload.get("indices")
                or payload.get("selection")
                or payload.get("archive_indices")
            )
            if selected_value is not None:
                selected_indices = _ensure_int_list(selected_value)
        elif isinstance(payload, (list, tuple)):
            selected_indices = _ensure_int_list(payload)
            if selected_indices:
                mode = mode or "branch"
        if mode is None:
            return None
        return {
            "mode": mode,
            "reason": reason,
            "selected_indices": selected_indices,
            "generation": generation,
            "timestamp": datetime.now().isoformat(),
        }

    def apply_pending_restart(self) -> Optional[Dict[str, Any]]:
        if not self._pending_restart_request:
            return None
        request = self._pending_restart_request
        self._pending_restart_request = None
        decision = self._build_restart_decision(request)
        restart_context = {
            "mode": request.get("mode"),
            "reason": request.get("reason"),
            "generation": request.get("generation"),
            "timestamp": request.get("timestamp"),
            "count": self._restart_counter + 1,
        }
        decision["restart_context"] = restart_context
        self.branching_decision = decision
        self.initialise_population(decision)
        log_entry = {
            "agent_id": self.agent_id,
            "restart": restart_context,
            "decision_choice": decision.get("choice"),
            "selected_images": decision.get("selected_images"),
            "selected_entry_ids": decision.get("selected_entry_ids"),
            "timestamp": datetime.now().isoformat(),
        }
        self._restart_history.append(log_entry)
        self._append_restart_log(log_entry)
        self._restart_counter += 1
        return decision

    def _build_restart_decision(self, request: Dict[str, Any]) -> Dict[str, Any]:
        mode = request.get("mode") or "fresh"
        if mode == "branch":
            decision = self._build_restart_branch_decision(request)
        else:
            decision = self._build_restart_fresh_decision(request)
        if not decision.get("selected_entry_ids"):
            decision.setdefault("selected_entry_ids", [])
        return decision

    def _build_restart_fresh_decision(
        self,
        request: Dict[str, Any],
        rationale: Optional[str] = None,
    ) -> Dict[str, Any]:
        base_text = rationale or "Restart requested; starting from a fresh random population."
        reason = request.get("reason")
        if reason:
            base_text = f"{base_text} ({reason})"
        return {
            "choice": "fresh",
            "selected_images": [],
            "selected_entry_ids": [],
            "rationale": base_text,
            "raw_response": None,
            "timestamp": datetime.now().isoformat(),
            "archive_elite_names": list(self._initial_branch_elite_names),
            "archive_subset_counts": dict(self._initial_branch_subset_counts),
        }

    def _build_restart_branch_decision(self, request: Dict[str, Any]) -> Dict[str, Any]:
        if not self._initial_branch_archive_entries:
            return self._build_restart_fresh_decision(
                request,
                rationale="Restart requested but the original archive snapshot is unavailable; starting fresh instead.",
            )
        archive_entries = copy.deepcopy(self._initial_branch_archive_entries)
        subset_counts = dict(self._initial_branch_subset_counts)
        elite_names = list(self._initial_branch_elite_names)
        working_display_order = (
            list(self._initial_branch_display_order)
            if self._initial_branch_display_order
            else list(range(len(archive_entries)))
        )
        if len(working_display_order) != len(archive_entries):
            working_display_order = list(range(len(archive_entries)))

        display_to_archive_index: Dict[int, int] = {
            display_idx: archive_idx
            for display_idx, archive_idx in enumerate(working_display_order)
            if 0 <= archive_idx < len(archive_entries)
        }

        requested_display_indices = _ensure_int_list(request.get("selected_indices") or [])
        archive_indices: List[int] = []
        invalid_display_indices: List[int] = []
        for display_idx in requested_display_indices:
            archive_idx = display_to_archive_index.get(display_idx)
            if archive_idx is None:
                invalid_display_indices.append(display_idx)
                continue
            archive_indices.append(archive_idx)

        # Preserve order while removing duplicates.
        seen_indices: Set[int] = set()
        unique_archive_indices: List[int] = []
        for idx in archive_indices:
            if idx in seen_indices:
                continue
            seen_indices.add(idx)
            unique_archive_indices.append(idx)

        valid_indices, invalid_archive_indices = self._validate_branch_indices(
            unique_archive_indices,
            archive_entries,
        )

        if valid_indices:
            rationale_parts = ["Restart requested; branching from the archive using specified indices."]
            reason = request.get("reason")
            if reason:
                rationale_parts.append(f"Reason: {reason}")
            ignored: List[int] = []
            if invalid_display_indices:
                ignored.extend(invalid_display_indices)
            if invalid_archive_indices:
                ignored.extend(invalid_archive_indices)
            if ignored:
                rationale_parts.append(f"Ignored invalid indices: {ignored}")
            decision = {
                "choice": "branch",
                "selected_images": valid_indices,
                "selected_entry_ids": [
                    archive_entries[idx]["id"]
                    for idx in valid_indices
                    if 0 <= idx < len(archive_entries)
                ],
                "rationale": " ".join(rationale_parts),
                "raw_response": None,
                "timestamp": datetime.now().isoformat(),
                "archive_elite_names": elite_names,
                "archive_subset_counts": subset_counts,
                "archive_display_order": working_display_order,
                "selected_display_indices": [
                    idx
                    for idx in requested_display_indices
                    if idx in display_to_archive_index and display_to_archive_index[idx] in valid_indices
                ],
            }
            preview_path = self._save_archive_branch_preview(decision, archive_entries)
            decision["branch_preview_path"] = str(preview_path)
            return decision

        prompt_note = (
            "Restart request: this is the same archive grid you saw at the beginning of the session. "
            "Pick a different favorite (indices unchanged) or respond with null to start fresh."
        )
        if invalid_display_indices or invalid_archive_indices:
            prompt_note = (
                f"Restart requested with branch mode but the provided indices were invalid: "
                f"{invalid_display_indices + invalid_archive_indices}. Please reply with a valid index from the archive grid shown below."
            )
        elif not requested_display_indices:
            prompt_note = (
                "Restart requested with branch mode but no archive indices were provided. "
                "Please reply with a valid index from the archive grid shown below."
            )

        branch_attempts = 0
        max_branch_attempts = 2
        next_prompt_note = prompt_note

        while branch_attempts < max_branch_attempts:
            branch_attempts += 1
            decision = self._decide_branching_from_entries(
                copy.deepcopy(archive_entries),
                copy.deepcopy(self._initial_branch_subset_ranges),
                dict(self._initial_branch_subset_counts),
                list(elite_names),
                prompt_note=next_prompt_note,
                display_order=list(working_display_order) if working_display_order else None,
            )
            if decision.get("selected_images"):
                if request.get("reason"):
                    rationale_text = decision.get("rationale", "") or ""
                    decision["rationale"] = f"{rationale_text} (restart reason: {request['reason']})".strip()
                return decision

            next_prompt_note = (
                "No valid archive selection was detected in your restart request. "
                "Please choose a valid index from the archive grid shown again."
            )

        fallback_rationale = (
            "Restart requested with branch mode but no valid archive selection was provided after retrying; "
            "starting from a fresh random population instead."
        )
        if request.get("reason"):
            fallback_rationale = f"{fallback_rationale} (restart reason: {request['reason']})"
        return self._build_restart_fresh_decision(
            request,
            rationale=fallback_rationale,
        )

    def _append_restart_log(self, payload: Dict[str, Any]) -> None:
        try:
            with self._restart_log_path.open("a", encoding="utf-8") as fp:
                fp.write(json.dumps(payload))
                fp.write("\n")
        except OSError:
            pass

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
        generation_i = int(self.population.generation)
        run_label = self._current_run_label()
        name_prefix = f"{run_label}_"
        if len(genomes) != self.rows * self.cols:
            raise ValueError(
                f"Expected {self.rows * self.cols} genomes, received {len(genomes)}."
            )

        self._zero_color_weights(genome for _, genome in genomes)

        if self.render_genome_diagrams:
            diagram_paths = save_neat_genome_diagrams(genomes, config, self.population_dir, generation_i)
            if diagram_paths:
                diagram_dir = diagram_paths[0].parent
                print(f"Genome diagrams saved to {diagram_dir}")
            elif graphviz is None and not self._diagram_warning_emitted:
                print("Graphviz not available; skipping genome diagram export.")
                self._diagram_warning_emitted = True

        gray_images, color_images = render_genome_images(
            genomes,
            config,
            self.thumb_size,
        )

        image_paths = self._save_generation_images(generation_i, color_images, gray_images)

        system_instruction = self.system_instruction
        prompt_template = self.prompt_template
        require_selection = True
        if generation_i == self.generations - 1:
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

        if generation_i == 0 and self.chat_history_turns == 0:
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
        if generation_i == 0 and self.agent_id.endswith("0"):
            prompt_template += (
                " You are the first agent. This is an initial random population, and you may select one or more parents for the next step of evolution. "
            )

        prompt_template = self._apply_pending_prompt_notes(prompt_template)

        population_images = color_images if self._color_enabled else gray_images
        grid_path: Optional[Path] = None
        view_index = 0

        if self.selection_baseline == "none":
            selection_meta_raw: Dict[str, Any]
            if self.rand_select_prob > 0 and random.random() < self.rand_select_prob:
                grid_image = create_numbered_grid(population_images, self.rows, self.cols, self.thumb_size)
                grid_path = self.query_dir / f"{name_prefix}gen_{generation_i:03d}_view_{view_index:02d}_grid.png"
                grid_image.save(grid_path, format="PNG")
                self._update_latest_image(grid_path)
                selected_idx = random.randrange(len(population_images))
                selection_meta_raw = {
                    "selected": [selected_idx],
                    "grid_path": str(grid_path),
                    "short_circuit": "random_selection",
                }
                if self.request_rationale:
                    selection_meta_raw["rationale"] = f"Random selection triggered (p={self.rand_select_prob:.3f})."
                print(
                    f"[{self.agent_id}] Gen {generation_i}: random short-circuit selection -> {selected_idx} "
                    f"(p={self.rand_select_prob:.3f})"
                )
            else:
                while True:
                    prompt_with_settings = self._prompt_with_settings(prompt_template)
                    variant_key = "color" if self._color_enabled else "gray"
                    population_images = color_images if self._color_enabled else gray_images
                    variant_image_map = {
                        idx: paths.color if variant_key == "color" else paths.gray
                        for idx, paths in image_paths.items()
                    }
                    try:
                        selection_meta_candidate = select_parents_from_grid(
                            self.config.model,
                            population_images,
                            self.config.thinking_budget,
                            self.config.temperature,
                            generation_i,
                            prompt_with_settings,
                            self.query_dir,
                            self.select_k,
                            system_instruction,
                            self.chat_history_turns,
                            require_selection=require_selection,
                            request_rationale=self.request_rationale,
                            allow_color_toggle=True,
                            current_color=self._color_enabled,
                            view_index=view_index,
                            image_path_map=variant_image_map,
                            log_raw_response=self.log_raw_responses,
                            run_label=run_label,
                        )
                        fallback_invoked = False
                    except GeminiPromptBlockedError as exc:
                        reason = exc.block_reason or "unspecified"
                        print(
                            f"[{self.agent_id}] Gemini blocked generation {generation_i} prompt (reason: {reason}). Using fallback selection."
                        )
                        selection_meta_candidate = self._fallback_selection_after_block(
                            population_images,
                            generation_i,
                            exc,
                        )
                        fallback_invoked = True

                    grid_path_candidate = self._resolve_query_path(selection_meta_candidate.get("grid_path"))
                    if grid_path_candidate is not None and grid_path_candidate.exists():
                        self._update_latest_image(grid_path_candidate)
                    if fallback_invoked:
                        selection_meta_raw = selection_meta_candidate
                        grid_path = grid_path_candidate
                        break
                    requested_color = self._coerce_bool(selection_meta_candidate.get("color"))
                    if requested_color is not None and requested_color != self._color_enabled:
                        self._color_enabled = requested_color
                        self._update_mutation_mode(self._mutation_mode)
                        view_index += 1
                        continue
                    selection_meta_raw = selection_meta_candidate
                    grid_path = grid_path_candidate
                    break
        else:
            grid_image = create_numbered_grid(population_images, self.rows, self.cols, self.thumb_size)
            grid_path = self.query_dir / f"{name_prefix}gen_{generation_i:03d}_view_{view_index:02d}_grid.png"
            grid_image.save(grid_path, format="PNG")
            self._update_latest_image(grid_path)
            selection_meta_raw = self._select_parents_baseline(
                population_images,
                generation_i,
                genomes,
                config,
            )
            selection_meta_raw["grid_path"] = str(grid_path)
        selection_meta = dict(selection_meta_raw)
        if grid_path is None:
            resolved_grid_str = selection_meta.get("grid_path")
            if resolved_grid_str:
                grid_path = Path(resolved_grid_str)
            else:
                grid_path = self.query_dir / f"{name_prefix}gen_{generation_i:03d}_view_{view_index:02d}_grid.png"
        selection_meta["grid_path"] = str(grid_path)
        selection_meta["color"] = self._color_enabled
        
        # In fixed session mode, ignore restart and quit requests
        if self.fixed_session_lengths:
            restart_request = None
            quit_requested = False
            quit_reason = None
            if selection_meta_raw.get("restart"):
                print(f"[{self.agent_id}] Ignoring restart request in fixed session mode at generation {generation_i}")
            if selection_meta.get("quit"):
                print(f"[{self.agent_id}] Ignoring quit request in fixed session mode at generation {generation_i}")
        else:
            restart_request = self._normalize_restart_request(selection_meta_raw.get("restart"), generation_i)
            if restart_request:
                self._pending_restart_request = restart_request
            quit_requested = bool(selection_meta.get("quit"))
            quit_reason = selection_meta.get("quit_reason")
            if quit_requested:
                reason_text = str(quit_reason).strip() if quit_reason else ""
                if not self._quit_requested:
                    self._quit_reason = reason_text or None
                    self._quit_generation = generation_i
                self._quit_requested = True
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

        genome_snapshots: Dict[int, neat.DefaultGenome] = {}
        for idx, (_, genome) in enumerate(genomes):
            genome_snapshots[idx] = copy.deepcopy(genome)

        record = GenerationArtifacts(
            grid_path=grid_path,
            selection_path=selection_path,
            image_paths=image_paths,
            genome_snapshots=genome_snapshots,
        )
        print(f"Saved selection grid to {selection_path}")
        self._generation_records[generation_i] = record
        self._log_generation_lineage(generation_i, genomes)
        self._save_selected_genomes(generation_i, genomes, selected_indices)

        publish_payload = None
        if self.selection_baseline == "none":
            publish_payload = self._parse_publish_payload(selection_meta.get("response_text", ""))
        else:
            publish_interval = 20
            should_publish = (generation_i + 1) % publish_interval == 0
            if should_publish and selected_indices:
                publish_payload = {
                    "index": selected_indices[0],
                    "reason": f"Baseline publish every {publish_interval} generations.",
                    "title": "Baseline favorite",
                    "raw": None,
                }

        publication_scheduled = False
        publish_index_for_highlight: Optional[int] = None
        if publish_payload is not None:
            publish_index = publish_payload.get("index")
            if publish_index in record.image_paths:
                favorite_reason = publish_payload.get("reason", "")
                favorite_title = publish_payload.get("title", "")
                favorite = {
                    "generation": generation_i,
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
                )
                publication_scheduled = scheduled
                selection_meta["publish"] = {
                    "index": publish_index,
                    "reason": favorite_reason,
                }
                if scheduled:
                    publish_index_for_highlight = publish_index
                elif self._last_publish_rejection:
                    selection_meta["publish_error"] = dict(self._last_publish_rejection)
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
            and generation_i == self.generations - 1
            and not self._has_publication()
            and record.image_paths
        ):
            fallback_index = self._choose_forced_publication_index(selected_indices, record)
            forced_rationale = (
                "Forced publication from selected parents." if fallback_index in selected_indices else "Forced publication at final generation."
            )
            favorite = {
                "generation": generation_i,
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
            elif self._last_publish_rejection and "publish_error" not in selection_meta:
                selection_meta["publish_error"] = dict(self._last_publish_rejection)

        selection_path = Path(selection_meta.get("selection_path") or grid_path)
        self._render_selection_with_publication(
            population_images,
            selected_indices,
            publish_index_for_highlight,
            selection_path,
        )
        self._update_latest_image(selection_path)
        record.selection_path = selection_path

        self._print_selection_response(
            generation_i,
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
            self.progress_callback(generation_i, favorite_payload, archive_payload)
        self._append_selection_history(generation_i, selection_meta)
        if quit_requested:
            message = self._quit_reason or "Agent requested to end the session."
            print(f"[{self.agent_id}] Quit requested at generation {generation_i}: {message}")
            raise AgentQuitRequested(message)
        if restart_request:
            raise AgentRestartRequested(restart_request)

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

    def _current_run_label(self) -> str:
        return f"branch_{self._restart_counter:03d}"

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
        entry.setdefault("run_label", self._current_run_label())
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
        if isinstance(parsed, ValueError) or not isinstance(parsed, dict):
            return None
        payload = parsed.get("publish")
        if payload in (None, "", "none", "null"):
            return None
        index_value: Any
        rationale: str = ""
        title: str = ""
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

    def _restore_selection_settings(self) -> None:
        if not self.resume_mode:
            return
        if not self._selection_history_path.exists():
            return
        last_entry: Optional[Dict[str, Any]] = None
        with self._selection_history_path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                last_entry = payload
        if not last_entry:
            return

        color_value = last_entry.get("color")
        color_bool = self._coerce_bool(color_value)
        if color_bool is not None:
            self._color_enabled = color_bool

        mutation_mode_value = last_entry.get("mutation_mode")
        if mutation_mode_value is not None:
            self._update_mutation_mode(mutation_mode_value)

        mutation_strength_value = last_entry.get("mutation_strength")
        if mutation_strength_value is not None:
            self._update_mutation_strength(mutation_strength_value)

        print(
            f"[{self.agent_id}] Restored color/mutation settings from selection history."
        )

    def _apply_publication(
        self,
        favorite: Dict[str, Any],
        *,
        forced: bool,
        response_text: Optional[str],
        source: str,
    ) -> bool:
        self._last_publish_rejection = None
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

        # prefer_color = bool(self._color_enabled)
        # image_variants = record.image_paths.get(index_int)
        # if isinstance(image_variants, ImageVariantPaths):
        #     image_path = image_variants.for_color_mode(prefer_color)
        # else:
        #     image_path = image_variants
        # duplicate_entry, pixel_distance = self._find_duplicate_archive_entry(image_path)
        # if duplicate_entry is not None:
        #     rejection: Dict[str, Any] = {
        #         "index": index_int,
        #         "generation": generation_int,
        #         "reason": "duplicate_archive_image",
        #         "archive_entry_id": duplicate_entry.entry_id,
        #     }
        #     if pixel_distance is not None:
        #         rejection["pixel_distance"] = pixel_distance
        #         rejection["epsilon"] = self._duplicate_publish_epsilon
        #     self._last_publish_rejection = rejection
        #     distance_str = f"{pixel_distance:.6f}" if pixel_distance is not None else "unknown"
        #     note = (
        #         f"Publication rejected: image {index_int} is too similar to archived entry {duplicate_entry.entry_id} "
        #         f"(distance {distance_str} <= {self._duplicate_publish_epsilon:.6f}). "
        #         "Please publish something meaningfully different."
        #     )
        #     self._queue_prompt_note(note)
        #     print(
        #         f"[{self.agent_id}] Publication rejected as duplicate of {duplicate_entry.entry_id} "
        #         f"(distance {distance_str}, epsilon {self._duplicate_publish_epsilon:.6f})."
        #     )
        #     return False

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
        entry = self.publish_to_archive(payload)
        if entry is None:
            print(
                f"[{self.agent_id}] Publication failed: generation={generation_int}, index={index_int}, title='{title}', forced={payload.get('forced')}" 
            )
            return False

        payload["archive_entry_id"] = entry.entry_id
        genome_key_committed = payload.get("genome_key")
        if genome_key_committed is not None:
            self._archive_seed_map[genome_key_committed] = {
                "entry_id": entry.entry_id,
                "agent_id": self.agent_id,
                "generation": payload.get("generation"),
            }
        self._append_publication_history(payload)
        self.favorite_archive_entry = entry
        self.favorite_decision = payload
        print(
            f"[{self.agent_id}] Publication committed: generation={generation_int}, index={index_int}, title='{title}', forced={payload.get('forced')}" 
        )
        return True

    @property
    def quit_reason(self) -> Optional[str]:
        return self._quit_reason

    @property
    def quit_generation(self) -> Optional[int]:
        return self._quit_generation

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
        population_images: List[Image.Image],
        selected_indices: Sequence[int],
        publish_index: Optional[int],
        selection_path: Path,
    ) -> None:
        image = create_numbered_grid(population_images, rows=self.rows, cols=self.cols, thumb_size=self.thumb_size, selected=selected_indices)
        if publish_index is not None:
            row = publish_index // self.cols
            col = publish_index % self.cols
            thumb = self.thumb_size
            margin = 12
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
        if not archive_entries:
            raise ValueError("No archive entries available for preview generation.")

        archive_root = Path(getattr(self.archive_manager, "archive_dir", self.agent_dir))

        def _resolve_image_path(path_value: Any) -> Path:
            if not path_value:
                raise ValueError("Missing image path in archive entry.")
            candidates: List[Path] = []
            raw_path = Path(path_value)
            candidates.append(raw_path)
            if not raw_path.is_absolute():
                candidates.append((archive_root / raw_path).resolve())
            relative_suffix = relative_suffix_after_dir(raw_path, "archive")
            if relative_suffix is not None:
                candidates.append((archive_root / relative_suffix).resolve())
            seen: Set[Path] = set()
            for candidate in candidates:
                if candidate in seen:
                    continue
                seen.add(candidate)
                if candidate.exists():
                    return candidate
            raise FileNotFoundError(f"Archive image not found: {raw_path}")

        grouped: Dict[str, List[Tuple[int, Dict[str, Any]]]] = {}
        label_order: List[str] = []
        for idx, entry in enumerate(archive_entries):
            label = str(entry.get("branching_subset_label") or entry.get("branching_subset") or "Archive").strip() or "Archive"
            if label not in grouped:
                grouped[label] = []
                label_order.append(label)
            grouped[label].append((idx, entry))

        background_color = (18, 18, 22)
        margin = ARCHIVE_GRID_MARGIN
        header_gap = max(4, margin // 3)
        font = ImageFont.load_default()
        panes: List[Dict[str, Any]] = []

        for label in label_order:
            subset = grouped[label]
            subset_images: List[Image.Image] = []
            subset_indices: List[int] = []
            for idx, entry in subset:
                path_value = entry.get("image_path")
                resolved_path = _resolve_image_path(path_value)
                with Image.open(resolved_path) as img:
                    processed = img.convert("RGB").resize((self.thumb_size, self.thumb_size), Image.Resampling.LANCZOS)
                subset_images.append(processed)
                subset_indices.append(idx)

            if not subset_images:
                continue

            columns = max(1, math.ceil(math.sqrt(len(subset_images))))
            rows = math.ceil(len(subset_images) / columns)
            tile_width, tile_height = subset_images[0].size
            pane_width = columns * tile_width + (columns + 1) * margin
            pane_height = rows * tile_height + (rows + 1) * margin
            pane_canvas = Image.new("RGB", (pane_width, pane_height), background_color)
            pane_positions: List[Dict[str, Any]] = []
            for img_idx, processed in enumerate(subset_images):
                col = img_idx % columns
                row = img_idx // columns
                x = margin + col * (tile_width + margin)
                y = margin + row * (tile_height + margin)
                pane_canvas.paste(processed, (x, y))
                pane_positions.append(
                    {
                        "index": subset_indices[img_idx],
                        "bbox": [x, y, x + tile_width, y + tile_height],
                    }
                )

            try:
                label_bbox = font.getbbox(f"{label}:")
                header_height = label_bbox[3] - label_bbox[1]
            except AttributeError:
                header_height = font.getsize(f"{label}:")[1]

            panes.append(
                {
                    "label": label,
                    "canvas": pane_canvas,
                    "width": pane_width,
                    "height": pane_height,
                    "header_height": header_height,
                    "positions": pane_positions,
                }
            )

        if not panes:
            raise ValueError("Failed to construct branch preview; no valid subset images found.")

        max_header_height = max(pane["header_height"] for pane in panes)
        max_grid_height = max(pane["height"] for pane in panes)
        total_width = sum(pane["width"] for pane in panes) + margin * (len(panes) + 1)
        total_height = margin + max_header_height + header_gap + max_grid_height + margin
        canvas = Image.new("RGB", (total_width, total_height), background_color)
        draw = ImageDraw.Draw(canvas)
        x_cursor = margin
        grid_y = margin + max_header_height + header_gap
        selected_set = set(selected)

        for pane in panes:
            header_text = f"{pane['label']}:"
            draw.text((x_cursor, margin), header_text, fill=(235, 235, 240), font=font)
            canvas.paste(pane["canvas"], (x_cursor, grid_y))
            for position in pane["positions"]:
                idx = position["index"]
                if idx not in selected_set:
                    continue
                x0 = x_cursor + position["bbox"][0]
                y0 = grid_y + position["bbox"][1]
                x1 = x_cursor + position["bbox"][2]
                y1 = grid_y + position["bbox"][3]
                _draw_dotted_rectangle(
                    draw,
                    (x0, y0, x1, y1),
                    color=(255, 0, 0),
                    width=5,
            )
            x_cursor += pane["width"] + margin

        output_path = self.query_dir / f"{self._current_run_label()}_archive_branch.png"
        canvas.save(output_path, format="PNG")
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

    # ------------------------------------------------------------------
    # CLIP noun baseline helpers
    # ------------------------------------------------------------------
    def _ensure_clip_noun_components(self) -> None:
        # Reuse a per-process cache so we only embed nouns once.
        cached = _CLIP_NOUN_CACHE.get("loaded", False)
        if cached:
            self._clip_model = _CLIP_NOUN_CACHE["model"]
            self._clip_preprocess = _CLIP_NOUN_CACHE["preprocess"]
            self._clip_tokenizer = _CLIP_NOUN_CACHE["tokenizer"]
            self._clip_device = _CLIP_NOUN_CACHE["device"]
            self._clip_text_embeddings = _CLIP_NOUN_CACHE["noun_embeddings"]
            self._clip_noun_labels = _CLIP_NOUN_CACHE["nouns"]
            self._clip_torch = _CLIP_NOUN_CACHE["torch"]
            return

        try:
            import torch  # type: ignore
            import open_clip  # type: ignore
            from compute_noun_similarity import embed_texts, format_prompts, load_nouns
        except Exception as exc:  # pragma: no cover - import guard
            raise RuntimeError(
                "CLIP noun baseline requires torch and open_clip (see compute_noun_similarity.py dependencies)."
            ) from exc

        noun_path = REPO_ROOT / "nounlist.txt"
        nouns = load_nouns(noun_path)
        prompts = format_prompts(nouns, "{label}")

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-H-14", pretrained="laion2b_s32b_b79k"
        )
        model.eval()
        model.to(device)
        tokenizer = open_clip.get_tokenizer("ViT-H-14")
        noun_embeddings = embed_texts(
            model,
            tokenizer,
            prompts,
            device,
            batch_size=512,
        )

        self._clip_model = model
        self._clip_preprocess = preprocess
        self._clip_tokenizer = tokenizer
        self._clip_device = device
        self._clip_text_embeddings = noun_embeddings
        self._clip_noun_labels = nouns
        self._clip_torch = torch

        _CLIP_NOUN_CACHE.update(
            {
                "loaded": True,
                "model": model,
                "preprocess": preprocess,
                "tokenizer": tokenizer,
                "device": device,
                "noun_embeddings": noun_embeddings,
                "nouns": nouns,
                "torch": torch,
            }
        )
        print(
            f"[{self.agent_id}] Initialized CLIP noun baseline with {len(nouns)} nouns on device {device}."
        )

    def _clip_embed_images(self, images: Sequence[Image.Image]) -> np.ndarray:
        self._ensure_clip_noun_components()
        assert self._clip_model is not None
        assert self._clip_preprocess is not None
        assert self._clip_device is not None
        assert self._clip_text_embeddings is not None
        assert self._clip_torch is not None

        if not images:
            return np.zeros((0, self._clip_text_embeddings.shape[1]), dtype=float)

        tensors = []
        for img in images:
            tensors.append(self._clip_preprocess(img.convert("RGB")))

        batch = self._clip_torch.stack(tensors, dim=0).to(self._clip_device)
        with self._clip_torch.no_grad():
            emb = self._clip_model.encode_image(batch)
        arr = emb.cpu().numpy()
        norm = np.linalg.norm(arr, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        return arr / norm

    def _clip_max_similarity_scores(self, images: Sequence[Image.Image]) -> List[float]:
        embeddings = self._clip_embed_images(images)
        assert self._clip_text_embeddings is not None
        if embeddings.size == 0:
            return []
        sims = embeddings @ self._clip_text_embeddings.T
        max_scores = np.max(sims, axis=1)
        return max_scores.tolist()

    @staticmethod
    def _normalize_similarity_scores(scores: Sequence[float]) -> List[float]:
        if not scores:
            return []
        weights = np.array(scores, dtype=float)
        weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        weights = weights - weights.min()
        weights = weights + 1e-6
        total = float(weights.sum())
        if total <= 0 or not math.isfinite(total):
            return [1.0 / len(scores)] * len(scores)
        return (weights / total).tolist()

    def _select_parents_clip_nouns(
        self,
        population_images: List[Image.Image],
        generation: int,
    ) -> Dict[str, Any]:
        scores = self._clip_max_similarity_scores(population_images)
        min_distances = [1.0 - score for score in scores]
        probabilities = self._normalize_similarity_scores(scores)
        if not probabilities or not population_images:
            selected_idx = 0 if population_images else -1
            rationale = "CLIP noun baseline fell back to index 0 (no scores available)."
        else:
            selected_idx = random.choices(range(len(population_images)), weights=probabilities, k=1)[0]
            prob_selected = probabilities[selected_idx] if selected_idx < len(probabilities) else 0.0
            selected_score = scores[selected_idx] if selected_idx < len(scores) else float("-inf")
            rationale = (
                f"CLIP noun baseline sampled index {selected_idx} "
                f"(max similarity={selected_score:.3f}, min distance={1.0 - selected_score:.3f}, p={prob_selected:.3f})."
            )

        metadata: Dict[str, Any] = {
            "selected": [selected_idx] if selected_idx >= 0 else [],
            "rationale": rationale,
            "baseline": self.selection_baseline,
            "max_similarities": scores,
            "min_distances": min_distances,
            "probabilities": probabilities,
            "noun_count": len(self._clip_noun_labels),
            "model": "ViT-H-14/laion2b_s32b_b79k",
            "selection_count": 1,
        }
        return self._write_baseline_artifacts(generation, population_images, metadata)

    def _select_parents_baseline(
        self,
        population_images: List[Image.Image],
        generation: int,
        genomes: List[Tuple[int, neat.DefaultGenome]],
        config: neat.Config,
    ) -> Dict[str, Any]:
        if self.selection_baseline == "clip-nouns":
            return self._select_parents_clip_nouns(population_images, generation)

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

        return self._write_baseline_artifacts(generation, population_images, metadata)

    def _write_baseline_artifacts(
        self,
        generation: int,
        population_images: List[Image.Image],
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        run_label = self._current_run_label()
        name_prefix = f"{run_label}_"
        self.query_dir.mkdir(parents=True, exist_ok=True)
        selected = metadata.get("selected", [])
        selection_image = create_numbered_grid(population_images=population_images, rows=self.rows, cols=self.cols, thumb_size=self.thumb_size, selected=selected)

        suffix = "_view_00"
        grid_path = self.query_dir / f"{name_prefix}gen_{generation:03d}{suffix}_grid.png"
        selection_path = self.query_dir / f"{name_prefix}gen_{generation:03d}{suffix}_selection.png"
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
                "color": self._color_enabled,
                "color_toggle_only": False,
                "run_label": run_label,
            }
        )

        metadata_dir = self.query_dir / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        meta_path = metadata_dir / f"{name_prefix}gen_{generation:03d}{suffix}_selection.json"
        meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        payload["metadata_path"] = str(meta_path)

        return payload
