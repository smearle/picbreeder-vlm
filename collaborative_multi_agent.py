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

import copy
import gzip
import json
import math
import pickle
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import shutil
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple
from itertools import count

import graphviz
import hydra
from hydra.conf import HelpConf, HydraConf
from hydra.core.config_store import ConfigStore
from hydra.utils import get_original_cwd

import neat
from neat.population import CompleteExtinctionException
from PIL import Image, ImageDraw, ImageFont

from auto_evolve import (
    DEFAULT_BASELINE_SELECTION_LIMIT,
    query_with_history,
    extract_json_object,
    ensure_gemini_key,
    gen_selection_prompt,
    reset_chat_session,
    select_parents_from_grid,
    restore_chat_history_from_metadata,
    summarize_genome_structure,
)
from artifacts import build_generation_state, save_neat_genome_diagrams, save_neat_population
from experiment_cli import SELECTION_BASELINES, cap_select_k_for_engine
from neat_components import (
    CHECKPOINT_SUFFIX,
    GenerationCheckpointer,
    InteractiveStagnation,
    PicbreederGenome,
    apply_picbreeder_config_defaults,
    seed_initial_population,
    sync_population_output_activations,
)
from picbreeder_reproduction import PicbreederReproduction
from rendering import _draw_dotted_rectangle, create_numbered_grid
from utils import apply_random_seed

ARCHIVE_DIR_NAME = "archive"
DEFAULT_AGENT_GENERATIONS = 20
DEFAULT_CHAT_HISTORY_TURNS = -1  # Unlimited conversational history.
AGENT_DIR_PREFIX = "agent_"
ARCHIVE_GRID_MARGIN = 12

GOAL_PROMPTS = {
    "familiar_objects": "Your goal is to evolve images that resemble familiar real-world objects.",
    "unfamiliar_objects": "Your goal is to evolve images that resemble unfamiliar objects that may or may not exist.",
    "lizards": "Your goal is to evolve images that resemble lizards.",
    "fish": "Your goal is to evolve images that resemble fish.",
    "skulls": "Your goal is to evolve images that resemble skulls.",
    "apples": "Your goal is to evolve images that resemble apples.",
    "butterflies": "Your goal is to evolve images that resemble butterflies.",
    "flowers": "Your goal is to evolve images that resemble flowers.",
}

DEFAULT_SYSTEM_INSTRUCTION = (
    "You are playing with a collaborative online platform which allows users to interactively evolve small neural networks called Compositional Pattern Producing Networks (CPPNs) for generating images. "
    "{goal_prompt} "
    "This is an open-ended search process. You may set out with certain goals in mind, but be willing to quickly adapt them as new forms arise. "
    "If a certain evolutionary direction is not progressing, do not choose the same partially-successful parent repeatedly. "
    "Be willing to give up on local optima and explore new areas of the search space, even without a pre-defined target. "
    "At the first generation the initial grid will display an archive of images published by prior users as favorites (unless you are the first user). "
    """You may choose to "branch" one of these images, or start instead from a random initial population. """
    "At each subsequent generation, you will be shown a grid of numbered images produced by CPPNs. "
    "{selection_prompt}"
    "If so inclined, you may also select an image to publish to the online archive. "
    "You must publish at least once during your session. Publishing multiple times is allowed, though the most recent publication overwrites any previous ones. "
    "Your session will run for {n_generations} generations. "
    "Try to contribute something novel, interesting or useful to the online archive. "
    "Do not add something to the archive that is identical to an existing image. "
    "{color_prompt}"
    "Respond with JSON only: {{\"selected\": [indices], \"rationale\": \"brief explanation\"}}. "
    "(During branching, you may select only one image from which to branch; "
    " set selected to null to start from a fresh population.) "
    "You may also include a \"publish\" field in the JSON response if you wish to publish an image from this grid. It should have the form: "
    '{{"index": image_index, "title": "image title", "reason": "brief publication note"}}. '
    # "(Also, for debugging, please tell me how many previous grids you see in the chat history, briefly describe in neutral, objective terms how the grids have changed over time, "
    # "and tell me if you see the archive from which you made your original branching decision; add this to the `rationale` text.) "
    "{archive_novelty_prompt}"
    "{mutation_mode_prompt}"
)

ARCHIVE_NOVELTY_PROMPT = (
    "When justifying your publication choice, explain why the selected contribution is valuable to the archive. "
    "Identify the most similar existing image in the archive and explain how your selection differs from it."
)

def gen_selection_prompt(select_k: Optional[int]) -> str:
    if select_k is None:
        return "Pick one or several images by their numeric labels--the corresponding CPPNs will be used as the parents of the next generation. "
    if select_k == 1:
        return "Pick one image by its numeric label--the corresponding CPPN will be used as the parent of the next generation. "
    return f"Pick up to {select_k} images by their numeric labels--the corresponding CPPNs will be used as the parents of the next generation. "


PARENT_SELECTION_PROMPT = "Above is the grid at generation {generation}."

ARCHIVE_BRANCHING_PROMPT = (
    "Above is the archive of images published by prior users. You may choose to branch from one of them, or start from a fresh population. "
    "Their names are, in raster order: {elite_names}."
)

REPO_ROOT = Path(__file__).resolve().parent


@dataclass
class ArchiveEntry:
    """Structured information recorded for each archived favourite."""

    entry_id: str
    title: str
    image_path: Path
    genome_path: Path
    agent_id: str
    generation: int
    image_index: int
    rationale: str
    source_experiment: Path
    added_at: datetime
    metadata_path: Optional[Path] = None
    selection_grid_path: Optional[Path] = None
    genome_key: Optional[int] = None
    parent_genome_keys: List[int] = field(default_factory=list)
    source_entry_ids: List[str] = field(default_factory=list)
    ancestor_genome_keys: List[int] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "id": self.entry_id,
            "title": self.title,
            "image_path": str(self.image_path),
            "genome_path": str(self.genome_path),
            "agent_id": self.agent_id,
            "generation": self.generation,
            "image_index": self.image_index,
            "rationale": self.rationale,
            "source_experiment": str(self.source_experiment),
            "added_at": self.added_at.isoformat(),
            "metadata_path": str(self.metadata_path) if self.metadata_path else None,
            "selection_grid_path": (
                str(self.selection_grid_path) if self.selection_grid_path else None
            ),
            "genome_key": self.genome_key,
            "parent_genome_keys": list(self.parent_genome_keys),
            "source_entry_ids": list(self.source_entry_ids),
            "ancestor_genome_keys": list(self.ancestor_genome_keys),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ArchiveEntry":
        added_at_raw = payload.get("added_at")
        added_at = (
            datetime.fromisoformat(added_at_raw)
            if isinstance(added_at_raw, str)
            else datetime.now()
        )
        metadata_path = payload.get("metadata_path")
        selection_grid_path = payload.get("selection_grid_path")
        title = payload.get("title") or ""
        genome_key_raw = payload.get("genome_key")
        try:
            genome_key = int(genome_key_raw) if genome_key_raw is not None else None
        except (TypeError, ValueError):
            genome_key = None
        return cls(
            entry_id=payload["id"],
            title=str(title),
            image_path=Path(payload["image_path"]),
            genome_path=Path(payload["genome_path"]),
            agent_id=payload["agent_id"],
            generation=int(payload["generation"]),
            image_index=int(payload["image_index"]),
            rationale=str(payload.get("rationale", "")),
            source_experiment=Path(payload["source_experiment"]),
            added_at=added_at,
            metadata_path=Path(metadata_path) if metadata_path else None,
            selection_grid_path=Path(selection_grid_path) if selection_grid_path else None,
            genome_key=genome_key,
            parent_genome_keys=_ensure_int_list(payload.get("parent_genome_keys", [])),
            source_entry_ids=[
                str(value) for value in payload.get("source_entry_ids", []) if value is not None
            ],
            ancestor_genome_keys=_ensure_int_list(payload.get("ancestor_genome_keys", [])),
        )


class ArchiveManager:
    """Manages the shared archive of published favourites."""

    def __init__(self, archive_dir: Path) -> None:
        self.archive_dir = archive_dir
        self.metadata_file = archive_dir / "archive_metadata.json"
        self.images_dir = archive_dir / "images"
        self.genomes_dir = archive_dir / "genomes"
        self.checkpoints_dir = archive_dir / "checkpoints"
        self.logs_dir = archive_dir / "logs"
        for directory in (
            archive_dir,
            self.images_dir,
            self.genomes_dir,
            self.checkpoints_dir,
            self.logs_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        self._metadata: Dict[str, Any] = {}
        self._load()

    # ------------------------------------------------------------------
    # Metadata management
    # ------------------------------------------------------------------
    def _load(self) -> None:
        if self.metadata_file.exists():
            with self.metadata_file.open("r", encoding="utf-8") as fp:
                self._metadata = json.load(fp)
            for entry in self._metadata.get("entries", []):
                entry.setdefault("title", "")
        else:
            self._metadata = {
                "created_at": datetime.now().isoformat(),
                "next_id": 1,
                "entries": [],
            }
            self._persist()

    def _persist(self) -> None:
        with self.metadata_file.open("w", encoding="utf-8") as fp:
            json.dump(self._metadata, fp, indent=2)

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    @property
    def entries(self) -> List[Dict[str, Any]]:
        return list(self._metadata.get("entries", []))

    def get_elite_names(self, max_length: int = 80) -> List[str]:
        names: List[str] = []
        for entry in self._metadata.get("entries", []):
            title = str(entry.get("title") or "").strip()
            candidates = [title, entry.get("rationale"), entry.get("id"), "untitled"]
            name = ""
            for candidate in candidates:
                candidate_str = str(candidate or "").strip()
                if candidate_str:
                    name = candidate_str
                    break
            single_line = " ".join(name.split())
            if max_length > 0 and len(single_line) > max_length:
                single_line = single_line[: max_length - 3] + "..."
            names.append(single_line or "untitled")
        return names

    def get_entry(self, entry_id: str) -> Optional[ArchiveEntry]:
        for raw in self._metadata.get("entries", []):
            if raw.get("id") == entry_id:
                try:
                    return ArchiveEntry.from_dict(raw)
                except Exception:
                    return None
        return None

    def add_entry(
        self,
        *,
        image_bytes: bytes,
        genome: neat.DefaultGenome,
        agent_id: str,
        generation: int,
        image_index: int,
        rationale: str,
        title: str,
        source_experiment: Path,
        favorite_log_path: Optional[Path] = None,
        selection_grid_path: Optional[Path] = None,
        genome_key: Optional[int] = None,
        parent_genome_keys: Optional[Sequence[int]] = None,
        source_entry_ids: Optional[Sequence[str]] = None,
        ancestor_genome_keys: Optional[Sequence[int]] = None,
    ) -> ArchiveEntry:
        """Persist a favourite image and genome into the shared archive."""

        entry_id = f"img_{self._metadata['next_id']:06d}"
        self._metadata["next_id"] += 1

        image_path = self.images_dir / f"{entry_id}.png"
        image_path.write_bytes(image_bytes)

        genome_path = self.genomes_dir / f"{entry_id}.pkl"
        with genome_path.open("wb") as handle:
            pickle.dump(genome, handle, protocol=pickle.HIGHEST_PROTOCOL)

        archive_entry = ArchiveEntry(
            entry_id=entry_id,
            title=title,
            image_path=image_path,
            genome_path=genome_path,
            agent_id=agent_id,
            generation=generation,
            image_index=image_index,
            rationale=rationale,
            source_experiment=source_experiment,
            added_at=datetime.now(),
            metadata_path=favorite_log_path,
            selection_grid_path=selection_grid_path,
            genome_key=genome_key,
            parent_genome_keys=list(parent_genome_keys or []),
            source_entry_ids=[str(value) for value in (source_entry_ids or [])],
            ancestor_genome_keys=list(ancestor_genome_keys or []),
        )

        self._metadata.setdefault("entries", []).append(archive_entry.as_dict())
        self._persist()
        self._write_checkpoint(archive_entry)
        return archive_entry

    def load_genome(self, entry_id: str) -> Optional[neat.DefaultGenome]:
        for entry in self._metadata.get("entries", []):
            if entry.get("id") != entry_id:
                continue
            genome_path = Path(entry["genome_path"])
            if not genome_path.exists():
                return None
            with genome_path.open("rb") as handle:
                return pickle.load(handle)
        return None

    def create_archive_grid(self, thumb_size: int = 200) -> Optional[Path]:
        entries = self.entries
        if not entries:
            return None

        images: List[Image.Image] = []
        captions: List[str] = []
        for index, entry in enumerate(entries):
            path = Path(entry["image_path"])
            if not path.exists():
                continue
            try:
                with Image.open(path) as img:
                    img = img.convert("RGB")
                    img = img.resize((thumb_size, thumb_size), Image.Resampling.LANCZOS)
                    images.append(img)
                    captions.append(f"{index}")
            except Exception:
                continue

        if not images:
            return None

        columns = max(1, math.ceil(math.sqrt(len(images))))
        rows = math.ceil(len(images) / columns)
        margin = ARCHIVE_GRID_MARGIN
        font = _try_load_font(18)

        tile_width, tile_height = images[0].size
        canvas = Image.new(
            "RGB",
            (
                columns * tile_width + (columns + 1) * margin,
                rows * tile_height + (rows + 1) * margin,
            ),
            (18, 18, 22),
        )
        draw = ImageDraw.Draw(canvas)

        for idx, img in enumerate(images):
            col = idx % columns
            row = idx // columns
            x = margin + col * (tile_width + margin)
            y = margin + row * (tile_height + margin)
            canvas.paste(img, (x, y))
            caption = captions[idx]
            draw.text((x + 8, y + 8), caption, font=font, fill=(255, 255, 0))

        output_path = self.archive_dir / "archive_grid.png"
        canvas.save(output_path, format="PNG")
        print(f"Archive grid saved to: {output_path}")
        return output_path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _write_checkpoint(self, entry: ArchiveEntry) -> None:
        snapshot = {
            "written_at": datetime.now().isoformat(),
            "new_entry": entry.as_dict(),
            "entries": self.entries,
        }
        checkpoint_name = f"checkpoint_{entry.entry_id}.json"
        checkpoint_path = self.checkpoints_dir / checkpoint_name
        with checkpoint_path.open("w", encoding="utf-8") as fp:
            json.dump(snapshot, fp, indent=2)

    def remove_entry(self, entry_id: str) -> bool:
        entries = self._metadata.get("entries", [])
        for index, entry in enumerate(entries):
            if entry.get("id") != entry_id:
                continue
            for key in ("image_path", "genome_path"):
                path_value = entry.get(key)
                if not path_value:
                    continue
                try:
                    Path(path_value).unlink(missing_ok=True)
                except OSError:
                    pass
            del entries[index]
            self._metadata["entries"] = entries
            self._persist()
            return True
        return False


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


def _try_load_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "DejaVuSans.ttf",
        "DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def _ensure_int_list(values: Iterable[Any]) -> List[int]:
    result: List[int] = []
    for value in values:
        try:
            idx = int(value)
        except (TypeError, ValueError):
            continue
        result.append(idx)
    return result


def _safe_int(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: int, lower: int, upper: int) -> int:
    return max(lower, min(upper, value))


@dataclass
class GenerationArtifacts:
    state_path: Path
    grid_path: Path
    selection_path: Path
    image_paths: Dict[int, Path] = field(default_factory=dict)
    genome_snapshots: Dict[int, neat.DefaultGenome] = field(default_factory=dict)


class CollaborativeAgentRunner:
    """Encapsulates the per-agent evolution workflow."""

    def __init__(
        self,
        agent_id: str,
        agent_dir: Path,
        config: CollaborativeConfig,
        neat_config: neat.Config,
        archive_manager: ArchiveManager,
        *,
        generations: int = DEFAULT_AGENT_GENERATIONS,
        rows: int,
        cols: int,
        thumb_size: int,
        scheme: str,
        palette: str,
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
    ) -> None:
        self.agent_id = agent_id
        self.agent_dir = agent_dir
        self.config = config
        # Save running latest image in parent of agent directory
        self.latest_img_path = agent_dir.parent / "latest_image.png"
        self.archive_manager = archive_manager
        self.generations = generations
        self.rows = rows
        self.cols = cols
        self.thumb_size = thumb_size
        self.scheme = scheme
        self.palette = palette
        self.select_k = select_k
        self.chat_history_turns = chat_history_turns
        self.selection_baseline = selection_baseline
        self.neat_config = neat_config
        self.progress_callback = progress_callback
        self.warm_start_active = bool(warm_start_active)
        self.resume_mode = resume_mode or (population is not None)

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

        self.prompt_template = PARENT_SELECTION_PROMPT
        if (self.scheme == "color" or self.scheme == "toggle"):
            color_prompt = "It would be nice to have color images in the online archive, but we do not want it to be domainated by high-frequency rainbow artefacts. "
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
                    "At each generation, you may choose to mutate only an isolated subnetwork of the CPPN affecting color or structure, "
                    "or to mutate the entire CPPN. Indicate your choice in a `mutation_mode` field in your JSON response, set to either `color_only`, `structure_only`, or `all`. "
                )
        else:
            mutation_mode_prompt = ""

        self.system_instruction = DEFAULT_SYSTEM_INSTRUCTION.format(
            goal_prompt=GOAL_PROMPTS[self.config.goal],
            selection_prompt=gen_selection_prompt(self.select_k),
            n_generations=self.generations,
            color_prompt=color_prompt,
            archive_novelty_prompt=archive_novelty_prompt,
            mutation_mode_prompt=mutation_mode_prompt,
        )
        if self.agent_id == 0:
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
        self._generation_records: Dict[int, GenerationArtifacts] = {}
        self._selection_history_path = self.logs_dir / "selection_history.jsonl"
        self._selection_history_path.touch(exist_ok=True)
        self.publication_history_path = self.logs_dir / "publication_history.jsonl"
        self.publication_history_path.touch(exist_ok=True)
        self._current_publication_entry_id: Optional[str] = None
        self._lineage_log_path = self.logs_dir / "lineage.jsonl"
        self._lineage_log_path.touch(exist_ok=True)
        self._archive_seed_map: Dict[int, Dict[str, Any]] = {}
        self._genome_lineage: Dict[int, Dict[str, Any]] = {}
        self.render_genome_diagrams = render_genome_diagrams
        self._diagram_warning_emitted = False
        if self.resume_mode:
            self._load_existing_publication_state()

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
        target = "structure_only" if self.warm_start_active else self._normalize_mutation_mode(requested_mode)
        if target != self._mutation_mode:
            self._mutation_mode = target
            self._apply_mutation_mode(self._mutation_mode)
        else:
            # Keep config in sync even if mode unchanged (useful after resume).
            self._apply_mutation_mode(self._mutation_mode)
        return self._mutation_mode

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

    # ------------------------------------------------------------------
    # Population initialisation and branching
    # ------------------------------------------------------------------
    def select_starting_point(self) -> Dict[str, Any]:
        archive_grid = self.archive_manager.create_archive_grid(self.thumb_size)
        archive_entries = self.archive_manager.entries
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
            elite_labels = [f"{idx}: {name}" for idx, name in enumerate(elite_name_list)]
            elite_names = "; ".join(elite_labels) if elite_labels else "none available"
            archive_prompt = ARCHIVE_BRANCHING_PROMPT.format(
                elite_names=elite_names,
            )

            with archive_grid.open("rb") as fp:
                grid_bytes = fp.read()

                response = query_with_history(
                    grid_bytes,
                    prompt=archive_prompt,
                    system_instruction=self.system_instruction,
                    chat_history_turns=self.chat_history_turns,
                )

                response_text = getattr(response, "text", "") or ""
                try:
                    parsed = extract_json_object(response_text)
                except Exception:
                    parsed = {}

                selected_images = parsed.get("selected", [])
                selected_images = [] if selected_images is None else selected_images
                selected_images = _ensure_int_list(selected_images)
                selected_images = [idx for idx in selected_images if 0 <= idx < len(archive_entries)][:1]
                if len(selected_images) > 1:
                    breakpoint()
                rationale = str(parsed.get("rationale", ""))
                choice = "branch" if selected_images else "fresh"
                decision = {
                    "choice": choice,
                    "selected_images": selected_images,
                    "rationale": rationale,
                    "raw_response": response_text,
                    "timestamp": datetime.now().isoformat(),
                    "archive_grid_path": str(archive_grid),
                    "archive_elite_names": elite_name_list,
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
            f"[{self.agent_id}] Branching decision: choice={choice}, selected={selected_images}, rationale='{rationale}'"
        )
        return decision

    def initialise_population(self, decision: Dict[str, Any]) -> None:
        self._archive_seed_map.clear()
        self._genome_lineage.clear()

        if decision.get("choice") != "branch" or not decision.get("selected_images"):
            seed_initial_population(self.population, self.neat_config.genome_config)
            self._enforce_structure_only_population()
            return

        archive_entries = self.archive_manager.entries
        selected_indices = decision.get("selected_images", [])
        selected_records: List[Tuple[Dict[str, Any], neat.DefaultGenome]] = []
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
            self._enforce_structure_only_population()
            return

        population_keys = list(self.population.population.keys())
        random.shuffle(population_keys)
        self.population.population.clear()

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
        self._write_branching_summary(decision, len(selected_records))

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

        state, cache = build_generation_state(
            genomes,
            config,
            generation,
            self.rows,
            self.cols,
            self.thumb_size,
            self.scheme,
            self.palette,
        )

        state_path = save_neat_population(state, self.population_dir, generation, cache)
        grid_image = create_numbered_grid(state)
        grid_path = self.query_dir / f"gen_{generation:03d}_grid.png"
        grid_image.save(grid_path, format="PNG")
        shutil.copy(grid_path, self.latest_img_path)

        system_instruction = self.system_instruction
        prompt_template = self.prompt_template
        require_selection = True
        if generation == self.generations - 1:
            require_selection = False
            require_publish = self.favorite_archive_entry is None
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

        if self.selection_baseline == "none":
            selection_meta_raw = select_parents_from_grid(
                state,
                prompt_template,
                self.query_dir,
                self.select_k,
                system_instruction,
                self.chat_history_turns,
                require_selection=require_selection,
            )
        else:
            selection_meta_raw = self._select_parents_baseline(
                generation,
                genomes,
                config,
                state,
            )
        selection_meta = dict(selection_meta_raw)
        resolved_mode = self._update_mutation_mode(selection_meta.get("mutation_mode"))
        selection_meta["mutation_mode"] = resolved_mode
        if self.warm_start_active:
            selection_meta["mutation_mode_forced"] = True
        selection_path = Path(selection_meta.get("selection_path", grid_path))
        shutil.copy(selection_path, self.latest_img_path)
        selected_indices: Sequence[int] = selection_meta["selected"]
        publish_details: Optional[Dict[str, Any]] = None
        for idx, (_, genome) in enumerate(genomes):
            genome.fitness = 1.0 if idx in selected_indices else 0.0

        images_dir = self.images_dir / f"gen_{generation:03d}"
        images_dir.mkdir(parents=True, exist_ok=True)
        image_paths: Dict[int, Path] = {}
        genome_snapshots: Dict[int, neat.DefaultGenome] = {}
        for idx, (_, genome) in enumerate(genomes):
            png_bytes = cache[idx]
            image_path = images_dir / f"idx_{idx:02d}.png"
            image_path.write_bytes(png_bytes)
            image_paths[idx] = image_path
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

        publication_committed = False
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
                entry = self._apply_publication(
                    favorite,
                    forced=False,
                    response_text=selection_meta.get("response_text"),
                    source="vlm",
                    replaced_entry_id=previous_entry_id,
                )
                publication_committed = entry is not None
                selection_meta["publish"] = {
                    "index": publish_index,
                    "reason": favorite_reason,
                }
                if publication_committed:
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
            not publication_committed
            and generation == self.generations - 1
            and self.favorite_archive_entry is None
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
            entry = self._apply_publication(
                favorite,
                forced=True,
                response_text=selection_meta.get("response_text"),
                source="forced",
                replaced_entry_id=previous_entry_id,
            )
            if entry is not None:
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
        record.selection_path = selection_path

        self._print_selection_response(
            generation,
            selected_indices,
            selection_meta.get("rationale", ""),
            selection_meta.get("mutation_mode"),
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

        image_path = record.image_paths.get(index)
        genome = record.genome_snapshots.get(index)
        if image_path is None or genome is None:
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
    ) -> Optional[ArchiveEntry]:
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

        if replaced_entry_id is not None:
            self.archive_manager.remove_entry(replaced_entry_id)
            self.favorite_archive_entry = None
            self._current_publication_entry_id = None

        entry = self.publish_to_archive(payload)
        if entry is None:
            return None

        payload["archive_entry_id"] = entry.entry_id
        if genome_key is not None:
            self._archive_seed_map[genome_key] = {
                "entry_id": entry.entry_id,
                "agent_id": self.agent_id,
                "generation": generation_int,
            }
        self._append_publication_history(payload)

        self.favorite_archive_entry = entry
        self.favorite_decision = payload
        self._current_publication_entry_id = entry.entry_id
        print(
            f"[{self.agent_id}] Publication committed: generation={generation_int}, index={index_int}, title='{title}', forced={payload.get('forced')}"
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
            shutil.copy(output_path, self.latest_img_path)

        return output_path

    def _print_selection_response(
        self,
        generation: int,
        selected_indices: Sequence[int],
        rationale: str,
        mutation_mode: Optional[str],
        publish_details: Optional[Dict[str, Any]],
    ) -> None:
        publish_index = publish_details.get("index") if publish_details else None
        publish_title = publish_details.get("title") if publish_details else None
        publish_rationale = publish_details.get("reason") if publish_details else None
        log_str = f"[{self.agent_id}] Gen {generation} selection: selected={list(selected_indices)}, rationale='{rationale}'"
        if mutation_mode is not None:
            log_str += f", mutation_mode='{mutation_mode}'"
        if publish_index is not None:
            log_str += f", publish_index={publish_index}, publish_title='{publish_title}', publish_rationale='{publish_rationale}'"
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

        grid_path = self.query_dir / f"gen_{generation:03d}_grid.png"
        selection_path = self.query_dir / f"gen_{generation:03d}_selection.png"
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
            }
        )

        metadata_dir = self.query_dir / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        meta_path = metadata_dir / f"gen_{generation:03d}_selection.json"
        meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        payload["metadata_path"] = str(meta_path)

        return payload


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
        palette: str,
        config_path: Path,
        select_k: Optional[int],
        agent_generations: int,
        warm_start_structure: int,
        enable_output_activations: bool,
        selection_baseline: str,
        seed: Optional[int],
        chat_history_turns: int,
        render_genome_diagrams: bool = False,
    ) -> None:
        self.config = config
        self.experiment_dir = experiment_dir
        self.chat_history_turns = chat_history_turns
        self.rows = rows
        self.cols = cols
        self.thumb_size = thumb_size
        self.scheme = scheme
        self.palette = palette
        self.config_path = config_path
        self.select_k = select_k
        self.agent_generations = agent_generations
        self.warm_start_structure = warm_start_structure
        self.enable_output_activations = enable_output_activations
        self.selection_baseline = selection_baseline
        self.seed = seed
        self.render_genome_diagrams = render_genome_diagrams

        self.archive_manager = ArchiveManager(self.experiment_dir / ARCHIVE_DIR_NAME)
        self.agents_dir = self.experiment_dir / "agents"
        self.agents_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_path = self.experiment_dir / "agents_metadata.json"
        self._metadata = self._load_metadata()
        self._ensure_run_config()

    def _load_metadata(self) -> Dict[str, Any]:
        if self.metadata_path.exists():
            with self.metadata_path.open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)
            metadata.setdefault("agents", [])
            metadata.setdefault("next_agent_number", 0)
            metadata.setdefault("run_config", None)
            changed = False
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
            if changed:
                self._persist_metadata(metadata)
            return metadata
        metadata = {
            "created_at": datetime.now().isoformat(),
            "next_agent_number": 0,
            "agents": [],
            "seed": self.seed,
            "run_config": None,
        }
        self._persist_metadata(metadata)
        return metadata

    def _ensure_run_config(self) -> None:
        run_config = {
            "rows": self.rows,
            "cols": self.cols,
            "thumb_size": self.thumb_size,
            "scheme": self.scheme,
            "palette": self.palette,
            "select_k": self.select_k,
            "agent_generations": self.agent_generations,
            "enable_output_activations": self.enable_output_activations,
            "warm_start_structure": self.warm_start_structure,
            "selection_baseline": self.selection_baseline,
        }
        existing = self._metadata.get("run_config")
        if existing is None:
            self._metadata["run_config"] = run_config
            self._persist_metadata(self._metadata)
            return
        if "warm_start_structure" not in existing:
            existing["warm_start_structure"] = 0
            self._persist_metadata(self._metadata)
        if "selection_baseline" not in existing:
            existing["selection_baseline"] = "none"
            self._persist_metadata(self._metadata)
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

    def _persist_metadata(self, metadata: Dict[str, Any]) -> None:
        with self.metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)

    def _allocate_agent_id(self) -> str:
        agent_number = self._metadata.get("next_agent_number", 0)
        agent_id = f"{AGENT_DIR_PREFIX}{agent_number:03d}"
        self._metadata["next_agent_number"] = agent_number + 1
        self._persist_metadata(self._metadata)
        return agent_id

    def _parse_agent_index(self, agent_id: str) -> Optional[int]:
        if not agent_id.startswith(AGENT_DIR_PREFIX):
            return None
        suffix = agent_id[len(AGENT_DIR_PREFIX) :]
        try:
            return int(suffix)
        except ValueError:
            return None

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
        self._metadata.setdefault("agents", []).append(record)
        self._persist_metadata(self._metadata)
        return record

    def _update_agent_record(self, agent_id: str, **updates: Any) -> None:
        record = self._find_agent_record(agent_id)
        if record is None:
            return
        record.update(updates)
        self._persist_metadata(self._metadata)

    def _count_finished_agents(self) -> int:
        finished = {"complete", "extinct"}
        return sum(1 for record in self._metadata.get("agents", []) if record.get("status") in finished)

    def _find_agent_to_resume(self, resume_agent_id: Optional[str]) -> Optional[Dict[str, Any]]:
        agents = self._metadata.get("agents", [])
        if resume_agent_id:
            record = next((entry for entry in agents if entry.get("agent_id") == resume_agent_id), None)
            if record is None:
                raise ValueError(f"Agent '{resume_agent_id}' not found in experiment metadata.")
            if record.get("status") in {"complete", "extinct"}:
                raise ValueError(f"Agent '{resume_agent_id}' has already completed.")
            return record
        for record in reversed(agents):
            if record.get("status") not in {"complete", "extinct"}:
                return record
        return None

    def _hydrate_agent_record_from_disk(self, record: Dict[str, Any]) -> None:
        agent_dir = Path(record["agent_dir"])
        logs_dir = agent_dir / "logs"
        changed = False
        branch_path = logs_dir / "branching_selection.json"
        if record.get("branching_decision") is None and branch_path.exists():
            try:
                record["branching_decision"] = json.loads(branch_path.read_text(encoding="utf-8"))
                changed = True
            except json.JSONDecodeError:
                pass
        favourite_path = logs_dir / "favorite_selection.json"
        if record.get("favorite_selection") is None and favourite_path.exists():
            try:
                record["favorite_selection"] = json.loads(favourite_path.read_text(encoding="utf-8"))
                changed = True
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
                    changed = True
        if changed:
            self._persist_metadata(self._metadata)

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
        config = neat.Config(
            PicbreederGenome,
            PicbreederReproduction,
            neat.DefaultSpeciesSet,
            InteractiveStagnation,
            str(self.config_path),
        )
        apply_picbreeder_config_defaults(
            config,
            enable_output_activations=self.enable_output_activations,
        )
        config.pop_size = self.rows * self.cols
        return config

    def _build_runner(
        self,
        agent_id: str,
        agent_dir: Path,
        neat_config: neat.Config,
        population: Optional[neat.Population],
        resume: bool,
    ) -> CollaborativeAgentRunner:
        callback = lambda generation, favorite, archive: self._on_generation_progress(
            agent_id, generation, favorite, archive
        )
        return CollaborativeAgentRunner(
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
            palette=self.palette,
            select_k=self.select_k,
            chat_history_turns=self.chat_history_turns,
            selection_baseline=self.selection_baseline,
            population=population,
            progress_callback=callback,
            resume_mode=resume,
            warm_start_active=self._is_warm_start_agent(agent_id),
            render_genome_diagrams=self.render_genome_diagrams,
        )

    def _on_generation_progress(
        self,
        agent_id: str,
        generation: int,
        favorite_payload: Optional[Dict[str, Any]],
        archive_payload: Optional[Dict[str, Any]],
    ) -> None:
        record = self._find_agent_record(agent_id)
        if record is None:
            return
        record["last_generation"] = generation
        if favorite_payload is not None:
            record["favorite_selection"] = favorite_payload
        if archive_payload is not None:
            record["archive_entry"] = archive_payload
        self._persist_metadata(self._metadata)

    def _execute_runner(
        self,
        agent_id: str,
        runner: CollaborativeAgentRunner,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[ArchiveEntry], bool, int]:
        remaining = max(0, runner.generations - runner.population.generation)
        extinct = False
        if remaining > 0:
            try:
                runner.population.run(runner.evaluate_generation, remaining)
            except CompleteExtinctionException:
                extinct = True
        favorite = runner.favorite_decision if runner.favorite_decision else None
        archive_entry = runner.favorite_archive_entry
        final_generation = runner.population.generation
        return favorite, archive_entry, extinct, final_generation

    def run_agents(self, total_agents: int, resume: bool, resume_agent_id: Optional[str]) -> None:
        resumed = False
        if resume:
            resumed = self._resume_agent(resume_agent_id)
        if resume and not resumed and resume_agent_id:
            # _resume_agent raises for unknown agent so this only occurs when no pending agents
            print(f"No in-progress agent found for resume request '{resume_agent_id}'.")
        while self._count_finished_agents() < total_agents:
            self._run_new_agent()

    def _run_new_agent(self) -> None:
        agent_id = self._allocate_agent_id()
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

    def _resume_agent(self, resume_agent_id: Optional[str]) -> bool:
        record = self._find_agent_to_resume(resume_agent_id)
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
        record = self._find_agent_record(agent_id)
        if record is None:
            return
        if branching_decision is not None:
            record["branching_decision"] = branching_decision
        if favourite is not None:
            record["favorite_selection"] = favourite
        if archive_entry is not None:
            record["archive_entry"] = archive_entry.as_dict()
        record["extinct"] = extinct
        record["status"] = "extinct" if extinct else "complete"
        record["completed_at"] = datetime.now().isoformat()
        record["last_generation"] = final_generation
        self._persist_metadata(self._metadata)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


@dataclass
class CollaborativeConfig:
    goal: str = "familiar_objects"
    rows: int = 4  # Rows in the CPPN grid
    cols: int = 5  # Columns in the CPPN grid
    thumb_size: int = 200  # Pixel size for rendered genome thumbnails
    chat_history_turns: int = DEFAULT_CHAT_HISTORY_TURNS  # How many prior turns each agent sees (-1 keeps all)
    scheme: str = "gray"  # Rendering scheme: color, gray, or mono
    color_palette: str = "hsb"  # Palette choice when using color or gray rendering
    config_path: Optional[Path] = None  # Optional override for the NEAT config file
    select_k: Optional[int] = None  # Max parents per generation (clamped to grid size when provided)
    agent_generations: int = DEFAULT_AGENT_GENERATIONS  # Generations executed for each agent
    num_agents: int = 100  # How many agents run sequentially in this session
    warm_start_structure: int = 0  # Number of initial agents restricted to structure-only mutation
    experiment_dir: Optional[Path] = None  # Output directory for logs and artefacts
    output_activations: bool = False  # Enable CPPN output activation mutations
    selection_baseline: str = "none"  # Parent-selection policy: none/random/max-depth/max-nodes
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
    config_name = "interactive_config_color" if (cfg.scheme == "color" or cfg.scheme == "toggle") else "interactive_config_gray"
    return base / config_name


def _ensure_absolute(path: Path, base: Path) -> Path:
    expanded = Path(path).expanduser()
    if expanded.is_absolute():
        return expanded.resolve()
    return (base / expanded).resolve()


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
            if cfg.scheme != "gray":
                experiment_name += f"_scheme-{cfg.scheme}"
            if cfg.warm_start_structure > 0:
                experiment_name += f"_warmstart{cfg.warm_start_structure}"
            if cfg.selection_baseline != "none":
                experiment_name += f"_baseline-{cfg.selection_baseline}"
            experiment_name += f"_{timestamp}"
            relative = Path("logs_collaborative") / experiment_name
            exp_dir = _ensure_absolute(relative, original_cwd)
        else:
            exp_dir = _ensure_absolute(Path(cfg.experiment_dir), original_cwd)
        exp_dir.mkdir(parents=True, exist_ok=True)
    cfg.experiment_dir = exp_dir

    if cfg.select_k is not None:
        max_possible = cfg.rows * cfg.cols
        cfg.select_k = min(max_possible, cap_select_k_for_engine("neat", cfg.select_k))

    if cfg.test_mode and not cfg.resume:
        cfg.agent_generations = min(3, cfg.agent_generations)
        cfg.num_agents = min(2, cfg.num_agents)

    return cfg


def run(cfg: CollaborativeConfig) -> None:
    apply_random_seed(cfg.seed)
    if cfg.selection_baseline == "none":
        ensure_gemini_key()
    orchestrator = CollaborativeMultiAgentOrchestrator(
        config=cfg,
        experiment_dir=cfg.experiment_dir,
        rows=cfg.rows,
        cols=cfg.cols,
        thumb_size=cfg.thumb_size,
        scheme=cfg.scheme,
        palette=cfg.color_palette,
        config_path=cfg.config_path,
        select_k=cfg.select_k,
        agent_generations=cfg.agent_generations,
        warm_start_structure=cfg.warm_start_structure,
        enable_output_activations=cfg.output_activations,
        selection_baseline=cfg.selection_baseline,
        seed=cfg.seed,
        chat_history_turns=cfg.chat_history_turns,
        render_genome_diagrams=cfg.render_genome_diagrams,
    )

    orchestrator.run_agents(
        cfg.num_agents,
        resume=cfg.resume,
        resume_agent_id=cfg.resume_agent_id,
    )
    orchestrator.archive_manager.create_archive_grid(cfg.thumb_size)


cs = ConfigStore.instance()
cs.store(name="collaborative_base", node=CollaborativeConfig)


@hydra.main(version_base=None, config_path=None, config_name="collaborative_base")
def main(cfg: CollaborativeConfig) -> None:
    original_cwd = Path(get_original_cwd())
    cfg = ensure_valid_config(cfg, original_cwd=original_cwd)
    run(cfg)


if __name__ == "__main__":
    main()
