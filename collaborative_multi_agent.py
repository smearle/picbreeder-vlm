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

import argparse
import copy
import gzip
import json
import math
import pickle
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import neat
from neat.population import CompleteExtinctionException
from PIL import Image, ImageDraw, ImageFont

from auto_evolve import (
    query_with_history,
    extract_json_object,
    ensure_gemini_key,
    gen_selection_prompt,
    reset_chat_session,
    select_parents_from_grid,
)
from artifacts import build_generation_state, save_neat_population
from experiment_cli import cap_select_k_for_engine
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

GOAL_PROMPT = (
    "Your goal is to evolve images that resemble familiar real-world objects."
    # "Your goal is to evolve images that resemble familiar real-world objects. We want the object to be colored, but try to move away from the high-frequency rainbow artefact. "
    # "Your goal is to evolve an image that looks like a fish."
    # "Your goal is to generate a lizard."
)

DEFAULT_SYSTEM_INSTRUCTION = (
    "You are playing with a collaborative online platform which allows users to interactively evolve small neural networks called Compositional Pattern Producing Networks (CPPNs) for generating images. "
    f"{GOAL_PROMPT} "
    """At the first generation the initial grid will display an archive of images published by prior users as favorites (unless you are the first user). You may choose to "branch" one of these images, or start instead from a random initial population. """
    "At each subsequent generation, you will be shown a grid of numbered images produced by CPPNs. "
    "{selection_prompt}"
    "If so inclined, you may also select an image to publish to the online archive . "
    "You must publish at least once during your session. Publishing multiple times is allowed, though the most recent publication replaces any prior favorite. "
    "Your session will run for {n_generations} generations. "
    "Try to contribute something novel, interesting or useful to the online archive. {color_prompt}"
    "Respond with JSON only: {{\"selected\": [indices], \"rationale\": \"brief explanation\"}}. "
    """(During branching, set `selected` to null to start from a fresh population.) """
    "You may also include a \"publish\" field in the JSON response if you wish to publish an image from this grid. It should have the form: "
    '{{"index": image_index, "title": "image title", "reason": "brief publication note"}}. '
    # "(Also, for debugging, please tell me how many previous grids you see in the chat history, briefly describe in neutral, objective terms how the grids have changed over time, "
    # "and tell me if you see the archive from which you made your original branching decision; add this to the `rationale` text.) "
)

def gen_selection_prompt(select_k: Optional[int]) -> str:
    if select_k is None:
        return "Pick one or several images by their numeric labels--the corresponding CPPNs will be used as the parents of the next generation. "
    if select_k == 1:
        return "Pick one image by its numeric label--the corresponding CPPN will be used as the parent of the next generation. "
    return f"Pick up to {select_k} images by their numeric labels--the corresponding CPPNs will be used as the parents of the next generation. "



DEFAULT_PROMPT = "Grid at generation {generation}."

ARCHIVE_SELECTION_PROMPT = (
    "Above is the archive of images published by prior users. You may choose to branch from one of them, or start from a fresh population. "
)

REPO_ROOT = Path(__file__).resolve().parent


@dataclass
class ArchiveEntry:
    """Structured information recorded for each archived favourite."""

    entry_id: str
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

    def as_dict(self) -> Dict[str, Any]:
        return {
            "id": self.entry_id,
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
        return cls(
            entry_id=payload["id"],
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
        source_experiment: Path,
        favorite_log_path: Optional[Path] = None,
        selection_grid_path: Optional[Path] = None,
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
                    captions.append(f"{index}\n{entry['agent_id']}")
            except Exception:
                continue

        if not images:
            return None

        columns = max(1, int(math.sqrt(len(images))))
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
        config: neat.Config,
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
        dry_run: bool,
        population: Optional[neat.Population] = None,
        progress_callback: Optional[
            Callable[[int, Optional[Dict[str, Any]], Optional[Dict[str, Any]]], None]
        ] = None,
        resume_mode: bool = False,
    ) -> None:
        self.agent_id = agent_id
        self.agent_dir = agent_dir
        self.archive_manager = archive_manager
        self.generations = generations
        self.rows = rows
        self.cols = cols
        self.thumb_size = thumb_size
        self.scheme = scheme
        self.palette = palette
        self.select_k = select_k
        self.chat_history_turns = chat_history_turns
        self.dry_run = dry_run
        self.config = config
        self.progress_callback = progress_callback
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
            self.population = neat.Population(self.config)
        else:
            self.population = population
        sync_population_output_activations(self.population, self.config.genome_config)
        self.population.add_reporter(GenerationCheckpointer(self.population_dir))

        self.prompt_template = DEFAULT_PROMPT
        if self.scheme == "color":
            color_prompt = "It would be nice to have color images in the online archive, but we do not want it to be domainated by high-frequency rainbow artefacts. "
        else:
            color_prompt = ""

        self.system_instruction = DEFAULT_SYSTEM_INSTRUCTION.format(
            selection_prompt=gen_selection_prompt(self.select_k),
            n_generations=self.generations,
            color_prompt=color_prompt,
        )

        self.branching_decision: Dict[str, Any] = {}
        self.favorite_decision: Dict[str, Any] = {}
        self.favorite_archive_entry: Optional[ArchiveEntry] = None
        self._generation_records: Dict[int, GenerationArtifacts] = {}
        self._selection_history_path = self.logs_dir / "selection_history.jsonl"
        self._selection_history_path.touch(exist_ok=True)
        self.publication_history_path = self.logs_dir / "publication_history.jsonl"
        self.publication_history_path.touch(exist_ok=True)
        self._current_publication_entry_id: Optional[str] = None
        if self.resume_mode:
            self._load_existing_publication_state()

    # ------------------------------------------------------------------
    # Population initialisation and branching
    # ------------------------------------------------------------------
    def select_starting_point(self) -> Dict[str, Any]:
        archive_grid = self.archive_manager.create_archive_grid(self.thumb_size)
        archive_entries = self.archive_manager.entries
        if not archive_entries:
            decision = {
                "choice": "fresh",
                "selected_images": [],
                "rationale": "Archive empty; defaulting to fresh population.",
                "raw_response": None,
                "timestamp": datetime.now().isoformat(),
            }
            self._write_branching_log(decision)
            self._print_branching_decision(decision)
            return decision

        if self.dry_run:
            decision = {
                "choice": "fresh" if random.random() < 0.5 else "branch",
                "selected_images": [],
                "rationale": "Dry-run random decision.",
                "raw_response": None,
                "timestamp": datetime.now().isoformat(),
            }
            if decision["choice"] == "branch":
                decision["selected_images"] = [random.randrange(len(archive_entries))]
            self._write_branching_log(decision)
            self._print_branching_decision(decision)
            return decision

        if archive_grid is None:
            decision = {
                "choice": "fresh",
                "selected_images": [],
                "rationale": "Archive grid unavailable; falling back to fresh population.",
                "raw_response": None,
                "timestamp": datetime.now().isoformat(),
            }
            self._write_branching_log(decision)
            self._print_branching_decision(decision)
            return decision

        with archive_grid.open("rb") as fp:
            grid_bytes = fp.read()

        response = query_with_history(
            grid_bytes,
            prompt=ARCHIVE_SELECTION_PROMPT,
            system_instruction=self.system_instruction,
            chat_history_turns=self.chat_history_turns,
        )

        response_text = getattr(response, "text", "") or ""
        try:
            parsed = extract_json_object(response_text)
        except Exception:
            parsed = {}

        choice = parsed.get("choice", "fresh")
        selected_images = _ensure_int_list(parsed.get("selected_images", []))
        selected_images = [idx for idx in selected_images if 0 <= idx < len(archive_entries)]
        decision = {
            "choice": "branch" if choice == "branch" and selected_images else "fresh",
            "selected_images": selected_images,
            "rationale": parsed.get("rationale", ""),
            "raw_response": response_text,
            "timestamp": datetime.now().isoformat(),
            "archive_grid_path": str(archive_grid),
        }
        preview_path = self._save_archive_branch_preview(decision, archive_entries)
        if preview_path is not None:
            decision["branch_preview_path"] = str(preview_path)
        self._write_branching_log(decision)
        self._print_branching_decision(decision)
        return decision

    def initialise_population(self, decision: Dict[str, Any]) -> None:
        if decision.get("choice") != "branch" or not decision.get("selected_images"):
            seed_initial_population(self.population, self.config.genome_config)
            return

        archive_entries = self.archive_manager.entries
        selected_indices = decision.get("selected_images", [])
        genomes: List[neat.DefaultGenome] = []
        for idx in selected_indices:
            entry = archive_entries[idx]
            genome = self.archive_manager.load_genome(entry["id"])
            if genome is None:
                continue
            genomes.append(genome)

        if not genomes:
            seed_initial_population(self.population, self.config.genome_config)
            return

        population_keys = list(self.population.population.keys())
        random.shuffle(population_keys)
        self.population.population.clear()

        for key, genome in zip(population_keys, genomes):
            clone = copy.deepcopy(genome)
            clone.key = key
            clone.fitness = None
            self.population.population[key] = clone

        sync_population_output_activations(self.population, self.config.genome_config)
        self.population.species.speciate(
            self.config,
            self.population.population,
            self.population.generation,
        )
        self.population.population = self.population.reproduction.reproduce(
            self.population.config, self.population.species, self.population.config.pop_size,
            self.population.generation)

        self._write_branching_summary(decision, len(genomes))

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

        require_publish = self.favorite_archive_entry is None and generation == self.generations - 1
        system_instruction = self.system_instruction
        prompt_template = self.prompt_template
        if require_publish:
            prompt_template += (
                " You have not published any favorite yet and this is the final generation; "
                "you must include a publish object selecting exactly one image to share. "
                "If you omit it, one of your selected parents will be published automatically."
            )

        selection_meta = select_parents_from_grid(
            state,
            prompt_template,
            self.query_dir,
            self.select_k,
            system_instruction,
            self.chat_history_turns,
        )
        selection_meta = dict(selection_meta)
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

        selection_path = Path(selection_meta.get("selection_path", grid_path))
        record = GenerationArtifacts(
            state_path=state_path,
            grid_path=grid_path,
            selection_path=selection_path,
            image_paths=image_paths,
            genome_snapshots=genome_snapshots,
        )
        print(f"Saved selection grid to {selection_path}")
        self._generation_records[generation] = record

        publish_payload = None
        if not self.dry_run:
            publish_payload = self._parse_publish_payload(selection_meta.get("response_text", ""))
        else:
            if self.favorite_archive_entry is None and random.random() < 0.25:
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
            source_experiment=self.agent_dir,
            favorite_log_path=log_path,
            selection_grid_path=record.grid_path,
        )
        return entry

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------
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
                or payload.get("explanation")
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
            "rationale": str(rationale).strip(),
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
        generation = favorite.get("generation")
        index = favorite.get("index")
        title = favorite.get("title", "")
        if generation is None or index is None:
            return None
        record = self._generation_records.get(int(generation))
        if record is None or int(index) not in record.image_paths:
            return None

        payload = dict(favorite)
        payload["generation"] = int(generation)
        payload["index"] = int(index)
        payload.setdefault("rationale", "")
        payload["timestamp"] = datetime.now().isoformat()
        payload["forced"] = forced
        payload["source"] = source
        if response_text is not None:
            payload["response_text"] = response_text
        if replaced_entry_id is not None:
            payload["replaced_entry_id"] = replaced_entry_id

        self._write_favorite_log(payload)

        if replaced_entry_id is not None:
            self.archive_manager.remove_entry(replaced_entry_id)
            self.favorite_archive_entry = None
            self._current_publication_entry_id = None

        entry = self.publish_to_archive(payload)
        if entry is None:
            return None

        payload["archive_entry_id"] = entry.entry_id
        self._append_publication_history(payload)

        self.favorite_archive_entry = entry
        self.favorite_decision = payload
        self._current_publication_entry_id = entry.entry_id
        print(
            f"[{self.agent_id}] Publication committed: generation={generation}, index={index}, title='{title}', forced={payload.get('forced')}"
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

    def _build_system_instruction(self, *, require_publish: bool) -> str:
        instruction = self.base_system_instruction
        return instruction

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

    def _print_branching_decision(self, decision: Dict[str, Any]) -> None:
        selected = decision.get("selected_images", [])
        rationale = decision.get("rationale", "")
        choice = decision.get("choice", "fresh")
        print(
            f"[{self.agent_id}] Branching decision: choice={choice}, selected={selected}, rationale='{rationale}'"
        )

    def _save_archive_branch_preview(
        self,
        decision: Dict[str, Any],
        archive_entries: List[Dict[str, Any]],
    ) -> Optional[Path]:
        if decision.get("choice") != "branch":
            return None
        selected = _ensure_int_list(decision.get("selected_images", []))
        if not selected:
            return None
        grid_path_str = decision.get("archive_grid_path")
        if not grid_path_str:
            return None
        grid_path = Path(grid_path_str)
        if not grid_path.exists():
            return None

        total_entries = len(archive_entries)
        if total_entries == 0:
            return None

        valid_indices: List[int] = []
        for entry_index, entry in enumerate(archive_entries):
            path_value = entry.get("image_path")
            if not path_value:
                continue
            if Path(path_value).exists():
                valid_indices.append(entry_index)

        if not valid_indices:
            return None

        columns = max(1, int(math.sqrt(len(valid_indices))))
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
                    continue
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

        return output_path

    def _print_selection_response(
        self,
        generation: int,
        selected_indices: Sequence[int],
        rationale: str,
        publish_details: Optional[Dict[str, Any]],
    ) -> None:
        publish_index = publish_details.get("index") if publish_details else None
        publish_title = publish_details.get("title") if publish_details else None
        publish_rationale = publish_details.get("rationale") if publish_details else None
        print((
            f"[{self.agent_id}] Gen {generation} selection: selected={list(selected_indices)}, rationale='{rationale}',"
            f" publish_index={publish_index}, publish_title='{publish_title}', publish_rationale='{publish_rationale}'"
        ))


class CollaborativeMultiAgentOrchestrator:
    """Coordinates sequential agent runs and archive management."""

    def __init__(
        self,
        experiment_dir: Path,
        rows: int,
        cols: int,
        thumb_size: int,
        scheme: str,
        palette: str,
        config_path: Path,
        select_k: Optional[int],
        agent_generations: int,
        enable_output_activations: bool,
        dry_run: bool,
        seed: Optional[int],
    ) -> None:
        self.experiment_dir = experiment_dir
        self.rows = rows
        self.cols = cols
        self.thumb_size = thumb_size
        self.scheme = scheme
        self.palette = palette
        self.config_path = config_path
        self.select_k = select_k
        self.agent_generations = agent_generations
        self.enable_output_activations = enable_output_activations
        self.dry_run = dry_run
        self.seed = seed

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
            metadata.setdefault("next_agent_number", 1)
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
            "next_agent_number": 1,
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
        }
        existing = self._metadata.get("run_config")
        if existing is None:
            self._metadata["run_config"] = run_config
            self._persist_metadata(self._metadata)
            return
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
        agent_number = self._metadata.get("next_agent_number", 1)
        agent_id = f"{AGENT_DIR_PREFIX}{agent_number:03d}"
        self._metadata["next_agent_number"] = agent_number + 1
        self._persist_metadata(self._metadata)
        return agent_id

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
        config: neat.Config,
        population: Optional[neat.Population],
        resume: bool,
    ) -> CollaborativeAgentRunner:
        callback = lambda generation, favorite, archive: self._on_generation_progress(
            agent_id, generation, favorite, archive
        )
        return CollaborativeAgentRunner(
            agent_id,
            agent_dir,
            config,
            self.archive_manager,
            generations=self.agent_generations,
            rows=self.rows,
            cols=self.cols,
            thumb_size=self.thumb_size,
            scheme=self.scheme,
            palette=self.palette,
            select_k=self.select_k,
            chat_history_turns=DEFAULT_CHAT_HISTORY_TURNS,
            dry_run=self.dry_run,
            population=population,
            progress_callback=callback,
            resume_mode=resume,
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

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collaborative multi-agent Picbreeder workflow with shared archive",
    )
    parser.add_argument("--rows", type=int, default=4, help="Rows in the CPPN grid")
    parser.add_argument("--cols", type=int, default=5, help="Columns in the CPPN grid")
    parser.add_argument(
        "--thumb-size",
        type=int,
        default=200,
        help="Thumbnail size for rendered genomes",
    )
    parser.add_argument(
        "--scheme",
        choices=("color", "gray", "mono"),
        default="gray",
        help="Rendering scheme",
    )
    parser.add_argument(
        "--color-palette",
        choices=("hsb", "sigmoid"),
        default="hsb",
        help="Colour palette for color/gray schemes",
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=None,
        help="Path to NEAT configuration file",
    )
    parser.add_argument(
        "--select-k",
        type=int,
        default=None,
        help="Maximum number of parents to select each generation",
    )
    parser.add_argument(
        "--agent-generations",
        type=int,
        default=DEFAULT_AGENT_GENERATIONS,
        help="Number of generations per agent run",
    )
    parser.add_argument(
        "--num-agents",
        type=int,
        default=20,
        help="Number of agents to execute sequentially",
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=None,
        help="Directory to store experiment artefacts",
    )
    parser.add_argument(
        "--output-activations",
        action="store_true",
        help="Enable CPPN output activation mutations",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip VLM calls; randomise branch/favourite decisions",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an interrupted experiment in the provided experiment directory",
    )
    parser.add_argument(
        "--resume-agent-id",
        type=str,
        default=None,
        help="Explicit agent identifier to resume (defaults to the newest in-progress agent)",
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Use short runs for quick validation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for deterministic runs",
    )
    return parser.parse_args()


def resolve_config_path(args: argparse.Namespace) -> Path:
    if args.config_path is not None:
        return args.config_path.resolve()
    base = REPO_ROOT / "picture2d"
    config_name = "interactive_config_color" if args.scheme == "color" else "interactive_config_gray"
    return (base / config_name).resolve()


def ensure_valid_args(args: argparse.Namespace) -> None:
    if args.resume_agent_id and not args.resume:
        raise ValueError("--resume-agent-id requires --resume")
    if args.rows < 1 or args.cols < 1:
        raise ValueError("rows and cols must be positive integers")
    if args.thumb_size < 8:
        raise ValueError("thumb-size must be at least 8")
    if args.agent_generations < 1:
        raise ValueError("agent-generations must be at least 1")
    if args.num_agents < 1:
        raise ValueError("num-agents must be at least 1")
    if args.select_k is not None and args.select_k < 1:
        raise ValueError("select-k must be at least 1 when provided")
    args.config_path = resolve_config_path(args)
    if not args.config_path.exists():
        raise FileNotFoundError(f"Config file not found at {args.config_path}")

    if args.resume:
        if args.experiment_dir is None:
            raise ValueError("--resume requires --experiment-dir pointing to an existing directory")
        args.experiment_dir = args.experiment_dir.resolve()
        if not args.experiment_dir.exists():
            raise FileNotFoundError(f"Experiment directory not found at {args.experiment_dir}")
    else:
        if args.experiment_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            seed_suffix = f"_seed_{args.seed}" if args.seed is not None else ""
            args.experiment_dir = Path("logs_collaborative") / f"collaborative_multi_agent_{timestamp}{seed_suffix}"
        args.experiment_dir = args.experiment_dir.resolve()
        args.experiment_dir.mkdir(parents=True, exist_ok=True)

    if args.select_k is not None:
        max_possible = args.rows * args.cols
        args.select_k = min(max_possible, cap_select_k_for_engine("neat", args.select_k))

    if args.test_mode and not args.resume:
        args.agent_generations = min(3, args.agent_generations)
        args.num_agents = min(2, args.num_agents)


def run(args: argparse.Namespace) -> None:
    ensure_valid_args(args)
    apply_random_seed(args.seed)
    if not args.dry_run:
        ensure_gemini_key()
    orchestrator = CollaborativeMultiAgentOrchestrator(
        args.experiment_dir,
        rows=args.rows,
        cols=args.cols,
        thumb_size=args.thumb_size,
        scheme=args.scheme,
        palette=args.color_palette,
        config_path=args.config_path,
        select_k=args.select_k,
        agent_generations=args.agent_generations,
        enable_output_activations=args.output_activations,
        dry_run=args.dry_run,
        seed=args.seed,
    )

    orchestrator.run_agents(
        args.num_agents,
        resume=args.resume,
        resume_agent_id=args.resume_agent_id,
    )
    orchestrator.archive_manager.create_archive_grid(args.thumb_size)


if __name__ == "__main__":
    run(parse_args())
