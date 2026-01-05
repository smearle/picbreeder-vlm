"""Archive management utilities for collaborative multi-agent Picbreeder runs."""

from __future__ import annotations

import contextlib
import copy
import heapq
import json
import math
import os
import pickle
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import neat
from PIL import Image, ImageDraw, ImageFont

from constants import RATING_BATCH_SIZE, RATE_EVERY, REPO_NAME
from rate_archive_with_vlm import RatingResult
from utils import atomic_write_json, _ensure_int_list, relative_suffix_after_dir

try:
    import fcntl  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - platform specific
    fcntl = None  # type: ignore[assignment]

try:
    import msvcrt  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - platform specific
    msvcrt = None  # type: ignore[assignment]

ARCHIVE_GRID_MARGIN = 12


def _lock_file_handle(handle: Any) -> bool:
    if fcntl is not None:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        return True
    if msvcrt is not None:  # pragma: no cover - windows specific
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        return True
    return False


def _unlock_file_handle(handle: Any) -> None:
    if fcntl is not None:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    elif msvcrt is not None:  # pragma: no cover - windows specific
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)


@contextlib.contextmanager
def interprocess_lock(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as handle:
        locked = _lock_file_handle(handle)
        try:
            yield
        finally:
            if locked:
                _unlock_file_handle(handle)


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
    color_enabled: bool = False
    n_published_children: int = 0

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
            "color_enabled": bool(self.color_enabled),
            "n_published_children": int(self.n_published_children),
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
        color_raw = payload.get("color_enabled", False)
        if isinstance(color_raw, str):
            lowered = color_raw.strip().lower()
            color_enabled = lowered in {"1", "true", "yes", "on"}
        else:
            color_enabled = bool(color_raw)
        children_raw = payload.get("n_published_children", 0)
        n_published_children = int(children_raw)

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
            color_enabled=color_enabled,
            n_published_children=n_published_children,
        )


class ArchiveManager:
    """Manages the shared archive of published favourites."""

    def __init__(self, archive_dir: Path, goal_prompt: str, thumb_size: int) -> None:
        self.archive_dir = archive_dir
        self.goal_prompt = goal_prompt
        self.thumb_size = thumb_size
        self.metadata_file = archive_dir / "archive_metadata.json"
        self.images_dir = archive_dir / "images"
        self.genomes_dir = archive_dir / "genomes"
        for directory in (
            archive_dir,
            self.images_dir,
            self.genomes_dir,
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
                "goal_prompt": self.goal_prompt,
            }
            self._persist()
            return

    def _persist(self) -> None:
        atomic_write_json(self.metadata_file, self._metadata)

    def _increment_parent_child_counts(self, parent_entry_ids: Sequence[str]) -> bool:
        """Increment child counters for the given parent entry IDs."""
        if not parent_entry_ids:
            return False
        parent_ids = {str(value or "").strip() for value in parent_entry_ids if str(value or "").strip()}
        if not parent_ids:
            return False

        changed = False
        for entry in self._metadata.get("entries", []):
            entry_id = str(entry.get("id") or "")
            if entry_id not in parent_ids:
                continue
            current_raw = entry.get("n_published_children", 0)
            try:
                current_value = int(current_raw)
            except (TypeError, ValueError):
                current_value = 0
            entry["n_published_children"] = current_value + 1
            changed = True
        return changed

    def refresh(self) -> None:
        """Reload metadata from disk to incorporate external updates."""
        self._load()

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    @property
    def entries(self) -> List[Dict[str, Any]]:
        self.refresh()
        return list(self._metadata.get("entries", []))

    def sample_entries(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        self.refresh()
        entries = list(self._metadata.get("entries", []))
        if limit is None or limit <= 0 or len(entries) <= limit:
            return entries
        return random.sample(entries, limit)

    def sample_branching_entries(
        self,
        top_count: int,
        random_count: int,
        *,
        best_new_count: int = 25,
        best_new_window_multiplier: int = 10,
        most_branched_count: int = 25,
        newest_count: Optional[int] = None,
    ) -> Tuple[
        List[Dict[str, Any]],
        List[Dict[str, Any]],
        List[Dict[str, Any]],
        List[Dict[str, Any]],
        List[Dict[str, Any]],
    ]:
        """Return top-rated, best-new, most-branched, newest, and random subsets for branching."""
        self.refresh()
        entries = list(self._metadata.get("entries", []))
        if not entries:
            return [], [], [], [], []

        newest_limit = best_new_count if newest_count is None else newest_count
        top_limit = max(0, int(top_count))
        random_limit = max(0, int(random_count))
        best_new_limit = max(0, int(best_new_count))
        most_branched_limit = max(0, int(most_branched_count))
        newest_limit = max(0, int(newest_limit))
        window_multiplier = max(1, int(best_new_window_multiplier))

        entry_infos: List[Dict[str, Any]] = []
        for idx, entry in enumerate(entries):
            image_path = Path(entry.get("image_path", ""))
            if not image_path.exists():
                continue
            ratings_raw = entry.get("vlm_ratings")
            if isinstance(ratings_raw, list) and ratings_raw:
                try:
                    rating_values = [float(value) for value in ratings_raw]
                except (TypeError, ValueError):
                    rating_values = []
                if rating_values:
                    average_rating = sum(rating_values) / len(rating_values)
                    rating_count = len(rating_values)
                else:
                    average_rating = float("-inf")
                    rating_count = 0
            else:
                average_rating = float("-inf")
                rating_count = 0

            added_at_raw = entry.get("added_at")
            if isinstance(added_at_raw, str):
                try:
                    added_at = datetime.fromisoformat(added_at_raw)
                except ValueError:
                    added_at = datetime.min
            else:
                added_at = datetime.min

            children_raw = entry.get("n_published_children", 0)
            try:
                published_children = int(children_raw)
            except (TypeError, ValueError):
                published_children = 0

            entry_id_raw = entry.get("id")
            entry_key = str(entry_id_raw) if entry_id_raw not in (None, "") else f"__idx_{idx}"

            entry_infos.append(
                {
                    "entry": entry,
                    "idx": idx,
                    "avg": average_rating,
                    "count": rating_count,
                    "added_at": added_at,
                    "key": entry_key,
                    "children": published_children,
                }
            )

        if not entry_infos:
            return [], [], [], [], []

        def _decorate(info: Dict[str, Any]) -> Dict[str, Any]:
            payload = copy.deepcopy(info["entry"])
            payload["_archive_index"] = info["idx"]
            payload["_average_rating"] = info["avg"] if math.isfinite(info["avg"]) else None
            payload["_rating_count"] = info["count"]
            payload["_added_at"] = info["added_at"].isoformat()
            payload["_published_children"] = info["children"]
            return payload

        used_keys: Set[str] = set()

        def _remaining_infos() -> List[Dict[str, Any]]:
            return [info for info in entry_infos if info["key"] not in used_keys]

        # Top rated subset
        decorated = list(entry_infos)
        random.shuffle(decorated)
        decorated.sort(key=lambda info: (info["avg"], info["count"]), reverse=True)
        top_infos = decorated[:top_limit]
        used_keys.update(info["key"] for info in top_infos)

        # Best new subset: highest-rated within most recent window of remaining entries
        best_new_infos: List[Dict[str, Any]] = []
        if best_new_limit > 0:
            remaining_infos = _remaining_infos()
            if remaining_infos:
                window_size = min(len(remaining_infos), best_new_limit * window_multiplier)
                recent_pool = list(remaining_infos)
                random.shuffle(recent_pool)
                recent_pool.sort(key=lambda info: info["added_at"], reverse=True)
                window = recent_pool[:window_size]
                random.shuffle(window)
                window.sort(key=lambda info: (info["avg"], info["count"]), reverse=True)
                best_new_infos = window[:best_new_limit]
                used_keys.update(info["key"] for info in best_new_infos)

        # Most branched subset: highest child counts among remaining entries
        most_branched_infos: List[Dict[str, Any]] = []
        if most_branched_limit > 0:
            remaining_infos = _remaining_infos()
            if remaining_infos:
                branched_pool = [info for info in remaining_infos if info["children"] > 0]
                if branched_pool:
                    random.shuffle(branched_pool)
                    branched_pool.sort(
                        key=lambda info: (info["children"], info["avg"], info["count"]),
                        reverse=True,
                    )
                    most_branched_infos = branched_pool[:most_branched_limit]
                    used_keys.update(info["key"] for info in most_branched_infos)

        # Newest subset: newest remaining entries regardless of rating
        newest_infos: List[Dict[str, Any]] = []
        if newest_limit > 0:
            remaining_infos = _remaining_infos()
            if remaining_infos:
                newest_pool = list(remaining_infos)
                random.shuffle(newest_pool)
                newest_pool.sort(key=lambda info: info["added_at"], reverse=True)
                newest_infos = newest_pool[:newest_limit]
                used_keys.update(info["key"] for info in newest_infos)

        # Random subset: draw randomly from the remainder
        random_infos: List[Dict[str, Any]] = []
        if random_limit > 0:
            remaining_infos = _remaining_infos()
            if remaining_infos:
                sample_size = min(random_limit, len(remaining_infos))
                random_infos = random.sample(remaining_infos, sample_size)

        top_subset = [_decorate(info) for info in top_infos]
        best_new_subset = [_decorate(info) for info in best_new_infos]
        most_branched_subset = [_decorate(info) for info in most_branched_infos]
        newest_subset = [_decorate(info) for info in newest_infos]
        random_subset = [_decorate(info) for info in random_infos]

        return top_subset, best_new_subset, most_branched_subset, newest_subset, random_subset

    def prepare_rating_batch(self, limit: int) -> List[Dict[str, str]]:
        if limit <= 0:
            return []
        # with interprocess_lock(self._lock_path):
        self.refresh()
        targets = self._select_entries_for_rating(limit)
        if targets:
            manual_meta = self._metadata.setdefault("manual_rating", {})
            manual_meta["in_progress_ids"] = [item.get("id") for item in targets]
            manual_meta["in_progress_started_at"] = datetime.now().isoformat()
            self._persist()
        return targets

    def apply_rating_results(self, ratings: Dict[str, RatingResult]) -> int:
        if not ratings:
            return 0
        # with interprocess_lock(self._lock_path):
        self.refresh()
        entries_by_id: Dict[str, Dict[str, Any]] = {}
        for entry in self._metadata.get("entries", []):
            entry_id = str(entry.get("id") or "")
            if entry_id:
                entries_by_id[entry_id] = entry

        timestamp = datetime.now().isoformat()
        applied_count = 0
        for entry_id, rating in ratings.items():
            entry = entries_by_id.get(entry_id)
            if entry is None:
                continue
            ratings_list = entry.get("vlm_ratings")
            if isinstance(ratings_list, list):
                pass
            elif ratings_list is None:
                ratings_list = []
            else:
                ratings_list = [ratings_list]
            ratings_list.append(float(rating.score))
            entry["vlm_ratings"] = ratings_list

            comments_list = entry.get("vlm_comments")
            if isinstance(comments_list, list):
                pass
            elif comments_list is None:
                comments_list = []
            else:
                comments_list = [comments_list]
            comments_list.append(str(rating.justification or ""))
            entry["vlm_comments"] = comments_list

            titles_list = entry.get("vlm_reported_titles")
            if isinstance(titles_list, list):
                pass
            elif titles_list is None:
                titles_list = []
            else:
                titles_list = [titles_list]
            titles_list.append(str(rating.reported_title or ""))
            entry["vlm_reported_titles"] = titles_list

            applied_count += 1

            manual_meta = self._metadata.setdefault("manual_rating", {})
            if applied_count:
                manual_meta["last_completed_at"] = timestamp
                manual_meta["last_completed_count"] = applied_count
            in_progress_ids = manual_meta.get("in_progress_ids")
            if isinstance(in_progress_ids, list) and in_progress_ids:
                remaining = [entry_id for entry_id in in_progress_ids if entry_id not in ratings]
                if remaining:
                    manual_meta["in_progress_ids"] = remaining
                else:
                    manual_meta.pop("in_progress_ids", None)
                    manual_meta.pop("in_progress_started_at", None)
            self._persist()
        print(f"Applied rating results for {applied_count} archive entries.")
        return applied_count

    def prepare_auto_rating_batch(self, limit: Optional[int] = None) -> Tuple[List[Dict[str, str]], int]:
        batch_size = RATING_BATCH_SIZE if limit is None else max(1, min(limit, RATING_BATCH_SIZE))
        if batch_size <= 0 or RATE_EVERY <= 0:
            print("Auto-rating is disabled due to non-positive batch size or rate interval.")
            return [], 0

        self.refresh()
        entries = self._metadata.get("entries", [])
        total_entries = len(entries)
        if total_entries < RATING_BATCH_SIZE:
            # print("Not enough entries in archive to trigger auto-rating.")
            return [], 0
        if total_entries % RATE_EVERY != 0:
            # print("Not the right time to trigger auto-rating based on RATE_EVERY setting.")
            return [], 0

        auto_meta = self._metadata.setdefault("auto_rating", {})

        targets = self._select_entries_for_rating(batch_size)
        if not targets:
            return [], 0

        auto_meta["trigger_entry_count"] = total_entries
        auto_meta["in_progress_batch_size"] = len(targets)
        auto_meta["in_progress_started_at"] = datetime.now().isoformat()
        self._persist()
        return targets, total_entries

    def mark_auto_rating_complete(self, trigger_entry_count: int, completed_count: int) -> None:
        if trigger_entry_count <= 0:
            return
        self.refresh()
        auto_meta = self._metadata.setdefault("auto_rating", {})
        if int(auto_meta.get("trigger_entry_count") or 0) != trigger_entry_count:
            return
        auto_meta["last_completed_count"] = trigger_entry_count
        auto_meta["last_completed_at"] = datetime.now().isoformat()
        auto_meta["last_completed_batch_size"] = int(completed_count)
        auto_meta.pop("trigger_entry_count", None)
        auto_meta.pop("in_progress_started_at", None)
        auto_meta.pop("in_progress_batch_size", None)
        self._persist()

    def mark_auto_rating_failed(self, trigger_entry_count: int) -> None:
        if trigger_entry_count <= 0:
            return
        self.refresh()
        auto_meta = self._metadata.setdefault("auto_rating", {})
        if int(auto_meta.get("trigger_entry_count") or 0) != trigger_entry_count:
            return
        auto_meta.pop("trigger_entry_count", None)
        auto_meta.pop("in_progress_started_at", None)
        auto_meta.pop("in_progress_batch_size", None)
        self._persist()

    def get_elite_names(self, max_length: int = 80) -> List[str]:
        self.refresh()
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
        self.refresh()
        for raw in self._metadata.get("entries", []):
            if raw.get("id") == entry_id:
                try:
                    return ArchiveEntry.from_dict(raw)
                except Exception:
                    return None
        return None

    def add_entry(
        self,
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
        color_enabled: bool = False,
    ) -> ArchiveEntry:
        """Persist a favorite image and genome into the shared archive."""

        # with interprocess_lock(self._lock_path):
        self.refresh()
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
            color_enabled=bool(color_enabled),
        )

        entries_list = self._metadata.setdefault("entries", [])
        entries_list.append(archive_entry.as_dict())

        self._increment_parent_child_counts(archive_entry.source_entry_ids)
        self._persist()
        self.create_archive_grid(self.thumb_size)
        # self._write_checkpoint(archive_entry)

        return archive_entry

    def load_genome(self, entry_id: str) -> Optional[neat.DefaultGenome]:
        self.refresh()
        for entry in self._metadata.get("entries", []):
            if entry.get("id") != entry_id:
                continue
            genome_path = Path(entry["genome_path"])
            if not genome_path.exists():
                return None
            with genome_path.open("rb") as handle:
                return pickle.load(handle)
        return None

    def create_archive_grid(
        self,
        thumb_size: int = 200,
        entries: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> Optional[Path]:
        if entries is None:
            self.refresh()
            entries = self.entries
            output_path = self.archive_dir / "archive_grid.png"
        else:
            entries = list(entries)
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            output_path = self.logs_dir / f"archive_grid_sample_{timestamp}.png"
            output_path.parent.mkdir(parents=True, exist_ok=True)
        if not entries:
            return None

        background_color = (18, 18, 22)
        margin = ARCHIVE_GRID_MARGIN
        labeled_entries: List[Dict[str, Any]] = []
        for idx, entry in enumerate(entries):
            payload = dict(entry)
            payload.setdefault("branching_subset_label", None)
            payload["_index"] = idx
            labeled_entries.append(payload)

        has_group_labels = any(
            bool(str(entry.get("branching_subset_label") or "").strip())
            for entry in labeled_entries
        )

        metadata_entries: List[Dict[str, Any]] = []
        subset_metadata: List[Dict[str, Any]] = []

        if not has_group_labels:
            images: List[Image.Image] = []
            valid_indices: List[int] = []
            for entry in labeled_entries:
                path = Path(entry["image_path"])
                path = relative_suffix_after_dir(path, "archive")
                path = self.archive_dir / path
                if not path.exists():
                    raise FileNotFoundError(f"Image path does not exist: {path}")
                try:
                    with Image.open(path) as img:
                        processed = img.convert("RGB").resize((thumb_size, thumb_size), Image.Resampling.LANCZOS)
                except Exception:
                    continue
                images.append(processed)
                valid_indices.append(entry["_index"])

            if not images:
                return None

            columns = max(1, math.ceil(math.sqrt(len(images))))
            rows = math.ceil(len(images) / columns)
            tile_width, tile_height = images[0].size
            canvas = Image.new(
                "RGB",
                (
                    columns * tile_width + (columns + 1) * margin,
                    rows * tile_height + (rows + 1) * margin,
                ),
                background_color,
            )
            for idx, img in enumerate(images):
                col = idx % columns
                row = idx // columns
                x = margin + col * (tile_width + margin)
                y = margin + row * (tile_height + margin)
                canvas.paste(img, (x, y))
                metadata_entries.append(
                    {
                        "index": valid_indices[idx],
                        "bbox": [x, y, x + tile_width, y + tile_height],
                        "subset_label": None,
                    }
                )

            canvas.save(output_path, format="PNG")
        else:
            groups: List[Tuple[str, List[Dict[str, Any]]]] = []
            current_label: Optional[str] = None
            current_group: List[Dict[str, Any]] = []
            for entry in labeled_entries:
                label = str(entry.get("branching_subset_label") or "").strip() or "Group"
                if current_label is None:
                    current_label = label
                if label != current_label:
                    groups.append((current_label, current_group))
                    current_group = []
                    current_label = label
                current_group.append(entry)
            if current_group:
                groups.append((current_label or "Group", current_group))

            panes: List[Dict[str, Any]] = []
            font = ImageFont.load_default()
            header_gap = max(4, margin // 3)
            for label, subset_entries in groups:
                subset_images: List[Image.Image] = []
                subset_indices: List[int] = []
                skip_entries: List[int] = []
                for entry in subset_entries:
                    path = Path(entry.get("image_path", ""))
                    path = relative_suffix_after_dir(path, "archive")
                    path = self.archive_dir / path
                    if not path.exists():
                        raise FileNotFoundError(f"Image path does not exist: {path}")
                    try:
                        with Image.open(path) as img:
                            processed = img.convert("RGB").resize((thumb_size, thumb_size), Image.Resampling.LANCZOS)
                    except Exception:
                        raise RuntimeError(f"Failed to open/process image at path: {path}")
                    subset_images.append(processed)
                    subset_indices.append(entry["_index"])

                if not subset_images:
                    continue

                columns = max(1, math.ceil(math.sqrt(len(subset_images))))
                rows = math.ceil(len(subset_images) / columns)
                tile_width, tile_height = subset_images[0].size
                pane_width = columns * tile_width + (columns + 1) * margin
                pane_height = rows * tile_height + (rows + 1) * margin
                pane_canvas = Image.new("RGB", (pane_width, pane_height), background_color)
                pane_positions: List[Dict[str, Any]] = []
                for idx, img in enumerate(subset_images):
                    col = idx % columns
                    row = idx // columns
                    x = margin + col * (tile_width + margin)
                    y = margin + row * (tile_height + margin)
                    pane_canvas.paste(img, (x, y))
                    pane_positions.append(
                        {
                            "index": subset_indices[idx],
                            "bbox": [x, y, x + tile_width, y + tile_height],
                            "subset_label": label,
                        }
                    )

                try:
                    bbox = font.getbbox(f"{label}:")
                    header_height = bbox[3] - bbox[1]
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
                subset_metadata.append(
                    {
                        "label": label,
                        "count": len(pane_positions),
                        "start_index": min(position["index"] for position in pane_positions),
                    }
                )

            if not panes:
                return None

            max_header_height = max(pane["header_height"] for pane in panes)
            max_grid_height = max(pane["height"] for pane in panes)
            total_width = sum(pane["width"] for pane in panes) + margin * (len(panes) + 1)
            total_height = margin + max_header_height + header_gap + max_grid_height + margin
            canvas = Image.new("RGB", (total_width, total_height), background_color)
            draw = ImageDraw.Draw(canvas)

            x_cursor = margin
            grid_y = margin + max_header_height + header_gap
            for pane in panes:
                header_text = f"{pane['label']}:"
                draw.text((x_cursor, margin), header_text, fill=(235, 235, 240), font=font)
                canvas.paste(pane["canvas"], (x_cursor, grid_y))
                for position in pane["positions"]:
                    x0, y0, x1, y1 = position["bbox"]
                    metadata_entries.append(
                        {
                            "index": position["index"],
                            "bbox": [x_cursor + x0, grid_y + y0, x_cursor + x1, grid_y + y1],
                            "subset_label": pane["label"],
                        }
                    )
                x_cursor += pane["width"] + margin

            canvas.save(output_path, format="PNG")

        print(f"Archive grid saved to: {output_path}")

        if entries is not None:
            metadata_path = output_path.with_suffix(".json")
            payload = {
                "generated_at": datetime.now().isoformat(),
                "entries": metadata_entries,
                "subsets": subset_metadata,
                "thumb_size": thumb_size,
                "margin": margin,
            }
            try:
                metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            except OSError:
                pass
        return output_path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _select_entries_for_rating(self, limit: int) -> List[Dict[str, str]]:
        if limit <= 0:
            return []
        entries = list(self._metadata.get("entries", []))
        if not entries or len(entries) < RATING_BATCH_SIZE:
            return []

        entries_with_images: List[Dict[str, Any]] = []
        keys: List[Tuple[int, datetime, str, int]] = []
        for idx, entry in enumerate(entries):
            image_path = entry.get("image_path")
            entry_id = str(entry.get("id") or "")
            if not entry_id or not image_path:
                continue
            rating_values = entry.get("vlm_ratings") or []
            rating_count = len(rating_values) if isinstance(rating_values, list) else 0
            added_at_raw = entry.get("added_at")
            if isinstance(added_at_raw, str):
                try:
                    added_at = datetime.fromisoformat(added_at_raw)
                except ValueError:
                    added_at = datetime.min
            else:
                added_at = datetime.min
            entries_with_images.append(entry)
            keys.append((rating_count, added_at, entry_id, len(entries_with_images) - 1))

        total_candidates = len(entries_with_images)
        if total_candidates < RATING_BATCH_SIZE:
            return []

        least_quota = min(limit // 2, total_candidates)
        random_quota = min(limit - least_quota, max(0, total_candidates - least_quota))

        selected_entries: List[Dict[str, Any]] = []
        selected_indices: Set[int] = set()

        if least_quota > 0:
            least_subset = heapq.nsmallest(least_quota, keys, key=lambda item: (item[0], item[1], item[2]))
            for _, _, _, entry_idx in least_subset:
                if entry_idx in selected_indices:
                    continue
                selected_indices.add(entry_idx)
                selected_entries.append(entries_with_images[entry_idx])

        available_count = total_candidates - len(selected_indices)
        random_quota = min(random_quota, available_count)
        if random_quota > 0:
            random_indices: Set[int] = set()
            while len(random_indices) < random_quota:
                candidate_idx = random.randrange(total_candidates)
                if candidate_idx in selected_indices or candidate_idx in random_indices:
                    continue
                random_indices.add(candidate_idx)
            for entry_idx in random_indices:
                selected_entries.append(entries_with_images[entry_idx])

        snapshot: List[Dict[str, str]] = []
        for entry in selected_entries:
            entry_id = str(entry.get("id") or "")
            image_path = entry.get("image_path")
            if not entry_id or not image_path:
                continue
            snapshot.append(
                {
                    "id": entry_id,
                    "title": str(entry.get("title") or entry_id),
                    "image_path": str(image_path),
                }
            )
        return snapshot


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
        # with interprocess_lock(self._lock_path):
        self.refresh()
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


__all__ = [
    "ARCHIVE_GRID_MARGIN",
    "ArchiveEntry",
    "ArchiveManager",
    "interprocess_lock",
    "atomic_write_json",
    "_ensure_int_list",
]
