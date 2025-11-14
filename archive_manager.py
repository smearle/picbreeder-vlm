"""Archive management utilities for collaborative multi-agent Picbreeder runs."""

from __future__ import annotations

import contextlib
import copy
import json
import math
import os
import pickle
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import neat
from PIL import Image, ImageDraw, ImageFont

from constants import RATE_EVERY
from im_query import query_images_with_captions
from rate_archive_with_vlm import (
    ArchiveEntry as RatingArchiveEntry,
    RatingResult,
    build_rating_system_prompt,
    format_rating_entry_label,
    parse_rating_batch_response,
)
from utils import atomic_write_json, _ensure_int_list

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
        )


class ArchiveManager:
    """Manages the shared archive of published favourites."""

    def __init__(self, archive_dir: Path, goal_prompt: str) -> None:
        self.archive_dir = archive_dir
        self.goal_prompt = goal_prompt
        self.metadata_file = archive_dir / "archive_metadata.json"
        # self._lock_path = self.metadata_file.with_suffix(".lock")
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
                "goal_prompt": self.goal_prompt,
            }
            self._persist()

    def _persist(self) -> None:
        atomic_write_json(self.metadata_file, self._metadata)

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
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Return top-rated and random archive subsets for branching."""
        self.refresh()
        entries = list(self._metadata.get("entries", []))
        if not entries:
            return [], []

        decorated: List[Tuple[float, int, int, Dict[str, Any]]] = []
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
            decorated.append((average_rating, rating_count, idx, entry))

        if not decorated:
            return [], []

        decorated.sort(key=lambda item: (item[0], item[1], -item[2]), reverse=True)

        top_subset: List[Dict[str, Any]] = []
        for average_rating, rating_count, idx, entry in decorated:
            if len(top_subset) >= max(0, top_count):
                break
            payload = copy.deepcopy(entry)
            payload["_archive_index"] = idx
            payload["_average_rating"] = average_rating if math.isfinite(average_rating) else None
            payload["_rating_count"] = rating_count
            top_subset.append(payload)

        selected_ids = {item.get("id") for item in top_subset if item.get("id")}
        eligible_random: List[Tuple[float, int, int, Dict[str, Any]]] = []
        for average_rating, rating_count, idx, entry in decorated:
            entry_id = entry.get("id")
            if entry_id in selected_ids:
                continue
            eligible_random.append((average_rating, rating_count, idx, entry))

        if eligible_random:
            random_pool = [item[3] for item in eligible_random]
        else:
            random_pool = []

        if random_count <= 0 or not random_pool:
            random_subset: List[Dict[str, Any]] = []
        else:
            sample_size = min(random_count, len(random_pool))
            sampled_entries = random.sample(random_pool, sample_size)
            random_subset = []
            id_to_meta = {entry.get("id"): (avg, cnt, idx) for avg, cnt, idx, entry in eligible_random}
            for entry in sampled_entries:
                avg, cnt, idx = id_to_meta.get(entry.get("id"), (float("nan"), 0, -1))
                payload = copy.deepcopy(entry)
                payload["_archive_index"] = idx
                payload["_average_rating"] = avg if math.isfinite(avg) else None
                payload["_rating_count"] = cnt
                random_subset.append(payload)

        return top_subset, random_subset

    def prepare_rating_batch(self, limit: int) -> List[Dict[str, str]]:
        if limit <= 0:
            return []
        # with interprocess_lock(self._lock_path):
        self.refresh()
        targets = self._select_lowest_rated_entries_locked(limit)
        if targets:
            manual_meta = self._metadata.setdefault("manual_rating", {})
            manual_meta["in_progress_ids"] = [item.get("id") for item in targets]
            manual_meta["in_progress_started_at"] = datetime.now().isoformat()
            self._persist()
        return targets

    def apply_rating_results(self, ratings: Dict[str, RatingResult]) -> None:
        if not ratings:
            return
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
        print(f"Applied ratings for {applied_count} archive entries.")

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
        should_auto_rate = False
        auto_rate_targets: List[Dict[str, str]] = []
        auto_rate_trigger_size = 0

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

        total_entries = len(entries_list)
        auto_rating_meta = self._metadata.setdefault("auto_rating", {})
        if total_entries > 0 and total_entries % RATE_EVERY == 0:
            last_completed = int(auto_rating_meta.get("last_completed_count", 0) or 0)
            in_progress = auto_rating_meta.get("in_progress_count")
            if total_entries > last_completed and in_progress != total_entries:
                auto_rate_targets = self._select_lowest_rated_entries_locked(100)
                if auto_rate_targets:
                    should_auto_rate = True
                    auto_rate_trigger_size = total_entries
                    auto_rating_meta["in_progress_count"] = len(auto_rate_targets)
                    auto_rating_meta["in_progress_started_at"] = datetime.now().isoformat()

        self._persist()
        self._write_checkpoint(archive_entry)

        if should_auto_rate:
            self._execute_auto_rating(auto_rate_targets, auto_rate_trigger_size)

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
                if not path.exists():
                    continue
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
            global_index_map: Dict[int, Dict[str, Any]] = {}
            for label, subset_entries in groups:
                subset_images: List[Image.Image] = []
                subset_indices: List[int] = []
                skip_entries: List[int] = []
                for entry in subset_entries:
                    path = Path(entry.get("image_path", ""))
                    if not path.exists():
                        skip_entries.append(entry["_index"])
                        continue
                    try:
                        with Image.open(path) as img:
                            processed = img.convert("RGB").resize((thumb_size, thumb_size), Image.Resampling.LANCZOS)
                    except Exception:
                        skip_entries.append(entry["_index"])
                        continue
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
    def _select_lowest_rated_entries_locked(self, limit: int) -> List[Dict[str, str]]:
        if limit <= 0:
            return []
        entries = list(self._metadata.get("entries", []))
        if not entries:
            return []

        decorated: List[Tuple[int, datetime, str, Dict[str, Any]]] = []
        for entry in entries:
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
            entry_id = str(entry.get("id") or "")
            decorated.append((rating_count, added_at, entry_id, entry))

        decorated.sort(key=lambda item: (item[0], item[1], item[2]))
        snapshot: List[Dict[str, str]] = []
        for _, _, _, entry in decorated[:limit]:
            image_path = entry.get("image_path")
            entry_id = str(entry.get("id") or "")
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

    def _execute_auto_rating(self, targets: Sequence[Dict[str, str]], trigger_size: int) -> None:
        if not targets:
            # with interprocess_lock(self._lock_path):
            self.refresh()
            auto_meta = self._metadata.setdefault("auto_rating", {})
            if auto_meta.get("in_progress_count") == trigger_size:
                auto_meta["in_progress_count"] = None
                auto_meta["last_completed_count"] = max(
                    int(auto_meta.get("last_completed_count", 0) or 0),
                    trigger_size,
                )
                auto_meta.pop("in_progress_started_at", None)
                auto_meta["last_completed_at"] = datetime.now().isoformat()
            self._persist()
            return

        vlm_entries: List[RatingArchiveEntry] = []
        for target in targets:
            image_path = Path(target.get("image_path", ""))
            if not image_path.exists():
                continue
            vlm_entries.append(
                RatingArchiveEntry(
                    image_id=str(target.get("id")),
                    title=str(target.get("title") or target.get("id") or ""),
                    image_path=image_path,
                )
            )

        if not vlm_entries:
            # with interprocess_lock(self._lock_path):
            self.refresh()
            auto_meta = self._metadata.setdefault("auto_rating", {})
            if auto_meta.get("in_progress_count") == trigger_size:
                auto_meta["in_progress_count"] = None
                auto_meta.pop("in_progress_started_at", None)
            self._persist()
            return

        rating_batch_size = 100
        include_titles = True
        require_titles = False
        results: Dict[str, RatingResult] = {}

        # Save rating system prompt to disk
        prompt_path = self.archive_dir / "vlm_ratings" / "system_prompt.txt"
        prompt_path.parent.mkdir(parents=True, exist_ok=True)

        for start in range(0, len(vlm_entries), rating_batch_size):
            batch = vlm_entries[start : start + rating_batch_size]
            if not batch:
                continue
            system_prompt = build_rating_system_prompt(
                batch,
                require_titles=require_titles,
                goal_prompt=self.goal_prompt,
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

        # with interprocess_lock(self._lock_path):
        self.refresh()
        entries_by_id: Dict[str, Dict[str, Any]] = {}
        for entry in self._metadata.get("entries", []):
            entry_id = str(entry.get("id") or "")
            if entry_id:
                entries_by_id[entry_id] = entry

        timestamp = datetime.now().isoformat()
        for entry_id, rating in results.items():
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

            auto_meta = self._metadata.setdefault("auto_rating", {})
            if auto_meta.get("in_progress_count") == trigger_size:
                auto_meta["last_completed_count"] = max(
                    int(auto_meta.get("last_completed_count", 0) or 0),
                    trigger_size,
                )
                auto_meta["last_completed_at"] = timestamp
                auto_meta["in_progress_count"] = None
                auto_meta.pop("in_progress_started_at", None)

            self._persist()

        print(
            f"Auto-rated {len(results)} of {len(vlm_entries)} archive entries (trigger size={trigger_size})."
        )

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
