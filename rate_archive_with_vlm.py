#!/usr/bin/env python3
"""Query a VLM to rate archive images—either individually or in batches—and visualize the results."""

from __future__ import annotations

import argparse
import base64
import json
import math
import random
import re
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from im_query import DEFAULT_MODEL, query_im, query_images_with_captions
from rendering import create_numbered_grid

SYSTEM_PROMPT_TEMPLATE = """You are evaluating generated artwork from a collaborative evolution archive.
The users were given the following objective: "{goal_prompt}"
You will be presented with {count} images.
For each image, provide a numeric score from 0 (poor) to 5 (exceptional) that reflects their success in achieving this objective.
{title_instruction}

Respond with JSON ONLY in this exact shape:
{ratings_schema}
Include exactly one rating per index shown in the grid."""


@dataclass
class ArchiveEntry:
    image_id: str
    title: str
    image_path: Path


@dataclass
class RatingResult:
    score: float
    justification: Optional[str]
    reported_title: Optional[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        type=Path,
        required=False,
        help="Path to an archive directory that contains archive_metadata.json",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="How many full passes to query the VLM (default: 3)",
    )
    parser.add_argument(
        "--image-limit",
        type=int,
        default=None,
        help="Optional cap on the number of images to rate",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of images to show to the VLM at once (-1 means use the full archive)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used for image sampling and order shuffling",
    )
    parser.add_argument(
        "--system-instruction",
        type=str,
        default=None,
        help="Optional system instruction passed to the VLM",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where logs and figures should be stored (defaults to archive/vlm_ratings/<timestamp>)",
    )
    parser.add_argument(
        "--resume-dir",
        type=Path,
        default=None,
        help="Existing vlm_ratings directory to continue from (skips creating a new timestamped folder)",
    )
    parser.add_argument(
        "--grid-thumb-size",
        type=int,
        default=256,
        help="Thumbnail size (pixels) used when building numbered grids (default: 256)",
    )
    parser.add_argument(
        "--include-titles-in-prompt",
        action="store_true",
        help="Include image titles in the prompt/reference list provided to the VLM",
    )
    parser.add_argument(
        "--use-multipart-input",
        dest="use_multipart_input",
        action="store_true",
        default=True,
        help="(default) Send images and captions as multi-part input instead of a stitched grid",
    )
    parser.add_argument(
        "--use-grid-input",
        dest="use_multipart_input",
        action="store_false",
        help="Send a single numbered grid image instead of multi-part input",
    )
    parser.add_argument(
        "--verify-titles",
        action="store_true",
        help="Ask the VLM to repeat titles for each rating and log mismatches for manual review",
    )
    return parser.parse_args()


def load_archive_entries(archive_dir: Path) -> List[ArchiveEntry]:
    metadata_path = archive_dir / "archive_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Could not find {metadata_path}")
    metadata = json.loads(metadata_path.read_text())
    entries: List[ArchiveEntry] = []
    goal_prompt = metadata.get("goal_prompt")
    for entry in metadata.get("entries", []):
        raw_path = Path(entry["image_path"]).expanduser()
        resolved = raw_path
        if "archive" in raw_path.parts:
            try:
                archive_idx = raw_path.parts.index("archive")
            except ValueError:
                archive_idx = -1
            if archive_idx >= 0:
                rel_path = Path(*raw_path.parts[archive_idx + 1 :])
                candidate = (archive_dir / rel_path).resolve()
                if candidate.exists():
                    resolved = candidate
        if not resolved.exists():
            continue
        entries.append(
            ArchiveEntry(
                image_id=entry["id"],
                title=entry.get("title") or entry["id"],
                image_path=resolved,
            )
        )
    if not entries:
        raise ValueError(f"No entries with images found in {metadata_path}")
    return entries, goal_prompt


def format_rating_entry_label(idx: int, entry: ArchiveEntry, include_titles: bool) -> str:
    if include_titles:
        return f"Image {idx}: {entry.title}"
    return f"Image {idx}"


def build_rating_system_prompt(
    batch: Sequence[ArchiveEntry],
    require_titles: bool,
    goal_prompt: str,
) -> str:
    title_instruction = ""
    ratings_schema = """{
  "ratings": [
    {"index": 0, "score": 4, "justification": "concise one-sentence reason"},
    ...
  ]
}"""
    if require_titles:
        title_instruction = "Repeat the exact title from the reference list within each rating object's `title` field."
        ratings_schema = """{
  "ratings": [
    {"index": 0, "title": "Exact title from the list", "score": 4, "justification": "concise one-sentence reason"},
    ...
  ]
}"""
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        count=len(batch),
        goal_prompt=goal_prompt,
        title_instruction=title_instruction,
        ratings_schema=ratings_schema,
    )
    return system_prompt


_SCORE_RE = re.compile(r"(?i)(?:score|rating)\s*[:=-]?\s*([1-5](?:\.\d+)?)\s*(?:/5)?")
_FALLBACK_NUMBER_RE = re.compile(r"\b([1-5](?:\.\d+)?)\b")


def extract_score(response_text: str) -> Optional[float]:
    for matcher in (_SCORE_RE, _FALLBACK_NUMBER_RE):
        match = matcher.search(response_text)
        if match:
            value = float(match.group(1))
            return float(min(5.0, max(1.0, value)))
    return None


def ensure_output_dir(archive_dir: Path, custom_dir: Optional[Path]) -> Path:
    if custom_dir:
        out_dir = custom_dir.expanduser().resolve()
    else:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        out_dir = archive_dir / "vlm_ratings" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def query_vlm(image_bytes: bytes, prompt: str, system_instruction: Optional[str]) -> str:
    response = query_im(
        image_bytes,
        prompt=prompt,
        mime_type="image/png",
        system_instruction=system_instruction,
    )
    return getattr(response, "text", "") or ""


def log_record(log_path: Path, record: Dict) -> None:
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def load_scores_from_log(log_path: Path) -> Tuple[Dict[str, List[float]], int, Dict[str, Dict[str, List[float]]]]:
    scores: Dict[str, List[float]] = defaultdict(list)
    mode_scores: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    max_run_index = -1
    if not log_path.exists():
        return scores, 0, mode_scores
    with log_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            entries = payload.get("entries")
            run_index_value = payload.get("run_index")
            if isinstance(run_index_value, int):
                max_run_index = max(max_run_index, run_index_value)
            mode_key = payload.get("input_mode") or "grid"
            if isinstance(entries, list):
                for entry in entries:
                    image_id = entry.get("image_id")
                    score = entry.get("score")
                    if image_id and isinstance(score, (int, float)):
                        scores[image_id].append(float(score))
                        mode_scores[mode_key][image_id].append(float(score))
                    else:
                        print(f"Skipping invalid log entry score: {entry}")
            else:
                image_id = payload.get("image_id")
                score = payload.get("score")
                if image_id and isinstance(score, (int, float)):
                    scores[image_id].append(float(score))
                    mode_scores[mode_key][image_id].append(float(score))
    completed_runs = max_run_index + 1 if max_run_index >= 0 else 0
    return scores, completed_runs, mode_scores


def summarize_scores(entries: Iterable[ArchiveEntry], scores: Dict[str, List[float]]) -> List[Dict]:
    summary = []
    for entry in entries:
        image_scores = scores.get(entry.image_id, [])
        if not image_scores:
            continue
        mean_score = float(np.mean(image_scores))
        std_score = float(statistics.pstdev(image_scores)) if len(image_scores) > 1 else 0.0
        summary.append(
            {
                "image_id": entry.image_id,
                "title": entry.title,
                "image_path": str(entry.image_path),
                "count": len(image_scores),
                "mean_score": mean_score,
                "std_score": std_score,
                "scores": image_scores,
            }
        )
    return summary


def render_ranked_figure(summary: List[Dict], output_path: Path) -> None:
    ordered = sorted(summary, key=lambda item: item["mean_score"], reverse=True)
    if not ordered:
        return
    per_cell_size = 4.0
    cols = max(1, int(math.ceil(math.sqrt(len(ordered)))))
    rows = int(math.ceil(len(ordered) / cols))
    fig_width = per_cell_size * cols
    fig_height = per_cell_size * rows
    grid_wspace = 0.3  # increase if image + bar still feel cramped horizontally
    fig = plt.figure(figsize=(fig_width, fig_height))
    grid = fig.add_gridspec(rows, cols, hspace=0.35, wspace=grid_wspace)
    bar_width_fraction = 0.2  # fraction of each cell dedicated to the bar subplot

    for rank, entry in enumerate(ordered):
        row = rank // cols
        col = rank % cols
        cell_spec = grid[row, col]
        cell_grid = cell_spec.subgridspec(
            1,
            2,
            width_ratios=[1 - bar_width_fraction, bar_width_fraction],
            wspace=0.08,
        )

        img_ax = fig.add_subplot(cell_grid[0, 0])
        img = Image.open(entry["image_path"]).convert("RGB")
        img_ax.imshow(img)
        img_ax.set_axis_off()
        img_ax.set_title(f"#{rank + 1}: {entry['title']} (n={entry['count']})", fontsize=10)

        bar_ax = fig.add_subplot(cell_grid[0, 1])
        bar_ax.bar(
            [0],
            [entry["mean_score"]],
            yerr=[[entry["std_score"]]],
            color="#4c72b0",
            alpha=0.85,
            align="center",
            width=0.5,
        )
        bar_ax.set_ylim(1, 5)
        bar_ax.set_xlim(-0.75, 0.75)
        bar_ax.set_xticks([])
        bar_ax.set_yticks([1, 2, 3, 4, 5])
        bar_ax.tick_params(axis="y", pad=6, labelsize=8)
        bar_ax.set_title(f"{entry['mean_score']:.2f} ± {entry['std_score']:.2f}", fontsize=8)
        bar_ax.axhline(entry["mean_score"], color="#1f4e79", linewidth=1.2)
        for spine in ("top", "right"):
            bar_ax.spines[spine].set_visible(False)

    for leftover in range(len(ordered), rows * cols):
        row = leftover // cols
        col = leftover % cols
        fig.add_subplot(grid[row, col]).set_axis_off()

    fig.suptitle("Archive image ratings", fontsize=14, y=0.92)
    fig.subplots_adjust(
        left=0.04,
        right=0.98,
        bottom=0.04,
        top=0.9,
        wspace=grid_wspace,
        hspace=0.35,
    )
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_mode_variability_report(
    entries: Sequence[ArchiveEntry],
    mode_scores: Dict[str, Dict[str, List[float]]],
    output_path: Path,
) -> None:
    if not mode_scores:
        return
    entry_lookup = {entry.image_id: entry for entry in entries}
    report: Dict[str, Any] = {}
    for mode, per_image_scores in mode_scores.items():
        per_image_stats: List[Dict[str, Any]] = []
        std_values: List[float] = []
        total_scores = 0
        for image_id, values in per_image_scores.items():
            if not values:
                continue
            mean_score = float(np.mean(values))
            std_score = float(statistics.pstdev(values)) if len(values) > 1 else 0.0
            std_values.append(std_score)
            total_scores += len(values)
            entry = entry_lookup.get(image_id)
            per_image_stats.append(
                {
                    "image_id": image_id,
                    "title": entry.title if entry else image_id,
                    "count": len(values),
                    "mean_score": mean_score,
                    "std_score": std_score,
                    "scores": values,
                }
            )
        mode_summary = {
            "image_count": len(per_image_stats),
            "score_count": total_scores,
            "mean_std": float(np.mean(std_values)) if std_values else 0.0,
            "median_std": float(np.median(std_values)) if std_values else 0.0,
            "max_std": float(max(std_values)) if std_values else 0.0,
        }
        report[mode] = {"summary": mode_summary, "per_image": per_image_stats}
    output_path.write_text(json.dumps(report, indent=2))


def _load_thumbnail(entry: ArchiveEntry, thumb_size: int, cache: Dict[str, bytes]) -> bytes:
    if entry.image_id in cache:
        return cache[entry.image_id]
    image = Image.open(entry.image_path).convert("RGB")
    if image.width != thumb_size or image.height != thumb_size:
        resample = getattr(Image, "LANCZOS", Image.BICUBIC)
        image = image.resize((thumb_size, thumb_size), resample=resample)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    data = buffer.getvalue()
    cache[entry.image_id] = data
    return data


def _grid_cols(count: int) -> int:
    if count <= 0:
        return 1
    return max(1, min(count, int(math.ceil(math.sqrt(count)))))


def build_grid_image(
    batch: Sequence[ArchiveEntry],
    thumb_size: int,
    thumb_cache: Dict[str, bytes],
) -> Image.Image:
    count = len(batch)
    cols = _grid_cols(count)
    rows = math.ceil(count / cols)
    state: Dict[str, Any] = {
        "generation": 0,
        "rows": rows,
        "cols": cols,
        "thumbSize": thumb_size,
        "images": [],
    }
    for idx, entry in enumerate(batch):
        row = idx // cols
        col = idx % cols
        thumb_bytes = _load_thumbnail(entry, thumb_size, thumb_cache)
        encoded = base64.b64encode(thumb_bytes).decode("ascii")
        state["images"].append(
            {
                "index": idx,
                "row": row,
                "col": col,
                "width": thumb_size,
                "height": thumb_size,
                "data": encoded,
                "encoding": "png",
                "mode": "RGB",
            }
        )
    return create_numbered_grid(state)


def extract_json_payload(response_text: str) -> Optional[Any]:
    cleaned = response_text.strip()
    if not cleaned:
        return None
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    brace_start = cleaned.find("{")
    brace_end = cleaned.rfind("}")
    candidate = None
    if brace_start != -1 and brace_end != -1 and brace_end >= brace_start:
        candidate = cleaned[brace_start : brace_end + 1]
    else:
        bracket_start = cleaned.find("[")
        bracket_end = cleaned.rfind("]")
        if bracket_start != -1 and bracket_end != -1 and bracket_end >= bracket_start:
            candidate = cleaned[bracket_start : bracket_end + 1]

    payload_str = candidate or cleaned
    try:
        return json.loads(payload_str)
    except json.JSONDecodeError:
        return None


def _resolve_index_from_payload(obj: Dict[str, Any], batch: Sequence[ArchiveEntry]) -> Optional[int]:
    for key in ("index", "idx", "label"):
        if key in obj:
            try:
                return int(obj[key])
            except (TypeError, ValueError):
                continue
    id_value = obj.get("image_id") or obj.get("imageId") or obj.get("id")
    if isinstance(id_value, str):
        for idx, entry in enumerate(batch):
            if entry.image_id == id_value:
                return idx
    title_value = obj.get("title") or obj.get("name")
    if isinstance(title_value, str):
        lowered = title_value.strip().lower()
        for idx, entry in enumerate(batch):
            if entry.title.strip().lower() == lowered:
                return idx
    return None


def _resolve_score_from_payload(obj: Dict[str, Any]) -> Optional[float]:
    for key in ("score", "rating", "value"):
        if key in obj:
            try:
                value = float(obj[key])
            except (TypeError, ValueError):
                continue
            return float(min(5.0, max(1.0, value)))
    return None


def _resolve_justification(obj: Dict[str, Any]) -> Optional[str]:
    for key in ("justification", "reason", "rationale", "explanation"):
        value = obj.get(key)
        if isinstance(value, str):
            return value.strip()
    return None


def parse_rating_batch_response(response_text: str, batch: Sequence[ArchiveEntry]) -> Dict[int, RatingResult]:
    parsed: Dict[int, RatingResult] = {}
    payload = extract_json_payload(response_text)
    candidates: List[Any] = []
    if isinstance(payload, dict):
        if isinstance(payload.get("ratings"), list):
            candidates = payload["ratings"]
        elif any(key in payload for key in ("score", "rating", "value")):
            candidates = [payload]
    elif isinstance(payload, list):
        candidates = payload

    for item in candidates:
        if not isinstance(item, dict):
            continue
        idx = _resolve_index_from_payload(item, batch)
        score = _resolve_score_from_payload(item)
        if idx is None or score is None:
            continue
        justification = _resolve_justification(item)
        parsed[idx] = RatingResult(
            score=score,
            justification=justification,
            reported_title=(item.get("title") or item.get("name")),
        )

    if not parsed and len(batch) == 1:
        fallback = extract_score(response_text)
        if fallback is not None:
            parsed[0] = RatingResult(
                score=fallback,
                justification=response_text.strip(),
                reported_title=None,
            )

    return parsed


def main() -> None:
    args = parse_args()
    if args.resume_dir is not None:
        args.archive = args.resume_dir.parent.parent
    archive_dir = args.archive.expanduser().resolve()
    entries, goal_prompt = load_archive_entries(archive_dir)

    rng = random.Random(args.seed)
    if args.image_limit:
        limit = min(args.image_limit, len(entries))
        entries = rng.sample(entries, limit)

    if args.resume_dir:
        output_dir = args.resume_dir.expanduser().resolve()
        if not output_dir.exists():
            raise FileNotFoundError(f"Resume directory not found: {output_dir}")
    else:
        output_dir = ensure_output_dir(archive_dir, args.output_dir)
    input_mode_tag = "multi_part_captions\n" if args.use_multipart_input else "grid_numbered\n"
    (output_dir / "input_mode.txt").write_text(input_mode_tag, encoding="utf-8")
    log_path = output_dir / "vlm_query_log.jsonl"
    stats_path = output_dir / "ratings_summary.json"
    figure_path = output_dir / "ratings_figure.png"

    scores: Dict[str, List[float]] = defaultdict(list)
    mode_scores: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    thumb_cache: Dict[str, bytes] = {}
    image_bytes_cache: Dict[str, bytes] = {}
    title_mismatches: List[Dict[str, Any]] = []
    title_checks = 0
    if not entries:
        print("No images to rate.")
        return

    include_titles = args.include_titles_in_prompt or args.verify_titles

    existing_runs = 0
    if log_path.exists():
        prior_scores, existing_runs, prior_mode_scores = load_scores_from_log(log_path)
        for image_id, values in prior_scores.items():
            scores[image_id].extend(values)
        if prior_scores:
            print(f"Loaded {sum(len(v) for v in prior_scores.values())} existing scores from {log_path}")
        for mode, per_image in prior_mode_scores.items():
            for image_id, values in per_image.items():
                mode_scores[mode][image_id].extend(values)

    if args.verify_titles:
        existing_verification = output_dir / "title_verification.json"
        if existing_verification.exists():
            try:
                previous = json.loads(existing_verification.read_text())
                title_checks = int(previous.get("checks", title_checks))
                existing_mismatch_list = previous.get("mismatches", [])
                if isinstance(existing_mismatch_list, list):
                    title_mismatches = list(existing_mismatch_list)
            except (json.JSONDecodeError, ValueError, TypeError):
                pass

    target_run_count = max(args.runs, existing_runs)
    additional_runs = max(target_run_count - existing_runs, 0)
    if additional_runs == 0:
        print(
            f"No additional runs needed (already have {existing_runs}, target {args.runs})."
        )

    if args.batch_size is None or args.batch_size <= 0:
        effective_batch_size = len(entries)
    else:
        effective_batch_size = min(args.batch_size, len(entries))

    for run_idx in range(existing_runs, target_run_count):
        run_rng = random.Random(args.seed + run_idx)
        ordered_entries = entries[:]
        run_rng.shuffle(ordered_entries)
        batches = [
            ordered_entries[i : i + effective_batch_size]
            for i in range(0, len(ordered_entries), effective_batch_size)
        ]
        system_prompt = build_rating_system_prompt(
            batch,
            require_titles=args.verify_titles,
            goal_prompt=goal_prompt,
        )
        for batch_idx, batch in enumerate(batches):
            response_text = ""
            error_message = None
            ratings: Dict[int, RatingResult] = {}
            grid_path: Optional[Path] = None
            input_mode = "multi_part" if args.use_multipart_input else "grid"
            query_start = time.time()
            try:
                if args.use_multipart_input:
                    image_bytes_list: List[bytes] = []
                    captions_list: List[str] = []
                    for position, entry in enumerate(batch):
                        if entry.image_id not in image_bytes_cache:
                            image_bytes_cache[entry.image_id] = entry.image_path.read_bytes()
                        image_bytes_list.append(image_bytes_cache[entry.image_id])
                        captions_list.append(format_rating_entry_label(position, entry, include_titles))
                    response = query_images_with_captions(
                        image_bytes_list,
                        captions_list,
                        prompt=None,
                        system_instruction=system_prompt,
                    )
                    response_text = getattr(response, "text", "") or ""
                else:
                    grid_path = output_dir / f"grid_run{run_idx:03d}_batch{batch_idx:03d}.png"
                    grid_image = build_grid_image(batch, args.grid_thumb_size, thumb_cache)
                    grid_image.save(grid_path, format="PNG")
                    buffer = BytesIO()
                    grid_image.save(buffer, format="PNG")
                    response_text = query_vlm(buffer.getvalue(), prompt, args.system_instruction)
                ratings = parse_rating_batch_response(response_text, batch)
            except Exception as exc:  # pylint: disable=broad-except
                error_message = str(exc)
            query_time_sec = time.time() - query_start

            batch_entries: List[Dict[str, Any]] = []
            for position, entry in enumerate(batch):
                rating = ratings.get(position)
                score_value = rating.score if rating else None
                if score_value is not None:
                    scores[entry.image_id].append(score_value)
                    mode_scores[input_mode][entry.image_id].append(score_value)
                reported_title = rating.reported_title if rating else None
                batch_entry: Dict[str, Any] = {
                    "grid_index": position,
                    "image_id": entry.image_id,
                    "title": entry.title,
                    "score": score_value,
                    "justification": rating.justification if rating else None,
                    "reported_title": reported_title,
                }
                if args.verify_titles and rating is not None and score_value is not None:
                    title_checks += 1
                    expected_clean = entry.title.strip()
                    reported_clean = (reported_title or "").strip()
                    matches = bool(reported_clean) and reported_clean.lower() == expected_clean.lower()
                    batch_entry["title_match"] = matches
                    if not matches:
                        title_mismatches.append(
                            {
                                "run_index": run_idx,
                                "batch_index": batch_idx,
                                "grid_index": position,
                                "image_id": entry.image_id,
                                "expected_title": entry.title,
                                "reported_title": reported_title,
                            }
                        )
                batch_entries.append(batch_entry)

            log_record(
                log_path,
                {
                    "timestamp": datetime.now().isoformat(),
                    "run_index": run_idx,
                    "batch_index": batch_idx,
                    "batch_size": len(batch),
                    "prompt": prompt,
                    "response_text": response_text,
                    "error": error_message,
                    "model": DEFAULT_MODEL,
                    "grid_path": str(grid_path) if grid_path else None,
                    "input_mode": input_mode,
                    "query_time_sec": query_time_sec,
                    "entries": batch_entries,
                },
            )

    summary = summarize_scores(entries, scores)
    stats_path.write_text(json.dumps(summary, indent=2))
    if summary:
        render_ranked_figure(summary, figure_path)
        print(f"Saved ranked figure to {figure_path}")
    else:
        print("No successful scores recorded; skipping figure generation.")
    print(f"Wrote detailed logs to {log_path}")
    print(f"Saved summary statistics to {stats_path}")
    stability_path = output_dir / "stability_metrics.json"
    write_mode_variability_report(entries, mode_scores, stability_path)
    if stability_path.exists():
        print(f"Saved stability metrics to {stability_path}")
    if args.verify_titles:
        verification_path = output_dir / "title_verification.json"
        payload = {
            "checks": title_checks,
            "mismatch_count": len(title_mismatches),
            "mismatches": title_mismatches,
        }
        verification_path.write_text(json.dumps(payload, indent=2))
        print(f"Saved title verification report to {verification_path}")


if __name__ == "__main__":
    main()
