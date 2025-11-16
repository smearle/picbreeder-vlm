import argparse
import base64
import binascii
import functools
import json
import os
import random
import re
import time
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from statistics import mean, pstdev

import sys

import graphviz 

REPO_ROOT = Path(__file__).resolve().parent

import neat
from rendering import create_numbered_grid, decode_image
import im_query  # type: ignore

DEFAULT_BASELINE_SELECTION_LIMIT = 1
_CHAT_SESSION: Optional[Any] = None
_CHAT_SESSION_MAX_TURNS: Optional[int] = None


def extract_json_object(text: str) -> Union[Dict[str, Any], ValueError]:
    cleaned = text.strip()
    if not cleaned:
        return ValueError("empty response")

    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    brace_start = cleaned.find("{")
    brace_end = cleaned.rfind("}")
    if brace_start != -1 and brace_end != -1 and brace_end >= brace_start:
        candidate = cleaned[brace_start : brace_end + 1]
    else:
        candidate = cleaned

    try:
        return json.loads(candidate)
    except json.decoder.JSONDecodeError:
        return ValueError("no valid JSON object found in response.")


def _session_max_turns(chat_history_turns: Optional[int]) -> Optional[int]:
    if chat_history_turns is None or chat_history_turns < 0:
        return None
    return chat_history_turns


def reset_chat_session() -> None:
    """Clear any cached chat session so the next request starts fresh."""

    global _CHAT_SESSION, _CHAT_SESSION_MAX_TURNS
    _CHAT_SESSION = None
    _CHAT_SESSION_MAX_TURNS = None


def _ensure_chat_session(chat_history_turns: Optional[int]) -> Any:
    global _CHAT_SESSION, _CHAT_SESSION_MAX_TURNS
    max_turns = _session_max_turns(chat_history_turns)
    if _CHAT_SESSION is None or _CHAT_SESSION_MAX_TURNS != max_turns:
        _CHAT_SESSION = im_query.create_chat_session(max_turns=max_turns)
        _CHAT_SESSION_MAX_TURNS = max_turns
    return _CHAT_SESSION


def query_with_history(
    image_caption_pairs: Sequence[im_query.ImageCaptionInput],
    prompt: str,
    system_instruction: Optional[str],
    chat_history_turns: Optional[int],
) -> Any:
    session: im_query.ImageChatSession = _ensure_chat_session(chat_history_turns)
    return session.send(
        image_caption_pairs=image_caption_pairs,
        prompt=prompt,
        history_turns=chat_history_turns,
        mime_type="image/png",
        system_instruction=system_instruction,
    )


def select_parents_from_grid(
    state: Dict[str, Any],
    prompt_template: str,
    query_dir: Path,
    select_k: Optional[int] = None,
    system_instruction: Optional[str] = None,
    chat_history_turns: Optional[int] = 0,
    require_selection: bool = True,
    allow_color_toggle: bool = False,
    current_color: Optional[bool] = None,
    view_index: Optional[int] = None,
    metadata_subdir: Optional[str] = None,
) -> Dict[str, Any]:
    generation = int(state["generation"])
    grid_image = create_numbered_grid(state)

    query_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_view_{view_index:02d}" if view_index is not None else ""

    metadata_dir = query_dir / "metadata"
    if metadata_subdir:
        metadata_dir = metadata_dir / metadata_subdir
    metadata_dir.mkdir(parents=True, exist_ok=True)
    parts_dir = metadata_dir / "parts" / f"gen_{generation:03d}{suffix}"
    parts_dir.mkdir(parents=True, exist_ok=True)

    base_grid_path = query_dir / f"gen_{generation:03d}{suffix}_grid.png"
    grid_image.save(base_grid_path, format="PNG")

    sorted_images = sorted(
        state["images"],
        key=lambda entry: int(entry.get("index", 0)),
    )
    state_variant = str(state.get("variant") or "") or None
    image_caption_pairs: List[im_query.ImageCaptionInput] = []
    input_parts_metadata: List[Dict[str, Any]] = []
    for entry in sorted_images:
        image = decode_image(entry)
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        part_bytes = buffer.getvalue()
        index_value = int(entry.get("index", len(image_caption_pairs)))
        title_value = str(entry.get("title") or "").strip()
        caption = f"Image {index_value}"
        if title_value:
            caption = f"{caption}: {title_value}"
        part_path = parts_dir / f"idx_{index_value:02d}.png"
        part_path.write_bytes(part_bytes)
        try:
            relative_part_path = part_path.relative_to(metadata_dir)
        except ValueError:
            relative_part_path = part_path
        image_caption_pairs.append((part_bytes, caption))
        input_parts_metadata.append(
            {
                "index": index_value,
                "caption": caption,
                "width": image.width,
                "height": image.height,
                "image_path": str(relative_part_path),
            }
        )

    total_images = len(state["images"])
    max_index = max(total_images - 1, 0)
    base_prompt = prompt_template.format(generation=generation)
    max_history_turns = _session_max_turns(chat_history_turns)
    prompt = base_prompt
    max_index = int(state["rows"]) * int(state["cols"]) - 1

    errors_dir = query_dir / "metadata" / "errors"
    max_attempts = 5
    response_text: str = ""
    parsed: Dict[str, Any] = {}
    raw_selected: Union[List[Any], None] = None
    cleaned: List[int] = []
    color_value: Optional[bool] = None

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

    color_value: Optional[bool] = None
    color_toggle_only = False

    attempt_latencies: List[float] = []

    for attempt in range(1, max_attempts + 1):
        start_time = time.perf_counter()
        response = query_with_history(
            image_caption_pairs,
            prompt=prompt,
            system_instruction=system_instruction,
            chat_history_turns=chat_history_turns,
        )
        attempt_latencies.append(time.perf_counter() - start_time)
        response_text = getattr(response, "text", "") or ""
        parse_result = extract_json_object(response_text)

        error_reason: Optional[str] = None
        if isinstance(parse_result, ValueError):
            error_reason = str(parse_result)
            parsed = {}
            raw_selected = None
        elif not isinstance(parse_result, dict):
            error_reason = "Response was not a JSON object."
            parsed = {}
            raw_selected = None
        else:
            parsed = parse_result
            raw_selected = parsed.get("selected")
            cleaned = []
            if isinstance(raw_selected, list):
                for value in raw_selected:
                    try:
                        idx = int(value)
                    except (TypeError, ValueError):
                        continue
                    if 0 <= idx <= max_index and idx not in cleaned:
                        cleaned.append(idx)
            elif raw_selected is None:
                cleaned = []
            else:
                error_reason = "Response missing 'selected' list."

            color_value = _coerce_bool(parsed.get("color"))
            color_toggle_requested = (
                allow_color_toggle
                and color_value is not None
                and (
                    current_color is None
                    or bool(color_value) != bool(current_color)
                )
            )
            if color_toggle_requested:
                cleaned = []
            if (
                error_reason is None
                and require_selection
                and not cleaned
                and not color_toggle_requested
            ):
                error_reason = "Response did not contain any valid selection indices."
            color_toggle_only = error_reason is None and color_toggle_requested

        if error_reason is None:
            break

        errors_dir.mkdir(parents=True, exist_ok=True)
        error_payload = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "generation": generation,
            "attempt": attempt,
            "prompt": prompt,
            "response_text": response_text,
            "error": error_reason,
        }
        if attempt_latencies:
            error_payload["latency_sec"] = attempt_latencies[-1]
        if isinstance(parsed, dict) and parsed:
            error_payload["parsed_json"] = parsed

        error_path = errors_dir / f"gen_{generation:03d}_attempt_{attempt:02d}.json"
        error_path.write_text(json.dumps(error_payload, indent=2), encoding="utf-8")

        if attempt >= max_attempts:
            raise ValueError(f"Exceeded {max_attempts} attempts while querying Gemini: {error_reason}")

        correction_instructions = [
            f"Your previous response could not be processed because: {error_reason}",
        ]
        include_previous_response = max_history_turns == 0
        if include_previous_response:
            correction_instructions.extend(
                [
                    "Previous response:",
                    response_text.strip() or "<empty response>",
                ]
            )
        correction_instructions.append("")
        correction_instructions.extend(
            [
                "Please reply with JSON only in the format "
                '{"selected": [indices], "rationale": "brief explanation"}',
                f"Use zero-based numeric indices between 0 and {max_index}.",
            ]
        )
        if select_k is not None:
            correction_instructions.append(f"Select at most {select_k} unique indices.")
        correction_instructions.append("Do not include code fences or extra commentary.")

        prompt = (
            f"{base_prompt}\n\n"
            + "\n".join(correction_instructions)
        )

    else:
        raise ValueError("Failed to obtain a valid Gemini response.")

    if select_k is not None and not color_toggle_only:
        cleaned = cleaned[:select_k]

    selection_path: Optional[Path] = None
    if not color_toggle_only:
        selection_image = create_numbered_grid(state, selected=cleaned)
        selection_path = query_dir / f"gen_{generation:03d}{suffix}_selection.png"
        selection_image.save(selection_path, format="PNG")

    metadata = {
        "selected": cleaned,
        "raw_selected": raw_selected,
        "rationale": parsed.get("rationale") or parsed.get("reason", ""),
        "response_text": response_text,
        "prompt": prompt,
        "generation": generation,
        "grid_path": str(base_grid_path),
        "selection_path": str(selection_path) if selection_path else None,
        "input_parts": input_parts_metadata,
        "select_k": select_k,
        "chat_history_turns": chat_history_turns,
        "mutation_mode": parsed.get("mutation_mode", None),
        "mutation_strength": parsed.get("mutation_strength", None),
        "color": color_value if color_value is not None else parsed.get("color", None),
        "view_index": view_index,
        "color_toggle_only": color_toggle_only,
        "response_attempts": attempt,
        "response_latencies_sec": attempt_latencies,
        "state_variant": state_variant,
    }
    meta_path = metadata_dir / f"gen_{generation:03d}{suffix}_selection.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    metadata["metadata_path"] = str(meta_path)

    return metadata


_METADATA_FILENAME_PATTERN = re.compile(
    r"gen_(\d+)(?:_view_(\d+))?_selection\.json$",
    re.IGNORECASE,
)

def _candidate_file_paths(path_value: str, *, meta_path: Path, query_dir: Path) -> List[Path]:
    raw_candidate = Path(path_value)
    candidates: List[Path] = [raw_candidate]
    if raw_candidate.is_absolute():
        return candidates

    possible_bases = [meta_path.parent, meta_path.parent.parent, query_dir]
    relative_variants = [raw_candidate, Path(raw_candidate.name)]

    for base in possible_bases:
        if base is None:
            continue
        for variant in relative_variants:
            candidate = (base / variant).resolve()
            if candidate not in candidates:
                candidates.append(candidate)

    return candidates


def _iter_query_metadata(query_dir: Path) -> List[Tuple[int, Dict[str, Any], Path]]:
    metadata_dir = query_dir / "metadata"
    if not metadata_dir.exists():
        return []

    candidate_dirs = [metadata_dir]
    views_dir = metadata_dir / "views"
    if views_dir.exists():
        candidate_dirs.append(views_dir)

    interim: List[Tuple[int, Optional[int], Dict[str, Any], Path]] = []
    for directory in candidate_dirs:
        for meta_path in sorted(directory.glob("gen_*_selection.json")):
            match = _METADATA_FILENAME_PATTERN.match(meta_path.name)
            generation_value: Optional[int]
            view_index: Optional[int]
            if match is not None:
                try:
                    generation_value = int(match.group(1))
                except ValueError:
                    generation_value = None
                try:
                    view_index = int(match.group(2)) if match.group(2) is not None else None
                except ValueError:
                    view_index = None
            else:
                generation_value = None
                view_index = None

            try:
                payload = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue

            if isinstance(payload, dict):
                if generation_value is None:
                    stored_generation = payload.get("generation")
                    generation_value = int(stored_generation) if isinstance(stored_generation, int) else None
                if generation_value is None:
                    continue
                interim.append((generation_value, view_index, payload, meta_path))

    interim.sort(
        key=lambda item: (
            item[0],
            item[1] if item[1] is not None else -1,
            item[3].stat().st_mtime,
            item[3].name,
        )
    )

    return [(generation, payload, meta_path) for generation, _, payload, meta_path in interim]


def restore_chat_history_from_metadata(
    query_dir: Path,
    *,
    chat_history_turns: Optional[int],
    prompt_template: Optional[str],
) -> int:
    """Rehydrate the Gemini chat session from saved selection metadata."""

    max_turns = _session_max_turns(chat_history_turns)
    if max_turns == 0:
        return 0

    records = _iter_query_metadata(query_dir)

    if not records:
        return 0

    turns: List[im_query.HistoryTurnInput] = []
    for generation_value, payload, meta_path in records:
        response_text = payload.get("response_text")
        if not response_text:
            continue

        prompt_text = payload.get("prompt")
        if not prompt_text and prompt_template:
            try:
                prompt_text = prompt_template.format(generation=generation_value)
            except (KeyError, ValueError):
                prompt_text = prompt_template

        input_parts_payload = payload.get("input_parts")
        if isinstance(input_parts_payload, list) and input_parts_payload:
            image_pairs: List[im_query.ImageCaptionInput] = []
            for part in input_parts_payload:
                image_b64 = part.get("image_b64")
                if not image_b64:
                    continue
                try:
                    image_bytes = base64.b64decode(image_b64)
                except (ValueError, binascii.Error):
                    continue
                image_bytes: Optional[bytes] = None
                image_path_value = part.get("image_path")
                if image_path_value:
                    for candidate in _candidate_file_paths(image_path_value, meta_path=meta_path, query_dir=query_dir):
                        try:
                            image_bytes = candidate.read_bytes()
                        except OSError:
                            continue
                        if image_bytes is not None:
                            break
                if image_bytes is None:
                    image_b64 = part.get("image_b64")
                    if image_b64:
                        try:
                            image_bytes = base64.b64decode(image_b64)
                        except (ValueError, binascii.Error):
                            image_bytes = None
                if image_bytes is None:
                    continue
                caption_text = str(part.get("caption") or "")
                image_pairs.append((image_bytes, caption_text))
            if image_pairs:
                turns.append((image_pairs, prompt_text or "", str(response_text)))
                continue

        grid_path_value = payload.get("grid_path")
        if not grid_path_value:
            continue

        image_bytes: Optional[bytes] = None
        for candidate in _candidate_file_paths(grid_path_value, meta_path=meta_path, query_dir=query_dir):
            try:
                image_bytes = candidate.read_bytes()
            except OSError:
                continue
            if image_bytes is not None:
                break

        if image_bytes is None:
            continue

        image_pairs: List[im_query.ImageCaptionInput] = [(image_bytes, prompt_text or "")]
        turns.append((image_pairs, "", str(response_text)))

    if not turns:
        return 0

    if max_turns is not None and max_turns > 0:
        turns = turns[-max_turns:]

    session = _ensure_chat_session(chat_history_turns)
    stored_turns = session.load_history(turns)
    return stored_turns


def summarize_genome_structure(genome: neat.DefaultGenome, genome_config: neat.genome.DefaultGenomeConfig) -> Dict[str, Any]:
    input_keys = set(getattr(genome_config, "input_keys", ()))
    output_keys = set(getattr(genome_config, "output_keys", ()))

    enabled_connections = [
        conn.key for conn in genome.connections.values() if getattr(conn, "enabled", False)
    ]

    depth_layers = []
    if enabled_connections:
        depth_layers = neat.graphs.feed_forward_layers(
            list(input_keys),
            list(output_keys),
            enabled_connections,
        )

    depth = len(depth_layers) if depth_layers else 0

    required_nodes: set = set()
    if enabled_connections:
        required_nodes = neat.graphs.required_for_output(
            list(input_keys),
            list(output_keys),
            enabled_connections,
        )

    active_hidden_nodes = len(required_nodes - output_keys)
    hidden_node_count = sum(1 for key in genome.nodes if key not in input_keys and key not in output_keys)

    return {
        "depth": depth,
        "active_hidden_nodes": active_hidden_nodes,
        "hidden_node_count": hidden_node_count,
        "enabled_connection_count": len(enabled_connections),
    }


def _summarize_numeric_collection(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"min": 0.0, "max": 0.0, "avg": 0.0, "std": 0.0}
    return {
        "min": float(min(values)),
        "max": float(max(values)),
        "avg": float(mean(values)),
        "std": float(pstdev(values)) if len(values) > 1 else 0.0,
    }


def compute_population_structure_stats(
    genomes: List[Tuple[int, neat.DefaultGenome]],
    genome_config: neat.genome.DefaultGenomeConfig,
) -> Dict[str, Dict[str, float]]:
    node_counts: List[int] = []
    depths: List[int] = []

    for _, genome in genomes:
        structure = summarize_genome_structure(genome, genome_config)
        node_counts.append(len(genome.nodes))
        depths.append(structure["depth"])

    return {
        "node_count": _summarize_numeric_collection([float(value) for value in node_counts]),
        "depth": _summarize_numeric_collection([float(value) for value in depths]),
    }


def ensure_gemini_key() -> None:
    if im_query is None:
        raise ImportError("im_query module is not available. Install required dependencies.")
    if not getattr(im_query, "api_key", None):
        raise EnvironmentError("Environment variable GEMINI_API_KEY is not set.")
