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

import PIL
import graphviz 

REPO_ROOT = Path(__file__).resolve().parent

import neat
from rendering import create_numbered_grid, decode_image
import im_query  # type: ignore
from vlm_backends import create_vlm_backend, VLMBackend, VLMChatSession, is_local_model

DEFAULT_BASELINE_SELECTION_LIMIT = 1
_CHAT_SESSION: Optional[VLMChatSession] = None
_CHAT_SESSION_MAX_TURNS: Optional[int] = None
_CHAT_SESSION_MODEL: Optional[str] = None


class GeminiPromptBlockedError(RuntimeError):
    """Raised when Gemini explicitly blocks a prompt and returns no content."""

    def __init__(
        self,
        message: str,
        *,
        block_reason: Optional[str] = None,
        prompt_feedback: Optional[Dict[str, Any]] = None,
        attempts: Optional[int] = None,
        generation: Optional[int] = None,
    ) -> None:
        super().__init__(message)
        self.block_reason = block_reason
        self.prompt_feedback = prompt_feedback or {}
        self.attempts = attempts
        self.generation = generation


def _summarize_prompt_feedback(feedback: Any) -> Optional[Dict[str, Any]]:
    if feedback is None:
        return None

    summary: Dict[str, Any] = {}
    block_reason = getattr(feedback, "block_reason", None)
    if block_reason is not None:
        summary["block_reason"] = getattr(block_reason, "value", str(block_reason))

    block_message = getattr(feedback, "block_reason_message", None)
    if block_message:
        summary["block_reason_message"] = str(block_message)

    safety_ratings = getattr(feedback, "safety_ratings", None)
    if safety_ratings:
        rating_summaries: List[Dict[str, Any]] = []
        for rating in safety_ratings:
            rating_summary: Dict[str, Any] = {}
            category = getattr(rating, "category", None)
            if category is not None:
                rating_summary["category"] = getattr(category, "value", str(category))
            probability = getattr(rating, "probability", None)
            if probability is not None:
                rating_summary["probability"] = getattr(probability, "value", str(probability))
            blocked_flag = getattr(rating, "blocked", None)
            if blocked_flag is not None:
                rating_summary["blocked"] = bool(blocked_flag)
            if rating_summary:
                rating_summaries.append(rating_summary)
        if rating_summaries:
            summary["safety_ratings"] = rating_summaries

    return summary or None


def _extract_response_diagnostics(response: Any) -> Dict[str, Any]:
    diagnostics: Dict[str, Any] = {}
    prompt_feedback = _summarize_prompt_feedback(getattr(response, "prompt_feedback", None))
    if prompt_feedback:
        diagnostics["prompt_feedback"] = prompt_feedback

    finish_reasons: List[str] = []
    for candidate in getattr(response, "candidates", []) or []:
        reason = getattr(candidate, "finish_reason", None)
        if reason is not None:
            finish_reasons.append(getattr(reason, "value", str(reason)))
    if finish_reasons:
        diagnostics["finish_reasons"] = finish_reasons

    return diagnostics


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

    global _CHAT_SESSION, _CHAT_SESSION_MAX_TURNS, _CHAT_SESSION_MODEL
    _CHAT_SESSION = None
    _CHAT_SESSION_MAX_TURNS = None
    _CHAT_SESSION_MODEL = None


# Cache for VLM backends (avoid re-loading models)
_BACKEND_CACHE: Dict[str, VLMBackend] = {}


def _get_backend(model: str) -> VLMBackend:
    """Get or create a cached VLM backend."""
    if model not in _BACKEND_CACHE:
        _BACKEND_CACHE[model] = create_vlm_backend(model)
    return _BACKEND_CACHE[model]


def _ensure_chat_session(model: str, chat_history_turns: Optional[int]) -> VLMChatSession:
    global _CHAT_SESSION, _CHAT_SESSION_MAX_TURNS, _CHAT_SESSION_MODEL
    max_turns = _session_max_turns(chat_history_turns)
    if _CHAT_SESSION is None or _CHAT_SESSION_MAX_TURNS != max_turns or _CHAT_SESSION_MODEL != model:
        backend = _get_backend(model)
        _CHAT_SESSION = backend.create_chat_session(max_turns=max_turns)
        _CHAT_SESSION_MAX_TURNS = max_turns
        _CHAT_SESSION_MODEL = model
    return _CHAT_SESSION


def query_with_history(
    model: str,
    image_caption_pairs: Sequence[im_query.ImageCaptionInput],
    thinking_budget: int,
    prompt: str,
    system_instruction: Optional[str],
    chat_history_turns: Optional[int],
    temperature: float,
) -> Any:
    session = _ensure_chat_session(model, chat_history_turns)
    return session.send(
        image_caption_pairs=image_caption_pairs,
        thinking_budget=thinking_budget,
        prompt=prompt,
        history_turns=chat_history_turns,
        mime_type="image/png",
        system_instruction=system_instruction,
        temperature=temperature,
    )


def select_parents_from_grid(
    model: str,
    population_images: List[PIL.Image.Image],
    thinking_budget: int,
    temperature: float,
    generation: int,
    prompt_template: str,
    query_dir: Path,
    select_k: Optional[int] = None,
    system_instruction: Optional[str] = None,
    chat_history_turns: Optional[int] = 0,
    require_selection: bool = True,
    request_rationale: bool = True,
    allow_color_toggle: bool = False,
    current_color: Optional[bool] = None,
    view_index: Optional[int] = None,
    metadata_subdir: Optional[str] = None,
    image_path_map: Optional[Dict[int, Union[str, Path]]] = None,
    log_raw_response: bool = False,
    raw_response_dir: Optional[Path] = None,
    run_label: str = "",
) -> Dict[str, Any]:

    query_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_view_{view_index:02d}" if view_index is not None else ""
    name_prefix = f"{run_label}_" if run_label else ""

    metadata_dir = query_dir / "metadata"
    if metadata_subdir:
        metadata_dir = metadata_dir / metadata_subdir
    metadata_dir.mkdir(parents=True, exist_ok=True)
    raw_response_output_dir = raw_response_dir or (metadata_dir / "raw_responses")
    parts_dir: Optional[Path] = None
    if image_path_map is None:
        parts_dir = metadata_dir / "parts" / f"{name_prefix}gen_{generation:03d}{suffix}"
        parts_dir.mkdir(parents=True, exist_ok=True)

    image_caption_pairs: List[im_query.ImageCaptionInput] = []
    input_parts_metadata: List[Dict[str, Any]] = []
    for i, image in enumerate(population_images):
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        part_bytes = buffer.getvalue()
        caption = f"Image {i}"
        stored_image_path: Optional[str] = None
        image_caption_pairs.append((part_bytes, caption))
        input_parts_metadata.append(
            {
                "index": i,
                "caption": caption,
                "width": image.width,
                "height": image.height,
                "image_path": stored_image_path,
            }
        )

    total_images = len(population_images)
    max_index = max(total_images - 1, 0)
    base_prompt = prompt_template.format(generation=generation)
    max_history_turns = _session_max_turns(chat_history_turns)
    prompt = base_prompt
    max_index = i

    errors_dir = query_dir / "metadata" / "errors"
    max_attempts = 5
    response_text: str = ""
    parsed: Dict[str, Any] = {}
    raw_selected: Union[List[Any], None] = None
    cleaned: List[int] = []
    color_value: Optional[bool] = None
    quit_requested: bool = False
    quit_reason_value: Optional[str] = None

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
    last_error_reason: Optional[str] = None
    last_response_diagnostics: Optional[Dict[str, Any]] = None

    for attempt in range(1, max_attempts + 1):
        start_time = time.perf_counter()
        response = query_with_history(
            model,
            image_caption_pairs,
            thinking_budget=thinking_budget,
            prompt=prompt,
            system_instruction=system_instruction,
            chat_history_turns=chat_history_turns,
            temperature=temperature,
        )
        attempt_latencies.append(time.perf_counter() - start_time)
        response_diagnostics = _extract_response_diagnostics(response)
        prompt_feedback_details = response_diagnostics.get("prompt_feedback") if isinstance(response_diagnostics.get("prompt_feedback"), dict) else None
        block_reason = prompt_feedback_details.get("block_reason") if prompt_feedback_details else None
        response_text = getattr(response, "text", "") or ""
        if log_raw_response:
            raw_response_output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            raw_filename = raw_response_output_dir / f"{name_prefix}gen_{generation:03d}_attempt_{attempt:02d}_{timestamp}.txt"
            try:
                raw_filename.write_text(response_text, encoding="utf-8")
            except OSError:
                pass
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
            quit_requested_candidate = bool(
                _coerce_bool(
                    parsed.get("quit")
                    or parsed.get("terminate")
                    or parsed.get("stop")
                )
            )
            quit_reason_candidate = (
                parsed.get("quit_reason")
                or parsed.get("quit_rationale")
                or parsed.get("quit_message")
            )
            restart_requested = parsed.get("restart") is not None
            if color_toggle_requested:
                cleaned = []
            if (
                error_reason is None
                and require_selection
                and not cleaned
                and not color_toggle_requested
                and not quit_requested_candidate
                and not restart_requested
            ):
                error_reason = "Response did not contain any valid selection indices."
            color_toggle_only = error_reason is None and color_toggle_requested
            if error_reason is None:
                quit_requested = quit_requested_candidate
                quit_reason_value = quit_reason_candidate
        if error_reason is None and not response_text.strip():
            if block_reason:
                error_reason = f"Gemini blocked the prompt ({block_reason})."
            else:
                error_reason = "empty response"

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
        if response_diagnostics:
            error_payload["response_diagnostics"] = response_diagnostics

        error_path = errors_dir / f"{name_prefix}gen_{generation:03d}_attempt_{attempt:02d}.json"
        error_path.write_text(json.dumps(error_payload, indent=2), encoding="utf-8")

        last_error_reason = error_reason
        if response_diagnostics:
            last_response_diagnostics = response_diagnostics

        if attempt >= max_attempts:
            break

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
        schema_text = (
            '{"selected": [indices], "rationale": "brief explanation"}'
            if request_rationale
            else '{"selected": [indices]}'
        )
        correction_instructions.extend(
            [
                "Please reply with JSON only in the format "
                f"{schema_text}",
                f"Use zero-based numeric indices between 0 and {max_index}.",
            ]
        )
        if select_k is not None:
            correction_instructions.append(f"Select at most {select_k} unique indices.")
        correction_instructions.append("Do not include code fences or extra commentary.")
        if block_reason:
            correction_instructions.extend(
                [
                    "Safety note: these CPPN images are computer-generated abstract patterns with no real people or sensitive content.",
                    "It is safe to provide the requested JSON response describing which indices to explore next.",
                ]
            )
            if block_reason == "OTHER" and not response_text.strip():
                correction_instructions.extend(
                    [
                        "Your last reply came back empty because the model reported it was blocked for an OTHER reason.",
                        "If this happens again, it may be best to ask to quit the session so we can stop here.",
                    ]
                )

        if block_reason:
            time.sleep(min(2.0 * attempt, 6.0))

        prompt = (
            f"{base_prompt}\n\n"
            + "\n".join(correction_instructions)
        )

    else:
        block_reason = None
        prompt_feedback_details = None
        if isinstance(last_response_diagnostics, dict):
            prompt_feedback_details = last_response_diagnostics.get("prompt_feedback")
            if isinstance(prompt_feedback_details, dict):
                block_reason = prompt_feedback_details.get("block_reason")

        if block_reason:
            reset_chat_session()
            raise GeminiPromptBlockedError(
                f"Gemini blocked generation {generation} prompt after {max_attempts} attempts.",
                block_reason=block_reason,
                prompt_feedback=prompt_feedback_details,
                attempts=max_attempts,
                generation=generation,
            )

        error_detail = f" Last error: {last_error_reason}." if last_error_reason else ""
        raise ValueError(f"Failed to obtain a valid Gemini response.{error_detail}")

    if select_k is not None and not color_toggle_only:
        cleaned = cleaned[:select_k]

    selection_path: Optional[Path] = None

    metadata = {
        "selected": cleaned,
        "raw_selected": raw_selected,
        "rationale": parsed.get("rationale") or parsed.get("reason", ""),
        "response_text": response_text,
        "prompt": prompt,
        "generation": generation,
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
        "quit": quit_requested,
        "quit_reason": (str(quit_reason_value).strip() if quit_reason_value else None),
        "restart": parsed.get("restart"),
        "run_label": run_label or None,
    }

    return metadata


_METADATA_FILENAME_PATTERN = re.compile(
    r"(?:branch_\d+_)?gen_(\d+)(?:_view_(\d+))?_selection\.json$",
    re.IGNORECASE,
)

def _candidate_file_paths(path_value: str, *, meta_path: Path, query_dir: Path) -> List[Path]:
    raw_candidate = Path(path_value)
    candidates: List[Path] = [raw_candidate]
    if raw_candidate.is_absolute():
        return candidates

    possible_bases: List[Path] = []
    base_candidates = [
        meta_path.parent,
        meta_path.parent.parent,
        query_dir,
        query_dir.parent,
        query_dir.parent.parent,
    ]
    for base in base_candidates:
        if base is None:
            continue
        if base not in possible_bases:
            possible_bases.append(base)
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
        for meta_path in sorted(directory.glob("*gen_*_selection.json")):
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
    model: str,
    query_dir: Path,
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
                    raise ValueError(f"Could not restore image for chat history from metadata: {meta_path}")
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

    session = _ensure_chat_session(model, chat_history_turns)
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
