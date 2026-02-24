import base64
import json
import logging
import os
import re
import threading
import time
from functools import lru_cache
from queue import Empty, Queue
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, TypeVar

import dotenv
from google import genai
from google.genai import types


dotenv.load_dotenv()
logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-2.5-pro"
DEFAULT_VLM_API_PROVIDER = "portkey"
VLM_API_PROVIDER = os.environ.get("VLM_API_PROVIDER", DEFAULT_VLM_API_PROVIDER).strip().lower()

PORTKEY_BASE_URL = os.environ.get("PORTKEY_BASE_URL")
PORTKEY_MODEL_NAMESPACE = os.environ.get("PORTKEY_MODEL_NAMESPACE")

ImageCaptionInput = Tuple[bytes, Optional[str]]
ImageCaptionPair = Tuple[bytes, str]
StoredTurn = Tuple[List[ImageCaptionPair], str, str]
HistoryTurnInput = Tuple[Sequence[ImageCaptionInput], str, str]
T = TypeVar("T")


class VLMAPITimeoutError(TimeoutError):
    """Raised when an outbound VLM API call exceeds configured wall-clock timeout."""


class QueryResponse:
    """Small compatibility wrapper for backends that do not expose `.text`."""

    def __init__(self, text: str, raw_response: Any = None, provider: str = "") -> None:
        self.text = text
        self.raw_response = raw_response
        self.provider = provider


def get_vlm_provider() -> str:
    provider = VLM_API_PROVIDER or DEFAULT_VLM_API_PROVIDER
    provider = provider.strip().lower()
    if provider in {"google", "google-genai", "gemini"}:
        return "google-genai"
    if provider == "portkey":
        return "portkey"
    raise ValueError(
        f"Unsupported VLM_API_PROVIDER '{provider}'. Use 'portkey' or 'google-genai'."
    )


def set_vlm_provider(provider: str) -> str:
    """Set provider at runtime. Accepted values: portkey, google-genai."""
    global VLM_API_PROVIDER
    VLM_API_PROVIDER = provider
    return get_vlm_provider()


@lru_cache(maxsize=1)
def get_genai_client() -> genai.Client:
    """Create and cache a Gemini client on first use."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Configure it before running Gemini VLM queries."
        )
    return genai.Client(
        api_key=api_key,
        http_options={"api_version": "v1alpha"},
    )


@lru_cache(maxsize=1)
def get_portkey_client():
    """Create and cache a Portkey client on first use."""
    try:
        from portkey_ai import Portkey
    except ImportError as exc:
        raise RuntimeError(
            "portkey_ai is not installed. Install it to use VLM_API_PROVIDER=portkey."
        ) from exc

    api_key = os.environ.get("PORTKEY_BEARER") or os.environ.get("PORTKEY_API_KEY")
    if not api_key:
        raise RuntimeError(
            "PORTKEY_BEARER (or PORTKEY_API_KEY) is not set. Configure it before running Portkey VLM queries."
        )
    return Portkey(base_url=PORTKEY_BASE_URL, api_key=api_key)


def _resolve_portkey_model(model: str) -> str:
    if "/" in model or model.startswith("@"):
        return model
    return f"{PORTKEY_MODEL_NAMESPACE}/{model}"


def _is_token_limit_error(exc: Exception) -> bool:
    """Best-effort detection of token limit errors across providers."""
    message = str(exc).lower()
    if "token" in message and ("limit" in message or "exceed" in message or "maximum context length" in message):
        return True
    code = getattr(exc, "code", None)
    if isinstance(code, str) and code.lower() in {"400", "invalid_argument"} and "token" in message:
        return True
    return False


def _is_rate_limit_error(exc: Exception) -> bool:
    message = str(exc).lower()
    if "resource_exhausted" in message or "rate limit" in message or "quota" in message:
        return True
    code = getattr(exc, "code", None)
    if isinstance(code, int) and code == 429:
        return True
    if isinstance(code, str) and code.strip() == "429":
        return True
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int) and status_code == 429:
        return True
    return False


def _is_server_error(exc: Exception) -> bool:
    code = getattr(exc, "code", None)
    status_code = getattr(exc, "status_code", None)
    message = str(exc).lower()
    if isinstance(code, int) and 500 <= code < 600:
        return True
    if isinstance(status_code, int) and 500 <= status_code < 600:
        return True
    if "server error" in message or "internal error" in message:
        return True
    return False


def _extract_retry_delay_seconds(exc: Exception) -> Optional[float]:
    details = getattr(exc, "details", None)
    if isinstance(details, (list, tuple)):
        for item in details:
            if isinstance(item, dict) and item.get("@type", "").endswith("RetryInfo"):
                retry_delay = item.get("retryDelay")
                if isinstance(retry_delay, str):
                    match = re.search(r"(\d+)(?:\.(\d+))?s", retry_delay)
                    if match:
                        seconds = float(match.group(1))
                        if match.group(2):
                            seconds += float("0." + match.group(2))
                        return seconds
    message = str(exc)
    match = re.search(r"retry in\s+(\d+(?:\.\d+)?)s", message, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return None


def _truncate_exception_message(message: str, *, limit: int = 400) -> str:
    if len(message) <= limit:
        return message
    return message[: limit - 3] + "..."


@lru_cache(maxsize=1)
def _max_retry_attempts() -> Optional[int]:
    raw_value = os.environ.get("VLM_MAX_RETRY_ATTEMPTS")
    if raw_value is None or not str(raw_value).strip():
        return None
    try:
        parsed = int(raw_value)
    except ValueError:
        logger.warning("Ignoring invalid VLM_MAX_RETRY_ATTEMPTS value: %r", raw_value)
        return None
    return parsed if parsed > 0 else None


@lru_cache(maxsize=1)
def _retry_log_path() -> Optional[str]:
    value = os.environ.get("VLM_RETRY_LOG_PATH")
    if value is None:
        return None
    value = value.strip()
    return value or None


def _append_retry_event_to_file(payload: Dict[str, Any]) -> None:
    log_path = _retry_log_path()
    if not log_path:
        return
    try:
        with open(log_path, "a", encoding="utf-8") as fp:
            fp.write(json.dumps(payload, ensure_ascii=True))
            fp.write("\n")
    except OSError as file_error:
        logger.warning("Failed writing retry event to VLM_RETRY_LOG_PATH: %s", file_error)


@lru_cache(maxsize=1)
def _event_log_path() -> Optional[str]:
    explicit_path = os.environ.get("VLM_EVENT_LOG_PATH")
    if explicit_path and explicit_path.strip():
        return explicit_path.strip()
    retry_path = _retry_log_path()
    if not retry_path:
        return None
    parent = os.path.dirname(retry_path)
    if not parent:
        return "vlm_error_events.jsonl"
    return os.path.join(parent, "vlm_error_events.jsonl")


@lru_cache(maxsize=1)
def _api_timeout_seconds() -> float:
    raw_value = os.environ.get("VLM_API_TIMEOUT_SECONDS")
    if raw_value is None or not str(raw_value).strip():
        return 180.0
    try:
        parsed = float(raw_value)
    except ValueError:
        logger.warning("Ignoring invalid VLM_API_TIMEOUT_SECONDS value: %r", raw_value)
        return 180.0
    if parsed <= 0:
        logger.warning("Ignoring non-positive VLM_API_TIMEOUT_SECONDS value: %r", raw_value)
        return 180.0
    return parsed


def _append_event_to_file(payload: Dict[str, Any]) -> None:
    log_path = _event_log_path()
    if not log_path:
        return
    try:
        with open(log_path, "a", encoding="utf-8") as fp:
            fp.write(json.dumps(payload, ensure_ascii=True))
            fp.write("\n")
    except OSError as file_error:
        logger.warning("Failed writing event to VLM_EVENT_LOG_PATH: %s", file_error)


def _log_vlm_error_event(
    *,
    event_type: str,
    context: str,
    provider: str,
    model: str,
    error_category: str,
    exc: Exception,
) -> None:
    event = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "event": event_type,
        "context": context,
        "provider": provider,
        "model": model,
        "pid": os.getpid(),
        "error_category": error_category,
        "error_type": type(exc).__name__,
        "error": _truncate_exception_message(str(exc)),
    }
    _append_event_to_file(event)


def _call_with_timeout(
    *,
    call: Callable[[], T],
    context: str,
    provider: str,
    model: str,
) -> T:
    timeout_seconds = _api_timeout_seconds()
    result_queue: "Queue[Tuple[bool, Any]]" = Queue(maxsize=1)

    def _runner() -> None:
        try:
            result_queue.put((True, call()))
        except Exception as exc:  # pragma: no cover - API-layer passthrough
            result_queue.put((False, exc))

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    thread.join(timeout_seconds)
    if thread.is_alive():
        timeout_exc = VLMAPITimeoutError(
            f"{context} timed out after {timeout_seconds:.1f}s for {provider}:{model}."
        )
        _log_vlm_error_event(
            event_type="vlm_timeout",
            context=context,
            provider=provider,
            model=model,
            error_category="timeout",
            exc=timeout_exc,
        )
        raise timeout_exc

    try:
        ok, payload = result_queue.get_nowait()
    except Empty as exc:  # pragma: no cover - defensive guard
        unknown_exc = RuntimeError(f"{context} finished without returning a result.")
        _log_vlm_error_event(
            event_type="vlm_error",
            context=context,
            provider=provider,
            model=model,
            error_category="unknown",
            exc=unknown_exc,
        )
        raise unknown_exc from exc

    if ok:
        return payload
    raise payload


def _log_retry_event(
    *,
    context: str,
    provider: str,
    model: str,
    retry_count: int,
    retry_delay_seconds: float,
    category: str,
    exc: Exception,
) -> None:
    exc_message = _truncate_exception_message(str(exc))
    event = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "event": "vlm_retry",
        "context": context,
        "provider": provider,
        "model": model,
        "pid": os.getpid(),
        "retry_count": retry_count,
        "retry_delay_seconds": float(retry_delay_seconds),
        "error_category": category,
        "error_type": type(exc).__name__,
        "error": exc_message,
    }
    logger.warning(
        "[VLM RETRY] context=%s provider=%s model=%s pid=%s retry=%d delay=%.2fs category=%s error=%s",
        context,
        provider,
        model,
        event["pid"],
        retry_count,
        retry_delay_seconds,
        category,
        exc_message,
    )
    _append_retry_event_to_file(event)


def _apply_retry_delay_or_raise(
    *,
    context: str,
    provider: str,
    model: str,
    retry_count: int,
    default_delay_seconds: float,
    category: str,
    exc: Exception,
) -> int:
    delay = _extract_retry_delay_seconds(exc) or default_delay_seconds
    next_retry_count = retry_count + 1
    _log_retry_event(
        context=context,
        provider=provider,
        model=model,
        retry_count=next_retry_count,
        retry_delay_seconds=delay,
        category=category,
        exc=exc,
    )
    retry_limit = _max_retry_attempts()
    if retry_limit is not None and next_retry_count >= retry_limit:
        raise RuntimeError(
            f"{context} exceeded VLM retry limit ({retry_limit}) for {provider}:{model}."
        ) from exc
    time.sleep(delay)
    return next_retry_count


def _normalise_pairs(image_caption_pairs: Sequence[ImageCaptionInput]) -> List[ImageCaptionPair]:
    pair_list = list(image_caption_pairs or ())
    if not pair_list:
        raise ValueError("image_caption_pairs must include at least one entry.")

    normalized_pairs: List[ImageCaptionPair] = []
    for pair in pair_list:
        try:
            image_data, caption_text = pair
        except (TypeError, ValueError) as exc:
            raise ValueError("Each image_caption_pair must contain an image and caption.") from exc
        if image_data is None:
            raise ValueError("Image data must not be None.")
        caption_value = caption_text if caption_text is not None else ""
        if not isinstance(caption_value, str):
            caption_value = str(caption_value)
        normalized_pairs.append((bytes(image_data), caption_value))
    return normalized_pairs


def _encode_data_url(image_bytes: bytes, mime_type: str) -> str:
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _build_portkey_user_content(
    image_caption_pairs: Sequence[ImageCaptionPair],
    prompt: str,
    mime_type: str,
) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = []
    for idx, (image_data, caption_text) in enumerate(image_caption_pairs):
        caption_to_use = caption_text or f"Image {idx + 1}:"
        if caption_to_use:
            content.append({"type": "text", "text": caption_to_use})
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": _encode_data_url(image_data, mime_type),
                },
            }
        )
    if prompt:
        content.append({"type": "text", "text": prompt})
    return content


def _extract_portkey_response_text(response: Any) -> str:
    choices = getattr(response, "choices", None) or []
    if not choices:
        return ""
    first_choice = choices[0]
    message = getattr(first_choice, "message", None)
    if message is None and isinstance(first_choice, dict):
        message = first_choice.get("message")
    if message is None:
        return ""

    content = getattr(message, "content", None)
    if content is None and isinstance(message, dict):
        content = message.get("content")

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: List[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text" and item.get("text"):
                    chunks.append(str(item["text"]))
                elif isinstance(item.get("content"), str):
                    chunks.append(item["content"])
            elif isinstance(item, str):
                chunks.append(item)
        return "\n".join(chunk for chunk in chunks if chunk)
    return str(content or "")


class ImageChatSession:
    """Maintains limited-turn chat history for image conversations."""

    def __init__(self, model: str = DEFAULT_MODEL, max_turns: Optional[int] = None) -> None:
        self.model = model
        self._turn_history: List[StoredTurn] = []
        self._max_turns = max_turns if (max_turns is None or max_turns >= 0) else None

    def _resolve_requested_turns(self, history_turns: Optional[int]) -> Optional[int]:
        requested = history_turns if history_turns is not None else self._max_turns
        if requested is not None and requested < 0:
            return None
        return requested

    def _build_history_contents(self, start_index: int) -> List[types.Content]:
        contents: List[types.Content] = []
        if start_index < 0:
            start_index = 0
        for image_caption_pairs, trailing_prompt, response_text in self._turn_history[start_index:]:
            user_parts: List[types.Part] = []
            extra_args = {}
            if self.model == "gemini-3-pro-preview":
                extra_args["media_resolution"] = {"level": "media_resolution_high"}
            for image_data, caption_text in image_caption_pairs:
                if caption_text:
                    user_parts.append(types.Part(text=caption_text))
                user_parts.append(
                    types.Part(
                        inline_data=types.Blob(
                            mime_type="image/png",
                            data=image_data,
                        ),
                        **extra_args,
                    )
                )
            if trailing_prompt:
                user_parts.append(types.Part(text=trailing_prompt))
            contents.append(
                types.Content(
                    role="user",
                    parts=user_parts,
                )
            )
            if response_text:
                contents.append(
                    types.Content(
                        role="model",
                        parts=[types.Part(text=response_text)],
                    )
                )
        return contents

    def _build_portkey_messages(
        self,
        start_index: int,
        current_pairs: Sequence[ImageCaptionPair],
        prompt_value: str,
        system_instruction: Optional[str],
        mime_type: str,
    ) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})

        for image_caption_pairs, trailing_prompt, response_text in self._turn_history[start_index:]:
            user_content = _build_portkey_user_content(image_caption_pairs, trailing_prompt, mime_type)
            messages.append({"role": "user", "content": user_content})
            if response_text:
                messages.append({"role": "assistant", "content": response_text})

        current_user_content = _build_portkey_user_content(current_pairs, prompt_value, mime_type)
        messages.append({"role": "user", "content": current_user_content})
        return messages

    def load_history(self, turns: Iterable[HistoryTurnInput]) -> int:
        """Install a previously recorded sequence of multi-part chat turns."""
        normalised: List[StoredTurn] = []
        for turn in turns:
            try:
                image_payload, trailing_prompt, response_text = turn
            except (TypeError, ValueError):
                continue

            trailing_prompt_value = str(trailing_prompt or "")
            response_value = str(response_text or "")

            image_pairs: List[ImageCaptionPair] = []
            for pair in image_payload:
                try:
                    image_bytes, caption_text = pair
                except (TypeError, ValueError):
                    continue
                if image_bytes is None:
                    continue
                caption_value = caption_text if caption_text is not None else ""
                if not isinstance(caption_value, str):
                    caption_value = str(caption_value)
                image_pairs.append((bytes(image_bytes), caption_value))

            if not image_pairs:
                continue

            normalised.append((image_pairs, trailing_prompt_value, response_value))

        if self._max_turns is None:
            self._turn_history = normalised
        elif self._max_turns <= 0:
            self._turn_history = []
        else:
            self._turn_history = normalised[-self._max_turns :]
        return len(self._turn_history)

    def send(
        self,
        image_caption_pairs: Sequence[ImageCaptionInput],
        thinking_budget: int,
        prompt: Optional[str] = "",
        history_turns: Optional[int] = 0,
        mime_type: str = "image/png",
        system_instruction: Optional[str] = None,
        temperature: Optional[float] = None,
    ):
        del thinking_budget
        requested_turns = self._resolve_requested_turns(history_turns)
        if requested_turns is None:
            start_index = 0
        else:
            start_index = max(len(self._turn_history) - requested_turns, 0)

        prompt_value = "" if prompt is None else str(prompt)
        normalized_pairs = _normalise_pairs(image_caption_pairs)
        stored_pairs: List[ImageCaptionPair] = []
        for idx, (image_data, caption_text) in enumerate(normalized_pairs):
            caption_to_use = caption_text or f"Image {idx + 1}:"
            stored_pairs.append((bytes(image_data), caption_to_use))

        provider = get_vlm_provider()
        retry_count = 0

        while True:
            try:
                if provider == "portkey":
                    messages = self._build_portkey_messages(
                        start_index=start_index,
                        current_pairs=stored_pairs,
                        prompt_value=prompt_value,
                        system_instruction=system_instruction,
                        mime_type=mime_type,
                    )
                    request_kwargs: Dict[str, Any] = {
                        "model": _resolve_portkey_model(self.model),
                        "messages": messages,
                    }
                    if temperature is not None:
                        request_kwargs["temperature"] = float(temperature)
                    raw_response = _call_with_timeout(
                        call=lambda: get_portkey_client().chat.completions.create(**request_kwargs),
                        context="ImageChatSession.send",
                        provider=provider,
                        model=self.model,
                    )
                    response = QueryResponse(
                        text=_extract_portkey_response_text(raw_response),
                        raw_response=raw_response,
                        provider="portkey",
                    )
                else:
                    config = None
                    config_kwargs = {}
                    if system_instruction:
                        config_kwargs["system_instruction"] = system_instruction
                    if temperature is not None:
                        config_kwargs["temperature"] = float(temperature)
                    if config_kwargs:
                        config = types.GenerateContentConfig(**config_kwargs)

                    history_contents = self._build_history_contents(start_index)
                    create_kwargs: Dict[str, Any] = {"model": self.model}
                    if history_contents:
                        create_kwargs["history"] = history_contents
                    if config is not None:
                        create_kwargs["config"] = config
                    chat = get_genai_client().chats.create(**create_kwargs)

                    parts: List[types.Part] = []
                    for caption_to_use, image_data in [
                        (caption, image) for image, caption in stored_pairs
                    ]:
                        if caption_to_use:
                            parts.append(types.Part(text=caption_to_use))
                        parts.append(
                            types.Part(
                                inline_data=types.Blob(
                                    mime_type=mime_type,
                                    data=image_data,
                                ),
                            )
                        )
                    if prompt_value:
                        parts.append(types.Part(text=prompt_value))
                    response = _call_with_timeout(
                        call=lambda: chat.send_message(parts),
                        context="ImageChatSession.send",
                        provider=provider,
                        model=self.model,
                    )
                if retry_count > 0:
                    logger.info(
                        "[VLM RETRY RECOVERED] context=ImageChatSession.send provider=%s model=%s retries=%d pid=%s",
                        provider,
                        self.model,
                        retry_count,
                        os.getpid(),
                    )
                break
            except VLMAPITimeoutError as exc:
                retry_count = _apply_retry_delay_or_raise(
                    context="ImageChatSession.send",
                    provider=provider,
                    model=self.model,
                    retry_count=retry_count,
                    default_delay_seconds=5.0,
                    category="timeout",
                    exc=exc,
                )
                continue
            except genai.errors.ServerError as exc:
                retry_count = _apply_retry_delay_or_raise(
                    context="ImageChatSession.send",
                    provider=provider,
                    model=self.model,
                    retry_count=retry_count,
                    default_delay_seconds=10.0,
                    category="server_error",
                    exc=exc,
                )
                continue
            except genai.errors.ClientError as exc:
                if _is_rate_limit_error(exc):
                    retry_count = _apply_retry_delay_or_raise(
                        context="ImageChatSession.send",
                        provider=provider,
                        model=self.model,
                        retry_count=retry_count,
                        default_delay_seconds=5.0,
                        category="rate_limit",
                        exc=exc,
                    )
                    continue
                if _is_token_limit_error(exc) and start_index < len(self._turn_history):
                    logger.warning(
                        "[VLM TOKEN LIMIT] context=ImageChatSession.send provider=%s model=%s pid=%s trimming_history_from=%d",
                        provider,
                        self.model,
                        os.getpid(),
                        start_index,
                    )
                    start_index += 1
                    continue
                _log_vlm_error_event(
                    event_type="vlm_error",
                    context="ImageChatSession.send",
                    provider=provider,
                    model=self.model,
                    error_category="client_error",
                    exc=exc,
                )
                raise
            except Exception as exc:
                if _is_rate_limit_error(exc):
                    retry_count = _apply_retry_delay_or_raise(
                        context="ImageChatSession.send",
                        provider=provider,
                        model=self.model,
                        retry_count=retry_count,
                        default_delay_seconds=5.0,
                        category="rate_limit",
                        exc=exc,
                    )
                    continue
                if _is_server_error(exc):
                    retry_count = _apply_retry_delay_or_raise(
                        context="ImageChatSession.send",
                        provider=provider,
                        model=self.model,
                        retry_count=retry_count,
                        default_delay_seconds=5.0,
                        category="server_error",
                        exc=exc,
                    )
                    continue
                if _is_token_limit_error(exc) and start_index < len(self._turn_history):
                    logger.warning(
                        "[VLM TOKEN LIMIT] context=ImageChatSession.send provider=%s model=%s pid=%s trimming_history_from=%d",
                        provider,
                        self.model,
                        os.getpid(),
                        start_index,
                    )
                    start_index += 1
                    continue
                _log_vlm_error_event(
                    event_type="vlm_error",
                    context="ImageChatSession.send",
                    provider=provider,
                    model=self.model,
                    error_category="unhandled_error",
                    exc=exc,
                )
                raise

        response_text = getattr(response, "text", "") or ""
        if start_index > 0:
            self._turn_history = self._turn_history[start_index:]
        self._turn_history.append((stored_pairs, prompt_value, response_text))
        if self._max_turns is not None:
            if self._max_turns == 0:
                self._turn_history.clear()
            else:
                self._turn_history = self._turn_history[-self._max_turns :]
        return response


def _query_multimodal(
    image_caption_pairs: Sequence[ImageCaptionInput],
    prompt: str,
    *,
    mime_type: str = "image/png",
    system_instruction: Optional[str] = None,
    model: str = DEFAULT_MODEL,
):
    normalized_pairs = _normalise_pairs(image_caption_pairs)
    prompt_value = "" if prompt is None else str(prompt)
    provider = get_vlm_provider()
    retry_count = 0

    while True:
        try:
            if provider == "portkey":
                messages: List[Dict[str, Any]] = []
                if system_instruction:
                    messages.append({"role": "system", "content": system_instruction})
                user_content = _build_portkey_user_content(normalized_pairs, prompt_value, mime_type)
                messages.append({"role": "user", "content": user_content})
                raw_response = _call_with_timeout(
                    call=lambda: get_portkey_client().chat.completions.create(
                        model=_resolve_portkey_model(model),
                        messages=messages,
                    ),
                    context="_query_multimodal",
                    provider=provider,
                    model=model,
                )
                return QueryResponse(
                    text=_extract_portkey_response_text(raw_response),
                    raw_response=raw_response,
                    provider="portkey",
                )

            parts: List[types.Part] = []
            extra_args = {}
            if model == "gemini-3-pro-preview":
                extra_args["media_resolution"] = {"level": "media_resolution_high"}
            for idx, (image_bytes, caption) in enumerate(normalized_pairs):
                caption_text = caption or f"Image {idx + 1}:"
                parts.append(types.Part(text=caption_text))
                parts.append(
                    types.Part(
                        inline_data=types.Blob(
                            mime_type=mime_type,
                            data=image_bytes,
                            **extra_args,
                        )
                    )
                )
            if prompt_value:
                parts.append(types.Part(text=prompt_value))

            config = types.GenerateContentConfig(system_instruction=system_instruction)
            contents = [types.Content(role="user", parts=parts, **extra_args)]
            request_kwargs = {
                "model": model,
                "contents": contents,
                "config": config,
            }
            response = _call_with_timeout(
                call=lambda: get_genai_client().models.generate_content(**request_kwargs),
                context="_query_multimodal",
                provider=provider,
                model=model,
            )
            if retry_count > 0:
                logger.info(
                    "[VLM RETRY RECOVERED] context=_query_multimodal provider=%s model=%s retries=%d pid=%s",
                    provider,
                    model,
                    retry_count,
                    os.getpid(),
                )
            return response
        except VLMAPITimeoutError as exc:
            retry_count = _apply_retry_delay_or_raise(
                context="_query_multimodal",
                provider=provider,
                model=model,
                retry_count=retry_count,
                default_delay_seconds=5.0,
                category="timeout",
                exc=exc,
            )
            continue
        except genai.errors.ServerError as exc:
            retry_count = _apply_retry_delay_or_raise(
                context="_query_multimodal",
                provider=provider,
                model=model,
                retry_count=retry_count,
                default_delay_seconds=10.0,
                category="server_error",
                exc=exc,
            )
        except genai.errors.ClientError as exc:
            if _is_rate_limit_error(exc):
                retry_count = _apply_retry_delay_or_raise(
                    context="_query_multimodal",
                    provider=provider,
                    model=model,
                    retry_count=retry_count,
                    default_delay_seconds=5.0,
                    category="rate_limit",
                    exc=exc,
                )
                continue
            _log_vlm_error_event(
                event_type="vlm_error",
                context="_query_multimodal",
                provider=provider,
                model=model,
                error_category="client_error",
                exc=exc,
            )
            raise
        except Exception as exc:
            if _is_rate_limit_error(exc):
                retry_count = _apply_retry_delay_or_raise(
                    context="_query_multimodal",
                    provider=provider,
                    model=model,
                    retry_count=retry_count,
                    default_delay_seconds=5.0,
                    category="rate_limit",
                    exc=exc,
                )
                continue
            if _is_server_error(exc):
                retry_count = _apply_retry_delay_or_raise(
                    context="_query_multimodal",
                    provider=provider,
                    model=model,
                    retry_count=retry_count,
                    default_delay_seconds=5.0,
                    category="server_error",
                    exc=exc,
                )
                continue
            _log_vlm_error_event(
                event_type="vlm_error",
                context="_query_multimodal",
                provider=provider,
                model=model,
                error_category="unhandled_error",
                exc=exc,
            )
            raise


def create_chat_session(model: str = DEFAULT_MODEL, *, max_turns: Optional[int] = None) -> ImageChatSession:
    return ImageChatSession(model=model, max_turns=max_turns)


def query_im(
    image_bytes: bytes,
    prompt: str,
    mime_type: str = "image/png",
    system_instruction: Optional[str] = None,
    model: str = DEFAULT_MODEL,
):
    return _query_multimodal(
        image_caption_pairs=[(image_bytes, "Image 1:")],
        prompt=prompt,
        mime_type=mime_type,
        system_instruction=system_instruction,
        model=model,
    )


def query_images_with_captions(
    image_bytes_list: Sequence[bytes],
    captions: Sequence[str],
    prompt: str,
    mime_type: str = "image/png",
    system_instruction: Optional[str] = None,
    model: str = DEFAULT_MODEL,
):
    """
    Query the model with multiple images and interleaved captions.

    Each caption is paired with the image at the same index and fed as a multi-part
    request: [caption_1, image_1, caption_2, image_2, ..., prompt].
    """
    if len(image_bytes_list) != len(captions):
        raise ValueError("image_bytes_list and captions must be the same length.")
    if not image_bytes_list:
        raise ValueError("At least one image must be provided.")

    pairs: List[ImageCaptionInput] = []
    for image_bytes, caption in zip(image_bytes_list, captions):
        pairs.append((image_bytes, caption))
    return _query_multimodal(
        image_caption_pairs=pairs,
        prompt=prompt,
        mime_type=mime_type,
        system_instruction=system_instruction,
        model=model,
    )


def query_text(
    prompt: str,
    system_instruction: Optional[str] = None,
    model: str = DEFAULT_MODEL,
):
    provider = get_vlm_provider()
    prompt_value = "" if prompt is None else str(prompt)
    retry_count = 0

    while True:
        try:
            if provider == "portkey":
                messages: List[Dict[str, Any]] = []
                if system_instruction:
                    messages.append({"role": "system", "content": system_instruction})
                messages.append({"role": "user", "content": prompt_value})
                raw_response = _call_with_timeout(
                    call=lambda: get_portkey_client().chat.completions.create(
                        model=_resolve_portkey_model(model),
                        messages=messages,
                    ),
                    context="query_text",
                    provider=provider,
                    model=model,
                )
                return QueryResponse(
                    text=_extract_portkey_response_text(raw_response),
                    raw_response=raw_response,
                    provider="portkey",
                )

            content_parts = [types.Part(text=prompt_value)]
            config = types.GenerateContentConfig(system_instruction=system_instruction)
            response = _call_with_timeout(
                call=lambda: get_genai_client().models.generate_content(
                    model=model,
                    contents=[types.Content(role="user", parts=content_parts)],
                    config=config,
                ),
                context="query_text",
                provider=provider,
                model=model,
            )
            if retry_count > 0:
                logger.info(
                    "[VLM RETRY RECOVERED] context=query_text provider=%s model=%s retries=%d pid=%s",
                    provider,
                    model,
                    retry_count,
                    os.getpid(),
                )
            return response
        except VLMAPITimeoutError as exc:
            retry_count = _apply_retry_delay_or_raise(
                context="query_text",
                provider=provider,
                model=model,
                retry_count=retry_count,
                default_delay_seconds=5.0,
                category="timeout",
                exc=exc,
            )
            continue
        except genai.errors.ServerError as exc:
            retry_count = _apply_retry_delay_or_raise(
                context="query_text",
                provider=provider,
                model=model,
                retry_count=retry_count,
                default_delay_seconds=10.0,
                category="server_error",
                exc=exc,
            )
        except genai.errors.ClientError as exc:
            if _is_rate_limit_error(exc):
                retry_count = _apply_retry_delay_or_raise(
                    context="query_text",
                    provider=provider,
                    model=model,
                    retry_count=retry_count,
                    default_delay_seconds=5.0,
                    category="rate_limit",
                    exc=exc,
                )
                continue
            _log_vlm_error_event(
                event_type="vlm_error",
                context="query_text",
                provider=provider,
                model=model,
                error_category="client_error",
                exc=exc,
            )
            raise
        except Exception as exc:
            if _is_rate_limit_error(exc):
                retry_count = _apply_retry_delay_or_raise(
                    context="query_text",
                    provider=provider,
                    model=model,
                    retry_count=retry_count,
                    default_delay_seconds=5.0,
                    category="rate_limit",
                    exc=exc,
                )
                continue
            if _is_server_error(exc):
                retry_count = _apply_retry_delay_or_raise(
                    context="query_text",
                    provider=provider,
                    model=model,
                    retry_count=retry_count,
                    default_delay_seconds=5.0,
                    category="server_error",
                    exc=exc,
                )
                continue
            _log_vlm_error_event(
                event_type="vlm_error",
                context="query_text",
                provider=provider,
                model=model,
                error_category="unhandled_error",
                exc=exc,
            )
            raise


if __name__ == "__main__":
    with open("rendered/rendered-34589-116.png", "rb") as f:
        image_bytes = f.read()

    response = query_im(image_bytes, prompt="Describe this image in detail.")
    print(response.text)
