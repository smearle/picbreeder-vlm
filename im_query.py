import os
import re
import time
from typing import Iterable, List, Optional, Sequence, Tuple

from google import genai
from google.genai import types
import dotenv


dotenv.load_dotenv()
api_key = os.environ.get('GEMINI_API_KEY')
client = genai.Client(
    api_key=api_key,
    http_options={'api_version': 'v1alpha'}
)

DEFAULT_MODEL = 'gemini-2.5-pro'


def _is_token_limit_error(exc: Exception) -> bool:
    """Best-effort detection of Gemini token limit errors."""
    message = str(exc).lower()
    if "token" in message and ("limit" in message or "exceed" in message):
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


ImageCaptionInput = Tuple[bytes, Optional[str]]
ImageCaptionPair = Tuple[bytes, str]
StoredTurn = Tuple[List[ImageCaptionPair], str, str]
HistoryTurnInput = Tuple[Sequence[ImageCaptionInput], str, str]


class ImageChatSession:
    """Maintains limited-turn chat history for Gemini image conversations."""

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
            if self.model == 'gemini-3-pro-preview':
                extra_args["media_resolution"] = {"level": "media_resolution_high"}
            for image_data, caption_text in image_caption_pairs:
                if caption_text:
                    user_parts.append(types.Part(text=caption_text))
                user_parts.append(
                    # types.Part.from_bytes(
                    #     data=image_data,
                    #     mime_type="image/png",
                    # )
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
        requested_turns = self._resolve_requested_turns(history_turns)
        if requested_turns is None:
            start_index = 0
        else:
            start_index = max(len(self._turn_history) - requested_turns, 0)

        config = None
        config_kwargs = {}
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction
        if temperature is not None:
            config_kwargs["temperature"] = float(temperature)
        if config_kwargs:
            config = types.GenerateContentConfig(
                **config_kwargs,
                # thinking_config=types.ThinkingConfig(
                #     thinking_budget=thinking_budget
                # ),
            )

        pair_list = list(image_caption_pairs or ())
        if not pair_list:
            raise ValueError("image_caption_pairs must include at least one entry.")

        prompt_value = "" if prompt is None else str(prompt)

        normalized_pairs: List[Tuple[bytes, str]] = []
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

        parts: List[types.Part] = []
        stored_pairs: List[ImageCaptionPair] = []
        for idx, (image_data, caption_text) in enumerate(normalized_pairs):
            caption_to_use = caption_text or f"Image {idx + 1}:"
            if caption_to_use:
                parts.append(types.Part(text=caption_to_use))
            parts.append(
                # types.Part.from_bytes(
                #     data=image_data,
                #     mime_type=mime_type,
                # )
                types.Part(
                    inline_data=types.Blob(
                        mime_type=mime_type,
                        data=image_data,
                    ),
                )
            )
            stored_pairs.append((bytes(image_data), caption_to_use))

        if prompt_value:
            parts.append(types.Part(text=prompt_value))

        while True:
            history_contents = self._build_history_contents(start_index)
            create_kwargs = {"model": self.model}
            if history_contents:
                create_kwargs["history"] = history_contents
            if config is not None:
                create_kwargs["config"] = config
            chat = client.chats.create(**create_kwargs)
            try:
                response = chat.send_message(parts)
                break
            except genai.errors.ServerError:
                time.sleep(10)
                continue
            except genai.errors.ClientError as exc:
                if _is_rate_limit_error(exc):
                    delay = _extract_retry_delay_seconds(exc) or 5.0
                    time.sleep(delay)
                    continue
                if _is_token_limit_error(exc) and start_index < len(self._turn_history):
                    start_index += 1
                    continue
                raise
            except Exception as exc:
                if _is_token_limit_error(exc) and start_index < len(self._turn_history):
                    start_index += 1
                    continue
                raise

        response_text = getattr(response, "text", "") or ""
        if start_index > 0:
            self._turn_history = self._turn_history[start_index:]
        self._turn_history.append((stored_pairs, prompt_value, response_text))
        if self._max_turns is not None:
            if self._max_turns == 0:
                self._turn_history.clear()
            else:
                self._turn_history = self._turn_history[-self._max_turns:]
        return response


def create_chat_session(model: str = DEFAULT_MODEL, *, max_turns: Optional[int] = None) -> ImageChatSession:
    return ImageChatSession(model=model, max_turns=max_turns)


def query_im(image_bytes, prompt: str, mime_type="image/png", system_instruction: Optional[str] = None):
    parts = [
        types.Part.from_bytes(
            data=image_bytes,
            mime_type=mime_type,
        ),
    ]
    if prompt:
        parts.append(types.Part(text=prompt))
    config = types.GenerateContentConfig(
        system_instruction=system_instruction
    )
    contents = [
        types.Content(
            role="user",
            parts=parts,
            media_resolution={"level": "media_resolution_high"},
        )
    ]
    while True:
        try:
            request_kwargs = {
                "model": DEFAULT_MODEL,
                "contents": contents,
                "config": config,
            }
            response = client.models.generate_content(**request_kwargs)
            return response
        except genai.errors.ServerError:
            time.sleep(10)
        except genai.errors.ClientError as exc:
            if _is_rate_limit_error(exc):
                delay = _extract_retry_delay_seconds(exc) or 5.0
                time.sleep(delay)
                continue
            raise


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

    parts: List[types.Part] = []
    extra_args = {}
    if model == 'gemini-3-pro-preview':
        extra_args["media_resolution"] = {"level": "media_resolution_high"}
    for idx, (image_bytes, caption) in enumerate(zip(image_bytes_list, captions)):
        caption_text = caption or f"Image {idx + 1}:"
        parts.append(types.Part(text=caption_text))
        parts.append(
            # types.Part.from_bytes(
            #     data=image_bytes,
            #     mime_type=mime_type,
            # )
            types.Part(
                inline_data=types.Blob(
                    mime_type=mime_type,
                    data=image_bytes,
                    **extra_args,
                )
            )
        )

    if prompt:
        parts.append(types.Part(text=prompt))

    config = types.GenerateContentConfig(system_instruction=system_instruction)
    contents = [types.Content(role="user", parts=parts, **extra_args)]

    while True:
        try:
            request_kwargs = {
                "model": model,
                "contents": contents,
                "config": config,
            }
            response = client.models.generate_content(**request_kwargs)
            return response
        except genai.errors.ServerError:
            time.sleep(10)
        except genai.errors.ClientError as exc:
            if _is_rate_limit_error(exc):
                delay = _extract_retry_delay_seconds(exc) or 5.0
                time.sleep(delay)
                continue
            raise

if __name__ == '__main__':
    with open('rendered/rendered-34589-116.png', 'rb') as f:
        image_bytes = f.read()

    response = query_im(image_bytes, prompt="Describe this image in detail.")
    print(response.text)
