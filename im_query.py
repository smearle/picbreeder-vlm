import os
import time
from typing import List, Optional, Tuple

from google import genai
from google.genai import types
import dotenv


dotenv.load_dotenv()
api_key = os.environ.get('GEMINI_API_KEY')
client = genai.Client(api_key=api_key)

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


class ImageChatSession:
    """Maintains limited-turn chat history for Gemini image conversations."""

    def __init__(self, model: str = DEFAULT_MODEL, max_turns: Optional[int] = None) -> None:
        self.model = model
        self._turn_history: List[Tuple[bytes, str, str]] = []
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
        for image_data, prompt_text, response_text in self._turn_history[start_index:]:
            user_parts: List[types.Part] = []
            if prompt_text:
                user_parts.append(types.Part(text=prompt_text))
            user_parts.append(
                types.Part.from_bytes(
                    data=image_data,
                    mime_type="image/png",
                )
            )
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

    def send(
        self,
        image_bytes: bytes,
        prompt: str,
        *,
        history_turns: Optional[int] = 0,
        mime_type: str = "image/png",
        system_instruction: Optional[str] = None,
    ):
        requested_turns = self._resolve_requested_turns(history_turns)
        if requested_turns is None:
            start_index = 0
        else:
            start_index = max(len(self._turn_history) - requested_turns, 0)

        config = None
        if system_instruction:
            config = types.GenerateContentConfig(system_instruction=system_instruction)

        parts = []
        if prompt:
            parts.append(types.Part(text=prompt))
        parts.append(
            types.Part.from_bytes(
                data=image_bytes,
                mime_type=mime_type,
            )
        )

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
            except Exception as exc:
                if _is_token_limit_error(exc) and start_index < len(self._turn_history):
                    start_index += 1
                    continue
            except genai.errors.ServerError:
                time.sleep(10)
                continue

        response_text = getattr(response, "text", "") or ""
        if start_index > 0:
            self._turn_history = self._turn_history[start_index:]
        self._turn_history.append((bytes(image_bytes), prompt, response_text))
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
            parts=parts
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

if __name__ == '__main__':
    with open('rendered/rendered-34589-116.png', 'rb') as f:
        image_bytes = f.read()

    response = query_im(image_bytes, prompt="Describe this image in detail.")
    print(response.text)
