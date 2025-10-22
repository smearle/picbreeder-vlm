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


class ImageChatSession:
    """Maintains limited-turn chat history for Gemini image conversations."""

    def __init__(self, model: str = DEFAULT_MODEL, max_turns: Optional[int] = None) -> None:
        self.model = model
        self._turn_history: List[Tuple[str, str]] = []
        self._max_turns = max_turns if (max_turns is None or max_turns >= 0) else None

    def send(
        self,
        image_bytes: bytes,
        prompt: str,
        *,
        history_turns: int = 0,
        mime_type: str = "image/png",
    ):
        history: List[types.Content] = []
        if history_turns > 0 and self._turn_history:
            for prompt_text, response_text in self._turn_history[-history_turns:]:
                history.append(
                    types.Content(
                        role="user",
                        parts=[types.Part.from_text(prompt_text)],
                    )
                )
                history.append(
                    types.Content(
                        role="model",
                        parts=[types.Part.from_text(response_text)],
                    )
                )

        chat = client.chats.create(model=self.model, history=history or None)
        parts = [
            types.Part.from_bytes(
                data=image_bytes,
                mime_type=mime_type,
            ),
            types.Part.from_text(prompt),
        ]
        response = chat.send_message(parts)
        response_text = getattr(response, "text", "") or ""
        self._turn_history.append((prompt, response_text))
        if self._max_turns is not None:
            if self._max_turns == 0:
                self._turn_history.clear()
            else:
                self._turn_history = self._turn_history[-self._max_turns:]
        return response


def create_chat_session(model: str = DEFAULT_MODEL, *, max_turns: Optional[int] = None) -> ImageChatSession:
    return ImageChatSession(model=model, max_turns=max_turns)


def query_im(image_bytes, prompt="Caption this image.", mime_type="image/png"):
    contents = [
        types.Part.from_bytes(
            data=image_bytes,
            mime_type=mime_type,
        ),
        prompt,
    ]
    while True:
        try:
            response = client.models.generate_content(
                model=DEFAULT_MODEL,
                contents=contents,
            )
            return response
        except genai.errors.ServerError:
            time.sleep(10)

if __name__ == '__main__':
    with open('rendered/rendered-34589-116.png', 'rb') as f:
        image_bytes = f.read()

    response = query_im(image_bytes)
    print(response.text)
