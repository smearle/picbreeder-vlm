"""A zero-cost mock VLM backend (`model=mock`).

It needs no API key, no GPU, and no model download: it inspects each prompt and
returns a well-formed JSON reply of the shape the agent loop expects
(`{"selected": [...]}`, plus occasional `publish`, and the `color` toggle). It is
meant for exercising the full evolution / archive / resume plumbing in tests, CI,
and quick local smoke-runs -- NOT for producing meaningful images.

Selections are deterministic per session (seeded by the session index) so runs are
reproducible. Use it like any other model:

    python evolve_collaborative.py model=mock num_agents=2 agent_generations=3
    # or to extend a published archive pulled from HF:
    python evolve_collaborative.py model=mock resume=true \
        experiment_dir=<reconstructed_run_dir> num_agents=<orig+N>
"""
from __future__ import annotations

import json
import random
from typing import Any, Iterable, List, Optional, Sequence

from picbreeder_vlm.vlm.vlm_backends import (
    HistoryTurnInput,
    ImageCaptionInput,
    StoredTurn,
    VLMBackend,
    VLMChatSession,
    VLMResponse,
)

# Distinctive phrase from the grayscale/colour toggle prompt (prompts.py).
_COLOR_TOGGLE_MARKER = "switch between color/grayscale"


def _build_reply(prompt: str, n_options: int, rng: random.Random, step: int) -> str:
    """Return a JSON string answering whatever the prompt is asking for."""
    p = (prompt or "").lower()

    # Colour/grayscale display toggle: expects only a {"color": bool} object.
    if _COLOR_TOGGLE_MARKER in p:
        return json.dumps({"color": False})

    # Everything else is a selection step (branch-from-archive or per-generation
    # selection). Pick one valid option; occasionally publish so a "continue" run
    # demonstrably grows the archive. n_options == 0 -> start from a fresh population.
    if n_options <= 0:
        return json.dumps({"selected": None, "rationale": "mock: fresh population"})

    idx = rng.randrange(n_options)
    reply: dict[str, Any] = {"selected": [idx], "rationale": "mock backend selection"}
    # Publish on every other selection step so at least one new entry is added.
    if step % 2 == 1:
        reply["publish"] = {"index": idx, "title": f"Mock Artifact {step}"}
    return json.dumps(reply)


class MockChatSession(VLMChatSession):
    def __init__(self, seed: int, max_turns: Optional[int] = None):
        self._rng = random.Random(seed)
        self._max_turns = max_turns
        self._turn_history: List[StoredTurn] = []
        self._step = 0

    def send(
        self,
        image_caption_pairs: Sequence[ImageCaptionInput],
        prompt: Optional[str] = "",
        history_turns: Optional[int] = 0,
        mime_type: str = "image/png",
        system_instruction: Optional[str] = None,
        temperature: Optional[float] = None,
        thinking_budget: int = -1,
    ) -> VLMResponse:
        pairs = list(image_caption_pairs or [])
        text = _build_reply(prompt or "", len(pairs), self._rng, self._step)
        self._step += 1
        stored_images = [(img, cap or "") for img, cap in pairs]
        self._turn_history.append((stored_images, prompt or "", text))
        return VLMResponse(text=text)

    def load_history(self, turns: Iterable[HistoryTurnInput]) -> int:
        count = 0
        for images, user_text, assistant_text in turns:
            stored = [(img, cap or "") for img, cap in images]
            self._turn_history.append((stored, user_text, assistant_text))
            count += 1
        return count

    @property
    def turn_history(self) -> List[StoredTurn]:
        return list(self._turn_history)


class MockBackend(VLMBackend):
    """Deterministic, dependency-free stand-in for a real VLM."""

    def __init__(self, seed: int = 0):
        self._seed = seed
        self._session_count = 0
        self._rng = random.Random(seed)

    @property
    def name(self) -> str:
        return "mock"

    def query(
        self,
        image_bytes: bytes,
        prompt: str,
        mime_type: str = "image/png",
        system_instruction: Optional[str] = None,
    ) -> VLMResponse:
        return VLMResponse(text=_build_reply(prompt, 1, self._rng, self._rng.randint(0, 1)))

    def query_multiple(
        self,
        image_bytes_list: Sequence[bytes],
        captions: Sequence[str],
        prompt: str,
        mime_type: str = "image/png",
        system_instruction: Optional[str] = None,
    ) -> VLMResponse:
        return VLMResponse(
            text=_build_reply(prompt, len(list(image_bytes_list)), self._rng, self._rng.randint(0, 1))
        )

    def create_chat_session(self, max_turns: Optional[int] = None) -> VLMChatSession:
        # Seed each session distinctly-but-deterministically for reproducible runs.
        self._session_count += 1
        return MockChatSession(seed=self._seed + self._session_count, max_turns=max_turns)
