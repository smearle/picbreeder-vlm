"""
Unified VLM backend interface for picbreeder-vlm.

Supports multiple vision-language models through a common interface:
- Gemini (Google's API)
- Qwen3-VL (local HuggingFace model)

Usage:
    from vlm_backends import create_vlm_backend, VLMBackend

    # Create backend by name
    backend = create_vlm_backend("gemini-2.5-pro")  # or "qwen3-vl-8b"
    
    # Query with image
    response = backend.query(image_bytes, prompt="Describe this image.")
    print(response.text)
    
    # Chat session with history
    session = backend.create_chat_session(max_turns=5)
    response = session.send([(image_bytes, "Image 1:")], prompt="What do you see?")
"""

import importlib.util
import json
import os
import re
import requests
import httpx
import time
from abc import ABC, abstractmethod
from io import BytesIO
from typing import Any, Iterable, List, Optional, Sequence, Tuple, Union

import PIL.Image

from constants import MAX_MODEL_LEN
import im_query

# Type aliases
ImageCaptionInput = Tuple[bytes, Optional[str]]
ImageCaptionPair = Tuple[bytes, str]
StoredTurn = Tuple[List[ImageCaptionPair], str, str]
HistoryTurnInput = Tuple[Sequence[ImageCaptionInput], str, str]


class VLMResponse:
    """Unified response object from VLM backends."""
    
    def __init__(self, text: str, raw_response: Any = None):
        self._text = text
        self._raw_response = raw_response
    
    @property
    def text(self) -> str:
        return self._text
    
    @property
    def raw(self) -> Any:
        """Access the raw backend-specific response object."""
        return self._raw_response


class VLMBackend(ABC):
    """Abstract base class for VLM backends."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the backend/model name."""
        pass
    
    @abstractmethod
    def query(
        self,
        image_bytes: bytes,
        prompt: str,
        mime_type: str = "image/png",
        system_instruction: Optional[str] = None,
    ) -> VLMResponse:
        """Query the model with a single image and prompt."""
        pass
    
    @abstractmethod
    def query_multiple(
        self,
        image_bytes_list: Sequence[bytes],
        captions: Sequence[str],
        prompt: str,
        mime_type: str = "image/png",
        system_instruction: Optional[str] = None,
    ) -> VLMResponse:
        """Query the model with multiple images and interleaved captions."""
        pass
    
    @abstractmethod
    def create_chat_session(self, max_turns: Optional[int] = None) -> "VLMChatSession":
        """Create a chat session with conversation history support."""
        pass


class VLMChatSession(ABC):
    """Abstract base class for VLM chat sessions with history."""
    
    @abstractmethod
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
        """Send a message with images and get a response."""
        pass
    
    @abstractmethod
    def load_history(self, turns: Iterable[HistoryTurnInput]) -> int:
        """Load previous conversation history."""
        pass
    
    @property
    @abstractmethod
    def turn_history(self) -> List[StoredTurn]:
        """Access the current turn history."""
        pass


class ProviderAwareGeminiChatSession(VLMChatSession):
    """Gemini chat session delegated to im_query provider routing."""

    def __init__(self, model: str, max_turns: Optional[int] = None):
        self._session = im_query.create_chat_session(model=model, max_turns=max_turns)

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
        response = self._session.send(
            image_caption_pairs=image_caption_pairs,
            prompt=prompt,
            history_turns=history_turns,
            mime_type=mime_type,
            system_instruction=system_instruction,
            temperature=temperature,
            thinking_budget=thinking_budget,
        )
        return VLMResponse(text=getattr(response, "text", "") or "", raw_response=response)

    def load_history(self, turns: Iterable[HistoryTurnInput]) -> int:
        return self._session.load_history(turns)

    @property
    def turn_history(self) -> List[StoredTurn]:
        return list(getattr(self._session, "_turn_history", []))


class PortkeyGeminiBackend(VLMBackend):
    """Gemini models routed through Portkey via im_query."""

    def __init__(self, model: str = "gemini-2.5-pro", **_unused_kwargs):
        self._model = model

    @property
    def name(self) -> str:
        return self._model

    def query(
        self,
        image_bytes: bytes,
        prompt: str,
        mime_type: str = "image/png",
        system_instruction: Optional[str] = None,
    ) -> VLMResponse:
        response = im_query.query_im(
            image_bytes=image_bytes,
            prompt=prompt,
            mime_type=mime_type,
            system_instruction=system_instruction,
            model=self._model,
        )
        return VLMResponse(text=getattr(response, "text", "") or "", raw_response=response)

    def query_multiple(
        self,
        image_bytes_list: Sequence[bytes],
        captions: Sequence[str],
        prompt: str,
        mime_type: str = "image/png",
        system_instruction: Optional[str] = None,
    ) -> VLMResponse:
        response = im_query.query_images_with_captions(
            image_bytes_list=image_bytes_list,
            captions=captions,
            prompt=prompt,
            mime_type=mime_type,
            system_instruction=system_instruction,
            model=self._model,
        )
        return VLMResponse(text=getattr(response, "text", "") or "", raw_response=response)

    def create_chat_session(self, max_turns: Optional[int] = None) -> "ProviderAwareGeminiChatSession":
        return ProviderAwareGeminiChatSession(model=self._model, max_turns=max_turns)


def _create_gemini_backend(model: str, **kwargs) -> VLMBackend:
    if "max_model_len" in kwargs:
        kwargs.pop("max_model_len")
    provider = im_query.get_vlm_provider()
    if provider == "portkey":
        return PortkeyGeminiBackend(model, **kwargs)
    return GeminiBackend(model, **kwargs)


class GeminiBackend(VLMBackend):
    """Google Gemini API backend."""
    
    def __init__(self, model: str = "gemini-2.5-pro"):
        self._model = model
        self._client = None
        self._types = None
    
    def _ensure_client(self):
        if self._client is None:
            import dotenv
            from google import genai
            from google.genai import types
            
            dotenv.load_dotenv()
            api_key = os.environ.get('GEMINI_API_KEY')
            self._client = genai.Client(
                api_key=api_key,
                http_options={'api_version': 'v1alpha'}
            )
            self._types = types
            self._genai = genai
    
    @property
    def name(self) -> str:
        return self._model
    
    def query(
        self,
        image_bytes: bytes,
        prompt: str,
        mime_type: str = "image/png",
        system_instruction: Optional[str] = None,
    ) -> VLMResponse:
        self._ensure_client()
        types = self._types
        
        parts = [
            types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
        ]
        if prompt:
            parts.append(types.Part(text=prompt))
        
        config = types.GenerateContentConfig(system_instruction=system_instruction)
        contents = [types.Content(role="user", parts=parts)]
        
        while True:
            try:
                response = self._client.models.generate_content(
                    model=self._model,
                    contents=contents,
                    config=config,
                )
                return VLMResponse(
                    text=getattr(response, "text", "") or "",
                    raw_response=response,
                )
            except Exception as exc:
                if isinstance(exc, httpx.RemoteProtocolError):
                    print(f"RemoteProtocolError in query. Retrying in 5 seconds...")
                    time.sleep(5)
                    continue
                if isinstance(exc, self._genai.errors.ClientError) and _is_rate_limit_error(exc):
                    delay = _extract_retry_delay_seconds(exc) or 40.0
                    print(f"Gemini quota exceeded (429) in query. Retrying in {delay} seconds...")
                    time.sleep(delay)
                    continue
                if isinstance(exc, self._genai.errors.ServerError):
                    time.sleep(10)
                    continue
                raise
    
    def query_multiple(
        self,
        image_bytes_list: Sequence[bytes],
        captions: Sequence[str],
        prompt: str,
        mime_type: str = "image/png",
        system_instruction: Optional[str] = None,
    ) -> VLMResponse:
        self._ensure_client()
        types = self._types
        
        if len(image_bytes_list) != len(captions):
            raise ValueError("image_bytes_list and captions must be the same length.")
        if not image_bytes_list:
            raise ValueError("At least one image must be provided.")
        
        parts: List[types.Part] = []
        extra_args = {}
        if self._model == 'gemini-3-pro-preview':
            extra_args["media_resolution"] = {"level": "media_resolution_high"}
        
        for idx, (image_bytes, caption) in enumerate(zip(image_bytes_list, captions)):
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
        
        if prompt:
            parts.append(types.Part(text=prompt))
        
        config = types.GenerateContentConfig(system_instruction=system_instruction)
        contents = [types.Content(role="user", parts=parts, **extra_args)]
        
        while True:
            try:
                response = self._client.models.generate_content(
                    model=self._model,
                    contents=contents,
                    config=config,
                )
                return VLMResponse(
                    text=getattr(response, "text", "") or "",
                    raw_response=response,
                )
            except Exception as exc:
                if isinstance(exc, httpx.RemoteProtocolError):
                    print(f"RemoteProtocolError in query_multiple. Retrying in 5 seconds...")
                    time.sleep(5)
                    continue
                if isinstance(exc, self._genai.errors.ClientError) and _is_rate_limit_error(exc):
                    delay = _extract_retry_delay_seconds(exc) or 40.0
                    print(f"Gemini quota exceeded (429) in query_multiple. Retrying in {delay} seconds...")
                    time.sleep(delay)
                    continue
                if isinstance(exc, self._genai.errors.ServerError):
                    time.sleep(10)
                    continue
                raise
    
    def create_chat_session(self, max_turns: Optional[int] = None) -> "GeminiChatSession":
        """Create a Gemini chat session with history support."""
        self._ensure_client()
        return GeminiChatSession(
            model=self._model,
            client=self._client,
            types=self._types,
            genai=self._genai,
            max_turns=max_turns,
        )


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


class GeminiChatSession(VLMChatSession):
    """Gemini chat session with conversation history."""
    
    def __init__(
        self,
        model: str,
        client: Any,
        types: Any,
        genai: Any,
        max_turns: Optional[int] = None,
    ):
        self._model = model
        self._client = client
        self._types = types
        self._genai = genai
        self._turn_history: List[StoredTurn] = []
        self._max_turns = max_turns if (max_turns is None or max_turns >= 0) else None
    
    @property
    def turn_history(self) -> List[StoredTurn]:
        return self._turn_history
    
    def _resolve_requested_turns(self, history_turns: Optional[int]) -> Optional[int]:
        requested = history_turns if history_turns is not None else self._max_turns
        if requested is not None and requested < 0:
            return None
        return requested
    
    def _build_history_contents(self, start_index: int) -> List[Any]:
        types = self._types
        contents: List[Any] = []
        if start_index < 0:
            start_index = 0
        for image_caption_pairs, trailing_prompt, response_text in self._turn_history[start_index:]:
            user_parts: List[Any] = []
            extra_args = {}
            if self._model == 'gemini-3-pro-preview':
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
            contents.append(types.Content(role="user", parts=user_parts))
            if response_text:
                contents.append(
                    types.Content(role="model", parts=[types.Part(text=response_text)])
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
            self._turn_history = normalised[-self._max_turns:]
        return len(self._turn_history)
    
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
        types = self._types
        
        requested_turns = self._resolve_requested_turns(history_turns)
        if requested_turns is None:
            start_index = 0
        else:
            start_index = max(len(self._turn_history) - requested_turns, 0)
        
        config_kwargs = {}
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction
        if temperature is not None:
            config_kwargs["temperature"] = float(temperature)
        config = types.GenerateContentConfig(**config_kwargs) if config_kwargs else None
        
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
        
        parts: List[Any] = []
        stored_pairs: List[ImageCaptionPair] = []
        for idx, (image_data, caption_text) in enumerate(normalized_pairs):
            caption_to_use = caption_text or f"Image {idx + 1}:"
            if caption_to_use:
                parts.append(types.Part(text=caption_to_use))
            parts.append(
                types.Part(
                    inline_data=types.Blob(mime_type=mime_type, data=image_data)
                )
            )
            stored_pairs.append((bytes(image_data), caption_to_use))
        
        if prompt_value:
            parts.append(types.Part(text=prompt_value))
        
        while True:
            history_contents = self._build_history_contents(start_index)
            create_kwargs = {"model": self._model}
            if history_contents:
                create_kwargs["history"] = history_contents
            if config is not None:
                create_kwargs["config"] = config
            chat = self._client.chats.create(**create_kwargs)
            try:
                response = chat.send_message(parts)
                break
            except Exception as exc:
                if isinstance(exc, httpx.RemoteProtocolError):
                    print(f"RemoteProtocolError in chat session. Retrying in 5 seconds...")
                    time.sleep(5)
                    continue

                # Handle 429 Resource Exhausted (Rate Limit)
                if isinstance(exc, self._genai.errors.ClientError) and _is_rate_limit_error(exc):
                    delay = _extract_retry_delay_seconds(exc) or 40.0
                    print(f"Gemini quota exceeded (429). Retrying in {delay} seconds...")
                    time.sleep(delay)
                    continue

                if _is_token_limit_error(exc) and start_index < len(self._turn_history):
                    start_index += 1
                    continue
                
                if isinstance(exc, self._genai.errors.ServerError):
                    time.sleep(10)
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
        
        return VLMResponse(text=response_text, raw_response=response)


class Qwen3VLBackend(VLMBackend):
    """Qwen3-VL local HuggingFace model backend."""
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
        device_map: str = "auto",
        use_flash_attention: bool = False,
    ):
        self._model_name = model_name
        self._device_map = device_map
        self._use_flash_attention = use_flash_attention
        self._model = None
        self._processor = None
    
    def _ensure_model(self):
        if self._model is None:
            import torch
            from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
            
            load_kwargs = {
                "pretrained_model_name_or_path": self._model_name,
                "dtype": "auto",
                "device_map": self._device_map,
            }
            
            if self._use_flash_attention:
                load_kwargs["attn_implementation"] = "flash_attention_2"
                load_kwargs["dtype"] = torch.bfloat16
            
            self._model = Qwen3VLForConditionalGeneration.from_pretrained(**load_kwargs)
            self._processor = AutoProcessor.from_pretrained(self._model_name)
    
    @property
    def name(self) -> str:
        return self._model_name
    
    def _bytes_to_pil(self, image_bytes: bytes) -> PIL.Image.Image:
        """Convert image bytes to PIL Image."""
        return PIL.Image.open(BytesIO(image_bytes))
    
    def query(
        self,
        image_bytes: bytes,
        prompt: str,
        mime_type: str = "image/png",
        system_instruction: Optional[str] = None,
        max_new_tokens: int = 2048,
    ) -> VLMResponse:
        self._ensure_model()
        
        # Convert bytes to PIL Image
        image = self._bytes_to_pil(image_bytes)
        
        # Build messages
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": [{"type": "text", "text": system_instruction}]})
        
        messages.append({
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        })
        
        # Process inputs
        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self._model.device)
        
        # Generate
        generated_ids = self._model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] 
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self._processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False,
        )
        
        return VLMResponse(text=output_text[0] if output_text else "", raw_response=generated_ids)
    
    def query_multiple(
        self,
        image_bytes_list: Sequence[bytes],
        captions: Sequence[str],
        prompt: str,
        mime_type: str = "image/png",
        system_instruction: Optional[str] = None,
        max_new_tokens: int = 2048,
    ) -> VLMResponse:
        self._ensure_model()
        
        if len(image_bytes_list) != len(captions):
            raise ValueError("image_bytes_list and captions must be the same length.")
        if not image_bytes_list:
            raise ValueError("At least one image must be provided.")
        
        # Build content with interleaved images and captions
        content = []
        for idx, (image_bytes, caption) in enumerate(zip(image_bytes_list, captions)):
            caption_text = caption or f"Image {idx + 1}:"
            content.append({"type": "text", "text": caption_text})
            image = self._bytes_to_pil(image_bytes)
            content.append({"type": "image", "image": image})
        
        if prompt:
            content.append({"type": "text", "text": prompt})
        
        # Build messages
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": [{"type": "text", "text": system_instruction}]})
        messages.append({"role": "user", "content": content})
        
        # Process inputs
        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self._model.device)
        
        # Generate
        generated_ids = self._model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] 
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self._processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False,
        )
        
        return VLMResponse(text=output_text[0] if output_text else "", raw_response=generated_ids)
    
    def create_chat_session(self, max_turns: Optional[int] = None) -> "Qwen3VLChatSession":
        """Create a Qwen3-VL chat session with history support."""
        self._ensure_model()
        return Qwen3VLChatSession(
            model=self._model,
            processor=self._processor,
            model_name=self._model_name,
            max_turns=max_turns,
        )


class VLLMQwen3VLBackend(VLMBackend):
    """Qwen3-VL local vLLM backend."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
        tensor_parallel_size: int = 1,
        # max_model_len: int = 4096,
        # max_model_len: int = 8192,
        max_model_len: int = MAX_MODEL_LEN,
        gpu_memory_utilization: float = 0.9,
    ):
        self._model_name = model_name
        self._tensor_parallel_size = tensor_parallel_size
        self._max_model_len = max_model_len
        self._gpu_memory_utilization = gpu_memory_utilization
        self._llm = None
        self._processor = None
        self._SamplingParams = None

    def _ensure_model(self):
        if self._llm is None:
            from vllm import LLM, SamplingParams
            from transformers import AutoProcessor, AutoConfig

            # Patch Qwen3VLConfig if needed (workaround for vLLM compatibility)
            # vLLM expects 'num_attention_heads' but Qwen3-VL might use a different name
            try:
                # We trust remote code because we are using the model
                config = AutoConfig.from_pretrained(self._model_name, trust_remote_code=True)
                config_class = config.__class__
                
                if not hasattr(config, "num_attention_heads"):
                    print(f"Patching {config_class.__name__} to add 'num_attention_heads' for vLLM compatibility...")
                    
                    # Look for alternative attribute names
                    candidates = ["num_heads", "n_head", "n_heads", "attention_heads"]
                    found_attr = None
                    for attr in candidates:
                        if hasattr(config, attr):
                            found_attr = attr
                            break
                    
                    if found_attr:
                        print(f"  -> Mapping 'num_attention_heads' to '{found_attr}'")
                        # Patch the class so all instances get it
                        setattr(config_class, "num_attention_heads", property(lambda self, attr=found_attr: getattr(self, attr)))
                    else:
                        print(f"  -> Warning: Could not find candidate for num_attention_heads in {candidates}")
                        
            except Exception as e:
                print(f"Warning: Failed to patch config: {e}")

            # Start with full arguments
            kwargs = {
                "model": self._model_name,
                "tensor_parallel_size": self._tensor_parallel_size,
                "gpu_memory_utilization": self._gpu_memory_utilization,
                "trust_remote_code": True,
                "max_model_len": self._max_model_len,
            }

            # Helper to try initialization
            def try_init(current_kwargs):
                return LLM(**current_kwargs)

            # Iteratively try to initialize, removing unsupported arguments on failure
            try:
                self._llm = try_init(kwargs)
            except TypeError as e:
                import re
                current_err = e
                while True:
                    err_str = str(current_err)
                    if "unexpected keyword argument" in err_str:
                        match = re.search(r"unexpected keyword argument '([^']+)'", err_str)
                        if match:
                            bad_arg = match.group(1)
                            if bad_arg in kwargs:
                                print(f"Warning: vLLM init failed with argument '{bad_arg}'. Removing it and retrying.")
                                del kwargs[bad_arg]
                                try:
                                    self._llm = try_init(kwargs)
                                    break  # Success
                                except TypeError as next_e:
                                    current_err = next_e
                                    continue
                    
                    # If we can't handle the error, raise it
                    print(f"Failed to initialize vLLM with kwargs: {kwargs.keys()}")
                    raise current_err

            self._processor = AutoProcessor.from_pretrained(self._model_name)
            self._SamplingParams = SamplingParams

    @property
    def name(self) -> str:
        return self._model_name

    def _bytes_to_pil(self, image_bytes: bytes) -> PIL.Image.Image:
        return PIL.Image.open(BytesIO(image_bytes))

    def _build_messages(
        self,
        image_bytes_list: Sequence[bytes],
        captions: Sequence[str],
        prompt: str,
        system_instruction: Optional[str],
    ) -> Tuple[List[dict], List[PIL.Image.Image]]:
        images: List[PIL.Image.Image] = []
        content: List[dict] = []
        for idx, (image_bytes, caption) in enumerate(zip(image_bytes_list, captions)):
            caption_text = caption or f"Image {idx + 1}:"
            if caption_text:
                content.append({"type": "text", "text": caption_text})
            image = self._bytes_to_pil(image_bytes)
            content.append({"type": "image", "image": image})
            images.append(image)

        if prompt:
            content.append({"type": "text", "text": prompt})

        messages: List[dict] = []
        if system_instruction:
            messages.append({"role": "system", "content": [{"type": "text", "text": system_instruction}]})
        messages.append({"role": "user", "content": content})
        return messages, images

    def _generate(
        self,
        messages: List[dict],
        images: Sequence[PIL.Image.Image],
        max_new_tokens: int,
        temperature: Optional[float] = None,
    ) -> str:
        self._ensure_model()
        prompt_text = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        sampling_params = self._SamplingParams(
            max_tokens=max_new_tokens,
            temperature=0.7 if temperature is None else float(temperature),
            top_p=0.9,
        )
        image_payload: Any = images[0] if len(images) == 1 else list(images)
        outputs = self._llm.generate(
            [{
                "prompt": prompt_text,
                "multi_modal_data": {"image": image_payload},
            }],
            sampling_params=sampling_params,
        )
        return outputs[0].outputs[0].text if outputs else ""

    def query(
        self,
        image_bytes: bytes,
        prompt: str,
        mime_type: str = "image/png",
        system_instruction: Optional[str] = None,
        max_new_tokens: int = 2048,
    ) -> VLMResponse:
        messages, images = self._build_messages([image_bytes], [""], prompt, system_instruction)
        response_text = self._generate(messages, images, max_new_tokens)
        return VLMResponse(text=response_text)

    def query_multiple(
        self,
        image_bytes_list: Sequence[bytes],
        captions: Sequence[str],
        prompt: str,
        mime_type: str = "image/png",
        system_instruction: Optional[str] = None,
        max_new_tokens: int = 2048,
    ) -> VLMResponse:
        if len(image_bytes_list) != len(captions):
            raise ValueError("image_bytes_list and captions must be the same length.")
        if not image_bytes_list:
            raise ValueError("At least one image must be provided.")

        messages, images = self._build_messages(image_bytes_list, captions, prompt, system_instruction)
        response_text = self._generate(messages, images, max_new_tokens)
        return VLMResponse(text=response_text)
    
    def create_chat_session(self, max_turns: Optional[int] = None) -> "VLLMQwen3VLChatSession":
        """Create a Qwen3-VL chat session with history support."""
        self._ensure_model()
        return VLLMQwen3VLChatSession(
            llm=self._llm,
            processor=self._processor,
            model_name=self._model_name,
            max_turns=max_turns,
            sampling_params_cls=self._SamplingParams,
        )


class Qwen3VLChatSession(VLMChatSession):
    """Qwen3-VL chat session with conversation history."""
    
    def __init__(
        self,
        model: Any,
        processor: Any,
        model_name: str,
        max_turns: Optional[int] = None,
    ):
        self._model = model
        self._processor = processor
        self._model_name = model_name
        self._turn_history: List[StoredTurn] = []
        self._max_turns = max_turns if (max_turns is None or max_turns >= 0) else None
    
    @property
    def turn_history(self) -> List[StoredTurn]:
        return self._turn_history
    
    def _bytes_to_pil(self, image_bytes: bytes) -> PIL.Image.Image:
        """Convert image bytes to PIL Image."""
        return PIL.Image.open(BytesIO(image_bytes))
    
    def _resolve_requested_turns(self, history_turns: Optional[int]) -> Optional[int]:
        requested = history_turns if history_turns is not None else self._max_turns
        if requested is not None and requested < 0:
            return None
        return requested
    
    def _build_history_messages(self, start_index: int) -> List[dict]:
        """Build message history for Qwen3-VL."""
        messages: List[dict] = []
        if start_index < 0:
            start_index = 0
        for image_caption_pairs, trailing_prompt, response_text in self._turn_history[start_index:]:
            # Build user content with images and captions
            user_content = []
            for image_data, caption_text in image_caption_pairs:
                if caption_text:
                    user_content.append({"type": "text", "text": caption_text})
                image = self._bytes_to_pil(image_data)
                user_content.append({"type": "image", "image": image})
            if trailing_prompt:
                user_content.append({"type": "text", "text": trailing_prompt})
            messages.append({"role": "user", "content": user_content})
            if response_text:
                # Assistant content must also be list format for Qwen3-VL processor
                messages.append({"role": "assistant", "content": [{"type": "text", "text": response_text}]})
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
            self._turn_history = normalised[-self._max_turns:]
        return len(self._turn_history)
    
    def send(
        self,
        image_caption_pairs: Sequence[ImageCaptionInput],
        prompt: Optional[str] = "",
        history_turns: Optional[int] = 0,
        mime_type: str = "image/png",
        system_instruction: Optional[str] = None,
        temperature: Optional[float] = None,
        thinking_budget: int = -1,
        max_new_tokens: int = 2048,
    ) -> VLMResponse:
        requested_turns = self._resolve_requested_turns(history_turns)
        if requested_turns is None:
            start_index = 0
        else:
            start_index = max(len(self._turn_history) - requested_turns, 0)
        
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
        
        # Build current turn content
        current_content = []
        stored_pairs: List[ImageCaptionPair] = []
        for idx, (image_data, caption_text) in enumerate(normalized_pairs):
            caption_to_use = caption_text or f"Image {idx + 1}:"
            if caption_to_use:
                current_content.append({"type": "text", "text": caption_to_use})
            image = self._bytes_to_pil(image_data)
            current_content.append({"type": "image", "image": image})
            stored_pairs.append((bytes(image_data), caption_to_use))
        
        if prompt_value:
            current_content.append({"type": "text", "text": prompt_value})
        
        # Build full message list with history
        messages = []
        if system_instruction:
            # System content must also be list format for Qwen3-VL processor
            messages.append({"role": "system", "content": [{"type": "text", "text": system_instruction}]})
        
        # Add history
        history_messages = self._build_history_messages(start_index)
        messages.extend(history_messages)
        
        # Add current turn
        messages.append({"role": "user", "content": current_content})
        
        # Process and generate
        generate_kwargs = {"max_new_tokens": max_new_tokens}
        if temperature is not None:
            generate_kwargs["temperature"] = temperature
            generate_kwargs["do_sample"] = True
        
        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self._model.device)
        
        generated_ids = self._model.generate(**inputs, **generate_kwargs)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] 
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self._processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        
        response_text = output_text[0] if output_text else ""
        
        # Update history
        if start_index > 0:
            self._turn_history = self._turn_history[start_index:]
        self._turn_history.append((stored_pairs, prompt_value, response_text))
        if self._max_turns is not None:
            if self._max_turns == 0:
                self._turn_history.clear()
            else:
                self._turn_history = self._turn_history[-self._max_turns:]
        
        return VLMResponse(text=response_text, raw_response=generated_ids)


class VLLMQwen3VLChatSession(VLMChatSession):
    """Qwen3-VL vLLM chat session with conversation history."""

    def __init__(
        self,
        llm: Any,
        processor: Any,
        model_name: str,
        max_turns: Optional[int] = None,
        sampling_params_cls: Optional[Any] = None,
    ):
        self._llm = llm
        self._processor = processor
        self._model_name = model_name
        self._turn_history: List[StoredTurn] = []
        self._max_turns = max_turns if (max_turns is None or max_turns >= 0) else None
        self._SamplingParams = sampling_params_cls

    @property
    def turn_history(self) -> List[StoredTurn]:
        return self._turn_history

    def _bytes_to_pil(self, image_bytes: bytes) -> PIL.Image.Image:
        return PIL.Image.open(BytesIO(image_bytes))

    def _resolve_requested_turns(self, history_turns: Optional[int]) -> Optional[int]:
        requested = history_turns if history_turns is not None else self._max_turns
        if requested is not None and requested < 0:
            return None
        return requested

    def load_history(self, turns: Iterable[HistoryTurnInput]) -> int:
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
            self._turn_history = normalised[-self._max_turns:]
        return len(self._turn_history)

    def _build_messages_with_images(
        self,
        start_index: int,
        normalized_pairs: Sequence[ImageCaptionPair],
        prompt_value: str,
    ) -> Tuple[List[dict], List[PIL.Image.Image], List[ImageCaptionPair]]:
        messages: List[dict] = []
        images: List[PIL.Image.Image] = []
        if start_index < 0:
            start_index = 0
        for image_caption_pairs, trailing_prompt, response_text in self._turn_history[start_index:]:
            user_content = []
            for image_data, caption_text in image_caption_pairs:
                if caption_text:
                    user_content.append({"type": "text", "text": caption_text})
                image = self._bytes_to_pil(image_data)
                user_content.append({"type": "image", "image": image})
                images.append(image)
            if trailing_prompt:
                user_content.append({"type": "text", "text": trailing_prompt})
            messages.append({"role": "user", "content": user_content})
            if response_text:
                messages.append({"role": "assistant", "content": [{"type": "text", "text": response_text}]})

        current_content = []
        stored_pairs: List[ImageCaptionPair] = []
        for idx, (image_data, caption_text) in enumerate(normalized_pairs):
            caption_to_use = caption_text or f"Image {idx + 1}:"
            if caption_to_use:
                current_content.append({"type": "text", "text": caption_to_use})
            image = self._bytes_to_pil(image_data)
            current_content.append({"type": "image", "image": image})
            images.append(image)
            stored_pairs.append((bytes(image_data), caption_to_use))

        if prompt_value:
            current_content.append({"type": "text", "text": prompt_value})
        messages.append({"role": "user", "content": current_content})
        return messages, images, stored_pairs

    def send(
        self,
        image_caption_pairs: Sequence[ImageCaptionInput],
        prompt: Optional[str] = "",
        history_turns: Optional[int] = 0,
        mime_type: str = "image/png",
        system_instruction: Optional[str] = None,
        temperature: Optional[float] = None,
        thinking_budget: int = -1,
        max_new_tokens: int = 2048,
    ) -> VLMResponse:
        requested_turns = self._resolve_requested_turns(history_turns)
        if requested_turns is None:
            start_index = 0
        else:
            start_index = max(len(self._turn_history) - requested_turns, 0)

        pair_list = list(image_caption_pairs or ())
        if not pair_list:
            raise ValueError("image_caption_pairs must include at least one entry.")

        prompt_value = "" if prompt is None else str(prompt)

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

        messages, images, stored_pairs = self._build_messages_with_images(
            start_index,
            normalized_pairs,
            prompt_value,
        )
        if system_instruction:
            messages.insert(0, {"role": "system", "content": [{"type": "text", "text": system_instruction}]})

        prompt_text = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        sampling_params = self._SamplingParams(
            max_tokens=max_new_tokens,
            temperature=0.7 if temperature is None else float(temperature),
            top_p=0.9,
        )
        image_payload: Any = images[0] if len(images) == 1 else list(images)
        outputs = self._llm.generate(
            [{
                "prompt": prompt_text,
                "multi_modal_data": {"image": image_payload},
            }],
            sampling_params=sampling_params,
        )
        response_text = outputs[0].outputs[0].text if outputs else ""

        if start_index > 0:
            self._turn_history = self._turn_history[start_index:]
        self._turn_history.append((stored_pairs, prompt_value, response_text))
        if self._max_turns is not None:
            if self._max_turns == 0:
                self._turn_history.clear()
            else:
                self._turn_history = self._turn_history[-self._max_turns:]

        return VLMResponse(text=response_text, raw_response=outputs)


class VLLMClientChatSession(VLMChatSession):
    """Chat session for remote vLLM server (OpenAI-compatible)."""

    def __init__(
        self,
        client_backend: "VLLMClientBackend",
        max_turns: Optional[int] = None,
    ):
        self._client = client_backend
        self._turn_history: List[StoredTurn] = []
        self._max_turns = max_turns if (max_turns is None or max_turns >= 0) else None

    @property
    def turn_history(self) -> List[StoredTurn]:
        return self._turn_history

    def _resolve_requested_turns(self, history_turns: Optional[int]) -> Optional[int]:
        requested = history_turns if history_turns is not None else self._max_turns
        if requested is not None and requested < 0:
            return None
        return requested

    def load_history(self, turns: Iterable[HistoryTurnInput]) -> int:
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
            self._turn_history = normalised[-self._max_turns:]
        return len(self._turn_history)

    def send(
        self,
        image_caption_pairs: Sequence[ImageCaptionInput],
        prompt: Optional[str] = "",
        history_turns: Optional[int] = 0,
        mime_type: str = "image/png",
        system_instruction: Optional[str] = None,
        temperature: Optional[float] = None,
        thinking_budget: int = -1,
        max_new_tokens: int = 2048,
    ) -> VLMResponse:
        import base64

        requested_turns = self._resolve_requested_turns(history_turns)
        if requested_turns is None:
            start_index = 0
        else:
            start_index = max(len(self._turn_history) - requested_turns, 0)

        # Build messages
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})

        # Add history
        for image_pairs, trailing_prompt, response_text in self._turn_history[start_index:]:
            content = []
            for img_bytes, caption in image_pairs:
                if caption:
                    content.append({"type": "text", "text": caption})
                b64_img = base64.b64encode(img_bytes).decode("utf-8")
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{b64_img}"}
                })
            if trailing_prompt:
                content.append({"type": "text", "text": trailing_prompt})

            messages.append({"role": "user", "content": content})
            if response_text:
                messages.append({"role": "assistant", "content": response_text})

        # Current turn
        pair_list = list(image_caption_pairs or ())
        if not pair_list:
            raise ValueError("image_caption_pairs must include at least one entry.")

        content = []
        stored_pairs: List[ImageCaptionPair] = []
        for pair in pair_list:
            try:
                img_bytes, caption = pair
            except (TypeError, ValueError):
                continue
            if img_bytes is None:
                continue
            caption_val = str(caption or "")

            if caption_val:
                content.append({"type": "text", "text": caption_val})
            b64_img = base64.b64encode(img_bytes).decode("utf-8")
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{b64_img}"}
            })
            stored_pairs.append((bytes(img_bytes), caption_val))

        prompt_val = str(prompt or "")
        if prompt_val:
            content.append({"type": "text", "text": prompt_val})

        messages.append({"role": "user", "content": content})

        # Send request via backend
        response_text = self._client._send_request(
            messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )

        # Update history
        if start_index > 0:
            self._turn_history = self._turn_history[start_index:]
        self._turn_history.append((stored_pairs, prompt_val, response_text))
        if self._max_turns is not None:
            if self._max_turns == 0:
                self._turn_history.clear()
            else:
                self._turn_history = self._turn_history[-self._max_turns:]

        return VLMResponse(text=response_text)


class VLLMClientBackend(VLMBackend):
    """Client for a remote vLLM server (OpenAI-compatible)."""

    def __init__(self, model_name: str, base_url: str = "http://localhost:8000/v1"):
        env_url = os.environ.get("VLLM_BASE_URL")
        if env_url:
            base_url = env_url
        self._model_name = model_name
        self._base_url = base_url.rstrip("/")
        self._api_key = "EMPTY"  # vLLM usually doesn't require a key by default

    @property
    def name(self) -> str:
        return f"remote:{self._model_name}"

    def _send_request(
        self, 
        messages: List[dict], 
        max_new_tokens: int = 2048,
        temperature: Optional[float] = None
    ) -> str:
        payload = {
            "model": self._model_name,
            "messages": messages,
            "max_tokens": max_new_tokens,
            "temperature": 0.7 if temperature is None else float(temperature),
        }
        
        # Retry logic for connection errors
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    f"{self._base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    json=payload,
                    timeout=300 # 5 minutes timeout for large model processing
                )
                response.raise_for_status()
                result = response.json()
                return result["choices"][0]["message"]["content"]
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    print(f"VLLM Client Error (Final Attempt): {e}")
                    raise
                print(f"VLLM Client Error (Attempt {attempt+1}/{max_retries}): {e}. Retrying in 2s...")
                time.sleep(2)
        return "" # Should not reach here

    def query(
        self,
        image_bytes: bytes,
        prompt: str,
        mime_type: str = "image/png",
        system_instruction: Optional[str] = None,
        max_new_tokens: int = 2048,
    ) -> VLMResponse:
        import base64
        b64_image = base64.b64encode(image_bytes).decode("utf-8")
        image_url = f"data:{mime_type};base64,{b64_image}"

        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})

        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_url}},
            ]
        })

        text = self._send_request(messages, max_new_tokens=max_new_tokens)
        return VLMResponse(text=text)

    def query_multiple(
        self,
        image_bytes_list: Sequence[bytes],
        captions: Sequence[str],
        prompt: str,
        mime_type: str = "image/png",
        system_instruction: Optional[str] = None,
        max_new_tokens: int = 2048,
    ) -> VLMResponse:
        import base64

        content = []
        for img_bytes, caption in zip(image_bytes_list, captions):
            if caption:
                content.append({"type": "text", "text": caption})
            b64_img = base64.b64encode(img_bytes).decode("utf-8")
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{b64_img}"}
            })

        if prompt:
            content.append({"type": "text", "text": prompt})

        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": content})

        text = self._send_request(messages, max_new_tokens=max_new_tokens)
        return VLMResponse(text=text)

    def create_chat_session(self, max_turns: Optional[int] = None) -> "VLLMClientChatSession":
        return VLLMClientChatSession(client_backend=self, max_turns=max_turns)


# Registry of available backends
_BACKEND_REGISTRY = {
    # Gemini models
    "gemini-2.5-pro": lambda: GeminiBackend("gemini-2.5-pro"),
    "gemini-2.5-flash": lambda: GeminiBackend("gemini-2.5-flash"),
    "gemini-3-pro-preview": lambda: GeminiBackend("gemini-3-pro-preview"),
    # Qwen3-VL models (default backend selected in create_vlm_backend)
    "qwen3-vl-8b": lambda: Qwen3VLBackend("Qwen/Qwen3-VL-8B-Instruct"),
    "qwen3-vl-4b": lambda: Qwen3VLBackend("Qwen/Qwen3-VL-4B-Instruct"),
    "qwen3-vl-2b": lambda: Qwen3VLBackend("Qwen/Qwen3-VL-2B-Instruct"),
    "qwen3-vl-30b-fp8": lambda: Qwen3VLBackend("Qwen/Qwen3-VL-30B-A3B-Instruct-FP8"),
}


def _vllm_available() -> bool:
    return importlib.util.find_spec("vllm") is not None


def _resolve_qwen_model_name(model: str) -> str:
    model_map = {
        "qwen3-vl-8b": "Qwen/Qwen3-VL-8B-Instruct",
        "qwen3-vl-4b": "Qwen/Qwen3-VL-4B-Instruct",
        "qwen3-vl-2b": "Qwen/Qwen3-VL-2B-Instruct",
        "qwen3-vl-30b-fp8": "Qwen/Qwen3-VL-30B-A3B-Instruct-FP8",

    }
    return model_map.get(model, model)


def create_vlm_backend(model: str, backend: Optional[str] = None, **kwargs) -> VLMBackend:
    """
    Create a VLM backend by model name.
    
    Args:
        model: Model name (e.g., "gemini-2.5-pro", "qwen3-vl-8b")
        backend: Backend override ("auto", "vllm", "hf")
        **kwargs: Additional arguments passed to the backend constructor
    
    Returns:
        VLMBackend instance
    
    Examples:
        >>> backend = create_vlm_backend("gemini-2.5-pro")
        >>> backend = create_vlm_backend("qwen3-vl-8b", backend="hf", use_flash_attention=True)
    """
    backend_choice = (backend or "auto").lower()
    if backend_choice not in {"auto", "vllm", "hf", "remote"}:
        raise ValueError("backend must be one of: auto, vllm, hf, remote")

    if model.startswith("remote:"):
        actual_model = model[7:]
        return VLLMClientBackend(actual_model, **kwargs)

    if model.startswith("gemini"):
        return _create_gemini_backend(model, **kwargs)

    # Check registry first
    if model in _BACKEND_REGISTRY:
        if model.startswith("qwen"):
            resolved_model = _resolve_qwen_model_name(model)
            if backend_choice == "remote":
                return VLLMClientBackend(resolved_model, **kwargs)
            if backend_choice == "vllm":
                return VLLMQwen3VLBackend(resolved_model, **kwargs)
            if backend_choice == "auto" and _vllm_available():
                return VLLMQwen3VLBackend(resolved_model, **kwargs)
            return Qwen3VLBackend(resolved_model, **kwargs)
        return _BACKEND_REGISTRY[model]()
    
    # Try to infer backend from model name
    model_lower = model.lower()
    if "qwen" in model_lower:
        resolved_model = _resolve_qwen_model_name(model)
        if backend_choice == "remote":
            return VLLMClientBackend(resolved_model, **kwargs)
        if backend_choice == "vllm":
            return VLLMQwen3VLBackend(resolved_model, **kwargs)
        if backend_choice == "auto" and not kwargs and _vllm_available():
            return VLLMQwen3VLBackend(resolved_model)
        return Qwen3VLBackend(resolved_model, **kwargs)
    
    else:
        return _create_gemini_backend(model, **kwargs)


def list_available_models() -> List[str]:
    """Return list of available model names."""
    return list(_BACKEND_REGISTRY.keys())


def is_local_model(model: str) -> bool:
    """Check if a model runs locally (vs API-based)."""
    model_lower = model.lower()
    return "qwen" in model_lower


# Convenience singleton for backward compatibility
_default_backend: Optional[VLMBackend] = None


def get_default_backend(model: str = "gemini-2.5-pro") -> VLMBackend:
    """Get or create the default backend singleton."""
    global _default_backend
    if _default_backend is None or _default_backend.name != model:
        _default_backend = create_vlm_backend(model)
    return _default_backend


if __name__ == "__main__":
    # Test Qwen3-VL backend
    print("Testing Qwen3-VL backend...")
    backend = create_vlm_backend("qwen3-vl-8b")
    
    # Load test image
    test_image_path = "rendered/rendered-34589-116.png"
    if os.path.exists(test_image_path):
        with open(test_image_path, "rb") as f:
            image_bytes = f.read()
        
        response = backend.query(image_bytes, prompt="Describe this image in detail.")
        print(f"Response: {response.text}")
    else:
        print(f"Test image not found: {test_image_path}")
        print("Creating a simple test image...")
        
        # Create a simple test image
        import numpy as np
        test_img = PIL.Image.fromarray(
            np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        )
        buf = BytesIO()
        test_img.save(buf, format="PNG")
        image_bytes = buf.getvalue()
        
        response = backend.query(image_bytes, prompt="What do you see in this image?")
        print(f"Response: {response.text}")
