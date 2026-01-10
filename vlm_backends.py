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

import os
import time
from abc import ABC, abstractmethod
from io import BytesIO
from typing import Any, Iterable, List, Optional, Sequence, Tuple, Union

import PIL.Image

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
            except self._genai.errors.ServerError:
                time.sleep(10)
    
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
            except self._genai.errors.ServerError:
                time.sleep(10)
    
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
                if _is_token_limit_error(exc) and start_index < len(self._turn_history):
                    start_index += 1
                    continue
                raise
            except self._genai.errors.ServerError:
                time.sleep(10)
                continue
        
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
                "torch_dtype": "auto",
                "device_map": self._device_map,
            }
            
            if self._use_flash_attention:
                load_kwargs["attn_implementation"] = "flash_attention_2"
                load_kwargs["torch_dtype"] = torch.bfloat16
            
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


# Registry of available backends
_BACKEND_REGISTRY = {
    # Gemini models
    "gemini-2.5-pro": lambda: GeminiBackend("gemini-2.5-pro"),
    "gemini-2.5-flash": lambda: GeminiBackend("gemini-2.5-flash"),
    "gemini-3-pro-preview": lambda: GeminiBackend("gemini-3-pro-preview"),
    # Qwen3-VL models
    "qwen3-vl-8b": lambda: Qwen3VLBackend("Qwen/Qwen3-VL-8B-Instruct"),
    "qwen3-vl-4b": lambda: Qwen3VLBackend("Qwen/Qwen3-VL-4B-Instruct"),
    "qwen3-vl-2b": lambda: Qwen3VLBackend("Qwen/Qwen3-VL-2B-Instruct"),
}


def create_vlm_backend(model: str, **kwargs) -> VLMBackend:
    """
    Create a VLM backend by model name.
    
    Args:
        model: Model name (e.g., "gemini-2.5-pro", "qwen3-vl-8b")
        **kwargs: Additional arguments passed to the backend constructor
    
    Returns:
        VLMBackend instance
    
    Examples:
        >>> backend = create_vlm_backend("gemini-2.5-pro")
        >>> backend = create_vlm_backend("qwen3-vl-8b", use_flash_attention=True)
    """
    # Check registry first
    if model in _BACKEND_REGISTRY:
        if kwargs:
            # Custom kwargs - need to instantiate directly
            if model.startswith("gemini"):
                return GeminiBackend(model, **kwargs)
            elif model.startswith("qwen"):
                model_map = {
                    "qwen3-vl-8b": "Qwen/Qwen3-VL-8B-Instruct",
                    "qwen3-vl-4b": "Qwen/Qwen3-VL-4B-Instruct", 
                    "qwen3-vl-2b": "Qwen/Qwen3-VL-2B-Instruct",
                }
                return Qwen3VLBackend(model_map.get(model, model), **kwargs)
        return _BACKEND_REGISTRY[model]()
    
    # Try to infer backend from model name
    model_lower = model.lower()
    if "gemini" in model_lower:
        return GeminiBackend(model, **kwargs)
    elif "qwen" in model_lower:
        return Qwen3VLBackend(model, **kwargs)
    
    raise ValueError(
        f"Unknown model: {model}. "
        f"Available models: {list(_BACKEND_REGISTRY.keys())}. "
        "Or use a full HuggingFace model path for Qwen."
    )


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
