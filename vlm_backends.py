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
"""

import os
import time
from abc import ABC, abstractmethod
from io import BytesIO
from typing import Any, List, Optional, Sequence, Tuple, Union

import PIL.Image

# Type aliases
ImageCaptionInput = Tuple[bytes, Optional[str]]
ImageCaptionPair = Tuple[bytes, str]


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
            messages.append({"role": "system", "content": system_instruction})
        
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
            messages.append({"role": "system", "content": system_instruction})
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
