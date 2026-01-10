"""
Test Qwen3-VL using the unified VLM backend.

Usage:
    python test_qwen.py                    # Test with default image
    python test_qwen.py path/to/image.png  # Test with custom image
"""

import sys
from io import BytesIO
from pathlib import Path

import requests
import PIL.Image

from vlm_backends import create_vlm_backend


def load_image_bytes(source: str) -> bytes:
    """Load image bytes from URL or file path."""
    if source.startswith("http://") or source.startswith("https://"):
        response = requests.get(source)
        response.raise_for_status()
        return response.content
    else:
        with open(source, "rb") as f:
            return f.read()


def main():
    # Default test image URL (same as original)
    default_url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
    
    # Get image source from args or use default
    image_source = sys.argv[1] if len(sys.argv) > 1 else default_url
    
    print(f"Loading image from: {image_source}")
    image_bytes = load_image_bytes(image_source)
    
    # Determine mime type
    mime_type = "image/png" if image_source.endswith(".png") else "image/jpeg"
    
    # Create Qwen3-VL backend
    print("Creating Qwen3-VL backend...")
    backend = create_vlm_backend("qwen3-vl-8b")
    
    # Query the model
    print("Querying model...")
    response = backend.query(
        image_bytes,
        prompt="Describe this image.",
        mime_type=mime_type,
    )
    
    print("\nResponse:")
    print(response.text)


if __name__ == "__main__":
    main()
