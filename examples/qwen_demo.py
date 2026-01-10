"""
Example: Using Qwen3-VL with picbreeder-vlm.

This demonstrates how to use the unified VLM backend interface
to switch between Gemini and Qwen3-VL models.
"""

from pathlib import Path
from io import BytesIO

import PIL.Image
import numpy as np

from vlm_backends import create_vlm_backend, list_available_models, is_local_model


def demo_single_image():
    """Demonstrate single image query with both backends."""
    
    print("Available models:", list_available_models())
    print()
    
    # Create a simple test image (red square)
    img_array = np.zeros((128, 128, 3), dtype=np.uint8)
    img_array[32:96, 32:96] = [255, 0, 0]  # Red square
    test_img = PIL.Image.fromarray(img_array)
    
    buf = BytesIO()
    test_img.save(buf, format="PNG")
    image_bytes = buf.getvalue()
    
    # Test with Qwen3-VL
    print("=" * 50)
    print("Testing Qwen3-VL-8B...")
    print("=" * 50)
    
    qwen_backend = create_vlm_backend("qwen3-vl-8b")
    print(f"Backend: {qwen_backend.name}")
    print(f"Is local model: {is_local_model('qwen3-vl-8b')}")
    
    response = qwen_backend.query(
        image_bytes,
        prompt="What shape and color do you see in this image? Be brief.",
    )
    print(f"Response: {response.text}")
    print()


def demo_multiple_images():
    """Demonstrate multi-image query."""
    
    # Create test images
    images = []
    captions = []
    
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    names = ["red", "green", "blue"]
    
    for color, name in zip(colors, names):
        img_array = np.zeros((64, 64, 3), dtype=np.uint8)
        img_array[16:48, 16:48] = color
        img = PIL.Image.fromarray(img_array)
        
        buf = BytesIO()
        img.save(buf, format="PNG")
        images.append(buf.getvalue())
        captions.append(f"Image showing a {name} square:")
    
    print("=" * 50)
    print("Testing multi-image query...")
    print("=" * 50)
    
    backend = create_vlm_backend("qwen3-vl-8b")
    response = backend.query_multiple(
        images,
        captions,
        prompt="What colors do you see across all images? List them.",
    )
    print(f"Response: {response.text}")


def demo_with_system_instruction():
    """Demonstrate using system instructions."""
    
    # Create test image
    img_array = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    test_img = PIL.Image.fromarray(img_array)
    
    buf = BytesIO()
    test_img.save(buf, format="PNG")
    image_bytes = buf.getvalue()
    
    print("=" * 50)
    print("Testing with system instruction...")
    print("=" * 50)
    
    backend = create_vlm_backend("qwen3-vl-8b")
    response = backend.query(
        image_bytes,
        prompt="What do you see?",
        system_instruction="You are an art critic. Respond in exactly one sentence.",
    )
    print(f"Response: {response.text}")


if __name__ == "__main__":
    demo_single_image()
    print("\n")
    demo_multiple_images()
    print("\n")
    demo_with_system_instruction()
