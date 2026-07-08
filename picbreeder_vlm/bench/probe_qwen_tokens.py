
import os
import sys
import torch
from PIL import Image
try:
    from transformers import AutoProcessor
except ImportError:
    print("Transformers not installed.")
    sys.exit(0)

model_id = "Qwen/Qwen2.5-VL-7B-Instruct"

try:
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
except Exception as e:
    print(f"Failed to load processor for {model_id}: {e}")
    # Try Qwen2-VL
    model_id = "Qwen/Qwen2-VL-7B-Instruct"
    try:
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    except Exception as e:
        print(f"Failed to load processor for {model_id}: {e}")
        sys.exit(1)

print(f"Loaded processor for {model_id}")

image = Image.new("RGB", (128, 128), color=(128, 128, 128))
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "Image 1"},
        ],
    }
]

text_prompt = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
inputs = processor(
    text=[text_prompt],
    images=[image],
    padding=True,
    return_tensors="pt",
)

print(f"Token count for 128x128 image + 'Image 1': {inputs.input_ids.shape[1]}")

# Check Archive size (100 images)
images = [Image.new("RGB", (128, 128)) for _ in range(100)]
content = []
for i in range(100):
    content.append({"type": "text", "text": f"Image {i}"})
    content.append({"type": "image", "image": images[i]})
content.append({"type": "text", "text": "Archive prompt..."})

conversation_archive = [
    {"role": "user", "content": content}
]

text_prompt_archive = processor.apply_chat_template(conversation_archive, tokenize=False, add_generation_prompt=True)
# Processing 100 images might be slow or OOM on cpu if not careful, but we just want token count.
# We can estimate by doing 1 image and multiplying? No, Qwen packs images.
# Let's try 10 images.
images_10 = images[:10]
content_10 = []
for i in range(10):
    content_10.append({"type": "text", "text": f"Image {i}"})
    content_10.append({"type": "image", "image": images_10[i]})

conversation_10 = [{"role": "user", "content": content_10}]
text_prompt_10 = processor.apply_chat_template(conversation_10, tokenize=False, add_generation_prompt=True)
inputs_10 = processor(text=[text_prompt_10], images=images_10, return_tensors="pt")
print(f"Token count for 10 images: {inputs_10.input_ids.shape[1]}")

