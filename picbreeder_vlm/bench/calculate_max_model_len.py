
import json
import torch
from PIL import Image
from transformers import AutoProcessor

# User-provided parameters
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
THUMB_SIZE = 128
ARCHIVE_IMAGES = 100
GEN_IMAGES = 15
TOTAL_GENERATIONS = 20
RATIONALE_TEXT = (
    "Images 1, 8, and 13 represent a balanced selection of diverse visual qualities. "
    "Image 1 offers a clean, rhythmic spiral with vibrant, harmonious color gradients, providing a strong structural foundation. "
    "Image 8 presents a dynamic, colorful vortex with intense saturation and a sense of motion, adding energy and complexity. "
    "Image 13 introduces a swirling, organic pattern with bold contrasts and a focal point that draws the eye, enhancing visual intrigue. "
    "These selections combine structural clarity, energetic motion, and organic complexity, ensuring a rich and innovative evolutionary path."
)

# Prompts (Simplified versions of what is in prompts.py/agent_runner.py)
ARCHIVE_PROMPT = (
    "Above is the archive of images published by prior users. You may choose to branch from one of them, or start from a fresh population.\n"
    "Top Rated: images 0-19.\n"
    "Best New Images: images 20-39.\n"
    "Most Branched: images 40-59.\n"
    "Newest: images 60-79.\n"
    "Random: images 80-99."
)

GEN_PROMPT = (
    "Above is the grid at generation {generation}.\n"
    "Pick one or several images by their numeric labels--the corresponding CPPNs will be used as the parents of the next generation (using both mutation and crossover). "
    "Respond with JSON only: {{\"selected\": [indices]}}."
)

SYSTEM_PROMPT = (
    "You are playing with a collaborative online platform which allows users to interactively evolve small neural networks called Compositional Pattern Producing Networks (CPPNs) for generating images. "
    "Your goal is to evolve images that resemble familiar real-world objects. "
    "At the first generation the initial grid will display an archive of images published by prior users as favorites (unless you are the first user). "
    "You may choose to \"branch\" one of these images, or start instead from a random initial population. "
    "At each subsequent generation, you will be shown a set of numbered images produced by CPPNs. "
    "Respond with JSON only."
)

def create_dummy_image(size=THUMB_SIZE):
    return Image.new("RGB", (size, size), color=(100, 100, 100))

def build_turn_content(images, caption_prefix, prompt_text):
    content = []
    for i, img in enumerate(images):
        content.append({"type": "text", "text": f"{caption_prefix} {i}"})
        content.append({"type": "image", "image": img})
    content.append({"type": "text", "text": prompt_text})
    return content

def main():
    print(f"Loading processor for {MODEL_ID}...")
    try:
        processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    except Exception as e:
        print(f"Error loading processor: {e}")
        return

    # Pre-create images to reuse (optimization)
    archive_images = [create_dummy_image() for _ in range(ARCHIVE_IMAGES)]
    gen_images = [create_dummy_image() for _ in range(GEN_IMAGES)]
    
    # Construct base turns
    # 1. Archive Turn
    archive_content = build_turn_content(archive_images, "Image", ARCHIVE_PROMPT)
    # 2. Generation Turn (Generic)
    gen_content = build_turn_content(gen_images, "Image", GEN_PROMPT.format(generation=5))
    
    # Response
    response_content = [{"type": "text", "text": json.dumps({"selected": [1, 8, 13], "rationale": RATIONALE_TEXT})}]

    # We will estimate tokens for components
    # But apply_chat_template expects a list of messages.
    # To get accurate counts, we should construct the full history for each scenario.

    results = {}
    
    # Scenarios to test
    # 0, 1, 2, 3, 4, 5, 10, 20, -1
    turns_to_test = [0, 1, 2, 3, 4, 5, 10, 20, -1]

    for turns in turns_to_test:
        print(f"Calculating for chat_history_turns={turns}...")
        
        # Determine the worst-case history sequence
        # Worst case usually involves the Archive turn if permitted by 'turns'.
        
        # Max history length allowed
        max_history_items = turns if turns != -1 else TOTAL_GENERATIONS
        
        # Current turn is always a Generation Turn (except turn 0, but turn 0 has no history).
        # Wait, if we are at Turn 0 (Archive Branching), history is empty. Input is Archive Grid.
        # If we are at Turn 1 (Gen 1), history is Archive. Input is Gen 1 Grid.
        # If we are at Turn 20, history is Archive + Gen 1...19. Input is Gen 20 Grid.
        
        # We need the MAX over the whole session.
        # Step 0: Archive Input. History: 0.
        # Step 1: Gen 1 Input. History: 1 (Archive).
        # Step K: Gen K Input. History: min(K, max_history_items).
        
        # If history includes Archive, it's heavy.
        # We should check:
        # 1. Archive Step (Input=Archive, Hist=0)
        # 2. Step where history is maxed out AND includes Archive if possible.
        
        max_tokens_for_setting = 0
        
        # We iterate through possible steps in a session to find the peak.
        # Steps 0 to TOTAL_GENERATIONS.
        for step in range(TOTAL_GENERATIONS + 1):
            messages = []
            if SYSTEM_PROMPT:
                 messages.append({"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]})

            # History
            # History consists of previous steps.
            # E.g. at step 5 (Gen 5), previous steps are 0 (Archive), 1 (Gen 1), 2, 3, 4.
            # History items = step.
            # We keep only 'max_history_items' most recent.
            
            history_start = max(0, step - max_history_items)
            history_indices = range(history_start, step)
            
            for hist_idx in history_indices:
                if hist_idx == 0:
                    # Archive Turn
                    messages.append({"role": "user", "content": archive_content})
                    messages.append({"role": "assistant", "content": response_content})
                else:
                    # Gen Turn
                    messages.append({"role": "user", "content": gen_content})
                    messages.append({"role": "assistant", "content": response_content})
            
            # Current Input
            if step == 0:
                messages.append({"role": "user", "content": archive_content})
            else:
                messages.append({"role": "user", "content": gen_content})
            
            # Calculate tokens
            # We use a batch of 1 to simulate
            try:
                # Note: apply_chat_template might be slow with many images. 
                # Qwen processor handles it.
                inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")
                token_count = inputs["input_ids"].shape[1]
                if token_count > max_tokens_for_setting:
                    max_tokens_for_setting = token_count
            except Exception as e:
                print(f"Error calculating tokens for step {step}: {e}")
        
        print(f"  Max tokens: {max_tokens_for_setting}")
        
        # Add buffer (e.g. 500 tokens) and round up to nearest 1024 or similar?
        # User said "add some extra length to this inferred value to be safe".
        safe_limit = int(max_tokens_for_setting * 1.1) + 500
        results[turns] = safe_limit
        print(f"  Safe limit: {safe_limit}")

    # Write to JSON
    with open("data/max_model_len_map.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved map to data/max_model_len_map.json")

if __name__ == "__main__":
    main()
