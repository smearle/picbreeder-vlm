"""Grounding-accuracy probe with known ground truth.

For gen_008 (viewed directly): images 7 and 14 are nearly-empty tiny dots;
image 8 is the most complex/detailed (multi-ring layered eye); most others are
crescent+sphere/eye shapes. We ask the model factual questions and check.
"""
import sys, time
from pathlib import Path
from picbreeder_vlm.vlm.vlm_backends import create_vlm_backend

RUN = Path("logs_collaborative/th2_ag20_model-qwen3-vl-8b_tb-1_scheme-toggle_nopersonalities_fixed-sesh_s0/agents/agent_000")

def load_cells(gen, color=False):
    d = RUN / "images" / "branch_000" / f"gen_{gen:03d}"
    pairs, i = [], 0
    while True:
        p = d / f"idx_{i:02d}{'' if color else '_gray'}.png"
        if not p.exists():
            break
        pairs.append((p.read_bytes(), f"Image {i}")); i += 1
    return pairs

PROBES = [
    "You are shown 15 numbered images. Which TWO image numbers are almost entirely empty (just a tiny dot or faint mark, very little content)? Answer with just the two numbers and a 5-word reason each.",
    "Among these 15 images, which ONE is the most visually complex / detailed (most layers, rings, or structure)? Give the number and a brief reason.",
]

def main():
    model = sys.argv[1]
    backend = create_vlm_backend(model)
    pairs = load_cells(8)
    print(f"\n===== {model} =====")
    for probe in PROBES:
        sess = backend.create_chat_session(max_turns=0)
        t0 = time.perf_counter()
        r = sess.send(image_caption_pairs=pairs, prompt=probe, temperature=0.2, max_new_tokens=300)
        dt = time.perf_counter() - t0
        print(f"\nQ: {probe[:70]}...\n[{dt:.1f}s] {r.text.strip()[:500]}")

if __name__ == "__main__":
    main()
