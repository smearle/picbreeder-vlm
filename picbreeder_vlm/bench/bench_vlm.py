"""Faithful local-VLM benchmark for the picbreeder-vlm selection task.

Feeds a real generation's 15 individual cell images (each captioned "Image N",
exactly as production does via select_parents_from_grid) to a model served on an
OpenAI-compatible endpoint, using the project's own remote: backend and the real
system instruction + prompt. Judges JSON validity, schema adherence, index
grounding, and latency.

Usage:
  .venv/bin/python bench_vlm.py remote:Qwen/Qwen3-VL-8B-Instruct
"""
import json
import sys
import time
from pathlib import Path

from picbreeder_vlm.vlm.vlm_backends import create_vlm_backend
from picbreeder_vlm.vlm.chat import extract_json_object

RUN = Path(
    "logs_collaborative/th2_ag20_model-qwen3-vl-8b_tb-1_scheme-toggle_"
    "nopersonalities_fixed-sesh_s0/agents/agent_000"
)
SYS_INSTR = (RUN / "system_instruction.txt").read_text(encoding="utf-8").strip()

# A few representative generations (different visual character) to avoid cherry-picking.
CASES = [
    {
        "gen": 8,
        "prompt": "Above is the grid at generation 8.\nCurrent settings: color=OFF, mutation_mode=structure_only, mutation_strength=0.50.",
        "color": False,
    },
    {
        "gen": 2,
        "prompt": "Above is the grid at generation 2.\nCurrent settings: color=OFF, mutation_mode=structure_only, mutation_strength=0.50.",
        "color": False,
    },
    {
        "gen": 15,
        "prompt": "Above is the grid at generation 15.\nCurrent settings: color=OFF, mutation_mode=structure_only, mutation_strength=0.50.",
        "color": False,
    },
]


def load_cells(gen: int, color: bool):
    d = RUN / "images" / "branch_000" / f"gen_{gen:03d}"
    pairs = []
    i = 0
    while True:
        suffix = "" if color else "_gray"
        p = d / f"idx_{i:02d}{suffix}.png"
        if not p.exists():
            break
        pairs.append((p.read_bytes(), f"Image {i}"))
        i += 1
    return pairs


def judge(parsed, n_images):
    """Score schema adherence + index validity. Returns (ok, notes)."""
    notes = []
    if isinstance(parsed, ValueError) or not isinstance(parsed, dict):
        return False, ["NOT VALID JSON"]
    if "selected" not in parsed and "color" not in parsed:
        notes.append("missing 'selected'")
    sel = parsed.get("selected")
    if isinstance(sel, list):
        bad = [s for s in sel if not (isinstance(s, int) and 0 <= s < n_images)]
        if bad:
            notes.append(f"out-of-range indices: {bad}")
    elif sel is not None and "color" not in parsed:
        notes.append(f"'selected' not a list/null: {sel!r}")
    has_rationale = bool(str(parsed.get("rationale", "")).strip())
    notes.append(f"rationale={'yes' if has_rationale else 'NO'}")
    ok = not isinstance(parsed, ValueError) and not any(
        x.startswith(("NOT", "out-of-range", "missing", "'selected'")) for x in notes
    )
    return ok, notes


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "remote:Qwen/Qwen3-VL-8B-Instruct"
    backend = create_vlm_backend(model)
    print(f"\n========== MODEL: {model} ==========")
    for case in CASES:
        pairs = load_cells(case["gen"], case["color"])
        session = backend.create_chat_session(max_turns=0)
        t0 = time.perf_counter()
        try:
            resp = session.send(
                image_caption_pairs=pairs,
                prompt=case["prompt"],
                system_instruction=SYS_INSTR,
                temperature=0.7,
                max_new_tokens=2048,
            )
            dt = time.perf_counter() - t0
            text = resp.text
        except Exception as e:
            print(f"\n--- gen {case['gen']} ({len(pairs)} imgs): EXCEPTION {e}")
            continue
        parsed = extract_json_object(text)
        ok, notes = judge(parsed, len(pairs))
        print(f"\n--- gen {case['gen']}  ({len(pairs)} imgs)  {dt:.1f}s  "
              f"{'OK' if ok else 'PROBLEM'}  [{', '.join(notes)}]")
        print(f"    raw ({len(text)} chars): {text[:600]}")
        if isinstance(parsed, dict):
            print(f"    parsed: {json.dumps(parsed)[:400]}")


if __name__ == "__main__":
    main()
