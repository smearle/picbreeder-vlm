"""Reliability stress test: hammer the selection task many times at production
temperature and measure JSON validity, index-in-range rate, schema adherence,
and latency. This is the metric that matters for an unattended pipeline.
"""
import json, sys, time, statistics
from pathlib import Path
from vlm_backends import create_vlm_backend
from chat import extract_json_object

RUN = Path("logs_collaborative/th2_ag20_model-qwen3-vl-8b_tb-1_scheme-toggle_nopersonalities_fixed-sesh_s0/agents/agent_000")
SYS = (RUN / "system_instruction.txt").read_text(encoding="utf-8").strip()
GENS = [2, 5, 8, 11, 15, 18]
N_PER_GEN = 4  # -> 24 calls

def load_cells(gen, color=False):
    d = RUN / "images" / "branch_000" / f"gen_{gen:03d}"
    pairs, i = [], 0
    while True:
        p = d / f"idx_{i:02d}{'' if color else '_gray'}.png"
        if not p.exists(): break
        pairs.append((p.read_bytes(), f"Image {i}")); i += 1
    return pairs

def main():
    model = sys.argv[1]
    backend = create_vlm_backend(model)
    n=valid_json=in_range=has_sel=has_rat=0
    lats=[]
    bad=[]
    for gen in GENS:
        pairs = load_cells(gen)
        prompt = f"Above is the grid at generation {gen}.\nCurrent settings: color=OFF, mutation_mode=structure_only, mutation_strength=0.50."
        for _ in range(N_PER_GEN):
            n+=1
            sess = backend.create_chat_session(max_turns=0)
            t0=time.perf_counter()
            try:
                txt = sess.send(image_caption_pairs=pairs, prompt=prompt, system_instruction=SYS, temperature=0.7, max_new_tokens=2048).text
            except Exception as e:
                bad.append(f"gen{gen}: EXC {e}"); continue
            lats.append(time.perf_counter()-t0)
            parsed = extract_json_object(txt)
            if isinstance(parsed, dict):
                valid_json+=1
                sel = parsed.get("selected")
                if sel is not None or "color" in parsed: has_sel+=1
                if str(parsed.get("rationale","")).strip(): has_rat+=1
                if isinstance(sel,list) and all(isinstance(s,int) and 0<=s<len(pairs) for s in sel):
                    in_range+=1
                elif sel is None and "color" in parsed:
                    in_range+=1
                else:
                    bad.append(f"gen{gen}: sel={sel!r}")
            else:
                bad.append(f"gen{gen}: BAD JSON: {txt[:120]!r}")
    print(f"\n===== {model} =====")
    print(f"calls={n}  valid_json={valid_json}/{n}  indices_in_range={in_range}/{n}  has_selected={has_sel}/{n}  has_rationale={has_rat}/{n}")
    if lats:
        print(f"latency: mean={statistics.mean(lats):.2f}s  p50={statistics.median(lats):.2f}s  max={max(lats):.2f}s")
    if bad:
        print("PROBLEMS:")
        for b in bad[:12]: print("  -", b)

if __name__ == "__main__":
    main()
