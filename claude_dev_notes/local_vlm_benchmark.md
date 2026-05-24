# Local VLM to replace Gemini — benchmark & setup

**Date:** 2026-05-24
**Why:** Out of Gemini tokens; need a local VLM to keep running the picbreeder-vlm
agent queries (the selection/publish loop in `collaborative_multi_agent.py` →
`agent_runner.py` → `chat.select_parents_from_grid`).

## TL;DR
- **Recommended / now running:** `Qwen/Qwen3-VL-30B-A3B-Instruct-FP8` (MoE, 3B active),
  sharded across both RTX 4090s. ~0.8 s/query, 24/24 valid JSON, best visual grounding.
- **Lighter fallback:** `Qwen/Qwen3-VL-8B-Instruct` — also perfect JSON, ~0.8 s, slightly
  more generic rationales.
- **Newer but NOT a good fit here:** `Qwen3.5-27B-FP8` (Feb 2026, natively multimodal) — it's
  a *reasoning* model that thinks for thousands of tokens by default → 25–155 s/query and
  frequent invalid (truncated) JSON. Usable only with `enable_thinking=False` (still ~3× slower,
  dense), and needs a backend code change to pass that flag. See "Qwen3.5" section.

## Hardware
- 2× NVIDIA RTX 4090 (24 GB each), 125 GB RAM, 36 CPU. CUDA 12.8, driver 570.
- `vllm 0.18.0` + `torch 2.10.0+cu128` + `transformers 4.57.6` in `.venv` — all working
  (the old `vllm_server_*.log` ABI errors in the repo root are stale from March).

## How the system talks to a local model
- `vlm_backends.create_vlm_backend("remote:<hf-id>")` → `VLLMClientBackend`, an
  OpenAI-compatible client. It reads `VLLM_BASE_URL` (default `http://localhost:8000/v1`).
- Each agent turn sends **~15 individual 128×128 images** (captioned "Image N"), NOT the
  composed grid (the grid PNG is only for logging). History adds prior turns.
- The model must return **strict JSON**: `{"selected": [...], "rationale": "..."}` plus
  optional `publish` / `quit` / `restart` / `mutation_mode` / `mutation_strength` / `color`.
- vLLM default allows 999 images/prompt, so no `--limit-mm-per-prompt` is strictly needed
  (we set it to 60 anyway).

## Candidates & results
All served TP=2 across both GPUs, `--max-model-len 50000`, `--max-num-seqs 16`.
Tested through the project's own `remote:` backend + `chat.extract_json_object`, on real
generation grids from a logged run (`logs_collaborative/th2_ag20_model-qwen3-vl-8b_..._s0/agents/agent_000`).

| Model | Valid JSON | Idx in-range | Grounding probe* | Latency (mean) | GPU | Notes |
|---|---|---|---|---|---|---|
| **Qwen3-VL-30B-A3B-FP8** (MoE) | 24/24 | 24/24 | ✅ best, most specific | **0.77 s** | ~22 GB/GPU (TP2) | **chosen default** |
| Qwen3-VL-8B | 24/24 | 24/24 | ✅ correct | 0.82 s | ~22 GB/GPU (TP2)† | strong lighter fallback |
| Qwen3.5-27B-FP8 (dense, reasoning) | ~1/3 (thinking overruns) | n/a | partial (didn't finish) | 25–155 s | ~22 GB/GPU (TP2) | thinking-on breaks it; see below |

\* Grounding probe on gen 8 (image viewed directly): correct answer is "images 7 & 14 are
nearly empty; image 8 is most complex." Both Qwen3-VL models got this right; the 30B's
selection rationales were consistently more visually specific (e.g. correctly described the
"central eye with a crescent below" structure vs the 8B's generic "circular patterns").

† 8B needs TP=2 to get the full 50k context: at 50k on a single 24 GB card the KV cache
doesn't fit (weights leave only ~2 GB, need ~7 GB → caps at ~15k context). On one GPU it's
fine up to ~15k context, which would let you run two 8B servers for 2× throughput.

## Qwen3.5 (the "newer model" check)
- `Qwen3.5` (released Feb 2026) is a **natively multimodal** unified family that reportedly
  beats Qwen3-VL on benchmarks. The 27B-FP8 (29 GB) and 9B (19 GB) checkpoints are **already
  in the HF cache** and vLLM 0.18 supports the arch (`Qwen3_5ForConditionalGeneration`).
- BUT it is a **reasoning model**: by default it emits long chain-of-thought before the JSON.
  In this high-throughput, JSON-only task that means 25–155 s/query and lots of truncated
  (invalid) JSON when it overruns the token budget.
- `chat_template_kwargs={"enable_thinking": false}` fixes output quality (2.5 s, valid JSON,
  good rationale). `reasoning_effort="none"` does NOT work. The project's `VLLMClientBackend`
  does not currently send `chat_template_kwargs`, so using Qwen3.5 would require a backend
  change — and it'd still be ~3× slower than the 30B MoE (dense vs 3B-active).
- Verdict: keep Qwen3-VL-30B-A3B-FP8 as default. Revisit Qwen3.5 only if max single-response
  quality matters more than throughput (then add the enable_thinking=False plumbing).
- Other comparable open models considered: InternVL3.5-30B-A3B (similar MoE class, worth an
  A/B if downloaded), GLM-4.5V / Qwen3-VL-235B / InternVL3.5-241B (too big for 2×24 GB).

## WHERE THINGS RUN / WHERE TO SEE OUTPUTS
Everything ran **locally on this machine** (both RTX 4090s), nothing on SLURM/cloud. No
picbreeder agent experiment is currently running — only the model server.

- **Model server (live):** `http://localhost:8000/v1`, PID in `vllm_server.pid`,
  log at `vllm_server.log` (repo root). Start/stop with `./serve_local_vlm.sh [stop]`.
- **Captured benchmark output:** `claude_dev_notes/results_qwen3vl-30b-a3b-fp8.txt`.
- **Earlier per-model server logs (ephemeral /tmp):** `/tmp/vllm_8b.log`,
  `/tmp/vllm_30b.log`, `/tmp/vllm_qwen35_27b.log`.
- **Benchmark scripts (re-runnable):**
  - `bench_vlm.py <model>` — faithful selection task on 3 real grids.
  - `probe_grounding.py <model>` — grounding-accuracy probe with known ground truth.
  - `stress_vlm.py <model>` — 24-call reliability test (valid-JSON / index / latency rates).
  - Pass e.g. `remote:Qwen/Qwen3-VL-30B-A3B-Instruct-FP8` (hits the running server).
- **Real source data used:** logged agent run under `logs_collaborative/th2_ag20_model-qwen3-vl-8b_tb-1_scheme-toggle_nopersonalities_fixed-sesh_s0/agents/agent_000/` (per-cell PNGs in `images/branch_000/gen_NNN/`, system prompt in `system_instruction.txt`).

## At-scale run on the local model (launched 2026-05-24)
- Command: `.venv/bin/python collaborative_multi_agent.py num_agents=1000 num_proc=10 seed=0 keep_query_images=true compress_completed_agents=true`
  (model defaults to the remote 30B; 10 agents in parallel hit the shared server; ~1.1 global gen/s).
- **Logs to:** `logs_collaborative/th1_ag20_model-qwen3-vl-30b-fp8_tb-1_scheme-toggle_nopersonalities_fixed-sesh_s0/`
  (auto-named to match the existing run scheme); stdout → `run_th1_qwen3vl30b_s0.log`.
- 1000 agents × 20 generations ≈ 20k generations (~5 h). Resumable via `resume=true experiment_dir=...`.
- Naming: `config.py` now maps known model ids to short aliases (`_MODEL_DIR_ALIASES`) and
  sanitizes the rest, so `remote:Qwen/Qwen3-VL-30B-A3B-Instruct-FP8` → `model-qwen3-vl-30b-fp8`.
- Completed agent dirs are zipped (compress=true) but query images are kept inside each zip.
- Verify the model's visual reasoning here:
  - `agents/agent_NNN/queries/branch_000_gen_NNN_view_00_selection.png` — numbered grid with the
    model's chosen cells boxed in red.
  - `agents/agent_NNN/logs/selection_history.jsonl` — `selected` indices + `rationale` per generation.
  - `agents/agent_NNN/images/branch_000/gen_NNN/idx_NN*.png` — the exact per-cell images sent.
  - `archive/images/` + archive grid — published images.

## How to use it with the system
1. Server is up (or run `./serve_local_vlm.sh`; `MODEL=Qwen/Qwen3-VL-8B-Instruct ./serve_local_vlm.sh` for the fallback).
2. Set the agent model to the remote string:
   - `config.py`: `model = "remote:Qwen/Qwen3-VL-30B-A3B-Instruct-FP8"` (or per-sweep in `sweep_configs.py`).
   - Optionally `.env`: `VLLM_BASE_URL=http://localhost:8000/v1` (8000 is the default anyway).

### Changes applied (2026-05-24)
- `config.py`: default `model` → `"remote:Qwen/Qwen3-VL-30B-A3B-Instruct-FP8"` (was
  `"gemini-2.5-pro"`). All un-overridden runs now use the local server.
- `.env`: added `VLLM_BASE_URL="http://localhost:8000/v1"`.
- `vlm_backends.is_local_model`: now returns False for `remote:`-prefixed models, so the
  orchestrator does **not** auto-start a second competing vLLM server when a `remote:` model
  is configured — all workers share the one server kept warm by `serve_local_vlm.sh`.
- `collaborative_multi_agent.py::_start_vlm_server`: honors env vars `VLLM_TP_SIZE` (default 1),
  `VLLM_MAX_NUM_SEQS`, `VLLM_GPU_MEM_UTIL` (default 0.9) so the orchestrator's *auto-start* path
  (used for registry-name models like `qwen3-vl-30b-fp8`, not `remote:`) can also run on this
  2-GPU box. Defaults unchanged → single-GPU / SLURM runs are unaffected.
- New files: `serve_local_vlm.sh` (start/stop the server), `bench_vlm.py`, `probe_grounding.py`,
  `stress_vlm.py` (benchmarks).

### NOT changed: sweep_configs.py
Sweep entries that hardcode `model=["qwen3-vl-30b-fp8"]` (registry name) still take the
*auto-start* path → they'd launch their own server and conflict with a manually-kept one on
this 2-GPU box. To make a sweep use the shared server instead, set its model list to
`["remote:Qwen/Qwen3-VL-30B-A3B-Instruct-FP8"]`. (Left for you — the file is open in your IDE.)

### Caveat for `pkill`
Don't `pkill -f` on the model name from a shell command that contains the model name — it
self-matches and kills its own shell (exit 144). Use `./serve_local_vlm.sh stop`, or
`kill $(cat vllm_server.pid)`.
