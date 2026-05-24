#!/usr/bin/env bash
# Serve the local VLM that replaces Gemini, on an OpenAI-compatible endpoint.
#
# Default: Qwen3-VL-30B-A3B-Instruct-FP8 (MoE, 3B active) sharded across both
# RTX 4090s. ~22 GB/GPU, ~0.8 s/query, full 50k context. This is what the
# picbreeder agents talk to via the `remote:` backend (config.model =
# "remote:Qwen/Qwen3-VL-30B-A3B-Instruct-FP8", VLLM_BASE_URL=http://localhost:8000/v1).
#
# Usage:
#   ./serve_local_vlm.sh                 # start the 30B MoE (default)
#   MODEL=Qwen/Qwen3-VL-8B-Instruct ./serve_local_vlm.sh   # lighter fallback
#   ./serve_local_vlm.sh stop            # stop the running server
set -euo pipefail
cd "$(dirname "$0")"

MODEL="${MODEL:-Qwen/Qwen3-VL-30B-A3B-Instruct-FP8}"
PORT="${PORT:-8000}"
TP="${VLLM_TP_SIZE:-2}"
MAX_SEQS="${VLLM_MAX_NUM_SEQS:-16}"
GPU_MEM="${VLLM_GPU_MEM_UTIL:-0.93}"
MAX_LEN="${VLLM_MAX_MODEL_LEN:-50000}"
PIDFILE="vllm_server.pid"
LOGFILE="vllm_server.log"
PY=".venv/bin/python"

if [[ "${1:-}" == "stop" ]]; then
  if [[ -f "$PIDFILE" ]]; then kill "$(cat "$PIDFILE")" 2>/dev/null || true; rm -f "$PIDFILE"; echo "stopped"; else echo "no pidfile"; fi
  exit 0
fi

echo "Serving $MODEL on :$PORT (TP=$TP, max_num_seqs=$MAX_SEQS, max_len=$MAX_LEN)"
setsid $PY -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --port "$PORT" --trust-remote-code \
  --tensor-parallel-size "$TP" --gpu-memory-utilization "$GPU_MEM" \
  --max-model-len "$MAX_LEN" --max-num-seqs "$MAX_SEQS" \
  --limit-mm-per-prompt '{"image": 60}' \
  > "$LOGFILE" 2>&1 < /dev/null &
echo $! > "$PIDFILE"
echo "pid $(cat "$PIDFILE") -> $LOGFILE ; waiting for readiness..."
until curl -s "http://localhost:$PORT/v1/models" >/dev/null 2>&1 || grep -qiE "error:|traceback|out of memory|valueerror|runtimeerror" "$LOGFILE"; do sleep 4; done
if curl -s "http://localhost:$PORT/v1/models" >/dev/null 2>&1; then echo "READY: $(curl -s http://localhost:$PORT/v1/models | $PY -c 'import sys,json;print(json.load(sys.stdin)["data"][0]["id"])')"; else echo "FAILED — see $LOGFILE"; tail -15 "$LOGFILE"; exit 1; fi
