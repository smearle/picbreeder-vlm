#!/bin/bash
#SBATCH --job-name=vllm_daemon_235b
#SBATCH --account=torch_pr_84_tandon_advanced
#SBATCH --partition=h200_tandon
#SBATCH --qos=gpu48
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=480G
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --output=daemon_logs/vllm_daemon_%j.out
#SBATCH --error=daemon_logs/vllm_daemon_%j.err

# Persistent vLLM server for the qwen_235b_bf16 sweep. Agent jobs are CPU-only and
# connect via remote:<hf-id> + VLLM_BASE_URL. Avoids the 30-min per-job model load
# and dodges the GPU-util killer that fires ~2h17 (the load tail drags trailing-2h
# average below 60%). Agents keep this daemon busy so its util stays high.
#
# The #SBATCH lines above are the torch defaults. They can't reference shell
# variables (SLURM reads them before the shell runs), but SLURM applies them with
# LOWEST precedence -- override for another cluster without editing this file via
# CLI flags or the SBATCH_* env vars, e.g.
#     SBATCH_ACCOUNT=myacct SBATCH_PARTITION=gpu SBATCH_QOS=normal \
#         sbatch --gres=gpu:2 scripts/daemon_vllm_235b.sbatch
# The --output/--error paths are relative, so submit from the repo root (or a dir
# with a daemon_logs/ subdir). Everything in the body below is env-overridable with
# the torch value as the default.

set -euo pipefail

REPO_DIR="${PB_DAEMON_REPO:-/scratch/se2161/picbreeder-vlm}"
MODEL="${PB_DAEMON_MODEL:-Qwen/Qwen3-VL-235B-A22B-Instruct}"
PORT="${PB_DAEMON_PORT:-49152}"
TP_SIZE="${PB_DAEMON_TP:-4}"
MAX_MODEL_LEN="${PB_DAEMON_MAX_MODEL_LEN:-131072}"
MAX_NUM_SEQS="${PB_DAEMON_MAX_NUM_SEQS:-16}"
GPU_MEM_UTIL="${PB_DAEMON_GPU_MEM_UTIL:-0.9}"
PYTHON_BIN="${PB_DAEMON_PYTHON:-.venv/bin/python}"

cd "$REPO_DIR"
mkdir -p daemon_logs

# Respect an inherited HF_HOME; fall back to the torch scratch cache.
export HF_HOME="${HF_HOME:-/scratch/se2161/hf_cache}"

HOST=$(hostname -s)
# Agents discover the endpoint via $VLLM_ENDPOINT_FILE, then $CWD/vllm_daemon.endpoint
# (see _wait_for_remote_vllm_daemon). Keep it at the repo root, or export the same
# VLLM_ENDPOINT_FILE for both this job and the agent jobs.
ENDPOINT_FILE="${VLLM_ENDPOINT_FILE:-$REPO_DIR/vllm_daemon.endpoint}"

# Write endpoint atomically (tmp + mv) so agents can poll for it.
TMP_ENDPOINT="${ENDPOINT_FILE}.tmp.$$"
echo "http://${HOST}:${PORT}/v1" > "$TMP_ENDPOINT"
mv "$TMP_ENDPOINT" "$ENDPOINT_FILE"

trap 'rm -f "$ENDPOINT_FILE"' EXIT

echo "[daemon] $(date) starting vLLM ($MODEL) on ${HOST}:${PORT}, endpoint -> $ENDPOINT_FILE"

# max-model-len=131072 supports th=-1 (full chat history) which can blow past 50k.
# max-num-seqs=16 keeps KV cache in budget (bf16 weights ~117GB/GPU leave ~23GB for KV).
"$PYTHON_BIN" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --port "${PORT}" \
  --host 0.0.0.0 \
  --trust-remote-code \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --max-model-len "$MAX_MODEL_LEN" \
  --tensor-parallel-size "$TP_SIZE" \
  --max-num-seqs "$MAX_NUM_SEQS"
