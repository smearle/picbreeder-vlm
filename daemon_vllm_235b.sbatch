#!/bin/bash
#SBATCH --job-name=vllm_daemon_235b
#SBATCH --account=torch_pr_84_tandon_advanced
#SBATCH --partition=h200_tandon
#SBATCH --qos=gpu168
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=480G
#SBATCH --time=49:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/se2161/picbreeder-vlm/daemon_logs/vllm_daemon_%j.out
#SBATCH --error=/scratch/se2161/picbreeder-vlm/daemon_logs/vllm_daemon_%j.err

# Persistent vLLM server for the qwen_235b_bf16 sweep. Agent jobs are CPU-only and
# connect via remote:<hf-id> + VLLM_BASE_URL. Avoids the 30-min per-job model load
# and dodges the GPU-util killer that fires ~2h17 (the load tail drags trailing-2h
# average below 60%). Agents keep this daemon busy so its util stays high.

set -euo pipefail
cd /scratch/se2161/picbreeder-vlm
mkdir -p daemon_logs

export HF_HOME=/scratch/se2161/hf_cache

PORT=49152
HOST=$(hostname -s)
ENDPOINT_FILE=/scratch/se2161/picbreeder-vlm/vllm_daemon.endpoint

# Write endpoint atomically (tmp + mv) so agents can poll for it.
TMP_ENDPOINT="${ENDPOINT_FILE}.tmp.$$"
echo "http://${HOST}:${PORT}/v1" > "$TMP_ENDPOINT"
mv "$TMP_ENDPOINT" "$ENDPOINT_FILE"

trap 'rm -f "$ENDPOINT_FILE"' EXIT

echo "[daemon] $(date) starting vLLM on ${HOST}:${PORT}, endpoint -> $ENDPOINT_FILE"

# max-model-len=131072 supports th=-1 (full chat history) which can blow past 50k.
# max-num-seqs=16 keeps KV cache in budget (bf16 weights ~117GB/GPU leave ~23GB for KV).
.venv/bin/python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-VL-235B-A22B-Instruct \
  --port "${PORT}" \
  --host 0.0.0.0 \
  --trust-remote-code \
  --gpu-memory-utilization 0.9 \
  --max-model-len 131072 \
  --tensor-parallel-size 4 \
  --max-num-seqs 16
