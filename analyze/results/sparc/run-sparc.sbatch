#!/usr/bin/env bash

###############################################################################
# Lightweight SLURM helper to start a local vLLM server and run `sparc` client
# This variant assumes you use the shared `vllm` environment (conda/virtualenv)
# and intentionally keeps the script minimal: it does not create venvs.
# Usage: sbatch run-sparc.sbatch -- <sparc args>
###############################################################################
#SBATCH --job-name=sparc
#SBATCH --partition=scc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A100:4
#SBATCH --time=48:00:00
#SBATCH --mem=128G
#SBATCH -c 32
#SBATCH --output=logs/%x-%j.out
#SBATCH -C inet

set -euo pipefail
echo "[$(date +%F %T)] Job starting on $HOSTNAME"

###############################################################################
# 1 · User-configurable parameters (override with environment variables)
###############################################################################
MODEL="${MODEL:-lkaesberg/Qwen3-14B-SPaRC-GRPO-8E}"
PORT="${LLM_PORT:-8000}"
TP_SIZE="${SLURM_GPUS_ON_NODE:-4}"            # tensor-parallel size (GPUs on node)
GPU_UTIL="${LLM_GPU_UTIL:-0.90}"
HF_HOME="${HF_HOME:-$SCRATCH/huggingface}"

# The environment name / activate command used on the cluster. Keep this as the
# single source of truth; change if you use a different env name or activation.
ACTIVATE_CMD="source activate vllm"

export HF_HOME
mkdir -p "$HF_HOME" logs

echo "Model: $MODEL | Port: $PORT | TP size: $TP_SIZE | HF_HOME: $HF_HOME"

###############################################################################
# 2 · Activate pre-provisioned vllm environment
###############################################################################
echo "Activating vllm environment..."
eval "$ACTIVATE_CMD"

###############################################################################
# 3 · Launch vLLM (background) and capture logs
###############################################################################
LOG_DIR="$PWD/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/vllm_${SLURM_JOB_ID:-$$}.log"
touch "$LOG_FILE"

echo "Starting vLLM server (log -> $LOG_FILE)"
python -u -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --port "$PORT" \
  --tensor-parallel-size "$TP_SIZE" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --trust-remote-code \
  >"$LOG_FILE" 2>&1 &
SERVER_PID=$!

# Tail log while server boots so the SLURM job shows progress
tail -n +1 -f "$LOG_FILE" &
TAIL_PID=$!

cleanup() {
  echo "[$(date +%T)] Cleaning up…"
  kill -SIGINT "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
  kill "$TAIL_PID"   2>/dev/null || true
}
trap cleanup EXIT INT TERM

###############################################################################
# 4 · Health probe (wait until /v1/models responds)
###############################################################################
echo -n "⌛ Waiting for vLLM to become ready"
for _ in {1..3600}; do
  if curl -s "http://127.0.0.1:${PORT}/v1/models" >/dev/null; then
    echo " … ready."
    break
  fi
  printf "."; sleep 1
done

# Stop tailing logs now (server keeps running in background)
kill "$TAIL_PID" 2>/dev/null || true
echo "[$(date +%T)] Server started (log saved to $LOG_FILE)"

###############################################################################
# 5 · Run sparc client against local server
###############################################################################
export OPENAI_API_BASE="http://127.0.0.1:${PORT}/v1"
export OPENAI_API_KEY="${OPENAI_API_KEY:-LOCALHOST_ONLY}"

echo "Running sparc client..."
sparc --api-key "$OPENAI_API_KEY" \
      --base-url "$OPENAI_API_BASE" \
      --model "$MODEL" \
      --batch-size 20 \
      "$@"

echo "[$(date +%F %T)] sparc finished — job complete."