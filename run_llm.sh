#!/bin/bash
###############################################################################
# SLURM DIRECTIVES
###############################################################################
#SBATCH --job-name=sparc
#SBATCH --partition=scc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -G A100:4
#SBATCH --time=48:00:00
#SBATCH --mem=128G
#SBATCH -c 32
#SBATCH --output=logs/%x-%j.out
#SBATCH -C inet

set -euo pipefail
echo "[$(date +%F\ %T)] Job starting on $HOSTNAME"

###############################################################################
# 1 · Parameters you might tweak
###############################################################################
MODEL="${MODEL:-lkaesberg/Qwen3-32B-SPaRC-GRPO-L}"
PORT="${LLM_PORT:-8000}"
TP_SIZE="${SLURM_GPUS_ON_NODE}"   # == 4
GPU_UTIL="${LLM_GPU_UTIL:-0.90}"
HF_HOME="${HF_HOME:-$SCRATCH/huggingface}"
VENV_VLLM="$HOME/.venv_llm"       # server env
VENV_SPARC="$HOME/.venv_sparc"    # client env

###############################################################################
# 2 · (Optional) module load / environment setup
###############################################################################
# module purge
# module load cuda/12.4  gcc/12  python/3.11
export HF_HOME; mkdir -p "$HF_HOME" logs
source activate vllm
###############################################################################
# 3 · vLLM virtual‑env
###############################################################################
#if [[ ! -d "$VENV_VLLM" ]]; then
#  python3 -m venv "$VENV_VLLM"
#fi
#source "$VENV_VLLM/bin/activate"
#
#python - <<'PY'
#import importlib.util, subprocess, sys
#if importlib.util.find_spec("vllm") is None:
#    subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "vllm[serve]"])
#PY

###############################################################################
# 4 · Launch vLLM — write to a temp log file, tail only during startup
###############################################################################
LOG_DIR="$PWD/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/vllm_${SLURM_JOB_ID:-$$}.log"
touch $LOG_FILE

python -u -m vllm.entrypoints.openai.api_server \
       --model "$MODEL" \
       --port "$PORT" \
       --tensor-parallel-size "$TP_SIZE" \
       --gpu-memory-utilization "$GPU_UTIL" \
       --trust-remote-code \
       >"$LOG_FILE" 2>&1 &              # <-- background; output → file only
SERVER_PID=$!

# Show the log *for now* so you can watch weights load
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
# 5 · Health probe (quiet)
###############################################################################
echo -n "⌛ Waiting for vLLM"
for _ in {1..3600}; do
  if curl -s "http://127.0.0.1:${PORT}/v1/models" >/dev/null; then
    echo " … ready."
    break
  fi
  printf "."; sleep 1
done

# ---- server is healthy: stop showing its log ------------------------------
kill "$TAIL_PID" 2>/dev/null || true   # server keeps running, log stays on disk
echo "[`date +%T`] Server output silenced (see $LOG_FILE if needed)"

###############################################################################
# 6 · sparc‑puzzle virtual‑env (separate!)
###############################################################################
#if [[ ! -d "$VENV_SPARC" ]]; then
#  python3 -m venv "$VENV_SPARC"
#fi
# Activate sparc env *without* touching the running server
#source "$VENV_SPARC/bin/activate"

#python - <<'PY'
#import importlib.util, subprocess, sys
# sparc‑puzzle pulls in openai, pandas, etc. — keep it isolated here
#pkg = "sparc-puzzle"
#if importlib.util.find_spec("sparc") is None:
#    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", pkg])
#PY


###############################################################################
# 7 · Activate sparc env and run payload  (unchanged)
###############################################################################

export OPENAI_API_BASE="http://127.0.0.1:${PORT}/v1"
export OPENAI_API_KEY="${OPENAI_API_KEY:-LOCALHOST_ONLY}"

sparc --api-key "$OPENAI_API_KEY" \
      --base-url "$OPENAI_API_BASE" \
      --model "$MODEL" \
      --batch-size 20 \
      "$@"

echo "[$(date +%F\ %T)] sparc finished — job complete."

