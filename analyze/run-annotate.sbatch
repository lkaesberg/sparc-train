#!/usr/bin/env bash

#SBATCH --job-name=sparc-annotate
#SBATCH --partition=scc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A100:4
#SBATCH --time=04:00:00
#SBATCH --mem=128G
#SBATCH -c 32
#SBATCH -C inet&80gb
#SBATCH --output=results/annotate/logs/%x-%j.out
#SBATCH --error=results/annotate/logs/%x-%j.out

set -euo pipefail

# defaults (can be overridden via env or sbatch --export)
INPUT="${INPUT:-/mnt/vast-standard/home/kaesberg1/u16096/sparc-train/analyze/results/sparc/lkaesberg_Qwen3-32B-SPaRC-GRPO-L.jsonl}"
OUTPUT="${OUTPUT:-}"
MODEL="${MODEL:-Qwen/Qwen3-32B}"
PORT="${LLM_PORT:-8000}"
# how many final sentences from the solver trace to include in the prompt
LAST_N_SENTENCES="${LAST_N_SENTENCES:-100}"

TP_SIZE="${SLURM_GPUS_ON_NODE:-4}"            # tensor-parallel size (GPUs on node)
GPU_UTIL="${LLM_GPU_UTIL:-0.90}"

if [[ -z "$INPUT" ]]; then
  echo "ERROR: INPUT not set. Provide --input <file.jsonl> or set INPUT env var." >&2
  exit 1
fi

if [[ -z "$OUTPUT" ]]; then
  BASENAME=$(basename "$INPUT" .jsonl)
  OUTPUT="results/annotate/${BASENAME}.annotated.jsonl"
fi

mkdir -p results/annotate
mkdir -p logs

source activate vllm

LOG_FILE="results/annotate/logs/vllm_annotate_${SLURM_JOB_ID:-$$}.log"
echo "Starting vLLM server (model=$MODEL) -> $LOG_FILE"
python -u -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --port "$PORT" \
  --tensor-parallel-size "$TP_SIZE" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --trust-remote-code \
  >"$LOG_FILE" 2>&1 &
SERVER_PID=$!

# Wait for server readiness
for i in {1..3600}; do
  if curl -s "http://127.0.0.1:${PORT}/v1/models" >/dev/null; then
    echo "vLLM ready"
    break
  fi
  printf "."; sleep 1
done

echo
echo "Running annotator: input=$INPUT output=$OUTPUT model=$MODEL port=$PORT last_n=$LAST_N_SENTENCES"

python annotate.py --input "$INPUT" --output "$OUTPUT" --model "$MODEL" --port $PORT --last-n-sentences $LAST_N_SENTENCES

# cleanup: politely stop server and wait a short time
if kill -0 $SERVER_PID 2>/dev/null; then
  kill -SIGINT $SERVER_PID || true
  sleep 2
  wait $SERVER_PID 2>/dev/null || true
fi

echo "Annotation finished: $OUTPUT"
