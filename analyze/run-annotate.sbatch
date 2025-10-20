#!/usr/bin/env bash
###############################################################################
# SBATCH helper to run annotation: starts a local vLLM server and runs the
# `analyze/annotate.py` script against a JSONL result file. Results are saved
# to analyze/results/annotate/<basename>.annotated.jsonl
#
# Usage (examples):
# sbatch analyze/run-annotate.sbatch --input analyze/results/sparc/myfile.jsonl --categories analyze/categories.txt
# or set environment variables INPUT/OUTPUT/CATEGORIES when calling sbatch.
###############################################################################
#SBATCH --job-name=sparc-annotate
#SBATCH --partition=scc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A100:1
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH -c 4
#SBATCH --output=analyze/results/annotate/%x-%j.out
#SBATCH --error=analyze/results/annotate/%x-%j.err

set -euo pipefail

# defaults (can be overridden via env or sbatch --export)
INPUT="${INPUT:-}"
OUTPUT="${OUTPUT:-}"
CATEGORIES="${CATEGORIES:-analyze/categories.txt}"
MODEL="${MODEL:-lkaesberg/Qwen3-14B-SPaRC-GRPO-8E}"
PORT="${LLM_PORT:-8000}"

if [[ -z "$INPUT" ]]; then
  echo "ERROR: INPUT not set. Provide --input <file.jsonl> or set INPUT env var." >&2
  exit 1
fi

#!/usr/bin/env bash
###############################################################################
# SBATCH helper to run annotation: starts a local vLLM server and runs the
# `analyze/annotate.py` script against a JSONL result file. Results are saved
# to analyze/results/annotate/<basename>.annotated.jsonl
#
# Usage examples:
# sbatch analyze/run-annotate.sbatch --input analyze/results/sparc/myfile.jsonl
# or set environment variables INPUT/OUTPUT when calling sbatch.
#
# Notes:
# - The vLLM server runs in the `vllm` conda environment.
# - The annotator runs in the `sparc` conda environment (required for
#   `sparc.generate_prompt`).
###############################################################################
#SBATCH --job-name=sparc-annotate
#SBATCH --partition=scc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A100:1
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH -c 4
#SBATCH --output=analyze/results/annotate/%x-%j.out
#SBATCH --error=analyze/results/annotate/%x-%j.err

set -euo pipefail

# defaults (can be overridden via env or sbatch --export)
INPUT="${INPUT:-}"
OUTPUT="${OUTPUT:-}"
# categories file is no longer required; annotator uses embedded categories.
MODEL="${MODEL:-lkaesberg/Qwen3-14B-SPaRC-GRPO-8E}"
PORT="${LLM_PORT:-8000}"
# how many final sentences from the solver trace to include in the prompt
LAST_N_SENTENCES="${LAST_N_SENTENCES:-100}"

if [[ -z "$INPUT" ]]; then
  echo "ERROR: INPUT not set. Provide --input <file.jsonl> or set INPUT env var." >&2
  exit 1
fi

if [[ -z "$OUTPUT" ]]; then
  BASENAME=$(basename "$INPUT" .jsonl)
  OUTPUT="analyze/results/annotate/${BASENAME}.annotated.jsonl"
fi

mkdir -p analyze/results/annotate
mkdir -p logs

# Start vLLM server in the `vllm` conda env. Keep CUDA modules if available.
module load gcc/13.2.0 || true
module load cuda/12.6.2 || true
source activate vllm

LOG_FILE="logs/vllm_annotate_${SLURM_JOB_ID:-$$}.log"
echo "Starting vLLM server (model=$MODEL) -> $LOG_FILE"
python -u -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --port "$PORT" \
  --trust-remote-code \
  >"$LOG_FILE" 2>&1 &
SERVER_PID=$!

# Wait for server readiness
for i in {1..300}; do
  if curl -s "http://127.0.0.1:${PORT}/v1/models" >/dev/null; then
    echo "vLLM ready"
    break
  fi
  printf "."; sleep 1
done

echo
echo "Running annotator: input=$INPUT output=$OUTPUT model=$MODEL port=$PORT last_n=$LAST_N_SENTENCES"

# Run annotator under the `sparc` conda env (required for `sparc.generate_prompt`).
conda deactivate 2>/dev/null || true
source activate sparc || true
python analyze/annotate.py --input "$INPUT" --output "$OUTPUT" --model "$MODEL" --port $PORT --last-n-sentences $LAST_N_SENTENCES

# cleanup: politely stop server and wait a short time
if kill -0 $SERVER_PID 2>/dev/null; then
  kill -SIGINT $SERVER_PID || true
  sleep 2
  wait $SERVER_PID 2>/dev/null || true
fi

echo "Annotation finished: $OUTPUT"
