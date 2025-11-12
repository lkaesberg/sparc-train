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
# For single file mode:
INPUT="${INPUT:-}"
OUTPUT="${OUTPUT:-}"

# For multi-file mode:
INPUT_DIR="${INPUT_DIR:-/mnt/vast-standard/home/kaesberg1/u16096/sparc-train/analyze/results/sparc}"
FILE_PATTERN="${FILE_PATTERN:-Qwen_Qwen3-4B.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-results/annotate}"

MODEL="${MODEL:-openai/gpt-oss-120b}"
PORT="${LLM_PORT:-8000}"
# how many final sentences from the solver trace to include in the prompt
LAST_N_SENTENCES="${LAST_N_SENTENCES:-100}"

TP_SIZE="${SLURM_GPUS_ON_NODE:-4}"            # tensor-parallel size (GPUs on node)
GPU_UTIL="${LLM_GPU_UTIL:-0.90}"

# Determine mode: single file or multi-file
if [[ -n "$INPUT" ]]; then
  # Single file mode (backward compatible)
  MODE="single"
  echo "Running in SINGLE FILE mode"
  echo "  Input: $INPUT"
  
  if [[ ! -f "$INPUT" ]]; then
    echo "ERROR: INPUT file does not exist: $INPUT" >&2
    exit 1
  fi
  
  if [[ -z "$OUTPUT" ]]; then
    BASENAME=$(basename "$INPUT" .jsonl)
    MODEL_SAFE=$(echo "$MODEL" | tr '/:' '__')
    OUTPUT="results/annotate/${BASENAME}.annotated_by_${MODEL_SAFE}.jsonl"
  fi
  
  # Create array with single file for unified processing
  INPUT_FILES=("$INPUT")
  OUTPUT_FILES=("$OUTPUT")
else
  # Multi-file mode
  MODE="multi"
  echo "Running in MULTI-FILE mode"
  echo "  Input directory: $INPUT_DIR"
  echo "  File pattern: $FILE_PATTERN"
  
  if [[ ! -d "$INPUT_DIR" ]]; then
    echo "ERROR: INPUT_DIR does not exist: $INPUT_DIR" >&2
    exit 1
  fi
  
  # Find all matching files
  shopt -s nullglob
  INPUT_FILES=("$INPUT_DIR"/$FILE_PATTERN)
  shopt -u nullglob
  
  if [[ ${#INPUT_FILES[@]} -eq 0 ]]; then
    echo "ERROR: No files matching pattern '$FILE_PATTERN' found in $INPUT_DIR" >&2
    exit 1
  fi
  
  echo "  Found ${#INPUT_FILES[@]} files to process:"
  for f in "${INPUT_FILES[@]}"; do
    echo "    - $(basename "$f")"
  done
  
  # Generate output paths
  OUTPUT_FILES=()
  MODEL_SAFE=$(echo "$MODEL" | tr '/:' '__')
  for INPUT_FILE in "${INPUT_FILES[@]}"; do
    BASENAME=$(basename "$INPUT_FILE" .jsonl)
    OUTPUT_FILE="$OUTPUT_DIR/${BASENAME}.annotated_by_${MODEL_SAFE}.jsonl"
    OUTPUT_FILES+=("$OUTPUT_FILE")
  done
fi

mkdir -p results/annotate
mkdir -p results/annotate/logs
mkdir -p "$OUTPUT_DIR"

source activate vllm
module load gcc/13.2.0 cuda/12.6.2 apptainer

LOG_FILE="results/annotate/logs/vllm_annotate_${SLURM_JOB_ID:-$$}.log"
echo "Starting vLLM server (model=$MODEL) -> $LOG_FILE"
apptainer exec -B $HF_HOME:$HF_HOME:rw --nv --cleanenv ~/vllm-gptoss.sif vllm serve "$MODEL" \
  --port "$PORT" \
  --tensor-parallel-size "$TP_SIZE" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --trust-remote-code \
  --download-dir $HF_HOME \
  --max-model-len 100000 \
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
echo "Running annotator with model=$MODEL port=$PORT last_n=$LAST_N_SENTENCES"
echo "Processing ${#INPUT_FILES[@]} file(s)..."
echo

# Process all files
SUCCESS_COUNT=0
FAIL_COUNT=0
TOTAL_FILES=${#INPUT_FILES[@]}

for i in "${!INPUT_FILES[@]}"; do
  INPUT_FILE="${INPUT_FILES[$i]}"
  OUTPUT_FILE="${OUTPUT_FILES[$i]}"
  
  FILE_NUM=$((i + 1))
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "Processing file $FILE_NUM/$TOTAL_FILES"
  echo "  Input:  $INPUT_FILE"
  echo "  Output: $OUTPUT_FILE"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  
  if python annotate.py --input "$INPUT_FILE" --output "$OUTPUT_FILE" --model "$MODEL" --port "$PORT" --last-n-sentences "$LAST_N_SENTENCES"; then
    echo "✓ Successfully annotated: $(basename "$INPUT_FILE")"
    SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
  else
    echo "✗ Failed to annotate: $(basename "$INPUT_FILE")" >&2
    FAIL_COUNT=$((FAIL_COUNT + 1))
  fi
  echo
done

# cleanup: politely stop server and wait a short time
if kill -0 $SERVER_PID 2>/dev/null; then
  echo "Stopping vLLM server..."
  kill -SIGINT $SERVER_PID || true
  sleep 2
  wait $SERVER_PID 2>/dev/null || true
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "ANNOTATION BATCH COMPLETE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Total files:      $TOTAL_FILES"
echo "Successful:       $SUCCESS_COUNT"
echo "Failed:           $FAIL_COUNT"
echo "Output directory: $OUTPUT_DIR"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [[ $FAIL_COUNT -gt 0 ]]; then
  exit 1
fi
