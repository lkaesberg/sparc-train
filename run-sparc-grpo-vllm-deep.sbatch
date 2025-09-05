#!/bin/bash
#SBATCH --job-name=sparc-grpo-vllm-2n
#SBATCH --partition=scc-gpu
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=4          # 1 task per GPU
#SBATCH --gres=gpu:A100:4
#SBATCH --time=48:00:00
#SBATCH --mem=256G
#SBATCH -c 8                         # ~8 CPUs per GPU

module load gcc/13.2.0
module load cuda/12.6.2

source activate vllm

# Resolve node list
NODELIST=( $(scontrol show hostnames ${SLURM_JOB_NODELIST}) )
TRAIN_NODES=("${NODELIST[@]:0:2}")   # first 2 nodes for training
VLLM_NODE="${NODELIST[2]}"           # 3rd node for vLLM

echo "Train nodes: ${TRAIN_NODES[*]}"
echo "vLLM node:  ${VLLM_NODE}"

# Central model selection (override by exporting MODEL_NAME before sbatch)
MODEL_NAME=${MODEL_NAME:-Qwen/Qwen3-4B}
echo "Model: ${MODEL_NAME}"

# --- Launch vLLM server on node 3 ---
srun --nodes=1 --ntasks=1 --nodelist=${VLLM_NODE} \
  bash -lc "\
    module load gcc/13.2.0; \
    module load cuda/12.6.2; \
    nvidia-smi; \
    source activate vllm; \
    trl vllm-serve \
      --model ${MODEL_NAME} \
      --host 0.0.0.0 --port 8000 \
      --tensor_parallel_size 4 \
  " &

# Give the server time to start
sleep 30

# --- Training on 2 nodes (2×4 GPUs = 8 procs) ---
srun --nodes=2 --ntasks=8 --ntasks-per-node=4 --nodelist="${TRAIN_NODES[*]}" \
  bash -lc "\
    module load gcc/13.2.0; \
    module load cuda/12.6.2; \
    nvidia-smi; \
    source activate vllm; \
    accelerate launch \
      --config_file deepspeed_zero3.yaml \
      train_grpo_vllm_min.py \
        --model ${MODEL_NAME} \
        --vllm_server_host ${VLLM_NODE} \
        --wandb_project sparc-grpo \
  "
