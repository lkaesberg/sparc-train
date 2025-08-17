#!/bin/bash
#SBATCH --job-name=sparc-grpo-vllm
#SBATCH --partition=gpu
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --time=48:00:00
#SBATCH -A p200941
#SBATCH -q default

module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.6.0
module load NCCL/2.22.3-GCCcore-13.3.0-CUDA-12.6.0

source sparc/bin/activate

# Resolve node list
NODELIST=( $(scontrol show hostnames ${SLURM_JOB_NODELIST}) )
TRAIN_NODES=("${NODELIST[@]:0:2}")     # nodes 0,1 for training
VLLM_NODE="${NODELIST[2]}"             # node 2 for vLLM

echo "Train nodes: ${TRAIN_NODES[*]}"
echo "vLLM node:  ${VLLM_NODE}"

# Central model selection (override by exporting MODEL_NAME before sbatch)
MODEL_NAME=${MODEL_NAME:-Qwen/Qwen3-14B}
echo "Model: ${MODEL_NAME}"

# Launch vLLM server on node 3 as a background step
srun --nodes=1 --ntasks=1 --nodelist=${VLLM_NODE} \
  bash -lc "\
    module load Python/3.12.3-GCCcore-13.3.0; \
    module load CUDA/12.6.0; \
    module load NCCL/2.22.3-GCCcore-13.3.0-CUDA-12.6.0; \
    nvidia-smi; \
    source sparc/bin/activate; \
    trl vllm-serve \
      --model "${MODEL_NAME}" \
      --host 0.0.0.0 --port 8000 \
      --tensor_parallel_size 4 \
  " &

# Give the server time to start
sleep 30

# Start distributed training across the 2 train nodes (16 total procs if 8 GPUs per node)
srun --nodes=2 --ntasks=2 --nodelist="${TRAIN_NODES[*]}" \
  bash -lc "\
    module load Python/3.12.3-GCCcore-13.3.0; \
    module load CUDA/12.6.0; \
    module load NCCL/2.22.3-GCCcore-13.3.0-CUDA-12.6.0; \
    nvidia-smi; \
    source sparc/bin/activate; \
    accelerate launch \
      --config_file deepspeed_zero3.yaml \
      --num_processes 8 \
      --num_machines 2 \
      --main_process_ip ${TRAIN_NODES[0]} \
      --machine_rank $SLURM_PROCID \
      --rdzv_backend c10d \
      train_grpo_vllm_min.py \
        --model "${MODEL_NAME}" \
        --vllm_server_host ${VLLM_NODE} \
        --wandb_project sparc-grpo \
  "