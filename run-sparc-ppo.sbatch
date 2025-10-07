#!/bin/bash
#SBATCH --job-name=sparc-ppo
#SBATCH --partition=scc-gpu
#SBATCH --time=48:00:00
#SBATCH --mem=256G
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1   
#SBATCH --gres=gpu:A100:4       # 4 GPUs per node
#SBATCH -c 32
#SBATCH -C inet&80gb
#SBATCH --output=logs_ppo/%x_%j.log
#SBATCH --error=logs_ppo/%x_%j.log

module load gcc/13.2.0
module load cuda/12.6.2

source activate vllm

# Central model selection (override by exporting MODEL_NAME before sbatch)
MODEL_NAME=${MODEL_NAME:-Qwen/Qwen3-0.6B}
echo "Model: ${MODEL_NAME}"
RUN_NAME_ADDITION=${RUN_NAME_ADDITION:-""}

# Discover allocated hosts
NODELIST=( $(scontrol show hostnames ${SLURM_JOB_NODELIST}) )
TOTAL_NODES=${#NODELIST[@]}

# User-tunable knobs (with safe defaults)
GPUS_PER_NODE=${GPUS_PER_NODE:-4}

# Derived distributed settings - use all nodes for training
NUM_MACHINES=${TOTAL_NODES}
NUM_PROCESSES=$(( TOTAL_NODES * GPUS_PER_NODE ))
MAIN_ADDR="${NODELIST[0]}"
MAIN_PORT=${MAIN_PORT:-29500}

echo "Train nodes (${NUM_MACHINES}): ${NODELIST[*]}"
echo "GPUs per node: ${GPUS_PER_NODE} | Total processes: ${NUM_PROCESSES}"

# Start distributed training across all nodes
srun --nodes=${NUM_MACHINES} --ntasks=${NUM_MACHINES} \
  bash -lc "\
    module load gcc/13.2.0; \
    module load cuda/12.6.2; \
    nvidia-smi; \
    source activate vllm; \
    accelerate launch \
      --config_file deepspeed_zero3.yaml \
      --num_processes ${NUM_PROCESSES} \
      --num_machines ${NUM_MACHINES} \
      --main_process_ip ${MAIN_ADDR} \
      --main_process_port ${MAIN_PORT} \
      --machine_rank \$SLURM_PROCID \
      --rdzv_backend c10d \
      train_ppo.py \
        --model \"${MODEL_NAME}\" \
        --wandb_project sparc-ppo \
        --run_name_addition ${RUN_NAME_ADDITION} \
  "