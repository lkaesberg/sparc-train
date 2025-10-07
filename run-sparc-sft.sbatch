#!/bin/bash
#SBATCH --job-name=sparc-sft
#SBATCH --partition=scc-gpu
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A100:4
#SBATCH --time=48:00:00 
#SBATCH --mem=256G
#SBATCH -c 32
#SBATCH -C inet
#SBATCH --output=logs_sft/%x_%j.log
#SBATCH --error=logs_sft/%x_%j.log

module load gcc/13.2.0
module load cuda/12.6.2

source activate sparc
nvidia-smi

# Central model selection (override by exporting MODEL_NAME before sbatch)
MODEL_NAME=${MODEL_NAME:-Qwen/Qwen3-32B}
echo "Model: ${MODEL_NAME}"
RUN_NAME_ADDITION=${RUN_NAME_ADDITION:-""}

# Discover allocated hosts for multi-node training
NODELIST=( $(scontrol show hostnames ${SLURM_JOB_NODELIST}) )
NUM_MACHINES=${#NODELIST[@]}

# User-tunable knobs (with safe defaults)
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
NUM_PROCESSES=$(( NUM_MACHINES * GPUS_PER_NODE ))
MAIN_ADDR="${NODELIST[0]}"
MAIN_PORT=${MAIN_PORT:-29500}

echo "Nodes: ${NODELIST[*]}"
echo "Machines: ${NUM_MACHINES} | GPUs per node: ${GPUS_PER_NODE} | Processes: ${NUM_PROCESSES}"
echo "Main addr: ${MAIN_ADDR}:${MAIN_PORT}"
echo "Model: ${MODEL_NAME}"

# Launch distributed SFT across the allocated nodes
srun --nodes=${NUM_MACHINES} --ntasks=${NUM_MACHINES} --nodelist="${NODELIST[*]}" \
  bash -lc "\
    module load gcc/13.2.0; \
    module load cuda/12.6.2; \
    source activate sparc; \
    nvidia-smi; \
    accelerate launch \
      --config_file deepspeed_zero3.yaml \
      --num_processes ${NUM_PROCESSES} \
      --num_machines ${NUM_MACHINES} \
      --main_process_ip ${MAIN_ADDR} \
      --main_process_port ${MAIN_PORT} \
      --machine_rank \$SLURM_PROCID \
      --rdzv_backend c10d \
      train_sft.py \
  "
