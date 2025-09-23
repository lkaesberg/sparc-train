#!/bin/bash
#SBATCH --job-name=sparc-grpo-vllm-2n
#SBATCH --partition=scc-gpu
#SBATCH --time=48:00:00
#SBATCH --mem=256G
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1   
#SBATCH --gres=gpu:A100:4       # 4 GPUs per node
#SBATCH -c 8                    # ~8 CPUs per task is common

module load gcc/13.2.0
module load cuda/12.6.2

source activate vllm

# Resolve node list
NODELIST=( $(scontrol show hostnames ${SLURM_JOB_NODELIST}) )
TRAIN_NODES=("${NODELIST[@]:0:2}")     # nodes 0,1 for training
VLLM_NODE="${NODELIST[2]}"             # node 2 for vLLM

echo "Train nodes: ${TRAIN_NODES[*]}"
echo "vLLM node:  ${VLLM_NODE}"

# Central model selection (override by exporting MODEL_NAME before sbatch)
MODEL_NAME=${MODEL_NAME:-/mnt/vast-standard/home/kaesberg1/u16096/sparc-train/models/Qwen3-4B}
echo "Model: ${MODEL_NAME}"

# Launch vLLM server on node 3 as a background step
srun --nodes=1 --ntasks=1 --nodelist=${VLLM_NODE} \
  bash -lc "\
    module load gcc/13.2.0; \
    module load cuda/12.6.2; \
    nvidia-smi; \
    source activate vllm; \
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
    module load gcc/13.2.0; \
    module load cuda/12.6.2; \
    nvidia-smi; \
    source activate vllm; \
    accelerate launch \
      --config_file deepspeed_zero3.yaml \
      --num_processes 4 \
      --num_machines 2 \
      --main_process_ip ${TRAIN_NODES[0]} \
      --machine_rank $SLURM_NODEID \
      --rdzv_backend c10d \
      train_grpo_vllm_min.py \
        --model \"${MODEL_NAME}\" \
        --vllm_server_host ${VLLM_NODE} \
        --wandb_project sparc-grpo \
  "