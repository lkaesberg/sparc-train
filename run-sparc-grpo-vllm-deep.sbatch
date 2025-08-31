#!/bin/bash
#SBATCH --job-name=sparc-grpo-vllm-2n
#SBATCH --partition=scc-gpu
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1        # will use srun overrides for the multi-proc launch
#SBATCH --gres=gpu:A100:4
#SBATCH --time=48:00:00
#SBATCH --mem=256G
#SBATCH -c 32

module load gcc/13.2.0
module load cuda/12.6.2

source activate vllm

# Resolve node list
NODELIST=( $(scontrol show hostnames ${SLURM_JOB_NODELIST}) )
TRAIN_NODES=("${NODELIST[@]:0:2}")     # nodes 0,1 for training
VLLM_NODE="${NODELIST[2]}"             # node 2 for vLLM

echo "Train nodes: ${TRAIN_NODES[*]}"
echo "vLLM node:  ${VLLM_NODE}"

MODEL_NAME=${MODEL_NAME:-Qwen/Qwen3-0.6B}
echo "Model: ${MODEL_NAME}"

# Recommended NCCL/Torch envs (tune to your cluster)
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=^lo,docker0      # or set to your actual interface like enp1s0f0 or ib0
export NCCL_IB_DISABLE=0                   # 0 = enable IB if available; set =1 to disable IB
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=3600                  # ms (*1000) — increase if you expect long collectives
export TORCH_NCCL_TRACE_BUFFER_SIZE=1048576
export PYTHONFAULTHANDLER=1

# Launch vLLM server on node 3 (background)
srun --nodes=1 --ntasks=1 --nodelist=${VLLM_NODE} \
  bash -lc "\
    module load gcc/13.2.0; \
    module load cuda/12.6.2; \
    source activate vllm; \
    nvidia-smi; \
    trl vllm-serve \
      --model \"${MODEL_NAME}\" \
      --host 0.0.0.0 --port 8000 \
      --tensor_parallel_size 4 \
  " &

sleep 30

# --- TRAINING LAUNCH ---
# We have 2 train nodes × 4 GPUs/node = 8 processes total.
TOTAL_GPUS=$(( 2 * 4 ))   # adjust to match training nodes & gpus
NTASKS=${TOTAL_GPUS}
NTASKS_PER_NODE=4        # 4 GPUs per node

# Launch training with one task per GPU (so accelerate can spawn per-GPU processes)
srun --nodes=2 --ntasks=${NTASKS} --ntasks-per-node=${NTASKS_PER_NODE} --nodelist="${TRAIN_NODES[*]}" \
  --kill-on-bad-exit=1 --overlap \
  bash -lc "\
    module load gcc/13.2.0; \
    module load cuda/12.6.2; \
    source activate vllm; \
    nvidia-smi; \
    # machine_rank should be the node index (0 or 1)
    MACHINE_RANK=\${SLURM_NODEID}; \
    echo SLURM_PROCID=\$SLURM_PROCID SLURM_NODEID=\$SLURM_NODEID MACHINE_RANK=\$MACHINE_RANK; \
    accelerate launch \
      --config_file deepspeed_zero3.yaml \
      --num_processes ${TOTAL_GPUS} \
      --num_machines 2 \
      --main_process_ip ${TRAIN_NODES[0]} \
      --machine_rank \$MACHINE_RANK \
      --rdzv_backend c10d \
      --rdzv_endpoint ${TRAIN_NODES[0]}:29400 \
      train_grpo_vllm_min.py \
        --model \"${MODEL_NAME}\" \
        --vllm_server_host ${VLLM_NODE} \
        --wandb_project sparc-grpo \
  "
