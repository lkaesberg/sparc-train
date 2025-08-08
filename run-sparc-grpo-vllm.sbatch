#!/bin/bash
#SBATCH --job-name=sparc-grpo-vllm
#SBATCH --partition=scc-gpu
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH -G A100:4
#SBATCH --time=48:00:00
#SBATCH --mem=256G
#SBATCH -c 32

module load gcc/13.2.0
module load cuda/12.6.2

source activate sparc

# Resolve node list
NODELIST=( $(scontrol show hostnames ${SLURM_JOB_NODELIST}) )
TRAIN_NODE=${NODELIST[0]}
VLLM_NODE=${NODELIST[1]}

echo "Train node: $TRAIN_NODE"
echo "vLLM node:  $VLLM_NODE"

# Launch vLLM server on node 2
srun --nodes=1 --ntasks=1 --nodelist=${VLLM_NODE} \
  bash -lc "\
    nvidia-smi; \
    vllm serve Qwen/Qwen3-0.6B --host 0.0.0.0 --port 8000 --tensor-parallel-size 4 --gpu-memory-utilization 0.9 \
  " &

sleep 30

# Launch GRPO trainer on node 1 pointing to vLLM server
srun --nodes=1 --ntasks=1 --nodelist=${TRAIN_NODE} \
  bash -lc "\
    nvidia-smi; \
    accelerate launch --config_file fsdp.config \
      train_grpo_vllm_min.py --model Qwen/Qwen3-0.6B --vllm_server_host ${VLLM_NODE} \
      --wandb_project sparc-grpo --wandb_entity larskaesberg-university-of-g-ttingen \
  "

wait


