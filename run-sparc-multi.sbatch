#!/bin/bash
#SBATCH --job-name=sparc
#SBATCH --partition=scc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -G A100:4
#SBATCH --time=48:00:00 
#SBATCH --mem=256G
#SBATCH -c 32
#SBATCH -C inet

module load gcc/13.2.0
module load cuda/12.6.2

source activate sparc
nvidia-smi

# Set model name here; can be overridden per-submission with: sbatch --export=ALL,MODEL_NAME=Your/Model run-sparc-multi.sbatch
export MODEL_NAME=${MODEL_NAME:-Qwen/Qwen3-0.6B}

accelerate launch \
    --config_file deepspeed_zero3.yaml \
    --num_processes 4 \
    --num_machines 1 \
    train_sft.py
