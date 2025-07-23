#!/bin/bash
#SBATCH --job-name=sparc
#SBATCH --partition=scc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -G A100:4
#SBATCH --time=48:00:00 
#SBATCH --mem=256G
#SBATCH -c 32

module load gcc/13.2.0
module load cuda/12.6.2

source activate sparc
nvidia-smi
accelerate launch \
    --config_file fsdp.config \
    train_sft.py
