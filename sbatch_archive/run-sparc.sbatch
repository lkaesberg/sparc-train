#!/bin/bash
#SBATCH --job-name=sparc
#SBATCH --partition=scc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -G A100:4
#SBATCH --time=48:00:00 
#SBATCH --mem=128G
#SBATCH -c 32

module load gcc/13.2.0-nvptx
module load cuda/12.6.2

source activate vllm
nvidia-smi
python3 train_sft.py
