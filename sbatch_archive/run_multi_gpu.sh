#!/bin/bash

# Multi-GPU Training Script for SPaRC SFT
# Usage: bash run_multi_gpu.sh [num_gpus]

NUM_GPUS=${1:-4}  # Default to 2 GPUs if not specified

echo "Starting multi-GPU training with $NUM_GPUS GPUs..."

# Option 1: Simple multi-GPU (recommended for most cases)
accelerate launch \
    --multi_gpu \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    train_sft.py