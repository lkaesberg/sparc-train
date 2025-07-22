#!/bin/bash

# Multi-GPU training launch script for SPaRC SFT training
# Based on TRL documentation examples

# Number of GPUs to use (adjust as needed)
NUM_GPUS=4

echo "Launching SPaRC SFT training on $NUM_GPUS GPUs..."

# Check if DeepSpeed should be used (for larger models)
USE_DEEPSPEED=${1:-"false"}

if [ "$USE_DEEPSPEED" = "true" ]; then
    echo "Using DeepSpeed for memory-efficient training..."
    # Method 1: DeepSpeed ZeRO Stage 2
    accelerate launch \
        --config_file deepspeed_config.yaml \
        --num_processes $NUM_GPUS \
        train_sft.py
else
    echo "Using standard multi-GPU training..."
    # Method 2: Standard multi-GPU
    accelerate launch \
        --config_file accelerate_config.yaml \
        --num_processes $NUM_GPUS \
        train_sft.py
fi

# Alternative Method 3: Direct command line (uncomment to use instead)
# accelerate launch \
#     --multi_gpu \
#     --num_machines 1 \
#     --num_processes $NUM_GPUS \
#     train_sft.py

echo "Training completed!"

# Usage examples:
# ./launch_multi_gpu.sh          # Standard multi-GPU
# ./launch_multi_gpu.sh true     # DeepSpeed multi-GPU 