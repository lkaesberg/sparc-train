#!/bin/bash

# DeepSpeed Multi-GPU Training Script for SPaRC SFT  
# Usage: bash run_deepspeed.sh

echo "Starting DeepSpeed multi-GPU training..."

# Create simple DeepSpeed config for ZeRO Stage 2
cat > deepspeed_config.json << EOF
{
    "fp16": {
        "enabled": false
    },
    "bf16": {
        "enabled": true
    },
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "none"
        },
        "offload_param": {
            "device": "none"
        },
        "overlap_comm": true,
        "contiguous_gradients": true,
        "sub_group_size": 1e9,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9
    },
    "gradient_accumulation_steps": 16,
    "gradient_clipping": 1.0,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": 1
}
EOF

# Launch with DeepSpeed
accelerate launch \
    --use_deepspeed \
    --deepspeed_config_file deepspeed_config.json \
    train_sft.py 