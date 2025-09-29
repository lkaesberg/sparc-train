#!/bin/bash


# Launch with DeepSpeed
accelerate launch \
    --use_deepspeed \
    --deepspeed_config_file deepspeed_zero3.yaml \
    train_sft.py 