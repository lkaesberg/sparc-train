import torch
import gc
import numpy as np
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer, setup_chat_format
from sparc.prompt import generate_prompt
from sparc.validation import extract_solution_path, validate_solution, analyze_path
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import PartialState  # Add for multi-GPU support
import re
from transformers import TrainerCallback

model_name = "Qwen/Qwen3-8B"

# Multi-GPU device setup - get this early to check if we're main process
device_string = PartialState().process_index
is_main_process = PartialState().is_main_process

# Initialize wandb only on main process
if is_main_process:
    wandb.init(
        entity="larskaesberg-university-of-g-ttingen",
        project="sparc-grpo",
        name=f"{model_name}-sparc-grpo",
        config={
            "model": model_name,
            "dataset": "lkaesberg/SPaRC",
            "task": "group_relative_policy_optimization"
        }
    )
    print(f"DEBUG: Wandb initialized on main process (rank {device_string})")
else:
    print(f"DEBUG: Skipping wandb initialization on worker process (rank {device_string})")

original_dataset = load_dataset("lkaesberg/SPaRC", "all", split="train")
original_eval_dataset = load_dataset("lkaesberg/SPaRC", "all", split="test")

def expand_dataset_with_individual_solutions(dataset):
    """
    Expand dataset so each puzzle appears once per solution.
    Instead of: 1 puzzle with 3 solutions
    Creates: 3 samples, each with the same puzzle but 1 unique solution
    
    BENEFITS FOR TRAINING:
    - Model sees each valid solution path as a separate training example
    - Learns multiple valid approaches to the same puzzle
    - Better pattern recognition for different solution strategies
    - Increased training data diversity without collecting new puzzles
    - More focused learning: each sample has exactly one target solution
    """
    expanded_samples = []
    original_count = len(dataset)
    total_solutions = 0
    
    for sample in dataset:
        puzzle_data = sample.copy()
        solutions = puzzle_data.get('solutions', [])
        
        if len(solutions) == 0:
            # Keep samples with no solutions as-is
            expanded_samples.append(puzzle_data)
            continue
        
        total_solutions += len(solutions)
        
        # Create one sample per solution
        for solution in solutions:
            new_sample = puzzle_data.copy()
            # Replace the solutions array with just this one solution
            new_sample['solutions'] = [solution]
            expanded_samples.append(new_sample)
    
    # Convert back to HuggingFace dataset format
    from datasets import Dataset
    
    if is_main_process:
        print(f"DEBUG: Dataset expansion - {original_count} puzzles with {total_solutions} total solutions")
        print(f"DEBUG: Average solutions per puzzle: {total_solutions/original_count:.2f}")
    
    return Dataset.from_list(expanded_samples)

# Expand both training and evaluation datasets
if is_main_process:
    print(f"DEBUG: Original train dataset size: {len(original_dataset)}")
    print(f"DEBUG: Original eval dataset size: {len(original_eval_dataset)}")

# For GRPO, we can optionally expand the dataset
# original_dataset = expand_dataset_with_individual_solutions(original_dataset)

if is_main_process:
    print(f"DEBUG: Train dataset size: {len(original_dataset)}")

# GRPO Training Configuration
training_args = GRPOConfig(
    output_dir="./tmp",
    report_to="wandb",
    logging_steps=10,
    save_steps=None,  # Disable intermediate checkpoint saving
    eval_steps=100,  # Evaluate less frequently to save memory
    warmup_steps=100,
    max_steps=10000,
    learning_rate=5e-6,  # Lower learning rate for GRPO
    per_device_train_batch_size=1,  # Reduce per-device batch size to save memory
    per_device_eval_batch_size=1,
    eval_accumulation_steps=1,
    gradient_accumulation_steps=4,  # Higher gradient accumulation
    remove_unused_columns=False,
    group_by_length=False,
    optim="adamw_torch_fused",
    gradient_checkpointing=False,
    bf16=True,
    save_strategy="no",
    eval_strategy="steps",
    do_eval=True,
    load_best_model_at_end=False,
    metric_for_best_model="eval_solution_accuracy",
    greater_is_better=True,
    save_total_limit=1,
    ddp_find_unused_parameters=False,
    logging_dir="./logs",
    logging_first_step=True,
    dataloader_pin_memory=False,
    fp16_full_eval=False,
    dataloader_num_workers=0,
    prediction_loss_only=False,
    
    # GRPO-specific parameters
    max_completion_length=30000,  # Maximum completion length
    temperature=0.7,  # Sampling temperature
    num_generations=4,  # Reduced from default 8 for memory efficiency
    
    # vLLM acceleration for generation
    use_vllm=True,  # Enable vLLM for faster generation
    vllm_mode="colocate",  # Run vLLM in same process, sharing GPU memory
    vllm_gpu_memory_utilization=0.3,  # Lower vLLM GPU memory utilization to reduce OOM risk
    
    # REGULARIZATION
    weight_decay=0.01,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    dataloader_drop_last=True,
    
    # FSDP optimizations for GRPO
    use_liger_loss=True,  # 40% memory savings
)

# Multi-GPU device setup
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    #device_map={'': device_string},  # Proper device placement for multi-GPU
    attn_implementation="sdpa",
    torch_dtype=torch.bfloat16,
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Disable cache during training to save memory
model.config.use_cache = False


def transform_to_prompt_format(dataset):
    """Transform dataset to the format expected by GRPO (prompt-only format)"""
    def format_sample(example):
        # For GRPO, we need the prompt and preserve puzzle data for reward validation
        prompt = generate_prompt(example)
        return {
            "prompt": [{"role": "user", "content": prompt}],
            "puzzle_data": example  # Preserve original puzzle data for reward function
        }
    
    return dataset.map(format_sample)

# Transform datasets to prompt format for GRPO
dataset = transform_to_prompt_format(original_dataset)
eval_dataset_grpo = transform_to_prompt_format(original_eval_dataset)

if is_main_process:
    print(f"DEBUG: Transformed datasets to GRPO prompt format")
    print(f"DEBUG: Sample prompt: {dataset[0]['prompt'][:200]}...")

# Wrapper to force left padding/truncation for all tokenization calls used by TRL
class LeftPadTokenizerWrapper:
    def __init__(self, base_tokenizer):
        self._t = base_tokenizer
        # Ensure base tokenizer is configured for left padding/truncation
        self._t.padding_side = "left"
        self._t.truncation_side = "left"

    def __call__(self, *args, **kwargs):
        # Default to padding and truncation if not specified
        kwargs.setdefault("padding", True)
        kwargs.setdefault("truncation", True)
        # Enforce left padding on every call
        self._t.padding_side = "left"
        self._t.truncation_side = "left"
        return self._t(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._t, name)

def create_sparc_validation_reward(original_dataset):
    """
    Create a reward function that has access to the original puzzle data.
    This is a closure that captures the original dataset for proper validation.
    """
    # Create a mapping from prompts to puzzle data for quick lookup
    prompt_to_puzzle = {}
    for example in original_dataset:
        prompt = generate_prompt(example)
        prompt_to_puzzle[prompt] = example
    
    def sparc_validation_reward(completions, prompts, **kwargs):
        """
        Reward function using actual SPaRC validation metrics.
        Uses the same validation functions as the SFT trainer for consistency.
        """
        rewards = []
        
        for i, (completion, prompt) in enumerate(zip(completions, prompts)):
            try:
                # Handle both string prompts and conversation format
                if isinstance(prompt, list) and len(prompt) > 0 and 'content' in prompt[0]:
                    prompt_text = prompt[0]['content']
                else:
                    prompt_text = str(prompt)
                
                # Handle both string completions and conversation format
                if isinstance(completion, list) and len(completion) > 0 and 'content' in completion[0]:
                    completion_text = completion[0]['content']
                else:
                    completion_text = str(completion)
                
                # Look up the original puzzle data
                puzzle = prompt_to_puzzle.get(prompt_text)
                
                if puzzle is None:
                    # Fallback: try basic format validation
                    if is_main_process and i < 3:  # Only print first few for debugging
                        print(f"DEBUG: No puzzle found for prompt:")
                        print(f"DEBUG: Prompt text (first 100 chars): {prompt_text[:100]}...")
                        print(f"DEBUG: Available puzzle prompts: {len(prompt_to_puzzle)} total")
                        # Show first few available prompts for comparison
                        available_prompts = list(prompt_to_puzzle.keys())[:2]
                        for j, avail_prompt in enumerate(available_prompts):
                            print(f"DEBUG: Available prompt {j} (first 100 chars): {avail_prompt[:100]}...")
                    
                    if "####" in completion_text and "(" in completion_text and ")" in completion_text:
                        reward = 0.2  # Small reward for format
                    else:
                        reward = 0.0
                    rewards.append(reward)
                    continue
                
                # Try to extract solution path from completion
                extracted_path = extract_solution_path(completion_text, puzzle)
                
                if extracted_path is not None:
                    # Use the same validation logic as SFT trainer
                    is_correct = validate_solution(extracted_path, puzzle)
                    
                    if is_correct:
                        reward = 1.0  # Perfect solution
                    else:
                        # Get detailed analysis for partial rewards
                        analysis = analyze_path(extracted_path, puzzle)
                        
                        reward = 0.0
                        # Give partial rewards based on validation criteria (same as SFT)
                        if analysis.get("starts_at_start_ends_at_exit", False):
                            reward += 0.25
                        if analysis.get("connected_line", False):
                            reward += 0.25
                        if analysis.get("non_intersecting_line", False):
                            reward += 0.25
                        if analysis.get("no_rule_crossing", False):
                            reward += 0.25
                else:
                    # No valid path extracted - give small reward for format
                    if "####" in completion_text and "(" in completion_text and ")" in completion_text:
                        reward = 0.1  # Small reward for attempting correct format
                    else:
                        reward = 0.0
                
                rewards.append(reward)
                
            except Exception as e:
                if is_main_process and i < 5:  # Only print first few errors
                    print(f"Error in SPaRC validation reward for completion {i}: {e}")
                rewards.append(0.0)
        
        return rewards
    
    return sparc_validation_reward

# GRPO doesn't use compute_metrics - it tracks rewards internally

if is_main_process:
    print(f"DEBUG: Eval dataset size: {len(original_eval_dataset)}")
    print(f"DEBUG: Train dataset size: {len(original_dataset)}")
    print(f"DEBUG: Model device: {model.device}")
    print(f"DEBUG: Wandb project: {wandb.run.project if wandb.run else 'No active run'}")

# Create the SPaRC validation reward function with access to original puzzle data
# Combine both train and eval datasets for comprehensive puzzle lookup
combined_dataset = []
for example in original_dataset:  # Use original dataset
    combined_dataset.append(example)
for example in original_eval_dataset:  # Use original eval dataset
    combined_dataset.append(example)

sparc_reward_func = create_sparc_validation_reward(combined_dataset)

# Initialize GRPO Trainer
# Wrap tokenizer to enforce left padding from within TRL calls
tokenizer = LeftPadTokenizerWrapper(tokenizer)

trainer = GRPOTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=eval_dataset_grpo,  # Use the transformed eval dataset
    processing_class=tokenizer,
    reward_funcs=sparc_reward_func,  # Use SPaRC validation-based reward function
)

if is_main_process:
    print("DEBUG: Starting GRPO training...")
trainer.train()

# Save only the final model
if is_main_process:
    print("DEBUG: GRPO training completed. Saving final model...")
    
    # Save the final model in standard format
    final_model_dir = "./final_grpo_model"
    trainer.save_model(final_model_dir)
    
    # Also save the tokenizer
    tokenizer.save_pretrained(final_model_dir)
    
    print(f"DEBUG: Final GRPO model saved to {final_model_dir}")

# Clean up memory after training
if is_main_process:
    print("DEBUG: Cleaning up memory...")
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    if is_main_process:
        print(f"DEBUG: CUDA memory allocated after training: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"DEBUG: CUDA memory reserved after training: {torch.cuda.memory_reserved() / 1024**3:.2f} GB") 