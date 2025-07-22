import torch
import gc
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer, setup_chat_format
from sparc.prompt import generate_prompt
from sparc.validation import extract_solution_path, validate_solution, analyze_path
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import PartialState  # Add for multi-GPU support
import re
import numpy as np
from transformers import TrainerCallback

class DebugEvaluationCallback(TrainerCallback):
    """Custom callback to debug evaluation events and manage memory aggressively"""
    
    def on_evaluate(self, args, state, control, model=None, tokenizer=None, **kwargs):
        if is_main_process:  # Only print on main process
            print(f"DEBUG: Evaluation started at step {state.global_step}")
        
        # Aggressive memory cleanup before evaluation
        gc.collect()  # Force garbage collection
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()  # Wait for all operations to complete
            if is_main_process:
                print(f"DEBUG: Aggressive memory cleanup before evaluation")
                print(f"DEBUG: CUDA memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    
    def on_log(self, args, state, control, model=None, tokenizer=None, logs=None, **kwargs):
        if logs and any(key.startswith('eval_') for key in logs.keys()) and is_main_process:
            print(f"DEBUG: Evaluation metrics logged at step {state.global_step}: {logs}")
        
        # Aggressive memory cleanup after evaluation
        if logs and any(key.startswith('eval_') for key in logs.keys()):
            gc.collect()  # Force garbage collection
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()  # Wait for all operations to complete
                if is_main_process:
                    print(f"DEBUG: Aggressive memory cleanup after evaluation")
                    print(f"DEBUG: CUDA memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

model_name = "Qwen/Qwen3-0.6B"

# Multi-GPU device setup - get this early to check if we're main process
device_string = PartialState().process_index
is_main_process = PartialState().is_main_process

# Initialize wandb only on main process
if is_main_process:
    wandb.init(
        entity="larskaesberg-university-of-g-ttingen",
        project="sparc-sft",
        name=f"{model_name}-sparc-sft",
        config={
            "model": model_name,
            "dataset": "lkaesberg/SPaRC",
            "task": "supervised_fine_tuning"
        }
    )
    print(f"DEBUG: Wandb initialized on main process (rank {device_string})")
else:
    print(f"DEBUG: Skipping wandb initialization on worker process (rank {device_string})")


dataset = load_dataset("lkaesberg/SPaRC", "all", split="train")
eval_dataset = load_dataset("lkaesberg/SPaRC", "all", split="test")

# Limit evaluation dataset size to prevent memory issues
max_eval_size = 100  # Limit to 100 samples for evaluation
if len(eval_dataset) > max_eval_size:
    if is_main_process:  # Only print on main process
        print(f"DEBUG: Limiting eval dataset from {len(eval_dataset)} to {max_eval_size} samples for memory efficiency")
    eval_dataset = eval_dataset.select(range(max_eval_size))

# Print memory info if available (only on main process)
if torch.cuda.is_available() and is_main_process:
    print(f"DEBUG: CUDA memory allocated before training: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"DEBUG: CUDA memory reserved before training: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    print(f"DEBUG: CUDA memory total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Force cleanup before training
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print(f"DEBUG: After cleanup - CUDA memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

training_args = SFTConfig(
    output_dir="./tmp",
    report_to="wandb",
    logging_steps=10,
    save_steps=500,
    eval_steps=250,  # Evaluate less frequently to save memory
    warmup_steps=100,
    max_steps=1000,
    learning_rate=5e-5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,  # Keep eval batch size at absolute minimum
    eval_accumulation_steps=1,  # Process eval in smallest possible steps
    gradient_accumulation_steps=16,
    max_seq_length=2048,  # Reduce sequence length to save memory
    remove_unused_columns=False,
    group_by_length=True,
    optim="adamw_torch_fused",  # Better performance than adamw_torch
    gradient_checkpointing=True,  # Reduce memory usage
    bf16=True,  # Use bfloat16 for better performance if supported
    save_strategy="steps",  # Save based on steps, not epochs
    eval_strategy="steps",  # Evaluate based on steps
    do_eval=True,  # Explicitly enable evaluation
    load_best_model_at_end=False,  # Disable to save memory
    metric_for_best_model="eval_solution_accuracy",  # Use your custom metric with eval_ prefix
    greater_is_better=True,  # Higher solution_accuracy is better
    save_total_limit=2,  # Keep only 2 best checkpoints to save disk space
    ddp_find_unused_parameters=False,  # Optimize for multi-GPU
    logging_dir="./logs",  # Add logging directory
    logging_first_step=True,  # Log the first step
    dataloader_pin_memory=False,  # Disable pin memory to avoid potential issues
    fp16_full_eval=False,  # Don't use fp16 during eval to avoid precision issues
    dataloader_num_workers=0,  # Disable multiprocessing to save memory
    prediction_loss_only=False,  # We need predictions for custom metrics
    
    # CRITICAL: Enable padding-free batching for massive memory savings
    padding_free=True,  # Eliminates padding completely, huge memory reduction
    
    # Use flash attention for better memory efficiency
    model_init_kwargs={"attn_implementation": "flash_attention_2"},
    
    # Additional memory optimizations
    eval_on_start=False,  # Don't evaluate at start
    include_inputs_for_metrics=False,  # Don't include inputs in metrics computation
    eval_do_concat_batches=False
)

# Multi-GPU device setup
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map={'': device_string}  # Proper device placement for multi-GPU
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Set pad token if not present
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def formatting_prompts_func(examples):
    # Handle both individual and batched calls from SFTTrainer
    
    # Check if this is a single example (batched=False) or batch (batched=True)
    if isinstance(examples, dict) and len(examples) > 0:
        first_key = list(examples.keys())[0]
        first_value = examples[first_key]
        
        # Case 1: Individual example (values are not lists)
        if not isinstance(first_value, list):
            messages = [
                  {
                    "role": "system",
                    "content": "You are an expert at solving puzzles games.",
                  },
                  {
                    "role": "user", 
                    "content": generate_prompt(examples)
                  },
                {
                    "role": "assistant",
                    "content": f"#### ({', '.join(map(lambda x: f'({x["x"]}, {x["y"]})', examples['solutions'][0]['path']))})"
                }
            ]
            
            formatted_text = tokenizer.apply_chat_template(messages, tokenize=False)
            return formatted_text
        
        # Case 2: Batched examples (values are lists)
        else:
            output_text = []
            
            # Iterate through all examples in the batch
            for i in range(len(examples["solutions"])):
                # Extract individual example from batch
                example = {key: values[i] for key, values in examples.items()}
                
                messages = [
                      {
                        "role": "system",
                        "content": "You are an expert at solving puzzles games.",
                      },
                      {
                        "role": "user", 
                        "content": generate_prompt(example)
                      },
                    {
                        "role": "assistant",
                        "content": f"#### ({', '.join(map(lambda x: f'({x["x"]}, {x["y"]})', example['solutions'][0]['path']))})"
                    }
                ]
                
                formatted_text = tokenizer.apply_chat_template(messages, tokenize=False)
                output_text.append(formatted_text)
            
            return output_text

def create_compute_metrics(eval_dataset):
    """Create a compute_metrics function that has access to the dataset"""
    def compute_metrics(eval_pred):
        # EXTREME memory cleanup before evaluation
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Ensure no gradients are computed during evaluation
        with torch.no_grad():
            if is_main_process:  # Only print debug on main process
                print(f"\nDEBUG: compute_metrics called with eval_pred type: {type(eval_pred)}")
                if torch.cuda.is_available():
                    print(f"DEBUG: CUDA memory at start of compute_metrics: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            
            predictions, labels = eval_pred
            
            if is_main_process:
                print(f"DEBUG: predictions shape: {predictions.shape if hasattr(predictions, 'shape') else 'no shape'}")
                print(f"DEBUG: predictions type: {type(predictions)}")
                if torch.cuda.is_available():
                    print(f"DEBUG: CUDA memory before detaching: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            
            # Handle the case where predictions might be nested tuples/lists from SFTTrainer
            if isinstance(predictions, tuple):
                predictions = predictions[0]
            
            # CRUCIAL: Detach predictions from computational graph to free GPU memory
            if hasattr(predictions, 'detach'):
                predictions = predictions.detach()
                if is_main_process:
                    print(f"DEBUG: Detached predictions from computational graph")
            
            # Also detach labels if they have gradients
            if hasattr(labels, 'detach'):
                labels = labels.detach()
                if is_main_process:
                    print(f"DEBUG: Detached labels from computational graph")
            
            # Check memory after detaching
            if is_main_process and torch.cuda.is_available():
                print(f"DEBUG: CUDA memory after detaching: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            

            decoded_preds = []
            num_predictions = len(predictions) if hasattr(predictions, '__len__') else predictions.shape[0]
            
            if is_main_process:
                print(f"DEBUG: Processing only {num_predictions} predictions")
            
            for i in range(num_predictions):
                try:
                    # Get single prediction
                    if hasattr(predictions, '__getitem__'):
                        single_pred = predictions[i]
                    else:
                        single_pred = predictions[i:i+1]
                    
                    # Ensure this single prediction is also detached
                    if hasattr(single_pred, 'detach'):
                        single_pred = single_pred.detach()
                    
                    # Move this single prediction to CPU immediately
                    if hasattr(single_pred, 'cpu'):
                        single_pred_cpu = single_pred.cpu().numpy()
                    else:
                        single_pred_cpu = single_pred
                    
                    # Aggressive cleanup after each prediction
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    # Decode this single prediction
                    if len(single_pred_cpu.shape) > 1:
                        # If prediction is logits [seq_len, vocab_size]
                        predicted_ids = single_pred_cpu.argmax(axis=-1)
                    else:
                        # If prediction is already token IDs
                        predicted_ids = single_pred_cpu
                    
                    # Decode this single sequence
                    try:
                        decoded = tokenizer.decode(predicted_ids, skip_special_tokens=True)
                        decoded_preds.append(decoded)
                    except Exception as decode_e:
                        if is_main_process:
                            print(f"DEBUG: Error decoding prediction {i}: {decode_e}")
                        decoded_preds.append("")
                    
                    # Aggressive cleanup after each step
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                except Exception as e:
                    if is_main_process:
                        print(f"DEBUG: Error processing prediction {i}: {e}")
                    decoded_preds.append("")
                    # Aggressive cleanup even on error
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            # Debug: Print some examples to see what we're getting (only main process)
            if is_main_process:
                print(f"\nDEBUG: Number of predictions decoded: {len(decoded_preds)}")
                if len(decoded_preds) > 0:
                    print(f"DEBUG: First prediction: {decoded_preds[0][:200]}...")
            
            # Evaluate ALL decoded predictions (very few)
            max_eval_samples = len(decoded_preds)
            
            # Initialize counters
            correct_solutions = 0
            valid_paths = 0
            starts_correctly = 0
            ends_correctly = 0
            connected_paths = 0
            non_intersecting = 0
            no_rule_crossing = 0
            total_paths = max_eval_samples
            
            if total_paths == 0:
                if is_main_process:
                    print("DEBUG: No predictions to evaluate")
                return {
                    "eval_solution_accuracy": 0.0,
                    "eval_valid_path_rate": 0.0,
                    "eval_start_end_accuracy": 0.0,
                    "eval_connection_rate": 0.0,
                    "eval_non_intersection_rate": 0.0,
                    "eval_rule_compliance_rate": 0.0,
                    "eval_correct_solutions": 0,
                    "eval_valid_paths": 0,
                    "eval_total_evaluated": 0
                }
            
            if is_main_process:
                print(f"DEBUG: Evaluating {total_paths} samples")
            
            # Process each evaluation sample individually
            for i in range(total_paths):
                pred = decoded_preds[i]
                # Get the original dataset entry (puzzle)
                puzzle = eval_dataset[i]
                
                try:
                    # Extract the path from model response using SPaRC validation
                    extracted_path = extract_solution_path(pred, puzzle)
                    
                    if extracted_path is not None:
                        # Validate against ground truth
                        is_correct = validate_solution(extracted_path, puzzle)
                        if is_correct:
                            correct_solutions += 1
                        
                        # Get detailed analysis
                        analysis = analyze_path(extracted_path, puzzle)
                        
                        # Count individual validation criteria
                        if analysis.get("starts_at_start_ends_at_exit", False):
                            starts_correctly += 1
                            ends_correctly += 1
                        if analysis.get("connected_line", False):
                            connected_paths += 1
                        if analysis.get("non_intersecting_line", False):
                            non_intersecting += 1
                        if analysis.get("no_rule_crossing", False):
                            no_rule_crossing += 1
                        if analysis.get("fully_valid_path", False):
                            valid_paths += 1
                
                except Exception as e:
                    # Handle cases where path extraction fails
                    if is_main_process:
                        print(f"Error in metrics computation for sample {i}: {e}")
                    continue
            
            # Calculate metrics as percentages
            metrics = {
                "eval_solution_accuracy": correct_solutions / total_paths if total_paths > 0 else 0,
                "eval_valid_path_rate": valid_paths / total_paths if total_paths > 0 else 0,
                "eval_start_end_accuracy": starts_correctly / total_paths if total_paths > 0 else 0,
                "eval_connection_rate": connected_paths / total_paths if total_paths > 0 else 0,
                "eval_non_intersection_rate": non_intersecting / total_paths if total_paths > 0 else 0,
                "eval_rule_compliance_rate": no_rule_crossing / total_paths if total_paths > 0 else 0,
                "eval_correct_solutions": correct_solutions,
                "eval_valid_paths": valid_paths,
                "eval_total_evaluated": total_paths
            }
            
            # Debug: Print metrics (only main process)
            if is_main_process:
                print(f"DEBUG: Computed metrics: {metrics}")
            
            # Only log to wandb from main process
            if wandb.run and is_main_process:
                wandb.log(metrics)
                print(f"DEBUG: Metrics logged to wandb")
            
            # Final aggressive cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            return metrics
    
    return compute_metrics

# Create the compute_metrics function with access to eval dataset
compute_metrics_fn = create_compute_metrics(eval_dataset)

if is_main_process:  # Only print debug info on main process
    print(f"DEBUG: Eval dataset size: {len(eval_dataset)}")
    print(f"DEBUG: Train dataset size: {len(dataset)}")
    print(f"DEBUG: Model device: {model.device}")
    print(f"DEBUG: Wandb project: {wandb.run.project if wandb.run else 'No active run'}")

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=eval_dataset,
    formatting_func=formatting_prompts_func,
    compute_metrics=compute_metrics_fn,
    callbacks=[DebugEvaluationCallback()],
)

if is_main_process:
    print("DEBUG: Starting training...")
trainer.train()

# Clean up memory after training
if is_main_process:
    print("DEBUG: Training completed. Cleaning up memory...")
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    if is_main_process:
        print(f"DEBUG: CUDA memory allocated after training: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"DEBUG: CUDA memory reserved after training: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")