from datasets import load_dataset
from trl import SFTConfig, SFTTrainer, setup_chat_format
from sparc.prompt import generate_prompt
from sparc.validation import extract_solution_path, validate_solution, analyze_path
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
import numpy as np

model_name = "Qwen/Qwen3-0.6B"

# Initialize wandb
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


dataset = load_dataset("lkaesberg/SPaRC", "all", split="train")
eval_dataset = load_dataset("lkaesberg/SPaRC", "all", split="test")

training_args = SFTConfig(
    output_dir="./tmp",
    report_to="wandb",
    logging_steps=10,
    save_steps=500,
    eval_steps=500,
    warmup_steps=100,
    max_steps=1000,
    learning_rate=5e-5,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    max_seq_length=4096,
    remove_unused_columns=False,
    group_by_length=True,
    optim="adamw_torch_fused",  # Better performance than adamw_torch
    gradient_checkpointing=True,  # Reduce memory usage
    bf16=True,  # Use bfloat16 for better performance if supported
    save_strategy="steps",  # Save based on steps, not epochs
    evaluation_strategy="steps",  # Evaluate based on steps
    load_best_model_at_end=True,  # Load best model at end of training
    metric_for_best_model="solution_accuracy",  # Use your custom metric
    greater_is_better=True,  # Higher solution_accuracy is better
    save_total_limit=3,  # Keep only 3 best checkpoints
)

model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Set up the chat format with default 'chatml' format (modern approach)
model, tokenizer = setup_chat_format(model, tokenizer)

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
        predictions, labels = eval_pred
        
        # Decode predictions only (we'll get ground truth from dataset)
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        
        # Initialize counters
        correct_solutions = 0
        valid_paths = 0
        starts_correctly = 0
        ends_correctly = 0
        connected_paths = 0
        non_intersecting = 0
        no_rule_crossing = 0
        total_paths = len(decoded_preds)
        
        for i, pred in enumerate(decoded_preds):
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
                print(f"Error: {e}")
                continue
        
        # Calculate metrics as percentages
        return {
            "solution_accuracy": correct_solutions / total_paths if total_paths > 0 else 0,
            "valid_path_rate": valid_paths / total_paths if total_paths > 0 else 0,
            "start_end_accuracy": starts_correctly / total_paths if total_paths > 0 else 0,
            "connection_rate": connected_paths / total_paths if total_paths > 0 else 0,
            "non_intersection_rate": non_intersecting / total_paths if total_paths > 0 else 0,
            "rule_compliance_rate": no_rule_crossing / total_paths if total_paths > 0 else 0,
            "correct_solutions": correct_solutions,
            "valid_paths": valid_paths,
            "total_evaluated": total_paths
        }
    
    return compute_metrics

# Create the compute_metrics function with access to eval dataset
compute_metrics_fn = create_compute_metrics(eval_dataset)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=eval_dataset,
    formatting_func=formatting_prompts_func,
    compute_metrics=compute_metrics_fn,
)

trainer.train()