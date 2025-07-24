import torch
import gc
import numpy as np
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer, setup_chat_format
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
    print(f"DEBUG: Original train dataset size: {len(dataset)}")
    print(f"DEBUG: Original eval dataset size: {len(eval_dataset)}")

# dataset = expand_dataset_with_individual_solutions(dataset)

if is_main_process:
    print(f"DEBUG: Expanded train dataset size: {len(dataset)}")



training_args = SFTConfig(
    output_dir="./tmp",
    report_to="wandb",
    logging_steps=10,
    save_steps=None,  # Disable intermediate checkpoint saving
    eval_steps=100,  # Evaluate less frequently to save memory
    warmup_steps=100,
    max_steps=10000,
    learning_rate=1e-6,  # DRASTICALLY reduced from 5e-5 to prevent overfitting
    per_device_train_batch_size=4,
    per_device_eval_batch_size=1,  # CRITICAL: Set to 1 to prevent sequence packing
    eval_accumulation_steps=1,  # Process eval in smallest possible steps
    gradient_accumulation_steps=2,
    max_seq_length=4096,  # Reduce sequence length to save memory
    remove_unused_columns=False,
    group_by_length=False,  # IMPORTANT: Disable to prevent length-based batching
    optim="adamw_torch_fused",  # Better performance than adamw_torch
    gradient_checkpointing=True,  # Reduce memory usage
    bf16=True,  # Use bfloat16 for better performance if supported
    save_strategy="no",  # Disable checkpoint saving during training
    eval_strategy="steps",  # Evaluate based on steps
    do_eval=True,  # Explicitly enable evaluation
    load_best_model_at_end=False,  # Disable to save memory
    metric_for_best_model="eval_solution_accuracy",  # Use your custom metric with eval_ prefix
    greater_is_better=True,  # Higher solution_accuracy is better
    save_total_limit=1,  # Keep only final model when we save manually
    ddp_find_unused_parameters=False,  # Optimize for multi-GPU
    logging_dir="./logs",  # Add logging directory
    logging_first_step=True,  # Log the first step
    dataloader_pin_memory=False,  # Disable pin memory to avoid potential issues
    fp16_full_eval=False,  # Don't use fp16 during eval to avoid precision issues
    dataloader_num_workers=0,  # Disable multiprocessing to save memory
    prediction_loss_only=False,  # We need predictions for custom metrics
    
    # CRITICAL: Enable padding-free batching for massive memory savings
    padding_free=True,  # Eliminates padding completely, huge memory reduction
    
    # REGULARIZATION: Add these to prevent overfitting
    weight_decay=0.01,  # L2 regularization
    warmup_ratio=0.1,  # Longer warmup to stabilize training
    lr_scheduler_type="cosine",  # Cosine decay instead of linear
    dataloader_drop_last=True,  # Ensure consistent batch sizes
    
    # Additional memory optimizations
    eval_on_start=True,  # Don't evaluate at start
    include_inputs_for_metrics=False,  # Don't include inputs in metrics computation
    eval_do_concat_batches=False,
    packing=False,
    eval_packing=False,
    assistant_only_loss=True
)

# Multi-GPU device setup
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map={'': device_string},  # Proper device placement for multi-GPU
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,  # Fix Flash Attention warning by specifying dtype
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.chat_template = "{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0].role == 'system' %}\n        {{- messages[0].content + '\\n\\n' }}\n    {%- endif %}\n    {{- \"# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n    {%- for tool in tools %}\n        {{- \"\\n\" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n{%- else %}\n    {%- if messages[0].role == 'system' %}\n        {{- '<|im_start|>system\\n' + messages[0].content + '<|im_end|>\\n' }}\n    {%- endif %}\n{%- endif %}\n{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}\n{%- for message in messages[::-1] %}\n    {%- set index = (messages|length - 1) - loop.index0 %}\n    {%- if ns.multi_step_tool and message.role == \"user\" and message.content is string and not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>')) %}\n        {%- set ns.multi_step_tool = false %}\n        {%- set ns.last_query_index = index %}\n    {%- endif %}\n{%- endfor %}\n{%- for message in messages %}\n    {%- if message.content is string %}\n        {%- set content = message.content %}\n    {%- else %}\n        {%- set content = '' %}\n    {%- endif %}\n    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) %}\n        {{- '<|im_start|>' + message.role + '\\n' + content + '<|im_end|>' + '\\n' }}\n    {%- elif message.role == \"assistant\" %}\n        {%- set reasoning_content = '' %}\n        {%- if message.reasoning_content is string %}\n            {%- set reasoning_content = message.reasoning_content %}\n        {%- else %}\n            {%- if '</think>' in content %}\n                {%- set reasoning_content = content.split('</think>')[0].rstrip('\\n').split('<think>')[-1].lstrip('\\n') %}\n                {%- set content = content.split('</think>')[-1].lstrip('\\n') %}\n            {%- endif %}\n        {%- endif %}\n\n        {{- '<|im_start|>' + message.role }}\n        {% generation %}\n        {%- if loop.index0 > ns.last_query_index %}\n            {%- if loop.last or (not loop.last and reasoning_content) %}\n                {{- '<think>\\n' + reasoning_content.strip('\\n') + '\\n</think>\\n\\n' + content.lstrip('\\n') }}\n            {%- else %}\n                {{- content }}\n            {%- endif %}\n        {%- else %}\n            {{- content }}\n        {%- endif %}\n        {%- if message.tool_calls %}\n            {%- for tool_call in message.tool_calls %}\n                {%- if (loop.first and content) or (not loop.first) %}\n                    {{- '\\n' }}\n                {%- endif %}\n                {%- if tool_call.function %}\n                    {%- set tool_call = tool_call.function %}\n                {%- endif %}\n                {{- '<tool_call>\\n{\"name\": \"' }}\n                {{- tool_call.name }}\n                {{- '\", \"arguments\": ' }}\n                {%- if tool_call.arguments is string %}\n                    {{- tool_call.arguments }}\n                {%- else %}\n                    {{- tool_call.arguments | tojson }}\n                {%- endif %}\n                {{- '}\\n</tool_call>' }}\n            {%- endfor %}\n        {%- endif %}\n        {{- '<|im_end|>' }}\n        {% endgeneration %}\n    {%- elif message.role == \"tool\" %}\n        {%- if loop.first or (messages[loop.index0 - 1].role != \"tool\") %}\n            {{- '<|im_start|>user' }}\n        {%- endif %}\n        {{- '\\n<tool_response>\\n' }}\n        {{- content }}\n        {{- '\\n</tool_response>' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n            {{- '<|im_end|>\\n' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n    {%- if enable_thinking is defined and enable_thinking is false %}\n        {{- '<think>\\n\\n</think>\\n\\n' }}\n    {%- endif %}\n{%- endif %}"

def transform_to_conversational_format(dataset):
    """
    Transform dataset to conversational format required for assistant_only_loss=True.
    
    TRL expects datasets with 'messages' field containing:
    [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."}
    ]
    """
    def format_sample(example):
        # Create the solution path string without nested f-strings
        path_coords = ', '.join([f"({point['x']}, {point['y']})" for point in example['solutions'][0]['path']])
        solution_text = f"#### ({path_coords})"
        
        messages = [
            {
                "role": "system",
                "content": "You are an expert at solving puzzles games."
            },
            {
                "role": "user", 
                "content": generate_prompt(example)
            },
            {
                "role": "assistant",
                "content": solution_text
            }
        ]
        return {"messages": messages}
    
    return dataset.map(format_sample, remove_columns=dataset.column_names)

# Transform datasets to conversational format
dataset = transform_to_conversational_format(dataset)

# Create conversational eval dataset for SFTTrainer (needs assistant_only_loss=True)
eval_dataset_conversational = transform_to_conversational_format(eval_dataset)
# Keep original eval_dataset for metrics computation (unchanged)

if is_main_process:
    print(f"DEBUG: Transformed train dataset to conversational format")
    print(f"DEBUG: Sample conversation: {dataset[0]['messages']}")
    print(f"DEBUG: Created conversational eval dataset for SFTTrainer")
    print(f"DEBUG: Kept original eval_dataset for metrics computation")
    print(f"DEBUG: Original eval entry fields: {list(eval_dataset[0].keys())}")

def preprocess_logits_for_metrics(logits, labels):
    """
    Preprocess logits to reduce memory usage during evaluation.
    This function converts logits to token IDs immediately, reducing memory footprint.
    """
    with torch.no_grad():
        # Detach from computational graph immediately
        if hasattr(logits, 'detach'):
            logits = logits.detach()
        
        # Convert logits to predicted token IDs (argmax along vocab dimension)
        # This dramatically reduces size: [batch, seq_len, vocab_size] -> [batch, seq_len]
        predicted_ids = logits.argmax(dim=-1)
        
        # IMPORTANT: Keep on GPU for multi-GPU gather operation
        # Don't move to CPU here - let Accelerate handle the gather first
        # predicted_ids stays on GPU for now
        
        # Clear the original logits from GPU memory
        del logits
        
        # Light memory cleanup (but keep predicted_ids on GPU)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return predicted_ids

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
            
            # Now predictions are already processed token IDs (not logits!)
            # After multi-GPU gather, they may still be on GPU
            predicted_ids, labels = eval_pred
            
            if is_main_process:
                print(f"DEBUG: predicted_ids shape: {predicted_ids.shape if hasattr(predicted_ids, 'shape') else 'no shape'}")
                print(f"DEBUG: predicted_ids type: {type(predicted_ids)}")
                print(f"DEBUG: predicted_ids device: {predicted_ids.device if hasattr(predicted_ids, 'device') else 'no device'}")
            
            # Handle different formats after multi-GPU gather
            if isinstance(predicted_ids, list):
                # Multi-GPU gather sometimes returns nested lists
                if is_main_process:
                    print(f"DEBUG: Converting list format, length: {len(predicted_ids)}")
                
                # Handle variable-length sequences - don't convert to single array
                # Instead, process each sequence individually
                flat_sequences = []
                for item in predicted_ids:
                    if isinstance(item, np.ndarray):
                        # Each item is a numpy array of shape (num_gpus, seq_len)
                        # Extract each GPU's sequences
                        for gpu_sequences in item:
                            flat_sequences.append(gpu_sequences)
                    elif isinstance(item, (list, tuple)):
                        # Handle nested lists/tuples
                        for subitem in item:
                            if hasattr(subitem, 'flatten'):
                                flat_sequences.append(subitem.flatten())
                            else:
                                flat_sequences.append(subitem)
                    else:
                        flat_sequences.append(item)
                
                # Now flat_sequences contains individual sequence arrays
                predicted_ids = flat_sequences
                if is_main_process:
                    print(f"DEBUG: Flattened to {len(flat_sequences)} individual sequences")
                    print(f"DEBUG: Sequence lengths: {[len(seq) if hasattr(seq, '__len__') else 'unknown' for seq in flat_sequences[:5]]}...")
                
            else:
                # Handle tensor format
                if hasattr(predicted_ids, 'cpu'):
                    predicted_ids = predicted_ids.cpu().numpy()
                elif hasattr(predicted_ids, 'numpy'):
                    predicted_ids = predicted_ids.numpy()
                
                # Convert tensor to list of sequences
                if len(predicted_ids.shape) == 3:
                    # Multi-GPU gather created extra dimension: [num_samples, num_gpus, seq_len]
                    # Flatten to list of individual sequences
                    flat_sequences = []
                    for sample in predicted_ids:
                        for gpu_seq in sample:
                            flat_sequences.append(gpu_seq)
                    predicted_ids = flat_sequences
                    if is_main_process:
                        print(f"DEBUG: Converted 3D tensor to {len(flat_sequences)} individual sequences")
                elif len(predicted_ids.shape) == 2:
                    # Convert 2D array to list of sequences
                    predicted_ids = [predicted_ids[i] for i in range(predicted_ids.shape[0])]
            
            # Additional memory cleanup after processing
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Process ALL predictions instead of limiting to avoid incomplete evaluation
            decoded_preds = []
            num_predictions = len(predicted_ids)
            
            if is_main_process:
                print(f"DEBUG: Processing ALL {num_predictions} predictions for comprehensive evaluation")
            
            for i in range(num_predictions):
                try:
                    # Get single prediction (already token IDs)
                    # predicted_ids is now a list of individual sequences
                    single_pred_ids = predicted_ids[i]
                    
                    # Ensure it's a 1D array of integers
                    if hasattr(single_pred_ids, 'flatten'):
                        single_pred_ids = single_pred_ids.flatten()
                    
                    # Convert to Python list of integers for tokenizer
                    if hasattr(single_pred_ids, 'tolist'):
                        single_pred_ids = single_pred_ids.tolist()
                    elif not isinstance(single_pred_ids, list):
                        single_pred_ids = list(single_pred_ids)
                    
                    # Ensure all elements are integers
                    single_pred_ids = [int(x) for x in single_pred_ids if isinstance(x, (int, float, np.integer, np.floating))]
                    
                    if is_main_process and i == 0:  # Debug first prediction
                        print(f"DEBUG: First sequence length: {len(single_pred_ids)}")
                        print(f"DEBUG: First few tokens: {single_pred_ids[:10]}")
                    
                    # Decode this single sequence
                    try:
                        decoded = tokenizer.decode(single_pred_ids, skip_special_tokens=True)
                        decoded_preds.append(decoded)
                        
                        if is_main_process and i == 0:  # Debug first decoded prediction
                            print(f"DEBUG: First decoded text (100 chars): {decoded[:100]}...")
                            
                    except Exception as decode_e:
                        if is_main_process:
                            print(f"DEBUG: Error decoding prediction {i}: {decode_e}")
                            print(f"DEBUG: Token sequence type: {type(single_pred_ids)}, length: {len(single_pred_ids) if hasattr(single_pred_ids, '__len__') else 'no length'}")
                        decoded_preds.append("")
                    
                    # Light cleanup after each prediction
                    if i % 10 == 0:  # Every 10 predictions
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                
                except Exception as e:
                    if is_main_process:
                        print(f"DEBUG: Error processing prediction {i}: {e}")
                    decoded_preds.append("")
            
            # Debug: Print some examples to see what we're getting (only main process)
            if is_main_process:
                print(f"\nDEBUG: Number of predictions decoded: {len(decoded_preds)}")
                print(f"DEBUG: Eval dataset size: {len(eval_dataset)}")
                if len(decoded_preds) > 0:
                    print(f"DEBUG: First prediction length: {len(decoded_preds[0])} chars")
                    print(f"DEBUG: First prediction: {decoded_preds[0][:400]}...")
                    
                    # Check if sequences are packed (contain multiple samples)
                    # Look for multiple system/user message pairs which indicate packing
                    sample_markers = decoded_preds[0].count("<|im_start|>system")
                    if sample_markers > 1:
                        print(f"WARNING: Detected {sample_markers} system messages in first prediction!")
                        print(f"WARNING: Sequences appear to be PACKED despite packing=False!")
                        print(f"WARNING: This will cause incorrect puzzle-to-path matching!")
            
            # Handle the packing issue: split packed sequences if detected
            if len(decoded_preds) < len(eval_dataset):
                if is_main_process:
                    print(f"WARNING: Mismatch - {len(decoded_preds)} predictions vs {len(eval_dataset)} eval samples")
                    print(f"WARNING: Attempting to split packed sequences...")
                
                # Try to split packed sequences
                split_predictions = []
                for pred in decoded_preds:
                    # Split on system message markers
                    if "<|im_start|>system" in pred:
                        parts = pred.split("<|im_start|>system")
                        # First part might be empty, skip it
                        for i, part in enumerate(parts):
                            if part.strip():
                                # Reconstruct the system message
                                if i > 0:  # Add back the system start token
                                    reconstructed = "<|im_start|>system" + part
                                else:
                                    reconstructed = part
                                split_predictions.append(reconstructed)
                    else:
                        split_predictions.append(pred)
                
                if is_main_process:
                    print(f"DEBUG: Split {len(decoded_preds)} packed predictions into {len(split_predictions)} individual predictions")
                
                # Use split predictions if we got a better match
                if len(split_predictions) > len(decoded_preds):
                    decoded_preds = split_predictions
            
            # Ensure we don't exceed eval dataset size
            max_eval_samples = min(len(decoded_preds), len(eval_dataset))
            
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
                print(f"DEBUG: Evaluating {total_paths} samples with corrected puzzle matching")
            
            # Process each evaluation sample individually
            for i in range(total_paths):
                pred = decoded_preds[i]
                # Get the original dataset entry (puzzle)
                puzzle = eval_dataset[i]
                
                # Removed detailed per-sample debug output for cleaner logs
                
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
    args=training_args,
    train_dataset=dataset,
    eval_dataset=eval_dataset_conversational, # Pass the conversational eval dataset
    compute_metrics=compute_metrics_fn,
    processing_class=tokenizer,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
)

if is_main_process:
    print("DEBUG: Starting training...")
trainer.train()

# Save only the final model (not FSDP checkpoints)
if is_main_process:
    print("DEBUG: Training completed. Saving final model...")
    
    # Save the final model in standard format
    final_model_dir = "./final_model"
    trainer.save_model(final_model_dir)
    
    # Also save the tokenizer
    tokenizer.save_pretrained(final_model_dir)
    
    print(f"DEBUG: Final model saved to {final_model_dir}")

# Clean up memory after training
if is_main_process:
    print("DEBUG: Cleaning up memory...")
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    if is_main_process:
        print(f"DEBUG: CUDA memory allocated after training: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"DEBUG: CUDA memory reserved after training: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")