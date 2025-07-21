from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from sparc.prompt import generate_prompt
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl.trainer.sft_trainer import clone_chat_template
import json

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

training_args = SFTConfig(
    output_dir="./tmp",
    report_to="wandb",
    logging_steps=10,
    save_steps=500,
    eval_steps=500,
    warmup_steps=100,
    max_steps=10000,
    learning_rate=5e-5,
    max_seq_length=4096,
)

model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model, tokenizer = clone_chat_template(model, tokenizer, model_name)

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

trainer = SFTTrainer(
    model=model_name,
    args=training_args,
    train_dataset=dataset,
    formatting_func=formatting_prompts_func,
    processing_class=tokenizer
)

trainer.train()