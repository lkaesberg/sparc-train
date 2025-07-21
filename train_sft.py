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
    max_steps=1000,
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    dataloader_drop_last=True,
    max_seq_length=4096,
    packing=False,
)

model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model, tokenizer = clone_chat_template(model, tokenizer, model_name)

def formatting_prompts_func(example):
    # Debug: Print the type and structure of the example
    print(f"Example type: {type(example)}")
    print(f"Example keys: {list(example.keys()) if isinstance(example, dict) else 'Not a dict'}")
    
    # Check if it's a batched format (lists) or individual format
    if isinstance(example, dict) and len(example) > 0:
        first_key = list(example.keys())[0]
        first_value = example[first_key]
        print(f"First value type: {type(first_value)}")
        if isinstance(first_value, list):
            print(f"First value length: {len(first_value)}")
            print("This appears to be BATCHED data")
            
            # Handle batched data - return list of formatted texts
            formatted_texts = []
            batch_size = len(first_value)
            
            for i in range(batch_size):
                # Extract individual example from batch
                individual_example = {key: values[i] for key, values in example.items()}
                
                puzzle_prompt = {
                    "messages": [
                              {
                                "role": "system",
                                "content": "You are an expert at solving puzzles games.",
                              },
                              {
                                "role": "user", 
                                "content": generate_prompt(individual_example)
                              },
                            {
                                "role": "assistant",
                                "content": f"#### ({', '.join(map(lambda x: f'({x["x"]}, {x["y"]})', individual_example['solutions'][0]['path']))})"
                            }
                            ]
                }
                
                formatted_text = tokenizer.apply_chat_template(puzzle_prompt, tokenize=False)
                formatted_texts.append(formatted_text)
            
            print(f"Returning {len(formatted_texts)} formatted texts")
            return formatted_texts
        else:
            print("This appears to be INDIVIDUAL data")
    
    # Handle individual example (not batched)
    puzzle_prompt = {
        "messages": [
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
    }
    
    formatted_text = tokenizer.apply_chat_template(puzzle_prompt, tokenize=False)
    print(f"Formatted text preview: {formatted_text[:100]}...")
    return formatted_text

trainer = SFTTrainer(
    model=model_name,
    args=training_args,
    train_dataset=dataset,
    formatting_func=formatting_prompts_func,
    processing_class=tokenizer
)

trainer.train()