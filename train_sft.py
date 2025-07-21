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
)

model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model, tokenizer = clone_chat_template(model, tokenizer, model_name)

def formatting_prompts_func(example):
    example["polyshape"] = json.dumps(example["polyshape"])
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

    return tokenizer.apply_chat_template(puzzle_prompt, tokenize=False)

trainer = SFTTrainer(
    model=model_name,
    args=training_args,
    train_dataset=dataset,
    formatting_func=formatting_prompts_func,
    processing_class=tokenizer
)

trainer.train()