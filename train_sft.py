from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from sparc.prompt import generate_prompt
import wandb

model = "Qwen/Qwen3-0.6B"

# Initialize wandb
wandb.init(
    project="sparc-sft",
    name=f"{model}-sparc-sft",
    config={
        "model": model,
        "dataset": "lkaesberg/SPaRC",
        "task": "supervised_fine_tuning"
    }
)


dataset = load_dataset("lkaesberg/SPaRC", "all", split="train")

def formatting_prompts_func(example):
    puzzle_prompt = [
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

    return puzzle_prompt

training_args = SFTConfig(
    output_dir="/tmp",
    report_to="wandb",
    logging_steps=10,
    save_steps=500,
    eval_steps=500,
    warmup_steps=100,
    max_steps=1000,
    learning_rate=5e-5,
)

trainer = SFTTrainer(
    model,
    args=training_args,
    train_dataset=dataset,
    formatting_func=formatting_prompts_func,
)

trainer.train()