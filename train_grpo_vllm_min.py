import argparse
from typing import List, Any, Dict

import wandb
from accelerate import PartialState
from datasets import load_dataset, Dataset
from trl import GRPOConfig, GRPOTrainer

from sparc.prompt import generate_prompt
from sparc.validation import extract_solution_path, validate_solution, analyze_path


def build_sparc_reward(original_examples: List[Dict[str, Any]]):
    prompt_to_puzzle: Dict[str, Dict[str, Any]] = {}
    for ex in original_examples:
        prompt_text = generate_prompt(ex)
        prompt_to_puzzle[prompt_text] = ex

    is_main = PartialState().is_main_process

    def reward_func(completions, prompts, **kwargs):
        rewards: List[float] = []

        for i, (completion, prompt) in enumerate(zip(completions, prompts)):
            # Normalize prompt/completion formats (chat vs str)
            if isinstance(prompt, list) and len(prompt) > 0 and isinstance(prompt[0], dict):
                prompt_text = prompt[0].get("content", str(prompt))
            else:
                prompt_text = str(prompt)

            if isinstance(completion, list) and len(completion) > 0 and isinstance(completion[0], dict):
                completion_text = completion[0].get("content", str(completion))
            else:
                completion_text = str(completion)

            puzzle = prompt_to_puzzle.get(prompt_text)
            reward_val = 0.0

            try:
                if puzzle is None:
                    # Fallback light format reward
                    reward_val = 0.2 if ("####" in completion_text and "(" in completion_text and ")" in completion_text) else 0.0
                else:
                    extracted_path = extract_solution_path(completion_text, puzzle)
                    if extracted_path is not None:
                        if validate_solution(extracted_path, puzzle):
                            reward_val = 1.0
                        else:
                            analysis = analyze_path(extracted_path, puzzle)
                            reward_val = 0.0
                            if analysis.get("starts_at_start_ends_at_exit", False):
                                reward_val += 0.25
                            if analysis.get("connected_line", False):
                                reward_val += 0.25
                            if analysis.get("non_intersecting_line", False):
                                reward_val += 0.25
                            if analysis.get("no_rule_crossing", False):
                                reward_val += 0.25
                    else:
                        reward_val = 0.1 if ("####" in completion_text and "(" in completion_text and ")" in completion_text) else 0.0
            except Exception:
                reward_val = 0.0

            rewards.append(float(reward_val))

        # Log per-sample rewards minimally on main process
        if is_main and len(rewards) > 0 and wandb.run is not None:
            table = wandb.Table(columns=["idx", "reward"], data=[[i, r] for i, r in enumerate(rewards)])
            wandb.log({
                "rewards/single_hist": wandb.Histogram(rewards),
                "rewards/single_table": table,
                "rewards/mean": sum(rewards) / len(rewards),
            })

        return rewards

    return reward_func


def to_grpo_prompt_format(dataset: Dataset) -> Dataset:
    def _map_fn(ex):
        prompt = generate_prompt(ex)
        return {
            "prompt": [{"role": "user", "content": prompt}],
            "puzzle_data": ex,
        }

    return dataset.map(_map_fn)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--vllm_server_host", type=str, required=True, help="Hostname or IP of the vLLM server (port 8000)")
    parser.add_argument("--wandb_project", type=str, default="sparc-grpo")
    parser.add_argument("--wandb_entity", type=str, default=None)
    args = parser.parse_args()

    state = PartialState()
    is_main = state.is_main_process

    if is_main:
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, config={
            "model": args.model,
            "dataset": "lkaesberg/SPaRC",
            "trainer": "GRPO",
            "use_vllm": True,
            "vllm_mode": "server",
            "vllm_server_host": args.vllm_server_host,
        })

    # Load datasets
    train_raw = load_dataset("lkaesberg/SPaRC", "all", split="train")
    eval_raw = load_dataset("lkaesberg/SPaRC", "all", split="test")

    # Transform to prompt format
    train_ds = to_grpo_prompt_format(train_raw)
    eval_ds = to_grpo_prompt_format(eval_raw)

    # Build reward function with access to original examples
    combined_examples: List[Dict[str, Any]] = list(train_raw) + list(eval_raw)
    reward = build_sparc_reward(combined_examples)

    # Minimal GRPO config per docs; vLLM in server mode
    config = GRPOConfig(
        output_dir="./grpo_outputs",
        report_to="wandb",
        per_device_train_batch_size=4,
        bf16=True,
        logging_steps=10,
        save_strategy="no",
        use_vllm=True,
        vllm_mode="server",
        vllm_server_host=args.vllm_server_host,
        gradient_checkpointing=True,
        max_completion_length=30000, 
        max_prompt_length=5000,
    )

    trainer = GRPOTrainer(
        model=args.model,
        args=config,
        reward_funcs=reward,
        train_dataset=train_ds,
        eval_dataset=None,
    )

    trainer.train()

    if is_main:
        trainer.save_model("./final_grpo_model")
        wandb.finish()


if __name__ == "__main__":
    main()


