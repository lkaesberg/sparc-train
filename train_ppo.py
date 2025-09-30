import argparse
import os
from typing import List, Any, Dict
import torch
import torch.distributed as dist

import wandb
from accelerate import PartialState
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import PPOConfig, PPOTrainer

from sparc.prompt import generate_prompt
from sparc.validation import extract_solution_path, validate_solution, analyze_path


def build_sparc_reward_functions(original_examples: List[Dict[str, Any]]):
    """Builds a list of reward functions, one per reward component.

    Returns a list[callable], each taking (completions, prompts, **kwargs) -> list[float].
    """
    prompt_to_puzzle: Dict[str, Dict[str, Any]] = {}
    for example in original_examples:
        prompt_text = generate_prompt(example)
        prompt_to_puzzle[prompt_text] = example

    is_main = PartialState().is_main_process

    def _normalize_texts(completion, prompt):
        if isinstance(prompt, list) and len(prompt) > 0 and isinstance(prompt[0], dict):
            # Prefer the first user message content if present; fallback to first message content
            try:
                prompt_text_local = next(
                    (m.get("content", "") for m in prompt if isinstance(m, dict) and m.get("role") == "user"),
                    prompt[0].get("content", str(prompt)),
                )
            except Exception:
                prompt_text_local = prompt[0].get("content", str(prompt))
        else:
            prompt_text_local = str(prompt)

        if isinstance(completion, list) and len(completion) > 0 and isinstance(completion[0], dict):
            completion_text_local = completion[0].get("content", str(completion))
        else:
            completion_text_local = str(completion)

        return prompt_text_local, completion_text_local

    # 1) Perfect solution reward (1.0 if fully correct, else 0.0)
    def reward_perfect_solution(completions, prompts, **kwargs):
        rewards_local: List[float] = []
        for completion, prompt in zip(completions, prompts):
            prompt_text, completion_text = _normalize_texts(completion, prompt)
            puzzle = prompt_to_puzzle.get(prompt_text)
            if puzzle is None:
                rewards_local.append(0.0)
                continue
            try:
                extracted_path = extract_solution_path(completion_text, puzzle)
                if extracted_path is not None and validate_solution(extracted_path, puzzle):
                    rewards_local.append(1.0)
                else:
                    rewards_local.append(0.0)
            except Exception:
                rewards_local.append(0.0)
        if is_main and len(rewards_local) > 0 and wandb.run is not None:
            wandb.log({"rewards/perfect_mean": sum(rewards_local) / len(rewards_local)})
        return rewards_local

    # Helper to compute analysis dict safely
    def _extract_path(completion_text: str, puzzle: Dict[str, Any]):
        try:
            return extract_solution_path(completion_text, puzzle)
        except Exception:
            return None

    def _is_perfect(completion_text: str, puzzle: Dict[str, Any]) -> bool:
        try:
            extracted_path = extract_solution_path(completion_text, puzzle)
            return bool(extracted_path is not None and validate_solution(extracted_path, puzzle))
        except Exception:
            return False

    def _safe_analysis(completion_text: str, puzzle: Dict[str, Any]):
        try:
            extracted_path = extract_solution_path(completion_text, puzzle)
            if extracted_path is None:
                return None
            return analyze_path(extracted_path, puzzle)
        except Exception:
            return None

    # 2) Starts at start and ends at exit (0.25)
    def reward_starts_and_ends(completions, prompts, **kwargs):
        rewards_local: List[float] = []
        for completion, prompt in zip(completions, prompts):
            prompt_text, completion_text = _normalize_texts(completion, prompt)
            puzzle = prompt_to_puzzle.get(prompt_text)
            if puzzle is None:
                rewards_local.append(0.0)
                continue
            if _is_perfect(completion_text, puzzle):
                rewards_local.append(0.0)
            else:
                analysis = _safe_analysis(completion_text, puzzle)
                rewards_local.append(1.0 if analysis and analysis.get("starts_at_start_ends_at_exit", False) else 0.0)
        return rewards_local

    # 3) Connected line (0.25)
    def reward_connected_line(completions, prompts, **kwargs):
        rewards_local: List[float] = []
        for completion, prompt in zip(completions, prompts):
            prompt_text, completion_text = _normalize_texts(completion, prompt)
            puzzle = prompt_to_puzzle.get(prompt_text)
            if puzzle is None:
                rewards_local.append(0.0)
                continue
            if _is_perfect(completion_text, puzzle):
                rewards_local.append(0.0)
            else:
                analysis = _safe_analysis(completion_text, puzzle)
                rewards_local.append(1.0 if analysis and analysis.get("connected_line", False) else 0.0)
        return rewards_local

    # 4) Non-intersecting line (0.25)
    def reward_non_intersecting(completions, prompts, **kwargs):
        rewards_local: List[float] = []
        for completion, prompt in zip(completions, prompts):
            prompt_text, completion_text = _normalize_texts(completion, prompt)
            puzzle = prompt_to_puzzle.get(prompt_text)
            if puzzle is None:
                rewards_local.append(0.0)
                continue
            if _is_perfect(completion_text, puzzle):
                rewards_local.append(0.0)
            else:
                analysis = _safe_analysis(completion_text, puzzle)
                rewards_local.append(1.0 if analysis and analysis.get("non_intersecting_line", False) else 0.0)
        return rewards_local

    # 5) No rule crossing (0.25)
    def reward_no_rule_crossing(completions, prompts, **kwargs):
        rewards_local: List[float] = []
        for completion, prompt in zip(completions, prompts):
            prompt_text, completion_text = _normalize_texts(completion, prompt)
            puzzle = prompt_to_puzzle.get(prompt_text)
            if puzzle is None:
                rewards_local.append(0.0)
                continue
            if _is_perfect(completion_text, puzzle):
                rewards_local.append(0.0)
            else:
                analysis = _safe_analysis(completion_text, puzzle)
                rewards_local.append(1.0 if analysis and analysis.get("no_rule_crossing", False) else 0.0)
        return rewards_local

    # 6) Format hint reward (small reward for emitting expected format when path not valid)
    def reward_format_hint(completions, prompts, **kwargs):
        rewards_local: List[float] = []
        for completion, prompt in zip(completions, prompts):
            prompt_text, completion_text = _normalize_texts(completion, prompt)
            puzzle = prompt_to_puzzle.get(prompt_text)
            if puzzle is not None and _is_perfect(completion_text, puzzle):
                rewards_local.append(0.0)
                continue
            has_format = ("####" in completion_text and "(" in completion_text and ")" in completion_text)
            rewards_local.append(1.0 if has_format else 0.0)
        return rewards_local

    # Return all reward functions as a list
    return [
        reward_perfect_solution,
        reward_starts_and_ends,
        reward_connected_line,
        reward_non_intersecting,
        reward_no_rule_crossing,
        reward_format_hint,
    ]


def to_ppo_prompt_format(dataset: Dataset) -> Dataset:
    def _map_fn(ex):
        prompt = generate_prompt(ex)
        system_msg = (
            "Use <think>...</think> to reason privately. Keep thinking concise. "
            "After thinking, output ONLY the final SPaRC path in the exact format '####(x0,y0)->(x1,y1)->...'. "
            "Do not include any extra text outside the final path line."
        )
        return {
            "prompt": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt},
            ],
            "puzzle_data": ex,
        }

    return dataset.map(_map_fn)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--wandb_project", type=str, default="sparc-ppo")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run_id", type=str, default=None, help="Optional W&B run id to resume")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--mini_batch_size", type=int, default=4)
    parser.add_argument("--ppo_epochs", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    args = parser.parse_args()

    state = PartialState()
    is_main = state.is_main_process

    if is_main:
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, id=args.wandb_run_id, resume="allow" if args.wandb_run_id else None, config={
            "model": args.model,
            "dataset": "lkaesberg/SPaRC",
            "trainer": "PPO"},
            settings=wandb.Settings(init_timeout=3600)
        )

    # Load datasets
    train_raw = load_dataset("lkaesberg/SPaRC", "all", split="train")
    eval_raw = load_dataset("lkaesberg/SPaRC", "all", split="test[:100]")

    # Transform to prompt format
    train_ds = to_ppo_prompt_format(train_raw)
    eval_ds = to_ppo_prompt_format(eval_raw)

    # Build reward functions with access to original examples
    combined_examples: List[Dict[str, Any]] = list(train_raw) + list(eval_raw)
    rewards = build_sparc_reward_functions(combined_examples)

    # PPO setup
    config = PPOConfig(
        model_name=args.model,
        learning_rate=args.learning_rate,
        log_with="wandb",
        batch_size=args.batch_size,
        mini_batch_size=args.mini_batch_size,
        ppo_epochs=args.ppo_epochs,
        target_kl=0.1,
        seed=42,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(args.model)

    ppo_trainer = PPOTrainer(
        config=config,
        model=model,
        tokenizer=tokenizer,
        dataset=train_ds,
    )

    # Reward aggregation weights (aligned with GRPO setup)
    reward_weights = [1.0, 0.25, 0.25, 0.25, 0.25, 0.1]

    def _aggregate_scalar_rewards(completions: List[str], prompts: List[Any]) -> List[float]:
        # Compute each component, then weighted sum
        component_funcs = rewards
        component_values = [func(completions, prompts) for func in component_funcs]
        scalars: List[float] = []
        for i in range(len(completions)):
            total = 0.0
            for w, vals in zip(reward_weights, component_values):
                total += w * float(vals[i])
            scalars.append(total)
        if is_main and len(scalars) > 0 and wandb.run is not None:
            wandb.log({
                "rewards/scalar_mean": sum(scalars) / len(scalars)
            })
        return scalars

    # Training loop using PPO
    max_new_tokens = int(args.max_new_tokens)
    for epoch in range(args.ppo_epochs):
        for batch in ppo_trainer.dataloader:
            prompts_messages: List[Any] = batch["prompt"]

            # Build chat-formatted input strings
            prompts_texts: List[str] = []
            for msgs in prompts_messages:
                if hasattr(tokenizer, "apply_chat_template") and isinstance(msgs, list) and (len(msgs) == 0 or isinstance(msgs[0], dict)):
                    try:
                        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                    except Exception:
                        # Fallback: concatenate message contents
                        if isinstance(msgs, list):
                            text = "\n".join(str(m.get("content", m)) if isinstance(m, dict) else str(m) for m in msgs)
                        else:
                            text = str(msgs)
                else:
                    if isinstance(msgs, list):
                        text = "\n".join(str(m.get("content", m)) if isinstance(m, dict) else str(m) for m in msgs)
                    else:
                        text = str(msgs)
                prompts_texts.append(text)

            # Tokenize inputs
            query_toks = tokenizer(prompts_texts, return_tensors="pt", padding=True, truncation=True)
            query_input_ids = query_toks.input_ids.to(ppo_trainer.accelerator.device)
            query_attention_mask = query_toks.attention_mask.to(ppo_trainer.accelerator.device)

            # Generate responses
            gen_out = ppo_trainer.generate(
                query_input_ids,
                attention_mask=query_attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.7,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            # Split out just the generated portion per sample
            input_lengths = (query_attention_mask.sum(dim=1)).tolist()
            responses_decoded: List[str] = []
            response_tensors: List[Any] = []
            for i in range(gen_out.shape[0]):
                start = int(input_lengths[i])
                resp_ids = gen_out[i][start:]
                response_tensors.append(resp_ids)
                responses_decoded.append(tokenizer.decode(resp_ids, skip_special_tokens=True))

            # Compute scalar rewards
            rewards_scalar = _aggregate_scalar_rewards(responses_decoded, prompts_messages)

            # Convert scalar rewards to per-token reward tensors
            rewards_tokenwise = []
            for i, resp in enumerate(response_tensors):
                if hasattr(resp, "shape"):
                    length = int(resp.shape[0])
                else:
                    length = len(resp)
                r = float(rewards_scalar[i])
                rewards_tokenwise.append(torch.full((length,), r, device=ppo_trainer.accelerator.device))

            # Run PPO step
            ppo_trainer.step(list(query_input_ids), response_tensors, rewards_tokenwise)

        if is_main:
            wandb.log({"epoch": epoch})

    # Save model
    save_dir = f"./models/{args.model.split('/')[-1]}-SPaRC-PPO"
    os.makedirs(save_dir, exist_ok=True)
    ppo_trainer.model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    if is_main:
        wandb.finish()


if __name__ == "__main__":
    main()


