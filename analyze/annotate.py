#!/usr/bin/env python3
"""
Annotate SPaRC JSONL results using a local vLLM OpenAI-compatible server.

Usage:
  python analyze/annotate.py --input results/sparc/<file>.jsonl \
      --output analyze/results/annotate/<file>.annotated.jsonl \
      --categories categories.txt \
      --model lkaesberg/Qwen3-14B-SPaRC-GRPO-8E \
      --port 8000

The script expects the local vLLM server to be reachable at
http://127.0.0.1:<port>/v1 and uses the OpenAI-compatible /v1/chat/completions
or /v1/completions endpoint depending on model (it talks to the standard
OpenAI-compatible inference API).

Input JSONL: each line is a JSON object representing a puzzle/sample. The
entire object will be forwarded to the LLM judge in the prompt. The output
JSONL will contain the original sample plus a new key `llm_annotation` with
an array of assigned categories (0..8 items) and optional `llm_raw` text.

This script is intentionally minimal and synchronous; it is designed for
moderate-size batches. For large-scale annotation, batching and rate limiting
should be added.
"""

import argparse
import json
import os
import re
import sys
from typing import List, Optional

from openai import OpenAI

from sparc.prompt import generate_prompt


def load_categories(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


# Default categories: if the user does not pass a --categories file, these
# will be used. Customize to match your desired 8 categories.
DEFAULT_CATEGORIES = [
    "Solved correctly (exact match)",
    "Solved but minor path deviation",
    "Incorrect path (valid shape but wrong exit)",
    "Crossing or rule violation",
    "Partial solution (incomplete)",
    "No attempt / empty reasoning",
    "Nonsensical / hallucinated path",
    "Other / ambiguous"
]


def get_field(sample: dict, field_path: str):
    # support simple dotted path like 'result.message'
    cur = sample
    for part in field_path.split('.'):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def get_last_n_sentences(text: str, n: int) -> str:
    if not text:
        return ""
    # First try splitting into sentences by punctuation
    sentences = re.split(r'(?<=[.!?])\s+', text.replace("\r", "\n"))
    # If splitting produced few sentences, also split by newlines and commas as a fallback
    if len(sentences) < n:
        more = []
        for s in sentences:
            pieces = [p.strip() for p in s.split('\n') if p.strip()]
            more.extend(pieces)
        sentences = more if more else sentences
    # Filter empty
    sentences = [s.strip() for s in sentences if s and not s.isspace()]
    if not sentences:
        return text.strip()
    last = sentences[-n:]
    return "\n".join(last)


def build_prompt(sample: dict, categories: List[str], last_n: int) -> str:
    # Auto-select puzzle fields (avoid sending huge arrays): use text_visualization
    # (human-friendly), id, and solutions if present. Trace field is assumed
    # to be `result.message` which we've seen in the JSONL files.
    puzzle_fields = ["id", "text_visualization", "solutions"]
    trace_field = "result.message"

    cat_lines = "\n".join(f"{i+1}. {c}" for i, c in enumerate(categories))

    parts = []
    for f in puzzle_fields:
        if '.' in f:
            val = get_field(sample, f)
        else:
            val = sample.get(f)
        if val is not None:
            # Keep the representation compact; solutions may be large so we
            # pretty-print only their first item when available.
            if f == "solutions" and isinstance(val, list) and val:
                s0 = val[0].copy()
                # show only path length and first/last few coords to keep prompt small
                if "path" in s0 and isinstance(s0["path"], list):
                    path = s0["path"]
                    preview = path[:10]
                    if len(path) > 10:
                        preview = preview + ["...", path[-3:]]
                    s0["path_preview"] = preview
                    s0.pop("path", None)
                parts.append(f"{f}:\n{json.dumps(s0, ensure_ascii=False)}")
            else:
                parts.append(f"{f}:\n{json.dumps(val, ensure_ascii=False)}")

    puzzle_text = "\n\n".join(parts) if parts else json.dumps(sample, ensure_ascii=False)

    raw_trace = get_field(sample, trace_field) or sample.get("message") or ""
    trace_excerpt = get_last_n_sentences(str(raw_trace), last_n)

    # Build user-specified prompt following the requested template.
    gp = generate_prompt(sample)
    orig_prompt_block = f"TASK\n{gp}\n\n"

    analysis = get_field(sample, "result.analysis") or sample.get("result", {}).get("analysis") or {}
    analysis_text = json.dumps(analysis, ensure_ascii=False)

    prompt = (
        "You are tasked to annotate the failure reason for a 2D pathfinding task.\n\n"
        "The original task looked like this:\n\n"
        f"{orig_prompt_block}\n"
        "The final summary and decision is this:\n\n"
        f"{trace_excerpt}\n\n"
        "The path has these problems:\n\n"
        f"PATH ANALYSIS\n{analysis_text}\n\n"
        "Categorize the failure reason. Provide as a valid JSON object with two keys:\n"
        "  - 'categories' (an array of integer indices starting at 1, you may return an empty array if none apply)\n"
        "  - 'explanation' (short justification).\n\n"
        "You can select no reason if nothing is fitting, or up to all reasons.\n\n"
        "Categories:\n"
        f"{cat_lines}\n\n"
        "Answer as valid JSON ONLY, e.g. {\"categories\": [1,3], \"explanation\": \"...\"}."
    )
    return prompt


def call_vllm(prompt: str, model: str, port: int = 8000, timeout: int = 60, api_key: Optional[str] = None) -> Optional[str]:
    """
    Call the local OpenAI-compatible vLLM using the openai Python library.

    This sets openai.api_base to the local server (e.g. http://127.0.0.1:8000/v1)
    and openai.api_key to the provided api_key (or empty string for anonymous).
    """
    base = f"http://127.0.0.1:{port}/v1"
    key = api_key or ""
    client = OpenAI(api_key=key, base_url=base)

    try:
        # Use the new client.chat.completions.create interface
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0.0,
        )
        # Extract content safely
        if resp and hasattr(resp, "get") and resp.get("choices"):
            ch = resp.get("choices")[0]
            # choice may be a mapping-like object
            if isinstance(ch, dict):
                msg = ch.get("message") or {}
                if isinstance(msg, dict) and "content" in msg:
                    return msg.get("content")
                if "text" in ch:
                    return ch.get("text")
            else:
                # try attribute access
                try:
                    return ch.message.content
                except Exception:
                    pass
        # fallback
        try:
            return resp.output
        except Exception:
            return None
    except Exception as e:
        print(f"Error calling vLLM (openai client): {e}", file=sys.stderr)
        return None


def parse_llm_response(text: str, num_categories: int) -> dict:
    # The prompt requests a JSON object with either 'category' (int) or
    # 'categories' (list) but we enforce exactly one selected category.
    text = (text or "").strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return {"categories": [-2], "explanation": f"failed_to_parse_raw_response: {text}"}
    try:
        obj = json.loads(text[start:end+1])
    except Exception:
        return {"categories": [-3], "explanation": f"failed_to_json_parse: {text}"}

    # If the model provided a 'categories' array, validate and return it (allow empty)
    if "categories" in obj:
        cats = obj.get("categories") or []
        if isinstance(cats, int):
            cats = [cats]
        valid = []
        try:
            for x in cats:
                xi = int(x)
                if 1 <= xi <= num_categories:
                    valid.append(xi)
        except Exception:
            # ignore malformed entries
            pass
        return {"categories": valid, "explanation": obj.get("explanation", "")}

    # Fallback: return empty categories with raw explanation
    return {"categories": [], "explanation": f"failed_to_parse_expected_keys: {json.dumps(obj)[:200]}"}


def annotate_file(input_path: str, output_path: str, categories: List[str], model: str, port: int, api_key: Optional[str], last_n: int):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    n = 0
    with open(input_path, "r", encoding="utf-8") as inf, open(output_path, "w", encoding="utf-8") as outf:
        for line in inf:
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
            except Exception as e:
                print(f"Skipping invalid JSON line: {e}", file=sys.stderr)
                continue
            n += 1
            prompt = build_prompt(sample, categories, last_n)
            raw = call_vllm(prompt, model=model, port=port)
            if raw is None:
                ann = {"categories": [-1], "explanation": "vllm_error_forced_choice"}
            else:
                parsed = parse_llm_response(raw, num_categories=len(categories))
                ann = parsed
            # Keep 1-based indices in output (single choice enforced)
            sample["llm_annotation"] = {"categories": ann.get("categories", []), "explanation": ann.get("explanation", ""), "llm_raw": raw}
            outf.write(json.dumps(sample, ensure_ascii=False) + "\n")
            if n % 10 == 0:
                print(f"Annotated {n} samples...")
    print(f"Done — annotated {n} samples -> {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input JSONL file (single file)")
    parser.add_argument("--output", required=True, help="Output JSONL file")
    parser.add_argument("--model", default="lkaesberg/Qwen3-14B-SPaRC-GRPO-8E")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--api-key", default=None, help="OPENAI_API_KEY if needed; default none (localhost)")
    parser.add_argument("--last-n-sentences", type=int, default=100, help="How many of the solver's last sentences to include in the judge prompt")
    args = parser.parse_args()

    cats = DEFAULT_CATEGORIES
    annotate_file(args.input, args.output, cats, args.model, args.port, args.api_key, args.last_n_sentences)


if __name__ == "__main__":
    main()
