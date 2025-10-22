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

# Global client to avoid reinitializing on every call
_vllm_client = None


def load_categories(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


# Default categories: if the user does not pass a --categories file, these
# will be used. These represent common failure modes in pathfinding reasoning.
DEFAULT_CATEGORIES = [
    "A — Planning / Logical Reasoning Flaw: Focuses on local moves or inconsistent logic. Often builds a path that works step-by-step but cannot satisfy all constraints in the end.",
    "B — Misunderstood or Invented Rule: Misinterprets what a rule means, ignores it, or invents new constraints not part of the puzzle (e.g., assumes illegal symmetry or shape rules).",
    "C — Spatial / Geometric Misjudgment: Makes geometric mistakes such as wrong shape size, rotation, or region estimation. Often traps itself in areas too small for the required pattern.",
    "D — Premature Verification / Overconfidence: Claims the solution is correct or fully verified without checking key rules. Typical statements include 'this should work' while violations remain.",
    "E — No Correction Despite Noticing Issue: Recognizes a contradiction or error in reasoning but never adjusts the plan or recomputes the path.",
    "F — Grid / Coordinate Error: Uses incorrect coordinates or indexing (off-by-one, swapped x/y, or path steps outside the defined board) due to coordinate confusion."
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
        "You are an expert at analyzing reasoning failures in pathfinding tasks.\n\n"
        "The original task was:\n\n"
        f"{orig_prompt_block}\n"
        "The model's final reasoning and decision:\n\n"
        f"{trace_excerpt}\n\n"
        "Path analysis (detected issues):\n\n"
        f"{analysis_text}\n\n"
        "Your task: Identify which failure mode(s) best explain why the solution failed.\n\n"
        "Failure Categories:\n"
        f"{cat_lines}\n\n"
        "Instructions:\n"
        "- Select one or more categories (A-F) that apply, or return empty array [] if none fit\n"
        "- Provide a brief explanation citing specific evidence from the reasoning\n"
        "- Be precise: distinguish between misunderstanding rules (B) vs poor planning (A) vs geometric errors (C)\n\n"
        "Return ONLY a valid JSON object with this exact format:\n"
        '{{"categories": ["A", "C"], "explanation": "The model exhibits... because..."}}\n\n'
        "Your response (JSON only):"
    )
    return prompt


def get_vllm_client(port: int = 8000, api_key: Optional[str] = None) -> OpenAI:
    """
    Get or create a singleton OpenAI client for vLLM.
    Reuses the client across calls to avoid overhead.
    """
    global _vllm_client
    if _vllm_client is None:
        base = f"http://127.0.0.1:{port}/v1"
        key = api_key or "EMPTY"  # vLLM typically requires a non-empty key
        print(f"[INFO] Initializing vLLM client at {base}", file=sys.stderr)
        _vllm_client = OpenAI(api_key=key, base_url=base)
    return _vllm_client


def call_vllm(prompt: str, model: str, port: int = 8000, timeout: int = 60, api_key: Optional[str] = None, max_retries: int = 5) -> Optional[str]:
    """
    Call the local OpenAI-compatible vLLM using the openai Python library v1.0+.

    Uses attribute access (not dict-like .get()) for the response object.
    Retries up to max_retries times on failure.
    """
    client = get_vllm_client(port=port, api_key=api_key)

    for attempt in range(max_retries):
        try:
            if attempt > 0:
                print(f"[INFO] Retry attempt {attempt}/{max_retries-1}", file=sys.stderr)
            print(f"[INFO] Calling vLLM with model '{model}' (prompt length: {len(prompt)} chars)", file=sys.stderr)
            
            # Use the chat.completions.create interface (OpenAI v1.0+)
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10000,
                temperature=0.0,
                timeout=timeout,
            )
            
            # OpenAI v1.0+ uses attribute access, not dict-like access
            if resp and resp.choices:
                choice = resp.choices[0]
                if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                    content = choice.message.content
                    print(f"[INFO] Received response ({len(content)} chars)", file=sys.stderr)
                    return content
                # Fallback for completion endpoint (if used)
                if hasattr(choice, 'text'):
                    text = choice.text
                    print(f"[INFO] Received text response ({len(text)} chars)", file=sys.stderr)
                    return text
            
            print(f"[WARN] No content in response (attempt {attempt+1}/{max_retries})", file=sys.stderr)
            if attempt < max_retries - 1:
                continue  # Try again
            return None
            
        except Exception as e:
            print(f"[ERROR] Error calling vLLM (attempt {attempt+1}/{max_retries}): {e}", file=sys.stderr)
            if attempt < max_retries - 1:
                import time
                wait_time = 2 ** attempt  # Exponential backoff: 1, 2, 4, 8 seconds
                print(f"[INFO] Waiting {wait_time}s before retry...", file=sys.stderr)
                time.sleep(wait_time)
            else:
                print(f"[ERROR] All {max_retries} attempts failed", file=sys.stderr)
                return None
    
    return None


def parse_llm_response(text: str, num_categories: int) -> dict:
    # The prompt requests a JSON object with 'categories' (array of letters A-F)
    # and 'explanation' (string).
    text = (text or "").strip()
    text = text.split("</think>")[-1]
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        print(f"[WARN] Failed to find JSON object in response: {text[:100]}...", file=sys.stderr)
        return {"categories": ["ERROR_PARSE"], "explanation": f"failed_to_parse_raw_response: {text}"}
    try:
        obj = json.loads(text[start:end+1])
    except Exception as e:
        print(f"[WARN] Failed to parse JSON: {e}", file=sys.stderr)
        return {"categories": ["ERROR_JSON"], "explanation": f"failed_to_json_parse: {text}"}

    # If the model provided a 'categories' array, validate and return it (allow empty)
    if "categories" in obj:
        cats = obj.get("categories") or []
        if isinstance(cats, str):
            cats = [cats]
        valid = []
        allowed_letters = [chr(65 + i) for i in range(num_categories)]  # A, B, C, D, E, F
        try:
            for x in cats:
                x_upper = str(x).upper().strip()
                if x_upper in allowed_letters:
                    valid.append(x_upper)
        except Exception:
            # ignore malformed entries
            pass
        print(f"[INFO] Parsed categories: {valid}", file=sys.stderr)
        return {"categories": valid, "explanation": obj.get("explanation", "")}

    # Fallback: return empty categories with raw explanation
    print(f"[WARN] No 'categories' key found in response", file=sys.stderr)
    return {"categories": [], "explanation": f"failed_to_parse_expected_keys: {json.dumps(obj)[:200]}"}


def annotate_file(input_path: str, output_path: str, categories: List[str], model: str, port: int, api_key: Optional[str], last_n: int):
    print(f"[INFO] Starting annotation process", file=sys.stderr)
    print(f"[INFO] Input file: {input_path}", file=sys.stderr)
    print(f"[INFO] Output file: {output_path}", file=sys.stderr)
    print(f"[INFO] Model: {model}", file=sys.stderr)
    print(f"[INFO] Port: {port}", file=sys.stderr)
    print(f"[INFO] Categories: {len(categories)} categories defined", file=sys.stderr)
    print(f"[INFO] Using last {last_n} sentences from trace", file=sys.stderr)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    n = 0
    errors = 0
    with open(input_path, "r", encoding="utf-8") as inf, open(output_path, "w", encoding="utf-8") as outf:
        for line in inf:
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
            except Exception as e:
                print(f"[ERROR] Skipping invalid JSON line: {e}", file=sys.stderr)
                continue
            n += 1
            sample_id = sample.get("id", f"sample_{n}")
            print(f"\n[INFO] Processing sample {n} (id: {sample_id})", file=sys.stderr)
            
            prompt = build_prompt(sample, categories, last_n)
            raw = call_vllm(prompt, model=model, port=port, api_key=api_key)
            if raw is None:
                print(f"[WARN] No response from vLLM for sample {n}", file=sys.stderr)
                ann = {"categories": ["ERROR_VLLM"], "explanation": "vllm_error_no_response"}
                errors += 1
            else:
                parsed = parse_llm_response(raw, num_categories=len(categories))
                ann = parsed
            # Keep 1-based indices in output (single choice enforced)
            sample["llm_annotation"] = {"categories": ann.get("categories", []), "explanation": ann.get("explanation", ""), "llm_raw": raw}
            outf.write(json.dumps(sample, ensure_ascii=False) + "\n")
            if n % 10 == 0:
                print(f"[PROGRESS] Annotated {n} samples ({errors} errors so far)...", file=sys.stderr)
    
    print(f"\n[SUCCESS] Done — annotated {n} samples -> {output_path}", file=sys.stderr)
    print(f"[STATS] Total errors: {errors}/{n} ({100*errors/n:.1f}%)" if n > 0 else "[STATS] No samples processed", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input JSONL file (single file)")
    parser.add_argument("--output", required=True, help="Output JSONL file")
    parser.add_argument("--model", default="lkaesberg/Qwen3-14B-SPaRC-GRPO-8E")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--api-key", default=None, help="OPENAI_API_KEY if needed; default none (localhost)")
    parser.add_argument("--last-n-sentences", type=int, default=100, help="How many of the solver's last sentences to include in the judge prompt")
    args = parser.parse_args()

    print(f"\n{'='*60}", file=sys.stderr)
    print(f"SPaRC Annotation Script", file=sys.stderr)
    print(f"{'='*60}\n", file=sys.stderr)

    cats = DEFAULT_CATEGORIES
    annotate_file(args.input, args.output, cats, args.model, args.port, args.api_key, args.last_n_sentences)
    
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"Annotation complete!", file=sys.stderr)
    print(f"{'='*60}\n", file=sys.stderr)


if __name__ == "__main__":
    main()
