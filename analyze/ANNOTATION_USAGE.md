# Annotation Script Usage Guide

The `run-annotate-container.sbatch` script now supports two modes:

## 1. Single File Mode (Original Behavior)

Process a single file by setting the `INPUT` variable:

```bash
sbatch --export=INPUT=/path/to/file.jsonl,MODEL=RedHatAI/Llama-4-Scout-17B-16E analyze/run-annotate-container.sbatch
```

Or with a custom output path:

```bash
sbatch --export=INPUT=/path/to/file.jsonl,OUTPUT=/path/to/output.jsonl,MODEL=YourModel analyze/run-annotate-container.sbatch
```

## 2. Multi-File Mode (New Feature)

Process multiple files matching a pattern by **not** setting `INPUT` (or setting it to empty):

### Example 1: Annotate all files containing "lkaesberg_Qwen3-32B"

```bash
sbatch --export=INPUT_DIR=/path/to/files,FILE_PATTERN='*lkaesberg_Qwen3-32B*.jsonl',MODEL=YourModel analyze/run-annotate-container.sbatch
```

### Example 2: Annotate all JSONL files in a directory

```bash
sbatch --export=INPUT_DIR=/path/to/files,FILE_PATTERN='*.jsonl',MODEL=YourModel analyze/run-annotate-container.sbatch
```

### Example 3: Using the defaults

The script has sensible defaults. If you don't set anything, it will look for files matching `*lkaesberg_Qwen3-32B*.jsonl` in the default directory:

```bash
sbatch analyze/run-annotate-container.sbatch
```

## Configuration Variables

### Single File Mode Variables
- `INPUT`: Path to a single input JSONL file (if set, enables single file mode)
- `OUTPUT`: Path to output file (optional, auto-generated if not set)

### Multi-File Mode Variables
- `INPUT_DIR`: Directory to search for files (default: `/mnt/vast-standard/home/kaesberg1/u16096/sparc-train/analyze/results/human_annotation`)
- `FILE_PATTERN`: Glob pattern to match files (default: `*lkaesberg_Qwen3-32B*.jsonl`)
- `OUTPUT_DIR`: Directory for output files (default: `results/annotate`)

### Common Variables
- `MODEL`: HuggingFace model name (default: `RedHatAI/Llama-4-Scout-17B-16E-Instruct-quantized.w4a16`)
- `LLM_PORT`: Port for vLLM server (default: `8000`)
- `LAST_N_SENTENCES`: Number of sentences from solver trace to include (default: `100`)
- `LLM_GPU_UTIL`: GPU memory utilization (default: `0.90`)

## Key Features

1. **Single vLLM Instance**: The vLLM server starts once and processes all files, saving significant time
2. **Backward Compatible**: Works exactly like before if you set the `INPUT` variable
3. **Pattern Matching**: Use shell glob patterns to select multiple files
4. **Progress Tracking**: Shows progress as each file is processed
5. **Error Handling**: Continues processing if one file fails, reports summary at the end

## Output

For multi-file mode, output files are automatically named:
```
{output_dir}/{input_basename}.annotated_by_{model_name}.jsonl
```

Example:
- Input: `annotation_samples.jsonl`
- Model: `lkaesberg/Qwen3-32B`
- Output: `results/annotate/annotation_samples.annotated_by_lkaesberg_Qwen3-32B.jsonl`

## Example: Annotate Multiple Model Outputs

If you have files like:
```
results/sparc/lkaesberg_Qwen3-32B_v1.jsonl
results/sparc/lkaesberg_Qwen3-32B_v2.jsonl
results/sparc/lkaesberg_Qwen3-32B_final.jsonl
```

Run:
```bash
sbatch --export=INPUT_DIR=results/sparc,FILE_PATTERN='lkaesberg_Qwen3-32B*.jsonl',MODEL=meta-llama/Llama-3.3-70B-Instruct analyze/run-annotate-container.sbatch
```

This will:
1. Start vLLM server once
2. Process all three files
3. Output to `results/annotate/` with annotated versions
4. Stop the vLLM server
5. Report success/failure summary

