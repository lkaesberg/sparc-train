# sparc-train — quick start (vLLM environment)First load the modules for python (>=3.12), gcc and cuda (>=12.6.0)

python -m venv sparc

This repository contains training and evaluation helpers for SPaRC experiments.source sparc/bin/activate

The included SBATCH scripts assume you use a shared `vllm` environment on the

compute nodes. There is one exception: SFT training should run inside the```

`sparc` environment — the `run-sparc-sft.sbatch` script is configured topip install vllm

activate that env for the launched training processes.pip install trl

pip install trl[vllm]

Two example conda environment files are included:pip install sparc-puzzle

- `environment-vllm.yml` — for the vLLM server and orchestration tools.pip install psutil

- `environment-sparc.yml` — for the training / sparc client (SFT) environment.pip install flash_attn --no-build-isolation

```

## Create the environments

Symlink the models folder to your scratch disk with more disk memory

Conda (recommended):```

ln -s /scratch/..../models $HOME/sparc-train/models

```bash``
# vllm server env
conda env create -f environment-vllm.yml
conda activate vllm

# sparc-train — quick start (vLLM environment)

This repository contains training and evaluation helpers for SPaRC experiments.
Top-level SBATCH scripts assume a shared `vllm` conda environment on compute
nodes. There is one exception: SFT worker processes are run inside a separate
`sparc` environment (see note in the SFT SBATCH).

## Included environment files

Two example conda environment files are provided:

- `environment-vllm.yml` — for the vLLM server and orchestration tools.
- `environment-sparc.yml` — for the training / sparc client (SFT) environment.

Create the environments (conda recommended):

```bash
# create vllm env
conda env create -f environment-vllm.yml
conda activate vllm

# create sparc env (used for SFT workers)
conda env create -f environment-sparc.yml
conda activate sparc
```

If you prefer virtualenvs, create them and install the corresponding pip
packages listed in each YAML.

## SBATCH scripts and usage

Files and purpose:

- `analyze/results/sparc/run-sparc.sbatch`: single-node vLLM server + `sparc`
  client for evaluation/debugging. Uses the `vllm` env.
- `run-sparc-grpo.sbatch`: multi-node GRPO training; launches vLLM on a
  dedicated node and runs training across the remaining nodes.
- `run-sparc-ppo.sbatch`: multi-node PPO training.
- `run-sparc-sft.sbatch`: multi-node SFT. Training worker processes run inside
  the `sparc` conda env (see note above).

Submit examples:

```bash
# single-node evaluation
sbatch analyze/results/sparc/run-sparc.sbatch -- --help

# multi-node jobs
sbatch run-sparc-grpo.sbatch
sbatch run-sparc-ppo.sbatch
sbatch run-sparc-sft.sbatch
```

Override env vars during submission, e.g.:

```bash
MODEL_NAME=lkaesberg/Qwen3-32B-SPaRC-GRPO RUN_NAME_ADDITION=8epoch sbatch run-sparc-grpo.sbatch
```

## Annotation pipeline (new)

You can annotate existing JSONL result files using an LLM judge hosted by a
local vLLM server. The repository contains a small annotator and an SBATCH
helper:

- `analyze/annotate.py` — Python script that reads a JSONL file, asks the LLM
  which categories (0–8) apply to each sample, and writes an annotated JSONL
  to `analyze/results/annotate/`.
- `analyze/run-annotate.sbatch` — starts a local vLLM server and runs the
  annotator; outputs are stored under `analyze/results/annotate/`.

Example use (submit with `sbatch`):

```bash
# Provide --input and optionally --categories (newline file) or set INPUT/CATEGORIES env vars
sbatch analyze/run-annotate.sbatch --input analyze/results/sparc/myfile.jsonl --categories analyze/categories.txt
```

Output format:

Each line in the annotated output is the original JSON object with a new key
`llm_annotation` containing:

```json
{
  "categories": [1,4],      // array of selected category indices (1-based)
  "explanation": "...",   // human-readable short explanation (optional)
  "llm_raw": "..."        // raw LLM text
}
```

## Annotation Comparison

### Basic Comparison (Single Annotator)

To compare human annotations with machine (LLM) annotations:

```bash
python analyze/compare_annotations.py
```

This script:
- Compares human annotations from `analyze/results/human_annotation/hagenkort_sparc_annotated.jsonl`
- Against all machine-annotated files with prefix `annotation_samples` in `analyze/results/annotate/`
- Matches samples by line number (validates that puzzle IDs match at each line)
- Calculates macro F1, precision, and recall across all 6 failure categories (A-F)
- Outputs per-category metrics and a summary comparison table

### Enhanced Comparison with Inter-Annotator Agreement

For comprehensive analysis with multiple human annotators:

```bash
python analyze/compare_annotations_with_iaa.py

# Exclude specific categories (useful for low-support categories)
python analyze/compare_annotations_with_iaa.py --exclude-categories A
python analyze/compare_annotations_with_iaa.py --exclude-categories A,C
```

This enhanced script:
- **Calculates inter-annotator agreement** between 3 human annotators using:
  - Pairwise F1 scores between all annotator pairs
  - Fleiss' Kappa for multi-rater agreement (per category)
- **Creates majority-vote baseline** (annotations appearing in ≥2/3 annotators)
- **Compares LLM models** against individual annotators and majority vote
- **Ranks models** in a summary table sorted by F1 score
- Shows per-category performance breakdown
- **Calculates Hamming loss** - fraction of incorrectly predicted labels (multi-label metric)
- **Supports category filtering** via `--exclude-categories` to remove low-support or problematic categories

Example output:
```
MODEL COMPARISON TABLE (vs Majority Vote ≥2/3)
================================================================================
Model                                         F1         Precision    Recall     Hamming   
---------------------------------------------------------------------------------------
openai_gpt-oss-120b                           0.5991     0.5771       0.6992     0.2200    
openai_gpt-oss-20b                            0.5533     0.4337       0.7784     0.2740    
Qwen_Qwen3-32B                                0.4568     0.5157       0.5151     0.2180    
...
---------------------------------------------------------------------------------------
Human Baseline (Avg Pairwise)                 0.4553     N/A          N/A        N/A       

Note: Hamming Loss = fraction of incorrectly predicted labels (lower is better, 0 = perfect)

PER-CATEGORY F1 SCORES (vs Majority Vote ≥2/3)
================================================================================
Model                                         B        C        D        E        F       
openai_gpt-oss-120b                           0.6607   0.2857   0.5714   0.6316   0.8462
...

Best Performing Models (vs Majority Vote):
  1. openai_gpt-oss-120b: F1 = 0.5991 (above human baseline)
  2. openai_gpt-oss-20b: F1 = 0.5533 (above human baseline)
  3. Qwen_Qwen3-32B: F1 = 0.4568 (below human baseline)

(Results shown with --exclude-categories A)
```

**Key Insights:**
- Human annotators achieve ~39% average pairwise F1 (46% when excluding Category A), showing annotation difficulty
- Top LLM models (50% F1, 60% without Category A) exceed human baseline, validating the approach
- **Hamming loss** ranges from 0.22-0.34 for top models (~22-34% of labels incorrect), complementing F1 scores
- Categories E and F show moderate human agreement (Kappa ~0.5-0.6)
- Categories A, B, C show low human agreement, indicating ambiguity
- **Category A** has very low support (4 samples) and poor agreement - consider excluding with `--exclude-categories A`
- Excluding low-support categories can significantly improve reported metrics (e.g., +20% F1 for top models)
- Best models achieve **~22% Hamming loss** (openai_gpt-oss-120b, Qwen_Qwen3-32B) = 78% label accuracy

The script automatically converts human annotation codes (e.g., `a_planning_logical_flaw`)
to LLM letter codes (e.g., `A`) for comparison.

## Dataset Sample Structure

Each entry in the JSONL result files (`analyze/results/sparc/*.jsonl`) contains the following keys:

### Puzzle Information
- **`id`**: Unique puzzle identifier (string)
- **`difficulty_level`**: Integer difficulty rating (1-5, where 1 is easiest and 5 is hardest)
- **`difficulty_score`**: Float difficulty score (continuous measure)
- **`grid_size`**: Object with `height` and `width` keys (puzzle dimensions)
- **`polyshapes`**: JSON string containing polyomino shape definitions
- **`puzzle_array`**: 2D array representing the puzzle grid with cell types and constraints
- **`solution_count`**: Number of valid solutions for this puzzle
- **`solutions`**: Array of solution objects, each containing `index`, `path` (list of {x, y} coordinates), and `pathLength`
- **`text_visualization`**: Human-readable YAML representation of the puzzle

### Model Results
- **`result`**: Object containing the model's attempt and analysis
  - **`puzzle_id`**: Reference to the puzzle ID
  - **`solved`**: Boolean indicating if the puzzle was correctly solved
  - **`analysis`**: Object with validation metrics:
    - `starts_at_start_ends_at_exit`: Path begins at start and ends at exit
    - `connected_line`: Path forms a connected line
    - `non_intersecting_line`: Path doesn't cross itself
    - `start_to_exit_connected`: Start and exit are connected
    - `no_rule_crossing`: No puzzle rules were violated
    - `fully_valid_path`: Overall validity (all checks passed)
  - **`processing_time`**: Time in seconds to generate the solution
  - **`extracted_path`**: List of {x, y} coordinates extracted from model output
  - **`message`**: Raw model output (typically includes thinking process and answer)
  - **`error`**: Error message if processing failed (null otherwise)

### Optional Annotations
- **`failure_annotation`**: (Optional) Manual annotation for failed attempts
  - **`completed`**: Boolean indicating if annotation is complete
  - **`failure_reasons`**: Array of failure category codes:
    - `a_planning_logical_flaw`: Logical or planning errors in approach
    - `b_misunderstood_invented_rule`: Misinterpreted or invented puzzle rules
    - `c_spatial_geometric_misjudgment`: Spatial reasoning or geometric errors
    - `d_premature_verification`: Claims solution is correct without checking key rules
    - `e_no_correction_despite_noticing`: Recognizes error but doesn't adjust the plan
    - `f_grid_coordinate_error`: Incorrect coordinates or indexing (off-by-one, swapped x/y, out of bounds)
  - **`other_reason`**: Free-text field for additional context
  - **`puzzle_id`**: Reference to the puzzle ID

- **`llm_annotation`**: (Optional) LLM judge annotation (see Annotation pipeline section)
  - **`categories`**: Array of letter codes for failure categories (e.g., ["C", "A"]):
    - `A`: Planning/logical flaw in the reasoning approach
    - `B`: Misunderstood or invented puzzle rules
    - `C`: Spatial/geometric misjudgment or miscalculation
    - `D`: Premature verification - claims correctness without checking key rules
    - `E`: No correction despite noticing - recognizes errors but doesn't adjust
    - `F`: Grid/coordinate error - off-by-one, swapped x/y, or out-of-bounds steps
  - **`explanation`**: Human-readable explanation of the failure modes
  - **`llm_raw`**: Complete raw LLM output including thinking process

### Example Sample
```json
{
  "difficulty_level": 3,
  "difficulty_score": 2.94,
  "id": "aa31d05ed8fdb273",
  "grid_size": {"height": 6, "width": 3},
  "result": {
    "puzzle_id": "aa31d05ed8fdb273",
    "solved": false,
    "analysis": {
      "fully_valid_path": false,
      "no_rule_crossing": false
    },
    "processing_time": 4.74,
    "extracted_path": [{"x": 6, "y": 0}, {"x": 6, "y": 1}]
  },
  "failure_annotation": {
    "completed": true,
    "failure_reasons": ["a_planning_logical_flaw", "c_spatial_geometric_misjudgment"],
    "other_reason": "",
    "puzzle_id": "aa31d05ed8fdb273"
  },
  "llm_annotation": {
    "categories": ["C", "A"],
    "explanation": "The model exhibits a spatial/geometric misjudgment (C) by assuming regions can fit polyshapes without verifying their actual size and layout. Additionally, the model shows a planning flaw (A) by constructing a path that appears to work step-by-step but fails to account for polyshape constraints.",
    "llm_raw": "<think>\n...\n</think>\n\n{\"categories\": [\"C\", \"A\"], \"explanation\": \"...\"}"
  }
}
```

## Logs and troubleshooting

- SBATCH output is controlled by each script's `#SBATCH --output` setting.
- The annotation SBATCH writes vLLM server log to `logs/vllm_annotate_<jobid>.log`.
- If the model cannot start or returns invalid JSON, the annotator will store
  an empty `categories` array and the raw LLM output under `llm_raw`.

If you'd like, I can:
- Pin exact package versions in the environment YAMLs.
- Add batching, parallel annotation, or retries to `annotate.py`.
- Change the annotator to use a different OpenAI-compatible path (e.g., the
  completions endpoint) if your model requires it.
