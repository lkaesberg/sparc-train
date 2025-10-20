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
