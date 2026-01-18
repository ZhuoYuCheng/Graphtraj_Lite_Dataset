# GraphTraj-lite Builder

This folder contains a minimal pipeline to generate a GraphTraj-lite style dataset
per the requirements in `autodl-tmp/.autodl/AGENT.md`.

## Output
- JSONL: `/root/autodl-tmp/graphtraj_lite/outputs/gtl_1000.jsonl`

## Requirements
- Python 3.12
- Torch already available in the environment
- Install deps:
  - `pip install -r /root/autodl-tmp/graphtraj_lite/requirements.txt`

## Run
```
python /root/autodl-tmp/graphtraj_lite/scripts/build_dataset.py \
  --output /root/autodl-tmp/graphtraj_lite/outputs/gtl_1000.jsonl
```

## Notes
- GAIA is optional. If it cannot be accessed, quotas are redistributed across MBPP+, MATH, GSM8K.
- To avoid attempting GAIA at all, pass `--skip-gaia`.
- GAIA on Hugging Face is gated; authenticate via `huggingface-cli login` or set `HF_TOKEN`/`HUGGINGFACE_TOKEN`.
- You must also request/accept access on the GAIA dataset page for the account tied to your token.
- Sparse attention graphs are computed from a small causal LM (default: distilgpt2) using
  step-level aggregation and top-k pruning.
