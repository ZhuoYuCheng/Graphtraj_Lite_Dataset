# GraphTraj-lite Builder

This folder contains a minimal pipeline to generate a GraphTraj-lite style dataset
per the requirements in `autodl-tmp/.autodl/AGENT.md`.

## Output
- JSONL: `/root/autodl-tmp/graphtraj_lite/outputs/gtl_300.jsonl`

## Requirements
- Python 3.12
- Torch already available in the environment
- Install deps:
  - `pip install -r /root/autodl-tmp/graphtraj_lite/requirements.txt`

## Run
```
python /root/autodl-tmp/graphtraj_lite/scripts/build_dataset.py \
  --output /root/autodl-tmp/graphtraj_lite/outputs/gtl_300.jsonl
```

## Notes
- GAIA is optional. If it cannot be accessed, quotas are redistributed across MBPP+, MATH, GSM8K.
- Sparse attention graphs are computed from a small causal LM (default: distilgpt2) using
  step-level aggregation and top-k pruning.
