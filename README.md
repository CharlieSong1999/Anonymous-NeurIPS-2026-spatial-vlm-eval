# Spatial-VLM Eval (anonymous release)

> Anonymized release for double-blind NeurIPS 2026 Evaluations &
> Datasets review. De-anonymized version will be linked from the
> dataset cards on Hugging Face after acceptance.

This repo contains the canonical evaluation pipeline for the
**Q1 (bin prediction)** and **Q2 (cubemap MCQ)** spatial-reasoning
benchmarks. It is the official reference implementation for the
results reported in the paper.

## Quick start

```bash
# 1. Install deps (Python 3.11+ recommended)
pip install -r requirements.txt

# 2. Set environment variables for source-frame paths
export EK_FRAMES_ROOT=/path/to/epic-kitchens
export HD_EPIC_FRAMES_ROOT=/path/to/hd-epic/Participants
export HD_EPIC_VRS_REFERENCE=/path/to/hd-epic/vrs/P01-20240202-110250_anonymized.vrs
export PROJECT_ROOT=/path/to/this/repo  # optional; defaults to repo root

# 3. Download datasets from Hugging Face
huggingface-cli download nipsedtrack2026/q1-bin-prediction --repo-type=dataset --local-dir data/q1
huggingface-cli download nipsedtrack2026/q2-cubemap-mcq    --repo-type=dataset --local-dir data/q2

# Direct dataset URLs:
#   https://huggingface.co/datasets/nipsedtrack2026/q1-bin-prediction
#   https://huggingface.co/datasets/nipsedtrack2026/q2-cubemap-mcq

# 4. Run a smoke test (3 queries, one local model)
python -m src.q2_eval --max 3 --models qwen3.5-9b --conditions A_cubemap_only
```

## Documentation

- `docs/run-experiments-on-setA-extended.md` — comprehensive
  onboarding doc covering data layout, schema, loaders, Q1 + Q2 eval
  recipes, metric computation, common pitfalls.
- `docs/cautious-on-hd-epic.md` — HD-EPIC-specific quirks (Aria pose
  convention, fisheye undistortion). Mandatory pre-read before any
  pipeline that touches HD frames.

## Provenance

Source datasets:
- Epic-Kitchens (CC-BY-NC 4.0): https://epic-kitchens.github.io/
- HD-EPIC (CC-BY-NC 4.0): https://hd-epic.github.io/

This repository's code is licensed CC-BY-NC 4.0 to match the source
datasets and the derived Q1+Q2 dataset.

## Reproducing the published results

The published headline numbers (mode accuracy on Q2 frame+cubemap
condition) are reproducible end-to-end:

```bash
# Generate Q2 cubemaps from scratch (~5 min CPU)
python -m src.q2_generate_full

# Run eval — local models in parallel
python -m src.q2_eval --models qwen3.5-9b gemma-4-31b qwen3-vl-30b     --conditions A_cubemap_only A_frame_plus_cubemap

# Compute metrics
python -m src.q2_metrics
```

API models (Gemini, GPT) require API keys (`GEMINI_API_KEY`,
`OPENAI_API_KEY`). Cost: ~$35 total for the full headline cell at the
quoted prices in the paper.

See `docs/run-experiments-on-setA-extended.md` for the full recipe.
