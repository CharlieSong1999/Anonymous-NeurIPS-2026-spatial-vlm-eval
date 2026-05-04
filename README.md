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

# 2. Download datasets from Hugging Face
huggingface-cli download nipsedtrack2026/q1-bin-prediction --repo-type=dataset --local-dir data/q1
huggingface-cli download nipsedtrack2026/q2-cubemap-mcq    --repo-type=dataset --local-dir data/q2

# Direct dataset URLs:
#   https://huggingface.co/datasets/nipsedtrack2026/q1-bin-prediction
#   https://huggingface.co/datasets/nipsedtrack2026/q2-cubemap-mcq

# 3. Run a smoke test (10 queries, one API model — needs GEMINI_API_KEY)
export GEMINI_API_KEY=...        # for gemini-3-flash
# export OPENAI_API_KEY=...      # for gpt-5.4

# Q1 sighted, gemini-3-flash, M=1, 10 queries from the human pool
python -m src.eval_v2 --models gemini-3-flash --conditions sighted     --eval-set 100 --m-repeats 1

# Q2 sighted, gemini-3-flash, 10 queries
python -m src.q2_eval --models gemini-3-flash     --conditions A_frame_plus_cubemap --max 10
```

## Environment variables

The defaults assume you've downloaded the HF datasets into `./data/q1`
and `./data/q2` (i.e. you ran the quick-start above). Override via env
vars only when your layout differs:

| Variable | Default | Purpose |
|---|---|---|
| `Q1_DATASET_ROOT` | `./data/q1` | HF Q1 dataset root (`queries.parquet`, `frames/`, `human_labels/`) |
| `Q2_DATASET_ROOT` | `./data/q2` | HF Q2 dataset root (one subdir per strategy) |
| `TESTSET_ROOT` | parent of `src/` | Where logs and runs are written |
| `RUNS_DIR` | `<TESTSET_ROOT>/runs` | Q1 eval JSONL output |
| `Q2_RUNS_DIR` | `<TESTSET_ROOT>/runs/q2` | Q2 eval JSONL output |
| `EK_FRAMES_ROOT` | upstream EK layout | Only used if `bundled_frame_path` is missing |
| `HD_EPIC_FRAMES_ROOT` | upstream HD layout | Only used if `bundled_frame_path` is missing |
| `GEMINI_API_KEY`, `OPENAI_API_KEY` | unset | Required for the respective API models |

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
