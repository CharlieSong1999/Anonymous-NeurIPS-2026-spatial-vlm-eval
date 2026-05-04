# Spatial-VLM Eval (anonymous release)

> Anonymized release for double-blind NeurIPS 2026 Evaluations &
> Datasets review. De-anonymized version will be linked from the
> dataset cards on Hugging Face after acceptance.

This repo contains the canonical evaluation pipeline for the
**Q1 (bin prediction)** spatial-reasoning benchmark. It is the official
reference implementation for the results reported in the paper.

## Quick start

```bash
# 1. Install deps (Python 3.11+ recommended)
pip install -r requirements.txt

# 2. Download the dataset from Hugging Face
huggingface-cli download nipsedtrack2026/q1-bin-prediction --repo-type=dataset --local-dir data/q1

# Direct dataset URL:
#   https://huggingface.co/datasets/nipsedtrack2026/q1-bin-prediction

# 3. Run a smoke test (one API model — needs GEMINI_API_KEY)
export GEMINI_API_KEY=...        # for gemini-3-flash
# export OPENAI_API_KEY=...      # for gpt-5.4

# Q1 sighted, gemini-3-flash, M=1, 100 queries from the small subset
python -m src.eval_v2 --models gemini-3-flash --conditions sighted     --eval-set 100 --m-repeats 1
```

## Environment variables

The default assumes you've downloaded the HF dataset into `./data/q1`
(i.e. you ran the quick-start above). Override via env vars only when
your layout differs:

| Variable | Default | Purpose |
|---|---|---|
| `Q1_DATASET_ROOT` | `./data/q1` | HF Q1 dataset root (`queries.parquet`, `frames/`) |
| `TESTSET_ROOT` | parent of `src/` | Where logs and runs are written |
| `RUNS_DIR` | `<TESTSET_ROOT>/runs` | Q1 eval JSONL output |
| `EK_FRAMES_ROOT` | upstream EK layout | Only used if `bundled_frame_path` is missing |
| `HD_EPIC_FRAMES_ROOT` | upstream HD layout | Only used if `bundled_frame_path` is missing |
| `GEMINI_API_KEY`, `OPENAI_API_KEY` | unset | Required for the respective API models |

## Documentation

- `docs/run-experiments-on-setA-extended.md` — comprehensive
  onboarding doc covering data layout, schema, loaders, Q1 eval
  recipes, metric computation, common pitfalls.
- `docs/cautious-on-hd-epic.md` — HD-EPIC-specific quirks (Aria pose
  convention, fisheye undistortion). Mandatory pre-read before any
  pipeline that touches HD frames.

## Provenance

Source datasets:
- Epic-Kitchens (CC-BY-NC 4.0): https://epic-kitchens.github.io/
- HD-EPIC (CC-BY-NC 4.0): https://hd-epic.github.io/

This repository's code is licensed CC-BY-NC 4.0 to match the source
datasets and the derived Q1 dataset.

## Reproducing the published results

```bash
# Run a sighted eval with API models (~$X for the headline cell)
python -m src.eval_v2 --models gemini-3-flash --conditions sighted     --eval-set full --m-repeats 20

# Compute the headline metrics (calibrated NLL family + group-KL)
python -m src.compare_averaging
python -m src.group_kl --eval-set full
```

API models (Gemini, GPT) require API keys (`GEMINI_API_KEY`,
`OPENAI_API_KEY`).

See `docs/run-experiments-on-setA-extended.md` for the full recipe.
