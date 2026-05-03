"""
M-ablation eval on the extended Set A test set.

Subsets 500 queries from setA_extended_filtered.parquet, runs M=25
baseline_sighted on 3 local models, then post-hoc M-ablation analysis
takes the first {5, 10, 15, 20, 25} responses to compute drift.

Output:
  exp/m_ablation_001/runs_extended/{model}_sighted_M25.jsonl
  exp/m_ablation_001/m_ablation_extended.csv

Run:
  conda run -n slam python3 -m src.m_ablation_eval [--models qwen3.5-9b gemma-4-31b]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    datefmt="%H:%M:%S")

sys.path.insert(0, str(Path(__file__).parent))
from eval_v2 import (
    MODELS, TEMPERATURE, MAX_TOKENS,
    CONCURRENCY_PER_MODEL,
    build_sighted_prompt, call_vlm, parse_response,
    img_to_jpeg_bytes, resolve_frame_path,
    append_jsonl, load_existing,
)
M_REPEATS = 25  # default; overridden by --M arg

TESTSET = Path("/path/to/this/repo")
DATA_DIR = TESTSET / "data"
RUNS_DIR = TESTSET / "exp" / "m_ablation_001" / "runs_extended"
RUNS_DIR.mkdir(parents=True, exist_ok=True)

N_SUBSET = 500
SUBSET_SEED = 42


def select_subset():
    """Pick N_SUBSET queries from setA_extended_filtered (or
    setA_extended if filter not yet done) with bin-balanced sampling."""
    filt_path = DATA_DIR / 'setA_extended_filtered.parquet'
    raw_path = DATA_DIR / 'setA_extended.parquet'
    if filt_path.exists():
        sa = pd.read_parquet(filt_path)
        logger.info(f'Using filtered: {filt_path} ({len(sa)} rows)')
    else:
        sa = pd.read_parquet(raw_path)
        logger.info(f'Using unfiltered: {raw_path} ({len(sa)} rows)')

    # Bin-balanced subset
    rng = np.random.RandomState(SUBSET_SEED)
    yaw_bins = sorted(sa.yaw_bin_4.unique())
    h_bins = sorted(sa.height_bin.unique())
    n_bins = len(yaw_bins) * len(h_bins)
    base = N_SUBSET // n_bins
    selected, leftover = [], []
    for yb in yaw_bins:
        for hb in h_bins:
            sub = sa[(sa.yaw_bin_4 == yb) & (sa.height_bin == hb)]
            idx = list(sub.index)
            rng.shuffle(idx)
            take = min(base, len(idx))
            selected.extend(idx[:take])
            leftover.extend(idx[take:])
    rng.shuffle(leftover)
    needed = N_SUBSET - len(selected)
    if needed > 0:
        selected.extend(leftover[:needed])
    subset = sa.loc[selected[:N_SUBSET]].reset_index(drop=True)
    logger.info(f'Subset: {len(subset)} queries, {subset.canonical_label.nunique()} labels')
    logger.info(f'  yaw bins: {subset.yaw_bin_4.value_counts().sort_index().to_dict()}')
    logger.info(f'  height bins: {subset.height_bin.value_counts().sort_index().to_dict()}')
    return subset


async def run_call(sem, prompt, img_bytes_list, model_cfg,
                   sample_id, repeat_id, label, gt_yaw, gt_height,
                   out_path, max_tokens):
    async with sem:
        raw = await call_vlm(prompt, img_bytes_list, model_cfg,
                             TEMPERATURE, max_tokens)
    parsed = parse_response(raw)
    rec = {'sample_id': sample_id, 'repeat_id': repeat_id,
           'label': label, 'gt_yaw': int(gt_yaw),
           'gt_height': int(gt_height), 'raw': raw, 'parsed': parsed}
    append_jsonl(out_path, rec)
    return rec


async def eval_model(model_name, model_cfg, subset, frame_cache, sem,
                     M_total, rep_offset=0, out_filename_M=None):
    """Run M_total reps starting at repeat_id=rep_offset."""
    fname_M = out_filename_M if out_filename_M is not None else (rep_offset + M_total)
    out_path = RUNS_DIR / f'{model_name}_sighted_M{fname_M}.jsonl'
    done = load_existing(out_path)
    expected = len(subset) * (rep_offset + M_total)
    logger.info(f'[{model_name}] {out_path.name}: {len(done)}/{expected} done '
                f'(rep_offset={rep_offset}, M_new={M_total})')

    tasks = []
    for _, row in subset.iterrows():
        label = row['canonical_label']
        prompt = build_sighted_prompt(label)
        fp = resolve_frame_path(row['dataset'], row['video_id'],
                                int(row['frame_index']))
        if fp not in frame_cache:
            if not Path(fp).exists():
                logger.warning(f'  Frame missing: {fp}')
                continue
            img_bgr = cv2.imread(fp)
            if img_bgr is None:
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            frame_cache[fp] = img_to_jpeg_bytes(img_rgb)
        for r in range(rep_offset, rep_offset + M_total):
            if (row['sample_id'], r) in done:
                continue
            tasks.append(run_call(
                sem, prompt, [frame_cache[fp]], model_cfg,
                row['sample_id'], r, label,
                row['yaw_bin_4'], row['height_bin'],
                out_path, MAX_TOKENS))

    if not tasks:
        return

    logger.info(f'[{model_name}] Queueing {len(tasks)} calls')
    t0 = time.time()
    completed = 0
    for batch_start in range(0, len(tasks), 200):
        batch = tasks[batch_start:batch_start + 200]
        await asyncio.gather(*batch)
        completed += len(batch)
        elapsed = time.time() - t0
        rate = completed / elapsed
        eta = (len(tasks) - completed) / rate if rate > 0 else 0
        logger.info(f'[{model_name}] {completed}/{len(tasks)} '
                    f'({rate:.1f}/s, ETA {eta/60:.1f} min)')


async def main_async(args):
    if args.eval_set == 'full':
        subset = select_subset()
        # Save for reproducibility
        subset.to_parquet(DATA_DIR / 'setA_extended_subset500.parquet', index=False)
    else:
        from src._eval_set import filter_by_eval_set
        sa = pd.read_parquet(DATA_DIR / 'setA_extended.parquet')
        subset = filter_by_eval_set(sa, args.eval_set).reset_index(drop=True)
        logger.info(f'Subset (eval_set={args.eval_set}): {len(subset)} queries')

    models = {m: MODELS[m] for m in args.models if m in MODELS}
    if not models:
        logger.error(f'No valid models: {args.models}')
        sys.exit(1)
    logger.info(f'Models: {list(models.keys())}')

    # Server health
    import httpx
    async with httpx.AsyncClient(timeout=10) as client:
        for mn, mc in models.items():
            url = mc['base_url'].rsplit('/v1/', 1)[0] + '/v1/models'
            try:
                r = await client.get(url)
                if r.status_code != 200:
                    logger.warning(f'{mn} unhealthy HTTP {r.status_code}')
                else:
                    served = [m['id'] for m in r.json().get('data', [])]
                    logger.info(f'{mn} OK ({served})')
            except Exception as e:
                logger.warning(f'{mn} check failed: {e}')

    frame_cache = {}
    sems = {m: asyncio.Semaphore(CONCURRENCY_PER_MODEL) for m in models}

    # Run all models in parallel
    await asyncio.gather(*[
        eval_model(mn, mc, subset, frame_cache, sems[mn],
                   M_total=args.M, rep_offset=args.rep_offset,
                   out_filename_M=args.out_filename_M)
        for mn, mc in models.items()
    ])
    logger.info('All models done')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+',
                        default=['gemma-4-31b', 'qwen3.5-9b', 'qwen3-vl-30b'])
    parser.add_argument('--M', type=int, default=25,
                        help='Number of reps to run (this batch)')
    parser.add_argument('--rep-offset', type=int, default=0,
                        help='Starting repeat_id (offset). For appending, use prior M total.')
    parser.add_argument('--out-filename-M', type=int, default=None,
                        help='Force output filename to use this M number')
    from src._eval_set import add_eval_set_arg
    add_eval_set_arg(parser)
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == '__main__':
    main()
