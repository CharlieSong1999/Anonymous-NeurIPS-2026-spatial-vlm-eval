"""
Q2 metrics + Q1 ↔ Q2 comparison.

For each (model, condition) cell:
  - Mode accuracy (% queries with predicted letter == gt_letter)
  - Per-letter answer distribution (sanity check on letter shuffling)
  - Per-face accuracy (face-position bias check)
  - Parse rate (% records with parseable answer)

Headline comparisons:
  - A_random_diff_face vs C_no_target_cluster (distractor strategy effect)
  - cubemap_only vs frame_plus_cubemap (input format effect)

Q1 ↔ Q2 cross-format comparison (per-query, joined by sample_id):
  - For each query, was Q1 correct on yaw? Was Q2 correct on letter?
  - Joint contingency: P(Q1✓,Q2✓), P(Q1✓,Q2✗), P(Q1✗,Q2✓), P(Q1✗,Q2✗)
  - Tests "does the model know the answer when forced-choice but answer
    incorrectly when open-ended?" (the headline framing question)

Output:
  meta/testset/exp/q2_eval_001/results/aggregate.csv
  meta/testset/exp/q2_eval_001/results/per_letter.csv
  meta/testset/exp/q2_eval_001/results/per_face.csv
  meta/testset/exp/q2_eval_001/results/q1_q2_joint.csv
  meta/testset/exp/q2_eval_001/results.md

Run:
  conda run -n slam python3 -m src.q2_metrics
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

TESTSET = Path("/path/to/this/repo")
DATA_DIR = TESTSET / "data"
Q2_EVAL_DIR = TESTSET / "exp" / "q2_eval_001"
RUNS_DIR = Q2_EVAL_DIR / "runs"
RESULTS_DIR = Q2_EVAL_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Q1 reference run paths (M=25 sighted on the 300-query sets;
# may not directly join to the 2000-query test set — flag if missing)
Q1_RUNS_DIR = TESTSET / "exp" / "sampling_ablation_001" / "runs"

MODELS = ['qwen3.5-9b', 'gemma-4-31b', 'qwen3-vl-30b',
          'gemini-3-flash', 'gpt-5.4']
CONDITIONS = [
    'A_cubemap_only', 'A_frame_plus_cubemap',
    'C_cubemap_only', 'C_frame_plus_cubemap',
]


# ── Q2 record loader ─────────────────────────────────────────────────
def load_q2_records(condition: str, model: str) -> pd.DataFrame:
    paths = list(RUNS_DIR.glob(f'{condition}_{model}_M*.jsonl'))
    if not paths:
        return pd.DataFrame()
    rows = []
    for p in paths:
        with open(p) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                parsed = r.get('parsed') or {}
                ans = parsed.get('answer', None)
                rows.append({
                    'sample_id': r['sample_id'],
                    'repeat_id': r['repeat_id'],
                    'gt_letter': r['gt_letter'],
                    'gt_face': r['gt_face'],
                    'pred_letter': ans if isinstance(ans, str) and ans in 'ABCD' else None,
                    'parse_ok': isinstance(ans, str) and ans in 'ABCD',
                })
    return pd.DataFrame(rows)


# ── Per-cell aggregation ─────────────────────────────────────────────
def aggregate_cell(condition: str, model: str) -> Dict:
    df = load_q2_records(condition, model)
    if df.empty:
        return {'condition': condition, 'model': model, 'n_records': 0}
    n = len(df)
    valid = df[df.parse_ok]
    n_valid = len(valid)
    correct = (valid.pred_letter == valid.gt_letter).sum()
    acc = correct / n_valid if n_valid else float('nan')
    parse_rate = n_valid / n if n else 0.0

    # Per-letter answer distribution (across all valid responses)
    letter_dist = valid.pred_letter.value_counts(normalize=True).to_dict()
    for L in 'ABCD':
        letter_dist.setdefault(L, 0.0)

    # Per-letter correctness (when predicted X, was X the GT?)
    letter_precision = {}
    for L in 'ABCD':
        sub = valid[valid.pred_letter == L]
        letter_precision[L] = (sub.gt_letter == L).sum() / len(sub) if len(sub) else float('nan')

    # GT-letter recall (when GT was X, did we predict X?)
    letter_recall = {}
    for L in 'ABCD':
        sub = valid[valid.gt_letter == L]
        letter_recall[L] = (sub.pred_letter == L).sum() / len(sub) if len(sub) else float('nan')

    return {
        'condition': condition, 'model': model,
        'n_records': n, 'n_valid': n_valid,
        'parse_rate': parse_rate,
        'mode_acc': acc,
        **{f'pred_{L}_frac': letter_dist[L] for L in 'ABCD'},
        **{f'precision_{L}': letter_precision[L] for L in 'ABCD'},
        **{f'recall_{L}': letter_recall[L] for L in 'ABCD'},
    }


def aggregate_per_face(condition: str, model: str) -> List[Dict]:
    df = load_q2_records(condition, model)
    if df.empty:
        return []
    valid = df[df.parse_ok]
    rows = []
    for face, sub in valid.groupby('gt_face'):
        n = len(sub)
        correct = (sub.pred_letter == sub.gt_letter).sum()
        rows.append({
            'condition': condition, 'model': model, 'gt_face': face,
            'n': n, 'mode_acc': correct / n if n else float('nan'),
        })
    return rows


# ── Q1 ↔ Q2 join ─────────────────────────────────────────────────────
def load_q1_records(set_name: str, condition: str, model: str) -> pd.DataFrame:
    """Load Q1 records (M=25 typically). condition='sighted' for headline."""
    paths = list(Q1_RUNS_DIR.glob(f'{set_name}_{condition}_{model}_M*.jsonl'))
    if not paths:
        return pd.DataFrame()
    rows = []
    PITCH_TO_VBIN = {'UP': 2, 'LEVEL': 1, 'DOWN': 0}
    for p in paths:
        with open(p) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                parsed = r.get('parsed') or {}
                try:
                    yaw = int(parsed.get('yaw_bin_id'))
                except (TypeError, ValueError):
                    yaw = None
                v = PITCH_TO_VBIN.get(str(parsed.get('pitch', '')).upper())
                rows.append({
                    'sample_id': r['sample_id'],
                    'repeat_id': r.get('repeat_id', 0),
                    'q1_pred_yaw': yaw,
                    'q1_pred_v': v,
                    'q1_gt_yaw': r['gt_yaw'],
                    'q1_gt_v': r['gt_height'],
                    'q1_parse_ok': yaw in (1, 2, 3) and v in (0, 1, 2),
                })
    return pd.DataFrame(rows)


def q1_mode_per_query(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce M repeats to per-query mode prediction + correctness."""
    out = []
    for sid, sub in df.groupby('sample_id'):
        valid = sub[sub.q1_parse_ok]
        if valid.empty:
            continue
        # mode yaw prediction
        yaw_mode = int(valid.q1_pred_yaw.mode().iloc[0])
        v_mode = int(valid.q1_pred_v.mode().iloc[0])
        gt_yaw = int(valid.q1_gt_yaw.iloc[0])
        gt_v = int(valid.q1_gt_v.iloc[0])
        out.append({
            'sample_id': sid,
            'q1_yaw_mode': yaw_mode, 'q1_v_mode': v_mode,
            'q1_gt_yaw': gt_yaw, 'q1_gt_v': gt_v,
            'q1_yaw_correct': yaw_mode == gt_yaw,
            'q1_joint_correct': yaw_mode == gt_yaw and v_mode == gt_v,
        })
    return pd.DataFrame(out)


# ── Face → (yaw_bin, vert_bin) mapping for Q1↔Q2 cross-comparison ────
FACE_TO_YAW_VERT = {
    # face: (yaw_bin, vert_bin) — None means "any"
    'Right': (1, 1),    # Right face ≈ yaw=1, level
    'Back':  (2, 1),
    'Left':  (3, 1),
    'Up':    (None, 2),
    'Down':  (None, 0),
    'Front': (0, 1),    # Excluded yaw bin
}


def main():
    print('=== Q2 metrics ===')

    # 1. Per-cell aggregate
    rows = []
    for cond in CONDITIONS:
        for model in MODELS:
            row = aggregate_cell(cond, model)
            rows.append(row)
            if row.get('n_records', 0) > 0:
                print(f'  {cond:30s} {model:14s} '
                      f'acc={row["mode_acc"]:.3f} '
                      f'parse={row["parse_rate"]:.3f} '
                      f'n={row["n_records"]}')
            else:
                print(f'  {cond:30s} {model:14s}  (no records)')
    agg_df = pd.DataFrame(rows)
    agg_path = RESULTS_DIR / 'aggregate.csv'
    agg_df.to_csv(agg_path, index=False)
    print(f'\nSaved: {agg_path}')

    # 2. Per-face
    face_rows = []
    for cond in CONDITIONS:
        for model in MODELS:
            face_rows.extend(aggregate_per_face(cond, model))
    face_df = pd.DataFrame(face_rows)
    face_df.to_csv(RESULTS_DIR / 'per_face.csv', index=False)

    # 3. Q1 ↔ Q2 join (only on shared sample_ids)
    print('\n=== Q1 ↔ Q2 cross-format comparison ===')
    q1_q2_rows = []
    for model in MODELS:
        # Try loading Q1 sighted on Set A (300-query)
        q1_df = load_q1_records('A', 'sighted', model)
        if q1_df.empty:
            print(f'  {model}: no Q1 sighted records found at {Q1_RUNS_DIR}')
            continue
        q1_mode = q1_mode_per_query(q1_df)
        for cond in CONDITIONS:
            q2 = load_q2_records(cond, model)
            if q2.empty:
                continue
            q2_valid = q2[q2.parse_ok].copy()
            q2_valid['q2_correct'] = q2_valid.pred_letter == q2_valid.gt_letter
            # Reduce Q2 to per-query (M=1 currently, so trivially)
            q2_mode = q2_valid.groupby('sample_id').agg(
                q2_correct=('q2_correct', 'first'),
                q2_pred_letter=('pred_letter', 'first'),
                gt_letter=('gt_letter', 'first'),
                gt_face=('gt_face', 'first'),
            ).reset_index()
            joined = q1_mode.merge(q2_mode, on='sample_id', how='inner')
            n = len(joined)
            if n == 0:
                print(f'  {model}/{cond}: no shared sample_ids with Q1 sighted')
                continue
            both_ok = (joined.q1_yaw_correct & joined.q2_correct).sum()
            q1_only = (joined.q1_yaw_correct & ~joined.q2_correct).sum()
            q2_only = (~joined.q1_yaw_correct & joined.q2_correct).sum()
            both_no = (~joined.q1_yaw_correct & ~joined.q2_correct).sum()
            row = {
                'model': model, 'condition': cond,
                'n_shared': n,
                'q1_acc': joined.q1_yaw_correct.mean(),
                'q2_acc': joined.q2_correct.mean(),
                'p_both_correct': both_ok / n,
                'p_q1_only': q1_only / n,
                'p_q2_only': q2_only / n,
                'p_neither': both_no / n,
                # Headline disagreement: Q2 right when Q1 wrong → format-bias
                'q2_only_when_q1_wrong': q2_only / max(both_no + q2_only, 1),
            }
            q1_q2_rows.append(row)
            print(f'  {model}/{cond}: n={n}, q1_acc={row["q1_acc"]:.3f}, '
                  f'q2_acc={row["q2_acc"]:.3f}, both={row["p_both_correct"]:.3f}, '
                  f'q2_rescues_q1={row["q2_only_when_q1_wrong"]:.3f}')
    q1_q2_df = pd.DataFrame(q1_q2_rows)
    q1_q2_df.to_csv(RESULTS_DIR / 'q1_q2_joint.csv', index=False)

    # 4. Write the summary .md
    md = ['# Q2 Eval Results — first pass\n',
          'Generated by `src/q2_metrics.py`. M=1 per query.\n',
          '\n## Per-cell aggregate\n']
    md.append('| Condition | Model | n | parse_rate | Mode Acc |')
    md.append('|---|---|---:|---:|---:|')
    for _, r in agg_df.iterrows():
        if r.get('n_records', 0) == 0:
            continue
        md.append(f'| {r["condition"]} | {r["model"]} | {int(r["n_records"])} '
                  f'| {r["parse_rate"]:.3f} | **{r["mode_acc"]:.3f}** |')

    md.append('\n## Per-letter answer distribution (sanity: should ≈ 0.25 each)\n')
    md.append('| Condition | Model | Pred A | Pred B | Pred C | Pred D |')
    md.append('|---|---|---:|---:|---:|---:|')
    for _, r in agg_df.iterrows():
        if r.get('n_records', 0) == 0:
            continue
        md.append(f'| {r["condition"]} | {r["model"]} | '
                  f'{r["pred_A_frac"]:.3f} | {r["pred_B_frac"]:.3f} | '
                  f'{r["pred_C_frac"]:.3f} | {r["pred_D_frac"]:.3f} |')

    md.append('\n## Per-face accuracy\n')
    md.append('| Condition | Model | Face | n | Mode Acc |')
    md.append('|---|---|---|---:|---:|')
    for _, r in face_df.iterrows():
        md.append(f'| {r["condition"]} | {r["model"]} | {r["gt_face"]} | '
                  f'{int(r["n"])} | {r["mode_acc"]:.3f} |')

    md.append('\n## Strategy A vs C (same model, same input variant)\n')
    md.append('| Model | Variant | Acc(A) | Acc(C) | Δ (C − A) |')
    md.append('|---|---|---:|---:|---:|')
    for model in MODELS:
        for variant in ['cubemap_only', 'frame_plus_cubemap']:
            a_row = agg_df[(agg_df.model == model) & (agg_df.condition == f'A_{variant}')]
            c_row = agg_df[(agg_df.model == model) & (agg_df.condition == f'C_{variant}')]
            if a_row.empty or c_row.empty:
                continue
            a_acc = a_row.iloc[0]['mode_acc']
            c_acc = c_row.iloc[0]['mode_acc']
            md.append(f'| {model} | {variant} | {a_acc:.3f} | {c_acc:.3f} | '
                      f'{c_acc - a_acc:+.3f} |')

    md.append('\n## Input variant: cubemap_only vs frame_plus_cubemap (same model, same strategy)\n')
    md.append('| Model | Strategy | Acc(cube) | Acc(frame+cube) | Δ (f+c − c) |')
    md.append('|---|---|---:|---:|---:|')
    for model in MODELS:
        for strat in ['A', 'C']:
            c_row = agg_df[(agg_df.model == model) & (agg_df.condition == f'{strat}_cubemap_only')]
            f_row = agg_df[(agg_df.model == model) & (agg_df.condition == f'{strat}_frame_plus_cubemap')]
            if c_row.empty or f_row.empty:
                continue
            c_acc = c_row.iloc[0]['mode_acc']
            f_acc = f_row.iloc[0]['mode_acc']
            md.append(f'| {model} | {strat} | {c_acc:.3f} | {f_acc:.3f} | '
                      f'{f_acc - c_acc:+.3f} |')

    if not q1_q2_df.empty:
        md.append('\n## Q1 ↔ Q2 cross-format (per-query, joined by sample_id)\n')
        md.append('Q1 source: `sampling_ablation_001/runs/A_sighted_<model>_M*.jsonl` (300-query Set A)\n')
        md.append('| Model | Condition | n shared | Q1 acc | Q2 acc | both ✓ | Q1 only | Q2 only | neither | Q2 rescues Q1 |')
        md.append('|---|---|---:|---:|---:|---:|---:|---:|---:|---:|')
        for _, r in q1_q2_df.iterrows():
            md.append(f'| {r["model"]} | {r["condition"]} | {int(r["n_shared"])} | '
                      f'{r["q1_acc"]:.3f} | {r["q2_acc"]:.3f} | '
                      f'{r["p_both_correct"]:.3f} | {r["p_q1_only"]:.3f} | '
                      f'{r["p_q2_only"]:.3f} | {r["p_neither"]:.3f} | '
                      f'**{r["q2_only_when_q1_wrong"]:.3f}** |')
        md.append('\n*"Q2 rescues Q1" = P(Q2 correct | Q1 wrong). High value '
                  'means the model knows the answer under MCQ even when its '
                  'open-ended Q1 prediction is biased — the headline '
                  'format-vs-knowledge finding.*')

    md_path = RESULTS_DIR / 'results.md'
    md_path.write_text('\n'.join(md))
    print(f'\nWrote: {md_path}')


if __name__ == '__main__':
    main()
