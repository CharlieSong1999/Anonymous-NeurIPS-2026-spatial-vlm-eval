"""
M-ablation: how stable are pairwise/distributional metrics as M decreases?

Approach: take the existing M=25 jsonl outputs and recompute metrics by
including only the first M' ∈ {5, 10, 15, 20, 25} responses per query.
This is post-hoc (no extra VLM calls) — just a re-aggregation. Equivalent
to running M=M' if responses are i.i.d., which they are at temperature=1.

Outputs per (model, condition, M', set):
  - Mean JSD_y, KLpwm_y, KLmpw_y, JSD_j, KLpwm_j, KLmpw_j on Filter B and Filter C
  - Mean NLL_y, NLL_v, NLL_j
  - Mean mode accuracy

Compares to M=25 baseline; flags if metrics drift > tolerance.

Run:
  conda run -n slam python3 -m src.m_ablation
  conda run -n slam python3 -m src.m_ablation --runs-dir <other-dir>
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import rel_entr

sys.path.insert(0, str(Path(__file__).parent))
from metrics_v2 import (
    build_pairwise_priors, filter_pairs, jsd, kl, load_captions,
    smooth_dirichlet, ALPHA, N_YAW, N_VERT, N_JOINT, joint_index,
    PITCH_TO_VBIN, ALLOWED_YAW, EPS, empirical_distribution,
)

TESTSET = Path("/path/to/this/repo")
DATA_DIR = TESTSET / "data"
DEFAULT_RUNS_DIR = TESTSET / "exp" / "sampling_ablation_001" / "runs"

M_VALUES = [5, 10, 15, 20, 25]


def load_run_responses(path):
    """Return dict sample_id -> list of (yaw_bin, vert_bin) in original order."""
    by_sid = defaultdict(list)
    if not path.exists():
        return by_sid
    with open(path) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            sid = rec['sample_id']
            parsed = rec.get('parsed') or {}
            try:
                y = int(parsed.get('yaw_bin_id'))
            except (TypeError, ValueError):
                y = None
            v = PITCH_TO_VBIN.get(str(parsed.get('pitch', '')).upper()) if parsed.get('pitch') else None
            if y in ALLOWED_YAW and v is not None:
                # rep_id might or might not be present
                rep = rec.get('repeat_id', len(by_sid[sid]))
                by_sid[sid].append((rep, y, v))
    # Sort by repeat_id and keep just (y, v)
    out = {}
    for sid, samples in by_sid.items():
        samples.sort(key=lambda x: x[0])
        out[sid] = [(y, v) for _, y, v in samples]
    return out


def per_query_metrics_at_M(by_sid, set_df, M_target):
    """Compute per-query metrics using only the first M_target responses.

    If a query has < M_target valid responses, use all available.
    """
    sid_to_row = {r['sample_id']: r for _, r in set_df.iterrows()}
    out = {}
    for sid, all_samples in by_sid.items():
        samples = all_samples[:M_target]
        if not samples:
            continue
        row = sid_to_row.get(sid)
        if row is None:
            continue
        gt_y = int(row['yaw_bin_4'])
        gt_v = int(row['height_bin'])
        gt_j = (gt_y - 1) * N_VERT + gt_v
        py, pv, pj = empirical_distribution(samples)
        nll_y = -np.log(max(py[gt_y - 1], EPS))
        nll_v = -np.log(max(pv[gt_v], EPS))
        nll_j = -np.log(max(pj[gt_j], EPS))
        mode_y = int(np.argmax(py)) + 1
        mode_v = int(np.argmax(pv))
        mode_j = int(np.argmax(pj))
        out[sid] = {
            'p_yaw': py, 'p_vert': pv, 'p_joint': pj,
            'nll_y': nll_y, 'nll_v': nll_v, 'nll_j': nll_j,
            'mode_y_exact': int(mode_y == gt_y),
            'mode_v_exact': int(mode_v == gt_v),
            'mode_j_exact': int(mode_j == gt_j),
            'm_eff': len(samples),
        }
    return out


def aggregate_filter(per_q, set_df, captions, pw_y, pw_v, pw_j, pairs):
    """Aggregate per-pair JSD / KL / asym for given pairs list."""
    sid_to_anchors = {}
    for _, row in set_df.iterrows():
        from metrics_v2 import detect_anchors_in_caption
        ds, vid, fidx = row['dataset'], row['video_id'], int(row['frame_index'])
        fk = f'{ds}:{vid}:{fidx}'
        sid_to_anchors[row['sample_id']] = detect_anchors_in_caption(
            captions.get(fk, ''))
    sid_to_label = dict(zip(set_df.sample_id, set_df.canonical_label))

    jsds_y, kls_pwm_y, kls_mpw_y = [], [], []
    jsds_j, kls_pwm_j, kls_mpw_j = [], [], []
    for tgt, anc in pairs:
        m_yaws, m_verts, m_joints = [], [], []
        for sid, q in per_q.items():
            if sid_to_label.get(sid) != tgt:
                continue
            if anc not in sid_to_anchors.get(sid, set()):
                continue
            m_yaws.append(q['p_yaw'])
            m_verts.append(q['p_vert'])
            m_joints.append(q['p_joint'])
        if not m_yaws:
            continue
        m_y = np.mean(m_yaws, axis=0); m_y /= m_y.sum()
        m_j = np.mean(m_joints, axis=0); m_j /= m_j.sum()
        ppy = pw_y.get((tgt, anc)); ppj = pw_j.get((tgt, anc))
        if ppy is None:
            continue
        jsds_y.append(jsd(m_y, ppy)); kls_pwm_y.append(kl(ppy, m_y)); kls_mpw_y.append(kl(m_y, ppy))
        jsds_j.append(jsd(m_j, ppj)); kls_pwm_j.append(kl(ppj, m_j)); kls_mpw_j.append(kl(m_j, ppj))
    return {
        'n_pairs_used': len(jsds_y),
        'mean_jsd_y': float(np.mean(jsds_y)) if jsds_y else float('nan'),
        'mean_kl_pwm_y': float(np.mean(kls_pwm_y)) if kls_pwm_y else float('nan'),
        'mean_kl_mpw_y': float(np.mean(kls_mpw_y)) if kls_mpw_y else float('nan'),
        'mean_jsd_j': float(np.mean(jsds_j)) if jsds_j else float('nan'),
        'mean_kl_pwm_j': float(np.mean(kls_pwm_j)) if kls_pwm_j else float('nan'),
        'mean_kl_mpw_j': float(np.mean(kls_mpw_j)) if kls_mpw_j else float('nan'),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs-dir', type=str, default=str(DEFAULT_RUNS_DIR))
    parser.add_argument('--set', type=str, default='A',
                        help='Which set: A, B, or extended (or arbitrary parquet path)')
    parser.add_argument('--out-dir', type=str,
                        default=str(TESTSET / 'exp' / 'm_ablation_001'))
    parser.add_argument('--captions', type=str, default='captions_v2.jsonl',
                        help='Captions file under data/ (default captions_v2.jsonl)')
    from src._eval_set import add_eval_set_arg, filter_by_eval_set
    add_eval_set_arg(parser)
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load set
    if args.set == 'A':
        set_df = pd.read_parquet(DATA_DIR / 'setA_300_filtered.parquet')
    elif args.set == 'B':
        set_df = pd.read_parquet(DATA_DIR / 'setB_300_filtered.parquet')
    elif args.set == 'extended':
        set_df = pd.read_parquet(DATA_DIR / 'setA_extended_subset500.parquet')
    else:
        set_df = pd.read_parquet(DATA_DIR / args.set)

    if args.eval_set != 'full':
        set_df = filter_by_eval_set(set_df, args.eval_set)

    # Load captions (allow custom path)
    cap_path = DATA_DIR / args.captions
    captions = {}
    if cap_path.exists():
        with open(cap_path) as f:
            for line in f:
                r = json.loads(line)
                captions[r['frame_key']] = r['caption']
    # Also merge default captions if separate file specified
    if args.captions != 'captions_v2.jsonl':
        default_cap = DATA_DIR / 'captions_v2.jsonl'
        if default_cap.exists():
            with open(default_cap) as f:
                for line in f:
                    r = json.loads(line)
                    if r['frame_key'] not in captions:
                        captions[r['frame_key']] = r['caption']
    print(f'Loaded {len(captions)} captions')
    print(f'Set {args.set}: {len(set_df)} queries')

    # Build pairwise priors and filters
    pw_y, pw_v, pw_j, cnt = build_pairwise_priors(set_df, captions)
    pairs_B = filter_pairs(cnt, n_min=5, filter_level='B')
    pairs_C = filter_pairs(cnt, n_min=5, filter_level='C')
    print(f'Filter B pairs: {len(pairs_B)}, Filter C pairs: {len(pairs_C)}')

    # Iterate JSONL files in runs_dir.
    # Two filename conventions supported:
    #   {set}_{cond}_{model}_M{N}.jsonl   (sampling_ablation_001/runs/)
    #   {model}_{cond}_M{N}.jsonl         (m_ablation_001/runs_extended/)
    rows = []
    if args.set in ('A', 'B'):
        pattern = f'{args.set}_*.jsonl'
    else:
        pattern = '*.jsonl'
    files = sorted(runs_dir.glob(pattern))
    print(f'\nFound {len(files)} JSONL files in {runs_dir}')

    for fp in files:
        stem = fp.stem
        parts = stem.split('_')
        # Detect convention
        if parts[0] in ('A', 'B'):
            # {set}_{cond}_{model}_M{N}
            cond = parts[1]
            model = '_'.join(parts[2:-1])
        else:
            # {model}_{cond}_M{N}
            model = '_'.join(parts[:-2])
            cond = parts[-2]
        by_sid = load_run_responses(fp)
        if not by_sid:
            print(f'  {fp.name}: empty')
            continue

        for M in M_VALUES:
            per_q = per_query_metrics_at_M(by_sid, set_df, M)
            if not per_q:
                continue
            # Per-query aggregates
            nll_y_arr = [q['nll_y'] for q in per_q.values()]
            nll_v_arr = [q['nll_v'] for q in per_q.values()]
            nll_j_arr = [q['nll_j'] for q in per_q.values()]
            mode_y_arr = [q['mode_y_exact'] for q in per_q.values()]
            mode_v_arr = [q['mode_v_exact'] for q in per_q.values()]
            mode_j_arr = [q['mode_j_exact'] for q in per_q.values()]
            m_eff_arr = [q['m_eff'] for q in per_q.values()]

            # Filter B & C
            agg_B = aggregate_filter(per_q, set_df, captions, pw_y, pw_v, pw_j, pairs_B)
            agg_C = aggregate_filter(per_q, set_df, captions, pw_y, pw_v, pw_j, pairs_C)

            row = {
                'model': model, 'condition': cond, 'set': args.set,
                'M': M,
                'n_queries': len(per_q),
                'm_eff_mean': float(np.mean(m_eff_arr)),
                'NLL_y': float(np.mean(nll_y_arr)),
                'NLL_v': float(np.mean(nll_v_arr)),
                'NLL_j': float(np.mean(nll_j_arr)),
                'ModeY%': 100 * float(np.mean(mode_y_arr)),
                'ModeV%': 100 * float(np.mean(mode_v_arr)),
                'ModeJ%': 100 * float(np.mean(mode_j_arr)),
            }
            for k, v in agg_B.items():
                row[f'{k}_B'] = v
            for k, v in agg_C.items():
                row[f'{k}_C'] = v
            rows.append(row)
        print(f'  {fp.name}: done {len(M_VALUES)} M values')

    df = pd.DataFrame(rows)
    out_path = out_dir / f'm_ablation_set{args.set}.csv'
    df.to_csv(out_path, index=False)
    print(f'\nSaved: {out_path}')
    print(f'Rows: {len(df)}')

    # Also dump a focused summary: each (model, condition) at each M, comparing to M=25
    print('\n=== Drift table (Δ vs M=25, Filter B JSD_y) ===')
    print(f'{"model":>13} {"cond":>8} {"set":>3} {"M":>3} {"NLL_j":>5} {"ModeJ%":>6} {"JSDy_B":>6} {"ΔJSDy_B":>8}')
    for (mdl, cnd), sub in df.groupby(['model', 'condition']):
        sub = sub.sort_values('M')
        ref = sub[sub.M == 25].iloc[0] if (sub.M == 25).any() else None
        for _, r in sub.iterrows():
            d = r['mean_jsd_y_B'] - ref['mean_jsd_y_B'] if ref is not None else float('nan')
            print(f'{mdl:>13} {cnd:>8} {r["set"]:>3} {int(r["M"]):>3} {r["NLL_j"]:>5.2f} {r["ModeJ%"]:>6.1f} {r["mean_jsd_y_B"]:>6.3f} {d:>+8.4f}')


if __name__ == '__main__':
    main()
