"""
Dense M-ablation analysis: recompute metrics at every integer M from
1 to (M_max - 1) and produce metric-vs-M curves per (model, metric).

After running M=50 worth of responses per query, this script:
  1. Loads each model's M=50 JSONL
  2. For each M' ∈ {1, 2, ..., 49}, computes all metrics using only
     the first M' responses per query
  3. Saves a long-form CSV (one row per model × M × metric)
  4. Plots metric-vs-M curves for each metric, one panel per model

Outputs:
  exp/m_ablation_001/m_ablation_dense.csv
  exp/m_ablation_001/figs/dense_<metric>.png

Run:
  conda run -n slam python3 -m src.m_ablation_dense
  conda run -n slam python3 -m src.m_ablation_dense --m-max 50 --m-min 1
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from m_ablation import (
    load_run_responses, per_query_metrics_at_M, aggregate_filter,
    DEFAULT_RUNS_DIR, DATA_DIR, TESTSET,
)
from metrics_v2 import (
    build_pairwise_priors, filter_pairs, load_captions,
    N_YAW, N_VERT, N_JOINT, joint_index, smooth_dirichlet,
    jsd, kl, EPS, circular_rps,
)
from scipy.special import rel_entr

# Action40 train prior path
PRIOR_PATH = Path('/path/to/this/repo/track-f/'
                  'outputs/action40/prior_probs_4bin_height.parquet')

OUT_DIR = TESTSET / 'exp' / 'm_ablation_001'
FIG_DIR = OUT_DIR / 'figs'
FIG_DIR.mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs-dir', type=str,
                        default=str(OUT_DIR / 'runs_extended'))
    parser.add_argument('--captions', type=str,
                        default='captions_v2_extended.jsonl')
    parser.add_argument('--set-parquet', type=str,
                        default='setA_extended_subset500.parquet')
    parser.add_argument('--m-min', type=int, default=1)
    parser.add_argument('--m-max', type=int, default=50,
                        help='Max M to evaluate (inclusive)')
    parser.add_argument('--m-step', type=int, default=1)
    parser.add_argument('--file-pattern', type=str, default='*_M50.jsonl',
                        help='Glob pattern for M=max JSONLs (e.g., *_M50.jsonl)')
    from src._eval_set import add_eval_set_arg, filter_by_eval_set
    add_eval_set_arg(parser)
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)

    # Load set
    set_df = pd.read_parquet(DATA_DIR / args.set_parquet)
    if args.eval_set != 'full':
        set_df = filter_by_eval_set(set_df, args.eval_set)
    print(f'Set: {len(set_df)} queries (eval_set={args.eval_set})')

    # Captions
    captions = {}
    cap_path = DATA_DIR / args.captions
    if cap_path.exists():
        with open(cap_path) as f:
            for line in f:
                r = json.loads(line)
                captions[r['frame_key']] = r['caption']
    # also merge default
    default = DATA_DIR / 'captions_v2.jsonl'
    if default.exists():
        with open(default) as f:
            for line in f:
                r = json.loads(line)
                if r['frame_key'] not in captions:
                    captions[r['frame_key']] = r['caption']
    print(f'Loaded {len(captions)} captions')

    # Pairwise priors and filters
    pw_y, pw_v, pw_j, cnt = build_pairwise_priors(set_df, captions)
    pairs_B = filter_pairs(cnt, n_min=5, filter_level='B')
    pairs_C = filter_pairs(cnt, n_min=5, filter_level='C')
    print(f'Filter B pairs: {len(pairs_B)}, Filter C pairs: {len(pairs_C)}')

    # Find files
    files = sorted(runs_dir.glob(args.file_pattern))
    print(f'\nFound {len(files)} files matching {args.file_pattern}')

    rows = []
    M_values = list(range(args.m_min, args.m_max + 1, args.m_step))
    print(f'Computing metrics at M = {args.m_min}..{args.m_max} step {args.m_step} ({len(M_values)} values)')

    for fp in files:
        # Filename: {model}_{cond}_M{N}.jsonl
        stem = fp.stem
        parts = stem.split('_')
        model = '_'.join(parts[:-2])
        cond = parts[-2]
        by_sid = load_run_responses(fp)
        if not by_sid:
            print(f'  {fp.name}: empty')
            continue
        max_resps = max(len(v) for v in by_sid.values())
        avg_resps = np.mean([len(v) for v in by_sid.values()])
        print(f'  {fp.name}: {len(by_sid)} sids, max_resps={max_resps}, avg={avg_resps:.1f}')

        for M in M_values:
            per_q = per_query_metrics_at_M(by_sid, set_df, M)
            if not per_q:
                continue
            # Per-query aggregates
            row = {
                'model': model, 'condition': cond, 'M': M,
                'n_queries': len(per_q),
                'm_eff_mean': float(np.mean([q['m_eff'] for q in per_q.values()])),
                'NLL_y': float(np.mean([q['nll_y'] for q in per_q.values()])),
                'NLL_v': float(np.mean([q['nll_v'] for q in per_q.values()])),
                'NLL_j': float(np.mean([q['nll_j'] for q in per_q.values()])),
                'ModeY%': 100 * float(np.mean([q['mode_y_exact'] for q in per_q.values()])),
                'ModeV%': 100 * float(np.mean([q['mode_v_exact'] for q in per_q.values()])),
                'ModeJ%': 100 * float(np.mean([q['mode_j_exact'] for q in per_q.values()])),
            }
            agg_B = aggregate_filter(per_q, set_df, captions, pw_y, pw_v, pw_j, pairs_B)
            agg_C = aggregate_filter(per_q, set_df, captions, pw_y, pw_v, pw_j, pairs_C)
            for k, v in agg_B.items():
                row[f'{k}_B'] = v
            for k, v in agg_C.items():
                row[f'{k}_C'] = v
            rows.append(row)
        print(f'    Done {len(M_values)} M values for {model}')

    df = pd.DataFrame(rows)
    out_path = OUT_DIR / 'm_ablation_dense.csv'
    df.to_csv(out_path, index=False)
    print(f'\nSaved: {out_path} ({len(df)} rows)')

    # Compute baselines (M-independent)
    print('\nComputing uniform + train_prior baselines...')
    baselines = compute_baselines(set_df, captions, pw_y, pw_v, pw_j,
                                  pairs_B, pairs_C)
    bl_path = OUT_DIR / 'm_ablation_baselines.csv'
    pd.DataFrame([baselines]).to_csv(bl_path, index=False)
    print(f'Saved: {bl_path}')
    print('Baselines:')
    for k, v in baselines.items():
        print(f'  {k:30s} = {v:.4f}')

    # Plots
    print('\nGenerating plots...')
    plot_metrics(df, baselines)


def compute_baselines(set_df, captions, pw_y, pw_v, pw_j, pairs_B, pairs_C):
    """Return dict of baseline values for each metric (M-independent)."""
    # Load train prior
    prior = pd.read_parquet(PRIOR_PATH)
    GLOBAL_YAW = np.array(prior[prior.fallback].iloc[0].p_yaw)
    GLOBAL_VERT = np.array(prior[prior.fallback].iloc[0].p_vert)
    GLOBAL_JOINT = np.outer(GLOBAL_YAW, GLOBAL_VERT).flatten()
    GLOBAL_JOINT = GLOBAL_JOINT / GLOBAL_JOINT.sum()
    prior_y = {r.canonical_label: np.array(r.p_yaw) for _, r in prior.iterrows()}
    prior_j = {r.canonical_label: np.array(r.p_joint) for _, r in prior.iterrows()}

    def get_prior(label):
        py = prior_y.get(label, GLOBAL_YAW)
        pj = prior_j.get(label, GLOBAL_JOINT)
        return py, pj

    uy = np.full(N_YAW, 1/N_YAW)
    uj = np.full(N_JOINT, 1/N_JOINT)
    uv = np.full(N_VERT, 1/N_VERT)

    out = {}

    # Per-query metrics: uniform & train_prior
    for bname, get_dist_fn in [
        ('uniform', lambda r: (uy, uv, uj)),
        ('train_prior', lambda r: (
            get_prior(r['canonical_label'])[0],
            np.array([sum(get_prior(r['canonical_label'])[1][v::N_VERT][:N_YAW])
                     for v in range(N_VERT)]),
            get_prior(r['canonical_label'])[1])),
    ]:
        nll_y_arr, nll_v_arr, nll_j_arr = [], [], []
        my_e, mv_e, mj_e = [], [], []
        for _, r in set_df.iterrows():
            gy, gv = int(r['yaw_bin_4']), int(r['height_bin'])
            gj = (gy - 1) * N_VERT + gv
            py, pv, pj = get_dist_fn(r)
            pv = pv / pv.sum() if pv.sum() > 0 else uv
            nll_y_arr.append(-np.log(max(py[gy - 1], EPS)))
            nll_v_arr.append(-np.log(max(pv[gv], EPS)))
            nll_j_arr.append(-np.log(max(pj[gj], EPS)))
            my = int(np.argmax(py)) + 1
            mv = int(np.argmax(pv))
            mj = int(np.argmax(pj))
            my_e.append(int(my == gy))
            mv_e.append(int(mv == gv))
            mj_e.append(int(mj == gj))
        out[f'{bname}_NLL_y'] = float(np.mean(nll_y_arr))
        out[f'{bname}_NLL_v'] = float(np.mean(nll_v_arr))
        out[f'{bname}_NLL_j'] = float(np.mean(nll_j_arr))
        out[f'{bname}_ModeY%'] = 100 * float(np.mean(my_e))
        out[f'{bname}_ModeV%'] = 100 * float(np.mean(mv_e))
        out[f'{bname}_ModeJ%'] = 100 * float(np.mean(mj_e))

    # Filter B/C divergences for uniform & train_prior
    for fl, pairs in [('B', pairs_B), ('C', pairs_C)]:
        for bname in ['uniform', 'train_prior']:
            jsds_y, kls_pwm_y, kls_mpw_y = [], [], []
            jsds_j, kls_pwm_j, kls_mpw_j = [], [], []
            for tgt, anc in pairs:
                ppy, ppj = pw_y[(tgt, anc)], pw_j[(tgt, anc)]
                if bname == 'uniform':
                    m_y, m_j = uy, uj
                else:
                    m_y, m_j = get_prior(tgt)
                jsds_y.append(jsd(m_y, ppy))
                kls_pwm_y.append(kl(ppy, m_y))
                kls_mpw_y.append(kl(m_y, ppy))
                jsds_j.append(jsd(m_j, ppj))
                kls_pwm_j.append(kl(ppj, m_j))
                kls_mpw_j.append(kl(m_j, ppj))
            out[f'{bname}_mean_jsd_y_{fl}'] = float(np.mean(jsds_y)) if jsds_y else float('nan')
            out[f'{bname}_mean_kl_pwm_y_{fl}'] = float(np.mean(kls_pwm_y)) if kls_pwm_y else float('nan')
            out[f'{bname}_mean_kl_mpw_y_{fl}'] = float(np.mean(kls_mpw_y)) if kls_mpw_y else float('nan')
            out[f'{bname}_mean_jsd_j_{fl}'] = float(np.mean(jsds_j)) if jsds_j else float('nan')
            out[f'{bname}_mean_kl_pwm_j_{fl}'] = float(np.mean(kls_pwm_j)) if kls_pwm_j else float('nan')
            out[f'{bname}_mean_kl_mpw_j_{fl}'] = float(np.mean(kls_mpw_j)) if kls_mpw_j else float('nan')

    return out


def plot_metrics(df, baselines):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    metrics = [
        ('mean_jsd_y_B', 'JSD_y (Filter B)'),
        ('mean_kl_pwm_y_B', 'KL(P_pw||P_m)_y (Filter B)'),
        ('mean_kl_mpw_y_B', 'KL(P_m||P_pw)_y (Filter B)'),
        ('mean_jsd_j_B', 'JSD_j (Filter B)'),
        ('mean_kl_pwm_j_B', 'KL(P_pw||P_m)_j (Filter B)'),
        ('mean_kl_mpw_j_B', 'KL(P_m||P_pw)_j (Filter B)'),
        ('mean_jsd_y_C', 'JSD_y (Filter C)'),
        ('mean_kl_pwm_y_C', 'KL(P_pw||P_m)_y (Filter C)'),
        ('mean_kl_mpw_y_C', 'KL(P_m||P_pw)_y (Filter C)'),
        ('mean_jsd_j_C', 'JSD_j (Filter C)'),
        ('mean_kl_pwm_j_C', 'KL(P_pw||P_m)_j (Filter C)'),
        ('mean_kl_mpw_j_C', 'KL(P_m||P_pw)_j (Filter C)'),
        ('NLL_y', 'NLL yaw (per-query)'),
        ('NLL_v', 'NLL vert (per-query)'),
        ('NLL_j', 'NLL joint (per-query)'),
        ('ModeJ%', 'Mode accuracy joint %'),
        ('ModeY%', 'Mode accuracy yaw %'),
        ('ModeV%', 'Mode accuracy vert %'),
    ]

    models = sorted(df.model.unique())
    colors = {'gemma-4-31b': 'tab:red', 'qwen3.5-9b': 'tab:blue',
              'qwen3-vl-30b': 'tab:green'}
    bl_colors = {'uniform': 'gray', 'train_prior': 'black'}
    bl_styles = {'uniform': '--', 'train_prior': ':'}

    def _draw_baselines(ax, col):
        for bname in ['uniform', 'train_prior']:
            key = f'{bname}_{col}'
            if key in baselines and not np.isnan(baselines[key]):
                ax.axhline(baselines[key],
                           color=bl_colors[bname], linestyle=bl_styles[bname],
                           linewidth=1.4, label=bname, alpha=0.9)

    # Combined figure: all metrics in one grid
    n = len(metrics)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.5 * rows))
    axes = axes.flatten()
    for i, (col, label) in enumerate(metrics):
        ax = axes[i]
        for m in models:
            sub = df[df.model == m].sort_values('M')
            if col not in sub.columns:
                continue
            ax.plot(sub.M, sub[col], 'o-', color=colors.get(m, None),
                    label=m, markersize=3)
        _draw_baselines(ax, col)
        ax.set_xlabel('M (samples per query)')
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'dense_all_metrics.png', dpi=120, bbox_inches='tight')
    fig.savefig(FIG_DIR / 'dense_all_metrics.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {FIG_DIR / "dense_all_metrics.png"}')

    # Individual metric figures (cleaner)
    for col, label in metrics:
        if col not in df.columns:
            continue
        fig, ax = plt.subplots(figsize=(7, 5))
        for m in models:
            sub = df[df.model == m].sort_values('M')
            ax.plot(sub.M, sub[col], 'o-', color=colors.get(m, None),
                    label=m, markersize=4)
        _draw_baselines(ax, col)
        ax.set_xlabel('M (samples per query)')
        ax.set_ylabel(label)
        ax.set_title(f'{label} vs M (extended Set A, baseline_sighted)')
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        safe_col = col.replace('/', '_')
        fig.savefig(FIG_DIR / f'dense_{safe_col}.png', dpi=120, bbox_inches='tight')
        plt.close(fig)
    print(f'  Saved {len(metrics)} individual plots')


if __name__ == '__main__':
    main()
