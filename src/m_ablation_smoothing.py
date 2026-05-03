"""
Smoothing-config sweep on the dense M-ablation.

Re-runs `m_ablation_dense` analysis with three different smoothing schemes
to test whether the monotonic climb of JSD/KL/NLL with M is driven by the
choice of Dirichlet prior strength rather than by the model itself.

Configs:
  - alpha_jeffrey_05: Jeffrey's prior, alpha=0.5 (current baseline). Strong
    Bayesian smoothing — pulls every distribution toward uniform at low M.
  - alpha_tiny_1e-3:  alpha=1e-3. Numerical floor only — distribution is
    essentially MLE / plug-in.
  - mle_miller_madow: alpha=1e-3 for the distribution itself, plus the
    Miller-Madow first-order bias correction subtracted from the plug-in
    KL/JSD estimates. Targets the inherent finite-sample bias of plug-in
    distributional estimators.

Per-pair distributions are built by pooling raw responses across queries
(rather than averaging smoothed per-query distributions), so the smoothing
choice is applied once per pair rather than per query then averaged.
This is the cleaner statistical estimator and what MM correction assumes.

Outputs (per config):
  exp/m_ablation_001/m_ablation_dense_<config>.csv
  exp/m_ablation_001/m_ablation_baselines_<config>.csv
  exp/m_ablation_001/figs/<config>/dense_<metric>.png
  exp/m_ablation_001/figs/<config>/dense_all_metrics.{png,pdf}

Run:
  conda run -n slam python3 -m src.m_ablation_smoothing --config alpha_jeffrey_05
  conda run -n slam python3 -m src.m_ablation_smoothing --config alpha_tiny_1e-3
  conda run -n slam python3 -m src.m_ablation_smoothing --config mle_miller_madow
  conda run -n slam python3 -m src.m_ablation_smoothing --config all
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
from m_ablation import load_run_responses, DATA_DIR, TESTSET
from metrics_v2 import (
    detect_anchors_in_caption, filter_pairs,
    N_YAW, N_VERT, N_JOINT, joint_index,
    PITCH_TO_VBIN, ALLOWED_YAW,
)

PRIOR_PATH = Path('/path/to/this/repo/track-f/'
                  'outputs/action40/prior_probs_4bin_height.parquet')

OUT_DIR = TESTSET / 'exp' / 'm_ablation_001'
FIG_BASE = OUT_DIR / 'figs'
NLL_FLOOR_TINY = 1e-3  # used as the NLL log-floor for tiny-alpha and MM configs

CONFIGS = {
    'alpha_jeffrey_05': dict(
        alpha=0.5,
        miller_madow=False,
        nll_floor=0.5,  # same as alpha for the smoothing
        label='alpha=0.5 (Jeffrey)',
    ),
    'alpha_tiny_1e-3': dict(
        alpha=1e-3,
        miller_madow=False,
        nll_floor=1e-3,
        label='alpha=1e-3 (numerical floor)',
    ),
    'mle_miller_madow': dict(
        alpha=1e-3,
        miller_madow=True,
        nll_floor=1e-3,
        label='MLE + Miller-Madow correction',
    ),
}


# ── Distribution & divergence under a given smoothing config ────────────
def smooth(counts, alpha):
    counts = np.asarray(counts, dtype=float)
    return (counts + alpha) / (counts.sum() + len(counts) * alpha)


def kl_plugin(p, q):
    p = np.clip(p, 1e-12, 1); p /= p.sum()
    q = np.clip(q, 1e-12, 1); q /= q.sum()
    return float(np.sum(rel_entr(p, q)))


def jsd_plugin(p, q):
    p = np.clip(p, 1e-12, 1); p /= p.sum()
    q = np.clip(q, 1e-12, 1); q /= q.sum()
    m = 0.5 * (p + q)
    return 0.5 * float(np.sum(rel_entr(p, m))) + \
           0.5 * float(np.sum(rel_entr(q, m)))


def mm_kl_correction(B, N_P):
    """Miller-Madow bias correction for KL(P_hat || Q): subtract (B-1)/(2 N_P)."""
    if N_P <= 0:
        return 0.0
    return (B - 1) / (2.0 * N_P)


def mm_jsd_correction(B, N_P, N_Q):
    """Miller-Madow bias correction for JSD(P_hat, Q_hat).

    Derivation: JSD = H(M) - 0.5 H(P) - 0.5 H(Q). Plug-in entropy bias is
    -(B-1)/(2N) per estimate. Treating M as having effective sample size
    (N_P + N_Q)/2 (single pooled estimate):
        E[JSD_plug] - JSD = -(B-1)/(2 N_M) + 0.5*(B-1)/(2 N_P) + 0.5*(B-1)/(2 N_Q)
                          = (B-1)/2 * [0.5/N_P + 0.5/N_Q - 1/N_M]
    Subtract this bias from the plug-in JSD.
    """
    if N_P <= 0 or N_Q <= 0:
        return 0.0
    N_M = 0.5 * (N_P + N_Q)
    return (B - 1) / 2.0 * (0.5 / N_P + 0.5 / N_Q - 1.0 / N_M)


def kl_with_correction(p, q, N_P, B, miller_madow):
    val = kl_plugin(p, q)
    if miller_madow:
        val -= mm_kl_correction(B, N_P)
    return max(0.0, val)


def jsd_with_correction(p, q, N_P, N_Q, B, miller_madow):
    val = jsd_plugin(p, q)
    if miller_madow:
        val -= mm_jsd_correction(B, N_P, N_Q)
    return max(0.0, val)


# ── Per-query (NLL, mode acc) — depends only on per-query alpha ────────
def per_query_metrics(by_sid, set_df, M_target, alpha, nll_floor):
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
        # Per-query distribution (smoothed with alpha)
        yaw_counts = np.zeros(N_YAW)
        vert_counts = np.zeros(N_VERT)
        joint_counts = np.zeros(N_JOINT)
        for y, v in samples:
            yaw_counts[y - 1] += 1
            vert_counts[v] += 1
            joint_counts[joint_index(y, v)] += 1
        py = smooth(yaw_counts, alpha)
        pv = smooth(vert_counts, alpha)
        pj = smooth(joint_counts, alpha)
        # NLL with floor to avoid log(0)
        nll_y = -np.log(max(py[gt_y - 1], nll_floor / (len(samples) + N_YAW * nll_floor)))
        nll_v = -np.log(max(pv[gt_v], nll_floor / (len(samples) + N_VERT * nll_floor)))
        nll_j = -np.log(max(pj[gt_j], nll_floor / (len(samples) + N_JOINT * nll_floor)))
        out[sid] = {
            'samples': samples,
            'nll_y': nll_y, 'nll_v': nll_v, 'nll_j': nll_j,
            'mode_y_exact': int(int(np.argmax(py)) + 1 == gt_y),
            'mode_v_exact': int(int(np.argmax(pv)) == gt_v),
            'mode_j_exact': int(int(np.argmax(pj)) == gt_j),
            'm_eff': len(samples),
        }
    return out


# ── Pair-wise priors built once (per smoothing config) ─────────────────
def build_pairwise_priors_pooled(set_df, captions, alpha):
    """Return dicts of P_pw(y), P_pw(v), P_pw(j), and sample counts.

    P_pw is built by pooling GT bins across queries in each (target, anchor)
    pair. Smoothed with given alpha at the pair level (one smoothing per
    pair rather than per query).
    """
    pw_y_counts, pw_v_counts, pw_j_counts = {}, {}, {}
    for _, row in set_df.iterrows():
        ds = row['dataset']; vid = row['video_id']
        fidx = int(row['frame_index'])
        fk = f'{ds}:{vid}:{fidx}'
        cap = captions.get(fk, '')
        anchors = detect_anchors_in_caption(cap)
        if not anchors:
            continue
        target = row['canonical_label']
        y_bin = int(row['yaw_bin_4'])
        v_bin = int(row['height_bin'])
        for a in anchors:
            key = (target, a)
            pw_y_counts.setdefault(key, np.zeros(N_YAW))[y_bin - 1] += 1
            pw_v_counts.setdefault(key, np.zeros(N_VERT))[v_bin] += 1
            pw_j_counts.setdefault(key, np.zeros(N_JOINT))[joint_index(y_bin, v_bin)] += 1
    pw_y = {k: smooth(v, alpha) for k, v in pw_y_counts.items()}
    pw_v = {k: smooth(v, alpha) for k, v in pw_v_counts.items()}
    pw_j = {k: smooth(v, alpha) for k, v in pw_j_counts.items()}
    counts = {k: int(v.sum()) for k, v in pw_y_counts.items()}
    return pw_y, pw_v, pw_j, counts


# ── Per-pair model distribution from pooled raw responses ──────────────
def build_pair_model_dist(per_q, set_df, captions, pair_target, pair_anchor, alpha):
    """Pool all raw responses for queries matching this pair → smooth once."""
    sid_to_anchors = {}
    sid_to_label = {}
    for _, row in set_df.iterrows():
        ds = row['dataset']; vid = row['video_id']
        fidx = int(row['frame_index'])
        fk = f'{ds}:{vid}:{fidx}'
        sid_to_anchors[row['sample_id']] = detect_anchors_in_caption(
            captions.get(fk, ''))
        sid_to_label[row['sample_id']] = row['canonical_label']
    yaw_counts = np.zeros(N_YAW)
    vert_counts = np.zeros(N_VERT)
    joint_counts = np.zeros(N_JOINT)
    n_q = 0
    n_responses = 0
    for sid, q in per_q.items():
        if sid_to_label.get(sid) != pair_target:
            continue
        if pair_anchor not in sid_to_anchors.get(sid, set()):
            continue
        n_q += 1
        for y, v in q['samples']:
            yaw_counts[y - 1] += 1
            vert_counts[v] += 1
            joint_counts[joint_index(y, v)] += 1
            n_responses += 1
    if n_responses == 0:
        return None, 0, 0
    py = smooth(yaw_counts, alpha)
    pv = smooth(vert_counts, alpha)
    pj = smooth(joint_counts, alpha)
    return (py, pv, pj), n_q, n_responses


def aggregate_filter(per_q, set_df, captions, pw_y, pw_v, pw_j, pw_counts,
                     pairs, alpha, miller_madow):
    """Per-pair JSD/KL with the given smoothing config."""
    sid_to_anchors = {}
    sid_to_label = {}
    for _, row in set_df.iterrows():
        ds = row['dataset']; vid = row['video_id']
        fidx = int(row['frame_index'])
        fk = f'{ds}:{vid}:{fidx}'
        sid_to_anchors[row['sample_id']] = detect_anchors_in_caption(
            captions.get(fk, ''))
        sid_to_label[row['sample_id']] = row['canonical_label']

    jsds_y, kls_pwm_y, kls_mpw_y = [], [], []
    jsds_j, kls_pwm_j, kls_mpw_j = [], [], []
    for tgt, anc in pairs:
        # Pool model responses across queries in this pair
        yaw_counts = np.zeros(N_YAW)
        joint_counts = np.zeros(N_JOINT)
        n_responses = 0
        for sid, q in per_q.items():
            if sid_to_label.get(sid) != tgt:
                continue
            if anc not in sid_to_anchors.get(sid, set()):
                continue
            for y, v in q['samples']:
                yaw_counts[y - 1] += 1
                joint_counts[joint_index(y, v)] += 1
                n_responses += 1
        if n_responses == 0:
            continue
        m_y = smooth(yaw_counts, alpha)
        m_j = smooth(joint_counts, alpha)
        ppy = pw_y.get((tgt, anc)); ppj = pw_j.get((tgt, anc))
        n_pw = pw_counts.get((tgt, anc), 0)
        if ppy is None or n_pw == 0:
            continue
        # JSD/KL with optional Miller-Madow correction
        jsds_y.append(jsd_with_correction(m_y, ppy, n_responses, n_pw, N_YAW,
                                          miller_madow))
        kls_pwm_y.append(kl_with_correction(ppy, m_y, n_pw, N_YAW, miller_madow))
        kls_mpw_y.append(kl_with_correction(m_y, ppy, n_responses, N_YAW,
                                            miller_madow))
        jsds_j.append(jsd_with_correction(m_j, ppj, n_responses, n_pw, N_JOINT,
                                          miller_madow))
        kls_pwm_j.append(kl_with_correction(ppj, m_j, n_pw, N_JOINT, miller_madow))
        kls_mpw_j.append(kl_with_correction(m_j, ppj, n_responses, N_JOINT,
                                            miller_madow))
    return {
        'n_pairs_used': len(jsds_y),
        'mean_jsd_y': float(np.mean(jsds_y)) if jsds_y else float('nan'),
        'mean_kl_pwm_y': float(np.mean(kls_pwm_y)) if kls_pwm_y else float('nan'),
        'mean_kl_mpw_y': float(np.mean(kls_mpw_y)) if kls_mpw_y else float('nan'),
        'mean_jsd_j': float(np.mean(jsds_j)) if jsds_j else float('nan'),
        'mean_kl_pwm_j': float(np.mean(kls_pwm_j)) if kls_pwm_j else float('nan'),
        'mean_kl_mpw_j': float(np.mean(kls_mpw_j)) if kls_mpw_j else float('nan'),
    }


def compute_baselines(set_df, captions, pw_y, pw_v, pw_j, pw_counts,
                      pairs_B, pairs_C, alpha, miller_madow, nll_floor):
    """Reference baselines: uniform & train_prior (M-independent)."""
    prior = pd.read_parquet(PRIOR_PATH)
    GLOBAL_YAW = np.array(prior[prior.fallback].iloc[0].p_yaw)
    GLOBAL_JOINT = np.array(prior[prior.fallback].iloc[0].p_joint)
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

    # Per-query NLL & mode acc
    floor_y = nll_floor / N_YAW  # exact for uniform/prior — no smoothing artifacts
    floor_j = nll_floor / N_JOINT
    for bname, fn in [
        ('uniform', lambda r: (uy, uv, uj)),
        ('train_prior', lambda r: (
            get_prior(r['canonical_label'])[0],
            np.array([sum(get_prior(r['canonical_label'])[1][v::N_VERT][:N_YAW])
                     for v in range(N_VERT)]),
            get_prior(r['canonical_label'])[1])),
    ]:
        nll_y_a, nll_v_a, nll_j_a = [], [], []
        my_e, mv_e, mj_e = [], [], []
        for _, r in set_df.iterrows():
            gy, gv = int(r['yaw_bin_4']), int(r['height_bin'])
            gj = (gy - 1) * N_VERT + gv
            py, pv, pj = fn(r)
            pv = pv / pv.sum() if pv.sum() > 0 else uv
            nll_y_a.append(-np.log(max(py[gy - 1], 1e-12)))
            nll_v_a.append(-np.log(max(pv[gv], 1e-12)))
            nll_j_a.append(-np.log(max(pj[gj], 1e-12)))
            my_e.append(int(int(np.argmax(py)) + 1 == gy))
            mv_e.append(int(int(np.argmax(pv)) == gv))
            mj_e.append(int(int(np.argmax(pj)) == gj))
        out[f'{bname}_NLL_y'] = float(np.mean(nll_y_a))
        out[f'{bname}_NLL_v'] = float(np.mean(nll_v_a))
        out[f'{bname}_NLL_j'] = float(np.mean(nll_j_a))
        out[f'{bname}_ModeY%'] = 100 * float(np.mean(my_e))
        out[f'{bname}_ModeV%'] = 100 * float(np.mean(mv_e))
        out[f'{bname}_ModeJ%'] = 100 * float(np.mean(mj_e))

    for fl, pairs in [('B', pairs_B), ('C', pairs_C)]:
        for bname in ['uniform', 'train_prior']:
            jsds_y, kls_pwm_y, kls_mpw_y = [], [], []
            jsds_j, kls_pwm_j, kls_mpw_j = [], [], []
            for tgt, anc in pairs:
                ppy, ppj = pw_y[(tgt, anc)], pw_j[(tgt, anc)]
                n_pw = pw_counts.get((tgt, anc), 1)
                if bname == 'uniform':
                    m_y, m_j = uy, uj
                else:
                    m_y, m_j = get_prior(tgt)
                # Treat reference as exact (no MM correction on the reference side)
                jsds_y.append(jsd_with_correction(m_y, ppy, 10**9, n_pw, N_YAW,
                                                   miller_madow))
                kls_pwm_y.append(kl_with_correction(ppy, m_y, n_pw, N_YAW,
                                                     miller_madow))
                kls_mpw_y.append(kl_with_correction(m_y, ppy, 10**9, N_YAW,
                                                     miller_madow))
                jsds_j.append(jsd_with_correction(m_j, ppj, 10**9, n_pw, N_JOINT,
                                                   miller_madow))
                kls_pwm_j.append(kl_with_correction(ppj, m_j, n_pw, N_JOINT,
                                                     miller_madow))
                kls_mpw_j.append(kl_with_correction(m_j, ppj, 10**9, N_JOINT,
                                                     miller_madow))
            out[f'{bname}_mean_jsd_y_{fl}'] = float(np.mean(jsds_y)) if jsds_y else float('nan')
            out[f'{bname}_mean_kl_pwm_y_{fl}'] = float(np.mean(kls_pwm_y)) if kls_pwm_y else float('nan')
            out[f'{bname}_mean_kl_mpw_y_{fl}'] = float(np.mean(kls_mpw_y)) if kls_mpw_y else float('nan')
            out[f'{bname}_mean_jsd_j_{fl}'] = float(np.mean(jsds_j)) if jsds_j else float('nan')
            out[f'{bname}_mean_kl_pwm_j_{fl}'] = float(np.mean(kls_pwm_j)) if kls_pwm_j else float('nan')
            out[f'{bname}_mean_kl_mpw_j_{fl}'] = float(np.mean(kls_mpw_j)) if kls_mpw_j else float('nan')
    return out


def plot_metrics(df, baselines, fig_dir, config_label):
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

    # Combined figure
    n = len(metrics); cols = 3
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
    fig.suptitle(f'Smoothing config: {config_label}', y=1.001, fontsize=12)
    fig.tight_layout()
    fig.savefig(fig_dir / 'dense_all_metrics.png', dpi=120, bbox_inches='tight')
    fig.savefig(fig_dir / 'dense_all_metrics.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {fig_dir / "dense_all_metrics.png"}')

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
        ax.set_title(f'{label} vs M  —  {config_label}')
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        safe_col = col.replace('/', '_')
        fig.savefig(fig_dir / f'dense_{safe_col}.png', dpi=120, bbox_inches='tight')
        plt.close(fig)
    print(f'  Saved {len(metrics)} individual plots')


def run_one_config(config_name, args):
    cfg = CONFIGS[config_name]
    alpha = cfg['alpha']
    miller_madow = cfg['miller_madow']
    nll_floor = cfg['nll_floor']
    print(f'\n=== Running config: {config_name} ===')
    print(f'  alpha={alpha}, miller_madow={miller_madow}, nll_floor={nll_floor}')

    fig_dir = FIG_BASE / config_name
    fig_dir.mkdir(parents=True, exist_ok=True)

    set_df = pd.read_parquet(DATA_DIR / args.set_parquet)
    if args.eval_set != 'full':
        from src._eval_set import filter_by_eval_set
        set_df = filter_by_eval_set(set_df, args.eval_set)

    captions = {}
    cap_path = DATA_DIR / args.captions
    if cap_path.exists():
        with open(cap_path) as f:
            for line in f:
                r = json.loads(line)
                captions[r['frame_key']] = r['caption']
    default = DATA_DIR / 'captions_v2.jsonl'
    if default.exists():
        with open(default) as f:
            for line in f:
                r = json.loads(line)
                if r['frame_key'] not in captions:
                    captions[r['frame_key']] = r['caption']
    print(f'  Loaded {len(captions)} captions, {len(set_df)} queries')

    pw_y, pw_v, pw_j, pw_counts = build_pairwise_priors_pooled(
        set_df, captions, alpha)
    pairs_B = filter_pairs(pw_counts, n_min=5, filter_level='B')
    pairs_C = filter_pairs(pw_counts, n_min=5, filter_level='C')
    print(f'  Filter B pairs: {len(pairs_B)}, Filter C: {len(pairs_C)}')

    runs_dir = Path(args.runs_dir)
    files = sorted(runs_dir.glob(args.file_pattern))
    print(f'  Found {len(files)} run files')

    rows = []
    M_values = list(range(args.m_min, args.m_max + 1, args.m_step))

    for fp in files:
        stem = fp.stem
        parts = stem.split('_')
        model = '_'.join(parts[:-2])
        cond = parts[-2]
        by_sid = load_run_responses(fp)
        if not by_sid:
            continue
        print(f'  {fp.name}: {len(by_sid)} sids')
        for M in M_values:
            per_q = per_query_metrics(by_sid, set_df, M, alpha, nll_floor)
            if not per_q:
                continue
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
            agg_B = aggregate_filter(per_q, set_df, captions, pw_y, pw_v, pw_j,
                                     pw_counts, pairs_B, alpha, miller_madow)
            agg_C = aggregate_filter(per_q, set_df, captions, pw_y, pw_v, pw_j,
                                     pw_counts, pairs_C, alpha, miller_madow)
            for k, v in agg_B.items(): row[f'{k}_B'] = v
            for k, v in agg_C.items(): row[f'{k}_C'] = v
            rows.append(row)
        print(f'    Done {len(M_values)} M values')

    df = pd.DataFrame(rows)
    df_path = OUT_DIR / f'm_ablation_dense_{config_name}.csv'
    df.to_csv(df_path, index=False)
    print(f'  Saved: {df_path}')

    print('  Computing baselines...')
    baselines = compute_baselines(set_df, captions, pw_y, pw_v, pw_j, pw_counts,
                                  pairs_B, pairs_C, alpha, miller_madow, nll_floor)
    bl_path = OUT_DIR / f'm_ablation_baselines_{config_name}.csv'
    pd.DataFrame([baselines]).to_csv(bl_path, index=False)
    print(f'  Saved: {bl_path}')

    print('  Plotting...')
    plot_metrics(df, baselines, fig_dir, cfg['label'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='all',
                        choices=list(CONFIGS.keys()) + ['all'])
    parser.add_argument('--runs-dir', type=str,
                        default=str(OUT_DIR / 'runs_extended'))
    parser.add_argument('--captions', type=str,
                        default='captions_v2_extended.jsonl')
    parser.add_argument('--set-parquet', type=str,
                        default='setA_extended_subset500.parquet')
    parser.add_argument('--m-min', type=int, default=1)
    parser.add_argument('--m-max', type=int, default=49)
    parser.add_argument('--m-step', type=int, default=1)
    parser.add_argument('--file-pattern', type=str, default='*_M50.jsonl')
    from src._eval_set import add_eval_set_arg
    add_eval_set_arg(parser)
    args = parser.parse_args()

    configs = list(CONFIGS.keys()) if args.config == 'all' else [args.config]
    for c in configs:
        run_one_config(c, args)


if __name__ == '__main__':
    main()
