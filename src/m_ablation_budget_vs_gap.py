"""
Budget-vs-gap figure for the smoothing-config M-ablation.

For each (config, model, metric), compute:
  cost_frac(M)     = M / M_max
  gap_remaining(M) = |metric(M) - metric(M_max)| / |metric(M_min) - metric(M_max)|
                     (clipped to [0, ~1.5] so overshoots are visible)

A "good" estimator hugs the bottom-left: low cost, gap already eliminated.
A "bad" estimator stays near the top: even at high cost, the gap to the
asymptote isn't closed.

Key insight: y measures how close metric(M) is to its M=M_max asymptote
within that config. Different configs can have different asymptotes
(see smoothing_comparison.md), so this figure does NOT compare config
asymptotes — it compares **convergence rate** of each estimator to its
own asymptote.

Outputs:
  figs/budget_vs_gap/budget_vs_gap_<metric>.png    (one per metric)
  figs/budget_vs_gap/budget_vs_gap_grid.png        (combined grid)
  figs/budget_vs_gap/budget_vs_gap_summary.png     (averaged across
                                                    distributional metrics)

Run:
  conda run -n slam python3 -m src.m_ablation_budget_vs_gap
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT_DIR = Path('/path/to/this/repo/exp/m_ablation_001')
FIG_DIR = OUT_DIR / 'figs' / 'budget_vs_gap'
FIG_DIR.mkdir(parents=True, exist_ok=True)

CONFIGS = ['alpha_jeffrey_05', 'alpha_tiny_1e-3', 'mle_miller_madow']
CONFIG_LABELS = {
    'alpha_jeffrey_05': r'Jeffrey $\alpha=0.5$',
    'alpha_tiny_1e-3':  r'Tiny $\alpha=10^{-3}$',
    'mle_miller_madow': 'MLE + Miller-Madow',
}
CONFIG_COLORS = {
    'alpha_jeffrey_05': 'tab:red',
    'alpha_tiny_1e-3':  'tab:orange',
    'mle_miller_madow': 'tab:green',
}
MODEL_STYLES = {
    'gemma-4-31b':  '-',
    'qwen3-vl-30b': '--',
    'qwen3.5-9b':   ':',
}

METRICS = [
    ('mean_jsd_y_B', 'JSD_y (Filter B)'),
    ('mean_jsd_j_B', 'JSD_j (Filter B)'),
    ('mean_kl_pwm_y_B', 'KL(P_pw||P_m)_y (B)'),
    ('mean_kl_mpw_y_B', 'KL(P_m||P_pw)_y (B)'),
    ('mean_kl_pwm_j_B', 'KL(P_pw||P_m)_j (B)'),
    ('mean_kl_mpw_j_B', 'KL(P_m||P_pw)_j (B)'),
    ('mean_jsd_y_C', 'JSD_y (Filter C)'),
    ('mean_jsd_j_C', 'JSD_j (Filter C)'),
    ('NLL_y', 'NLL yaw'),
    ('NLL_v', 'NLL vert'),
    ('NLL_j', 'NLL joint'),
    ('ModeJ%', 'Mode acc joint %'),
]


def load_all():
    """Return dict: config -> DataFrame."""
    out = {}
    for cfg in CONFIGS:
        path = OUT_DIR / f'm_ablation_dense_{cfg}.csv'
        out[cfg] = pd.read_csv(path)
    return out


def compute_gap_remaining(df, metric):
    """Return DataFrame with M, model, gap_remaining, cost_frac for the given metric."""
    if metric not in df.columns:
        return None
    rows = []
    M_max = int(df.M.max())
    M_min = int(df.M.min())
    for model, sub in df.groupby('model'):
        sub = sub.sort_values('M')
        asymptote = float(sub[sub.M == M_max][metric].iloc[0])
        starting = float(sub[sub.M == M_min][metric].iloc[0])
        denom = abs(starting - asymptote)
        if denom < 1e-12:
            denom = 1.0  # degenerate — flat curve; gap is already 0
        for _, r in sub.iterrows():
            gap = abs(float(r[metric]) - asymptote) / denom
            cost = float(r['M']) / M_max
            rows.append({'model': model, 'M': int(r['M']),
                         'cost_frac': cost, 'gap_remaining': gap,
                         'value': float(r[metric]),
                         'asymptote': asymptote})
    return pd.DataFrame(rows)


def plot_per_metric(all_data, metric, label, fig_dir):
    """One figure per metric. 9 lines = 3 configs × 3 models."""
    fig, ax = plt.subplots(figsize=(8, 6))
    # Reference: linear cost = gap eliminated
    ax.plot([0, 1], [1, 0], color='gray', linestyle='-', linewidth=0.8,
            alpha=0.5, label='linear (1 unit cost = 1 unit gap closed)')
    for cfg in CONFIGS:
        df = all_data[cfg]
        gap_df = compute_gap_remaining(df, metric)
        if gap_df is None:
            continue
        for model, sub in gap_df.groupby('model'):
            sub = sub.sort_values('cost_frac')
            ax.plot(sub.cost_frac, sub.gap_remaining,
                    color=CONFIG_COLORS[cfg],
                    linestyle=MODEL_STYLES.get(model, '-'),
                    linewidth=1.5, alpha=0.85,
                    label=f'{CONFIG_LABELS[cfg]} | {model}')
    ax.set_xlabel('Cost fraction (M / M_max)')
    ax.set_ylabel('Gap to asymptote remaining (normalized)')
    ax.set_title(f'{label} — convergence rate vs cost')
    ax.set_xlim(0, 1.02)
    ax.set_ylim(-0.05, 1.5)
    ax.axhline(0, color='black', linewidth=0.5, alpha=0.5)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc='upper right')
    fig.tight_layout()
    safe = metric.replace('/', '_')
    fig.savefig(fig_dir / f'budget_vs_gap_{safe}.png', dpi=120, bbox_inches='tight')
    plt.close(fig)


def plot_grid(all_data, fig_dir):
    """Combined grid: one panel per metric."""
    n = len(METRICS); cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.5 * rows))
    axes = axes.flatten()
    for i, (metric, label) in enumerate(METRICS):
        ax = axes[i]
        ax.plot([0, 1], [1, 0], color='gray', linestyle='-', linewidth=0.8,
                alpha=0.5)
        for cfg in CONFIGS:
            df = all_data[cfg]
            gap_df = compute_gap_remaining(df, metric)
            if gap_df is None:
                continue
            for model, sub in gap_df.groupby('model'):
                sub = sub.sort_values('cost_frac')
                ax.plot(sub.cost_frac, sub.gap_remaining,
                        color=CONFIG_COLORS[cfg],
                        linestyle=MODEL_STYLES.get(model, '-'),
                        linewidth=1.2, alpha=0.85)
        ax.set_xlabel('Cost (M / M_max)')
        ax.set_ylabel('Gap remaining')
        ax.set_title(label, fontsize=10)
        ax.set_xlim(0, 1.02)
        ax.set_ylim(-0.05, 1.5)
        ax.axhline(0, color='black', linewidth=0.5, alpha=0.5)
        ax.grid(alpha=0.3)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    # Single legend at top
    handles = [plt.Line2D([0], [0], color=CONFIG_COLORS[c], lw=2,
                          label=CONFIG_LABELS[c]) for c in CONFIGS]
    handles.extend([plt.Line2D([0], [0], color='black',
                                linestyle=MODEL_STYLES[m], lw=1.5, label=m)
                    for m in MODEL_STYLES])
    handles.append(plt.Line2D([0], [0], color='gray', lw=0.8,
                              label='linear reference (1:1)'))
    fig.legend(handles=handles, loc='upper center',
               bbox_to_anchor=(0.5, 1.005), ncol=4, fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(fig_dir / 'budget_vs_gap_grid.png', dpi=120, bbox_inches='tight')
    fig.savefig(fig_dir / 'budget_vs_gap_grid.pdf', bbox_inches='tight')
    plt.close(fig)


def plot_summary(all_data, fig_dir):
    """Averaged-over-distributional-metrics summary, one line per (config, model)."""
    # Use only distributional Filter B metrics for the summary
    summary_metrics = ['mean_jsd_y_B', 'mean_jsd_j_B',
                       'mean_kl_pwm_y_B', 'mean_kl_mpw_y_B',
                       'mean_kl_pwm_j_B', 'mean_kl_mpw_j_B']
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [1, 0], color='gray', linestyle='-', linewidth=0.8,
            alpha=0.5, label='linear reference (1:1)')
    for cfg in CONFIGS:
        df = all_data[cfg]
        # Average gap_remaining across the metrics, per (model, M)
        all_gaps = []
        for metric in summary_metrics:
            g = compute_gap_remaining(df, metric)
            if g is None:
                continue
            g = g.assign(metric=metric)
            all_gaps.append(g)
        if not all_gaps:
            continue
        gap_df = pd.concat(all_gaps, ignore_index=True)
        avg = gap_df.groupby(['model', 'cost_frac'], as_index=False)[
            'gap_remaining'].mean()
        for model, sub in avg.groupby('model'):
            sub = sub.sort_values('cost_frac')
            ax.plot(sub.cost_frac, sub.gap_remaining,
                    color=CONFIG_COLORS[cfg],
                    linestyle=MODEL_STYLES.get(model, '-'),
                    linewidth=1.7, alpha=0.85,
                    label=f'{CONFIG_LABELS[cfg]} | {model}')
    ax.set_xlabel('Cost fraction (M / M_max)')
    ax.set_ylabel('Mean gap to asymptote remaining (normalized)')
    ax.set_title('Convergence rate vs cost — averaged across distributional metrics')
    ax.set_xlim(0, 1.02)
    ax.set_ylim(-0.05, 1.5)
    ax.axhline(0, color='black', linewidth=0.5, alpha=0.5)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc='upper right')
    fig.tight_layout()
    fig.savefig(fig_dir / 'budget_vs_gap_summary.png', dpi=120, bbox_inches='tight')
    fig.savefig(fig_dir / 'budget_vs_gap_summary.pdf', bbox_inches='tight')
    plt.close(fig)


def emit_breakeven_table(all_data):
    """For each (config, model, metric), find smallest cost_frac at which
    gap_remaining drops below 10%, 5%, 1%. Print as a table."""
    print('\n=== Cost fraction needed to close X% of the gap ===')
    print(f'{"metric":<22} {"config":<22} {"model":<13} '
          f'{"≤10%":>6} {"≤5%":>6} {"≤1%":>6}')
    print('-' * 78)
    for metric in ['mean_jsd_j_B', 'mean_jsd_y_B', 'mean_kl_mpw_j_B']:
        for cfg in CONFIGS:
            df = all_data[cfg]
            gap_df = compute_gap_remaining(df, metric)
            if gap_df is None:
                continue
            for model, sub in gap_df.groupby('model'):
                sub = sub.sort_values('cost_frac')
                cuts = {}
                for thresh in [0.10, 0.05, 0.01]:
                    below = sub[sub.gap_remaining <= thresh]
                    cuts[thresh] = (below.cost_frac.min()
                                    if len(below) else float('nan'))
                print(f'{metric:<22} {CONFIG_LABELS[cfg]:<22} {model:<13} '
                      f'{cuts[0.10]:>6.2f} {cuts[0.05]:>6.2f} {cuts[0.01]:>6.2f}')


def main():
    all_data = load_all()
    print(f'Loaded {len(all_data)} configs')

    # Per-metric figures
    for metric, label in METRICS:
        plot_per_metric(all_data, metric, label, FIG_DIR)
    print(f'Saved {len(METRICS)} per-metric figures')

    # Combined grid
    plot_grid(all_data, FIG_DIR)
    print('Saved combined grid')

    # Summary
    plot_summary(all_data, FIG_DIR)
    print('Saved summary figure')

    emit_breakeven_table(all_data)


if __name__ == '__main__':
    main()
