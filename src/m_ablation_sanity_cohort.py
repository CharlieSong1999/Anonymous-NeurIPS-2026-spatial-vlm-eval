"""Sanity check: split per-query NLL_j into 'GT observed' vs 'GT missed'
cohorts at each M, under alpha=1e-3.

The hypothesis: Gemma's rising NLL_j with M is driven by a large 'GT missed'
cohort whose mean NLL tracks log(M) + log(1/alpha) by construction. Other
models either have a smaller missed cohort (better-calibrated mass on GT)
or have it shrink with M.

Reads the same M=50 dense JSONLs used by the smoothing M-ablation, so
results are directly comparable to figs/alpha_tiny_1e-3/dense_all_metrics.pdf.

Outputs:
  exp/m_ablation_001/figs/sanity_cohort/
    cohort_fraction_vs_M.{pdf,png}    miss-rate vs M, per model
    cohort_meanNLL_vs_M.{pdf,png}     mean NLL_j within each cohort vs M
    overall_NLL_vs_M.{pdf,png}        re-derived overall NLL (sanity-check vs csv)
    summary.csv                       numeric backing data
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from m_ablation import load_run_responses, DATA_DIR, TESTSET
from metrics_v2 import N_YAW, N_VERT, N_JOINT, joint_index

RUNS_DIR = TESTSET / "exp" / "m_ablation_001" / "runs_extended"
OUT_DIR = TESTSET / "exp" / "m_ablation_001" / "figs" / "sanity_cohort"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALPHA = 1e-3
M_VALUES = [1, 2, 3, 5, 7, 10, 15, 20, 25, 30, 40, 49]
MODELS = [
    ("gemma-4-31b",   "gemma-4-31b_sighted_M50.jsonl"),
    ("qwen3-vl-30b",  "qwen3-vl-30b_sighted_M50.jsonl"),
    ("qwen3.5-9b",    "qwen3.5-9b_sighted_M50.jsonl"),
]


def cohort_metrics(by_sid, set_df, M, alpha):
    """For a given M and alpha, return per-query (n_gt, nll_j) and cohort
    masks. nll_j uses the same plug-in MLE+alpha formula as the locked
    estimator (no extra floor needed; alpha already prevents log(0))."""
    sid_to_row = {r["sample_id"]: r for _, r in set_df.iterrows()}
    n_gt_list = []
    nll_list = []
    m_eff_list = []
    for sid, all_samples in by_sid.items():
        row = sid_to_row.get(sid)
        if row is None:
            continue
        samples = all_samples[:M]
        if not samples:
            continue
        gt_y = int(row["yaw_bin_4"])
        gt_v = int(row["height_bin"])
        gt_j = (gt_y - 1) * N_VERT + gt_v
        joint_counts = np.zeros(N_JOINT)
        for y, v in samples:
            joint_counts[joint_index(y, v)] += 1
        n_gt = int(joint_counts[gt_j])
        m_eff = len(samples)
        p_gt = (n_gt + alpha) / (m_eff + N_JOINT * alpha)
        nll_list.append(-np.log(p_gt))
        n_gt_list.append(n_gt)
        m_eff_list.append(m_eff)
    n_gt_arr = np.array(n_gt_list, dtype=int)
    nll_arr = np.array(nll_list, dtype=float)
    m_eff_arr = np.array(m_eff_list, dtype=int)
    return n_gt_arr, nll_arr, m_eff_arr


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--set-parquet', type=str,
                        default='setA_extended_subset500.parquet')
    from src._eval_set import add_eval_set_arg, filter_by_eval_set
    add_eval_set_arg(parser)
    args = parser.parse_args()

    set_df = pd.read_parquet(DATA_DIR / args.set_parquet)
    if args.eval_set != 'full':
        set_df = filter_by_eval_set(set_df, args.eval_set)
    print(f"Loaded set: {len(set_df)} queries (eval_set={args.eval_set})")

    rows = []
    per_model_curves = {}

    for model_name, fname in MODELS:
        path = RUNS_DIR / fname
        if not path.exists():
            print(f"MISSING: {path}")
            continue
        by_sid = load_run_responses(path)
        print(f"{model_name}: loaded {len(by_sid)} sids, "
              f"first sid has {len(next(iter(by_sid.values())))} samples")

        curves = {
            "M": [],
            "frac_miss": [],
            "n_total": [],
            "n_miss": [],
            "n_obs": [],
            "mean_nll_miss": [],
            "mean_nll_obs": [],
            "mean_nll_overall": [],
            "predicted_floor": [],
        }
        for M in M_VALUES:
            n_gt, nll, m_eff = cohort_metrics(by_sid, set_df, M, ALPHA)
            if len(n_gt) == 0:
                continue
            miss = (n_gt == 0)
            obs = ~miss
            n_total = len(n_gt)
            n_miss = int(miss.sum())
            n_obs = int(obs.sum())
            mean_nll_miss = float(nll[miss].mean()) if n_miss > 0 else np.nan
            mean_nll_obs = float(nll[obs].mean()) if n_obs > 0 else np.nan
            mean_nll_overall = float(nll.mean())
            # Predicted floor for n_gt=0 queries with effective m. Use the
            # mean of -log(alpha / (m_eff + N_JOINT*alpha)) across the
            # missed cohort (m_eff might differ per query if some had < M).
            if n_miss > 0:
                pred_floor = float(np.mean(
                    -np.log(ALPHA / (m_eff[miss] + N_JOINT * ALPHA))))
            else:
                pred_floor = np.nan

            curves["M"].append(M)
            curves["frac_miss"].append(n_miss / n_total)
            curves["n_total"].append(n_total)
            curves["n_miss"].append(n_miss)
            curves["n_obs"].append(n_obs)
            curves["mean_nll_miss"].append(mean_nll_miss)
            curves["mean_nll_obs"].append(mean_nll_obs)
            curves["mean_nll_overall"].append(mean_nll_overall)
            curves["predicted_floor"].append(pred_floor)
            rows.append({
                "model": model_name,
                "M": M,
                "n_total": n_total,
                "n_miss": n_miss,
                "n_obs": n_obs,
                "frac_miss": n_miss / n_total,
                "mean_nll_miss": mean_nll_miss,
                "mean_nll_obs": mean_nll_obs,
                "mean_nll_overall": mean_nll_overall,
                "predicted_floor_miss": pred_floor,
            })
        per_model_curves[model_name] = curves

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_DIR / "summary.csv", index=False)
    print(f"Wrote {OUT_DIR / 'summary.csv'}")
    print(summary.to_string(index=False))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "gemma-4-31b":   "tab:red",
        "qwen3-vl-30b":  "tab:blue",
        "qwen3.5-9b":    "tab:green",
    }

    # 1. Miss-rate vs M
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for m, c in per_model_curves.items():
        ax.plot(c["M"], c["frac_miss"], "o-",
                color=colors.get(m, "k"), label=m, linewidth=1.5)
    ax.set_xlabel("M (samples per query)")
    ax.set_ylabel("Fraction of queries with GT bin never sampled")
    ax.set_title("Miss-rate cohort fraction vs M (alpha=1e-3)")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "cohort_fraction_vs_M.pdf")
    fig.savefig(OUT_DIR / "cohort_fraction_vs_M.png", dpi=150)
    plt.close(fig)

    # 2. Mean NLL within each cohort vs M (+ predicted floor)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    ax = axes[0]
    for m, c in per_model_curves.items():
        ax.plot(c["M"], c["mean_nll_miss"], "o-",
                color=colors.get(m, "k"), label=f"{m} (miss cohort)",
                linewidth=1.5)
    # Theoretical floor: -log(alpha / (M + N_JOINT*alpha)) ≈ log(M) + log(1/alpha)
    M_arr = np.array(M_VALUES, dtype=float)
    pred = -np.log(ALPHA / (M_arr + N_JOINT * ALPHA))
    ax.plot(M_arr, pred, "k--", linewidth=1.0,
            label=r"$-\log(\alpha/(M + 9\alpha))$ (theory)")
    ax.set_xlabel("M")
    ax.set_ylabel("Mean per-query NLL_j")
    ax.set_title("GT-missed cohort: NLL grows as log(M)+const")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    for m, c in per_model_curves.items():
        ax.plot(c["M"], c["mean_nll_obs"], "s-",
                color=colors.get(m, "k"), label=f"{m} (obs cohort)",
                linewidth=1.5)
    ax.set_xlabel("M")
    ax.set_title("GT-observed cohort: NLL falls (variance shrinks)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "cohort_meanNLL_vs_M.pdf")
    fig.savefig(OUT_DIR / "cohort_meanNLL_vs_M.png", dpi=150)
    plt.close(fig)

    # 3. Overall NLL re-derivation (sanity vs published curves)
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for m, c in per_model_curves.items():
        ax.plot(c["M"], c["mean_nll_overall"], "o-",
                color=colors.get(m, "k"), label=m, linewidth=1.5)
    ax.set_xlabel("M")
    ax.set_ylabel("Mean per-query NLL_j (overall)")
    ax.set_title("Overall NLL_j vs M (re-derived from raw JSONL, alpha=1e-3)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "overall_NLL_vs_M.pdf")
    fig.savefig(OUT_DIR / "overall_NLL_vs_M.png", dpi=150)
    plt.close(fig)

    print(f"Plots saved under {OUT_DIR}")


if __name__ == "__main__":
    main()
