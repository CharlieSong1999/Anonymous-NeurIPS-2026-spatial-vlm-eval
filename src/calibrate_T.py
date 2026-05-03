"""Temperature-scaling calibration of Q1 NLL across VLMs.

See plan.md for design. Outputs results.md with primary + sensitivity
tables, plus a stub §3 for hand-written analysis.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

# ── Paths ─────────────────────────────────────────────────────────────
META = Path("/path/to/this/repo")
CALIB_DIR = META / "metric" / "NLL" / "calibration"
TESTSET = META / "testset"
DATA_DIR = TESTSET / "data"
RUNS_DIR = TESTSET / "exp" / "sampling_ablation_001" / "runs"

SET_PARQUET = DATA_DIR / "setA_300_filtered.parquet"
TRAIN_POOL = DATA_DIR / "extended_pool_balanced.parquet"
DEFAULT_SPLIT_JSON = CALIB_DIR / "split.json"
PER_SAMPLE_DIR = CALIB_DIR / "per_sample"
PER_SAMPLE_DIR.mkdir(exist_ok=True)
PER_CATEGORY_DIR = CALIB_DIR / "per_category"
PER_CATEGORY_DIR.mkdir(exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────
N_YAW = 3      # yaw_bin_4 ∈ {1,2,3} (front=0 excluded)
N_VERT = 3     # height_bin ∈ {0,1,2}
N_JOINT = 9    # 3 × 3 — module-level default for backward compat. Reset by
               # set_task() at runtime when --task yaw or --task pitch is used.
M_TARGET = 20
ALPHAS = [0.01, 0.05, 0.1, 0.5, 1.0]
ALPHA_DEFAULT = 0.1
EPS_FLOOR = 1e-12

PITCH_TO_VBIN = {"UP": 2, "LEVEL": 1, "DOWN": 0}

# Task selection (mutated by set_task at runtime)
TASK = "joint"


def joint_index(yaw_bin_4: int, height_bin: int) -> int:
    """Joint 9-bin index. Used by default; replaced via set_task() for
    yaw-only / pitch-only tasks (kept named for backward compat)."""
    return (yaw_bin_4 - 1) * N_VERT + height_bin


def set_task(task: str) -> None:
    """Re-bind module globals N_JOINT and joint_index for the chosen task."""
    global N_JOINT, joint_index, TASK
    TASK = task
    if task == "joint":
        N_JOINT = N_YAW * N_VERT
        joint_index = lambda yaw_bin_4, height_bin: (yaw_bin_4 - 1) * N_VERT + height_bin
    elif task == "yaw":
        N_JOINT = N_YAW
        joint_index = lambda yaw_bin_4, height_bin: yaw_bin_4 - 1
    elif task == "pitch":
        N_JOINT = N_VERT
        joint_index = lambda yaw_bin_4, height_bin: height_bin
    else:
        raise ValueError(f"Unknown task: {task}")


# ── Loaders ───────────────────────────────────────────────────────────
def load_run_responses(path: Path):
    """Return (dict[sample_id] -> list of (yaw, height) up to first 20,
              dict[sample_id] -> count of valid samples used)."""
    by_sid: dict[str, list] = {}
    with open(path) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            sid = rec["sample_id"]
            parsed = rec.get("parsed") or {}
            try:
                y = int(parsed.get("yaw_bin_id"))
            except (TypeError, ValueError):
                continue
            v_raw = parsed.get("pitch", "")
            v = PITCH_TO_VBIN.get(str(v_raw).upper()) if v_raw else None
            if y not in (1, 2, 3) or v is None:
                continue
            rep = rec.get("repeat_id")
            by_sid.setdefault(sid, []).append((rep, y, v))
    out = {}
    M_per = {}
    for sid, lst in by_sid.items():
        lst.sort(key=lambda x: (x[0] if x[0] is not None else 0))
        keep = [(y, v) for _, y, v in lst[:M_TARGET]]
        out[sid] = keep
        M_per[sid] = len(keep)
    return out, M_per


def counts_per_query(samples: list[tuple[int, int]]) -> np.ndarray:
    c = np.zeros(N_JOINT, dtype=np.float64)
    for y, v in samples:
        c[joint_index(y, v)] += 1
    return c


# ── Smoothing ─────────────────────────────────────────────────────────
def smooth_from_counts(counts: np.ndarray, alpha: float, M: float) -> np.ndarray:
    return (counts + alpha) / (M + N_JOINT * alpha)


# ── T scaling (logit-space) ───────────────────────────────────────────
def tscale_log_p(log_p: np.ndarray, T: float) -> np.ndarray:
    """Return log of T-scaled distribution. log_p shape: (..., B).
    Handles T = inf (β = 0 → uniform predictive) by computing in
    inverse-temperature space:  z = log_p * β  with  β = 1/T  (β = 0
    when T is inf or non-positive)."""
    if T is None or T <= 0 or math.isinf(T):
        beta = 0.0
    else:
        beta = 1.0 / T
    z = log_p * beta
    return z - logsumexp(z, axis=-1, keepdims=True)


BETA_MAX = 100.0  # inverse temperature upper bound
BETA_MIN = 0.0    # β = 0 ↔ uniform predictive (T → ∞)


def _cat_indices(cats):
    """Group sample indices by category, return list of np.ndarray index
    groups (one per unique category, sorted)."""
    by_cat = {}
    for i, c in enumerate(cats):
        by_cat.setdefault(c, []).append(i)
    return [np.array(v, dtype=int) for k, v in sorted(by_cat.items())]


def aggregate(per_sample_vals: np.ndarray, cats=None) -> float:
    """Per-sample mean (cats=None) or per-category-then-across-categories
    mean (cats provided)."""
    if cats is None:
        return float(np.mean(per_sample_vals))
    groups = _cat_indices(cats)
    means = [float(np.mean(per_sample_vals[idx])) for idx in groups]
    return float(np.mean(means))


def tune_T(log_ps: np.ndarray, gts: np.ndarray, cats=None) -> float:
    """log_ps: (N, B); gts: (N,) GT indices. Optimise inverse temperature
    β ∈ [BETA_MIN, BETA_MAX]; return T = 1/β.

    β = 0 means "predict uniform" (T = ∞, returned as float('inf')).
    β = 1 means "no scaling" (T = 1, identity).
    Larger β = sharper than the smoothed input.

    If cats is provided (list of category labels, one per row), the loss
    is per-category averaged: NLL is meaned within each category, then
    across categories — matches the per-category reported metric.

    Optimising over β is preferred to optimising over T because it lets
    the optimum reach the uniform-predictive regime without an artificial
    upper bound on T (which previously capped T at 20)."""
    groups = _cat_indices(cats) if cats is not None else None

    def loss(beta):
        z = log_ps * beta
        nll = -(z[np.arange(len(gts)), gts] - logsumexp(z, axis=1))
        if groups is None:
            return float(np.mean(nll))
        means = [float(np.mean(nll[idx])) for idx in groups]
        return float(np.mean(means))

    res = minimize_scalar(loss, bounds=(BETA_MIN, BETA_MAX),
                            method="bounded",
                            options={"xatol": 1e-6})
    beta_star = float(res.x)
    # Any β < 1e-3 corresponds to T > 1000 — predicted distribution is
    # numerically indistinguishable from uniform. Return T = ∞ so it
    # displays cleanly and the downstream tscale_log_p maps to uniform.
    if beta_star < 1e-3:
        return float("inf")
    return 1.0 / beta_star


# ── Category prior (LOSO by participant) ──────────────────────────────
def build_category_prior_loso(set_df: pd.DataFrame, train_df: pd.DataFrame):
    """Return dict[sample_id] -> counts (size B). Counts come from training-pool
    entries with the same canonical_label but a different participant."""
    out = {}
    M_per = {}
    grouped = {}
    for (pid, lbl), g in train_df.groupby(["participant_id", "canonical_label"]):
        grouped[(pid, lbl)] = g[["yaw_bin_4", "height_bin"]].values
    pids = sorted(train_df["participant_id"].unique())
    for _, row in set_df.iterrows():
        sid = row["sample_id"]
        pid = row["participant_id"]
        lbl = row["canonical_label"]
        counts = np.zeros(N_JOINT, dtype=np.float64)
        for tpid in pids:
            if tpid == pid:
                continue
            arr = grouped.get((tpid, lbl))
            if arr is None:
                continue
            for yb, hb in arr:
                yb, hb = int(yb), int(hb)
                if yb in (1, 2, 3) and hb in (0, 1, 2):
                    counts[joint_index(yb, hb)] += 1
        out[sid] = counts
        M_per[sid] = float(counts.sum())
    return out, M_per


# ── Per-model evaluation ──────────────────────────────────────────────
def evaluate_model(name: str, counts_d: dict, M_per: dict, set_df: pd.DataFrame,
                   split: dict, alpha: float, averaging: str = "sample"):
    """If averaging='category', NLL/acc/entropy are aggregated per
    canonical_label first, then meaned across categories. T is tuned
    against the same loss."""
    sid_to_gt = {r["sample_id"]: joint_index(int(r["yaw_bin_4"]), int(r["height_bin"]))
                  for _, r in set_df.iterrows()}
    sid_to_cat = {r["sample_id"]: r["canonical_label"] for _, r in set_df.iterrows()}
    cal = [s for s in split["calibration_sample_ids"] if s in counts_d]
    ev = [s for s in split["eval_sample_ids"] if s in counts_d]

    def smooth_log(sid):
        c = counts_d[sid]; M = M_per[sid]
        p = smooth_from_counts(c, alpha, M)
        return np.log(p)

    cal_log = np.stack([smooth_log(s) for s in cal])
    cal_gts = np.array([sid_to_gt[s] for s in cal], dtype=int)
    ev_log = np.stack([smooth_log(s) for s in ev])
    ev_gts = np.array([sid_to_gt[s] for s in ev], dtype=int)

    cal_cats = [sid_to_cat[s] for s in cal] if averaging == "category" else None
    ev_cats = [sid_to_cat[s] for s in ev] if averaging == "category" else None

    T_star = tune_T(cal_log, cal_gts, cal_cats)
    T_oracle = tune_T(ev_log, ev_gts, ev_cats)

    per_smooth = -ev_log[np.arange(len(ev_gts)), ev_gts]
    nll_smooth = aggregate(per_smooth, ev_cats)
    ev_log_T = tscale_log_p(ev_log, T_star)
    per_calib = -ev_log_T[np.arange(len(ev_gts)), ev_gts]
    nll_calib = aggregate(per_calib, ev_cats)
    ev_log_oracle = tscale_log_p(ev_log, T_oracle)
    per_oracle = -ev_log_oracle[np.arange(len(ev_gts)), ev_gts]
    nll_oracle = aggregate(per_oracle, ev_cats)
    delta_oracle = nll_calib - nll_oracle  # ≥ 0 by construction (oracle is min)
    # Entropy pre-calibration (T=1, just smoothed)
    per_H_pre = -np.sum(np.exp(ev_log) * ev_log, axis=1)
    entropy_pre = aggregate(per_H_pre, ev_cats)
    entropy_norm_pre = entropy_pre / math.log(N_JOINT)
    # Entropy post-calibration (T=T*)
    per_H_post = -np.sum(np.exp(ev_log_T) * ev_log_T, axis=1)
    entropy_post = aggregate(per_H_post, ev_cats)
    entropy_norm_post = entropy_post / math.log(N_JOINT)
    per_correct = (np.argmax(ev_log, axis=1) == ev_gts).astype(float)
    acc = aggregate(per_correct, ev_cats)

    out = {
        "name": name, "T_star": T_star, "T_oracle": T_oracle,
        "eval_acc": acc, "eval_nll_smooth": nll_smooth,
        "eval_nll_calib": nll_calib, "eval_nll_oracle": nll_oracle,
        "delta_oracle": delta_oracle,
        "eval_entropy_pre": entropy_pre,
        "eval_entropy_norm_pre": entropy_norm_pre,
        "eval_entropy_post": entropy_post,
        "eval_entropy_norm_post": entropy_norm_post,
        "n_cal": len(cal), "n_eval": len(ev),
    }
    if averaging == "category":
        # Build the intermediate per-(model,category) NLL_calib breakdown.
        per_cat = {}
        groups = {}
        for i, c in enumerate(ev_cats):
            groups.setdefault(c, []).append(i)
        for c, idx in groups.items():
            idx = np.array(idx, dtype=int)
            per_cat[c] = {
                "n": int(len(idx)),
                "nll_smooth": float(np.mean(per_smooth[idx])),
                "nll_calib": float(np.mean(per_calib[idx])),
                "nll_oracle": float(np.mean(per_oracle[idx])),
                "acc": float(np.mean(per_correct[idx])),
            }
        out["per_category"] = per_cat
        out["n_cal_categories"] = len(set(cal_cats)) if cal_cats else 0
        out["n_eval_categories"] = len(per_cat)
    return out


def floored_nll(counts_d: dict, M_per: dict, set_df: pd.DataFrame,
                  split: dict, eps: float = EPS_FLOOR,
                  averaging: str = "sample") -> float:
    sid_to_gt = {r["sample_id"]: joint_index(int(r["yaw_bin_4"]), int(r["height_bin"]))
                  for _, r in set_df.iterrows()}
    sid_to_cat = {r["sample_id"]: r["canonical_label"] for _, r in set_df.iterrows()}
    nlls = []
    cats = []
    for sid in split["eval_sample_ids"]:
        if sid not in counts_d:
            continue
        c = counts_d[sid]; M = M_per[sid]
        if M <= 0:
            # No data → uniform fallback as floored prediction
            p = np.full(N_JOINT, 1.0 / N_JOINT)
        else:
            p = c / M
            p = np.clip(p, eps, 1.0)
            p = p / p.sum()
        nlls.append(-math.log(p[sid_to_gt[sid]]))
        cats.append(sid_to_cat[sid])
    if not nlls:
        return float("nan")
    if averaging == "category":
        return aggregate(np.array(nlls), cats)
    return float(np.mean(nlls))


# ── Main ──────────────────────────────────────────────────────────────
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=str, default=str(DEFAULT_SPLIT_JSON),
                      help="Path to split JSON (default: split.json). Output "
                           "filenames are derived from the split basename.")
    ap.add_argument("--task", type=str, default="joint",
                      choices=["joint", "yaw", "pitch"],
                      help="Which target to evaluate NLL on. joint=9-bin "
                           "yaw×pitch (default); yaw=3-bin yaw_bin_4 ∈ "
                           "{1,2,3}; pitch=3-bin height_bin ∈ {0,1,2}.")
    ap.add_argument("--averaging", type=str, default="sample",
                      choices=["sample", "category"],
                      help="How to average per-query metrics. 'sample': "
                           "mean over all eval queries (default; outputs "
                           "to per_sample/). 'category': mean within each "
                           "canonical_label, then mean across categories "
                           "(outputs to per_category/). T is tuned against "
                           "the same loss.")
    args = ap.parse_args()
    set_task(args.task)
    split_path = Path(args.split)
    suffix_split = split_path.stem.replace("split", "").lstrip("_")
    suffix_task = "" if args.task == "joint" else f"_{args.task}"
    suffix = suffix_split + suffix_task
    out_dir = (PER_CATEGORY_DIR if args.averaging == "category"
                 else PER_SAMPLE_DIR)
    if suffix:
        out_md = out_dir / f"results_{suffix}.md"
        out_raw = out_dir / f"results_raw_{suffix}.json"
    else:
        out_md = out_dir / "results.md"
        out_raw = out_dir / "results_raw.json"
    print(f"Split: {split_path}  •  Task: {args.task}  •  "
          f"B = {N_JOINT}  •  averaging = {args.averaging}")
    print(f"Output: {out_md}")

    print("Loading data...")
    set_df = pd.read_parquet(SET_PARQUET)
    train_df = pd.read_parquet(TRAIN_POOL)
    split = json.loads(split_path.read_text())

    model_runs = {
        "gemma-4-31b/blind":    RUNS_DIR / "A_blind_gemma-4-31b_M25.jsonl",
        "gemma-4-31b/sighted":  RUNS_DIR / "A_sighted_gemma-4-31b_M25.jsonl",
        "qwen3-vl-30b/blind":   RUNS_DIR / "A_blind_qwen3-vl-30b_M25.jsonl",
        "qwen3-vl-30b/sighted": RUNS_DIR / "A_sighted_qwen3-vl-30b_M25.jsonl",
        "qwen3.5-9b/blind":     RUNS_DIR / "A_blind_qwen3.5-9b_M25.jsonl",
        "qwen3.5-9b/sighted":   RUNS_DIR / "A_sighted_qwen3.5-9b_M25.jsonl",
    }
    print("Building model count distributions...")
    model_counts = {}
    model_M = {}
    for name, p in model_runs.items():
        runs, M_per = load_run_responses(p)
        d = {sid: counts_per_query(s) for sid, s in runs.items()}
        model_counts[name] = d
        model_M[name] = M_per
        avg_M = np.mean(list(M_per.values()))
        print(f"  {name}: {len(d)} queries, mean M={avg_M:.1f}")

    print("Building category prior (LOSO)...")
    cat_counts, cat_M = build_category_prior_loso(set_df, train_df)
    n_zero = sum(1 for v in cat_M.values() if v == 0)
    print(f"  Category prior: {len(cat_counts)} queries, "
          f"{n_zero} with no LOSO support (fall back to uniform)")

    # Floored NLL (eval set only, T=1)
    print("Computing floored NLL...")
    floored = {}
    for name, d in model_counts.items():
        floored[name] = floored_nll(d, model_M[name], set_df, split,
                                       averaging=args.averaging)
    floored["Category prior"] = floored_nll(cat_counts, cat_M, set_df, split,
                                                averaging=args.averaging)

    # Primary table (α=0.1)
    print(f"\nPrimary evaluation at α={ALPHA_DEFAULT}...")
    rows = []
    for name, d in model_counts.items():
        r = evaluate_model(name, d, model_M[name], set_df, split,
                              ALPHA_DEFAULT, averaging=args.averaging)
        r["floored_nll"] = floored[name]
        rows.append(r)
    r = evaluate_model("Category prior", cat_counts, cat_M, set_df, split,
                          ALPHA_DEFAULT, averaging=args.averaging)
    r["floored_nll"] = floored["Category prior"]
    rows.append(r)
    rows.append({
        "name": "Uniform",
        "eval_acc": 1.0 / N_JOINT,
        "floored_nll": math.log(N_JOINT),
        "eval_nll_smooth": math.log(N_JOINT),
        "eval_nll_calib": math.log(N_JOINT),
        "eval_nll_oracle": math.log(N_JOINT),
        "delta_oracle": 0.0,
        "T_star": float("nan"), "T_oracle": float("nan"),
        "eval_entropy_pre": math.log(N_JOINT),
        "eval_entropy_norm_pre": 1.0,
        "eval_entropy_post": math.log(N_JOINT),
        "eval_entropy_norm_post": 1.0,
        "n_cal": 0, "n_eval": len(split["eval_sample_ids"]),
    })

    # Sensitivity sweep
    print(f"\nSensitivity sweep over α ∈ {ALPHAS}...")
    sens = {}
    for alpha in ALPHAS:
        per_alpha = {}
        for name, d in model_counts.items():
            per_alpha[name] = evaluate_model(name, d, model_M[name], set_df,
                                                 split, alpha,
                                                 averaging=args.averaging)
        per_alpha["Category prior"] = evaluate_model(
            "Category prior", cat_counts, cat_M, set_df, split, alpha,
            averaging=args.averaging)
        sens[alpha] = per_alpha
        print(f"  α={alpha} done")

    # Write results
    order = (["Uniform", "Category prior"] + list(model_runs.keys()))
    rows_by = {r["name"]: r for r in rows}
    with open(out_md, "w") as f:
        f.write("# Calibration results (auto-generated by calibrate_T.py)\n\n")
        f.write(f"**Task:** `{TASK}` — {N_JOINT}-bin target. "
                  f"Uniform NLL = log {N_JOINT} = {math.log(N_JOINT):.3f}.\n\n")
        if args.averaging == "category":
            n_cats = rows[0].get("n_eval_categories", "?")
            f.write(f"**Averaging:** `category` — NLL/acc/entropy meaned "
                      f"within each `canonical_label`, then meaned across "
                      f"the {n_cats} eval categories. T tuned against the "
                      f"same per-category loss.\n\n")
        else:
            f.write("**Averaging:** `sample` — mean over all eval queries.\n\n")
        f.write(f"Calibration set: **{len(split['calibration_sample_ids'])}** "
                  f"queries from {split['calibration_participants']}.\n")
        f.write(f"Eval set: **{len(split['eval_sample_ids'])}** "
                  f"queries from {split['eval_participants']}.\n\n")

        # §1
        f.write(f"## §1 Primary table (α = {ALPHA_DEFAULT})\n\n")
        f.write("`δ_oracle = NLL_eval(T*_cal) − NLL_eval(T*_oracle)` is the "
                  "extra NLL paid by using the calibration-tuned T instead "
                  "of the oracle-tuned T on the eval set. By construction "
                  "≥ 0; closer to 0 = calibration generalises better.  \n")
        f.write("`H_norm = H / log B` ∈ [0, 1] — uniform → 1, "
                  "deterministic → 0. *pre* = at T=1 (smoothed only); "
                  "*post* = at T=T* (calibrated).  \n")
        f.write("**Bold** = best in column; <u>underline</u> = second best. "
                  "Acc: higher better. NLLs and δ_oracle: lower better. "
                  "T*, Oracle T, H_norm: not ranked (informational).\n\n")

        # Build value lists for ranking
        all_acc = [r["eval_acc"] for r in rows]
        all_floored = [r["floored_nll"] for r in rows]
        all_smooth = [r["eval_nll_smooth"] for r in rows]
        all_calib = [r["eval_nll_calib"] for r in rows]
        all_delta = [r["delta_oracle"] for r in rows]

        def rank_markup(s_fmt, v, all_vs, lower_better=True):
            if isinstance(v, float) and math.isnan(v):
                return s_fmt
            valid = sorted({x for x in all_vs
                              if not (isinstance(x, float) and math.isnan(x))},
                            reverse=not lower_better)
            if not valid:
                return s_fmt
            if v == valid[0]:
                return f"**{s_fmt}**"
            if len(valid) > 1 and v == valid[1]:
                return f"<u>{s_fmt}</u>"
            return s_fmt

        f.write("| Model | Acc | Floored NLL (eval) | Smooth NLL | "
                  "Calibrated NLL | T* | H_norm pre | H_norm post | "
                  "Oracle T | δ_oracle |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for name in order:
            r = rows_by[name]
            def _fmt(x):
                if isinstance(x, float):
                    if math.isnan(x):
                        return "—"
                    if math.isinf(x):
                        return "∞"
                return f"{x:.3f}"
            tstar = _fmt(r["T_star"])
            torac = _fmt(r["T_oracle"])
            acc_s = rank_markup(f"{r['eval_acc']:.1%}", r["eval_acc"],
                                  all_acc, lower_better=False)
            flo_s = rank_markup(f"{r['floored_nll']:.3f}", r["floored_nll"],
                                  all_floored)
            smo_s = rank_markup(f"{r['eval_nll_smooth']:.3f}",
                                  r["eval_nll_smooth"], all_smooth)
            cal_s = rank_markup(f"{r['eval_nll_calib']:.3f}",
                                  r["eval_nll_calib"], all_calib)
            del_s = rank_markup(f"{r['delta_oracle']:.4f}", r["delta_oracle"],
                                  all_delta)
            f.write(f"| {name} | {acc_s} | {flo_s} | {smo_s} | {cal_s} | "
                      f"{tstar} | {r['eval_entropy_norm_pre']:.3f} | "
                      f"{r['eval_entropy_norm_post']:.3f} | "
                      f"{torac} | {del_s} |\n")

        # §2
        f.write(f"\n## §2 Sensitivity sweep (α ∈ {ALPHAS})\n\n")
        f.write("### Tuned T*\n\n")
        f.write("| Model | " + " | ".join(f"α={a}" for a in ALPHAS) + " |\n")
        f.write("|" + "---|" * (len(ALPHAS) + 1) + "\n")
        for name in (list(model_runs.keys()) + ["Category prior"]):
            cells = [f"{sens[a][name]['T_star']:.3f}" for a in ALPHAS]
            f.write("| " + name + " | " + " | ".join(cells) + " |\n")
        f.write("\n### Eval Calibrated NLL  "
                  "(**bold** = best in column, <u>underline</u> = second)\n\n")
        f.write("| Model | " + " | ".join(f"α={a}" for a in ALPHAS) + " |\n")
        f.write("|" + "---|" * (len(ALPHAS) + 1) + "\n")
        sens_models = list(model_runs.keys()) + ["Category prior"]
        for name in sens_models:
            cells = []
            for a in ALPHAS:
                col_vals = [sens[a][m]["eval_nll_calib"] for m in sens_models]
                v = sens[a][name]["eval_nll_calib"]
                cells.append(rank_markup(f"{v:.3f}", v, col_vals))
            f.write("| " + name + " | " + " | ".join(cells) + " |\n")

        # §3 — per-category breakdown (only meaningful when averaging=category)
        if args.averaging == "category":
            f.write("\n## §3 Per-(model, category) NLL_calib breakdown "
                      "(intermediate)\n\n")
            f.write("Each cell is the **within-category mean Calibrated "
                      "NLL** before being meaned across categories. "
                      "Categories are sorted by total eval count (most "
                      "populous first).\n\n")
            named_rows = [r for r in rows if r.get("per_category")]
            if named_rows:
                # Build category list: union across models, ordered by
                # total n in the first row that has them.
                first = named_rows[0]
                cats_sorted = sorted(first["per_category"].keys(),
                                       key=lambda c: -first["per_category"][c]["n"])
                f.write("| Category (n) | "
                          + " | ".join(r["name"] for r in named_rows) + " |\n")
                f.write("|---|" + "---:|" * len(named_rows) + "\n")
                for c in cats_sorted:
                    n = first["per_category"][c]["n"]
                    cells = []
                    for r in named_rows:
                        pc = r["per_category"].get(c)
                        cells.append(f"{pc['nll_calib']:.3f}" if pc else "—")
                    f.write(f"| {c} ({n}) | " + " | ".join(cells) + " |\n")
            f.write("\n")

        f.write("\n## §4 Analysis\n\n")
        f.write("See [`analysis.md`](analysis.md).\n")

    print(f"\nWrote {out_md}")
    # Also dump raw numeric results for downstream use
    raw = {
        "split_path": str(split_path),
        "primary_alpha": ALPHA_DEFAULT,
        "primary_rows": rows,
        "sensitivity": {str(a): {n: v for n, v in sens[a].items()}
                         for a in ALPHAS},
        "split": {"n_cal": len(split["calibration_sample_ids"]),
                  "n_eval": len(split["eval_sample_ids"])},
    }
    out_raw.write_text(
        json.dumps(raw, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else str(x)))


if __name__ == "__main__":
    main()
