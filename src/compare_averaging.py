"""Side-by-side per-sample vs per-category metrics for the human
ensembles and all API models on the 100-sample human pool. Outputs:

  per_sample_vs_per_category_joint.md
  per_sample_vs_per_category_yaw.md
  per_sample_vs_per_category_height.md   (height = pitch task)

Each file also contains a **Calibrate-first-then-average-per-category**
section: T* is tuned on the per-sample calibration loss, then the
metrics (Acc / Floored NLL / Smooth NLL / Calib NLL) are aggregated
within each canonical_label and averaged across categories. This
sub-table is computed for both the 69-sample human pool (with humans
+ APIs + baselines) and the 1383-sample full-setA eval (APIs +
baselines).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from calibrate_humans import (
    USERS_DIR, USERS_BK_DIR, MIN_COMPLETION, ALPHA, CAL_PIDS, ADMIN_NAME,
    SET_PARQUET, API_MODELS,
    load_user_distributions, load_api_counts_at_M,
    evaluate_vlm_at_M, build_ensemble_row_impl,
    task_b, task_gt, maybe_marginalize_to_yaw, smooth_for_B,
    split_sids,
)
from calibrate_T import (
    set_task, tune_T, tscale_log_p, EPS_FLOOR, CALIB_DIR, aggregate,
    TESTSET, TRAIN_POOL, build_category_prior_loso,
)

POOL_STATE_PATH = TESTSET / "data" / "human_eval" / "pool_state.json"
SETA_FULL_PARQUET = TESTSET / "data" / "setA_extended_filtered.parquet"

# Filename suffix per task (user requested "height" for pitch task)
TASK_FILENAME = {"joint": "joint", "yaw": "yaw", "pitch": "height"}


def fmt(x, prec=3):
    if isinstance(x, float):
        if math.isnan(x):
            return "—"
        if math.isinf(x):
            return "∞"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    return f"{x:.{prec}f}"


def make_uniform_row(task: str, n_eval) -> dict:
    """Analytic uniform row — task-aware so log B matches the bin count.
    Mode-invariant: per-sample and cat-avg metrics are identical."""
    B = task_b(task)
    return {
        "name": "Uniform",
        "T_star": float("nan"), "T_oracle": float("nan"),
        "eval_acc": 1.0 / B,
        "floored_nll": math.log(B),
        "eval_nll_smooth": math.log(B),
        "eval_nll_calib": math.log(B),
        "eval_nll_oracle": math.log(B),
        "delta_oracle": 0.0,
        "eval_entropy_norm_pre": 1.0,
        "eval_entropy_norm_post": 1.0,
        "n_cal": 0, "n_eval": n_eval,
        "eval_acc_cat_avg": 1.0 / B,
        "floored_nll_cat_avg": math.log(B),
        "eval_nll_smooth_cat_avg": math.log(B),
        "eval_nll_calib_cat_avg": math.log(B),
    }


def make_category_prior_row(task: str, averaging: str,
                                set_df: pd.DataFrame, split: dict,
                                train_df: pd.DataFrame) -> dict | None:
    """LOSO category-prior counts → smoothed → T-tuned, both averaging
    modes. Counts always built in 9-bin form, then marginalised inside
    evaluate_vlm_at_M for yaw/pitch."""
    set_task("joint")  # produce 9-bin counts regardless of target task
    cat_counts, cat_M = build_category_prior_loso(set_df, train_df)
    set_task(task)  # restore target task before evaluation
    r = evaluate_vlm_at_M(cat_counts, cat_M, set_df, split, ALPHA,
                              task=task, averaging=averaging)
    if r is None:
        return None
    r["name"] = "Category prior"
    return r


def collect_rows(task: str, averaging: str, set_df: pd.DataFrame,
                  files: dict, pool_sids: set):
    """Return {ensemble_M7, ensemble_M6_no_admin, api: {(name, M): row}}
    for the given task and averaging mode."""
    set_task(task)

    annotator_joint_idx: dict[str, dict[str, int]] = {}

    for upath in files.values():
        rec = json.loads(upath.read_text())
        n_single = sum(1 for v in (rec.get("q1_single_answers") or {}).values()
                          if "yaw_bin_4" in v)
        n_probs = sum(1 for v in (rec.get("q1_probs_answers") or {}).values()
                         if "joint_probs" in v)
        if n_single < MIN_COMPLETION and n_probs < MIN_COMPLETION:
            continue
        for spec in load_user_distributions(upath):
            # Mirror calibrate_humans.py: skip a spec whose own kind has
            # < MIN_COMPLETION answers. This is what makes (single from
            # probs) win over a sparse (single) for partial-single users.
            n = (n_probs if spec["kind"] in ("probs", "derived_single")
                 else n_single)
            if n < MIN_COMPLETION:
                continue
            if spec["kind"] in ("single", "derived_single"):
                joints = {sid: int(np.argmax(v))
                            for sid, v in spec["dist"].items()}
                annotator_joint_idx.setdefault(spec["name"], joints)

    ensemble_all = build_ensemble_row_impl(
        annotator_joint_idx, f"human:ensemble M={len(annotator_joint_idx)}",
        task, averaging, set_df, ALPHA)
    no_admin = {k: v for k, v in annotator_joint_idx.items()
                  if k.lower() != ADMIN_NAME}
    ensemble_no_admin = build_ensemble_row_impl(
        no_admin,
        f"human:ensemble (no admin) M={len(no_admin)}",
        task, averaging, set_df, ALPHA)

    # APIs at M=20 and M=6 on the same scene-disjoint pool split.
    api_split = {
        "calibration_sample_ids": sorted(
            s for s in pool_sids
            if s in set_df.index
            and set_df.loc[s, "participant_id"] in CAL_PIDS),
        "eval_sample_ids": sorted(
            s for s in pool_sids
            if s in set_df.index
            and set_df.loc[s, "participant_id"] not in CAL_PIDS),
    }
    api_rows = {}
    api_counts = {}
    for aname, apath in API_MODELS.items():
        if not apath.exists():
            continue
        for M_target in (20, 6):
            counts, M_per = load_api_counts_at_M(
                apath, M_target, sids_filter=pool_sids)
            if not counts:
                continue
            api_counts[(aname, M_target)] = (counts, M_per)
            r = evaluate_vlm_at_M(counts, M_per, set_df, api_split,
                                      ALPHA, task=task, averaging=averaging)
            if r is None:
                continue
            r["name"] = f"{aname} (M={M_target})"
            api_rows[(aname, M_target)] = r

    return {
        "ensemble_M7": ensemble_all,
        "ensemble_M6_no_admin": ensemble_no_admin,
        "api": api_rows,
        "api_counts": api_counts,
        "annotator_joint_idx": annotator_joint_idx,
        "api_split": api_split,
    }


def collect_api_rows_full_setA(task: str, averaging: str,
                                  setA_df: pd.DataFrame):
    """Compute API rows on the full setA (~1951 queries) under the
    same scene-disjoint 30/70 split as the human pool. Returns
    {(api_name, M_target): row}. Only includes APIs whose JSONL fully
    covers setA (otherwise eval coverage is ambiguous)."""
    set_task(task)
    setA_sids = set(setA_df["sample_id"])
    setA_split = {
        "calibration_sample_ids": sorted(
            setA_df.loc[setA_df["participant_id"].isin(CAL_PIDS), "sample_id"]),
        "eval_sample_ids": sorted(
            setA_df.loc[~setA_df["participant_id"].isin(CAL_PIDS), "sample_id"]),
    }
    rows = {}
    for aname, apath in API_MODELS.items():
        if not apath.exists():
            continue
        for M_target in (20, 6):
            counts, M_per = load_api_counts_at_M(
                apath, M_target, sids_filter=setA_sids)
            if not counts:
                continue
            covered = len(counts) / len(setA_sids)
            if covered < 0.99:
                # Skip partial coverage so the eval baseline is consistent.
                print(f"  skip {aname} M={M_target} on full setA — "
                        f"covers {len(counts)}/{len(setA_sids)} = "
                        f"{covered:.1%}")
                continue
            r = evaluate_vlm_at_M(counts, M_per, setA_df, setA_split,
                                      ALPHA, task=task, averaging=averaging)
            if r is None:
                continue
            r["name"] = f"{aname} (M={M_target})"
            rows[(aname, M_target)] = r
    return rows, setA_split


def compute_kl_jsd_block(task: str, set_df: pd.DataFrame,
                            s_collected: dict, train_df):
    """Per-query, both-side KL plus symmetric JSD vs the human ensemble
    (M=N) on the human pool. Distributions are the **smoothed (T=1)**
    Dirichlet plug-ins — no temperature calibration. Per-query KL/JSD
    therefore depend only on smoothing; aggregation (`_s` per-sample
    mean vs `_c` per-category mean) is the only thing that differs
    between the two columns. Returns ordered list of dicts:
    [{name, KL_hm_s, KL_hm_c, KL_mh_s, KL_mh_c, JSD_s, JSD_c}]."""
    set_task(task)
    B = task_b(task)
    api_split = s_collected["api_split"]
    eval_sids = api_split["eval_sample_ids"]
    sid_to_cat = {r["sample_id"]: r["canonical_label"]
                    for _, r in set_df.iterrows()}

    # ── Build ensemble counts per sid (M=N) ─────────────────────────
    annotator_joint_idx = s_collected["annotator_joint_idx"]
    if not annotator_joint_idx:
        return []
    all_sids = set.union(*[set(d.keys())
                              for d in annotator_joint_idx.values()])
    sid_to_counts_h, sid_to_M_h = {}, {}
    for sid in all_sids:
        counts = np.zeros(9, dtype=np.float64)
        n = 0
        for d in annotator_joint_idx.values():
            if sid in d:
                counts[d[sid]] += 1
                n += 1
        if n > 0:
            sid_to_counts_h[sid] = counts
            sid_to_M_h[sid] = n

    # ── Build (no-admin) ensemble counts per sid (M=N−1) ────────────
    no_admin_idx = {k: v for k, v in annotator_joint_idx.items()
                      if k.lower() != ADMIN_NAME}
    sid_to_counts_na, sid_to_M_na = {}, {}
    if no_admin_idx:
        union_na = set.union(*[set(d.keys()) for d in no_admin_idx.values()])
        for sid in union_na:
            counts = np.zeros(9, dtype=np.float64)
            n = 0
            for d in no_admin_idx.values():
                if sid in d:
                    counts[d[sid]] += 1
                    n += 1
            if n > 0:
                sid_to_counts_na[sid] = counts
                sid_to_M_na[sid] = n

    # ── Build category prior counts (always 9-bin, marginalised on use) ─
    set_task("joint")
    cat_counts9, cat_M = build_category_prior_loso(set_df, train_df)
    set_task(task)

    def per_query_p_smoothed(counts_d, M_per):
        """Smoothed (T=1) Dirichlet plug-in for each eval sid; no
        temperature scaling. Returns dict[sid] -> p (B-vec)."""
        out = {}
        for sid in eval_sids:
            if sid not in counts_d:
                continue
            c = maybe_marginalize_to_yaw(counts_d[sid], task)
            out[sid] = smooth_for_B(c, ALPHA, M_per[sid], B)
        return out

    def kl(p, q, eps=1e-12):
        p = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0)
        q = np.clip(np.asarray(q, dtype=np.float64), eps, 1.0)
        p = p / p.sum()
        q = q / q.sum()
        return float(np.sum(p * (np.log(p) - np.log(q))))

    def kl_jsd(p, q):
        m = 0.5 * (np.asarray(p) + np.asarray(q))
        return kl(p, q), kl(q, p), 0.5 * (kl(p, m) + kl(q, m))

    def aggregate_per_sid(values_by_sid: dict, mode: str):
        if not values_by_sid:
            return float("nan")
        if mode == "sample":
            return float(np.mean(list(values_by_sid.values())))
        by_cat = {}
        for sid, v in values_by_sid.items():
            by_cat.setdefault(sid_to_cat[sid], []).append(v)
        return float(np.mean([float(np.mean(vs)) for vs in by_cat.values()]))

    # ── Reference: M=N ensemble (smoothed only, no T) ───────────────
    p_h = per_query_p_smoothed(sid_to_counts_h, sid_to_M_h)

    def compare_one(p_m: dict):
        kl_hm, kl_mh, jsd_v = {}, {}, {}
        for sid in p_h.keys() & p_m.keys():
            a, b, j = kl_jsd(p_h[sid], p_m[sid])
            kl_hm[sid] = a
            kl_mh[sid] = b
            jsd_v[sid] = j
        return {
            "KL_hm_s": aggregate_per_sid(kl_hm, "sample"),
            "KL_hm_c": aggregate_per_sid(kl_hm, "category"),
            "KL_mh_s": aggregate_per_sid(kl_mh, "sample"),
            "KL_mh_c": aggregate_per_sid(kl_mh, "category"),
            "JSD_s": aggregate_per_sid(jsd_v, "sample"),
            "JSD_c": aggregate_per_sid(jsd_v, "category"),
        }

    rows = []

    # Uniform: p_m = full(B, 1/B) for every eval sid.
    p_uniform = np.full(B, 1.0 / B)
    p_m_uni = {sid: p_uniform.copy() for sid in eval_sids}
    d = compare_one(p_m_uni)
    d["name"] = "Uniform"
    rows.append(d)

    # No-admin ensemble row dropped per spec (M=10 humans only)

    # Category prior
    if cat_M:
        p_m = per_query_p_smoothed(cat_counts9, cat_M)
        d = compare_one(p_m)
        d["name"] = "Category prior"
        rows.append(d)

    # API rows in canonical order
    api_counts = s_collected.get("api_counts", {})
    for kk in sorted(api_counts.keys(), key=lambda x: (x[0], -x[1])):
        if kk not in s_collected["api"]:
            continue
        counts_d, M_per = api_counts[kk]
        p_m = per_query_p_smoothed(counts_d, M_per)
        d = compare_one(p_m)
        d["name"] = s_collected["api"][kk]["name"]
        rows.append(d)

    return rows


# ── Main ──────────────────────────────────────────────────────────────
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--eval-set", "--eval_set",
        dest="eval_set",
        choices=["full", "100", "all"],
        default="all",
        help=(
            "Which subset(s) to write tables for. 'full' = full setA "
            "(~1383 eval queries, APIs only); '100' = human pool "
            "(~69 eval queries, humans + APIs); 'all' (default) = both. "
            "Note: '300' is not a valid choice for the calibrated-NLL "
            "tables — calibration requires the cal/eval split which is "
            "defined on the 100-pool and the full setA."
        ),
    )
    args = ap.parse_args()
    do_pool = args.eval_set in ("100", "all")
    do_setA = args.eval_set in ("full", "all")

    set_df = pd.read_parquet(SET_PARQUET).set_index("sample_id", drop=False)
    setA_df = pd.read_parquet(SETA_FULL_PARQUET)
    train_df = pd.read_parquet(TRAIN_POOL)

    files: dict[str, Path] = {}
    if USERS_DIR.exists():
        for f in sorted(USERS_DIR.glob("*.json")):
            files.setdefault(f.stem, f)
    if USERS_BK_DIR.exists():
        for f in sorted(USERS_BK_DIR.glob("*.json")):
            files.setdefault(f.stem, f)

    pool_sids = set(json.loads(POOL_STATE_PATH.read_text())["active"])

    # Restrict the human-pool set_df to the 100 active sids — this is
    # what the category prior / uniform / split computations operate on.
    pool_set_df = set_df.loc[set_df["sample_id"].isin(pool_sids)].copy()

    # API split (same scene-disjoint rule as humans)
    api_split = {
        "calibration_sample_ids": sorted(
            s for s in pool_sids
            if s in set_df.index
            and set_df.loc[s, "participant_id"] in CAL_PIDS),
        "eval_sample_ids": sorted(
            s for s in pool_sids
            if s in set_df.index
            and set_df.loc[s, "participant_id"] not in CAL_PIDS),
    }

    if not (do_pool and do_setA):
        print(f"NOTE: --eval-set={args.eval_set!r} is accepted but currently "
              "always renders BOTH human-pool and full-setA tables (their "
              "rendering is interlocked). Per-subset rendering can be added "
              "later; outputs are identical to default for now.")

    for task in ("joint", "yaw", "pitch"):
        print(f"\n=== Task: {task} ===")
        s = collect_rows(task, "sample", set_df, files, pool_sids)
        c = collect_rows(task, "category", set_df, files, pool_sids)
        # Full setA (~1951 queries) — APIs only.
        api_setA_s, setA_split = collect_api_rows_full_setA(
            task, "sample", setA_df)
        api_setA_c, _ = collect_api_rows_full_setA(
            task, "category", setA_df)

        # Baselines: uniform + category prior, both averaging modes,
        # for the human pool eval and the full-setA eval.
        set_task(task)  # ensure task-aware B for uniform
        n_eval_pool = len(api_split["eval_sample_ids"])
        n_eval_setA = len(setA_split["eval_sample_ids"])
        unif_pool = make_uniform_row(task, n_eval_pool)
        unif_setA = make_uniform_row(task, n_eval_setA)
        cat_pool_s = make_category_prior_row(
            task, "sample", pool_set_df, api_split, train_df)
        cat_pool_c = make_category_prior_row(
            task, "category", pool_set_df, api_split, train_df)
        cat_setA_s = make_category_prior_row(
            task, "sample", setA_df, setA_split, train_df)
        cat_setA_c = make_category_prior_row(
            task, "category", setA_df, setA_split, train_df)

        # Build the markdown
        fname = f"per_sample_vs_per_category_{TASK_FILENAME[task]}.md"
        out = CALIB_DIR / fname
        B = task_b(task)
        with open(out, "w") as f:
            f.write(f"# Per-sample vs Per-category — task `{task}` "
                      f"(B = {B}, log B = {math.log(B):.3f})\n\n")

            # ── Method (read once, then refer back) ───────────────────
            n_cal_pool = len(api_split["calibration_sample_ids"])
            n_cal_setA = len(setA_split["calibration_sample_ids"])
            f.write("## Method\n\n")
            f.write("Read this section once; every subsequent table refers "
                      "back here for column definitions and the recipe used "
                      "to compute each cell.\n\n")
            f.write("### Eval subsets\n\n")
            f.write(
                f"- **Human pool** (the 100-sample annotation pool, "
                f"`pool_state.json::active`): scene-disjoint 30/70 split "
                f"by participant → cal n = {n_cal_pool}, eval n = "
                f"{n_eval_pool}. The same eval subset is used for human "
                f"ensembles and API models — true head-to-head.\n")
            f.write(
                f"- **Full setA** (`setA_extended_filtered.parquet`, "
                f"{len(setA_df)} queries): same scene-disjoint rule by "
                f"participant → cal n = {n_cal_setA}, eval n = "
                f"{n_eval_setA}. APIs only (humans did not label the full "
                f"set). API rows shown at M=20 only on full setA (M=6 is "
                f"reserved for the budget-matched human-pool comparison).\n\n")
            f.write("Cal participants = `{P07, P19, P22, P31, P33, P35}`; "
                      "eval participants = `{P01, P02, P06, P08, P09}`.\n\n")
            f.write("### Calibration recipe\n\n")
            f.write(
                f"For each (model, query) we observe `M` parsed responses "
                f"(M_target ∈ {{20, 6}}; humans contribute M=1 single "
                f"picks or M=100 probability allocations). Counts are "
                f"smoothed with Dirichlet plug-in:\n\n"
                f"```\n"
                f"  p_i = (n_i + α) / (M + B·α),   α = 0.1\n"
                f"```\n\n"
                f"Temperature scaling is parameterised by inverse "
                f"temperature β ∈ [0, 100]:\n\n"
                f"```\n"
                f"  z_i  = log p_i\n"
                f"  p_i^T = softmax(β · z),       T = 1/β\n"
                f"```\n\n"
                f"`β = 0` ↔ `T = ∞` ↔ uniform predictive (reported as "
                f"`∞`); `β = 1` ↔ `T = 1` ↔ identity. We minimise the "
                f"calibration NLL via `scipy.optimize.minimize_scalar` "
                f"(bounded Brent, `xatol = 1e-6`).\n\n")
            f.write("### Three views of the same metric\n\n")
            f.write(
                "All three views use the same smoothing and the same "
                "log-prob inputs `log p`. They differ only in (i) what "
                "loss T* is tuned against and (ii) how the per-query "
                "metric is aggregated:\n\n"
                "1. **per-sample** — T* minimises `mean over cal queries "
                "of NLL`, the eval metric is `mean over eval queries of "
                "NLL`.\n"
                "2. **per-category** — T* minimises `mean over categories "
                "of (mean over cal queries within category) of NLL`, the "
                "eval metric uses the same nested aggregator on eval. T "
                "tuned and metric reported are aligned.\n"
                "3. **calibrate-first-then-average-per-category** — T* "
                "is the per-sample T* (= mode 1's T*), but the eval "
                "metric uses mode 2's nested-mean aggregator. Isolates "
                "*aggregation of the metric* from *aggregation of the "
                "calibration objective*.\n\n")
            f.write("### Per-query metrics (for the per-sample numerator)\n\n")
            f.write(
                "Let `j*` be the GT bin for query `q` and `B` the bin "
                "count. Per-query values that get aggregated:\n\n"
                "- **Acc** — `1[argmax_i log p_{q,i} == j*]` (T-invariant, "
                "since softmax preserves argmax).\n"
                "- **Floored NLL** — `−log max(n_{j*}/M, ε) / Σ ...` "
                "(empirical with ε-floor; T-independent, M-sensitive).\n"
                "- **Smooth NLL** — `−log p_{q,j*}` at `T = 1` (Dirichlet "
                "plug-in only).\n"
                "- **Calib NLL** — `−log p_{q,j*}^T` at the chosen T*.\n"
                "- **H_pre / H_post** — `−Σ_i p_{q,i} log p_{q,i}` at "
                "T=1 / T=T*, normalised by `log B` so 0 = peaky and "
                "1 = uniform.\n"
                "- **Oracle T**, **δ_oracle** — `T_oracle` minimises NLL "
                "directly on eval (so calibration over-fits whenever "
                "`δ_oracle = NLL(T*) − NLL(T_oracle) > 0`).\n\n")
            f.write("### KL & JSD vs human ensemble\n\n")
            f.write(
                "Right after the human-pool main table there is a section "
                "comparing each row's predicted distribution against the "
                "**human ensemble (M=N)** distribution per query, using "
                "**smoothed (T = 1) Dirichlet plug-ins only — no "
                "temperature calibration**. We report both directed KL "
                "divergences plus the symmetric Jensen-Shannon "
                "divergence:\n\n"
                "- `KL(h‖m) = Σ_i p_h(i) · log(p_h(i)/p_m(i))` (forward).\n"
                "- `KL(m‖h) = Σ_i p_m(i) · log(p_m(i)/p_h(i))` (reverse).\n"
                "- `JSD(h, m) = ½·KL(p_h‖M) + ½·KL(p_m‖M)` with "
                "`M = ½·(p_h + p_m)`. Symmetric, bounded in [0, log 2].\n\n"
                "Per-query values depend only on smoothing; the `_s` and "
                "`_c` columns of the table differ in aggregation alone "
                "(per-sample mean vs mean-within-category-then-across-"
                "categories).\n\n")
            f.write("### Per-category NLL breakdown\n\n")
            f.write(
                "Below the main tables is a (category × model) breakdown. "
                "Each cell is the **within-category mean Calibrated NLL** "
                "under the per-sample T* (view 3 above). Reading the "
                "same row across columns compares models head-to-head on "
                "a single object category; reading the same column down "
                "lists the per-category contributions whose unweighted "
                "mean is the row's `Calib NLL` cell in the *Calibrate-"
                "first-then-average-per-category* tables.\n\n")
            f.write("### Category difficulty ranking\n\n")
            f.write(
                "After the breakdown there are two ranking views: an "
                "**Overall ranking** that averages each metric across "
                "all model rows (excluding Uniform) and sorts categories "
                "by ascending mean Calib NLL — useful for asking *which "
                "object categories are universally easier?* — and a "
                "**Per-model top-3** that lists, per model and per "
                "metric, the three easiest and three hardest categories "
                "*for that model*. Per-metric top-3 sub-tables cover "
                "Mode acc (easiest = highest), Calib NLL, Smooth NLL, "
                "and Floored NLL (easiest = lowest for the three NLL "
                "variants).\n\n")
            f.write(
                "Conventions: best-in-column markers are omitted because "
                "the eval subset is fixed within each table — direct "
                "numeric compare. `T = ∞` is shown when β collapses to "
                "≤ 1e-3 (predict uniform). Cells outside a row's eval "
                "support show `—`.\n\n")

            # Header
            cols = (["Model", "Mode", "n_eval", "Acc", "Floored NLL",
                       "Smooth NLL", "Calib NLL", "T*", "H_pre", "H_post",
                       "Oracle T", "δ_oracle"])
            f.write(f"## Main table — Human pool (n_eval = {n_eval_pool})\n\n")
            f.write(
                "Per-sample and per-category modes (views 1 and 2 in the "
                "**Method** section) for each row, on the human-pool eval "
                "subset. Uniform is mode-invariant (single row).\n\n")
            f.write("| " + " | ".join(cols) + " |\n")
            f.write("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")

            def write_row(row, mode_label, display_name=None):
                if row is None:
                    return
                name = display_name or row["name"]
                f.write(
                    f"| {name} | {mode_label} | {row['n_eval']} | "
                    f"{row['eval_acc']:.1%} | "
                    f"{fmt(row['floored_nll'])} | "
                    f"{fmt(row['eval_nll_smooth'])} | "
                    f"{fmt(row['eval_nll_calib'])} | "
                    f"{fmt(row['T_star'])} | "
                    f"{fmt(row.get('eval_entropy_norm_pre', float('nan')))} | "
                    f"{fmt(row.get('eval_entropy_norm_post', float('nan')))} | "
                    f"{fmt(row['T_oracle'])} | "
                    f"{fmt(row['delta_oracle'], prec=4)} |\n")

            # Baselines first (uniform is mode-invariant — written once).
            write_row(unif_pool, "—")
            if cat_pool_s and cat_pool_c:
                write_row(cat_pool_s, "sample")
                write_row(cat_pool_c, "category")

            # Ensembles
            for key in ("ensemble_M7",):  # (no admin) row dropped per spec
                row_s = s[key]; row_c = c[key]
                if row_s and row_c:
                    write_row(row_s, "sample")
                    write_row(row_c, "category")

            # APIs in stable order
            api_order = sorted(c["api"].keys(),
                                  key=lambda kk: (kk[0], -kk[1]))
            for kk in api_order:
                if kk in s["api"] and kk in c["api"]:
                    write_row(s["api"][kk], "sample")
                    write_row(c["api"][kk], "category")

            # ── KL & JSD vs human ensemble (human pool) ─────────────
            kl_jsd_rows = compute_kl_jsd_block(task, set_df, s, train_df)
            if kl_jsd_rows:
                ens_size = len(s["annotator_joint_idx"])
                f.write(f"\n## KL & JSD vs human ensemble "
                          f"(reference = `humans (ensemble) M={ens_size}`)\n\n")
                f.write(
                    "For each query in the human-pool eval subset "
                    f"(n_eval = {n_eval_pool}) we form `p_human` and "
                    "`p_model` as the **smoothed Dirichlet plug-in "
                    "(T = 1, no temperature calibration)** and compute "
                    "the two directed KL divergences plus the symmetric "
                    "JSD. Per-query KL/JSD therefore depend only on the "
                    "smoothing recipe; the `_s` vs `_c` columns differ "
                    "only in **aggregation**:\n\n"
                    "- `_s` columns: mean over the 69 eval queries.\n"
                    "- `_c` columns: mean within each `canonical_label` "
                    "first, then mean across categories.\n\n"
                    "- **KL(h‖m)** = `Σ_i p_h log(p_h/p_m)` — penalises "
                    "the model where it puts low mass on bins humans "
                    "favour.\n"
                    "- **KL(m‖h)** = `Σ_i p_m log(p_m/p_h)` — penalises "
                    "humans where they put low mass on bins the model "
                    "favours.\n"
                    "- **JSD** = `½·KL(p_h‖M) + ½·KL(p_m‖M)` with "
                    "`M = ½·(p_h + p_m)`. Symmetric, bounded in "
                    "[0, log 2 ≈ 0.693] (nats).\n\n")
                f.write("| Model | KL(h‖m)_s | KL(h‖m)_c | "
                          "KL(m‖h)_s | KL(m‖h)_c | "
                          "JSD_s | JSD_c |\n")
                f.write("|---|---:|---:|---:|---:|---:|---:|\n")
                for r in kl_jsd_rows:
                    f.write(
                        f"| {r['name']} | "
                        f"{fmt(r['KL_hm_s'])} | {fmt(r['KL_hm_c'])} | "
                        f"{fmt(r['KL_mh_s'])} | {fmt(r['KL_mh_c'])} | "
                        f"{fmt(r['JSD_s'])} | {fmt(r['JSD_c'])} |\n")

            # ── API rows on the full setA (M=20 only) ────────────
            if api_setA_s:
                f.write(f"\n## Main table — Full setA "
                          f"(n_eval = {n_eval_setA})\n\n")
                f.write(
                    "APIs at **M=20 only** on full setA (M=6 reserved "
                    "for the budget-matched human-pool comparison "
                    "above). See **Method** for column definitions.\n\n")
                f.write("| " + " | ".join(cols) + " |\n")
                f.write("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
                # Baselines first
                write_row(unif_setA, "—")
                if cat_setA_s and cat_setA_c:
                    write_row(cat_setA_s, "sample")
                    write_row(cat_setA_c, "category")
                api_order_setA_M20 = sorted(
                    [kk for kk in api_setA_c if kk[1] == 20],
                    key=lambda kk: kk[0])
                for kk in api_order_setA_M20:
                    if kk in api_setA_s and kk in api_setA_c:
                        write_row(api_setA_s[kk], "sample")
                        write_row(api_setA_c[kk], "category")

            # ── Calibrate-first-then-average-per-category ───────────────
            f.write("\n## Calibrate-first-then-average-per-category\n\n")
            f.write(
                "View 3 from the **Method** section: T* tuned per-sample, "
                "then every metric averaged within `canonical_label` and "
                "finally across categories.\n\n")

            cat_avg_cols = ["Model", "n_eval", "Acc", "Floored NLL",
                              "Smooth NLL", "Calib NLL", "T*"]

            def write_cat_avg_row(row, display_name=None):
                if row is None:
                    return
                name = display_name or row["name"]
                f.write(
                    f"| {name} | {row['n_eval']} | "
                    f"{row['eval_acc_cat_avg']:.1%} | "
                    f"{fmt(row['floored_nll_cat_avg'])} | "
                    f"{fmt(row['eval_nll_smooth_cat_avg'])} | "
                    f"{fmt(row['eval_nll_calib_cat_avg'])} | "
                    f"{fmt(row['T_star'])} |\n")

            f.write(f"### Human pool (n_eval = {n_eval_pool})\n\n")
            f.write("| " + " | ".join(cat_avg_cols) + " |\n")
            f.write("|---|---:|---:|---:|---:|---:|---:|\n")
            write_cat_avg_row(unif_pool)
            if cat_pool_s:
                write_cat_avg_row(cat_pool_s)
            for key in ("ensemble_M7",):  # (no admin) row dropped per spec
                if s.get(key):
                    write_cat_avg_row(s[key])
            for kk in api_order:
                if kk in s["api"]:
                    write_cat_avg_row(s["api"][kk])

            f.write(f"\n### Full setA (n_eval = {n_eval_setA}, M=20 only)\n\n")
            f.write("| " + " | ".join(cat_avg_cols) + " |\n")
            f.write("|---|---:|---:|---:|---:|---:|---:|\n")
            write_cat_avg_row(unif_setA)
            if cat_setA_s:
                write_cat_avg_row(cat_setA_s)
            api_order_setA_M20 = sorted(
                [kk for kk in api_setA_s if kk[1] == 20],
                key=lambda kk: kk[0])
            for kk in api_order_setA_M20:
                if kk in api_setA_s:
                    write_cat_avg_row(api_setA_s[kk])

            # ── Per-category NLL breakdown ──────────────────────────
            f.write("\n## Per-category NLL breakdown\n\n")
            f.write(
                "See **Method → Per-category NLL breakdown** for what "
                "each cell means and how it relates to the rows above. "
                "Categories sorted by total eval count (descending).\n\n")

            def short_name(name: str) -> str:
                return name.replace(" (sighted, API)", "").replace(
                    "human:", "")

            def write_breakdown(rows_dict: dict, header: str):
                """rows_dict: ordered dict-like of name -> row dict."""
                # Keep only rows with a per_cat_breakdown
                kept = [(n, r) for n, r in rows_dict.items()
                          if r and r.get("per_cat_breakdown")]
                if not kept:
                    return
                # Build category list = union, sorted by total n
                # (using counts from the first row; all rows on the same
                # eval set should have the same category set).
                ref_counts = {}
                for _, r in kept:
                    for c, d in r["per_cat_breakdown"].items():
                        ref_counts[c] = max(ref_counts.get(c, 0), d["n"])
                cats_sorted = sorted(ref_counts,
                                        key=lambda c: -ref_counts[c])
                f.write(f"### {header}\n\n")
                col_names = [short_name(n) for n, _ in kept]
                f.write("| Category (n) | " + " | ".join(col_names) + " |\n")
                f.write("|---|" + "---:|" * len(kept) + "\n")
                for cat in cats_sorted:
                    n = ref_counts[cat]
                    cells = []
                    for _, r in kept:
                        d = r["per_cat_breakdown"].get(cat)
                        cells.append(fmt(d["nll_calib"]) if d else "—")
                    f.write(f"| {cat} ({n}) | " + " | ".join(cells) + " |\n")
                f.write("\n")

            # Human pool sub-table (humans + APIs + Cat prior, both Ms)
            human_pool_rows = {}
            if cat_pool_s:
                human_pool_rows[cat_pool_s["name"]] = cat_pool_s
            for key in ("ensemble_M7",):  # (no admin) row dropped per spec
                if s.get(key):
                    human_pool_rows[s[key]["name"]] = s[key]
            for kk in api_order:
                if kk in s["api"]:
                    human_pool_rows[s["api"][kk]["name"]] = s["api"][kk]
            write_breakdown(
                human_pool_rows,
                f"Human pool (n_eval = {n_eval_pool})")

            # Full setA sub-table (Cat prior + APIs at M=20 only)
            setA_rows = {}
            if cat_setA_s:
                setA_rows[cat_setA_s["name"]] = cat_setA_s
            for kk in api_order_setA_M20:
                if kk in api_setA_s:
                    setA_rows[api_setA_s[kk]["name"]] = api_setA_s[kk]
            write_breakdown(
                setA_rows,
                f"Full setA (n_eval = {n_eval_setA}, M=20 only)")

            # ── Category difficulty ranking ─────────────────────────
            f.write("\n## Category difficulty ranking\n\n")
            f.write(
                "Two views derived from the per-category breakdown above:\n\n"
                "1. **Overall ranking** — for each eval subset, "
                "categories are sorted by `mean Calib NLL across all "
                "model rows in that subset` (lowest = easiest). The "
                "other metrics (Acc, Smooth NLL, Floored NLL) are also "
                "averaged across models for the same category. Excludes "
                "Uniform (mode-invariant); includes Category prior, "
                "humans, and APIs.\n"
                "2. **Per-model top-3** — for each model, lists the "
                "three easiest and three hardest categories *for that "
                "model* by Calib NLL. The full sorted list per model is "
                "implicit in reading the model's column in the "
                "breakdown table above.\n\n")

            def overall_rank_table(rows_dict, header):
                kept = [(n, r) for n, r in rows_dict.items()
                          if r and r.get("per_cat_breakdown")]
                if not kept:
                    return
                # Aggregate per-category across models
                cat_stats = {}  # cat -> list of dicts
                cat_n = {}
                for _, r in kept:
                    for c, d in r["per_cat_breakdown"].items():
                        cat_stats.setdefault(c, []).append(d)
                        cat_n[c] = max(cat_n.get(c, 0), d["n"])
                rows_to_print = []
                for c, ds in cat_stats.items():
                    rows_to_print.append({
                        "cat": c,
                        "n": cat_n[c],
                        "mean_acc": float(np.mean([d["acc"] for d in ds])),
                        "mean_calib": float(np.mean([d["nll_calib"] for d in ds])),
                        "mean_smooth": float(np.mean([d["nll_smooth"] for d in ds])),
                        "mean_floored": float(np.mean([d["floored_nll"] for d in ds])),
                    })
                rows_to_print.sort(key=lambda r: r["mean_calib"])
                f.write(f"### Overall ranking — {header}\n\n")
                f.write("Sorted by **mean Calib NLL** ascending "
                          "(easiest = rank 1).\n\n")
                f.write("| Rank | Category (n) | mean Acc | mean Calib NLL | "
                          "mean Smooth NLL | mean Floored NLL |\n")
                f.write("|---:|---|---:|---:|---:|---:|\n")
                for i, rr in enumerate(rows_to_print, 1):
                    f.write(
                        f"| {i} | {rr['cat']} ({rr['n']}) | "
                        f"{rr['mean_acc']:.1%} | "
                        f"{fmt(rr['mean_calib'])} | "
                        f"{fmt(rr['mean_smooth'])} | "
                        f"{fmt(rr['mean_floored'])} |\n")
                f.write("\n")

            METRIC_SPECS = [
                ("acc", "Mode acc", False, lambda v: f"{v:.0%}"),
                ("nll_calib", "Calib NLL", True, lambda v: fmt(v)),
                ("nll_smooth", "Smooth NLL", True, lambda v: fmt(v)),
                ("floored_nll", "Floored NLL", True, lambda v: fmt(v)),
            ]

            def per_model_topk_one_metric(rows_dict, metric_key,
                                              metric_label,
                                              lower_better, fmt_v, k=3):
                kept = [(n, r) for n, r in rows_dict.items()
                          if r and r.get("per_cat_breakdown")]
                if not kept:
                    return
                direction = "lowest" if lower_better else "highest"
                inverse = "highest" if lower_better else "lowest"
                f.write(f"#### {metric_label}\n\n")
                f.write(f"Easiest = **{direction}** {metric_label}; "
                          f"hardest = {inverse}.\n\n")
                f.write(f"| Model | Easiest {k} ({metric_label}) | "
                          f"Hardest {k} ({metric_label}) |\n")
                f.write("|---|---|---|\n")
                for n, r in kept:
                    cats = sorted(r["per_cat_breakdown"].items(),
                                    key=lambda kv: kv[1][metric_key],
                                    reverse=not lower_better)
                    easiest = cats[:k]
                    hardest = cats[-k:][::-1]
                    e_str = ", ".join(f"{c} ({fmt_v(d[metric_key])})"
                                          for c, d in easiest)
                    h_str = ", ".join(f"{c} ({fmt_v(d[metric_key])})"
                                          for c, d in hardest)
                    f.write(f"| {short_name(n)} | {e_str} | {h_str} |\n")
                f.write("\n")

            def per_model_topk_table(rows_dict, header, k=3):
                kept = [(n, r) for n, r in rows_dict.items()
                          if r and r.get("per_cat_breakdown")]
                if not kept:
                    return
                f.write(f"### Per-model top-{k} — {header}\n\n")
                f.write(f"Per-model lists of the {k} **easiest** and {k} "
                          f"**hardest** categories *for that model*, "
                          f"shown for four metrics. The metric header "
                          f"says which direction is "
                          f"\"easy\".\n\n")
                for key, label, lo, fv in METRIC_SPECS:
                    per_model_topk_one_metric(
                        rows_dict, key, label, lo, fv, k=k)

            overall_rank_table(human_pool_rows,
                                  f"Human pool (n_eval = {n_eval_pool})")
            overall_rank_table(setA_rows,
                                  f"Full setA (n_eval = {n_eval_setA}, "
                                  f"M=20 only)")
            per_model_topk_table(human_pool_rows,
                                    f"Human pool (n_eval = {n_eval_pool})",
                                    k=3)
            per_model_topk_table(setA_rows,
                                    f"Full setA (n_eval = {n_eval_setA}, "
                                    f"M=20 only)",
                                    k=3)

        print(f"  Wrote {out}")


if __name__ == "__main__":
    main()
