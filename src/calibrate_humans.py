"""Calibrate every human annotator on the human pool and combine with VLMs.

Inputs:
  meta/testset/data/human_eval/users/<uid>.json
  meta/testset/data/human_eval/users_bk/<uid>.json

For each annotator:
  - q1_single_answers → one-hot 9-vector at the picked joint bin, M=1.
  - q1_probs_answers  → 9-vector of integer probs summing to 100, M=100.
  - q1_probs ALSO derives a "single from probs" row whose distribution is
    one-hot at the top bin (random tie-breaking, deterministic per sid).

Calibration recipe matches `calibrate_T.py`: α=0.1, scene-disjoint
participant split (cal = {P07, P19, P22, P31, P33, P35}), 1-D T tuning,
oracle T on eval, δ_oracle, H_norm pre/post.

Output:
  meta/metric/NLL/calibration/results_human_combined.md
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from calibrate_T import (
    N_JOINT, joint_index, smooth_from_counts, tscale_log_p, tune_T,
    aggregate, set_task, ALPHA_DEFAULT, EPS_FLOOR, CALIB_DIR, TESTSET,
    PITCH_TO_VBIN, DATA_DIR, PER_CATEGORY_DIR,
)

ALPHA = ALPHA_DEFAULT  # 0.1
M_SINGLE = 1
M_PROBS = 100

SET_PARQUET = (Path("/path/to/this/repo/release") /
                 "q1-bin-prediction" / "queries.parquet")
USERS_DIR = TESTSET / "data" / "human_eval" / "users"
USERS_BK_DIR = TESTSET / "data" / "human_eval" / "users_bk"
PER_SAMPLE_DIR = CALIB_DIR / "per_sample"
PER_SAMPLE_DIR.mkdir(exist_ok=True)
VLM_RAW = PER_SAMPLE_DIR / "results_raw_30_70.json"

CAL_PIDS = {"P07", "P19", "P22", "P31", "P33", "P35"}
ADMIN_NAME = "admin"  # filename for the no-admin ensemble exclusion

# Task config (joint = 9-bin yaw×height, yaw = 3-bin yaw only,
# pitch = 3-bin height only).
B_FOR_TASK = {"joint": 9, "yaw": 3, "pitch": 3}


def task_b(task: str) -> int:
    return B_FOR_TASK[task]


def task_gt(row, task: str) -> int:
    if task == "yaw":
        return int(row["yaw_bin_4"]) - 1
    if task == "pitch":
        return int(row["height_bin"])
    return joint_index(int(row["yaw_bin_4"]), int(row["height_bin"]))


def maybe_marginalize_to_yaw(counts9: np.ndarray, task: str) -> np.ndarray:
    """Marginalise 9-bin counts to 3-bin counts for yaw or pitch task.
    (Name kept for backward compat; now also handles pitch.)
    Index = (yaw_bin_4 - 1) * 3 + height_bin."""
    if task == "joint":
        return counts9
    if task == "yaw":
        return np.array([counts9[0] + counts9[1] + counts9[2],   # RIGHT
                         counts9[3] + counts9[4] + counts9[5],   # BACK
                         counts9[6] + counts9[7] + counts9[8]],  # LEFT
                        dtype=np.float64)
    # pitch: sum over yaw → height_bin = j % 3
    return np.array([counts9[0] + counts9[3] + counts9[6],   # BELOW
                     counts9[1] + counts9[4] + counts9[7],   # ON
                     counts9[2] + counts9[5] + counts9[8]],  # ABOVE
                    dtype=np.float64)


def smooth_for_B(counts: np.ndarray, alpha: float, M: float,
                  B: int) -> np.ndarray:
    """B-aware Dirichlet smoothing (avoids relying on calibrate_T's
    module-global N_JOINT)."""
    return (counts + alpha) / (M + B * alpha)

# VLM M=5 rows (subsampled from sampling_ablation_001 M=25 JSONLs on
# setA_300_filtered) for apples-to-apples comparison with the M=5
# human ensemble.
M_VLM_COMPARE = 5
VLM_SET_PARQUET = DATA_DIR / "setA_300_filtered.parquet"
VLM_SPLIT_30_70 = CALIB_DIR / "split_30_70.json"
VLM_RUNS_DIR = TESTSET / "exp" / "sampling_ablation_001" / "runs"
VLM_MODELS = {
    "gemma-4-31b/blind":    VLM_RUNS_DIR / "A_blind_gemma-4-31b_M25.jsonl",
    "gemma-4-31b/sighted":  VLM_RUNS_DIR / "A_sighted_gemma-4-31b_M25.jsonl",
    "qwen3-vl-30b/blind":   VLM_RUNS_DIR / "A_blind_qwen3-vl-30b_M25.jsonl",
    "qwen3-vl-30b/sighted": VLM_RUNS_DIR / "A_sighted_qwen3-vl-30b_M25.jsonl",
    "qwen3.5-9b/blind":     VLM_RUNS_DIR / "A_blind_qwen3.5-9b_M25.jsonl",
    "qwen3.5-9b/sighted":   VLM_RUNS_DIR / "A_sighted_qwen3.5-9b_M25.jsonl",
}


# ── API model JSONLs (different parsed format: yaw_bin/height as labels) ─
API_RUNS_DIR = Path("/path/to/this/repo/results/"
                       "setA_extention/records")
API_MODELS = {
    "gpt-5.4 (sighted, API)":
        API_RUNS_DIR / "sighted_gpt-5.4_M20.jsonl",
    "gemini-3-flash (sighted, API)":
        API_RUNS_DIR / "sighted_gemini-3-flash_M20.jsonl",
    "qwen3-vl-235b (sighted, API)":
        API_RUNS_DIR / "sighted_qwen3-vl-235b_M20.jsonl",
    "qwen3.5-397b-a17b (sighted, API)":
        API_RUNS_DIR / "sighted_qwen3.5-397b-a17b_M20.jsonl",
    "qwen3-vl-30b (sighted, API)":
        API_RUNS_DIR / "sighted_qwen3-vl-30b_M20.jsonl",
    "qwen3.5-35b-a3b (sighted, API)":
        API_RUNS_DIR / "sighted_qwen3.5-35b-a3b_M20.jsonl",
    "gemma-4-31b (sighted, API)":
        API_RUNS_DIR / "sighted_gemma-4-31b_M20.jsonl",
    "ministral-3-14b (sighted, API)":
        API_RUNS_DIR / "sighted_ministral-3-14b_M20.jsonl",
    "vlm-3r (sighted, API)":
        API_RUNS_DIR / "sighted_vlm-3r_M20.jsonl",
    "robobrain-2.5-8b (sighted, API)":
        API_RUNS_DIR / "sighted_robobrain-2.5-8b_M20.jsonl",
}
YAW_STR_TO_BIN = {"RIGHT": 1, "BACK": 2, "LEFT": 3}
HEIGHT_STR_TO_BIN = {"BELOW": 0, "ON": 1, "ABOVE": 2}


def load_api_counts_at_M(path: Path, M_target: int, sids_filter: set | None = None):
    """API JSONL parser. Each rec has parsed.yaw_bin (label string) and
    parsed.height (label string). Restrict to sids_filter if given."""
    by_sid = {}
    with open(path) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            sid = rec.get("sample_id")
            if sids_filter is not None and sid not in sids_filter:
                continue
            parsed = rec.get("parsed") or {}
            y_str = str(parsed.get("yaw_bin", "")).upper()
            h_str = str(parsed.get("height", "")).upper()
            y = YAW_STR_TO_BIN.get(y_str)
            h = HEIGHT_STR_TO_BIN.get(h_str)
            if y is None or h is None:
                continue
            rep = rec.get("repeat_id")
            by_sid.setdefault(sid, []).append((rep, y, h))
    counts_d, M_per = {}, {}
    for sid, lst in by_sid.items():
        lst.sort(key=lambda x: x[0] if x[0] is not None else 0)
        keep = lst[:M_target]
        c = np.zeros(9, dtype=np.float64)
        for _, y, h in keep:
            c[joint_index(y, h)] += 1
        counts_d[sid] = c
        M_per[sid] = len(keep)
    return counts_d, M_per


def load_vlm_counts_at_M(path: Path, M_target: int):
    """Parse a VLM JSONL, take the first M_target valid samples per sid
    (sorted by repeat_id), return (dict[sid] -> 9-vector counts,
    dict[sid] -> M_eff)."""
    by_sid = {}
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
            if y not in (1, 2, 3) or v not in (0, 1, 2):
                continue
            rep = rec.get("repeat_id")
            by_sid.setdefault(sid, []).append((rep, y, v))
    counts_d, M_per = {}, {}
    for sid, lst in by_sid.items():
        lst.sort(key=lambda x: (x[0] if x[0] is not None else 0))
        keep = lst[:M_target]
        c = np.zeros(9, dtype=np.float64)
        for _, y, v in keep:
            c[joint_index(y, v)] += 1
        counts_d[sid] = c
        M_per[sid] = len(keep)
    return counts_d, M_per


def evaluate_vlm_at_M(counts_d, M_per, set_df, split, alpha,
                        task: str = "joint", averaging: str = "sample"):
    B = task_b(task)
    sid_to_gt = {r["sample_id"]: task_gt(r, task)
                  for _, r in set_df.iterrows()}
    sid_to_cat = {r["sample_id"]: r["canonical_label"]
                    for _, r in set_df.iterrows()}
    cal = [s for s in split["calibration_sample_ids"] if s in counts_d]
    ev = [s for s in split["eval_sample_ids"] if s in counts_d]

    def smooth_log(sid):
        c = maybe_marginalize_to_yaw(counts_d[sid], task)
        return np.log(smooth_for_B(c, alpha, M_per[sid], B))

    if not cal or not ev:
        return None
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
    delta_oracle = nll_calib - nll_oracle
    H_pre = aggregate(-np.sum(np.exp(ev_log) * ev_log, axis=1), ev_cats)
    H_post = aggregate(-np.sum(np.exp(ev_log_T) * ev_log_T, axis=1), ev_cats)
    per_correct = (np.argmax(ev_log, axis=1) == ev_gts).astype(float)
    acc = aggregate(per_correct, ev_cats)

    floored_nlls = []
    for sid in ev:
        c = maybe_marginalize_to_yaw(counts_d[sid], task)
        s = c.sum()
        p = c / s if s > 0 else np.full(B, 1 / B)
        p = np.clip(p, EPS_FLOOR, 1.0)
        p = p / p.sum()
        floored_nlls.append(-math.log(p[sid_to_gt[sid]]))
    floored_nll = aggregate(np.array(floored_nlls), ev_cats)

    # Always also compute the cat-averaged version of every per-sample
    # metric, using *whatever T* was just tuned* (per-sample T when
    # averaging="sample"; per-category T when averaging="category").
    # When averaging="sample" these give the "calibrate-then-average-
    # per-category" view.
    cat_keys_for_avg = [sid_to_cat[s] for s in ev]
    floored_arr = np.array(floored_nlls)
    # Per-(category) breakdown of within-cat means using the just-tuned T*
    breakdown_groups = {}
    for i, c in enumerate(cat_keys_for_avg):
        breakdown_groups.setdefault(c, []).append(i)
    per_cat_breakdown = {}
    for c, idx in breakdown_groups.items():
        idx_arr = np.array(idx, dtype=int)
        per_cat_breakdown[c] = {
            "n": int(len(idx_arr)),
            "nll_calib": float(np.mean(per_calib[idx_arr])),
            "nll_smooth": float(np.mean(per_smooth[idx_arr])),
            "floored_nll": float(np.mean(floored_arr[idx_arr])),
            "acc": float(np.mean(per_correct[idx_arr])),
        }
    out = {
        "T_star": T_star, "T_oracle": T_oracle,
        "eval_acc": acc, "floored_nll": floored_nll,
        "eval_nll_smooth": nll_smooth, "eval_nll_calib": nll_calib,
        "eval_nll_oracle": nll_oracle, "delta_oracle": delta_oracle,
        "eval_entropy_norm_pre": H_pre / math.log(B),
        "eval_entropy_norm_post": H_post / math.log(B),
        "n_cal": len(cal), "n_eval": len(ev),
        "eval_acc_cat_avg": aggregate(per_correct, cat_keys_for_avg),
        "floored_nll_cat_avg": aggregate(floored_arr, cat_keys_for_avg),
        "eval_nll_smooth_cat_avg": aggregate(per_smooth, cat_keys_for_avg),
        "eval_nll_calib_cat_avg": aggregate(per_calib, cat_keys_for_avg),
        "per_cat_breakdown": per_cat_breakdown,
    }
    if averaging == "category":
        per_cat = {}
        groups = {}
        for i, c in enumerate(ev_cats):
            groups.setdefault(c, []).append(i)
        for c, idx in groups.items():
            idx = np.array(idx, dtype=int)
            per_cat[c] = {
                "n": int(len(idx)),
                "nll_calib": float(np.mean(per_calib[idx])),
                "acc": float(np.mean(per_correct[idx])),
            }
        out["per_category"] = per_cat
        out["n_eval_categories"] = len(per_cat)
    return out


def hash_to_int(s: str) -> int:
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16)


# ── Per-annotator distribution loaders ────────────────────────────────
def load_user_distributions(user_path: Path):
    """Return list of dicts, each describing one row to evaluate.
    Keys: name, label (display), kind ('single'|'probs'|'derived_single'),
    M, dist (dict[sid] -> 9-vector counts)."""
    rec = json.loads(user_path.read_text())
    name = rec.get("name", "?")
    # Anonymous display ID = first 8 chars of the user-file stem (already
    # an md5-derived UID). Real `name` is kept only for the no-admin
    # exclusion check.
    anon = user_path.stem[:8]
    rows = []

    # q1_single → one-hot
    sa = rec.get("q1_single_answers") or {}
    if any("yaw_bin_4" in v for v in sa.values()):
        ds = {}
        for sid, ans in sa.items():
            if "yaw_bin_4" not in ans:
                continue
            yaw = int(ans["yaw_bin_4"])
            h = int(ans["height_bin"])
            if yaw in (1, 2, 3) and h in (0, 1, 2):
                v = np.zeros(9, dtype=np.float64)
                v[joint_index(yaw, h)] = 1.0
                ds[sid] = v
        if ds:
            rows.append({
                "name": name, "label": f"human:{anon} (single)",
                "kind": "single", "M": M_SINGLE, "dist": ds,
            })

    # q1_probs → rich + derived-single (top bin, random tie-break)
    pa = rec.get("q1_probs_answers") or {}
    if any("joint_probs" in v for v in pa.values()):
        dp = {}
        dps = {}
        seed_base = hash_to_int(name)
        for sid, ans in pa.items():
            if "joint_probs" not in ans:
                continue
            v = np.array(ans["joint_probs"], dtype=np.float64)
            if len(v) != 9:
                continue
            dp[sid] = v
            # Top bin with random tie-breaking (deterministic per sid)
            mx = v.max()
            top = [i for i, x in enumerate(v) if x == mx]
            rng = random.Random(seed_base ^ hash_to_int(sid))
            pick = rng.choice(top)
            oh = np.zeros(9, dtype=np.float64)
            oh[pick] = 1.0
            dps[sid] = oh
        if dp:
            rows.append({
                "name": name, "label": f"human:{anon} (probs)",
                "kind": "probs", "M": M_PROBS, "dist": dp,
            })
            rows.append({
                "name": name, "label": f"human:{anon} (single from probs)",
                "kind": "derived_single", "M": M_SINGLE, "dist": dps,
            })
    return rows


# ── Calibration on a (dist, M) row ────────────────────────────────────
def split_sids(set_df: pd.DataFrame, sids):
    cal = sorted(s for s in sids
                  if set_df.loc[s, "participant_id"] in CAL_PIDS)
    ev = sorted(s for s in sids
                  if set_df.loc[s, "participant_id"] not in CAL_PIDS)
    return cal, ev


def evaluate_row(dist: dict, M: int, set_df: pd.DataFrame, alpha: float,
                   task: str = "joint", averaging: str = "sample"):
    B = task_b(task)
    sid_to_gt = {r["sample_id"]: task_gt(r, task)
                  for _, r in set_df.iterrows()}
    sid_to_cat = {r["sample_id"]: r["canonical_label"]
                    for _, r in set_df.iterrows()}
    sids = set(dist) & set(set_df.index)
    cal_sids, eval_sids = split_sids(set_df, sids)

    def smooth_log(sid):
        c = maybe_marginalize_to_yaw(dist[sid], task)
        return np.log(smooth_for_B(c, alpha, M, B))

    cal_log = np.stack([smooth_log(s) for s in cal_sids])
    cal_gts = np.array([sid_to_gt[s] for s in cal_sids], dtype=int)
    ev_log = np.stack([smooth_log(s) for s in eval_sids])
    ev_gts = np.array([sid_to_gt[s] for s in eval_sids], dtype=int)

    cal_cats = [sid_to_cat[s] for s in cal_sids] if averaging == "category" else None
    ev_cats = [sid_to_cat[s] for s in eval_sids] if averaging == "category" else None

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
    delta_oracle = nll_calib - nll_oracle

    H_pre = aggregate(-np.sum(np.exp(ev_log) * ev_log, axis=1), ev_cats)
    H_post = aggregate(-np.sum(np.exp(ev_log_T) * ev_log_T, axis=1), ev_cats)
    per_correct = (np.argmax(ev_log, axis=1) == ev_gts).astype(float)
    acc = aggregate(per_correct, ev_cats)

    # Floored NLL on marginalised counts (B-aware)
    floored_nlls = []
    for sid in eval_sids:
        c = maybe_marginalize_to_yaw(dist[sid], task)
        s = c.sum()
        p = c / s if s > 0 else np.full(B, 1 / B)
        p = np.clip(p, EPS_FLOOR, 1.0)
        p = p / p.sum()
        floored_nlls.append(-math.log(p[sid_to_gt[sid]]))
    floored_nll = aggregate(np.array(floored_nlls), ev_cats)

    cat_keys_for_avg = [sid_to_cat[s] for s in eval_sids]
    floored_arr = np.array(floored_nlls)
    breakdown_groups = {}
    for i, c in enumerate(cat_keys_for_avg):
        breakdown_groups.setdefault(c, []).append(i)
    per_cat_breakdown = {}
    for c, idx in breakdown_groups.items():
        idx_arr = np.array(idx, dtype=int)
        per_cat_breakdown[c] = {
            "n": int(len(idx_arr)),
            "nll_calib": float(np.mean(per_calib[idx_arr])),
            "nll_smooth": float(np.mean(per_smooth[idx_arr])),
            "floored_nll": float(np.mean(floored_arr[idx_arr])),
            "acc": float(np.mean(per_correct[idx_arr])),
        }
    out = {
        "T_star": T_star, "T_oracle": T_oracle,
        "eval_acc": acc,
        "floored_nll": floored_nll,
        "eval_nll_smooth": nll_smooth,
        "eval_nll_calib": nll_calib,
        "eval_nll_oracle": nll_oracle,
        "delta_oracle": delta_oracle,
        "eval_entropy_norm_pre": H_pre / math.log(B),
        "eval_entropy_norm_post": H_post / math.log(B),
        "n_cal": len(cal_sids), "n_eval": len(eval_sids),
        "eval_acc_cat_avg": aggregate(per_correct, cat_keys_for_avg),
        "floored_nll_cat_avg": aggregate(floored_arr, cat_keys_for_avg),
        "eval_nll_smooth_cat_avg": aggregate(per_smooth, cat_keys_for_avg),
        "eval_nll_calib_cat_avg": aggregate(per_calib, cat_keys_for_avg),
        "per_cat_breakdown": per_cat_breakdown,
    }
    if averaging == "category":
        per_cat = {}
        groups = {}
        for i, c in enumerate(ev_cats):
            groups.setdefault(c, []).append(i)
        for c, idx in groups.items():
            idx = np.array(idx, dtype=int)
            per_cat[c] = {
                "n": int(len(idx)),
                "nll_calib": float(np.mean(per_calib[idx])),
                "acc": float(np.mean(per_correct[idx])),
            }
        out["per_category"] = per_cat
        out["n_eval_categories"] = len(per_cat)
    return out


def build_ensemble_row_impl(included, label, task, averaging, set_df, alpha):
    """Module-level ensemble builder (was nested in main()). `included` =
    {annotator_name: {sid: joint_idx}}. Returns a row dict or None.
    Each annotator's pick is stored as a 9-bin joint index; for yaw/pitch
    we marginalise to 3 bins."""
    if not included:
        return None
    all_sids = set.union(*[set(d.keys()) for d in included.values()])
    sid_to_counts, sid_to_M = {}, {}
    for sid in all_sids:
        counts = np.zeros(9, dtype=np.float64)
        n = 0
        for d in included.values():
            if sid in d:
                counts[d[sid]] += 1
                n += 1
        if n == 0:
            continue
        sid_to_counts[sid] = counts
        sid_to_M[sid] = n
    B = task_b(task)
    sid_to_gt = {r["sample_id"]: task_gt(r, task)
                  for _, r in set_df.iterrows()}
    sid_to_cat = {r["sample_id"]: r["canonical_label"]
                    for _, r in set_df.iterrows()}
    sids = set(sid_to_counts) & set(set_df.index)
    cal_sids, eval_sids = split_sids(set_df, sids)
    if not cal_sids or not eval_sids:
        return None

    def smooth_log_var(sid):
        c = maybe_marginalize_to_yaw(sid_to_counts[sid], task)
        return np.log(smooth_for_B(c, alpha, sid_to_M[sid], B))

    cal_log = np.stack([smooth_log_var(s) for s in cal_sids])
    cal_gts = np.array([sid_to_gt[s] for s in cal_sids], dtype=int)
    ev_log = np.stack([smooth_log_var(s) for s in eval_sids])
    ev_gts = np.array([sid_to_gt[s] for s in eval_sids], dtype=int)

    cal_cats = ([sid_to_cat[s] for s in cal_sids]
                  if averaging == "category" else None)
    ev_cats = ([sid_to_cat[s] for s in eval_sids]
                 if averaging == "category" else None)

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
    delta_oracle = nll_calib - nll_oracle
    H_pre = aggregate(-np.sum(np.exp(ev_log) * ev_log, axis=1), ev_cats)
    H_post = aggregate(-np.sum(np.exp(ev_log_T) * ev_log_T, axis=1), ev_cats)
    per_correct = (np.argmax(ev_log, axis=1) == ev_gts).astype(float)
    acc = aggregate(per_correct, ev_cats)

    floored_nlls = []
    for sid in eval_sids:
        c = maybe_marginalize_to_yaw(sid_to_counts[sid], task)
        s = c.sum()
        p = c / s if s > 0 else np.full(B, 1 / B)
        p = np.clip(p, EPS_FLOOR, 1.0)
        p = p / p.sum()
        floored_nlls.append(-math.log(p[sid_to_gt[sid]]))
    floored_nll = aggregate(np.array(floored_nlls), ev_cats)

    cat_keys_for_avg = [sid_to_cat[s] for s in eval_sids]
    floored_arr = np.array(floored_nlls)
    breakdown_groups = {}
    for i, c in enumerate(cat_keys_for_avg):
        breakdown_groups.setdefault(c, []).append(i)
    per_cat_breakdown = {}
    for c, idx in breakdown_groups.items():
        idx_arr = np.array(idx, dtype=int)
        per_cat_breakdown[c] = {
            "n": int(len(idx_arr)),
            "nll_calib": float(np.mean(per_calib[idx_arr])),
            "nll_smooth": float(np.mean(per_smooth[idx_arr])),
            "floored_nll": float(np.mean(floored_arr[idx_arr])),
            "acc": float(np.mean(per_correct[idx_arr])),
        }
    out = {
        "name": label,
        "T_star": T_star, "T_oracle": T_oracle,
        "eval_acc": acc, "floored_nll": floored_nll,
        "eval_nll_smooth": nll_smooth, "eval_nll_calib": nll_calib,
        "eval_nll_oracle": nll_oracle, "delta_oracle": delta_oracle,
        "eval_entropy_norm_pre": H_pre / math.log(B),
        "eval_entropy_norm_post": H_post / math.log(B),
        "n_cal": len(cal_sids), "n_eval": len(eval_sids),
        "eval_acc_cat_avg": aggregate(per_correct, cat_keys_for_avg),
        "floored_nll_cat_avg": aggregate(floored_arr, cat_keys_for_avg),
        "eval_nll_smooth_cat_avg": aggregate(per_smooth, cat_keys_for_avg),
        "eval_nll_calib_cat_avg": aggregate(per_calib, cat_keys_for_avg),
        "per_cat_breakdown": per_cat_breakdown,
    }
    if averaging == "category":
        per_cat = {}
        groups = {}
        for i, c in enumerate(ev_cats):
            groups.setdefault(c, []).append(i)
        for c, idx in groups.items():
            idx = np.array(idx, dtype=int)
            per_cat[c] = {
                "n": int(len(idx)),
                "nll_calib": float(np.mean(per_calib[idx])),
                "acc": float(np.mean(per_correct[idx])),
            }
        out["per_category"] = per_cat
        out["n_eval_categories"] = len(per_cat)
    return out


def fmt_t(x):
    if isinstance(x, float):
        if math.isnan(x):
            return "—"
        if math.isinf(x):
            return "∞"
    return f"{x:.3f}"


def fmt_row(r):
    return (
        f"| {r['name']} | {r.get('n_eval', '—')} | "
        f"{r['eval_acc']:.1%} | {r['floored_nll']:.3f} | "
        f"{r['eval_nll_smooth']:.3f} | {r['eval_nll_calib']:.3f} | "
        f"{fmt_t(r['T_star'])} | "
        f"{r.get('eval_entropy_norm_pre', float('nan')):.3f} | "
        f"{r.get('eval_entropy_norm_post', float('nan')):.3f} | "
        f"{fmt_t(r['T_oracle'])} | {r['delta_oracle']:.4f} |"
    )


MIN_COMPLETION = 80  # at least this many answered in the active task


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["joint", "yaw", "pitch"],
                      default="joint",
                      help="Compute calibration on the joint 9-bin target "
                           "(default), marginalised yaw 3-bin target, or "
                           "marginalised pitch (height) 3-bin target.")
    ap.add_argument("--averaging", choices=["sample", "category"],
                      default="sample",
                      help="How to average per-query metrics. 'sample': "
                           "mean over all eval queries (default; outputs "
                           "to per_sample/). 'category': mean within each "
                           "canonical_label, then mean across categories "
                           "(outputs to per_category/). T tuned against "
                           "the same loss.")
    args = ap.parse_args()
    task = args.task
    averaging = args.averaging
    out_suffix = "" if task == "joint" else f"_{task}"
    out_dir = (PER_CATEGORY_DIR if averaging == "category"
                 else PER_SAMPLE_DIR)

    set_task(task)
    print(f"Task: {task} (B={task_b(task)})")
    print("Loading set parquet…")
    set_df = pd.read_parquet(SET_PARQUET).set_index("sample_id", drop=False)

    # Dedupe by uid (filename stem). users/ wins over users_bk/.
    files: dict[str, Path] = {}
    if USERS_DIR.exists():
        for f in sorted(USERS_DIR.glob("*.json")):
            files.setdefault(f.stem, f)
    if USERS_BK_DIR.exists():
        for f in sorted(USERS_BK_DIR.glob("*.json")):
            files.setdefault(f.stem, f)
    print(f"Considering {len(files)} unique user files (after dedupe).")

    # ──────────────── Per-annotator single-pick joint indices ────────
    # Used both for the per-row evaluation and the M=5 ensemble.
    annotator_joint_idx: dict[str, dict[str, int]] = {}

    human_rows = []
    for upath in files.values():
        rec = json.loads(upath.read_text())
        n_single = sum(1 for v in (rec.get("q1_single_answers") or {}).values()
                          if "yaw_bin_4" in v)
        n_probs = sum(1 for v in (rec.get("q1_probs_answers") or {}).values()
                         if "joint_probs" in v)
        if n_single < MIN_COMPLETION and n_probs < MIN_COMPLETION:
            print(f"  skip {upath.stem[:8]} "
                  f"(single={n_single}, probs={n_probs})")
            continue
        for spec in load_user_distributions(upath):
            # Skip the row if the *kind* doesn't have enough answers.
            n = (n_probs if spec["kind"] in ("probs", "derived_single")
                 else n_single)
            if n < MIN_COMPLETION:
                continue
            r = evaluate_row(spec["dist"], spec["M"], set_df, ALPHA,
                                task=task, averaging=averaging)
            r["name"] = spec["label"]
            human_rows.append(r)
            print(f"  {spec['label']:>40s}  acc={r['eval_acc']:.1%}  "
                  f"NLL_calib={r['eval_nll_calib']:.3f}  "
                  f"T*={r['T_star']:.2f}  n_eval={r['n_eval']}")

            # Record the single-pick joint index per sid for the ensemble.
            # For 'probs' rows, we use the derived top-bin (kind ==
            # 'derived_single'), which is the single-pick equivalent of
            # the rich distribution.
            if spec["kind"] in ("single", "derived_single"):
                joints = {sid: int(np.argmax(v))
                            for sid, v in spec["dist"].items()}
                # Avoid double-counting the same annotator under a
                # potential second kind: prefer 'single' if present, else
                # derived_single. So only set if not already present.
                annotator_joint_idx.setdefault(spec["name"], joints)

    # ──────────────── Ensembles: humans as a single M=N VLM ─────────
    def build_ensemble_row(included, label):
        return build_ensemble_row_impl(included, label, task, averaging,
                                            set_df, ALPHA)

    # M=N (everyone)
    if len(annotator_joint_idx) >= 2:
        names_all = sorted(annotator_joint_idx.keys())
        row_all = build_ensemble_row(
            annotator_joint_idx,
            f"human:ensemble M={len(names_all)}",
        )
        if row_all is not None:
            human_rows.insert(0, row_all)
            print(f"  {row_all['name']:>40s}  acc={row_all['eval_acc']:.1%}  "
                  f"NLL_calib={row_all['eval_nll_calib']:.3f}  "
                  f"T*={row_all['T_star']:.2f}  n_eval={row_all['n_eval']}")

    # (no admin) ensemble row dropped per spec — keep the M=10 row only.

    # VLM rows.
    # - For joint: pull M=20 rows from results_raw_30_70.json (already
    #   computed by calibrate_T) and also compute M=5 from JSONLs for
    #   the human comparison.
    # - For yaw: results_raw is joint-only, so recompute M=20 yaw from
    #   the JSONLs.
    # When averaging=category, we must always recompute M=20 VLM rows
    # because the per_sample VLM_RAW JSON has per-sample-averaged
    # numbers (and no per-category breakdown).
    vlm_rows_by_name = {}
    use_raw_vlm = (task == "joint" and averaging == "sample"
                     and VLM_RAW.exists())
    if use_raw_vlm:
        vlm_raw = json.loads(VLM_RAW.read_text())
        for r in vlm_raw["primary_rows"]:
            vlm_rows_by_name[r["name"]] = r

    print(f"\nComputing VLM rows at M={M_VLM_COMPARE} (and M=20 if "
          f"task!=joint or averaging=category)…")
    vlm_m5_rows = []
    vlm_m20_rows = []
    if VLM_SET_PARQUET.exists() and VLM_SPLIT_30_70.exists():
        vlm_set_df = pd.read_parquet(VLM_SET_PARQUET)
        vlm_split = json.loads(VLM_SPLIT_30_70.read_text())
        for vname, vpath in VLM_MODELS.items():
            if not vpath.exists():
                continue
            counts5, M5 = load_vlm_counts_at_M(vpath, M_VLM_COMPARE)
            r5 = evaluate_vlm_at_M(counts5, M5, vlm_set_df, vlm_split,
                                       ALPHA, task=task, averaging=averaging)
            if r5 is not None:
                r5["name"] = f"{vname} (M={M_VLM_COMPARE})"
                vlm_m5_rows.append(r5)
                print(f"  {r5['name']:>40s}  acc={r5['eval_acc']:.1%}  "
                      f"NLL_calib={r5['eval_nll_calib']:.3f}  "
                      f"T*={r5['T_star']:.2f}  n_eval={r5['n_eval']}")
            if not use_raw_vlm:
                counts20, M20 = load_vlm_counts_at_M(vpath, 20)
                r20 = evaluate_vlm_at_M(counts20, M20, vlm_set_df, vlm_split,
                                            ALPHA, task=task,
                                            averaging=averaging)
                if r20 is not None:
                    r20["name"] = vname  # bare name, mirrors M=20 default
                    vlm_m20_rows.append(r20)

    # ─── API models on the SAME 100-sample human pool (head-to-head) ───
    pool_state = json.loads((TESTSET / "data" / "human_eval" /
                                "pool_state.json").read_text())
    pool_sids = set(pool_state["active"])

    print(f"\nAPI rows on the 100-sample human pool (eval=human's "
          f"69-sample subset) at M=20, M=10 (humans-budget match), and M=6…")
    api_split = {  # apply same scene-disjoint split to the pool
        "calibration_sample_ids": sorted(
            s for s in pool_sids
            if s in set_df.index
            and set_df.loc[s, "participant_id"] in CAL_PIDS),
        "eval_sample_ids": sorted(
            s for s in pool_sids
            if s in set_df.index
            and set_df.loc[s, "participant_id"] not in CAL_PIDS),
    }
    api_m20_rows = []
    api_m10_rows = []
    api_m6_rows = []
    for aname, apath in API_MODELS.items():
        if not apath.exists():
            continue
        for M_target, dst in (
                (20, api_m20_rows),
                (10, api_m10_rows),
                (6, api_m6_rows)):
            counts, M_per = load_api_counts_at_M(
                apath, M_target, sids_filter=pool_sids)
            if not counts:
                continue
            # Need full coverage of the pool to be a fair comparison.
            covered = len(counts)
            r = evaluate_vlm_at_M(counts, M_per, set_df, api_split,
                                      ALPHA, task=task, averaging=averaging)
            if r is None:
                continue
            r["name"] = f"{aname} (M={M_target})"
            dst.append(r)
            print(f"  {r['name']:<48s}  "
                  f"acc={r['eval_acc']:.1%}  "
                  f"NLL_calib={r['eval_nll_calib']:.3f}  "
                  f"T*={r['T_star']:.2f}  "
                  f"covered={covered}/100  n_eval={r['n_eval']}")

    # Render combined md
    out = out_dir / f"results_human_combined{out_suffix}.md"
    UNIFORM = "Uniform"
    CAT = "Category prior"
    VLM_ORDER = [
        "gemma-4-31b/blind", "gemma-4-31b/sighted",
        "qwen3-vl-30b/blind", "qwen3-vl-30b/sighted",
        "qwen3.5-9b/blind", "qwen3.5-9b/sighted",
    ]

    with open(out, "w") as f:
        f.write("# Human + VLM combined calibration table\n\n")
        if averaging == "category":
            f.write("**Averaging:** `category` — NLL/acc/entropy meaned "
                      "within each `canonical_label`, then meaned across "
                      "categories. T tuned against the same per-category "
                      "loss.\n\n")
        else:
            f.write("**Averaging:** `sample` — mean over all eval queries.\n\n")
        f.write("**Caveat**: human and VLM rows use *different* eval "
                  "subsets. Humans label the **100-sample human pool** "
                  "(`pool_state.json::active`); VLM rows are reproduced "
                  "from `results_30_70.md` on `setA_300_filtered` (their "
                  "eval subset = 203 queries from {P01, P02, P06, P08, "
                  "P09}, mostly *non-overlapping* with the human pool). "
                  "Same calibration recipe (α=0.1, scene-disjoint "
                  "participants, 1-D T tuning) applies to both. Treat "
                  "the comparison as a context check, not a head-to-head "
                  "benchmark on identical samples.\n\n")
        if human_rows:
            n_eval_human = human_rows[0]["n_eval"]
            n_cal_human = human_rows[0]["n_cal"]
            f.write(f"- Human eval subset: **{n_eval_human}** samples "
                      f"(participants ∉ {sorted(CAL_PIDS)}; calibration "
                      f"set = {n_cal_human} samples).\n")
        f.write("- VLM eval subset: **203** samples (from `results_30_70.md`).\n\n")

        f.write("Human rows shown: **ensemble M=N** and **ensemble "
                  "(no admin) M=N−1** — per-query majority across "
                  "annotators, smoothed and calibrated as if the "
                  "ensemble were a single M=N VLM. Individual annotator "
                  "rows are saved in `results_raw_*.json` but omitted "
                  "from this table for legibility.\n\n")

        f.write("| Model | n_eval | Acc | Floored NLL | Smooth NLL | "
                  "Calib NLL | T* | H_pre | H_post | Oracle T | "
                  "δ_oracle |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")

        # Uniform analytic — task-aware
        B_now = task_b(task)
        unif = ({
            "name": UNIFORM, "eval_acc": 1.0 / B_now,
            "floored_nll": math.log(B_now),
            "eval_nll_smooth": math.log(B_now),
            "eval_nll_calib": math.log(B_now),
            "T_star": float("nan"), "T_oracle": float("nan"),
            "delta_oracle": 0.0,
            "eval_entropy_norm_pre": 1.0, "eval_entropy_norm_post": 1.0,
            "n_eval": "—",
        })
        f.write(fmt_row(unif) + "\n")

        # Human rows — only the ensemble rows (individual annotators are
        # in the per-(model, category) breakdown / raw JSON if needed).
        ensemble_rows = [r for r in human_rows if "ensemble" in r["name"]]
        for r in ensemble_rows:
            f.write(fmt_row(r) + "\n")

        # API model rows on the SAME 100-sample human pool — head-to-head
        # with humans (n_eval = same 69 sids).
        if api_m20_rows or api_m10_rows or api_m6_rows:
            f.write("\n**API models on the human pool** "
                      "(SAME 69-sample eval subset as the humans above; "
                      "first true head-to-head). M=10 is the budget-"
                      "matched cut to humans (10 annotators); M=6 is "
                      "tighter still; M=20 is the locked default.\n\n")
            f.write("| Model | n_eval | Acc | Floored NLL | Smooth NLL | "
                      "Calib NLL | T* | H_pre | H_post | Oracle T | "
                      "δ_oracle |\n")
            f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
            for r in api_m20_rows + api_m10_rows + api_m6_rows:
                f.write(fmt_row(r) + "\n")

        # VLM rows at M=5 (apples-to-apples with the human ensembles)
        if vlm_m5_rows:
            f.write(f"\n*VLM rows at M={M_VLM_COMPARE} (subsampled from "
                      "M=25 JSONLs on `setA_300_filtered`; same 30/70 "
                      "split as the M=20 rows below).*\n\n")
            f.write("| Model | n_eval | Acc | Floored NLL | Smooth NLL | "
                      "Calib NLL | T* | H_pre | H_post | Oracle T | "
                      "δ_oracle |\n")
            f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
            for r in vlm_m5_rows:
                f.write(fmt_row(r) + "\n")
            f.write("\n*VLM rows at M=20 (original locked default).*\n\n")
            f.write("| Model | n_eval | Acc | Floored NLL | Smooth NLL | "
                      "Calib NLL | T* | H_pre | H_post | Oracle T | "
                      "δ_oracle |\n")
            f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")

        # Category prior + VLMs (M=20). For joint+sample we pull from
        # results_raw_30_70.json; for everything else we use the
        # freshly-computed M=20 rows.
        if use_raw_vlm:
            if CAT in vlm_rows_by_name:
                f.write(fmt_row(vlm_rows_by_name[CAT]) + "\n")
            for n in VLM_ORDER:
                if n in vlm_rows_by_name:
                    f.write(fmt_row(vlm_rows_by_name[n]) + "\n")
        else:
            for r in vlm_m20_rows:
                f.write(fmt_row(r) + "\n")

        # Per-(model, category) intermediate breakdown (only meaningful
        # when averaging=category).
        if averaging == "category":
            named_rows = [r for r in (ensemble_rows + api_m20_rows + api_m10_rows + api_m6_rows
                                          + vlm_m5_rows + vlm_m20_rows)
                            if r.get("per_category")]
            if named_rows:
                f.write("\n## Per-(model, category) NLL_calib breakdown "
                          "(intermediate)\n\n")
                f.write("Each cell is the **within-category mean Calibrated "
                          "NLL** before being meaned across categories. "
                          "Categories sorted by total eval count in the "
                          "first row.\n\n")
                # Different rows have different eval subsets (humans use
                # the 69-sample pool; VLMs use the 203-sample setA split).
                # Group rows by eval-subset signature so the breakdown
                # tables are coherent.
                sigs = {}
                human_pool_rows = ensemble_rows + api_m20_rows + api_m10_rows + api_m6_rows
                for r in named_rows:
                    sig = ("human-pool" if r in human_pool_rows
                              else "setA-203")
                    sigs.setdefault(sig, []).append(r)
                for sig, rs in sigs.items():
                    cats_sorted = sorted(rs[0]["per_category"].keys(),
                                            key=lambda c: -rs[0]["per_category"][c]["n"])
                    f.write(f"### eval subset: {sig} "
                              f"(n_eval = {rs[0]['n_eval']}, "
                              f"{rs[0].get('n_eval_categories', '?')} cats)\n\n")
                    f.write("| Category (n) | "
                              + " | ".join(r["name"] for r in rs) + " |\n")
                    f.write("|---|" + "---:|" * len(rs) + "\n")
                    for c in cats_sorted:
                        n = rs[0]["per_category"][c]["n"]
                        cells = []
                        for r in rs:
                            pc = r["per_category"].get(c)
                            cells.append(f"{pc['nll_calib']:.3f}" if pc else "—")
                        f.write(f"| {c} ({n}) | " + " | ".join(cells) + " |\n")
                    f.write("\n")

    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
