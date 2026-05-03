"""Group KL & JSD vs GT pairwise distribution.

For each (target_category c, visible_object vo) pair where multiple
frames in the eval subset share the (c, vo) condition:

  P_gt(y | c, vo)    = empirical yaw distribution over GT bins of those frames
  P_model(y | c, vo) = aggregated yaw distribution over the model's
                       responses for those frames (counts pooled across
                       frames × repeats × annotators).

We then compute:
  - KL(P_gt || P_model)    — model blind-spot penalty
  - KL(P_model || P_gt)    — model overconfidence penalty
  - JSD(P_gt, P_model)     — symmetric

and aggregate (mean) across all qualifying (c, vo) pairs in each eval
subset. Three subsets: full setA (1951 queries), 300-set
(setA_300_filtered), and the 100-sample human pool.

Pair filtering follows pairwise-metric.md Filter B: only **same-zone**
pairs (target c and anchor vo in the same kitchen functional zone),
generic anchors {counter, cabinet, backsplash, window, door} excluded.

Yaw is the primary 3-bin task. Pitch (height) and joint (9-bin) are
also reported for completeness.

Outputs:
  meta/metric/GroupKL/groupkl_results.md
  meta/metric/GroupKL/groupkl_pairs.json   (per-pair raw data, audit trail)
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

META = Path("/path/to/this/repo")
TESTSET = META / "testset"
DATA_DIR = TESTSET / "data"
OUT_DIR = META / "metric" / "GroupKL"
OUT_DIR.mkdir(exist_ok=True, parents=True)

# Eval subsets
SET_FULL = DATA_DIR / "setA_extended_filtered.parquet"
SET_300 = DATA_DIR / "setA_300_filtered.parquet"
POOL_PATH = DATA_DIR / "human_eval" / "pool_state.json"

# Captions
CAPTIONS_JSONL = DATA_DIR / "captions_v2_extended.jsonl"

# Models (APIs that fully cover all subsets; same set as in
# calibrate_humans.py / compare_averaging.py)
API_RUNS_DIR = META / "results" / "setA_extention" / "records"
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

# Humans (only on the 100-pool subset)
USERS_DIR = TESTSET / "data" / "human_eval" / "users"
USERS_BK_DIR = TESTSET / "data" / "human_eval" / "users_bk"
MIN_HUMAN_COMPLETION = 80
ADMIN_NAME = "admin"

# Training pool for category prior
TRAIN_POOL = DATA_DIR / "extended_pool_balanced.parquet"

# Yaw / pitch label maps (data idx)
YAW_DATA = {1: 0, 2: 1, 3: 2}  # yaw_bin_4 (1=R, 2=B, 3=L) → idx (0,1,2)
EPS = 1e-12

YAW_STR_TO_BIN = {"RIGHT": 1, "BACK": 2, "LEFT": 3}
HEIGHT_STR_TO_BIN = {"BELOW": 0, "ON": 1, "ABOVE": 2}

# ── Anchor detection patterns (from track-f/work/scripts/pairwise_metric.py) ─
ANCHOR_PATTERNS = {
    "sink":          [r"\bsink\b", r"\bbasin\b"],
    "stove":         [r"\bstove\b", r"\bcooktop\b", r"\bburner\b", r"\bhob\b"],
    "oven":          [r"\boven\b"],
    "fridge":        [r"\bfridge\b", r"\brefrigerator\b", r"\bfreezer\b"],
    "microwave":     [r"\bmicrowave\b"],
    "hood":          [r"\bhood\b", r"\bextractor\b"],
    "counter":       [r"\bcounter(?:top)?\b", r"\bworktop\b"],
    "cabinet":       [r"\bcabinet\b", r"\bcupboard\b"],
    "window":        [r"\bwindow\b"],
    "door":          [r"\bdoor(?:way)?\b"],
    "shelf":         [r"\bshelf\b", r"\bshelves\b"],
    "drawer":        [r"\bdrawer\b"],
    "draining_rack": [r"\bdrain(?:ing)?\s*(?:rack|board)\b",
                       r"\bdish\s*rack\b", r"\bdrying\s*rack\b"],
    "faucet":        [r"\bfaucet\b", r"\btap\b"],
    "backsplash":    [r"\bbacksplash\b", r"\btile[ds]?\s*wall\b",
                       r"\bsplashback\b"],
    "washing_machine": [r"\bwashing\s*machine\b", r"\bdishwasher\b",
                          r"\bwasher\b"],
    "radiator":      [r"\bradiator\b"],
}

# ── Filter B: per-target related-anchor allowlist ───────────────────
# Faithful to `meta/work/discussion/pairwise-metric.md`'s Filter B:
# a (target, anchor) pair is kept iff the anchor is in the target's
# allowlist AND the anchor is not a generic kitchen feature
# {counter, cabinet, backsplash}. Allows cross-zone relations
# (e.g. cutting board ← sink, plant ← window).
TARGET_RELATED: dict[str, set[str]] = {
    # ── sink zone targets ────────────────────────
    "faucet":          {"sink", "faucet", "draining_rack", "window"},
    "sink":            {"sink", "faucet", "draining_rack", "window"},
    "sponge":          {"sink", "faucet", "draining_rack"},
    "draining rack":   {"sink", "faucet", "draining_rack", "window"},
    "scrub brush":     {"sink", "faucet", "draining_rack"},
    "rubber glove":    {"sink", "faucet", "draining_rack"},
    "basin":           {"sink", "faucet", "draining_rack"},
    "soap bottle":     {"sink", "faucet", "draining_rack"},
    "strainer":        {"sink", "faucet", "draining_rack", "drawer"},
    # ── stove zone targets ───────────────────────
    "stove":           {"stove", "oven", "hood"},
    "oven":            {"stove", "oven", "hood"},
    "pot":             {"stove", "oven", "hood"},
    "frying pan":      {"stove", "oven", "hood"},
    "kettle":          {"stove", "oven", "faucet"},
    "saucepan":        {"stove", "oven"},
    "pressure cooker": {"stove", "oven"},
    "baking tray":     {"stove", "oven"},
    "baking pan":      {"stove", "oven"},
    "baking dish":     {"stove", "oven"},
    "coffee pot":      {"stove", "oven"},
    # ── counter / prep targets (cross-zone to sink/stove) ─
    "cutting board":   {"sink", "faucet", "stove", "drawer"},
    "knife":           {"drawer", "shelf", "sink", "stove"},
    "knife holder":    {"drawer", "sink", "stove"},
    "toaster":         {"window", "sink"},
    "coffee machine":  {"sink", "faucet"},
    "food processor":  {"sink", "drawer"},
    "food blender":    {"sink", "drawer"},
    "paper towel":     {"sink", "faucet", "window"},
    # ── periphery ────────────────────────────────
    "fridge":          {"fridge", "door", "window"},
    "washing machine": {"washing_machine", "fridge", "door"},
    "trash bin":       {"door", "fridge"},
    "plant":           {"window", "shelf"},
    "wall socket":     {"door", "window"},
    "radiator":        {"radiator", "window"},
    # ── storage ──────────────────────────────────
    "basket":          {"shelf", "drawer", "door"},
    "scissor":         {"drawer", "shelf"},
    # ── food / produce ───────────────────────────
    "banana":          {"fridge", "draining_rack", "window"},
    "apple":           {"fridge", "draining_rack"},
    "carrot":          {"fridge", "draining_rack"},
    "avocado":         {"fridge", "draining_rack"},
    "bread":           {"oven"},
    # ── misc kitchenware ─────────────────────────
    "kitchen scale":   {"stove", "oven", "drawer"},
    "milk bottle":     {"fridge"},
    "tray":            {"sink", "oven", "drawer"},
    "colander":        {"sink", "faucet", "draining_rack", "drawer"},
}
# Generic kitchen-everywhere anchors — excluded by Filter B because
# they appear in 47–88% of captions and produce near-uniform P_pairwise
# for almost every target. Re-included by Filter C.
GENERIC_ANCHORS = {"counter", "cabinet", "backsplash"}

# ── Filter A (strict same-zone, sink+stove only, per pairwise-metric.md) ─
# Filter A is the most restrictive: only kitchen pairs where target and
# anchor are both in the same primary work-zone (sink-zone or stove-zone).
SINK_ZONE_TARGETS = {"faucet", "sink", "sponge", "draining rack",
                       "scrub brush", "rubber glove", "basin",
                       "soap bottle", "strainer"}
STOVE_ZONE_TARGETS = {"stove", "oven", "pot", "frying pan", "kettle",
                        "saucepan", "pressure cooker", "baking tray",
                        "baking pan", "baking dish", "coffee pot"}
SINK_ZONE_ANCHORS = {"sink", "faucet", "draining_rack"}
STOVE_ZONE_ANCHORS = {"stove", "oven", "hood"}


def is_filter_a(target: str, anchor: str) -> bool:
    """Filter A — strict same-zone: only sink-zone or stove-zone pairs."""
    if anchor == target:
        return False
    if target in SINK_ZONE_TARGETS and anchor in SINK_ZONE_ANCHORS:
        return True
    if target in STOVE_ZONE_TARGETS and anchor in STOVE_ZONE_ANCHORS:
        return True
    return False


def is_filter_b(target: str, anchor: str) -> bool:
    """Filter B — per-target allowlist (cross-zone OK), generic anchors
    {counter, cabinet, backsplash} excluded. The recommended primary
    filter per `pairwise-metric.md` (Filter B has the best balance of
    coverage and non-uniformity in the original analysis)."""
    if anchor in GENERIC_ANCHORS:
        return False
    if anchor == target:
        return False
    related = TARGET_RELATED.get(target)
    if related is None:
        return False
    return anchor in related


def is_filter_c(target: str, anchor: str) -> bool:
    """Filter C — Filter B universe plus the generic anchors. Every
    target with a TARGET_RELATED entry is allowed to pair with any
    generic anchor (counter / cabinet / backsplash). Most permissive."""
    if anchor == target:
        return False
    related = TARGET_RELATED.get(target)
    if related is None:
        return False
    if anchor in related:
        return True
    return anchor in GENERIC_ANCHORS


# Backward-compat alias for ablation_compare.py and earlier consumers.
is_same_zone = is_filter_b


# ── Helpers ──────────────────────────────────────────────────────────
def detect_anchors(caption: str) -> set:
    if not caption:
        return set()
    cl = caption.lower()
    out = set()
    for a, pats in ANCHOR_PATTERNS.items():
        if any(re.search(p, cl) for p in pats):
            out.add(a)
    return out


def kl(p, q):
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1.0)
    q = np.clip(np.asarray(q, dtype=np.float64), EPS, 1.0)
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * (np.log(p) - np.log(q))))


def kl_jsd(p, q):
    m = 0.5 * (np.asarray(p) + np.asarray(q))
    return kl(p, q), kl(q, p), 0.5 * (kl(p, m) + kl(q, m))


# ── Loaders ──────────────────────────────────────────────────────────
def load_captions() -> dict:
    """Return dict[frame_key] -> caption."""
    out = {}
    with open(CAPTIONS_JSONL) as f:
        for line in f:
            r = json.loads(line)
            out[r["frame_key"]] = r.get("caption", "")
    return out


def task_b(task: str) -> int:
    return {"yaw": 3, "pitch": 3, "joint": 9}[task]


def task_idx(yaw_bin_1to3: int, height_bin_0to2: int, task: str) -> int:
    """Convert (yaw_bin_4 ∈ {1,2,3}, height_bin ∈ {0,1,2}) to a bin idx."""
    if task == "yaw":
        return yaw_bin_1to3 - 1            # 0..2
    if task == "pitch":
        return height_bin_0to2             # 0..2
    return (yaw_bin_1to3 - 1) * 3 + height_bin_0to2  # 0..8


def load_api_predictions(path: Path, sids_filter: set | None = None,
                            M_target: int = 20):
    """Return dict[sid] -> list of (yaw_bin_1to3, height_bin_0to2),
    capped at M_target valid responses per sid."""
    by_sid = {}
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            sid = r.get("sample_id")
            if sids_filter is not None and sid not in sids_filter:
                continue
            parsed = r.get("parsed") or {}
            y = YAW_STR_TO_BIN.get(str(parsed.get("yaw_bin", "")).upper())
            h = HEIGHT_STR_TO_BIN.get(str(parsed.get("height", "")).upper())
            if y is None or h is None:
                continue
            rep = r.get("repeat_id")
            by_sid.setdefault(sid, []).append((rep, y, h))
    out = {}
    for sid, lst in by_sid.items():
        lst.sort(key=lambda x: x[0] if x[0] is not None else 0)
        out[sid] = [(y, h) for _, y, h in lst[:M_target]]
    return out


def hash_to_int(s: str) -> int:
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16)


def load_humans_predictions(sids_filter: set | None = None):
    """Return dict[sid] -> list of (yaw_bin_1to3, height_bin_0to2). Each
    annotator with ≥ MIN_HUMAN_COMPLETION single picks contributes one
    entry per sid; annotators with only probs use the top bin (with stable
    sid-hash tie-break)."""
    files = {}
    for d in (USERS_DIR, USERS_BK_DIR):
        if d.exists():
            for f in sorted(d.glob("*.json")):
                files.setdefault(f.stem, f)
    by_sid = {}
    for stem, p in files.items():
        rec = json.loads(p.read_text())
        anon = stem[:8]
        n_single = sum(1 for v in (rec.get("q1_single_answers") or {}).values()
                          if "yaw_bin_4" in v)
        n_probs = sum(1 for v in (rec.get("q1_probs_answers") or {}).values()
                         if "joint_probs" in v)
        if n_single < MIN_HUMAN_COMPLETION and n_probs < MIN_HUMAN_COMPLETION:
            continue
        if n_single >= MIN_HUMAN_COMPLETION:
            for sid, ans in (rec.get("q1_single_answers") or {}).items():
                if sids_filter is not None and sid not in sids_filter:
                    continue
                if "yaw_bin_4" not in ans:
                    continue
                yaw = int(ans["yaw_bin_4"])
                h = int(ans["height_bin"])
                if yaw in (1, 2, 3) and h in (0, 1, 2):
                    by_sid.setdefault(sid, []).append((yaw, h))
        else:
            seed_base = hash_to_int(anon)
            for sid, ans in (rec.get("q1_probs_answers") or {}).items():
                if sids_filter is not None and sid not in sids_filter:
                    continue
                v = ans.get("joint_probs")
                if v is None or len(v) != 9:
                    continue
                v = np.array(v, dtype=np.float64)
                mx = v.max()
                top = [i for i, x in enumerate(v) if x == mx]
                rng = random.Random(seed_base ^ hash_to_int(sid))
                pick = rng.choice(top)
                yaw_bin = pick // 3 + 1
                pitch_bin = pick % 3
                by_sid.setdefault(sid, []).append((yaw_bin, pitch_bin))
    return by_sid


def load_category_prior(set_df: pd.DataFrame, train_df: pd.DataFrame,
                          task: str):
    """LOSO category prior. For each sid in set_df, returns a (B,) prob
    vector built from training-pool entries with the same canonical_label
    but a different participant_id. Returns dict[sid] -> p (B-vec)."""
    B = task_b(task)
    grouped = {}
    for (pid, lbl), g in train_df.groupby(["participant_id", "canonical_label"]):
        grouped[(pid, lbl)] = g[["yaw_bin_4", "height_bin"]].values
    pids = sorted(train_df["participant_id"].unique())
    out = {}
    for _, row in set_df.iterrows():
        sid = row["sample_id"]
        pid = row["participant_id"]
        lbl = row["canonical_label"]
        counts = np.zeros(B, dtype=np.float64)
        for tpid in pids:
            if tpid == pid:
                continue
            arr = grouped.get((tpid, lbl))
            if arr is None:
                continue
            for yb, hb in arr:
                yb, hb = int(yb), int(hb)
                if yb in (1, 2, 3) and hb in (0, 1, 2):
                    counts[task_idx(yb, hb, task)] += 1
        if counts.sum() > 0:
            out[sid] = counts / counts.sum()
    return out


# ── Build (c, vo) groups and compute distributions ───────────────────
def build_pair_groups(set_df: pd.DataFrame, captions: dict,
                          min_n: int = 5, filter_fn=None):
    """Iterate frames in set_df; for each frame and each detected anchor
    that is accepted by `filter_fn(target, anchor)`, register the sid
    under the (target, anchor) pair. Pairs with fewer than `min_n`
    frames are dropped. Default filter is Filter B (the recommended
    primary filter)."""
    if filter_fn is None:
        filter_fn = is_filter_b
    fk_for_sid = {}
    for _, r in set_df.iterrows():
        fk_for_sid[r["sample_id"]] = (
            f"{r['dataset']}:{r['video_id']}:{r['frame_index']}")
    pair_to_sids = {}
    for _, row in set_df.iterrows():
        sid = row["sample_id"]
        target = row["canonical_label"]
        cap = captions.get(fk_for_sid[sid], "")
        if not cap:
            continue
        anchors = detect_anchors(cap)
        for a in anchors:
            if not filter_fn(target, a):
                continue
            pair_to_sids.setdefault((target, a), []).append(sid)
    return {p: sids for p, sids in pair_to_sids.items() if len(sids) >= min_n}


def gt_distribution(set_df_indexed: pd.DataFrame, sids: list, task: str):
    """Empirical GT distribution over these sids."""
    B = task_b(task)
    counts = np.zeros(B, dtype=np.float64)
    for sid in sids:
        if sid not in set_df_indexed.index:
            continue
        r = set_df_indexed.loc[sid]
        counts[task_idx(int(r["yaw_bin_4"]), int(r["height_bin"]), task)] += 1
    if counts.sum() == 0:
        return None
    return counts / counts.sum()


def model_predictions_distribution(predictions: dict, sids: list, task: str):
    """Empirical model distribution over the pooled responses of these
    sids (counts the model's predicted bins across all reps × sids).
    `predictions[sid]` = list of (yaw_bin_1to3, height_bin_0to2)."""
    B = task_b(task)
    counts = np.zeros(B, dtype=np.float64)
    for sid in sids:
        for y, h in predictions.get(sid, []):
            counts[task_idx(y, h, task)] += 1
    if counts.sum() == 0:
        return None
    return counts / counts.sum()


def cat_prior_distribution(cat_prior_p: dict, sids: list, task: str):
    """Average the per-sid LOSO category-prior distributions across sids
    in the group (each sid contributes its own prior since LOSO yields a
    sid-specific prior; mean is the natural pooled distribution)."""
    B = task_b(task)
    if not sids:
        return None
    ps = [cat_prior_p[sid] for sid in sids if sid in cat_prior_p]
    if not ps:
        return None
    return np.mean(np.stack(ps), axis=0)


# ── Main per-subset evaluation ───────────────────────────────────────
FILTERS = {
    "A": is_filter_a,
    "B": is_filter_b,
    "C": is_filter_c,
}


def evaluate_subset(name: str, set_df: pd.DataFrame, captions: dict,
                       min_n: int = 5, include_humans: bool = False,
                       include_old_models: bool = False):
    """Evaluate all rows on the given subset under all three filters.
    Returns a dict with per-filter pair data and per-(filter, task)
    aggregated rows."""
    set_idx = set_df.set_index("sample_id", drop=False)
    eval_sids = set(set_df["sample_id"])
    print(f"\n=== Subset: {name}  (n_sids = {len(eval_sids)}) ===")

    # Build pair sets under each filter
    pairs_per_filter = {}
    for fname, ffn in FILTERS.items():
        groups = build_pair_groups(set_df, captions, min_n=min_n,
                                       filter_fn=ffn)
        pairs_per_filter[fname] = groups
        print(f"  Filter {fname} (n >= {min_n}): {len(groups)} pairs")

    # Load model predictions, restricted to this subset's sids
    predictions = {}
    for mname, mpath in API_MODELS.items():
        if not mpath.exists():
            continue
        preds = load_api_predictions(mpath, sids_filter=eval_sids,
                                          M_target=20)
        if preds:
            predictions[mname] = preds
    if include_humans:
        h_preds = load_humans_predictions(sids_filter=eval_sids)
        if h_preds:
            predictions["humans (ensemble)"] = h_preds

    # Category prior
    train_df = pd.read_parquet(TRAIN_POOL)
    cat_priors = {
        "yaw": load_category_prior(set_df, train_df, "yaw"),
        "pitch": load_category_prior(set_df, train_df, "pitch"),
        "joint": load_category_prior(set_df, train_df, "joint"),
    }

    # Build a master pair-membership view (union across filters)
    union_pairs = {}
    for fname, groups in pairs_per_filter.items():
        for pair, sids in groups.items():
            union_pairs.setdefault(pair, {"n": len(sids), "filters": set()})
            union_pairs[pair]["filters"].add(fname)
            # n_frames is identical across filters (same caption→sid map)
    union_pair_list = sorted(
        [(p, info["n"], "".join(sorted(info["filters"])))
            for p, info in union_pairs.items()],
        key=lambda r: -r[1])

    results = {"subset": name, "n_sids": len(eval_sids),
                  "min_n": min_n,
                  "n_pairs_per_filter": {f: len(g)
                                              for f, g in pairs_per_filter.items()},
                  "union_pair_list": [
                      {"pair": f"{p[0]} \\| {p[1]}", "n": n,
                        "filters": fs}
                      for p, n, fs in union_pair_list],
                  "filters": {}}

    for fname, pair_to_sids in pairs_per_filter.items():
        per_task = {}
        for task in ("yaw", "pitch", "joint"):
            B = task_b(task)
            rows = []
            # Pre-compute per-pair P_gt
            pair_gt = {}
            for pair, sids in pair_to_sids.items():
                p_gt = gt_distribution(set_idx, sids, task)
                if p_gt is not None:
                    pair_gt[pair] = p_gt

            def aggregate_for_predictor(pred_p_per_pair):
                kls_hm, kls_mh, jsds = [], [], []
                for pair, p_gt in pair_gt.items():
                    p_m = pred_p_per_pair.get(pair)
                    if p_m is None:
                        continue
                    kl_hm, kl_mh, j = kl_jsd(p_gt, p_m)
                    kls_hm.append(kl_hm)
                    kls_mh.append(kl_mh)
                    jsds.append(j)
                if not jsds:
                    return float("nan"), float("nan"), float("nan"), 0
                return (float(np.mean(kls_hm)),
                        float(np.mean(kls_mh)),
                        float(np.mean(jsds)),
                        len(jsds))

            # Uniform
            unif = np.full(B, 1.0 / B)
            unif_per_pair = {p: unif for p in pair_gt}
            kl_gm, kl_mg, jsd, n_used = aggregate_for_predictor(unif_per_pair)
            rows.append({"name": "Uniform", "n_pairs": n_used,
                          "kl_gm": kl_gm, "kl_mg": kl_mg, "jsd": jsd})

            # Category prior
            cat_p_per_pair = {}
            for pair, sids in pair_to_sids.items():
                p_cat = cat_prior_distribution(cat_priors[task], sids, task)
                if p_cat is not None:
                    cat_p_per_pair[pair] = p_cat
            kl_gm, kl_mg, jsd, n_used = aggregate_for_predictor(cat_p_per_pair)
            rows.append({"name": "Category prior", "n_pairs": n_used,
                          "kl_gm": kl_gm, "kl_mg": kl_mg, "jsd": jsd})

            # Models (APIs + humans)
            for mname in predictions:
                preds = predictions[mname]
                m_per_pair = {}
                for pair, sids in pair_to_sids.items():
                    p_m = model_predictions_distribution(preds, sids, task)
                    if p_m is not None:
                        m_per_pair[pair] = p_m
                kl_gm, kl_mg, jsd, n_used = aggregate_for_predictor(m_per_pair)
                rows.append({"name": mname, "n_pairs": n_used,
                              "kl_gm": kl_gm, "kl_mg": kl_mg, "jsd": jsd})

            per_task[task] = {"rows": rows}
        results["filters"][fname] = {
            "n_pairs": len(pair_to_sids),
            "tasks": per_task,
        }
        print(f"  Filter {fname}: tasks done")
    return results


def fmt(v, prec=3):
    if isinstance(v, float):
        if math.isnan(v):
            return "—"
        if math.isinf(v):
            return "∞"
    return f"{v:.{prec}f}"


def _summary_block(all_results: list, f):
    """Lead-of-document summary: best/worst rows under Filter B per
    (subset, task)."""
    f.write("## TL;DR\n\n")
    f.write(
        "**Pairwise group divergence** = for each `(target c, visible-"
        "object vo)` pair, build `P_gt(y | c, vo)` and `P_model(y | c, "
        "vo)` from raw bin counts pooled over the pair's frames, "
        "compute KL/JSD per pair, then average across pairs. Smaller "
        "JSD = closer to GT pairwise prior. Three filter levels are "
        "evaluated side-by-side (see **Filter levels** in Method).\n\n")
    f.write("Headline (under Filter B, the recommended primary filter):\n\n")
    f.write("| Subset | n_pairs (B) | task | best (lowest JSD) | "
              "worst (highest JSD) | uniform JSD |\n")
    f.write("|---|---:|---|---|---|---:|\n")
    for res in all_results:
        nb = res["n_pairs_per_filter"].get("B", 0)
        if nb == 0:
            f.write(f"| {res['subset']} | 0 | — | — | — | — |\n")
            continue
        for task in ("yaw", "pitch", "joint"):
            rows = res["filters"]["B"]["tasks"][task]["rows"]
            usable = [r for r in rows if not math.isnan(r["jsd"])]
            if not usable:
                continue
            usable.sort(key=lambda r: r["jsd"])
            best = usable[0]
            worst = usable[-1]
            unif_row = next((r for r in rows if r["name"] == "Uniform"),
                                None)
            f.write(
                f"| {res['subset']} | {nb} | {task} | "
                f"{best['name']} ({fmt(best['jsd'])}) | "
                f"{worst['name']} ({fmt(worst['jsd'])}) | "
                f"{fmt(unif_row['jsd']) if unif_row else '—'} |\n")
    f.write("\n")


def render_md(all_results: list, out_path: Path):
    with open(out_path, "w") as f:
        f.write("# Group KL & JSD vs GT pairwise distribution\n\n")
        _summary_block(all_results, f)
        f.write("## Method\n\n")
        f.write(
            "For each (target_category `c`, visible_object `vo`) pair, "
            "we pool **all frames** in the eval subset that share the "
            "same condition `(c, vo)` (i.e. target = `c` and the frame's "
            "caption mentions `vo`). With ≥ `min_n` frames per pair we "
            "estimate two distributions over yaw / pitch / joint bins:\n\n"
            "- `P_gt(y | c, vo)` — empirical from those frames' GT bins.\n"
            "- `P_model(y | c, vo)` — empirical from the model's responses "
            "pooled over those frames × repeats × annotators (raw "
            "predicted-bin counts; no temperature calibration).\n\n"
            "Then we compute the two directed KL divergences plus the "
            "symmetric JSD per pair, and **average across pairs**:\n\n"
            "- `KL(g‖m) = Σ_i P_gt(i) · log(P_gt(i)/P_model(i))` — model "
            "blind-spot penalty.\n"
            "- `KL(m‖g) = Σ_i P_model(i) · log(P_model(i)/P_gt(i))` — "
            "model overconfidence penalty.\n"
            "- `JSD(g, m) = ½·KL(P_gt‖M) + ½·KL(P_model‖M)` with "
            "`M = ½(P_gt + P_model)`. Bounded in [0, log 2 ≈ 0.693] (nats).\n\n"
            "All distributions are clipped at ε = 1e-12 then re-normalised "
            "before KL computation; no Dirichlet smoothing or temperature "
            "scaling is applied.\n\n")
        f.write("### Filter levels (matching `pairwise-metric.md`)\n\n")
        f.write(
            "Three filter levels for the `(target, anchor)` pair set, "
            "applied independently to the same caption→anchor "
            "extraction pipeline:\n\n"
            "**Filter A — strict same-zone.** Only pairs where target "
            "and anchor are both in the **same primary kitchen work "
            "zone** (sink-zone or stove-zone). Most restrictive; "
            "highest expected `P_pairwise` non-uniformity. Per "
            "`pairwise-metric.md` line 984: \"Sink-zone and stove-zone "
            "pairs only.\"\n\n")
        f.write(
            f"  - sink-zone targets = `{sorted(SINK_ZONE_TARGETS)}`\n"
            f"  - sink-zone anchors = `{sorted(SINK_ZONE_ANCHORS)}`\n"
            f"  - stove-zone targets = `{sorted(STOVE_ZONE_TARGETS)}`\n"
            f"  - stove-zone anchors = `{sorted(STOVE_ZONE_ANCHORS)}`\n"
            "  - A pair `(c, vo)` is in Filter A iff `c` is in one "
            "zone's target list AND `vo` is in the **same zone**'s "
            "anchor list (and `vo ≠ c`).\n\n")
        f.write(
            "**Filter B — per-target allowlist, generic anchors "
            "excluded.** The recommended primary filter. Each target "
            "has a hand-curated list of related anchors that may be "
            "**cross-zone** (e.g. `cutting board ← sink`, "
            "`plant ← window`, `banana ← draining_rack`). The "
            "generic anchor set "
            f"`{sorted(GENERIC_ANCHORS)}` is excluded — these "
            "appear in 47–88 % of captions and produce near-uniform "
            "`P_pairwise`. Filter A ⊂ Filter B.\n\n")
        f.write(
            "**Filter C — Filter B + generic anchors.** The same per-"
            "target allowlists as B, but with "
            f"`{sorted(GENERIC_ANCHORS)}` re-added for every target. "
            "Most permissive; covers near-uniform pairs that dilute "
            "the discriminative signal. Filter A ⊂ Filter B ⊂ "
            "Filter C.\n\n")
        # Inline TARGET_RELATED for transparency
        f.write("**Per-target related anchors (used by Filters B and C):**\n\n")
        f.write("| Target | Related anchors |\n|---|---|\n")
        for tgt in sorted(TARGET_RELATED.keys()):
            anchors = ", ".join(f"`{a}`" for a in sorted(TARGET_RELATED[tgt]))
            f.write(f"| {tgt} | {anchors} |\n")
        f.write("\n")
        f.write("### Eval subsets\n\n")
        f.write(
            "- **full setA** = `setA_extended_filtered.parquet` (1951 "
            "queries, 11 participants).\n"
            "- **300** = `setA_300_filtered.parquet` (the locked 300-sample "
            "core).\n"
            "- **100** = the 100-sample human-evaluation pool "
            "(`pool_state.json::active`). Humans included as a "
            "'humans (ensemble)' row on this subset only.\n\n")

        for res in all_results:
            n_per_filter = res["n_pairs_per_filter"]
            f.write(f"## Subset: {res['subset']} "
                      f"(n_sids = {res['n_sids']}, "
                      f"min_n = {res['min_n']}; pair counts: "
                      f"A={n_per_filter.get('A', 0)} / "
                      f"B={n_per_filter.get('B', 0)} / "
                      f"C={n_per_filter.get('C', 0)})\n\n")
            if all(v == 0 for v in n_per_filter.values()):
                f.write(
                    "**No (c, vo) pairs satisfy `n ≥ "
                    f"{res['min_n']}` under any filter.** The metric "
                    "is undefined here.\n\n")
                continue

            # Combined pair list with filter-membership column
            if res["union_pair_list"]:
                f.write("### (c, vo) pairs available "
                          "(membership across filters)\n\n")
                f.write(
                    "Each row shows where the pair first appears: "
                    "`A` = strict same-zone (also in B and C), "
                    "`AB`/`B` = added by B, `BC`/`C` = added by C. "
                    "Pairs with `n_frames < min_n` are dropped.\n\n")
                f.write("| Pair (target \\| anchor) | n_frames | Filters |\n")
                f.write("|---|---:|---|\n")
                for entry in res["union_pair_list"]:
                    f.write(f"| {entry['pair']} | {entry['n']} | "
                              f"{entry['filters']} |\n")
                f.write("\n")

            # Side-by-side JSD table per task
            for task in ("yaw", "pitch", "joint"):
                B = task_b(task)
                f.write(f"### Task: {task} (B = {B}, log B = "
                          f"{math.log(B):.3f})\n\n")
                f.write(
                    "JSD vs `P_gt` averaged across pairs, shown side-"
                    "by-side under all three filter levels. `n=A/B/C` "
                    "in column headers is the number of pairs the row "
                    "could evaluate on (matches the subset filter "
                    "counts at the section header). Δ-vs-Uniform-B is "
                    "the most-natural ranking metric (negative = beats "
                    "Uniform under the recommended primary filter).\n\n")

                # Aggregate rows across filters by model name.
                # Use Filter B's row order as canonical (insertion order),
                # then sort ascending by JSD-B.
                rows_b = res["filters"]["B"]["tasks"][task]["rows"]
                rows_a = {r["name"]: r
                              for r in res["filters"]["A"]["tasks"][task]["rows"]}
                rows_c = {r["name"]: r
                              for r in res["filters"]["C"]["tasks"][task]["rows"]}

                unif_b = next((r for r in rows_b
                                  if r["name"] == "Uniform"
                                  and not math.isnan(r["jsd"])), None)
                unif_b_jsd = unif_b["jsd"] if unif_b else float("nan")

                sorted_b = sorted(
                    rows_b,
                    key=lambda r: (math.isnan(r["jsd"]), r["jsd"]))

                f.write(
                    f"| Rank | Model | "
                    f"JSD-A (n={n_per_filter.get('A', 0)}) | "
                    f"JSD-B (n={n_per_filter.get('B', 0)}) | "
                    f"JSD-C (n={n_per_filter.get('C', 0)}) | "
                    f"Δ-B vs Uniform-B |\n")
                f.write("|---:|---|---:|---:|---:|---:|\n")
                for i, rb in enumerate(sorted_b, 1):
                    name = rb["name"]
                    ra = rows_a.get(name, {"jsd": float("nan")})
                    rc = rows_c.get(name, {"jsd": float("nan")})
                    delta = (rb["jsd"] - unif_b_jsd
                                if not (math.isnan(rb["jsd"])
                                          or math.isnan(unif_b_jsd))
                                else float("nan"))
                    delta_str = (f"{delta:+.3f}"
                                    if not math.isnan(delta) else "—")
                    f.write(
                        f"| {i} | {name} | "
                        f"{fmt(ra['jsd'])} | {fmt(rb['jsd'])} | "
                        f"{fmt(rc['jsd'])} | {delta_str} |\n")
                f.write("\n")

                # KL diagnostic block — all three filters side by side
                f.write(
                    "<details><summary>Diagnostic: both directed KLs "
                    "under Filters A / B / C</summary>\n\n"
                    f"`A: n={n_per_filter.get('A', 0)}`, "
                    f"`B: n={n_per_filter.get('B', 0)}`, "
                    f"`C: n={n_per_filter.get('C', 0)}`. "
                    "Rows in Filter B's order (sorted by JSD-B "
                    "ascending). `KL(g‖m)` = blind-spot penalty; "
                    "`KL(m‖g)` = overconfidence penalty.\n\n"
                    "| Model | "
                    "KL(g‖m) [A] | KL(m‖g) [A] | "
                    "KL(g‖m) [B] | KL(m‖g) [B] | "
                    "KL(g‖m) [C] | KL(m‖g) [C] |\n"
                    "|---|---:|---:|---:|---:|---:|---:|\n")
                for rb in sorted_b:
                    name = rb["name"]
                    ra = rows_a.get(name, {"kl_gm": float("nan"),
                                                 "kl_mg": float("nan")})
                    rc = rows_c.get(name, {"kl_gm": float("nan"),
                                                 "kl_mg": float("nan")})
                    f.write(
                        f"| {name} | "
                        f"{fmt(ra['kl_gm'])} | {fmt(ra['kl_mg'])} | "
                        f"{fmt(rb['kl_gm'])} | {fmt(rb['kl_mg'])} | "
                        f"{fmt(rc['kl_gm'])} | {fmt(rc['kl_mg'])} |\n")
                f.write("\n</details>\n\n")

        # Trailing discussion
        _discussion_block(all_results, f)

    print(f"Wrote {out_path}")


def _discussion_block(all_results: list, f):
    full = next((r for r in all_results if r["subset"] == "full setA"), None)
    n_pairs_full = (full["n_pairs_per_filter"].get("B", "?")
                       if full else "?")
    f.write("## Discussion\n\n")
    f.write(
        "1. **Data scale matters.** Filter B same-zone (c, vo) pairs "
        f"are sparse: the full setA (1951 sids) yields {n_pairs_full} "
        "pairs at `n ≥ 5`, the 300-set yields a smaller number at "
        "`n ≥ 3`, and the 100-pool yields fewer still even at `n ≥ 2`. "
        "The metric is informative on full setA; numbers on the smaller "
        "subsets are presented for completeness but should be treated "
        "as directional.\n\n"
        "2. **Yaw on full setA — uniform is hard to beat.** Same "
        "finding as the original `pairwise-metric.md` analysis: most "
        "rows sit close to Uniform's JSD because raw model distributions "
        "are over-peaked on bins that disagree with the pairwise prior. "
        "Only Category prior, gpt-5.4, gemini-3-flash, and the smaller "
        "qwen3.5 / vlm-3r rows tie with or narrowly beat Uniform; the "
        "larger qwen3-vl-235b and qwen3.5-397b-a17b sit well above it.\n\n"
        "3. **Pitch & joint — Category prior wins.** Kitchen-object "
        "heights are stable priors, so the LOSO category prior is the "
        "best non-uniform reference. On pitch, Category prior and "
        "`vlm-3r` beat Uniform by wide margins; on joint, Category "
        "prior, `vlm-3r`, `gpt-5.4`, `ministral-3-14b`, and "
        "`qwen3.5-35b-a3b` all beat Uniform. This shows that *pitch "
        "has discriminative signal* the metric can pick up — yaw "
        "doesn't.\n\n"
        "4. **The two KL directions tell the failure mode.** "
        "`KL(g‖m)` ≪ `KL(m‖g)` for almost every row: the models are "
        "overconfident on bins that the GT pairwise prior assigns near-"
        "zero probability. Uniform reverses this asymmetry on yaw — "
        "its uniform support means it has no blind spots, but it pays "
        "the price by spreading mass everywhere.\n\n"
        "5. **Caveats reusable from `pairwise-metric.md`.** Smoothing "
        "on the GT empirical distribution is light (ε = 1e-12 floor + "
        "renormalisation); per-pair sample sizes (n = 5–47 frames) make "
        "the GT distribution itself noisy. The metric measures *raw "
        "predictive shape match* against a small-n empirical reference, "
        "which is why it favours rows that are diffuse-and-correct over "
        "rows that are peaky-and-correct.\n")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--eval-set", "--eval_set",
        dest="eval_set",
        choices=["full", "300", "100", "all"],
        default="all",
        help=(
            "Which subset(s) to evaluate. 'full' = full setA (1951 sids), "
            "'300' = setA-300 (300 sids), '100' = human pool (100 sids), "
            "'all' (default) = all three."
        ),
    )
    args = ap.parse_args()

    print("Loading captions...")
    captions = load_captions()
    print(f"  {len(captions)} captioned frame_keys")

    pool_sids = set(json.loads(POOL_PATH.read_text())["active"])

    full_df = pd.read_parquet(SET_FULL)
    set300_df = pd.read_parquet(SET_300)
    pool_df = full_df[full_df["sample_id"].isin(pool_sids)].copy()

    # `min_n` = minimum frames per pair. Bumped on the larger subsets
    # for better per-pair P_gt estimates; the 100-pool keeps min_n=2
    # because there's no other choice given its size.
    all_results = []
    if args.eval_set in ("full", "all"):
        all_results.append(evaluate_subset(
            "full setA", full_df, captions, min_n=10, include_humans=False))
    if args.eval_set in ("300", "all"):
        all_results.append(evaluate_subset(
            "setA-300", set300_df, captions, min_n=5, include_humans=False))
    if args.eval_set in ("100", "all"):
        all_results.append(evaluate_subset(
            "human pool (100)", pool_df, captions, min_n=2,
            include_humans=True))

    suffix = "" if args.eval_set == "all" else f"_{args.eval_set}"

    # Save raw per-pair JSON for audit
    raw_path = OUT_DIR / f"groupkl_pairs{suffix}.json"
    with open(raw_path, "w") as f:
        json.dump(all_results, f, indent=2,
                    default=lambda v: float(v) if isinstance(v, np.floating)
                    else str(v))
    print(f"Wrote {raw_path}")

    render_md(all_results, OUT_DIR / f"groupkl_results{suffix}.md")


if __name__ == "__main__":
    main()
