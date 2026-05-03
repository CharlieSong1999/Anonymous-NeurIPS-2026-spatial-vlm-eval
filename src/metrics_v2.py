"""
Quick metrics from eval_v2.py JSONLs — Mode accuracy + Brier + RPS only.

The distributional NLL / JSD / KL families have moved to two canonical
pipelines (so the smoothing recipe + calibration split are locked):

  Calibrated NLL (per-sample, per-category, calibrate-first views):
    meta/metric/NLL/calibration/compare_averaging.py
    → per_sample_vs_per_category_{yaw,height,joint}.md

  Group-KL / Group-JSD vs GT pairwise distribution (Filters A/B/C):
    meta/metric/GroupKL/group_kl.py
    → groupkl_results.md

Both accept --eval-set; see docs/run-experiments-on-setA-extended.md §4.5.

This file produces:
  - aggregate_metrics.csv : per-(set, condition, model) Mode acc + Brier + RPS
  - set_summaries.json    : scene/label coverage diagnostics

Run:
  conda run -n slam python3 -m src.metrics_v2
  conda run -n slam python3 -m src.metrics_v2 --partial   # tolerate incomplete runs
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import rel_entr

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

# ── Paths ─────────────────────────────────────────────────────────────
TESTSET = Path("/path/to/this/repo")
DATA_DIR = TESTSET / "data"
EXP_DIR = TESTSET / "exp" / "sampling_ablation_001"
RUNS_DIR = EXP_DIR / "runs"
OUT_DIR = EXP_DIR / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Bin constants ────────────────────────────────────────────────────
EPS = 1e-9
ALPHA = 0.5  # Dirichlet smoothing
PITCH_TO_VBIN = {'UP': 2, 'LEVEL': 1, 'DOWN': 0}
N_YAW = 3   # bins 1, 2, 3 (front excluded)
N_VERT = 3  # 0=DOWN, 1=LEVEL, 2=UP
N_JOINT = N_YAW * N_VERT  # 9
ALLOWED_YAW = [1, 2, 3]


# ── Semantic rules (Filter A/B/C) — lifted from pairwise-metric ────
SAME_ZONE_PAIRS = {
    'sink_zone': {
        'targets': {'faucet', 'sink', 'sponge', 'draining rack', 'scrub brush',
                    'rubber glove', 'basin', 'soap bottle'},
        'anchors': {'sink', 'faucet', 'draining_rack'},
    },
    'stove_zone': {
        'targets': {'stove', 'oven', 'pot', 'frying pan', 'kettle',
                    'saucepan', 'pressure cooker', 'baking tray',
                    'baking pan', 'baking dish'},
        'anchors': {'stove', 'oven', 'hood'},
    },
    'counter_zone': {
        'targets': {'cutting board', 'knife holder', 'toaster', 'coffee machine',
                    'coffee pot', 'food processor', 'food blender', 'paper towel',
                    'colander', 'kitchen scale', 'strainer', 'tray', 'grater',
                    'knife'},
        'anchors': {'counter'},
    },
}

GENERIC_ANCHORS = {'counter', 'cabinet', 'backsplash'}

# Label-specific related-anchor map. Filter B EXCLUDES the GENERIC_ANCHORS;
# Filter C INCLUDES them. So include counter/cabinet/backsplash here for
# any target that plausibly co-occurs with generic kitchen surfaces (i.e.
# everything that lives in a kitchen that's not a wall fixture).
RELATED_RULES = {
    'faucet': ['sink', 'faucet', 'draining_rack', 'window', 'sink_or_faucet',
               'counter', 'cabinet', 'backsplash'],
    'sink': ['sink', 'faucet', 'draining_rack', 'window',
             'counter', 'cabinet', 'backsplash'],
    'sponge': ['sink', 'faucet', 'draining_rack',
               'counter', 'cabinet', 'backsplash'],
    'draining rack': ['sink', 'faucet', 'draining_rack', 'window',
                      'counter', 'cabinet', 'backsplash'],
    'scrub brush': ['sink', 'faucet', 'draining_rack',
                    'counter', 'cabinet', 'backsplash'],
    'rubber glove': ['sink', 'faucet', 'draining_rack',
                     'counter', 'cabinet', 'backsplash'],
    'basin': ['sink', 'faucet', 'draining_rack',
              'counter', 'cabinet', 'backsplash'],
    'soap bottle': ['sink', 'faucet', 'draining_rack',
                    'counter', 'cabinet', 'backsplash'],
    'stove': ['stove', 'oven', 'hood',
              'counter', 'cabinet', 'backsplash'],
    'oven': ['stove', 'oven', 'hood',
             'counter', 'cabinet', 'backsplash'],
    'pot': ['stove', 'oven', 'hood',
            'counter', 'cabinet', 'backsplash'],
    'frying pan': ['stove', 'oven', 'hood',
                   'counter', 'cabinet', 'backsplash'],
    'kettle': ['stove', 'oven', 'hood',
               'counter', 'cabinet', 'backsplash'],
    'saucepan': ['stove', 'oven', 'hood',
                 'counter', 'cabinet', 'backsplash'],
    'pressure cooker': ['stove', 'oven', 'hood',
                        'counter', 'cabinet', 'backsplash'],
    'baking tray': ['stove', 'oven', 'hood',
                    'counter', 'cabinet', 'backsplash'],
    'baking pan': ['stove', 'oven', 'hood',
                   'counter', 'cabinet', 'backsplash'],
    'baking dish': ['stove', 'oven', 'hood',
                    'counter', 'cabinet', 'backsplash'],
    'cutting board': ['stove', 'oven', 'sink', 'faucet',
                      'counter', 'cabinet', 'backsplash'],
    'knife holder': ['stove', 'cutting_board',
                     'counter', 'cabinet', 'backsplash'],
    'toaster': ['stove', 'oven', 'microwave',
                'counter', 'cabinet', 'backsplash'],
    'coffee machine': ['stove', 'oven', 'microwave', 'window',
                       'counter', 'cabinet', 'backsplash'],
    'coffee pot': ['stove', 'oven', 'microwave',
                   'counter', 'cabinet', 'backsplash'],
    'food processor': ['stove', 'oven', 'microwave',
                       'counter', 'cabinet', 'backsplash'],
    'food blender': ['stove', 'oven', 'microwave',
                     'counter', 'cabinet', 'backsplash'],
    'paper towel': ['sink', 'faucet', 'draining_rack', 'stove', 'oven',
                    'counter', 'cabinet', 'backsplash'],
    'colander': ['sink', 'stove', 'oven', 'draining_rack',
                 'counter', 'cabinet', 'backsplash'],
    'kitchen scale': ['stove', 'oven', 'sink',
                      'counter', 'cabinet', 'backsplash'],
    'strainer': ['sink', 'draining_rack',
                 'counter', 'cabinet', 'backsplash'],
    'tray': ['stove', 'oven', 'sink',
             'counter', 'cabinet', 'backsplash'],
    'grater': ['stove', 'sink', 'cutting_board',
               'counter', 'cabinet', 'backsplash'],
    'knife': ['stove', 'sink', 'cutting_board',
              'counter', 'cabinet', 'backsplash'],
    'plant': ['window', 'shelf', 'counter', 'cabinet'],
    'trash bin': ['door', 'cabinet'],
    'washing machine': ['draining_rack', 'door'],
    'wall socket': ['counter', 'backsplash'],
    'milk bottle': ['fridge', 'counter', 'cabinet'],
    'window': ['window'],
    # Food items
    'apple': ['fridge', 'counter'], 'avocado': ['fridge', 'counter'],
    'banana': ['counter'], 'bread': ['counter', 'cabinet'],
    'beer bottle': ['fridge', 'counter'], 'carrot': ['fridge', 'counter'],
}


# ── Caption-based anchor extraction ──────────────────────────────────
ANCHOR_REGEX = {
    'sink': r'\bsink\b|\bbasin\b',
    'stove': r'\bstove\b|\bcooktop\b|\bhob\b|\bstovetop\b|\bstove burner|\bgas burner|\bgas ring\b',
    'oven': r'\boven\b',
    'fridge': r'\bfridge\b|\brefrigerator\b|\bfreezer\b',
    'microwave': r'\bmicrowave\b',
    'dishwasher': r'\bdishwasher\b',
    'hood': r'\brange hood\b|\bvent hood\b|\bextractor hood\b|\bhood\b',
    'faucet': r'\bfaucet\b|\btap\b(?!e)',
    'draining_rack': r'\bdrain(?:ing)? rack\b|\bdish rack\b|\bdrying rack\b|\bdraining board\b',
    'washing_machine': r'\bwashing machine\b|\bwasher\b',
    'window': r'\bwindow\b',
    'door': r'\bdoor(?:way)?\b',
    'drawer': r'\bdrawer\b',
    'shelf': r'\bshelf\b|\bshelves\b|\bshelving\b',
    'radiator': r'\bradiator\b|\bheater\b',
    'counter': r'\bcounter(?:top)?\b|\bworktop\b|\bwork surface\b',
    'cabinet': r'\bcabinet\b|\bcupboard\b',
    'backsplash': r'\bbacksplash\b|\btile(?:d|s)? wall\b|\bsplashback\b',
}


def detect_anchors_in_caption(caption: str) -> set:
    """Return set of anchor names mentioned in caption."""
    hits = set()
    if not caption:
        return hits
    text = caption.lower()
    for a, pat in ANCHOR_REGEX.items():
        if re.search(pat, text):
            hits.add(a)
    return hits


# ── Distribution + smoothing ─────────────────────────────────────────
def smooth_dirichlet(counts: np.ndarray, alpha: float = ALPHA) -> np.ndarray:
    counts = np.asarray(counts, dtype=float)
    n = counts.sum()
    B = len(counts)
    return (counts + alpha) / (n + B * alpha)


def joint_index(yaw_bin: int, vert_bin: int) -> int:
    """Map (yaw_bin in {1,2,3}, vert_bin in {0,1,2}) to joint index 0..8."""
    return (yaw_bin - 1) * N_VERT + vert_bin


def yaw_marginal(joint_dist: np.ndarray) -> np.ndarray:
    """3-bin yaw from 9-bin joint."""
    return np.array([joint_dist[i*N_VERT:(i+1)*N_VERT].sum() for i in range(N_YAW)])


def vert_marginal(joint_dist: np.ndarray) -> np.ndarray:
    return np.array([joint_dist[v::N_VERT][:N_YAW].sum() for v in range(N_VERT)])


# ── Divergences ──────────────────────────────────────────────────────
def kl(p: np.ndarray, q: np.ndarray) -> float:
    p = np.clip(p, EPS, 1); p = p / p.sum()
    q = np.clip(q, EPS, 1); q = q / q.sum()
    return float(np.sum(rel_entr(p, q)))


def jsd(p: np.ndarray, q: np.ndarray) -> float:
    p = np.clip(p, EPS, 1); p = p / p.sum()
    q = np.clip(q, EPS, 1); q = q / q.sum()
    m = 0.5 * (p + q)
    return 0.5 * float(np.sum(rel_entr(p, m))) + \
           0.5 * float(np.sum(rel_entr(q, m)))


def tv_distance(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p) / np.asarray(p).sum()
    q = np.asarray(q) / np.asarray(q).sum()
    return 0.5 * float(np.abs(p - q).sum())


def circular_rps(prob: np.ndarray, gt_idx: int, n_bins: int = N_YAW) -> float:
    """Circular ranked probability score on n_bins yaw."""
    # Use minimum cumulative distance over n_bins rotations
    # Simplified: for K=3, just average squared error in CDF
    # under min-rotation alignment.
    best = float('inf')
    for shift in range(n_bins):
        rolled = np.roll(prob, shift)
        gt_shifted = (gt_idx + shift) % n_bins
        gt_one_hot = np.zeros(n_bins); gt_one_hot[gt_shifted] = 1
        cdf_p = np.cumsum(rolled)
        cdf_g = np.cumsum(gt_one_hot)
        score = float(np.sum((cdf_p - cdf_g) ** 2))
        best = min(best, score)
    return best


# ── Load eval JSONL → per-query distribution ─────────────────────────
def load_run(path: Path):
    """Load a JSONL into a dict: sample_id -> list of (yaw_bin, vert_bin) parses."""
    by_sid = defaultdict(list)
    n_total = 0
    n_valid = 0
    if not path.exists():
        return by_sid, 0, 0
    with open(path) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            n_total += 1
            sid = rec['sample_id']
            parsed = rec.get('parsed') or {}
            y = parsed.get('yaw_bin_id')
            p = parsed.get('pitch')
            try:
                y = int(y)
            except (TypeError, ValueError):
                y = None
            v = PITCH_TO_VBIN.get(str(p).upper()) if p else None
            if y in ALLOWED_YAW and v is not None:
                by_sid[sid].append((y, v))
                n_valid += 1
    return by_sid, n_total, n_valid


def empirical_distribution(samples: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (yaw_dist 3, vert_dist 3, joint_dist 9) with Dirichlet smoothing."""
    yaw_counts = np.zeros(N_YAW)
    vert_counts = np.zeros(N_VERT)
    joint_counts = np.zeros(N_JOINT)
    for y, v in samples:
        yaw_counts[y - 1] += 1
        vert_counts[v] += 1
        joint_counts[joint_index(y, v)] += 1
    return (smooth_dirichlet(yaw_counts), smooth_dirichlet(vert_counts),
            smooth_dirichlet(joint_counts))


# ── Pair-wise prior P_pairwise(y | c, a) ─────────────────────────────
def build_pairwise_priors(set_df: pd.DataFrame, captions: dict,
                          anchor_subset: list = None):
    """Return:
      pw_yaw[(target, anchor)]   = (3,) prob distribution
      pw_vert[(target, anchor)]  = (3,)
      pw_joint[(target, anchor)] = (9,)
      counts[(target, anchor)]   = n samples used
    Estimated from THIS set's GT yaw_bin_4 / height_bin under anchor-visibility.
    """
    pw_yaw, pw_vert, pw_joint, counts = {}, {}, {}, {}
    for _, row in set_df.iterrows():
        sid = row['sample_id']
        # Use frame_key to look up caption
        ds = row['dataset']
        vid = row['video_id']
        fidx = int(row['frame_index'])
        fk = f'{ds}:{vid}:{fidx}'
        cap = captions.get(fk, '')
        anchors = detect_anchors_in_caption(cap)
        if anchor_subset is not None:
            anchors = anchors & set(anchor_subset)
        if not anchors:
            continue
        target = row['canonical_label']
        y_bin = int(row['yaw_bin_4'])
        v_bin = int(row['height_bin'])
        for a in anchors:
            key = (target, a)
            counts[key] = counts.get(key, 0) + 1
            pw_yaw.setdefault(key, np.zeros(N_YAW))[y_bin - 1] += 1
            pw_vert.setdefault(key, np.zeros(N_VERT))[v_bin] += 1
            pw_joint.setdefault(key, np.zeros(N_JOINT))[joint_index(y_bin, v_bin)] += 1
    # Smooth
    pw_yaw = {k: smooth_dirichlet(v) for k, v in pw_yaw.items()}
    pw_vert = {k: smooth_dirichlet(v) for k, v in pw_vert.items()}
    pw_joint = {k: smooth_dirichlet(v) for k, v in pw_joint.items()}
    return pw_yaw, pw_vert, pw_joint, counts


# ── Filter levels ────────────────────────────────────────────────────
def filter_pairs(counts: dict, n_min: int = 5,
                 filter_level: str = 'B') -> list:
    """Return list of (target, anchor) pairs satisfying n>=n_min and the
    semantic filter level.

    A: same zone (sink_zone or stove_zone or counter_zone, if anchor in zone's anchor set)
    B: semantically related per RELATED_RULES, exclude generic anchors
    C: semantically related per RELATED_RULES, include generic anchors
    """
    keep = []
    for (target, anchor), n in counts.items():
        if n < n_min:
            continue
        # Filter A: same zone
        if filter_level == 'A':
            in_zone = False
            for zone in SAME_ZONE_PAIRS.values():
                if target in zone['targets'] and anchor in zone['anchors']:
                    in_zone = True
                    break
            if not in_zone:
                continue
        # Filter B / C: related per rules
        related = RELATED_RULES.get(target, [])
        if anchor not in related:
            continue
        # Filter B excludes generic anchors
        if filter_level == 'B' and anchor in GENERIC_ANCHORS:
            continue
        keep.append((target, anchor))
    return keep


# ── Per-query metrics ────────────────────────────────────────────────
def per_query_metrics(by_sid: dict, set_df: pd.DataFrame):
    """Return per-query metrics dict keyed by sample_id."""
    out = {}
    sid_to_row = {r['sample_id']: r for _, r in set_df.iterrows()}
    for sid, samples in by_sid.items():
        if not samples:
            continue
        row = sid_to_row.get(sid)
        if row is None:
            continue
        gt_y = int(row['yaw_bin_4'])
        gt_v = int(row['height_bin'])
        gt_j = joint_index(gt_y, gt_v)
        py, pv, pj = empirical_distribution(samples)
        # NLL
        nll_y = -np.log(max(py[gt_y - 1], EPS))
        nll_v = -np.log(max(pv[gt_v], EPS))
        nll_j = -np.log(max(pj[gt_j], EPS))
        # Brier joint
        gt_oh = np.zeros(N_JOINT); gt_oh[gt_j] = 1
        brier_j = float(np.sum((pj - gt_oh) ** 2))
        # Circular RPS yaw
        rps_y = circular_rps(py, gt_y - 1)
        # Mode acc
        mode_y = int(np.argmax(py)) + 1  # 1-3
        mode_v = int(np.argmax(pv))      # 0-2
        mode_j = int(np.argmax(pj))
        mode_y_exact = int(mode_y == gt_y)
        mode_v_exact = int(mode_v == gt_v)
        mode_j_exact = int(mode_j == gt_j)
        # Yaw ±1 (circular over 3 bins: any non-front yaw considered ±1 here)
        # In 3-bin yaw [R, B, L], distance is min(|m-g|, 3-|m-g|). ±1 means dist≤1.
        d_y = abs(mode_y - gt_y)
        d_y_circ = min(d_y, N_YAW - d_y)
        mode_y_pm1 = int(d_y_circ <= 1)

        out[sid] = {
            'gt_yaw': gt_y, 'gt_vert': gt_v, 'gt_joint': gt_j,
            'p_yaw': py, 'p_vert': pv, 'p_joint': pj,
            'nll_y': nll_y, 'nll_v': nll_v, 'nll_j': nll_j,
            'brier_j': brier_j, 'rps_y': rps_y,
            'mode_y_exact': mode_y_exact, 'mode_v_exact': mode_v_exact,
            'mode_j_exact': mode_j_exact, 'mode_y_pm1': mode_y_pm1,
            'm_eff': len(samples),
        }
    return out


# ── Per-pair pairwise metrics ────────────────────────────────────────
def per_pair_pairwise_metrics(per_query: dict, set_df: pd.DataFrame,
                              captions: dict, pw_joint: dict,
                              pw_yaw: dict, pw_vert: dict,
                              pairs: list, anchor_subset: list = None):
    """For each pair, average model distributions over queries hitting that pair,
    then compute JSD/KL vs P_pairwise."""
    # Build query -> visible anchors
    sid_to_anchors = {}
    for _, row in set_df.iterrows():
        ds = row['dataset']; vid = row['video_id']; fidx = int(row['frame_index'])
        fk = f'{ds}:{vid}:{fidx}'
        cap = captions.get(fk, '')
        anchors = detect_anchors_in_caption(cap)
        if anchor_subset is not None:
            anchors = anchors & set(anchor_subset)
        sid_to_anchors[row['sample_id']] = anchors

    sid_to_label = dict(zip(set_df.sample_id, set_df.canonical_label))

    rows = []
    for target, anchor in pairs:
        # Find queries in this pair
        m_yaws, m_verts, m_joints, n = [], [], [], 0
        for sid, q in per_query.items():
            if sid_to_label.get(sid) != target:
                continue
            if anchor not in sid_to_anchors.get(sid, set()):
                continue
            m_yaws.append(q['p_yaw'])
            m_verts.append(q['p_vert'])
            m_joints.append(q['p_joint'])
            n += 1
        if n == 0:
            continue
        # Average model dist over queries in this pair
        m_yaw = np.mean(m_yaws, axis=0); m_yaw = m_yaw / m_yaw.sum()
        m_vert = np.mean(m_verts, axis=0); m_vert = m_vert / m_vert.sum()
        m_joint = np.mean(m_joints, axis=0); m_joint = m_joint / m_joint.sum()
        # P_pairwise
        ppw_yaw = pw_yaw.get((target, anchor))
        ppw_vert = pw_vert.get((target, anchor))
        ppw_joint = pw_joint.get((target, anchor))
        if ppw_yaw is None:
            continue
        rows.append({
            'target': target, 'anchor': anchor, 'n_pair': n,
            'jsd_yaw': jsd(m_yaw, ppw_yaw),
            'kl_pw_m_yaw': kl(ppw_yaw, m_yaw),
            'kl_m_pw_yaw': kl(m_yaw, ppw_yaw),
            'jsd_joint': jsd(m_joint, ppw_joint),
            'kl_pw_m_joint': kl(ppw_joint, m_joint),
            'kl_m_pw_joint': kl(m_joint, ppw_joint),
            'h_pw_yaw': float(-np.sum(rel_entr(ppw_yaw, np.ones(N_YAW) / N_YAW))),  # neg KL from uniform
        })
    return pd.DataFrame(rows)


# ── Set-level summary ────────────────────────────────────────────────
def set_summary(set_df: pd.DataFrame, captions: dict, pw_counts: dict):
    """Set-level facts: bin distributions, scene/label coverage, pair counts."""
    out = {}
    n = len(set_df)
    out['n'] = n
    out['n_ek'] = int((set_df.dataset == 'epic_kitchens').sum())
    out['n_hd'] = int((set_df.dataset == 'hd_epic').sum())
    out['labels'] = int(set_df.canonical_label.nunique())
    # GT yaw / vert / joint
    yaw_counts = np.array([(set_df.yaw_bin_4 == y).sum() for y in ALLOWED_YAW])
    vert_counts = np.array([(set_df.height_bin == v).sum() for v in range(N_VERT)])
    out['gt_yaw'] = (yaw_counts / yaw_counts.sum()).tolist()
    out['gt_vert'] = (vert_counts / vert_counts.sum()).tolist()
    out['tv_yaw_uniform'] = tv_distance(yaw_counts, np.ones(N_YAW))
    out['tv_vert_uniform'] = tv_distance(vert_counts, np.ones(N_VERT))
    # Scene coverage
    out['scenes'] = sorted(set_df.participant_id.unique().tolist())
    out['per_scene'] = set_df.participant_id.value_counts().sort_index().to_dict()
    # Pair counts at thresholds
    for fl in ['A', 'B', 'C']:
        for n_min in [5, 10, 15, 20]:
            pairs = filter_pairs(pw_counts, n_min=n_min, filter_level=fl)
            out[f'n_pairs_filter{fl}_n{n_min}'] = len(pairs)
    return out


# ── Aggregate over filter ────────────────────────────────────────────
def aggregate_metrics_filter(pair_df: pd.DataFrame, pairs: list) -> dict:
    pair_df = pair_df[pair_df.apply(
        lambda r: (r['target'], r['anchor']) in pairs, axis=1)]
    if len(pair_df) == 0:
        return {'n_pairs': 0}
    return {
        'n_pairs': len(pair_df),
        'mean_jsd_yaw': float(pair_df.jsd_yaw.mean()),
        'mean_kl_pw_m_yaw': float(pair_df.kl_pw_m_yaw.mean()),
        'mean_kl_m_pw_yaw': float(pair_df.kl_m_pw_yaw.mean()),
        'mean_jsd_joint': float(pair_df.jsd_joint.mean()),
        'mean_kl_pw_m_joint': float(pair_df.kl_pw_m_joint.mean()),
        'mean_kl_m_pw_joint': float(pair_df.kl_m_pw_joint.mean()),
    }


# ── Main ─────────────────────────────────────────────────────────────
def load_captions():
    caps = {}
    p = DATA_DIR / 'captions_v2.jsonl'
    if not p.exists():
        return caps
    with open(p) as f:
        for line in f:
            r = json.loads(line)
            caps[r['frame_key']] = r['caption']
    return caps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--partial', action='store_true',
                        help='Tolerate incomplete eval runs')
    args = parser.parse_args()

    sets = {}
    for s in ['A', 'B']:
        p = DATA_DIR / f'set{s}_300_filtered.parquet'
        if not p.exists():
            logger.error(f'Missing: {p}')
            return
        sets[s] = pd.read_parquet(p)

    captions = load_captions()
    logger.info(f'Loaded {len(captions)} captions')

    # Set-level summaries
    set_summaries = {}
    set_pw = {}  # (set) -> (pw_yaw, pw_vert, pw_joint, counts)
    for sname, sdf in sets.items():
        pw_yaw, pw_vert, pw_joint, counts = build_pairwise_priors(
            sdf, captions)
        set_pw[sname] = (pw_yaw, pw_vert, pw_joint, counts)
        set_summaries[sname] = set_summary(sdf, captions, counts)
        logger.info(f'\n=== Set {sname} summary ===')
        for k, v in set_summaries[sname].items():
            logger.info(f'  {k}: {v}')

    # Find all eval runs
    runs = sorted(RUNS_DIR.glob('*.jsonl'))
    logger.info(f'\nFound {len(runs)} eval JSONL files in {RUNS_DIR}')

    # Aggregate per (set, condition, model)
    rows = []
    pair_rows = []
    for path in runs:
        m = re.match(r'(A|B)_(\w[\w-]*)_(\w[\w.-]*)_M25\.jsonl', path.name)
        if not m:
            logger.warning(f'  Skip unparseable filename: {path.name}')
            continue
        s, cond, model = m.group(1), m.group(2), m.group(3)
        by_sid, n_total, n_valid = load_run(path)
        if not by_sid:
            logger.info(f'  {path.name}: empty')
            continue
        valid_rate = n_valid / n_total if n_total else 0
        m_eff_avg = np.mean([len(v) for v in by_sid.values()])

        per_q = per_query_metrics(by_sid, sets[s])

        # Mode accuracy + Brier + RPS only — distributional NLL/JSD are
        # produced by the canonical pipelines (see header note).
        brier_j = np.mean([q['brier_j'] for q in per_q.values()])
        rps_y = np.mean([q['rps_y'] for q in per_q.values()])
        mode_y = np.mean([q['mode_y_exact'] for q in per_q.values()])
        mode_v = np.mean([q['mode_v_exact'] for q in per_q.values()])
        mode_j = np.mean([q['mode_j_exact'] for q in per_q.values()])
        mode_y_pm1 = np.mean([q['mode_y_pm1'] for q in per_q.values()])

        row = {
            'set': s, 'condition': cond, 'model': model,
            'n_queries': len(per_q),
            'valid_rate': valid_rate, 'm_eff_avg': m_eff_avg,
            'Brier_j': brier_j, 'RPS_y': rps_y,
            'ModeY%': mode_y * 100, 'ModeV%': mode_v * 100,
            'ModeJ%': mode_j * 100, 'ModeY±1%': mode_y_pm1 * 100,
        }
        rows.append(row)
        logger.info(f'  {path.name}: n={len(per_q)}, '
                    f'ModeJ%={mode_j*100:.1f}')

    # Save
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / 'aggregate_metrics.csv', index=False)
    logger.info(f'\nSaved: {OUT_DIR / "aggregate_metrics.csv"}')

    # Save set summaries
    with open(OUT_DIR / 'set_summaries.json', 'w') as f:
        json.dump(set_summaries, f, indent=2, default=str)
    logger.info(f'Saved: {OUT_DIR / "set_summaries.json"}')


if __name__ == '__main__':
    main()
