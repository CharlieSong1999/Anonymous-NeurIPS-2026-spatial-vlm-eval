# Running Experiments on the `setA_extended` Test Set

**Date**: 2026-04-27
**Status**: living document
**Audience**: anyone who wants to run a new VLM evaluation,
add a model, build a new metric, or extend the test set.

---

## TL;DR

Everything in this project's "v2" evaluation pipeline lives under
`meta/testset/`. The canonical test set is **`setA_extended`** — 2000
queries (1455 HD-Epic + 545 Epic-Kitchens).

The task is **open-ended bin distribution**: a 4-bin yaw × 3-bin
height = 12-bin joint label. Models output JSON with `yaw_bin_id` and
`pitch`; we report metrics over M=20 samples per query.

To add a new evaluation: copy `src/eval_v2.py`, change the prompt
builder / model list / condition logic, run against the local vLLM
servers or API endpoints. To compute new metrics: read JSONL output,
group by (sample_id, condition, model), aggregate.

**Mandatory pre-read**: [`cautious-on-hd-epic.md`](./cautious-on-hd-epic.md)
covers two non-obvious data quirks (Aria pose convention, fisheye
undistortion). Anything that processes HD frames or poses outside the
canonical loaders MUST account for them.

---

## §1 Data layout

```
meta/testset/
├── data/                                       # all test-set artifacts
│   ├── setA_extended.parquet                   # ★ 2000-query canonical test set ("full")
│   ├── setA_extended_filtered.parquet          # same after caption-based filter
│   ├── setA_extended_subset500.parquet         # 500-query subset (legacy M-ablation)
│   ├── setA_tiny_300.parquet                   # ★ 300-query stratified subset (covers the 100-pool, matches 2k distribution)
│   ├── subset_membership.json                  # ★ {"100": [...], "300": [...]} mapping for --eval-set
│   ├── human_eval/pool_state.json              # 100 sample_ids in the human-eval active pool
│   ├── captions_v2.jsonl                       # Gemini-3-flash captions for setA (300-version)
│   ├── captions_v2_extended.jsonl              # captions for setA_extended frames
│   └── (legacy 300-query sets, build artifacts, candidate pools)
├── src/                                        # all production scripts
│   ├── build_extended_testset.py               # builds setA_extended from candidate pool
│   ├── caption_and_filter_extended.py          # captions + drops queries where target visible
│   ├── eval_v2.py                              # ★ Q1 eval pipeline
│   ├── metrics_v2.py                           # ★ Q1 metrics
│   ├── purge_hd_jsonl.py                       # remove HD records from eval JSONLs
│   ├── m_ablation*.py                          # M (samples-per-query) ablations
│   ├── build_tiny_subset.py                    # builds setA_tiny_300.parquet + subset_membership.json
│   ├── _eval_set.py                            # shared --eval-set helper used by every eval script
│   └── …
└── exp/                                        # all evaluation outputs
    ├── m_ablation_001/                         # M-ablation experiment + smoothing comparison
    ├── q2_pilot_001/                           # Q2 design / strategy review
    │   ├── strategy_comparison.md
    │   └── rotation_fix/                       # before/after HD pose+undistort comparisons
    ├── q2_eval_001/                            # Q2 evaluation (current results live here)
    │   ├── runs/{condition}_{model}_M{M}.jsonl
    │   └── results/{aggregate,per_face,q1_q2_joint}.csv + results.md
    ├── sampling_ablation_001/                  # 300-query Q1 evaluation (sighted/blind/cot-e)
    │   ├── runs/{set}_{cond}_{model}_M25.jsonl
    │   └── results/
    └── sampling_ablation_001.md
```

The original Q1 eval was on the 300-query `setA_300_filtered` test
set; the M-ablation used `setA_extended_subset500`. New full
evaluations should use `setA_extended` (2000 queries).

### Subset switch — `--eval-set {full,300,100}`

Every eval and ablation script accepts a `--eval-set` flag that filters
the loaded parquet down to one of three nested subsets:

| value | rows | use case |
|---|---|---|
| `full` (default) | whatever the input parquet has (typically 2000 for Q1, 1975 per Q2 strategy) | headline results |
| `300` | 300 rows from `setA_tiny_300.parquet`, stratified on `(dataset, yaw_bin_4, height_bin)` to match the 2k distribution; **superset of `100`** | fast ablations |
| `100` | the 100 ids in `data/human_eval/pool_state.json` `active` field; the human-trial sample | comparing model vs. human |

`100 ⊂ 300 ⊂ full`, so a model run on `full` already contains the
answers for the smaller subsets — you can re-aggregate metrics by
subset without re-querying.

Source of truth: `meta/testset/data/subset_membership.json`. To
regenerate (e.g. after a new test-set build), run

```bash
conda run -n slam python3 -m src.build_tiny_subset
```

Q1 scripts filter on the `sample_id` column; Q2 scripts filter on the
`q1_sample_id` column (the foreign key into Q1). The `_eval_set.py`
helper handles both — see its docstring.

---

## §2 The test set schema

`setA_extended.parquet` columns (one row per query):

| Column | Type | Description |
|---|---|---|
| `sample_id` | str | Stable id, e.g. `q1_epic_kitchens_…_2_1501_1` or `hd_epic_P01-…_yb6` |
| `dataset` | str | `epic_kitchens` / `hd_epic` / `hd_extended` |
| `participant_id` | str | e.g. `P02`, `P01` |
| `video_id` | str | e.g. `P02_03` (EK) or `P01-20240204-142301` (HD) |
| `frame_index` | int | Frame number — note **EK uses 10-digit, HD uses 6-digit** padding |
| `canonical_label` | str | Target object name (e.g. `washing machine`) |
| `target_world_xyz` | array(3) | Target 3D location, world coords (meters) |
| `camera_position` | array(3) | Camera origin in world coords (meters) |
| `camera_rotation_flat` | array(9) | Row-major flattened R_wc rotation matrix |
| `world_up` | array(3) | Unit world-up vector (gravity direction) |
| `hfov`, `vfov` | float | Camera FOV in degrees (see HD undistortion caveat below) |
| `yaw_deg`, `pitch_deg` | float | Camera-relative target direction (continuous) |
| `yaw_bin_4` | int 1-3 | 4-bin yaw bin ID (front=0 excluded; 1=Right, 2=Back, 3=Left) |
| `pitch_bin` | int 0-2 | 3-bin pitch (0=down, 1=level, 2=up) |
| `height_bin` | int 0-2 | Same as pitch_bin (alias) |
| `yaw_bin_12` | int 0-11 | 12-bin yaw (used by Track E experiments) |
| `height_above_floor` | float | Target height above floor (meters) |
| `tilt_from_vert` | float | Camera tilt from vertical (degrees) |

**Joint bin** (used in Q1 metrics) = `(yaw_bin_4 - 1) * 3 + height_bin`,
range 0-8 (9 bins).

**Allowed yaw bins** for the "unseen" task: 1, 2, 3 (forward bin 0
is excluded — target must NOT be in front of camera).

---

## §3 Loading data correctly (the boring but critical part)

### 3.1 Loading a frame

EK and HD-Epic differ in path layout, frame index padding, AND
preprocessing. Always use the canonical resolver:

```python
EK_ROOT = Path("<EK_FRAMES_ROOT>")
HD_ROOT = Path(
    "<HD_EPIC_REFINED_ROOT>/data_refinement/"
    "ego_pipeline_test/data/Participants"
)

def resolve_frame_path(dataset, video_id, frame_index, participant_id):
    if dataset == 'epic_kitchens':
        return EK_ROOT / participant_id / video_id / 'frames' / f'frame_{frame_index:010d}.jpg'
    return HD_ROOT / participant_id / video_id / 'images' / f'frame_{frame_index:06d}.jpg'
```

For HD-Epic frames, **frames are raw Aria fisheye** — see §3.2 for
the undistortion recipe if your work needs accurate geometry. The
plain `Image.open(path)` is sufficient for "send the original frame
to a VLM" (Q1 sighted condition); it is NOT sufficient for any
projection / cubemap rendering.

### 3.2 Loading HD-Epic with undistortion (for projection work)

The reference implementation lives in
[`src/q2_pilot.py`](../../testset/src/q2_pilot.py) (`_get_hd_calibrations`,
`load_frame`). The short version:

```python
from projectaria_tools.core import data_provider
from projectaria_tools.core.calibration import (
    distort_by_calibration, get_linear_camera_calibration,
    rotate_camera_calib_cw90deg,
)

HD_VRS = ("<HD_EPIC_SOURCE_ROOT>/video/hd-epic/"
          "vrs/HD-EPIC/VRS/P01/P01-20240202-110250_anonymized.vrs")
_CACHE = {}

def get_hd_calib():
    if 'src_rot' in _CACHE:
        return _CACHE['src_rot'], _CACHE['dst'], _CACHE['K']
    src = (data_provider.create_vrs_data_provider(HD_VRS)
           .get_device_calibration().get_camera_calib('camera-rgb'))
    src_rot = rotate_camera_calib_cw90deg(src)            # match upright frame orientation
    dst = get_linear_camera_calibration(1408, 1408, 612.0, 'rgb-linear')
    fx, fy = dst.get_focal_lengths(); cx, cy = dst.get_principal_point()
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    _CACHE.update({'src_rot': src_rot, 'dst': dst, 'K': K})
    return src_rot, dst, K

def load_hd_frame_undistorted(path) -> Image.Image:
    img = np.asarray(Image.open(path).convert('RGB'))
    src_rot, dst, _ = get_hd_calib()
    out = distort_by_calibration(img, dst, src_rot)
    if out.dtype != np.uint8:
        out = np.clip(out, 0, 255).astype(np.uint8)
    return Image.fromarray(out)
```

After undistortion: image is 1408×1408, true pinhole, `K=(612, 612,
703.5, 703.5)`, HFOV ≈ 98°. **Use this K for downstream projection,
NOT the `hfov`/`vfov` fields stored in the parquet** — those describe
the original fisheye and are wrong post-rectification.

### 3.3 Loading the camera pose

The pose in `camera_rotation_flat` is **OpenCV convention for EK,
Aria-native for HD-Epic**. For HD, you must remap before doing any
OpenCV-style projection:

```python
ARIA_TO_OPENCV_POSE_REMAP = np.array([
    [0,  1, 0],
    [-1, 0, 0],
    [0,  0, 1],
], dtype=np.float64)

def get_pose(row):
    R_wc = np.asarray(row['camera_rotation_flat']).reshape(3, 3)
    if row['dataset'] != 'epic_kitchens':
        R_wc = R_wc @ ARIA_TO_OPENCV_POSE_REMAP
    return R_wc
```

After remap, `R_wc[:,0] · world_up ≈ 0` (X is horizontal=right),
`R_wc[:,1] · world_up ≈ -sin(tilt)` (Y is mostly down).

**Sanity-check** before any HD batch run:

```python
assert abs(np.dot(R_wc[:,0], world_up)) < 0.3, \
    "HD pose appears Aria-native — apply remap"
```

See `cautious-on-hd-epic.md` for the full background on why this is
required and which bugs it caused.

### 3.4 Loading the captions

Captions for visible-anchor extraction (used by the pairwise-metric
evaluation):

```python
import json
captions = {}
with open('meta/testset/data/captions_v2_extended.jsonl') as f:
    for line in f:
        r = json.loads(line)
        # frame_key format: "{dataset}:{video_id}:{frame_index}"
        captions[r['frame_key']] = r['caption']
```

The captions are from Gemini-3-flash and describe what's visible
in the frame (anchors, layout, gravity orientation). Used for the
pairwise spatial prior — see `meta/work/discussion/pairwise-metric.md`.

---

## §4 Running a Q1 evaluation (open-ended bin prediction)

### 4.1 Reference: `src/eval_v2.py`

Q1 sends a single perspective frame + a prompt asking the model to
predict the target's yaw_bin_id ∈ {1,2,3} and pitch ∈ {UP, LEVEL,
DOWN}. Output is JSON.

### 4.2 The canonical sighted prompt

From `src/eval_v2.py:build_sighted_prompt`:

```text
You are given ONE egocentric RGB image (first-person view).
The target object "{label}" is OUT OF VIEW (not visible in the image).
Predict its most likely direction based on visible layout cues
(walls, counters, appliances, doorways, free space) and spatial commonsense.

There are 4 direction bins (90° each, clockwise from camera forward):
  Bin 0 = Front (12 o'clock — EXCLUDED, target must be out of view)
  Bin 1 = Right (3 o'clock)
  Bin 2 = Back  (6 o'clock)
  Bin 3 = Left  (9 o'clock)
And 3 height levels relative to the kitchen benchtop:
  UP    = above benchtop ...
  LEVEL = on benchtop / work zone ...
  DOWN  = below benchtop ...

Output a single JSON object:
  {"justification": "<1-2 sentences>",
   "yaw_bin_id": <integer 1-3>,
   "pitch": "<UP|LEVEL|DOWN>"}
```

There are also `build_blind_prompt` (no image) and `build_cot_e_prompt`
(chain-of-thought + image). All use the same JSON output schema.

### 4.3 Running it

```bash
# Headline (300-query setA_300_filtered + setB_300_filtered, all conditions)
conda run -n slam python3 -m src.eval_v2 \
    --sets A B \
    --conditions sighted blind cot-e \
    --models qwen3.5-9b gemma-4-31b qwen3-vl-30b

# Tiny ablation (300 rows from the 2k via stratified sampling — fast)
conda run -n slam python3 -m src.eval_v2 \
    --sets A --conditions sighted --models qwen3.5-9b \
    --eval-set 300

# Smoke test on the 100-query human-trial pool
conda run -n slam python3 -m src.eval_v2 \
    --sets A --conditions sighted --models qwen3.5-9b \
    --eval-set 100
```

`--eval-set 300` and `--eval-set 100` filter the loaded set parquet by
`sample_id`. Default is `full` (no filtering, equivalent to the
historical behaviour).

For the **extended (2000-query) test set**, point `--sets` at the
extended parquet directly: `--sets extended_filtered` would resolve to
`setA_extended_filtered.parquet`. The `--eval-set` filter applies
identically to whichever parquet was loaded.

### 4.4 Output

JSONL at `exp/sampling_ablation_001/runs/{set}_{cond}_{model}_M{M}.jsonl`,
one record per query × repeat. Schema:
```json
{
  "sample_id": "...",
  "repeat_id": 0..M-1,
  "label": "washing machine",
  "gt_yaw": 1, "gt_height": 0,
  "raw": "<full model response text>",
  "parsed": {"yaw_bin_id": 2, "pitch": "DOWN", "justification": "..."}
}
```

### 4.5 Compute Q1 metrics

The metric pipeline is split into three scripts. Each accepts
`--eval-set {full,300,100,...}` (see per-script choices below).

#### 4.5.1 Mode accuracy + Brier + RPS — quick sanity table

```bash
conda run -n slam python3 -m src.metrics_v2
```

Outputs `aggregate_metrics.csv` + `set_summaries.json` under
`exp/sampling_ablation_001/results/`. Per-(set, condition, model) Mode%
on yaw / vert / joint (plus `±1` yaw and Brier / RPS).

#### 4.5.2 Calibrated NLL family — `compare_averaging.py`

```bash
# Both human-pool and full-setA tables (default)
conda run -n slam python <PROJECT_ROOT>/metric/NLL/calibration/compare_averaging.py

# (Subset filter accepted; renders both tables identically for now.)
conda run -n slam python .../compare_averaging.py --eval-set 100
```

For each task ∈ {yaw, height, joint} this writes
`per_sample_vs_per_category_{yaw,height,joint}.md` under
`meta/metric/NLL/calibration/`. Per row × per task × per
averaging-mode (sample / category) it reports:

| Column | Meaning |
|---|---|
| `Acc` | argmax accuracy (T-invariant) |
| `Floored NLL` | empirical with ε-floor; T-independent, M-sensitive |
| `Smooth NLL` | `−log p_{j*}` at T=1 (Dirichlet plug-in only) |
| `Calib NLL` | `−log p_{j*}^T` at chosen T* — **headline** |
| `T*` | tuned inverse-temp (∞ when β collapses to 0) |
| `H_pre / H_post` | normalised entropy at T=1 / T=T* (0=peaky, 1=uniform) |
| `Oracle T`, `δ_oracle` | overfit detector |

Plus a side block: `KL(h‖m)`, `KL(m‖h)`, `JSD(h,m)` of each row's
smoothed distribution against the **`humans (ensemble) M=10`**
reference. Both `_s` (per-sample mean) and `_c` (per-category mean)
columns. Reported only on the human-pool table.

Plus a per-category breakdown + difficulty ranking on the calibrated
NLL.

Locked params:
- Cal/eval split: `meta/metric/NLL/calibration/split_30_70.json`
  (cal: P07/P19/P22/P31/P33/P35; eval: P01/P02/P06/P08/P09)
- Dirichlet α: 0.1
- Human ensemble: M=10 only (no admin row dropped per spec)

#### 4.5.3 Group-KL / Group-JSD vs GT pairwise distribution — `group_kl.py`

```bash
# All three subsets in one run (default)
conda run -n slam python <PROJECT_ROOT>/metric/GroupKL/group_kl.py

# Just one subset (cheaper)
conda run -n slam python .../group_kl.py --eval-set 100
conda run -n slam python .../group_kl.py --eval-set 300
conda run -n slam python .../group_kl.py --eval-set full
```

Writes `groupkl_results.md` (or `groupkl_results_<subset>.md`) under
`meta/metric/GroupKL/`. For each task ∈ {yaw, pitch, joint} × each
subset, reports:

| Filter | Definition |
|---|---|
| **A** — strict same-zone | sink-zone OR stove-zone, target+anchor in same zone |
| **B** — recommended primary | per-target allowlist, generic anchors `{backsplash, cabinet, counter}` excluded |
| **C** — B + generic anchors | most permissive |

Per filter: per-pair `JSD(P_gt, P_model)`, `KL(g‖m)`, `KL(m‖g)`,
averaged across pairs. Pairs with `n_frames < min_n` dropped
(`min_n` = 10 / 5 / 2 for full / 300 / 100 respectively).

Includes `humans (ensemble)` row on the 100-pool only.

#### 4.5.4 What was discarded (was NLL/JSD inside `metrics_v2.py`)

The old smoothed-NLL block (NLL_y, NLL_v, NLL_j) and the old pairwise
JSD vs `P_pairwise` block in `metrics_v2.py` are **gone**. Old
`exp/sampling_ablation_001/results/{aggregate.csv, per_pair_metrics.csv,
set_summaries.json}` will be overwritten on the next `metrics_v2` run
with the new (mode-only) schema.

### 4.6 Running M / smoothing ablations on a subset

The five `m_ablation_*.py` scripts all accept the same
`--eval-set {full,300,100}` flag. Typical workflow:

```bash
# 1. Run model on the 300 subset (fast — ~6× cheaper than the 2k)
conda run -n slam python3 -m src.m_ablation_eval \
    --models qwen3.5-9b gemma-4-31b \
    --M 25 --eval-set 300

# 2. Compute baseline metrics on the same 300
conda run -n slam python3 -m src.m_ablation \
    --set extended --eval-set 300

# 3. Sweep smoothing configs on the same 300
conda run -n slam python3 -m src.m_ablation_smoothing \
    --config all --set-parquet setA_extended_subset500.parquet \
    --eval-set 300

# 4. Dense M-sweep on the same 300
conda run -n slam python3 -m src.m_ablation_dense \
    --m-min 1 --m-max 50 --eval-set 300
```

Because `100 ⊂ 300 ⊂ full`, you can run with `--eval-set full` once
and re-aggregate any metric script with `--eval-set 100` or
`--eval-set 300` against the same JSONL outputs to see the metric
restricted to that sub-pool — no re-querying needed.


## §6 Adding a new model

Local model (vLLM-served, OpenAI-compatible):

```python
# In src/eval_v2.py, add to MODELS dict:
'newmodel-name': {
    'api': 'openai_compat',
    'base_url': 'http://127.0.0.1:PORT/v1/chat/completions',
    'model_id': 'org/model-id',
    'extra_body': {},   # e.g. {'chat_template_kwargs': {'enable_thinking': False}} for some Qwen
},
```

Spin up the server (assumes `vllm` conda env, weights in the
shared cache):

```bash
HF_HUB_CACHE=<HF_HUB_CACHE> \
  conda run -n vllm vllm serve org/model-id \
  --port PORT --max-model-len 4096 \
  --gpu-memory-utilization 0.92 --enforce-eager --trust-remote-code
```

API model (Gemini or OpenAI):

```python
'newapi-model': {
    'api': 'gemini',  # or 'openai'
    'model_id': 'real-model-id',
    'env_key': 'GEMINI_API_KEY',  # or 'OPENAI_API_KEY'
},
```

Adjust `API_CONCURRENCY` (default 4) if the API has different rate
limits. **Do NOT add API models to a routine eval without explicit
sign-off** — they cost real money.

---

## §7 Adding a new condition / prompt variant

For Q1, add a new builder in `src/eval_v2.py`:

```python
def build_my_condition_prompt(label: str) -> str:
    return f"""... your prompt with {label} substituted ..."""

CONDITIONS['my-condition'] = {
    'prompt_fn': build_my_condition_prompt,
    'send_image': True,
    'max_tokens': MAX_TOKENS,
}
```

Then run with `--conditions my-condition`.

For Q2, similar pattern in `src/q2_eval.py`:

```python
PROMPT_BUILDERS['my-variant'] = build_my_variant_prompt
INPUT_VARIANTS.append('my-variant')
```

The condition name in `q2_eval.py` is `{strategy}_{variant}`, e.g.
`A_my-variant`. Eval is then `--conditions A_my-variant`.

---

## §8 Adding a new metric

The metrics scripts (`metrics_v2.py`, `q2_metrics.py`) all follow the
same shape:

1. Iterate over `runs/{cell}.jsonl` files.
2. Load records into a DataFrame.
3. Group / aggregate per (model, condition).
4. Write a CSV + add a section to the `.md` report.

For new metrics, prefer adding to the existing scripts rather than
forking. The existing aggregate functions (`load_q2_records`,
`aggregate_cell`, `aggregate_per_face`, etc.) are generic enough to
be reused.

For distributional metrics (JSD, KL, NLL): use **plug-in MLE with
α = 1e-3** per the locked smoothing decision in
`exp/m_ablation_001/smoothing_comparison.md`. If you want
Miller-Madow correction, the helper functions are in
`src/m_ablation_smoothing.py` (`kl_with_correction`,
`jsd_with_correction`).

---

## §9 Running the human-trial Gradio app

```bash
conda run -n slam python3 -m src.q2_human_gradio --port 17861 --n 100
```

Picks 100 queries from `setA_extended_q2/A_random_diff_face`
deterministically (seed=2026), serves them at `http://localhost:17861`.
User picks A/B/C/D, sees ✓/✗ feedback (toggleable for blind trial),
running accuracy + per-letter pick distribution at the top. Trial
log saved to `exp/q2_eval_001/human_trial/trial_<timestamp>.jsonl`
on completion.

---

## §10 Common workflows

### 10.1 "I want to add a new model and run it on the standard Q2 setup"

1. Spin up the model server (or get API key in env).
2. Edit `MODELS` dict in `src/q2_eval.py`.
3. Smoke test: `--max 5 --models newmodel-name --conditions A_cubemap_only`.
4. Inspect a JSONL record — confirm `parsed.answer ∈ {A,B,C,D}`.
5. Full run: drop `--max`, queue the 4 conditions you want.
6. `python3 -m src.q2_metrics` (regenerates the rollup).

### 10.2 "I want to extend the test set with new labels"

1. Extend `pairs_master.parquet` (live in `query_gen/data/shared_labels/`)
   with new (cluster, frame) candidates.
2. Run `src/build_extended_testset.py` — handles label filter,
   synonym map, OOV/tilt filters, height-bin assignment.
3. Run `src/caption_and_filter_extended.py` — captions all new frames
   with Gemini-3-flash, drops queries where the target is mentioned
   in caption (i.e. visible after all).
4. Update doc, regenerate Q2 cubemaps if needed.
5. Re-run any evaluation that covers the changed label set.

### 10.3 "I want to make Q1 use the 2000-query set instead of 300"

Today, `eval_v2.py` only knows about `setA_300_filtered.parquet`
and `setB_300_filtered.parquet` (sets A and B from the original
300-query design). To use `setA_extended.parquet`:

1. Extend the `--sets` arg parser to accept `extended` as a value.
2. In `main_async`, branch on the set name and load
   `setA_extended_filtered.parquet`.
3. Output JSONL prefix becomes `extended_{cond}_{model}_M{M}.jsonl`.
4. The `metrics_v2.py` rollup reads everything in the runs dir, so
   it'll auto-pick up the new files. Add an `extended` row to the
   set-summary aggregation if you want a separate header.

---

## §11 Background: why things are the way they are

If you see something that looks weird, here are the top 5 things
worth knowing the history of:

| Quirk | Where to read |
|---|---|
| HD-Epic Aria pose convention; HD frame undistortion | [`cautious-on-hd-epic.md`](./cautious-on-hd-epic.md) |
| Why M=20 + α=1e-3 (and NOT α=0.5 / M=25) | [`testset/exp/m_ablation_001/smoothing_comparison.md`](../../testset/exp/m_ablation_001/smoothing_comparison.md) |
| Q2 distractor strategy choice (A vs B vs C); 2x3 layout | [`testset/exp/q2_pilot_001/strategy_comparison.md`](../../testset/exp/q2_pilot_001/strategy_comparison.md) |
| Why Set A (bin-balanced) over Set B (anchor-stratified) | [`testset/test_set_extension_plan.md`](../../testset/test_set_extension_plan.md) §"2026-04-26 — Decision: stick with Set A" |
| Pairwise spatial-prior metric (Filter A/B/C; what beats uniform) | [`work/discussion/pairwise-metric.md`](../discussion/pairwise-metric.md) |

---

## §12 Environment and infrastructure

| Resource | Detail |
|---|---|
| Conda env for eval scripts | `slam` (has projectaria_tools, numpy, pandas, PIL, gradio, httpx, etc.) |
| Conda env for vLLM server | `vllm` (v0.19.0) |
| GPU | Single NVIDIA RTX 5090, ~31 GB usable VRAM |
| HF cache | `<HF_HUB_CACHE>` (3.4 TB SSD) |
| Local model serving | One vLLM process per model (Qwen3.5-9B → :8001, Gemma-4-31B → :9002, Qwen3-VL-30B → :18002) |
| API keys | `GEMINI_API_KEY`, `OPENAI_API_KEY` in env (do NOT print these to logs) |
| Frame data — EK | `<EK_FRAMES_ROOT>/{P}/{V}/frames/frame_*.jpg` |
| Frame data — HD | `<HD_EPIC_FRAMES_ROOT>/{P}/{V}/images/frame_*.jpg` |
| HD VRS calibration | `<HD_EPIC_VRS_REFERENCE>` (used as generic for all participants) |

---

## §13 Pre-flight checklist for a new experiment

Before launching anything that takes more than 10 min wall-time, verify:

- [ ] You're using the canonical loaders (frame, pose, captions). If
  you wrote your own, it handles the EK vs HD path differences AND
  the HD pose remap AND the HD undistortion (if doing projection).
- [ ] Smoke test with `--max 5` (or equivalent) first. Inspect at
  least one JSONL record to confirm parsing works.
- [ ] Server health-check passes (`curl :PORT/v1/models`) for any
  vLLM server you'll hit.
- [ ] Resume-from-existing-JSONL is enabled (it is by default in
  `eval_v2.py` and `q2_eval.py`). Don't accidentally overwrite a
  cell — the prefix/path is unique per (set, condition, model, M).
- [ ] If running API models: explicit user sign-off, ballpark cost
  estimate written down.
- [ ] If processing HD frames: orientation invariant assertion
  (`abs(R_wc[:,0] · world_up) < 0.3` AFTER your loader). Catches
  Aria-pose-not-remapped silently.
- [ ] Background log file specified (`> /tmp/myexp.log 2>&1 &`).
  Set up a `Monitor` or `until` loop to detect completion.

---

## §14 Open issues / TODOs that affect new experiments

- Q1 has not been re-run on `setA_extended` (2000 queries) yet —
  current Q1 numbers are on the 300-query Set A. If you want Q1↔Q2
  comparison on the FULL set, run Q1 on `setA_extended` first
  (see §10.3).
- API models (gemini, gpt) on Q2 have only the `A_frame_plus_cubemap`
  condition. If you want them on cubemap-only or Strategy C, you'll
  need to launch those — current results are local-models-only for
  the other 3 cells.
- The `meta/outpaint/` panorama generation pipeline has NOT been
  audited for the HD pose+undistortion fix (it predates the bug
  discovery). If you re-use those panoramas in a new eval, audit them
  first against `cautious-on-hd-epic.md`.
- No CI / unit tests yet. `cautious-on-hd-epic.md` §7 has a TODO for
  a load-and-assert smoke test that should run on every commit.
