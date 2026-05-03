# Cautious-on-HD-Epic — Loading and Projecting HD-Epic Data Correctly

**Date**: 2026-04-27
**Status**: living document
**Audience**: anyone consuming HD-Epic frames and poses for spatial-reasoning
work in this project (testset, eval, panorama generation, anchor analysis,
etc.)

---

## TL;DR

HD-Epic frames + poses look "almost OpenCV" but have **two non-obvious quirks**.
Both have bitten us. Read this doc before you build a new pipeline that
reads HD-Epic data.

| # | Quirk | Severity | What to do |
|---|---|---|---|
| 1 | **Pose is in Aria native convention** (R[:,0]=down, R[:,1]=left, R[:,2]=forward), not OpenCV. The frame is pre-rotated to upright at preprocessing time, so frame orientation alone looks fine. | High — silently produces wrong cubemap projections / wrong viewing-direction maths | Apply a 90° pose remap before doing any OpenCV-style projection (see §1). |
| 2 | **Frames are raw Aria fisheye (1408×1408, ~100° HFOV)** — not undistorted. The project's canonical pinhole approximation (fx=fy=610) produces visually-acceptable but quantitatively-wrong projections, especially near the frame periphery. | Medium — visible distortion, slight error in spatial reasoning. Currently accepted. | Either (a) undistort with cv2.fisheye + Aria intrinsics, or (b) explicitly document that a pinhole approximation is being used. See §2. |

A third quirk worth noting (not Aria-specific):

| 3 | The HD-Epic frames live at `.../images/frame_*.jpg` (NOT `.../frames/frame_*.jpg` which is EK's convention) and use a **6-digit zero-padded** frame index (NOT EK's 10-digit). | Low — `FileNotFoundError` when loading | Use the dataset-aware path resolver in §3. |

---

## §1 The pose-convention quirk (the silent killer)

### What the docs say

From `meta/track-e/context.md:411–426`:

> Camera axes (Aria-specific — **NOT standard OpenCV**): Aria has a 90° CW
> roll relative to standard OpenCV.
> - `R[:,2]` = camera forward (Z) — same as OpenCV
> - `R[:,0]` points **downward** in the scene (not rightward).
>   `dot(R[:,0], world_up) ≈ −0.96`
> - `R[:,1]` points **leftward** in the scene (not downward).
>   `dot(R[:,1], world_up) ≈ 0`
> - Physical cam_right = `−R[:,1]` (negate Y column)
> - Physical cam_up = `−R[:,0]` (negate X column)

This is **about the pose only**. The pose stays in Aria native convention
even after the data_refinement pipeline pre-rotates the frames to upright
OpenCV-style orientation. **Pose and frame orientation do not match by
default**.

### How to recognise the bug visually

If you build a cubemap from an HD-Epic frame and project it through
standard OpenCV maths, the visible RGB region will:
- be positioned in the *wrong cubemap face* (e.g., visible content lands
  on the LEFT face when the camera was pointed FORWARD), or
- be split awkwardly across multiple faces, or
- have its content rotated 90° relative to the original frame.

Compare with EK in the same script — if EK looks fine and HD looks broken,
it's almost certainly this quirk.

### How to verify on actual data

```python
import numpy as np, pandas as pd
df = pd.read_parquet('meta/testset/data/setA_extended.parquet')
hd_row = df[df.dataset == 'hd_epic'].iloc[0]
R = np.asarray(hd_row['camera_rotation_flat']).reshape(3, 3)
world_up = np.asarray(hd_row['world_up'])

print('cam X . world_up:', np.dot(R[:, 0], world_up))   # should be ≈ -0.93 if Aria
print('cam Y . world_up:', np.dot(R[:, 1], world_up))   # should be ≈ 0
print('cam Z . world_up:', np.dot(R[:, 2], world_up))   # ≈ sin(camera tilt below horizon)
```

If `cam X · world_up` is strongly negative, the pose is Aria-native and
needs the remap below.

### The fix — apply once at load time

```python
# Maps Aria-native R_wc to OpenCV-convention R_wc.
# OpenCV X (right) = -Aria Y;  OpenCV Y (down) = +Aria X;  OpenCV Z (fwd) = +Aria Z.
ARIA_TO_OPENCV_POSE_REMAP = np.array([
    [0,  1, 0],
    [-1, 0, 0],
    [0,  0, 1],
], dtype=np.float64)

def load_hd_pose(R_wc_aria: np.ndarray) -> np.ndarray:
    """Convert HD-Epic Aria-native pose to OpenCV convention.
    Apply this once at load time; downstream code can then assume OpenCV."""
    return R_wc_aria @ ARIA_TO_OPENCV_POSE_REMAP
```

Verify after the remap:
```python
R_ok = load_hd_pose(R)
print('after remap, cam X . world_up:', np.dot(R_ok[:, 0], world_up))   # should be ≈ 0
print('after remap, cam Y . world_up:', np.dot(R_ok[:, 1], world_up))   # should be ≈ -0.93
```

### Do NOT also rotate the frame

The frame at `.../images/frame_NNNNNN.jpg` is already in upright OpenCV
orientation. **Do not apply `cv2.rotate` / `np.rot90` / `Image.rotate` to
the frame on load** — that produces a sideways frame and a broken cubemap.

### Where this bit us

- 2026-04-27: All 1455 HD-Epic queries in `data/setA_extended_q2/{A,C}/`
  cubemaps were rendered without the remap. Cubemap visible regions
  ended up in the wrong faces / orientations. Local-model Q2 results
  for HD frames are tainted; rerun required after regen. EK frames
  unaffected.
- Likely same issue in `outpaint/` panorama generation if the script
  pre-dates this discovery — needs an audit before the next outpaint
  run if HD frames are involved.

### Sanity-check command before any HD-Epic batch run

```python
# Add to the top of any pipeline that loads HD-Epic poses:
assert abs(np.dot(R_wc[:, 0], world_up)) < 0.3, (
    f"HD-Epic pose appears to be Aria-native (cam X . world_up = "
    f"{np.dot(R_wc[:,0], world_up):.2f}); apply ARIA_TO_OPENCV_POSE_REMAP")
```

---

## §2 The fisheye-distortion quirk — RESOLVED 2026-04-27

### Summary

Use **`projectaria_tools`** (already installed in the `slam` conda env) to
undistort HD-Epic frames at load time. The undistortion uses the
real Aria `FISHEYE624` calibration read from a representative VRS file.
After undistortion the frame is a true linear pinhole (no curved-line
artefacts) and standard OpenCV projection math applies cleanly.

### What the docs originally said

> Aria RGB camera is 1408×1408, fisheye623 model. Pinhole approximation
> focal length ≈ 610 px (from Aria MPS calibration).
>     — `data_refinement/query_gen/src/config.py:68–75`

This was the project-canonical *approximation* (effectively no
undistortion). It produced visible barrel distortion in cubemap visible
regions and ~5–10 % angular error at the periphery. Acceptable for the
direction-only task at coarse bin granularity, but inaccurate for the
visual review pass and for any pixel-accurate downstream work.

### The implementation

Aria's `FISHEYE624` model has 15 parameters: f, cx, cy, six radial
coefficients (k1–k6), two tangential (p1, p2), and four thin-prism
(s1–s4). The SDK exposes the full model via `CameraCalibration` plus
`distort_by_calibration(srcImage, dstCalib, srcCalib)` which performs
the inverse mapping in C++.

Two pieces are needed:

1. **Source calibration** (the actual Aria FISHEYE624 with its 15
   parameters). Loaded once from a representative VRS file. Per-device
   calibrations differ slightly (~5 % in focal length) across HD-Epic's
   small batch of glasses, so a single device's calibration is a fine
   generic for the whole dataset.

2. **Destination calibration** (a linear pinhole) — built via
   `get_linear_camera_calibration(W, H, focal, label)`. Choose a focal
   length close to the source's (~612 px) to preserve content scale.

Important: the stored upright frames are 90° rotated from the Aria
native sensor orientation. To make `distort_by_calibration` consume
upright pixels, **rotate the source calibration first** with
`rotate_camera_calib_cw90deg(src_calib)`.

### Reference implementation (production)

Lifted from `meta/testset/src/q2_pilot.py` (`_get_hd_calibrations`,
`load_frame`):

```python
from projectaria_tools.core import data_provider
from projectaria_tools.core.calibration import (
    get_linear_camera_calibration, distort_by_calibration,
    rotate_camera_calib_cw90deg,
)

HD_VRS_REFERENCE = ("<HD_EPIC_SOURCE_ROOT>/video/hd-epic/"
                    "vrs/HD-EPIC/VRS/P01/P01-20240202-110250_anonymized.vrs")

_HD_CALIB_CACHE = {}

def _get_hd_calibrations():
    """Return (src_calib_rotated, dst_linear_calib, K) — cached after first call."""
    if _HD_CALIB_CACHE:
        return _HD_CALIB_CACHE['src_rot'], _HD_CALIB_CACHE['dst'], _HD_CALIB_CACHE['K']
    provider = data_provider.create_vrs_data_provider(HD_VRS_REFERENCE)
    src = provider.get_device_calibration().get_camera_calib('camera-rgb')
    src_rot = rotate_camera_calib_cw90deg(src)
    W, H = 1408, 1408
    focal = float(src.get_focal_lengths()[0])  # 612.20
    dst = get_linear_camera_calibration(W, H, focal, 'rgb-linear')
    fx, fy = dst.get_focal_lengths()
    cx, cy = dst.get_principal_point()
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    _HD_CALIB_CACHE.update({'src_rot': src_rot, 'dst': dst, 'K': K})
    return src_rot, dst, K

def load_hd_epic_frame(path: str) -> Image.Image:
    img = np.asarray(Image.open(path).convert('RGB'))
    src_rot, dst, _ = _get_hd_calibrations()
    undist = distort_by_calibration(img, dst, src_rot)
    if undist.dtype != np.uint8:
        undist = np.clip(undist, 0, 255).astype(np.uint8)
    return Image.fromarray(undist)
```

### Geometry after undistortion

| Quantity | Value |
|---|---:|
| Image size | 1408 × 1408 (unchanged) |
| Focal length | 612.20 px (matches source) |
| Principal point | (703.5, 703.5) — image centre |
| HFOV / VFOV | 2 · atan(704/612) ≈ **98°** |
| Visible (non-black) circle radius | ~600 px (≈ 88° HFOV; outside this is black because fisheye captures wider FOV than pinhole can re-render) |
| Distortion | zero (true pinhole) |

Use this K (or pull from `dst.get_focal_lengths()` /
`dst.get_principal_point()`) for any downstream projection on the
undistorted frame. Do **not** use the `hfov`/`vfov` fields stored in
the v2 test set parquet for HD-Epic — those describe the original
fisheye and are wrong for the rectified frame.

### Caveats

- **Single-device calibration used for all participants.** P01's VRS
  calibration is applied to P02–P09 frames as well. Per-device error
  is ~5 % in focal length — acceptable for direction prediction, marker
  placement, and visual review. If you ever need higher fidelity:
  iterate over `vrs/HD-EPIC/VRS/{participant}/...vrs` files and cache
  per-participant calibrations. Some participants have no VRS in this
  data tree; for those, fall back to P01's.
- **Periphery is now black** (vs the old fisheye which filled the
  whole frame). Visible content is the central ~88° HFOV circle.
  Cubemap projections sample only this region; markers go in the
  black periphery + dark unseen-region of the cubemap face.
- **Per-frame cost**: ~50 ms for the `distort_by_calibration` call on
  a 1408×1408 frame (single-threaded). Calibrations are cached at
  module level, so the VRS load (~2 s) only happens once.
- **Old assets baked with the un-undistorted pipeline are now stale**
  for HD-Epic. Anything in `meta/testset/data/setA_extended_q2/` that
  was rendered before 2026-04-27 should be regenerated. EK assets are
  unaffected.

### Verification

Render any HD-Epic frame through this pipeline and check that originally-
straight lines (cabinets, door frames, window mullions, counter edges)
appear straight. See `meta/testset/exp/q2_pilot_001/rotation_fix/` for
4 worked examples (raw fisheye | undistorted | cubemap before fix |
cubemap after fix).

---

## §3 The frame-path quirk (just be careful)

EK and HD-Epic frames live in different folder structures:

| Dataset | Path template |
|---|---|
| `epic_kitchens` | `<EK_FRAMES_ROOT>/{participant_id}/{video_id}/frames/frame_{frame_index:010d}.jpg` |
| `hd_epic` and `hd_extended` | `<HD_EPIC_FRAMES_ROOT>/{participant_id}/{video_id}/images/frame_{frame_index:06d}.jpg` |

Note the **`frames/` vs `images/` subdirectory** and the **10-digit vs
6-digit padding** of `frame_index`. A naïve unified resolver will silently
fail to find HD-Epic frames.

### Correct resolver

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

This resolver appears in `meta/testset/src/eval_v2.py` and
`meta/testset/src/q2_pilot.py`. Reuse it; do not write a new one.

### Where this bit us

- 2026-04-27: First Q2-asset generation skipped all 1455 HD-Epic frames
  silently because the path used `.../images/` was missing — the
  resolver was using `.../frames/` for HD too. The bug was caught by
  noticing only 545 (= EK count) cubemaps were generated. **Always
  print `n_loaded` and assert `>= expected` after a batch load.**

---

## §4 Canonical loader for HD-Epic in this project

Putting it all together — the function every script should use to
load an HD-Epic (frame, pose, intrinsics) tuple correctly:

```python
import numpy as np
from pathlib import Path
from PIL import Image

ARIA_TO_OPENCV_POSE_REMAP = np.array([
    [0,  1, 0],
    [-1, 0, 0],
    [0,  0, 1],
], dtype=np.float64)

def load_hd_epic(row) -> tuple:
    """Return (frame_pil, R_wc_opencv, K, world_up).

    Assumes `row` has fields:
      dataset, participant_id, video_id, frame_index,
      camera_rotation_flat (9), world_up (3), hfov (deg), vfov (deg).

    Notes:
      - Frame is loaded raw — fisheye distortion is NOT removed.
      - Pose is remapped from Aria-native to OpenCV convention.
      - K uses the project's pinhole approximation. Document the
        approximation in any quantitative report.
    """
    assert row['dataset'] in ('hd_epic', 'hd_extended'), (
        "load_hd_epic is for HD-Epic only")
    path = (HD_ROOT / row['participant_id'] / row['video_id']
            / 'images' / f"frame_{int(row['frame_index']):06d}.jpg")
    frame = Image.open(path).convert('RGB')

    R_aria = np.asarray(row['camera_rotation_flat']).reshape(3, 3)
    R_wc_opencv = R_aria @ ARIA_TO_OPENCV_POSE_REMAP

    W, H = frame.size  # 1408 × 1408
    hfov = np.radians(float(row['hfov']))
    vfov = np.radians(float(row['vfov']))
    K = np.array([
        [(W / 2) / np.tan(hfov / 2), 0, W / 2],
        [0, (H / 2) / np.tan(vfov / 2), H / 2],
        [0, 0, 1],
    ], dtype=np.float64)

    world_up = np.asarray(row['world_up'])
    world_up = world_up / np.linalg.norm(world_up)
    return frame, R_wc_opencv, K, world_up
```

For mixed EK + HD pipelines, dispatch on `dataset` and use `load_hd_epic`
or `load_ek` per row.

---

## §5 Pre-flight checklist for any new HD-Epic pipeline

Before merging any code that loads HD-Epic data, verify:

- [ ] `dataset == 'hd_epic'` (or `hd_extended`) check exists somewhere
  in the load path.
- [ ] `ARIA_TO_OPENCV_POSE_REMAP` (or equivalent) is applied to the pose.
- [ ] Frame path uses `.../images/...` and `:06d` padding.
- [ ] **NO** rotation / transpose is applied to the frame itself.
- [ ] After-load assertion: `abs(np.dot(R_wc[:, 0], world_up)) < 0.3`.
- [ ] Loaded count matches expected count (catch silent path-mismatch).
- [ ] Quantitative reports cite "pinhole approximation, fisheye not
  corrected" footnote.
- [ ] If your work involves rendering (panoramas, cubemaps, marker
  overlays), eyeball ≥ 3 HD-Epic samples to confirm visible content
  ends up in the expected cubemap face / panorama region. EK alone
  is not a sufficient sanity check.

---

## §6 Where the affected production code lives

These scripts handle HD-Epic data and should be updated whenever this
doc changes:

| File | What it does | Status |
|---|---|---|
| `meta/testset/src/q2_pilot.py` | Renders Q2 cubemaps | ✓ remap applied (2026-04-27) |
| `meta/testset/src/q2_generate_full.py` | Batch Q2 generation, calls `q2_pilot` | ✓ inherits fix |
| `meta/testset/src/q2_eval.py` | Evaluates Q2 with VLMs | uses pre-rendered cubemaps; if they're wrong it reads wrong cubemaps |
| `meta/testset/src/eval_v2.py` | Q1 sighted eval; sends raw frame to VLM | not affected (no projection involved; frame is correct) |
| `meta/outpaint/...` | Panorama / cubemap generation for outpainting eval | unverified — audit needed before next outpaint run; also lacks the §2 fisheye undistortion |

When you fix or change anything pose-related for HD-Epic, update this
table.

---

## §7 Open issues / TODOs

- [x] ~~Decide whether to undistort HD-Epic frames before downstream use.~~
  Done 2026-04-27. See §2.
- [ ] Audit `meta/outpaint/` for the pose remap AND the fisheye
  undistortion. Panoramas generated before 2026-04-27 may have both
  bugs.
- [ ] Add a unit test that loads one HD-Epic and one EK sample, runs
  them through the canonical load function, and asserts the orientation
  invariants (cam X horizontal, cam Y mostly-down). Wire into CI so
  this doc's assumptions can't silently rot.
- [ ] Optional: cache per-participant calibrations (instead of P01-as-
  generic). Worth it only if absolute accuracy matters more than the
  ~5 % per-device variation.
