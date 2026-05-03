"""
Q2 pilot — distractor strategy comparison for visual review.

Generates Q2 multiple-choice images for a small number of frames using
several distractor sampling strategies, in 2x3 cubemap layout. Saves
side-by-side comparison panels and the actual VLM-ready prompt text.

Distractor strategies tested:
  A. random_diff_face    — uniform random in any face != GT face, in the
                           BLACK (unseen) region only.
  B. random_far_3d       — uniform random direction on sphere, constrained
                           to >= 90° from GT and outside the visible FOV.
  C. no_target_cluster   — geometric far-from-GT, with the additional
                           semantic constraint that no cluster of the
                           target class lies within 30° angular window of
                           the chosen direction (uses step7_results.json
                           for EK; falls back to A for HD-Epic).

Output:
  meta/testset/exp/q2_pilot_001/
    samples/
      <sample_id>__<strategy>__cubemap.png   # full 2x3 cubemap with markers
      <sample_id>__<strategy>__prompt.txt    # the VLM prompt text
    comparison/
      <sample_id>__side_by_side.png          # all 3 strategies side-by-side
    pilot_log.md                             # what was generated, choices made

Run:
  conda run -n slam python3 -m src.q2_pilot --n-samples 5
"""
from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

# ── Paths ─────────────────────────────────────────────────────────────────────
TESTSET = Path("/path/to/this/repo")
DATA_DIR = TESTSET / "data"
OUT_DIR = TESTSET / "exp" / "q2_pilot_001"
SAMPLES_DIR = OUT_DIR / "samples"
COMPARISON_DIR = OUT_DIR / "comparison"
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
COMPARISON_DIR.mkdir(parents=True, exist_ok=True)

EK_ROOT = "/path/to/epic-kitchens"
HD_ROOT = ("/path/to/hd-epic/Participants")
# A representative HD-EPIC participant VRS used to bootstrap the Aria
# FISHEYE624 calibration. Per-device variation across HD-EPIC's small batch
# of Aria glasses is small (~5% in focal length) — using P01's calibration
# for all participants is an acceptable approximation.
HD_VRS_REFERENCE = ("/path/to/hd-epic/vrs/P01-20240202-110250_anonymized.vrs")

# ── Config ─────────────────────────────────────────────────────────────────────
FACE_SIZE = 320           # cubemap face resolution (smaller for faster pilot)
N_OPTIONS = 4             # A, B, C, D
LETTERS = ["A", "B", "C", "D"]
MARKER_RADIUS = 22
MIN_ANGULAR_GT_DIST_DEG = 90.0   # distractors must be ≥ this far from GT
NO_VISIBLE_MARGIN_PX = 8         # marker centre must be ≥ this far from any visible pixel
CLUSTER_EXCLUSION_DEG = 30.0     # for strategy C: no cluster of target class within this window

# 2×3 layout: row 0 = Up, Front, Down; row 1 = Left, Behind, Right
# (Stacks the verticals on top, the horizontals on bottom — keeps "front" centred top.)
LAYOUT_2x3 = [
    ["Up",   "Front", "Down"],
    ["Left", "Back",  "Right"],
]
# All faces, in canonical processing order
FACE_ORDER = ["Up", "Front", "Down", "Left", "Back", "Right"]

# Each face is defined by (centre, right_axis, down_axis) unit vectors in person frame.
# Person frame: X=right, Y=up, Z=forward.
FACE_DIRS: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {
    "Front": (np.array([0,  0,  1.0]), np.array([1.0,  0,  0]), np.array([0, -1.0, 0])),
    "Back":  (np.array([0,  0, -1.0]), np.array([-1.0, 0,  0]), np.array([0, -1.0, 0])),
    "Right": (np.array([1.0, 0, 0]),   np.array([0,  0, -1.0]), np.array([0, -1.0, 0])),
    "Left":  (np.array([-1.0, 0, 0]),  np.array([0,  0,  1.0]), np.array([0, -1.0, 0])),
    "Up":    (np.array([0,  1.0, 0]),  np.array([1.0, 0, 0]),   np.array([0,  0,  1.0])),
    "Down":  (np.array([0, -1.0, 0]),  np.array([1.0, 0, 0]),   np.array([0,  0, -1.0])),
}

MARKER_COLORS = {
    "A": (255, 80, 80),    # red
    "B": (80, 180, 255),   # blue
    "C": (80, 255, 80),    # green
    "D": (255, 200, 50),   # yellow
}
DARK_BG = 40  # same value used by query_gen renderer for unseen pixels


# ── Geometry ───────────────────────────────────────────────────────────────────
def compute_person_rotation(R_wc: np.ndarray, world_up: np.ndarray) -> np.ndarray:
    """Gravity-aligned person rotation. Person columns: [right, up, forward].

    Camera forward = R_wc[:,2] in world. Project to ground plane (perp to
    world_up); world_up points up. Same convention as panorama_projector.py.
    """
    cam_fwd = R_wc[:, 2]
    fwd_flat = cam_fwd - np.dot(cam_fwd, world_up) * world_up
    n = np.linalg.norm(fwd_flat)
    if n < 1e-6:
        cam_right = R_wc[:, 0]
        fwd_flat = cam_right - np.dot(cam_right, world_up) * world_up
        n = np.linalg.norm(fwd_flat)
        if n < 1e-6:
            fwd_flat = np.array([1.0, 0, 0])
            n = 1.0
    forward = fwd_flat / n
    right = np.cross(forward, world_up)
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)
    up = up / np.linalg.norm(up)
    return np.column_stack((right, up, forward))


def render_face(
    rgb: np.ndarray,
    R_wc: np.ndarray,
    R_person: np.ndarray,
    K: np.ndarray,
    face_name: str,
    face_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Render one face by ray-casting; return (face_image, visible_mask).

    visible_mask is (face_size, face_size) bool — True where the ray hit
    the source frame (so face pixel has real RGB content).
    """
    h, w = rgb.shape[:2]
    centre, right_dir, down_dir = FACE_DIRS[face_name]
    coords = (2.0 * (np.arange(face_size) + 0.5) / face_size) - 1.0
    uu, vv = np.meshgrid(coords, coords)
    rays_p = (centre[None, None, :]
              + uu[:, :, None] * right_dir[None, None, :]
              + vv[:, :, None] * down_dir[None, None, :])
    rays_p = rays_p / np.linalg.norm(rays_p, axis=2, keepdims=True)
    flat = rays_p.reshape(-1, 3)
    rays_w = (R_person @ flat.T).T
    rays_c = (R_wc.T @ rays_w.T).T
    z_c = rays_c[:, 2]
    valid = z_c > 0.01
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    px = np.full(len(z_c), -1.0)
    py = np.full(len(z_c), -1.0)
    px[valid] = fx * (rays_c[valid, 0] / z_c[valid]) + cx
    py[valid] = fy * (rays_c[valid, 1] / z_c[valid]) + cy
    pxi = np.round(px).astype(np.int32)
    pyi = np.round(py).astype(np.int32)
    in_bounds = valid & (pxi >= 0) & (pxi < w) & (pyi >= 0) & (pyi < h)
    face_flat = np.full((len(z_c), 3), DARK_BG, dtype=np.uint8)
    face_flat[in_bounds] = rgb[pyi[in_bounds], pxi[in_bounds]]
    return (face_flat.reshape(face_size, face_size, 3),
            in_bounds.reshape(face_size, face_size))


def direction_to_face_xy(d: np.ndarray, face_size: int) -> Tuple[str, int, int]:
    """Person-frame direction → (face_name, x, y) in that face's local image coords."""
    x, y, z = d
    ax, ay, az = abs(x), abs(y), abs(z)
    if az >= ax and az >= ay:
        if z > 0:
            face = "Front"; u = x / z;  v = y / z
        else:
            face = "Back";  u = -x / -z; v = y / -z
    elif ax >= ay and ax >= az:
        if x > 0:
            face = "Right"; u = -z / x; v = y / x
        else:
            face = "Left";  u = z / -x; v = y / -x
    else:
        if y > 0:
            face = "Up";   u = x / y;   v = -z / y
        else:
            face = "Down"; u = x / -y;  v = z / -y
    px = int((u + 1) * 0.5 * (face_size - 1))
    py = int((1 - (v + 1) * 0.5) * (face_size - 1))
    px = max(0, min(face_size - 1, px))
    py = max(0, min(face_size - 1, py))
    return face, px, py


def face_xy_to_direction(face: str, px: int, py: int, face_size: int) -> np.ndarray:
    """Inverse of above. Returns unit person-frame direction."""
    centre, right_dir, down_dir = FACE_DIRS[face]
    u = (2.0 * (px + 0.5) / face_size) - 1.0
    v = 1.0 - 2.0 * (py + 0.5) / face_size  # invert: small py = top = +v
    d = centre + u * right_dir - v * down_dir  # down_dir is the "v negative" axis
    return d / np.linalg.norm(d)


def angular_distance_deg(d1: np.ndarray, d2: np.ndarray) -> float:
    cos_a = float(np.clip(np.dot(d1, d2), -1.0, 1.0))
    return np.degrees(np.arccos(cos_a))


# ── Frame and pose loading ────────────────────────────────────────────────────
# ── HD-Epic Aria fisheye → rectified pinhole undistortion ───────────────
# Cached source (rotated to upright) + dest (linear pinhole) calibrations.
# Bootstrapped from P01's VRS calibration; reused for all HD-Epic participants
# (per-device variation in HD-EPIC is small).
_HD_CALIB_CACHE: dict = {}


def _get_hd_calibrations():
    """Lazily build (src_calib_rotated, dst_linear_calib, K_matrix) for HD-Epic
    undistortion. The output linear calibration matches the upright frame:
      W=H=1408, focal=612, principal point ~(703.5, 703.5)."""
    if _HD_CALIB_CACHE:
        return (_HD_CALIB_CACHE["src_rot"], _HD_CALIB_CACHE["dst"],
                _HD_CALIB_CACHE["K"])
    from projectaria_tools.core import data_provider
    from projectaria_tools.core.calibration import (
        get_linear_camera_calibration, rotate_camera_calib_cw90deg,
    )
    provider = data_provider.create_vrs_data_provider(HD_VRS_REFERENCE)
    src_calib = provider.get_device_calibration().get_camera_calib("camera-rgb")
    # Stored upright frames are 90° CW from the Aria native orientation; rotate
    # the calibration to match (so distort_by_calibration consumes upright pixels).
    src_rot = rotate_camera_calib_cw90deg(src_calib)
    W, H = 1408, 1408
    focal = float(src_calib.get_focal_lengths()[0])  # 612.20
    dst = get_linear_camera_calibration(W, H, focal, "rgb-linear")
    fx, fy = dst.get_focal_lengths()
    cx, cy = dst.get_principal_point()
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    _HD_CALIB_CACHE["src_rot"] = src_rot
    _HD_CALIB_CACHE["dst"] = dst
    _HD_CALIB_CACHE["K"] = K
    return src_rot, dst, K


def load_frame(dataset: str, video_id: str, frame_index: int,
               participant_id: str, undistort_hd: bool = True) -> Optional[Image.Image]:
    if dataset == "epic_kitchens":
        path = f"{EK_ROOT}/{participant_id}/{video_id}/frames/frame_{frame_index:010d}.jpg"
    else:
        # HD-Epic uses 'images/' subdir, not 'frames/'
        path = f"{HD_ROOT}/{participant_id}/{video_id}/images/frame_{frame_index:06d}.jpg"
    if not os.path.isfile(path):
        return None
    img = Image.open(path).convert("RGB")
    if dataset != "epic_kitchens" and undistort_hd:
        from projectaria_tools.core.calibration import distort_by_calibration
        src_rot, dst, _ = _get_hd_calibrations()
        arr = np.asarray(img)
        undist = distort_by_calibration(arr, dst, src_rot)
        if undist.dtype != np.uint8:
            undist = np.clip(undist, 0, 255).astype(np.uint8)
        img = Image.fromarray(undist)
    return img


# HD-Epic poses are in Aria-native convention (per track-e/context.md):
#   R_wc[:,0] = down in scene   (NOT right)
#   R_wc[:,1] = left in scene   (NOT down)
#   R_wc[:,2] = forward
# But the HD-Epic FRAMES at .../images/frame_*.jpg have already been
# pre-rotated to upright (OpenCV-like) orientation by the data_refinement
# pipeline. So the pose and frame orientations DO NOT MATCH and projection
# math (which assumes OpenCV camera frame: X=right, Y=down, Z=forward)
# produces incorrect cubemaps.
#
# Fix: remap pose from Aria→OpenCV. Frame stays as-is.
#   OpenCV X (right) = -Aria Y
#   OpenCV Y (down)  = +Aria X
#   OpenCV Z (fwd)   = +Aria Z
# So R_opencv = R_aria @ M where columns of M select the right Aria columns:
ARIA_TO_OPENCV_POSE_REMAP = np.array([
    [0,  1, 0],
    [-1, 0, 0],
    [0,  0, 1],
], dtype=np.float64)


def reconstruct_camera_pose(row) -> Tuple[np.ndarray, np.ndarray]:
    """Return (R_wc, (hfov_rad, vfov_rad)) from the test set row.

    For HD-Epic samples, remap the Aria-native pose to OpenCV convention.
    The frame itself is left untouched (it is already pre-rotated upright).
    """
    R_wc = np.asarray(row["camera_rotation_flat"]).reshape(3, 3)
    hfov_rad = np.radians(float(row["hfov"]))
    vfov_rad = np.radians(float(row["vfov"]))
    if row["dataset"] != "epic_kitchens":
        R_wc = R_wc @ ARIA_TO_OPENCV_POSE_REMAP
        # Override stored hfov/vfov for HD-Epic — after Aria undistortion, the
        # frame is a true linear pinhole at focal=612 / size=1408×1408 →
        # HFOV=VFOV ≈ 2*atan(704/612) ≈ 98°. The K used downstream is
        # reconstructed from these via make_intrinsics_for_frame().
        _, _, K = _get_hd_calibrations()
        # Reverse-derive HFOV from K (W/2)/fx
        W = 1408; fx = K[0, 0]; fy = K[1, 1]
        hfov_rad = 2 * np.arctan((W / 2.0) / fx)
        vfov_rad = 2 * np.arctan((W / 2.0) / fy)
    return R_wc, (hfov_rad, vfov_rad)


def make_intrinsics_for_frame(
    frame_size: Tuple[int, int],   # (W, H)
    hfov_rad: float, vfov_rad: float,
) -> np.ndarray:
    W, H = frame_size
    fx = (W / 2.0) / np.tan(hfov_rad / 2.0)
    fy = (H / 2.0) / np.tan(vfov_rad / 2.0)
    cx, cy = W / 2.0, H / 2.0
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


# ── Cluster loader (for strategy C) ────────────────────────────────────────────
def load_target_cluster_directions(
    row, R_wc: np.ndarray, R_person: np.ndarray,
) -> List[np.ndarray]:
    """For strategy C: return list of person-frame directions of OTHER clusters
    of the same target class in this scene (excluding the GT cluster itself)."""
    if row["dataset"] != "epic_kitchens":
        return []  # HD-Epic: no step7_results in same path; fall back to A
    path = f"{EK_ROOT}/{row['participant_id']}/{row['video_id']}/3d/step7_results.json"
    if not os.path.isfile(path):
        return []
    with open(path) as f:
        clusters = json.load(f)
    target = str(row["canonical_label"]).lower()
    cam_pos = np.asarray(row["camera_position"], dtype=np.float64)
    out = []
    for c in clusters:
        label = str(c.get("label", "")).lower()
        if label != target:
            continue
        cluster_pos = np.asarray(c["mean_position"], dtype=np.float64)
        # Skip the GT cluster: same world position as test row's target_world_xyz
        if np.linalg.norm(cluster_pos - np.asarray(row["target_world_xyz"])) < 0.1:
            continue
        d_world = cluster_pos - cam_pos
        n = np.linalg.norm(d_world)
        if n < 1e-6:
            continue
        d_world = d_world / n
        # World → person frame: d_person = R_person.T @ d_world
        d_person = R_person.T @ d_world
        d_person = d_person / np.linalg.norm(d_person)
        out.append(d_person)
    return out


# ── Distractor strategies ─────────────────────────────────────────────────────
def gt_direction_person_frame(row, R_person: np.ndarray) -> np.ndarray:
    cam_pos = np.asarray(row["camera_position"], dtype=np.float64)
    target_world = np.asarray(row["target_world_xyz"], dtype=np.float64)
    d_world = target_world - cam_pos
    d_world = d_world / np.linalg.norm(d_world)
    d_person = R_person.T @ d_world
    return d_person / np.linalg.norm(d_person)


def is_in_visible_region(
    d: np.ndarray, masks: Dict[str, np.ndarray], face_size: int,
    margin: int = NO_VISIBLE_MARGIN_PX,
) -> bool:
    face, px, py = direction_to_face_xy(d, face_size)
    mask = masks[face]
    # Window check: any visible pixel within margin?
    y0 = max(0, py - margin); y1 = min(face_size, py + margin + 1)
    x0 = max(0, px - margin); x1 = min(face_size, px + margin + 1)
    return bool(mask[y0:y1, x0:x1].any())


def sample_uniform_unit_vector(rng: np.random.RandomState) -> np.ndarray:
    v = rng.randn(3)
    return v / np.linalg.norm(v)


def sample_uniform_in_face(face: str, rng: np.random.RandomState) -> np.ndarray:
    """Uniform random direction within a single cubemap face (not uniform on sphere
    but sufficient for distractor sampling)."""
    centre, right_dir, down_dir = FACE_DIRS[face]
    u = rng.uniform(-0.85, 0.85)  # margin from edges to avoid corner distortion
    v = rng.uniform(-0.85, 0.85)
    d = centre + u * right_dir + v * down_dir
    return d / np.linalg.norm(d)


def strategy_A_random_diff_face(
    gt_dir: np.ndarray, masks: Dict[str, np.ndarray], face_size: int,
    rng: np.random.RandomState, n: int = 3, max_tries: int = 200,
) -> List[np.ndarray]:
    """Distractors: uniform-in-face on faces != GT face, must land in black region."""
    gt_face, _, _ = direction_to_face_xy(gt_dir, face_size)
    other_faces = [f for f in FACE_ORDER if f != gt_face]
    out = []
    rng.shuffle(other_faces)
    chosen_faces = []
    for face in other_faces:
        if len(out) >= n:
            break
        for _ in range(max_tries // len(other_faces) + 1):
            d = sample_uniform_in_face(face, rng)
            if is_in_visible_region(d, masks, face_size):
                continue
            # angular separation from existing options
            if any(angular_distance_deg(d, o) < 25.0 for o in out + [gt_dir]):
                continue
            out.append(d)
            chosen_faces.append(face)
            break
    return out[:n]


def strategy_B_random_far_3d(
    gt_dir: np.ndarray, masks: Dict[str, np.ndarray], face_size: int,
    rng: np.random.RandomState, n: int = 3, max_tries: int = 500,
) -> List[np.ndarray]:
    """Distractors: uniform random on sphere; ≥ 90° from GT, not in visible region."""
    out = []
    for _ in range(max_tries):
        if len(out) >= n:
            break
        d = sample_uniform_unit_vector(rng)
        if angular_distance_deg(d, gt_dir) < MIN_ANGULAR_GT_DIST_DEG:
            continue
        if is_in_visible_region(d, masks, face_size):
            continue
        if any(angular_distance_deg(d, o) < 25.0 for o in out):
            continue
        out.append(d)
    return out[:n]


def strategy_C_no_target_cluster(
    gt_dir: np.ndarray, target_cluster_dirs: List[np.ndarray],
    masks: Dict[str, np.ndarray], face_size: int,
    rng: np.random.RandomState, n: int = 3, max_tries: int = 500,
) -> List[np.ndarray]:
    """Distractors: uniform-in-face on faces != GT face, in black region, AND
    ≥ CLUSTER_EXCLUSION_DEG away from any *other* cluster of the target class.
    Falls back to strategy A when no target_cluster_dirs available."""
    if not target_cluster_dirs:
        return strategy_A_random_diff_face(gt_dir, masks, face_size, rng, n)
    gt_face, _, _ = direction_to_face_xy(gt_dir, face_size)
    other_faces = [f for f in FACE_ORDER if f != gt_face]
    out = []
    rng.shuffle(other_faces)
    for face in other_faces:
        if len(out) >= n:
            break
        for _ in range(max_tries // len(other_faces) + 1):
            d = sample_uniform_in_face(face, rng)
            if is_in_visible_region(d, masks, face_size):
                continue
            if any(angular_distance_deg(d, o) < 25.0 for o in out + [gt_dir]):
                continue
            # Reject if any same-class cluster is within exclusion window
            if any(angular_distance_deg(d, c) < CLUSTER_EXCLUSION_DEG
                   for c in target_cluster_dirs):
                continue
            out.append(d)
            break
    # If we couldn't fill, top up with strategy A
    if len(out) < n:
        extra = strategy_A_random_diff_face(gt_dir, masks, face_size, rng,
                                            n - len(out))
        out.extend(extra)
    return out[:n]


# ── Marker drawing & 2x3 layout ───────────────────────────────────────────────
def get_font(size: int):
    try:
        return ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except Exception:
        return ImageFont.load_default()


def draw_marker(draw, px: int, py: int, letter: str,
                font: ImageFont.FreeTypeFont, radius: int = MARKER_RADIUS):
    color = MARKER_COLORS[letter]
    draw.ellipse([px - radius, py - radius, px + radius, py + radius],
                 fill=color, outline=(255, 255, 255), width=3)
    bb = draw.textbbox((0, 0), letter, font=font)
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    draw.text((px - tw // 2, py - th // 2 - 2), letter,
              fill=(0, 0, 0), font=font)


def label_face_image(face_arr: np.ndarray, face_name: str) -> np.ndarray:
    """Stamp a small face label at the top-left."""
    img = Image.fromarray(face_arr.copy())
    draw = ImageDraw.Draw(img)
    font = get_font(20)
    txt = face_name.upper()
    # White text with black outline
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            draw.text((8 + dx, 6 + dy), txt, fill=(0, 0, 0), font=font)
    draw.text((8, 6), txt, fill=(255, 255, 255), font=font)
    return np.asarray(img)


def assemble_2x3(faces_marked: Dict[str, np.ndarray]) -> np.ndarray:
    """Stack into 2×3 grid per LAYOUT_2x3, with thin separators and labels."""
    rows = []
    sep_w = 4
    for row in LAYOUT_2x3:
        cells = []
        for fname in row:
            labelled = label_face_image(faces_marked[fname], fname)
            cells.append(labelled)
            cells.append(np.full((labelled.shape[0], sep_w, 3), 0, dtype=np.uint8))
        cells.pop()  # remove trailing sep
        rows.append(np.concatenate(cells, axis=1))
        rows.append(np.full((sep_w, rows[-1].shape[1], 3), 0, dtype=np.uint8))
    rows.pop()
    return np.concatenate(rows, axis=0)


# ── Prompt text ───────────────────────────────────────────────────────────────
# Mirrors the Q1 sighted prompt structure (eval_v2.py:build_sighted_prompt) so
# Q1 / Q2 differ only where the answer format inherently must.
LAYOUT_DESC = (
    "The image is a 2x3 cubemap showing every direction from the camera:\n"
    "  top row:    UP   | FRONT  | DOWN\n"
    "  bottom row: LEFT | BEHIND | RIGHT\n"
    "The dark-grey region in each tile is OUTSIDE the camera's original field "
    "of view (not seen by the camera). The visible RGB region shows what the "
    "camera actually saw."
)

OPTIONS_DESC = (
    "Four candidate locations are marked in the unseen (dark-grey) region "
    "with coloured letter markers: A, B, C, and D. Exactly one marker is at "
    "the true 3D location of the target object; the other three are distractors."
)


def build_prompt_text(target: str) -> str:
    return f"""\
You are given ONE egocentric RGB image (first-person view), projected as a 2x3 cubemap from the camera's viewpoint.
The target object "{target}" is OUT OF VIEW (not visible in any region of the cubemap).
Predict its most likely direction based on visible layout cues \
(walls, counters, appliances, doorways, free space) and spatial commonsense.

{LAYOUT_DESC}

{OPTIONS_DESC}

Output a single JSON object:
  {{"justification": "<1-2 sentences of spatial reasoning>",
    "answer": "<A|B|C|D>"}}

Example:
  {{"justification": "The sink is typically behind and to the left in this layout, and marker C is in that area.",
    "answer": "C"}}
"""


# ── Main per-sample pipeline ──────────────────────────────────────────────────
@dataclass
class CandidateRender:
    letter: str
    direction: np.ndarray
    face: str
    px: int
    py: int
    is_gt: bool


def run_one_sample(
    row, strategy_name: str, rng: np.random.RandomState,
) -> Optional[Tuple[np.ndarray, str, dict]]:
    """Returns (cubemap_image_2x3, prompt_text, meta) or None on failure."""
    frame = load_frame(row["dataset"], row["video_id"], int(row["frame_index"]),
                       row["participant_id"])
    if frame is None:
        return None

    R_wc, (hfov, vfov) = reconstruct_camera_pose(row)
    world_up = np.asarray(row["world_up"], dtype=np.float64)
    world_up = world_up / np.linalg.norm(world_up)
    R_person = compute_person_rotation(R_wc, world_up)
    K = make_intrinsics_for_frame(frame.size, hfov, vfov)

    # Render 6 faces and visible masks
    rgb = np.asarray(frame)
    faces, masks = {}, {}
    for fname in FACE_ORDER:
        face_img, mask = render_face(rgb, R_wc, R_person, K, fname, FACE_SIZE)
        faces[fname] = face_img
        masks[fname] = mask

    # GT direction (in person frame) and check it lands in the unseen region
    gt_dir = gt_direction_person_frame(row, R_person)
    if is_in_visible_region(gt_dir, masks, FACE_SIZE, margin=2):
        # Target lies inside FOV — should not happen for our "unseen" test set
        return None

    # Distractors per strategy
    if strategy_name == "A_random_diff_face":
        distractors = strategy_A_random_diff_face(gt_dir, masks, FACE_SIZE, rng)
    elif strategy_name == "B_random_far_3d":
        distractors = strategy_B_random_far_3d(gt_dir, masks, FACE_SIZE, rng)
    elif strategy_name == "C_no_target_cluster":
        cluster_dirs = load_target_cluster_directions(row, R_wc, R_person)
        distractors = strategy_C_no_target_cluster(
            gt_dir, cluster_dirs, masks, FACE_SIZE, rng)
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")

    if len(distractors) < N_OPTIONS - 1:
        return None  # could not generate enough distractors

    # Build candidates: GT + distractors, shuffle letters
    options = [(gt_dir, True)] + [(d, False) for d in distractors]
    rng.shuffle(options)
    candidates: List[CandidateRender] = []
    for letter, (d, is_gt) in zip(LETTERS, options):
        face, px, py = direction_to_face_xy(d, FACE_SIZE)
        candidates.append(CandidateRender(letter=letter, direction=d,
                                          face=face, px=px, py=py, is_gt=is_gt))

    # Draw markers on faces
    faces_pil = {fname: Image.fromarray(faces[fname].copy()) for fname in FACE_ORDER}
    font = get_font(22)
    for cand in candidates:
        d = ImageDraw.Draw(faces_pil[cand.face])
        draw_marker(d, cand.px, cand.py, cand.letter, font)

    faces_marked = {fname: np.asarray(img) for fname, img in faces_pil.items()}
    cubemap_2x3 = assemble_2x3(faces_marked)

    prompt = build_prompt_text(row["canonical_label"])
    gt_letter = next(c.letter for c in candidates if c.is_gt)

    meta = {
        "sample_id": row["sample_id"],
        "strategy": strategy_name,
        "target": row["canonical_label"],
        "dataset": row["dataset"],
        "video_id": row["video_id"],
        "frame_index": int(row["frame_index"]),
        "gt_letter": gt_letter,
        "gt_face": next(c.face for c in candidates if c.is_gt),
        "candidates": [
            {"letter": c.letter, "face": c.face, "px": c.px, "py": c.py,
             "is_gt": c.is_gt} for c in candidates
        ],
    }
    return cubemap_2x3, prompt, meta


def make_side_by_side(images: Dict[str, np.ndarray], prompt: str,
                      sample_id: str, gt_letters: Dict[str, str]) -> np.ndarray:
    """Stack 3 strategy panels vertically with strategy headers per row.

    Layout (for K strategies):
       [HEADER]
       [strategy A header]
       [cubemap A]
       [strategy B header]
       [cubemap B]
       ...
       [PROMPT footer]
    """
    pad = 14
    head_h = 36
    panels = []
    title_font = get_font(22)
    sub_font = get_font(15)
    for strat, img in images.items():
        H, W = img.shape[:2]
        # Strategy header bar
        bar = np.full((head_h, W, 3), 245, dtype=np.uint8)
        pil = Image.fromarray(bar)
        d = ImageDraw.Draw(pil)
        gt = gt_letters.get(strat, "?")
        d.text((10, 6), f"{strat}   (GT = {gt})", fill=(0, 0, 0), font=sub_font)
        bar = np.asarray(pil)
        panels.append(bar)
        panels.append(img)
        panels.append(np.full((pad, W, 3), 250, dtype=np.uint8))
    panels.pop()
    col_img = np.concatenate(panels, axis=0)
    W = col_img.shape[1]

    # Top header
    top_h = 56
    top = np.full((top_h, W, 3), 230, dtype=np.uint8)
    pil = Image.fromarray(top)
    d = ImageDraw.Draw(pil)
    d.text((10, 8), sample_id, fill=(0, 0, 0), font=title_font)
    d.text((10, 36), "Same frame, three distractor strategies stacked vertically.",
           fill=(60, 60, 60), font=sub_font)
    top = np.asarray(pil)

    # Bottom: prompt
    fp = get_font(14)
    line_w_px = W - 20
    pil_dummy = Image.new("RGB", (10, 10))
    draw_dummy = ImageDraw.Draw(pil_dummy)
    words = prompt.replace("\n", " \n ").split()
    lines, cur = [], ""
    for word in words:
        if word == "\n":
            lines.append(cur); cur = ""; continue
        trial = (cur + " " + word).strip()
        bb = draw_dummy.textbbox((0, 0), trial, font=fp)
        if bb[2] - bb[0] > line_w_px:
            lines.append(cur); cur = word
        else:
            cur = trial
    if cur: lines.append(cur)
    line_h = 18
    bot_h = 24 + line_h * len(lines)
    bot = np.full((bot_h, W, 3), 240, dtype=np.uint8)
    pil = Image.fromarray(bot)
    d = ImageDraw.Draw(pil)
    d.text((10, 4), "Prompt sent to VLM:", fill=(0, 0, 0), font=sub_font)
    for i, line in enumerate(lines):
        d.text((10, 22 + i * line_h), line, fill=(30, 30, 30), font=fp)
    bot = np.asarray(pil)

    return np.concatenate([top, col_img, bot], axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ek-only", action="store_true",
                        help="Only sample EK frames (so strategy C has cluster data).")
    args = parser.parse_args()

    df = pd.read_parquet(DATA_DIR / "setA_extended.parquet")
    if args.ek_only:
        df = df[df.dataset == "epic_kitchens"]
    df = df.reset_index(drop=True)

    rng = np.random.RandomState(args.seed)
    indices = rng.choice(len(df), size=args.n_samples, replace=False)
    samples = df.iloc[indices]

    log_lines = ["# Q2 Pilot — distractor strategy comparison\n"]
    log_lines.append(f"Date: 2026-04-27 — generated by `src/q2_pilot.py`\n")
    log_lines.append(f"\n## Samples\n")

    strategies = ["A_random_diff_face", "B_random_far_3d", "C_no_target_cluster"]

    for _, row in samples.iterrows():
        sid = row["sample_id"]
        log_lines.append(f"\n### {sid}")
        log_lines.append(f"- target: **{row['canonical_label']}**, "
                         f"dataset: {row['dataset']}, frame: {row['frame_index']}")
        log_lines.append(f"- yaw_deg: {float(row['yaw_deg']):.1f}, "
                         f"pitch_deg: {float(row['pitch_deg']):.1f}, "
                         f"hfov: {row['hfov']}, vfov: {row['vfov']}")

        rendered: Dict[str, np.ndarray] = {}
        prompt_text = ""
        for strat in strategies:
            sub_rng = np.random.RandomState(args.seed + hash(sid + strat) % 10**6)
            res = run_one_sample(row, strat, sub_rng)
            if res is None:
                log_lines.append(f"  - **{strat}**: FAILED (target visible in FOV "
                                 f"or insufficient distractors)")
                continue
            img, prompt, meta = res
            prompt_text = prompt
            # save individual files
            stub = f"{sid}__{strat}"
            Image.fromarray(img).save(SAMPLES_DIR / f"{stub}__cubemap.png")
            (SAMPLES_DIR / f"{stub}__prompt.txt").write_text(prompt)
            (SAMPLES_DIR / f"{stub}__meta.json").write_text(
                json.dumps(meta, default=str, indent=2))
            log_lines.append(f"  - **{strat}**: GT={meta['gt_letter']} on {meta['gt_face']}, "
                             f"distractor faces=" +
                             ", ".join(c["face"] for c in meta["candidates"]
                                       if not c["is_gt"]))
            rendered[strat] = img
        if len(rendered) >= 2 and prompt_text:
            gt_letters = {}
            for strat in rendered:
                meta_path = SAMPLES_DIR / f"{sid}__{strat}__meta.json"
                if meta_path.exists():
                    gt_letters[strat] = json.loads(meta_path.read_text())["gt_letter"]
            sbs = make_side_by_side(rendered, prompt_text, sid, gt_letters)
            Image.fromarray(sbs).save(COMPARISON_DIR / f"{sid}__side_by_side.png")
            log_lines.append(f"  - side-by-side: `comparison/{sid}__side_by_side.png`")

    (OUT_DIR / "pilot_log.md").write_text("\n".join(log_lines))
    print(f"\nDone. Output in {OUT_DIR}")
    print(f"  Per-sample images: {SAMPLES_DIR}")
    print(f"  Side-by-side comparisons: {COMPARISON_DIR}")
    print(f"  Log: {OUT_DIR / 'pilot_log.md'}")


if __name__ == "__main__":
    main()
