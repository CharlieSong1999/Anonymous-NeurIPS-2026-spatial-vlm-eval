"""
Generate Q2 cubemap MCQ assets for the full setA_extended (2000 queries)
under both distractor strategies A (random_diff_face) and C
(no_target_cluster, falls back to A on HD-Epic).

For each (query, strategy):
  data/setA_extended_q2/<strategy>/images/<sample_id>.jpg   — 2x3 cubemap PNG-as-JPG
  data/setA_extended_q2/<strategy>/queries.parquet          — one row per sample

Re-uses run_one_sample() from q2_pilot.py for the actual rendering.

Run:
  conda run -n slam python3 -m src.q2_generate_full
  conda run -n slam python3 -m src.q2_generate_full --strategies A
  conda run -n slam python3 -m src.q2_generate_full --max 100  # subset
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from q2_pilot import (
    DATA_DIR, run_one_sample, build_prompt_text,
)

OUT_BASE = DATA_DIR / "setA_extended_q2"
STRATEGIES_AVAILABLE = ["A_random_diff_face", "C_no_target_cluster"]


def deterministic_seed(sid: str, strategy: str, base_seed: int) -> int:
    """Reproducible seed (replaces hash()). hash() is randomized per Python
    process via PYTHONHASHSEED — using md5 makes the seed stable across runs."""
    h = hashlib.md5(f"{sid}|{strategy}".encode("utf-8")).hexdigest()
    return base_seed + int(h[:8], 16)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategies", nargs="+",
                        default=STRATEGIES_AVAILABLE,
                        choices=STRATEGIES_AVAILABLE)
    parser.add_argument("--max", type=int, default=None,
                        help="Limit number of queries for testing.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quality", type=int, default=92,
                        help="JPEG quality.")
    parser.add_argument("--datasets", nargs="+", default=None,
                        choices=["epic_kitchens", "hd_epic", "hd_extended"],
                        help="If set, only regenerate queries from these "
                             "datasets and merge with existing parquet rows "
                             "for other datasets.")
    args = parser.parse_args()

    df = pd.read_parquet(DATA_DIR / "setA_extended.parquet")
    if args.datasets:
        df = df[df.dataset.isin(args.datasets)].reset_index(drop=True)
        print(f"Filtered to datasets {args.datasets}: {len(df)} queries")
    if args.max:
        df = df.head(args.max).reset_index(drop=True)
    print(f"Working set: {len(df)} queries; strategies={args.strategies}")

    overall_start = time.time()

    for strategy in args.strategies:
        out_dir = OUT_BASE / strategy
        img_dir = out_dir / "images"
        img_dir.mkdir(parents=True, exist_ok=True)

        rows = []
        n_done, n_skipped = 0, 0
        t0 = time.time()
        last_progress = t0

        for i, (_, row) in enumerate(df.iterrows()):
            sid = row["sample_id"]
            # Deterministic per-(sid, strategy) RNG (hashlib, not Python's hash())
            seed = deterministic_seed(sid, strategy, args.seed)
            sub_rng = np.random.RandomState(seed)
            res = run_one_sample(row, strategy, sub_rng)
            if res is None:
                n_skipped += 1
                continue
            img, prompt, meta = res
            jpg_path = img_dir / f"{sid}.jpg"
            Image.fromarray(img).save(jpg_path, format="JPEG",
                                       quality=args.quality)
            row_meta = {
                "sample_id": sid,
                "q2_strategy": strategy,
                "image_path": str(jpg_path.relative_to(out_dir)),
                "prompt": prompt,
                "gt_letter": meta["gt_letter"],
                "gt_face": meta["gt_face"],
                "candidates_json": json.dumps(meta["candidates"], default=str),
                # Echo Q1 fields for joining
                "q1_sample_id": sid,
                "dataset": row["dataset"],
                "video_id": row["video_id"],
                "participant_id": row["participant_id"],
                "frame_index": int(row["frame_index"]),
                "canonical_label": row["canonical_label"],
                "yaw_bin_4": int(row["yaw_bin_4"]),
                "height_bin": int(row["height_bin"]),
                "yaw_deg": float(row["yaw_deg"]),
                "pitch_deg": float(row["pitch_deg"]),
                "rng_seed": int(seed),
            }
            rows.append(row_meta)
            n_done += 1

            if time.time() - last_progress > 30 or i + 1 == len(df):
                rate = (i + 1) / (time.time() - t0)
                eta = (len(df) - (i + 1)) / max(rate, 1e-6)
                print(f"  [{strategy}] {i+1}/{len(df)} done={n_done} skip={n_skipped} "
                      f"rate={rate:.2f}/s eta={eta/60:.1f} min", flush=True)
                last_progress = time.time()

        df_out = pd.DataFrame(rows)
        # If --datasets filter was used and a previous queries.parquet exists,
        # merge: keep rows from datasets we DIDN'T regenerate.
        parquet_path = out_dir / "queries.parquet"
        if args.datasets and parquet_path.exists():
            old = pd.read_parquet(parquet_path)
            keep = old[~old.dataset.isin(args.datasets)]
            print(f"[{strategy}] Merging {len(keep)} preserved rows from "
                  f"datasets NOT in {args.datasets}")
            df_out = pd.concat([keep, df_out], ignore_index=True)
        df_out.to_parquet(parquet_path, index=False)
        elapsed = time.time() - t0
        print(f"\n[{strategy}] DONE in {elapsed/60:.1f} min — "
              f"{len(rows)} queries (re)generated, {n_skipped} skipped "
              f"(target visible). Total parquet rows: {len(df_out)}")
        print(f"  Parquet: {parquet_path}")
        print(f"  Images:  {img_dir}")

    print(f"\nAll strategies done. Total wall: "
          f"{(time.time() - overall_start)/60:.1f} min")


if __name__ == "__main__":
    main()
