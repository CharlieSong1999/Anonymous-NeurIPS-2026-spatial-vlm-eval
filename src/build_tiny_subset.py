"""Build a 300-row stratified subset of setA_extended that covers the
100-query human-eval pool, and emit a JSON membership mapping that
downstream eval scripts can use to filter on `--eval_set [full|300|100]`.

Stratification key: (dataset, yaw_bin_4, height_bin) — i.e. the 9-bin
joint task plus dataset, giving ~18 cells. The 100-active-pool ids are
included verbatim; the remaining 200 slots are filled by largest-
remainder allocation against the 2k joint frequency, then random
sampling within each cell.

Outputs:
  meta/testset/data/setA_tiny_300.parquet   — 300 rows, same schema as setA_extended
  meta/testset/data/subset_membership.json  — { "100": [...], "300": [...], "full_n": 2000 }
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("/path/to/this/repo/data")
SOURCE = DATA / "setA_extended.parquet"
POOL = DATA / "human_eval" / "pool_state.json"
OUT_PARQUET = DATA / "setA_tiny_300.parquet"
OUT_MAPPING = DATA / "subset_membership.json"

N_TARGET = 300
SEED = 2026


def stratum(row) -> tuple:
    return (row.dataset, int(row.yaw_bin_4), int(row.height_bin))


def main() -> None:
    full = pd.read_parquet(SOURCE)
    full["_stratum"] = full.apply(stratum, axis=1)
    print(f"loaded {len(full)} rows from {SOURCE.name}")

    active = json.loads(POOL.read_text())["active"]
    active = [sid for sid in active if sid in set(full["sample_id"])]
    assert len(active) == 100, f"expected 100 active in source, got {len(active)}"
    print(f"100-active pool fully present in setA_extended")

    # Target counts per stratum from the 2k distribution, scaled to 300
    cell_freq = full["_stratum"].value_counts(normalize=True).to_dict()
    raw_targets = {s: N_TARGET * f for s, f in cell_freq.items()}
    floor_targets = {s: int(np.floor(v)) for s, v in raw_targets.items()}
    deficit = N_TARGET - sum(floor_targets.values())
    # Largest-remainder method: distribute the leftover seats by fractional part
    remainders = sorted(
        ((s, raw_targets[s] - floor_targets[s]) for s in raw_targets),
        key=lambda x: -x[1],
    )
    targets = dict(floor_targets)
    for s, _ in remainders[:deficit]:
        targets[s] += 1
    assert sum(targets.values()) == N_TARGET
    print(f"strata: {len(targets)}, target sum: {sum(targets.values())}")

    # How many active fall in each stratum?
    active_df = full[full["sample_id"].isin(active)]
    active_per_cell = Counter(active_df["_stratum"])

    rng = np.random.default_rng(SEED)
    chosen: set[str] = set(active)

    # Per-cell: top up to target with a non-active sample
    overshoot_strata: list[tuple] = []
    for s, tgt in targets.items():
        active_in_cell = active_per_cell.get(s, 0)
        need = tgt - active_in_cell
        pool = full[(full["_stratum"] == s) & (~full["sample_id"].isin(active))]
        if need <= 0:
            overshoot_strata.append((s, -need))
            continue
        take = min(need, len(pool))
        if take < need:
            print(f"  stratum {s}: pool too small ({len(pool)} < {need})")
        picks = pool.sample(n=take, random_state=int(rng.integers(2**31)))[
            "sample_id"
        ].tolist()
        chosen.update(picks)

    # Reconcile overshoot: if some active overshot, total > N_TARGET; sample down
    if len(chosen) > N_TARGET:
        print(f"  overshoot before reconciliation: {len(chosen)} > {N_TARGET}")
        overshoot = len(chosen) - N_TARGET
        # Drop only from the cells where active overshot, preserving smaller cells' targets
        droppable = [
            sid
            for sid in chosen
            if sid not in active
            and active_per_cell.get(
                full.loc[full.sample_id == sid, "_stratum"].iloc[0], 0
            )
            > targets.get(full.loc[full.sample_id == sid, "_stratum"].iloc[0], 0)
        ]
        if len(droppable) >= overshoot:
            drop = rng.choice(droppable, size=overshoot, replace=False).tolist()
        else:
            extra = [sid for sid in chosen if sid not in active]
            drop = rng.choice(extra, size=overshoot, replace=False).tolist()
        chosen.difference_update(drop)

    # Backfill if short (rare, only when some cells had insufficient pool)
    if len(chosen) < N_TARGET:
        short = N_TARGET - len(chosen)
        remaining = full[~full["sample_id"].isin(chosen)]
        backfill = remaining.sample(n=short, random_state=int(rng.integers(2**31)))[
            "sample_id"
        ].tolist()
        chosen.update(backfill)

    assert len(chosen) == N_TARGET, f"final size {len(chosen)} != {N_TARGET}"
    assert set(active).issubset(chosen), "100-active not fully covered"

    tiny = full[full["sample_id"].isin(chosen)].drop(columns=["_stratum"]).reset_index(drop=True)
    tiny.to_parquet(OUT_PARQUET, index=False)
    print(f"\nwrote {len(tiny)} rows to {OUT_PARQUET}")

    # Distribution comparison
    print("\n=== distribution check (full 2k vs tiny 300) ===")
    for col in ["dataset", "yaw_bin_4", "height_bin"]:
        full_d = full[col].value_counts(normalize=True).sort_index().round(3).to_dict()
        tiny_d = tiny[col].value_counts(normalize=True).sort_index().round(3).to_dict()
        print(f"  {col}:  full={full_d}  tiny={tiny_d}")

    # Per-class coverage
    cls_cov = (tiny["canonical_label"].value_counts() > 0).sum()
    print(f"  canonical_label: {cls_cov} / {full['canonical_label'].nunique()} classes covered")

    # Membership mapping
    mapping = {
        "100": sorted(active),
        "300": sorted(tiny["sample_id"].tolist()),
        "full_n": int(len(full)),
        "source_parquet": "setA_extended.parquet",
        "300_parquet": "setA_tiny_300.parquet",
        "stratification_key": ["dataset", "yaw_bin_4", "height_bin"],
        "seed": SEED,
    }
    OUT_MAPPING.write_text(json.dumps(mapping, indent=2))
    print(f"\nwrote membership mapping to {OUT_MAPPING}")
    print(f"  100: {len(mapping['100'])} ids")
    print(f"  300: {len(mapping['300'])} ids  (covers 100/100 of active = "
          f"{len(set(mapping['100']) & set(mapping['300']))}/100)")


if __name__ == "__main__":
    main()
