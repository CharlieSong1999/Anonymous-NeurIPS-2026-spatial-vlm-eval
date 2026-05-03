"""Tiny shared helper for the `--eval_set [full|300|100]` flag.

Reads the membership mapping at `meta/testset/data/subset_membership.json`
and exposes a `filter_by_eval_set` that works for both Q1 parquets
(filter on `sample_id`) and Q2 parquets (filter on `q1_sample_id`,
since Q2 queries inherit Q1's sample_id via the foreign key).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

EVAL_SET_CHOICES = ("full", "300", "100")

_MAPPING_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "subset_membership.json"
)


def load_mapping() -> dict:
    return json.loads(_MAPPING_PATH.read_text())


def filter_by_eval_set(
    df: pd.DataFrame,
    eval_set: str,
    *,
    id_column: str = "sample_id",
) -> pd.DataFrame:
    """Filter a queries dataframe to the chosen eval-set membership.

    `id_column` is `sample_id` for Q1 parquets and `q1_sample_id` for Q2
    parquets (whose own `sample_id` is a Q2-specific id).
    """
    if eval_set == "full":
        return df.reset_index(drop=True)
    if eval_set not in ("100", "300"):
        raise ValueError(f"unknown eval_set={eval_set!r}; expected one of {EVAL_SET_CHOICES}")

    mapping = load_mapping()
    keep = set(mapping[eval_set])
    out = df[df[id_column].isin(keep)].reset_index(drop=True)

    missing = len(keep) - len(out)
    if missing > 0:
        print(
            f"[eval_set={eval_set}] note: {missing}/{len(keep)} ids in the "
            f"subset are not present in this dataframe (column={id_column}); "
            f"kept {len(out)} rows."
        )
    return out


def add_eval_set_arg(parser, default: str = "full") -> None:
    """Convenience: add a uniform --eval-set / --eval_set flag to a parser."""
    parser.add_argument(
        "--eval-set",
        "--eval_set",
        dest="eval_set",
        choices=EVAL_SET_CHOICES,
        default=default,
        help=(
            "Which subset of setA to evaluate on. "
            "'full' = all rows in the input parquet; "
            "'300' = stratified 300-row tiny subset; "
            "'100' = the 100-query human-eval pool. "
            "Membership is defined by data/subset_membership.json."
        ),
    )
