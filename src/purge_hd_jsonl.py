"""
Purge HD-Epic records from existing q2_eval JSONL files. After this,
restart `q2_eval` and the resume-from-existing-jsonl logic will only
re-run the HD queries (EK records remain valid).

Run:
  conda run -n slam python3 -m src.purge_hd_jsonl
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

RUNS_DIR = Path("/path/to/this/repo/exp/q2_eval_001/runs")


def is_hd_sample_id(sid: str) -> bool:
    return sid.startswith("hd_epic_") or sid.startswith("hd_extended_")


def main():
    files = sorted(RUNS_DIR.glob("*.jsonl"))
    print(f"Found {len(files)} JSONL files in {RUNS_DIR}")

    total_removed = 0
    total_kept = 0
    for fp in files:
        kept = []
        n_removed = 0
        with open(fp) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if is_hd_sample_id(r.get("sample_id", "")):
                    n_removed += 1
                else:
                    kept.append(line)
        with open(fp, "w") as f:
            for line in kept:
                f.write(line + "\n")
        total_removed += n_removed
        total_kept += len(kept)
        print(f"  {fp.name}: removed {n_removed} HD, kept {len(kept)} EK")

    print(f"\nTotal: removed {total_removed} HD records, "
          f"kept {total_kept} EK records across {len(files)} files.")


if __name__ == "__main__":
    main()
