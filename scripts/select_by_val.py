#!/usr/bin/env python3
"""Pick a snapshot from an eval_mehldau_snapshots report: lowest target-val dCE among
snapshots whose generic dCE <= limit. Prints the chosen update (empty if none)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def choose(report: dict, generic_limit: float = 0.02):
    rows = [r for r in report["rows"] if r["update"] > 0 and r["d_ce_generic"] <= generic_limit]
    return min(rows, key=lambda r: r["d_ce_target_val"]) if rows else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("report", type=Path)
    ap.add_argument("--generic-limit", type=float, default=0.02)
    args = ap.parse_args(argv)
    row = choose(json.loads(args.report.read_text()), args.generic_limit)
    print(row["update"] if row else "")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
