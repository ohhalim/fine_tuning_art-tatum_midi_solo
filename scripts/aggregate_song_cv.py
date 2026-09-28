#!/usr/bin/env python3
"""C1: pick the update budget from song-level CV (docs/experiments/MEHLDAU_CLEAN_BASE.md).

Among updates whose fold-mean dCE_generic <= +0.02, pick the lowest fold-mean
specialisation on the held-out songs. Also evaluates completion criterion 1.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

GENERIC_LIMIT = 0.02
SPEC_TARGET = -0.02


def aggregate(fold_reports: list[dict]) -> dict:
    updates = sorted({r["update"] for rep in fold_reports for r in rep["rows"] if r["update"] > 0})
    table = []
    for u in updates:
        rows = [next(r for r in rep["rows"] if r["update"] == u) for rep in fold_reports]
        specs = [r["specialisation_val"] for r in rows]
        table.append({"update": u,
                      "mean_specialisation": statistics.mean(specs),
                      "mean_d_ce_heldout": statistics.mean(r["d_ce_target_val"] for r in rows),
                      "mean_d_ce_generic": statistics.mean(r["d_ce_generic"] for r in rows),
                      "folds_negative": sum(s < 0 for s in specs), "per_fold": specs})
    eligible = [t for t in table if t["mean_d_ce_generic"] <= GENERIC_LIMIT]
    chosen = min(eligible, key=lambda t: t["mean_specialisation"]) if eligible else None
    criterion = bool(chosen and chosen["mean_specialisation"] <= SPEC_TARGET
                     and chosen["folds_negative"] >= len(fold_reports) - 1)
    return {"table": table, "chosen": chosen, "criterion1_met": criterion}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fold-report", type=Path, action="append", required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args(argv)
    out = {"schema": "song_cv_v1", "generic_limit": GENERIC_LIMIT, "spec_target": SPEC_TARGET,
           "fold_reports": [str(p) for p in args.fold_report],
           **aggregate([json.loads(p.read_text()) for p in args.fold_report])}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    for t in out["table"]:
        print(f"u{t['update']:>4} spec {t['mean_specialisation']:+.4f} heldout {t['mean_d_ce_heldout']:+.4f} "
              f"generic {t['mean_d_ce_generic']:+.4f} neg {t['folds_negative']}/{len(args.fold_report)}")
    print("chosen", out["chosen"] and out["chosen"]["update"], "criterion1", out["criterion1_met"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
