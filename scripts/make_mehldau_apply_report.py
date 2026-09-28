#!/usr/bin/env python3
"""M-A1: pick each run's snapshot by the pre-registered rule and write the comparison report.

Rule (docs/experiments/SEED_REPEAT_AND_MEHLDAU_APPLY.md): among snapshots with
dCE_generic <= +0.02, the one with the lowest dCE on the Mehldau val songs.
No listening: musical_quality_verified / style_verified stay false.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

GENERIC_LIMIT = 0.02
IMPROVEMENT = 0.01


def select(report: dict) -> dict:
    rows = [r for r in report["rows"] if r["update"] > 0 and r["d_ce_generic"] <= GENERIC_LIMIT]
    if not rows:
        raise ValueError("no snapshot satisfies the generic-loss limit")
    return min(rows, key=lambda r: r["d_ce_target_val"])


def ce_table(report: dict) -> list[dict]:
    return [{k: r[k] for k in ("update", "d_ce_target_train", "d_ce_target_val", "d_ce_generic",
                               "specialisation_val")}
            for r in sorted(report["rows"], key=lambda r: r["update"])]


def gen_summary(report: dict, update: int) -> dict:
    row = next(r for r in report["rows"] if r["update"] == update)
    per = row["per_seed"]
    return {"update": update, "sequences": len(per),
            "grammar_valid": sum(p["grammar_valid"] for p in per),
            "notes_mean": sum(p["notes"] for p in per) / len(per),
            "copy8_mean": sum((p["copy8"] or 0) for p in per) / len(per),
            "copy16_max": max((p["copy16"] or 0) for p in per)}


def runtime_summary(report: dict) -> dict:
    prod = report["production"]
    return {"completed_bars": report["completed_bars"], "model_bars": prod["model_bar_count"],
            "fallback": prod["fallback_bar_count"], "errors": prod["error_count"],
            "deadline_misses": report["scheduler_dispatch_deadline_miss_count"],
            "gen_ms_p50": prod["generation_ms"].get("p50"),
            "played_notes": report.get("played_note_count")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ce-outproj", type=Path, required=True)
    ap.add_argument("--ce-qkv", type=Path, required=True)
    ap.add_argument("--gen-outproj", type=Path, required=True)
    ap.add_argument("--gen-qkv", type=Path, required=True)
    ap.add_argument("--runtime", type=Path, action="append", required=True, metavar="NAME=REPORT")
    ap.add_argument("--density", type=Path, default=None)
    ap.add_argument("--midi-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args(argv)

    ce = {"out_proj": json.loads(args.ce_outproj.read_text()),
          "out_proj+qkv": json.loads(args.ce_qkv.read_text())}
    picked = {name: select(rep) for name, rep in ce.items()}
    diff = picked["out_proj+qkv"]["specialisation_val"] - picked["out_proj"]["specialisation_val"]
    gens = {"out_proj": json.loads(args.gen_outproj.read_text()),
            "out_proj+qkv": json.loads(args.gen_qkv.read_text())}
    runtime = {}
    for spec in args.runtime:
        name, path = str(spec).split("=", 1)
        runtime[name] = runtime_summary(json.loads(Path(path).read_text()))
    out = {
        "schema": "mehldau_apply_v1",
        "musical_quality_verified": False, "style_verified": False,
        "listening_done": False,
        "rule": {"generic_limit": GENERIC_LIMIT, "select": "lowest dCE on Mehldau val",
                 "improvement_threshold": IMPROVEMENT},
        "ce_tables": {name: ce_table(rep) for name, rep in ce.items()},
        "selected": {name: {"update": r["update"], "d_ce_train": r["d_ce_target_train"],
                            "d_ce_val": r["d_ce_target_val"], "d_ce_generic": r["d_ce_generic"],
                            "specialisation_val": r["specialisation_val"]}
                     for name, r in picked.items()},
        "specialisation_diff_qkv_minus_outproj": diff,
        "qkv_improves_by_rule": diff <= -IMPROVEMENT,
        "generation": {
            "base_update0": gen_summary(gens["out_proj"], 0),
            "out_proj": gen_summary(gens["out_proj"], picked["out_proj"]["update"]),
            "out_proj+qkv": gen_summary(gens["out_proj+qkv"], picked["out_proj+qkv"]["update"]),
        },
        "runtime_128bpm": runtime,
        "run_density": json.loads(args.density.read_text()) if args.density else None,
        "comparison_midi": sorted(str(p) for p in args.midi_dir.glob("*.mid")),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: out[k] for k in ("selected", "specialisation_diff_qkv_minus_outproj",
                                          "qkv_improves_by_rule", "generation", "runtime_128bpm")},
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
