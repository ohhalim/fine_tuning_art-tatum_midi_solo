"""Decide the next solo-yield repair target after residual tension review."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from scripts.assess_stage_b_generic_base_readiness import write_json, write_text  # noqa: E402
from scripts.build_music_transformer_solo_yield_dead_air_density_repair_package import (  # noqa: E402
    _bool_token,
    _dict,
    _float,
    _int,
    _list,
    compact_probe_sample,
    read_json,
)


SCHEMA_VERSION = "music_transformer_solo_yield_residual_tension_target_decision_v1"


class SoloYieldResidualTensionTargetDecisionError(ValueError):
    pass


def _reject_quality_claim(report: dict[str, Any]) -> None:
    readiness = _dict(report.get("readiness"))
    claimed = [
        key
        for key in ("musical_quality_claimed", "artist_style_claimed", "production_ready_claimed")
        if bool(readiness.get(key, False))
    ]
    if claimed:
        raise SoloYieldResidualTensionTargetDecisionError(
            f"unexpected quality claim: {claimed}"
        )


def package_candidates(report: dict[str, Any]) -> list[Any]:
    candidates = _list(report.get("candidates"))
    if candidates:
        return candidates
    return _list(report.get("selected_candidates"))


def residual_label_counts(
    package: dict[str, Any],
    *,
    dead_air_watch_max: float,
    min_syncopation_ratio: float,
    min_tension_ratio: float,
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for item in package_candidates(package):
        candidate = _dict(item)
        if _float(candidate.get("tension_ratio")) < float(min_tension_ratio):
            counts["low_tension_color"] += 1
        if _float(candidate.get("syncopated_onset_ratio")) < float(min_syncopation_ratio):
            counts["low_syncopation"] += 1
        if _float(candidate.get("dead_air_ratio")) > float(dead_air_watch_max):
            counts["dead_air_watch"] += 1
    return dict(sorted(counts.items()))


def case_feasibility_rows(
    sweep_report: dict[str, Any],
    *,
    selected_per_case: int,
    max_dead_air_ratio: float,
    min_direction_change_ratio: float,
    min_tension_ratio: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in _list(sweep_report.get("cases")):
        case_dict = _dict(case)
        probe_report = read_json(Path(str(case_dict.get("probe_report_path") or "")))
        guarded: list[dict[str, Any]] = []
        tension_safe: list[dict[str, Any]] = []
        for sample in _list(probe_report.get("samples")):
            sample_dict = _dict(sample)
            if not bool(sample_dict.get("strict_valid", False)):
                continue
            compact = compact_probe_sample(case_dict, sample_dict)
            if not Path(str(compact["source_midi_path"])).exists():
                continue
            if _float(compact.get("dead_air_ratio")) > float(max_dead_air_ratio):
                continue
            if _float(compact.get("direction_change_ratio")) < float(min_direction_change_ratio):
                continue
            guarded.append(compact)
            if _float(compact.get("tension_ratio")) >= float(min_tension_ratio):
                tension_safe.append(compact)
        rows.append(
            {
                "case_label": str(case_dict.get("label") or ""),
                "selected_per_case": int(selected_per_case),
                "guarded_candidate_count": len(guarded),
                "tension_safe_candidate_count": len(tension_safe),
                "tension_safe_feasible": len(tension_safe) >= int(selected_per_case),
                "guard": {
                    "max_dead_air_ratio": float(max_dead_air_ratio),
                    "min_direction_change_ratio": float(min_direction_change_ratio),
                    "min_tension_ratio": float(min_tension_ratio),
                },
            }
        )
    return rows


def select_next_target(label_counts: dict[str, int], feasibility_rows: list[dict[str, Any]]) -> str:
    low_tension_count = int(label_counts.get("low_tension_color", 0))
    tension_feasible = all(bool(row.get("tension_safe_feasible")) for row in feasibility_rows)
    if low_tension_count and tension_feasible:
        return "tension_color_balance_repair"
    if low_tension_count and int(label_counts.get("low_syncopation", 0)):
        return "rhythm_syncopation_balance_repair"
    if low_tension_count:
        return "guard_or_candidate_count_decision"
    if int(label_counts.get("low_syncopation", 0)):
        return "rhythm_syncopation_balance_repair"
    return "listening_review_or_broader_repeatability"


def next_boundary_for_target(target: str) -> str:
    return {
        "tension_color_balance_repair": "music_transformer_solo_yield_tension_color_balance_repair",
        "rhythm_syncopation_balance_repair": "music_transformer_solo_yield_rhythm_syncopation_balance_repair",
        "guard_or_candidate_count_decision": "music_transformer_solo_yield_guard_or_candidate_count_decision",
    }.get(target, "music_transformer_solo_yield_listening_review_or_broader_repeatability")


def build_decision_report(
    *,
    sweep_report: dict[str, Any],
    source_package: dict[str, Any],
    output_dir: Path,
    selected_per_case: int,
    max_dead_air_ratio: float,
    min_direction_change_ratio: float,
    min_tension_ratio: float,
    min_syncopation_ratio: float,
    dead_air_watch_max: float,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    _reject_quality_claim(source_package)
    label_counts = residual_label_counts(
        source_package,
        dead_air_watch_max=float(dead_air_watch_max),
        min_syncopation_ratio=float(min_syncopation_ratio),
        min_tension_ratio=float(min_tension_ratio),
    )
    feasibility_rows = case_feasibility_rows(
        sweep_report,
        selected_per_case=int(selected_per_case),
        max_dead_air_ratio=float(max_dead_air_ratio),
        min_direction_change_ratio=float(min_direction_change_ratio),
        min_tension_ratio=float(min_tension_ratio),
    )
    selected_target = select_next_target(label_counts, feasibility_rows)
    blocked_cases = [
        row["case_label"] for row in feasibility_rows if not bool(row["tension_safe_feasible"])
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "source_reports": {
            "sweep_report": sweep_report.get("output_dir"),
            "source_package": source_package.get("output_dir"),
        },
        "request": {
            "selected_per_case": int(selected_per_case),
            "max_dead_air_ratio": float(max_dead_air_ratio),
            "min_direction_change_ratio": float(min_direction_change_ratio),
            "min_tension_ratio": float(min_tension_ratio),
            "min_syncopation_ratio": float(min_syncopation_ratio),
            "dead_air_watch_max": float(dead_air_watch_max),
        },
        "residual_label_counts": label_counts,
        "feasibility_rows": feasibility_rows,
        "decision": {
            "current_boundary": "music_transformer_solo_yield_residual_tension_target_decision",
            "selected_repair_target": selected_target,
            "next_boundary": next_boundary_for_target(selected_target),
            "critical_user_input_required": False,
            "tension_repeat_feasible": not blocked_cases,
            "tension_repeat_blocked_cases": blocked_cases,
            "reason": (
                "tension-safe candidate count below selected_per_case for guarded cases; route to next residual major label"
                if blocked_cases and selected_target == "rhythm_syncopation_balance_repair"
                else "tension-safe candidate count supports another tension repair"
            ),
        },
        "readiness": {
            "residual_tension_target_decision_completed": True,
            "musical_quality_claimed": False,
            "artist_style_claimed": False,
            "production_ready_claimed": False,
        },
        "not_proven": [
            "human_audio_preference",
            "stable_jazz_solo_quality",
            "threshold_calibration",
            "guard_relaxation_safety",
        ],
    }


def validate_decision_report(
    report: dict[str, Any],
    *,
    expected_target: str,
    require_no_quality_claim: bool,
) -> dict[str, Any]:
    if str(report.get("schema_version")) != SCHEMA_VERSION:
        raise SoloYieldResidualTensionTargetDecisionError("schema version mismatch")
    decision = _dict(report.get("decision"))
    readiness = _dict(report.get("readiness"))
    if str(decision.get("selected_repair_target") or "") != str(expected_target):
        raise SoloYieldResidualTensionTargetDecisionError("selected target mismatch")
    if require_no_quality_claim:
        claimed = [
            key
            for key in ("musical_quality_claimed", "artist_style_claimed", "production_ready_claimed")
            if bool(readiness.get(key, True))
        ]
        if claimed:
            raise SoloYieldResidualTensionTargetDecisionError(
                f"unexpected quality claim: {claimed}"
            )
    return {
        "schema_version": str(report.get("schema_version")),
        "residual_label_counts": _dict(report.get("residual_label_counts")),
        "selected_repair_target": str(decision.get("selected_repair_target") or ""),
        "next_boundary": str(decision.get("next_boundary") or ""),
        "tension_repeat_feasible": bool(decision.get("tension_repeat_feasible", True)),
        "tension_repeat_blocked_cases": _list(decision.get("tension_repeat_blocked_cases")),
        "critical_user_input_required": bool(decision.get("critical_user_input_required", True)),
        "musical_quality_claimed": bool(readiness.get("musical_quality_claimed", True)),
    }


def _format_counts(value: dict[str, Any]) -> str:
    if not value:
        return "none"
    return ", ".join(f"{key}={value[key]}" for key in sorted(value))


def markdown_report(report: dict[str, Any]) -> str:
    decision = report["decision"]
    readiness = report["readiness"]
    lines = [
        "# Music Transformer Solo Yield Residual Tension Target Decision",
        "",
        "## Summary",
        "",
        f"- residual label counts: `{_format_counts(report['residual_label_counts'])}`",
        f"- tension repeat feasible: `{_bool_token(decision['tension_repeat_feasible'])}`",
        f"- tension repeat blocked cases: `{', '.join(decision['tension_repeat_blocked_cases']) or 'none'}`",
        f"- selected repair target: `{decision['selected_repair_target']}`",
        f"- next boundary: `{decision['next_boundary']}`",
        f"- critical user input required: `{_bool_token(decision['critical_user_input_required'])}`",
        f"- musical quality claimed: `{_bool_token(readiness['musical_quality_claimed'])}`",
        "",
        "## Feasibility Rows",
        "",
        "| case | selected per case | guarded | tension-safe | feasible |",
        "|---|---:|---:|---:|---|",
    ]
    for row in report.get("feasibility_rows", []):
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['case_label']}`",
                    str(row["selected_per_case"]),
                    str(row["guarded_candidate_count"]),
                    str(row["tension_safe_candidate_count"]),
                    f"`{_bool_token(row['tension_safe_feasible'])}`",
                ]
            )
            + " |"
        )
    lines.extend(["", "## Not Proven", ""])
    for item in report.get("not_proven", []):
        lines.append(f"- `{item}`")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Decide residual tension target")
    parser.add_argument("--sweep_report", type=str, required=True)
    parser.add_argument("--source_package", type=str, required=True)
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/music_transformer_finetune_mvp/solo_yield_residual_tension_decision",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--selected_per_case", type=int, default=2)
    parser.add_argument("--max_dead_air_ratio", type=float, default=0.68)
    parser.add_argument("--min_direction_change_ratio", type=float, default=0.50)
    parser.add_argument("--min_tension_ratio", type=float, default=0.20)
    parser.add_argument("--min_syncopation_ratio", type=float, default=0.70)
    parser.add_argument("--dead_air_watch_max", type=float, default=0.66)
    parser.add_argument("--expected_target", type=str, default="rhythm_syncopation_balance_repair")
    parser.add_argument("--require_no_quality_claim", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    report = build_decision_report(
        sweep_report=read_json(Path(args.sweep_report)),
        source_package=read_json(Path(args.source_package)),
        output_dir=output_dir,
        selected_per_case=int(args.selected_per_case),
        max_dead_air_ratio=float(args.max_dead_air_ratio),
        min_direction_change_ratio=float(args.min_direction_change_ratio),
        min_tension_ratio=float(args.min_tension_ratio),
        min_syncopation_ratio=float(args.min_syncopation_ratio),
        dead_air_watch_max=float(args.dead_air_watch_max),
    )
    summary = validate_decision_report(
        report,
        expected_target=str(args.expected_target),
        require_no_quality_claim=bool(args.require_no_quality_claim),
    )
    write_json(output_dir / "residual_tension_target_decision.json", report)
    write_json(output_dir / "residual_tension_target_decision_summary.json", summary)
    markdown = markdown_report(report)
    write_text(output_dir / "residual_tension_target_decision.md", markdown)
    if args.doc_path:
        write_text(Path(args.doc_path), markdown)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
