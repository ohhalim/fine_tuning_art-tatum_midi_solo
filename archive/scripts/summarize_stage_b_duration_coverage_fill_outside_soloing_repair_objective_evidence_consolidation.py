"""Consolidate objective evidence for outside-soloing repair candidates."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class StageBDurationCoverageOutsideSoloingRepairObjectiveEvidenceConsolidationError(ValueError):
    pass


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def selected_rows(repair_sweep: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for result in _list(repair_sweep.get("source_repair_results")):
        if not isinstance(result, dict):
            continue
        selected = _dict(result.get("selected_candidate"))
        rows.append(
            {
                "source_candidate_id": str(result.get("source_candidate_id") or ""),
                "sample_seed": int(result.get("sample_seed", 0) or 0),
                "source_selected_dead_air_ratio": float(result.get("source_selected_dead_air_ratio", 1.0) or 1.0),
                "source_selected_max_interval": int(result.get("source_selected_max_interval", 0) or 0),
                "candidate_id": str(selected.get("candidate_id") or ""),
                "repair_policy": str(selected.get("repair_policy") or ""),
                "qualified": bool(_dict(selected.get("outside_soloing_gate")).get("qualified", False)),
                "flags": list(_dict(selected.get("outside_soloing_gate")).get("flags") or []),
                "dead_air_ratio": float(_dict(selected.get("metrics")).get("dead_air_ratio", 1.0) or 1.0),
                "chord_tone_ratio": float(_dict(selected.get("metrics")).get("chord_tone_ratio", 0.0) or 0.0),
                "focused_unique_pitch_count": int(
                    _dict(selected.get("focused_solo_metrics")).get("focused_unique_pitch_count", 0) or 0
                ),
                "focused_max_interval": int(
                    _dict(selected.get("focused_solo_metrics")).get("focused_max_interval", 0) or 0
                ),
                "max_non_chord_tone_run": int(
                    _dict(selected.get("pitch_role_metrics")).get("max_non_chord_tone_run", 0) or 0
                ),
                "midi_path": str(selected.get("midi_path") or ""),
            }
        )
    return rows


def validate_inputs(
    *,
    repair_sweep: dict[str, Any],
    user_review_fill: dict[str, Any],
) -> None:
    repair_summary = _dict(repair_sweep.get("repair_summary"))
    repair_claim = _dict(repair_sweep.get("claim_boundary"))
    review_claim = _dict(user_review_fill.get("claim_boundary"))
    review_decision = _dict(user_review_fill.get("decision"))
    if str(repair_summary.get("boundary") or "") != "outside_soloing_pitch_role_repair_candidates":
        raise StageBDurationCoverageOutsideSoloingRepairObjectiveEvidenceConsolidationError(
            "outside-soloing repair candidate boundary required"
        )
    if int(repair_summary.get("repaired_source_candidate_count", 0) or 0) < 2:
        raise StageBDurationCoverageOutsideSoloingRepairObjectiveEvidenceConsolidationError(
            "two repaired source candidates required"
        )
    if bool(repair_claim.get("broad_model_quality_claimed", True)):
        raise StageBDurationCoverageOutsideSoloingRepairObjectiveEvidenceConsolidationError(
            "broad model quality must not be claimed"
        )
    if str(review_claim.get("boundary") or "") != "outside_soloing_repair_audio_review_pending":
        raise StageBDurationCoverageOutsideSoloingRepairObjectiveEvidenceConsolidationError(
            "pending audio review boundary required"
        )
    if bool(review_claim.get("human_audio_preference_claimed", True)):
        raise StageBDurationCoverageOutsideSoloingRepairObjectiveEvidenceConsolidationError(
            "human/audio preference must not be claimed"
        )
    if not bool(review_decision.get("objective_auto_progress_allowed", False)):
        raise StageBDurationCoverageOutsideSoloingRepairObjectiveEvidenceConsolidationError(
            "objective auto progress must be allowed"
        )


def build_objective_evidence_consolidation(
    *,
    repair_sweep: dict[str, Any],
    user_review_fill: dict[str, Any],
    output_dir: Path,
    min_repaired_source_candidates: int,
    min_chord_tone_ratio: float,
    max_non_chord_run: int,
    max_interval: int,
) -> dict[str, Any]:
    validate_inputs(repair_sweep=repair_sweep, user_review_fill=user_review_fill)
    rows = selected_rows(repair_sweep)
    if len(rows) < int(min_repaired_source_candidates):
        raise StageBDurationCoverageOutsideSoloingRepairObjectiveEvidenceConsolidationError(
            f"expected {int(min_repaired_source_candidates)} selected rows, got {len(rows)}"
        )

    qualified_count = sum(1 for row in rows if row["qualified"])
    dead_air_preserved_count = sum(
        1 for row in rows if row["dead_air_ratio"] <= row["source_selected_dead_air_ratio"] + 1e-9
    )
    chord_tone_pass_count = sum(1 for row in rows if row["chord_tone_ratio"] >= float(min_chord_tone_ratio))
    non_chord_run_pass_count = sum(1 for row in rows if row["max_non_chord_tone_run"] <= int(max_non_chord_run))
    interval_pass_count = sum(1 for row in rows if row["focused_max_interval"] <= int(max_interval))
    max_interval_improved_count = sum(
        1
        for row in rows
        if row["source_selected_max_interval"] > 0
        and row["focused_max_interval"] < row["source_selected_max_interval"]
    )
    source_count = len(rows)
    objective_support = (
        qualified_count >= int(min_repaired_source_candidates)
        and dead_air_preserved_count >= int(min_repaired_source_candidates)
        and chord_tone_pass_count >= int(min_repaired_source_candidates)
        and non_chord_run_pass_count >= int(min_repaired_source_candidates)
        and interval_pass_count >= int(min_repaired_source_candidates)
    )
    boundary = (
        "outside_soloing_repair_objective_evidence_support"
        if objective_support
        else "outside_soloing_repair_objective_evidence_incomplete"
    )

    return {
        "schema_version": "stage_b_duration_coverage_fill_outside_soloing_repair_objective_evidence_consolidation_v1",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "source_schemas": {
            "repair_sweep": str(repair_sweep.get("schema_version") or ""),
            "user_review_fill": str(user_review_fill.get("schema_version") or ""),
        },
        "thresholds": {
            "min_repaired_source_candidates": int(min_repaired_source_candidates),
            "min_chord_tone_ratio": float(min_chord_tone_ratio),
            "max_non_chord_run": int(max_non_chord_run),
            "max_interval": int(max_interval),
        },
        "selected_candidates": rows,
        "objective_evidence_summary": {
            "boundary": boundary,
            "source_candidate_count": int(source_count),
            "qualified_source_candidate_count": int(qualified_count),
            "dead_air_preserved_source_candidate_count": int(dead_air_preserved_count),
            "chord_tone_pass_source_candidate_count": int(chord_tone_pass_count),
            "non_chord_run_pass_source_candidate_count": int(non_chord_run_pass_count),
            "interval_pass_source_candidate_count": int(interval_pass_count),
            "max_interval_improved_source_candidate_count": int(max_interval_improved_count),
            "selected_min_chord_tone_ratio": min(row["chord_tone_ratio"] for row in rows),
            "selected_max_non_chord_tone_run": max(row["max_non_chord_tone_run"] for row in rows),
            "selected_max_interval": max(row["focused_max_interval"] for row in rows),
            "human_audio_preference_claimed": False,
            "broad_model_quality_claimed": False,
        },
        "claim_boundary": {
            "boundary": boundary,
            "objective_midi_evidence_claimed": bool(objective_support),
            "dead_air_gain_preserved_claimed": dead_air_preserved_count >= int(min_repaired_source_candidates),
            "pitch_role_repair_claimed": chord_tone_pass_count >= int(min_repaired_source_candidates)
            and non_chord_run_pass_count >= int(min_repaired_source_candidates),
            "human_audio_preference_claimed": False,
            "multi_reviewer_preference_claimed": False,
            "broad_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
            "production_ready_improviser_claimed": False,
        },
        "proven": [
            "objective_pitch_role_gate_on_selected_repaired_sources",
            "dead_air_gain_preserved_on_selected_repaired_sources",
            "audio_review_pending_state_recorded",
        ]
        if objective_support
        else [],
        "not_proven": [
            "human_audio_preference",
            "multi_reviewer_preference",
            "broad_trained_model_quality",
            "brad_style_adaptation",
            "production_ready_improviser",
        ],
        "next_recommended_issue": (
            "Stage B margin-recovered phrase/vocabulary duration coverage fill outside-soloing repair next decision"
        ),
    }


def validate_objective_evidence_consolidation(
    report: dict[str, Any],
    *,
    expected_boundary: str | None,
    min_repaired_source_candidates: int,
    require_no_preference_claim: bool,
    require_no_broad_quality_claim: bool,
) -> dict[str, Any]:
    summary = _dict(report.get("objective_evidence_summary"))
    claim = _dict(report.get("claim_boundary"))
    boundary = str(summary.get("boundary") or "")
    if expected_boundary and boundary != expected_boundary:
        raise StageBDurationCoverageOutsideSoloingRepairObjectiveEvidenceConsolidationError(
            f"expected boundary {expected_boundary}, got {boundary}"
        )
    if int(summary.get("source_candidate_count", 0) or 0) < int(min_repaired_source_candidates):
        raise StageBDurationCoverageOutsideSoloingRepairObjectiveEvidenceConsolidationError(
            "not enough source candidates"
        )
    if require_no_preference_claim and bool(claim.get("human_audio_preference_claimed", True)):
        raise StageBDurationCoverageOutsideSoloingRepairObjectiveEvidenceConsolidationError(
            "human/audio preference must not be claimed"
        )
    if require_no_broad_quality_claim and bool(claim.get("broad_model_quality_claimed", True)):
        raise StageBDurationCoverageOutsideSoloingRepairObjectiveEvidenceConsolidationError(
            "broad model quality must not be claimed"
        )
    return {
        "boundary": boundary,
        "source_candidate_count": int(summary.get("source_candidate_count", 0) or 0),
        "qualified_source_candidate_count": int(summary.get("qualified_source_candidate_count", 0) or 0),
        "dead_air_preserved_source_candidate_count": int(
            summary.get("dead_air_preserved_source_candidate_count", 0) or 0
        ),
        "chord_tone_pass_source_candidate_count": int(
            summary.get("chord_tone_pass_source_candidate_count", 0) or 0
        ),
        "non_chord_run_pass_source_candidate_count": int(
            summary.get("non_chord_run_pass_source_candidate_count", 0) or 0
        ),
        "interval_pass_source_candidate_count": int(summary.get("interval_pass_source_candidate_count", 0) or 0),
        "selected_min_chord_tone_ratio": float(summary.get("selected_min_chord_tone_ratio", 0.0) or 0.0),
        "selected_max_non_chord_tone_run": int(summary.get("selected_max_non_chord_tone_run", 0) or 0),
        "selected_max_interval": int(summary.get("selected_max_interval", 0) or 0),
        "human_audio_preference_claimed": bool(summary.get("human_audio_preference_claimed", True)),
        "broad_model_quality_claimed": bool(summary.get("broad_model_quality_claimed", True)),
        "next_recommended_issue": str(report.get("next_recommended_issue") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    summary = report["objective_evidence_summary"]
    lines = [
        "# Stage B Duration Coverage Fill Outside-Soloing Repair Objective Evidence Consolidation",
        "",
        f"- boundary: `{summary['boundary']}`",
        f"- source candidates: `{summary['source_candidate_count']}`",
        f"- qualified source candidates: `{summary['qualified_source_candidate_count']}`",
        f"- dead-air preserved source candidates: `{summary['dead_air_preserved_source_candidate_count']}`",
        f"- chord-tone pass source candidates: `{summary['chord_tone_pass_source_candidate_count']}`",
        f"- non-chord run pass source candidates: `{summary['non_chord_run_pass_source_candidate_count']}`",
        f"- interval pass source candidates: `{summary['interval_pass_source_candidate_count']}`",
        f"- selected min chord-tone ratio: `{summary['selected_min_chord_tone_ratio']:.3f}`",
        f"- selected max non-chord run: `{summary['selected_max_non_chord_tone_run']}`",
        f"- selected max interval: `{summary['selected_max_interval']}`",
        f"- human/audio preference claimed: `{summary['human_audio_preference_claimed']}`",
        f"- broad model quality claimed: `{summary['broad_model_quality_claimed']}`",
        "",
        "| sample seed | policy | qualified | dead-air | chord-tone | non-chord run | max interval | interval delta |",
        "|---:|---|:---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["selected_candidates"]:
        lines.append(
            "| {sample_seed} | `{policy}` | {qualified} | {dead_air:.4f} | {chord_tone:.3f} | "
            "{non_chord_run} | {interval} | {interval_delta:+d} |".format(
                sample_seed=int(row["sample_seed"]),
                policy=row["repair_policy"],
                qualified=bool(row["qualified"]),
                dead_air=float(row["dead_air_ratio"]),
                chord_tone=float(row["chord_tone_ratio"]),
                non_chord_run=int(row["max_non_chord_tone_run"]),
                interval=int(row["focused_max_interval"]),
                interval_delta=int(row["focused_max_interval"]) - int(row["source_selected_max_interval"]),
            )
        )
    lines.extend(["", "## Not Proven", ""])
    for item in report.get("not_proven", []):
        lines.append(f"- `{item}`")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Consolidate outside-soloing repair objective evidence")
    parser.add_argument(
        "--repair_sweep",
        type=str,
        default="outputs/stage_b_duration_coverage_fill_outside_soloing_repair_sweep/"
        "harness_stage_b_duration_coverage_fill_outside_soloing_repair_sweep/"
        "stage_b_duration_coverage_fill_outside_soloing_repair_sweep.json",
    )
    parser.add_argument(
        "--user_review_fill",
        type=str,
        default="outputs/stage_b_duration_coverage_fill_outside_soloing_repair_user_listening_review_fill/"
        "harness_stage_b_duration_coverage_fill_outside_soloing_repair_user_listening_review_fill/"
        "stage_b_duration_coverage_fill_outside_soloing_repair_user_listening_review_fill.json",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/stage_b_duration_coverage_fill_outside_soloing_repair_objective_evidence_consolidation",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--min_repaired_source_candidates", type=int, default=2)
    parser.add_argument("--min_chord_tone_ratio", type=float, default=0.72)
    parser.add_argument("--max_non_chord_run", type=int, default=1)
    parser.add_argument("--max_interval", type=int, default=7)
    parser.add_argument("--expected_boundary", type=str, default="")
    parser.add_argument("--require_no_preference_claim", action="store_true")
    parser.add_argument("--require_no_broad_quality_claim", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    report = build_objective_evidence_consolidation(
        repair_sweep=read_json(Path(args.repair_sweep)),
        user_review_fill=read_json(Path(args.user_review_fill)),
        output_dir=output_dir,
        min_repaired_source_candidates=int(args.min_repaired_source_candidates),
        min_chord_tone_ratio=float(args.min_chord_tone_ratio),
        max_non_chord_run=int(args.max_non_chord_run),
        max_interval=int(args.max_interval),
    )
    summary = validate_objective_evidence_consolidation(
        report,
        expected_boundary=str(args.expected_boundary or ""),
        min_repaired_source_candidates=int(args.min_repaired_source_candidates),
        require_no_preference_claim=bool(args.require_no_preference_claim),
        require_no_broad_quality_claim=bool(args.require_no_broad_quality_claim),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "stage_b_duration_coverage_fill_outside_soloing_repair_objective_evidence_consolidation.json"
    markdown_path = output_dir / "stage_b_duration_coverage_fill_outside_soloing_repair_objective_evidence_consolidation.md"
    write_json(report_path, report)
    write_json(
        output_dir
        / "stage_b_duration_coverage_fill_outside_soloing_repair_objective_evidence_consolidation_validation_summary.json",
        summary,
    )
    markdown_path.write_text(markdown_report(report), encoding="utf-8")
    print(json.dumps({**summary, "report_path": str(report_path), "markdown_path": str(markdown_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
