"""Consolidate duration/coverage fill repeatability evidence boundaries."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class StageBDurationCoverageRepeatabilityConsolidationError(ValueError):
    pass


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def build_repeatability_consolidation_report(
    *,
    user_listening_consolidation: dict[str, Any],
    dead_air_gain_repair: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    user_boundary = _dict(user_listening_consolidation.get("consolidated_claim_boundary"))
    user_alignment = _dict(user_listening_consolidation.get("evidence_alignment"))
    repair_summary = _dict(dead_air_gain_repair.get("repair_summary"))
    repair_claim = _dict(dead_air_gain_repair.get("claim_boundary"))

    if str(user_boundary.get("preferred_candidate") or "") != "duration_coverage_fill_keep":
        raise StageBDurationCoverageRepeatabilityConsolidationError("unexpected user preferred candidate")
    if not bool(user_boundary.get("single_user_human_audio_preference_claimed", False)):
        raise StageBDurationCoverageRepeatabilityConsolidationError("single-user preference is required")
    if bool(user_boundary.get("broad_model_quality_claimed", True)):
        raise StageBDurationCoverageRepeatabilityConsolidationError("user consolidation must not claim broad quality")
    if str(repair_summary.get("boundary") or "") != "qualified_gate_repeatability_with_dead_air_gain":
        raise StageBDurationCoverageRepeatabilityConsolidationError("dead-air gain repeatability boundary is required")
    if not bool(repair_claim.get("selected_distinct_source_dead_air_gain_claimed", False)):
        raise StageBDurationCoverageRepeatabilityConsolidationError("dead-air gain claim is required")
    if bool(repair_claim.get("broad_model_quality_claimed", True)):
        raise StageBDurationCoverageRepeatabilityConsolidationError("repair report must not claim broad quality")

    boundary = "current_keep_and_distinct_source_dead_air_gain_midi_support"
    return {
        "schema_version": "stage_b_duration_coverage_fill_repeatability_consolidation_v1",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "source_schemas": {
            "user_listening_consolidation": str(user_listening_consolidation.get("schema_version") or ""),
            "dead_air_gain_repair": str(dead_air_gain_repair.get("schema_version") or ""),
        },
        "current_keep_anchor": {
            "candidate_id": str(user_listening_consolidation.get("candidate_id") or ""),
            "preferred_candidate": str(user_boundary.get("preferred_candidate") or ""),
            "midi_and_user_preference_aligned": bool(user_alignment.get("same_preferred_candidate", False)),
            "single_user_human_audio_preference_claimed": True,
            "rendered_audio_file_count": int(user_alignment.get("rendered_audio_file_count", 0) or 0),
        },
        "distinct_source_repeatability": {
            "boundary": str(repair_summary.get("boundary") or ""),
            "source_candidate_count": int(repair_summary.get("source_candidate_count", 0) or 0),
            "qualified_source_candidate_count": int(repair_summary.get("qualified_source_candidate_count", 0) or 0),
            "dead_air_gain_source_candidate_count": int(
                repair_summary.get("dead_air_gain_source_candidate_count", 0) or 0
            ),
            "total_variant_count": int(repair_summary.get("total_variant_count", 0) or 0),
            "total_qualified_variant_count": int(repair_summary.get("total_qualified_variant_count", 0) or 0),
            "total_dead_air_gain_variant_count": int(
                repair_summary.get("total_dead_air_gain_variant_count", 0) or 0
            ),
            "selected_fill_additions": list(repair_summary.get("selected_fill_additions") or []),
        },
        "consolidated_claim_boundary": {
            "boundary": boundary,
            "current_keep_single_user_preference_claimed": True,
            "distinct_source_midi_gate_repeatability_claimed": True,
            "distinct_source_dead_air_gain_claimed": True,
            "new_source_human_audio_preference_claimed": False,
            "multi_reviewer_preference_claimed": False,
            "broad_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
            "production_ready_improviser_claimed": False,
        },
        "proven": [
            "current_keep_midi_and_single_user_listening_preference",
            "distinct_source_qualified_midi_gate_repeatability",
            "distinct_source_selected_dead_air_gain",
        ],
        "not_proven": [
            "new_source_human_audio_preference",
            "multi_reviewer_preference",
            "broad_trained_model_quality",
            "brad_style_adaptation",
            "production_ready_improviser",
        ],
        "next_recommended_issue": (
            "Stage B margin-recovered phrase/vocabulary duration coverage fill repeatability audio review package"
        ),
    }


def validate_repeatability_consolidation(
    report: dict[str, Any],
    *,
    expected_boundary: str | None,
    require_no_broad_quality_claim: bool,
) -> dict[str, Any]:
    boundary = _dict(report.get("consolidated_claim_boundary"))
    repeatability = _dict(report.get("distinct_source_repeatability"))
    boundary_name = str(boundary.get("boundary") or "")
    if expected_boundary and boundary_name != expected_boundary:
        raise StageBDurationCoverageRepeatabilityConsolidationError(
            f"expected boundary {expected_boundary}, got {boundary_name}"
        )
    if int(repeatability.get("dead_air_gain_source_candidate_count", 0) or 0) < 2:
        raise StageBDurationCoverageRepeatabilityConsolidationError("dead-air gain source count is too low")
    if require_no_broad_quality_claim:
        blocked = [
            "new_source_human_audio_preference_claimed",
            "multi_reviewer_preference_claimed",
            "broad_model_quality_claimed",
            "brad_style_adaptation_claimed",
            "production_ready_improviser_claimed",
        ]
        claimed = [name for name in blocked if bool(boundary.get(name, True))]
        if claimed:
            raise StageBDurationCoverageRepeatabilityConsolidationError(f"unexpected broad claim: {claimed}")
    return {
        "boundary": boundary_name,
        "current_keep_single_user_preference_claimed": bool(
            boundary.get("current_keep_single_user_preference_claimed", False)
        ),
        "distinct_source_midi_gate_repeatability_claimed": bool(
            boundary.get("distinct_source_midi_gate_repeatability_claimed", False)
        ),
        "distinct_source_dead_air_gain_claimed": bool(
            boundary.get("distinct_source_dead_air_gain_claimed", False)
        ),
        "source_candidate_count": int(repeatability.get("source_candidate_count", 0) or 0),
        "dead_air_gain_source_candidate_count": int(
            repeatability.get("dead_air_gain_source_candidate_count", 0) or 0
        ),
        "broad_model_quality_claimed": bool(boundary.get("broad_model_quality_claimed", True)),
        "next_recommended_issue": str(report.get("next_recommended_issue") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    boundary = report["consolidated_claim_boundary"]
    anchor = report["current_keep_anchor"]
    repeatability = report["distinct_source_repeatability"]
    lines = [
        "# Stage B Duration Coverage Fill Repeatability Consolidation",
        "",
        f"- boundary: `{boundary['boundary']}`",
        f"- current keep anchor: `{anchor['candidate_id']}`",
        f"- current keep single-user preference: `{boundary['current_keep_single_user_preference_claimed']}`",
        f"- distinct source MIDI repeatability: `{boundary['distinct_source_midi_gate_repeatability_claimed']}`",
        f"- distinct source dead-air gain: `{boundary['distinct_source_dead_air_gain_claimed']}`",
        f"- new source human/audio preference claimed: `{boundary['new_source_human_audio_preference_claimed']}`",
        f"- broad model quality claimed: `{boundary['broad_model_quality_claimed']}`",
        "",
        "## Repeatability Evidence",
        "",
        f"- source candidates: `{repeatability['source_candidate_count']}`",
        f"- qualified source candidates: `{repeatability['qualified_source_candidate_count']}`",
        f"- dead-air gain source candidates: `{repeatability['dead_air_gain_source_candidate_count']}`",
        f"- total variants: `{repeatability['total_variant_count']}`",
        f"- qualified variants: `{repeatability['total_qualified_variant_count']}`",
        f"- dead-air gain variants: `{repeatability['total_dead_air_gain_variant_count']}`",
        f"- selected fill additions: `{repeatability['selected_fill_additions']}`",
        "",
        "## Not Proven",
        "",
    ]
    for item in report.get("not_proven", []):
        lines.append(f"- `{item}`")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Consolidate duration coverage fill repeatability evidence")
    parser.add_argument(
        "--user_listening_consolidation",
        type=str,
        default="outputs/stage_b_duration_coverage_fill_user_listening_review_consolidation/"
        "harness_stage_b_duration_coverage_fill_user_listening_review_consolidation/"
        "stage_b_duration_coverage_fill_user_listening_review_consolidation.json",
    )
    parser.add_argument(
        "--dead_air_gain_repair",
        type=str,
        default="outputs/stage_b_duration_coverage_fill_dead_air_gain_repeatability_repair/"
        "harness_stage_b_duration_coverage_fill_dead_air_gain_repeatability_repair/"
        "stage_b_duration_coverage_fill_dead_air_gain_repeatability_repair.json",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/stage_b_duration_coverage_fill_repeatability_consolidation",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--expected_boundary", type=str, default="")
    parser.add_argument("--require_no_broad_quality_claim", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    report = build_repeatability_consolidation_report(
        user_listening_consolidation=read_json(Path(args.user_listening_consolidation)),
        dead_air_gain_repair=read_json(Path(args.dead_air_gain_repair)),
        output_dir=output_dir,
    )
    summary = validate_repeatability_consolidation(
        report,
        expected_boundary=str(args.expected_boundary or ""),
        require_no_broad_quality_claim=bool(args.require_no_broad_quality_claim),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "stage_b_duration_coverage_fill_repeatability_consolidation.json"
    markdown_path = output_dir / "stage_b_duration_coverage_fill_repeatability_consolidation.md"
    write_json(report_path, report)
    write_json(output_dir / "stage_b_duration_coverage_fill_repeatability_consolidation_validation_summary.json", summary)
    markdown_path.write_text(markdown_report(report), encoding="utf-8")
    print(json.dumps({**summary, "report_path": str(report_path), "markdown_path": str(markdown_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
