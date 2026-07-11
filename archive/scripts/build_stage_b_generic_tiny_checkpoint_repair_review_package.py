"""Build a review package for generic tiny checkpoint repair candidates."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from scripts.assess_stage_b_generic_base_readiness import read_json, write_json, write_text  # noqa: E402
from scripts.run_stage_b_generic_tiny_checkpoint_generation_probe import (  # noqa: E402
    _bool_token,
    _dict,
    _float,
    _int,
)


class StageBGenericTinyCheckpointRepairReviewPackageError(ValueError):
    pass


def metric_value(row: dict[str, Any], key: str, default: Any = 0) -> Any:
    return _dict(row.get("metrics")).get(key, default)


def build_candidate_row(row: dict[str, Any], *, package_midi_path: Path | None = None) -> dict[str, Any]:
    temporal = _dict(row.get("temporal_coverage"))
    phrase = _dict(row.get("phrase_contour"))
    pitch_roles = _dict(row.get("pitch_roles"))
    rhythm = _dict(row.get("rhythm_profile"))
    collapse = _dict(row.get("collapse"))
    return {
        "sample_index": _int(row.get("sample_index")),
        "sample_seed": _int(row.get("sample_seed")),
        "source_midi_path": str(row.get("midi_path") or ""),
        "package_midi_path": str(package_midi_path) if package_midi_path else "",
        "valid": bool(row.get("valid", False)),
        "strict_valid": bool(row.get("strict_valid", False)),
        "grammar_gate_passed": bool(row.get("grammar_gate_passed", False)),
        "note_count": _int(metric_value(row, "note_count")),
        "unique_pitch_count": _int(metric_value(row, "unique_pitch_count")),
        "dead_air_ratio": _float(metric_value(row, "dead_air_ratio")),
        "phrase_coverage_ratio": _float(metric_value(row, "phrase_coverage_ratio")),
        "chord_tone_ratio": _float(metric_value(row, "chord_tone_ratio")),
        "max_simultaneous_notes": _int(metric_value(row, "max_simultaneous_notes")),
        "max_note_duration_ratio": _float(metric_value(row, "max_note_duration_ratio")),
        "onset_coverage_ratio": _float(temporal.get("onset_coverage_ratio")),
        "sustained_coverage_ratio": _float(temporal.get("sustained_coverage_ratio")),
        "adjacent_repeated_pitch_ratio": _float(phrase.get("adjacent_repeated_pitch_ratio")),
        "direction_change_ratio": _float(phrase.get("direction_change_ratio")),
        "root_tone_ratio": _float(pitch_roles.get("root_tone_ratio")),
        "tension_ratio": _float(pitch_roles.get("tension_ratio")),
        "syncopated_onset_ratio": _float(rhythm.get("syncopated_onset_ratio")),
        "postprocess_removal_ratio": _float(collapse.get("postprocess_removal_ratio")),
        "failure_reason": row.get("diagnostic_failure_reason"),
    }


def sort_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        candidates,
        key=lambda row: (
            _float(row.get("dead_air_ratio")),
            -_float(row.get("phrase_coverage_ratio")),
            -_int(row.get("unique_pitch_count")),
            _int(row.get("sample_index")),
        ),
    )


def copy_candidate_midis(candidates: list[dict[str, Any]], midi_dir: Path) -> list[dict[str, Any]]:
    midi_dir.mkdir(parents=True, exist_ok=True)
    copied: list[dict[str, Any]] = []
    for rank, candidate in enumerate(candidates, start=1):
        source = Path(str(candidate["source_midi_path"]))
        target = midi_dir / f"rank_{rank:02d}_seed_{candidate['sample_seed']}_sample_{candidate['sample_index']}.mid"
        if source.exists():
            shutil.copy2(source, target)
        copied.append({**candidate, "review_rank": rank, "package_midi_path": str(target)})
    return copied


def build_review_package_report(
    *,
    run_dir: Path,
    repeatability_report_path: Path,
    repeatability_report: dict[str, Any],
    generation_report: dict[str, Any],
    candidates: list[dict[str, Any]],
    failed_rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    summary = _dict(_dict(repeatability_report.get("generation")).get("summary"))
    candidate_count = len(candidates)
    required_count = int(args.min_candidate_count)
    review_package_ready = candidate_count >= required_count
    return {
        "schema_version": "stage_b_generic_tiny_checkpoint_repair_review_package_v1",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "run_dir": str(run_dir),
        "repeatability_report_path": str(repeatability_report_path),
        "generation_report_path": str(_dict(repeatability_report.get("generation")).get("report_path") or ""),
        "input": {
            "issue_number": int(args.issue_number),
            "min_candidate_count": required_count,
        },
        "source_summary": {
            "sample_count": _int(summary.get("sample_count")),
            "valid_sample_count": _int(summary.get("valid_sample_count")),
            "strict_valid_sample_count": _int(summary.get("strict_valid_sample_count")),
            "grammar_gate_sample_count": _int(summary.get("grammar_gate_sample_count")),
            "valid_sample_rate": _float(summary.get("valid_sample_rate")),
            "strict_valid_sample_rate": _float(summary.get("strict_valid_sample_rate")),
            "grammar_gate_sample_rate": _float(summary.get("grammar_gate_sample_rate")),
            "collapse_warning_sample_rate": _float(summary.get("collapse_warning_sample_rate")),
        },
        "review_package": {
            "candidate_count": candidate_count,
            "failed_candidate_count": len(failed_rows),
            "review_package_ready": review_package_ready,
            "midi_dir": str(run_dir / "midi"),
            "candidates": candidates,
            "failed_rows": failed_rows,
        },
        "readiness": {
            "boundary": "stage_b_generic_tiny_checkpoint_repair_review_package",
            "review_package_ready": review_package_ready,
            "musical_quality_claimed": False,
            "raw_generation_quality_claimed": False,
            "constrained_generation_quality_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
        },
        "decision": {
            "current_boundary": "stage_b_generic_tiny_checkpoint_repair_review_package",
            "next_boundary": (
                "stage_b_generic_tiny_checkpoint_repair_listening_notes"
                if review_package_ready
                else "stage_b_generic_tiny_checkpoint_repair_review_package_rebuild"
            ),
            "auto_progress_allowed": True,
            "critical_user_input_required": False,
            "reason": "strict-valid repair candidates are packaged for review without musical quality claims",
        },
        "not_proven": [
            "musical_quality",
            "unconstrained_raw_generation_quality",
            "broad_trained_model_quality",
            "brad_style_adaptation",
        ],
        "next_recommended_issue": (
            "Stage B generic tiny checkpoint repair listening notes"
            if review_package_ready
            else "Stage B generic tiny checkpoint repair review package rebuild"
        ),
    }


def validate_review_package_report(
    report: dict[str, Any],
    *,
    expected_boundary: str | None,
    require_review_package_ready: bool,
    require_no_musical_quality_claim: bool,
    require_no_broad_quality_claim: bool,
    require_no_brad_style_claim: bool,
) -> dict[str, Any]:
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    package = _dict(report.get("review_package"))
    source = _dict(report.get("source_summary"))
    boundary = str(readiness.get("boundary") or "")
    if expected_boundary and boundary != expected_boundary:
        raise StageBGenericTinyCheckpointRepairReviewPackageError(
            f"expected boundary {expected_boundary}, got {boundary}"
        )
    if require_review_package_ready and not bool(readiness.get("review_package_ready", False)):
        raise StageBGenericTinyCheckpointRepairReviewPackageError("review package should be ready")
    if require_no_musical_quality_claim and bool(readiness.get("musical_quality_claimed", True)):
        raise StageBGenericTinyCheckpointRepairReviewPackageError("musical quality must not be claimed")
    if require_no_broad_quality_claim and bool(readiness.get("broad_trained_model_quality_claimed", True)):
        raise StageBGenericTinyCheckpointRepairReviewPackageError("broad trained-model quality must not be claimed")
    if require_no_brad_style_claim and bool(readiness.get("brad_style_adaptation_claimed", True)):
        raise StageBGenericTinyCheckpointRepairReviewPackageError("Brad style adaptation must not be claimed")
    if _int(package.get("candidate_count")) <= 0:
        raise StageBGenericTinyCheckpointRepairReviewPackageError("review candidate count must be positive")
    return {
        "boundary": boundary,
        "next_boundary": str(decision.get("next_boundary") or ""),
        "source_sample_count": _int(source.get("sample_count")),
        "source_strict_valid_sample_count": _int(source.get("strict_valid_sample_count")),
        "source_grammar_gate_sample_count": _int(source.get("grammar_gate_sample_count")),
        "candidate_count": _int(package.get("candidate_count")),
        "failed_candidate_count": _int(package.get("failed_candidate_count")),
        "review_package_ready": bool(readiness.get("review_package_ready", False)),
        "musical_quality_claimed": bool(readiness.get("musical_quality_claimed", True)),
        "raw_generation_quality_claimed": bool(readiness.get("raw_generation_quality_claimed", True)),
        "constrained_generation_quality_claimed": bool(
            readiness.get("constrained_generation_quality_claimed", True)
        ),
        "broad_trained_model_quality_claimed": bool(
            readiness.get("broad_trained_model_quality_claimed", True)
        ),
        "brad_style_adaptation_claimed": bool(readiness.get("brad_style_adaptation_claimed", True)),
        "critical_user_input_required": bool(decision.get("critical_user_input_required", True)),
        "next_recommended_issue": str(report.get("next_recommended_issue") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    readiness = report["readiness"]
    decision = report["decision"]
    source = report["source_summary"]
    package = report["review_package"]
    lines = [
        "# Stage B Generic Tiny Checkpoint Repair Review Package",
        "",
        "## Summary",
        "",
        f"- boundary: `{readiness['boundary']}`",
        f"- next boundary: `{decision['next_boundary']}`",
        f"- review package ready: `{_bool_token(readiness['review_package_ready'])}`",
        f"- musical quality claimed: `{_bool_token(readiness['musical_quality_claimed'])}`",
        f"- broad trained-model quality claimed: `{_bool_token(readiness['broad_trained_model_quality_claimed'])}`",
        f"- Brad style adaptation claimed: `{_bool_token(readiness['brad_style_adaptation_claimed'])}`",
        "",
        "## Source",
        "",
        f"- source sample count: `{source['sample_count']}`",
        f"- source strict valid sample count: `{source['strict_valid_sample_count']}`",
        f"- source grammar gate sample count: `{source['grammar_gate_sample_count']}`",
        f"- source valid / strict / grammar rate: `{source['valid_sample_rate']}/"
        f"{source['strict_valid_sample_rate']}/{source['grammar_gate_sample_rate']}`",
        "",
        "## Candidates",
        "",
        f"- candidate count: `{package['candidate_count']}`",
        f"- failed candidate count: `{package['failed_candidate_count']}`",
        f"- midi dir: `{package['midi_dir']}`",
        "",
    ]
    for candidate in package["candidates"]:
        lines.append(
            "- "
            f"rank `{candidate['review_rank']}` "
            f"seed `{candidate['sample_seed']}` "
            f"sample `{candidate['sample_index']}` "
            f"dead_air `{candidate['dead_air_ratio']}` "
            f"coverage `{candidate['phrase_coverage_ratio']}` "
            f"chord_tone `{candidate['chord_tone_ratio']}` "
            f"midi `{candidate['package_midi_path']}`"
        )
    lines.extend(["", "## Failed Rows", ""])
    for row in package["failed_rows"]:
        lines.append(
            f"- seed `{row['sample_seed']}` sample `{row['sample_index']}` reason `{row['failure_reason']}`"
        )
    lines.extend(["", "## Not Proven", ""])
    for item in report["not_proven"]:
        lines.append(f"- `{item}`")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Stage B generic tiny checkpoint repair review package")
    parser.add_argument(
        "--repeatability_report",
        type=str,
        default="outputs/stage_b_generic_tiny_checkpoint_repair_repeatability/"
        "harness_stage_b_generic_tiny_checkpoint_repair_repeatability/"
        "stage_b_generic_tiny_checkpoint_repair_repeatability.json",
    )
    parser.add_argument("--output_root", type=str, default="outputs/stage_b_generic_tiny_checkpoint_repair_review_package")
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--issue_number", type=int, default=399)
    parser.add_argument("--min_candidate_count", type=int, default=5)
    parser.add_argument("--expected_boundary", type=str, default="")
    parser.add_argument("--require_review_package_ready", action="store_true")
    parser.add_argument("--require_no_musical_quality_claim", action="store_true")
    parser.add_argument("--require_no_broad_quality_claim", action="store_true")
    parser.add_argument("--require_no_brad_style_claim", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    repeatability_report_path = Path(args.repeatability_report)
    if not repeatability_report_path.exists():
        raise StageBGenericTinyCheckpointRepairReviewPackageError("repeatability report required")
    repeatability_report = read_json(repeatability_report_path)
    generation_report_path = Path(str(_dict(repeatability_report.get("generation")).get("report_path") or ""))
    if not generation_report_path.exists():
        raise StageBGenericTinyCheckpointRepairReviewPackageError("generation report required")
    generation_report = read_json(generation_report_path)

    strict_rows = [row for row in generation_report.get("samples", []) if row.get("strict_valid")]
    failed_rows = [
        build_candidate_row(row)
        for row in generation_report.get("samples", [])
        if not row.get("strict_valid")
    ]
    candidates = sort_candidates([build_candidate_row(row) for row in strict_rows])
    packaged_candidates = copy_candidate_midis(candidates, run_dir / "midi")
    report = build_review_package_report(
        run_dir=run_dir,
        repeatability_report_path=repeatability_report_path,
        repeatability_report=repeatability_report,
        generation_report=generation_report,
        candidates=packaged_candidates,
        failed_rows=failed_rows,
        args=args,
    )
    summary = validate_review_package_report(
        report,
        expected_boundary=str(args.expected_boundary or ""),
        require_review_package_ready=bool(args.require_review_package_ready),
        require_no_musical_quality_claim=bool(args.require_no_musical_quality_claim),
        require_no_broad_quality_claim=bool(args.require_no_broad_quality_claim),
        require_no_brad_style_claim=bool(args.require_no_brad_style_claim),
    )
    write_json(run_dir / "stage_b_generic_tiny_checkpoint_repair_review_package.json", report)
    write_json(run_dir / "stage_b_generic_tiny_checkpoint_repair_review_package_validation_summary.json", summary)
    markdown = markdown_report(report)
    write_text(run_dir / "stage_b_generic_tiny_checkpoint_repair_review_package.md", markdown)
    if args.doc_path:
        write_text(Path(args.doc_path), markdown)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
