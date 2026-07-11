"""Decide range/interval guard targets after phrase-continuation MIDI note failure."""

from __future__ import annotations

import argparse
import json
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


class StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardDecisionError(ValueError):
    pass


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def validate_failure_review(report: dict[str, Any]) -> dict[str, Any]:
    review = _dict(report.get("user_listening_review"))
    failure = _dict(report.get("midi_note_failure"))
    claim = _dict(report.get("claim_boundary"))
    decision = _dict(report.get("decision"))
    if str(claim.get("boundary") or "") != "generic_tiny_checkpoint_repair_phrase_continuation_midi_note_failure_reject_all":
        raise StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardDecisionError(
            "unexpected MIDI note failure boundary"
        )
    if str(review.get("overall_decision") or "") != "reject_all":
        raise StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardDecisionError(
            "reject_all decision required"
        )
    if str(review.get("primary_failure") or "") != "midi_note_random_large_leaps":
        raise StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardDecisionError(
            "midi_note_random_large_leaps failure required"
        )
    if not bool(failure.get("all_reviewed_candidates_failed", False)):
        raise StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardDecisionError(
            "all reviewed candidates must fail note audit"
        )
    if bool(claim.get("human_audio_keep_claimed", True)):
        raise StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardDecisionError(
            "human/audio keep must not be claimed"
        )
    if bool(claim.get("musical_quality_claimed", True)):
        raise StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardDecisionError(
            "musical quality must not be claimed"
        )
    if str(decision.get("next_boundary") or "") != (
        "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_decision"
    ):
        raise StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardDecisionError(
            "unexpected next boundary"
        )
    reviewed = _list(report.get("reviewed_audio_files"))
    if not reviewed:
        raise StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardDecisionError(
            "reviewed MIDI evidence required"
        )
    return _dict(_dict(reviewed[0]).get("midi_note_audit"))


def build_guard_decision(
    failure_review: dict[str, Any],
    *,
    output_dir: Path,
    target_max_pitch_span: int,
    target_max_abs_interval: int,
    target_max_large_interval_ratio: float,
    target_max_severe_interval_count: int,
) -> dict[str, Any]:
    audit = validate_failure_review(failure_review)
    observed = {
        "note_count": _int(audit.get("note_count")),
        "pitch_min": audit.get("pitch_min"),
        "pitch_max": audit.get("pitch_max"),
        "pitch_span": _int(audit.get("pitch_span")),
        "max_abs_interval": _int(audit.get("max_abs_interval")),
        "large_interval_ratio": _float(audit.get("large_interval_ratio")),
        "severe_interval_count": _int(audit.get("severe_interval_count")),
        "intervals": _list(audit.get("intervals")),
        "pitch_name_sequence": _list(audit.get("pitch_name_sequence")),
    }
    targets = {
        "max_pitch_span": int(target_max_pitch_span),
        "max_abs_interval": int(target_max_abs_interval),
        "max_large_interval_ratio": float(target_max_large_interval_ratio),
        "max_severe_interval_count": int(target_max_severe_interval_count),
        "preferred_pitch_floor": 48,
        "preferred_pitch_ceiling": 84,
        "large_interval_threshold": 12,
        "severe_interval_threshold": 24,
    }
    return {
        "schema_version": "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_decision_v1",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "source_failure_review_schema": str(failure_review.get("schema_version") or ""),
        "observed_failure": observed,
        "guard_targets": targets,
        "repair_targets": [
            "filter_pitch_candidates_to_preferred_solo_range",
            "reject_or_repair_adjacent_interval_above_target",
            "penalize_large_register_jumps_during_candidate_ranking",
            "require_small_leap_or_stepwise_support_ratio",
            "fail_audio_package_when_range_interval_guard_fails",
        ],
        "readiness": {
            "boundary": "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_decision",
            "range_interval_guard_decision_recorded": True,
            "musical_quality_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
        },
        "decision": {
            "current_boundary": "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_decision",
            "next_boundary": "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sweep",
            "auto_progress_allowed": True,
            "critical_user_input_required": False,
            "reason": "MIDI note audit failure maps to objective range/interval guard repair",
        },
        "not_proven": [
            "repaired_candidate_exists",
            "audio_rendered_quality",
            "musical_quality",
            "broad_trained_model_quality",
            "brad_style_adaptation",
        ],
        "next_recommended_issue": "Stage B generic tiny checkpoint repair phrase continuation range interval guard sweep",
    }


def validate_guard_decision(
    report: dict[str, Any],
    *,
    expected_boundary: str | None,
    expected_next_boundary: str | None,
    require_no_quality_claim: bool,
) -> dict[str, Any]:
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    observed = _dict(report.get("observed_failure"))
    targets = _dict(report.get("guard_targets"))
    boundary = str(readiness.get("boundary") or "")
    next_boundary = str(decision.get("next_boundary") or "")
    if expected_boundary and boundary != expected_boundary:
        raise StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardDecisionError(
            f"expected boundary {expected_boundary}, got {boundary}"
        )
    if expected_next_boundary and next_boundary != expected_next_boundary:
        raise StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardDecisionError(
            f"expected next boundary {expected_next_boundary}, got {next_boundary}"
        )
    if _int(observed.get("max_abs_interval")) <= _int(targets.get("max_abs_interval")):
        raise StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardDecisionError(
            "observed max interval must exceed target"
        )
    if _int(observed.get("pitch_span")) <= _int(targets.get("max_pitch_span")):
        raise StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardDecisionError(
            "observed pitch span must exceed target"
        )
    if require_no_quality_claim:
        claimed = [
            bool(readiness.get("musical_quality_claimed", True)),
            bool(readiness.get("broad_trained_model_quality_claimed", True)),
            bool(readiness.get("brad_style_adaptation_claimed", True)),
        ]
        if any(claimed):
            raise StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardDecisionError(
                "quality claims must not be set"
            )
    return {
        "boundary": boundary,
        "next_boundary": next_boundary,
        "observed_pitch_span": _int(observed.get("pitch_span")),
        "observed_max_abs_interval": _int(observed.get("max_abs_interval")),
        "observed_large_interval_ratio": _float(observed.get("large_interval_ratio")),
        "target_max_pitch_span": _int(targets.get("max_pitch_span")),
        "target_max_abs_interval": _int(targets.get("max_abs_interval")),
        "target_max_large_interval_ratio": _float(targets.get("max_large_interval_ratio")),
        "repair_target_count": len(_list(report.get("repair_targets"))),
        "auto_progress_allowed": bool(decision.get("auto_progress_allowed", False)),
        "critical_user_input_required": bool(decision.get("critical_user_input_required", True)),
        "musical_quality_claimed": bool(readiness.get("musical_quality_claimed", True)),
        "next_recommended_issue": str(report.get("next_recommended_issue") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    readiness = report["readiness"]
    decision = report["decision"]
    observed = report["observed_failure"]
    targets = report["guard_targets"]
    lines = [
        "# Stage B Generic Tiny Checkpoint Repair Phrase Continuation Range Interval Guard Decision",
        "",
        "## Summary",
        "",
        f"- boundary: `{readiness['boundary']}`",
        f"- next boundary: `{decision['next_boundary']}`",
        f"- auto progress allowed: `{_bool_token(decision['auto_progress_allowed'])}`",
        f"- musical quality claimed: `{_bool_token(readiness['musical_quality_claimed'])}`",
        "",
        "## Observed Failure",
        "",
        f"- note count: `{observed['note_count']}`",
        f"- pitch range: `{observed['pitch_min']}-{observed['pitch_max']}`",
        f"- pitch span: `{observed['pitch_span']}`",
        f"- max abs interval: `{observed['max_abs_interval']}`",
        f"- large interval ratio: `{observed['large_interval_ratio']}`",
        f"- severe interval count: `{observed['severe_interval_count']}`",
        f"- intervals: `{observed['intervals']}`",
        "",
        "## Guard Targets",
        "",
        f"- max pitch span: `{targets['max_pitch_span']}`",
        f"- max abs interval: `{targets['max_abs_interval']}`",
        f"- max large interval ratio: `{targets['max_large_interval_ratio']}`",
        f"- max severe interval count: `{targets['max_severe_interval_count']}`",
        f"- preferred pitch range: `{targets['preferred_pitch_floor']}-{targets['preferred_pitch_ceiling']}`",
        "",
        "## Repair Targets",
        "",
    ]
    for item in report.get("repair_targets", []):
        lines.append(f"- `{item}`")
    lines.extend(["", "## Not Proven", ""])
    for item in report.get("not_proven", []):
        lines.append(f"- `{item}`")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Decide phrase-continuation range/interval guard targets")
    parser.add_argument(
        "--failure_review_report",
        type=str,
        default="outputs/stage_b_generic_tiny_checkpoint_repair_phrase_continuation_midi_note_failure_review/"
        "harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_midi_note_failure_review/"
        "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_midi_note_failure_review.json",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_decision",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--target_max_pitch_span", type=int, default=24)
    parser.add_argument("--target_max_abs_interval", type=int, default=12)
    parser.add_argument("--target_max_large_interval_ratio", type=float, default=0.35)
    parser.add_argument("--target_max_severe_interval_count", type=int, default=0)
    parser.add_argument("--expected_boundary", type=str, default="")
    parser.add_argument("--expected_next_boundary", type=str, default="")
    parser.add_argument("--require_no_quality_claim", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    failure_review_path = Path(args.failure_review_report)
    if not failure_review_path.exists():
        raise StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardDecisionError(
            "failure review report required"
        )
    report = build_guard_decision(
        read_json(failure_review_path),
        output_dir=output_dir,
        target_max_pitch_span=int(args.target_max_pitch_span),
        target_max_abs_interval=int(args.target_max_abs_interval),
        target_max_large_interval_ratio=float(args.target_max_large_interval_ratio),
        target_max_severe_interval_count=int(args.target_max_severe_interval_count),
    )
    summary = validate_guard_decision(
        report,
        expected_boundary=str(args.expected_boundary or ""),
        expected_next_boundary=str(args.expected_next_boundary or ""),
        require_no_quality_claim=bool(args.require_no_quality_claim),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        output_dir / "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_decision.json",
        report,
    )
    write_json(
        output_dir
        / "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_decision_validation_summary.json",
        summary,
    )
    markdown = markdown_report(report)
    write_text(output_dir / "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_decision.md", markdown)
    if args.doc_path:
        write_text(Path(args.doc_path), markdown)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
