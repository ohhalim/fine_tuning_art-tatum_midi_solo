"""Record the residual-aware listening review pending boundary."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from scripts.assess_stage_b_generic_base_readiness import write_json, write_text  # noqa: E402


SCHEMA_VERSION = "music_transformer_solo_yield_residual_aware_listening_review_pending_v1"
HANDOFF_FREEZE_SCHEMA_VERSION = "music_transformer_solo_yield_residual_aware_mvp_handoff_freeze_v1"
REVIEW_INPUT_SCHEMA_VERSION = "music_transformer_solo_yield_residual_aware_review_input_v1"
BOUNDARY = "music_transformer_solo_yield_residual_aware_listening_review_pending"
NEXT_BOUNDARY = "music_transformer_solo_yield_residual_aware_completion_audit"


class SoloYieldResidualAwareListeningReviewPendingError(ValueError):
    pass


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _bool_token(value: Any) -> str:
    return "true" if bool(value) else "false"


def _format_counts(value: dict[str, Any]) -> str:
    if not value:
        return "none"
    return ", ".join(f"{key}={value[key]}" for key in sorted(value))


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SoloYieldResidualAwareListeningReviewPendingError(f"json not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _require_no_quality_claim(container: dict[str, Any], *, label: str) -> None:
    claimed = [
        key
        for key in (
            "human_audio_preference_claimed",
            "midi_to_solo_musical_quality_claimed",
            "musical_quality_claimed",
            "artist_style_claimed",
            "production_ready_claimed",
        )
        if bool(container.get(key, False))
    ]
    if claimed:
        raise SoloYieldResidualAwareListeningReviewPendingError(
            f"unexpected quality claim in {label}: {claimed}"
        )


def validate_handoff_freeze(report: dict[str, Any]) -> dict[str, Any]:
    if str(report.get("schema_version") or "") != HANDOFF_FREEZE_SCHEMA_VERSION:
        raise SoloYieldResidualAwareListeningReviewPendingError("handoff freeze schema mismatch")
    aggregate = _dict(report.get("aggregate"))
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    artifact_paths = _dict(report.get("artifact_paths"))
    if not bool(readiness.get("residual_aware_mvp_handoff_freeze_completed", False)):
        raise SoloYieldResidualAwareListeningReviewPendingError("handoff freeze completion required")
    if not bool(readiness.get("local_candidate_artifacts_verified", False)):
        raise SoloYieldResidualAwareListeningReviewPendingError("local candidate artifacts verification required")
    if not bool(readiness.get("review_input_template_available", False)):
        raise SoloYieldResidualAwareListeningReviewPendingError("review input template availability required")
    if bool(aggregate.get("raw_artifact_upload_required", True)):
        raise SoloYieldResidualAwareListeningReviewPendingError("raw artifact upload should not be required")
    if _int(aggregate.get("missing_file_count")) != 0:
        raise SoloYieldResidualAwareListeningReviewPendingError("missing file count must be zero")
    if _int(aggregate.get("checksum_mismatch_count")) != 0:
        raise SoloYieldResidualAwareListeningReviewPendingError("checksum mismatch count must be zero")
    if str(decision.get("next_boundary") or "") != BOUNDARY:
        raise SoloYieldResidualAwareListeningReviewPendingError("handoff freeze must route to pending boundary")
    if bool(decision.get("critical_user_input_required", True)):
        raise SoloYieldResidualAwareListeningReviewPendingError("critical user input should not be required")
    _require_no_quality_claim(readiness, label="handoff freeze readiness")
    template_path = Path(str(artifact_paths.get("review_input_template_json") or ""))
    if not template_path.exists() or not template_path.is_file():
        raise SoloYieldResidualAwareListeningReviewPendingError(
            f"review input template missing: {template_path}"
        )
    return {
        "candidate_count": _int(aggregate.get("candidate_count")),
        "midi_count": _int(aggregate.get("midi_count")),
        "wav_count": _int(aggregate.get("wav_count")),
        "quality_proxy_pass_count": _int(aggregate.get("quality_proxy_pass_count")),
        "quality_proxy_fail_count": _int(aggregate.get("quality_proxy_fail_count")),
        "major_label_counts": _dict(aggregate.get("major_label_counts")),
        "watch_label_counts": _dict(aggregate.get("watch_label_counts")),
        "review_input_template_path": str(template_path),
        "validated_listening_input_present": bool(
            readiness.get("validated_listening_input_present", True)
        ),
        "preference_fill_allowed": bool(readiness.get("preference_fill_allowed", True)),
        "listening_review_completed": bool(readiness.get("listening_review_completed", True)),
    }


def validate_review_input_template(path: Path, *, expected_candidate_count: int) -> dict[str, Any]:
    report = read_json(path)
    if str(report.get("schema_version") or "") != REVIEW_INPUT_SCHEMA_VERSION:
        raise SoloYieldResidualAwareListeningReviewPendingError("review input schema mismatch")
    candidates = _list(report.get("candidates"))
    if len(candidates) != expected_candidate_count:
        raise SoloYieldResidualAwareListeningReviewPendingError("review input candidate count mismatch")
    if str(report.get("review_status") or "") != "pending":
        raise SoloYieldResidualAwareListeningReviewPendingError("review status must remain pending")
    if str(report.get("overall_decision") or "") != "pending":
        raise SoloYieldResidualAwareListeningReviewPendingError("overall decision must remain pending")
    filled = [
        row
        for row in candidates
        if str(_dict(row).get("decision") or "") != "pending"
        or _dict(row).get("usable_as_jazz_solo_phrase") is not None
        or _dict(row).get("primary_failure") is not None
    ]
    if filled:
        raise SoloYieldResidualAwareListeningReviewPendingError("review input should not be filled")
    return {
        "schema_version": str(report.get("schema_version")),
        "review_status": str(report.get("review_status")),
        "overall_decision": str(report.get("overall_decision")),
        "candidate_count": len(candidates),
        "pending_candidate_count": len(candidates),
    }


def build_pending_report(
    *,
    handoff_freeze_report: dict[str, Any],
    output_dir: Path,
    issue_number: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    handoff = validate_handoff_freeze(handoff_freeze_report)
    review_input = validate_review_input_template(
        Path(str(handoff["review_input_template_path"])),
        expected_candidate_count=int(handoff["candidate_count"]),
    )
    pending_input = not bool(handoff["validated_listening_input_present"])
    preference_blocked = not bool(handoff["preference_fill_allowed"])
    listening_pending = not bool(handoff["listening_review_completed"])
    report = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "issue_number": int(issue_number),
        "source_reports": {
            "handoff_freeze_report": handoff_freeze_report.get("output_dir"),
        },
        "aggregate": {
            "candidate_count": handoff["candidate_count"],
            "midi_count": handoff["midi_count"],
            "wav_count": handoff["wav_count"],
            "quality_proxy_pass_count": handoff["quality_proxy_pass_count"],
            "quality_proxy_fail_count": handoff["quality_proxy_fail_count"],
            "major_label_counts": handoff["major_label_counts"],
            "watch_label_counts": handoff["watch_label_counts"],
            "pending_candidate_count": review_input["pending_candidate_count"],
        },
        "review_input": {
            "path": handoff["review_input_template_path"],
            "schema_version": review_input["schema_version"],
            "review_status": review_input["review_status"],
            "overall_decision": review_input["overall_decision"],
            "candidate_count": review_input["candidate_count"],
            "pending_candidate_count": review_input["pending_candidate_count"],
        },
        "readiness": {
            "residual_aware_listening_review_pending_recorded": True,
            "local_mvp_handoff_ready": True,
            "review_input_template_pending": True,
            "manual_review_required_for_quality_claim": True,
            "validated_listening_input_present": pending_input is False,
            "preference_fill_allowed": preference_blocked is False,
            "listening_review_completed": listening_pending is False,
            "human_audio_preference_claimed": False,
            "midi_to_solo_musical_quality_claimed": False,
            "musical_quality_claimed": False,
            "artist_style_claimed": False,
            "production_ready_claimed": False,
        },
        "decision": {
            "current_boundary": BOUNDARY,
            "next_boundary": NEXT_BOUNDARY,
            "critical_user_input_required": False,
            "reason": "listening review input remains pending; technical completion audit can proceed without quality claim",
        },
        "not_proven": [
            "listening_review_completed",
            "human_audio_preference",
            "midi_to_solo_musical_quality",
            "stable_jazz_solo_quality",
            "production_ready_improviser",
        ],
    }
    write_json(output_dir / "residual_aware_listening_review_pending.json", report)
    write_json(
        output_dir / "residual_aware_listening_review_pending_summary.json",
        validate_pending_report(report),
    )
    write_text(output_dir / "residual_aware_listening_review_pending.md", markdown_report(report))
    return report


def validate_pending_report(
    report: dict[str, Any],
    *,
    expected_next_boundary: str | None = None,
    require_pending_review: bool = False,
    require_no_quality_claim: bool = False,
) -> dict[str, Any]:
    if str(report.get("schema_version") or "") != SCHEMA_VERSION:
        raise SoloYieldResidualAwareListeningReviewPendingError("pending report schema mismatch")
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    aggregate = _dict(report.get("aggregate"))
    if not bool(readiness.get("residual_aware_listening_review_pending_recorded", False)):
        raise SoloYieldResidualAwareListeningReviewPendingError("pending boundary record required")
    if expected_next_boundary and str(decision.get("next_boundary") or "") != expected_next_boundary:
        raise SoloYieldResidualAwareListeningReviewPendingError("unexpected next boundary")
    if require_pending_review:
        if not bool(readiness.get("review_input_template_pending", False)):
            raise SoloYieldResidualAwareListeningReviewPendingError("review input template must be pending")
        if not bool(readiness.get("manual_review_required_for_quality_claim", False)):
            raise SoloYieldResidualAwareListeningReviewPendingError("manual review requirement should be recorded")
        if bool(readiness.get("validated_listening_input_present", True)):
            raise SoloYieldResidualAwareListeningReviewPendingError("validated listening input should remain pending")
        if bool(readiness.get("preference_fill_allowed", True)):
            raise SoloYieldResidualAwareListeningReviewPendingError("preference fill should remain blocked")
        if bool(readiness.get("listening_review_completed", True)):
            raise SoloYieldResidualAwareListeningReviewPendingError("listening review should remain pending")
    if require_no_quality_claim:
        _require_no_quality_claim(readiness, label="pending readiness")
    if bool(decision.get("critical_user_input_required", True)):
        raise SoloYieldResidualAwareListeningReviewPendingError("critical user input should not be required")
    return {
        "schema_version": str(report.get("schema_version")),
        "local_mvp_handoff_ready": bool(readiness.get("local_mvp_handoff_ready", False)),
        "review_input_template_pending": bool(readiness.get("review_input_template_pending", False)),
        "manual_review_required_for_quality_claim": bool(
            readiness.get("manual_review_required_for_quality_claim", False)
        ),
        "candidate_count": _int(aggregate.get("candidate_count")),
        "midi_count": _int(aggregate.get("midi_count")),
        "wav_count": _int(aggregate.get("wav_count")),
        "pending_candidate_count": _int(aggregate.get("pending_candidate_count")),
        "validated_listening_input_present": bool(
            readiness.get("validated_listening_input_present", True)
        ),
        "preference_fill_allowed": bool(readiness.get("preference_fill_allowed", True)),
        "listening_review_completed": bool(readiness.get("listening_review_completed", True)),
        "midi_to_solo_musical_quality_claimed": bool(
            readiness.get("midi_to_solo_musical_quality_claimed", True)
        ),
        "next_boundary": str(decision.get("next_boundary") or ""),
        "critical_user_input_required": bool(decision.get("critical_user_input_required", True)),
    }


def markdown_report(report: dict[str, Any]) -> str:
    aggregate = report["aggregate"]
    readiness = report["readiness"]
    decision = report["decision"]
    review_input = report["review_input"]
    lines = [
        "# Music Transformer Solo Yield Residual-Aware Listening Review Pending",
        "",
        "## Summary",
        "",
        f"- issue: `#{report['issue_number']}`",
        f"- candidate count: `{aggregate['candidate_count']}`",
        f"- MIDI/WAV: `{aggregate['midi_count']}` / `{aggregate['wav_count']}`",
        (
            "- quality proxy pass/fail: "
            f"`{aggregate['quality_proxy_pass_count']}` / `{aggregate['quality_proxy_fail_count']}`"
        ),
        f"- major labels: `{_format_counts(aggregate['major_label_counts'])}`",
        f"- watch labels: `{_format_counts(aggregate['watch_label_counts'])}`",
        f"- review input template: `{review_input['path']}`",
        f"- review status: `{review_input['review_status']}`",
        f"- pending candidate count: `{aggregate['pending_candidate_count']}`",
        (
            "- manual review required for quality claim: "
            f"`{_bool_token(readiness['manual_review_required_for_quality_claim'])}`"
        ),
        f"- validated listening input present: `{_bool_token(readiness['validated_listening_input_present'])}`",
        f"- preference fill allowed: `{_bool_token(readiness['preference_fill_allowed'])}`",
        f"- listening review completed: `{_bool_token(readiness['listening_review_completed'])}`",
        f"- musical quality claimed: `{_bool_token(readiness['midi_to_solo_musical_quality_claimed'])}`",
        f"- next boundary: `{decision['next_boundary']}`",
        "",
        "## Not Proven",
        "",
    ]
    for item in report.get("not_proven", []):
        lines.append(f"- `{item}`")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Record residual-aware listening review pending boundary")
    parser.add_argument("--handoff_freeze_report", type=str, required=True)
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/music_transformer_finetune_mvp/solo_yield_residual_aware_listening_review_pending",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--issue_number", type=int, default=0)
    parser.add_argument("--expected_next_boundary", type=str, default=NEXT_BOUNDARY)
    parser.add_argument("--require_pending_review", action="store_true")
    parser.add_argument("--require_no_quality_claim", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    report = build_pending_report(
        handoff_freeze_report=read_json(Path(args.handoff_freeze_report)),
        output_dir=output_dir,
        issue_number=int(args.issue_number),
    )
    summary = validate_pending_report(
        report,
        expected_next_boundary=str(args.expected_next_boundary),
        require_pending_review=bool(args.require_pending_review),
        require_no_quality_claim=bool(args.require_no_quality_claim),
    )
    if args.doc_path:
        write_text(Path(args.doc_path), markdown_report(report))
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
