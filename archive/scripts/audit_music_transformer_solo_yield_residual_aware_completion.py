"""Audit residual-aware technical MVP completion for solo-yield candidates."""

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


SCHEMA_VERSION = "music_transformer_solo_yield_residual_aware_completion_audit_v1"
PENDING_SCHEMA_VERSION = "music_transformer_solo_yield_residual_aware_listening_review_pending_v1"
HANDOFF_SCHEMA_VERSION = "music_transformer_solo_yield_residual_aware_mvp_handoff_freeze_v1"
BOUNDARY = "music_transformer_solo_yield_residual_aware_completion_audit"
NEXT_BOUNDARY = "music_transformer_solo_yield_residual_aware_final_status_sync"


class SoloYieldResidualAwareCompletionAuditError(ValueError):
    pass


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


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
        raise SoloYieldResidualAwareCompletionAuditError(f"json not found: {path}")
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
        raise SoloYieldResidualAwareCompletionAuditError(
            f"unexpected quality claim in {label}: {claimed}"
        )


def validate_pending_report(report: dict[str, Any]) -> dict[str, Any]:
    if str(report.get("schema_version") or "") != PENDING_SCHEMA_VERSION:
        raise SoloYieldResidualAwareCompletionAuditError("pending report schema mismatch")
    aggregate = _dict(report.get("aggregate"))
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    review_input = _dict(report.get("review_input"))
    if not bool(readiness.get("residual_aware_listening_review_pending_recorded", False)):
        raise SoloYieldResidualAwareCompletionAuditError("pending boundary record required")
    if not bool(readiness.get("local_mvp_handoff_ready", False)):
        raise SoloYieldResidualAwareCompletionAuditError("local MVP handoff readiness required")
    if not bool(readiness.get("review_input_template_pending", False)):
        raise SoloYieldResidualAwareCompletionAuditError("review input template pending required")
    if not bool(readiness.get("manual_review_required_for_quality_claim", False)):
        raise SoloYieldResidualAwareCompletionAuditError("manual review requirement required")
    if str(decision.get("next_boundary") or "") != BOUNDARY:
        raise SoloYieldResidualAwareCompletionAuditError("pending report must route to completion audit")
    if bool(decision.get("critical_user_input_required", True)):
        raise SoloYieldResidualAwareCompletionAuditError("critical user input should not be required")
    if str(review_input.get("review_status") or "") != "pending":
        raise SoloYieldResidualAwareCompletionAuditError("review status must remain pending")
    if _int(aggregate.get("pending_candidate_count")) != _int(aggregate.get("candidate_count")):
        raise SoloYieldResidualAwareCompletionAuditError("pending candidate count mismatch")
    _require_no_quality_claim(readiness, label="pending readiness")
    return {
        "candidate_count": _int(aggregate.get("candidate_count")),
        "midi_count": _int(aggregate.get("midi_count")),
        "wav_count": _int(aggregate.get("wav_count")),
        "quality_proxy_pass_count": _int(aggregate.get("quality_proxy_pass_count")),
        "quality_proxy_fail_count": _int(aggregate.get("quality_proxy_fail_count")),
        "major_label_counts": _dict(aggregate.get("major_label_counts")),
        "watch_label_counts": _dict(aggregate.get("watch_label_counts")),
        "pending_candidate_count": _int(aggregate.get("pending_candidate_count")),
        "review_input_path": str(review_input.get("path") or ""),
    }


def validate_handoff_freeze(report: dict[str, Any]) -> dict[str, Any]:
    if str(report.get("schema_version") or "") != HANDOFF_SCHEMA_VERSION:
        raise SoloYieldResidualAwareCompletionAuditError("handoff freeze schema mismatch")
    aggregate = _dict(report.get("aggregate"))
    readiness = _dict(report.get("readiness"))
    if not bool(readiness.get("residual_aware_mvp_handoff_freeze_completed", False)):
        raise SoloYieldResidualAwareCompletionAuditError("handoff freeze completion required")
    if not bool(readiness.get("local_candidate_artifacts_verified", False)):
        raise SoloYieldResidualAwareCompletionAuditError("local candidate artifacts verification required")
    if _int(aggregate.get("missing_file_count")) != 0:
        raise SoloYieldResidualAwareCompletionAuditError("missing file count must be zero")
    if _int(aggregate.get("checksum_mismatch_count")) != 0:
        raise SoloYieldResidualAwareCompletionAuditError("checksum mismatch count must be zero")
    if bool(aggregate.get("raw_artifact_upload_required", True)):
        raise SoloYieldResidualAwareCompletionAuditError("raw artifact upload should not be required")
    _require_no_quality_claim(readiness, label="handoff readiness")
    return {
        "candidate_count": _int(aggregate.get("candidate_count")),
        "midi_count": _int(aggregate.get("midi_count")),
        "wav_count": _int(aggregate.get("wav_count")),
        "missing_file_count": _int(aggregate.get("missing_file_count")),
        "checksum_mismatch_count": _int(aggregate.get("checksum_mismatch_count")),
        "raw_artifact_upload_required": bool(aggregate.get("raw_artifact_upload_required", True)),
        "local_candidate_artifacts_verified": bool(
            readiness.get("local_candidate_artifacts_verified", False)
        ),
    }


def build_completion_audit_report(
    *,
    pending_report: dict[str, Any],
    handoff_freeze_report: dict[str, Any],
    output_dir: Path,
    issue_number: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pending = validate_pending_report(pending_report)
    handoff = validate_handoff_freeze(handoff_freeze_report)
    counts_match = bool(
        pending["candidate_count"] == handoff["candidate_count"]
        and pending["midi_count"] == handoff["midi_count"]
        and pending["wav_count"] == handoff["wav_count"]
    )
    technical_complete = bool(
        counts_match
        and handoff["local_candidate_artifacts_verified"]
        and pending["pending_candidate_count"] == pending["candidate_count"]
        and handoff["missing_file_count"] == 0
        and handoff["checksum_mismatch_count"] == 0
        and not handoff["raw_artifact_upload_required"]
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "issue_number": int(issue_number),
        "source_reports": {
            "pending_report": pending_report.get("output_dir"),
            "handoff_freeze_report": handoff_freeze_report.get("output_dir"),
        },
        "aggregate": {
            "candidate_count": pending["candidate_count"],
            "midi_count": pending["midi_count"],
            "wav_count": pending["wav_count"],
            "pending_candidate_count": pending["pending_candidate_count"],
            "quality_proxy_pass_count": pending["quality_proxy_pass_count"],
            "quality_proxy_fail_count": pending["quality_proxy_fail_count"],
            "major_label_counts": pending["major_label_counts"],
            "watch_label_counts": pending["watch_label_counts"],
            "missing_file_count": handoff["missing_file_count"],
            "checksum_mismatch_count": handoff["checksum_mismatch_count"],
            "raw_artifact_upload_required": handoff["raw_artifact_upload_required"],
            "counts_match": counts_match,
        },
        "completion": {
            "technical_mvp_complete": technical_complete,
            "local_review_ready": technical_complete,
            "review_input_path": pending["review_input_path"],
            "completion_scope": "input MIDI/chord context to ranked solo MIDI/WAV review candidates",
            "quality_scope": "not a musical quality or artist-style claim",
        },
        "readiness": {
            "residual_aware_completion_audit_completed": True,
            "technical_mvp_complete": technical_complete,
            "local_review_ready": technical_complete,
            "review_input_template_pending": True,
            "manual_review_required_for_quality_claim": True,
            "validated_listening_input_present": False,
            "preference_fill_allowed": False,
            "listening_review_completed": False,
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
            "reason": "technical MVP completion audited; final status sync can record completion without quality claim",
        },
        "not_proven": [
            "listening_review_completed",
            "human_audio_preference",
            "midi_to_solo_musical_quality",
            "stable_jazz_solo_quality",
            "production_ready_improviser",
        ],
    }
    write_json(output_dir / "residual_aware_completion_audit.json", report)
    write_json(
        output_dir / "residual_aware_completion_audit_summary.json",
        validate_completion_audit_report(report),
    )
    write_text(output_dir / "residual_aware_completion_audit.md", markdown_report(report))
    return report


def validate_completion_audit_report(
    report: dict[str, Any],
    *,
    expected_next_boundary: str | None = None,
    require_technical_complete: bool = False,
    require_no_quality_claim: bool = False,
) -> dict[str, Any]:
    if str(report.get("schema_version") or "") != SCHEMA_VERSION:
        raise SoloYieldResidualAwareCompletionAuditError("completion audit schema mismatch")
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    aggregate = _dict(report.get("aggregate"))
    if not bool(readiness.get("residual_aware_completion_audit_completed", False)):
        raise SoloYieldResidualAwareCompletionAuditError("completion audit record required")
    if expected_next_boundary and str(decision.get("next_boundary") or "") != expected_next_boundary:
        raise SoloYieldResidualAwareCompletionAuditError("unexpected next boundary")
    if require_technical_complete and not bool(readiness.get("technical_mvp_complete", False)):
        raise SoloYieldResidualAwareCompletionAuditError("technical MVP completion required")
    if require_no_quality_claim:
        _require_no_quality_claim(readiness, label="completion readiness")
    if bool(decision.get("critical_user_input_required", True)):
        raise SoloYieldResidualAwareCompletionAuditError("critical user input should not be required")
    return {
        "schema_version": str(report.get("schema_version")),
        "technical_mvp_complete": bool(readiness.get("technical_mvp_complete", False)),
        "local_review_ready": bool(readiness.get("local_review_ready", False)),
        "candidate_count": _int(aggregate.get("candidate_count")),
        "midi_count": _int(aggregate.get("midi_count")),
        "wav_count": _int(aggregate.get("wav_count")),
        "pending_candidate_count": _int(aggregate.get("pending_candidate_count")),
        "quality_proxy_pass_count": _int(aggregate.get("quality_proxy_pass_count")),
        "quality_proxy_fail_count": _int(aggregate.get("quality_proxy_fail_count")),
        "missing_file_count": _int(aggregate.get("missing_file_count")),
        "checksum_mismatch_count": _int(aggregate.get("checksum_mismatch_count")),
        "raw_artifact_upload_required": bool(aggregate.get("raw_artifact_upload_required", True)),
        "manual_review_required_for_quality_claim": bool(
            readiness.get("manual_review_required_for_quality_claim", False)
        ),
        "midi_to_solo_musical_quality_claimed": bool(
            readiness.get("midi_to_solo_musical_quality_claimed", True)
        ),
        "next_boundary": str(decision.get("next_boundary") or ""),
        "critical_user_input_required": bool(decision.get("critical_user_input_required", True)),
    }


def markdown_report(report: dict[str, Any]) -> str:
    aggregate = report["aggregate"]
    completion = report["completion"]
    readiness = report["readiness"]
    decision = report["decision"]
    lines = [
        "# Music Transformer Solo Yield Residual-Aware Completion Audit",
        "",
        "## Summary",
        "",
        f"- issue: `#{report['issue_number']}`",
        f"- technical MVP complete: `{_bool_token(completion['technical_mvp_complete'])}`",
        f"- local review ready: `{_bool_token(completion['local_review_ready'])}`",
        f"- candidate count: `{aggregate['candidate_count']}`",
        f"- MIDI/WAV: `{aggregate['midi_count']}` / `{aggregate['wav_count']}`",
        (
            "- quality proxy pass/fail: "
            f"`{aggregate['quality_proxy_pass_count']}` / `{aggregate['quality_proxy_fail_count']}`"
        ),
        f"- pending candidate count: `{aggregate['pending_candidate_count']}`",
        f"- major labels: `{_format_counts(aggregate['major_label_counts'])}`",
        f"- watch labels: `{_format_counts(aggregate['watch_label_counts'])}`",
        f"- missing file count: `{aggregate['missing_file_count']}`",
        f"- checksum mismatch count: `{aggregate['checksum_mismatch_count']}`",
        f"- raw artifact upload required: `{_bool_token(aggregate['raw_artifact_upload_required'])}`",
        f"- manual review required for quality claim: `{_bool_token(readiness['manual_review_required_for_quality_claim'])}`",
        f"- musical quality claimed: `{_bool_token(readiness['midi_to_solo_musical_quality_claimed'])}`",
        f"- next boundary: `{decision['next_boundary']}`",
        "",
        "## Completion Scope",
        "",
        f"- technical scope: `{completion['completion_scope']}`",
        f"- quality scope: `{completion['quality_scope']}`",
        f"- review input: `{completion['review_input_path']}`",
        "",
        "## Not Proven",
        "",
    ]
    for item in report.get("not_proven", []):
        lines.append(f"- `{item}`")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit residual-aware technical MVP completion")
    parser.add_argument("--pending_report", type=str, required=True)
    parser.add_argument("--handoff_freeze_report", type=str, required=True)
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/music_transformer_finetune_mvp/solo_yield_residual_aware_completion_audit",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--issue_number", type=int, default=0)
    parser.add_argument("--expected_next_boundary", type=str, default=NEXT_BOUNDARY)
    parser.add_argument("--require_technical_complete", action="store_true")
    parser.add_argument("--require_no_quality_claim", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    report = build_completion_audit_report(
        pending_report=read_json(Path(args.pending_report)),
        handoff_freeze_report=read_json(Path(args.handoff_freeze_report)),
        output_dir=output_dir,
        issue_number=int(args.issue_number),
    )
    summary = validate_completion_audit_report(
        report,
        expected_next_boundary=str(args.expected_next_boundary),
        require_technical_complete=bool(args.require_technical_complete),
        require_no_quality_claim=bool(args.require_no_quality_claim),
    )
    if args.doc_path:
        write_text(Path(args.doc_path), markdown_report(report))
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
