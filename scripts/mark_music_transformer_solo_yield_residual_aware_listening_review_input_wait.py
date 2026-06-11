"""Record residual-aware listening review input wait state."""

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


SCHEMA_VERSION = "music_transformer_solo_yield_residual_aware_listening_review_input_wait_v1"
FINAL_STATUS_SCHEMA_VERSION = "music_transformer_solo_yield_residual_aware_final_status_sync_v1"
BOUNDARY = "music_transformer_solo_yield_residual_aware_listening_review_input_wait"
NEXT_BOUNDARY = "music_transformer_solo_yield_residual_aware_user_listening_review_fill"


class SoloYieldResidualAwareListeningReviewInputWaitError(ValueError):
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
        raise SoloYieldResidualAwareListeningReviewInputWaitError(f"json not found: {path}")
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
        raise SoloYieldResidualAwareListeningReviewInputWaitError(
            f"unexpected quality claim in {label}: {claimed}"
        )


def validate_final_status_sync(report: dict[str, Any]) -> dict[str, Any]:
    if str(report.get("schema_version") or "") != FINAL_STATUS_SCHEMA_VERSION:
        raise SoloYieldResidualAwareListeningReviewInputWaitError("final status sync schema mismatch")
    aggregate = _dict(report.get("aggregate"))
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    if not bool(readiness.get("residual_aware_final_status_sync_completed", False)):
        raise SoloYieldResidualAwareListeningReviewInputWaitError("final status sync completion required")
    if not bool(readiness.get("residual_aware_final_status_synced", False)):
        raise SoloYieldResidualAwareListeningReviewInputWaitError("final status synced required")
    if not bool(readiness.get("technical_mvp_complete", False)):
        raise SoloYieldResidualAwareListeningReviewInputWaitError("technical MVP completion required")
    if not bool(readiness.get("local_review_ready", False)):
        raise SoloYieldResidualAwareListeningReviewInputWaitError("local review readiness required")
    if str(decision.get("next_boundary") or "") != BOUNDARY:
        raise SoloYieldResidualAwareListeningReviewInputWaitError("final status must route to input wait")
    if bool(decision.get("critical_user_input_required", True)):
        raise SoloYieldResidualAwareListeningReviewInputWaitError("critical user input should not be required")
    if _int(aggregate.get("candidate_count")) <= 0:
        raise SoloYieldResidualAwareListeningReviewInputWaitError("candidate count required")
    if _int(aggregate.get("pending_candidate_count")) != _int(aggregate.get("candidate_count")):
        raise SoloYieldResidualAwareListeningReviewInputWaitError("pending candidate count mismatch")
    _require_no_quality_claim(readiness, label="final status readiness")
    return {
        "candidate_count": _int(aggregate.get("candidate_count")),
        "midi_count": _int(aggregate.get("midi_count")),
        "wav_count": _int(aggregate.get("wav_count")),
        "pending_candidate_count": _int(aggregate.get("pending_candidate_count")),
        "quality_proxy_pass_count": _int(aggregate.get("quality_proxy_pass_count")),
        "quality_proxy_fail_count": _int(aggregate.get("quality_proxy_fail_count")),
        "major_label_counts": _dict(aggregate.get("major_label_counts")),
        "watch_label_counts": _dict(aggregate.get("watch_label_counts")),
    }


def build_input_wait_report(
    *,
    final_status_sync_report: dict[str, Any],
    output_dir: Path,
    issue_number: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = validate_final_status_sync(final_status_sync_report)
    report = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "issue_number": int(issue_number),
        "source_reports": {
            "final_status_sync": final_status_sync_report.get("output_dir"),
        },
        "aggregate": {
            "candidate_count": summary["candidate_count"],
            "midi_count": summary["midi_count"],
            "wav_count": summary["wav_count"],
            "pending_candidate_count": summary["pending_candidate_count"],
            "quality_proxy_pass_count": summary["quality_proxy_pass_count"],
            "quality_proxy_fail_count": summary["quality_proxy_fail_count"],
            "major_label_counts": summary["major_label_counts"],
            "watch_label_counts": summary["watch_label_counts"],
        },
        "readiness": {
            "residual_aware_listening_review_input_wait_recorded": True,
            "technical_mvp_complete": True,
            "final_status_synced": True,
            "local_review_ready": True,
            "user_listening_input_required_for_quality_claim": True,
            "automated_quality_claim_blocked": True,
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
            "reason": "technical MVP complete; wait for user listening input before review fill or quality claim",
        },
        "not_proven": [
            "listening_review_completed",
            "human_audio_preference",
            "midi_to_solo_musical_quality",
            "stable_jazz_solo_quality",
            "production_ready_improviser",
        ],
    }
    write_json(output_dir / "residual_aware_listening_review_input_wait.json", report)
    write_json(
        output_dir / "residual_aware_listening_review_input_wait_summary.json",
        validate_input_wait_report(report),
    )
    write_text(output_dir / "residual_aware_listening_review_input_wait.md", markdown_report(report))
    return report


def validate_input_wait_report(
    report: dict[str, Any],
    *,
    expected_next_boundary: str | None = None,
    require_wait_recorded: bool = False,
    require_no_quality_claim: bool = False,
) -> dict[str, Any]:
    if str(report.get("schema_version") or "") != SCHEMA_VERSION:
        raise SoloYieldResidualAwareListeningReviewInputWaitError("input wait schema mismatch")
    aggregate = _dict(report.get("aggregate"))
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    if not bool(readiness.get("residual_aware_listening_review_input_wait_recorded", False)):
        raise SoloYieldResidualAwareListeningReviewInputWaitError("input wait record required")
    if expected_next_boundary and str(decision.get("next_boundary") or "") != expected_next_boundary:
        raise SoloYieldResidualAwareListeningReviewInputWaitError("unexpected next boundary")
    if require_wait_recorded:
        if not bool(readiness.get("technical_mvp_complete", False)):
            raise SoloYieldResidualAwareListeningReviewInputWaitError("technical MVP completion required")
        if not bool(readiness.get("user_listening_input_required_for_quality_claim", False)):
            raise SoloYieldResidualAwareListeningReviewInputWaitError("user listening input requirement required")
        if not bool(readiness.get("automated_quality_claim_blocked", False)):
            raise SoloYieldResidualAwareListeningReviewInputWaitError("automated quality claim block required")
        if bool(readiness.get("validated_listening_input_present", True)):
            raise SoloYieldResidualAwareListeningReviewInputWaitError("validated listening input should remain false")
        if bool(readiness.get("preference_fill_allowed", True)):
            raise SoloYieldResidualAwareListeningReviewInputWaitError("preference fill should remain false")
        if bool(readiness.get("listening_review_completed", True)):
            raise SoloYieldResidualAwareListeningReviewInputWaitError("listening review should remain false")
    if require_no_quality_claim:
        _require_no_quality_claim(readiness, label="input wait readiness")
    if bool(decision.get("critical_user_input_required", True)):
        raise SoloYieldResidualAwareListeningReviewInputWaitError("critical user input should not be required")
    return {
        "schema_version": str(report.get("schema_version")),
        "technical_mvp_complete": bool(readiness.get("technical_mvp_complete", False)),
        "final_status_synced": bool(readiness.get("final_status_synced", False)),
        "local_review_ready": bool(readiness.get("local_review_ready", False)),
        "user_listening_input_required_for_quality_claim": bool(
            readiness.get("user_listening_input_required_for_quality_claim", False)
        ),
        "automated_quality_claim_blocked": bool(
            readiness.get("automated_quality_claim_blocked", False)
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
    lines = [
        "# Music Transformer Solo Yield Residual-Aware Listening Review Input Wait",
        "",
        "## Summary",
        "",
        f"- issue: `#{report['issue_number']}`",
        f"- technical MVP complete: `{_bool_token(readiness['technical_mvp_complete'])}`",
        f"- final status synced: `{_bool_token(readiness['final_status_synced'])}`",
        f"- local review ready: `{_bool_token(readiness['local_review_ready'])}`",
        f"- candidate count: `{aggregate['candidate_count']}`",
        f"- MIDI/WAV: `{aggregate['midi_count']}` / `{aggregate['wav_count']}`",
        f"- pending candidate count: `{aggregate['pending_candidate_count']}`",
        (
            "- quality proxy pass/fail: "
            f"`{aggregate['quality_proxy_pass_count']}` / `{aggregate['quality_proxy_fail_count']}`"
        ),
        f"- major labels: `{_format_counts(aggregate['major_label_counts'])}`",
        f"- watch labels: `{_format_counts(aggregate['watch_label_counts'])}`",
        (
            "- user listening input required for quality claim: "
            f"`{_bool_token(readiness['user_listening_input_required_for_quality_claim'])}`"
        ),
        f"- automated quality claim blocked: `{_bool_token(readiness['automated_quality_claim_blocked'])}`",
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
    parser = argparse.ArgumentParser(description="Record residual-aware listening review input wait")
    parser.add_argument("--final_status_sync", type=str, required=True)
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/music_transformer_finetune_mvp/solo_yield_residual_aware_listening_review_input_wait",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--issue_number", type=int, default=0)
    parser.add_argument("--expected_next_boundary", type=str, default=NEXT_BOUNDARY)
    parser.add_argument("--require_wait_recorded", action="store_true")
    parser.add_argument("--require_no_quality_claim", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    report = build_input_wait_report(
        final_status_sync_report=read_json(Path(args.final_status_sync)),
        output_dir=output_dir,
        issue_number=int(args.issue_number),
    )
    summary = validate_input_wait_report(
        report,
        expected_next_boundary=str(args.expected_next_boundary),
        require_wait_recorded=bool(args.require_wait_recorded),
        require_no_quality_claim=bool(args.require_no_quality_claim),
    )
    if args.doc_path:
        write_text(Path(args.doc_path), markdown_report(report))
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
