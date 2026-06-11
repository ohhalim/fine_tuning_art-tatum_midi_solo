"""Sync residual-aware final status after technical MVP completion audit."""

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


SCHEMA_VERSION = "music_transformer_solo_yield_residual_aware_final_status_sync_v1"
COMPLETION_SCHEMA_VERSION = "music_transformer_solo_yield_residual_aware_completion_audit_v1"
BOUNDARY = "music_transformer_solo_yield_residual_aware_final_status_sync"
NEXT_BOUNDARY = "music_transformer_solo_yield_residual_aware_listening_review_input_wait"


class SoloYieldResidualAwareFinalStatusSyncError(ValueError):
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
        raise SoloYieldResidualAwareFinalStatusSyncError(f"json not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def read_text(path: Path) -> str:
    if not path.exists():
        raise SoloYieldResidualAwareFinalStatusSyncError(f"text not found: {path}")
    return path.read_text(encoding="utf-8")


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
        raise SoloYieldResidualAwareFinalStatusSyncError(
            f"unexpected quality claim in {label}: {claimed}"
        )


def validate_completion_audit(report: dict[str, Any]) -> dict[str, Any]:
    if str(report.get("schema_version") or "") != COMPLETION_SCHEMA_VERSION:
        raise SoloYieldResidualAwareFinalStatusSyncError("completion audit schema mismatch")
    aggregate = _dict(report.get("aggregate"))
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    if not bool(readiness.get("residual_aware_completion_audit_completed", False)):
        raise SoloYieldResidualAwareFinalStatusSyncError("completion audit record required")
    if not bool(readiness.get("technical_mvp_complete", False)):
        raise SoloYieldResidualAwareFinalStatusSyncError("technical MVP completion required")
    if not bool(readiness.get("local_review_ready", False)):
        raise SoloYieldResidualAwareFinalStatusSyncError("local review readiness required")
    if str(decision.get("next_boundary") or "") != BOUNDARY:
        raise SoloYieldResidualAwareFinalStatusSyncError("completion audit must route to final status sync")
    if bool(decision.get("critical_user_input_required", True)):
        raise SoloYieldResidualAwareFinalStatusSyncError("critical user input should not be required")
    if _int(aggregate.get("candidate_count")) <= 0:
        raise SoloYieldResidualAwareFinalStatusSyncError("candidate count required")
    if _int(aggregate.get("midi_count")) != _int(aggregate.get("candidate_count")):
        raise SoloYieldResidualAwareFinalStatusSyncError("MIDI count mismatch")
    if _int(aggregate.get("wav_count")) != _int(aggregate.get("candidate_count")):
        raise SoloYieldResidualAwareFinalStatusSyncError("WAV count mismatch")
    if _int(aggregate.get("pending_candidate_count")) != _int(aggregate.get("candidate_count")):
        raise SoloYieldResidualAwareFinalStatusSyncError("pending candidate count mismatch")
    if _int(aggregate.get("missing_file_count")) != 0:
        raise SoloYieldResidualAwareFinalStatusSyncError("missing file count must be zero")
    if _int(aggregate.get("checksum_mismatch_count")) != 0:
        raise SoloYieldResidualAwareFinalStatusSyncError("checksum mismatch count must be zero")
    if bool(aggregate.get("raw_artifact_upload_required", True)):
        raise SoloYieldResidualAwareFinalStatusSyncError("raw artifact upload should not be required")
    _require_no_quality_claim(readiness, label="completion readiness")
    return {
        "candidate_count": _int(aggregate.get("candidate_count")),
        "midi_count": _int(aggregate.get("midi_count")),
        "wav_count": _int(aggregate.get("wav_count")),
        "pending_candidate_count": _int(aggregate.get("pending_candidate_count")),
        "quality_proxy_pass_count": _int(aggregate.get("quality_proxy_pass_count")),
        "quality_proxy_fail_count": _int(aggregate.get("quality_proxy_fail_count")),
        "major_label_counts": _dict(aggregate.get("major_label_counts")),
        "watch_label_counts": _dict(aggregate.get("watch_label_counts")),
        "technical_mvp_complete": bool(readiness.get("technical_mvp_complete", False)),
        "local_review_ready": bool(readiness.get("local_review_ready", False)),
        "manual_review_required_for_quality_claim": bool(
            readiness.get("manual_review_required_for_quality_claim", False)
        ),
    }


def required_readme_snippets(summary: dict[str, Any]) -> list[str]:
    return [
        "residual-aware completion audit: technical MVP complete `true`, local review ready `true`",
        "completion audit doc: `docs/STAGE_B_MIDI_TO_SOLO_RESIDUAL_AWARE_COMPLETION_AUDIT_2026-06-11.md`",
        f"최신 review package: MIDI `{summary['midi_count']}`, WAV `{summary['wav_count']}`",
        f"objective rubric: pass/fail `{summary['quality_proxy_pass_count']} / {summary['quality_proxy_fail_count']}`",
        f"남은 major label: `{_format_counts(summary['major_label_counts'])}`",
        f"남은 watch label: `{_format_counts(summary['watch_label_counts'])}`",
        "validated listening input: `false`",
        "musical quality claim: `false`",
    ]


def required_current_status_snippets(*, issue_number: int, summary: dict[str, Any]) -> list[str]:
    return [
        f"current issue: Issue #{issue_number}, Stage B MIDI-to-solo residual-aware final status sync",
        "residual-aware completion audit technical MVP complete: `true`",
        "residual-aware completion audit local review ready: `true`",
        "residual-aware completion audit quality claim: `false`",
        "residual-aware completion audit next boundary: `music_transformer_solo_yield_residual_aware_final_status_sync`",
        f"residual-aware final status sync candidate count: `{summary['candidate_count']}`",
        f"residual-aware final status sync MIDI/WAV: `{summary['midi_count']} / {summary['wav_count']}`",
        "stable jazz solo quality: `not_proven`",
        "human listening preference input: `false`",
    ]


def validate_documentation_sync(
    *,
    readme_text: str,
    current_status_text: str,
    issue_number: int,
    summary: dict[str, Any],
) -> dict[str, Any]:
    readme_required = required_readme_snippets(summary)
    current_status_required = required_current_status_snippets(
        issue_number=issue_number,
        summary=summary,
    )
    readme_missing = [snippet for snippet in readme_required if snippet not in readme_text]
    current_status_missing = [
        snippet for snippet in current_status_required if snippet not in current_status_text
    ]
    return {
        "readme_required_snippets": readme_required,
        "current_status_required_snippets": current_status_required,
        "readme_missing_snippets": readme_missing,
        "current_status_missing_snippets": current_status_missing,
        "readme_final_status_synced": not readme_missing,
        "current_status_final_synced": not current_status_missing,
        "final_status_synced": not readme_missing and not current_status_missing,
    }


def build_final_status_sync_report(
    *,
    completion_audit: dict[str, Any],
    readme_text: str,
    current_status_text: str,
    output_dir: Path,
    issue_number: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = validate_completion_audit(completion_audit)
    docs = validate_documentation_sync(
        readme_text=readme_text,
        current_status_text=current_status_text,
        issue_number=int(issue_number),
        summary=summary,
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "issue_number": int(issue_number),
        "source_reports": {
            "completion_audit": completion_audit.get("output_dir"),
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
        "documentation_audit": docs,
        "readiness": {
            "residual_aware_final_status_sync_completed": True,
            "residual_aware_final_status_synced": bool(docs["final_status_synced"]),
            "technical_mvp_complete": summary["technical_mvp_complete"],
            "local_review_ready": summary["local_review_ready"],
            "manual_review_required_for_quality_claim": summary[
                "manual_review_required_for_quality_claim"
            ],
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
            "reason": "final status synced; wait for listening review input before any quality claim",
        },
        "not_proven": [
            "listening_review_completed",
            "human_audio_preference",
            "midi_to_solo_musical_quality",
            "stable_jazz_solo_quality",
            "production_ready_improviser",
        ],
    }
    write_json(output_dir / "residual_aware_final_status_sync.json", report)
    write_json(
        output_dir / "residual_aware_final_status_sync_summary.json",
        validate_final_status_sync_report(report),
    )
    write_text(output_dir / "residual_aware_final_status_sync.md", markdown_report(report))
    return report


def validate_final_status_sync_report(
    report: dict[str, Any],
    *,
    expected_next_boundary: str | None = None,
    require_final_status_synced: bool = False,
    require_no_quality_claim: bool = False,
) -> dict[str, Any]:
    if str(report.get("schema_version") or "") != SCHEMA_VERSION:
        raise SoloYieldResidualAwareFinalStatusSyncError("final status sync schema mismatch")
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    aggregate = _dict(report.get("aggregate"))
    if not bool(readiness.get("residual_aware_final_status_sync_completed", False)):
        raise SoloYieldResidualAwareFinalStatusSyncError("final status sync completion required")
    if expected_next_boundary and str(decision.get("next_boundary") or "") != expected_next_boundary:
        raise SoloYieldResidualAwareFinalStatusSyncError("unexpected next boundary")
    if require_final_status_synced and not bool(
        readiness.get("residual_aware_final_status_synced", False)
    ):
        raise SoloYieldResidualAwareFinalStatusSyncError("final status sync required")
    if require_no_quality_claim:
        _require_no_quality_claim(readiness, label="final status readiness")
    if bool(decision.get("critical_user_input_required", True)):
        raise SoloYieldResidualAwareFinalStatusSyncError("critical user input should not be required")
    return {
        "schema_version": str(report.get("schema_version")),
        "residual_aware_final_status_synced": bool(
            readiness.get("residual_aware_final_status_synced", False)
        ),
        "technical_mvp_complete": bool(readiness.get("technical_mvp_complete", False)),
        "local_review_ready": bool(readiness.get("local_review_ready", False)),
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
    docs = report["documentation_audit"]
    lines = [
        "# Music Transformer Solo Yield Residual-Aware Final Status Sync",
        "",
        "## Summary",
        "",
        f"- issue: `#{report['issue_number']}`",
        f"- final status synced: `{_bool_token(readiness['residual_aware_final_status_synced'])}`",
        f"- technical MVP complete: `{_bool_token(readiness['technical_mvp_complete'])}`",
        f"- local review ready: `{_bool_token(readiness['local_review_ready'])}`",
        f"- candidate count: `{aggregate['candidate_count']}`",
        f"- MIDI/WAV: `{aggregate['midi_count']}` / `{aggregate['wav_count']}`",
        (
            "- quality proxy pass/fail: "
            f"`{aggregate['quality_proxy_pass_count']}` / `{aggregate['quality_proxy_fail_count']}`"
        ),
        f"- pending candidate count: `{aggregate['pending_candidate_count']}`",
        f"- major labels: `{_format_counts(aggregate['major_label_counts'])}`",
        f"- watch labels: `{_format_counts(aggregate['watch_label_counts'])}`",
        f"- README missing snippet count: `{len(docs['readme_missing_snippets'])}`",
        f"- current status missing snippet count: `{len(docs['current_status_missing_snippets'])}`",
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
    parser = argparse.ArgumentParser(description="Sync residual-aware final status")
    parser.add_argument("--completion_audit", type=str, required=True)
    parser.add_argument("--readme_path", type=str, default="README.md")
    parser.add_argument("--current_status_path", type=str, default="docs/CURRENT_STATUS_AND_PLAN.md")
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/music_transformer_finetune_mvp/solo_yield_residual_aware_final_status_sync",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--issue_number", type=int, default=0)
    parser.add_argument("--expected_next_boundary", type=str, default=NEXT_BOUNDARY)
    parser.add_argument("--require_final_status_synced", action="store_true")
    parser.add_argument("--require_no_quality_claim", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    report = build_final_status_sync_report(
        completion_audit=read_json(Path(args.completion_audit)),
        readme_text=read_text(Path(args.readme_path)),
        current_status_text=read_text(Path(args.current_status_path)),
        output_dir=output_dir,
        issue_number=int(args.issue_number),
    )
    summary = validate_final_status_sync_report(
        report,
        expected_next_boundary=str(args.expected_next_boundary),
        require_final_status_synced=bool(args.require_final_status_synced),
        require_no_quality_claim=bool(args.require_no_quality_claim),
    )
    if args.doc_path:
        write_text(Path(args.doc_path), markdown_report(report))
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
