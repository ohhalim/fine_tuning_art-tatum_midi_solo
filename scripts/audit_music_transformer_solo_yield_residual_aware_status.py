"""Audit residual-aware status sync for the Music Transformer solo-yield MVP."""

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


SCHEMA_VERSION = "music_transformer_solo_yield_residual_aware_status_audit_v1"
FINAL_REVIEW_SCHEMA_VERSION = "music_transformer_solo_yield_residual_aware_final_review_package_v1"
INPUT_GUARD_SCHEMA_VERSION = "music_transformer_solo_yield_residual_aware_listening_input_guard_v1"
STATUS_AUDIT_BOUNDARY = "music_transformer_solo_yield_residual_aware_status_audit"
NEXT_BOUNDARY = "music_transformer_solo_yield_residual_aware_mvp_handoff_freeze"
STATUS_SYNC_REPAIR_BOUNDARY = "music_transformer_solo_yield_residual_aware_status_sync_repair"


class SoloYieldResidualAwareStatusAuditError(ValueError):
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
        raise SoloYieldResidualAwareStatusAuditError(f"json not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def read_text(path: Path) -> str:
    if not path.exists():
        raise SoloYieldResidualAwareStatusAuditError(f"text not found: {path}")
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
        raise SoloYieldResidualAwareStatusAuditError(
            f"unexpected quality claim in {label}: {claimed}"
        )


def validate_final_review_package(report: dict[str, Any]) -> dict[str, Any]:
    if str(report.get("schema_version") or "") != FINAL_REVIEW_SCHEMA_VERSION:
        raise SoloYieldResidualAwareStatusAuditError("final review schema mismatch")
    aggregate = _dict(report.get("aggregate"))
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    candidate_count = _int(aggregate.get("candidate_count"))
    midi_count = _int(aggregate.get("midi_count"))
    wav_count = _int(aggregate.get("wav_count"))
    pass_count = _int(aggregate.get("quality_proxy_pass_count"))
    fail_count = _int(aggregate.get("quality_proxy_fail_count"))
    if not bool(readiness.get("residual_aware_final_review_package_ready", False)):
        raise SoloYieldResidualAwareStatusAuditError("final review package readiness required")
    if candidate_count <= 0:
        raise SoloYieldResidualAwareStatusAuditError("candidate count required")
    if midi_count != candidate_count:
        raise SoloYieldResidualAwareStatusAuditError("MIDI count mismatch")
    if wav_count != candidate_count:
        raise SoloYieldResidualAwareStatusAuditError("WAV count mismatch")
    if pass_count + fail_count != candidate_count:
        raise SoloYieldResidualAwareStatusAuditError("proxy pass/fail count mismatch")
    if _int(aggregate.get("checksum_mismatch_count")) != 0:
        raise SoloYieldResidualAwareStatusAuditError("checksum mismatch count must be zero")
    if _int(aggregate.get("missing_file_count")) != 0:
        raise SoloYieldResidualAwareStatusAuditError("missing file count must be zero")
    if str(decision.get("next_boundary") or "") != (
        "music_transformer_solo_yield_residual_aware_listening_input_guard"
    ):
        raise SoloYieldResidualAwareStatusAuditError("final review next boundary mismatch")
    if bool(decision.get("critical_user_input_required", True)):
        raise SoloYieldResidualAwareStatusAuditError("critical user input should not be required")
    _require_no_quality_claim(readiness, label="final review readiness")
    return {
        "candidate_count": candidate_count,
        "midi_count": midi_count,
        "wav_count": wav_count,
        "quality_proxy_pass_count": pass_count,
        "quality_proxy_fail_count": fail_count,
        "major_label_counts": _dict(aggregate.get("major_label_counts")),
        "watch_label_counts": _dict(aggregate.get("watch_label_counts")),
        "validated_listening_input_present": bool(
            readiness.get("validated_listening_input_present", True)
        ),
        "review_input_template_written": bool(readiness.get("review_input_template_written", False)),
    }


def validate_input_guard_report(
    report: dict[str, Any],
    *,
    expected_candidate_count: int,
) -> dict[str, Any]:
    if str(report.get("schema_version") or "") != INPUT_GUARD_SCHEMA_VERSION:
        raise SoloYieldResidualAwareStatusAuditError("input guard schema mismatch")
    source = _dict(report.get("source_package_summary"))
    guard = _dict(report.get("guard_result"))
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    if not bool(readiness.get("residual_aware_listening_input_guard_completed", False)):
        raise SoloYieldResidualAwareStatusAuditError("input guard completion required")
    if _int(source.get("candidate_count")) != expected_candidate_count:
        raise SoloYieldResidualAwareStatusAuditError("guard source candidate count mismatch")
    if _int(source.get("midi_count")) != expected_candidate_count:
        raise SoloYieldResidualAwareStatusAuditError("guard source MIDI count mismatch")
    if _int(source.get("wav_count")) != expected_candidate_count:
        raise SoloYieldResidualAwareStatusAuditError("guard source WAV count mismatch")
    if _int(guard.get("review_item_count")) != expected_candidate_count:
        raise SoloYieldResidualAwareStatusAuditError("guard review item count mismatch")
    if bool(decision.get("critical_user_input_required", True)):
        raise SoloYieldResidualAwareStatusAuditError("critical user input should not be required")
    if str(decision.get("next_boundary") or "") != (
        "music_transformer_solo_yield_residual_aware_status_sync"
    ):
        raise SoloYieldResidualAwareStatusAuditError("guard next boundary mismatch")
    _require_no_quality_claim(readiness, label="input guard readiness")
    return {
        "candidate_count": _int(source.get("candidate_count")),
        "midi_count": _int(source.get("midi_count")),
        "wav_count": _int(source.get("wav_count")),
        "quality_proxy_pass_count": _int(source.get("quality_proxy_pass_count")),
        "quality_proxy_fail_count": _int(source.get("quality_proxy_fail_count")),
        "major_label_counts": _dict(source.get("major_label_counts")),
        "watch_label_counts": _dict(source.get("watch_label_counts")),
        "validated_listening_input_present": bool(guard.get("validated_listening_input_present", True)),
        "preference_fill_allowed": bool(guard.get("preference_fill_allowed", True)),
        "listening_review_completed": bool(readiness.get("listening_review_completed", True)),
        "review_item_count": _int(guard.get("review_item_count")),
    }


def required_readme_snippets(final_summary: dict[str, Any], guard_summary: dict[str, Any]) -> list[str]:
    major = _dict(final_summary.get("major_label_counts"))
    watch = _dict(final_summary.get("watch_label_counts"))
    return [
        f"최신 review package: MIDI `{final_summary['midi_count']}`, WAV `{final_summary['wav_count']}`",
        (
            "objective rubric: pass/fail "
            f"`{final_summary['quality_proxy_pass_count']} / {final_summary['quality_proxy_fail_count']}`"
        ),
        f"남은 major label: `{_format_counts(major)}`",
        f"남은 watch label: `{_format_counts(watch)}`",
        "tension 추가 repair 가능성: current guard 기준 `false`",
        f"validated listening input: `{_bool_token(guard_summary['validated_listening_input_present'])}`",
        "human/audio preference claim: `false`",
        "musical quality claim: `false`",
        "residual_aware_final_review_package.md",
        "residual_aware_listening_input_guard",
    ]


def required_current_status_snippets(
    *,
    issue_number: int,
    final_summary: dict[str, Any],
    guard_summary: dict[str, Any],
) -> list[str]:
    major = _dict(final_summary.get("major_label_counts"))
    watch = _dict(final_summary.get("watch_label_counts"))
    return [
        f"current issue: Issue #{issue_number}, Stage B MIDI-to-solo residual-aware status audit",
        f"residual-aware final review package candidate count: `{final_summary['candidate_count']}`",
        f"residual-aware final review MIDI/WAV: `{final_summary['midi_count']} / {final_summary['wav_count']}`",
        (
            "residual-aware objective rubric pass/fail: "
            f"`{final_summary['quality_proxy_pass_count']} / {final_summary['quality_proxy_fail_count']}`"
        ),
        f"residual-aware residual major label: `{_format_counts(major)}`",
        f"residual-aware residual watch label: `{_format_counts(watch)}`",
        "residual tension repeat feasible under current guard: `false`",
        (
            "residual-aware listening input guard validated input: "
            f"`{_bool_token(guard_summary['validated_listening_input_present'])}`"
        ),
        (
            "residual-aware listening input guard preference fill: "
            f"`{_bool_token(guard_summary['preference_fill_allowed'])}`"
        ),
        f"residual-aware next boundary: `{STATUS_AUDIT_BOUNDARY}`",
    ]


def validate_documentation_sync(
    *,
    readme_text: str,
    current_status_text: str,
    issue_number: int,
    final_summary: dict[str, Any],
    guard_summary: dict[str, Any],
) -> dict[str, Any]:
    readme_required = required_readme_snippets(final_summary, guard_summary)
    current_status_required = required_current_status_snippets(
        issue_number=issue_number,
        final_summary=final_summary,
        guard_summary=guard_summary,
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
        "readme_status_synced": not readme_missing,
        "current_status_synced": not current_status_missing,
        "documentation_synced": not readme_missing and not current_status_missing,
    }


def build_status_audit_report(
    *,
    final_review_package: dict[str, Any],
    input_guard_report: dict[str, Any],
    readme_text: str,
    current_status_text: str,
    output_dir: Path,
    issue_number: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    final_summary = validate_final_review_package(final_review_package)
    guard_summary = validate_input_guard_report(
        input_guard_report,
        expected_candidate_count=int(final_summary["candidate_count"]),
    )
    docs = validate_documentation_sync(
        readme_text=readme_text,
        current_status_text=current_status_text,
        issue_number=int(issue_number),
        final_summary=final_summary,
        guard_summary=guard_summary,
    )
    counts_match = all(
        final_summary[key] == guard_summary[key]
        for key in (
            "candidate_count",
            "midi_count",
            "wav_count",
            "quality_proxy_pass_count",
            "quality_proxy_fail_count",
            "major_label_counts",
            "watch_label_counts",
        )
    )
    pending_input = not bool(guard_summary["validated_listening_input_present"])
    preference_blocked = not bool(guard_summary["preference_fill_allowed"])
    listening_pending = not bool(guard_summary["listening_review_completed"])
    status_ready = bool(
        counts_match
        and docs["documentation_synced"]
        and pending_input
        and preference_blocked
        and listening_pending
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "issue_number": int(issue_number),
        "source_reports": {
            "final_review_package": final_review_package.get("output_dir"),
            "input_guard_report": input_guard_report.get("output_dir"),
        },
        "aggregate": {
            "candidate_count": final_summary["candidate_count"],
            "midi_count": final_summary["midi_count"],
            "wav_count": final_summary["wav_count"],
            "quality_proxy_pass_count": final_summary["quality_proxy_pass_count"],
            "quality_proxy_fail_count": final_summary["quality_proxy_fail_count"],
            "major_label_counts": final_summary["major_label_counts"],
            "watch_label_counts": final_summary["watch_label_counts"],
            "review_item_count": guard_summary["review_item_count"],
            "counts_match": counts_match,
        },
        "documentation_audit": docs,
        "readiness": {
            "residual_aware_status_audit_completed": True,
            "residual_aware_status_synced": status_ready,
            "readme_status_synced": bool(docs["readme_status_synced"]),
            "current_status_synced": bool(docs["current_status_synced"]),
            "validated_listening_input_present": bool(guard_summary["validated_listening_input_present"]),
            "preference_fill_allowed": bool(guard_summary["preference_fill_allowed"]),
            "listening_review_completed": bool(guard_summary["listening_review_completed"]),
            "human_audio_preference_claimed": False,
            "midi_to_solo_musical_quality_claimed": False,
            "musical_quality_claimed": False,
            "artist_style_claimed": False,
            "production_ready_claimed": False,
        },
        "decision": {
            "current_boundary": STATUS_AUDIT_BOUNDARY,
            "next_boundary": NEXT_BOUNDARY if status_ready else STATUS_SYNC_REPAIR_BOUNDARY,
            "critical_user_input_required": False,
            "reason": (
                "residual-aware status synced; freeze MVP handoff boundary"
                if status_ready
                else "residual-aware status sync mismatch; repair documentation boundary"
            ),
        },
        "not_proven": [
            "listening_review_completed",
            "human_audio_preference",
            "midi_to_solo_musical_quality",
            "stable_jazz_solo_quality",
            "production_ready_improviser",
        ],
    }
    write_json(output_dir / "residual_aware_status_audit.json", report)
    write_json(output_dir / "residual_aware_status_audit_summary.json", validate_status_audit_report(report))
    write_text(output_dir / "residual_aware_status_audit.md", markdown_report(report))
    return report


def validate_status_audit_report(
    report: dict[str, Any],
    *,
    expected_next_boundary: str | None = None,
    require_docs_synced: bool = False,
    require_pending_input: bool = False,
    require_no_quality_claim: bool = False,
) -> dict[str, Any]:
    if str(report.get("schema_version") or "") != SCHEMA_VERSION:
        raise SoloYieldResidualAwareStatusAuditError("status audit schema mismatch")
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    aggregate = _dict(report.get("aggregate"))
    if not bool(readiness.get("residual_aware_status_audit_completed", False)):
        raise SoloYieldResidualAwareStatusAuditError("status audit completion required")
    if expected_next_boundary and str(decision.get("next_boundary") or "") != expected_next_boundary:
        raise SoloYieldResidualAwareStatusAuditError("unexpected next boundary")
    if require_docs_synced and not bool(readiness.get("residual_aware_status_synced", False)):
        raise SoloYieldResidualAwareStatusAuditError("residual-aware status sync required")
    if require_pending_input and bool(readiness.get("validated_listening_input_present", True)):
        raise SoloYieldResidualAwareStatusAuditError("validated listening input should remain pending")
    if require_pending_input and bool(readiness.get("preference_fill_allowed", True)):
        raise SoloYieldResidualAwareStatusAuditError("preference fill should remain blocked")
    if require_pending_input and bool(readiness.get("listening_review_completed", True)):
        raise SoloYieldResidualAwareStatusAuditError("listening review should remain pending")
    if require_no_quality_claim:
        _require_no_quality_claim(readiness, label="status audit readiness")
    if bool(decision.get("critical_user_input_required", True)):
        raise SoloYieldResidualAwareStatusAuditError("critical user input should not be required")
    return {
        "schema_version": str(report.get("schema_version")),
        "residual_aware_status_synced": bool(readiness.get("residual_aware_status_synced", False)),
        "readme_status_synced": bool(readiness.get("readme_status_synced", False)),
        "current_status_synced": bool(readiness.get("current_status_synced", False)),
        "candidate_count": _int(aggregate.get("candidate_count")),
        "midi_count": _int(aggregate.get("midi_count")),
        "wav_count": _int(aggregate.get("wav_count")),
        "quality_proxy_pass_count": _int(aggregate.get("quality_proxy_pass_count")),
        "quality_proxy_fail_count": _int(aggregate.get("quality_proxy_fail_count")),
        "validated_listening_input_present": bool(
            readiness.get("validated_listening_input_present", True)
        ),
        "preference_fill_allowed": bool(readiness.get("preference_fill_allowed", True)),
        "listening_review_completed": bool(readiness.get("listening_review_completed", True)),
        "human_audio_preference_claimed": bool(
            readiness.get("human_audio_preference_claimed", True)
        ),
        "midi_to_solo_musical_quality_claimed": bool(
            readiness.get("midi_to_solo_musical_quality_claimed", True)
        ),
        "next_boundary": str(decision.get("next_boundary") or ""),
        "critical_user_input_required": bool(decision.get("critical_user_input_required", True)),
    }


def markdown_report(report: dict[str, Any]) -> str:
    aggregate = report["aggregate"]
    docs = report["documentation_audit"]
    readiness = report["readiness"]
    decision = report["decision"]
    lines = [
        "# Music Transformer Solo Yield Residual-Aware Status Audit",
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
        f"- README synced: `{_bool_token(readiness['readme_status_synced'])}`",
        f"- current status synced: `{_bool_token(readiness['current_status_synced'])}`",
        f"- validated listening input present: `{_bool_token(readiness['validated_listening_input_present'])}`",
        f"- preference fill allowed: `{_bool_token(readiness['preference_fill_allowed'])}`",
        f"- listening review completed: `{_bool_token(readiness['listening_review_completed'])}`",
        f"- musical quality claimed: `{_bool_token(readiness['midi_to_solo_musical_quality_claimed'])}`",
        f"- next boundary: `{decision['next_boundary']}`",
        "",
        "## Documentation Missing Snippets",
        "",
        f"- README missing count: `{len(docs['readme_missing_snippets'])}`",
        f"- current status missing count: `{len(docs['current_status_missing_snippets'])}`",
        "",
        "## Not Proven",
        "",
    ]
    for item in report.get("not_proven", []):
        lines.append(f"- `{item}`")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit residual-aware solo-yield status sync")
    parser.add_argument("--final_review_package", type=str, required=True)
    parser.add_argument("--input_guard_report", type=str, required=True)
    parser.add_argument("--readme_path", type=str, default="README.md")
    parser.add_argument("--current_status_path", type=str, default="docs/CURRENT_STATUS_AND_PLAN.md")
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/music_transformer_finetune_mvp/solo_yield_residual_aware_status_audit",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--issue_number", type=int, default=0)
    parser.add_argument("--expected_next_boundary", type=str, default=NEXT_BOUNDARY)
    parser.add_argument("--require_docs_synced", action="store_true")
    parser.add_argument("--require_pending_input", action="store_true")
    parser.add_argument("--require_no_quality_claim", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    report = build_status_audit_report(
        final_review_package=read_json(Path(args.final_review_package)),
        input_guard_report=read_json(Path(args.input_guard_report)),
        readme_text=read_text(Path(args.readme_path)),
        current_status_text=read_text(Path(args.current_status_path)),
        output_dir=output_dir,
        issue_number=int(args.issue_number),
    )
    summary = validate_status_audit_report(
        report,
        expected_next_boundary=str(args.expected_next_boundary),
        require_docs_synced=bool(args.require_docs_synced),
        require_pending_input=bool(args.require_pending_input),
        require_no_quality_claim=bool(args.require_no_quality_claim),
    )
    if args.doc_path:
        write_text(Path(args.doc_path), markdown_report(report))
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
