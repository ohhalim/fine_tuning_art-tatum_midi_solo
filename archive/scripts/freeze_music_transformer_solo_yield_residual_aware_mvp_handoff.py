"""Freeze residual-aware local MVP handoff for solo-yield candidates."""

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
from scripts.render_stage_b_midi_to_solo_candidate_audio import sha256_file  # noqa: E402


SCHEMA_VERSION = "music_transformer_solo_yield_residual_aware_mvp_handoff_freeze_v1"
FINAL_REVIEW_SCHEMA_VERSION = "music_transformer_solo_yield_residual_aware_final_review_package_v1"
INPUT_GUARD_SCHEMA_VERSION = "music_transformer_solo_yield_residual_aware_listening_input_guard_v1"
STATUS_AUDIT_SCHEMA_VERSION = "music_transformer_solo_yield_residual_aware_status_audit_v1"
BOUNDARY = "music_transformer_solo_yield_residual_aware_mvp_handoff_freeze"
NEXT_BOUNDARY = "music_transformer_solo_yield_residual_aware_listening_review_pending"


class SoloYieldResidualAwareMvpHandoffFreezeError(ValueError):
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


def _float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _bool_token(value: Any) -> str:
    return "true" if bool(value) else "false"


def _format_counts(value: dict[str, Any]) -> str:
    if not value:
        return "none"
    return ", ".join(f"{key}={value[key]}" for key in sorted(value))


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SoloYieldResidualAwareMvpHandoffFreezeError(f"json not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _require_file(path_value: Any, *, label: str) -> Path:
    path = Path(str(path_value or ""))
    if not str(path) or not path.exists() or not path.is_file():
        raise SoloYieldResidualAwareMvpHandoffFreezeError(f"{label} missing: {path}")
    return path


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
        raise SoloYieldResidualAwareMvpHandoffFreezeError(
            f"unexpected quality claim in {label}: {claimed}"
        )


def _require_checksum(path: Path, expected: str, *, label: str) -> None:
    actual = sha256_file(path)
    if str(expected or "") != actual:
        raise SoloYieldResidualAwareMvpHandoffFreezeError(
            f"{label} checksum mismatch: expected={expected} actual={actual}"
        )


def validate_status_audit(report: dict[str, Any]) -> dict[str, Any]:
    if str(report.get("schema_version") or "") != STATUS_AUDIT_SCHEMA_VERSION:
        raise SoloYieldResidualAwareMvpHandoffFreezeError("status audit schema mismatch")
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    aggregate = _dict(report.get("aggregate"))
    if not bool(readiness.get("residual_aware_status_audit_completed", False)):
        raise SoloYieldResidualAwareMvpHandoffFreezeError("status audit completion required")
    if not bool(readiness.get("residual_aware_status_synced", False)):
        raise SoloYieldResidualAwareMvpHandoffFreezeError("status sync required before freeze")
    if str(decision.get("next_boundary") or "") != BOUNDARY:
        raise SoloYieldResidualAwareMvpHandoffFreezeError("status audit must route to handoff freeze")
    if bool(decision.get("critical_user_input_required", True)):
        raise SoloYieldResidualAwareMvpHandoffFreezeError("critical user input should not be required")
    _require_no_quality_claim(readiness, label="status audit readiness")
    return {
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
    }


def validate_final_review_package(report: dict[str, Any]) -> dict[str, Any]:
    if str(report.get("schema_version") or "") != FINAL_REVIEW_SCHEMA_VERSION:
        raise SoloYieldResidualAwareMvpHandoffFreezeError("final review schema mismatch")
    aggregate = _dict(report.get("aggregate"))
    readiness = _dict(report.get("readiness"))
    candidates = _list(report.get("candidate_handoff"))
    candidate_count = _int(aggregate.get("candidate_count"))
    if not bool(readiness.get("residual_aware_final_review_package_ready", False)):
        raise SoloYieldResidualAwareMvpHandoffFreezeError("final review readiness required")
    if candidate_count <= 0 or len(candidates) != candidate_count:
        raise SoloYieldResidualAwareMvpHandoffFreezeError("candidate handoff count mismatch")
    if _int(aggregate.get("midi_count")) != candidate_count:
        raise SoloYieldResidualAwareMvpHandoffFreezeError("MIDI count mismatch")
    if _int(aggregate.get("wav_count")) != candidate_count:
        raise SoloYieldResidualAwareMvpHandoffFreezeError("WAV count mismatch")
    if _int(aggregate.get("missing_file_count")) != 0:
        raise SoloYieldResidualAwareMvpHandoffFreezeError("missing file count must be zero")
    if _int(aggregate.get("checksum_mismatch_count")) != 0:
        raise SoloYieldResidualAwareMvpHandoffFreezeError("checksum mismatch count must be zero")
    _require_no_quality_claim(readiness, label="final review readiness")
    return {
        "candidate_count": candidate_count,
        "midi_count": _int(aggregate.get("midi_count")),
        "wav_count": _int(aggregate.get("wav_count")),
        "quality_proxy_pass_count": _int(aggregate.get("quality_proxy_pass_count")),
        "quality_proxy_fail_count": _int(aggregate.get("quality_proxy_fail_count")),
        "major_label_counts": _dict(aggregate.get("major_label_counts")),
        "watch_label_counts": _dict(aggregate.get("watch_label_counts")),
        "output_dir": str(report.get("output_dir") or ""),
    }


def validate_input_guard(report: dict[str, Any], *, expected_candidate_count: int) -> dict[str, Any]:
    if str(report.get("schema_version") or "") != INPUT_GUARD_SCHEMA_VERSION:
        raise SoloYieldResidualAwareMvpHandoffFreezeError("input guard schema mismatch")
    guard = _dict(report.get("guard_result"))
    readiness = _dict(report.get("readiness"))
    if not bool(readiness.get("residual_aware_listening_input_guard_completed", False)):
        raise SoloYieldResidualAwareMvpHandoffFreezeError("input guard completion required")
    if _int(guard.get("review_item_count")) != expected_candidate_count:
        raise SoloYieldResidualAwareMvpHandoffFreezeError("review item count mismatch")
    _require_no_quality_claim(readiness, label="input guard readiness")
    return {
        "review_item_count": _int(guard.get("review_item_count")),
        "validated_listening_input_present": bool(guard.get("validated_listening_input_present", True)),
        "preference_fill_allowed": bool(guard.get("preference_fill_allowed", True)),
        "listening_review_completed": bool(readiness.get("listening_review_completed", True)),
        "output_dir": str(report.get("output_dir") or ""),
    }


def _handoff_support_paths(final_review_package: dict[str, Any]) -> dict[str, str]:
    output_dir = Path(str(final_review_package.get("output_dir") or ""))
    return {
        "final_review_package_md": str(output_dir / "residual_aware_final_review_package.md"),
        "final_review_package_json": str(output_dir / "residual_aware_final_review_package.json"),
        "review_input_template_json": str(output_dir / "residual_aware_review_input_template.json"),
    }


def build_candidate_manifest(final_review_package: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate in _list(final_review_package.get("candidate_handoff")):
        item = _dict(candidate)
        midi_path = _require_file(item.get("review_midi_path"), label="candidate MIDI")
        wav_path = _require_file(item.get("review_wav_path"), label="candidate WAV")
        midi_sha = str(item.get("review_midi_sha256") or "")
        wav_sha = str(item.get("review_wav_sha256") or "")
        _require_checksum(midi_path, midi_sha, label="MIDI")
        _require_checksum(wav_path, wav_sha, label="WAV")
        rows.append(
            {
                "review_index": _int(item.get("review_index")),
                "case_label": str(item.get("case_label") or ""),
                "sample_index": _int(item.get("sample_index")),
                "sample_seed": _int(item.get("sample_seed")),
                "review_midi_path": str(midi_path),
                "review_wav_path": str(wav_path),
                "review_midi_sha256": midi_sha,
                "review_wav_sha256": wav_sha,
                "duration_seconds": _float(item.get("duration_seconds")),
                "sample_rate": _int(item.get("sample_rate")),
                "quality_proxy_pass": bool(item.get("quality_proxy_pass", False)),
                "rubric_major_labels": _list(item.get("rubric_major_labels")),
                "rubric_watch_labels": _list(item.get("rubric_watch_labels")),
            }
        )
    return rows


def _verify_support_paths(paths: dict[str, str]) -> dict[str, Any]:
    missing = [path for path in paths.values() if not Path(path).exists()]
    return {
        "support_file_count": len(paths),
        "missing_support_files": missing,
        "support_files_verified": not missing,
    }


def build_handoff_freeze_report(
    *,
    final_review_package: dict[str, Any],
    input_guard_report: dict[str, Any],
    status_audit_report: dict[str, Any],
    output_dir: Path,
    issue_number: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    status = validate_status_audit(status_audit_report)
    final_summary = validate_final_review_package(final_review_package)
    guard_summary = validate_input_guard(
        input_guard_report,
        expected_candidate_count=int(final_summary["candidate_count"]),
    )
    candidates = build_candidate_manifest(final_review_package)
    support_paths = _handoff_support_paths(final_review_package)
    support_validation = _verify_support_paths(support_paths)
    counts_match = bool(
        status["candidate_count"]
        == final_summary["candidate_count"]
        == guard_summary["review_item_count"]
        == len(candidates)
        and status["midi_count"] == final_summary["midi_count"]
        and status["wav_count"] == final_summary["wav_count"]
    )
    pending_input = not bool(guard_summary["validated_listening_input_present"])
    preference_blocked = not bool(guard_summary["preference_fill_allowed"])
    local_artifacts_verified = bool(
        counts_match
        and support_validation["support_files_verified"]
        and len(candidates) == int(final_summary["candidate_count"])
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "issue_number": int(issue_number),
        "source_reports": {
            "final_review_package": final_review_package.get("output_dir"),
            "input_guard_report": input_guard_report.get("output_dir"),
            "status_audit_report": status_audit_report.get("output_dir"),
        },
        "artifact_paths": support_paths,
        "aggregate": {
            "candidate_count": len(candidates),
            "midi_count": final_summary["midi_count"],
            "wav_count": final_summary["wav_count"],
            "review_item_count": guard_summary["review_item_count"],
            "quality_proxy_pass_count": final_summary["quality_proxy_pass_count"],
            "quality_proxy_fail_count": final_summary["quality_proxy_fail_count"],
            "major_label_counts": final_summary["major_label_counts"],
            "watch_label_counts": final_summary["watch_label_counts"],
            "missing_file_count": 0,
            "checksum_mismatch_count": 0,
            "missing_support_file_count": len(support_validation["missing_support_files"]),
            "counts_match": counts_match,
            "raw_artifact_upload_required": False,
        },
        "candidate_manifest": candidates,
        "support_file_validation": support_validation,
        "readiness": {
            "residual_aware_mvp_handoff_freeze_completed": True,
            "local_candidate_artifacts_verified": local_artifacts_verified,
            "review_input_template_available": Path(
                support_paths["review_input_template_json"]
            ).exists(),
            "validated_listening_input_present": pending_input is False,
            "preference_fill_allowed": preference_blocked is False,
            "listening_review_completed": bool(guard_summary["listening_review_completed"]),
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
            "reason": "local MVP candidate handoff frozen; listening review input remains pending",
        },
        "not_proven": [
            "listening_review_completed",
            "human_audio_preference",
            "midi_to_solo_musical_quality",
            "stable_jazz_solo_quality",
            "production_ready_improviser",
        ],
    }
    write_json(output_dir / "residual_aware_mvp_handoff_freeze.json", report)
    write_json(
        output_dir / "residual_aware_mvp_handoff_freeze_summary.json",
        validate_handoff_freeze_report(report),
    )
    write_text(output_dir / "residual_aware_mvp_handoff_freeze.md", markdown_report(report))
    return report


def validate_handoff_freeze_report(
    report: dict[str, Any],
    *,
    expected_next_boundary: str | None = None,
    require_local_artifacts_verified: bool = False,
    require_pending_input: bool = False,
    require_no_quality_claim: bool = False,
) -> dict[str, Any]:
    if str(report.get("schema_version") or "") != SCHEMA_VERSION:
        raise SoloYieldResidualAwareMvpHandoffFreezeError("handoff freeze schema mismatch")
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    aggregate = _dict(report.get("aggregate"))
    if not bool(readiness.get("residual_aware_mvp_handoff_freeze_completed", False)):
        raise SoloYieldResidualAwareMvpHandoffFreezeError("handoff freeze completion required")
    if expected_next_boundary and str(decision.get("next_boundary") or "") != expected_next_boundary:
        raise SoloYieldResidualAwareMvpHandoffFreezeError("unexpected next boundary")
    if require_local_artifacts_verified and not bool(
        readiness.get("local_candidate_artifacts_verified", False)
    ):
        raise SoloYieldResidualAwareMvpHandoffFreezeError("local candidate artifacts verification required")
    if require_pending_input and bool(readiness.get("validated_listening_input_present", True)):
        raise SoloYieldResidualAwareMvpHandoffFreezeError("validated listening input should remain pending")
    if require_pending_input and bool(readiness.get("preference_fill_allowed", True)):
        raise SoloYieldResidualAwareMvpHandoffFreezeError("preference fill should remain blocked")
    if require_pending_input and bool(readiness.get("listening_review_completed", True)):
        raise SoloYieldResidualAwareMvpHandoffFreezeError("listening review should remain pending")
    if require_no_quality_claim:
        _require_no_quality_claim(readiness, label="handoff freeze readiness")
    if bool(decision.get("critical_user_input_required", True)):
        raise SoloYieldResidualAwareMvpHandoffFreezeError("critical user input should not be required")
    return {
        "schema_version": str(report.get("schema_version")),
        "local_candidate_artifacts_verified": bool(
            readiness.get("local_candidate_artifacts_verified", False)
        ),
        "candidate_count": _int(aggregate.get("candidate_count")),
        "midi_count": _int(aggregate.get("midi_count")),
        "wav_count": _int(aggregate.get("wav_count")),
        "quality_proxy_pass_count": _int(aggregate.get("quality_proxy_pass_count")),
        "quality_proxy_fail_count": _int(aggregate.get("quality_proxy_fail_count")),
        "missing_file_count": _int(aggregate.get("missing_file_count")),
        "checksum_mismatch_count": _int(aggregate.get("checksum_mismatch_count")),
        "missing_support_file_count": _int(aggregate.get("missing_support_file_count")),
        "validated_listening_input_present": bool(
            readiness.get("validated_listening_input_present", True)
        ),
        "preference_fill_allowed": bool(readiness.get("preference_fill_allowed", True)),
        "listening_review_completed": bool(readiness.get("listening_review_completed", True)),
        "midi_to_solo_musical_quality_claimed": bool(
            readiness.get("midi_to_solo_musical_quality_claimed", True)
        ),
        "raw_artifact_upload_required": bool(aggregate.get("raw_artifact_upload_required", True)),
        "next_boundary": str(decision.get("next_boundary") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    aggregate = report["aggregate"]
    readiness = report["readiness"]
    decision = report["decision"]
    lines = [
        "# Music Transformer Solo Yield Residual-Aware MVP Handoff Freeze",
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
        f"- local candidate artifacts verified: `{_bool_token(readiness['local_candidate_artifacts_verified'])}`",
        f"- missing file count: `{aggregate['missing_file_count']}`",
        f"- checksum mismatch count: `{aggregate['checksum_mismatch_count']}`",
        f"- review input template available: `{_bool_token(readiness['review_input_template_available'])}`",
        f"- validated listening input present: `{_bool_token(readiness['validated_listening_input_present'])}`",
        f"- preference fill allowed: `{_bool_token(readiness['preference_fill_allowed'])}`",
        f"- listening review completed: `{_bool_token(readiness['listening_review_completed'])}`",
        f"- musical quality claimed: `{_bool_token(readiness['midi_to_solo_musical_quality_claimed'])}`",
        f"- raw artifact upload required: `{_bool_token(aggregate['raw_artifact_upload_required'])}`",
        f"- next boundary: `{decision['next_boundary']}`",
        "",
        "## Artifact Paths",
        "",
    ]
    for name, path in sorted(report["artifact_paths"].items()):
        lines.append(f"- {name}: `{path}`")
    lines.extend(["", "## Not Proven", ""])
    for item in report.get("not_proven", []):
        lines.append(f"- `{item}`")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Freeze residual-aware local MVP handoff")
    parser.add_argument("--final_review_package", type=str, required=True)
    parser.add_argument("--input_guard_report", type=str, required=True)
    parser.add_argument("--status_audit_report", type=str, required=True)
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/music_transformer_finetune_mvp/solo_yield_residual_aware_mvp_handoff_freeze",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--issue_number", type=int, default=0)
    parser.add_argument("--expected_next_boundary", type=str, default=NEXT_BOUNDARY)
    parser.add_argument("--require_local_artifacts_verified", action="store_true")
    parser.add_argument("--require_pending_input", action="store_true")
    parser.add_argument("--require_no_quality_claim", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    report = build_handoff_freeze_report(
        final_review_package=read_json(Path(args.final_review_package)),
        input_guard_report=read_json(Path(args.input_guard_report)),
        status_audit_report=read_json(Path(args.status_audit_report)),
        output_dir=output_dir,
        issue_number=int(args.issue_number),
    )
    summary = validate_handoff_freeze_report(
        report,
        expected_next_boundary=str(args.expected_next_boundary),
        require_local_artifacts_verified=bool(args.require_local_artifacts_verified),
        require_pending_input=bool(args.require_pending_input),
        require_no_quality_claim=bool(args.require_no_quality_claim),
    )
    if args.doc_path:
        write_text(Path(args.doc_path), markdown_report(report))
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
