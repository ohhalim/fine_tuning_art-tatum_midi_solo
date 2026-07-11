"""Guard residual-aware solo-yield review fill against missing listening input."""

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


SOURCE_SCHEMA_VERSION = "music_transformer_solo_yield_residual_aware_final_review_package_v1"
SCHEMA_VERSION = "music_transformer_solo_yield_residual_aware_listening_input_guard_v1"
BOUNDARY = "music_transformer_solo_yield_residual_aware_listening_input_guard"
FILL_BOUNDARY = "music_transformer_solo_yield_residual_aware_listening_review_fill"
STATUS_SYNC_BOUNDARY = "music_transformer_solo_yield_residual_aware_status_sync"


class SoloYieldResidualAwareListeningInputGuardError(ValueError):
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
        raise SoloYieldResidualAwareListeningInputGuardError(f"json not found: {path}")
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
        raise SoloYieldResidualAwareListeningInputGuardError(
            f"unexpected quality claim in {label}: {claimed}"
        )


def validate_source_package(report: dict[str, Any]) -> dict[str, Any]:
    if str(report.get("schema_version") or "") != SOURCE_SCHEMA_VERSION:
        raise SoloYieldResidualAwareListeningInputGuardError("source schema mismatch")
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    aggregate = _dict(report.get("aggregate"))
    if not bool(readiness.get("residual_aware_final_review_package_ready", False)):
        raise SoloYieldResidualAwareListeningInputGuardError("final review package readiness required")
    if _int(aggregate.get("candidate_count")) <= 0:
        raise SoloYieldResidualAwareListeningInputGuardError("candidate count required")
    if _int(aggregate.get("midi_count")) != _int(aggregate.get("candidate_count")):
        raise SoloYieldResidualAwareListeningInputGuardError("MIDI count mismatch")
    if _int(aggregate.get("wav_count")) != _int(aggregate.get("candidate_count")):
        raise SoloYieldResidualAwareListeningInputGuardError("WAV count mismatch")
    if str(decision.get("next_boundary") or "") != BOUNDARY:
        raise SoloYieldResidualAwareListeningInputGuardError("source must route to listening input guard")
    if bool(decision.get("critical_user_input_required", True)):
        raise SoloYieldResidualAwareListeningInputGuardError("critical user input should not be required")
    _require_no_quality_claim(readiness, label="source readiness")
    return {
        "candidate_count": _int(aggregate.get("candidate_count")),
        "midi_count": _int(aggregate.get("midi_count")),
        "wav_count": _int(aggregate.get("wav_count")),
        "quality_proxy_pass_count": _int(aggregate.get("quality_proxy_pass_count")),
        "quality_proxy_fail_count": _int(aggregate.get("quality_proxy_fail_count")),
        "major_label_counts": _dict(aggregate.get("major_label_counts")),
        "watch_label_counts": _dict(aggregate.get("watch_label_counts")),
        "validated_listening_input_present": bool(
            readiness.get("validated_listening_input_present", False)
        ),
        "review_input_template_written": bool(readiness.get("review_input_template_written", False)),
        "review_item_count": len(_list(report.get("candidate_handoff"))),
    }


def build_guard_report(
    source_package: dict[str, Any],
    *,
    output_dir: Path,
    issue_number: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    source = validate_source_package(source_package)
    validated_input = bool(source["validated_listening_input_present"])
    next_boundary = FILL_BOUNDARY if validated_input else STATUS_SYNC_BOUNDARY
    reason = (
        "validated listening input present; preference fill allowed"
        if validated_input
        else "listening input pending; preference fill blocked"
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "issue_number": int(issue_number),
        "boundary": BOUNDARY,
        "source_package_summary": source,
        "guard_result": {
            "validated_listening_input_present": validated_input,
            "preference_fill_allowed": validated_input,
            "review_item_count": int(source["review_item_count"]),
            "missing_validated_input_reason": "" if validated_input else "validated_listening_input_present=false",
        },
        "readiness": {
            "residual_aware_listening_input_guard_completed": True,
            "validated_listening_input_present": validated_input,
            "preference_fill_allowed": validated_input,
            "listening_review_completed": False,
            "human_audio_preference_claimed": False,
            "midi_to_solo_musical_quality_claimed": False,
            "musical_quality_claimed": False,
            "artist_style_claimed": False,
            "production_ready_claimed": False,
        },
        "decision": {
            "current_boundary": BOUNDARY,
            "next_boundary": next_boundary,
            "critical_user_input_required": False,
            "reason": reason,
        },
        "not_proven": [
            "listening_review_completed",
            "human_audio_preference",
            "midi_to_solo_musical_quality",
            "production_ready_improviser",
        ],
    }
    write_json(output_dir / "residual_aware_listening_input_guard.json", report)
    write_text(output_dir / "residual_aware_listening_input_guard.md", markdown_report(report))
    return report


def validate_guard_report(
    report: dict[str, Any],
    *,
    expected_next_boundary: str,
    require_pending_input: bool,
    require_no_quality_claim: bool,
) -> dict[str, Any]:
    if str(report.get("schema_version") or "") != SCHEMA_VERSION:
        raise SoloYieldResidualAwareListeningInputGuardError("schema version mismatch")
    guard = _dict(report.get("guard_result"))
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    if str(decision.get("next_boundary") or "") != str(expected_next_boundary):
        raise SoloYieldResidualAwareListeningInputGuardError("unexpected next boundary")
    if require_pending_input and bool(guard.get("validated_listening_input_present", True)):
        raise SoloYieldResidualAwareListeningInputGuardError("listening input should remain pending")
    if require_no_quality_claim:
        _require_no_quality_claim(readiness, label="guard readiness")
    if bool(decision.get("critical_user_input_required", True)):
        raise SoloYieldResidualAwareListeningInputGuardError("critical user input should not be required")
    return {
        "schema_version": str(report.get("schema_version")),
        "validated_listening_input_present": bool(guard.get("validated_listening_input_present", True)),
        "preference_fill_allowed": bool(guard.get("preference_fill_allowed", True)),
        "review_item_count": _int(guard.get("review_item_count")),
        "listening_review_completed": bool(readiness.get("listening_review_completed", True)),
        "human_audio_preference_claimed": bool(readiness.get("human_audio_preference_claimed", True)),
        "midi_to_solo_musical_quality_claimed": bool(
            readiness.get("midi_to_solo_musical_quality_claimed", True)
        ),
        "critical_user_input_required": bool(decision.get("critical_user_input_required", True)),
        "next_boundary": str(decision.get("next_boundary") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    guard = report["guard_result"]
    source = report["source_package_summary"]
    readiness = report["readiness"]
    decision = report["decision"]
    lines = [
        "# Music Transformer Solo Yield Residual-Aware Listening Input Guard",
        "",
        "## Summary",
        "",
        f"- boundary: `{report['boundary']}`",
        f"- candidate count: `{source['candidate_count']}`",
        f"- MIDI/WAV: `{source['midi_count']}` / `{source['wav_count']}`",
        f"- quality proxy pass/fail: `{source['quality_proxy_pass_count']}` / `{source['quality_proxy_fail_count']}`",
        f"- review item count: `{guard['review_item_count']}`",
        f"- validated listening input present: `{_bool_token(guard['validated_listening_input_present'])}`",
        f"- preference fill allowed: `{_bool_token(guard['preference_fill_allowed'])}`",
        f"- listening review completed: `{_bool_token(readiness['listening_review_completed'])}`",
        f"- human audio preference claimed: `{_bool_token(readiness['human_audio_preference_claimed'])}`",
        f"- musical quality claimed: `{_bool_token(readiness['midi_to_solo_musical_quality_claimed'])}`",
        f"- next boundary: `{decision['next_boundary']}`",
        f"- reason: `{decision['reason']}`",
        "",
        "## Residual Labels",
        "",
        f"- major labels: `{_format_counts(source['major_label_counts'])}`",
        f"- watch labels: `{_format_counts(source['watch_label_counts'])}`",
        "",
        "## Not Proven",
        "",
    ]
    for item in report.get("not_proven", []):
        lines.append(f"- `{item}`")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Guard residual-aware listening input")
    parser.add_argument("--source_package", type=str, required=True)
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/music_transformer_finetune_mvp/solo_yield_residual_aware_listening_input_guard",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--issue_number", type=int, default=0)
    parser.add_argument("--expected_next_boundary", type=str, default=STATUS_SYNC_BOUNDARY)
    parser.add_argument("--require_pending_input", action="store_true")
    parser.add_argument("--require_no_quality_claim", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    report = build_guard_report(
        read_json(Path(args.source_package)),
        output_dir=output_dir,
        issue_number=int(args.issue_number),
    )
    summary = validate_guard_report(
        report,
        expected_next_boundary=str(args.expected_next_boundary),
        require_pending_input=bool(args.require_pending_input),
        require_no_quality_claim=bool(args.require_no_quality_claim),
    )
    write_json(output_dir / "residual_aware_listening_input_guard_summary.json", summary)
    if args.doc_path:
        write_text(Path(args.doc_path), markdown_report(report))
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
