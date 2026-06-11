"""Guard listening input for chord-tone landing repaired solo-yield candidates."""

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
from scripts.build_music_transformer_solo_yield_chord_tone_landing_listening_package import (  # noqa: E402
    SCHEMA_VERSION as SOURCE_PACKAGE_SCHEMA_VERSION,
    build_listening_package,
    load_or_build_audio_package,
)


SCHEMA_VERSION = "music_transformer_solo_yield_chord_tone_landing_repair_listening_input_guard_v1"
INPUT_SCHEMA_VERSION = "music_transformer_solo_yield_chord_tone_landing_listening_input_v1"
BOUNDARY = "music_transformer_solo_yield_chord_tone_landing_repair_listening_input_guard"
NEXT_BOUNDARY_PENDING = "music_transformer_solo_yield_chord_tone_landing_repair_objective_only_next_decision"
NEXT_BOUNDARY_REVIEWED = "music_transformer_solo_yield_chord_tone_landing_repair_listening_review_fill"


class SoloYieldChordToneLandingListeningInputGuardError(ValueError):
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


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SoloYieldChordToneLandingListeningInputGuardError(f"json not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _require_no_quality_claim(readiness: dict[str, Any]) -> None:
    claimed = [
        key
        for key in (
            "audio_rendered_quality_claimed",
            "musical_quality_claimed",
            "artist_style_claimed",
            "production_ready_claimed",
        )
        if bool(readiness.get(key, False))
    ]
    if claimed:
        raise SoloYieldChordToneLandingListeningInputGuardError(f"unexpected quality claim: {claimed}")


def validate_package(package_report: dict[str, Any]) -> dict[str, Any]:
    if str(package_report.get("schema_version") or "") != SOURCE_PACKAGE_SCHEMA_VERSION:
        raise SoloYieldChordToneLandingListeningInputGuardError("source package schema required")
    readiness = _dict(package_report.get("readiness"))
    _require_no_quality_claim(readiness)
    candidate_count = _int(package_report.get("candidate_count"))
    if candidate_count <= 0:
        raise SoloYieldChordToneLandingListeningInputGuardError("candidate count required")
    if not bool(readiness.get("listening_package_ready", False)):
        raise SoloYieldChordToneLandingListeningInputGuardError("source package readiness required")
    if _int(readiness.get("candidate_midi_files_copied")) != candidate_count:
        raise SoloYieldChordToneLandingListeningInputGuardError("MIDI copy count mismatch")
    if _int(readiness.get("candidate_wav_files_copied")) != candidate_count:
        raise SoloYieldChordToneLandingListeningInputGuardError("WAV copy count mismatch")
    if not bool(readiness.get("review_input_template_written", False)):
        raise SoloYieldChordToneLandingListeningInputGuardError("review input template required")
    return {
        "candidate_count": candidate_count,
        "output_dir": str(package_report.get("output_dir") or ""),
        "validated_listening_input_present": bool(readiness.get("validated_listening_input_present", False)),
        "preference_fill_allowed": bool(readiness.get("preference_fill_allowed", False)),
    }


def pending_candidate_fields(candidate: dict[str, Any]) -> list[str]:
    pending: list[str] = []
    if str(candidate.get("decision") or "pending") == "pending":
        pending.append("decision")
    if candidate.get("usable_as_jazz_solo_phrase") is None:
        pending.append("usable_as_jazz_solo_phrase")
    if candidate.get("primary_failure") is None:
        pending.append("primary_failure")
    return pending


def validate_input(input_report: dict[str, Any], *, candidate_count: int) -> dict[str, Any]:
    schema_matched = str(input_report.get("schema_version") or "") == INPUT_SCHEMA_VERSION
    candidates = [_dict(row) for row in _list(input_report.get("candidates"))]
    pending_status = str(input_report.get("review_status") or "pending") != "reviewed"
    pending_overall = str(input_report.get("overall_decision") or "pending") == "pending"
    pending_fields = {
        str(_int(row.get("review_index"))): pending_candidate_fields(row)
        for row in candidates
        if pending_candidate_fields(row)
    }
    candidate_count_matched = len(candidates) == int(candidate_count)
    validated = bool(
        schema_matched
        and candidate_count_matched
        and not pending_status
        and not pending_overall
        and not pending_fields
    )
    return {
        "schema_matched": schema_matched,
        "validated_listening_input_present": validated,
        "candidate_count_matched": candidate_count_matched,
        "input_candidate_count": len(candidates),
        "expected_candidate_count": int(candidate_count),
        "pending_status": pending_status,
        "pending_overall_decision": pending_overall,
        "pending_candidate_field_count": sum(len(fields) for fields in pending_fields.values()),
        "pending_candidate_fields": pending_fields,
    }


def load_or_build_package(
    *,
    package_report_path: Path,
    output_dir: Path,
    audio_package_report_path: Path,
    repair_sweep_report_path: Path,
    source_package_report_path: Path,
    renderer_path: str,
    soundfont_path: str,
    sample_rate: int,
    min_candidates: int,
) -> dict[str, Any]:
    if package_report_path.exists():
        return read_json(package_report_path)
    audio_package = load_or_build_audio_package(
        audio_package_report_path=audio_package_report_path,
        output_dir=output_dir / "source_audio_build",
        repair_sweep_report_path=repair_sweep_report_path,
        package_report_path=source_package_report_path,
        renderer_path=renderer_path,
        soundfont_path=soundfont_path,
        sample_rate=int(sample_rate),
    )
    return build_listening_package(
        audio_package,
        output_dir=output_dir / "source_listening_package",
        min_candidates=int(min_candidates),
    )


def build_guard_report(
    package_report: dict[str, Any],
    *,
    output_dir: Path,
    listening_input_path: Path | None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    package_validation = validate_package(package_report)
    default_input_path = Path(package_validation["output_dir"]) / "listening_review_input_template.json"
    input_path = listening_input_path or default_input_path
    input_validation = validate_input(
        read_json(input_path),
        candidate_count=_int(package_validation["candidate_count"]),
    )
    preference_fill_allowed = bool(input_validation["validated_listening_input_present"])
    readiness = {
        "listening_input_guard_completed": True,
        "source_package_ready": True,
        "validated_listening_input_present": preference_fill_allowed,
        "preference_fill_allowed": preference_fill_allowed,
        "objective_only_next_decision_required": not preference_fill_allowed,
        "audio_rendered_quality_claimed": False,
        "musical_quality_claimed": False,
        "artist_style_claimed": False,
        "production_ready_claimed": False,
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "boundary": BOUNDARY,
        "source_package": {
            "schema_version": package_report.get("schema_version"),
            "output_dir": package_validation["output_dir"],
            "candidate_count": package_validation["candidate_count"],
            "source_validated_listening_input_present": package_validation[
                "validated_listening_input_present"
            ],
            "source_preference_fill_allowed": package_validation["preference_fill_allowed"],
        },
        "listening_input_path": str(input_path),
        "input_validation": input_validation,
        "readiness": readiness,
        "decision": {
            "current_boundary": BOUNDARY,
            "next_boundary": NEXT_BOUNDARY_REVIEWED if preference_fill_allowed else NEXT_BOUNDARY_PENDING,
            "critical_user_input_required": False,
            "reason": (
                "validated listening input present"
                if preference_fill_allowed
                else "listening input pending; route to objective-only next decision without quality claim"
            ),
        },
        "not_proven": [
            "human_audio_preference",
            "stable_jazz_solo_quality",
            "artist_level_long_solo_generation",
            "production_ready_improviser",
        ],
    }
    write_json(output_dir / "chord_tone_landing_repair_listening_input_guard.json", report)
    write_json(
        output_dir / "chord_tone_landing_repair_listening_input_guard_summary.json",
        validate_report(report, require_no_quality_claim=True),
    )
    write_text(output_dir / "chord_tone_landing_repair_listening_input_guard.md", markdown_report(report))
    return report


def validate_report(report: dict[str, Any], *, require_no_quality_claim: bool) -> dict[str, Any]:
    if str(report.get("schema_version") or "") != SCHEMA_VERSION:
        raise SoloYieldChordToneLandingListeningInputGuardError("schema version mismatch")
    readiness = _dict(report.get("readiness"))
    if require_no_quality_claim:
        _require_no_quality_claim(readiness)
    validation = _dict(report.get("input_validation"))
    decision = _dict(report.get("decision"))
    return {
        "schema_version": str(report.get("schema_version") or ""),
        "boundary": str(report.get("boundary") or ""),
        "candidate_count": _int(_dict(report.get("source_package")).get("candidate_count")),
        "schema_matched": bool(validation.get("schema_matched", False)),
        "validated_listening_input_present": bool(validation.get("validated_listening_input_present", False)),
        "preference_fill_allowed": bool(readiness.get("preference_fill_allowed", True)),
        "objective_only_next_decision_required": bool(
            readiness.get("objective_only_next_decision_required", False)
        ),
        "pending_status": bool(validation.get("pending_status", False)),
        "pending_overall_decision": bool(validation.get("pending_overall_decision", False)),
        "pending_candidate_field_count": _int(validation.get("pending_candidate_field_count")),
        "musical_quality_claimed": bool(readiness.get("musical_quality_claimed", True)),
        "next_boundary": str(decision.get("next_boundary") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    validation = report["input_validation"]
    readiness = report["readiness"]
    decision = report["decision"]
    lines = [
        "# Music Transformer Solo Yield Chord-Tone Landing Repair Listening Input Guard",
        "",
        "## Summary",
        "",
        f"- boundary: `{report['boundary']}`",
        f"- source candidate count: `{report['source_package']['candidate_count']}`",
        f"- listening input path: `{report['listening_input_path']}`",
        f"- schema matched: `{_bool_token(validation['schema_matched'])}`",
        f"- validated listening input present: `{_bool_token(validation['validated_listening_input_present'])}`",
        f"- preference fill allowed: `{_bool_token(readiness['preference_fill_allowed'])}`",
        f"- objective-only next decision required: `{_bool_token(readiness['objective_only_next_decision_required'])}`",
        f"- pending status: `{_bool_token(validation['pending_status'])}`",
        f"- pending overall decision: `{_bool_token(validation['pending_overall_decision'])}`",
        f"- pending candidate field count: `{validation['pending_candidate_field_count']}`",
        f"- musical quality claimed: `{_bool_token(readiness['musical_quality_claimed'])}`",
        f"- next boundary: `{decision['next_boundary']}`",
        "",
        "## Pending Candidate Fields",
        "",
    ]
    pending = _dict(validation.get("pending_candidate_fields"))
    if pending:
        for review_index, fields in pending.items():
            lines.append(f"- review `{review_index}`: `{','.join(str(field) for field in fields)}`")
    else:
        lines.append("- `none`")
    lines.extend(["", "## Not Proven", ""])
    for item in report.get("not_proven", []):
        lines.append(f"- `{item}`")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Guard chord-tone landing repair listening input")
    parser.add_argument(
        "--package_report",
        type=str,
        default=(
            "outputs/music_transformer_finetune_mvp/solo_yield_chord_tone_landing_repair_listening_review/"
            "issue_1268_chord_tone_landing_listening_package/listening_review_package.json"
        ),
    )
    parser.add_argument(
        "--audio_package_report",
        type=str,
        default=(
            "outputs/music_transformer_finetune_mvp/solo_yield_chord_tone_landing_repair_audio/"
            "issue_1266_chord_tone_landing_audio_package/chord_tone_landing_repair_audio_package.json"
        ),
    )
    parser.add_argument(
        "--repair_sweep_report",
        type=str,
        default=(
            "outputs/music_transformer_finetune_mvp/solo_yield_chord_tone_landing_repair/"
            "issue_1264_chord_tone_landing_repair/chord_tone_landing_repair_sweep.json"
        ),
    )
    parser.add_argument(
        "--source_package_report",
        type=str,
        default=(
            "outputs/music_transformer_finetune_mvp/solo_yield_listening_review/"
            "issue_1250_4bar_repaired_top8_listening_package/listening_review_package.json"
        ),
    )
    parser.add_argument("--listening_input", type=str, default="")
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/music_transformer_finetune_mvp/solo_yield_chord_tone_landing_repair_listening_input_guard",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--renderer", type=str, default="")
    parser.add_argument("--soundfont", type=str, default="")
    parser.add_argument("--sample_rate", type=int, default=44100)
    parser.add_argument("--min_candidates", type=int, default=8)
    parser.add_argument("--require_no_quality_claim", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    package_report = load_or_build_package(
        package_report_path=Path(args.package_report),
        output_dir=output_dir,
        audio_package_report_path=Path(args.audio_package_report),
        repair_sweep_report_path=Path(args.repair_sweep_report),
        source_package_report_path=Path(args.source_package_report),
        renderer_path=str(args.renderer or ""),
        soundfont_path=str(args.soundfont or ""),
        sample_rate=int(args.sample_rate),
        min_candidates=int(args.min_candidates),
    )
    listening_input_path = Path(args.listening_input) if args.listening_input else None
    report = build_guard_report(
        package_report,
        output_dir=output_dir,
        listening_input_path=listening_input_path,
    )
    summary = validate_report(report, require_no_quality_claim=bool(args.require_no_quality_claim))
    if args.doc_path:
        write_text(Path(args.doc_path), markdown_report(report))
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
