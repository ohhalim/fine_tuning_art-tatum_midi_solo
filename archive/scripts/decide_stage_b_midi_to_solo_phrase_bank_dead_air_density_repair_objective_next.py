"""Select the next step after repaired phrase-bank evidence without listening input."""

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
from scripts.guard_stage_b_midi_to_solo_phrase_bank_dead_air_density_repair_listening_review_input import (  # noqa: E402
    BOUNDARY as INPUT_GUARD_BOUNDARY,
    OBJECTIVE_NEXT_BOUNDARY as INPUT_GUARD_NEXT_BOUNDARY,
)
from scripts.render_stage_b_midi_to_solo_phrase_bank_dead_air_density_repair_audio import (  # noqa: E402
    BOUNDARY as AUDIO_PACKAGE_BOUNDARY,
)
from scripts.run_stage_b_midi_to_solo_phrase_bank_dead_air_density_repair_probe import (  # noqa: E402
    BOUNDARY as REPAIR_PROBE_BOUNDARY,
)


class StageBMidiToSoloPhraseBankDeadAirDensityRepairObjectiveNextError(ValueError):
    pass


BOUNDARY = "stage_b_midi_to_solo_phrase_bank_dead_air_density_repair_objective_only_next_decision"
NEXT_BOUNDARY = "stage_b_midi_to_solo_phrase_bank_cli_mvp_package"
SCHEMA_VERSION = "stage_b_midi_to_solo_phrase_bank_dead_air_density_repair_objective_next_v1"

QUALITY_CLAIM_KEYS = [
    "human_audio_preference_claimed",
    "midi_to_solo_musical_quality_claimed",
    "musical_quality_claimed",
    "audio_rendered_quality_claimed",
    "phrase_bank_musical_quality_claimed",
    "model_checkpoint_generation_quality_claimed",
    "broad_trained_model_quality_claimed",
    "brad_style_adaptation_claimed",
    "production_ready_claimed",
]


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


def _require_no_quality_claim(container: dict[str, Any], *, label: str) -> None:
    claimed = [name for name in QUALITY_CLAIM_KEYS if bool(container.get(name, False))]
    if claimed:
        raise StageBMidiToSoloPhraseBankDeadAirDensityRepairObjectiveNextError(
            f"unexpected quality claim in {label}: {claimed}"
        )


def validate_input_guard_report(report: dict[str, Any]) -> dict[str, Any]:
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    guard = _dict(report.get("guard_result"))
    if str(report.get("boundary") or "") != INPUT_GUARD_BOUNDARY:
        raise StageBMidiToSoloPhraseBankDeadAirDensityRepairObjectiveNextError(
            "repaired input guard boundary required"
        )
    if str(decision.get("next_boundary") or "") != INPUT_GUARD_NEXT_BOUNDARY:
        raise StageBMidiToSoloPhraseBankDeadAirDensityRepairObjectiveNextError(
            "input guard must route to objective-only next decision"
        )
    if not bool(readiness.get("listening_review_input_guard_completed", False)):
        raise StageBMidiToSoloPhraseBankDeadAirDensityRepairObjectiveNextError("input guard completion required")
    if bool(guard.get("validated_review_input_present", True)):
        raise StageBMidiToSoloPhraseBankDeadAirDensityRepairObjectiveNextError(
            "objective-only decision requires pending review input"
        )
    if bool(guard.get("preference_fill_allowed", True)):
        raise StageBMidiToSoloPhraseBankDeadAirDensityRepairObjectiveNextError(
            "preference fill must remain blocked"
        )
    if bool(decision.get("critical_user_input_required", True)):
        raise StageBMidiToSoloPhraseBankDeadAirDensityRepairObjectiveNextError(
            "critical user input should not be required"
        )
    _require_no_quality_claim(readiness, label="input guard readiness")
    return {
        "boundary": INPUT_GUARD_BOUNDARY,
        "review_item_count": _int(guard.get("review_item_count")),
        "validated_review_input_present": bool(guard.get("validated_review_input_present", False)),
        "preference_fill_allowed": bool(guard.get("preference_fill_allowed", False)),
    }


def validate_repair_probe_report(report: dict[str, Any]) -> dict[str, Any]:
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    summary = _dict(report.get("summary"))
    if str(report.get("boundary") or "") != REPAIR_PROBE_BOUNDARY:
        raise StageBMidiToSoloPhraseBankDeadAirDensityRepairObjectiveNextError(
            "dead-air density repair probe boundary required"
        )
    if not bool(readiness.get("dead_air_density_repair_probe_completed", False)):
        raise StageBMidiToSoloPhraseBankDeadAirDensityRepairObjectiveNextError("repair probe completion required")
    if not bool(summary.get("repair_probe_target_passed", False)):
        raise StageBMidiToSoloPhraseBankDeadAirDensityRepairObjectiveNextError("repair target pass required")
    if _int(summary.get("qualified_repaired_candidate_count")) <= 0:
        raise StageBMidiToSoloPhraseBankDeadAirDensityRepairObjectiveNextError(
            "qualified repaired candidates required"
        )
    if bool(decision.get("critical_user_input_required", True)):
        raise StageBMidiToSoloPhraseBankDeadAirDensityRepairObjectiveNextError(
            "critical user input should not be required"
        )
    _require_no_quality_claim(readiness, label="repair probe readiness")
    candidates = [_dict(item) for item in _list(report.get("repaired_candidates"))]
    if not candidates:
        raise StageBMidiToSoloPhraseBankDeadAirDensityRepairObjectiveNextError("repaired candidates required")
    return {
        "boundary": REPAIR_PROBE_BOUNDARY,
        "summary": summary,
        "repaired_candidates": candidates,
    }


def validate_audio_package_report(report: dict[str, Any]) -> dict[str, Any]:
    boundary = _dict(report.get("audio_render_boundary"))
    decision = _dict(report.get("decision"))
    if str(boundary.get("boundary") or "") != AUDIO_PACKAGE_BOUNDARY:
        raise StageBMidiToSoloPhraseBankDeadAirDensityRepairObjectiveNextError(
            "dead-air density repair audio boundary required"
        )
    if not bool(boundary.get("technical_wav_validation", False)):
        raise StageBMidiToSoloPhraseBankDeadAirDensityRepairObjectiveNextError("technical WAV validation required")
    if _int(boundary.get("rendered_audio_file_count")) <= 0:
        raise StageBMidiToSoloPhraseBankDeadAirDensityRepairObjectiveNextError("rendered audio files required")
    if bool(decision.get("critical_user_input_required", True)):
        raise StageBMidiToSoloPhraseBankDeadAirDensityRepairObjectiveNextError(
            "critical user input should not be required"
        )
    _require_no_quality_claim(boundary, label="audio package boundary")
    rendered = [_dict(item) for item in _list(report.get("rendered_audio_files"))]
    if not rendered:
        raise StageBMidiToSoloPhraseBankDeadAirDensityRepairObjectiveNextError("rendered audio rows required")
    return {
        "boundary": AUDIO_PACKAGE_BOUNDARY,
        "technical_wav_validation": bool(boundary.get("technical_wav_validation", False)),
        "rendered_audio_file_count": _int(boundary.get("rendered_audio_file_count")),
        "rendered_audio_files": rendered,
    }


def build_candidate_rows(
    *,
    repair_probe: dict[str, Any],
    audio_package: dict[str, Any],
) -> list[dict[str, Any]]:
    audio_by_seed = {
        _int(item.get("sample_seed")): item for item in _list(audio_package.get("rendered_audio_files"))
    }
    rows: list[dict[str, Any]] = []
    for item in _list(repair_probe.get("repaired_candidates")):
        gate = _dict(item.get("repair_gate"))
        metrics = _dict(item.get("repaired_metrics"))
        seed = _int(item.get("sample_seed"))
        audio = _dict(audio_by_seed.get(seed))
        wav = _dict(audio.get("wav_file"))
        rows.append(
            {
                "rank": _int(item.get("rank")),
                "sample_seed": seed,
                "repaired_midi_path": str(item.get("repaired_midi_path") or ""),
                "wav_path": str(wav.get("path") or ""),
                "objective_supported": bool(gate.get("qualified", False)),
                "repair_flags": [str(flag) for flag in _list(gate.get("flags"))],
                "note_count": _int(metrics.get("note_count")),
                "unique_pitch_count": _int(metrics.get("unique_pitch_count")),
                "dead_air_ratio": _float(metrics.get("dead_air_ratio")),
                "phrase_coverage_ratio": _float(metrics.get("phrase_coverage_ratio")),
                "max_simultaneous_notes": _int(metrics.get("max_simultaneous_notes")),
                "wav_duration_seconds": _float(wav.get("duration_seconds")),
                "wav_sample_rate": _int(wav.get("sample_rate")),
                "wav_sha256_prefix": str(wav.get("sha256") or "")[:12],
            }
        )
    return rows


def build_objective_next_report(
    *,
    input_guard_report: dict[str, Any],
    repair_probe_report: dict[str, Any],
    audio_package_report: dict[str, Any],
    output_dir: Path,
    issue_number: int,
) -> dict[str, Any]:
    input_guard = validate_input_guard_report(input_guard_report)
    repair_probe = validate_repair_probe_report(repair_probe_report)
    audio_package = validate_audio_package_report(audio_package_report)
    candidate_rows = build_candidate_rows(repair_probe=repair_probe, audio_package=audio_package)
    objective_supported_count = sum(1 for item in candidate_rows if bool(item["objective_supported"]))
    dead_air_values = [_float(item.get("dead_air_ratio")) for item in candidate_rows]
    cli_package_ready = (
        objective_supported_count == len(candidate_rows)
        and bool(audio_package["technical_wav_validation"])
        and not bool(input_guard["preference_fill_allowed"])
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "issue_number": int(issue_number),
        "boundary": BOUNDARY,
        "source_boundaries": {
            "input_guard": input_guard["boundary"],
            "repair_probe": repair_probe["boundary"],
            "audio_package": audio_package["boundary"],
        },
        "objective_summary": {
            "candidate_count": len(candidate_rows),
            "objective_supported_candidate_count": objective_supported_count,
            "all_repaired_candidates_objective_supported": objective_supported_count == len(candidate_rows),
            "min_dead_air_ratio": min(dead_air_values) if dead_air_values else 0.0,
            "max_dead_air_ratio": max(dead_air_values) if dead_air_values else 0.0,
            "technical_wav_validation": bool(audio_package["technical_wav_validation"]),
            "rendered_audio_file_count": _int(audio_package["rendered_audio_file_count"]),
            "validated_review_input_present": bool(input_guard["validated_review_input_present"]),
            "preference_fill_allowed": bool(input_guard["preference_fill_allowed"]),
            "cli_mvp_package_ready": bool(cli_package_ready),
        },
        "candidate_reviews": candidate_rows,
        "readiness": {
            "boundary": BOUNDARY,
            "objective_only_next_decision_completed": True,
            "cli_mvp_package_ready": bool(cli_package_ready),
            "human_audio_preference_claimed": False,
            "midi_to_solo_musical_quality_claimed": False,
            "audio_rendered_quality_claimed": False,
            "phrase_bank_musical_quality_claimed": False,
            "model_checkpoint_generation_quality_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
        },
        "decision": {
            "current_boundary": BOUNDARY,
            "next_boundary": NEXT_BOUNDARY,
            "auto_progress_allowed": True,
            "critical_user_input_required": False,
            "reason": "objective-supported repaired MIDI/WAV evidence is ready for CLI MVP packaging without preference claim",
        },
        "not_proven": [
            "listening_review_completed",
            "human_audio_preference",
            "midi_to_solo_musical_quality",
            "phrase_bank_musical_quality",
            "broad_trained_model_quality",
            "brad_style_adaptation",
        ],
        "next_recommended_issue": "Stage B MIDI-to-solo phrase-bank CLI MVP package",
    }


def validate_objective_next_report(
    report: dict[str, Any],
    *,
    expected_boundary: str | None,
    expected_next_boundary: str | None,
    require_objective_decision: bool,
    require_cli_ready: bool,
    require_no_quality_claim: bool,
) -> dict[str, Any]:
    boundary = str(report.get("boundary") or "")
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    summary = _dict(report.get("objective_summary"))
    if expected_boundary and boundary != expected_boundary:
        raise StageBMidiToSoloPhraseBankDeadAirDensityRepairObjectiveNextError(
            f"expected boundary {expected_boundary}, got {boundary}"
        )
    if expected_next_boundary and str(decision.get("next_boundary") or "") != expected_next_boundary:
        raise StageBMidiToSoloPhraseBankDeadAirDensityRepairObjectiveNextError("unexpected next boundary")
    if require_objective_decision and not bool(readiness.get("objective_only_next_decision_completed", False)):
        raise StageBMidiToSoloPhraseBankDeadAirDensityRepairObjectiveNextError(
            "objective-only decision completion required"
        )
    if require_cli_ready and not bool(readiness.get("cli_mvp_package_ready", False)):
        raise StageBMidiToSoloPhraseBankDeadAirDensityRepairObjectiveNextError("CLI package readiness expected")
    if bool(summary.get("preference_fill_allowed", True)):
        raise StageBMidiToSoloPhraseBankDeadAirDensityRepairObjectiveNextError(
            "preference fill must remain blocked"
        )
    if bool(decision.get("critical_user_input_required", True)):
        raise StageBMidiToSoloPhraseBankDeadAirDensityRepairObjectiveNextError(
            "critical user input should not be required"
        )
    if require_no_quality_claim:
        _require_no_quality_claim(readiness, label="objective-next readiness")
    return {
        "boundary": boundary,
        "next_boundary": str(decision.get("next_boundary") or ""),
        "candidate_count": _int(summary.get("candidate_count")),
        "objective_supported_candidate_count": _int(summary.get("objective_supported_candidate_count")),
        "all_repaired_candidates_objective_supported": bool(
            summary.get("all_repaired_candidates_objective_supported", False)
        ),
        "min_dead_air_ratio": _float(summary.get("min_dead_air_ratio")),
        "max_dead_air_ratio": _float(summary.get("max_dead_air_ratio")),
        "technical_wav_validation": bool(summary.get("technical_wav_validation", False)),
        "cli_mvp_package_ready": bool(summary.get("cli_mvp_package_ready", False)),
        "human_audio_preference_claimed": bool(readiness.get("human_audio_preference_claimed", True)),
        "midi_to_solo_musical_quality_claimed": bool(
            readiness.get("midi_to_solo_musical_quality_claimed", True)
        ),
        "critical_user_input_required": bool(decision.get("critical_user_input_required", True)),
        "next_recommended_issue": str(report.get("next_recommended_issue") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    summary = report["objective_summary"]
    readiness = report["readiness"]
    decision = report["decision"]
    lines = [
        "# Stage B MIDI-to-Solo Phrase-Bank Dead-Air Density Repair Objective-Only Next Decision",
        "",
        "## Summary",
        "",
        f"- boundary: `{report['boundary']}`",
        f"- next boundary: `{decision['next_boundary']}`",
        f"- candidate count: `{summary['candidate_count']}`",
        f"- objective supported candidate count: `{summary['objective_supported_candidate_count']}`",
        f"- all repaired candidates objective supported: `{_bool_token(summary['all_repaired_candidates_objective_supported'])}`",
        f"- dead-air range: `{summary['min_dead_air_ratio']:.4f} - {summary['max_dead_air_ratio']:.4f}`",
        f"- technical WAV validation: `{_bool_token(summary['technical_wav_validation'])}`",
        f"- CLI MVP package ready: `{_bool_token(summary['cli_mvp_package_ready'])}`",
        f"- preference fill allowed: `{_bool_token(summary['preference_fill_allowed'])}`",
        f"- human/audio preference claimed: `{_bool_token(readiness['human_audio_preference_claimed'])}`",
        f"- MIDI-to-solo musical quality claimed: `{_bool_token(readiness['midi_to_solo_musical_quality_claimed'])}`",
        "",
        "## Candidate Review",
        "",
    ]
    for item in report["candidate_reviews"]:
        lines.extend(
            [
                f"### Rank {item['rank']}",
                "",
                f"- seed: `{item['sample_seed']}`",
                f"- objective supported: `{_bool_token(item['objective_supported'])}`",
                f"- notes / unique pitches / max simultaneous: `{item['note_count']} / {item['unique_pitch_count']} / {item['max_simultaneous_notes']}`",
                f"- dead-air / phrase coverage: `{item['dead_air_ratio']:.4f} / {item['phrase_coverage_ratio']:.4f}`",
                f"- WAV duration / sample rate / sha256 prefix: `{item['wav_duration_seconds']:.3f}s / {item['wav_sample_rate']} / {item['wav_sha256_prefix']}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Decision",
            "",
            f"- auto progress allowed: `{_bool_token(decision['auto_progress_allowed'])}`",
            f"- critical user input required: `{_bool_token(decision['critical_user_input_required'])}`",
            f"- reason: `{decision['reason']}`",
            f"- next recommended issue: `{report['next_recommended_issue']}`",
            "",
            "## Claim Boundary",
            "",
        ]
    )
    for item in report["not_proven"]:
        lines.append(f"- `{item}`")
    lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Decide repaired phrase-bank objective-only next step")
    parser.add_argument("--input_guard_report", type=str, required=True)
    parser.add_argument("--repair_probe_report", type=str, required=True)
    parser.add_argument("--audio_package_report", type=str, required=True)
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/stage_b_midi_to_solo_phrase_bank_dead_air_density_repair_objective_only_next_decision",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--issue_number", type=int, default=650)
    parser.add_argument("--expected_boundary", type=str, default="")
    parser.add_argument("--expected_next_boundary", type=str, default="")
    parser.add_argument("--require_objective_decision", action="store_true")
    parser.add_argument("--require_cli_ready", action="store_true")
    parser.add_argument("--require_no_quality_claim", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    report = build_objective_next_report(
        input_guard_report=read_json(Path(args.input_guard_report)),
        repair_probe_report=read_json(Path(args.repair_probe_report)),
        audio_package_report=read_json(Path(args.audio_package_report)),
        output_dir=output_dir,
        issue_number=int(args.issue_number),
    )
    summary = validate_objective_next_report(
        report,
        expected_boundary=str(args.expected_boundary or ""),
        expected_next_boundary=str(args.expected_next_boundary or ""),
        require_objective_decision=bool(args.require_objective_decision),
        require_cli_ready=bool(args.require_cli_ready),
        require_no_quality_claim=bool(args.require_no_quality_claim),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        output_dir / "stage_b_midi_to_solo_phrase_bank_dead_air_density_repair_objective_only_next_decision.json",
        report,
    )
    write_json(
        output_dir
        / "stage_b_midi_to_solo_phrase_bank_dead_air_density_repair_objective_only_next_decision_validation_summary.json",
        summary,
    )
    markdown = markdown_report(report)
    write_text(
        output_dir / "stage_b_midi_to_solo_phrase_bank_dead_air_density_repair_objective_only_next_decision.md",
        markdown,
    )
    if args.doc_path:
        write_text(Path(args.doc_path), markdown)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
