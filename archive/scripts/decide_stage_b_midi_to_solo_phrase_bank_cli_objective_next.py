"""Select the next step after CLI phrase-bank evidence without listening input."""

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
from scripts.check_stage_b_midi_to_solo_phrase_bank_cli_user_input_smoke import (  # noqa: E402
    BOUNDARY as USER_INPUT_BOUNDARY,
    NEXT_BOUNDARY as USER_INPUT_NEXT_BOUNDARY,
)
from scripts.guard_stage_b_midi_to_solo_phrase_bank_cli_listening_review_input import (  # noqa: E402
    BOUNDARY as INPUT_GUARD_BOUNDARY,
    OBJECTIVE_NEXT_BOUNDARY as INPUT_GUARD_NEXT_BOUNDARY,
)
from scripts.render_stage_b_midi_to_solo_phrase_bank_cli_audio_smoke import (  # noqa: E402
    BOUNDARY as AUDIO_RENDER_BOUNDARY,
    NEXT_BOUNDARY as AUDIO_RENDER_NEXT_BOUNDARY,
)


class StageBMidiToSoloPhraseBankCliObjectiveNextError(ValueError):
    pass


BOUNDARY = "stage_b_midi_to_solo_phrase_bank_cli_objective_only_next_decision"
NEXT_BOUNDARY = "stage_b_midi_to_solo_mvp_current_evidence_consolidation"
SCHEMA_VERSION = "stage_b_midi_to_solo_phrase_bank_cli_objective_next_v1"

QUALITY_CLAIM_KEYS = [
    "human_audio_preference_claimed",
    "midi_to_solo_musical_quality_claimed",
    "musical_quality_claimed",
    "audio_rendered_quality_claimed",
    "phrase_bank_musical_quality_claimed",
    "model_checkpoint_generation_quality_claimed",
    "model_direct_generation_quality_claimed",
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
        raise StageBMidiToSoloPhraseBankCliObjectiveNextError(
            f"unexpected quality claim in {label}: {claimed}"
        )


def validate_input_guard_report(report: dict[str, Any]) -> dict[str, Any]:
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    guard = _dict(report.get("guard_result"))
    if str(report.get("boundary") or "") != INPUT_GUARD_BOUNDARY:
        raise StageBMidiToSoloPhraseBankCliObjectiveNextError("CLI input guard boundary required")
    if str(decision.get("next_boundary") or "") != INPUT_GUARD_NEXT_BOUNDARY:
        raise StageBMidiToSoloPhraseBankCliObjectiveNextError(
            "input guard must route to CLI objective-only next decision"
        )
    if not bool(readiness.get("listening_review_input_guard_completed", False)):
        raise StageBMidiToSoloPhraseBankCliObjectiveNextError("input guard completion required")
    if bool(guard.get("validated_review_input_present", True)):
        raise StageBMidiToSoloPhraseBankCliObjectiveNextError(
            "objective-only decision requires pending review input"
        )
    if bool(guard.get("preference_fill_allowed", True)):
        raise StageBMidiToSoloPhraseBankCliObjectiveNextError("preference fill must remain blocked")
    if bool(decision.get("critical_user_input_required", True)):
        raise StageBMidiToSoloPhraseBankCliObjectiveNextError("critical user input should not be required")
    _require_no_quality_claim(readiness, label="CLI input guard readiness")
    return {
        "boundary": INPUT_GUARD_BOUNDARY,
        "review_item_count": _int(guard.get("review_item_count")),
        "validated_review_input_present": bool(guard.get("validated_review_input_present", False)),
        "preference_fill_allowed": bool(guard.get("preference_fill_allowed", False)),
    }


def validate_user_input_smoke_report(report: dict[str, Any], *, min_candidate_count: int) -> dict[str, Any]:
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    summary = _dict(report.get("objective_summary"))
    input_info = _dict(report.get("input"))
    candidates = [_dict(item) for item in _list(report.get("candidate_manifest"))]
    if str(report.get("boundary") or "") != USER_INPUT_BOUNDARY:
        raise StageBMidiToSoloPhraseBankCliObjectiveNextError("CLI user-input smoke boundary required")
    if str(decision.get("next_boundary") or "") != USER_INPUT_NEXT_BOUNDARY:
        raise StageBMidiToSoloPhraseBankCliObjectiveNextError(
            "user-input smoke must route to audio render smoke"
        )
    if not bool(readiness.get("user_input_smoke_completed", False)):
        raise StageBMidiToSoloPhraseBankCliObjectiveNextError("user-input smoke completion required")
    if not bool(readiness.get("explicit_input_path_used", False)):
        raise StageBMidiToSoloPhraseBankCliObjectiveNextError("explicit input path required")
    if not bool(readiness.get("ranked_repaired_midi_exported", False)):
        raise StageBMidiToSoloPhraseBankCliObjectiveNextError("ranked repaired MIDI export required")
    if _int(summary.get("candidate_count")) < int(min_candidate_count):
        raise StageBMidiToSoloPhraseBankCliObjectiveNextError("candidate count below threshold")
    if _int(summary.get("repaired_midi_file_count")) < int(min_candidate_count):
        raise StageBMidiToSoloPhraseBankCliObjectiveNextError("repaired MIDI file count below threshold")
    if not bool(summary.get("all_candidates_objective_supported", False)):
        raise StageBMidiToSoloPhraseBankCliObjectiveNextError("objective-supported candidates required")
    if _int(summary.get("input_context_bars")) <= 0:
        raise StageBMidiToSoloPhraseBankCliObjectiveNextError("input context bars required")
    if bool(decision.get("critical_user_input_required", True)):
        raise StageBMidiToSoloPhraseBankCliObjectiveNextError("critical user input should not be required")
    _require_no_quality_claim(readiness, label="CLI user-input readiness")
    return {
        "boundary": USER_INPUT_BOUNDARY,
        "input_midi": str(input_info.get("midi_path") or ""),
        "explicit_input_used": bool(input_info.get("explicit_input_used", False)),
        "candidate_count": _int(summary.get("candidate_count")),
        "objective_supported_candidate_count": _int(summary.get("objective_supported_candidate_count")),
        "repaired_midi_file_count": _int(summary.get("repaired_midi_file_count")),
        "input_context_bars": _int(summary.get("input_context_bars")),
        "min_dead_air_ratio": _float(summary.get("min_dead_air_ratio")),
        "max_dead_air_ratio": _float(summary.get("max_dead_air_ratio")),
        "candidates": candidates[: int(min_candidate_count)],
    }


def validate_audio_render_report(report: dict[str, Any], *, expected_count: int) -> dict[str, Any]:
    boundary = _dict(report.get("audio_render_boundary"))
    decision = _dict(report.get("decision"))
    if str(report.get("source_boundary") or "") != USER_INPUT_BOUNDARY:
        raise StageBMidiToSoloPhraseBankCliObjectiveNextError(
            "audio render source must be CLI user-input smoke"
        )
    if str(boundary.get("boundary") or "") != AUDIO_RENDER_BOUNDARY:
        raise StageBMidiToSoloPhraseBankCliObjectiveNextError("CLI audio render boundary required")
    if str(decision.get("next_boundary") or "") != AUDIO_RENDER_NEXT_BOUNDARY:
        raise StageBMidiToSoloPhraseBankCliObjectiveNextError(
            "audio render must route to listening review package"
        )
    if not bool(boundary.get("technical_wav_validation", False)):
        raise StageBMidiToSoloPhraseBankCliObjectiveNextError("technical WAV validation required")
    if not bool(boundary.get("cli_user_input_audio_render_completed", False)):
        raise StageBMidiToSoloPhraseBankCliObjectiveNextError("CLI user-input audio render completion required")
    if _int(boundary.get("rendered_audio_file_count")) < int(expected_count):
        raise StageBMidiToSoloPhraseBankCliObjectiveNextError("rendered audio count below threshold")
    if bool(decision.get("critical_user_input_required", True)):
        raise StageBMidiToSoloPhraseBankCliObjectiveNextError("critical user input should not be required")
    _require_no_quality_claim(boundary, label="CLI audio render boundary")
    rendered = [_dict(item) for item in _list(report.get("rendered_audio_files"))]
    if len(rendered) < int(expected_count):
        raise StageBMidiToSoloPhraseBankCliObjectiveNextError("rendered audio rows required")
    return {
        "boundary": AUDIO_RENDER_BOUNDARY,
        "technical_wav_validation": bool(boundary.get("technical_wav_validation", False)),
        "rendered_audio_file_count": _int(boundary.get("rendered_audio_file_count")),
        "rendered_audio_files": rendered[: int(expected_count)],
    }


def build_candidate_rows(*, user_input: dict[str, Any], audio_render: dict[str, Any]) -> list[dict[str, Any]]:
    audio_by_seed = {
        _int(item.get("sample_seed")): item for item in _list(audio_render.get("rendered_audio_files"))
    }
    rows: list[dict[str, Any]] = []
    for item in _list(user_input.get("candidates")):
        seed = _int(item.get("sample_seed"))
        audio = _dict(audio_by_seed.get(seed))
        wav = _dict(audio.get("wav_file"))
        rows.append(
            {
                "rank": _int(item.get("rank")),
                "sample_seed": seed,
                "midi_path": str(item.get("repaired_midi_path") or ""),
                "wav_path": str(wav.get("path") or ""),
                "objective_supported": bool(item.get("objective_supported", False)),
                "note_count": _int(item.get("note_count")),
                "unique_pitch_count": _int(item.get("unique_pitch_count")),
                "max_simultaneous_notes": _int(item.get("max_simultaneous_notes")),
                "dead_air_ratio": _float(item.get("dead_air_ratio")),
                "phrase_coverage_ratio": _float(item.get("phrase_coverage_ratio")),
                "wav_duration_seconds": _float(wav.get("duration_seconds")),
                "wav_sample_rate": _int(wav.get("sample_rate")),
                "wav_sha256_prefix": str(wav.get("sha256") or "")[:12],
            }
        )
    return rows


def build_objective_next_report(
    *,
    input_guard_report: dict[str, Any],
    user_input_smoke_report: dict[str, Any],
    audio_render_report: dict[str, Any],
    output_dir: Path,
    issue_number: int,
    min_candidate_count: int,
) -> dict[str, Any]:
    input_guard = validate_input_guard_report(input_guard_report)
    user_input = validate_user_input_smoke_report(
        user_input_smoke_report,
        min_candidate_count=int(min_candidate_count),
    )
    audio_render = validate_audio_render_report(audio_render_report, expected_count=int(min_candidate_count))
    candidate_rows = build_candidate_rows(user_input=user_input, audio_render=audio_render)
    objective_supported_count = sum(1 for item in candidate_rows if bool(item["objective_supported"]))
    technical_cli_path_ready = (
        bool(user_input["explicit_input_used"])
        and objective_supported_count >= int(min_candidate_count)
        and _int(user_input["repaired_midi_file_count"]) >= int(min_candidate_count)
        and bool(audio_render["technical_wav_validation"])
        and not bool(input_guard["validated_review_input_present"])
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
            "user_input_smoke": user_input["boundary"],
            "audio_render": audio_render["boundary"],
        },
        "objective_summary": {
            "technical_midi_to_solo_cli_path_ready": bool(technical_cli_path_ready),
            "explicit_input_used": bool(user_input["explicit_input_used"]),
            "candidate_count": _int(user_input["candidate_count"]),
            "objective_supported_candidate_count": int(objective_supported_count),
            "repaired_midi_file_count": _int(user_input["repaired_midi_file_count"]),
            "rendered_audio_file_count": _int(audio_render["rendered_audio_file_count"]),
            "technical_wav_validation": bool(audio_render["technical_wav_validation"]),
            "input_context_bars": _int(user_input["input_context_bars"]),
            "dead_air_range": [
                _float(user_input["min_dead_air_ratio"]),
                _float(user_input["max_dead_air_ratio"]),
            ],
            "validated_review_input_present": bool(input_guard["validated_review_input_present"]),
            "preference_fill_allowed": bool(input_guard["preference_fill_allowed"]),
            "mvp_current_evidence_consolidation_ready": bool(technical_cli_path_ready),
        },
        "candidate_reviews": candidate_rows,
        "readiness": {
            "boundary": BOUNDARY,
            "cli_objective_only_next_decision_completed": True,
            "technical_midi_to_solo_cli_path_ready": bool(technical_cli_path_ready),
            "mvp_current_evidence_consolidation_ready": bool(technical_cli_path_ready),
            "human_audio_preference_claimed": False,
            "midi_to_solo_musical_quality_claimed": False,
            "audio_rendered_quality_claimed": False,
            "phrase_bank_musical_quality_claimed": False,
            "model_checkpoint_generation_quality_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
            "production_ready_claimed": False,
        },
        "decision": {
            "current_boundary": BOUNDARY,
            "next_boundary": NEXT_BOUNDARY,
            "auto_progress_allowed": True,
            "critical_user_input_required": False,
            "reason": "explicit-input CLI ranked MIDI and WAV technical path ready; preference remains blocked",
        },
        "not_proven": [
            "listening_review_completed",
            "human_audio_preference",
            "midi_to_solo_musical_quality",
            "phrase_bank_musical_quality",
            "broad_trained_model_quality",
            "brad_style_adaptation",
            "production_ready",
        ],
        "next_recommended_issue": "Stage B MIDI-to-solo MVP current evidence consolidation",
    }


def validate_objective_next_report(
    report: dict[str, Any],
    *,
    expected_boundary: str | None,
    expected_next_boundary: str | None,
    require_objective_decision: bool,
    require_current_evidence_ready: bool,
    require_no_quality_claim: bool,
) -> dict[str, Any]:
    boundary = str(report.get("boundary") or "")
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    summary = _dict(report.get("objective_summary"))
    if expected_boundary and boundary != expected_boundary:
        raise StageBMidiToSoloPhraseBankCliObjectiveNextError(
            f"expected boundary {expected_boundary}, got {boundary}"
        )
    if expected_next_boundary and str(decision.get("next_boundary") or "") != expected_next_boundary:
        raise StageBMidiToSoloPhraseBankCliObjectiveNextError("unexpected next boundary")
    if require_objective_decision and not bool(
        readiness.get("cli_objective_only_next_decision_completed", False)
    ):
        raise StageBMidiToSoloPhraseBankCliObjectiveNextError("objective decision completion required")
    if require_current_evidence_ready and not bool(
        summary.get("mvp_current_evidence_consolidation_ready", False)
    ):
        raise StageBMidiToSoloPhraseBankCliObjectiveNextError("current evidence readiness expected")
    if bool(summary.get("preference_fill_allowed", True)):
        raise StageBMidiToSoloPhraseBankCliObjectiveNextError("preference fill must remain blocked")
    if bool(decision.get("critical_user_input_required", True)):
        raise StageBMidiToSoloPhraseBankCliObjectiveNextError("critical user input should not be required")
    if require_no_quality_claim:
        _require_no_quality_claim(readiness, label="CLI objective-next readiness")
    return {
        "boundary": boundary,
        "next_boundary": str(decision.get("next_boundary") or ""),
        "technical_midi_to_solo_cli_path_ready": bool(
            summary.get("technical_midi_to_solo_cli_path_ready", False)
        ),
        "mvp_current_evidence_consolidation_ready": bool(
            summary.get("mvp_current_evidence_consolidation_ready", False)
        ),
        "explicit_input_used": bool(summary.get("explicit_input_used", False)),
        "candidate_count": _int(summary.get("candidate_count")),
        "objective_supported_candidate_count": _int(summary.get("objective_supported_candidate_count")),
        "repaired_midi_file_count": _int(summary.get("repaired_midi_file_count")),
        "rendered_audio_file_count": _int(summary.get("rendered_audio_file_count")),
        "technical_wav_validation": bool(summary.get("technical_wav_validation", False)),
        "input_context_bars": _int(summary.get("input_context_bars")),
        "validated_review_input_present": bool(summary.get("validated_review_input_present", True)),
        "preference_fill_allowed": bool(summary.get("preference_fill_allowed", True)),
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
    dead_air = summary["dead_air_range"]
    lines = [
        "# Stage B MIDI-to-Solo Phrase-Bank CLI Objective-Only Next Decision",
        "",
        "## Summary",
        "",
        f"- boundary: `{report['boundary']}`",
        f"- next boundary: `{decision['next_boundary']}`",
        f"- technical MIDI-to-solo CLI path ready: `{_bool_token(summary['technical_midi_to_solo_cli_path_ready'])}`",
        f"- explicit input used: `{_bool_token(summary['explicit_input_used'])}`",
        f"- candidate count: `{summary['candidate_count']}`",
        f"- objective supported candidate count: `{summary['objective_supported_candidate_count']}`",
        f"- repaired MIDI file count: `{summary['repaired_midi_file_count']}`",
        f"- rendered audio file count: `{summary['rendered_audio_file_count']}`",
        f"- technical WAV validation: `{_bool_token(summary['technical_wav_validation'])}`",
        f"- input context bars: `{summary['input_context_bars']}`",
        f"- dead-air range: `{dead_air[0]:.4f} - {dead_air[1]:.4f}`",
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
                f"- MIDI: `{item['midi_path']}`",
                f"- WAV: `{item['wav_path']}`",
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
    parser = argparse.ArgumentParser(description="Decide CLI phrase-bank objective-only next step")
    parser.add_argument("--input_guard_report", type=str, required=True)
    parser.add_argument("--user_input_smoke_report", type=str, required=True)
    parser.add_argument("--audio_render_report", type=str, required=True)
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/stage_b_midi_to_solo_phrase_bank_cli_objective_only_next_decision",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--issue_number", type=int, default=662)
    parser.add_argument("--min_candidate_count", type=int, default=3)
    parser.add_argument("--expected_boundary", type=str, default="")
    parser.add_argument("--expected_next_boundary", type=str, default="")
    parser.add_argument("--require_objective_decision", action="store_true")
    parser.add_argument("--require_current_evidence_ready", action="store_true")
    parser.add_argument("--require_no_quality_claim", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    report = build_objective_next_report(
        input_guard_report=read_json(Path(args.input_guard_report)),
        user_input_smoke_report=read_json(Path(args.user_input_smoke_report)),
        audio_render_report=read_json(Path(args.audio_render_report)),
        output_dir=output_dir,
        issue_number=int(args.issue_number),
        min_candidate_count=int(args.min_candidate_count),
    )
    summary = validate_objective_next_report(
        report,
        expected_boundary=str(args.expected_boundary or ""),
        expected_next_boundary=str(args.expected_next_boundary or ""),
        require_objective_decision=bool(args.require_objective_decision),
        require_current_evidence_ready=bool(args.require_current_evidence_ready),
        require_no_quality_claim=bool(args.require_no_quality_claim),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "stage_b_midi_to_solo_phrase_bank_cli_objective_only_next_decision.json", report)
    write_json(
        output_dir / "stage_b_midi_to_solo_phrase_bank_cli_objective_only_next_decision_validation_summary.json",
        summary,
    )
    markdown = markdown_report(report)
    write_text(output_dir / "stage_b_midi_to_solo_phrase_bank_cli_objective_only_next_decision.md", markdown)
    if args.doc_path:
        write_text(Path(args.doc_path), markdown)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
