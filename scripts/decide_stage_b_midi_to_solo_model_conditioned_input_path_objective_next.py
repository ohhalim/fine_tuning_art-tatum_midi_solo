"""Select the next step after model-conditioned input-path evidence without listening input."""

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
from scripts.export_stage_b_midi_to_solo_model_conditioned_input_path_candidates import (  # noqa: E402
    BOUNDARY as CANDIDATE_EXPORT_BOUNDARY,
)
from scripts.guard_stage_b_midi_to_solo_model_conditioned_input_path_listening_review_input import (  # noqa: E402
    BOUNDARY as INPUT_GUARD_BOUNDARY,
    OBJECTIVE_NEXT_BOUNDARY as INPUT_GUARD_NEXT_BOUNDARY,
)
from scripts.render_stage_b_midi_to_solo_model_conditioned_input_path_audio import (  # noqa: E402
    BOUNDARY as AUDIO_RENDER_BOUNDARY,
)


class StageBMidiToSoloModelConditionedInputPathObjectiveNextError(ValueError):
    pass


BOUNDARY = "stage_b_midi_to_solo_model_conditioned_input_path_objective_only_next_decision"
REPAIR_NEXT_BOUNDARY = "stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_decision"
CURRENT_EVIDENCE_NEXT_BOUNDARY = "stage_b_midi_to_solo_mvp_current_evidence_consolidation"
SCHEMA_VERSION = "stage_b_midi_to_solo_model_conditioned_input_path_objective_next_v1"

QUALITY_CLAIM_KEYS = [
    "human_audio_preference_claimed",
    "midi_to_solo_musical_quality_claimed",
    "musical_quality_claimed",
    "audio_rendered_quality_claimed",
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
        raise StageBMidiToSoloModelConditionedInputPathObjectiveNextError(
            f"unexpected quality claim in {label}: {claimed}"
        )


def _validate_cli_source(source: dict[str, Any], *, label: str) -> dict[str, Any]:
    if not bool(source.get("phrase_bank_cli_technical_path_completed", False)):
        raise StageBMidiToSoloModelConditionedInputPathObjectiveNextError(
            f"{label} phrase-bank CLI technical path completion required"
        )
    if _int(source.get("cli_candidate_count")) < 3:
        raise StageBMidiToSoloModelConditionedInputPathObjectiveNextError(
            f"{label} CLI candidate count below 3"
        )
    if _int(source.get("cli_rendered_audio_file_count")) < 3:
        raise StageBMidiToSoloModelConditionedInputPathObjectiveNextError(
            f"{label} CLI rendered WAV count below 3"
        )
    if _int(source.get("cli_input_context_bars")) <= 0:
        raise StageBMidiToSoloModelConditionedInputPathObjectiveNextError(
            f"{label} CLI input context bars required"
        )
    if bool(source.get("cli_preference_fill_allowed", True)):
        raise StageBMidiToSoloModelConditionedInputPathObjectiveNextError(
            f"{label} CLI preference fill should remain blocked"
        )
    return {
        "phrase_bank_cli_technical_path_completed": True,
        "cli_candidate_count": _int(source.get("cli_candidate_count")),
        "cli_rendered_audio_file_count": _int(source.get("cli_rendered_audio_file_count")),
        "cli_input_context_bars": _int(source.get("cli_input_context_bars")),
        "cli_preference_fill_allowed": bool(source.get("cli_preference_fill_allowed", True)),
    }


def validate_input_guard_report(report: dict[str, Any]) -> dict[str, Any]:
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    guard = _dict(report.get("guard_result"))
    source = _validate_cli_source(_dict(guard.get("source_evidence")), label="input guard source")
    if str(report.get("boundary") or "") != INPUT_GUARD_BOUNDARY:
        raise StageBMidiToSoloModelConditionedInputPathObjectiveNextError(
            "model-conditioned input guard boundary required"
        )
    if str(decision.get("next_boundary") or "") != INPUT_GUARD_NEXT_BOUNDARY:
        raise StageBMidiToSoloModelConditionedInputPathObjectiveNextError(
            "input guard must route to objective-only next decision"
        )
    if not bool(readiness.get("listening_review_input_guard_completed", False)):
        raise StageBMidiToSoloModelConditionedInputPathObjectiveNextError(
            "input guard completion required"
        )
    if bool(guard.get("validated_review_input_present", True)):
        raise StageBMidiToSoloModelConditionedInputPathObjectiveNextError(
            "objective-only decision requires pending review input"
        )
    if bool(guard.get("preference_fill_allowed", True)):
        raise StageBMidiToSoloModelConditionedInputPathObjectiveNextError(
            "preference fill must remain blocked"
        )
    if bool(decision.get("critical_user_input_required", True)):
        raise StageBMidiToSoloModelConditionedInputPathObjectiveNextError(
            "critical user input should not be required"
        )
    _require_no_quality_claim(readiness, label="input guard readiness")
    return {
        "boundary": INPUT_GUARD_BOUNDARY,
        "review_item_count": _int(guard.get("review_item_count")),
        "required_input_field_count": _int(guard.get("required_input_field_count")),
        "validated_review_input_present": bool(guard.get("validated_review_input_present", False)),
        "preference_fill_allowed": bool(guard.get("preference_fill_allowed", False)),
        "source_evidence": source,
    }


def validate_candidate_export_report(report: dict[str, Any], *, expected_count: int) -> dict[str, Any]:
    readiness = _dict(report.get("readiness"))
    summary = _dict(report.get("summary"))
    source = _validate_cli_source(_dict(report.get("probe_source")), label="candidate export source")
    candidates = [_dict(item) for item in _list(report.get("top_candidates"))]
    if str(report.get("boundary") or readiness.get("boundary") or "") != CANDIDATE_EXPORT_BOUNDARY:
        raise StageBMidiToSoloModelConditionedInputPathObjectiveNextError(
            "candidate export boundary required"
        )
    required_true = [
        "model_conditioned_input_path_candidate_export_completed",
        "ranked_midi_candidates_exported",
        "model_conditioned_ranked_input_path_contract_matched",
        "fallback_replacement_candidate_export_ready",
    ]
    missing = [name for name in required_true if not bool(readiness.get(name, False))]
    if missing:
        raise StageBMidiToSoloModelConditionedInputPathObjectiveNextError(
            f"missing candidate export readiness: {missing}"
        )
    if _int(summary.get("exported_candidate_count")) < int(expected_count):
        raise StageBMidiToSoloModelConditionedInputPathObjectiveNextError(
            "exported candidate count below expected"
        )
    if len(candidates) < int(expected_count):
        raise StageBMidiToSoloModelConditionedInputPathObjectiveNextError(
            "top candidate rows below expected"
        )
    _require_no_quality_claim(readiness, label="candidate export readiness")
    return {
        "boundary": CANDIDATE_EXPORT_BOUNDARY,
        "exported_candidate_count": _int(summary.get("exported_candidate_count")),
        "best_note_count": _int(summary.get("best_note_count")),
        "best_unique_pitch_count": _int(summary.get("best_unique_pitch_count")),
        "best_dead_air_ratio": _float(summary.get("best_dead_air_ratio")),
        "top_candidates": candidates[: int(expected_count)],
        "source_evidence": source,
    }


def validate_audio_render_report(report: dict[str, Any], *, expected_count: int) -> dict[str, Any]:
    boundary = _dict(report.get("audio_render_boundary"))
    decision = _dict(report.get("decision"))
    source = _validate_cli_source(_dict(report.get("candidate_export_source")), label="audio render source")
    if str(report.get("source_boundary") or "") != CANDIDATE_EXPORT_BOUNDARY:
        raise StageBMidiToSoloModelConditionedInputPathObjectiveNextError(
            "audio render source must be candidate export"
        )
    if str(boundary.get("boundary") or "") != AUDIO_RENDER_BOUNDARY:
        raise StageBMidiToSoloModelConditionedInputPathObjectiveNextError(
            "audio render boundary required"
        )
    required_true = [
        "render_attempted",
        "technical_wav_validation",
        "model_conditioned_ranked_audio_render_completed",
        "fallback_replacement_candidate_export_ready",
        "fallback_replacement_technical_path_ready",
        "fallback_replacement_ready",
    ]
    missing = [name for name in required_true if not bool(boundary.get(name, False))]
    if missing:
        raise StageBMidiToSoloModelConditionedInputPathObjectiveNextError(
            f"missing audio render readiness: {missing}"
        )
    if bool(decision.get("critical_user_input_required", True)):
        raise StageBMidiToSoloModelConditionedInputPathObjectiveNextError(
            "critical user input should not be required"
        )
    _require_no_quality_claim(boundary, label="audio render boundary")
    rows = [_dict(item) for item in _list(report.get("rendered_audio_files"))]
    if len(rows) < int(expected_count):
        raise StageBMidiToSoloModelConditionedInputPathObjectiveNextError(
            "rendered audio rows below expected"
        )
    return {
        "boundary": AUDIO_RENDER_BOUNDARY,
        "rendered_audio_file_count": _int(boundary.get("rendered_audio_file_count")),
        "technical_wav_validation": bool(boundary.get("technical_wav_validation", False)),
        "rendered_audio_files": rows[: int(expected_count)],
        "source_evidence": source,
    }


def build_candidate_rows(
    *,
    candidate_export: dict[str, Any],
    audio_render: dict[str, Any],
    dead_air_threshold: float,
) -> list[dict[str, Any]]:
    audio_by_rank = {
        _int(item.get("rank")): item for item in _list(audio_render.get("rendered_audio_files"))
    }
    rows: list[dict[str, Any]] = []
    for candidate in _list(candidate_export.get("top_candidates")):
        rank = _int(candidate.get("rank"))
        audio = _dict(audio_by_rank.get(rank))
        wav = _dict(audio.get("wav_file"))
        dead_air = max(
            _float(candidate.get("dead_air_ratio")),
            _float(audio.get("source_dead_air_ratio")),
        )
        rows.append(
            {
                "rank": rank,
                "sample_index": _int(candidate.get("sample_index")),
                "sample_seed": _int(candidate.get("sample_seed")),
                "midi_path": str(candidate.get("export_midi_path") or audio.get("source_midi_path") or ""),
                "wav_path": str(wav.get("path") or ""),
                "note_count": _int(candidate.get("note_count") or audio.get("source_note_count")),
                "unique_pitch_count": _int(
                    candidate.get("unique_pitch_count") or audio.get("source_unique_pitch_count")
                ),
                "max_simultaneous_notes": _int(candidate.get("max_simultaneous_notes")),
                "chord_tone_ratio": _float(
                    candidate.get("chord_tone_ratio") or audio.get("source_chord_tone_ratio")
                ),
                "dead_air_ratio": dead_air,
                "phrase_coverage_ratio": _float(candidate.get("phrase_coverage_ratio")),
                "position_span_ratio": _float(candidate.get("position_span_ratio")),
                "postprocess_removal_ratio": _float(candidate.get("postprocess_removal_ratio")),
                "wav_duration_seconds": _float(wav.get("duration_seconds")),
                "wav_sample_rate": _int(wav.get("sample_rate")),
                "wav_sha256_prefix": str(wav.get("sha256") or "")[:12],
                "dead_air_failure": dead_air >= float(dead_air_threshold),
            }
        )
    return rows


def build_objective_next_report(
    *,
    input_guard_report: dict[str, Any],
    candidate_export_report: dict[str, Any],
    audio_render_report: dict[str, Any],
    output_dir: Path,
    issue_number: int,
    expected_count: int,
    dead_air_threshold: float,
) -> dict[str, Any]:
    input_guard = validate_input_guard_report(input_guard_report)
    candidate_export = validate_candidate_export_report(
        candidate_export_report,
        expected_count=int(expected_count),
    )
    audio_render = validate_audio_render_report(audio_render_report, expected_count=int(expected_count))
    if input_guard["source_evidence"] != candidate_export["source_evidence"]:
        raise StageBMidiToSoloModelConditionedInputPathObjectiveNextError(
            "input guard and candidate export source evidence mismatch"
        )
    if candidate_export["source_evidence"] != audio_render["source_evidence"]:
        raise StageBMidiToSoloModelConditionedInputPathObjectiveNextError(
            "candidate export and audio render source evidence mismatch"
        )
    candidate_rows = build_candidate_rows(
        candidate_export=candidate_export,
        audio_render=audio_render,
        dead_air_threshold=float(dead_air_threshold),
    )
    dead_air_values = [_float(item["dead_air_ratio"]) for item in candidate_rows]
    dead_air_failure_count = sum(1 for item in candidate_rows if bool(item["dead_air_failure"]))
    repair_required = dead_air_failure_count > 0
    next_boundary = REPAIR_NEXT_BOUNDARY if repair_required else CURRENT_EVIDENCE_NEXT_BOUNDARY
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "issue_number": int(issue_number),
        "boundary": BOUNDARY,
        "source_boundaries": {
            "input_guard": input_guard["boundary"],
            "candidate_export": candidate_export["boundary"],
            "audio_render": audio_render["boundary"],
        },
        "source_evidence": input_guard["source_evidence"],
        "objective_summary": {
            "model_conditioned_technical_path_ready": True,
            "candidate_count": int(len(candidate_rows)),
            "exported_candidate_count": _int(candidate_export["exported_candidate_count"]),
            "rendered_audio_file_count": _int(audio_render["rendered_audio_file_count"]),
            "technical_wav_validation": bool(audio_render["technical_wav_validation"]),
            "dead_air_threshold": float(dead_air_threshold),
            "dead_air_failure_count": int(dead_air_failure_count),
            "all_candidates_dead_air_failure": dead_air_failure_count == len(candidate_rows),
            "dead_air_min": min(dead_air_values) if dead_air_values else 0.0,
            "dead_air_max": max(dead_air_values) if dead_air_values else 0.0,
            "best_note_count": _int(candidate_export["best_note_count"]),
            "best_unique_pitch_count": _int(candidate_export["best_unique_pitch_count"]),
            "validated_review_input_present": bool(input_guard["validated_review_input_present"]),
            "preference_fill_allowed": bool(input_guard["preference_fill_allowed"]),
            "dead_air_timing_repair_required": bool(repair_required),
            "current_evidence_consolidation_ready": not bool(repair_required),
        },
        "candidate_reviews": candidate_rows,
        "readiness": {
            "boundary": BOUNDARY,
            "objective_only_next_decision_completed": True,
            "model_conditioned_technical_path_ready": True,
            "dead_air_timing_repair_required": bool(repair_required),
            "current_evidence_consolidation_ready": not bool(repair_required),
            "human_audio_preference_claimed": False,
            "midi_to_solo_musical_quality_claimed": False,
            "audio_rendered_quality_claimed": False,
            "model_checkpoint_generation_quality_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
            "production_ready_claimed": False,
        },
        "decision": {
            "current_boundary": BOUNDARY,
            "next_boundary": next_boundary,
            "auto_progress_allowed": True,
            "critical_user_input_required": False,
            "reason": (
                "model-conditioned ranked MIDI/WAV technical path is ready; dead-air objective risk requires repair"
                if repair_required
                else "model-conditioned ranked MIDI/WAV technical path has no dead-air repair trigger"
            ),
        },
        "not_proven": [
            "listening_review_completed",
            "human_audio_preference",
            "midi_to_solo_musical_quality",
            "audio_rendered_quality",
            "broad_trained_model_quality",
            "brad_style_adaptation",
        ],
        "next_recommended_issue": (
            "Stage B MIDI-to-solo model-conditioned input path dead-air timing repair decision"
            if repair_required
            else "Stage B MIDI-to-solo MVP current evidence consolidation"
        ),
    }


def validate_objective_next_report(
    report: dict[str, Any],
    *,
    expected_boundary: str | None,
    expected_next_boundary: str | None,
    require_objective_decision: bool,
    require_repair_required: bool,
    require_no_quality_claim: bool,
) -> dict[str, Any]:
    boundary = str(report.get("boundary") or "")
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    summary = _dict(report.get("objective_summary"))
    if expected_boundary and boundary != expected_boundary:
        raise StageBMidiToSoloModelConditionedInputPathObjectiveNextError(
            f"expected boundary {expected_boundary}, got {boundary}"
        )
    if expected_next_boundary and str(decision.get("next_boundary") or "") != expected_next_boundary:
        raise StageBMidiToSoloModelConditionedInputPathObjectiveNextError("unexpected next boundary")
    if require_objective_decision and not bool(readiness.get("objective_only_next_decision_completed", False)):
        raise StageBMidiToSoloModelConditionedInputPathObjectiveNextError(
            "objective decision completion required"
        )
    if require_repair_required and not bool(summary.get("dead_air_timing_repair_required", False)):
        raise StageBMidiToSoloModelConditionedInputPathObjectiveNextError(
            "dead-air timing repair should be required"
        )
    if bool(summary.get("preference_fill_allowed", True)):
        raise StageBMidiToSoloModelConditionedInputPathObjectiveNextError(
            "preference fill must remain blocked"
        )
    if bool(decision.get("critical_user_input_required", True)):
        raise StageBMidiToSoloModelConditionedInputPathObjectiveNextError(
            "critical user input should not be required"
        )
    if require_no_quality_claim:
        _require_no_quality_claim(readiness, label="objective-next readiness")
    return {
        "boundary": boundary,
        "next_boundary": str(decision.get("next_boundary") or ""),
        "model_conditioned_technical_path_ready": bool(
            summary.get("model_conditioned_technical_path_ready", False)
        ),
        "candidate_count": _int(summary.get("candidate_count")),
        "exported_candidate_count": _int(summary.get("exported_candidate_count")),
        "rendered_audio_file_count": _int(summary.get("rendered_audio_file_count")),
        "technical_wav_validation": bool(summary.get("technical_wav_validation", False)),
        "dead_air_threshold": _float(summary.get("dead_air_threshold")),
        "dead_air_failure_count": _int(summary.get("dead_air_failure_count")),
        "all_candidates_dead_air_failure": bool(summary.get("all_candidates_dead_air_failure", False)),
        "dead_air_min": _float(summary.get("dead_air_min")),
        "dead_air_max": _float(summary.get("dead_air_max")),
        "validated_review_input_present": bool(summary.get("validated_review_input_present", True)),
        "preference_fill_allowed": bool(summary.get("preference_fill_allowed", True)),
        "dead_air_timing_repair_required": bool(summary.get("dead_air_timing_repair_required", False)),
        "current_evidence_consolidation_ready": bool(
            summary.get("current_evidence_consolidation_ready", True)
        ),
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
    source = report["source_evidence"]
    lines = [
        "# Stage B MIDI-to-Solo Model-Conditioned Input Path Objective-Only Next Decision",
        "",
        "## Summary",
        "",
        f"- boundary: `{report['boundary']}`",
        f"- next boundary: `{decision['next_boundary']}`",
        f"- model-conditioned technical path ready: `{_bool_token(summary['model_conditioned_technical_path_ready'])}`",
        f"- candidate / exported / rendered: `{summary['candidate_count']} / {summary['exported_candidate_count']} / {summary['rendered_audio_file_count']}`",
        f"- technical WAV validation: `{_bool_token(summary['technical_wav_validation'])}`",
        f"- dead-air threshold: `{summary['dead_air_threshold']:.4f}`",
        f"- dead-air failure count: `{summary['dead_air_failure_count']}`",
        f"- dead-air min / max: `{summary['dead_air_min']:.4f} / {summary['dead_air_max']:.4f}`",
        f"- dead-air timing repair required: `{_bool_token(summary['dead_air_timing_repair_required'])}`",
        f"- current evidence consolidation ready: `{_bool_token(summary['current_evidence_consolidation_ready'])}`",
        f"- preference fill allowed: `{_bool_token(summary['preference_fill_allowed'])}`",
        f"- human/audio preference claimed: `{_bool_token(readiness['human_audio_preference_claimed'])}`",
        f"- MIDI-to-solo musical quality claimed: `{_bool_token(readiness['midi_to_solo_musical_quality_claimed'])}`",
        "",
        "## Source Evidence",
        "",
        f"- phrase-bank CLI technical path completed: `{_bool_token(source['phrase_bank_cli_technical_path_completed'])}`",
        f"- CLI candidate / rendered WAV: `{source['cli_candidate_count']}` / `{source['cli_rendered_audio_file_count']}`",
        f"- CLI input context bars: `{source['cli_input_context_bars']}`",
        f"- CLI preference fill allowed: `{_bool_token(source['cli_preference_fill_allowed'])}`",
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
                f"- notes / unique pitches / max simultaneous: `{item['note_count']} / {item['unique_pitch_count']} / {item['max_simultaneous_notes']}`",
                f"- chord-tone / dead-air / phrase coverage: `{item['chord_tone_ratio']:.4f} / {item['dead_air_ratio']:.4f} / {item['phrase_coverage_ratio']:.4f}`",
                f"- position span / postprocess removal: `{item['position_span_ratio']:.4f} / {item['postprocess_removal_ratio']:.4f}`",
                f"- dead-air failure: `{_bool_token(item['dead_air_failure'])}`",
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
    parser = argparse.ArgumentParser(
        description="Decide model-conditioned input-path objective-only next step"
    )
    parser.add_argument("--input_guard_report", type=str, required=True)
    parser.add_argument("--candidate_export_report", type=str, required=True)
    parser.add_argument("--audio_render_report", type=str, required=True)
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/stage_b_midi_to_solo_model_conditioned_input_path_objective_only_next_decision",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--issue_number", type=int, default=686)
    parser.add_argument("--expected_count", type=int, default=3)
    parser.add_argument("--dead_air_threshold", type=float, default=0.5)
    parser.add_argument("--expected_boundary", type=str, default="")
    parser.add_argument("--expected_next_boundary", type=str, default="")
    parser.add_argument("--require_objective_decision", action="store_true")
    parser.add_argument("--require_repair_required", action="store_true")
    parser.add_argument("--require_no_quality_claim", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    report = build_objective_next_report(
        input_guard_report=read_json(Path(args.input_guard_report)),
        candidate_export_report=read_json(Path(args.candidate_export_report)),
        audio_render_report=read_json(Path(args.audio_render_report)),
        output_dir=output_dir,
        issue_number=int(args.issue_number),
        expected_count=int(args.expected_count),
        dead_air_threshold=float(args.dead_air_threshold),
    )
    summary = validate_objective_next_report(
        report,
        expected_boundary=str(args.expected_boundary or ""),
        expected_next_boundary=str(args.expected_next_boundary or ""),
        require_objective_decision=bool(args.require_objective_decision),
        require_repair_required=bool(args.require_repair_required),
        require_no_quality_claim=bool(args.require_no_quality_claim),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        output_dir / "stage_b_midi_to_solo_model_conditioned_input_path_objective_only_next_decision.json",
        report,
    )
    write_json(
        output_dir
        / "stage_b_midi_to_solo_model_conditioned_input_path_objective_only_next_decision_validation_summary.json",
        summary,
    )
    markdown = markdown_report(report)
    write_text(
        output_dir / "stage_b_midi_to_solo_model_conditioned_input_path_objective_only_next_decision.md",
        markdown,
    )
    if args.doc_path:
        write_text(Path(args.doc_path), markdown)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
