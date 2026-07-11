"""Run model-direct timing phrase repair for MIDI-to-solo candidates."""

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
from scripts.diagnose_stage_b_midi_to_solo_model_direct_phrase_quality import (  # noqa: E402
    LISTENING_REVIEW_BOUNDARY,
)
from scripts.run_stage_b_midi_to_solo_model_direct_8bar_generation_probe import (  # noqa: E402
    build_generation_command,
    run_command,
    summarize_context,
    summarize_generation,
    summarize_repaired_scale_smoke,
    summarize_sequence_budget_repair,
)
from scripts.run_stage_b_midi_to_solo_model_direct_pitch_contour_repair import (  # noqa: E402
    BOUNDARY as SOURCE_BOUNDARY,
    TIMING_REPAIR_BOUNDARY as BOUNDARY,
    summarize_repaired_diagnostics,
)


class StageBMidiToSoloModelDirectTimingPhraseRepairError(ValueError):
    pass


FAIL_NEXT_BOUNDARY = "stage_b_midi_to_solo_model_direct_timing_phrase_repair_followup"
SCHEMA_VERSION = "stage_b_midi_to_solo_model_direct_timing_phrase_repair_v1"


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise StageBMidiToSoloModelDirectTimingPhraseRepairError(f"report missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _bool_token(value: Any) -> str:
    return "true" if bool(value) else "false"


def validate_source_pitch_repair(report: dict[str, Any]) -> dict[str, Any]:
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    result = _dict(report.get("repair_result"))
    diagnostics = _dict(report.get("repaired_diagnostics_summary"))
    boundary = str(report.get("boundary") or readiness.get("boundary") or "")
    if boundary != SOURCE_BOUNDARY:
        raise StageBMidiToSoloModelDirectTimingPhraseRepairError("pitch contour repair boundary required")
    if str(decision.get("next_boundary") or "") != BOUNDARY:
        raise StageBMidiToSoloModelDirectTimingPhraseRepairError("pitch repair must route to timing repair")
    if not bool(readiness.get("pitch_contour_repair_passed", False)):
        raise StageBMidiToSoloModelDirectTimingPhraseRepairError("pitch contour repair pass required")
    if bool(readiness.get("model_direct_generation_quality_claimed", True)):
        raise StageBMidiToSoloModelDirectTimingPhraseRepairError("upstream model quality must not be claimed")
    return {
        "boundary": boundary,
        "candidate_count": _int(diagnostics.get("candidate_count")),
        "flag_counts": _dict(diagnostics.get("flag_counts")),
        "max_interval_max": _int(diagnostics.get("max_interval_max")),
        "max_pitch_span": _int(diagnostics.get("max_pitch_span")),
        "max_dead_air_ratio": _float(diagnostics.get("max_dead_air_ratio")),
        "previous_dead_air_flag_count": _int(result.get("repaired_dead_air_flag_count")),
    }


def build_timing_phrase_repair_report(
    *,
    source_pitch_repair: dict[str, Any],
    sequence_budget_repair: dict[str, Any],
    context_report: dict[str, Any],
    repaired_training_scale_smoke: dict[str, Any],
    generation_result: dict[str, Any],
    generation_report: dict[str, Any],
    generation_report_path: Path,
    output_dir: Path,
    issue_number: int,
    target_bars: int,
    note_groups_per_bar: int,
    jazz_rhythm_profile: str,
    constrained_pitch_min: int,
    constrained_pitch_max: int,
    constrained_max_adjacent_interval: int,
    dead_air_threshold_seconds: float,
) -> dict[str, Any]:
    source = validate_source_pitch_repair(source_pitch_repair)
    sequence = summarize_sequence_budget_repair(sequence_budget_repair)
    context = summarize_context(context_report, target_bars=int(target_bars))
    scale = summarize_repaired_scale_smoke(repaired_training_scale_smoke)
    generation = summarize_generation(generation_report)
    diagnostics = summarize_repaired_diagnostics(
        [str(path) for path in _list(generation.get("midi_paths"))],
        dead_air_threshold_seconds=float(dead_air_threshold_seconds),
    )
    command_succeeded = _int(generation_result.get("returncode")) == 0
    flag_counts = _dict(diagnostics.get("flag_counts"))
    dead_air_removed = _int(flag_counts.get("dead_air_gap", 0)) == 0
    wide_interval_still_removed = _int(flag_counts.get("wide_interval_contour", 0)) == 0
    wide_register_still_removed = _int(flag_counts.get("wide_register_span", 0)) == 0
    max_interval_guard_kept = _int(diagnostics.get("max_interval_max")) <= _int(source.get("max_interval_max"))
    dead_air_ratio_reduced = _float(diagnostics.get("max_dead_air_ratio")) < _float(source.get("max_dead_air_ratio"))
    repair_passed = bool(
        command_succeeded
        and generation["passed_strict_review_gate"]
        and generation["all_midi_paths_exist"]
        and dead_air_removed
        and wide_interval_still_removed
        and wide_register_still_removed
        and max_interval_guard_kept
        and dead_air_ratio_reduced
    )
    next_boundary = LISTENING_REVIEW_BOUNDARY if repair_passed else FAIL_NEXT_BOUNDARY
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "issue_number": int(issue_number),
        "boundary": BOUNDARY,
        "source_boundaries": {
            "pitch_contour_repair": source["boundary"],
            "sequence_budget_repair": sequence["boundary"],
            "context": context["boundary"],
            "repaired_training_scale_smoke": scale["boundary"],
        },
        "source_pitch_repair_summary": source,
        "sequence_budget_summary": sequence,
        "context_summary": context,
        "repaired_scale_smoke_summary": scale,
        "repair_config": {
            "generation_source": "model_checkpoint_direct_constrained",
            "target_bars": int(target_bars),
            "note_groups_per_bar": int(note_groups_per_bar),
            "max_sequence": int(scale["max_sequence"]),
            "jazz_rhythm_positions": True,
            "jazz_duration_tokens": False,
            "jazz_rhythm_profile": str(jazz_rhythm_profile),
            "cap_duration_to_next_position": False,
            "fill_duration_to_next_position": True,
            "constrained_pitch_min": int(constrained_pitch_min),
            "constrained_pitch_max": int(constrained_pitch_max),
            "constrained_max_adjacent_interval": int(constrained_max_adjacent_interval),
        },
        "generation_report_path": str(generation_report_path),
        "generation_command": generation_result,
        "generation_summary": generation,
        "repaired_diagnostics_summary": diagnostics,
        "repair_result": {
            "previous_dead_air_flag_count": _int(source.get("previous_dead_air_flag_count")),
            "repaired_dead_air_flag_count": _int(flag_counts.get("dead_air_gap", 0)),
            "previous_max_dead_air_ratio": _float(source.get("max_dead_air_ratio")),
            "repaired_max_dead_air_ratio": _float(diagnostics.get("max_dead_air_ratio")),
            "previous_max_interval_max": _int(source.get("max_interval_max")),
            "repaired_max_interval_max": _int(diagnostics.get("max_interval_max")),
            "previous_max_pitch_span": _int(source.get("max_pitch_span")),
            "repaired_max_pitch_span": _int(diagnostics.get("max_pitch_span")),
            "dead_air_removed": bool(dead_air_removed),
            "dead_air_ratio_reduced": bool(dead_air_ratio_reduced),
            "max_interval_guard_kept": bool(max_interval_guard_kept),
            "timing_phrase_repair_passed": bool(repair_passed),
        },
        "readiness": {
            "boundary": BOUNDARY,
            "timing_phrase_repair_completed": bool(command_succeeded),
            "direct_generated_midi_written": bool(generation["all_midi_paths_exist"]),
            "strict_review_gate_passed": bool(generation["passed_strict_review_gate"]),
            "timing_phrase_repair_passed": bool(repair_passed),
            "model_direct_generation_quality_claimed": False,
            "midi_to_solo_musical_quality_claimed": False,
            "human_audio_preference_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
        },
        "decision": {
            "current_boundary": BOUNDARY,
            "next_boundary": next_boundary,
            "auto_progress_allowed": True,
            "critical_user_input_required": False,
            "reason": (
                "timing dead-air diagnostics repaired; route to listening review package"
                if repair_passed
                else "timing dead-air diagnostics still require follow-up"
            ),
        },
        "not_proven": [
            "model_direct_generation_quality",
            "midi_to_solo_musical_quality",
            "human_audio_preference",
            "broad_trained_model_quality",
            "brad_style_adaptation",
        ],
        "next_recommended_issue": (
            "Stage B MIDI-to-solo model-direct listening review package"
            if repair_passed
            else "Stage B MIDI-to-solo model-direct timing phrase repair follow-up"
        ),
    }


def validate_timing_phrase_repair_report(
    report: dict[str, Any],
    *,
    expected_boundary: str | None,
    expected_next_boundary: str | None,
    require_repair_completed: bool,
    require_timing_repair_passed: bool,
    require_no_quality_claim: bool,
) -> dict[str, Any]:
    boundary = str(report.get("boundary") or "")
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    result = _dict(report.get("repair_result"))
    generation = _dict(report.get("generation_summary"))
    diagnostics = _dict(report.get("repaired_diagnostics_summary"))
    if expected_boundary and boundary != expected_boundary:
        raise StageBMidiToSoloModelDirectTimingPhraseRepairError(
            f"expected boundary {expected_boundary}, got {boundary}"
        )
    if expected_next_boundary and str(decision.get("next_boundary") or "") != expected_next_boundary:
        raise StageBMidiToSoloModelDirectTimingPhraseRepairError("unexpected next boundary")
    if require_repair_completed and not bool(readiness.get("timing_phrase_repair_completed", False)):
        raise StageBMidiToSoloModelDirectTimingPhraseRepairError("repair command must complete")
    if not bool(readiness.get("direct_generated_midi_written", False)):
        raise StageBMidiToSoloModelDirectTimingPhraseRepairError("generated MIDI files required")
    if not bool(generation.get("passed_strict_review_gate", False)):
        raise StageBMidiToSoloModelDirectTimingPhraseRepairError("strict review gate required")
    if require_timing_repair_passed and not bool(result.get("timing_phrase_repair_passed", False)):
        raise StageBMidiToSoloModelDirectTimingPhraseRepairError("timing phrase repair should pass")
    if _int(diagnostics.get("candidate_count")) < 3:
        raise StageBMidiToSoloModelDirectTimingPhraseRepairError("repaired diagnostics candidate count required")
    if bool(decision.get("critical_user_input_required", True)):
        raise StageBMidiToSoloModelDirectTimingPhraseRepairError("critical user input should not be required")
    if require_no_quality_claim:
        blocked = [
            "model_direct_generation_quality_claimed",
            "midi_to_solo_musical_quality_claimed",
            "human_audio_preference_claimed",
            "broad_trained_model_quality_claimed",
            "brad_style_adaptation_claimed",
        ]
        claimed = [name for name in blocked if bool(readiness.get(name, True))]
        if claimed:
            raise StageBMidiToSoloModelDirectTimingPhraseRepairError(f"unexpected quality claim: {claimed}")
    return {
        "boundary": boundary,
        "next_boundary": str(decision.get("next_boundary") or ""),
        "strict_valid_sample_count": _int(generation.get("strict_valid_sample_count")),
        "previous_dead_air_flag_count": _int(result.get("previous_dead_air_flag_count")),
        "repaired_dead_air_flag_count": _int(result.get("repaired_dead_air_flag_count")),
        "previous_max_dead_air_ratio": _float(result.get("previous_max_dead_air_ratio")),
        "repaired_max_dead_air_ratio": _float(result.get("repaired_max_dead_air_ratio")),
        "previous_max_interval_max": _int(result.get("previous_max_interval_max")),
        "repaired_max_interval_max": _int(result.get("repaired_max_interval_max")),
        "dead_air_removed": bool(result.get("dead_air_removed", False)),
        "dead_air_ratio_reduced": bool(result.get("dead_air_ratio_reduced", False)),
        "max_interval_guard_kept": bool(result.get("max_interval_guard_kept", False)),
        "timing_phrase_repair_passed": bool(result.get("timing_phrase_repair_passed", False)),
        "model_direct_generation_quality_claimed": bool(
            readiness.get("model_direct_generation_quality_claimed", True)
        ),
        "human_audio_preference_claimed": bool(readiness.get("human_audio_preference_claimed", True)),
        "critical_user_input_required": bool(decision.get("critical_user_input_required", True)),
        "next_recommended_issue": str(report.get("next_recommended_issue") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    readiness = report["readiness"]
    decision = report["decision"]
    result = report["repair_result"]
    config = report["repair_config"]
    lines = [
        "# Stage B MIDI-to-Solo Model-Direct Timing Phrase Repair",
        "",
        "## Summary",
        "",
        f"- boundary: `{report['boundary']}`",
        f"- next boundary: `{decision['next_boundary']}`",
        f"- strict review gate passed: `{_bool_token(readiness['strict_review_gate_passed'])}`",
        f"- timing phrase repair passed: `{_bool_token(readiness['timing_phrase_repair_passed'])}`",
        f"- jazz rhythm profile: `{config['jazz_rhythm_profile']}`",
        f"- fill duration to next position: `{_bool_token(config['fill_duration_to_next_position'])}`",
        f"- model-direct generation quality claimed: `{_bool_token(readiness['model_direct_generation_quality_claimed'])}`",
        f"- human/audio preference claimed: `{_bool_token(readiness['human_audio_preference_claimed'])}`",
        "",
        "## Before / After",
        "",
        f"- dead-air flag count: `{result['previous_dead_air_flag_count']}` -> `{result['repaired_dead_air_flag_count']}`",
        f"- max dead-air ratio: `{result['previous_max_dead_air_ratio']:.4f}` -> `{result['repaired_max_dead_air_ratio']:.4f}`",
        f"- max interval max: `{result['previous_max_interval_max']}` -> `{result['repaired_max_interval_max']}`",
        f"- max pitch span: `{result['previous_max_pitch_span']}` -> `{result['repaired_max_pitch_span']}`",
        "",
        "## Not Proven",
        "",
    ]
    for item in report.get("not_proven", []):
        lines.append(f"- `{item}`")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run model-direct timing phrase repair")
    parser.add_argument("--source_pitch_repair", type=str, required=True)
    parser.add_argument("--sequence_budget_repair", type=str, required=True)
    parser.add_argument("--context_report", type=str, required=True)
    parser.add_argument("--repaired_training_scale_smoke", type=str, required=True)
    parser.add_argument("--output_root", type=str, default="outputs/stage_b_midi_to_solo_model_direct_timing_phrase_repair")
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--issue_number", type=int, default=510)
    parser.add_argument("--target_bars", type=int, default=8)
    parser.add_argument("--note_groups_per_bar", type=int, default=4)
    parser.add_argument("--num_samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=510)
    parser.add_argument("--max_sequence", type=int, default=160)
    parser.add_argument("--chord_pitch_mode", type=str, default="tones_tensions")
    parser.add_argument("--max_simultaneous_notes", type=int, default=1)
    parser.add_argument("--min_valid_samples", type=int, default=1)
    parser.add_argument("--min_strict_valid_samples", type=int, default=1)
    parser.add_argument("--constrained_pitch_min", type=int, default=55)
    parser.add_argument("--constrained_pitch_max", type=int, default=79)
    parser.add_argument("--constrained_max_adjacent_interval", type=int, default=9)
    parser.add_argument("--jazz_rhythm_positions", action="store_true", default=True)
    parser.add_argument("--jazz_duration_tokens", action="store_true", default=False)
    parser.add_argument("--jazz_rhythm_profile", type=str, default="compact_phrase")
    parser.add_argument("--cap_duration_to_next_position", action="store_true", default=False)
    parser.add_argument("--fill_duration_to_next_position", action="store_true", default=True)
    parser.add_argument("--dead_air_threshold_seconds", type=float, default=0.5)
    parser.add_argument("--expected_boundary", type=str, default="")
    parser.add_argument("--expected_next_boundary", type=str, default="")
    parser.add_argument("--require_repair_completed", action="store_true")
    parser.add_argument("--require_timing_repair_passed", action="store_true")
    parser.add_argument("--require_no_quality_claim", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    source_pitch_repair = read_json(Path(args.source_pitch_repair))
    sequence_budget_repair = read_json(Path(args.sequence_budget_repair))
    context_report = read_json(Path(args.context_report))
    repaired_training_scale_smoke = read_json(Path(args.repaired_training_scale_smoke))
    context_summary = summarize_context(context_report, target_bars=int(args.target_bars))
    scale_summary = summarize_repaired_scale_smoke(repaired_training_scale_smoke)
    generation_output_root = output_dir / "generation_probe"
    generation_run_id = "timing_phrase_repair"
    generation_result = run_command(
        build_generation_command(
            args=args,
            checkpoint_dir=Path(scale_summary["checkpoint_dir"]),
            generation_output_root=generation_output_root,
            generation_run_id=generation_run_id,
            context_summary=context_summary,
        )
    )
    generation_report_path = generation_output_root / generation_run_id / "report.json"
    generation_report = read_json(generation_report_path) if generation_report_path.exists() else {}
    report = build_timing_phrase_repair_report(
        source_pitch_repair=source_pitch_repair,
        sequence_budget_repair=sequence_budget_repair,
        context_report=context_report,
        repaired_training_scale_smoke=repaired_training_scale_smoke,
        generation_result=generation_result,
        generation_report=generation_report,
        generation_report_path=generation_report_path,
        output_dir=output_dir,
        issue_number=int(args.issue_number),
        target_bars=int(args.target_bars),
        note_groups_per_bar=int(args.note_groups_per_bar),
        jazz_rhythm_profile=str(args.jazz_rhythm_profile),
        constrained_pitch_min=int(args.constrained_pitch_min),
        constrained_pitch_max=int(args.constrained_pitch_max),
        constrained_max_adjacent_interval=int(args.constrained_max_adjacent_interval),
        dead_air_threshold_seconds=float(args.dead_air_threshold_seconds),
    )
    summary = validate_timing_phrase_repair_report(
        report,
        expected_boundary=str(args.expected_boundary or ""),
        expected_next_boundary=str(args.expected_next_boundary or ""),
        require_repair_completed=bool(args.require_repair_completed),
        require_timing_repair_passed=bool(args.require_timing_repair_passed),
        require_no_quality_claim=bool(args.require_no_quality_claim),
    )
    write_json(output_dir / "stage_b_midi_to_solo_model_direct_timing_phrase_repair.json", report)
    write_json(
        output_dir / "stage_b_midi_to_solo_model_direct_timing_phrase_repair_validation_summary.json",
        summary,
    )
    markdown = markdown_report(report)
    write_text(output_dir / "stage_b_midi_to_solo_model_direct_timing_phrase_repair.md", markdown)
    if args.doc_path:
        write_text(Path(args.doc_path), markdown)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
