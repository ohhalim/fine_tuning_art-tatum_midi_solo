"""Run pitch-contour repair probe for dead-air/timing repaired MIDI candidates."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pretty_midi

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from scripts.assess_stage_b_generic_base_readiness import read_json, write_json, write_text  # noqa: E402
from scripts.decide_stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_pitch_contour import (  # noqa: E402
    BOUNDARY as DECISION_BOUNDARY,
    NEXT_BOUNDARY as DECISION_NEXT_BOUNDARY,
    validate_pitch_contour_decision_report,
)
from scripts.run_stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_probe import (  # noqa: E402
    BOUNDARY as DEAD_AIR_REPAIR_BOUNDARY,
    load_midi_notes,
    max_simultaneous_notes,
    objective_metrics_for_path,
)


class StageBMidiToSoloModelConditionedInputPathDeadAirTimingRepairPitchContourProbeError(
    ValueError
):
    pass


BOUNDARY = (
    "stage_b_midi_to_solo_model_conditioned_input_path_"
    "dead_air_timing_repair_pitch_contour_probe"
)
NEXT_BOUNDARY = (
    "stage_b_midi_to_solo_model_conditioned_input_path_"
    "dead_air_timing_repair_pitch_contour_audio_package"
)
FAIL_NEXT_BOUNDARY = (
    "stage_b_midi_to_solo_model_conditioned_input_path_"
    "dead_air_timing_repair_pitch_contour_followup"
)
SCHEMA_VERSION = (
    "stage_b_midi_to_solo_model_conditioned_input_path_"
    "dead_air_timing_repair_pitch_contour_probe_v1"
)

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
        raise StageBMidiToSoloModelConditionedInputPathDeadAirTimingRepairPitchContourProbeError(
            f"unexpected quality claim in {label}: {claimed}"
        )


def validate_pitch_contour_decision_source(report: dict[str, Any]) -> dict[str, Any]:
    try:
        summary = validate_pitch_contour_decision_report(
            report,
            expected_boundary=DECISION_BOUNDARY,
            expected_next_boundary=DECISION_NEXT_BOUNDARY,
            require_pitch_contour_decision=True,
            require_repair_probe=True,
            require_no_quality_claim=True,
        )
    except ValueError as exc:
        raise StageBMidiToSoloModelConditionedInputPathDeadAirTimingRepairPitchContourProbeError(
            str(exc)
        ) from exc
    if _int(summary["source_max_interval"]) <= _int(summary["target_max_interval"]):
        raise StageBMidiToSoloModelConditionedInputPathDeadAirTimingRepairPitchContourProbeError(
            "pitch-contour repair source must exceed target interval"
        )
    return summary


def validate_dead_air_repair_probe_source(
    report: dict[str, Any],
    *,
    min_candidate_count: int,
) -> dict[str, Any]:
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    summary = _dict(report.get("summary"))
    candidate_repairs = [_dict(item) for item in _list(report.get("candidate_repairs"))]
    if str(report.get("boundary") or "") != DEAD_AIR_REPAIR_BOUNDARY:
        raise StageBMidiToSoloModelConditionedInputPathDeadAirTimingRepairPitchContourProbeError(
            "dead-air timing repair probe boundary required"
        )
    if not bool(readiness.get("dead_air_timing_repair_passed", False)):
        raise StageBMidiToSoloModelConditionedInputPathDeadAirTimingRepairPitchContourProbeError(
            "dead-air timing repair pass required"
        )
    if bool(decision.get("critical_user_input_required", True)):
        raise StageBMidiToSoloModelConditionedInputPathDeadAirTimingRepairPitchContourProbeError(
            "critical user input should not be required"
        )
    if len(candidate_repairs) < int(min_candidate_count):
        raise StageBMidiToSoloModelConditionedInputPathDeadAirTimingRepairPitchContourProbeError(
            "candidate repair count below minimum"
        )
    for row in candidate_repairs[: int(min_candidate_count)]:
        repaired_path = Path(str(row.get("repaired_midi_path") or ""))
        if not repaired_path.exists():
            raise StageBMidiToSoloModelConditionedInputPathDeadAirTimingRepairPitchContourProbeError(
                f"dead-air repaired MIDI missing: {repaired_path}"
            )
        if not bool(row.get("candidate_repair_passed", False)):
            raise StageBMidiToSoloModelConditionedInputPathDeadAirTimingRepairPitchContourProbeError(
                "dead-air repaired candidate must pass"
            )
    _require_no_quality_claim(readiness, label="dead-air repair probe readiness")
    return {
        "boundary": DEAD_AIR_REPAIR_BOUNDARY,
        "candidate_repairs": candidate_repairs[: int(min_candidate_count)],
        "source_max_interval": _int(summary.get("max_repaired_interval")),
        "source_dead_air_max": _float(summary.get("repaired_dead_air_max")),
        "max_added_note_ratio": _float(summary.get("max_added_note_ratio")),
        "max_postprocess_removal_ratio": _float(summary.get("max_postprocess_removal_ratio")),
        "max_repaired_simultaneous_notes": _int(summary.get("max_repaired_simultaneous_notes")),
    }


def map_to_preferred_range(
    pitch: int,
    *,
    preferred_pitch_min: int,
    preferred_pitch_max: int,
) -> int:
    mapped = int(pitch)
    while mapped < int(preferred_pitch_min):
        mapped += 12
    while mapped > int(preferred_pitch_max):
        mapped -= 12
    return max(int(preferred_pitch_min), min(int(preferred_pitch_max), int(mapped)))


def choose_contour_pitch(
    original_pitch: int,
    *,
    previous_pitch: int | None,
    preferred_pitch_min: int,
    preferred_pitch_max: int,
    max_adjacent_interval: int,
) -> int:
    base = map_to_preferred_range(
        int(original_pitch),
        preferred_pitch_min=int(preferred_pitch_min),
        preferred_pitch_max=int(preferred_pitch_max),
    )
    if previous_pitch is None:
        return int(base)
    candidates: list[int] = []
    for octave_shift in range(-8, 9):
        candidate = int(original_pitch) + (12 * octave_shift)
        if int(preferred_pitch_min) <= candidate <= int(preferred_pitch_max):
            candidates.append(candidate)
    if not candidates:
        candidates.append(base)
    candidates.sort(
        key=lambda candidate: (
            abs(candidate - int(previous_pitch)),
            abs(candidate - int(original_pitch)),
            candidate,
        )
    )
    best = candidates[0]
    if abs(best - int(previous_pitch)) <= int(max_adjacent_interval):
        return int(best)
    direction = 1 if best >= int(previous_pitch) else -1
    bounded = int(previous_pitch) + (direction * int(max_adjacent_interval))
    return map_to_preferred_range(
        bounded,
        preferred_pitch_min=int(preferred_pitch_min),
        preferred_pitch_max=int(preferred_pitch_max),
    )


def write_pitch_contour_repaired_midi(
    source_midi_path: str | Path,
    repaired_midi_path: str | Path,
    *,
    preferred_pitch_min: int,
    preferred_pitch_max: int,
    max_adjacent_interval: int,
) -> dict[str, Any]:
    source_midi_path = Path(source_midi_path)
    repaired_midi_path = Path(repaired_midi_path)
    source = pretty_midi.PrettyMIDI(str(source_midi_path))
    source_notes = load_midi_notes(source_midi_path)
    if not source_notes:
        raise StageBMidiToSoloModelConditionedInputPathDeadAirTimingRepairPitchContourProbeError(
            f"source MIDI has no notes: {source_midi_path}"
        )
    repaired = pretty_midi.PrettyMIDI(initial_tempo=float(source.estimate_tempo() or 120.0))
    program = 0
    for instrument in source.instruments:
        if not instrument.is_drum:
            program = int(instrument.program)
            break
    repaired_instrument = pretty_midi.Instrument(
        program=program,
        is_drum=False,
        name="pitch_contour_repaired_solo",
    )
    previous_pitch: int | None = None
    changed_note_count = 0
    max_pitch_shift_abs = 0
    repaired_notes: list[pretty_midi.Note] = []
    for note in source_notes:
        pitch = choose_contour_pitch(
            int(note.pitch),
            previous_pitch=previous_pitch,
            preferred_pitch_min=int(preferred_pitch_min),
            preferred_pitch_max=int(preferred_pitch_max),
            max_adjacent_interval=int(max_adjacent_interval),
        )
        if pitch != int(note.pitch):
            changed_note_count += 1
            max_pitch_shift_abs = max(max_pitch_shift_abs, abs(pitch - int(note.pitch)))
        repaired_notes.append(
            pretty_midi.Note(
                velocity=max(1, min(127, int(note.velocity or 80))),
                pitch=int(pitch),
                start=float(note.start),
                end=float(note.end),
            )
        )
        previous_pitch = int(pitch)
    repaired_instrument.notes = repaired_notes
    repaired.instruments.append(repaired_instrument)
    repaired_midi_path.parent.mkdir(parents=True, exist_ok=True)
    repaired.write(str(repaired_midi_path))
    return {
        "source_note_count": int(len(source_notes)),
        "repaired_note_count": int(len(repaired_notes)),
        "changed_note_count": int(changed_note_count),
        "pitch_changed_ratio": float(changed_note_count / max(1, len(source_notes))),
        "max_pitch_shift_abs": int(max_pitch_shift_abs),
    }


def build_pitch_contour_probe_report(
    *,
    pitch_contour_decision_report: dict[str, Any],
    dead_air_repair_probe_report: dict[str, Any],
    output_dir: Path,
    issue_number: int,
    min_repaired_candidates: int,
    dead_air_threshold_seconds: float,
    preferred_pitch_min: int,
    preferred_pitch_max: int,
    max_adjacent_interval: int,
    min_unique_pitch_count: int,
) -> dict[str, Any]:
    decision_source = validate_pitch_contour_decision_source(pitch_contour_decision_report)
    dead_air_source = validate_dead_air_repair_probe_source(
        dead_air_repair_probe_report,
        min_candidate_count=int(min_repaired_candidates),
    )
    repaired_dir = output_dir / "midi"
    candidate_repairs: list[dict[str, Any]] = []
    for row in dead_air_source["candidate_repairs"]:
        rank = _int(row.get("rank"))
        sample_index = _int(row.get("sample_index"))
        source_path = Path(str(row.get("repaired_midi_path") or ""))
        repaired_path = repaired_dir / (
            f"rank_{rank:02d}_sample_{sample_index:02d}_pitch_contour_repair.mid"
        )
        before = objective_metrics_for_path(
            source_path,
            dead_air_threshold_seconds=float(dead_air_threshold_seconds),
        )
        pitch_stats = write_pitch_contour_repaired_midi(
            source_path,
            repaired_path,
            preferred_pitch_min=int(preferred_pitch_min),
            preferred_pitch_max=int(preferred_pitch_max),
            max_adjacent_interval=int(max_adjacent_interval),
        )
        after = objective_metrics_for_path(
            repaired_path,
            dead_air_threshold_seconds=float(dead_air_threshold_seconds),
        )
        repaired_notes = load_midi_notes(repaired_path)
        candidate_passed = bool(
            _int(after.get("max_interval")) <= _int(decision_source["target_max_interval"])
            and _float(after.get("dead_air_ratio")) <= _float(decision_source["target_dead_air_max"])
            and _int(after.get("note_count")) >= _int(before.get("note_count"))
            and _int(after.get("unique_pitch_count")) >= int(min_unique_pitch_count)
            and max_simultaneous_notes(repaired_notes) <= 1
        )
        candidate_repairs.append(
            {
                "rank": rank,
                "sample_index": sample_index,
                "sample_seed": _int(row.get("sample_seed")),
                "source_midi_path": str(source_path),
                "repaired_midi_path": str(repaired_path),
                "source_metrics": {
                    "note_count": _int(before.get("note_count")),
                    "unique_pitch_count": _int(before.get("unique_pitch_count")),
                    "max_simultaneous_notes": _int(before.get("max_simultaneous_notes")),
                    "dead_air_ratio": _float(before.get("dead_air_ratio")),
                    "max_interval": _int(before.get("max_interval")),
                    "pitch_span": _int(before.get("pitch_span")),
                },
                "repaired_metrics": {
                    "note_count": _int(after.get("note_count")),
                    "unique_pitch_count": _int(after.get("unique_pitch_count")),
                    "max_simultaneous_notes": max_simultaneous_notes(repaired_notes),
                    "dead_air_ratio": _float(after.get("dead_air_ratio")),
                    "max_interval": _int(after.get("max_interval")),
                    "pitch_span": _int(after.get("pitch_span")),
                },
                "pitch_repair_stats": pitch_stats,
                "candidate_repair_passed": bool(candidate_passed),
            }
        )

    repaired_pass_count = sum(
        1 for row in candidate_repairs if bool(row.get("candidate_repair_passed", False))
    )
    source_max_interval = max(
        (_int(row.get("source_metrics", {}).get("max_interval")) for row in candidate_repairs),
        default=0,
    )
    repaired_max_interval = max(
        (_int(row.get("repaired_metrics", {}).get("max_interval")) for row in candidate_repairs),
        default=0,
    )
    repaired_dead_air_max = max(
        (_float(row.get("repaired_metrics", {}).get("dead_air_ratio")) for row in candidate_repairs),
        default=0.0,
    )
    pitch_contour_target_met = bool(
        repaired_pass_count >= int(min_repaired_candidates)
        and repaired_max_interval <= _int(decision_source["target_max_interval"])
        and repaired_dead_air_max <= _float(decision_source["target_dead_air_max"])
    )
    next_boundary = NEXT_BOUNDARY if pitch_contour_target_met else FAIL_NEXT_BOUNDARY
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
            "+00:00", "Z"
        ),
        "output_dir": str(output_dir),
        "issue_number": int(issue_number),
        "boundary": BOUNDARY,
        "source_boundaries": {
            "pitch_contour_decision": DECISION_BOUNDARY,
            "dead_air_repair_probe": DEAD_AIR_REPAIR_BOUNDARY,
        },
        "repair_config": {
            "strategy": "pitch_class_octave_contour_fold",
            "dead_air_threshold_seconds": float(dead_air_threshold_seconds),
            "preferred_pitch_min": int(preferred_pitch_min),
            "preferred_pitch_max": int(preferred_pitch_max),
            "max_adjacent_interval": int(max_adjacent_interval),
            "min_repaired_candidates": int(min_repaired_candidates),
            "min_unique_pitch_count": int(min_unique_pitch_count),
        },
        "guardrails": {
            "target_max_interval": _int(decision_source["target_max_interval"]),
            "source_decision_max_interval": _int(decision_source["source_max_interval"]),
            "required_interval_reduction_min": _int(
                decision_source["required_interval_reduction_min"]
            ),
            "target_dead_air_max": _float(decision_source["target_dead_air_max"]),
            "source_dead_air_max": _float(dead_air_source["source_dead_air_max"]),
            "max_simultaneous_notes": 1,
            "min_unique_pitch_count": int(min_unique_pitch_count),
            "source_max_added_note_ratio": _float(decision_source["source_max_added_note_ratio"]),
            "added_note_ratio_review_required": bool(
                decision_source["added_note_ratio_review_required"]
            ),
        },
        "candidate_repairs": candidate_repairs,
        "summary": {
            "source_candidate_count": int(len(candidate_repairs)),
            "repaired_candidate_count": int(len(candidate_repairs)),
            "repaired_pass_count": int(repaired_pass_count),
            "source_max_interval": int(source_max_interval),
            "repaired_max_interval": int(repaired_max_interval),
            "target_max_interval": _int(decision_source["target_max_interval"]),
            "interval_reduction": int(source_max_interval - repaired_max_interval),
            "required_interval_reduction_min": _int(
                decision_source["required_interval_reduction_min"]
            ),
            "source_dead_air_max": _float(dead_air_source["source_dead_air_max"]),
            "repaired_dead_air_max": float(repaired_dead_air_max),
            "target_dead_air_max": _float(decision_source["target_dead_air_max"]),
            "max_repaired_simultaneous_notes": max(
                (
                    _int(row.get("repaired_metrics", {}).get("max_simultaneous_notes"))
                    for row in candidate_repairs
                ),
                default=0,
            ),
            "min_repaired_unique_pitch_count": min(
                (
                    _int(row.get("repaired_metrics", {}).get("unique_pitch_count"))
                    for row in candidate_repairs
                ),
                default=0,
            ),
            "max_pitch_changed_ratio": max(
                (
                    _float(row.get("pitch_repair_stats", {}).get("pitch_changed_ratio"))
                    for row in candidate_repairs
                ),
                default=0.0,
            ),
            "pitch_contour_repair_passed": bool(pitch_contour_target_met),
        },
        "readiness": {
            "boundary": BOUNDARY,
            "pitch_contour_repair_probe_completed": True,
            "repaired_ranked_midi_written": bool(len(candidate_repairs) >= int(min_repaired_candidates)),
            "pitch_contour_repair_passed": bool(pitch_contour_target_met),
            "pitch_contour_audio_render_required": bool(pitch_contour_target_met),
            "current_evidence_consolidation_ready": False,
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
                "pitch-contour objective target passed; render repaired MIDI to WAV next"
                if pitch_contour_target_met
                else "pitch-contour objective target still requires follow-up"
            ),
        },
        "not_proven": [
            "human_audio_preference",
            "midi_to_solo_musical_quality",
            "audio_rendered_quality",
            "model_checkpoint_generation_quality",
            "broad_trained_model_quality",
            "brad_style_adaptation",
        ],
        "next_recommended_issue": (
            "Stage B MIDI-to-solo model-conditioned input path dead-air timing repair pitch contour audio package"
            if pitch_contour_target_met
            else "Stage B MIDI-to-solo model-conditioned input path dead-air timing repair pitch contour follow-up"
        ),
    }


def validate_pitch_contour_probe_report(
    report: dict[str, Any],
    *,
    expected_boundary: str | None,
    expected_next_boundary: str | None,
    min_repaired_candidates: int,
    require_repair_passed: bool,
    require_no_quality_claim: bool,
) -> dict[str, Any]:
    boundary = str(report.get("boundary") or "")
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    summary = _dict(report.get("summary"))
    candidate_repairs = [_dict(item) for item in _list(report.get("candidate_repairs"))]
    if expected_boundary and boundary != expected_boundary:
        raise StageBMidiToSoloModelConditionedInputPathDeadAirTimingRepairPitchContourProbeError(
            f"expected boundary {expected_boundary}, got {boundary}"
        )
    if expected_next_boundary and str(decision.get("next_boundary") or "") != expected_next_boundary:
        raise StageBMidiToSoloModelConditionedInputPathDeadAirTimingRepairPitchContourProbeError(
            "unexpected next boundary"
        )
    if len(candidate_repairs) < int(min_repaired_candidates):
        raise StageBMidiToSoloModelConditionedInputPathDeadAirTimingRepairPitchContourProbeError(
            "candidate repair count below threshold"
        )
    for row in candidate_repairs[: int(min_repaired_candidates)]:
        if not Path(str(row.get("repaired_midi_path") or "")).exists():
            raise StageBMidiToSoloModelConditionedInputPathDeadAirTimingRepairPitchContourProbeError(
                "repaired MIDI path missing"
            )
        if require_repair_passed and not bool(row.get("candidate_repair_passed", False)):
            raise StageBMidiToSoloModelConditionedInputPathDeadAirTimingRepairPitchContourProbeError(
                "candidate repair should pass"
            )
    if require_repair_passed and not bool(readiness.get("pitch_contour_repair_passed", False)):
        raise StageBMidiToSoloModelConditionedInputPathDeadAirTimingRepairPitchContourProbeError(
            "pitch-contour repair should pass"
        )
    if _int(summary.get("repaired_max_interval")) > _int(summary.get("target_max_interval")):
        raise StageBMidiToSoloModelConditionedInputPathDeadAirTimingRepairPitchContourProbeError(
            "repaired max interval exceeds target"
        )
    if _float(summary.get("repaired_dead_air_max")) > _float(summary.get("target_dead_air_max")):
        raise StageBMidiToSoloModelConditionedInputPathDeadAirTimingRepairPitchContourProbeError(
            "repaired dead-air exceeds target"
        )
    if bool(decision.get("critical_user_input_required", True)):
        raise StageBMidiToSoloModelConditionedInputPathDeadAirTimingRepairPitchContourProbeError(
            "critical user input should not be required"
        )
    if require_no_quality_claim:
        _require_no_quality_claim(readiness, label="pitch-contour probe readiness")
    return {
        "boundary": boundary,
        "next_boundary": str(decision.get("next_boundary") or ""),
        "pitch_contour_repair_probe_completed": bool(
            readiness.get("pitch_contour_repair_probe_completed", False)
        ),
        "repaired_ranked_midi_written": bool(
            readiness.get("repaired_ranked_midi_written", False)
        ),
        "pitch_contour_repair_passed": bool(
            readiness.get("pitch_contour_repair_passed", False)
        ),
        "pitch_contour_audio_render_required": bool(
            readiness.get("pitch_contour_audio_render_required", False)
        ),
        "source_candidate_count": _int(summary.get("source_candidate_count")),
        "repaired_candidate_count": _int(summary.get("repaired_candidate_count")),
        "repaired_pass_count": _int(summary.get("repaired_pass_count")),
        "source_max_interval": _int(summary.get("source_max_interval")),
        "repaired_max_interval": _int(summary.get("repaired_max_interval")),
        "target_max_interval": _int(summary.get("target_max_interval")),
        "interval_reduction": _int(summary.get("interval_reduction")),
        "required_interval_reduction_min": _int(summary.get("required_interval_reduction_min")),
        "source_dead_air_max": _float(summary.get("source_dead_air_max")),
        "repaired_dead_air_max": _float(summary.get("repaired_dead_air_max")),
        "target_dead_air_max": _float(summary.get("target_dead_air_max")),
        "max_repaired_simultaneous_notes": _int(
            summary.get("max_repaired_simultaneous_notes")
        ),
        "min_repaired_unique_pitch_count": _int(
            summary.get("min_repaired_unique_pitch_count")
        ),
        "max_pitch_changed_ratio": _float(summary.get("max_pitch_changed_ratio")),
        "human_audio_preference_claimed": bool(
            readiness.get("human_audio_preference_claimed", True)
        ),
        "midi_to_solo_musical_quality_claimed": bool(
            readiness.get("midi_to_solo_musical_quality_claimed", True)
        ),
        "critical_user_input_required": bool(decision.get("critical_user_input_required", True)),
        "next_recommended_issue": str(report.get("next_recommended_issue") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    summary = report["summary"]
    readiness = report["readiness"]
    decision = report["decision"]
    config = report["repair_config"]
    guardrails = report["guardrails"]
    lines = [
        "# Stage B MIDI-to-Solo Model-Conditioned Input Path Dead-Air Timing Repair Pitch Contour Probe",
        "",
        "## Summary",
        "",
        f"- boundary: `{report['boundary']}`",
        f"- next boundary: `{decision['next_boundary']}`",
        f"- repair passed: `{_bool_token(summary['pitch_contour_repair_passed'])}`",
        f"- source candidate count: `{summary['source_candidate_count']}`",
        f"- repaired candidate count: `{summary['repaired_candidate_count']}`",
        f"- repaired pass count: `{summary['repaired_pass_count']}`",
        f"- source max interval: `{summary['source_max_interval']}`",
        f"- repaired max interval: `{summary['repaired_max_interval']}`",
        f"- target max interval: `{summary['target_max_interval']}`",
        f"- interval reduction: `{summary['interval_reduction']}`",
        f"- required interval reduction min: `{summary['required_interval_reduction_min']}`",
        f"- source dead-air max: `{summary['source_dead_air_max']:.4f}`",
        f"- repaired dead-air max: `{summary['repaired_dead_air_max']:.4f}`",
        f"- min repaired unique pitch count: `{summary['min_repaired_unique_pitch_count']}`",
        f"- max pitch changed ratio: `{summary['max_pitch_changed_ratio']:.4f}`",
        "",
        "## Repair Config",
        "",
        f"- strategy: `{config['strategy']}`",
        f"- preferred pitch range: `{config['preferred_pitch_min']}`-`{config['preferred_pitch_max']}`",
        f"- max adjacent interval: `{config['max_adjacent_interval']}`",
        f"- min unique pitch count: `{config['min_unique_pitch_count']}`",
        "",
        "## Guardrails",
        "",
        f"- target max interval: `{guardrails['target_max_interval']}`",
        f"- target dead-air max: `{guardrails['target_dead_air_max']:.4f}`",
        f"- max simultaneous notes: `{guardrails['max_simultaneous_notes']}`",
        f"- source max added-note ratio: `{guardrails['source_max_added_note_ratio']:.4f}`",
        f"- added-note ratio review required: `{_bool_token(guardrails['added_note_ratio_review_required'])}`",
        "",
        "## Repaired MIDI",
        "",
    ]
    for row in report["candidate_repairs"]:
        source = row["source_metrics"]
        repaired = row["repaired_metrics"]
        stats = row["pitch_repair_stats"]
        lines.append(
            f"- rank `{row['rank']}` sample `{row['sample_index']}`: "
            f"`{row['repaired_midi_path']}`, max interval `{source['max_interval']}` -> "
            f"`{repaired['max_interval']}`, unique pitch `{source['unique_pitch_count']}` -> "
            f"`{repaired['unique_pitch_count']}`, pitch changed ratio "
            f"`{stats['pitch_changed_ratio']:.4f}`, pass `{_bool_token(row['candidate_repair_passed'])}`"
        )
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            f"- human/audio preference claimed: `{_bool_token(readiness['human_audio_preference_claimed'])}`",
            f"- MIDI-to-solo musical quality claimed: `{_bool_token(readiness['midi_to_solo_musical_quality_claimed'])}`",
            f"- audio rendered quality claimed: `{_bool_token(readiness['audio_rendered_quality_claimed'])}`",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run pitch-contour repair probe for dead-air/timing repaired MIDI candidates"
    )
    parser.add_argument("--pitch_contour_decision_report", type=str, required=True)
    parser.add_argument("--dead_air_repair_probe_report", type=str, required=True)
    parser.add_argument(
        "--output_root",
        type=str,
        default=(
            "outputs/stage_b_midi_to_solo_model_conditioned_input_path_"
            "dead_air_timing_repair_pitch_contour_probe"
        ),
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--issue_number", type=int, default=698)
    parser.add_argument("--min_repaired_candidates", type=int, default=3)
    parser.add_argument("--dead_air_threshold_seconds", type=float, default=0.5)
    parser.add_argument("--preferred_pitch_min", type=int, default=48)
    parser.add_argument("--preferred_pitch_max", type=int, default=88)
    parser.add_argument("--max_adjacent_interval", type=int, default=12)
    parser.add_argument("--min_unique_pitch_count", type=int, default=8)
    parser.add_argument("--expected_boundary", type=str, default="")
    parser.add_argument("--expected_next_boundary", type=str, default="")
    parser.add_argument("--require_repair_passed", action="store_true")
    parser.add_argument("--require_no_quality_claim", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    report = build_pitch_contour_probe_report(
        pitch_contour_decision_report=read_json(Path(args.pitch_contour_decision_report)),
        dead_air_repair_probe_report=read_json(Path(args.dead_air_repair_probe_report)),
        output_dir=output_dir,
        issue_number=int(args.issue_number),
        min_repaired_candidates=int(args.min_repaired_candidates),
        dead_air_threshold_seconds=float(args.dead_air_threshold_seconds),
        preferred_pitch_min=int(args.preferred_pitch_min),
        preferred_pitch_max=int(args.preferred_pitch_max),
        max_adjacent_interval=int(args.max_adjacent_interval),
        min_unique_pitch_count=int(args.min_unique_pitch_count),
    )
    summary = validate_pitch_contour_probe_report(
        report,
        expected_boundary=str(args.expected_boundary or ""),
        expected_next_boundary=str(args.expected_next_boundary or ""),
        min_repaired_candidates=int(args.min_repaired_candidates),
        require_repair_passed=bool(args.require_repair_passed),
        require_no_quality_claim=bool(args.require_no_quality_claim),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        output_dir
        / "stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_pitch_contour_probe.json",
        report,
    )
    write_json(
        output_dir
        / "stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_pitch_contour_probe_validation_summary.json",
        summary,
    )
    markdown = markdown_report(report)
    write_text(
        output_dir
        / "stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_pitch_contour_probe.md",
        markdown,
    )
    if args.doc_path:
        write_text(Path(args.doc_path), markdown)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
