"""Run dead-air/timing repair probe for model-conditioned input-path MIDI candidates."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pretty_midi

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from scripts.assess_stage_b_generic_base_readiness import read_json, write_json, write_text  # noqa: E402
from scripts.decide_stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_decision import (  # noqa: E402
    BOUNDARY as DECISION_BOUNDARY,
    NEXT_BOUNDARY as DECISION_NEXT_BOUNDARY,
)
from scripts.diagnose_stage_b_midi_to_solo_model_direct_phrase_quality import (  # noqa: E402
    note_metrics_for_path,
)
from scripts.export_stage_b_midi_to_solo_model_conditioned_input_path_candidates import (  # noqa: E402
    BOUNDARY as CANDIDATE_EXPORT_BOUNDARY,
)


class StageBMidiToSoloModelConditionedInputPathDeadAirTimingRepairProbeError(
    ValueError
):
    pass


BOUNDARY = (
    "stage_b_midi_to_solo_model_conditioned_input_path_"
    "dead_air_timing_repair_probe"
)
NEXT_BOUNDARY = (
    "stage_b_midi_to_solo_model_conditioned_input_path_"
    "dead_air_timing_repair_audio_package"
)
FAIL_NEXT_BOUNDARY = (
    "stage_b_midi_to_solo_model_conditioned_input_path_"
    "dead_air_timing_repair_followup"
)
SCHEMA_VERSION = (
    "stage_b_midi_to_solo_model_conditioned_input_path_"
    "dead_air_timing_repair_probe_v1"
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
        raise StageBMidiToSoloModelConditionedInputPathDeadAirTimingRepairProbeError(
            f"unexpected quality claim in {label}: {claimed}"
        )


def max_simultaneous_notes(notes: list[pretty_midi.Note]) -> int:
    events: list[tuple[float, int]] = []
    for note in notes:
        events.append((float(note.start), 1))
        events.append((float(note.end), -1))
    active = 0
    maximum = 0
    for _time, delta in sorted(events, key=lambda item: (item[0], item[1])):
        active = max(0, active + delta)
        maximum = max(maximum, active)
    return maximum


def load_midi_notes(path: str | Path) -> list[pretty_midi.Note]:
    midi = pretty_midi.PrettyMIDI(str(path))
    notes: list[pretty_midi.Note] = []
    for instrument in midi.instruments:
        if not instrument.is_drum:
            notes.extend(instrument.notes)
    return sorted(notes, key=lambda note: (float(note.start), int(note.pitch), float(note.end)))


def objective_metrics_for_path(
    path: str | Path,
    *,
    dead_air_threshold_seconds: float,
) -> dict[str, Any]:
    metrics = dict(
        note_metrics_for_path(
            path,
            dead_air_threshold_seconds=float(dead_air_threshold_seconds),
        )
    )
    notes = load_midi_notes(path)
    metrics["max_simultaneous_notes"] = max_simultaneous_notes(notes)
    return metrics


def map_pitch_to_preferred_range(
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
    return mapped


def repair_candidate_midi(
    source_midi_path: str | Path,
    repaired_midi_path: str | Path,
    *,
    bpm: float,
    max_start_gap_seconds: float,
    min_note_duration_seconds: float,
    fill_note_duration_seconds: float,
    preferred_pitch_min: int,
    preferred_pitch_max: int,
) -> dict[str, Any]:
    source_midi_path = Path(source_midi_path)
    repaired_midi_path = Path(repaired_midi_path)
    source = pretty_midi.PrettyMIDI(str(source_midi_path))
    source_notes: list[pretty_midi.Note] = []
    for instrument in source.instruments:
        if not instrument.is_drum:
            source_notes.extend(instrument.notes)
    source_notes.sort(key=lambda note: (float(note.start), int(note.pitch), float(note.end)))
    if not source_notes:
        raise StageBMidiToSoloModelConditionedInputPathDeadAirTimingRepairProbeError(
            f"source MIDI has no notes: {source_midi_path}"
        )

    repaired = pretty_midi.PrettyMIDI(initial_tempo=float(bpm))
    program = 0
    for instrument in source.instruments:
        if not instrument.is_drum:
            program = int(instrument.program)
            break
    repaired_instrument = pretty_midi.Instrument(
        program=program,
        is_drum=False,
        name="dead_air_timing_repaired_solo",
    )

    repaired_notes: list[pretty_midi.Note] = [
        pretty_midi.Note(
            velocity=max(1, min(127, int(note.velocity or 80))),
            pitch=int(note.pitch),
            start=float(note.start),
            end=float(note.end),
        )
        for note in source_notes
        if float(note.end) > float(note.start)
    ]
    added_notes: list[pretty_midi.Note] = []
    for previous_note, next_note in zip(source_notes, source_notes[1:]):
        gap = float(next_note.start) - float(previous_note.start)
        if gap <= float(max_start_gap_seconds):
            continue
        insert_count = max(1, int(math.ceil(gap / float(max_start_gap_seconds))) - 1)
        for insert_index in range(1, insert_count + 1):
            fraction = float(insert_index) / float(insert_count + 1)
            start = float(previous_note.start) + (gap * fraction)
            if start <= float(previous_note.start) + 0.03:
                continue
            if start >= float(next_note.start) - 0.03:
                continue
            interpolated_pitch = round(
                int(previous_note.pitch)
                + ((int(next_note.pitch) - int(previous_note.pitch)) * fraction)
            )
            pitch = map_pitch_to_preferred_range(
                int(interpolated_pitch),
                preferred_pitch_min=int(preferred_pitch_min),
                preferred_pitch_max=int(preferred_pitch_max),
            )
            duration = min(
                float(fill_note_duration_seconds),
                max(float(min_note_duration_seconds), (gap / float(insert_count + 1)) * 0.75),
            )
            end = min(float(next_note.start) - 0.01, start + duration)
            if end <= start:
                continue
            velocity = round((int(previous_note.velocity or 80) + int(next_note.velocity or 80)) / 2)
            added_notes.append(
                pretty_midi.Note(
                    velocity=max(1, min(127, int(velocity))),
                    pitch=int(pitch),
                    start=float(start),
                    end=float(end),
                )
            )

    all_notes = sorted(
        repaired_notes + added_notes,
        key=lambda note: (float(note.start), int(note.pitch), float(note.end)),
    )
    final_notes: list[pretty_midi.Note] = []
    overlap_adjusted_count = 0
    for index, note in enumerate(all_notes):
        start = float(note.start)
        end = float(note.end)
        if index + 1 < len(all_notes):
            next_start = float(all_notes[index + 1].start)
            if end > next_start - 0.005:
                adjusted_end = max(start + 0.005, next_start - 0.005)
                if adjusted_end < end:
                    overlap_adjusted_count += 1
                end = adjusted_end
        if end - start < 0.005:
            continue
        final_notes.append(
            pretty_midi.Note(
                velocity=int(note.velocity),
                pitch=int(note.pitch),
                start=start,
                end=end,
            )
        )

    repaired_instrument.notes = final_notes
    repaired.instruments.append(repaired_instrument)
    repaired_midi_path.parent.mkdir(parents=True, exist_ok=True)
    repaired.write(str(repaired_midi_path))

    removed_note_count = max(0, len(repaired_notes) + len(added_notes) - len(final_notes))
    return {
        "source_note_count": int(len(source_notes)),
        "added_note_count": int(len(added_notes)),
        "removed_note_count": int(removed_note_count),
        "repaired_note_count": int(len(final_notes)),
        "overlap_adjusted_note_count": int(overlap_adjusted_count),
        "added_note_ratio": float(len(added_notes) / max(1, len(source_notes))),
        "postprocess_removal_ratio": float(removed_note_count / max(1, len(source_notes))),
    }


def validate_repair_decision(report: dict[str, Any]) -> dict[str, Any]:
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    target = _dict(report.get("repair_target"))
    guardrails = _dict(report.get("guardrails"))
    source = _dict(report.get("source_objective_summary"))
    if str(report.get("boundary") or "") != DECISION_BOUNDARY:
        raise StageBMidiToSoloModelConditionedInputPathDeadAirTimingRepairProbeError(
            "dead-air timing repair decision boundary required"
        )
    if str(decision.get("next_boundary") or "") != DECISION_NEXT_BOUNDARY:
        raise StageBMidiToSoloModelConditionedInputPathDeadAirTimingRepairProbeError(
            "decision must route to repair probe"
        )
    if str(target.get("selected_target") or "") != "dead_air_timing_continuity":
        raise StageBMidiToSoloModelConditionedInputPathDeadAirTimingRepairProbeError(
            "dead-air timing continuity target required"
        )
    if not bool(target.get("repair_probe_required", False)):
        raise StageBMidiToSoloModelConditionedInputPathDeadAirTimingRepairProbeError(
            "repair probe requirement missing"
        )
    if _int(target.get("source_dead_air_failure_count")) <= 0:
        raise StageBMidiToSoloModelConditionedInputPathDeadAirTimingRepairProbeError(
            "source dead-air failures required"
        )
    if _float(target.get("required_dead_air_gain_min")) <= 0:
        raise StageBMidiToSoloModelConditionedInputPathDeadAirTimingRepairProbeError(
            "positive dead-air gain requirement required"
        )
    if bool(source.get("preference_fill_allowed", True)):
        raise StageBMidiToSoloModelConditionedInputPathDeadAirTimingRepairProbeError(
            "preference fill must remain blocked"
        )
    if bool(decision.get("critical_user_input_required", True)):
        raise StageBMidiToSoloModelConditionedInputPathDeadAirTimingRepairProbeError(
            "critical user input should not be required"
        )
    _require_no_quality_claim(readiness, label="repair decision readiness")
    return {
        "boundary": DECISION_BOUNDARY,
        "target_dead_air_max": _float(target.get("target_dead_air_max")),
        "required_dead_air_gain_min": _float(target.get("required_dead_air_gain_min")),
        "source_dead_air_failure_count": _int(target.get("source_dead_air_failure_count")),
        "source_dead_air_max": _float(target.get("source_dead_air_max")),
        "min_note_count": _int(guardrails.get("min_note_count")),
        "min_unique_pitch_count": _int(guardrails.get("min_unique_pitch_count")),
        "max_simultaneous_notes": _int(guardrails.get("max_simultaneous_notes")),
        "max_postprocess_removal_ratio": _float(
            guardrails.get("max_postprocess_removal_ratio")
        ),
        "require_preference_fill_blocked": bool(
            guardrails.get("require_preference_fill_blocked", False)
        ),
    }


def validate_candidate_export(report: dict[str, Any], *, min_count: int) -> dict[str, Any]:
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    context = _dict(report.get("input_context"))
    top_candidates = [_dict(item) for item in _list(report.get("top_candidates"))]
    if str(report.get("boundary") or "") != CANDIDATE_EXPORT_BOUNDARY:
        raise StageBMidiToSoloModelConditionedInputPathDeadAirTimingRepairProbeError(
            "candidate export boundary required"
        )
    required_true = [
        "model_conditioned_input_path_candidate_export_completed",
        "ranked_midi_candidates_exported",
        "model_conditioned_ranked_input_path_contract_matched",
    ]
    missing = [name for name in required_true if not bool(readiness.get(name, False))]
    if missing:
        raise StageBMidiToSoloModelConditionedInputPathDeadAirTimingRepairProbeError(
            f"missing candidate export readiness: {missing}"
        )
    if bool(decision.get("critical_user_input_required", False)):
        raise StageBMidiToSoloModelConditionedInputPathDeadAirTimingRepairProbeError(
            "critical user input should not be required"
        )
    if len(top_candidates) < int(min_count):
        raise StageBMidiToSoloModelConditionedInputPathDeadAirTimingRepairProbeError(
            "top candidate count below minimum"
        )
    for row in top_candidates[: int(min_count)]:
        export_path = Path(str(row.get("export_midi_path") or ""))
        if not export_path.exists():
            raise StageBMidiToSoloModelConditionedInputPathDeadAirTimingRepairProbeError(
                f"exported MIDI missing: {export_path}"
            )
        if not bool(row.get("contract_gate_passed", False)):
            raise StageBMidiToSoloModelConditionedInputPathDeadAirTimingRepairProbeError(
                "candidate contract gate must be passed"
            )
    _require_no_quality_claim(readiness, label="candidate export readiness")
    return {
        "boundary": CANDIDATE_EXPORT_BOUNDARY,
        "bpm": _float(context.get("bpm")) or 120.0,
        "bars": _int(context.get("bars")),
        "chord_progression": list(_list(context.get("chord_progression"))),
        "top_candidates": top_candidates[: int(min_count)],
    }


def build_dead_air_timing_repair_probe_report(
    *,
    repair_decision_report: dict[str, Any],
    candidate_export_report: dict[str, Any],
    output_dir: Path,
    issue_number: int,
    min_repaired_candidates: int,
    dead_air_threshold_seconds: float,
    max_start_gap_seconds: float,
    fill_note_duration_seconds: float,
    min_note_duration_seconds: float,
    preferred_pitch_min: int,
    preferred_pitch_max: int,
) -> dict[str, Any]:
    decision_source = validate_repair_decision(repair_decision_report)
    export_source = validate_candidate_export(
        candidate_export_report,
        min_count=int(min_repaired_candidates),
    )
    repaired_dir = output_dir / "midi"
    candidate_repairs: list[dict[str, Any]] = []
    for row in export_source["top_candidates"]:
        rank = _int(row.get("rank"))
        sample_index = _int(row.get("sample_index"))
        source_path = Path(str(row.get("export_midi_path") or ""))
        repaired_path = repaired_dir / (
            f"rank_{rank:02d}_sample_{sample_index:02d}_dead_air_timing_repair.mid"
        )
        before = objective_metrics_for_path(
            source_path,
            dead_air_threshold_seconds=float(dead_air_threshold_seconds),
        )
        repair_stats = repair_candidate_midi(
            source_path,
            repaired_path,
            bpm=float(export_source["bpm"]),
            max_start_gap_seconds=float(max_start_gap_seconds),
            min_note_duration_seconds=float(min_note_duration_seconds),
            fill_note_duration_seconds=float(fill_note_duration_seconds),
            preferred_pitch_min=int(preferred_pitch_min),
            preferred_pitch_max=int(preferred_pitch_max),
        )
        after = objective_metrics_for_path(
            repaired_path,
            dead_air_threshold_seconds=float(dead_air_threshold_seconds),
        )
        candidate_passed = bool(
            _float(after.get("dead_air_ratio")) <= _float(decision_source["target_dead_air_max"])
            and _int(after.get("note_count")) >= _int(decision_source["min_note_count"])
            and _int(after.get("unique_pitch_count"))
            >= _int(decision_source["min_unique_pitch_count"])
            and _int(after.get("max_simultaneous_notes"))
            <= _int(decision_source["max_simultaneous_notes"])
            and _float(repair_stats.get("postprocess_removal_ratio"))
            <= _float(decision_source["max_postprocess_removal_ratio"])
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
                    "max_simultaneous_notes": _int(after.get("max_simultaneous_notes")),
                    "dead_air_ratio": _float(after.get("dead_air_ratio")),
                    "max_interval": _int(after.get("max_interval")),
                    "pitch_span": _int(after.get("pitch_span")),
                },
                "repair_stats": repair_stats,
                "dead_air_gain": _float(before.get("dead_air_ratio"))
                - _float(after.get("dead_air_ratio")),
                "candidate_repair_passed": bool(candidate_passed),
            }
        )

    source_dead_air_values = [
        _float(row.get("source_metrics", {}).get("dead_air_ratio")) for row in candidate_repairs
    ]
    repaired_dead_air_values = [
        _float(row.get("repaired_metrics", {}).get("dead_air_ratio"))
        for row in candidate_repairs
    ]
    repaired_pass_count = sum(
        1 for row in candidate_repairs if bool(row.get("candidate_repair_passed", False))
    )
    source_dead_air_max = max(source_dead_air_values) if source_dead_air_values else 0.0
    repaired_dead_air_max = max(repaired_dead_air_values) if repaired_dead_air_values else 0.0
    dead_air_gain_max = source_dead_air_max - repaired_dead_air_max
    target_met = bool(
        repaired_pass_count >= int(min_repaired_candidates)
        and repaired_dead_air_max <= _float(decision_source["target_dead_air_max"])
        and dead_air_gain_max >= _float(decision_source["required_dead_air_gain_min"])
    )
    next_boundary = NEXT_BOUNDARY if target_met else FAIL_NEXT_BOUNDARY
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "issue_number": int(issue_number),
        "boundary": BOUNDARY,
        "source_boundaries": {
            "repair_decision": DECISION_BOUNDARY,
            "candidate_export": CANDIDATE_EXPORT_BOUNDARY,
        },
        "repair_config": {
            "strategy": "timing_gap_fill_and_duration_compaction",
            "dead_air_threshold_seconds": float(dead_air_threshold_seconds),
            "max_start_gap_seconds": float(max_start_gap_seconds),
            "fill_note_duration_seconds": float(fill_note_duration_seconds),
            "min_note_duration_seconds": float(min_note_duration_seconds),
            "preferred_fill_pitch_min": int(preferred_pitch_min),
            "preferred_fill_pitch_max": int(preferred_pitch_max),
            "min_repaired_candidates": int(min_repaired_candidates),
        },
        "guardrails": {
            "target_dead_air_max": _float(decision_source["target_dead_air_max"]),
            "required_dead_air_gain_min": _float(decision_source["required_dead_air_gain_min"]),
            "min_note_count": _int(decision_source["min_note_count"]),
            "min_unique_pitch_count": _int(decision_source["min_unique_pitch_count"]),
            "max_simultaneous_notes": _int(decision_source["max_simultaneous_notes"]),
            "max_postprocess_removal_ratio": _float(
                decision_source["max_postprocess_removal_ratio"]
            ),
        },
        "input_context": {
            "bars": _int(export_source["bars"]),
            "bpm": _float(export_source["bpm"]),
            "chord_progression": list(export_source["chord_progression"]),
        },
        "candidate_repairs": candidate_repairs,
        "summary": {
            "source_candidate_count": int(len(candidate_repairs)),
            "repaired_candidate_count": int(len(candidate_repairs)),
            "repaired_pass_count": int(repaired_pass_count),
            "source_dead_air_max": float(source_dead_air_max),
            "repaired_dead_air_max": float(repaired_dead_air_max),
            "dead_air_gain_max": float(dead_air_gain_max),
            "target_dead_air_max": _float(decision_source["target_dead_air_max"]),
            "required_dead_air_gain_min": _float(decision_source["required_dead_air_gain_min"]),
            "required_dead_air_gain_met": bool(
                dead_air_gain_max >= _float(decision_source["required_dead_air_gain_min"])
            ),
            "max_added_note_ratio": max(
                (
                    _float(row.get("repair_stats", {}).get("added_note_ratio"))
                    for row in candidate_repairs
                ),
                default=0.0,
            ),
            "max_postprocess_removal_ratio": max(
                (
                    _float(row.get("repair_stats", {}).get("postprocess_removal_ratio"))
                    for row in candidate_repairs
                ),
                default=0.0,
            ),
            "max_repaired_simultaneous_notes": max(
                (
                    _int(row.get("repaired_metrics", {}).get("max_simultaneous_notes"))
                    for row in candidate_repairs
                ),
                default=0,
            ),
            "max_repaired_interval": max(
                (
                    _int(row.get("repaired_metrics", {}).get("max_interval"))
                    for row in candidate_repairs
                ),
                default=0,
            ),
            "dead_air_timing_repair_passed": bool(target_met),
        },
        "readiness": {
            "boundary": BOUNDARY,
            "dead_air_timing_repair_probe_completed": True,
            "repaired_ranked_midi_written": bool(len(candidate_repairs) >= int(min_repaired_candidates)),
            "dead_air_timing_repair_passed": bool(target_met),
            "dead_air_timing_audio_render_required": bool(target_met),
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
                "repaired ranked MIDI candidates pass the dead-air/timing objective gate; render repaired MIDI to WAV next"
                if target_met
                else "dead-air/timing repair objective gate still requires follow-up"
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
            "Stage B MIDI-to-solo model-conditioned input path dead-air timing repair audio package"
            if target_met
            else "Stage B MIDI-to-solo model-conditioned input path dead-air timing repair follow-up"
        ),
    }


def validate_dead_air_timing_repair_probe_report(
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
        raise StageBMidiToSoloModelConditionedInputPathDeadAirTimingRepairProbeError(
            f"expected boundary {expected_boundary}, got {boundary}"
        )
    if expected_next_boundary and str(decision.get("next_boundary") or "") != expected_next_boundary:
        raise StageBMidiToSoloModelConditionedInputPathDeadAirTimingRepairProbeError(
            "unexpected next boundary"
        )
    if len(candidate_repairs) < int(min_repaired_candidates):
        raise StageBMidiToSoloModelConditionedInputPathDeadAirTimingRepairProbeError(
            "candidate repair count below threshold"
        )
    for row in candidate_repairs[: int(min_repaired_candidates)]:
        if not Path(str(row.get("repaired_midi_path") or "")).exists():
            raise StageBMidiToSoloModelConditionedInputPathDeadAirTimingRepairProbeError(
                "repaired MIDI path missing"
            )
        if not bool(row.get("candidate_repair_passed", False)) and require_repair_passed:
            raise StageBMidiToSoloModelConditionedInputPathDeadAirTimingRepairProbeError(
                "candidate repair should pass"
            )
    if require_repair_passed and not bool(
        readiness.get("dead_air_timing_repair_passed", False)
    ):
        raise StageBMidiToSoloModelConditionedInputPathDeadAirTimingRepairProbeError(
            "dead-air timing repair should pass"
        )
    if bool(decision.get("critical_user_input_required", True)):
        raise StageBMidiToSoloModelConditionedInputPathDeadAirTimingRepairProbeError(
            "critical user input should not be required"
        )
    if require_no_quality_claim:
        _require_no_quality_claim(readiness, label="repair probe readiness")
    return {
        "boundary": boundary,
        "next_boundary": str(decision.get("next_boundary") or ""),
        "dead_air_timing_repair_probe_completed": bool(
            readiness.get("dead_air_timing_repair_probe_completed", False)
        ),
        "repaired_ranked_midi_written": bool(
            readiness.get("repaired_ranked_midi_written", False)
        ),
        "dead_air_timing_repair_passed": bool(
            readiness.get("dead_air_timing_repair_passed", False)
        ),
        "dead_air_timing_audio_render_required": bool(
            readiness.get("dead_air_timing_audio_render_required", False)
        ),
        "source_candidate_count": _int(summary.get("source_candidate_count")),
        "repaired_candidate_count": _int(summary.get("repaired_candidate_count")),
        "repaired_pass_count": _int(summary.get("repaired_pass_count")),
        "source_dead_air_max": _float(summary.get("source_dead_air_max")),
        "repaired_dead_air_max": _float(summary.get("repaired_dead_air_max")),
        "dead_air_gain_max": _float(summary.get("dead_air_gain_max")),
        "target_dead_air_max": _float(summary.get("target_dead_air_max")),
        "required_dead_air_gain_met": bool(summary.get("required_dead_air_gain_met", False)),
        "max_added_note_ratio": _float(summary.get("max_added_note_ratio")),
        "max_postprocess_removal_ratio": _float(
            summary.get("max_postprocess_removal_ratio")
        ),
        "max_repaired_simultaneous_notes": _int(summary.get("max_repaired_simultaneous_notes")),
        "max_repaired_interval": _int(summary.get("max_repaired_interval")),
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
        "# Stage B MIDI-to-Solo Model-Conditioned Input Path Dead-Air Timing Repair Probe",
        "",
        "## Summary",
        "",
        f"- boundary: `{report['boundary']}`",
        f"- next boundary: `{decision['next_boundary']}`",
        f"- repair passed: `{_bool_token(summary['dead_air_timing_repair_passed'])}`",
        f"- source candidate count: `{summary['source_candidate_count']}`",
        f"- repaired candidate count: `{summary['repaired_candidate_count']}`",
        f"- repaired pass count: `{summary['repaired_pass_count']}`",
        f"- source dead-air max: `{summary['source_dead_air_max']:.4f}`",
        f"- repaired dead-air max: `{summary['repaired_dead_air_max']:.4f}`",
        f"- dead-air gain max: `{summary['dead_air_gain_max']:.4f}`",
        f"- max added-note ratio: `{summary['max_added_note_ratio']:.4f}`",
        f"- max postprocess removal ratio: `{summary['max_postprocess_removal_ratio']:.4f}`",
        f"- max repaired simultaneous notes: `{summary['max_repaired_simultaneous_notes']}`",
        f"- max repaired interval: `{summary['max_repaired_interval']}`",
        "",
        "## Repair Config",
        "",
        f"- strategy: `{config['strategy']}`",
        f"- dead-air threshold seconds: `{config['dead_air_threshold_seconds']:.4f}`",
        f"- max start gap seconds: `{config['max_start_gap_seconds']:.4f}`",
        f"- fill note duration seconds: `{config['fill_note_duration_seconds']:.4f}`",
        f"- preferred fill pitch range: `{config['preferred_fill_pitch_min']}`-`{config['preferred_fill_pitch_max']}`",
        "",
        "## Guardrails",
        "",
        f"- target dead-air max: `{guardrails['target_dead_air_max']:.4f}`",
        f"- required dead-air gain min: `{guardrails['required_dead_air_gain_min']:.4f}`",
        f"- min note count: `{guardrails['min_note_count']}`",
        f"- min unique pitch count: `{guardrails['min_unique_pitch_count']}`",
        f"- max simultaneous notes: `{guardrails['max_simultaneous_notes']}`",
        f"- max postprocess removal ratio: `{guardrails['max_postprocess_removal_ratio']:.4f}`",
        "",
        "## Repaired MIDI",
        "",
    ]
    for row in report["candidate_repairs"]:
        source = row["source_metrics"]
        repaired = row["repaired_metrics"]
        stats = row["repair_stats"]
        lines.append(
            f"- rank `{row['rank']}` sample `{row['sample_index']}`: "
            f"`{row['repaired_midi_path']}`, dead-air `{source['dead_air_ratio']:.4f}` -> "
            f"`{repaired['dead_air_ratio']:.4f}`, notes `{source['note_count']}` -> "
            f"`{repaired['note_count']}`, added ratio `{stats['added_note_ratio']:.4f}`, "
            f"pass `{_bool_token(row['candidate_repair_passed'])}`"
        )
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            f"- human/audio preference claimed: `{_bool_token(readiness['human_audio_preference_claimed'])}`",
            f"- MIDI-to-solo musical quality claimed: `{_bool_token(readiness['midi_to_solo_musical_quality_claimed'])}`",
            f"- audio-rendered quality claimed: `{_bool_token(readiness['audio_rendered_quality_claimed'])}`",
            f"- model checkpoint generation quality claimed: `{_bool_token(readiness['model_checkpoint_generation_quality_claimed'])}`",
            "",
            "## Decision",
            "",
            f"- auto progress allowed: `{_bool_token(decision['auto_progress_allowed'])}`",
            f"- critical user input required: `{_bool_token(decision['critical_user_input_required'])}`",
            f"- next recommended issue: `{report['next_recommended_issue']}`",
            "",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run model-conditioned input-path dead-air timing repair probe"
    )
    parser.add_argument("--repair_decision_report", type=str, required=True)
    parser.add_argument("--candidate_export_report", type=str, required=True)
    parser.add_argument(
        "--output_root",
        type=str,
        default=(
            "outputs/stage_b_midi_to_solo_model_conditioned_input_path_"
            "dead_air_timing_repair_probe"
        ),
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--issue_number", type=int, default=690)
    parser.add_argument("--min_repaired_candidates", type=int, default=3)
    parser.add_argument("--dead_air_threshold_seconds", type=float, default=0.5)
    parser.add_argument("--max_start_gap_seconds", type=float, default=0.49)
    parser.add_argument("--fill_note_duration_seconds", type=float, default=0.18)
    parser.add_argument("--min_note_duration_seconds", type=float, default=0.04)
    parser.add_argument("--preferred_pitch_min", type=int, default=48)
    parser.add_argument("--preferred_pitch_max", type=int, default=88)
    parser.add_argument("--expected_boundary", type=str, default="")
    parser.add_argument("--expected_next_boundary", type=str, default="")
    parser.add_argument("--require_repair_passed", action="store_true")
    parser.add_argument("--require_no_quality_claim", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    report = build_dead_air_timing_repair_probe_report(
        repair_decision_report=read_json(Path(args.repair_decision_report)),
        candidate_export_report=read_json(Path(args.candidate_export_report)),
        output_dir=output_dir,
        issue_number=int(args.issue_number),
        min_repaired_candidates=int(args.min_repaired_candidates),
        dead_air_threshold_seconds=float(args.dead_air_threshold_seconds),
        max_start_gap_seconds=float(args.max_start_gap_seconds),
        fill_note_duration_seconds=float(args.fill_note_duration_seconds),
        min_note_duration_seconds=float(args.min_note_duration_seconds),
        preferred_pitch_min=int(args.preferred_pitch_min),
        preferred_pitch_max=int(args.preferred_pitch_max),
    )
    summary = validate_dead_air_timing_repair_probe_report(
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
        / "stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_probe.json",
        report,
    )
    write_json(
        output_dir
        / "stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_probe_validation_summary.json",
        summary,
    )
    markdown = markdown_report(report)
    write_text(
        output_dir
        / "stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_probe.md",
        markdown,
    )
    if args.doc_path:
        write_text(Path(args.doc_path), markdown)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
