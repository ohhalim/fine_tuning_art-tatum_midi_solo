"""Run a minimum-change pitch-contour repair probe for model-conditioned MIDI-to-solo output."""

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
from scripts.decide_stage_b_midi_to_solo_model_conditioned_pitch_contour_changed_ratio_review import (  # noqa: E402
    BOUNDARY as DECISION_BOUNDARY,
    NEXT_BOUNDARY as DECISION_NEXT_BOUNDARY,
    SELECTED_TARGET as DECISION_TARGET,
)
from scripts.run_stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_pitch_contour_probe import (  # noqa: E402
    BOUNDARY as PITCH_CONTOUR_PROBE_BOUNDARY,
    load_midi_notes,
    max_simultaneous_notes,
    objective_metrics_for_path,
)


class StageBMidiToSoloPitchContourChangedRatioRepairProbeError(ValueError):
    pass


BOUNDARY = "stage_b_midi_to_solo_model_conditioned_pitch_contour_changed_ratio_repair_probe"
NEXT_BOUNDARY = "stage_b_midi_to_solo_model_conditioned_pitch_contour_changed_ratio_repair_audio_package"
FAIL_NEXT_BOUNDARY = "stage_b_midi_to_solo_model_conditioned_pitch_contour_changed_ratio_repair_decision"
SCHEMA_VERSION = "stage_b_midi_to_solo_model_conditioned_pitch_contour_changed_ratio_repair_probe_v1"

QUALITY_CLAIM_KEYS = [
    "human_audio_preference_claimed",
    "midi_to_solo_musical_quality_claimed",
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
        raise StageBMidiToSoloPitchContourChangedRatioRepairProbeError(
            f"unexpected quality claim in {label}: {claimed}"
        )


def pitch_class_options(
    pitch: int,
    *,
    preferred_pitch_min: int,
    preferred_pitch_max: int,
) -> list[int]:
    candidates: list[int] = []
    for octave_shift in range(-8, 9):
        candidate = int(pitch) + (12 * octave_shift)
        if int(preferred_pitch_min) <= candidate <= int(preferred_pitch_max):
            candidates.append(candidate)
    return sorted(set(candidates))


def minimum_change_contour_pitches(
    source_pitches: list[int],
    *,
    preferred_pitch_min: int,
    preferred_pitch_max: int,
    max_adjacent_interval: int,
) -> list[int]:
    if not source_pitches:
        return []
    states: dict[int, tuple[int, int, list[int]]] = {}
    for candidate in pitch_class_options(
        source_pitches[0],
        preferred_pitch_min=int(preferred_pitch_min),
        preferred_pitch_max=int(preferred_pitch_max),
    ):
        states[candidate] = (
            0 if candidate == int(source_pitches[0]) else 1,
            abs(candidate - int(source_pitches[0])),
            [candidate],
        )
    for source_pitch in source_pitches[1:]:
        next_states: dict[int, tuple[int, int, list[int]]] = {}
        for previous_pitch, (changed_count, total_shift, path) in states.items():
            for candidate in pitch_class_options(
                source_pitch,
                preferred_pitch_min=int(preferred_pitch_min),
                preferred_pitch_max=int(preferred_pitch_max),
            ):
                if abs(candidate - int(previous_pitch)) > int(max_adjacent_interval):
                    continue
                value = (
                    changed_count + (0 if candidate == int(source_pitch) else 1),
                    total_shift + abs(candidate - int(source_pitch)),
                    [*path, candidate],
                )
                current = next_states.get(candidate)
                if current is None or value[:2] < current[:2]:
                    next_states[candidate] = value
        states = next_states
        if not states:
            raise StageBMidiToSoloPitchContourChangedRatioRepairProbeError(
                "no minimum-change contour path satisfies interval guard"
            )
    return min(states.values(), key=lambda item: (item[0], item[1]))[2]


def write_minimum_change_repaired_midi(
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
        raise StageBMidiToSoloPitchContourChangedRatioRepairProbeError(
            f"source MIDI has no notes: {source_midi_path}"
        )
    repaired_pitches = minimum_change_contour_pitches(
        [int(note.pitch) for note in source_notes],
        preferred_pitch_min=int(preferred_pitch_min),
        preferred_pitch_max=int(preferred_pitch_max),
        max_adjacent_interval=int(max_adjacent_interval),
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
        name="pitch_change_ratio_repaired_solo",
    )
    changed_count = 0
    max_pitch_shift_abs = 0
    for note, pitch in zip(source_notes, repaired_pitches, strict=True):
        if int(pitch) != int(note.pitch):
            changed_count += 1
            max_pitch_shift_abs = max(max_pitch_shift_abs, abs(int(pitch) - int(note.pitch)))
        repaired_instrument.notes.append(
            pretty_midi.Note(
                velocity=max(1, min(127, int(note.velocity or 80))),
                pitch=int(pitch),
                start=float(note.start),
                end=float(note.end),
            )
        )
    repaired.instruments.append(repaired_instrument)
    repaired_midi_path.parent.mkdir(parents=True, exist_ok=True)
    repaired.write(str(repaired_midi_path))
    return {
        "source_note_count": int(len(source_notes)),
        "repaired_note_count": int(len(repaired_pitches)),
        "changed_note_count": int(changed_count),
        "pitch_changed_ratio": float(changed_count / max(1, len(source_notes))),
        "max_pitch_shift_abs": int(max_pitch_shift_abs),
    }


def validate_changed_ratio_decision(report: dict[str, Any]) -> dict[str, Any]:
    if str(report.get("boundary") or "") != DECISION_BOUNDARY:
        raise StageBMidiToSoloPitchContourChangedRatioRepairProbeError(
            "changed-ratio review decision boundary required"
        )
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    review = _dict(report.get("changed_ratio_review"))
    if str(decision.get("next_boundary") or "") != DECISION_NEXT_BOUNDARY:
        raise StageBMidiToSoloPitchContourChangedRatioRepairProbeError(
            "changed-ratio decision should route to repair probe"
        )
    if str(readiness.get("selected_target") or "") != DECISION_TARGET:
        raise StageBMidiToSoloPitchContourChangedRatioRepairProbeError(
            "changed-ratio selected target mismatch"
        )
    if not bool(readiness.get("changed_ratio_review_decision_completed", False)):
        raise StageBMidiToSoloPitchContourChangedRatioRepairProbeError(
            "changed-ratio decision completion required"
        )
    if not bool(readiness.get("repair_probe_required", False)):
        raise StageBMidiToSoloPitchContourChangedRatioRepairProbeError(
            "repair probe requirement required"
        )
    if bool(review.get("model_conditioned_input_path_alignment_required", True)):
        raise StageBMidiToSoloPitchContourChangedRatioRepairProbeError(
            "input-path alignment should not be required"
        )
    if _int(review.get("max_interval")) > _int(review.get("max_interval_threshold")):
        raise StageBMidiToSoloPitchContourChangedRatioRepairProbeError(
            "source interval target should already be supported"
        )
    if not bool(review.get("changed_ratio_review_required", False)):
        raise StageBMidiToSoloPitchContourChangedRatioRepairProbeError(
            "changed-ratio review requirement expected"
        )
    if bool(decision.get("critical_user_input_required", True)):
        raise StageBMidiToSoloPitchContourChangedRatioRepairProbeError(
            "critical user input should not be required"
        )
    _require_no_quality_claim(readiness, label="changed-ratio decision readiness")
    return {
        "boundary": DECISION_BOUNDARY,
        "max_interval_threshold": _int(review.get("max_interval_threshold")),
        "changed_ratio_review_threshold": _float(
            review.get("changed_ratio_review_threshold")
        ),
        "audio_review_required": bool(review.get("audio_review_required", False)),
    }


def validate_pitch_contour_probe_source(
    report: dict[str, Any],
    *,
    min_candidate_count: int,
) -> dict[str, Any]:
    if str(report.get("boundary") or "") != PITCH_CONTOUR_PROBE_BOUNDARY:
        raise StageBMidiToSoloPitchContourChangedRatioRepairProbeError(
            "pitch-contour probe boundary required"
        )
    readiness = _dict(report.get("readiness"))
    summary = _dict(report.get("summary"))
    repairs = [_dict(item) for item in _list(report.get("candidate_repairs")) if isinstance(item, dict)]
    if not bool(readiness.get("pitch_contour_repair_passed", False)):
        raise StageBMidiToSoloPitchContourChangedRatioRepairProbeError(
            "pitch-contour repair pass required"
        )
    if _float(summary.get("max_pitch_changed_ratio")) <= 0:
        raise StageBMidiToSoloPitchContourChangedRatioRepairProbeError(
            "source pitch changed ratio required"
        )
    if len(repairs) < int(min_candidate_count):
        raise StageBMidiToSoloPitchContourChangedRatioRepairProbeError(
            "pitch-contour repair candidate count below threshold"
        )
    for row in repairs[: int(min_candidate_count)]:
        if not Path(str(row.get("source_midi_path") or "")).exists():
            raise StageBMidiToSoloPitchContourChangedRatioRepairProbeError(
                f"source MIDI missing: {row.get('source_midi_path')}"
            )
    if bool(_dict(report.get("decision")).get("critical_user_input_required", True)):
        raise StageBMidiToSoloPitchContourChangedRatioRepairProbeError(
            "critical user input should not be required"
        )
    _require_no_quality_claim(readiness, label="pitch-contour probe readiness")
    return {
        "boundary": PITCH_CONTOUR_PROBE_BOUNDARY,
        "candidate_repairs": repairs[: int(min_candidate_count)],
        "source_max_pitch_changed_ratio": _float(summary.get("max_pitch_changed_ratio")),
        "source_repaired_max_interval": _int(summary.get("repaired_max_interval")),
        "source_repaired_dead_air_max": _float(summary.get("repaired_dead_air_max")),
    }


def build_changed_ratio_repair_probe_report(
    *,
    changed_ratio_decision: dict[str, Any],
    pitch_contour_probe: dict[str, Any],
    output_dir: Path,
    issue_number: int,
    min_repaired_candidates: int,
    dead_air_threshold_seconds: float,
    preferred_pitch_min: int,
    preferred_pitch_max: int,
    max_adjacent_interval: int,
    max_pitch_changed_ratio: float,
    min_unique_pitch_count: int,
) -> dict[str, Any]:
    decision_source = validate_changed_ratio_decision(changed_ratio_decision)
    probe_source = validate_pitch_contour_probe_source(
        pitch_contour_probe,
        min_candidate_count=int(min_repaired_candidates),
    )
    repaired_dir = output_dir / "midi"
    candidate_repairs: list[dict[str, Any]] = []
    for row in probe_source["candidate_repairs"]:
        rank = _int(row.get("rank"))
        sample_index = _int(row.get("sample_index"))
        source_path = Path(str(row.get("source_midi_path") or ""))
        repaired_path = repaired_dir / (
            f"rank_{rank:02d}_sample_{sample_index:02d}_changed_ratio_repair.mid"
        )
        before = objective_metrics_for_path(
            source_path,
            dead_air_threshold_seconds=float(dead_air_threshold_seconds),
        )
        pitch_stats = write_minimum_change_repaired_midi(
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
            _float(pitch_stats.get("pitch_changed_ratio")) <= float(max_pitch_changed_ratio)
            and _int(after.get("max_interval")) <= int(max_adjacent_interval)
            and _float(after.get("dead_air_ratio")) <= 0.35
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
                    "dead_air_ratio": _float(before.get("dead_air_ratio")),
                    "max_interval": _int(before.get("max_interval")),
                },
                "repaired_metrics": {
                    "note_count": _int(after.get("note_count")),
                    "unique_pitch_count": _int(after.get("unique_pitch_count")),
                    "dead_air_ratio": _float(after.get("dead_air_ratio")),
                    "max_interval": _int(after.get("max_interval")),
                    "max_simultaneous_notes": max_simultaneous_notes(repaired_notes),
                },
                "pitch_repair_stats": pitch_stats,
                "candidate_repair_passed": bool(candidate_passed),
            }
        )
    repaired_pass_count = sum(
        1 for row in candidate_repairs if bool(row.get("candidate_repair_passed", False))
    )
    max_ratio = max(
        (_float(row.get("pitch_repair_stats", {}).get("pitch_changed_ratio")) for row in candidate_repairs),
        default=0.0,
    )
    repaired_max_interval = max(
        (_int(row.get("repaired_metrics", {}).get("max_interval")) for row in candidate_repairs),
        default=0,
    )
    repaired_dead_air_max = max(
        (_float(row.get("repaired_metrics", {}).get("dead_air_ratio")) for row in candidate_repairs),
        default=0.0,
    )
    min_unique = min(
        (_int(row.get("repaired_metrics", {}).get("unique_pitch_count")) for row in candidate_repairs),
        default=0,
    )
    repair_passed = bool(
        repaired_pass_count >= int(min_repaired_candidates)
        and max_ratio <= float(max_pitch_changed_ratio)
        and repaired_max_interval <= int(max_adjacent_interval)
        and repaired_dead_air_max <= 0.35
    )
    next_boundary = NEXT_BOUNDARY if repair_passed else FAIL_NEXT_BOUNDARY
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "issue_number": int(issue_number),
        "boundary": BOUNDARY,
        "source_boundaries": {
            "changed_ratio_decision": decision_source["boundary"],
            "pitch_contour_probe": probe_source["boundary"],
        },
        "repair_config": {
            "strategy": "minimum_change_pitch_class_dynamic_programming",
            "dead_air_threshold_seconds": float(dead_air_threshold_seconds),
            "preferred_pitch_min": int(preferred_pitch_min),
            "preferred_pitch_max": int(preferred_pitch_max),
            "max_adjacent_interval": int(max_adjacent_interval),
            "max_pitch_changed_ratio": float(max_pitch_changed_ratio),
            "min_repaired_candidates": int(min_repaired_candidates),
            "min_unique_pitch_count": int(min_unique_pitch_count),
        },
        "source_summary": {
            "source_max_pitch_changed_ratio": _float(
                probe_source["source_max_pitch_changed_ratio"]
            ),
            "decision_changed_ratio_review_threshold": _float(
                decision_source["changed_ratio_review_threshold"]
            ),
            "source_repaired_max_interval": _int(probe_source["source_repaired_max_interval"]),
            "source_repaired_dead_air_max": _float(
                probe_source["source_repaired_dead_air_max"]
            ),
        },
        "candidate_repairs": candidate_repairs,
        "summary": {
            "source_candidate_count": int(len(candidate_repairs)),
            "repaired_candidate_count": int(len(candidate_repairs)),
            "repaired_pass_count": int(repaired_pass_count),
            "source_max_pitch_changed_ratio": _float(
                probe_source["source_max_pitch_changed_ratio"]
            ),
            "repaired_max_pitch_changed_ratio": float(max_ratio),
            "max_pitch_changed_ratio": float(max_pitch_changed_ratio),
            "pitch_changed_ratio_reduction": float(
                _float(probe_source["source_max_pitch_changed_ratio"]) - max_ratio
            ),
            "repaired_max_interval": int(repaired_max_interval),
            "target_max_interval": int(max_adjacent_interval),
            "repaired_dead_air_max": float(repaired_dead_air_max),
            "min_repaired_unique_pitch_count": int(min_unique),
            "changed_ratio_repair_passed": bool(repair_passed),
        },
        "readiness": {
            "boundary": BOUNDARY,
            "changed_ratio_repair_probe_completed": True,
            "repaired_ranked_midi_written": bool(len(candidate_repairs) >= int(min_repaired_candidates)),
            "changed_ratio_repair_passed": bool(repair_passed),
            "changed_ratio_repair_audio_render_required": bool(repair_passed),
            "human_audio_preference_claimed": False,
            "midi_to_solo_musical_quality_claimed": False,
            "audio_rendered_quality_claimed": False,
            "model_checkpoint_generation_quality_claimed": False,
            "model_direct_generation_quality_claimed": False,
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
                "changed-ratio objective repair passed; render repaired MIDI to WAV next"
                if repair_passed
                else "changed-ratio objective repair needs follow-up"
            ),
        },
        "not_proven": [
            "human_audio_preference",
            "midi_to_solo_musical_quality",
            "audio_rendered_quality",
            "broad_trained_model_quality",
            "brad_style_adaptation",
        ],
        "next_recommended_issue": (
            "Stage B MIDI-to-solo model-conditioned pitch-contour changed-ratio repair audio package"
            if repair_passed
            else "Stage B MIDI-to-solo model-conditioned pitch-contour changed-ratio repair decision"
        ),
    }


def validate_changed_ratio_repair_probe_report(
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
    repairs = [_dict(item) for item in _list(report.get("candidate_repairs"))]
    if expected_boundary and boundary != expected_boundary:
        raise StageBMidiToSoloPitchContourChangedRatioRepairProbeError(
            f"expected boundary {expected_boundary}, got {boundary}"
        )
    if expected_next_boundary and str(decision.get("next_boundary") or "") != expected_next_boundary:
        raise StageBMidiToSoloPitchContourChangedRatioRepairProbeError(
            "unexpected next boundary"
        )
    if len(repairs) < int(min_repaired_candidates):
        raise StageBMidiToSoloPitchContourChangedRatioRepairProbeError(
            "repaired candidate count below threshold"
        )
    if require_repair_passed and not bool(readiness.get("changed_ratio_repair_passed", False)):
        raise StageBMidiToSoloPitchContourChangedRatioRepairProbeError(
            "changed-ratio repair pass required"
        )
    if bool(decision.get("critical_user_input_required", True)):
        raise StageBMidiToSoloPitchContourChangedRatioRepairProbeError(
            "critical user input should not be required"
        )
    for row in repairs[: int(min_repaired_candidates)]:
        if not Path(str(row.get("repaired_midi_path") or "")).exists():
            raise StageBMidiToSoloPitchContourChangedRatioRepairProbeError(
                f"repaired MIDI missing: {row.get('repaired_midi_path')}"
            )
    if require_no_quality_claim:
        _require_no_quality_claim(readiness, label="changed-ratio repair readiness")
    return {
        "boundary": boundary,
        "next_boundary": str(decision.get("next_boundary") or ""),
        "changed_ratio_repair_probe_completed": bool(
            readiness.get("changed_ratio_repair_probe_completed", False)
        ),
        "changed_ratio_repair_passed": bool(
            readiness.get("changed_ratio_repair_passed", False)
        ),
        "changed_ratio_repair_audio_render_required": bool(
            readiness.get("changed_ratio_repair_audio_render_required", False)
        ),
        "source_candidate_count": _int(summary.get("source_candidate_count")),
        "repaired_candidate_count": _int(summary.get("repaired_candidate_count")),
        "repaired_pass_count": _int(summary.get("repaired_pass_count")),
        "source_max_pitch_changed_ratio": _float(
            summary.get("source_max_pitch_changed_ratio")
        ),
        "repaired_max_pitch_changed_ratio": _float(
            summary.get("repaired_max_pitch_changed_ratio")
        ),
        "max_pitch_changed_ratio": _float(summary.get("max_pitch_changed_ratio")),
        "pitch_changed_ratio_reduction": _float(
            summary.get("pitch_changed_ratio_reduction")
        ),
        "repaired_max_interval": _int(summary.get("repaired_max_interval")),
        "target_max_interval": _int(summary.get("target_max_interval")),
        "repaired_dead_air_max": _float(summary.get("repaired_dead_air_max")),
        "min_repaired_unique_pitch_count": _int(summary.get("min_repaired_unique_pitch_count")),
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
    lines = [
        "# Stage B MIDI-to-Solo Model-Conditioned Pitch-Contour Changed-Ratio Repair Probe",
        "",
        "## Summary",
        "",
        f"- boundary: `{report['boundary']}`",
        f"- next boundary: `{decision['next_boundary']}`",
        f"- changed-ratio repair passed: `{_bool_token(readiness['changed_ratio_repair_passed'])}`",
        f"- repaired candidates: `{summary['repaired_candidate_count']}`",
        f"- repaired pass count: `{summary['repaired_pass_count']}`",
        "",
        "## Evidence",
        "",
        f"- source max pitch changed ratio: `{summary['source_max_pitch_changed_ratio']:.4f}`",
        f"- repaired max pitch changed ratio: `{summary['repaired_max_pitch_changed_ratio']:.4f}`",
        f"- max pitch changed ratio: `{summary['max_pitch_changed_ratio']:.4f}`",
        f"- pitch changed ratio reduction: `{summary['pitch_changed_ratio_reduction']:.4f}`",
        f"- repaired max interval / target: `{summary['repaired_max_interval']}` / `{summary['target_max_interval']}`",
        f"- repaired dead-air max: `{summary['repaired_dead_air_max']:.4f}`",
        f"- min repaired unique pitch count: `{summary['min_repaired_unique_pitch_count']}`",
        "",
        "## Claim Boundary",
        "",
        f"- human/audio preference claimed: `{_bool_token(readiness['human_audio_preference_claimed'])}`",
        f"- MIDI-to-solo musical quality claimed: `{_bool_token(readiness['midi_to_solo_musical_quality_claimed'])}`",
        f"- broad trained model quality claimed: `{_bool_token(readiness['broad_trained_model_quality_claimed'])}`",
        f"- Brad style adaptation claimed: `{_bool_token(readiness['brad_style_adaptation_claimed'])}`",
        "",
        "## Next",
        "",
        f"- `{report['next_recommended_issue']}`",
    ]
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run pitch-contour changed-ratio repair probe")
    parser.add_argument("--changed_ratio_decision", type=str, required=True)
    parser.add_argument("--pitch_contour_probe", type=str, required=True)
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/stage_b_midi_to_solo_model_conditioned_pitch_contour_changed_ratio_repair_probe",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--issue_number", type=int, default=718)
    parser.add_argument("--min_repaired_candidates", type=int, default=3)
    parser.add_argument("--dead_air_threshold_seconds", type=float, default=0.5)
    parser.add_argument("--preferred_pitch_min", type=int, default=48)
    parser.add_argument("--preferred_pitch_max", type=int, default=88)
    parser.add_argument("--max_adjacent_interval", type=int, default=12)
    parser.add_argument("--max_pitch_changed_ratio", type=float, default=0.5)
    parser.add_argument("--min_unique_pitch_count", type=int, default=20)
    parser.add_argument("--expected_boundary", type=str, default="")
    parser.add_argument("--expected_next_boundary", type=str, default="")
    parser.add_argument("--require_repair_passed", action="store_true")
    parser.add_argument("--require_no_quality_claim", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    report = build_changed_ratio_repair_probe_report(
        changed_ratio_decision=read_json(Path(args.changed_ratio_decision)),
        pitch_contour_probe=read_json(Path(args.pitch_contour_probe)),
        output_dir=output_dir,
        issue_number=int(args.issue_number),
        min_repaired_candidates=int(args.min_repaired_candidates),
        dead_air_threshold_seconds=float(args.dead_air_threshold_seconds),
        preferred_pitch_min=int(args.preferred_pitch_min),
        preferred_pitch_max=int(args.preferred_pitch_max),
        max_adjacent_interval=int(args.max_adjacent_interval),
        max_pitch_changed_ratio=float(args.max_pitch_changed_ratio),
        min_unique_pitch_count=int(args.min_unique_pitch_count),
    )
    summary = validate_changed_ratio_repair_probe_report(
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
        / "stage_b_midi_to_solo_model_conditioned_pitch_contour_changed_ratio_repair_probe.json",
        report,
    )
    write_json(
        output_dir
        / (
            "stage_b_midi_to_solo_model_conditioned_pitch_contour_"
            "changed_ratio_repair_probe_validation_summary.json"
        ),
        summary,
    )
    markdown = markdown_report(report)
    write_text(
        output_dir
        / "stage_b_midi_to_solo_model_conditioned_pitch_contour_changed_ratio_repair_probe.md",
        markdown,
    )
    if args.doc_path:
        write_text(Path(args.doc_path), markdown)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
