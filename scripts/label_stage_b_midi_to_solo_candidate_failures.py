"""Label current MIDI-to-solo candidates with the quality rubric baseline."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

import pretty_midi

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from scripts.assess_stage_b_generic_base_readiness import read_json, write_json, write_text  # noqa: E402
from scripts.audit_stage_b_midi_to_solo_final_status import (  # noqa: E402
    BRIDGE_SOURCE_CONTEXT_KEYS,
)
from scripts.build_stage_b_midi_to_solo_mvp_delivery_package import (  # noqa: E402
    BOUNDARY as DELIVERY_BOUNDARY,
)
from scripts.build_stage_b_midi_to_solo_quality_rubric_baseline import (  # noqa: E402
    BOUNDARY as RUBRIC_BOUNDARY,
    NEXT_BOUNDARY as RUBRIC_NEXT_BOUNDARY,
    SELECTED_TARGET as RUBRIC_SELECTED_TARGET,
    StageBMidiToSoloQualityRubricBaselineError,
    validate_quality_rubric_baseline_report,
)


class StageBMidiToSoloCandidateFailureLabelingError(ValueError):
    pass


BOUNDARY = "stage_b_midi_to_solo_candidate_failure_labeling"
REPAIR_NEXT_BOUNDARY = "stage_b_midi_to_solo_targeted_quality_repair_sweep"
AUDIO_REVIEW_NEXT_BOUNDARY = "stage_b_midi_to_solo_audio_review_package"
REPAIR_TARGET = "targeted_quality_repair_sweep"
AUDIO_REVIEW_TARGET = "audio_review_package"
SCHEMA_VERSION = "stage_b_midi_to_solo_candidate_failure_labeling_v3"

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
        raise StageBMidiToSoloCandidateFailureLabelingError(
            f"unexpected quality claim in {label}: {claimed}"
        )


def _source_context_fields(container: dict[str, Any], *, label: str) -> dict[str, Any]:
    for key in BRIDGE_SOURCE_CONTEXT_KEYS:
        if key not in container or container[key] is None:
            raise StageBMidiToSoloCandidateFailureLabelingError(
                f"{label} source-context field required: {key}"
            )
    return {key: container[key] for key in BRIDGE_SOURCE_CONTEXT_KEYS}


def validate_rubric_baseline(report: dict[str, Any]) -> dict[str, Any]:
    if str(report.get("boundary") or "") != RUBRIC_BOUNDARY:
        raise StageBMidiToSoloCandidateFailureLabelingError("rubric baseline boundary required")
    try:
        summary = validate_quality_rubric_baseline_report(
            report,
            expected_boundary=RUBRIC_BOUNDARY,
            expected_next_boundary=RUBRIC_NEXT_BOUNDARY,
            expected_target=RUBRIC_SELECTED_TARGET,
            min_rubric_item_count=8,
            require_candidate_labeling_ready=True,
            require_no_quality_claim=True,
        )
    except StageBMidiToSoloQualityRubricBaselineError as exc:
        raise StageBMidiToSoloCandidateFailureLabelingError(str(exc)) from exc
    rubric_items = [_dict(item) for item in _list(report.get("rubric_items"))]
    if len(rubric_items) < 8:
        raise StageBMidiToSoloCandidateFailureLabelingError("rubric items below 8")
    if not bool(summary.get("outside_soloing_repair_evidence_ready", False)):
        raise StageBMidiToSoloCandidateFailureLabelingError(
            "outside-soloing repair evidence readiness required"
        )
    if not bool(summary.get("outside_soloing_repair_source_context_preserved", False)):
        raise StageBMidiToSoloCandidateFailureLabelingError(
            "outside-soloing repair source context preservation required"
        )
    _source_context_fields(summary, label="quality rubric baseline")
    if _int(summary.get("outside_soloing_repair_wav_count")) < 6:
        raise StageBMidiToSoloCandidateFailureLabelingError(
            "outside-soloing repair WAV count below 6"
        )
    if _int(summary.get("outside_soloing_repair_pitch_role_risk_count_after")) != 0:
        raise StageBMidiToSoloCandidateFailureLabelingError(
            "outside-soloing residual pitch-role risk should be zero"
        )
    source_objective_risk = _int(
        summary.get("outside_soloing_repair_source_objective_pitch_role_risk_count")
    )
    source_risk_before = _int(
        summary.get("outside_soloing_repair_source_pitch_role_risk_count_before")
    )
    source_risk_after = _int(
        summary.get("outside_soloing_repair_source_pitch_role_risk_count_after")
    )
    source_risk_delta = _int(summary.get("outside_soloing_repair_source_pitch_role_risk_delta"))
    if source_objective_risk <= 0:
        raise StageBMidiToSoloCandidateFailureLabelingError(
            "outside-soloing source objective pitch-role risk count required"
        )
    if source_risk_after > source_risk_before:
        raise StageBMidiToSoloCandidateFailureLabelingError(
            "outside-soloing source pitch-role risk should not increase"
        )
    if source_risk_delta != source_risk_before - source_risk_after:
        raise StageBMidiToSoloCandidateFailureLabelingError(
            "outside-soloing source pitch-role risk delta mismatch"
        )
    if bool(summary.get("outside_soloing_repair_source_targeted", True)):
        raise StageBMidiToSoloCandidateFailureLabelingError(
            "outside-soloing source repair should remain non-targeted"
        )
    if not bool(summary.get("outside_soloing_repair_source_residual_risk_preserved", False)):
        raise StageBMidiToSoloCandidateFailureLabelingError(
            "outside-soloing source residual risk preservation required"
        )
    return {**summary, "rubric_items": rubric_items}


def _require_existing_file(path: str, *, label: str) -> str:
    if not path:
        raise StageBMidiToSoloCandidateFailureLabelingError(f"{label} path required")
    if not Path(path).exists():
        raise StageBMidiToSoloCandidateFailureLabelingError(f"{label} missing: {path}")
    return path


def validate_delivery_package(report: dict[str, Any]) -> list[dict[str, Any]]:
    readiness = _dict(report.get("readiness"))
    package = _dict(report.get("delivery_package"))
    manifest = _dict(report.get("artifact_manifest"))
    if str(report.get("boundary") or "") != DELIVERY_BOUNDARY:
        raise StageBMidiToSoloCandidateFailureLabelingError("MVP delivery package boundary required")
    if not bool(readiness.get("mvp_delivery_package_completed", False)):
        raise StageBMidiToSoloCandidateFailureLabelingError("MVP delivery package completion required")
    if not bool(package.get("input_to_ranked_midi_ready", False)):
        raise StageBMidiToSoloCandidateFailureLabelingError("input to ranked MIDI readiness required")
    if not bool(package.get("input_to_rendered_wav_evidence_ready", False)):
        raise StageBMidiToSoloCandidateFailureLabelingError("input to rendered WAV evidence required")
    _require_no_quality_claim(readiness, label="delivery readiness")

    candidates: list[dict[str, Any]] = []
    for item in [_dict(row) for row in _list(manifest.get("cli_repaired_midi_candidates"))]:
        candidates.append(
            {
                "source": "cli_repaired_midi",
                "rank": _int(item.get("rank")),
                "midi_path": _require_existing_file(
                    str(item.get("repaired_midi_path") or ""),
                    label="CLI repaired MIDI",
                ),
                "source_note_count": _int(item.get("note_count")),
                "source_unique_pitch_count": _int(item.get("unique_pitch_count")),
                "source_dead_air_ratio": _float(item.get("dead_air_ratio")),
                "source_objective_supported": bool(item.get("objective_supported", False)),
            }
        )
    for item in [_dict(row) for row in _list(manifest.get("changed_ratio_repair_audio_candidates"))]:
        candidates.append(
            {
                "source": "changed_ratio_repair",
                "rank": _int(item.get("rank")),
                "midi_path": _require_existing_file(
                    str(item.get("repaired_midi_path") or ""),
                    label="changed-ratio repaired MIDI",
                ),
                "source_note_count": 0,
                "source_unique_pitch_count": _int(item.get("repaired_unique_pitch_count")),
                "source_dead_air_ratio": 0.0,
                "source_objective_supported": True,
                "wav_path": str(item.get("wav_path") or ""),
                "pitch_changed_ratio": _float(item.get("pitch_changed_ratio")),
                "repaired_max_interval": _int(item.get("repaired_max_interval")),
            }
        )
    if len(candidates) < 3:
        raise StageBMidiToSoloCandidateFailureLabelingError("at least 3 MIDI candidates required")
    return candidates


def most_common_ratio(values: list[float | int], *, precision: int = 3) -> float:
    if not values:
        return 0.0
    rounded = [round(float(value), precision) for value in values]
    return max(Counter(rounded).values()) / len(rounded)


def repeated_cycle(values: list[tuple[float, ...]], *, cycle_length: int) -> bool:
    if len(values) < cycle_length * 2:
        return False
    return values[:cycle_length] == values[cycle_length : cycle_length * 2]


def load_midi_notes(path: str | Path) -> tuple[pretty_midi.PrettyMIDI, list[pretty_midi.Note]]:
    midi = pretty_midi.PrettyMIDI(str(path))
    notes: list[pretty_midi.Note] = []
    for instrument in midi.instruments:
        if not instrument.is_drum:
            notes.extend(instrument.notes)
    return midi, sorted(notes, key=lambda note: (float(note.start), int(note.pitch), float(note.end)))


def beat_length_seconds(midi: pretty_midi.PrettyMIDI) -> float:
    beats = [float(value) for value in midi.get_beats()]
    diffs = [beats[index] - beats[index - 1] for index in range(1, len(beats))]
    positive = [value for value in diffs if value > 0.0]
    if positive:
        return float(median(positive))
    tempo_times, tempi = midi.get_tempo_changes()
    if len(tempi) > 0 and float(tempi[0]) > 0.0:
        return 60.0 / float(tempi[0])
    return 0.5


def bar_grid(midi: pretty_midi.PrettyMIDI, notes: list[pretty_midi.Note]) -> tuple[list[float], float]:
    end_time = max(float(midi.get_end_time()), max((float(note.end) for note in notes), default=0.0))
    downbeats = [float(value) for value in midi.get_downbeats()]
    if len(downbeats) >= 2:
        while downbeats[-1] < end_time:
            downbeats.append(downbeats[-1] + max(0.0001, downbeats[-1] - downbeats[-2]))
        return downbeats, max(0.0001, downbeats[1] - downbeats[0])
    bar_length = max(0.0001, beat_length_seconds(midi) * 4.0)
    grid = [0.0]
    while grid[-1] < end_time:
        grid.append(grid[-1] + bar_length)
    if len(grid) < 2:
        grid.append(bar_length)
    return grid, bar_length


def bar_metrics(
    midi: pretty_midi.PrettyMIDI,
    notes: list[pretty_midi.Note],
) -> dict[str, Any]:
    grid, _bar_length = bar_grid(midi, notes)
    note_counts: list[int] = []
    onset_patterns: list[tuple[float, ...]] = []
    for index in range(len(grid) - 1):
        start = grid[index]
        end = grid[index + 1]
        length = max(0.0001, end - start)
        bar_notes = [note for note in notes if start <= float(note.start) < end]
        note_counts.append(len(bar_notes))
        onset_patterns.append(
            tuple(round((float(note.start) - start) / length * 4.0, 3) for note in bar_notes)
        )
    active_bar_count = sum(1 for count in note_counts if count > 0)
    return {
        "bar_count": len(note_counts),
        "active_bar_count": active_bar_count,
        "empty_bar_count": sum(1 for count in note_counts if count == 0),
        "note_counts_per_bar": note_counts,
        "note_count_per_bar_most_common_ratio": most_common_ratio(note_counts, precision=0),
        "most_common_note_count_per_bar": Counter(note_counts).most_common(1)[0][0]
        if note_counts
        else 0,
        "unique_onset_pattern_count": len(set(onset_patterns)),
        "four_bar_rhythm_cycle_repeated": repeated_cycle(onset_patterns, cycle_length=4),
        "four_notes_per_bar_template": bool(note_counts)
        and Counter(note_counts).most_common(1)[0][0] == 4,
    }


def rhythm_signature(notes: list[pretty_midi.Note]) -> tuple[tuple[float, ...], tuple[float, ...]]:
    starts = [round(float(note.start), 3) for note in notes]
    durations = [round(max(0.0, float(note.end) - float(note.start)), 3) for note in notes]
    iois = [round(starts[index] - starts[index - 1], 3) for index in range(1, len(starts))]
    return tuple(iois), tuple(durations)


def max_simultaneous_notes(notes: list[pretty_midi.Note]) -> int:
    events: list[tuple[float, int]] = []
    for note in notes:
        events.append((float(note.start), 1))
        events.append((float(note.end), -1))
    active = 0
    max_active = 0
    for _time, delta in sorted(events, key=lambda item: (item[0], item[1])):
        active += delta
        max_active = max(max_active, active)
    return max_active


def contour_turn_count(pitches: list[int]) -> int:
    signs: list[int] = []
    for index in range(1, len(pitches)):
        delta = pitches[index] - pitches[index - 1]
        if delta > 0:
            signs.append(1)
        elif delta < 0:
            signs.append(-1)
    return sum(1 for index in range(1, len(signs)) if signs[index] != signs[index - 1])


def analyze_candidate_midi(candidate: dict[str, Any]) -> dict[str, Any]:
    midi, notes = load_midi_notes(candidate["midi_path"])
    if not notes:
        return {
            "source": candidate["source"],
            "rank": candidate["rank"],
            "midi_path": candidate["midi_path"],
            "note_count": 0,
            "rhythm_signature": ((), ()),
            "metrics": {
                "note_count": 0,
                "active_bar_count": 0,
                "empty_bar_count": 0,
                "max_simultaneous_notes": 0,
                "grammar_valid": False,
                "strict_valid": False,
            },
        }
    pitches = [int(note.pitch) for note in notes]
    starts = [float(note.start) for note in notes]
    durations = [max(0.0, float(note.end) - float(note.start)) for note in notes]
    iois = [max(0.0, starts[index] - starts[index - 1]) for index in range(1, len(starts))]
    intervals = [abs(pitches[index] - pitches[index - 1]) for index in range(1, len(pitches))]
    beat_length = beat_length_seconds(midi)
    long_gap_threshold = beat_length * 2.0
    dead_air_events = sum(1 for gap in iois if gap >= long_gap_threshold)
    max_gap_beats = max((gap / max(0.0001, beat_length) for gap in iois), default=0.0)
    bar = bar_metrics(midi, notes)
    small_interval_ratio = sum(1 for value in intervals if value <= 4) / max(1, len(intervals))
    large_interval_ratio = sum(1 for value in intervals if value >= 12) / max(1, len(intervals))
    max_simultaneous = max_simultaneous_notes(notes)
    peak_index = max(range(len(pitches)), key=lambda index: pitches[index]) if pitches else 0
    peak_position_ratio = peak_index / max(1, len(pitches) - 1)
    metrics = {
        **bar,
        "note_count": len(notes),
        "unique_pitch_count": len(set(pitches)),
        "unique_pitch_class_count": len({pitch % 12 for pitch in pitches}),
        "pitch_span": max(pitches) - min(pitches),
        "max_abs_interval": max(intervals) if intervals else 0,
        "large_interval_ratio_gte12": large_interval_ratio,
        "small_interval_ratio_le4": small_interval_ratio,
        "duration_most_common_ratio": most_common_ratio(durations),
        "ioi_most_common_ratio": most_common_ratio(iois),
        "dead_air_ratio": dead_air_events / max(1, len(iois)),
        "source_dead_air_ratio": _float(candidate.get("source_dead_air_ratio")),
        "max_gap_beats": max_gap_beats,
        "contour_turn_count": contour_turn_count(pitches),
        "phrase_peak_position_ratio": peak_position_ratio,
        "max_simultaneous_notes": max_simultaneous,
        "grammar_valid": bool(candidate.get("source_objective_supported", False)),
        "strict_valid": max_simultaneous <= 1,
        "chord_context_available": False,
        "cadence_resolution_present": None,
        "strong_beat_chord_tone_ratio": None,
        "cadence_landing_chord_tone": None,
    }
    return {
        "source": candidate["source"],
        "rank": candidate["rank"],
        "midi_path": candidate["midi_path"],
        "wav_path": str(candidate.get("wav_path") or ""),
        "rhythm_signature": rhythm_signature(notes),
        "metrics": metrics,
    }


def threshold_for(rubric_items: list[dict[str, Any]], rubric_id: str) -> dict[str, Any]:
    for item in rubric_items:
        if str(item.get("id") or "") == rubric_id:
            return _dict(item.get("threshold"))
    return {}


def label_candidate(
    candidate: dict[str, Any],
    *,
    rubric_items: list[dict[str, Any]],
    shared_rhythm_signature_count: int,
) -> dict[str, Any]:
    metrics = _dict(candidate.get("metrics"))
    failure_labels: list[str] = []
    not_evaluable_labels: list[str] = []
    evidence: dict[str, Any] = {}

    sparse = threshold_for(rubric_items, "sparse_or_empty_output")
    if _int(metrics.get("note_count")) < _int(sparse.get("min_note_count")) or _int(
        metrics.get("active_bar_count")
    ) < _int(sparse.get("min_active_bar_count")):
        failure_labels.append("sparse_or_empty_output")

    dead_air = threshold_for(rubric_items, "dead_air_or_density_gap")
    if _float(metrics.get("dead_air_ratio")) > _float(dead_air.get("max_dead_air_ratio")) or _int(
        metrics.get("empty_bar_count")
    ) > _int(dead_air.get("max_empty_bar_count")):
        failure_labels.append("dead_air_or_density_gap")

    rhythm = threshold_for(rubric_items, "rhythmic_monotony")
    if (
        _float(metrics.get("duration_most_common_ratio"))
        >= _float(rhythm.get("max_duration_most_common_ratio"))
        or _float(metrics.get("ioi_most_common_ratio"))
        >= _float(rhythm.get("max_ioi_most_common_ratio"))
        or _float(metrics.get("note_count_per_bar_most_common_ratio"))
        >= _float(rhythm.get("max_note_count_per_bar_most_common_ratio"))
    ):
        failure_labels.append("rhythmic_monotony")

    songlike = threshold_for(rubric_items, "songlike_melody_not_soloing")
    if (
        bool(metrics.get("four_notes_per_bar_template", False))
        or bool(metrics.get("four_bar_rhythm_cycle_repeated", False))
        or int(shared_rhythm_signature_count) > _int(songlike.get("max_shared_rhythm_signature_count"))
        or _float(metrics.get("small_interval_ratio_le4"))
        > _float(songlike.get("max_small_interval_ratio_le4"))
    ):
        failure_labels.append("songlike_melody_not_soloing")

    if not bool(metrics.get("chord_context_available", False)):
        not_evaluable_labels.extend(
            ["outside_soloing_without_context", "weak_chord_tone_landing"]
        )

    phrase = threshold_for(rubric_items, "phrase_shape_missing_tension_release")
    peak_range = _list(phrase.get("phrase_peak_position_ratio_range"))
    peak_min = _float(peak_range[0]) if len(peak_range) >= 2 else 0.25
    peak_max = _float(peak_range[1]) if len(peak_range) >= 2 else 0.85
    if (
        _int(metrics.get("contour_turn_count")) < _int(phrase.get("min_contour_turn_count"))
        or _float(metrics.get("large_interval_ratio_gte12"))
        > _float(phrase.get("max_large_interval_ratio_gte12"))
        or not (peak_min <= _float(metrics.get("phrase_peak_position_ratio")) <= peak_max)
    ):
        failure_labels.append("phrase_shape_missing_tension_release")

    technical = threshold_for(rubric_items, "technical_gate_regression")
    if (
        not bool(metrics.get("grammar_valid", False))
        or not bool(metrics.get("strict_valid", False))
        or _int(metrics.get("max_simultaneous_notes"))
        > _int(technical.get("max_simultaneous_notes"))
    ):
        failure_labels.append("technical_gate_regression")

    evidence["shared_rhythm_signature_count"] = int(shared_rhythm_signature_count)
    evidence["failure_label_count"] = len(failure_labels)
    evidence["not_evaluable_label_count"] = len(not_evaluable_labels)
    return {
        **candidate,
        "failure_labels": sorted(set(failure_labels)),
        "not_evaluable_labels": sorted(set(not_evaluable_labels)),
        "label_evidence": evidence,
    }


def build_candidate_failure_labeling_report(
    *,
    rubric_baseline: dict[str, Any],
    mvp_delivery_package: dict[str, Any],
    output_dir: Path,
    issue_number: int,
) -> dict[str, Any]:
    rubric_summary = validate_rubric_baseline(rubric_baseline)
    rubric_items = [_dict(item) for item in _list(rubric_summary.get("rubric_items"))]
    candidates = validate_delivery_package(mvp_delivery_package)
    analyzed = [analyze_candidate_midi(candidate) for candidate in candidates]
    rhythm_counts = Counter(tuple(item.get("rhythm_signature", ((), ()))) for item in analyzed)
    labeled = [
        label_candidate(
            item,
            rubric_items=rubric_items,
            shared_rhythm_signature_count=rhythm_counts[
                tuple(item.get("rhythm_signature", ((), ())))
            ],
        )
        for item in analyzed
    ]
    failure_counts = Counter(label for item in labeled for label in _list(item.get("failure_labels")))
    not_evaluable_counts = Counter(
        label for item in labeled for label in _list(item.get("not_evaluable_labels"))
    )
    outside_context = {
        "outside_soloing_repair_evidence_ready": bool(
            rubric_summary["outside_soloing_repair_evidence_ready"]
        ),
        "outside_soloing_repair_source_context_preserved": bool(
            rubric_summary["outside_soloing_repair_source_context_preserved"]
        ),
        "outside_soloing_repair_wav_count": _int(
            rubric_summary["outside_soloing_repair_wav_count"]
        ),
        "outside_soloing_repair_source_objective_pitch_role_risk_count": _int(
            rubric_summary["outside_soloing_repair_source_objective_pitch_role_risk_count"]
        ),
        "outside_soloing_repair_source_pitch_role_risk_count_before": _int(
            rubric_summary["outside_soloing_repair_source_pitch_role_risk_count_before"]
        ),
        "outside_soloing_repair_source_pitch_role_risk_count_after": _int(
            rubric_summary["outside_soloing_repair_source_pitch_role_risk_count_after"]
        ),
        "outside_soloing_repair_source_pitch_role_risk_delta": _int(
            rubric_summary["outside_soloing_repair_source_pitch_role_risk_delta"]
        ),
        "outside_soloing_repair_source_targeted": bool(
            rubric_summary["outside_soloing_repair_source_targeted"]
        ),
        "outside_soloing_repair_source_residual_risk_preserved": bool(
            rubric_summary["outside_soloing_repair_source_residual_risk_preserved"]
        ),
        "outside_soloing_repair_pitch_role_risk_count_after": _int(
            rubric_summary["outside_soloing_repair_pitch_role_risk_count_after"]
        ),
        "outside_soloing_repair_pitch_role_risk_delta": _int(
            rubric_summary.get("outside_soloing_repair_pitch_role_risk_delta")
        ),
        "outside_soloing_not_evaluable_count": _int(
            not_evaluable_counts.get("outside_soloing_without_context", 0)
        ),
        **{key: rubric_summary[key] for key in BRIDGE_SOURCE_CONTEXT_KEYS},
        "outside_soloing_label_scope": "not_evaluable chord-context gap after objective pitch-role repair",
    }
    failed_candidate_count = sum(1 for item in labeled if _list(item.get("failure_labels")))
    selected_target = REPAIR_TARGET if failed_candidate_count > 0 else AUDIO_REVIEW_TARGET
    next_boundary = REPAIR_NEXT_BOUNDARY if failed_candidate_count > 0 else AUDIO_REVIEW_NEXT_BOUNDARY
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "issue_number": int(issue_number),
        "boundary": BOUNDARY,
        "source_boundary": RUBRIC_BOUNDARY,
        "delivery_source_boundary": DELIVERY_BOUNDARY,
        "candidate_labels": labeled,
        "aggregate": {
            "candidate_count": len(labeled),
            "failed_candidate_count": failed_candidate_count,
            "failure_counts": dict(sorted(failure_counts.items())),
            "not_evaluable_counts": dict(sorted(not_evaluable_counts.items())),
            "max_failure_label_count": max(
                (len(_list(item.get("failure_labels"))) for item in labeled),
                default=0,
            ),
            "candidate_sources": dict(
                sorted(Counter(str(item.get("source") or "") for item in labeled).items())
            ),
            **outside_context,
        },
        "selected_next_target": {
            "selected_target": selected_target,
            "selected_next_boundary": next_boundary,
            "reason": "failure labels exist" if failed_candidate_count > 0 else "no objective failure labels found",
        },
        "readiness": {
            "boundary": BOUNDARY,
            "candidate_failure_labeling_completed": True,
            "candidate_count": len(labeled),
            "failed_candidate_count": failed_candidate_count,
            "targeted_quality_repair_sweep_ready": failed_candidate_count > 0,
            "audio_review_package_ready": failed_candidate_count == 0,
            "outside_soloing_repair_source_context_preserved": bool(
                outside_context["outside_soloing_repair_source_context_preserved"]
            ),
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
            "reason": "candidate MIDI evidence labeled against quality rubric without quality claim",
        },
        "not_proven": [
            "targeted_quality_repair_completed",
            "human_audio_preference",
            "midi_to_solo_musical_quality",
            "broad_trained_model_quality",
        ],
        "next_recommended_issue": (
            "Stage B MIDI-to-solo targeted quality repair sweep source-context refresh"
            if failed_candidate_count > 0
            else "Stage B MIDI-to-solo audio review package source-context refresh"
        ),
    }


def validate_candidate_failure_labeling_report(
    report: dict[str, Any],
    *,
    expected_boundary: str | None,
    min_candidate_count: int,
    require_labeling_completed: bool,
    require_no_quality_claim: bool,
) -> dict[str, Any]:
    boundary = str(report.get("boundary") or "")
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    selected = _dict(report.get("selected_next_target"))
    aggregate = _dict(report.get("aggregate"))
    labels = _list(report.get("candidate_labels"))
    if expected_boundary and boundary != expected_boundary:
        raise StageBMidiToSoloCandidateFailureLabelingError(
            f"expected boundary {expected_boundary}, got {boundary}"
        )
    if require_labeling_completed and not bool(
        readiness.get("candidate_failure_labeling_completed", False)
    ):
        raise StageBMidiToSoloCandidateFailureLabelingError("labeling completion required")
    if len(labels) < int(min_candidate_count):
        raise StageBMidiToSoloCandidateFailureLabelingError("candidate label count below minimum")
    if _int(aggregate.get("candidate_count")) != len(labels):
        raise StageBMidiToSoloCandidateFailureLabelingError("candidate count mismatch")
    if str(selected.get("selected_target") or "") not in {REPAIR_TARGET, AUDIO_REVIEW_TARGET}:
        raise StageBMidiToSoloCandidateFailureLabelingError("unexpected selected target")
    if str(decision.get("next_boundary") or "") not in {
        REPAIR_NEXT_BOUNDARY,
        AUDIO_REVIEW_NEXT_BOUNDARY,
    }:
        raise StageBMidiToSoloCandidateFailureLabelingError("unexpected next boundary")
    if bool(decision.get("critical_user_input_required", True)):
        raise StageBMidiToSoloCandidateFailureLabelingError(
            "critical user input should not be required"
        )
    if require_no_quality_claim:
        _require_no_quality_claim(readiness, label="candidate labeling readiness")
    return {
        "boundary": boundary,
        "next_boundary": str(decision.get("next_boundary") or ""),
        "selected_target": str(selected.get("selected_target") or ""),
        "candidate_failure_labeling_completed": bool(
            readiness.get("candidate_failure_labeling_completed", False)
        ),
        "candidate_count": len(labels),
        "failed_candidate_count": _int(aggregate.get("failed_candidate_count")),
        "failure_label_type_count": len(_dict(aggregate.get("failure_counts"))),
        "not_evaluable_label_type_count": len(_dict(aggregate.get("not_evaluable_counts"))),
        "outside_soloing_repair_evidence_ready": bool(
            aggregate.get("outside_soloing_repair_evidence_ready", False)
        ),
        "outside_soloing_repair_source_context_preserved": bool(
            aggregate.get("outside_soloing_repair_source_context_preserved", False)
        ),
        "outside_soloing_repair_wav_count": _int(
            aggregate.get("outside_soloing_repair_wav_count")
        ),
        "outside_soloing_repair_source_objective_pitch_role_risk_count": _int(
            aggregate.get("outside_soloing_repair_source_objective_pitch_role_risk_count")
        ),
        "outside_soloing_repair_source_pitch_role_risk_count_before": _int(
            aggregate.get("outside_soloing_repair_source_pitch_role_risk_count_before")
        ),
        "outside_soloing_repair_source_pitch_role_risk_count_after": _int(
            aggregate.get("outside_soloing_repair_source_pitch_role_risk_count_after")
        ),
        "outside_soloing_repair_source_pitch_role_risk_delta": _int(
            aggregate.get("outside_soloing_repair_source_pitch_role_risk_delta")
        ),
        "outside_soloing_repair_source_targeted": bool(
            aggregate.get("outside_soloing_repair_source_targeted", True)
        ),
        "outside_soloing_repair_source_residual_risk_preserved": bool(
            aggregate.get("outside_soloing_repair_source_residual_risk_preserved", False)
        ),
        "outside_soloing_repair_pitch_role_risk_count_after": _int(
            aggregate.get("outside_soloing_repair_pitch_role_risk_count_after")
        ),
        "outside_soloing_repair_pitch_role_risk_delta": _int(
            aggregate.get("outside_soloing_repair_pitch_role_risk_delta")
        ),
        "outside_soloing_not_evaluable_count": _int(
            aggregate.get("outside_soloing_not_evaluable_count")
        ),
        **{
            key: aggregate.get(key)
            for key in BRIDGE_SOURCE_CONTEXT_KEYS
        },
        "targeted_quality_repair_sweep_ready": bool(
            readiness.get("targeted_quality_repair_sweep_ready", False)
        ),
        "audio_review_package_ready": bool(readiness.get("audio_review_package_ready", False)),
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
    aggregate = report["aggregate"]
    selected = report["selected_next_target"]
    readiness = report["readiness"]
    lines = [
        "# Stage B MIDI-to-Solo Candidate Failure Labeling Source Context Refresh",
        "",
        "## Summary",
        "",
        f"- boundary: `{report['boundary']}`",
        f"- source boundary: `{report['source_boundary']}`",
        f"- next boundary: `{selected['selected_next_boundary']}`",
        f"- selected target: `{selected['selected_target']}`",
        f"- candidate count: `{aggregate['candidate_count']}`",
        f"- failed candidate count: `{aggregate['failed_candidate_count']}`",
        f"- outside-soloing repair evidence ready: `{_bool_token(aggregate['outside_soloing_repair_evidence_ready'])}`",
        f"- outside-soloing repair source context preserved: `{_bool_token(aggregate['outside_soloing_repair_source_context_preserved'])}`",
        f"- outside-soloing repair WAV count: `{aggregate['outside_soloing_repair_wav_count']}`",
        f"- outside-soloing source objective pitch-role risk: `{aggregate['outside_soloing_repair_source_objective_pitch_role_risk_count']}`",
        f"- outside-soloing source pitch-role risk before / after / delta: `{aggregate['outside_soloing_repair_source_pitch_role_risk_count_before']}` / `{aggregate['outside_soloing_repair_source_pitch_role_risk_count_after']}` / `{aggregate['outside_soloing_repair_source_pitch_role_risk_delta']}`",
        f"- outside-soloing source repair targeted: `{_bool_token(aggregate['outside_soloing_repair_source_targeted'])}`",
        f"- outside-soloing source residual risk preserved: `{_bool_token(aggregate['outside_soloing_repair_source_residual_risk_preserved'])}`",
        f"- outside-soloing current repair pitch-role risk after / delta: `{aggregate['outside_soloing_repair_pitch_role_risk_count_after']}` / `{aggregate['outside_soloing_repair_pitch_role_risk_delta']}`",
        f"- follow-up objective source outside-soloing source pitch-role risk: `{aggregate['followup_objective_source_outside_soloing_source_pitch_role_risk_count_before']} -> {aggregate['followup_objective_source_outside_soloing_source_pitch_role_risk_count_after']}`",
        f"- follow-up objective source outside-soloing current repair pitch-role risk after/delta: `{aggregate['followup_objective_source_outside_soloing_current_pitch_role_risk_count_after']} / {aggregate['followup_objective_source_outside_soloing_current_pitch_role_risk_delta']}`",
        f"- follow-up repair sweep source outside-soloing source pitch-role risk: `{aggregate['followup_repair_sweep_source_outside_soloing_source_pitch_role_risk_count_before']} -> {aggregate['followup_repair_sweep_source_outside_soloing_source_pitch_role_risk_count_after']}`",
        f"- follow-up repair sweep source outside-soloing current repair pitch-role risk after/delta: `{aggregate['followup_repair_sweep_source_outside_soloing_current_pitch_role_risk_count_after']} / {aggregate['followup_repair_sweep_source_outside_soloing_current_pitch_role_risk_delta']}`",
        f"- bridge repair sweep source outside-soloing source pitch-role risk: `{aggregate['repair_sweep_source_outside_soloing_source_pitch_role_risk_count_before']} -> {aggregate['repair_sweep_source_outside_soloing_source_pitch_role_risk_count_after']}`",
        f"- bridge repair sweep source outside-soloing current repair pitch-role risk after/delta: `{aggregate['repair_sweep_source_outside_soloing_current_pitch_role_risk_count_after']} / {aggregate['repair_sweep_source_outside_soloing_current_pitch_role_risk_delta']}`",
        f"- outside-soloing not evaluable count: `{aggregate['outside_soloing_not_evaluable_count']}`",
        f"- targeted quality repair sweep ready: `{_bool_token(readiness['targeted_quality_repair_sweep_ready'])}`",
        "",
        "## Failure Counts",
        "",
    ]
    for label, count in aggregate["failure_counts"].items():
        lines.append(f"- `{label}`: `{count}`")
    if not aggregate["failure_counts"]:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Not Evaluable",
            "",
        ]
    )
    for label, count in aggregate["not_evaluable_counts"].items():
        lines.append(f"- `{label}`: `{count}`")
    if not aggregate["not_evaluable_counts"]:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            f"- human/audio preference claimed: `{_bool_token(readiness['human_audio_preference_claimed'])}`",
            f"- MIDI-to-solo musical quality claimed: `{_bool_token(readiness['midi_to_solo_musical_quality_claimed'])}`",
            "",
            "## Next",
            "",
            f"- `{report['next_recommended_issue']}`",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Label MIDI-to-solo candidate failures")
    parser.add_argument("--rubric_baseline", type=str, required=True)
    parser.add_argument("--mvp_delivery_package", type=str, required=True)
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/stage_b_midi_to_solo_candidate_failure_labeling",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--issue_number", type=int, default=748)
    parser.add_argument("--expected_boundary", type=str, default="")
    parser.add_argument("--min_candidate_count", type=int, default=6)
    parser.add_argument("--require_labeling_completed", action="store_true")
    parser.add_argument("--require_no_quality_claim", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    report = build_candidate_failure_labeling_report(
        rubric_baseline=read_json(Path(args.rubric_baseline)),
        mvp_delivery_package=read_json(Path(args.mvp_delivery_package)),
        output_dir=output_dir,
        issue_number=int(args.issue_number),
    )
    summary = validate_candidate_failure_labeling_report(
        report,
        expected_boundary=str(args.expected_boundary or ""),
        min_candidate_count=int(args.min_candidate_count),
        require_labeling_completed=bool(args.require_labeling_completed),
        require_no_quality_claim=bool(args.require_no_quality_claim),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "stage_b_midi_to_solo_candidate_failure_labeling.json", report)
    write_json(
        output_dir / "stage_b_midi_to_solo_candidate_failure_labeling_validation_summary.json",
        summary,
    )
    markdown = markdown_report(report)
    write_text(output_dir / "stage_b_midi_to_solo_candidate_failure_labeling.md", markdown)
    if args.doc_path:
        write_text(Path(args.doc_path), markdown)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
