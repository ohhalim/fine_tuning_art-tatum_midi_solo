"""Run a phrase/rhythm repair sweep for songlike contour repaired candidates."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pretty_midi

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from scripts.assess_stage_b_generic_base_readiness import read_json, write_json, write_text  # noqa: E402
from scripts.audit_stage_b_midi_to_solo_final_status import (  # noqa: E402
    BRIDGE_SOURCE_CONTEXT_KEYS,
)
from scripts.decide_stage_b_midi_to_solo_songlike_melody_contour_repair_followup import (  # noqa: E402
    BOUNDARY as FOLLOWUP_BOUNDARY,
    NEXT_BOUNDARY as FOLLOWUP_NEXT_BOUNDARY,
    SELECTED_TARGET as FOLLOWUP_SELECTED_TARGET,
)
from scripts.label_stage_b_midi_to_solo_candidate_failures import (  # noqa: E402
    analyze_candidate_midi,
    label_candidate,
    validate_rubric_baseline,
)
from scripts.run_stage_b_midi_to_solo_songlike_melody_contour_repair_sweep import (  # noqa: E402
    BOUNDARY as SOURCE_REPAIR_SWEEP_BOUNDARY,
)


class StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairSweepError(ValueError):
    pass


BOUNDARY = "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_repair_sweep"
NEXT_BOUNDARY = "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_repair_audio_package"
SELECTED_TARGET = "songlike_melody_contour_phrase_rhythm_repair_audio_package"
SCHEMA_VERSION = "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_repair_sweep_v3"
PHRASE_LABEL = "phrase_shape_missing_tension_release"
RHYTHM_LABEL = "rhythmic_monotony"
TARGET_LABELS = (PHRASE_LABEL, RHYTHM_LABEL)
BAR_SECONDS = 2.0

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
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairSweepError(
            f"unexpected quality claim in {label}: {claimed}"
        )


def _bridge_source_context_fields(
    source: dict[str, Any],
    *,
    bridge_prefix: str,
    label: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in BRIDGE_SOURCE_CONTEXT_KEYS:
        source_key = f"{bridge_prefix}_{key}" if bridge_prefix else key
        if source_key not in source or source[source_key] is None:
            raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairSweepError(
                f"{label} source-context field required: {source_key}"
            )
        result[key] = source[source_key]
    return result


def _source_context_fields(
    source: dict[str, Any],
    *,
    prefix: str = "source_outside_soloing_repair",
    bridge_prefix: str = "",
    label: str,
) -> dict[str, Any]:
    return {
        "source_outside_soloing_repair_source_objective_pitch_role_risk_count": _int(
            source.get(f"{prefix}_source_objective_pitch_role_risk_count")
        ),
        "source_outside_soloing_repair_source_context_preserved": bool(
            source.get(f"{prefix}_source_context_preserved", False)
        ),
        "source_outside_soloing_repair_source_pitch_role_risk_count_before": _int(
            source.get(f"{prefix}_source_pitch_role_risk_count_before")
        ),
        "source_outside_soloing_repair_source_pitch_role_risk_count_after": _int(
            source.get(f"{prefix}_source_pitch_role_risk_count_after")
        ),
        "source_outside_soloing_repair_source_pitch_role_risk_delta": _int(
            source.get(f"{prefix}_source_pitch_role_risk_delta")
        ),
        "source_outside_soloing_repair_source_targeted": bool(
            source.get(f"{prefix}_source_targeted", True)
        ),
        "source_outside_soloing_repair_source_residual_risk_preserved": bool(
            source.get(f"{prefix}_source_residual_risk_preserved", False)
        ),
        "source_outside_soloing_repair_pitch_role_risk_count_after": _int(
            source.get(f"{prefix}_pitch_role_risk_count_after")
        ),
        "source_outside_soloing_repair_pitch_role_risk_delta": _int(
            source.get(f"{prefix}_pitch_role_risk_delta")
        ),
        **_bridge_source_context_fields(
            source,
            bridge_prefix=bridge_prefix,
            label=label,
        ),
    }


def _validate_source_context(context: dict[str, Any], *, label: str) -> None:
    objective_risk = _int(
        context.get("source_outside_soloing_repair_source_objective_pitch_role_risk_count")
    )
    source_before = _int(
        context.get("source_outside_soloing_repair_source_pitch_role_risk_count_before")
    )
    source_after = _int(
        context.get("source_outside_soloing_repair_source_pitch_role_risk_count_after")
    )
    source_delta = _int(
        context.get("source_outside_soloing_repair_source_pitch_role_risk_delta")
    )
    current_after = _int(
        context.get("source_outside_soloing_repair_pitch_role_risk_count_after")
    )
    current_delta = _int(context.get("source_outside_soloing_repair_pitch_role_risk_delta"))
    if objective_risk <= 0 or source_before <= 0:
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairSweepError(
            f"{label} source pitch-role risk context required"
        )
    if objective_risk != source_before:
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairSweepError(
            f"{label} objective/source risk mismatch"
        )
    if source_before - source_after != source_delta:
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairSweepError(
            f"{label} source pitch-role risk delta mismatch"
        )
    if source_delta <= 0:
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairSweepError(
            f"{label} positive source pitch-role risk delta required"
        )
    if bool(context.get("source_outside_soloing_repair_source_targeted", True)):
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairSweepError(
            f"{label} source-targeted flag should remain false"
        )
    if not bool(context.get("source_outside_soloing_repair_source_context_preserved", False)):
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairSweepError(
            f"{label} source context preservation required"
        )
    if not bool(
        context.get("source_outside_soloing_repair_source_residual_risk_preserved", False)
    ):
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairSweepError(
            f"{label} source residual risk preservation required"
        )
    if current_after != 0:
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairSweepError(
            f"{label} current repair residual pitch-role risk should be zero"
        )
    if current_delta <= 0:
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairSweepError(
            f"{label} positive current repair pitch-role risk delta required"
        )


def _require_matching_source_context(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    label: str,
) -> None:
    for key, left_value in left.items():
        right_value = right.get(key)
        mismatch = (
            bool(left_value) != bool(right_value)
            if isinstance(left_value, bool)
            else left_value != right_value
        )
        if mismatch:
            raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairSweepError(
                f"{label} mismatch for {key}"
            )


def _prefixed_context_for_output(
    context: dict[str, Any],
    *,
    prefix: str,
) -> dict[str, Any]:
    return {
        **{
            f"{prefix}_{key}": value
            for key, value in context.items()
            if key.startswith("source_outside_soloing_repair_")
        },
        **{f"{prefix}_{key}": context[key] for key in BRIDGE_SOURCE_CONTEXT_KEYS},
    }


def validate_followup_decision(report: dict[str, Any]) -> dict[str, Any]:
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    selected = _dict(report.get("selected_next_target"))
    sweep = _dict(report.get("repair_sweep_summary"))
    if str(report.get("boundary") or "") != FOLLOWUP_BOUNDARY:
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairSweepError(
            "songlike contour follow-up decision boundary required"
        )
    if str(decision.get("next_boundary") or "") != FOLLOWUP_NEXT_BOUNDARY:
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairSweepError(
            "follow-up decision must route to phrase/rhythm repair sweep"
        )
    if str(selected.get("selected_target") or "") != FOLLOWUP_SELECTED_TARGET:
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairSweepError(
            "phrase/rhythm repair target required"
        )
    if not bool(readiness.get("followup_decision_completed", False)):
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairSweepError(
            "follow-up decision completion required"
        )
    if not bool(readiness.get("phrase_rhythm_tie_target_selected", False)):
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairSweepError(
            "phrase/rhythm tie target selection required"
        )
    if _int(readiness.get("technical_regression_count")) != 0:
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairSweepError(
            "source technical regression must remain zero"
        )
    if not bool(readiness.get("objective_source_outside_soloing_repair_evidence_ready", False)):
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairSweepError(
            "objective outside-soloing repair evidence readiness required"
        )
    if _int(readiness.get("objective_source_outside_soloing_repair_pitch_role_risk_count_after")) != 0:
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairSweepError(
            "objective outside-soloing residual pitch-role risk should be zero"
        )
    if _int(readiness.get("objective_source_outside_soloing_not_evaluable_count")) <= 0:
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairSweepError(
            "objective source outside-soloing not-evaluable boundary required"
        )
    if _int(readiness.get("objective_repaired_outside_soloing_not_evaluable_count")) <= 0:
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairSweepError(
            "objective repaired outside-soloing not-evaluable boundary required"
        )
    if not bool(readiness.get("repair_sweep_source_outside_soloing_repair_evidence_ready", False)):
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairSweepError(
            "repair sweep outside-soloing repair evidence readiness required"
        )
    if _int(readiness.get("repair_sweep_source_outside_soloing_repair_pitch_role_risk_count_after")) != 0:
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairSweepError(
            "repair sweep outside-soloing residual pitch-role risk should be zero"
        )
    if _int(readiness.get("repair_sweep_source_outside_soloing_not_evaluable_count")) <= 0:
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairSweepError(
            "repair sweep source outside-soloing not-evaluable boundary required"
        )
    if _int(readiness.get("repair_sweep_repaired_outside_soloing_not_evaluable_count")) <= 0:
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairSweepError(
            "repair sweep repaired outside-soloing not-evaluable boundary required"
        )
    objective_context = _source_context_fields(
        readiness,
        prefix="objective_source_outside_soloing_repair",
        bridge_prefix="objective",
        label="follow-up objective source context",
    )
    _validate_source_context(
        objective_context,
        label="follow-up objective source context",
    )
    source_context = _source_context_fields(
        readiness,
        prefix="repair_sweep_source_outside_soloing_repair",
        bridge_prefix="repair_sweep",
        label="follow-up repair sweep source context",
    )
    _validate_source_context(source_context, label="follow-up repair sweep source context")
    primary_labels = [str(label) for label in _list(selected.get("primary_remaining_failure_labels"))]
    if not all(label in primary_labels for label in TARGET_LABELS):
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairSweepError(
            "phrase/rhythm primary labels required"
        )
    if _int(selected.get("primary_remaining_failure_count")) <= 0:
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairSweepError(
            "primary remaining label count required"
        )
    _require_no_quality_claim(readiness, label="follow-up readiness")
    return {
        "candidate_count": _int(readiness.get("candidate_count")),
        "source_total_failure_label_count": _int(
            readiness.get("source_total_failure_label_count")
        ),
        "repaired_total_failure_label_count": _int(
            readiness.get("repaired_total_failure_label_count")
        ),
        "source_failure_label_delta": _int(readiness.get("failure_label_delta")),
        "remaining_failure_counts": _dict(sweep.get("remaining_failure_counts")),
        "primary_remaining_failure_labels": primary_labels,
        "primary_remaining_failure_count": _int(selected.get("primary_remaining_failure_count")),
        **_prefixed_context_for_output(objective_context, prefix="objective"),
        "objective_source_outside_soloing_repair_evidence_ready": bool(
            readiness.get("objective_source_outside_soloing_repair_evidence_ready", False)
        ),
        "objective_source_outside_soloing_repair_pitch_role_risk_count_after": _int(
            readiness.get("objective_source_outside_soloing_repair_pitch_role_risk_count_after")
        ),
        "objective_source_outside_soloing_not_evaluable_count": _int(
            readiness.get("objective_source_outside_soloing_not_evaluable_count")
        ),
        "objective_repaired_outside_soloing_not_evaluable_count": _int(
            readiness.get("objective_repaired_outside_soloing_not_evaluable_count")
        ),
        "repair_sweep_source_outside_soloing_repair_evidence_ready": bool(
            readiness.get("repair_sweep_source_outside_soloing_repair_evidence_ready", False)
        ),
        "repair_sweep_source_outside_soloing_repair_pitch_role_risk_count_after": _int(
            readiness.get("repair_sweep_source_outside_soloing_repair_pitch_role_risk_count_after")
        ),
        "repair_sweep_source_outside_soloing_not_evaluable_count": _int(
            readiness.get("repair_sweep_source_outside_soloing_not_evaluable_count")
        ),
        "repair_sweep_repaired_outside_soloing_not_evaluable_count": _int(
            readiness.get("repair_sweep_repaired_outside_soloing_not_evaluable_count")
        ),
        **source_context,
    }


def validate_source_repair_sweep(report: dict[str, Any]) -> dict[str, Any]:
    readiness = _dict(report.get("readiness"))
    aggregate = _dict(report.get("aggregate"))
    rows = [_dict(item) for item in _list(report.get("candidate_repairs"))]
    if str(report.get("boundary") or "") != SOURCE_REPAIR_SWEEP_BOUNDARY:
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairSweepError(
            "songlike contour repair sweep source required"
        )
    if not bool(readiness.get("songlike_melody_contour_repair_sweep_completed", False)):
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairSweepError(
            "songlike contour repair sweep completion required"
        )
    if not bool(readiness.get("songlike_melody_contour_repair_target_supported", False)):
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairSweepError(
            "songlike contour repair target support required"
        )
    if _int(aggregate.get("technical_regression_count")) != 0:
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairSweepError(
            "source technical regression must remain zero"
        )
    if not bool(aggregate.get("source_outside_soloing_repair_evidence_ready", False)):
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairSweepError(
            "source outside-soloing repair evidence readiness required"
        )
    if _int(aggregate.get("source_outside_soloing_repair_pitch_role_risk_count_after")) != 0:
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairSweepError(
            "source outside-soloing residual pitch-role risk should be zero"
        )
    source_context = _source_context_fields(
        aggregate,
        label="source repair sweep context",
    )
    _validate_source_context(source_context, label="source repair sweep context")
    if _int(aggregate.get("source_outside_soloing_not_evaluable_count")) <= 0:
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairSweepError(
            "source outside-soloing not-evaluable boundary required"
        )
    if _int(aggregate.get("repaired_outside_soloing_not_evaluable_count")) <= 0:
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairSweepError(
            "repaired outside-soloing not-evaluable boundary required"
        )
    if len(rows) < 6:
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairSweepError(
            "source repair row count below 6"
        )
    _require_no_quality_claim(readiness, label="songlike contour repair readiness")
    return {"rows": rows, **source_context}


def density_patterns() -> list[list[int]]:
    return [
        [5, 4, 6, 3, 5, 6, 4, 5],
        [6, 3, 5, 4, 6, 5, 4, 3],
        [4, 6, 5, 3, 6, 4, 5, 6],
        [5, 6, 3, 5, 4, 6, 5, 4],
        [6, 5, 4, 6, 3, 5, 6, 4],
        [4, 5, 6, 4, 5, 3, 6, 5],
    ]


def onset_patterns_by_density(variant: int) -> dict[int, list[list[float]]]:
    banks = [
        {
            3: [[0.0, 1.25, 2.75], [0.5, 1.5, 3.25]],
            4: [[0.0, 0.75, 2.0, 3.0], [0.25, 1.0, 2.25, 3.5]],
            5: [[0.0, 0.5, 1.5, 2.5, 3.25], [0.25, 1.0, 1.75, 2.75, 3.5]],
            6: [[0.0, 0.5, 1.0, 1.75, 2.5, 3.25], [0.25, 0.75, 1.5, 2.0, 2.75, 3.5]],
        },
        {
            3: [[0.25, 1.75, 3.0], [0.0, 1.0, 2.5]],
            4: [[0.5, 1.25, 2.0, 3.25], [0.0, 1.5, 2.25, 3.5]],
            5: [[0.0, 0.75, 1.25, 2.5, 3.25], [0.5, 1.0, 1.75, 2.75, 3.5]],
            6: [[0.0, 0.5, 1.25, 1.75, 2.5, 3.5], [0.25, 0.75, 1.5, 2.25, 2.75, 3.75]],
        },
        {
            3: [[0.5, 1.5, 2.75], [0.0, 1.25, 3.25]],
            4: [[0.0, 1.0, 2.0, 3.25], [0.25, 1.25, 2.5, 3.5]],
            5: [[0.0, 0.5, 1.25, 2.25, 3.5], [0.25, 1.0, 1.75, 2.5, 3.25]],
            6: [[0.0, 0.75, 1.25, 2.0, 2.5, 3.5], [0.25, 0.5, 1.5, 2.25, 3.0, 3.75]],
        },
    ]
    return banks[variant % len(banks)]


def durations_for_count(variant: int, density: int, bar_index: int) -> list[float]:
    base = {
        3: [0.31, 0.47, 0.59],
        4: [0.24, 0.39, 0.53, 0.33],
        5: [0.21, 0.36, 0.27, 0.49, 0.41],
        6: [0.18, 0.29, 0.23, 0.37, 0.26, 0.46],
    }[density]
    return [
        round(value + 0.011 * ((variant + bar_index + index) % 5), 3)
        for index, value in enumerate(base)
    ]


def contour_cells(variant: int, density: int, bar_index: int) -> list[int]:
    banks = {
        3: [[0, 7, 2], [4, -3, 8], [7, 0, 11]],
        4: [[0, 7, 2, 10], [5, -2, 9, 3], [7, 12, 5, 10]],
        5: [[0, 7, 2, 10, 5], [4, -3, 8, 1, 11], [7, 12, 5, 9, 2]],
        6: [[0, 7, 2, 10, 5, 12], [4, -3, 8, 1, 11, 6], [7, 12, 5, 9, 2, 10]],
    }
    return banks[density][(variant + bar_index) % len(banks[density])]


def raw_contour_pitches(variant: int, densities: list[int]) -> list[int]:
    total_notes = sum(densities)
    peak_index = max(1, int(total_notes * (0.52 + 0.02 * (variant % 3))))
    oscillation = [0, 6, -3, 8, -5, 5, -2, 7, -4, 4, 9, -1]
    pitches: list[int] = []
    for index in range(total_notes):
        if index <= peak_index:
            center = 58 + (index / max(1, peak_index)) * 16
        else:
            center = 74 - ((index - peak_index) / max(1, total_notes - peak_index - 1)) * 11
        pitch = int(round(center)) + oscillation[(index + variant * 2) % len(oscillation)]
        pitches.append(max(54, min(80, pitch)))
    return pitches


def fit_interval_limit(raw_pitches: list[int], *, max_interval: int = 12) -> list[int]:
    if not raw_pitches:
        return []
    fitted = [int(raw_pitches[0])]
    for raw in raw_pitches[1:]:
        pitch = int(raw)
        previous = fitted[-1]
        while pitch - previous > max_interval:
            pitch -= 12
        while previous - pitch > max_interval:
            pitch += 12
        fitted.append(pitch)
    return fitted


def write_phrase_rhythm_repair_midi(
    *,
    output_path: Path,
    rank: int,
    variant: int,
) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    densities = density_patterns()[variant % len(density_patterns())]
    onset_bank = onset_patterns_by_density(variant)
    pitches = fit_interval_limit(raw_contour_pitches(variant, densities), max_interval=12)
    midi = pretty_midi.PrettyMIDI(initial_tempo=120)
    midi.time_signature_changes.append(pretty_midi.TimeSignature(4, 4, 0.0))
    instrument = pretty_midi.Instrument(
        program=0,
        is_drum=False,
        name=f"phrase_rhythm_repair_rank_{rank}",
    )
    pitch_index = 0
    for bar_index, density in enumerate(densities):
        onset_options = onset_bank[density]
        offsets = onset_options[(bar_index + variant) % len(onset_options)]
        durations = durations_for_count(variant, density, bar_index)
        starts = [bar_index * BAR_SECONDS + beat_offset * 0.5 for beat_offset in offsets]
        for note_index, start in enumerate(starts):
            next_start = (
                starts[note_index + 1]
                if note_index + 1 < len(starts)
                else (bar_index + 1) * BAR_SECONDS
            )
            duration = min(durations[note_index], max(0.08, next_start - start - 0.03))
            instrument.notes.append(
                pretty_midi.Note(
                    velocity=84,
                    pitch=int(pitches[pitch_index]),
                    start=float(start),
                    end=float(start + duration),
                )
            )
            pitch_index += 1
    midi.instruments.append(instrument)
    midi.write(str(output_path))
    return {
        "rank": int(rank),
        "midi_path": str(output_path),
        "density_pattern": densities,
        "repair_source": "phrase_rhythm_arc_cells",
    }


def relabel_generated_candidates(
    generated_rows: list[dict[str, Any]],
    *,
    rubric_items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    analyzed = [
        analyze_candidate_midi(
            {
                "source": str(row.get("source") or ""),
                "rank": _int(row.get("rank")),
                "midi_path": str(row.get("phrase_rhythm_repaired_midi_path") or ""),
                "source_objective_supported": True,
                "source_dead_air_ratio": 0.0,
            }
        )
        for row in generated_rows
    ]
    rhythm_counts = Counter(tuple(item.get("rhythm_signature", ((), ()))) for item in analyzed)
    relabeled: list[dict[str, Any]] = []
    for row, item in zip(generated_rows, analyzed, strict=False):
        labeled = label_candidate(
            item,
            rubric_items=rubric_items,
            shared_rhythm_signature_count=rhythm_counts[
                tuple(item.get("rhythm_signature", ((), ())))
            ],
        )
        relabeled.append({**row, "phrase_rhythm_repaired_labeling": labeled})
    return relabeled


def failure_counts(rows: list[dict[str, Any]], *, key: str) -> Counter[str]:
    return Counter(
        label
        for row in rows
        for label in _list(_dict(row.get(key)).get("failure_labels"))
    )


def not_evaluable_counts(rows: list[dict[str, Any]], *, key: str) -> Counter[str]:
    return Counter(
        label
        for row in rows
        for label in _list(_dict(row.get(key)).get("not_evaluable_labels"))
    )


def build_songlike_melody_contour_phrase_rhythm_repair_sweep_report(
    *,
    followup_decision: dict[str, Any],
    source_repair_sweep: dict[str, Any],
    rubric_baseline: dict[str, Any],
    output_dir: Path,
    issue_number: int,
) -> dict[str, Any]:
    followup = validate_followup_decision(followup_decision)
    source = validate_source_repair_sweep(source_repair_sweep)
    source_rows = [_dict(item) for item in _list(source.get("rows"))]
    source_context = {
        key: value
        for key, value in source.items()
        if key.startswith("source_outside_soloing_repair_")
        or key in BRIDGE_SOURCE_CONTEXT_KEYS
    }
    followup_context = {
        key: value
        for key, value in followup.items()
        if key.startswith("source_outside_soloing_repair_")
        or key in BRIDGE_SOURCE_CONTEXT_KEYS
    }
    objective_context = {
        key: value
        for key, value in followup.items()
        if key.startswith("objective_source_outside_soloing_repair_")
        or key in {f"objective_{source_key}" for source_key in BRIDGE_SOURCE_CONTEXT_KEYS}
    }
    _require_matching_source_context(
        followup_context,
        source_context,
        label="follow-up/source repair sweep source context",
    )
    rubric_summary = validate_rubric_baseline(rubric_baseline)
    rubric_items = [_dict(item) for item in _list(rubric_summary.get("rubric_items"))]
    if len(source_rows) < followup["candidate_count"]:
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairSweepError(
            "source row count below follow-up candidate count"
        )

    midi_dir = output_dir / "midi"
    generated_rows: list[dict[str, Any]] = []
    for index, source in enumerate(source_rows[: followup["candidate_count"]], start=1):
        source_labeling = _dict(source.get("contour_repaired_labeling"))
        output_path = midi_dir / f"{index:02d}_songlike_melody_contour_phrase_rhythm_repair.mid"
        generated = write_phrase_rhythm_repair_midi(
            output_path=output_path,
            rank=index,
            variant=index - 1,
        )
        source_failure_labels = _list(source_labeling.get("failure_labels"))
        generated_rows.append(
            {
                "source": str(source.get("source") or ""),
                "rank": index,
                "source_rank": _int(source.get("rank")) or index,
                "source_midi_path": str(source.get("contour_repaired_midi_path") or ""),
                "source_failure_labels": source_failure_labels,
                "source_failure_label_count": len(source_failure_labels),
                "source_phrase_rhythm_label_count": sum(
                    1 for label in source_failure_labels if label in TARGET_LABELS
                ),
                "phrase_rhythm_repaired_midi_path": generated["midi_path"],
                "density_pattern": generated["density_pattern"],
                "repair_source": generated["repair_source"],
            }
        )

    relabeled = relabel_generated_candidates(generated_rows, rubric_items=rubric_items)
    total_before = sum(_int(item.get("source_failure_label_count")) for item in relabeled)
    total_after = sum(
        len(_list(_dict(item.get("phrase_rhythm_repaired_labeling")).get("failure_labels")))
        for item in relabeled
    )
    source_phrase_rhythm_count = sum(
        _int(item.get("source_phrase_rhythm_label_count")) for item in relabeled
    )
    repaired_failures = failure_counts(relabeled, key="phrase_rhythm_repaired_labeling")
    repaired_not_evaluable = not_evaluable_counts(
        relabeled,
        key="phrase_rhythm_repaired_labeling",
    )
    repaired_phrase_rhythm_count = sum(repaired_failures.get(label, 0) for label in TARGET_LABELS)
    repaired_outside_soloing_count = repaired_not_evaluable.get(
        "outside_soloing_without_context",
        0,
    )
    improved_count = sum(
        1
        for item in relabeled
        if len(_list(_dict(item.get("phrase_rhythm_repaired_labeling")).get("failure_labels")))
        < _int(item.get("source_failure_label_count"))
    )
    technical_regression_count = repaired_failures.get("technical_gate_regression", 0)
    target_supported = (
        total_after < total_before
        and repaired_phrase_rhythm_count < source_phrase_rhythm_count
        and technical_regression_count == 0
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "issue_number": int(issue_number),
        "boundary": BOUNDARY,
        "source_boundary": FOLLOWUP_BOUNDARY,
        "source_repair_sweep_boundary": SOURCE_REPAIR_SWEEP_BOUNDARY,
        "candidate_repairs": relabeled,
        "aggregate": {
            "candidate_count": len(relabeled),
            "source_total_failure_label_count": total_before,
            "repaired_total_failure_label_count": total_after,
            "failure_label_delta": total_before - total_after,
            "source_phrase_rhythm_failure_count": source_phrase_rhythm_count,
            "repaired_phrase_rhythm_failure_count": repaired_phrase_rhythm_count,
            "phrase_rhythm_failure_delta": (
                source_phrase_rhythm_count - repaired_phrase_rhythm_count
            ),
            "improved_candidate_count": improved_count,
            "technical_regression_count": technical_regression_count,
            "repaired_failure_counts": dict(sorted(repaired_failures.items())),
            "source_outside_soloing_repair_evidence_ready": bool(
                followup["repair_sweep_source_outside_soloing_repair_evidence_ready"]
            ),
            "source_outside_soloing_repair_pitch_role_risk_count_after": _int(
                followup["repair_sweep_source_outside_soloing_repair_pitch_role_risk_count_after"]
            ),
            **objective_context,
            **source_context,
            "source_outside_soloing_not_evaluable_count": _int(
                followup["repair_sweep_repaired_outside_soloing_not_evaluable_count"]
            ),
            "repaired_outside_soloing_not_evaluable_count": _int(
                repaired_outside_soloing_count
            ),
            "repaired_not_evaluable_counts": dict(sorted(repaired_not_evaluable.items())),
            "target_supported": target_supported,
        },
        "selected_next_target": {
            "selected_target": SELECTED_TARGET
            if target_supported
            else "songlike_melody_contour_repair_followup_decision",
            "selected_next_boundary": NEXT_BOUNDARY
            if target_supported
            else "stage_b_midi_to_solo_songlike_melody_contour_repair_followup_decision",
            "reason": "phrase/rhythm labels reduced without technical regression"
            if target_supported
            else "phrase/rhythm labels were not reduced enough",
        },
        "readiness": {
            "boundary": BOUNDARY,
            "songlike_melody_contour_phrase_rhythm_repair_sweep_completed": True,
            "songlike_melody_contour_phrase_rhythm_repair_target_supported": target_supported,
            "candidate_count": len(relabeled),
            "failure_label_delta": total_before - total_after,
            "phrase_rhythm_failure_delta": (
                source_phrase_rhythm_count - repaired_phrase_rhythm_count
            ),
            "technical_regression_count": technical_regression_count,
            "audio_package_ready": target_supported,
            "source_outside_soloing_repair_evidence_ready": bool(
                followup["repair_sweep_source_outside_soloing_repair_evidence_ready"]
            ),
            "source_outside_soloing_repair_pitch_role_risk_count_after": _int(
                followup["repair_sweep_source_outside_soloing_repair_pitch_role_risk_count_after"]
            ),
            **objective_context,
            **source_context,
            "source_outside_soloing_not_evaluable_count": _int(
                followup["repair_sweep_repaired_outside_soloing_not_evaluable_count"]
            ),
            "repaired_outside_soloing_not_evaluable_count": _int(
                repaired_outside_soloing_count
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
            "next_boundary": NEXT_BOUNDARY
            if target_supported
            else "stage_b_midi_to_solo_songlike_melody_contour_repair_followup_decision",
            "auto_progress_allowed": True,
            "critical_user_input_required": False,
            "reason": "phrase/rhythm repair sweep completed without musical quality claim",
        },
        "not_proven": [
            "human_audio_preference",
            "midi_to_solo_musical_quality",
            "audio_rendered_quality",
            "outside_soloing_without_context",
            "weak_chord_tone_landing",
            "broad_trained_model_quality",
        ],
        "next_recommended_issue": (
            "Stage B MIDI-to-solo songlike melody contour phrase/rhythm repair audio package source-context refresh"
        )
        if target_supported
        else "Stage B MIDI-to-solo songlike melody contour phrase/rhythm repair follow-up decision",
        "source_summary": followup,
    }


def validate_songlike_melody_contour_phrase_rhythm_repair_sweep_report(
    report: dict[str, Any],
    *,
    expected_boundary: str | None,
    expected_next_boundary: str | None,
    min_candidate_count: int,
    require_sweep_completed: bool,
    require_target_supported: bool,
    require_phrase_rhythm_delta: bool,
    require_no_quality_claim: bool,
) -> dict[str, Any]:
    boundary = str(report.get("boundary") or "")
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    aggregate = _dict(report.get("aggregate"))
    repairs = _list(report.get("candidate_repairs"))
    if expected_boundary and boundary != expected_boundary:
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairSweepError(
            f"expected boundary {expected_boundary}, got {boundary}"
        )
    if expected_next_boundary and str(decision.get("next_boundary") or "") != expected_next_boundary:
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairSweepError(
            "unexpected next boundary"
        )
    if require_sweep_completed and not bool(
        readiness.get(
            "songlike_melody_contour_phrase_rhythm_repair_sweep_completed", False
        )
    ):
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairSweepError(
            "phrase/rhythm repair sweep completion required"
        )
    if len(repairs) < int(min_candidate_count):
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairSweepError(
            "candidate count below minimum"
        )
    if require_target_supported and not bool(aggregate.get("target_supported", False)):
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairSweepError(
            "target support required"
        )
    if require_phrase_rhythm_delta and _int(aggregate.get("phrase_rhythm_failure_delta")) <= 0:
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairSweepError(
            "positive phrase/rhythm failure delta required"
        )
    if _int(aggregate.get("technical_regression_count")) != 0:
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairSweepError(
            "technical regression should be zero"
        )
    if not bool(aggregate.get("source_outside_soloing_repair_evidence_ready", False)):
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairSweepError(
            "outside-soloing repair evidence readiness required"
        )
    if _int(aggregate.get("source_outside_soloing_repair_pitch_role_risk_count_after")) != 0:
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairSweepError(
            "outside-soloing residual pitch-role risk should be zero"
        )
    objective_context = _source_context_fields(
        aggregate,
        prefix="objective_source_outside_soloing_repair",
        bridge_prefix="objective",
        label="aggregate objective source context",
    )
    _validate_source_context(
        objective_context,
        label="aggregate objective source context",
    )
    source_context = _source_context_fields(
        aggregate,
        label="aggregate source context",
    )
    _validate_source_context(source_context, label="aggregate source context")
    if _int(aggregate.get("source_outside_soloing_not_evaluable_count")) <= 0:
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairSweepError(
            "source outside-soloing not-evaluable boundary required"
        )
    if _int(aggregate.get("repaired_outside_soloing_not_evaluable_count")) <= 0:
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairSweepError(
            "repaired outside-soloing not-evaluable boundary required"
        )
    if bool(decision.get("critical_user_input_required", True)):
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairSweepError(
            "critical user input should not be required"
        )
    if require_no_quality_claim:
        _require_no_quality_claim(readiness, label="songlike contour repair readiness")
    return {
        "boundary": boundary,
        "source_boundary": str(report.get("source_boundary") or ""),
        "source_repair_sweep_boundary": str(
            report.get("source_repair_sweep_boundary") or ""
        ),
        "next_boundary": str(decision.get("next_boundary") or ""),
        "selected_target": str(
            _dict(report.get("selected_next_target")).get("selected_target") or ""
        ),
        "songlike_melody_contour_phrase_rhythm_repair_sweep_completed": bool(
            readiness.get(
                "songlike_melody_contour_phrase_rhythm_repair_sweep_completed", False
            )
        ),
        "songlike_melody_contour_phrase_rhythm_repair_target_supported": bool(
            readiness.get(
                "songlike_melody_contour_phrase_rhythm_repair_target_supported", False
            )
        ),
        "candidate_count": len(repairs),
        "source_total_failure_label_count": _int(
            aggregate.get("source_total_failure_label_count")
        ),
        "repaired_total_failure_label_count": _int(
            aggregate.get("repaired_total_failure_label_count")
        ),
        "failure_label_delta": _int(aggregate.get("failure_label_delta")),
        "source_phrase_rhythm_failure_count": _int(
            aggregate.get("source_phrase_rhythm_failure_count")
        ),
        "repaired_phrase_rhythm_failure_count": _int(
            aggregate.get("repaired_phrase_rhythm_failure_count")
        ),
        "phrase_rhythm_failure_delta": _int(aggregate.get("phrase_rhythm_failure_delta")),
        "improved_candidate_count": _int(aggregate.get("improved_candidate_count")),
        "technical_regression_count": _int(aggregate.get("technical_regression_count")),
        "repaired_failure_counts": _dict(aggregate.get("repaired_failure_counts")),
        "source_outside_soloing_repair_evidence_ready": bool(
            aggregate.get("source_outside_soloing_repair_evidence_ready", False)
        ),
        "source_outside_soloing_repair_pitch_role_risk_count_after": _int(
            aggregate.get("source_outside_soloing_repair_pitch_role_risk_count_after")
        ),
        "source_outside_soloing_repair_source_objective_pitch_role_risk_count": _int(
            aggregate.get("source_outside_soloing_repair_source_objective_pitch_role_risk_count")
        ),
        "objective_source_outside_soloing_repair_source_context_preserved": bool(
            aggregate.get("objective_source_outside_soloing_repair_source_context_preserved", False)
        ),
        **{f"objective_{key}": aggregate.get(f"objective_{key}") for key in BRIDGE_SOURCE_CONTEXT_KEYS},
        "source_outside_soloing_repair_source_context_preserved": bool(
            aggregate.get("source_outside_soloing_repair_source_context_preserved", False)
        ),
        "source_outside_soloing_repair_source_pitch_role_risk_count_before": _int(
            aggregate.get("source_outside_soloing_repair_source_pitch_role_risk_count_before")
        ),
        "source_outside_soloing_repair_source_pitch_role_risk_count_after": _int(
            aggregate.get("source_outside_soloing_repair_source_pitch_role_risk_count_after")
        ),
        "source_outside_soloing_repair_source_pitch_role_risk_delta": _int(
            aggregate.get("source_outside_soloing_repair_source_pitch_role_risk_delta")
        ),
        "source_outside_soloing_repair_source_targeted": bool(
            aggregate.get("source_outside_soloing_repair_source_targeted", True)
        ),
        "source_outside_soloing_repair_source_residual_risk_preserved": bool(
            aggregate.get("source_outside_soloing_repair_source_residual_risk_preserved", False)
        ),
        "source_outside_soloing_repair_pitch_role_risk_delta": _int(
            aggregate.get("source_outside_soloing_repair_pitch_role_risk_delta")
        ),
        **{key: aggregate.get(key) for key in BRIDGE_SOURCE_CONTEXT_KEYS},
        "source_outside_soloing_not_evaluable_count": _int(
            aggregate.get("source_outside_soloing_not_evaluable_count")
        ),
        "repaired_outside_soloing_not_evaluable_count": _int(
            aggregate.get("repaired_outside_soloing_not_evaluable_count")
        ),
        "repaired_not_evaluable_counts": _dict(
            aggregate.get("repaired_not_evaluable_counts")
        ),
        "audio_package_ready": bool(readiness.get("audio_package_ready", False)),
        "human_audio_preference_claimed": bool(
            readiness.get("human_audio_preference_claimed", True)
        ),
        "midi_to_solo_musical_quality_claimed": bool(
            readiness.get("midi_to_solo_musical_quality_claimed", True)
        ),
        "critical_user_input_required": bool(
            decision.get("critical_user_input_required", True)
        ),
        "next_recommended_issue": str(report.get("next_recommended_issue") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    aggregate = report["aggregate"]
    readiness = report["readiness"]
    decision = report["decision"]
    selected = report["selected_next_target"]
    lines = [
        "# Stage B MIDI-to-Solo Songlike Melody Contour Phrase/Rhythm Repair Sweep",
        "",
        "## Summary",
        "",
        f"- boundary: `{report['boundary']}`",
        f"- source boundary: `{report['source_boundary']}`",
        f"- next boundary: `{decision['next_boundary']}`",
        f"- selected target: `{selected['selected_target']}`",
        f"- candidate count: `{aggregate['candidate_count']}`",
        f"- total failure labels: `{aggregate['source_total_failure_label_count']} -> {aggregate['repaired_total_failure_label_count']}`",
        f"- failure label delta: `{aggregate['failure_label_delta']}`",
        f"- phrase/rhythm failure count: `{aggregate['source_phrase_rhythm_failure_count']} -> {aggregate['repaired_phrase_rhythm_failure_count']}`",
        f"- phrase/rhythm failure delta: `{aggregate['phrase_rhythm_failure_delta']}`",
        f"- improved candidate count: `{aggregate['improved_candidate_count']}`",
        f"- technical regression count: `{aggregate['technical_regression_count']}`",
        f"- source outside-soloing repair evidence ready: `{_bool_token(aggregate['source_outside_soloing_repair_evidence_ready'])}`",
        f"- objective source outside-soloing source context preserved: `{_bool_token(aggregate['objective_source_outside_soloing_repair_source_context_preserved'])}`",
        f"- source outside-soloing source context preserved: `{_bool_token(aggregate['source_outside_soloing_repair_source_context_preserved'])}`",
        f"- source outside-soloing source pitch-role risk before / after / delta: `{aggregate['source_outside_soloing_repair_source_pitch_role_risk_count_before']}` / `{aggregate['source_outside_soloing_repair_source_pitch_role_risk_count_after']}` / `{aggregate['source_outside_soloing_repair_source_pitch_role_risk_delta']}`",
        f"- source outside-soloing source repair targeted: `{_bool_token(aggregate['source_outside_soloing_repair_source_targeted'])}`",
        f"- source outside-soloing source residual risk preserved: `{_bool_token(aggregate['source_outside_soloing_repair_source_residual_risk_preserved'])}`",
        f"- source outside-soloing repair pitch-role risk after: `{aggregate['source_outside_soloing_repair_pitch_role_risk_count_after']}`",
        f"- source outside-soloing current repair pitch-role risk delta: `{aggregate['source_outside_soloing_repair_pitch_role_risk_delta']}`",
        f"- source outside-soloing not evaluable count: `{aggregate['source_outside_soloing_not_evaluable_count']}`",
        f"- repaired outside-soloing not evaluable count: `{aggregate['repaired_outside_soloing_not_evaluable_count']}`",
        f"- target supported: `{_bool_token(aggregate['target_supported'])}`",
        f"- human/audio preference claimed: `{_bool_token(readiness['human_audio_preference_claimed'])}`",
        f"- MIDI-to-solo musical quality claimed: `{_bool_token(readiness['midi_to_solo_musical_quality_claimed'])}`",
        "",
        "## Repaired Failure Counts",
        "",
    ]
    for label, count in aggregate["repaired_failure_counts"].items():
        lines.append(f"- `{label}`: `{count}`")
    lines.extend(["", "## Repaired Not Evaluable Counts", ""])
    for label, count in aggregate["repaired_not_evaluable_counts"].items():
        lines.append(f"- `{label}`: `{count}`")
    lines.extend(["", "## MIDI Files", ""])
    for item in report.get("candidate_repairs", []):
        lines.append(
            f"- rank `{item['rank']}`: `{item['phrase_rhythm_repaired_midi_path']}`"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- reason: `{decision['reason']}`",
            f"- auto progress allowed: `{_bool_token(decision['auto_progress_allowed'])}`",
            f"- critical user input required: `{_bool_token(decision['critical_user_input_required'])}`",
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
        description="Run songlike melody contour phrase/rhythm repair sweep"
    )
    parser.add_argument("--followup_decision", type=str, required=True)
    parser.add_argument("--source_repair_sweep", type=str, required=True)
    parser.add_argument("--rubric_baseline", type=str, required=True)
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_repair_sweep",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--issue_number", type=int, default=944)
    parser.add_argument("--expected_boundary", type=str, default="")
    parser.add_argument("--expected_next_boundary", type=str, default="")
    parser.add_argument("--min_candidate_count", type=int, default=6)
    parser.add_argument("--require_sweep_completed", action="store_true")
    parser.add_argument("--require_target_supported", action="store_true")
    parser.add_argument("--require_phrase_rhythm_delta", action="store_true")
    parser.add_argument("--require_no_quality_claim", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    report = build_songlike_melody_contour_phrase_rhythm_repair_sweep_report(
        followup_decision=read_json(Path(args.followup_decision)),
        source_repair_sweep=read_json(Path(args.source_repair_sweep)),
        rubric_baseline=read_json(Path(args.rubric_baseline)),
        output_dir=output_dir,
        issue_number=int(args.issue_number),
    )
    summary = validate_songlike_melody_contour_phrase_rhythm_repair_sweep_report(
        report,
        expected_boundary=str(args.expected_boundary or ""),
        expected_next_boundary=str(args.expected_next_boundary or ""),
        min_candidate_count=int(args.min_candidate_count),
        require_sweep_completed=bool(args.require_sweep_completed),
        require_target_supported=bool(args.require_target_supported),
        require_phrase_rhythm_delta=bool(args.require_phrase_rhythm_delta),
        require_no_quality_claim=bool(args.require_no_quality_claim),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        output_dir
        / "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_repair_sweep.json",
        report,
    )
    write_json(
        output_dir
        / "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_repair_sweep_validation_summary.json",
        summary,
    )
    markdown = markdown_report(report)
    write_text(
        output_dir
        / "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_repair_sweep.md",
        markdown,
    )
    if args.doc_path:
        write_text(Path(args.doc_path), markdown)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
