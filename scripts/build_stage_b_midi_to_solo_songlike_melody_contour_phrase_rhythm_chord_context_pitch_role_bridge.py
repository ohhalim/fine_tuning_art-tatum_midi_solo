"""Build chord-context pitch-role metrics for phrase/rhythm repair candidates."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from scripts.assess_stage_b_generic_base_readiness import read_json, write_json, write_text  # noqa: E402
from scripts.decide_stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_repair_followup import (  # noqa: E402
    BOUNDARY as FOLLOWUP_BOUNDARY,
    NEXT_BOUNDARY as FOLLOWUP_NEXT_BOUNDARY,
)
from scripts.evaluate_chord_labeled_subset import (  # noqa: E402
    analyze_sample,
    load_sample_groups,
)
from scripts.run_stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_repair_sweep import (  # noqa: E402
    BOUNDARY as REPAIR_SWEEP_BOUNDARY,
)
from scripts.run_stage_b_reference_stats import pitch_role_for_group  # noqa: E402


class StageBMidiToSoloPhraseRhythmChordContextPitchRoleBridgeError(ValueError):
    pass


BOUNDARY = (
    "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_context_pitch_role_bridge"
)
NEXT_BOUNDARY = (
    "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_context_pitch_role_objective_decision"
)
SELECTED_TARGET = (
    "songlike_melody_contour_phrase_rhythm_chord_context_pitch_role_objective_decision"
)
SCHEMA_VERSION = "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_context_pitch_role_bridge_v4"
DEFAULT_CHORDS = ["Cm7", "Fm7", "Bb7", "Ebmaj7"]
CONTEXT_LABELS = ["outside_soloing_without_context", "weak_chord_tone_landing"]
SOURCE_CONTEXT_SUFFIXES = [
    "source_objective_pitch_role_risk_count",
    "source_context_preserved",
    "source_pitch_role_risk_count_before",
    "source_pitch_role_risk_count_after",
    "source_pitch_role_risk_delta",
    "source_targeted",
    "source_residual_risk_preserved",
    "pitch_role_risk_count_after",
    "pitch_role_risk_delta",
]
BRIDGE_SOURCE_CONTEXT_KEYS = [
    "followup_objective_source_outside_soloing_source_pitch_role_risk_count_before",
    "followup_objective_source_outside_soloing_source_pitch_role_risk_count_after",
    "followup_objective_source_outside_soloing_source_pitch_role_risk_delta",
    "followup_objective_source_outside_soloing_source_targeted",
    "followup_objective_source_outside_soloing_source_residual_risk_preserved",
    "followup_objective_source_outside_soloing_current_pitch_role_risk_count_after",
    "followup_objective_source_outside_soloing_current_pitch_role_risk_delta",
    "followup_repair_sweep_source_outside_soloing_source_pitch_role_risk_count_before",
    "followup_repair_sweep_source_outside_soloing_source_pitch_role_risk_count_after",
    "followup_repair_sweep_source_outside_soloing_source_pitch_role_risk_delta",
    "followup_repair_sweep_source_outside_soloing_source_targeted",
    "followup_repair_sweep_source_outside_soloing_source_residual_risk_preserved",
    "followup_repair_sweep_source_outside_soloing_current_pitch_role_risk_count_after",
    "followup_repair_sweep_source_outside_soloing_current_pitch_role_risk_delta",
    "repair_sweep_source_outside_soloing_source_pitch_role_risk_count_before",
    "repair_sweep_source_outside_soloing_source_pitch_role_risk_count_after",
    "repair_sweep_source_outside_soloing_source_pitch_role_risk_delta",
    "repair_sweep_source_outside_soloing_source_targeted",
    "repair_sweep_source_outside_soloing_source_residual_risk_preserved",
    "repair_sweep_source_outside_soloing_current_pitch_role_risk_count_after",
    "repair_sweep_source_outside_soloing_current_pitch_role_risk_delta",
]
BRIDGE_SOURCE_CONTEXT_PRESERVED_KEYS = [
    "followup_objective_source_outside_soloing_source_context_preserved",
    "followup_repair_sweep_source_outside_soloing_source_context_preserved",
    "repair_sweep_source_outside_soloing_source_context_preserved",
]
BRIDGE_REQUIRED_SOURCE_CONTEXT_KEYS = [
    *BRIDGE_SOURCE_CONTEXT_KEYS,
    *BRIDGE_SOURCE_CONTEXT_PRESERVED_KEYS,
]
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
        raise StageBMidiToSoloPhraseRhythmChordContextPitchRoleBridgeError(
            f"unexpected quality claim in {label}: {claimed}"
        )


def _source_context_fields(
    container: dict[str, Any], *, base: str, label: str
) -> dict[str, Any]:
    objective = _int(container.get(f"{base}_source_objective_pitch_role_risk_count"))
    before = _int(container.get(f"{base}_source_pitch_role_risk_count_before"))
    after = _int(container.get(f"{base}_source_pitch_role_risk_count_after"))
    delta = _int(container.get(f"{base}_source_pitch_role_risk_delta"))
    context_preserved = bool(container.get(f"{base}_source_context_preserved", False))
    source_targeted = bool(container.get(f"{base}_source_targeted", True))
    residual_preserved = bool(container.get(f"{base}_source_residual_risk_preserved", False))
    current_after = _int(container.get(f"{base}_pitch_role_risk_count_after"))
    current_delta = _int(container.get(f"{base}_pitch_role_risk_delta"))
    if objective <= 0:
        raise StageBMidiToSoloPhraseRhythmChordContextPitchRoleBridgeError(
            f"{label} source objective pitch-role risk count required"
        )
    if before != objective:
        raise StageBMidiToSoloPhraseRhythmChordContextPitchRoleBridgeError(
            f"{label} source before count must match objective risk count"
        )
    if before - after != delta:
        raise StageBMidiToSoloPhraseRhythmChordContextPitchRoleBridgeError(
            f"{label} source pitch-role risk delta mismatch"
        )
    if not context_preserved:
        raise StageBMidiToSoloPhraseRhythmChordContextPitchRoleBridgeError(
            f"{label} source context preservation required"
        )
    if source_targeted:
        raise StageBMidiToSoloPhraseRhythmChordContextPitchRoleBridgeError(
            f"{label} source should remain untargeted"
        )
    if not residual_preserved:
        raise StageBMidiToSoloPhraseRhythmChordContextPitchRoleBridgeError(
            f"{label} source residual risk should be preserved"
        )
    if current_after != 0:
        raise StageBMidiToSoloPhraseRhythmChordContextPitchRoleBridgeError(
            f"{label} current repair pitch-role risk should be resolved before bridge"
        )
    if current_delta <= 0:
        raise StageBMidiToSoloPhraseRhythmChordContextPitchRoleBridgeError(
            f"{label} current repair pitch-role risk delta required"
        )
    return {
        f"{base}_source_objective_pitch_role_risk_count": objective,
        f"{base}_source_pitch_role_risk_count_before": before,
        f"{base}_source_pitch_role_risk_count_after": after,
        f"{base}_source_pitch_role_risk_delta": delta,
        f"{base}_source_context_preserved": context_preserved,
        f"{base}_source_targeted": source_targeted,
        f"{base}_source_residual_risk_preserved": residual_preserved,
        f"{base}_pitch_role_risk_count_after": current_after,
        f"{base}_pitch_role_risk_delta": current_delta,
    }


def _validate_bridge_source_context_consistency(
    followup: dict[str, Any], sweep: dict[str, Any]
) -> None:
    for suffix in SOURCE_CONTEXT_SUFFIXES:
        followup_key = f"repair_sweep_source_outside_soloing_repair_{suffix}"
        sweep_key = f"source_outside_soloing_repair_{suffix}"
        if followup.get(followup_key) != sweep.get(sweep_key):
            raise StageBMidiToSoloPhraseRhythmChordContextPitchRoleBridgeError(
                f"repair sweep source context mismatch for {suffix}"
            )


def expand_chords(chords: list[str], bar_count: int) -> list[str]:
    if not chords:
        raise StageBMidiToSoloPhraseRhythmChordContextPitchRoleBridgeError(
            "chord progression required"
        )
    return [str(chords[index % len(chords)]) for index in range(int(bar_count))]


def parse_chords(raw: str) -> list[str]:
    chords = [item.strip() for item in str(raw or "").split(",") if item.strip()]
    return chords or list(DEFAULT_CHORDS)


def validate_followup_source(report: dict[str, Any]) -> dict[str, Any]:
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    selected = _dict(report.get("selected_next_target"))
    if str(report.get("boundary") or "") != FOLLOWUP_BOUNDARY:
        raise StageBMidiToSoloPhraseRhythmChordContextPitchRoleBridgeError(
            "phrase/rhythm repair follow-up decision boundary required"
        )
    if str(decision.get("next_boundary") or "") != FOLLOWUP_NEXT_BOUNDARY:
        raise StageBMidiToSoloPhraseRhythmChordContextPitchRoleBridgeError(
            "follow-up decision must route to chord-context pitch-role bridge"
        )
    if str(selected.get("selected_target") or "") != BOUNDARY.replace("stage_b_midi_to_solo_", ""):
        raise StageBMidiToSoloPhraseRhythmChordContextPitchRoleBridgeError(
            "selected target must be chord-context pitch-role bridge"
        )
    if not bool(readiness.get("followup_decision_completed", False)):
        raise StageBMidiToSoloPhraseRhythmChordContextPitchRoleBridgeError(
            "follow-up decision completion required"
        )
    if not bool(readiness.get("chord_context_pitch_role_bridge_selected", False)):
        raise StageBMidiToSoloPhraseRhythmChordContextPitchRoleBridgeError(
            "chord-context pitch-role bridge selection required"
        )
    if bool(decision.get("critical_user_input_required", True)):
        raise StageBMidiToSoloPhraseRhythmChordContextPitchRoleBridgeError(
            "critical user input should not be required"
        )
    if not bool(readiness.get("objective_source_outside_soloing_repair_evidence_ready", False)):
        raise StageBMidiToSoloPhraseRhythmChordContextPitchRoleBridgeError(
            "objective outside-soloing repair evidence should be ready"
        )
    if _int(readiness.get("objective_source_outside_soloing_repair_pitch_role_risk_count_after")) != 0:
        raise StageBMidiToSoloPhraseRhythmChordContextPitchRoleBridgeError(
            "objective outside-soloing repair pitch-role risk should be resolved"
        )
    if _int(readiness.get("objective_source_outside_soloing_not_evaluable_count")) <= 0:
        raise StageBMidiToSoloPhraseRhythmChordContextPitchRoleBridgeError(
            "objective outside-soloing not-evaluable count should be preserved"
        )
    if _int(readiness.get("repair_sweep_source_outside_soloing_not_evaluable_count")) <= 0:
        raise StageBMidiToSoloPhraseRhythmChordContextPitchRoleBridgeError(
            "repair sweep outside-soloing not-evaluable count should be preserved"
        )
    _require_no_quality_claim(readiness, label="follow-up readiness")
    objective_source_context = _source_context_fields(
        readiness,
        base="objective_source_outside_soloing_repair",
        label="follow-up objective outside-soloing repair",
    )
    repair_sweep_source_context = _source_context_fields(
        readiness,
        base="repair_sweep_source_outside_soloing_repair",
        label="follow-up repair sweep outside-soloing repair",
    )
    return {
        "boundary": FOLLOWUP_BOUNDARY,
        "candidate_count": _int(readiness.get("candidate_count")),
        "failure_label_delta": _int(readiness.get("failure_label_delta")),
        "phrase_rhythm_failure_delta": _int(readiness.get("phrase_rhythm_failure_delta")),
        "context_not_evaluable_min_count": _int(
            readiness.get("context_not_evaluable_min_count")
        ),
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
        **objective_source_context,
        **repair_sweep_source_context,
    }


def repaired_labeling(row: dict[str, Any]) -> dict[str, Any]:
    return _dict(
        row.get("phrase_rhythm_repaired_labeling")
        or row.get("contour_repaired_labeling")
        or row.get("repaired_labeling")
    )


def repaired_midi_path(row: dict[str, Any]) -> str:
    value = (
        row.get("phrase_rhythm_repaired_midi_path")
        or row.get("contour_repaired_midi_path")
        or row.get("repaired_midi_path")
        or row.get("midi_path")
    )
    if not value:
        raise StageBMidiToSoloPhraseRhythmChordContextPitchRoleBridgeError(
            "candidate repaired MIDI path required"
        )
    return str(value)


def validate_repair_sweep_source(report: dict[str, Any]) -> dict[str, Any]:
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    aggregate = _dict(report.get("aggregate"))
    rows = [_dict(row) for row in _list(report.get("candidate_repairs"))]
    if str(report.get("boundary") or "") != REPAIR_SWEEP_BOUNDARY:
        raise StageBMidiToSoloPhraseRhythmChordContextPitchRoleBridgeError(
            "phrase/rhythm repair sweep boundary required"
        )
    if not bool(
        readiness.get("songlike_melody_contour_phrase_rhythm_repair_sweep_completed", False)
    ):
        raise StageBMidiToSoloPhraseRhythmChordContextPitchRoleBridgeError(
            "phrase/rhythm repair sweep completion required"
        )
    if _int(aggregate.get("candidate_count")) < 6 or len(rows) < 6:
        raise StageBMidiToSoloPhraseRhythmChordContextPitchRoleBridgeError(
            "at least 6 repair candidates required"
        )
    if _int(aggregate.get("technical_regression_count")) != 0:
        raise StageBMidiToSoloPhraseRhythmChordContextPitchRoleBridgeError(
            "technical regression must remain zero"
        )
    if _int(aggregate.get("phrase_rhythm_failure_delta")) <= 0:
        raise StageBMidiToSoloPhraseRhythmChordContextPitchRoleBridgeError(
            "positive phrase/rhythm failure delta required"
        )
    if not bool(aggregate.get("source_outside_soloing_repair_evidence_ready", False)):
        raise StageBMidiToSoloPhraseRhythmChordContextPitchRoleBridgeError(
            "repair sweep outside-soloing repair evidence should be ready"
        )
    if _int(aggregate.get("source_outside_soloing_repair_pitch_role_risk_count_after")) != 0:
        raise StageBMidiToSoloPhraseRhythmChordContextPitchRoleBridgeError(
            "repair sweep outside-soloing repair pitch-role risk should be resolved"
        )
    if _int(aggregate.get("source_outside_soloing_not_evaluable_count")) <= 0:
        raise StageBMidiToSoloPhraseRhythmChordContextPitchRoleBridgeError(
            "repair sweep source outside-soloing not-evaluable count should be preserved"
        )
    if _int(aggregate.get("repaired_outside_soloing_not_evaluable_count")) <= 0:
        raise StageBMidiToSoloPhraseRhythmChordContextPitchRoleBridgeError(
            "repair sweep repaired outside-soloing not-evaluable count should be preserved"
        )
    if bool(decision.get("critical_user_input_required", True)):
        raise StageBMidiToSoloPhraseRhythmChordContextPitchRoleBridgeError(
            "critical user input should not be required"
        )
    _require_no_quality_claim(readiness, label="repair sweep readiness")
    source_context = _source_context_fields(
        aggregate,
        base="source_outside_soloing_repair",
        label="repair sweep outside-soloing repair",
    )
    not_evaluable_counts = Counter()
    for row in rows:
        labels = _list(repaired_labeling(row).get("not_evaluable_labels"))
        not_evaluable_counts.update(str(label) for label in labels)
    if any(_int(not_evaluable_counts.get(label)) < len(rows) for label in CONTEXT_LABELS):
        raise StageBMidiToSoloPhraseRhythmChordContextPitchRoleBridgeError(
            "context not-evaluable labels must cover all candidates before bridge"
        )
    return {
        "boundary": REPAIR_SWEEP_BOUNDARY,
        "candidate_count": len(rows),
        "rows": rows,
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
        "technical_regression_count": _int(aggregate.get("technical_regression_count")),
        "source_outside_soloing_repair_evidence_ready": bool(
            aggregate.get("source_outside_soloing_repair_evidence_ready", False)
        ),
        "source_outside_soloing_repair_pitch_role_risk_count_after": _int(
            aggregate.get("source_outside_soloing_repair_pitch_role_risk_count_after")
        ),
        "source_outside_soloing_not_evaluable_count": _int(
            aggregate.get("source_outside_soloing_not_evaluable_count")
        ),
        "repaired_outside_soloing_not_evaluable_count": _int(
            aggregate.get("repaired_outside_soloing_not_evaluable_count")
        ),
        "repaired_not_evaluable_counts": _dict(aggregate.get("repaired_not_evaluable_counts")),
        "not_evaluable_counts_before": dict(sorted(not_evaluable_counts.items())),
        **source_context,
    }


def strong_beat_chord_tone_ratio(analysis: dict[str, Any]) -> float:
    strong_counts = _dict(_dict(analysis.get("bucket_counts")).get("strong"))
    total = sum(_int(strong_counts.get(role)) for role in strong_counts)
    if total <= 0:
        return 0.0
    chord_hits = sum(_int(strong_counts.get(role)) for role in ("root", "guide", "chord"))
    return float(chord_hits / total)


def max_non_chord_tone_run(groups: list[dict[str, int]], chords: list[str]) -> int:
    current = 0
    maximum = 0
    for group in groups:
        role = pitch_role_for_group(group, chords[int(group["bar"])])
        if role in {"approach", "outside", "unknown_chord"}:
            current += 1
            maximum = max(maximum, current)
        else:
            current = 0
    return int(maximum)


def final_landing_role(groups: list[dict[str, int]], chords: list[str]) -> str:
    if not groups:
        return "unknown_chord"
    final = groups[-1]
    return pitch_role_for_group(final, chords[int(final["bar"])])


def bridge_candidate(
    *,
    row: dict[str, Any],
    rank: int,
    chords: list[str],
    bpm: float,
    manifest_path: Path,
) -> dict[str, Any]:
    labeling = repaired_labeling(row)
    metrics = _dict(labeling.get("metrics"))
    bar_count = _int(metrics.get("bar_count")) or 8
    expanded_chords = expand_chords(chords, bar_count)
    midi_path = repaired_midi_path(row)
    sample = {
        "sample_id": f"rank_{_int(row.get('rank')) or rank}",
        "bar_count": bar_count,
        "bpm": float(bpm),
        "chords": expanded_chords,
        "midi_path": midi_path,
    }
    analysis = analyze_sample(sample, manifest_path=manifest_path)
    groups = load_sample_groups(sample, manifest_path=manifest_path)
    role_ratios = _dict(analysis.get("role_ratios"))
    strong_ratio = strong_beat_chord_tone_ratio(analysis)
    final_role = final_landing_role(groups, expanded_chords)
    landing_chord_tone = final_role in {"root", "guide", "chord"}
    non_chord_run = max_non_chord_tone_run(groups, expanded_chords)
    old_not_evaluable = [str(label) for label in _list(labeling.get("not_evaluable_labels"))]
    remaining_not_evaluable = [
        label for label in old_not_evaluable if label not in set(CONTEXT_LABELS)
    ]
    bridge_flags: list[str] = []
    if _float(role_ratios.get("outside_ratio")) >= 0.35 or non_chord_run >= 4:
        bridge_flags.append("outside_soloing_pitch_role_risk")
    if strong_ratio < 0.40 or not landing_chord_tone:
        bridge_flags.append("weak_chord_tone_landing_risk")
    updated_metrics = {
        **metrics,
        "chord_context_available": True,
        "chord_progression": expanded_chords,
        "pitch_role_metrics_defined": True,
        "strong_beat_chord_tone_ratio": strong_ratio,
        "cadence_landing_chord_tone": landing_chord_tone,
        "cadence_landing_role": final_role,
        "max_non_chord_tone_run": non_chord_run,
        "chord_tone_ratio": _float(role_ratios.get("chord_tone_ratio")),
        "tension_ratio": _float(role_ratios.get("tension_ratio")),
        "approach_ratio": _float(role_ratios.get("approach_ratio")),
        "outside_ratio": _float(role_ratios.get("outside_ratio")),
    }
    return {
        "rank": _int(row.get("rank")) or rank,
        "midi_path": midi_path,
        "sample": sample,
        "analysis": analysis,
        "bridge_metrics": updated_metrics,
        "not_evaluable_before": old_not_evaluable,
        "not_evaluable_after": remaining_not_evaluable,
        "bridge_flags": bridge_flags,
    }


def build_bridge_report(
    *,
    followup_report: dict[str, Any],
    repair_sweep_report: dict[str, Any],
    chords: list[str],
    bpm: float,
    output_dir: Path,
    issue_number: int,
) -> dict[str, Any]:
    followup = validate_followup_source(followup_report)
    sweep = validate_repair_sweep_source(repair_sweep_report)
    _validate_bridge_source_context_consistency(followup, sweep)
    manifest_path = ROOT_DIR / "stage_b_midi_to_solo_phrase_rhythm_bridge_manifest.json"
    candidates = [
        bridge_candidate(
            row=row,
            rank=index,
            chords=chords,
            bpm=bpm,
            manifest_path=manifest_path,
        )
        for index, row in enumerate(sweep["rows"], start=1)
    ]
    sample_analyses = [candidate["analysis"] for candidate in candidates]
    chord_context_available_count = sum(
        1 for candidate in candidates if candidate["bridge_metrics"]["chord_context_available"]
    )
    pitch_role_metrics_defined_count = sum(
        1 for candidate in candidates if candidate["bridge_metrics"]["pitch_role_metrics_defined"]
    )
    not_evaluable_before_count = sum(
        len(candidate["not_evaluable_before"]) for candidate in candidates
    )
    not_evaluable_after_count = sum(len(candidate["not_evaluable_after"]) for candidate in candidates)
    bridge_flag_counts = Counter(
        flag for candidate in candidates for flag in _list(candidate.get("bridge_flags"))
    )
    min_chord_tone_ratio = min(
        _float(candidate["bridge_metrics"].get("chord_tone_ratio")) for candidate in candidates
    )
    max_outside_ratio = max(
        _float(candidate["bridge_metrics"].get("outside_ratio")) for candidate in candidates
    )
    max_non_chord_run = max(
        _int(candidate["bridge_metrics"].get("max_non_chord_tone_run"))
        for candidate in candidates
    )
    summary = {
        "candidate_count": len(candidates),
        "chord_context_available_count": chord_context_available_count,
        "pitch_role_metrics_defined_count": pitch_role_metrics_defined_count,
        "not_evaluable_before_count": not_evaluable_before_count,
        "not_evaluable_after_count": not_evaluable_after_count,
        "not_evaluable_counts_before": sweep["not_evaluable_counts_before"],
        "followup_objective_source_outside_soloing_not_evaluable_count": _int(
            followup["objective_source_outside_soloing_not_evaluable_count"]
        ),
        "followup_objective_repaired_outside_soloing_not_evaluable_count": _int(
            followup["objective_repaired_outside_soloing_not_evaluable_count"]
        ),
        "followup_repair_sweep_source_outside_soloing_not_evaluable_count": _int(
            followup["repair_sweep_source_outside_soloing_not_evaluable_count"]
        ),
        "followup_repair_sweep_repaired_outside_soloing_not_evaluable_count": _int(
            followup["repair_sweep_repaired_outside_soloing_not_evaluable_count"]
        ),
        "repair_sweep_source_outside_soloing_not_evaluable_count": _int(
            sweep["source_outside_soloing_not_evaluable_count"]
        ),
        "repair_sweep_repaired_outside_soloing_not_evaluable_count": _int(
            sweep["repaired_outside_soloing_not_evaluable_count"]
        ),
        "followup_objective_source_outside_soloing_source_pitch_role_risk_count_before": _int(
            followup[
                "objective_source_outside_soloing_repair_source_pitch_role_risk_count_before"
            ]
        ),
        "followup_objective_source_outside_soloing_source_pitch_role_risk_count_after": _int(
            followup[
                "objective_source_outside_soloing_repair_source_pitch_role_risk_count_after"
            ]
        ),
        "followup_objective_source_outside_soloing_source_pitch_role_risk_delta": _int(
            followup["objective_source_outside_soloing_repair_source_pitch_role_risk_delta"]
        ),
        "followup_objective_source_outside_soloing_source_context_preserved": bool(
            followup["objective_source_outside_soloing_repair_source_context_preserved"]
        ),
        "followup_objective_source_outside_soloing_source_targeted": bool(
            followup["objective_source_outside_soloing_repair_source_targeted"]
        ),
        "followup_objective_source_outside_soloing_source_residual_risk_preserved": bool(
            followup[
                "objective_source_outside_soloing_repair_source_residual_risk_preserved"
            ]
        ),
        "followup_objective_source_outside_soloing_current_pitch_role_risk_count_after": _int(
            followup["objective_source_outside_soloing_repair_pitch_role_risk_count_after"]
        ),
        "followup_objective_source_outside_soloing_current_pitch_role_risk_delta": _int(
            followup["objective_source_outside_soloing_repair_pitch_role_risk_delta"]
        ),
        "followup_repair_sweep_source_outside_soloing_source_pitch_role_risk_count_before": _int(
            followup[
                "repair_sweep_source_outside_soloing_repair_source_pitch_role_risk_count_before"
            ]
        ),
        "followup_repair_sweep_source_outside_soloing_source_pitch_role_risk_count_after": _int(
            followup[
                "repair_sweep_source_outside_soloing_repair_source_pitch_role_risk_count_after"
            ]
        ),
        "followup_repair_sweep_source_outside_soloing_source_pitch_role_risk_delta": _int(
            followup["repair_sweep_source_outside_soloing_repair_source_pitch_role_risk_delta"]
        ),
        "followup_repair_sweep_source_outside_soloing_source_context_preserved": bool(
            followup["repair_sweep_source_outside_soloing_repair_source_context_preserved"]
        ),
        "followup_repair_sweep_source_outside_soloing_source_targeted": bool(
            followup["repair_sweep_source_outside_soloing_repair_source_targeted"]
        ),
        "followup_repair_sweep_source_outside_soloing_source_residual_risk_preserved": bool(
            followup[
                "repair_sweep_source_outside_soloing_repair_source_residual_risk_preserved"
            ]
        ),
        "followup_repair_sweep_source_outside_soloing_current_pitch_role_risk_count_after": _int(
            followup["repair_sweep_source_outside_soloing_repair_pitch_role_risk_count_after"]
        ),
        "followup_repair_sweep_source_outside_soloing_current_pitch_role_risk_delta": _int(
            followup["repair_sweep_source_outside_soloing_repair_pitch_role_risk_delta"]
        ),
        "repair_sweep_source_outside_soloing_source_pitch_role_risk_count_before": _int(
            sweep["source_outside_soloing_repair_source_pitch_role_risk_count_before"]
        ),
        "repair_sweep_source_outside_soloing_source_pitch_role_risk_count_after": _int(
            sweep["source_outside_soloing_repair_source_pitch_role_risk_count_after"]
        ),
        "repair_sweep_source_outside_soloing_source_pitch_role_risk_delta": _int(
            sweep["source_outside_soloing_repair_source_pitch_role_risk_delta"]
        ),
        "repair_sweep_source_outside_soloing_source_context_preserved": bool(
            sweep["source_outside_soloing_repair_source_context_preserved"]
        ),
        "repair_sweep_source_outside_soloing_source_targeted": bool(
            sweep["source_outside_soloing_repair_source_targeted"]
        ),
        "repair_sweep_source_outside_soloing_source_residual_risk_preserved": bool(
            sweep["source_outside_soloing_repair_source_residual_risk_preserved"]
        ),
        "repair_sweep_source_outside_soloing_current_pitch_role_risk_count_after": _int(
            sweep["source_outside_soloing_repair_pitch_role_risk_count_after"]
        ),
        "repair_sweep_source_outside_soloing_current_pitch_role_risk_delta": _int(
            sweep["source_outside_soloing_repair_pitch_role_risk_delta"]
        ),
        "bridge_flag_counts": dict(sorted(bridge_flag_counts.items())),
        "min_chord_tone_ratio": min_chord_tone_ratio,
        "max_outside_ratio": max_outside_ratio,
        "max_non_chord_tone_run": max_non_chord_run,
        "role_summary": summarize_role_samples(sample_analyses),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "issue_number": int(issue_number),
        "boundary": BOUNDARY,
        "source_boundary": followup["boundary"],
        "repair_sweep_boundary": sweep["boundary"],
        "context": {
            "chord_progression": list(chords),
            "expanded_bar_count": max(_int(candidate["sample"]["bar_count"]) for candidate in candidates),
            "bpm": float(bpm),
            "context_source": "fallback_default_harness_chords",
        },
        "source_summary": {
            "followup": followup,
            "repair_sweep": {
                key: value for key, value in sweep.items() if key != "rows"
            },
        },
        "contextualized_candidates": candidates,
        "summary": summary,
        "readiness": {
            "boundary": BOUNDARY,
            "chord_context_pitch_role_bridge_completed": True,
            "candidate_count": len(candidates),
            "chord_context_available_count": chord_context_available_count,
            "pitch_role_metrics_defined_count": pitch_role_metrics_defined_count,
            "not_evaluable_before_count": not_evaluable_before_count,
            "not_evaluable_after_count": not_evaluable_after_count,
            "followup_objective_source_outside_soloing_not_evaluable_count": _int(
                followup["objective_source_outside_soloing_not_evaluable_count"]
            ),
            "followup_objective_repaired_outside_soloing_not_evaluable_count": _int(
                followup["objective_repaired_outside_soloing_not_evaluable_count"]
            ),
            "followup_repair_sweep_source_outside_soloing_not_evaluable_count": _int(
                followup["repair_sweep_source_outside_soloing_not_evaluable_count"]
            ),
            "followup_repair_sweep_repaired_outside_soloing_not_evaluable_count": _int(
                followup["repair_sweep_repaired_outside_soloing_not_evaluable_count"]
            ),
            "repair_sweep_source_outside_soloing_not_evaluable_count": _int(
                sweep["source_outside_soloing_not_evaluable_count"]
            ),
            "repair_sweep_repaired_outside_soloing_not_evaluable_count": _int(
                sweep["repaired_outside_soloing_not_evaluable_count"]
            ),
            **{key: summary[key] for key in BRIDGE_REQUIRED_SOURCE_CONTEXT_KEYS},
            "human_audio_preference_claimed": False,
            "audio_rendered_quality_claimed": False,
            "midi_to_solo_musical_quality_claimed": False,
            "model_checkpoint_generation_quality_claimed": False,
            "model_direct_generation_quality_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
            "production_ready_claimed": False,
        },
        "decision": {
            "current_boundary": BOUNDARY,
            "next_boundary": NEXT_BOUNDARY,
            "selected_target": SELECTED_TARGET,
            "auto_progress_allowed": True,
            "critical_user_input_required": False,
            "reason": "chord-context bridge produced pitch-role metrics without quality claim",
        },
        "not_proven": [
            "human_audio_preference",
            "midi_to_solo_musical_quality",
            "audio_rendered_quality",
            "broad_trained_model_quality",
            "brad_style_adaptation",
        ],
        "next_recommended_issue": (
            "Stage B MIDI-to-solo songlike melody contour phrase/rhythm chord-context pitch-role objective decision source-context refresh"
        ),
    }


def summarize_role_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    role_counts: Counter[str] = Counter()
    note_count = 0
    for sample in samples:
        note_count += _int(sample.get("note_count"))
        role_counts.update(
            {str(role): _int(count) for role, count in _dict(sample.get("role_counts")).items()}
        )
    total = max(1, note_count)
    chord_tone = (
        _int(role_counts.get("root"))
        + _int(role_counts.get("guide"))
        + _int(role_counts.get("chord"))
    )
    return {
        "sample_count": len(samples),
        "note_count": note_count,
        "role_counts": dict(sorted(role_counts.items())),
        "chord_tone_ratio": float(chord_tone / total),
        "tension_ratio": float(_int(role_counts.get("tension")) / total),
        "approach_ratio": float(_int(role_counts.get("approach")) / total),
        "outside_ratio": float(_int(role_counts.get("outside")) / total),
    }


def validate_bridge_report(
    report: dict[str, Any],
    *,
    expected_boundary: str | None,
    expected_next_boundary: str | None,
    require_bridge_completed: bool,
    require_context_available: bool,
    require_no_quality_claim: bool,
) -> dict[str, Any]:
    boundary = str(report.get("boundary") or "")
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    summary = _dict(report.get("summary"))
    if expected_boundary and boundary != expected_boundary:
        raise StageBMidiToSoloPhraseRhythmChordContextPitchRoleBridgeError(
            f"expected boundary {expected_boundary}, got {boundary}"
        )
    if expected_next_boundary and str(decision.get("next_boundary") or "") != expected_next_boundary:
        raise StageBMidiToSoloPhraseRhythmChordContextPitchRoleBridgeError(
            "unexpected next boundary"
        )
    if require_bridge_completed and not bool(
        readiness.get("chord_context_pitch_role_bridge_completed", False)
    ):
        raise StageBMidiToSoloPhraseRhythmChordContextPitchRoleBridgeError(
            "bridge completion required"
        )
    candidate_count = _int(readiness.get("candidate_count"))
    if candidate_count <= 0:
        raise StageBMidiToSoloPhraseRhythmChordContextPitchRoleBridgeError(
            "candidate count required"
        )
    if require_context_available:
        if _int(readiness.get("chord_context_available_count")) != candidate_count:
            raise StageBMidiToSoloPhraseRhythmChordContextPitchRoleBridgeError(
                "all candidates must have chord context"
            )
        if _int(readiness.get("pitch_role_metrics_defined_count")) != candidate_count:
            raise StageBMidiToSoloPhraseRhythmChordContextPitchRoleBridgeError(
                "all candidates must have pitch-role metrics"
            )
        if _int(readiness.get("not_evaluable_after_count")) != 0:
            raise StageBMidiToSoloPhraseRhythmChordContextPitchRoleBridgeError(
                "context not-evaluable labels must be cleared"
            )
    if bool(decision.get("critical_user_input_required", True)):
        raise StageBMidiToSoloPhraseRhythmChordContextPitchRoleBridgeError(
            "critical user input should not be required"
        )
    if require_no_quality_claim:
        _require_no_quality_claim(readiness, label="bridge readiness")
    for key in BRIDGE_REQUIRED_SOURCE_CONTEXT_KEYS:
        if key not in readiness:
            raise StageBMidiToSoloPhraseRhythmChordContextPitchRoleBridgeError(
                f"bridge source-context field required: {key}"
            )
    missing_preserved = [
        key for key in BRIDGE_SOURCE_CONTEXT_PRESERVED_KEYS if not bool(readiness.get(key))
    ]
    if missing_preserved:
        raise StageBMidiToSoloPhraseRhythmChordContextPitchRoleBridgeError(
            f"bridge source-context preserved field must be true: {missing_preserved}"
        )
    return {
        "boundary": boundary,
        "source_boundary": str(report.get("source_boundary") or ""),
        "repair_sweep_boundary": str(report.get("repair_sweep_boundary") or ""),
        "next_boundary": str(decision.get("next_boundary") or ""),
        "selected_target": str(decision.get("selected_target") or ""),
        "chord_context_pitch_role_bridge_completed": bool(
            readiness.get("chord_context_pitch_role_bridge_completed", False)
        ),
        "candidate_count": candidate_count,
        "chord_context_available_count": _int(
            readiness.get("chord_context_available_count")
        ),
        "pitch_role_metrics_defined_count": _int(
            readiness.get("pitch_role_metrics_defined_count")
        ),
        "not_evaluable_before_count": _int(readiness.get("not_evaluable_before_count")),
        "not_evaluable_after_count": _int(readiness.get("not_evaluable_after_count")),
        "followup_objective_source_outside_soloing_not_evaluable_count": _int(
            readiness.get("followup_objective_source_outside_soloing_not_evaluable_count")
        ),
        "followup_objective_repaired_outside_soloing_not_evaluable_count": _int(
            readiness.get("followup_objective_repaired_outside_soloing_not_evaluable_count")
        ),
        "followup_repair_sweep_source_outside_soloing_not_evaluable_count": _int(
            readiness.get("followup_repair_sweep_source_outside_soloing_not_evaluable_count")
        ),
        "followup_repair_sweep_repaired_outside_soloing_not_evaluable_count": _int(
            readiness.get("followup_repair_sweep_repaired_outside_soloing_not_evaluable_count")
        ),
        "repair_sweep_source_outside_soloing_not_evaluable_count": _int(
            readiness.get("repair_sweep_source_outside_soloing_not_evaluable_count")
        ),
        "repair_sweep_repaired_outside_soloing_not_evaluable_count": _int(
            readiness.get("repair_sweep_repaired_outside_soloing_not_evaluable_count")
        ),
        "min_chord_tone_ratio": _float(summary.get("min_chord_tone_ratio")),
        "max_outside_ratio": _float(summary.get("max_outside_ratio")),
        "max_non_chord_tone_run": _int(summary.get("max_non_chord_tone_run")),
        "bridge_flag_counts": _dict(summary.get("bridge_flag_counts")),
        **{key: readiness.get(key) for key in BRIDGE_REQUIRED_SOURCE_CONTEXT_KEYS},
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
    summary = report["summary"]
    readiness = report["readiness"]
    decision = report["decision"]
    context = report["context"]
    lines = [
        "# Stage B MIDI-to-Solo Phrase/Rhythm Chord-Context Pitch-Role Bridge",
        "",
        "## Summary",
        "",
        f"- boundary: `{report['boundary']}`",
        f"- source boundary: `{report['source_boundary']}`",
        f"- repair sweep boundary: `{report['repair_sweep_boundary']}`",
        f"- next boundary: `{decision['next_boundary']}`",
        f"- selected target: `{decision['selected_target']}`",
        f"- chord progression: `{','.join(context['chord_progression'])}`",
        f"- context source: `{context['context_source']}`",
        f"- candidate count: `{readiness['candidate_count']}`",
        f"- chord context available count: `{readiness['chord_context_available_count']}`",
        f"- pitch-role metrics defined count: `{readiness['pitch_role_metrics_defined_count']}`",
        f"- not evaluable count: `{readiness['not_evaluable_before_count']} -> {readiness['not_evaluable_after_count']}`",
        f"- follow-up objective source/repaired outside-soloing not evaluable count: `{readiness['followup_objective_source_outside_soloing_not_evaluable_count']}/{readiness['followup_objective_repaired_outside_soloing_not_evaluable_count']}`",
        f"- follow-up repair sweep source/repaired outside-soloing not evaluable count: `{readiness['followup_repair_sweep_source_outside_soloing_not_evaluable_count']}/{readiness['followup_repair_sweep_repaired_outside_soloing_not_evaluable_count']}`",
        f"- bridge repair sweep source/repaired outside-soloing not evaluable count: `{readiness['repair_sweep_source_outside_soloing_not_evaluable_count']}/{readiness['repair_sweep_repaired_outside_soloing_not_evaluable_count']}`",
        f"- follow-up objective source outside-soloing source pitch-role risk: `{readiness['followup_objective_source_outside_soloing_source_pitch_role_risk_count_before']} -> {readiness['followup_objective_source_outside_soloing_source_pitch_role_risk_count_after']}`",
        f"- follow-up objective source outside-soloing source pitch-role risk delta: `{readiness['followup_objective_source_outside_soloing_source_pitch_role_risk_delta']}`",
        f"- follow-up objective source outside-soloing source context preserved: `{_bool_token(readiness['followup_objective_source_outside_soloing_source_context_preserved'])}`",
        f"- follow-up objective source outside-soloing source targeted: `{_bool_token(readiness['followup_objective_source_outside_soloing_source_targeted'])}`",
        f"- follow-up objective source outside-soloing source residual risk preserved: `{_bool_token(readiness['followup_objective_source_outside_soloing_source_residual_risk_preserved'])}`",
        f"- follow-up objective source outside-soloing current repair pitch-role risk after/delta: `{readiness['followup_objective_source_outside_soloing_current_pitch_role_risk_count_after']} / {readiness['followup_objective_source_outside_soloing_current_pitch_role_risk_delta']}`",
        f"- follow-up repair sweep source outside-soloing source pitch-role risk: `{readiness['followup_repair_sweep_source_outside_soloing_source_pitch_role_risk_count_before']} -> {readiness['followup_repair_sweep_source_outside_soloing_source_pitch_role_risk_count_after']}`",
        f"- follow-up repair sweep source outside-soloing source pitch-role risk delta: `{readiness['followup_repair_sweep_source_outside_soloing_source_pitch_role_risk_delta']}`",
        f"- follow-up repair sweep source outside-soloing source context preserved: `{_bool_token(readiness['followup_repair_sweep_source_outside_soloing_source_context_preserved'])}`",
        f"- follow-up repair sweep source outside-soloing source targeted: `{_bool_token(readiness['followup_repair_sweep_source_outside_soloing_source_targeted'])}`",
        f"- follow-up repair sweep source outside-soloing source residual risk preserved: `{_bool_token(readiness['followup_repair_sweep_source_outside_soloing_source_residual_risk_preserved'])}`",
        f"- follow-up repair sweep source outside-soloing current repair pitch-role risk after/delta: `{readiness['followup_repair_sweep_source_outside_soloing_current_pitch_role_risk_count_after']} / {readiness['followup_repair_sweep_source_outside_soloing_current_pitch_role_risk_delta']}`",
        f"- bridge repair sweep source outside-soloing source pitch-role risk: `{readiness['repair_sweep_source_outside_soloing_source_pitch_role_risk_count_before']} -> {readiness['repair_sweep_source_outside_soloing_source_pitch_role_risk_count_after']}`",
        f"- bridge repair sweep source outside-soloing source pitch-role risk delta: `{readiness['repair_sweep_source_outside_soloing_source_pitch_role_risk_delta']}`",
        f"- bridge repair sweep source outside-soloing source context preserved: `{_bool_token(readiness['repair_sweep_source_outside_soloing_source_context_preserved'])}`",
        f"- bridge repair sweep source outside-soloing source targeted: `{_bool_token(readiness['repair_sweep_source_outside_soloing_source_targeted'])}`",
        f"- bridge repair sweep source outside-soloing source residual risk preserved: `{_bool_token(readiness['repair_sweep_source_outside_soloing_source_residual_risk_preserved'])}`",
        f"- bridge repair sweep source outside-soloing current repair pitch-role risk after/delta: `{readiness['repair_sweep_source_outside_soloing_current_pitch_role_risk_count_after']} / {readiness['repair_sweep_source_outside_soloing_current_pitch_role_risk_delta']}`",
        f"- min chord-tone ratio: `{summary['min_chord_tone_ratio']:.3f}`",
        f"- max outside ratio: `{summary['max_outside_ratio']:.3f}`",
        f"- max non-chord run: `{summary['max_non_chord_tone_run']}`",
        f"- human/audio preference claimed: `{_bool_token(readiness['human_audio_preference_claimed'])}`",
        f"- MIDI-to-solo musical quality claimed: `{_bool_token(readiness['midi_to_solo_musical_quality_claimed'])}`",
        "",
        "## Bridge Flags",
        "",
    ]
    for label, count in summary["bridge_flag_counts"].items():
        lines.append(f"- `{label}`: `{count}`")
    if not summary["bridge_flag_counts"]:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Candidates",
            "",
            "| rank | chord-tone | tension | approach | outside | strong beat chord-tone | final role | flags |",
            "|---:|---:|---:|---:|---:|---:|---|---|",
        ]
    )
    for candidate in report["contextualized_candidates"]:
        metrics = candidate["bridge_metrics"]
        lines.append(
            "| {rank} | {chord:.3f} | {tension:.3f} | {approach:.3f} | {outside:.3f} | {strong:.3f} | `{final}` | `{flags}` |".format(
                rank=candidate["rank"],
                chord=float(metrics["chord_tone_ratio"]),
                tension=float(metrics["tension_ratio"]),
                approach=float(metrics["approach_ratio"]),
                outside=float(metrics["outside_ratio"]),
                strong=float(metrics["strong_beat_chord_tone_ratio"]),
                final=str(metrics["cadence_landing_role"]),
                flags=",".join(candidate["bridge_flags"]) or "none",
            )
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
        description="Build chord-context pitch-role bridge for phrase/rhythm repair candidates"
    )
    parser.add_argument("--followup_report", type=str, required=True)
    parser.add_argument("--repair_sweep_report", type=str, required=True)
    parser.add_argument("--chords", type=str, default=",".join(DEFAULT_CHORDS))
    parser.add_argument("--bpm", type=float, default=124.0)
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_context_pitch_role_bridge",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--issue_number", type=int, default=1124)
    parser.add_argument("--expected_boundary", type=str, default="")
    parser.add_argument("--expected_next_boundary", type=str, default="")
    parser.add_argument("--require_bridge_completed", action="store_true")
    parser.add_argument("--require_context_available", action="store_true")
    parser.add_argument("--require_no_quality_claim", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    report = build_bridge_report(
        followup_report=read_json(Path(args.followup_report)),
        repair_sweep_report=read_json(Path(args.repair_sweep_report)),
        chords=parse_chords(args.chords),
        bpm=float(args.bpm),
        output_dir=output_dir,
        issue_number=int(args.issue_number),
    )
    summary = validate_bridge_report(
        report,
        expected_boundary=str(args.expected_boundary or ""),
        expected_next_boundary=str(args.expected_next_boundary or ""),
        require_bridge_completed=bool(args.require_bridge_completed),
        require_context_available=bool(args.require_context_available),
        require_no_quality_claim=bool(args.require_no_quality_claim),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        output_dir / "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_context_pitch_role_bridge.json",
        report,
    )
    write_json(
        output_dir
        / "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_context_pitch_role_bridge_validation_summary.json",
        summary,
    )
    markdown = markdown_report(report)
    write_text(
        output_dir / "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_context_pitch_role_bridge.md",
        markdown,
    )
    if args.doc_path:
        write_text(Path(args.doc_path), markdown)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
