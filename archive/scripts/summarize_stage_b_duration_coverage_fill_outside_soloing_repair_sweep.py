"""Build outside-soloing repair variants for duration/coverage repeatability sources."""

from __future__ import annotations

import argparse
import json
import numbers
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import pretty_midi

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from inference.app.fallback import (  # noqa: E402
    chord_for_time,
    chord_pitches_in_range,
    parse_chord,
    phrase_duration_sec,
)
from inference.app.metrics import compute_midi_metrics  # noqa: E402
from inference.app.schemas import GenerationRequest  # noqa: E402
from scripts.select_stage_b_margin_recovered_repair_candidate import (  # noqa: E402
    focused_solo_metrics,
    non_drum_notes,
)
from scripts.summarize_stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_repair import (  # noqa: E402
    copy_note,
    enforce_monophonic_note_ends,
    request_from_report,
    write_midi,
)
from scripts.summarize_stage_b_margin_recovered_phrase_vocabulary_repair import (  # noqa: E402
    focused_max_interval,
    float_value,
)


PREFERRED_SOLO_MIN = 48
PREFERRED_SOLO_MAX = 88
DEFAULT_REPAIR_POLICIES = (
    "chord_tone_snap",
    "guide_tone_landing",
    "contour_resolution",
)
GUIDE_INTERVALS = {3, 4, 10, 11}


class StageBDurationCoverageOutsideSoloingRepairSweepError(ValueError):
    pass


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        return float(value)
    return value


def safe_label(value: Any) -> str:
    text = str(value).lower().replace(".", "_").replace("-", "m")
    return "".join(char for char in text if char.isalnum() or char == "_") or "na"


def load_source_notes(midi_path: Path) -> list[pretty_midi.Note]:
    midi = pretty_midi.PrettyMIDI(str(midi_path))
    notes = [copy_note(note) for note in non_drum_notes(midi)]
    if not notes:
        raise StageBDurationCoverageOutsideSoloingRepairSweepError(f"source MIDI has no notes: {midi_path}")
    return sorted(notes, key=lambda note: (float(note.start), int(note.pitch), float(note.end)))


def chord_pitch_classes(request: GenerationRequest, start_sec: float) -> set[int]:
    root_pc, intervals = parse_chord(chord_for_time(request, start_sec))
    return {(root_pc + interval) % 12 for interval in intervals}


def guide_pitch_classes(request: GenerationRequest, start_sec: float) -> set[int]:
    root_pc, intervals = parse_chord(chord_for_time(request, start_sec))
    guide_intervals = [interval for interval in intervals if interval % 12 in GUIDE_INTERVALS]
    if not guide_intervals:
        guide_intervals = intervals[:1]
    return {(root_pc + interval) % 12 for interval in guide_intervals}


def pitches_for_time(
    request: GenerationRequest,
    start_sec: float,
    *,
    guide_only: bool,
) -> list[int]:
    root_pc, intervals = parse_chord(chord_for_time(request, start_sec))
    if guide_only:
        intervals = [interval for interval in intervals if interval % 12 in GUIDE_INTERVALS] or intervals[:1]
    pitches = chord_pitches_in_range(root_pc, intervals, PREFERRED_SOLO_MIN, PREFERRED_SOLO_MAX)
    if not pitches:
        pitches = chord_pitches_in_range(root_pc, [0], PREFERRED_SOLO_MIN, PREFERRED_SOLO_MAX)
    return sorted(set(int(pitch) for pitch in pitches))


def landing_note_indexes(notes: Sequence[pretty_midi.Note], request: GenerationRequest) -> set[int]:
    if not notes:
        return set()
    phrase_duration = phrase_duration_sec(request)
    segment_duration = phrase_duration / max(1, len(request.chord_progression))
    indexes: set[int] = {len(notes) - 1}
    for segment_index in range(len(request.chord_progression)):
        segment_start = segment_index * segment_duration
        segment_end = (segment_index + 1) * segment_duration
        candidates = [
            index
            for index, note in enumerate(notes)
            if float(note.start) >= segment_start and float(note.start) < segment_end
        ]
        if candidates:
            indexes.add(max(candidates, key=lambda index: float(notes[index].start)))
    return indexes


def choose_repair_pitch(
    *,
    original_pitch: int,
    pool: Sequence[int],
    previous_pitch: int | None,
    max_interval: int,
    force_move: bool,
) -> int:
    if not pool:
        return int(original_pitch)
    scored: list[tuple[bool, int, int, int, int, int]] = []
    for pitch in pool:
        interval = abs(int(pitch) - int(previous_pitch)) if previous_pitch is not None else 0
        overflow = max(0, interval - int(max_interval))
        repeat_penalty = 1 if previous_pitch is not None and int(pitch) == int(previous_pitch) else 0
        original_distance = abs(int(pitch) - int(original_pitch))
        if force_move and int(pitch) == int(original_pitch):
            original_distance += 4
        scored.append(
            (
                overflow > 0,
                overflow,
                repeat_penalty,
                original_distance,
                interval,
                int(pitch),
            )
        )
    scored.sort()
    return int(scored[0][-1])


def build_repaired_notes(
    source_notes: Sequence[pretty_midi.Note],
    *,
    request: GenerationRequest,
    policy: str,
    max_interval: int,
) -> list[pretty_midi.Note]:
    ordered = [copy_note(note) for note in sorted(source_notes, key=lambda note: (float(note.start), int(note.pitch)))]
    phrase_duration = phrase_duration_sec(request)
    landing_indexes = landing_note_indexes(ordered, request)
    repaired: list[pretty_midi.Note] = []
    previous_pitch: int | None = None

    for index, note in enumerate(ordered):
        original_pitch = int(note.pitch)
        is_landing = index in landing_indexes
        original_is_chord_tone = original_pitch % 12 in chord_pitch_classes(request, float(note.start))
        force_guide = policy in {"guide_tone_landing", "contour_resolution"} and is_landing
        guide_only = bool(force_guide)
        pool = pitches_for_time(request, float(note.start), guide_only=guide_only)
        keep_original = (
            original_is_chord_tone
            and not force_guide
            and (
                previous_pitch is None
                or abs(int(original_pitch) - int(previous_pitch)) <= int(max_interval)
            )
            and not (
                previous_pitch is not None
                and int(original_pitch) == int(previous_pitch)
                and policy == "contour_resolution"
            )
        )
        if keep_original:
            pitch = original_pitch
        else:
            pitch = choose_repair_pitch(
                original_pitch=original_pitch,
                pool=pool,
                previous_pitch=previous_pitch,
                max_interval=int(max_interval),
                force_move=policy == "contour_resolution",
            )
        repaired.append(
            pretty_midi.Note(
                velocity=int(note.velocity),
                pitch=int(pitch),
                start=float(note.start),
                end=min(float(phrase_duration), float(note.end)),
            )
        )
        previous_pitch = int(pitch)

    return enforce_monophonic_note_ends(repaired, max_duration_sec=phrase_duration)


def max_non_chord_run(notes: Sequence[pretty_midi.Note], request: GenerationRequest) -> int:
    longest = 0
    current = 0
    for note in sorted(notes, key=lambda item: (float(item.start), int(item.pitch))):
        if int(note.pitch) % 12 in chord_pitch_classes(request, float(note.start)):
            current = 0
        else:
            current += 1
            longest = max(longest, current)
    return int(longest)


def landing_metrics(notes: Sequence[pretty_midi.Note], request: GenerationRequest) -> dict[str, Any]:
    ordered = sorted(notes, key=lambda item: (float(item.start), int(item.pitch)))
    indexes = landing_note_indexes(ordered, request)
    landing_notes = [ordered[index] for index in sorted(indexes)]
    guide_count = sum(
        1
        for note in landing_notes
        if int(note.pitch) % 12 in guide_pitch_classes(request, float(note.start))
    )
    chord_count = sum(
        1
        for note in landing_notes
        if int(note.pitch) % 12 in chord_pitch_classes(request, float(note.start))
    )
    total = len(landing_notes)
    return {
        "landing_note_count": int(total),
        "guide_tone_landing_count": int(guide_count),
        "chord_tone_landing_count": int(chord_count),
        "guide_tone_landing_ratio": float(guide_count / total) if total else 0.0,
        "chord_tone_landing_ratio": float(chord_count / total) if total else 0.0,
    }


def pitch_role_metrics(midi_path: Path, request: GenerationRequest) -> dict[str, Any]:
    midi = pretty_midi.PrettyMIDI(str(midi_path))
    notes = non_drum_notes(midi)
    return {
        "max_non_chord_tone_run": max_non_chord_run(notes, request),
        **landing_metrics(notes, request),
    }


def outside_soloing_gate_flags(
    variant: dict[str, Any],
    source_result: dict[str, Any],
    *,
    min_chord_tone_ratio: float,
    max_non_chord_run_length: int,
    max_interval: int,
    min_unique_pitch_count: int,
    min_note_count: int,
    max_dead_air_ratio_exclusive: float,
    max_simultaneous_notes: int,
) -> list[str]:
    metrics = _dict(variant.get("metrics"))
    focused = _dict(variant.get("focused_solo_metrics"))
    pitch_role = _dict(variant.get("pitch_role_metrics"))
    source_selected_dead_air = float(source_result.get("selected_dead_air_ratio", 1.0) or 1.0)
    source_baseline_dead_air = float(source_result.get("baseline_dead_air_ratio", 1.0) or 1.0)
    repaired_dead_air = float_value(metrics.get("dead_air_ratio"), 1.0)
    flags: list[str] = []

    if repaired_dead_air > source_selected_dead_air + 1e-9:
        flags.append("dead_air_gain_not_preserved")
    if repaired_dead_air >= source_baseline_dead_air:
        flags.append("dead_air_gain_boundary_lost")
    if repaired_dead_air >= float(max_dead_air_ratio_exclusive):
        flags.append("dead_air_gate_failed")
    if int(focused.get("focused_max_simultaneous_notes", 0) or 0) > int(max_simultaneous_notes):
        flags.append("focused_polyphony")
    if int(focused.get("focused_note_count", 0) or 0) < int(min_note_count):
        flags.append("too_sparse_for_context_review")
    if int(focused.get("focused_unique_pitch_count", 0) or 0) < int(min_unique_pitch_count):
        flags.append("low_pitch_variety")
    if int(focused.get("focused_max_interval", 0) or 0) > int(max_interval):
        flags.append("wide_interval_after_repair")
    if float_value(metrics.get("chord_tone_ratio"), 0.0) < float(min_chord_tone_ratio):
        flags.append("low_chord_tone_ratio")
    if int(pitch_role.get("max_non_chord_tone_run", 0) or 0) > int(max_non_chord_run_length):
        flags.append("non_chord_run_not_repaired")
    return flags


def outside_soloing_score(variant: dict[str, Any], *, qualified: bool) -> float:
    metrics = _dict(variant.get("metrics"))
    focused = _dict(variant.get("focused_solo_metrics"))
    pitch_role = _dict(variant.get("pitch_role_metrics"))
    score = 1000.0 if qualified else 0.0
    score += float_value(metrics.get("chord_tone_ratio"), 0.0) * 240.0
    score += float_value(pitch_role.get("guide_tone_landing_ratio"), 0.0) * 90.0
    score += (1.0 - min(1.0, float_value(metrics.get("dead_air_ratio"), 1.0))) * 80.0
    score += min(12, int(focused.get("focused_unique_pitch_count", 0) or 0)) * 7.0
    score -= int(pitch_role.get("max_non_chord_tone_run", 0) or 0) * 60.0
    score -= max(0, int(focused.get("focused_max_interval", 0) or 0) - 5) * 14.0
    score -= int(focused.get("focused_adjacent_pitch_repeats", 0) or 0) * 8.0
    score -= int(focused.get("focused_duplicated_3_note_pitch_class_chunks", 0) or 0) * 12.0
    return round(float(score), 6)


def build_variant(
    *,
    source_result: dict[str, Any],
    source_notes: Sequence[pretty_midi.Note],
    source_report: dict[str, Any],
    output_dir: Path,
    policy: str,
    max_interval: int,
    min_chord_tone_ratio: float,
    max_non_chord_run_length: int,
    min_unique_pitch_count: int,
    min_note_count: int,
    max_dead_air_ratio_exclusive: float,
    max_simultaneous_notes: int,
) -> dict[str, Any]:
    request = request_from_report(source_report)
    selected_id = str(source_result.get("selected_candidate_id") or source_result.get("source_candidate_id") or "source")
    candidate_id = f"{selected_id}_outside_repair_{safe_label(policy)}"
    repaired_notes = build_repaired_notes(
        source_notes,
        request=request,
        policy=policy,
        max_interval=int(max_interval),
    )
    midi_path = write_midi(
        repaired_notes,
        output_dir / "midi" / f"{candidate_id}.mid",
        bpm=int(request.bpm),
    )
    focused = focused_solo_metrics(midi_path)
    focused["focused_max_interval"] = focused_max_interval(midi_path)
    row: dict[str, Any] = {
        "candidate_id": candidate_id,
        "source_candidate_id": str(source_result.get("source_candidate_id") or ""),
        "source_selected_candidate_id": str(source_result.get("selected_candidate_id") or ""),
        "source_run_id": str(source_result.get("source_run_id") or ""),
        "sample_seed": int(source_result.get("sample_seed", 0) or 0),
        "repair_policy": str(policy),
        "midi_path": str(midi_path),
        "metrics": compute_midi_metrics(midi_path, 0, False, request=request).to_dict(),
        "focused_solo_metrics": focused,
        "pitch_role_metrics": pitch_role_metrics(midi_path, request),
    }
    flags = outside_soloing_gate_flags(
        row,
        source_result,
        min_chord_tone_ratio=float(min_chord_tone_ratio),
        max_non_chord_run_length=int(max_non_chord_run_length),
        max_interval=int(max_interval),
        min_unique_pitch_count=int(min_unique_pitch_count),
        min_note_count=int(min_note_count),
        max_dead_air_ratio_exclusive=float(max_dead_air_ratio_exclusive),
        max_simultaneous_notes=int(max_simultaneous_notes),
    )
    row["outside_soloing_gate"] = {
        "qualified": not flags,
        "flags": flags,
    }
    row["outside_soloing_score"] = outside_soloing_score(row, qualified=not flags)
    return _json_safe(row)


def select_variant(variants: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = [dict(row) for row in variants]
    if not rows:
        raise StageBDurationCoverageOutsideSoloingRepairSweepError("no outside-soloing repair variants found")
    rows.sort(
        key=lambda row: (
            not bool(_dict(row.get("outside_soloing_gate")).get("qualified", False)),
            -float(row.get("outside_soloing_score", 0.0) or 0.0),
            -float_value(_dict(row.get("metrics")).get("chord_tone_ratio"), 0.0),
            int(_dict(row.get("pitch_role_metrics")).get("max_non_chord_tone_run", 99) or 99),
            int(_dict(row.get("focused_solo_metrics")).get("focused_max_interval", 99) or 99),
            str(row.get("repair_policy") or ""),
        )
    )
    return rows[0]


def validate_inputs(
    *,
    outside_soloing_decision: dict[str, Any],
    dead_air_gain_repair: dict[str, Any],
) -> None:
    decision = _dict(outside_soloing_decision.get("decision"))
    constraints = _dict(outside_soloing_decision.get("selection_constraints"))
    decision_claim = _dict(outside_soloing_decision.get("claim_boundary"))
    if str(decision.get("next_boundary") or "") != "outside_soloing_pitch_role_phrase_clarity_repair":
        raise StageBDurationCoverageOutsideSoloingRepairSweepError("outside-soloing repair decision boundary required")
    if not bool(decision.get("auto_progress_allowed", False)):
        raise StageBDurationCoverageOutsideSoloingRepairSweepError("auto progress is not allowed")
    if bool(decision.get("critical_user_input_required", True)):
        raise StageBDurationCoverageOutsideSoloingRepairSweepError("critical user input is still required")
    if not bool(constraints.get("keep_dead_air_gain_gate", False)):
        raise StageBDurationCoverageOutsideSoloingRepairSweepError("dead-air gain gate preservation is required")
    if not bool(constraints.get("keep_monophonic_gate", False)):
        raise StageBDurationCoverageOutsideSoloingRepairSweepError("monophonic gate preservation is required")
    if bool(decision_claim.get("broad_model_quality_claimed", True)):
        raise StageBDurationCoverageOutsideSoloingRepairSweepError("broad model quality must not be claimed")

    summary = _dict(dead_air_gain_repair.get("repair_summary"))
    claim = _dict(dead_air_gain_repair.get("claim_boundary"))
    if str(summary.get("boundary") or "") != "qualified_gate_repeatability_with_dead_air_gain":
        raise StageBDurationCoverageOutsideSoloingRepairSweepError("dead-air gain repeatability boundary required")
    if bool(claim.get("broad_model_quality_claimed", True)):
        raise StageBDurationCoverageOutsideSoloingRepairSweepError("broad model quality must not be claimed")


def build_outside_soloing_repair_sweep_report(
    *,
    outside_soloing_decision: dict[str, Any],
    dead_air_gain_repair: dict[str, Any],
    output_dir: Path,
    generation_output_root: Path,
    repair_policies: Sequence[str],
    min_source_candidates: int,
    min_repaired_source_candidates: int,
    min_chord_tone_ratio: float,
    max_non_chord_run_length: int,
    max_interval: int,
    min_unique_pitch_count: int,
    min_note_count: int,
    max_dead_air_ratio_exclusive: float,
    max_simultaneous_notes: int,
) -> dict[str, Any]:
    validate_inputs(
        outside_soloing_decision=outside_soloing_decision,
        dead_air_gain_repair=dead_air_gain_repair,
    )
    unknown_policies = sorted(set(repair_policies) - set(DEFAULT_REPAIR_POLICIES))
    if unknown_policies:
        raise StageBDurationCoverageOutsideSoloingRepairSweepError(
            f"unknown repair policies: {unknown_policies}"
        )
    source_results = [
        dict(row)
        for row in _list(dead_air_gain_repair.get("source_repeatability_results"))
        if isinstance(row, dict)
    ]
    if len(source_results) < int(min_source_candidates):
        raise StageBDurationCoverageOutsideSoloingRepairSweepError(
            f"expected at least {int(min_source_candidates)} source candidates, got {len(source_results)}"
        )

    repaired_results: list[dict[str, Any]] = []
    all_variants: list[dict[str, Any]] = []
    for index, source_result in enumerate(source_results, start=1):
        source_midi_path = Path(str(source_result.get("selected_midi_path") or ""))
        if not source_midi_path.exists():
            raise StageBDurationCoverageOutsideSoloingRepairSweepError(f"selected MIDI not found: {source_midi_path}")
        source_run_id = str(source_result.get("source_run_id") or "")
        source_report_path = generation_output_root / source_run_id / "report.json"
        if not source_report_path.exists():
            raise StageBDurationCoverageOutsideSoloingRepairSweepError(f"source report not found: {source_report_path}")
        if not bool(source_result.get("dead_air_gain_repaired", False)):
            raise StageBDurationCoverageOutsideSoloingRepairSweepError("source result must preserve dead-air gain")

        source_report = read_json(source_report_path)
        source_notes = load_source_notes(source_midi_path)
        source_output_dir = output_dir / "source_sweeps" / f"{index:02d}_{source_result.get('sample_seed', 0)}"
        variants = [
            build_variant(
                source_result=source_result,
                source_notes=source_notes,
                source_report=source_report,
                output_dir=source_output_dir,
                policy=policy,
                max_interval=int(max_interval),
                min_chord_tone_ratio=float(min_chord_tone_ratio),
                max_non_chord_run_length=int(max_non_chord_run_length),
                min_unique_pitch_count=int(min_unique_pitch_count),
                min_note_count=int(min_note_count),
                max_dead_air_ratio_exclusive=float(max_dead_air_ratio_exclusive),
                max_simultaneous_notes=int(max_simultaneous_notes),
            )
            for policy in repair_policies
        ]
        for rank, row in enumerate(
            sorted(
                variants,
                key=lambda item: (
                    not bool(_dict(item.get("outside_soloing_gate")).get("qualified", False)),
                    -float(item.get("outside_soloing_score", 0.0) or 0.0),
                    str(item.get("repair_policy") or ""),
                ),
            ),
            start=1,
        ):
            row["outside_soloing_rank"] = int(rank)
        selected = select_variant(variants)
        selected["outside_soloing_rank"] = min(
            int(row.get("outside_soloing_rank", 0) or 0)
            for row in variants
            if str(row.get("candidate_id") or "") == str(selected.get("candidate_id") or "")
        )
        all_variants.extend(variants)
        repaired_results.append(
            {
                "source_candidate_id": str(source_result.get("source_candidate_id") or ""),
                "source_selected_candidate_id": str(source_result.get("selected_candidate_id") or ""),
                "source_run_id": source_run_id,
                "source_seed": int(source_result.get("source_seed", 0) or 0),
                "sample_index": int(source_result.get("sample_index", 0) or 0),
                "sample_seed": int(source_result.get("sample_seed", 0) or 0),
                "source_selected_midi_path": str(source_midi_path),
                "source_report_path": str(source_report_path),
                "baseline_dead_air_ratio": float(source_result.get("baseline_dead_air_ratio", 1.0) or 1.0),
                "source_selected_dead_air_ratio": float(source_result.get("selected_dead_air_ratio", 1.0) or 1.0),
                "source_selected_max_interval": int(source_result.get("selected_max_interval", 0) or 0),
                "variant_count": int(len(variants)),
                "qualified_variant_count": sum(
                    1 for row in variants if bool(_dict(row.get("outside_soloing_gate")).get("qualified", False))
                ),
                "selected_candidate": selected,
                "variants": variants,
            }
        )

    selected_rows = [result["selected_candidate"] for result in repaired_results]
    repaired_source_count = sum(
        1 for row in selected_rows if bool(_dict(row.get("outside_soloing_gate")).get("qualified", False))
    )
    dead_air_preserved_source_count = sum(
        1
        for result in repaired_results
        if float_value(_dict(result["selected_candidate"].get("metrics")).get("dead_air_ratio"), 1.0)
        <= float(result.get("source_selected_dead_air_ratio", 1.0) or 1.0) + 1e-9
    )
    total_variant_count = len(all_variants)
    qualified_variant_count = sum(
        1 for row in all_variants if bool(_dict(row.get("outside_soloing_gate")).get("qualified", False))
    )
    boundary = (
        "outside_soloing_pitch_role_repair_candidates"
        if repaired_source_count >= int(min_repaired_source_candidates)
        else "outside_soloing_pitch_role_repair_not_recovered"
    )

    return _json_safe(
        {
            "schema_version": "stage_b_duration_coverage_fill_outside_soloing_repair_sweep_v1",
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
            "output_dir": str(output_dir),
            "previous_boundary": str(_dict(outside_soloing_decision.get("decision")).get("next_boundary") or ""),
            "source_dead_air_boundary": str(_dict(dead_air_gain_repair.get("repair_summary")).get("boundary") or ""),
            "repair_targets": list(outside_soloing_decision.get("repair_targets") or []),
            "repair_policies": list(repair_policies),
            "thresholds": {
                "min_source_candidates": int(min_source_candidates),
                "min_repaired_source_candidates": int(min_repaired_source_candidates),
                "min_chord_tone_ratio": float(min_chord_tone_ratio),
                "max_non_chord_run_length": int(max_non_chord_run_length),
                "max_interval": int(max_interval),
                "min_focused_unique_pitch_count": int(min_unique_pitch_count),
                "min_focused_note_count": int(min_note_count),
                "max_dead_air_ratio_exclusive": float(max_dead_air_ratio_exclusive),
                "max_focused_simultaneous_notes": int(max_simultaneous_notes),
            },
            "source_repair_results": repaired_results,
            "repair_summary": {
                "boundary": boundary,
                "source_candidate_count": int(len(repaired_results)),
                "repaired_source_candidate_count": int(repaired_source_count),
                "dead_air_preserved_source_candidate_count": int(dead_air_preserved_source_count),
                "total_variant_count": int(total_variant_count),
                "total_qualified_variant_count": int(qualified_variant_count),
                "selected_repair_policies": sorted(
                    {str(row.get("repair_policy") or "") for row in selected_rows}
                ),
                "selected_min_chord_tone_ratio": min(
                    float_value(_dict(row.get("metrics")).get("chord_tone_ratio"), 0.0) for row in selected_rows
                ),
                "selected_max_non_chord_tone_run": max(
                    int(_dict(row.get("pitch_role_metrics")).get("max_non_chord_tone_run", 0) or 0)
                    for row in selected_rows
                ),
                "selected_max_interval": max(
                    int(_dict(row.get("focused_solo_metrics")).get("focused_max_interval", 0) or 0)
                    for row in selected_rows
                ),
                "broad_model_quality_claimed": False,
            },
            "claim_boundary": {
                "boundary": boundary,
                "midi_pitch_role_repair_candidate_claimed": repaired_source_count
                >= int(min_repaired_source_candidates),
                "dead_air_gain_preserved_claimed": dead_air_preserved_source_count
                >= int(min_repaired_source_candidates),
                "human_audio_preference_claimed": False,
                "multi_reviewer_preference_claimed": False,
                "broad_model_quality_claimed": False,
                "brad_style_adaptation_claimed": False,
                "production_ready_improviser_claimed": False,
            },
            "not_proven": [
                "human_audio_preference",
                "multi_reviewer_preference",
                "broad_trained_model_quality",
                "brad_style_adaptation",
                "production_ready_improviser",
            ],
            "next_recommended_issue": (
                "Stage B margin-recovered phrase/vocabulary duration coverage fill outside-soloing repair "
                "audio review package"
            ),
        }
    )


def validate_outside_soloing_repair_sweep(
    report: dict[str, Any],
    *,
    expected_boundary: str | None,
    min_source_candidates: int,
    min_repaired_source_candidates: int,
    require_no_broad_quality_claim: bool,
) -> dict[str, Any]:
    summary = _dict(report.get("repair_summary"))
    claim = _dict(report.get("claim_boundary"))
    boundary = str(summary.get("boundary") or "")
    if expected_boundary and boundary != expected_boundary:
        raise StageBDurationCoverageOutsideSoloingRepairSweepError(
            f"expected boundary {expected_boundary}, got {boundary}"
        )
    if int(summary.get("source_candidate_count", 0) or 0) < int(min_source_candidates):
        raise StageBDurationCoverageOutsideSoloingRepairSweepError("not enough source candidates")
    if int(summary.get("repaired_source_candidate_count", 0) or 0) < int(min_repaired_source_candidates):
        raise StageBDurationCoverageOutsideSoloingRepairSweepError("not enough repaired source candidates")
    if require_no_broad_quality_claim:
        blocked = [
            "human_audio_preference_claimed",
            "multi_reviewer_preference_claimed",
            "broad_model_quality_claimed",
            "brad_style_adaptation_claimed",
            "production_ready_improviser_claimed",
        ]
        claimed = [name for name in blocked if bool(claim.get(name, True))]
        if claimed:
            raise StageBDurationCoverageOutsideSoloingRepairSweepError(f"unexpected broad claim: {claimed}")
    return {
        "boundary": boundary,
        "source_candidate_count": int(summary.get("source_candidate_count", 0) or 0),
        "repaired_source_candidate_count": int(summary.get("repaired_source_candidate_count", 0) or 0),
        "dead_air_preserved_source_candidate_count": int(
            summary.get("dead_air_preserved_source_candidate_count", 0) or 0
        ),
        "total_variant_count": int(summary.get("total_variant_count", 0) or 0),
        "total_qualified_variant_count": int(summary.get("total_qualified_variant_count", 0) or 0),
        "selected_repair_policies": list(summary.get("selected_repair_policies") or []),
        "selected_min_chord_tone_ratio": float(summary.get("selected_min_chord_tone_ratio", 0.0) or 0.0),
        "selected_max_non_chord_tone_run": int(summary.get("selected_max_non_chord_tone_run", 0) or 0),
        "selected_max_interval": int(summary.get("selected_max_interval", 0) or 0),
        "broad_model_quality_claimed": bool(summary.get("broad_model_quality_claimed", True)),
        "next_recommended_issue": str(report.get("next_recommended_issue") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    summary = report["repair_summary"]
    lines = [
        "# Stage B Duration Coverage Fill Outside-Soloing Repair Sweep",
        "",
        f"- previous boundary: `{report['previous_boundary']}`",
        f"- source dead-air boundary: `{report['source_dead_air_boundary']}`",
        f"- boundary: `{summary['boundary']}`",
        f"- source candidates: `{summary['source_candidate_count']}`",
        f"- repaired source candidates: `{summary['repaired_source_candidate_count']}`",
        f"- dead-air preserved source candidates: `{summary['dead_air_preserved_source_candidate_count']}`",
        f"- total variants: `{summary['total_variant_count']}`",
        f"- qualified variants: `{summary['total_qualified_variant_count']}`",
        f"- selected policies: `{summary['selected_repair_policies']}`",
        f"- selected min chord-tone ratio: `{summary['selected_min_chord_tone_ratio']:.3f}`",
        f"- selected max non-chord run: `{summary['selected_max_non_chord_tone_run']}`",
        f"- selected max interval: `{summary['selected_max_interval']}`",
        f"- broad model quality claimed: `{summary['broad_model_quality_claimed']}`",
        "",
        "| source | sample seed | selected | policy | qualified | chord-tone | non-chord run | "
        "dead-air | unique | max interval | flags |",
        "|---|---:|---|---|:---:|---:|---:|---:|---:|---:|---|",
    ]
    for result in report["source_repair_results"]:
        selected = result["selected_candidate"]
        metrics = selected["metrics"]
        focused = selected["focused_solo_metrics"]
        pitch_role = selected["pitch_role_metrics"]
        gate = selected["outside_soloing_gate"]
        lines.append(
            "| `{source}` | {sample_seed} | `{selected_id}` | `{policy}` | {qualified} | "
            "{chord_tone:.3f} | {non_chord_run} | {dead_air:.4f} | {unique} | {interval} | `{flags}` |".format(
                source=result["source_candidate_id"],
                sample_seed=int(result["sample_seed"]),
                selected_id=selected["candidate_id"],
                policy=selected["repair_policy"],
                qualified=bool(gate["qualified"]),
                chord_tone=float_value(metrics.get("chord_tone_ratio"), 0.0),
                non_chord_run=int(pitch_role["max_non_chord_tone_run"]),
                dead_air=float_value(metrics.get("dead_air_ratio"), 1.0),
                unique=int(focused["focused_unique_pitch_count"]),
                interval=int(focused["focused_max_interval"]),
                flags=list(gate["flags"]),
            )
        )
    lines.extend(["", "## Not Proven", ""])
    for item in report.get("not_proven", []):
        lines.append(f"- `{item}`")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Stage B outside-soloing repair sweep")
    parser.add_argument(
        "--outside_soloing_decision",
        type=str,
        default="outputs/stage_b_duration_coverage_fill_outside_soloing_repair_decision/"
        "harness_stage_b_duration_coverage_fill_outside_soloing_repair_decision/"
        "stage_b_duration_coverage_fill_outside_soloing_repair_decision.json",
    )
    parser.add_argument(
        "--dead_air_gain_repair",
        type=str,
        default="outputs/stage_b_duration_coverage_fill_dead_air_gain_repeatability_repair/"
        "harness_stage_b_duration_coverage_fill_dead_air_gain_repeatability_repair/"
        "stage_b_duration_coverage_fill_dead_air_gain_repeatability_repair.json",
    )
    parser.add_argument("--generation_output_root", type=str, default="outputs/stage_b_generation_probe")
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/stage_b_duration_coverage_fill_outside_soloing_repair_sweep",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--repair_policy", action="append", type=str, default=None)
    parser.add_argument("--min_source_candidates", type=int, default=2)
    parser.add_argument("--min_repaired_source_candidates", type=int, default=2)
    parser.add_argument("--min_chord_tone_ratio", type=float, default=0.72)
    parser.add_argument("--max_non_chord_run_length", type=int, default=1)
    parser.add_argument("--max_interval", type=int, default=7)
    parser.add_argument("--min_unique_pitch_count", type=int, default=6)
    parser.add_argument("--min_note_count", type=int, default=12)
    parser.add_argument("--max_dead_air_ratio_exclusive", type=float, default=0.376)
    parser.add_argument("--max_simultaneous_notes", type=int, default=1)
    parser.add_argument("--expected_boundary", type=str, default="")
    parser.add_argument("--require_no_broad_quality_claim", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    report = build_outside_soloing_repair_sweep_report(
        outside_soloing_decision=read_json(Path(args.outside_soloing_decision)),
        dead_air_gain_repair=read_json(Path(args.dead_air_gain_repair)),
        output_dir=output_dir,
        generation_output_root=Path(args.generation_output_root),
        repair_policies=args.repair_policy or list(DEFAULT_REPAIR_POLICIES),
        min_source_candidates=int(args.min_source_candidates),
        min_repaired_source_candidates=int(args.min_repaired_source_candidates),
        min_chord_tone_ratio=float(args.min_chord_tone_ratio),
        max_non_chord_run_length=int(args.max_non_chord_run_length),
        max_interval=int(args.max_interval),
        min_unique_pitch_count=int(args.min_unique_pitch_count),
        min_note_count=int(args.min_note_count),
        max_dead_air_ratio_exclusive=float(args.max_dead_air_ratio_exclusive),
        max_simultaneous_notes=int(args.max_simultaneous_notes),
    )
    summary = validate_outside_soloing_repair_sweep(
        report,
        expected_boundary=str(args.expected_boundary or ""),
        min_source_candidates=int(args.min_source_candidates),
        min_repaired_source_candidates=int(args.min_repaired_source_candidates),
        require_no_broad_quality_claim=bool(args.require_no_broad_quality_claim),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "stage_b_duration_coverage_fill_outside_soloing_repair_sweep.json"
    markdown_path = output_dir / "stage_b_duration_coverage_fill_outside_soloing_repair_sweep.md"
    write_json(report_path, report)
    write_json(output_dir / "stage_b_duration_coverage_fill_outside_soloing_repair_sweep_validation_summary.json", summary)
    markdown_path.write_text(markdown_report(report), encoding="utf-8")
    print(json.dumps({**summary, "report_path": str(report_path), "markdown_path": str(markdown_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
