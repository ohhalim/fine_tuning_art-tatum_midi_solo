"""Summarize duration/coverage fill repair for a Stage B phrase/vocabulary candidate."""

from __future__ import annotations

import argparse
import json
import math
import numbers
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

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
from scripts.build_stage_b_margin_recovered_listening_notes import write_json  # noqa: E402
from scripts.select_stage_b_margin_recovered_repair_candidate import (  # noqa: E402
    focused_solo_metrics,
    non_drum_notes,
    read_json,
)
from scripts.run_stage_b_generation_probe import postprocess_stage_b_midi  # noqa: E402
from scripts.summarize_stage_b_margin_recovered_phrase_vocabulary_repair import (  # noqa: E402
    candidate_gate_flags,
    focused_max_interval,
    float_value,
)


PREFERRED_SOLO_MIN = 48
PREFERRED_SOLO_MAX = 88
DEFAULT_FILL_MAX_ADDITIONS = (4, 6, 8, 10)


class DurationCoverageFillRepairError(ValueError):
    pass


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


def request_from_report(report: dict[str, Any]) -> GenerationRequest:
    raw = dict(report.get("request") or {})
    request = GenerationRequest(
        bpm=int(raw.get("bpm", 124) or 124),
        chord_progression=list(raw.get("chord_progression") or ["Cm7", "Fm7", "Bb7", "Ebmaj7"]),
        bars=int(raw.get("bars", 2) or 2),
        density=str(raw.get("density") or "medium"),
        energy=str(raw.get("energy") or "mid"),
        temperature=raw.get("temperature"),
        top_k=raw.get("top_k"),
        top_p=raw.get("top_p"),
        seed=int(raw.get("seed", 0) or 0),
    )
    request.validate()
    return request


def copy_note(note: pretty_midi.Note) -> pretty_midi.Note:
    return pretty_midi.Note(
        velocity=int(note.velocity),
        pitch=int(note.pitch),
        start=float(note.start),
        end=float(note.end),
    )


def monophonic_source_notes(midi_path: Path, simultaneous_limit: int) -> tuple[list[pretty_midi.Note], dict[str, Any]]:
    midi = pretty_midi.PrettyMIDI(str(midi_path))
    postprocess = postprocess_stage_b_midi(midi, simultaneous_limit=simultaneous_limit)
    return [copy_note(note) for note in non_drum_notes(midi)], postprocess


def pitch_candidates_for_time(request: GenerationRequest, start_sec: float) -> list[int]:
    chord = chord_for_time(request, start_sec)
    root_pc, intervals = parse_chord(chord)
    return chord_pitches_in_range(root_pc, intervals, PREFERRED_SOLO_MIN, PREFERRED_SOLO_MAX)


def choose_fill_pitch(
    *,
    request: GenerationRequest,
    start_sec: float,
    previous_pitch: int,
    next_pitch: int,
    target_pitch: float,
    local_pitches: Sequence[int],
    all_pitches: set[int],
) -> int:
    chord_tones = set(pitch_candidates_for_time(request, start_sec))
    bridge_min = max(PREFERRED_SOLO_MIN, min(previous_pitch, next_pitch) - 3)
    bridge_max = min(PREFERRED_SOLO_MAX, max(previous_pitch, next_pitch) + 3)
    passing_tones = set(range(bridge_min, bridge_max + 1))
    pool = sorted(chord_tones | passing_tones)
    scored: list[tuple[bool, bool, float, int, int]] = []
    recent = set(int(pitch) for pitch in local_pitches[-3:])
    for pitch in pool:
        if pitch == int(previous_pitch):
            continue
        if abs(int(pitch) - int(previous_pitch)) > 7:
            continue
        if abs(int(next_pitch) - int(pitch)) > 11:
            continue
        scored.append(
            (
                int(pitch) in recent,
                int(pitch) in all_pitches,
                abs(float(pitch) - float(target_pitch)),
                abs(int(next_pitch) - int(pitch)),
                int(pitch),
            )
        )
    if not scored:
        return int(previous_pitch)
    scored.sort()
    return int(scored[0][-1])


def enforce_monophonic_note_ends(
    notes: Sequence[pretty_midi.Note],
    *,
    max_duration_sec: float,
    min_duration_sec: float = 0.04,
    end_epsilon_sec: float = 0.001,
) -> list[pretty_midi.Note]:
    ordered = sorted(notes, key=lambda note: (float(note.start), int(note.pitch), float(note.end)))
    repaired: list[pretty_midi.Note] = []
    for index, note in enumerate(ordered):
        start = float(note.start)
        next_start = float(ordered[index + 1].start) if index + 1 < len(ordered) else float(max_duration_sec)
        latest_end = min(float(max_duration_sec), next_start - float(end_epsilon_sec))
        if latest_end <= start:
            continue
        original_end = min(float(note.end), float(max_duration_sec))
        end = min(max(original_end, start + float(min_duration_sec)), latest_end)
        if end <= start:
            continue
        repaired.append(
            pretty_midi.Note(
                velocity=int(note.velocity),
                pitch=int(note.pitch),
                start=start,
                end=end,
            )
        )
    return repaired


def build_filled_notes(
    source_notes: Sequence[pretty_midi.Note],
    *,
    request: GenerationRequest,
    max_additions: int,
    dead_air_threshold_sec: float,
) -> tuple[list[pretty_midi.Note], list[dict[str, Any]]]:
    if not source_notes:
        raise DurationCoverageFillRepairError("source MIDI has no notes")

    sixteenth_sec = 60.0 / float(request.bpm) / 4.0
    max_duration_sec = phrase_duration_sec(request)
    notes = [copy_note(note) for note in sorted(source_notes, key=lambda note: (float(note.start), int(note.pitch)))]
    additions: list[dict[str, Any]] = []
    all_pitches = {int(note.pitch) for note in notes}

    for previous_note, next_note in zip(notes, notes[1:]):
        gap_sec = float(next_note.start) - float(previous_note.start)
        if gap_sec < float(dead_air_threshold_sec):
            continue
        slot_count = max(0, math.ceil((gap_sec - 0.001) / sixteenth_sec) - 1)
        if slot_count <= 0:
            continue
        previous_pitch = int(previous_note.pitch)
        local_pitches = [previous_pitch]
        for slot_index in range(slot_count):
            if len(additions) >= int(max_additions):
                break
            start_sec = round(float(previous_note.start) + sixteenth_sec * (slot_index + 1), 6)
            if start_sec >= float(next_note.start) - 0.04:
                continue
            target_pitch = previous_note.pitch + (
                (int(next_note.pitch) - int(previous_note.pitch)) * ((slot_index + 1) / (slot_count + 1))
            )
            pitch = choose_fill_pitch(
                request=request,
                start_sec=start_sec,
                previous_pitch=previous_pitch,
                next_pitch=int(next_note.pitch),
                target_pitch=target_pitch,
                local_pitches=local_pitches,
                all_pitches=all_pitches,
            )
            duration_sec = max(0.06, sixteenth_sec * 0.85)
            notes.append(
                pretty_midi.Note(
                    velocity=76,
                    pitch=int(pitch),
                    start=float(start_sec),
                    end=min(float(max_duration_sec), float(start_sec) + duration_sec),
                )
            )
            additions.append(
                {
                    "start_sec": float(start_sec),
                    "pitch": int(pitch),
                    "between": [int(previous_note.pitch), int(next_note.pitch)],
                }
            )
            previous_pitch = int(pitch)
            local_pitches.append(int(pitch))
            all_pitches.add(int(pitch))
        if len(additions) >= int(max_additions):
            break

    return enforce_monophonic_note_ends(notes, max_duration_sec=max_duration_sec), additions


def write_midi(notes: Sequence[pretty_midi.Note], output_path: Path, bpm: int) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    midi = pretty_midi.PrettyMIDI(initial_tempo=float(bpm))
    instrument = pretty_midi.Instrument(program=0, is_drum=False, name="duration_coverage_fill_repair")
    instrument.notes = [copy_note(note) for note in notes]
    midi.instruments.append(instrument)
    midi.write(str(output_path))
    return output_path


def build_variant(
    *,
    base_candidate: dict[str, Any],
    source_notes: Sequence[pretty_midi.Note],
    output_dir: Path,
    request: GenerationRequest,
    max_additions: int,
    dead_air_threshold_sec: float,
    min_unique_pitch_count: int,
    max_dead_air_ratio_exclusive: float,
    min_note_count: int,
    max_simultaneous_notes: int,
    max_duplicated_3_note_chunks: int,
    max_adjacent_pitch_repeats_exclusive: int,
    max_interval_exclusive: int,
) -> dict[str, Any]:
    candidate_id = f"{base_candidate['candidate_id']}_duration_fill_maxadd_{int(max_additions)}"
    notes, additions = build_filled_notes(
        source_notes,
        request=request,
        max_additions=int(max_additions),
        dead_air_threshold_sec=float(dead_air_threshold_sec),
    )
    midi_path = write_midi(notes, output_dir / "midi" / f"{candidate_id}.mid", bpm=int(request.bpm))
    focused = focused_solo_metrics(midi_path)
    focused["focused_max_interval"] = focused_max_interval(midi_path)
    row: dict[str, Any] = {
        "candidate_id": candidate_id,
        "source_candidate_id": str(base_candidate["candidate_id"]),
        "midi_path": str(midi_path),
        "metrics": compute_midi_metrics(midi_path, 0, False, request=request).to_dict(),
        "focused_solo_metrics": focused,
        "fill_repair": {
            "max_additions": int(max_additions),
            "fill_addition_count": int(len(additions)),
            "dead_air_threshold_sec": float(dead_air_threshold_sec),
            "additions": additions,
        },
    }
    flags = candidate_gate_flags(
        row,
        min_unique_pitch_count=int(min_unique_pitch_count),
        max_dead_air_ratio_exclusive=float(max_dead_air_ratio_exclusive),
        min_note_count=int(min_note_count),
        max_simultaneous_notes=int(max_simultaneous_notes),
        max_duplicated_3_note_chunks=int(max_duplicated_3_note_chunks),
        max_adjacent_pitch_repeats_exclusive=int(max_adjacent_pitch_repeats_exclusive),
        max_interval_exclusive=int(max_interval_exclusive),
    )
    row["duration_coverage_gate"] = {
        "qualified": not flags,
        "flags": flags,
    }
    row["duration_coverage_score"] = duration_coverage_score(row, qualified=not flags)
    return _json_safe(row)


def duration_coverage_score(candidate: dict[str, Any], *, qualified: bool) -> float:
    metrics = candidate["metrics"]
    focused = candidate["focused_solo_metrics"]
    repair = candidate["fill_repair"]
    score = 1000.0 if qualified else 0.0
    score += (1.0 - min(1.0, float_value(metrics.get("dead_air_ratio"), 1.0))) * 120.0
    score += min(16, int(focused.get("focused_unique_pitch_count", 0) or 0)) * 8.0
    score += min(24, int(focused.get("focused_note_count", 0) or 0)) * 2.0
    score -= int(repair.get("fill_addition_count", 0) or 0) * 4.0
    score -= int(focused.get("focused_adjacent_pitch_repeats", 0) or 0) * 30.0
    score -= int(focused.get("focused_duplicated_3_note_pitch_class_chunks", 0) or 0) * 50.0
    score -= max(0, int(focused.get("focused_max_interval", 0) or 0) - 7) * 10.0
    return round(float(score), 6)


def build_duration_coverage_fill_report(
    previous_repair_summary: dict[str, Any],
    *,
    output_dir: Path,
    fill_max_additions: Sequence[int],
    dead_air_threshold_sec: float,
    simultaneous_limit: int,
    min_unique_pitch_count: int,
    max_dead_air_ratio_exclusive: float,
    min_note_count: int,
    max_simultaneous_notes: int,
    max_duplicated_3_note_chunks: int,
    max_adjacent_pitch_repeats_exclusive: int,
    max_interval_exclusive: int,
) -> dict[str, Any]:
    base_candidate = dict(previous_repair_summary.get("selected_candidate") or {})
    if not base_candidate:
        raise DurationCoverageFillRepairError("previous summary has no selected candidate")
    midi_path = Path(str(base_candidate.get("midi_path") or ""))
    if not midi_path.exists():
        raise DurationCoverageFillRepairError(f"selected candidate MIDI not found: {midi_path}")
    source_report_path = Path(str(base_candidate.get("source_report_path") or ""))
    if not source_report_path.exists():
        raise DurationCoverageFillRepairError(f"source report not found: {source_report_path}")

    source_report = read_json(source_report_path)
    request = request_from_report(source_report)
    source_notes, source_postprocess = monophonic_source_notes(midi_path, simultaneous_limit=int(simultaneous_limit))
    variants = [
        build_variant(
            base_candidate=base_candidate,
            source_notes=source_notes,
            output_dir=output_dir,
            request=request,
            max_additions=int(max_additions),
            dead_air_threshold_sec=float(dead_air_threshold_sec),
            min_unique_pitch_count=int(min_unique_pitch_count),
            max_dead_air_ratio_exclusive=float(max_dead_air_ratio_exclusive),
            min_note_count=int(min_note_count),
            max_simultaneous_notes=int(max_simultaneous_notes),
            max_duplicated_3_note_chunks=int(max_duplicated_3_note_chunks),
            max_adjacent_pitch_repeats_exclusive=int(max_adjacent_pitch_repeats_exclusive),
            max_interval_exclusive=int(max_interval_exclusive),
        )
        for max_additions in fill_max_additions
    ]
    if not variants:
        raise DurationCoverageFillRepairError("no duration/coverage fill variants were built")
    variants.sort(
        key=lambda row: (
            not bool(row["duration_coverage_gate"]["qualified"]),
            int(row["fill_repair"]["fill_addition_count"]),
            float_value(row["metrics"].get("dead_air_ratio"), 1.0),
            -int(row["focused_solo_metrics"].get("focused_unique_pitch_count", 0) or 0),
            str(row["candidate_id"]),
        )
    )
    for index, row in enumerate(variants, start=1):
        row["duration_coverage_rank"] = int(index)

    selected = variants[0]
    selected_metrics = selected["metrics"]
    selected_focused = selected["focused_solo_metrics"]
    base_metrics = dict(base_candidate.get("metrics") or {})
    base_focused = dict(base_candidate.get("focused_solo_metrics") or {})
    base_dead_air = float_value(base_metrics.get("dead_air_ratio"), 1.0)
    selected_dead_air = float_value(selected_metrics.get("dead_air_ratio"), 1.0)
    qualified_count = sum(1 for row in variants if row["duration_coverage_gate"]["qualified"])

    return _json_safe(
        {
            "schema_version": "stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_repair_v1",
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
            "output_dir": str(output_dir),
            "source_previous_summary": str(previous_repair_summary.get("output_dir") or ""),
            "source_candidate": {
                "candidate_id": str(base_candidate["candidate_id"]),
                "midi_path": str(midi_path),
                "source_report_path": str(source_report_path),
                "source_postprocess": source_postprocess,
                "metrics": base_metrics,
                "focused_solo_metrics": base_focused,
            },
            "thresholds": {
                "min_focused_unique_pitch_count": int(min_unique_pitch_count),
                "max_dead_air_ratio_exclusive": float(max_dead_air_ratio_exclusive),
                "min_focused_note_count": int(min_note_count),
                "max_focused_simultaneous_notes": int(max_simultaneous_notes),
                "max_duplicated_3_note_pitch_class_chunks": int(max_duplicated_3_note_chunks),
                "max_adjacent_pitch_repeats_exclusive": int(max_adjacent_pitch_repeats_exclusive),
                "max_interval_exclusive": int(max_interval_exclusive),
                "dead_air_threshold_sec": float(dead_air_threshold_sec),
            },
            "variant_count": int(len(variants)),
            "qualified_variant_count": int(qualified_count),
            "selected_candidate": selected,
            "repair_summary": {
                "selected_candidate_id": str(selected["candidate_id"]),
                "selected_midi_path": str(selected["midi_path"]),
                "selected_fill_addition_count": int(selected["fill_repair"]["fill_addition_count"]),
                "qualified": bool(selected["duration_coverage_gate"]["qualified"]),
                "remaining_flags": list(selected["duration_coverage_gate"]["flags"]),
                "baseline_dead_air_ratio": float(base_dead_air),
                "selected_dead_air_ratio": float(selected_dead_air),
                "dead_air_delta_from_baseline": round(float(base_dead_air - selected_dead_air), 6),
                "baseline_focused_unique_pitch_count": int(
                    base_focused.get("focused_unique_pitch_count", 0) or 0
                ),
                "selected_focused_unique_pitch_count": int(
                    selected_focused.get("focused_unique_pitch_count", 0) or 0
                ),
                "baseline_focused_note_count": int(base_focused.get("focused_note_count", 0) or 0),
                "selected_focused_note_count": int(selected_focused.get("focused_note_count", 0) or 0),
                "selected_adjacent_pitch_repeats": int(
                    selected_focused.get("focused_adjacent_pitch_repeats", 0) or 0
                ),
                "selected_duplicated_3_note_pitch_class_chunks": int(
                    selected_focused.get("focused_duplicated_3_note_pitch_class_chunks", 0) or 0
                ),
                "selected_max_interval": int(selected_focused.get("focused_max_interval", 0) or 0),
                "duration_coverage_fill_improved": bool(
                    selected_dead_air < base_dead_air
                    and selected_dead_air < float(max_dead_air_ratio_exclusive)
                    and bool(selected["duration_coverage_gate"]["qualified"])
                ),
                "claim_boundary": "postprocess_duration_coverage_fill_candidate",
            },
            "variants": variants,
        }
    )


def markdown_report(report: dict[str, Any]) -> str:
    summary = report["repair_summary"]
    lines = [
        "# Stage B Margin-Recovered Phrase/Vocabulary Duration Coverage Fill Repair",
        "",
        f"- source candidate: `{report['source_candidate']['candidate_id']}`",
        f"- variant count: `{report['variant_count']}`",
        f"- qualified variant count: `{report['qualified_variant_count']}`",
        f"- selected candidate: `{summary['selected_candidate_id']}`",
        f"- selected fill additions: `{summary['selected_fill_addition_count']}`",
        f"- qualified: `{summary['qualified']}`",
        f"- remaining flags: `{summary['remaining_flags']}`",
        f"- dead-air delta from baseline: `{summary['dead_air_delta_from_baseline']:.3f}`",
        f"- claim boundary: `{summary['claim_boundary']}`",
        "",
        "| rank | candidate | qualified | additions | score | notes | unique | dead-air | "
        "adj repeat | dup3 | max interval | flags |",
        "|---:|---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for candidate in report["variants"]:
        focused = candidate["focused_solo_metrics"]
        metrics = candidate["metrics"]
        gate = candidate["duration_coverage_gate"]
        repair = candidate["fill_repair"]
        lines.append(
            "| {rank} | `{candidate_id}` | {qualified} | {additions} | {score:.3f} | "
            "{notes} | {unique} | {dead_air:.3f} | {adj} | {dup3} | {max_interval} | `{flags}` |".format(
                rank=int(candidate["duration_coverage_rank"]),
                candidate_id=candidate["candidate_id"],
                qualified=bool(gate["qualified"]),
                additions=int(repair["fill_addition_count"]),
                score=float(candidate["duration_coverage_score"]),
                notes=int(focused["focused_note_count"]),
                unique=int(focused["focused_unique_pitch_count"]),
                dead_air=float_value(metrics.get("dead_air_ratio"), 0.0),
                adj=int(focused["focused_adjacent_pitch_repeats"]),
                dup3=int(focused["focused_duplicated_3_note_pitch_class_chunks"]),
                max_interval=int(focused["focused_max_interval"]),
                flags=list(gate["flags"]),
            )
        )
    return "\n".join(lines).rstrip() + "\n"


def validate_duration_coverage_fill(
    report: dict[str, Any],
    *,
    require_qualified: bool,
    require_dead_air_improvement: bool,
    expected_fill_addition_count: int | None,
) -> dict[str, Any]:
    summary = report["repair_summary"]
    if require_qualified and not bool(summary["qualified"]):
        raise DurationCoverageFillRepairError("selected duration/coverage fill candidate is not qualified")
    if require_dead_air_improvement and not bool(summary["duration_coverage_fill_improved"]):
        raise DurationCoverageFillRepairError("selected duration/coverage fill candidate did not improve dead-air")
    if expected_fill_addition_count is not None and int(summary["selected_fill_addition_count"]) != int(
        expected_fill_addition_count
    ):
        raise DurationCoverageFillRepairError(
            "expected fill addition count "
            f"{int(expected_fill_addition_count)}, got {summary['selected_fill_addition_count']}"
        )
    return {
        "variant_count": int(report["variant_count"]),
        "qualified_variant_count": int(report["qualified_variant_count"]),
        "selected_candidate_id": str(summary["selected_candidate_id"]),
        "selected_fill_addition_count": int(summary["selected_fill_addition_count"]),
        "qualified": bool(summary["qualified"]),
        "remaining_flags": list(summary["remaining_flags"]),
        "baseline_dead_air_ratio": float(summary["baseline_dead_air_ratio"]),
        "selected_dead_air_ratio": float(summary["selected_dead_air_ratio"]),
        "dead_air_delta_from_baseline": float(summary["dead_air_delta_from_baseline"]),
        "selected_focused_unique_pitch_count": int(summary["selected_focused_unique_pitch_count"]),
        "selected_focused_note_count": int(summary["selected_focused_note_count"]),
        "selected_adjacent_pitch_repeats": int(summary["selected_adjacent_pitch_repeats"]),
        "selected_duplicated_3_note_pitch_class_chunks": int(
            summary["selected_duplicated_3_note_pitch_class_chunks"]
        ),
        "selected_max_interval": int(summary["selected_max_interval"]),
        "claim_boundary": str(summary["claim_boundary"]),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize duration/coverage fill repair for Stage B candidate")
    parser.add_argument(
        "--summary_path",
        type=str,
        default=str(
            ROOT_DIR
            / "outputs"
            / "stage_b_margin_recovered_phrase_vocabulary_repair"
            / "harness_stage_b_margin_recovered_phrase_vocabulary_coverage_aware_adjacent_constrained_repair"
            / "phrase_vocabulary_repair_summary.json"
        ),
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=str(ROOT_DIR / "outputs" / "stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_repair"),
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--fill_max_additions", action="append", type=int, default=None)
    parser.add_argument("--dead_air_threshold_sec", type=float, default=0.18)
    parser.add_argument("--simultaneous_limit", type=int, default=1)
    parser.add_argument("--min_unique_pitch_count", type=int, default=7)
    parser.add_argument("--max_dead_air_ratio_exclusive", type=float, default=0.376)
    parser.add_argument("--min_note_count", type=int, default=12)
    parser.add_argument("--max_simultaneous_notes", type=int, default=1)
    parser.add_argument("--max_duplicated_3_note_chunks", type=int, default=0)
    parser.add_argument("--max_adjacent_pitch_repeats_exclusive", type=int, default=1)
    parser.add_argument("--max_interval_exclusive", type=int, default=12)
    parser.add_argument("--require_qualified", action="store_true")
    parser.add_argument("--require_dead_air_improvement", action="store_true")
    parser.add_argument("--expected_fill_addition_count", type=int, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    fill_max_additions = args.fill_max_additions or list(DEFAULT_FILL_MAX_ADDITIONS)
    report = build_duration_coverage_fill_report(
        read_json(Path(args.summary_path)),
        output_dir=output_dir,
        fill_max_additions=fill_max_additions,
        dead_air_threshold_sec=float(args.dead_air_threshold_sec),
        simultaneous_limit=int(args.simultaneous_limit),
        min_unique_pitch_count=int(args.min_unique_pitch_count),
        max_dead_air_ratio_exclusive=float(args.max_dead_air_ratio_exclusive),
        min_note_count=int(args.min_note_count),
        max_simultaneous_notes=int(args.max_simultaneous_notes),
        max_duplicated_3_note_chunks=int(args.max_duplicated_3_note_chunks),
        max_adjacent_pitch_repeats_exclusive=int(args.max_adjacent_pitch_repeats_exclusive),
        max_interval_exclusive=int(args.max_interval_exclusive),
    )
    summary = validate_duration_coverage_fill(
        report,
        require_qualified=bool(args.require_qualified),
        require_dead_air_improvement=bool(args.require_dead_air_improvement),
        expected_fill_addition_count=args.expected_fill_addition_count,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "duration_coverage_fill_repair_summary.json", report)
    write_json(output_dir / "duration_coverage_fill_repair_result.json", summary)
    (output_dir / "duration_coverage_fill_repair_summary.md").write_text(
        markdown_report(report),
        encoding="utf-8",
    )
    summary.update(
        {
            "summary_path": str(output_dir / "duration_coverage_fill_repair_summary.json"),
            "markdown_path": str(output_dir / "duration_coverage_fill_repair_summary.md"),
        }
    )
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
