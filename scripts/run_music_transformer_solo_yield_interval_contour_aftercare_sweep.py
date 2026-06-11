"""Repair residual wide intervals after density-aftercare repair."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

import pretty_midi

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from scripts.assess_stage_b_generic_base_readiness import write_json, write_text  # noqa: E402
from scripts.decide_music_transformer_solo_yield_density_aftercare_objective_next import (  # noqa: E402
    SCHEMA_VERSION as OBJECTIVE_DECISION_SCHEMA_VERSION,
)
from scripts.review_music_transformer_solo_yield_repaired_top8_objective_failures import (  # noqa: E402
    midi_profile,
)
from scripts.run_music_transformer_solo_yield_density_aftercare_sweep import (  # noqa: E402
    SCHEMA_VERSION as SOURCE_REPAIR_SWEEP_SCHEMA_VERSION,
    chord_contexts,
    chord_tone_ratio,
    direction_change_ratio,
    max_abs_interval,
    note_arrays,
)


SCHEMA_VERSION = "music_transformer_solo_yield_interval_contour_aftercare_sweep_v1"
BOUNDARY = "music_transformer_solo_yield_interval_contour_aftercare_sweep"
NEXT_BOUNDARY = "music_transformer_solo_yield_interval_contour_aftercare_audio_package"
SELECTED_TARGET = "interval_contour_aftercare_audio_package"


class SoloYieldIntervalContourAftercareSweepError(ValueError):
    pass


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


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SoloYieldIntervalContourAftercareSweepError(f"json not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _require_no_quality_claim(report: dict[str, Any]) -> None:
    readiness = _dict(report.get("readiness"))
    claimed = [
        key
        for key in (
            "audio_rendered_quality_claimed",
            "musical_quality_claimed",
            "artist_style_claimed",
            "production_ready_claimed",
        )
        if bool(readiness.get(key, False))
    ]
    if claimed:
        raise SoloYieldIntervalContourAftercareSweepError(f"unexpected quality claim: {claimed}")


def _all_notes(midi: pretty_midi.PrettyMIDI) -> list[pretty_midi.Note]:
    notes: list[pretty_midi.Note] = []
    for instrument in midi.instruments:
        if not instrument.is_drum:
            notes.extend(instrument.notes)
    return sorted(notes, key=lambda note: (float(note.start), int(note.pitch), float(note.end)))


def _target_midi_path(output_dir: Path, row: dict[str, Any]) -> Path:
    review_index = _int(row.get("review_index"))
    case_label = str(row.get("case_label") or "candidate")
    return output_dir / "midi" / f"candidate_{review_index:02d}_{case_label}_interval_contour_aftercare.mid"


def validate_sources(source_repair_sweep: dict[str, Any], objective_decision: dict[str, Any]) -> list[dict[str, Any]]:
    if str(source_repair_sweep.get("schema_version") or "") != SOURCE_REPAIR_SWEEP_SCHEMA_VERSION:
        raise SoloYieldIntervalContourAftercareSweepError("source density aftercare sweep schema required")
    if str(objective_decision.get("schema_version") or "") != OBJECTIVE_DECISION_SCHEMA_VERSION:
        raise SoloYieldIntervalContourAftercareSweepError("objective decision schema required")
    _require_no_quality_claim(source_repair_sweep)
    _require_no_quality_claim(objective_decision)
    decision = _dict(objective_decision.get("decision"))
    if str(decision.get("selected_next_target") or "") != "interval_contour_aftercare":
        raise SoloYieldIntervalContourAftercareSweepError("interval contour aftercare target required")
    rows = [_dict(row) for row in _list(source_repair_sweep.get("candidate_repairs"))]
    if not rows:
        raise SoloYieldIntervalContourAftercareSweepError("source repair rows required")
    return rows


def _note_chords(notes: list[pretty_midi.Note], chords: list[str]) -> list[str]:
    if not notes:
        return []
    pitches, starts, ends = note_arrays(notes)
    del pitches
    phrase_start = min(starts, default=0.0)
    phrase_duration = max(1e-6, max(ends, default=phrase_start) - phrase_start)
    note_times = [(start + end) / 2.0 for start, end in zip(starts, ends)]
    return chord_contexts(
        note_times,
        phrase_start=phrase_start,
        phrase_duration=phrase_duration,
        chords=chords,
    )


def _candidate_pitch_range(original_pitch: int, *, max_pitch_shift: int, pitch_min: int, pitch_max: int) -> Iterable[int]:
    low = max(int(pitch_min), int(original_pitch) - int(max_pitch_shift))
    high = min(int(pitch_max), int(original_pitch) + int(max_pitch_shift))
    return range(low, high + 1)


def _best_pitch_adjustment(
    notes: list[pretty_midi.Note],
    chords: list[str],
    *,
    max_interval: int,
    max_pitch_shift: int,
    pitch_min: int,
    pitch_max: int,
    min_chord_tone_ratio: float,
    min_direction_change: float,
) -> dict[str, Any] | None:
    pitches, _starts, _ends = note_arrays(notes)
    if len(pitches) < 2:
        return None
    note_chords = _note_chords(notes, chords)
    before_chord_ratio = chord_tone_ratio(pitches, note_chords)
    before_direction = direction_change_ratio(pitches)
    best: tuple[tuple[Any, ...], dict[str, Any]] | None = None
    for left_index, (left_pitch, right_pitch) in enumerate(zip(pitches, pitches[1:])):
        if abs(int(right_pitch) - int(left_pitch)) < int(max_interval):
            continue
        for target_index in (left_index + 1, left_index):
            if target_index == len(pitches) - 1:
                continue
            original_pitch = int(pitches[target_index])
            for trial_pitch in _candidate_pitch_range(
                original_pitch,
                max_pitch_shift=int(max_pitch_shift),
                pitch_min=int(pitch_min),
                pitch_max=int(pitch_max),
            ):
                if trial_pitch == original_pitch:
                    continue
                trial_pitches = list(pitches)
                trial_pitches[target_index] = int(trial_pitch)
                trial_interval = max_abs_interval(trial_pitches)
                trial_chord_ratio = chord_tone_ratio(trial_pitches, note_chords)
                trial_direction = direction_change_ratio(trial_pitches)
                if trial_interval >= int(max_interval):
                    continue
                if trial_chord_ratio + 1e-9 < max(float(min_chord_tone_ratio), before_chord_ratio):
                    continue
                if trial_direction + 1e-9 < max(float(min_direction_change), before_direction):
                    continue
                score = (
                    -trial_interval,
                    trial_chord_ratio,
                    trial_direction,
                    -abs(int(trial_pitch) - original_pitch),
                    target_index == left_index + 1,
                    -target_index,
                )
                candidate = {
                    "target_note_index": int(target_index),
                    "original_pitch": int(original_pitch),
                    "repaired_pitch": int(trial_pitch),
                    "pitch_shift": int(trial_pitch) - int(original_pitch),
                    "source_interval": abs(int(right_pitch) - int(left_pitch)),
                    "max_abs_interval_after": int(trial_interval),
                    "chord_tone_ratio_after": float(trial_chord_ratio),
                    "direction_change_ratio_after": float(trial_direction),
                }
                if best is None or score > best[0]:
                    best = (score, candidate)
    return best[1] if best else None


def _apply_adjustments(
    midi: pretty_midi.PrettyMIDI,
    chords: list[str],
    *,
    max_interval: int,
    max_pitch_shift: int,
    pitch_min: int,
    pitch_max: int,
    min_chord_tone_ratio: float,
    min_direction_change: float,
) -> list[dict[str, Any]]:
    adjustments: list[dict[str, Any]] = []
    for _ in range(8):
        notes = _all_notes(midi)
        if max_abs_interval([int(note.pitch) for note in notes]) < int(max_interval):
            break
        adjustment = _best_pitch_adjustment(
            notes,
            chords,
            max_interval=int(max_interval),
            max_pitch_shift=int(max_pitch_shift),
            pitch_min=int(pitch_min),
            pitch_max=int(pitch_max),
            min_chord_tone_ratio=float(min_chord_tone_ratio),
            min_direction_change=float(min_direction_change),
        )
        if adjustment is None:
            break
        target_note = notes[int(adjustment["target_note_index"])]
        target_note.pitch = int(adjustment["repaired_pitch"])
        adjustments.append(adjustment)
    return adjustments


def repair_candidate(
    row: dict[str, Any],
    *,
    output_dir: Path,
    max_interval: int,
    max_pitch_shift: int,
    pitch_min: int,
    pitch_max: int,
    min_chord_tone_ratio: float,
    min_direction_change: float,
    min_note_count: int,
) -> dict[str, Any]:
    source_path = Path(str(row.get("repaired_midi_path") or ""))
    if not source_path.exists():
        raise SoloYieldIntervalContourAftercareSweepError(f"source MIDI missing: {source_path}")
    before_profile = _dict(row.get("after_profile"))
    target_path = _target_midi_path(output_dir, row)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    needs_repair = _int(before_profile.get("midi_max_abs_interval")) >= int(max_interval)
    adjustments: list[dict[str, Any]] = []

    if needs_repair:
        midi = pretty_midi.PrettyMIDI(str(source_path))
        if not midi.instruments:
            raise SoloYieldIntervalContourAftercareSweepError("source MIDI instrument required")
        chords = [item.strip() for item in str(row.get("chords") or "").split(",") if item.strip()]
        adjustments = _apply_adjustments(
            midi,
            chords,
            max_interval=int(max_interval),
            max_pitch_shift=int(max_pitch_shift),
            pitch_min=int(pitch_min),
            pitch_max=int(pitch_max),
            min_chord_tone_ratio=float(min_chord_tone_ratio),
            min_direction_change=float(min_direction_change),
        )
        midi.write(str(target_path))
    else:
        shutil.copy2(source_path, target_path)

    repaired_candidate = {
        "review_midi_path": str(target_path),
        "chords": str(row.get("chords") or ""),
    }
    after_profile = midi_profile(repaired_candidate)
    return {
        "review_index": _int(row.get("review_index")),
        "case_label": str(row.get("case_label") or ""),
        "chords": str(row.get("chords") or ""),
        "source_midi_path": str(source_path),
        "repaired_midi_path": str(target_path),
        "source_wav_path": str(row.get("source_wav_path") or ""),
        "before_profile": before_profile,
        "after_profile": after_profile,
        "repair": {
            "wide_interval_before": bool(needs_repair),
            "wide_interval_after": _int(after_profile.get("midi_max_abs_interval")) >= int(max_interval),
            "adjusted_note_count": len(adjustments),
            "adjustments": adjustments,
            "max_pitch_shift": max((abs(_int(item.get("pitch_shift"))) for item in adjustments), default=0),
            "note_count_delta": _int(after_profile.get("midi_note_count")) - _int(before_profile.get("midi_note_count")),
            "chord_tone_ratio_delta": (
                _float(after_profile.get("midi_chord_tone_ratio"))
                - _float(before_profile.get("midi_chord_tone_ratio"))
            ),
            "direction_change_after_valid": _float(after_profile.get("midi_direction_change_ratio"))
            >= float(min_direction_change),
            "low_note_count_after": _int(after_profile.get("midi_note_count")) < int(min_note_count),
            "final_landing_preserved": bool(after_profile.get("final_landing_is_chord_tone", False)),
        },
    }


def build_repair_sweep(
    *,
    source_repair_sweep: dict[str, Any],
    objective_decision: dict[str, Any],
    output_dir: Path,
    max_interval: int = 8,
    max_pitch_shift: int = 2,
    pitch_min: int = 55,
    pitch_max: int = 84,
    min_chord_tone_ratio: float = 0.50,
    min_direction_change: float = 0.50,
    min_note_count: int = 30,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = validate_sources(source_repair_sweep, objective_decision)
    repairs = [
        repair_candidate(
            row,
            output_dir=output_dir,
            max_interval=int(max_interval),
            max_pitch_shift=int(max_pitch_shift),
            pitch_min=int(pitch_min),
            pitch_max=int(pitch_max),
            min_chord_tone_ratio=float(min_chord_tone_ratio),
            min_direction_change=float(min_direction_change),
            min_note_count=int(min_note_count),
        )
        for row in rows
    ]
    wide_before = sum(1 for row in repairs if bool(_dict(row.get("repair")).get("wide_interval_before")))
    wide_after = sum(1 for row in repairs if bool(_dict(row.get("repair")).get("wide_interval_after")))
    changed_total = sum(_int(_dict(row.get("repair")).get("adjusted_note_count")) for row in repairs)
    max_shift = max((_int(_dict(row.get("repair")).get("max_pitch_shift")) for row in repairs), default=0)
    chord_ratio_decrease = sum(
        1 for row in repairs if _float(_dict(row.get("repair")).get("chord_tone_ratio_delta")) < -1e-9
    )
    low_note_after = sum(1 for row in repairs if bool(_dict(row.get("repair")).get("low_note_count_after")))
    weak_direction_after = sum(
        1 for row in repairs if _float(_dict(row.get("after_profile")).get("midi_direction_change_ratio")) < float(min_direction_change)
    )
    final_landing_not = sum(
        1 for row in repairs if not bool(_dict(row.get("after_profile")).get("final_landing_is_chord_tone", False))
    )
    low_chord_after = sum(
        1 for row in repairs if _float(_dict(row.get("after_profile")).get("midi_chord_tone_ratio")) < float(min_chord_tone_ratio)
    )
    dead_air_after = sum(
        1 for row in repairs if _float(_dict(row.get("after_profile")).get("midi_max_gap_seconds")) >= 0.65
    )
    chord_ratios = [_float(_dict(row.get("after_profile")).get("midi_chord_tone_ratio")) for row in repairs]
    target_supported = bool(
        wide_before > wide_after
        and wide_after == 0
        and low_note_after == 0
        and chord_ratio_decrease == 0
        and low_chord_after == 0
        and weak_direction_after == 0
        and final_landing_not == 0
    )
    next_boundary = NEXT_BOUNDARY if target_supported else "music_transformer_solo_yield_objective_repair_decision"
    report = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "boundary": BOUNDARY,
        "source_repair_sweep": {
            "schema_version": source_repair_sweep.get("schema_version"),
            "output_dir": source_repair_sweep.get("output_dir"),
            "candidate_count": _int(_dict(source_repair_sweep.get("aggregate")).get("candidate_count")),
        },
        "source_objective_decision": {
            "schema_version": objective_decision.get("schema_version"),
            "output_dir": objective_decision.get("output_dir"),
            "selected_next_target": str(_dict(objective_decision.get("decision")).get("selected_next_target") or ""),
        },
        "candidate_repairs": repairs,
        "aggregate": {
            "candidate_count": len(repairs),
            "repaired_midi_count": len(repairs),
            "wide_interval_review_count_before": int(wide_before),
            "wide_interval_review_count_after": int(wide_after),
            "wide_interval_review_count_delta": int(wide_before - wide_after),
            "adjusted_note_total": int(changed_total),
            "max_pitch_shift": int(max_shift),
            "chord_tone_ratio_decrease_count": int(chord_ratio_decrease),
            "midi_low_chord_tone_ratio_count_after": int(low_chord_after),
            "low_note_count_after": int(low_note_after),
            "dead_air_aftercare_count_after": int(dead_air_after),
            "weak_direction_change_count_after": int(weak_direction_after),
            "final_landing_not_chord_tone_count_after": int(final_landing_not),
            "midi_chord_tone_ratio_min": min(chord_ratios, default=0.0),
            "midi_chord_tone_ratio_avg": float(mean(chord_ratios)) if chord_ratios else 0.0,
            "target_supported": bool(target_supported),
        },
        "readiness": {
            "interval_contour_aftercare_sweep_completed": True,
            "target_supported": bool(target_supported),
            "validated_listening_input_present": False,
            "audio_rendered_quality_claimed": False,
            "musical_quality_claimed": False,
            "artist_style_claimed": False,
            "production_ready_claimed": False,
        },
        "decision": {
            "current_boundary": BOUNDARY,
            "selected_next_target": SELECTED_TARGET if target_supported else "objective_repair_decision",
            "next_boundary": next_boundary,
            "critical_user_input_required": False,
            "reason": "wide interval residual reduced by bounded pitch adjustment while preserving objective guards",
        },
        "not_proven": [
            "human_audio_preference",
            "stable_jazz_solo_quality",
            "artist_level_long_solo_generation",
            "production_ready_improviser",
        ],
    }
    write_json(output_dir / "interval_contour_aftercare_sweep.json", report)
    write_json(output_dir / "interval_contour_aftercare_sweep_summary.json", validate_report(report))
    write_text(output_dir / "interval_contour_aftercare_sweep.md", markdown_report(report))
    return report


def validate_report(report: dict[str, Any], *, min_candidates: int = 1) -> dict[str, Any]:
    if str(report.get("schema_version") or "") != SCHEMA_VERSION:
        raise SoloYieldIntervalContourAftercareSweepError("schema version mismatch")
    readiness = _dict(report.get("readiness"))
    aggregate = _dict(report.get("aggregate"))
    decision = _dict(report.get("decision"))
    candidate_count = _int(aggregate.get("candidate_count"))
    if candidate_count < int(min_candidates):
        raise SoloYieldIntervalContourAftercareSweepError("candidate count below requirement")
    if not bool(readiness.get("interval_contour_aftercare_sweep_completed", False)):
        raise SoloYieldIntervalContourAftercareSweepError("repair sweep completion required")
    _require_no_quality_claim({"readiness": readiness})
    return {
        "schema_version": str(report.get("schema_version") or ""),
        "candidate_count": candidate_count,
        "repaired_midi_count": _int(aggregate.get("repaired_midi_count")),
        "wide_interval_review_count_before": _int(aggregate.get("wide_interval_review_count_before")),
        "wide_interval_review_count_after": _int(aggregate.get("wide_interval_review_count_after")),
        "wide_interval_review_count_delta": _int(aggregate.get("wide_interval_review_count_delta")),
        "adjusted_note_total": _int(aggregate.get("adjusted_note_total")),
        "max_pitch_shift": _int(aggregate.get("max_pitch_shift")),
        "chord_tone_ratio_decrease_count": _int(aggregate.get("chord_tone_ratio_decrease_count")),
        "midi_low_chord_tone_ratio_count_after": _int(aggregate.get("midi_low_chord_tone_ratio_count_after")),
        "low_note_count_after": _int(aggregate.get("low_note_count_after")),
        "dead_air_aftercare_count_after": _int(aggregate.get("dead_air_aftercare_count_after")),
        "weak_direction_change_count_after": _int(aggregate.get("weak_direction_change_count_after")),
        "final_landing_not_chord_tone_count_after": _int(
            aggregate.get("final_landing_not_chord_tone_count_after")
        ),
        "target_supported": bool(aggregate.get("target_supported", False)),
        "selected_next_target": str(decision.get("selected_next_target") or ""),
        "next_boundary": str(decision.get("next_boundary") or ""),
        "musical_quality_claimed": bool(readiness.get("musical_quality_claimed", True)),
    }


def markdown_report(report: dict[str, Any]) -> str:
    aggregate = report["aggregate"]
    decision = report["decision"]
    readiness = report["readiness"]
    lines = [
        "# Music Transformer Solo Yield Interval Contour Aftercare Sweep",
        "",
        "## Summary",
        "",
        f"- candidate count: `{aggregate['candidate_count']}`",
        f"- repaired MIDI count: `{aggregate['repaired_midi_count']}`",
        f"- wide interval review: `{aggregate['wide_interval_review_count_before']} -> {aggregate['wide_interval_review_count_after']}`",
        f"- adjusted note total: `{aggregate['adjusted_note_total']}`",
        f"- max pitch shift: `{aggregate['max_pitch_shift']}`",
        f"- chord-tone ratio decrease count: `{aggregate['chord_tone_ratio_decrease_count']}`",
        f"- MIDI low chord-tone ratio after: `{aggregate['midi_low_chord_tone_ratio_count_after']}`",
        f"- low note count after: `{aggregate['low_note_count_after']}`",
        f"- dead-air aftercare after: `{aggregate['dead_air_aftercare_count_after']}`",
        f"- weak direction-change after: `{aggregate['weak_direction_change_count_after']}`",
        f"- final landing not chord-tone after: `{aggregate['final_landing_not_chord_tone_count_after']}`",
        f"- chord-tone ratio min/avg: `{float(aggregate['midi_chord_tone_ratio_min']):.4f}` / `{float(aggregate['midi_chord_tone_ratio_avg']):.4f}`",
        f"- target supported: `{_bool_token(aggregate['target_supported'])}`",
        f"- selected next target: `{decision['selected_next_target']}`",
        f"- next boundary: `{decision['next_boundary']}`",
        f"- musical quality claimed: `{_bool_token(readiness['musical_quality_claimed'])}`",
        "",
        "## Candidate Repairs",
        "",
    ]
    for row in report.get("candidate_repairs", []):
        repair = row["repair"]
        before = row["before_profile"]
        after = row["after_profile"]
        lines.extend(
            [
                f"- candidate `{row['review_index']}` / `{row['case_label']}`",
                f"  - max interval: `{before['midi_max_abs_interval']}` -> `{after['midi_max_abs_interval']}`",
                f"  - note count: `{before['midi_note_count']}` -> `{after['midi_note_count']}`",
                f"  - chord-tone ratio: `{float(before['midi_chord_tone_ratio']):.4f}` -> `{float(after['midi_chord_tone_ratio']):.4f}`",
                f"  - direction-change ratio: `{float(before['midi_direction_change_ratio']):.4f}` -> `{float(after['midi_direction_change_ratio']):.4f}`",
                f"  - adjusted notes: `{repair['adjusted_note_count']}`",
                f"  - max pitch shift: `{repair['max_pitch_shift']}`",
                f"  - final landing preserved: `{_bool_token(repair['final_landing_preserved'])}`",
                f"  - repaired MIDI: `{row['repaired_midi_path']}`",
            ]
        )
    lines.extend(["", "## Not Proven", ""])
    for item in report.get("not_proven", []):
        lines.append(f"- `{item}`")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run interval contour aftercare repair sweep")
    parser.add_argument(
        "--source_repair_sweep_report",
        type=str,
        default=(
            "outputs/music_transformer_finetune_mvp/solo_yield_density_aftercare/"
            "issue_1294_density_aftercare_sweep/density_aftercare_sweep.json"
        ),
    )
    parser.add_argument(
        "--objective_decision_report",
        type=str,
        default=(
            "outputs/music_transformer_finetune_mvp/solo_yield_density_aftercare_objective_next/"
            "issue_1302_density_aftercare_objective_next/"
            "density_aftercare_objective_next_decision.json"
        ),
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/music_transformer_finetune_mvp/solo_yield_interval_contour_aftercare",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--max_interval", type=int, default=8)
    parser.add_argument("--max_pitch_shift", type=int, default=2)
    parser.add_argument("--pitch_min", type=int, default=55)
    parser.add_argument("--pitch_max", type=int, default=84)
    parser.add_argument("--min_chord_tone_ratio", type=float, default=0.50)
    parser.add_argument("--min_direction_change", type=float, default=0.50)
    parser.add_argument("--min_note_count", type=int, default=30)
    parser.add_argument("--min_candidates", type=int, default=8)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    report = build_repair_sweep(
        source_repair_sweep=read_json(Path(args.source_repair_sweep_report)),
        objective_decision=read_json(Path(args.objective_decision_report)),
        output_dir=output_dir,
        max_interval=int(args.max_interval),
        max_pitch_shift=int(args.max_pitch_shift),
        pitch_min=int(args.pitch_min),
        pitch_max=int(args.pitch_max),
        min_chord_tone_ratio=float(args.min_chord_tone_ratio),
        min_direction_change=float(args.min_direction_change),
        min_note_count=int(args.min_note_count),
    )
    summary = validate_report(report, min_candidates=int(args.min_candidates))
    if args.doc_path:
        write_text(Path(args.doc_path), markdown_report(report))
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
