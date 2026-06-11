"""Repair low chord-role balance after phrase-direction repair."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pretty_midi

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from inference.app.fallback import parse_chord  # noqa: E402
from scripts.assess_stage_b_generic_base_readiness import write_json, write_text  # noqa: E402
from scripts.decide_music_transformer_solo_yield_phrase_direction_objective_next import (  # noqa: E402
    SCHEMA_VERSION as OBJECTIVE_DECISION_SCHEMA_VERSION,
)
from scripts.review_music_transformer_solo_yield_repaired_top8_objective_failures import (  # noqa: E402
    midi_profile,
)
from scripts.run_music_transformer_solo_yield_phrase_direction_repair_sweep import (  # noqa: E402
    SCHEMA_VERSION as SOURCE_REPAIR_SWEEP_SCHEMA_VERSION,
)


SCHEMA_VERSION = "music_transformer_solo_yield_chord_role_balance_repair_sweep_v1"
BOUNDARY = "music_transformer_solo_yield_chord_role_balance_repair_sweep"
NEXT_BOUNDARY = "music_transformer_solo_yield_chord_role_balance_repair_audio_package"
SELECTED_TARGET = "chord_role_balance_repair_audio_package"


class SoloYieldChordRoleBalanceRepairSweepError(ValueError):
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
        raise SoloYieldChordRoleBalanceRepairSweepError(f"json not found: {path}")
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
        raise SoloYieldChordRoleBalanceRepairSweepError(f"unexpected quality claim: {claimed}")


def _all_notes(midi: pretty_midi.PrettyMIDI) -> list[pretty_midi.Note]:
    notes: list[pretty_midi.Note] = []
    for instrument in midi.instruments:
        if not instrument.is_drum:
            notes.extend(instrument.notes)
    return sorted(notes, key=lambda note: (float(note.start), int(note.pitch), float(note.end)))


def direction_change_ratio(pitches: list[int]) -> float:
    if len(pitches) < 3:
        return 0.0
    signs: list[int] = []
    for left, right in zip(pitches, pitches[1:]):
        delta = int(right) - int(left)
        if delta > 0:
            signs.append(1)
        elif delta < 0:
            signs.append(-1)
    if len(signs) < 2:
        return 0.0
    changes = sum(1 for left, right in zip(signs, signs[1:]) if left != right)
    return changes / max(1, len(signs) - 1)


def max_abs_interval(pitches: list[int]) -> int:
    return max((abs(int(right) - int(left)) for left, right in zip(pitches, pitches[1:])), default=0)


def _chord_tone_pcs(chord: str) -> set[int]:
    root_pc, intervals = parse_chord(chord)
    return {(root_pc + interval) % 12 for interval in intervals}


def _chord_index(start: float, *, phrase_start: float, phrase_duration: float, chord_count: int) -> int:
    if chord_count <= 1 or phrase_duration <= 0:
        return 0
    ratio = max(0.0, min(0.999999, (float(start) - phrase_start) / phrase_duration))
    return min(chord_count - 1, int(ratio * chord_count))


def chord_contexts(starts: list[float], ends: list[float], chords: list[str]) -> list[str]:
    phrase_start = min(starts, default=0.0)
    phrase_duration = max(1e-6, max(ends, default=phrase_start) - phrase_start)
    return [
        chords[
            _chord_index(
                start,
                phrase_start=phrase_start,
                phrase_duration=phrase_duration,
                chord_count=len(chords),
            )
        ]
        for start in starts
    ]


def chord_tone_ratio(pitches: list[int], note_chords: list[str]) -> float:
    if not pitches:
        return 0.0
    chord_tone_count = 0
    for pitch, chord in zip(pitches, note_chords):
        if int(pitch) % 12 in _chord_tone_pcs(chord):
            chord_tone_count += 1
    return chord_tone_count / max(1, len(pitches))


def chord_tone_shift_options(
    pitch: int,
    chord: str,
    *,
    max_shift: int,
    pitch_min: int,
    pitch_max: int,
) -> list[int]:
    chord_pcs = _chord_tone_pcs(chord)
    options = [
        candidate
        for candidate in range(int(pitch) - int(max_shift), int(pitch) + int(max_shift) + 1)
        if candidate != int(pitch)
        and int(pitch_min) <= candidate <= int(pitch_max)
        and candidate % 12 in chord_pcs
    ]
    return sorted(options, key=lambda value: (abs(value - int(pitch)), value))


def repair_chord_role_pitches(
    pitches: list[int],
    note_chords: list[str],
    *,
    min_chord_tone_ratio: float,
    min_direction_change: float,
    max_interval: int,
    max_shift: int,
    pitch_min: int,
    pitch_max: int,
) -> tuple[list[int], list[dict[str, Any]]]:
    repaired = [int(pitch) for pitch in pitches]
    changes: list[dict[str, Any]] = []
    for _ in range(len(repaired)):
        base_ratio = chord_tone_ratio(repaired, note_chords)
        if base_ratio >= float(min_chord_tone_ratio):
            break
        base_direction = direction_change_ratio(repaired)
        candidates: list[tuple[tuple[Any, ...], int, int, list[int], float, float, int]] = []
        for note_index, (pitch, chord) in enumerate(zip(repaired[:-1], note_chords[:-1])):
            if int(pitch) % 12 in _chord_tone_pcs(chord):
                continue
            for target_pitch in chord_tone_shift_options(
                int(pitch),
                chord,
                max_shift=int(max_shift),
                pitch_min=int(pitch_min),
                pitch_max=int(pitch_max),
            ):
                trial = list(repaired)
                trial[note_index] = int(target_pitch)
                trial_ratio = chord_tone_ratio(trial, note_chords)
                if trial_ratio <= base_ratio + 1e-9:
                    continue
                trial_direction = direction_change_ratio(trial)
                if trial_direction + 1e-9 < float(min_direction_change):
                    continue
                trial_interval = max_abs_interval(trial)
                if trial_interval > int(max_interval):
                    continue
                score = (
                    trial_ratio,
                    trial_direction >= base_direction - 1e-9,
                    trial_direction,
                    -trial_interval,
                    -abs(int(target_pitch) - int(pitch)),
                    -note_index,
                )
                candidates.append(
                    (
                        score,
                        note_index,
                        int(target_pitch),
                        trial,
                        trial_ratio,
                        trial_direction,
                        trial_interval,
                    )
                )
        if not candidates:
            break
        _, note_index, target_pitch, repaired, trial_ratio, trial_direction, trial_interval = max(
            candidates,
            key=lambda item: item[0],
        )
        changes.append(
            {
                "note_index": int(note_index),
                "original_pitch": int(pitches[note_index]),
                "repaired_pitch": int(target_pitch),
                "pitch_shift": int(target_pitch) - int(pitches[note_index]),
                "chord": note_chords[note_index],
                "chord_tone_ratio_after_shift": float(trial_ratio),
                "direction_change_ratio_after_shift": float(trial_direction),
                "max_abs_interval_after_shift": int(trial_interval),
            }
        )
    return repaired, changes


def _target_midi_path(output_dir: Path, row: dict[str, Any]) -> Path:
    review_index = _int(row.get("review_index"))
    case_label = str(row.get("case_label") or "candidate")
    return output_dir / "midi" / f"candidate_{review_index:02d}_{case_label}_chord_role_balance_repair.mid"


def validate_sources(source_repair_sweep: dict[str, Any], objective_decision: dict[str, Any]) -> list[dict[str, Any]]:
    if str(source_repair_sweep.get("schema_version") or "") != SOURCE_REPAIR_SWEEP_SCHEMA_VERSION:
        raise SoloYieldChordRoleBalanceRepairSweepError("source repair sweep schema required")
    if str(objective_decision.get("schema_version") or "") != OBJECTIVE_DECISION_SCHEMA_VERSION:
        raise SoloYieldChordRoleBalanceRepairSweepError("objective decision schema required")
    _require_no_quality_claim(source_repair_sweep)
    _require_no_quality_claim(objective_decision)
    decision = _dict(objective_decision.get("decision"))
    if str(decision.get("selected_next_target") or "") != "chord_role_balance_repair":
        raise SoloYieldChordRoleBalanceRepairSweepError("chord role balance target required")
    rows = [_dict(row) for row in _list(source_repair_sweep.get("candidate_repairs"))]
    if not rows:
        raise SoloYieldChordRoleBalanceRepairSweepError("source repair rows required")
    return rows


def repair_candidate(
    row: dict[str, Any],
    *,
    output_dir: Path,
    min_chord_tone_ratio: float,
    min_direction_change: float,
    max_interval: int,
    max_shift: int,
    pitch_min: int,
    pitch_max: int,
) -> dict[str, Any]:
    source_path = Path(str(row.get("repaired_midi_path") or ""))
    if not source_path.exists():
        raise SoloYieldChordRoleBalanceRepairSweepError(f"source MIDI missing: {source_path}")
    before_profile = _dict(row.get("after_profile"))
    target_path = _target_midi_path(output_dir, row)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    low_chord_role = _float(before_profile.get("midi_chord_tone_ratio")) < float(min_chord_tone_ratio)
    changes: list[dict[str, Any]] = []
    changed_note_count = 0

    if low_chord_role:
        midi = pretty_midi.PrettyMIDI(str(source_path))
        notes = _all_notes(midi)
        pitches = [int(note.pitch) for note in notes]
        starts = [float(note.start) for note in notes]
        ends = [float(note.end) for note in notes]
        chords = [item.strip() for item in str(row.get("chords") or "").split(",") if item.strip()]
        note_chords = chord_contexts(starts, ends, chords)
        repaired_pitches, changes = repair_chord_role_pitches(
            pitches,
            note_chords,
            min_chord_tone_ratio=float(min_chord_tone_ratio),
            min_direction_change=float(min_direction_change),
            max_interval=int(max_interval),
            max_shift=int(max_shift),
            pitch_min=int(pitch_min),
            pitch_max=int(pitch_max),
        )
        for note, pitch in zip(notes, repaired_pitches):
            note.pitch = int(pitch)
        changed_note_count = sum(1 for left, right in zip(pitches, repaired_pitches) if int(left) != int(right))
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
            "low_chord_role_before": bool(low_chord_role),
            "low_chord_role_after": _float(after_profile.get("midi_chord_tone_ratio")) < float(min_chord_tone_ratio),
            "changed_note_count": int(changed_note_count),
            "pitch_shifts": changes,
            "chord_tone_ratio_delta": (
                _float(after_profile.get("midi_chord_tone_ratio"))
                - _float(before_profile.get("midi_chord_tone_ratio"))
            ),
            "direction_change_after_valid": _float(after_profile.get("midi_direction_change_ratio"))
            >= float(min_direction_change),
            "max_interval_after_valid": _int(after_profile.get("midi_max_abs_interval")) <= int(max_interval),
            "final_landing_preserved": bool(after_profile.get("final_landing_is_chord_tone", False)),
        },
    }


def build_repair_sweep(
    *,
    source_repair_sweep: dict[str, Any],
    objective_decision: dict[str, Any],
    output_dir: Path,
    min_chord_tone_ratio: float = 0.50,
    min_direction_change: float = 0.50,
    max_interval: int = 8,
    max_shift: int = 2,
    pitch_min: int = 55,
    pitch_max: int = 84,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = validate_sources(source_repair_sweep, objective_decision)
    repairs = [
        repair_candidate(
            row,
            output_dir=output_dir,
            min_chord_tone_ratio=float(min_chord_tone_ratio),
            min_direction_change=float(min_direction_change),
            max_interval=int(max_interval),
            max_shift=int(max_shift),
            pitch_min=int(pitch_min),
            pitch_max=int(pitch_max),
        )
        for row in rows
    ]
    low_before = sum(1 for row in repairs if bool(_dict(row.get("repair")).get("low_chord_role_before")))
    low_after = sum(1 for row in repairs if bool(_dict(row.get("repair")).get("low_chord_role_after")))
    chord_ratio_decrease = sum(
        1 for row in repairs if _float(_dict(row.get("repair")).get("chord_tone_ratio_delta")) < -1e-9
    )
    weak_direction_after = sum(
        1 for row in repairs if _float(_dict(row.get("after_profile")).get("midi_direction_change_ratio")) < float(min_direction_change)
    )
    final_landing_not = sum(
        1 for row in repairs if not bool(_dict(row.get("after_profile")).get("final_landing_is_chord_tone", False))
    )
    wide_before = sum(
        1 for row in repairs if _int(_dict(row.get("before_profile")).get("midi_max_abs_interval")) >= int(max_interval)
    )
    wide_after = sum(
        1 for row in repairs if _int(_dict(row.get("after_profile")).get("midi_max_abs_interval")) >= int(max_interval)
    )
    changed_total = sum(_int(_dict(row.get("repair")).get("changed_note_count")) for row in repairs)
    max_shift_seen = max(
        (
            abs(_int(shift.get("pitch_shift")))
            for row in repairs
            for shift in _list(_dict(row.get("repair")).get("pitch_shifts"))
        ),
        default=0,
    )
    target_supported = bool(
        low_before > low_after
        and low_after == 0
        and chord_ratio_decrease == 0
        and weak_direction_after == 0
        and final_landing_not == 0
        and wide_after <= wide_before
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
            "low_chord_role_count_before": int(low_before),
            "low_chord_role_count_after": int(low_after),
            "low_chord_role_delta": int(low_before - low_after),
            "changed_note_total": int(changed_total),
            "max_abs_pitch_shift": int(max_shift_seen),
            "chord_tone_ratio_decrease_count": int(chord_ratio_decrease),
            "weak_direction_change_count_after": int(weak_direction_after),
            "final_landing_not_chord_tone_count_after": int(final_landing_not),
            "wide_interval_review_count_before": int(wide_before),
            "wide_interval_review_count_after": int(wide_after),
            "target_supported": bool(target_supported),
        },
        "readiness": {
            "chord_role_balance_repair_sweep_completed": True,
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
            "reason": "low chord-role ratio reduced by small pitch shifts while preserving objective guards",
        },
        "not_proven": [
            "human_audio_preference",
            "stable_jazz_solo_quality",
            "artist_level_long_solo_generation",
            "production_ready_improviser",
        ],
    }
    write_json(output_dir / "chord_role_balance_repair_sweep.json", report)
    write_json(output_dir / "chord_role_balance_repair_sweep_summary.json", validate_report(report))
    write_text(output_dir / "chord_role_balance_repair_sweep.md", markdown_report(report))
    return report


def validate_report(report: dict[str, Any], *, min_candidates: int = 1) -> dict[str, Any]:
    if str(report.get("schema_version") or "") != SCHEMA_VERSION:
        raise SoloYieldChordRoleBalanceRepairSweepError("schema version mismatch")
    readiness = _dict(report.get("readiness"))
    aggregate = _dict(report.get("aggregate"))
    decision = _dict(report.get("decision"))
    candidate_count = _int(aggregate.get("candidate_count"))
    if candidate_count < int(min_candidates):
        raise SoloYieldChordRoleBalanceRepairSweepError("candidate count below requirement")
    if not bool(readiness.get("chord_role_balance_repair_sweep_completed", False)):
        raise SoloYieldChordRoleBalanceRepairSweepError("repair sweep completion required")
    _require_no_quality_claim({"readiness": readiness})
    return {
        "schema_version": str(report.get("schema_version") or ""),
        "candidate_count": candidate_count,
        "repaired_midi_count": _int(aggregate.get("repaired_midi_count")),
        "low_chord_role_count_before": _int(aggregate.get("low_chord_role_count_before")),
        "low_chord_role_count_after": _int(aggregate.get("low_chord_role_count_after")),
        "low_chord_role_delta": _int(aggregate.get("low_chord_role_delta")),
        "changed_note_total": _int(aggregate.get("changed_note_total")),
        "max_abs_pitch_shift": _int(aggregate.get("max_abs_pitch_shift")),
        "chord_tone_ratio_decrease_count": _int(aggregate.get("chord_tone_ratio_decrease_count")),
        "weak_direction_change_count_after": _int(aggregate.get("weak_direction_change_count_after")),
        "final_landing_not_chord_tone_count_after": _int(
            aggregate.get("final_landing_not_chord_tone_count_after")
        ),
        "wide_interval_review_count_before": _int(aggregate.get("wide_interval_review_count_before")),
        "wide_interval_review_count_after": _int(aggregate.get("wide_interval_review_count_after")),
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
        "# Music Transformer Solo Yield Chord Role Balance Repair Sweep",
        "",
        "## Summary",
        "",
        f"- candidate count: `{aggregate['candidate_count']}`",
        f"- repaired MIDI count: `{aggregate['repaired_midi_count']}`",
        f"- low chord-role count: `{aggregate['low_chord_role_count_before']} -> {aggregate['low_chord_role_count_after']}`",
        f"- changed note total: `{aggregate['changed_note_total']}`",
        f"- max abs pitch shift: `{aggregate['max_abs_pitch_shift']}`",
        f"- chord-tone ratio decrease count: `{aggregate['chord_tone_ratio_decrease_count']}`",
        f"- weak direction-change after: `{aggregate['weak_direction_change_count_after']}`",
        f"- final landing not chord-tone after: `{aggregate['final_landing_not_chord_tone_count_after']}`",
        f"- wide interval review: `{aggregate['wide_interval_review_count_before']} -> {aggregate['wide_interval_review_count_after']}`",
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
                f"  - chord-tone ratio: `{float(before['midi_chord_tone_ratio']):.4f}` -> `{float(after['midi_chord_tone_ratio']):.4f}`",
                f"  - direction-change ratio: `{float(before['midi_direction_change_ratio']):.4f}` -> `{float(after['midi_direction_change_ratio']):.4f}`",
                f"  - max interval: `{before['midi_max_abs_interval']}` -> `{after['midi_max_abs_interval']}`",
                f"  - changed notes: `{repair['changed_note_count']}`",
                f"  - final landing preserved: `{_bool_token(repair['final_landing_preserved'])}`",
                f"  - repaired MIDI: `{row['repaired_midi_path']}`",
            ]
        )
    lines.extend(["", "## Not Proven", ""])
    for item in report.get("not_proven", []):
        lines.append(f"- `{item}`")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run chord role balance repair sweep")
    parser.add_argument(
        "--source_repair_sweep_report",
        type=str,
        default=(
            "outputs/music_transformer_finetune_mvp/solo_yield_phrase_direction_repair/"
            "issue_1274_phrase_direction_repair_sweep/phrase_direction_repair_sweep.json"
        ),
    )
    parser.add_argument(
        "--objective_decision_report",
        type=str,
        default=(
            "outputs/music_transformer_finetune_mvp/solo_yield_phrase_direction_repair_objective_next/"
            "issue_1282_phrase_direction_objective_next/"
            "phrase_direction_objective_next_decision.json"
        ),
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/music_transformer_finetune_mvp/solo_yield_chord_role_balance_repair",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--min_candidates", type=int, default=8)
    parser.add_argument("--min_chord_tone_ratio", type=float, default=0.50)
    parser.add_argument("--min_direction_change", type=float, default=0.50)
    parser.add_argument("--max_interval", type=int, default=8)
    parser.add_argument("--max_shift", type=int, default=2)
    parser.add_argument("--pitch_min", type=int, default=55)
    parser.add_argument("--pitch_max", type=int, default=84)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    report = build_repair_sweep(
        source_repair_sweep=read_json(Path(args.source_repair_sweep_report)),
        objective_decision=read_json(Path(args.objective_decision_report)),
        output_dir=output_dir,
        min_chord_tone_ratio=float(args.min_chord_tone_ratio),
        min_direction_change=float(args.min_direction_change),
        max_interval=int(args.max_interval),
        max_shift=int(args.max_shift),
        pitch_min=int(args.pitch_min),
        pitch_max=int(args.pitch_max),
    )
    summary = validate_report(report, min_candidates=int(args.min_candidates))
    write_json(output_dir / "chord_role_balance_repair_sweep_summary.json", summary)
    if args.doc_path:
        write_text(Path(args.doc_path), markdown_report(report))
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
