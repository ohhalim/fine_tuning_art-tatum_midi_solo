"""Repair final chord-tone landing for repaired top8 solo-yield candidates."""

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

from inference.app.fallback import parse_chord  # noqa: E402
from scripts.assess_stage_b_generic_base_readiness import write_json, write_text  # noqa: E402
from scripts.review_music_transformer_solo_yield_repaired_top8_objective_failures import (  # noqa: E402
    SCHEMA_VERSION as OBJECTIVE_REVIEW_SCHEMA_VERSION,
    build_objective_failure_review,
    midi_profile,
    read_json,
)


SCHEMA_VERSION = "music_transformer_solo_yield_chord_tone_landing_repair_sweep_v1"
BOUNDARY = "music_transformer_solo_yield_chord_tone_landing_repair_sweep"
NEXT_BOUNDARY = "music_transformer_solo_yield_chord_tone_landing_repair_audio_package"
SELECTED_TARGET = "chord_tone_landing_repair_audio_package"


class SoloYieldChordToneLandingRepairSweepError(ValueError):
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


def _require_no_quality_claim(report: dict[str, Any]) -> None:
    readiness = _dict(report.get("readiness"))
    claimed = [
        key
        for key in ("musical_quality_claimed", "artist_style_claimed", "production_ready_claimed")
        if bool(readiness.get(key, False))
    ]
    if claimed:
        raise SoloYieldChordToneLandingRepairSweepError(f"unexpected quality claim: {claimed}")


def _quality_safe_source(package_report: dict[str, Any], objective_review: dict[str, Any]) -> None:
    if str(package_report.get("schema_version") or "") != "music_transformer_solo_yield_listening_package_v1":
        raise SoloYieldChordToneLandingRepairSweepError("listening package schema required")
    if str(objective_review.get("schema_version") or "") != OBJECTIVE_REVIEW_SCHEMA_VERSION:
        raise SoloYieldChordToneLandingRepairSweepError("objective failure review schema required")
    _require_no_quality_claim(package_report)
    _require_no_quality_claim(objective_review)
    decision = _dict(objective_review.get("decision"))
    if str(decision.get("selected_next_target") or "") != "chord_tone_landing_repair":
        raise SoloYieldChordToneLandingRepairSweepError("chord-tone landing repair target required")


def _chord_tone_pitches_near(pitch: int, chord: str) -> list[int]:
    root_pc, intervals = parse_chord(chord)
    pitch_classes = {(root_pc + interval) % 12 for interval in intervals}
    candidates = [
        octave * 12 + pitch_class
        for octave in range(0, 11)
        for pitch_class in pitch_classes
        if 21 <= octave * 12 + pitch_class <= 108
    ]
    return sorted(candidates, key=lambda value: (abs(value - int(pitch)), value))


def nearest_chord_tone_pitch(pitch: int, chord: str) -> int:
    candidates = _chord_tone_pitches_near(int(pitch), chord)
    return int(candidates[0]) if candidates else int(pitch)


def _all_notes(midi: pretty_midi.PrettyMIDI) -> list[pretty_midi.Note]:
    notes: list[pretty_midi.Note] = []
    for instrument in midi.instruments:
        if not instrument.is_drum:
            notes.extend(instrument.notes)
    return sorted(notes, key=lambda note: (float(note.start), int(note.pitch), float(note.end)))


def _target_midi_path(output_dir: Path, candidate: dict[str, Any]) -> Path:
    review_index = _int(candidate.get("review_index"))
    case_label = str(candidate.get("case_label") or "candidate")
    return output_dir / "midi" / f"candidate_{review_index:02d}_{case_label}_chord_tone_landing_repair.mid"


def repair_candidate_final_landing(
    candidate: dict[str, Any],
    *,
    output_dir: Path,
) -> dict[str, Any]:
    source_path = Path(str(candidate.get("review_midi_path") or ""))
    if not source_path.exists():
        raise SoloYieldChordToneLandingRepairSweepError(f"source MIDI missing: {source_path}")
    before_profile = midi_profile(candidate)
    target_chord = str(before_profile.get("final_landing_chord") or "")
    original_pitch = before_profile.get("final_landing_pitch")
    if original_pitch is None:
        raise SoloYieldChordToneLandingRepairSweepError("final landing pitch required")
    repaired_pitch = nearest_chord_tone_pitch(int(original_pitch), target_chord)

    midi = pretty_midi.PrettyMIDI(str(source_path))
    notes = _all_notes(midi)
    if not notes:
        raise SoloYieldChordToneLandingRepairSweepError("source MIDI note required")
    final_note = notes[-1]
    final_note.pitch = int(repaired_pitch)
    target_path = _target_midi_path(output_dir, candidate)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    midi.write(str(target_path))

    repaired_candidate = dict(candidate)
    repaired_candidate["review_midi_path"] = str(target_path)
    after_profile = midi_profile(repaired_candidate)
    pitch_shift = int(repaired_pitch) - int(original_pitch)

    return {
        "review_index": _int(candidate.get("review_index")),
        "case_label": str(candidate.get("case_label") or ""),
        "chords": str(candidate.get("chords") or ""),
        "source_midi_path": str(source_path),
        "repaired_midi_path": str(target_path),
        "source_wav_path": str(candidate.get("review_wav_path") or ""),
        "before_profile": before_profile,
        "after_profile": after_profile,
        "repair": {
            "changed_note_count": int(pitch_shift != 0),
            "original_final_pitch": int(original_pitch),
            "repaired_final_pitch": int(repaired_pitch),
            "pitch_shift": int(pitch_shift),
            "abs_pitch_shift": abs(int(pitch_shift)),
            "target_chord": target_chord,
        },
    }


def build_repair_sweep(
    *,
    package_report: dict[str, Any],
    objective_review: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    _quality_safe_source(package_report, objective_review)
    candidates = [_dict(item) for item in _list(package_report.get("candidates"))]
    if not candidates:
        raise SoloYieldChordToneLandingRepairSweepError("candidate list required")

    repairs = [
        repair_candidate_final_landing(candidate, output_dir=output_dir)
        for candidate in candidates
    ]
    before_not = sum(
        1 for row in repairs if not bool(_dict(row.get("before_profile")).get("final_landing_is_chord_tone"))
    )
    after_not = sum(
        1 for row in repairs if not bool(_dict(row.get("after_profile")).get("final_landing_is_chord_tone"))
    )
    changed_total = sum(_int(_dict(row.get("repair")).get("changed_note_count")) for row in repairs)
    max_shift = max((_int(_dict(row.get("repair")).get("abs_pitch_shift")) for row in repairs), default=0)
    target_supported = before_not > after_not and after_not == 0
    next_boundary = NEXT_BOUNDARY if target_supported else "music_transformer_solo_yield_objective_repair_decision"

    report = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "boundary": BOUNDARY,
        "source_package": {
            "schema_version": package_report.get("schema_version"),
            "output_dir": package_report.get("output_dir"),
            "candidate_count": _int(package_report.get("candidate_count")),
        },
        "source_objective_review": {
            "schema_version": objective_review.get("schema_version"),
            "output_dir": objective_review.get("output_dir"),
            "selected_next_target": str(_dict(objective_review.get("decision")).get("selected_next_target") or ""),
        },
        "candidate_repairs": repairs,
        "aggregate": {
            "candidate_count": len(repairs),
            "repaired_midi_count": len(repairs),
            "changed_note_total": int(changed_total),
            "max_abs_pitch_shift": int(max_shift),
            "final_landing_not_chord_tone_count_before": int(before_not),
            "final_landing_not_chord_tone_count_after": int(after_not),
            "final_landing_not_chord_tone_delta": int(before_not - after_not),
            "target_supported": bool(target_supported),
        },
        "readiness": {
            "chord_tone_landing_repair_sweep_completed": True,
            "target_supported": bool(target_supported),
            "validated_listening_input_present": False,
            "musical_quality_claimed": False,
            "artist_style_claimed": False,
            "production_ready_claimed": False,
        },
        "decision": {
            "current_boundary": BOUNDARY,
            "selected_next_target": SELECTED_TARGET if target_supported else "objective_repair_decision",
            "next_boundary": next_boundary,
            "critical_user_input_required": False,
            "reason": "final landing chord-tone risk repaired by nearest chord-tone pitch shift without quality claim",
        },
        "not_proven": [
            "human_audio_preference",
            "stable_jazz_solo_quality",
            "artist_level_long_solo_generation",
            "production_ready_improviser",
        ],
    }
    write_json(output_dir / "chord_tone_landing_repair_sweep.json", report)
    write_json(output_dir / "chord_tone_landing_repair_sweep_summary.json", validate_report(report))
    write_text(output_dir / "chord_tone_landing_repair_sweep.md", markdown_report(report))
    return report


def validate_report(report: dict[str, Any], *, min_candidates: int = 1) -> dict[str, Any]:
    if str(report.get("schema_version") or "") != SCHEMA_VERSION:
        raise SoloYieldChordToneLandingRepairSweepError("schema version mismatch")
    readiness = _dict(report.get("readiness"))
    aggregate = _dict(report.get("aggregate"))
    decision = _dict(report.get("decision"))
    candidate_count = _int(aggregate.get("candidate_count"))
    if candidate_count < int(min_candidates):
        raise SoloYieldChordToneLandingRepairSweepError("candidate count below requirement")
    if not bool(readiness.get("chord_tone_landing_repair_sweep_completed", False)):
        raise SoloYieldChordToneLandingRepairSweepError("repair sweep completion required")
    claimed = [
        key
        for key in ("musical_quality_claimed", "artist_style_claimed", "production_ready_claimed")
        if bool(readiness.get(key, True))
    ]
    if claimed:
        raise SoloYieldChordToneLandingRepairSweepError(f"unexpected quality claim: {claimed}")
    return {
        "schema_version": str(report.get("schema_version") or ""),
        "candidate_count": candidate_count,
        "repaired_midi_count": _int(aggregate.get("repaired_midi_count")),
        "changed_note_total": _int(aggregate.get("changed_note_total")),
        "max_abs_pitch_shift": _int(aggregate.get("max_abs_pitch_shift")),
        "final_landing_not_chord_tone_count_before": _int(
            aggregate.get("final_landing_not_chord_tone_count_before")
        ),
        "final_landing_not_chord_tone_count_after": _int(
            aggregate.get("final_landing_not_chord_tone_count_after")
        ),
        "final_landing_not_chord_tone_delta": _int(
            aggregate.get("final_landing_not_chord_tone_delta")
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
        "# Music Transformer Solo Yield Chord-Tone Landing Repair Sweep",
        "",
        "## Summary",
        "",
        f"- candidate count: `{aggregate['candidate_count']}`",
        f"- repaired MIDI count: `{aggregate['repaired_midi_count']}`",
        f"- changed note total: `{aggregate['changed_note_total']}`",
        f"- max abs pitch shift: `{aggregate['max_abs_pitch_shift']}`",
        f"- final landing not chord-tone: `{aggregate['final_landing_not_chord_tone_count_before']} -> {aggregate['final_landing_not_chord_tone_count_after']}`",
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
                f"  - final landing: `{before['final_landing_pitch']}` -> `{after['final_landing_pitch']}` over `{repair['target_chord']}`",
                f"  - pitch shift: `{repair['pitch_shift']}`",
                f"  - chord-tone landing: `{_bool_token(before['final_landing_is_chord_tone'])}` -> `{_bool_token(after['final_landing_is_chord_tone'])}`",
                f"  - repaired MIDI: `{row['repaired_midi_path']}`",
            ]
        )
    lines.extend(["", "## Not Proven", ""])
    for item in report.get("not_proven", []):
        lines.append(f"- `{item}`")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run solo-yield chord-tone landing repair sweep")
    parser.add_argument(
        "--package_report",
        type=str,
        default=(
            "outputs/music_transformer_finetune_mvp/solo_yield_listening_review/"
            "issue_1250_4bar_repaired_top8_listening_package/listening_review_package.json"
        ),
    )
    parser.add_argument(
        "--objective_review_report",
        type=str,
        default=(
            "outputs/music_transformer_finetune_mvp/solo_yield_objective_failure_review/"
            "issue_1262_repaired_top8_objective_review/objective_failure_review.json"
        ),
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/music_transformer_finetune_mvp/solo_yield_chord_tone_landing_repair",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--min_candidates", type=int, default=8)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    package_report = read_json(Path(args.package_report))
    objective_path = Path(args.objective_review_report)
    if objective_path.exists():
        objective_review = read_json(objective_path)
    else:
        objective_review = build_objective_failure_review(
            package_report,
            output_dir=output_dir / "source_objective_review",
        )
    report = build_repair_sweep(
        package_report=package_report,
        objective_review=objective_review,
        output_dir=output_dir,
    )
    summary = validate_report(report, min_candidates=int(args.min_candidates))
    write_json(output_dir / "chord_tone_landing_repair_sweep_summary.json", summary)
    if args.doc_path:
        write_text(Path(args.doc_path), markdown_report(report))
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
