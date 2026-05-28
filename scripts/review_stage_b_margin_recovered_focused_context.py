"""Review a Stage B margin-recovered focused package against solo/context MIDI metrics."""

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

from scripts.build_stage_b_margin_recovered_focused_package import (  # noqa: E402
    max_simultaneous_notes,
    non_drum_notes,
    read_json,
)
from scripts.build_stage_b_margin_recovered_listening_notes import write_json  # noqa: E402


SCHEMA_VERSION = "stage_b_margin_recovered_focused_context_decision_v1"

ROOT_TO_PC = {
    "C": 0,
    "C#": 1,
    "Db": 1,
    "D": 2,
    "D#": 3,
    "Eb": 3,
    "E": 4,
    "F": 5,
    "F#": 6,
    "Gb": 6,
    "G": 7,
    "G#": 8,
    "Ab": 8,
    "A": 9,
    "A#": 10,
    "Bb": 10,
    "B": 11,
}

QUALITY_INTERVALS = {
    "maj7": {0, 4, 7, 11},
    "min7": {0, 3, 7, 10},
    "m7": {0, 3, 7, 10},
    "7": {0, 4, 7, 10},
    "dim7": {0, 3, 6, 9},
    "m7b5": {0, 3, 6, 10},
}

QUALITY_TENSIONS = {
    "maj7": {2, 6, 9},
    "min7": {2, 5, 9},
    "m7": {2, 5, 9},
    "7": {2, 5, 9},
    "dim7": {2},
    "m7b5": {2, 5},
}


class FocusedContextDecisionError(ValueError):
    pass


def pitch_name(pitch: int) -> str:
    names = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
    return f"{names[int(pitch) % 12]}{int(pitch) // 12 - 1}"


def infer_bpm(midi_path: Path, fallback_bpm: float = 124.0) -> float:
    midi = pretty_midi.PrettyMIDI(str(midi_path))
    _times, tempi = midi.get_tempo_changes()
    if len(tempi) == 0:
        return float(fallback_bpm)
    return float(tempi[0])


def seconds_to_beats(seconds: float, bpm: float) -> float:
    return float(seconds) * float(bpm) / 60.0


def parse_chord(chord: str) -> tuple[str, str]:
    chord = str(chord or "Cmaj7").strip()
    if len(chord) >= 2 and chord[1] in {"#", "b"}:
        return chord[:2], chord[2:] or "maj7"
    return chord[:1], chord[1:] or "maj7"


def chord_at_beat(beat: float, chords: list[str]) -> str:
    if not chords:
        return "Cmaj7"
    bar_index = max(0, int(float(beat) // 4.0))
    return chords[bar_index % len(chords)]


def pitch_role(pitch: int, chord: str) -> str:
    root, quality = parse_chord(chord)
    root_pc = ROOT_TO_PC.get(root, 0)
    chord_pcs = {(root_pc + interval) % 12 for interval in QUALITY_INTERVALS.get(quality, QUALITY_INTERVALS["maj7"])}
    tension_pcs = {(root_pc + interval) % 12 for interval in QUALITY_TENSIONS.get(quality, set())}
    pitch_pc = int(pitch) % 12
    if pitch_pc in chord_pcs:
        return "chord_tone"
    if pitch_pc in tension_pcs:
        return "tension"
    return "outside"


def duplicated_pitch_class_chunks(pitches: list[int], size: int) -> int:
    if len(pitches) < size:
        return 0
    chunks = [tuple(int(pitch) % 12 for pitch in pitches[index : index + size]) for index in range(len(pitches) - size + 1)]
    counts = Counter(chunks)
    return int(sum(count - 1 for count in counts.values() if count > 1))


def context_summary(context_path: Path) -> dict[str, Any]:
    if not context_path.exists():
        return {"context_exists": False, "instrument_count": 0, "instruments": []}
    midi = pretty_midi.PrettyMIDI(str(context_path))
    instruments = [
        {
            "name": str(instrument.name or ""),
            "program": int(instrument.program),
            "is_drum": bool(instrument.is_drum),
            "note_count": int(len(instrument.notes)),
        }
        for instrument in midi.instruments
    ]
    names = [item["name"].lower() for item in instruments]
    return {
        "context_exists": True,
        "instrument_count": int(len(instruments)),
        "instruments": instruments,
        "has_chord_guide": any("chord" in name for name in names),
        "has_bass_guide": any("bass" in name for name in names),
        "has_solo_track": any("solo" in name for name in names),
    }


def focused_context_flags(metrics: dict[str, Any], context: dict[str, Any]) -> list[str]:
    flags: list[str] = []
    if not context.get("context_exists"):
        flags.append("missing_context_midi")
    if not context.get("has_chord_guide"):
        flags.append("missing_chord_guide")
    if not context.get("has_bass_guide"):
        flags.append("missing_bass_guide")
    if not context.get("has_solo_track"):
        flags.append("missing_solo_track")
    if int(metrics.get("max_simultaneous_notes", 0) or 0) > 1:
        flags.append("polyphonic_solo_overlap")
    if int(metrics.get("note_count", 0) or 0) < 12:
        flags.append("too_sparse_for_context_review")
    if int(metrics.get("unique_pitch_count", 0) or 0) < 6:
        flags.append("low_pitch_variety")
    if float(metrics.get("dead_air_ratio", 0.0) or 0.0) > 0.40:
        flags.append("dead_air_needs_review")
    if float(metrics.get("phrase_span_beats", 0.0) or 0.0) < 6.0:
        flags.append("short_phrase_span")
    if float(metrics.get("max_end_beats", 0.0) or 0.0) > float(metrics.get("context_total_beats", 0.0) or 0.0) + 0.01:
        flags.append("solo_exceeds_context_guide")
    if str(metrics.get("final_note_role") or "") == "outside":
        flags.append("final_outside_context_chord")
    if int(metrics.get("duplicated_3_note_pitch_class_chunks", 0) or 0) > 0:
        flags.append("repeated_pitch_class_cell")
    return flags


def focused_context_decision(flags: list[str]) -> str:
    if flags:
        return "needs_followup"
    return "keep_for_focused_listening"


def analyze_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    files = candidate.get("review_files") if isinstance(candidate.get("review_files"), dict) else {}
    solo_path = Path(str(files.get("midi_path") or ""))
    context_path = Path(str(files.get("context_midi_path") or ""))
    if not solo_path.exists():
        raise FocusedContextDecisionError(f"solo MIDI does not exist: {solo_path}")
    transform = candidate.get("focused_package_transform") if isinstance(candidate.get("focused_package_transform"), dict) else {}
    chords = [str(chord) for chord in transform.get("context_chords", [])] or ["Cmaj7"]
    bpm = float(transform.get("context_bpm") or infer_bpm(solo_path))
    bars = int(transform.get("context_bars") or 2)
    notes = non_drum_notes(solo_path)
    if not notes:
        raise FocusedContextDecisionError(f"solo MIDI has no notes: {solo_path}")
    starts_beats = [seconds_to_beats(float(note.start), bpm) for note in notes]
    ends_beats = [seconds_to_beats(float(note.end), bpm) for note in notes]
    durations_beats = [max(0.0, end - start) for start, end in zip(starts_beats, ends_beats)]
    pitches = [int(note.pitch) for note in notes]
    intervals = [pitches[index + 1] - pitches[index] for index in range(len(pitches) - 1)]
    final_note = notes[-1]
    final_start_beats = starts_beats[-1]
    final_chord = chord_at_beat(final_start_beats, chords[:bars] or chords)
    final_role = pitch_role(int(final_note.pitch), final_chord)
    source_metrics = candidate.get("source_metrics") if isinstance(candidate.get("source_metrics"), dict) else {}
    metrics = {
        "note_count": int(len(notes)),
        "unique_pitch_count": int(len(set(pitches))),
        "pitch_min": int(min(pitches)),
        "pitch_max": int(max(pitches)),
        "range": f"{pitch_name(min(pitches))}-{pitch_name(max(pitches))}",
        "phrase_span_beats": round(float(max(ends_beats) - min(starts_beats)), 3),
        "max_end_beats": round(float(max(ends_beats)), 3),
        "context_total_beats": round(float(bars * 4.0), 3),
        "max_duration_beats": round(float(max(durations_beats)), 3),
        "max_interval": int(max((abs(interval) for interval in intervals), default=0)),
        "adjacent_pitch_repeats": int(sum(1 for interval in intervals if interval == 0)),
        "duplicated_3_note_pitch_class_chunks": duplicated_pitch_class_chunks(pitches, 3),
        "max_simultaneous_notes": max_simultaneous_notes(notes),
        "dead_air_ratio": float(source_metrics.get("dead_air_ratio", 0.0) or 0.0),
        "onset_coverage_ratio": float(source_metrics.get("onset_coverage_ratio", 0.0) or 0.0),
        "sustained_coverage_ratio": float(source_metrics.get("sustained_coverage_ratio", 0.0) or 0.0),
        "final_note": pitch_name(int(final_note.pitch)),
        "final_start_beats": round(float(final_start_beats), 3),
        "final_chord": final_chord,
        "final_note_role": final_role,
    }
    context = context_summary(context_path)
    flags = focused_context_flags(metrics, context)
    decision = focused_context_decision(flags)
    return {
        "candidate_id": str(candidate.get("candidate_id") or ""),
        "prior_proxy_decision": str(candidate.get("listening", {}).get("decision") or ""),
        "focused_context_decision": decision,
        "decision_flags": flags,
        "solo_midi_path": str(solo_path),
        "context_midi_path": str(context_path),
        "metrics": metrics,
        "context_summary": context,
        "rationale": rationale_for_decision(decision, flags, metrics, context),
    }


def rationale_for_decision(
    decision: str,
    flags: list[str],
    metrics: dict[str, Any],
    context: dict[str, Any],
) -> list[str]:
    if decision == "keep_for_focused_listening":
        return [
            "No focused context blocker was detected.",
            "Solo max simultaneous notes is 1.",
            "Context MIDI includes chord guide, bass guide, and solo track.",
        ]
    reasons = []
    if "low_pitch_variety" in flags:
        reasons.append(f"Unique pitch count is {metrics['unique_pitch_count']}, below the focused context keep threshold.")
    if "dead_air_needs_review" in flags:
        reasons.append(f"Dead-air ratio is {metrics['dead_air_ratio']:.3f}, so timing/space still needs review.")
    if "solo_exceeds_context_guide" in flags:
        reasons.append(
            f"Solo reaches {metrics['max_end_beats']:.3f} beats, beyond context guide {metrics['context_total_beats']:.3f} beats."
        )
    if "final_outside_context_chord" in flags:
        reasons.append(f"Final note {metrics['final_note']} is outside {metrics['final_chord']}.")
    if not context.get("has_chord_guide"):
        reasons.append("Context MIDI does not expose a chord guide track.")
    return reasons or ["Focused context flags require follow-up before listening-review promotion."]


def build_focused_context_decision(focused_package: dict[str, Any], *, output_dir: Path) -> dict[str, Any]:
    candidates = focused_package.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise FocusedContextDecisionError("focused package must contain non-empty candidates")
    reviewed = [analyze_candidate(candidate) for candidate in candidates]
    decision_counts = Counter(item["focused_context_decision"] for item in reviewed)
    flag_counts = Counter(flag for item in reviewed for flag in item["decision_flags"])
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "source_focused_package": str(focused_package.get("output_dir") or ""),
        "candidate_count": int(len(reviewed)),
        "decision_counts": dict(sorted(decision_counts.items())),
        "flag_counts": dict(sorted(flag_counts.items())),
        "candidates": reviewed,
    }


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Stage B Margin-Recovered Focused Context Decision",
        "",
        f"- candidate count: `{report['candidate_count']}`",
        f"- decisions: `{report['decision_counts']}`",
        f"- flags: `{report['flag_counts']}`",
        "",
        "This is a MIDI metric/context decision, not a human listening proof.",
        "",
        "| candidate | prior | decision | notes | unique | range | span | max active | dead-air | final | flags |",
        "|---|---|---|---:|---:|---|---:|---:|---:|---|---|",
    ]
    for candidate in report["candidates"]:
        metrics = candidate["metrics"]
        lines.append(
            "| "
            + " | ".join(
                [
                    candidate["candidate_id"],
                    candidate["prior_proxy_decision"],
                    candidate["focused_context_decision"],
                    str(metrics["note_count"]),
                    str(metrics["unique_pitch_count"]),
                    str(metrics["range"]),
                    f"{float(metrics['phrase_span_beats']):.3f}",
                    str(metrics["max_simultaneous_notes"]),
                    f"{float(metrics['dead_air_ratio']):.3f}",
                    f"{metrics['final_note']} over {metrics['final_chord']} ({metrics['final_note_role']})",
                    ",".join(candidate["decision_flags"]) if candidate["decision_flags"] else "ok",
                ]
            )
            + " |"
        )
    return "\n".join(lines).rstrip() + "\n"


def validate_decision(
    report: dict[str, Any],
    *,
    expected_candidate_id: str | None,
    expected_decision: str | None,
    expected_candidate_count: int | None = None,
) -> dict[str, Any]:
    candidates = report.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise FocusedContextDecisionError("report candidates must be non-empty")
    if expected_candidate_count is not None and len(candidates) != int(expected_candidate_count):
        raise FocusedContextDecisionError(f"expected {expected_candidate_count} candidates, got {len(candidates)}")
    if expected_candidate_id:
        actual = [str(candidate.get("candidate_id") or "") for candidate in candidates]
        if actual != [expected_candidate_id]:
            raise FocusedContextDecisionError(f"expected candidate {expected_candidate_id}, got {actual}")
    if expected_decision:
        decisions = [str(candidate.get("focused_context_decision") or "") for candidate in candidates]
        if decisions != [expected_decision]:
            raise FocusedContextDecisionError(f"expected decision {expected_decision}, got {decisions}")
    return {
        "candidate_count": int(len(candidates)),
        "candidate_ids": [str(candidate.get("candidate_id") or "") for candidate in candidates],
        "decisions": [str(candidate.get("focused_context_decision") or "") for candidate in candidates],
        "flag_counts": report.get("flag_counts", {}),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Review Stage B margin-recovered focused context package")
    parser.add_argument("--focused_package", type=str, required=True)
    parser.add_argument(
        "--output_root",
        type=str,
        default=str(ROOT_DIR / "outputs" / "stage_b_margin_recovered_focused_context_decision"),
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--expected_candidate_id", type=str, default="")
    parser.add_argument("--expected_decision", type=str, default="")
    parser.add_argument("--expected_candidate_count", type=int, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    focused_package_path = Path(args.focused_package)
    focused_package = read_json(focused_package_path)
    focused_package["output_dir"] = str(focused_package_path.parent)
    report = build_focused_context_decision(focused_package, output_dir=output_dir)
    summary = validate_decision(
        report,
        expected_candidate_id=str(args.expected_candidate_id or ""),
        expected_decision=str(args.expected_decision or ""),
        expected_candidate_count=args.expected_candidate_count,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "focused_context_decision.json", report)
    write_json(output_dir / "focused_context_decision_summary.json", summary)
    (output_dir / "focused_context_decision.md").write_text(markdown_report(report), encoding="utf-8")
    summary.update(
        {
            "decision_path": str(output_dir / "focused_context_decision.json"),
            "markdown_path": str(output_dir / "focused_context_decision.md"),
        }
    )
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
