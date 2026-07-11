"""Extract MIDI context for the Stage B MIDI-to-solo MVP."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pretty_midi

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from scripts.assess_stage_b_generic_base_readiness import write_json, write_text  # noqa: E402
from scripts.stage_b_tokens import (  # noqa: E402
    CHORD_QUALITIES,
    CHORD_ROOTS,
    PC_TO_CANONICAL_ROOT,
    POSITIONS_PER_BAR,
    parse_chord_symbol,
)


class StageBMidiToSoloContextExtractionError(ValueError):
    pass


BOUNDARY = "stage_b_midi_to_solo_context_extraction_mvp"
NEXT_BOUNDARY = "stage_b_midi_to_solo_training_resource_probe"
SCHEMA_VERSION = "stage_b_midi_to_solo_context_extraction_v1"

QUALITY_TEMPLATES = {
    "maj": {0, 4, 7},
    "min": {0, 3, 7},
    "dom7": {0, 4, 7, 10},
    "maj7": {0, 4, 7, 11},
    "min7": {0, 3, 7, 10},
    "halfdim": {0, 3, 6, 10},
    "dim": {0, 3, 6},
    "sus": {0, 5, 7},
}


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


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def load_tempo_bpm(pm: pretty_midi.PrettyMIDI) -> float:
    _, tempos = pm.get_tempo_changes()
    if len(tempos):
        return float(tempos[0])
    try:
        estimated = float(pm.estimate_tempo())
    except (ValueError, ZeroDivisionError):
        estimated = 120.0
    return estimated if estimated > 0 else 120.0


def bar_duration_sec(tempo_bpm: float, beats_per_bar: int = 4) -> float:
    return (60.0 / max(1e-6, float(tempo_bpm))) * int(beats_per_bar)


def iter_non_drum_notes(pm: pretty_midi.PrettyMIDI) -> Iterable[pretty_midi.Note]:
    for instrument in pm.instruments:
        if instrument.is_drum:
            continue
        for note in instrument.notes:
            yield note


def note_bar_index(note: pretty_midi.Note, *, tempo_bpm: float, beats_per_bar: int = 4) -> int:
    return max(0, int(math.floor(float(note.start) / bar_duration_sec(tempo_bpm, beats_per_bar))))


def normalize_chord_text(text: str) -> str:
    cleaned = str(text or "").strip()
    cleaned = re.sub(r"^(chord|changes?)\s*[:=]\s*", "", cleaned, flags=re.IGNORECASE)
    return cleaned


def explicit_chord_events(pm: pretty_midi.PrettyMIDI, *, tempo_bpm: float) -> dict[int, dict[str, Any]]:
    events: dict[int, dict[str, Any]] = {}
    for attr in ("text_events", "lyrics"):
        for event in getattr(pm, attr, []) or []:
            text = normalize_chord_text(getattr(event, "text", ""))
            root, quality = parse_chord_symbol(text)
            if root == "N" or quality == "unknown":
                continue
            bar_index = max(0, int(math.floor(float(getattr(event, "time", 0.0)) / bar_duration_sec(tempo_bpm))))
            events[bar_index] = {
                "root": root,
                "quality": quality,
                "source": "explicit_text_event",
                "confidence": 1.0,
                "symbol": text,
            }
    return events


def collect_bar_pitch_classes(
    pm: pretty_midi.PrettyMIDI,
    *,
    tempo_bpm: float,
    bar_count: int,
) -> list[Counter[int]]:
    counters = [Counter() for _ in range(bar_count)]
    for note in iter_non_drum_notes(pm):
        bar_index = note_bar_index(note, tempo_bpm=tempo_bpm)
        if 0 <= bar_index < bar_count:
            counters[bar_index][int(note.pitch) % 12] += 1
    return counters


def collect_bar_bass_notes(
    pm: pretty_midi.PrettyMIDI,
    *,
    tempo_bpm: float,
    bar_count: int,
) -> list[int | None]:
    pitches_by_bar: list[list[int]] = [[] for _ in range(bar_count)]
    for note in iter_non_drum_notes(pm):
        bar_index = note_bar_index(note, tempo_bpm=tempo_bpm)
        if 0 <= bar_index < bar_count:
            pitches_by_bar[bar_index].append(int(note.pitch))
    return [min(pitches) if pitches else None for pitches in pitches_by_bar]


def score_quality(pitch_classes: set[int], root_pc: int, quality: str) -> int:
    template = QUALITY_TEMPLATES[quality]
    normalized = {(pc - root_pc) % 12 for pc in pitch_classes}
    return len(normalized & template) * 2 - len(template - normalized)


def infer_chord_from_bar(
    pitch_counter: Counter[int],
    *,
    bass_note: int | None,
) -> dict[str, Any]:
    if not pitch_counter:
        return {
            "root": "N",
            "quality": "unknown",
            "source": "empty_bar_fallback",
            "confidence": 0.0,
            "symbol": "N",
        }

    pitch_classes = set(int(pc) for pc in pitch_counter)
    root_candidates = [int(bass_note) % 12] if bass_note is not None else []
    root_candidates.extend(pc for pc, _ in pitch_counter.most_common())

    best: tuple[int, int, str] | None = None
    for root_pc in dict.fromkeys(root_candidates):
        for quality in QUALITY_TEMPLATES:
            score = score_quality(pitch_classes, root_pc, quality)
            if bass_note is not None and root_pc == int(bass_note) % 12:
                score += 2
            candidate = (score, root_pc, quality)
            if best is None or candidate > best:
                best = candidate

    if best is None:
        return {
            "root": "N",
            "quality": "unknown",
            "source": "pitch_class_inference_failed",
            "confidence": 0.0,
            "symbol": "N",
        }

    score, root_pc, quality = best
    root = PC_TO_CANONICAL_ROOT.get(root_pc, "N")
    confidence = max(0.25, min(0.9, 0.45 + 0.08 * score))
    if root not in CHORD_ROOTS or quality not in CHORD_QUALITIES:
        return {
            "root": "N",
            "quality": "unknown",
            "source": "pitch_class_inference_unsupported",
            "confidence": 0.0,
            "symbol": "N",
        }
    return {
        "root": root,
        "quality": quality,
        "source": "pitch_class_inference",
        "confidence": round(float(confidence), 4),
        "symbol": f"{root}:{quality}",
    }


def build_fixture_midi(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pm = pretty_midi.PrettyMIDI(initial_tempo=120)
    comp = pretty_midi.Instrument(program=0, is_drum=False, name="piano_comping")
    bass = pretty_midi.Instrument(program=32, is_drum=False, name="bass")
    chord_pitches = [
        [48, 52, 55, 59],
        [53, 57, 60, 63],
        [55, 59, 62, 65],
        [48, 52, 55, 59],
    ]
    bass_pitches = [36, 41, 43, 36]
    for bar_index, pitches in enumerate(chord_pitches):
        start = float(bar_index * 2.0)
        for pitch in pitches:
            comp.notes.append(pretty_midi.Note(velocity=70, pitch=pitch, start=start, end=start + 1.75))
        bass.notes.append(
            pretty_midi.Note(velocity=82, pitch=bass_pitches[bar_index], start=start, end=start + 1.75)
        )
    pm.instruments.extend([comp, bass])
    pm.write(str(path))
    return path


def extract_context_from_midi(
    midi_path: Path,
    *,
    target_context_bars: int = 8,
    beats_per_bar: int = 4,
) -> dict[str, Any]:
    if not midi_path.exists():
        raise StageBMidiToSoloContextExtractionError(f"input MIDI does not exist: {midi_path}")
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    tempo_bpm = load_tempo_bpm(pm)
    end_time = max([float(note.end) for note in iter_non_drum_notes(pm)] or [0.0])
    detected_bars = max(1, int(math.ceil(end_time / bar_duration_sec(tempo_bpm, beats_per_bar))))
    bar_count = max(int(target_context_bars), detected_bars)

    explicit_events = explicit_chord_events(pm, tempo_bpm=tempo_bpm)
    pitch_counters = collect_bar_pitch_classes(pm, tempo_bpm=tempo_bpm, bar_count=bar_count)
    bass_notes = collect_bar_bass_notes(pm, tempo_bpm=tempo_bpm, bar_count=bar_count)

    bar_contexts: list[dict[str, Any]] = []
    for bar_index in range(bar_count):
        chord = explicit_events.get(bar_index)
        if chord is None:
            chord = infer_chord_from_bar(pitch_counters[bar_index], bass_note=bass_notes[bar_index])
        if (
            chord["root"] == "N"
            and not pitch_counters[bar_index]
            and bar_contexts
            and bar_contexts[-1]["chord_root"] != "N"
        ):
            previous = bar_contexts[-1]
            chord = {
                "root": previous["chord_root"],
                "quality": previous["chord_quality"],
                "source": "carry_forward_empty_bar",
                "confidence": round(min(float(previous["chord_confidence"]), 0.45), 4),
                "symbol": previous["chord_symbol"],
            }
        bar_contexts.append(
            {
                "bar_index": int(bar_index),
                "tempo": float(round(tempo_bpm, 4)),
                "chord_root": chord["root"],
                "chord_quality": chord["quality"],
                "chord_source": chord["source"],
                "chord_confidence": float(chord["confidence"]),
                "chord_symbol": chord["symbol"],
                "bass_note": bass_notes[bar_index],
                "pitch_class_count": int(sum(pitch_counters[bar_index].values())),
                "unique_pitch_class_count": int(len(pitch_counters[bar_index])),
            }
        )

    context_events: list[dict[str, Any]] = []
    for bar_index, bar_context in enumerate(bar_contexts):
        next_context = bar_contexts[bar_index + 1] if bar_index + 1 < len(bar_contexts) else bar_context
        for position_index in range(POSITIONS_PER_BAR):
            context_events.append(
                {
                    "bar_index": int(bar_index),
                    "position_index": int(position_index),
                    "tempo": float(bar_context["tempo"]),
                    "chord_root": str(bar_context["chord_root"]),
                    "chord_quality": str(bar_context["chord_quality"]),
                    "next_chord_root": str(next_context["chord_root"]),
                    "next_chord_quality": str(next_context["chord_quality"]),
                    "bass_note": bar_context["bass_note"],
                    "chord_confidence": float(bar_context["chord_confidence"]),
                    "chord_source": str(bar_context["chord_source"]),
                }
            )

    return {
        "midi_path": str(midi_path),
        "tempo_bpm": float(round(tempo_bpm, 4)),
        "beats_per_bar": int(beats_per_bar),
        "positions_per_bar": int(POSITIONS_PER_BAR),
        "detected_bars": int(detected_bars),
        "context_bars": int(bar_count),
        "bar_contexts": bar_contexts,
        "context_events": context_events,
    }


def build_context_report(
    *,
    midi_path: Path,
    output_dir: Path,
    target_context_bars: int,
    issue_number: int,
) -> dict[str, Any]:
    context = extract_context_from_midi(midi_path, target_context_bars=target_context_bars)
    bar_contexts = list(context["bar_contexts"])
    event_count = len(context["context_events"])
    unknown_bars = [
        item for item in bar_contexts if item["chord_root"] == "N" or item["chord_quality"] == "unknown"
    ]
    inferred_bars = [item for item in bar_contexts if item["chord_source"] == "pitch_class_inference"]
    explicit_bars = [item for item in bar_contexts if item["chord_source"] == "explicit_text_event"]
    carried_bars = [item for item in bar_contexts if item["chord_source"] == "carry_forward_empty_bar"]
    low_confidence_bars = [item for item in bar_contexts if float(item["chord_confidence"]) < 0.5]
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "issue_number": int(issue_number),
        "boundary": BOUNDARY,
        "input": {
            "midi_path": str(midi_path),
            "target_context_bars": int(target_context_bars),
        },
        "context": context,
        "summary": {
            "context_bars": int(context["context_bars"]),
            "positions_per_bar": int(context["positions_per_bar"]),
            "context_event_count": int(event_count),
            "explicit_chord_bar_count": int(len(explicit_bars)),
            "inferred_chord_bar_count": int(len(inferred_bars)),
            "carry_forward_chord_bar_count": int(len(carried_bars)),
            "unknown_chord_bar_count": int(len(unknown_bars)),
            "low_confidence_bar_count": int(len(low_confidence_bars)),
            "bass_note_bar_count": int(sum(1 for item in bar_contexts if item["bass_note"] is not None)),
        },
        "readiness": {
            "boundary": BOUNDARY,
            "context_extraction_completed": True,
            "required_context_fields_present": True,
            "midi_to_solo_mvp_claimed": False,
            "harmony_analysis_quality_claimed": False,
            "brad_style_fine_tuning_completed": False,
        },
        "decision": {
            "current_boundary": BOUNDARY,
            "next_boundary": NEXT_BOUNDARY,
            "auto_progress_allowed": True,
            "critical_user_input_required": False,
            "reason": (
                "MIDI context rows are available for conditioned generation; low-confidence chord "
                "inference remains a ranking penalty, not a blocker"
            ),
        },
        "next_recommended_issue": "Stage B MIDI-to-solo training resource probe",
    }


def validate_context_report(
    report: dict[str, Any],
    *,
    expected_boundary: str | None,
    expected_next_boundary: str | None,
    min_context_bars: int,
    require_no_final_claim: bool,
) -> dict[str, Any]:
    boundary = str(report.get("boundary") or "")
    decision = _dict(report.get("decision"))
    readiness = _dict(report.get("readiness"))
    summary = _dict(report.get("summary"))
    context = _dict(report.get("context"))
    next_boundary = str(decision.get("next_boundary") or "")
    if expected_boundary and boundary != expected_boundary:
        raise StageBMidiToSoloContextExtractionError(f"expected boundary {expected_boundary}, got {boundary}")
    if expected_next_boundary and next_boundary != expected_next_boundary:
        raise StageBMidiToSoloContextExtractionError(
            f"expected next boundary {expected_next_boundary}, got {next_boundary}"
        )
    if _int(summary.get("context_bars")) < int(min_context_bars):
        raise StageBMidiToSoloContextExtractionError("context bars below minimum")
    expected_events = _int(summary.get("context_bars")) * POSITIONS_PER_BAR
    if _int(summary.get("context_event_count")) != expected_events:
        raise StageBMidiToSoloContextExtractionError("context event count does not match bars * positions")
    if _int(summary.get("bass_note_bar_count")) <= 0:
        raise StageBMidiToSoloContextExtractionError("at least one bass-note bar required")
    required_fields = {
        "bar_index",
        "position_index",
        "tempo",
        "chord_root",
        "chord_quality",
        "next_chord_root",
        "next_chord_quality",
        "bass_note",
        "chord_confidence",
        "chord_source",
    }
    for event in context.get("context_events") or []:
        missing = required_fields - set(event)
        if missing:
            raise StageBMidiToSoloContextExtractionError(f"context event missing fields: {sorted(missing)}")
    if bool(decision.get("critical_user_input_required", True)):
        raise StageBMidiToSoloContextExtractionError("context extraction should not require critical user input")
    if require_no_final_claim:
        blocked = [
            "midi_to_solo_mvp_claimed",
            "harmony_analysis_quality_claimed",
            "brad_style_fine_tuning_completed",
        ]
        claimed = [name for name in blocked if bool(readiness.get(name, True))]
        if claimed:
            raise StageBMidiToSoloContextExtractionError(f"unexpected final claim: {claimed}")
    return {
        "boundary": boundary,
        "next_boundary": next_boundary,
        "context_bars": _int(summary.get("context_bars")),
        "context_event_count": _int(summary.get("context_event_count")),
        "inferred_chord_bar_count": _int(summary.get("inferred_chord_bar_count")),
        "carry_forward_chord_bar_count": _int(summary.get("carry_forward_chord_bar_count")),
        "unknown_chord_bar_count": _int(summary.get("unknown_chord_bar_count")),
        "low_confidence_bar_count": _int(summary.get("low_confidence_bar_count")),
        "bass_note_bar_count": _int(summary.get("bass_note_bar_count")),
        "midi_to_solo_mvp_claimed": bool(readiness.get("midi_to_solo_mvp_claimed", True)),
        "harmony_analysis_quality_claimed": bool(
            readiness.get("harmony_analysis_quality_claimed", True)
        ),
        "critical_user_input_required": bool(decision.get("critical_user_input_required", True)),
        "next_recommended_issue": str(report.get("next_recommended_issue") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    summary = report["summary"]
    decision = report["decision"]
    readiness = report["readiness"]
    context = report["context"]
    lines = [
        "# Stage B MIDI-to-Solo Context Extraction MVP",
        "",
        "## Summary",
        "",
        f"- boundary: `{report['boundary']}`",
        f"- next boundary: `{decision['next_boundary']}`",
        f"- input MIDI: `{report['input']['midi_path']}`",
        f"- context extraction completed: `{_bool_token(readiness['context_extraction_completed'])}`",
        f"- MIDI-to-solo MVP claimed: `{_bool_token(readiness['midi_to_solo_mvp_claimed'])}`",
        f"- harmony analysis quality claimed: `{_bool_token(readiness['harmony_analysis_quality_claimed'])}`",
        "",
        "## Context Summary",
        "",
        f"- tempo BPM: `{context['tempo_bpm']}`",
        f"- context bars: `{summary['context_bars']}`",
        f"- positions per bar: `{context['positions_per_bar']}`",
        f"- context event count: `{summary['context_event_count']}`",
        f"- explicit / inferred / carried / unknown chord bars: `{summary['explicit_chord_bar_count']}` / `{summary['inferred_chord_bar_count']}` / `{summary['carry_forward_chord_bar_count']}` / `{summary['unknown_chord_bar_count']}`",
        f"- low-confidence bars: `{summary['low_confidence_bar_count']}`",
        f"- bass-note bars: `{summary['bass_note_bar_count']}`",
        "",
        "## Bar Contexts",
        "",
    ]
    for item in context["bar_contexts"]:
        lines.append(
            f"- bar `{item['bar_index']}`: `{item['chord_root']}` `{item['chord_quality']}`, "
            f"bass `{item['bass_note']}`, source `{item['chord_source']}`, confidence `{item['chord_confidence']}`"
        )
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract MIDI-to-solo MVP context from an input MIDI")
    parser.add_argument("--input_midi", type=str, default="")
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/stage_b_midi_to_solo_context_extraction",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--issue_number", type=int, default=483)
    parser.add_argument("--target_context_bars", type=int, default=8)
    parser.add_argument("--expected_boundary", type=str, default="")
    parser.add_argument("--expected_next_boundary", type=str, default="")
    parser.add_argument("--min_context_bars", type=int, default=4)
    parser.add_argument("--require_no_final_claim", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    input_midi = Path(args.input_midi) if args.input_midi else build_fixture_midi(output_dir / "fixture.mid")
    report = build_context_report(
        midi_path=input_midi,
        output_dir=output_dir,
        target_context_bars=int(args.target_context_bars),
        issue_number=int(args.issue_number),
    )
    summary = validate_context_report(
        report,
        expected_boundary=str(args.expected_boundary or ""),
        expected_next_boundary=str(args.expected_next_boundary or ""),
        min_context_bars=int(args.min_context_bars),
        require_no_final_claim=bool(args.require_no_final_claim),
    )
    write_json(output_dir / "stage_b_midi_to_solo_context_extraction.json", report)
    write_json(output_dir / "stage_b_midi_to_solo_context_extraction_validation_summary.json", summary)
    markdown = markdown_report(report)
    write_text(output_dir / "stage_b_midi_to_solo_context_extraction.md", markdown)
    if args.doc_path:
        write_text(Path(args.doc_path), markdown)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
