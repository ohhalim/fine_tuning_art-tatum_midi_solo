"""
Offline MIDI quality metrics for Stage A quick checks.
"""

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Dict, List

import pretty_midi


def find_midi_files(input_dir: Path) -> List[Path]:
    files = list(input_dir.rglob("*.mid")) + list(input_dir.rglob("*.midi"))
    return sorted(files)


def repetition_ratio(pitches: List[int], n: int = 4) -> float:
    if len(pitches) < n * 2:
        return 0.0
    grams = [tuple(pitches[i : i + n]) for i in range(len(pitches) - n + 1)]
    unique = len(set(grams))
    return 1.0 - (unique / len(grams))


def evaluate_file(path: Path, dead_air_threshold_ms: float) -> Dict[str, float]:
    pm = pretty_midi.PrettyMIDI(str(path))

    notes = []
    for inst in pm.instruments:
        if not inst.is_drum:
            notes.extend(inst.notes)
    notes = sorted(notes, key=lambda x: (x.start, x.pitch))

    if not notes:
        return {
            "file": str(path),
            "note_count": 0,
            "duration_sec": 0.0,
            "note_density": 0.0,
            "dead_air_events": 0,
            "dead_air_ratio": 1.0,
            "pitch_mean": 0.0,
            "pitch_std": 0.0,
            "repetition_4gram": 0.0,
        }

    starts = [n.start for n in notes]
    pitches = [n.pitch for n in notes]
    duration_sec = max(1e-6, notes[-1].end - notes[0].start)

    gaps = [max(0.0, starts[i] - starts[i - 1]) for i in range(1, len(starts))]
    threshold = dead_air_threshold_ms / 1000.0
    dead_air_events = sum(1 for g in gaps if g >= threshold)
    dead_air_ratio = dead_air_events / max(1, len(gaps))

    pitch_mean = mean(pitches)
    pitch_std = (sum((p - pitch_mean) ** 2 for p in pitches) / len(pitches)) ** 0.5

    return {
        "file": str(path),
        "note_count": len(notes),
        "duration_sec": duration_sec,
        "note_density": len(notes) / duration_sec,
        "dead_air_events": dead_air_events,
        "dead_air_ratio": dead_air_ratio,
        "pitch_mean": pitch_mean,
        "pitch_std": pitch_std,
        "repetition_4gram": repetition_ratio(pitches, n=4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate generated MIDI files (Stage A)")
    parser.add_argument("--input", type=str, required=True, help="Directory containing MIDI files")
    parser.add_argument("--dead_air_threshold_ms", type=float, default=180.0)
    parser.add_argument("--output_json", type=str, default=None)
    args = parser.parse_args()

    input_dir = Path(args.input)
    files = find_midi_files(input_dir)
    if not files:
        raise ValueError(f"No MIDI files found in {input_dir}")

    rows = [evaluate_file(p, args.dead_air_threshold_ms) for p in files]

    agg = {
        "files": len(rows),
        "avg_note_density": mean(r["note_density"] for r in rows),
        "avg_dead_air_ratio": mean(r["dead_air_ratio"] for r in rows),
        "avg_repetition_4gram": mean(r["repetition_4gram"] for r in rows),
        "avg_pitch_std": mean(r["pitch_std"] for r in rows),
    }

    report = {"summary": agg, "files": rows}
    print(json.dumps(report["summary"], ensure_ascii=True, indent=2))

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, ensure_ascii=True, indent=2))
        print(f"Saved report: {out_path}")


if __name__ == "__main__":
    main()

