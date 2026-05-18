"""
Audit the Brad Mehldau MIDI dataset before control_v1 training.

The goal is to measure whether the actual dataset is usable for symbolic MIDI
fine-tuning before spending GPU time on full Stage A training.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

import pretty_midi


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "scripts"))

from scripts.control_tokens import SEQUENCE_FORMAT_CONTROL_V1  # noqa: E402
from scripts.prepare_role_dataset import (  # noqa: E402
    encode_midi_simple,
    find_midi_files,
    infer_tempo,
    notes_to_midi,
    split_conditioning_target,
)


def percentile(values: Sequence[int | float], ratio: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * float(ratio))))
    return ordered[index]


def basic_stats(values: Sequence[int | float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "min": None,
            "p50": None,
            "p90": None,
            "max": None,
            "mean": None,
        }
    return {
        "min": min(values),
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "max": max(values),
        "mean": statistics.mean(values),
    }


def non_drum_notes(pm: pretty_midi.PrettyMIDI) -> list[pretty_midi.Note]:
    notes: list[pretty_midi.Note] = []
    for instrument in pm.instruments:
        if not instrument.is_drum:
            notes.extend(instrument.notes)
    return sorted(notes, key=lambda note: (note.start, note.pitch))


def audit_file(
    midi_path: Path,
    split_pitch: int,
    min_conditioning_notes: int,
    min_target_notes: int,
    max_sequence: int,
    temp_root: Path,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "path": str(midi_path),
        "usable": False,
        "skip_reason": None,
    }

    try:
        pm = pretty_midi.PrettyMIDI(str(midi_path))
    except Exception as exc:
        row["skip_reason"] = f"unreadable MIDI: {exc}"
        return row

    notes = non_drum_notes(pm)
    row.update(
        {
            "instrument_count": len(pm.instruments),
            "non_drum_instrument_count": sum(1 for instrument in pm.instruments if not instrument.is_drum),
            "duration_sec": float(pm.get_end_time()),
            "tempo_bpm": float(infer_tempo(pm)),
            "note_count": len(notes),
            "pitch_min": min((note.pitch for note in notes), default=None),
            "pitch_max": max((note.pitch for note in notes), default=None),
        }
    )

    if len(notes) < max(min_conditioning_notes, min_target_notes):
        row["skip_reason"] = "too few notes"
        return row

    conditioning, target = split_conditioning_target(
        notes,
        split_pitch=split_pitch,
        min_conditioning_notes=min_conditioning_notes,
        min_target_notes=min_target_notes,
    )
    row["conditioning_notes"] = len(conditioning)
    row["target_notes"] = len(target)
    if len(conditioning) < 2 or len(target) < min_target_notes:
        row["skip_reason"] = "too few conditioning or target notes after split"
        return row

    sample_dir = temp_root / f"sample_{abs(hash(str(midi_path))) % 10_000_000}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    conditioning_path = sample_dir / "conditioning.mid"
    target_path = sample_dir / "target.mid"
    notes_to_midi(conditioning, row["tempo_bpm"]).write(str(conditioning_path))
    notes_to_midi(target, row["tempo_bpm"]).write(str(target_path))

    conditioning_tokens = encode_midi_simple(str(conditioning_path))
    target_tokens = encode_midi_simple(str(target_path))
    control_v1_token_count = 3 + len(conditioning_tokens) + 1 + len(target_tokens) + 1
    legacy_token_count = len(conditioning_tokens) + 1 + len(target_tokens) + 1

    row.update(
        {
            "conditioning_token_count": len(conditioning_tokens),
            "target_token_count": len(target_tokens),
            "control_v1_token_count": control_v1_token_count,
            "legacy_token_count": legacy_token_count,
            "exceeds_max_sequence": control_v1_token_count > max_sequence,
            "max_sequence": int(max_sequence),
            "usable": bool(conditioning_tokens and target_tokens),
        }
    )
    if not row["usable"]:
        row["skip_reason"] = "empty conditioning or target tokens"
    return row


def summarize(rows: Sequence[dict[str, Any]], max_sequence: int) -> dict[str, Any]:
    usable_rows = [row for row in rows if row.get("usable")]
    token_counts = [int(row["control_v1_token_count"]) for row in usable_rows]
    return {
        "file_count": len(rows),
        "usable_file_count": len(usable_rows),
        "unusable_file_count": len(rows) - len(usable_rows),
        "sequence_format": SEQUENCE_FORMAT_CONTROL_V1,
        "max_sequence": int(max_sequence),
        "control_v1_token_count": basic_stats(token_counts),
        "target_token_count": basic_stats([int(row["target_token_count"]) for row in usable_rows]),
        "conditioning_token_count": basic_stats([int(row["conditioning_token_count"]) for row in usable_rows]),
        "note_count": basic_stats([int(row["note_count"]) for row in usable_rows]),
        "duration_sec": basic_stats([float(row["duration_sec"]) for row in usable_rows]),
        "tempo_bpm": basic_stats([float(row["tempo_bpm"]) for row in usable_rows]),
        "files_exceeding_max_sequence": sum(1 for row in usable_rows if row.get("exceeds_max_sequence")),
        "skip_reasons": {
            reason: sum(1 for row in rows if row.get("skip_reason") == reason)
            for reason in sorted({str(row.get("skip_reason")) for row in rows if row.get("skip_reason")})
        },
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    summary = payload["summary"]

    def fmt(value: Any) -> str:
        if value is None:
            return "-"
        if isinstance(value, float):
            return f"{value:.2f}"
        return str(value)

    lines = [
        "# Brad Mehldau Dataset Audit",
        "",
        f"- input_dir: `{payload['input_dir']}`",
        f"- sequence_format: `{summary['sequence_format']}`",
        f"- files: `{summary['file_count']}`",
        f"- usable files: `{summary['usable_file_count']}`",
        f"- max_sequence: `{summary['max_sequence']}`",
        f"- files exceeding max_sequence: `{summary['files_exceeding_max_sequence']}`",
        "",
        "## Token Stats",
        "",
        "| Metric | Min | P50 | P90 | Max | Mean |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key in ["control_v1_token_count", "conditioning_token_count", "target_token_count", "note_count"]:
        stats = summary[key]
        lines.append(
            f"| {key} | {fmt(stats['min'])} | {fmt(stats['p50'])} | {fmt(stats['p90'])} | {fmt(stats['max'])} | {fmt(stats['mean'])} |"
        )

    lines.extend(
        [
            "",
            "## Decision",
            "",
            "Use subset/full training only after confirming these token lengths are compatible with the crop policy and chosen max_sequence.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit Brad Mehldau MIDI dataset for control_v1 training")
    parser.add_argument("--input_dir", type=str, default="./midi_dataset/midi/studio/Brad Mehldau")
    parser.add_argument("--output_json", type=str, default="./outputs/dataset_audit/brad_mehldau_control_v1_audit.json")
    parser.add_argument("--output_md", type=str, default="./outputs/dataset_audit/brad_mehldau_control_v1_audit.md")
    parser.add_argument("--split_pitch", type=int, default=60)
    parser.add_argument("--min_conditioning_notes", type=int, default=8)
    parser.add_argument("--min_target_notes", type=int, default=24)
    parser.add_argument("--max_sequence", type=int, default=512)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    input_dir = Path(args.input_dir)
    midi_files = find_midi_files(input_dir)
    if not midi_files:
        raise ValueError(f"No MIDI files found under: {input_dir}")

    with tempfile.TemporaryDirectory() as tmp_dir:
        rows = [
            audit_file(
                midi_path=path,
                split_pitch=args.split_pitch,
                min_conditioning_notes=args.min_conditioning_notes,
                min_target_notes=args.min_target_notes,
                max_sequence=args.max_sequence,
                temp_root=Path(tmp_dir),
            )
            for path in midi_files
        ]

    payload = {
        "input_dir": str(input_dir),
        "summary": summarize(rows, args.max_sequence),
        "files": rows,
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    write_markdown(Path(args.output_md), payload)
    print(json.dumps(payload["summary"], ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
