"""
Audit the full local jazz piano MIDI dataset before broad training.

This script is intentionally a dataset health check, not a training step. It
answers whether the full corpus can be used as a generic jazz-piano prior
before a smaller Brad Mehldau style-adaptation stage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import pretty_midi


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "scripts"))

from scripts.prepare_role_dataset import find_midi_files, infer_tempo  # noqa: E402


PIANO_PROGRAMS = set(range(0, 8))


def percentile(values: Sequence[int | float], ratio: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * float(ratio))))
    return ordered[index]


def basic_stats(values: Sequence[int | float]) -> dict[str, float | int | None]:
    if not values:
        return {"min": None, "p50": None, "p90": None, "p99": None, "max": None, "mean": None}
    return {
        "min": min(values),
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p99": percentile(values, 0.99),
        "max": max(values),
        "mean": statistics.mean(values),
    }


def stable_file_hash(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_dataset_path(midi_path: Path, input_dir: Path) -> dict[str, str | bool | None]:
    try:
        rel = midi_path.relative_to(input_dir)
    except ValueError:
        rel = midi_path

    parts = rel.parts
    source = parts[0] if len(parts) > 0 else None
    artist = parts[1] if len(parts) > 1 else None
    album = parts[2] if len(parts) > 2 else None

    if source == "midi" and len(parts) > 3:
        source = parts[1]
        artist = parts[2]
        album = parts[3]

    artist_name = artist or ""
    return {
        "source": source,
        "artist": artist,
        "album": album,
        "is_brad_mehldau": artist_name.lower() == "brad mehldau",
    }


def non_drum_instruments(pm: pretty_midi.PrettyMIDI) -> list[pretty_midi.Instrument]:
    return [instrument for instrument in pm.instruments if not instrument.is_drum]


def flatten_notes(instruments: Iterable[pretty_midi.Instrument]) -> list[pretty_midi.Note]:
    notes: list[pretty_midi.Note] = []
    for instrument in instruments:
        notes.extend(instrument.notes)
    return sorted(notes, key=lambda note: (note.start, note.pitch))


def recommendation_for(row: dict[str, Any], args: argparse.Namespace) -> str:
    if not row.get("readable"):
        return "reject_unreadable"
    if int(row.get("non_drum_note_count") or 0) < int(args.min_notes):
        return "reject_too_few_notes"
    if float(row.get("duration_sec") or 0.0) < float(args.min_duration_sec):
        return "reject_too_short"
    if row.get("pitch_out_of_piano_range"):
        return "review_pitch_range"
    if row.get("long_sustain_suspect"):
        return "review_long_sustain"
    if row.get("non_piano_program_suspect"):
        return "review_non_piano_program"
    if row.get("too_long"):
        return "review_too_long"
    return "candidate"


def audit_file(midi_path: Path, input_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    row: dict[str, Any] = {
        "path": str(midi_path),
        "readable": False,
        "error": None,
        "size_bytes": midi_path.stat().st_size,
        "extension": midi_path.suffix.lower(),
        "sha1": stable_file_hash(midi_path) if args.hash_files else None,
    }
    row.update(parse_dataset_path(midi_path, input_dir))

    try:
        pm = pretty_midi.PrettyMIDI(str(midi_path))
    except Exception as exc:
        row["error"] = str(exc)
        row["recommendation"] = recommendation_for(row, args)
        return row

    instruments = pm.instruments
    non_drum = non_drum_instruments(pm)
    drum_count = sum(1 for instrument in instruments if instrument.is_drum)
    non_drum_notes = flatten_notes(non_drum)
    drum_notes = flatten_notes(instrument for instrument in instruments if instrument.is_drum)
    piano_program_notes = flatten_notes(instrument for instrument in non_drum if instrument.program in PIANO_PROGRAMS)

    duration_sec = float(pm.get_end_time())
    note_durations = [max(0.0, float(note.end - note.start)) for note in non_drum_notes]
    max_note_duration_sec = max(note_durations, default=0.0)
    max_note_duration_ratio = max_note_duration_sec / duration_sec if duration_sec > 0 else 0.0
    piano_program_note_ratio = len(piano_program_notes) / len(non_drum_notes) if non_drum_notes else 0.0
    pitch_min = min((note.pitch for note in non_drum_notes), default=None)
    pitch_max = max((note.pitch for note in non_drum_notes), default=None)

    row.update(
        {
            "readable": True,
            "duration_sec": duration_sec,
            "tempo_bpm": float(infer_tempo(pm)),
            "instrument_count": len(instruments),
            "non_drum_instrument_count": len(non_drum),
            "drum_instrument_count": drum_count,
            "non_drum_note_count": len(non_drum_notes),
            "drum_note_count": len(drum_notes),
            "piano_program_note_count": len(piano_program_notes),
            "piano_program_note_ratio": piano_program_note_ratio,
            "pitch_min": pitch_min,
            "pitch_max": pitch_max,
            "mean_note_duration_sec": statistics.mean(note_durations) if note_durations else None,
            "max_note_duration_sec": max_note_duration_sec,
            "max_note_duration_ratio": max_note_duration_ratio,
            "too_long": duration_sec > float(args.max_duration_sec),
            "pitch_out_of_piano_range": (pitch_min is not None and pitch_min < 21) or (pitch_max is not None and pitch_max > 108),
            "long_sustain_suspect": max_note_duration_ratio > float(args.max_note_duration_ratio)
            and len(non_drum_notes) >= int(args.min_notes),
            "non_piano_program_suspect": piano_program_note_ratio < float(args.min_piano_program_ratio)
            and len(non_drum_notes) >= int(args.min_notes),
        }
    )
    row["recommendation"] = recommendation_for(row, args)
    return row


def counter_dict(counter: Counter[Any], limit: int | None = None) -> dict[str, int]:
    pairs = counter.most_common(limit)
    return {str(key): int(value) for key, value in pairs}


def summarize(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    readable = [row for row in rows if row.get("readable")]
    candidates = [row for row in readable if row.get("recommendation") == "candidate"]
    candidate_non_brad = [row for row in candidates if not row.get("is_brad_mehldau")]
    candidate_brad = [row for row in candidates if row.get("is_brad_mehldau")]

    hash_groups: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        if row.get("sha1"):
            hash_groups[str(row["sha1"])].append(str(row["path"]))
    duplicate_groups = {digest: paths for digest, paths in hash_groups.items() if len(paths) > 1}

    return {
        "file_count": len(rows),
        "readable_file_count": len(readable),
        "unreadable_file_count": len(rows) - len(readable),
        "candidate_file_count": len(candidates),
        "candidate_non_brad_file_count": len(candidate_non_brad),
        "candidate_brad_file_count": len(candidate_brad),
        "brad_file_count": sum(1 for row in rows if row.get("is_brad_mehldau")),
        "non_brad_file_count": sum(1 for row in rows if not row.get("is_brad_mehldau")),
        "source_counts": counter_dict(Counter(row.get("source") for row in rows)),
        "recommendation_counts": counter_dict(Counter(row.get("recommendation") for row in rows)),
        "top_artists": counter_dict(Counter(row.get("artist") for row in rows), limit=30),
        "top_candidate_artists": counter_dict(Counter(row.get("artist") for row in candidates), limit=30),
        "duration_sec": basic_stats([float(row["duration_sec"]) for row in readable]),
        "non_drum_note_count": basic_stats([int(row["non_drum_note_count"]) for row in readable]),
        "instrument_count": basic_stats([int(row["instrument_count"]) for row in readable]),
        "tempo_bpm": basic_stats([float(row["tempo_bpm"]) for row in readable]),
        "piano_program_note_ratio": basic_stats([float(row["piano_program_note_ratio"]) for row in readable]),
        "max_note_duration_ratio": basic_stats([float(row["max_note_duration_ratio"]) for row in readable]),
        "duplicate_exact_hash_group_count": len(duplicate_groups),
        "duplicate_exact_file_count": sum(len(paths) for paths in duplicate_groups.values()),
        "duplicate_exact_groups_preview": dict(list(duplicate_groups.items())[:20]),
    }


def split_manifests(rows: Sequence[dict[str, Any]]) -> dict[str, list[str]]:
    candidates = [row for row in rows if row.get("recommendation") == "candidate"]
    return {
        "candidate_all": [str(row["path"]) for row in candidates],
        "candidate_non_brad": [str(row["path"]) for row in candidates if not row.get("is_brad_mehldau")],
        "candidate_brad": [str(row["path"]) for row in candidates if row.get("is_brad_mehldau")],
        "review": [str(row["path"]) for row in rows if str(row.get("recommendation", "")).startswith("review_")],
        "rejected": [str(row["path"]) for row in rows if str(row.get("recommendation", "")).startswith("reject_")],
    }


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def stats_row(summary: dict[str, Any], key: str) -> str:
    stats = summary[key]
    return (
        f"| `{key}` | {fmt(stats['min'])} | {fmt(stats['p50'])} | {fmt(stats['p90'])} | "
        f"{fmt(stats['p99'])} | {fmt(stats['max'])} | {fmt(stats['mean'])} |"
    )


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    lines = [
        "# Jazz Piano Dataset Audit",
        "",
        f"- input_dir: `{payload['input_dir']}`",
        f"- files: `{summary['file_count']}`",
        f"- readable: `{summary['readable_file_count']}`",
        f"- unreadable: `{summary['unreadable_file_count']}`",
        f"- candidate files: `{summary['candidate_file_count']}`",
        f"- candidate non-Brad files: `{summary['candidate_non_brad_file_count']}`",
        f"- candidate Brad files: `{summary['candidate_brad_file_count']}`",
        f"- exact duplicate hash groups: `{summary['duplicate_exact_hash_group_count']}`",
        "",
        "## Recommendation Counts",
        "",
        "| Recommendation | Count |",
        "|---|---:|",
    ]
    for key, value in summary["recommendation_counts"].items():
        lines.append(f"| `{key}` | {value} |")

    lines.extend(
        [
            "",
            "## Source Counts",
            "",
            "| Source | Count |",
            "|---|---:|",
        ]
    )
    for key, value in summary["source_counts"].items():
        lines.append(f"| `{key}` | {value} |")

    lines.extend(
        [
            "",
            "## Distribution Stats",
            "",
            "| Metric | Min | P50 | P90 | P99 | Max | Mean |",
            "|---|---:|---:|---:|---:|---:|---:|",
            stats_row(summary, "duration_sec"),
            stats_row(summary, "non_drum_note_count"),
            stats_row(summary, "instrument_count"),
            stats_row(summary, "tempo_bpm"),
            stats_row(summary, "piano_program_note_ratio"),
            stats_row(summary, "max_note_duration_ratio"),
            "",
            "## Top Artists",
            "",
            "| Artist | Files |",
            "|---|---:|",
        ]
    )
    for key, value in summary["top_artists"].items():
        lines.append(f"| `{key}` | {value} |")

    lines.extend(
        [
            "",
            "## Training Decision",
            "",
            "- Use candidate non-Brad files for a generic jazz-piano prior only after tokenizer sanity is validated.",
            "- Use candidate Brad files for style adaptation and holdout evaluation, not for the generic-only baseline.",
            "- Review or reject files should not enter training until the reason is understood.",
            "- Exact duplicates should be deduplicated before train/val/test split.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit the full jazz piano MIDI dataset")
    parser.add_argument("--input_dir", type=str, default="./midi_dataset/midi")
    parser.add_argument("--output_json", type=str, default="./outputs/dataset_audit/jazz_piano_dataset_audit.json")
    parser.add_argument("--output_md", type=str, default="./outputs/dataset_audit/jazz_piano_dataset_audit.md")
    parser.add_argument("--min_notes", type=int, default=24)
    parser.add_argument("--min_duration_sec", type=float, default=5.0)
    parser.add_argument("--max_duration_sec", type=float, default=1800.0)
    parser.add_argument("--max_note_duration_ratio", type=float, default=0.85)
    parser.add_argument("--min_piano_program_ratio", type=float, default=0.80)
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--hash_files", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    input_dir = Path(args.input_dir)
    midi_files = find_midi_files(input_dir)
    if args.max_files is not None:
        midi_files = midi_files[: max(0, int(args.max_files))]
    if not midi_files:
        raise ValueError(f"No MIDI files found under: {input_dir}")

    rows = [audit_file(midi_path, input_dir, args) for midi_path in midi_files]
    payload = {
        "input_dir": str(input_dir),
        "audit_thresholds": {
            "min_notes": args.min_notes,
            "min_duration_sec": args.min_duration_sec,
            "max_duration_sec": args.max_duration_sec,
            "max_note_duration_ratio": args.max_note_duration_ratio,
            "min_piano_program_ratio": args.min_piano_program_ratio,
            "hash_files": args.hash_files,
        },
        "summary": summarize(rows),
        "manifests": split_manifests(rows),
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
