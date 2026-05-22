"""Evaluate a small chord-labeled MIDI/inline-note subset for pitch-role sanity."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import pretty_midi

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from scripts.run_stage_b_reference_stats import (  # noqa: E402
    PITCH_ROLE_KEYS,
    guide_tone_pitch_classes,
    pitch_role_for_group,
    pitch_role_ratio_metrics,
    position_bucket,
)
from scripts.stage_b_tokens import (  # noqa: E402
    PIANO_PITCH_MAX,
    PIANO_PITCH_MIN,
    POSITIONS_PER_BAR,
    parse_chord_symbol,
    quantize_note_position,
)


SCHEMA_VERSION = "stage_b_chord_labeled_eval_v1"


class ManifestError(ValueError):
    pass


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_chord_symbol(chord: Any, sample_id: str, bar_index: int) -> str:
    if not isinstance(chord, str) or not chord.strip():
        raise ManifestError(f"{sample_id}: chord at bar {bar_index} must be a non-empty string")
    root, quality = parse_chord_symbol(chord)
    if root == "N" or quality == "unknown":
        raise ManifestError(f"{sample_id}: unsupported chord symbol at bar {bar_index}: {chord!r}")
    return chord.strip()


def validate_manifest(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ManifestError(f"schema_version must be {SCHEMA_VERSION!r}")
    samples = manifest.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ManifestError("manifest.samples must be a non-empty list")

    validated: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for raw in samples:
        if not isinstance(raw, dict):
            raise ManifestError("each sample must be an object")
        sample = dict(raw)
        sample_id = str(sample.get("sample_id") or "").strip()
        if not sample_id:
            raise ManifestError("sample_id is required")
        if sample_id in seen_ids:
            raise ManifestError(f"duplicate sample_id: {sample_id}")
        seen_ids.add(sample_id)

        bar_count = int(sample.get("bar_count", 0) or 0)
        if bar_count <= 0:
            raise ManifestError(f"{sample_id}: bar_count must be positive")
        chords = sample.get("chords")
        if not isinstance(chords, list) or len(chords) != bar_count:
            raise ManifestError(f"{sample_id}: chords length must equal bar_count")
        sample["chords"] = [
            validate_chord_symbol(chord, sample_id=sample_id, bar_index=index)
            for index, chord in enumerate(chords)
        ]
        sample["bar_count"] = bar_count
        sample["beats_per_bar"] = int(sample.get("beats_per_bar", 4) or 4)
        sample["positions_per_bar"] = int(sample.get("positions_per_bar", POSITIONS_PER_BAR) or POSITIONS_PER_BAR)
        if sample["positions_per_bar"] != POSITIONS_PER_BAR:
            raise ManifestError(f"{sample_id}: positions_per_bar must be {POSITIONS_PER_BAR}")

        has_midi = bool(sample.get("midi_path"))
        has_inline = isinstance(sample.get("notes"), list)
        if has_midi == has_inline:
            raise ManifestError(f"{sample_id}: provide exactly one of midi_path or notes")
        if has_midi and float(sample.get("bpm", 0) or 0) <= 0:
            raise ManifestError(f"{sample_id}: bpm is required for midi_path samples")
        validated.append(sample)
    return validated


def resolve_midi_path(manifest_path: Path, midi_path: str) -> Path:
    path = Path(midi_path)
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path


def load_groups_from_midi(sample: dict[str, Any], manifest_path: Path) -> list[dict[str, int]]:
    path = resolve_midi_path(manifest_path, str(sample["midi_path"]))
    if not path.exists():
        raise ManifestError(f"{sample['sample_id']}: midi_path does not exist: {path}")
    midi = pretty_midi.PrettyMIDI(str(path))
    bpm = float(sample["bpm"])
    groups: list[dict[str, int]] = []
    for instrument in midi.instruments:
        for note in instrument.notes:
            if int(note.pitch) < PIANO_PITCH_MIN or int(note.pitch) > PIANO_PITCH_MAX:
                continue
            bar, position = quantize_note_position(float(note.start), bpm)
            if 0 <= bar < int(sample["bar_count"]):
                groups.append({"bar": int(bar), "position": int(position), "pitch": int(note.pitch)})
    return sorted(groups, key=lambda item: (item["bar"], item["position"], item["pitch"]))


def load_groups_from_inline_notes(sample: dict[str, Any]) -> list[dict[str, int]]:
    groups: list[dict[str, int]] = []
    sample_id = str(sample["sample_id"])
    for index, raw_note in enumerate(sample.get("notes", [])):
        if not isinstance(raw_note, dict):
            raise ManifestError(f"{sample_id}: note {index} must be an object")
        bar = int(raw_note.get("bar", -1))
        position = int(raw_note.get("position", -1))
        pitch = int(raw_note.get("pitch", -1))
        if not (0 <= bar < int(sample["bar_count"])):
            raise ManifestError(f"{sample_id}: note {index} bar out of range: {bar}")
        if not (0 <= position < POSITIONS_PER_BAR):
            raise ManifestError(f"{sample_id}: note {index} position out of range: {position}")
        if not (PIANO_PITCH_MIN <= pitch <= PIANO_PITCH_MAX):
            raise ManifestError(f"{sample_id}: note {index} pitch out of range: {pitch}")
        groups.append({"bar": bar, "position": position, "pitch": pitch})
    return sorted(groups, key=lambda item: (item["bar"], item["position"], item["pitch"]))


def load_sample_groups(sample: dict[str, Any], manifest_path: Path) -> list[dict[str, int]]:
    if sample.get("midi_path"):
        return load_groups_from_midi(sample, manifest_path=manifest_path)
    return load_groups_from_inline_notes(sample)


def _empty_role_counts() -> dict[str, int]:
    return {role: 0 for role in PITCH_ROLE_KEYS}


def analyze_sample(sample: dict[str, Any], manifest_path: Path) -> dict[str, Any]:
    groups = load_sample_groups(sample, manifest_path=manifest_path)
    role_counts: Counter[str] = Counter(_empty_role_counts())
    bucket_counts: dict[str, Counter[str]] = {
        "strong": Counter(_empty_role_counts()),
        "eighth": Counter(_empty_role_counts()),
        "offgrid": Counter(_empty_role_counts()),
    }
    per_bar_counts: dict[str, Counter[str]] = {}

    for group in groups:
        chord = sample["chords"][int(group["bar"])]
        role = pitch_role_for_group(group, chord)
        role_counts[role] += 1
        bucket_counts[position_bucket(int(group["position"]))][role] += 1
        bar_key = str(group["bar"])
        if bar_key not in per_bar_counts:
            per_bar_counts[bar_key] = Counter(_empty_role_counts())
        per_bar_counts[bar_key][role] += 1

    note_count = int(len(groups))
    return {
        "sample_id": sample["sample_id"],
        "source_type": "midi" if sample.get("midi_path") else "inline_notes",
        "bar_count": int(sample["bar_count"]),
        "chords": sample["chords"],
        "note_count": note_count,
        "unique_pitch_count": int(len({int(group["pitch"]) for group in groups})),
        "role_counts": {role: int(role_counts.get(role, 0)) for role in PITCH_ROLE_KEYS},
        "role_ratios": pitch_role_ratio_metrics(dict(role_counts)),
        "bucket_counts": {
            bucket: {role: int(counter.get(role, 0)) for role in PITCH_ROLE_KEYS}
            for bucket, counter in bucket_counts.items()
        },
        "per_bar_counts": {
            bar: {role: int(counter.get(role, 0)) for role in PITCH_ROLE_KEYS}
            for bar, counter in sorted(per_bar_counts.items(), key=lambda item: int(item[0]))
        },
        "guide_tone_classes_by_bar": {
            str(index): sorted(int(pc) for pc in guide_tone_pitch_classes(chord))
            for index, chord in enumerate(sample["chords"])
        },
    }


def summarize_samples(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    role_counts: Counter[str] = Counter(_empty_role_counts())
    note_count = 0
    for sample in samples:
        note_count += int(sample.get("note_count", 0) or 0)
        role_counts.update({role: int(count) for role, count in sample.get("role_counts", {}).items()})
    return {
        "sample_count": int(len(samples)),
        "note_count": int(note_count),
        "role_counts": {role: int(role_counts.get(role, 0)) for role in PITCH_ROLE_KEYS},
        "role_ratios": pitch_role_ratio_metrics(dict(role_counts)),
    }


def markdown_report(report: dict[str, Any]) -> str:
    summary = report["summary"]
    ratios = summary["role_ratios"]
    lines = [
        "# Stage B Chord-Labeled Evaluation Subset",
        "",
        f"- manifest: `{report['manifest_path']}`",
        f"- sample count: `{summary['sample_count']}`",
        f"- note count: `{summary['note_count']}`",
        f"- chord-tone ratio: `{ratios['chord_tone_ratio']:.3f}`",
        f"- tension ratio: `{ratios['tension_ratio']:.3f}`",
        f"- approach ratio: `{ratios['approach_ratio']:.3f}`",
        f"- outside ratio: `{ratios['outside_ratio']:.3f}`",
        "",
        "## Samples",
        "",
        "| sample | bars | notes | unique pitches | chord-tone | tension | approach | outside |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for sample in report["samples"]:
        sample_ratios = sample["role_ratios"]
        lines.append(
            "| {sample_id} | {bar_count} | {note_count} | {unique_pitch_count} | {chord:.3f} | {tension:.3f} | {approach:.3f} | {outside:.3f} |".format(
                sample_id=sample["sample_id"],
                bar_count=sample["bar_count"],
                note_count=sample["note_count"],
                unique_pitch_count=sample["unique_pitch_count"],
                chord=sample_ratios["chord_tone_ratio"],
                tension=sample_ratios["tension_ratio"],
                approach=sample_ratios["approach_ratio"],
                outside=sample_ratios["outside_ratio"],
            )
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "- This is an evaluation contract, not a claim that real Brad/reference phrases are labeled.",
            "- Add real phrases only when their bar-level chord labels are known or manually verified.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def build_report(manifest_path: Path) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    samples = validate_manifest(manifest)
    sample_reports = [analyze_sample(sample, manifest_path=manifest_path) for sample in samples]
    return {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "schema_version": SCHEMA_VERSION,
        "manifest_path": str(manifest_path),
        "samples": sample_reports,
        "summary": summarize_samples(sample_reports),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate chord-labeled Stage B subset")
    parser.add_argument(
        "--manifest",
        type=str,
        default=str(ROOT_DIR / "data" / "eval" / "stage_b_chord_labeled_tiny" / "manifest.json"),
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=str(ROOT_DIR / "outputs" / "stage_b_chord_labeled_eval"),
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--min_samples", type=int, default=1)
    parser.add_argument("--min_notes", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest)
    report = build_report(manifest_path)
    summary = report["summary"]
    if int(summary["sample_count"]) < int(args.min_samples):
        raise ManifestError(f"sample_count below minimum: {summary['sample_count']} < {args.min_samples}")
    if int(summary["note_count"]) < int(args.min_notes):
        raise ManifestError(f"note_count below minimum: {summary['note_count']} < {args.min_notes}")

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_root) / run_id
    write_json(run_dir / "chord_labeled_eval_report.json", report)
    (run_dir / "chord_labeled_eval_report.md").write_text(markdown_report(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "sample_count": summary["sample_count"],
                "note_count": summary["note_count"],
                "chord_tone_ratio": summary["role_ratios"]["chord_tone_ratio"],
                "tension_ratio": summary["role_ratios"]["tension_ratio"],
                "outside_ratio": summary["role_ratios"]["outside_ratio"],
                "report_path": str(run_dir / "chord_labeled_eval_report.md"),
            },
            ensure_ascii=True,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
