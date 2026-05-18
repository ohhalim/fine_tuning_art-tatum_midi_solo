"""
Analyze non-chord pitch classes in low chord-tone-ratio generation results.

Input is a generation contract sweep JSON produced by
scripts/eval_generation_contract_sweep.py. The report is diagnostic only; it is
not used as an acceptance gate.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT_PATH = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT_PATH))

from inference.app.fallback import chord_for_time, parse_chord
from inference.app.metrics import load_notes
from inference.app.schemas import GenerationRequest


PC_NAMES = ("C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B")


def pc_name(pitch_class: int) -> str:
    return PC_NAMES[int(pitch_class) % 12]


def chord_pitch_classes(chord: str) -> set[int]:
    root_pc, intervals = parse_chord(chord)
    return {(root_pc + interval) % 12 for interval in intervals}


def request_for_row(row: dict[str, Any], config: dict[str, Any]) -> GenerationRequest:
    return GenerationRequest(
        job_id=row["job_id"],
        bpm=int(config.get("bpm", 124)),
        chord_progression=list(row["chord_progression"]),
        bars=int(config.get("bars", 2)),
        time_signature=str(config.get("time_signature", "4/4")),
        section=str(config.get("section", "drop")),
        energy=str(config.get("energy", "high")),
        density=str(row.get("density", "medium")),
        seed=int(row.get("seed", 42)),
    )


def resolve_midi_path(row: dict[str, Any], root_dir: Path) -> Path | None:
    raw_path = row.get("midi_path")
    if not raw_path:
        return None
    midi_path = Path(raw_path)
    if not midi_path.is_absolute():
        midi_path = root_dir / midi_path
    return midi_path


def analyze_row(row: dict[str, Any], config: dict[str, Any], root_dir: Path, max_examples: int) -> dict[str, Any]:
    request = request_for_row(row, config)
    midi_path = resolve_midi_path(row, root_dir)
    if midi_path is None or not midi_path.exists():
        return {
            "job_id": row.get("job_id"),
            "error": f"missing MIDI path: {midi_path}",
        }

    notes = load_notes(midi_path)
    non_chord_by_chord: dict[str, Counter[str]] = defaultdict(Counter)
    chord_tone_count = 0
    non_chord_tone_count = 0
    examples: list[dict[str, Any]] = []

    for note in notes:
        chord = chord_for_time(request, float(note.start))
        allowed_pcs = chord_pitch_classes(chord)
        pitch_class = int(note.pitch) % 12
        if pitch_class in allowed_pcs:
            chord_tone_count += 1
            continue

        non_chord_tone_count += 1
        non_chord_by_chord[chord][pc_name(pitch_class)] += 1
        if len(examples) < max_examples:
            examples.append(
                {
                    "start_sec": round(float(note.start), 4),
                    "pitch": int(note.pitch),
                    "pitch_class": pc_name(pitch_class),
                    "active_chord": chord,
                    "allowed_pitch_classes": [pc_name(pc) for pc in sorted(allowed_pcs)],
                }
            )

    total = chord_tone_count + non_chord_tone_count
    recomputed_ratio = chord_tone_count / total if total else None
    top_non_chord = Counter()
    for counter in non_chord_by_chord.values():
        top_non_chord.update(counter)

    return {
        "job_id": row["job_id"],
        "density": row.get("density"),
        "seed": row.get("seed"),
        "progression_index": row.get("progression_index"),
        "chord_progression": row.get("chord_progression"),
        "midi_path": str(midi_path),
        "reported_chord_tone_ratio": row.get("chord_tone_ratio"),
        "recomputed_chord_tone_ratio": recomputed_ratio,
        "note_count": len(notes),
        "chord_tone_count": chord_tone_count,
        "non_chord_tone_count": non_chord_tone_count,
        "top_non_chord_pitch_classes": dict(top_non_chord.most_common()),
        "non_chord_pitch_classes_by_active_chord": {
            chord: dict(counter.most_common()) for chord, counter in sorted(non_chord_by_chord.items())
        },
        "non_chord_note_examples": examples,
    }


def select_rows(rows: list[dict[str, Any]], threshold: float, max_rows: int | None) -> list[dict[str, Any]]:
    selected = [
        row
        for row in rows
        if row.get("status") == "COMPLETED"
        and isinstance(row.get("chord_tone_ratio"), (int, float))
        and float(row["chord_tone_ratio"]) <= threshold
    ]
    selected.sort(key=lambda row: (float(row["chord_tone_ratio"]), str(row.get("density", "")), str(row["job_id"])))
    return selected[:max_rows] if max_rows is not None else selected


def summarize(analyses: list[dict[str, Any]], source_rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    ratio_values = [
        float(row["chord_tone_ratio"])
        for row in source_rows
        if isinstance(row.get("chord_tone_ratio"), (int, float))
    ]
    selected_ratios = [
        float(item["recomputed_chord_tone_ratio"])
        for item in analyses
        if isinstance(item.get("recomputed_chord_tone_ratio"), (int, float))
    ]
    top_non_chord = Counter()
    by_density: dict[str, dict[str, Any]] = {}

    for item in analyses:
        top_non_chord.update(item.get("top_non_chord_pitch_classes", {}))

    for density in sorted({str(item.get("density")) for item in analyses if item.get("density") is not None}):
        items = [item for item in analyses if item.get("density") == density]
        ratios = [
            float(item["recomputed_chord_tone_ratio"])
            for item in items
            if isinstance(item.get("recomputed_chord_tone_ratio"), (int, float))
        ]
        by_density[density] = {
            "low_sample_count": len(items),
            "avg_recomputed_chord_tone_ratio": mean(ratios) if ratios else None,
        }

    return {
        "threshold": threshold,
        "source_row_count": len(source_rows),
        "source_avg_chord_tone_ratio": mean(ratio_values) if ratio_values else None,
        "analyzed_low_sample_count": len(analyses),
        "analyzed_avg_chord_tone_ratio": mean(selected_ratios) if selected_ratios else None,
        "top_non_chord_pitch_classes": dict(top_non_chord.most_common()),
        "by_density": by_density,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n")


def fmt_ratio(value: Any) -> str:
    return f"{value:.3f}" if isinstance(value, (int, float)) else ""


def write_markdown(path: Path, report: dict[str, Any], detail_limit: int) -> None:
    summary = report["summary"]
    lines = [
        "# Chord-tone Error Analysis",
        "",
        "This report analyzes generated MIDI rows whose chord-tone ratio is at or below the configured threshold. The metric is a pitch-class proxy and is not an acceptance gate.",
        "",
        "## Summary",
        "",
        f"- source: `{report['source_path']}`",
        f"- threshold: `{summary['threshold']}`",
        f"- source rows: `{summary['source_row_count']}`",
        f"- source avg chord-tone ratio: `{fmt_ratio(summary['source_avg_chord_tone_ratio'])}`",
        f"- analyzed low samples: `{summary['analyzed_low_sample_count']}`",
        f"- analyzed avg chord-tone ratio: `{fmt_ratio(summary['analyzed_avg_chord_tone_ratio'])}`",
        f"- top non-chord pitch classes: `{summary['top_non_chord_pitch_classes']}`",
        "",
        "## By Density",
        "",
        "| Density | Low Samples | Avg Ratio |",
        "|---|---:|---:|",
    ]
    for density, density_summary in summary["by_density"].items():
        lines.append(
            "| {density} | {count} | {ratio} |".format(
                density=density,
                count=density_summary["low_sample_count"],
                ratio=fmt_ratio(density_summary["avg_recomputed_chord_tone_ratio"]),
            )
        )

    lines.extend(
        [
            "",
            "## Low Samples",
            "",
            "| Job | Density | Seed | Ratio | Notes | Top Non-chord PCs |",
            "|---|---|---:|---:|---:|---|",
        ]
    )
    for item in report["analyses"]:
        lines.append(
            "| {job} | {density} | {seed} | {ratio} | {notes} | `{top}` |".format(
                job=item["job_id"],
                density=item.get("density", ""),
                seed=item.get("seed", ""),
                ratio=fmt_ratio(item.get("recomputed_chord_tone_ratio")),
                notes=item.get("note_count", ""),
                top=item.get("top_non_chord_pitch_classes", {}),
            )
        )

    lines.extend(["", "## Details", ""])
    for item in report["analyses"][:detail_limit]:
        lines.extend(
            [
                f"### {item['job_id']}",
                "",
                f"- density: `{item.get('density')}`",
                f"- seed: `{item.get('seed')}`",
                f"- ratio: `{fmt_ratio(item.get('recomputed_chord_tone_ratio'))}`",
                f"- chord progression: `{item.get('chord_progression')}`",
                f"- non-chord by active chord: `{item.get('non_chord_pitch_classes_by_active_chord')}`",
                "",
                "| Start Sec | Pitch | Pitch Class | Active Chord | Allowed PCs |",
                "|---:|---:|---|---|---|",
            ]
        )
        for example in item["non_chord_note_examples"]:
            lines.append(
                "| {start} | {pitch} | {pc} | {chord} | `{allowed}` |".format(
                    start=example["start_sec"],
                    pitch=example["pitch"],
                    pc=example["pitch_class"],
                    chord=example["active_chord"],
                    allowed=example["allowed_pitch_classes"],
                )
            )
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze low chord-tone-ratio sweep outputs")
    parser.add_argument("--input_json", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--max_examples", type=int, default=16)
    parser.add_argument("--detail_limit", type=int, default=12)
    parser.add_argument(
        "--output_json",
        type=str,
        default=str(PROJECT_ROOT_PATH / "outputs" / "sweeps" / "chord_tone_error_analysis.json"),
    )
    parser.add_argument(
        "--output_md",
        type=str,
        default=str(PROJECT_ROOT_PATH / "outputs" / "sweeps" / "chord_tone_error_analysis.md"),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    input_path = Path(args.input_json)
    payload = json.loads(input_path.read_text())
    config = payload.get("config", {})
    rows = payload.get("rows", [])
    selected_rows = select_rows(rows, threshold=float(args.threshold), max_rows=args.max_rows)
    analyses = [
        analyze_row(row, config=config, root_dir=PROJECT_ROOT_PATH, max_examples=int(args.max_examples))
        for row in selected_rows
    ]
    report = {
        "source_path": str(input_path),
        "summary": summarize(analyses, rows, threshold=float(args.threshold)),
        "analyses": analyses,
    }

    write_json(Path(args.output_json), report)
    write_markdown(Path(args.output_md), report, detail_limit=int(args.detail_limit))
    print(json.dumps(report["summary"], ensure_ascii=True, indent=2))
    print(f"Saved JSON: {args.output_json}")
    print(f"Saved Markdown: {args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
