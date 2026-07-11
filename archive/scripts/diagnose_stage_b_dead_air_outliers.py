"""Diagnose Stage B raw generation samples that fail the dead-air gate."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import pretty_midi


ROOT_DIR = Path(__file__).resolve().parents[1]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def non_drum_notes(midi_path: Path) -> list[pretty_midi.Note]:
    midi = pretty_midi.PrettyMIDI(str(midi_path))
    notes: list[pretty_midi.Note] = []
    for instrument in midi.instruments:
        if not instrument.is_drum:
            notes.extend(instrument.notes)
    return sorted(notes, key=lambda note: (float(note.start), int(note.pitch), float(note.end)))


def note_rows(notes: Sequence[pretty_midi.Note]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, note in enumerate(notes, start=1):
        rows.append(
            {
                "index": int(index),
                "start_sec": float(note.start),
                "end_sec": float(note.end),
                "duration_sec": max(0.0, float(note.end) - float(note.start)),
                "pitch": int(note.pitch),
                "velocity": int(note.velocity),
            }
        )
    return rows


def start_gap_rows(notes: Sequence[pretty_midi.Note], threshold_sec: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ordered = sorted(notes, key=lambda note: (float(note.start), int(note.pitch), float(note.end)))
    for index in range(1, len(ordered)):
        previous = ordered[index - 1]
        current = ordered[index]
        start_gap = max(0.0, float(current.start) - float(previous.start))
        silent_gap = max(0.0, float(current.start) - float(previous.end))
        rows.append(
            {
                "from_note": int(index),
                "to_note": int(index + 1),
                "from_pitch": int(previous.pitch),
                "to_pitch": int(current.pitch),
                "from_start_sec": float(previous.start),
                "to_start_sec": float(current.start),
                "start_gap_sec": start_gap,
                "silent_gap_sec": silent_gap,
                "is_dead_air_gap": bool(start_gap >= float(threshold_sec)),
            }
        )
    return rows


def _nested_float(source: dict[str, Any], section: str, key: str, default: float = 0.0) -> float:
    value = source.get(section, {}).get(key, default)
    return float(value if value is not None else default)


def _nested_int(source: dict[str, Any], section: str, key: str, default: int = 0) -> int:
    value = source.get(section, {}).get(key, default)
    return int(value if value is not None else default)


def _reason_contains_dead_air(sample: dict[str, Any]) -> bool:
    reason = str(sample.get("failure_reason") or sample.get("diagnostic_failure_reason") or "")
    return "dead-air" in reason


def compact_sample(
    sample: dict[str, Any],
    threshold_sec: float,
    dead_air_gate: float,
) -> dict[str, Any]:
    midi_path = Path(str(sample.get("midi_path", "")))
    notes = non_drum_notes(midi_path) if midi_path.exists() else []
    gaps = start_gap_rows(notes, threshold_sec=threshold_sec)
    dead_air_gaps = [row for row in gaps if row["is_dead_air_gap"]]
    max_start_gap_sec = max((float(row["start_gap_sec"]) for row in gaps), default=0.0)
    dead_air_ratio = _nested_float(sample, "metrics", "dead_air_ratio")
    postprocess = sample.get("postprocess", {}) if isinstance(sample.get("postprocess"), dict) else {}
    removed_note_count = int(postprocess.get("removed_note_count", 0) or 0)

    return {
        "sample_index": int(sample.get("sample_index", 0) or 0),
        "valid": bool(sample.get("valid", False)),
        "strict_valid": bool(sample.get("strict_valid", False)),
        "failure_reason": sample.get("failure_reason"),
        "dead_air_outlier": bool(dead_air_ratio >= float(dead_air_gate) or _reason_contains_dead_air(sample)),
        "midi_path": str(midi_path),
        "note_count": _nested_int(sample, "metrics", "note_count"),
        "unique_pitch_count": _nested_int(sample, "metrics", "unique_pitch_count"),
        "dead_air_ratio": dead_air_ratio,
        "phrase_coverage_ratio": _nested_float(sample, "metrics", "phrase_coverage_ratio"),
        "onset_coverage_ratio": _nested_float(sample, "temporal_coverage", "onset_coverage_ratio"),
        "sustained_coverage_ratio": _nested_float(sample, "temporal_coverage", "sustained_coverage_ratio"),
        "position_span_ratio": _nested_float(sample, "temporal_coverage", "position_span_ratio"),
        "head_empty_steps": _nested_int(sample, "temporal_coverage", "head_empty_steps"),
        "tail_empty_steps": _nested_int(sample, "temporal_coverage", "tail_empty_steps"),
        "longest_onset_empty_run_steps": _nested_int(sample, "temporal_coverage", "longest_onset_empty_run_steps"),
        "longest_sustained_empty_run_steps": _nested_int(
            sample,
            "temporal_coverage",
            "longest_sustained_empty_run_steps",
        ),
        "postprocess_removed_note_count": removed_note_count,
        "postprocess_removal_ratio": _nested_float(sample, "collapse", "postprocess_removal_ratio"),
        "collapse_warning": bool(sample.get("collapse", {}).get("collapse_warning", False)),
        "repeated_position_pitch_pair_ratio": _nested_float(
            sample,
            "collapse",
            "repeated_position_pitch_pair_ratio",
        ),
        "grammar_valid": bool(sample.get("grammar", {}).get("grammar_valid", False)),
        "invalid_token_count": _nested_int(sample, "grammar", "invalid_token_count"),
        "invalid_tokens_head": sample.get("grammar", {}).get("invalid_tokens_head", []),
        "generated_token_names_head": sample.get("generated_token_names_head", []),
        "gap_count": int(len(gaps)),
        "dead_air_gap_count": int(len(dead_air_gaps)),
        "max_start_gap_sec": max_start_gap_sec,
        "dead_air_gaps": dead_air_gaps,
        "note_rows_head": note_rows(notes[:16]),
    }


def build_summary(sample_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    outliers = [row for row in sample_rows if row["dead_air_outlier"]]
    strict_rows = [row for row in sample_rows if row["strict_valid"]]
    dead_air_values = [float(row["dead_air_ratio"]) for row in sample_rows]
    return {
        "sample_count": int(len(sample_rows)),
        "dead_air_outlier_count": int(len(outliers)),
        "outlier_sample_indices": [int(row["sample_index"]) for row in outliers],
        "strict_valid_sample_count": int(len(strict_rows)),
        "max_dead_air_ratio": max(dead_air_values) if dead_air_values else 0.0,
        "min_dead_air_ratio": min(dead_air_values) if dead_air_values else 0.0,
        "max_start_gap_sec": max((float(row["max_start_gap_sec"]) for row in sample_rows), default=0.0),
        "max_head_empty_steps": max((int(row["head_empty_steps"]) for row in sample_rows), default=0),
        "max_tail_empty_steps": max((int(row["tail_empty_steps"]) for row in sample_rows), default=0),
        "max_longest_sustained_empty_run_steps": max(
            (int(row["longest_sustained_empty_run_steps"]) for row in sample_rows),
            default=0,
        ),
    }


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def markdown_report(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Stage B Dead-Air Outlier Diagnostics",
        "",
        f"- source report: `{report['source_report_path']}`",
        f"- dead-air gate: `{report['dead_air_gate']:.3f}`",
        f"- threshold sec: `{report['dead_air_threshold_sec']:.3f}`",
        f"- sample count: `{summary['sample_count']}`",
        f"- outlier count: `{summary['dead_air_outlier_count']}`",
        f"- outlier sample indices: `{summary['outlier_sample_indices']}`",
        "",
        "| sample | valid | strict | notes | pitches | dead-air | phrase | onset | sustained | span | head | tail | longest sustained empty | removed | max start gap | reason |",
        "|---:|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in report["samples"]:
        lines.append(
            "| {sample_index} | {valid} | {strict_valid} | {note_count} | {unique_pitch_count} | "
            "{dead_air_ratio} | {phrase_coverage_ratio} | {onset_coverage_ratio} | "
            "{sustained_coverage_ratio} | {position_span_ratio} | {head_empty_steps} | "
            "{tail_empty_steps} | {longest_sustained_empty_run_steps} | "
            "{postprocess_removed_note_count} | {max_start_gap_sec} | {reason} |".format(
                sample_index=row["sample_index"],
                valid=row["valid"],
                strict_valid=row["strict_valid"],
                note_count=row["note_count"],
                unique_pitch_count=row["unique_pitch_count"],
                dead_air_ratio=_fmt(row["dead_air_ratio"]),
                phrase_coverage_ratio=_fmt(row["phrase_coverage_ratio"]),
                onset_coverage_ratio=_fmt(row["onset_coverage_ratio"]),
                sustained_coverage_ratio=_fmt(row["sustained_coverage_ratio"]),
                position_span_ratio=_fmt(row["position_span_ratio"]),
                head_empty_steps=row["head_empty_steps"],
                tail_empty_steps=row["tail_empty_steps"],
                longest_sustained_empty_run_steps=row["longest_sustained_empty_run_steps"],
                postprocess_removed_note_count=row["postprocess_removed_note_count"],
                max_start_gap_sec=_fmt(row["max_start_gap_sec"]),
                reason=row.get("failure_reason") or "none",
            )
        )
    lines.extend(["", "## Outlier Gap Detail", ""])
    for row in report["samples"]:
        if not row["dead_air_outlier"]:
            continue
        lines.append(f"### Sample {row['sample_index']}")
        lines.append("")
        lines.append(
            "- pattern: "
            f"head empty `{row['head_empty_steps']}` steps, "
            f"tail empty `{row['tail_empty_steps']}` steps, "
            f"sustained coverage `{row['sustained_coverage_ratio']:.3f}`, "
            f"dead-air gaps `{row['dead_air_gap_count']}/{row['gap_count']}`"
        )
        lines.append("- first generated tokens:")
        token_names = row.get("generated_token_names_head", [])
        lines.append("  - " + ", ".join(str(token) for token in token_names[:16]))
        lines.append("")
        lines.append("| from | to | from pitch | to pitch | start gap sec | silent gap sec |")
        lines.append("|---:|---:|---:|---:|---:|---:|")
        for gap in row["dead_air_gaps"]:
            lines.append(
                "| {from_note} | {to_note} | {from_pitch} | {to_pitch} | {start_gap_sec:.3f} | {silent_gap_sec:.3f} |".format(
                    **gap
                )
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def build_diagnostic_report(
    source_report_path: Path,
    dead_air_threshold_ms: float,
    dead_air_gate: float,
) -> dict[str, Any]:
    source = read_json(source_report_path)
    threshold_sec = float(dead_air_threshold_ms) / 1000.0
    sample_rows = [
        compact_sample(sample, threshold_sec=threshold_sec, dead_air_gate=dead_air_gate)
        for sample in source.get("samples", [])
        if isinstance(sample, dict)
    ]
    return {
        "source_report_path": str(source_report_path),
        "source_run_id": str(source.get("run_id", "")),
        "source_issue": int(source.get("issue", 0) or 0),
        "request": source.get("request", {}),
        "dead_air_threshold_sec": threshold_sec,
        "dead_air_gate": float(dead_air_gate),
        "summary": build_summary(sample_rows),
        "samples": sample_rows,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose Stage B dead-air outlier samples")
    parser.add_argument("--report_path", type=str, required=True)
    parser.add_argument(
        "--output_root",
        type=str,
        default=str(ROOT_DIR / "outputs" / "stage_b_dead_air_diagnostics"),
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--dead_air_threshold_ms", type=float, default=180.0)
    parser.add_argument("--dead_air_gate", type=float, default=0.8)
    parser.add_argument("--expected_outliers", type=int, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    report = build_diagnostic_report(
        source_report_path=Path(args.report_path),
        dead_air_threshold_ms=args.dead_air_threshold_ms,
        dead_air_gate=args.dead_air_gate,
    )
    write_json(output_dir / "dead_air_diagnostics.json", report)
    (output_dir / "dead_air_diagnostics.md").write_text(markdown_report(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True, indent=2))
    if args.expected_outliers is not None and report["summary"]["dead_air_outlier_count"] != int(args.expected_outliers):
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
