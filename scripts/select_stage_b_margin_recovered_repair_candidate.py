"""Select a Stage B margin-recovered repair candidate from an expanded generation report."""

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

from scripts.build_stage_b_margin_recovered_listening_notes import write_json  # noqa: E402
from scripts.review_stage_b_margin_recovered_focused_context import (  # noqa: E402
    duplicated_pitch_class_chunks,
    max_simultaneous_notes,
)
from scripts.run_stage_b_generation_probe import postprocess_stage_b_midi  # noqa: E402


class MarginRecoveredRepairSelectionError(ValueError):
    pass


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def non_drum_notes(midi: pretty_midi.PrettyMIDI) -> list[pretty_midi.Note]:
    notes: list[pretty_midi.Note] = []
    for instrument in midi.instruments:
        if not instrument.is_drum:
            notes.extend(instrument.notes)
    return sorted(notes, key=lambda note: (float(note.start), int(note.pitch), float(note.end)))


def focused_solo_metrics(midi_path: Path) -> dict[str, Any]:
    midi = pretty_midi.PrettyMIDI(str(midi_path))
    postprocess = postprocess_stage_b_midi(midi, simultaneous_limit=1)
    notes = non_drum_notes(midi)
    pitches = [int(note.pitch) for note in notes]
    intervals = [pitches[index + 1] - pitches[index] for index in range(len(pitches) - 1)]
    return {
        "focused_note_count": int(len(notes)),
        "focused_unique_pitch_count": int(len(set(pitches))),
        "focused_pitch_min": int(min(pitches)) if pitches else None,
        "focused_pitch_max": int(max(pitches)) if pitches else None,
        "focused_max_simultaneous_notes": max_simultaneous_notes(notes),
        "focused_adjacent_pitch_repeats": int(sum(1 for interval in intervals if interval == 0)),
        "focused_duplicated_3_note_pitch_class_chunks": duplicated_pitch_class_chunks(pitches, 3),
        "focused_postprocess_removed_note_count": int(postprocess.get("removed_note_count", 0) or 0),
        "focused_postprocess_removal_ratio": (
            float(postprocess.get("removed_note_count", 0) or 0)
            / max(1.0, float(postprocess.get("before_note_count", 0) or 0))
        ),
    }


def candidate_score(candidate: dict[str, Any]) -> float:
    metrics = candidate["metrics"]
    focused = candidate["focused_solo_metrics"]
    score = 0.0
    score += (1.0 - min(1.0, float(metrics.get("dead_air_ratio", 1.0) or 1.0))) * 60.0
    score += min(8, int(focused.get("focused_unique_pitch_count", 0) or 0)) * 6.0
    score += min(14, int(focused.get("focused_note_count", 0) or 0)) * 1.0
    score += float(candidate.get("temporal_coverage", {}).get("onset_coverage_ratio", 0.0) or 0.0) * 10.0
    score += float(candidate.get("temporal_coverage", {}).get("sustained_coverage_ratio", 0.0) or 0.0) * 5.0
    score += float(metrics.get("phrase_coverage_ratio", 0.0) or 0.0) * 5.0
    score -= int(focused.get("focused_adjacent_pitch_repeats", 0) or 0) * 1.0
    score -= int(focused.get("focused_duplicated_3_note_pitch_class_chunks", 0) or 0) * 8.0
    score -= float(focused.get("focused_postprocess_removal_ratio", 0.0) or 0.0) * 6.0
    if not bool(candidate.get("strict_valid", False)):
        score -= 25.0
    return round(float(score), 6)


def build_candidates(report: dict[str, Any]) -> list[dict[str, Any]]:
    samples = report.get("samples")
    if not isinstance(samples, list) or not samples:
        raise MarginRecoveredRepairSelectionError("report must contain non-empty samples")
    rows: list[dict[str, Any]] = []
    seed = int(report.get("request", {}).get("seed", 0) or 0)
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        midi_path = Path(str(sample.get("midi_path") or ""))
        if not midi_path.exists():
            continue
        row = {
            "candidate_id": "margin_recovered_repair_seed_{seed}_sample_{sample}".format(
                seed=seed,
                sample=int(sample.get("sample_index", 0) or 0),
            ),
            "sample_index": int(sample.get("sample_index", 0) or 0),
            "sample_seed": int(sample.get("sample_seed", seed) or seed),
            "midi_path": str(midi_path),
            "valid": bool(sample.get("valid", False)),
            "strict_valid": bool(sample.get("strict_valid", False)),
            "grammar_gate_passed": bool(sample.get("grammar_gate_passed", False)),
            "metrics": dict(sample.get("metrics") or {}),
            "temporal_coverage": dict(sample.get("temporal_coverage") or {}),
            "focused_solo_metrics": focused_solo_metrics(midi_path),
        }
        row["repair_score"] = candidate_score(row)
        rows.append(row)
    if not rows:
        raise MarginRecoveredRepairSelectionError("no readable sample MIDI files found")
    rows.sort(
        key=lambda row: (
            -float(row["repair_score"]),
            float(row["metrics"].get("dead_air_ratio", 1.0) or 1.0),
            -int(row["focused_solo_metrics"].get("focused_unique_pitch_count", 0) or 0),
            int(row["sample_index"]),
        )
    )
    for index, row in enumerate(rows, start=1):
        row["repair_rank"] = int(index)
    return rows


def build_selection_report(
    generation_report: dict[str, Any],
    *,
    output_dir: Path,
    baseline_candidate_id: str,
    baseline_dead_air: float,
    baseline_unique_pitch_count: int,
) -> dict[str, Any]:
    candidates = build_candidates(generation_report)
    selected = candidates[0]
    selected_metrics = selected["metrics"]
    selected_focused = selected["focused_solo_metrics"]
    dead_air_delta = float(baseline_dead_air) - float(selected_metrics.get("dead_air_ratio", 0.0) or 0.0)
    unique_delta = int(selected_focused.get("focused_unique_pitch_count", 0) or 0) - int(baseline_unique_pitch_count)
    remaining_flags: list[str] = []
    if int(selected_focused.get("focused_unique_pitch_count", 0) or 0) < 6:
        remaining_flags.append("low_pitch_variety")
    if float(selected_metrics.get("dead_air_ratio", 0.0) or 0.0) > 0.40:
        remaining_flags.append("dead_air_needs_review")
    if int(selected_focused.get("focused_note_count", 0) or 0) < 12:
        remaining_flags.append("too_sparse_for_context_review")
    if int(selected_focused.get("focused_duplicated_3_note_pitch_class_chunks", 0) or 0) > 0:
        remaining_flags.append("repeated_pitch_class_cell")
    return {
        "schema_version": "stage_b_margin_recovered_pitch_dead_air_repair_selection_v1",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "source_generation_report": str(generation_report.get("run_dir") or generation_report.get("run_id") or ""),
        "baseline": {
            "candidate_id": baseline_candidate_id,
            "dead_air_ratio": float(baseline_dead_air),
            "focused_unique_pitch_count": int(baseline_unique_pitch_count),
        },
        "candidate_count": int(len(candidates)),
        "selected_candidate": selected,
        "repair_summary": {
            "selected_candidate_id": selected["candidate_id"],
            "selected_sample_index": int(selected["sample_index"]),
            "baseline_dead_air_ratio": float(baseline_dead_air),
            "selected_dead_air_ratio": float(selected_metrics.get("dead_air_ratio", 0.0) or 0.0),
            "dead_air_delta": round(float(dead_air_delta), 6),
            "baseline_focused_unique_pitch_count": int(baseline_unique_pitch_count),
            "selected_focused_unique_pitch_count": int(
                selected_focused.get("focused_unique_pitch_count", 0) or 0
            ),
            "focused_unique_pitch_delta": int(unique_delta),
            "remaining_flags": remaining_flags,
            "partial_repair": bool(dead_air_delta > 0.0 or unique_delta > 0),
            "focused_keep_ready": not remaining_flags,
        },
        "candidates": candidates,
    }


def markdown_report(report: dict[str, Any]) -> str:
    summary = report["repair_summary"]
    lines = [
        "# Stage B Margin-Recovered Pitch/Dead-Air Repair Selection",
        "",
        f"- candidate count: `{report['candidate_count']}`",
        f"- selected candidate: `{summary['selected_candidate_id']}`",
        f"- selected sample: `{summary['selected_sample_index']}`",
        f"- dead-air delta: `{summary['dead_air_delta']:.3f}`",
        f"- focused unique pitch delta: `{summary['focused_unique_pitch_delta']}`",
        f"- focused keep ready: `{summary['focused_keep_ready']}`",
        f"- remaining flags: `{summary['remaining_flags']}`",
        "",
        "| rank | candidate | strict | score | notes | unique | dead-air | onset | sustained | dup3 | adj repeat |",
        "|---:|---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for candidate in report["candidates"]:
        focused = candidate["focused_solo_metrics"]
        metrics = candidate["metrics"]
        temporal = candidate["temporal_coverage"]
        lines.append(
            "| {rank} | `{candidate_id}` | {strict} | {score:.3f} | {notes} | {unique} | "
            "{dead_air:.3f} | {onset:.3f} | {sustained:.3f} | {dup3} | {adj} |".format(
                rank=candidate["repair_rank"],
                candidate_id=candidate["candidate_id"],
                strict=candidate["strict_valid"],
                score=float(candidate["repair_score"]),
                notes=int(focused["focused_note_count"]),
                unique=int(focused["focused_unique_pitch_count"]),
                dead_air=float(metrics.get("dead_air_ratio", 0.0) or 0.0),
                onset=float(temporal.get("onset_coverage_ratio", 0.0) or 0.0),
                sustained=float(temporal.get("sustained_coverage_ratio", 0.0) or 0.0),
                dup3=int(focused["focused_duplicated_3_note_pitch_class_chunks"]),
                adj=int(focused["focused_adjacent_pitch_repeats"]),
            )
        )
    return "\n".join(lines).rstrip() + "\n"


def validate_selection(
    report: dict[str, Any],
    *,
    expected_sample_index: int | None,
    require_partial_repair: bool,
) -> dict[str, Any]:
    summary = report["repair_summary"]
    if expected_sample_index is not None and int(summary["selected_sample_index"]) != int(expected_sample_index):
        raise MarginRecoveredRepairSelectionError(
            f"expected sample {expected_sample_index}, got {summary['selected_sample_index']}"
        )
    if require_partial_repair and not bool(summary["partial_repair"]):
        raise MarginRecoveredRepairSelectionError("selected candidate did not improve baseline")
    return {
        "candidate_count": int(report["candidate_count"]),
        "selected_candidate_id": str(summary["selected_candidate_id"]),
        "selected_sample_index": int(summary["selected_sample_index"]),
        "dead_air_delta": float(summary["dead_air_delta"]),
        "focused_unique_pitch_delta": int(summary["focused_unique_pitch_delta"]),
        "focused_keep_ready": bool(summary["focused_keep_ready"]),
        "remaining_flags": list(summary["remaining_flags"]),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Select margin-recovered pitch/dead-air repair candidate")
    parser.add_argument("--report_path", type=str, required=True)
    parser.add_argument(
        "--output_root",
        type=str,
        default=str(ROOT_DIR / "outputs" / "stage_b_margin_recovered_pitch_dead_air_repair"),
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--baseline_candidate_id", type=str, default="margin_recovered_rank_2_seed_31_sample_5")
    parser.add_argument("--baseline_dead_air", type=float, default=0.4444444444444444)
    parser.add_argument("--baseline_unique_pitch_count", type=int, default=4)
    parser.add_argument("--expected_sample_index", type=int, default=None)
    parser.add_argument("--require_partial_repair", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    generation_report = read_json(Path(args.report_path))
    report = build_selection_report(
        generation_report,
        output_dir=output_dir,
        baseline_candidate_id=str(args.baseline_candidate_id),
        baseline_dead_air=float(args.baseline_dead_air),
        baseline_unique_pitch_count=int(args.baseline_unique_pitch_count),
    )
    summary = validate_selection(
        report,
        expected_sample_index=args.expected_sample_index,
        require_partial_repair=bool(args.require_partial_repair),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "repair_candidate_selection.json", report)
    write_json(output_dir / "repair_candidate_selection_summary.json", summary)
    (output_dir / "repair_candidate_selection.md").write_text(markdown_report(report), encoding="utf-8")
    summary.update(
        {
            "selection_path": str(output_dir / "repair_candidate_selection.json"),
            "markdown_path": str(output_dir / "repair_candidate_selection.md"),
        }
    )
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
