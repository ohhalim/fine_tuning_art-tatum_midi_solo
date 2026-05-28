"""Summarize a Stage B phrase/vocabulary repair sweep for the timing/repetition candidate."""

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
from scripts.select_stage_b_margin_recovered_repair_candidate import (  # noqa: E402
    build_candidates,
    non_drum_notes,
    read_json,
)
from scripts.run_stage_b_generation_probe import postprocess_stage_b_midi  # noqa: E402
from scripts.summarize_stage_b_margin_recovered_pitch_vocab_sweep import safe_label  # noqa: E402
from scripts.summarize_stage_b_margin_recovered_timing_repetition_repair import float_value  # noqa: E402


class MarginRecoveredPhraseVocabularyRepairError(ValueError):
    pass


def focused_max_interval(midi_path: Path) -> int:
    midi = pretty_midi.PrettyMIDI(str(midi_path))
    postprocess_stage_b_midi(midi, simultaneous_limit=1)
    pitches = [int(note.pitch) for note in non_drum_notes(midi)]
    intervals = [abs(pitches[index + 1] - pitches[index]) for index in range(len(pitches) - 1)]
    return int(max(intervals, default=0))


def candidate_gate_flags(
    candidate: dict[str, Any],
    *,
    min_unique_pitch_count: int,
    max_dead_air_ratio_exclusive: float,
    min_note_count: int,
    max_simultaneous_notes: int,
    max_duplicated_3_note_chunks: int,
    max_adjacent_pitch_repeats_exclusive: int,
    max_interval_exclusive: int,
) -> list[str]:
    focused = candidate["focused_solo_metrics"]
    metrics = candidate["metrics"]
    flags: list[str] = []
    if int(focused.get("focused_unique_pitch_count", 0) or 0) < min_unique_pitch_count:
        flags.append("low_pitch_variety")
    if float_value(metrics.get("dead_air_ratio"), 1.0) >= max_dead_air_ratio_exclusive:
        flags.append("dead_air_not_repaired")
    if int(focused.get("focused_note_count", 0) or 0) < min_note_count:
        flags.append("too_sparse_for_context_review")
    if int(focused.get("focused_max_simultaneous_notes", 0) or 0) > max_simultaneous_notes:
        flags.append("focused_polyphony")
    if int(focused.get("focused_duplicated_3_note_pitch_class_chunks", 0) or 0) > max_duplicated_3_note_chunks:
        flags.append("repeated_pitch_class_cell")
    if int(focused.get("focused_adjacent_pitch_repeats", 0) or 0) >= max_adjacent_pitch_repeats_exclusive:
        flags.append("adjacent_repetition_not_repaired")
    if int(focused.get("focused_max_interval", 0) or 0) >= max_interval_exclusive:
        flags.append("wide_interval_not_repaired")
    return flags


def repair_score(candidate: dict[str, Any], *, qualified: bool) -> float:
    focused = candidate["focused_solo_metrics"]
    metrics = candidate["metrics"]
    temporal = candidate["temporal_coverage"]
    score = 1000.0 if qualified else 0.0
    score += (1.0 - min(1.0, float_value(metrics.get("dead_air_ratio"), 1.0))) * 180.0
    score += min(10, int(focused.get("focused_unique_pitch_count", 0) or 0)) * 24.0
    score += min(16, int(focused.get("focused_note_count", 0) or 0)) * 3.0
    score += float(temporal.get("onset_coverage_ratio", 0.0) or 0.0) * 20.0
    score += float(temporal.get("sustained_coverage_ratio", 0.0) or 0.0) * 10.0
    score -= int(focused.get("focused_adjacent_pitch_repeats", 0) or 0) * 20.0
    score -= max(0, int(focused.get("focused_max_interval", 0) or 0) - 7) * 8.0
    score -= int(focused.get("focused_duplicated_3_note_pitch_class_chunks", 0) or 0) * 40.0
    return round(float(score), 6)


def enrich_candidates(
    report: dict[str, Any],
    *,
    report_path: Path,
    min_unique_pitch_count: int,
    max_dead_air_ratio_exclusive: float,
    min_note_count: int,
    max_simultaneous_notes: int,
    max_duplicated_3_note_chunks: int,
    max_adjacent_pitch_repeats_exclusive: int,
    max_interval_exclusive: int,
) -> list[dict[str, Any]]:
    source_run_id = str(report.get("run_id") or report_path.parent.name)
    request = dict(report.get("request") or {})
    summary = dict(report.get("summary") or {})
    request_seed = int(request.get("seed", 0) or 0)
    top_k = request.get("top_k")
    temperature = request.get("temperature")
    sample_count = int(summary.get("sample_count", len(report.get("samples") or [])) or 0)
    rows = build_candidates(report)
    enriched: list[dict[str, Any]] = []
    for row in rows:
        row = dict(row)
        focused = dict(row["focused_solo_metrics"])
        focused["focused_max_interval"] = focused_max_interval(Path(str(row["midi_path"])))
        row["focused_solo_metrics"] = focused
        row["source_run_id"] = source_run_id
        row["source_report_path"] = str(report_path)
        row["source_request"] = {
            "seed": request_seed,
            "top_k": top_k,
            "temperature": temperature,
            "sample_count": sample_count,
        }
        row["candidate_id"] = (
            "margin_recovered_phrase_vocab_seed_{seed}_topk_{top_k}_temp_{temperature}_n{count}_sample_{sample}".format(
                seed=request_seed,
                top_k=safe_label(top_k),
                temperature=safe_label(temperature),
                count=sample_count,
                sample=int(row.get("sample_index", 0) or 0),
            )
        )
        flags = candidate_gate_flags(
            row,
            min_unique_pitch_count=min_unique_pitch_count,
            max_dead_air_ratio_exclusive=max_dead_air_ratio_exclusive,
            min_note_count=min_note_count,
            max_simultaneous_notes=max_simultaneous_notes,
            max_duplicated_3_note_chunks=max_duplicated_3_note_chunks,
            max_adjacent_pitch_repeats_exclusive=max_adjacent_pitch_repeats_exclusive,
            max_interval_exclusive=max_interval_exclusive,
        )
        row["phrase_vocabulary_gate"] = {
            "qualified": not flags,
            "flags": flags,
        }
        row["phrase_vocabulary_score"] = repair_score(row, qualified=not flags)
        enriched.append(row)
    return enriched


def build_repair_report(
    report_paths: list[Path],
    *,
    output_dir: Path,
    previous_candidate_id: str,
    previous_dead_air: float,
    previous_unique_pitch_count: int,
    previous_note_count: int,
    previous_adjacent_pitch_repeats: int,
    previous_max_interval: int,
    min_unique_pitch_count: int,
    max_dead_air_ratio_exclusive: float,
    min_note_count: int,
    max_simultaneous_notes: int,
    max_duplicated_3_note_chunks: int,
    max_adjacent_pitch_repeats_exclusive: int,
    max_interval_exclusive: int,
) -> dict[str, Any]:
    if not report_paths:
        raise MarginRecoveredPhraseVocabularyRepairError("at least one report path is required")
    candidates: list[dict[str, Any]] = []
    for path in report_paths:
        report = read_json(path)
        candidates.extend(
            enrich_candidates(
                report,
                report_path=path,
                min_unique_pitch_count=min_unique_pitch_count,
                max_dead_air_ratio_exclusive=max_dead_air_ratio_exclusive,
                min_note_count=min_note_count,
                max_simultaneous_notes=max_simultaneous_notes,
                max_duplicated_3_note_chunks=max_duplicated_3_note_chunks,
                max_adjacent_pitch_repeats_exclusive=max_adjacent_pitch_repeats_exclusive,
                max_interval_exclusive=max_interval_exclusive,
            )
        )
    if not candidates:
        raise MarginRecoveredPhraseVocabularyRepairError("no candidates found")
    candidates.sort(
        key=lambda row: (
            not bool(row["phrase_vocabulary_gate"]["qualified"]),
            -float(row["phrase_vocabulary_score"]),
            int(row["focused_solo_metrics"].get("focused_adjacent_pitch_repeats", 0) or 0),
            int(row["focused_solo_metrics"].get("focused_max_interval", 0) or 0),
            float_value(row["metrics"].get("dead_air_ratio"), 1.0),
            -int(row["focused_solo_metrics"].get("focused_unique_pitch_count", 0) or 0),
            -int(row["focused_solo_metrics"].get("focused_note_count", 0) or 0),
            str(row["source_run_id"]),
            int(row["sample_index"]),
        )
    )
    for index, row in enumerate(candidates, start=1):
        row["repair_rank"] = int(index)
    selected = candidates[0]
    focused = selected["focused_solo_metrics"]
    metrics = selected["metrics"]
    selected_dead_air = float_value(metrics.get("dead_air_ratio"), 0.0)
    selected_unique = int(focused.get("focused_unique_pitch_count", 0) or 0)
    selected_note_count = int(focused.get("focused_note_count", 0) or 0)
    selected_adjacent_repeats = int(focused.get("focused_adjacent_pitch_repeats", 0) or 0)
    selected_max_interval = int(focused.get("focused_max_interval", 0) or 0)
    qualified_count = sum(1 for row in candidates if row["phrase_vocabulary_gate"]["qualified"])
    return {
        "schema_version": "stage_b_margin_recovered_phrase_vocabulary_repair_v1",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "source_report_paths": [str(path) for path in report_paths],
        "thresholds": {
            "min_focused_unique_pitch_count": int(min_unique_pitch_count),
            "max_dead_air_ratio_exclusive": float(max_dead_air_ratio_exclusive),
            "min_focused_note_count": int(min_note_count),
            "max_focused_simultaneous_notes": int(max_simultaneous_notes),
            "max_duplicated_3_note_pitch_class_chunks": int(max_duplicated_3_note_chunks),
            "max_adjacent_pitch_repeats_exclusive": int(max_adjacent_pitch_repeats_exclusive),
            "max_interval_exclusive": int(max_interval_exclusive),
        },
        "previous_timing_repetition_candidate": {
            "candidate_id": previous_candidate_id,
            "dead_air_ratio": float(previous_dead_air),
            "focused_unique_pitch_count": int(previous_unique_pitch_count),
            "focused_note_count": int(previous_note_count),
            "focused_adjacent_pitch_repeats": int(previous_adjacent_pitch_repeats),
            "focused_max_interval": int(previous_max_interval),
        },
        "report_count": int(len(report_paths)),
        "candidate_count": int(len(candidates)),
        "qualified_candidate_count": int(qualified_count),
        "selected_candidate": selected,
        "repair_summary": {
            "selected_candidate_id": selected["candidate_id"],
            "selected_source_run_id": selected["source_run_id"],
            "selected_sample_index": int(selected["sample_index"]),
            "selected_sample_seed": int(selected.get("sample_seed", 0) or 0),
            "qualified": bool(selected["phrase_vocabulary_gate"]["qualified"]),
            "remaining_flags": list(selected["phrase_vocabulary_gate"]["flags"]),
            "previous_dead_air_ratio": float(previous_dead_air),
            "selected_dead_air_ratio": selected_dead_air,
            "dead_air_delta_from_previous": round(float(previous_dead_air) - selected_dead_air, 6),
            "previous_focused_unique_pitch_count": int(previous_unique_pitch_count),
            "selected_focused_unique_pitch_count": selected_unique,
            "focused_unique_pitch_delta_from_previous": int(selected_unique - previous_unique_pitch_count),
            "previous_focused_note_count": int(previous_note_count),
            "selected_focused_note_count": selected_note_count,
            "focused_note_delta_from_previous": int(selected_note_count - previous_note_count),
            "previous_adjacent_pitch_repeats": int(previous_adjacent_pitch_repeats),
            "selected_adjacent_pitch_repeats": selected_adjacent_repeats,
            "adjacent_pitch_repeat_delta_from_previous": int(
                previous_adjacent_pitch_repeats - selected_adjacent_repeats
            ),
            "previous_max_interval": int(previous_max_interval),
            "selected_max_interval": selected_max_interval,
            "max_interval_delta_from_previous": int(previous_max_interval - selected_max_interval),
            "phrase_vocabulary_improved": bool(
                selected_adjacent_repeats < int(previous_adjacent_pitch_repeats)
                and selected_max_interval < int(previous_max_interval)
            ),
        },
        "candidates": candidates,
    }


def markdown_report(report: dict[str, Any]) -> str:
    summary = report["repair_summary"]
    lines = [
        "# Stage B Margin-Recovered Phrase/Vocabulary Repair",
        "",
        f"- report count: `{report['report_count']}`",
        f"- candidate count: `{report['candidate_count']}`",
        f"- qualified candidate count: `{report['qualified_candidate_count']}`",
        f"- selected candidate: `{summary['selected_candidate_id']}`",
        f"- selected source run: `{summary['selected_source_run_id']}`",
        f"- selected sample: `{summary['selected_sample_index']}`",
        f"- selected sample seed: `{summary['selected_sample_seed']}`",
        f"- qualified: `{summary['qualified']}`",
        f"- remaining flags: `{summary['remaining_flags']}`",
        f"- adjacent repeat delta from previous: `{summary['adjacent_pitch_repeat_delta_from_previous']}`",
        f"- max interval delta from previous: `{summary['max_interval_delta_from_previous']}`",
        "",
        "| rank | candidate | qualified | score | notes | unique | dead-air | onset | sustained | adj repeat | max interval | flags |",
        "|---:|---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for candidate in report["candidates"][:20]:
        focused = candidate["focused_solo_metrics"]
        metrics = candidate["metrics"]
        temporal = candidate["temporal_coverage"]
        gate = candidate["phrase_vocabulary_gate"]
        lines.append(
            "| {rank} | `{candidate_id}` | {qualified} | {score:.3f} | {notes} | {unique} | "
            "{dead_air:.3f} | {onset:.3f} | {sustained:.3f} | {adj} | {max_interval} | `{flags}` |".format(
                rank=int(candidate["repair_rank"]),
                candidate_id=candidate["candidate_id"],
                qualified=bool(gate["qualified"]),
                score=float(candidate["phrase_vocabulary_score"]),
                notes=int(focused["focused_note_count"]),
                unique=int(focused["focused_unique_pitch_count"]),
                dead_air=float_value(metrics.get("dead_air_ratio"), 0.0),
                onset=float(temporal.get("onset_coverage_ratio", 0.0) or 0.0),
                sustained=float(temporal.get("sustained_coverage_ratio", 0.0) or 0.0),
                adj=int(focused["focused_adjacent_pitch_repeats"]),
                max_interval=int(focused["focused_max_interval"]),
                flags=list(gate["flags"]),
            )
        )
    return "\n".join(lines).rstrip() + "\n"


def validate_repair(
    report: dict[str, Any],
    *,
    require_qualified: bool,
    require_phrase_vocabulary_improvement: bool,
    expected_source_run_id: str | None,
    expected_sample_index: int | None,
) -> dict[str, Any]:
    summary = report["repair_summary"]
    if require_qualified and not bool(summary["qualified"]):
        raise MarginRecoveredPhraseVocabularyRepairError("selected candidate is not qualified")
    if require_phrase_vocabulary_improvement and not bool(summary["phrase_vocabulary_improved"]):
        raise MarginRecoveredPhraseVocabularyRepairError(
            "selected candidate did not improve phrase/vocabulary blockers"
        )
    if expected_source_run_id is not None and summary["selected_source_run_id"] != expected_source_run_id:
        raise MarginRecoveredPhraseVocabularyRepairError(
            f"expected source run {expected_source_run_id}, got {summary['selected_source_run_id']}"
        )
    if expected_sample_index is not None and int(summary["selected_sample_index"]) != int(expected_sample_index):
        raise MarginRecoveredPhraseVocabularyRepairError(
            f"expected sample {expected_sample_index}, got {summary['selected_sample_index']}"
        )
    return {
        "report_count": int(report["report_count"]),
        "candidate_count": int(report["candidate_count"]),
        "qualified_candidate_count": int(report["qualified_candidate_count"]),
        "selected_candidate_id": str(summary["selected_candidate_id"]),
        "selected_source_run_id": str(summary["selected_source_run_id"]),
        "selected_sample_index": int(summary["selected_sample_index"]),
        "selected_sample_seed": int(summary["selected_sample_seed"]),
        "qualified": bool(summary["qualified"]),
        "phrase_vocabulary_improved": bool(summary["phrase_vocabulary_improved"]),
        "dead_air_delta_from_previous": float(summary["dead_air_delta_from_previous"]),
        "adjacent_pitch_repeat_delta_from_previous": int(summary["adjacent_pitch_repeat_delta_from_previous"]),
        "max_interval_delta_from_previous": int(summary["max_interval_delta_from_previous"]),
        "focused_unique_pitch_delta_from_previous": int(summary["focused_unique_pitch_delta_from_previous"]),
        "focused_note_delta_from_previous": int(summary["focused_note_delta_from_previous"]),
        "remaining_flags": list(summary["remaining_flags"]),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize margin-recovered phrase/vocabulary repair sweep")
    parser.add_argument("--report_path", action="append", required=True)
    parser.add_argument(
        "--output_root",
        type=str,
        default=str(ROOT_DIR / "outputs" / "stage_b_margin_recovered_phrase_vocabulary_repair"),
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument(
        "--previous_candidate_id",
        type=str,
        default="margin_recovered_timing_repetition_seed_37_topk_7_temp_086_n48_sample_39",
    )
    parser.add_argument("--previous_dead_air", type=float, default=0.35294117647058826)
    parser.add_argument("--previous_unique_pitch_count", type=int, default=7)
    parser.add_argument("--previous_note_count", type=int, default=14)
    parser.add_argument("--previous_adjacent_pitch_repeats", type=int, default=2)
    parser.add_argument("--previous_max_interval", type=int, default=16)
    parser.add_argument("--min_unique_pitch_count", type=int, default=6)
    parser.add_argument("--max_dead_air_ratio_exclusive", type=float, default=0.40)
    parser.add_argument("--min_note_count", type=int, default=12)
    parser.add_argument("--max_simultaneous_notes", type=int, default=1)
    parser.add_argument("--max_duplicated_3_note_chunks", type=int, default=0)
    parser.add_argument("--max_adjacent_pitch_repeats_exclusive", type=int, default=2)
    parser.add_argument("--max_interval_exclusive", type=int, default=12)
    parser.add_argument("--require_qualified", action="store_true")
    parser.add_argument("--require_phrase_vocabulary_improvement", action="store_true")
    parser.add_argument("--expected_source_run_id", type=str, default=None)
    parser.add_argument("--expected_sample_index", type=int, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    report_paths = [Path(path) for path in args.report_path]
    report = build_repair_report(
        report_paths,
        output_dir=output_dir,
        previous_candidate_id=str(args.previous_candidate_id),
        previous_dead_air=float(args.previous_dead_air),
        previous_unique_pitch_count=int(args.previous_unique_pitch_count),
        previous_note_count=int(args.previous_note_count),
        previous_adjacent_pitch_repeats=int(args.previous_adjacent_pitch_repeats),
        previous_max_interval=int(args.previous_max_interval),
        min_unique_pitch_count=int(args.min_unique_pitch_count),
        max_dead_air_ratio_exclusive=float(args.max_dead_air_ratio_exclusive),
        min_note_count=int(args.min_note_count),
        max_simultaneous_notes=int(args.max_simultaneous_notes),
        max_duplicated_3_note_chunks=int(args.max_duplicated_3_note_chunks),
        max_adjacent_pitch_repeats_exclusive=int(args.max_adjacent_pitch_repeats_exclusive),
        max_interval_exclusive=int(args.max_interval_exclusive),
    )
    summary = validate_repair(
        report,
        require_qualified=bool(args.require_qualified),
        require_phrase_vocabulary_improvement=bool(args.require_phrase_vocabulary_improvement),
        expected_source_run_id=args.expected_source_run_id,
        expected_sample_index=args.expected_sample_index,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "phrase_vocabulary_repair_summary.json", report)
    write_json(output_dir / "phrase_vocabulary_repair_result.json", summary)
    (output_dir / "phrase_vocabulary_repair_summary.md").write_text(markdown_report(report), encoding="utf-8")
    summary.update(
        {
            "summary_path": str(output_dir / "phrase_vocabulary_repair_summary.json"),
            "markdown_path": str(output_dir / "phrase_vocabulary_repair_summary.md"),
        }
    )
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
