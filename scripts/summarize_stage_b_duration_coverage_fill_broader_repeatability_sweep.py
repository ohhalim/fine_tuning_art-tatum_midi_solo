"""Run a broader duration/coverage fill repeatability sweep over existing candidates."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from scripts.summarize_stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_repair import (
    DEFAULT_FILL_MAX_ADDITIONS,
    build_duration_coverage_fill_report,
)


class StageBDurationCoverageBroaderRepeatabilitySweepError(ValueError):
    pass


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def current_anchor_summary(duration_fill_summary: dict[str, Any]) -> dict[str, Any]:
    repair = _dict(duration_fill_summary.get("repair_summary"))
    selected = _dict(duration_fill_summary.get("selected_candidate"))
    source = _dict(duration_fill_summary.get("source_candidate"))
    if not repair or not selected or not source:
        raise StageBDurationCoverageBroaderRepeatabilitySweepError("duration fill summary is missing anchor data")
    return {
        "source_candidate_id": str(source.get("candidate_id") or ""),
        "selected_candidate_id": str(repair.get("selected_candidate_id") or selected.get("candidate_id") or ""),
        "variant_count": int(duration_fill_summary.get("variant_count", 0) or 0),
        "qualified_variant_count": int(duration_fill_summary.get("qualified_variant_count", 0) or 0),
        "qualified": bool(repair.get("qualified", False)),
        "duration_coverage_fill_improved": bool(repair.get("duration_coverage_fill_improved", False)),
        "fill_addition_count": int(repair.get("selected_fill_addition_count", 0) or 0),
        "baseline_dead_air_ratio": float(repair.get("baseline_dead_air_ratio", 0.0) or 0.0),
        "selected_dead_air_ratio": float(repair.get("selected_dead_air_ratio", 0.0) or 0.0),
        "dead_air_delta_from_baseline": float(repair.get("dead_air_delta_from_baseline", 0.0) or 0.0),
        "selected_focused_note_count": int(repair.get("selected_focused_note_count", 0) or 0),
        "selected_focused_unique_pitch_count": int(repair.get("selected_focused_unique_pitch_count", 0) or 0),
        "selected_adjacent_pitch_repeats": int(repair.get("selected_adjacent_pitch_repeats", 0) or 0),
        "selected_max_interval": int(repair.get("selected_max_interval", 0) or 0),
        "claim_boundary": str(repair.get("claim_boundary") or ""),
    }


def qualified_distinct_candidates(distinct_sample_seed_sweep: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    candidates = [
        dict(candidate)
        for candidate in _list(distinct_sample_seed_sweep.get("top_candidates"))
        if isinstance(candidate, dict) and bool(candidate.get("qualified", False))
    ]
    candidates.sort(key=lambda row: (int(row.get("repair_rank", 999999) or 999999), str(row.get("candidate_id") or "")))
    return candidates[: int(limit)]


def previous_summary_from_candidate(
    candidate: dict[str, Any],
    *,
    generation_output_root: Path,
) -> dict[str, Any]:
    source_run_id = str(candidate.get("source_run_id") or "")
    sample_index = int(candidate.get("sample_index", -1))
    candidate_id = str(candidate.get("candidate_id") or "")
    if not source_run_id or sample_index < 0 or not candidate_id:
        raise StageBDurationCoverageBroaderRepeatabilitySweepError(f"invalid source candidate row: {candidate}")
    source_root = generation_output_root / source_run_id
    midi_path = source_root / "samples" / f"stage_b_sample_{sample_index}.mid"
    source_report_path = source_root / "report.json"
    if not midi_path.exists():
        raise StageBDurationCoverageBroaderRepeatabilitySweepError(f"source MIDI not found: {midi_path}")
    if not source_report_path.exists():
        raise StageBDurationCoverageBroaderRepeatabilitySweepError(f"source report not found: {source_report_path}")
    return {
        "output_dir": str(source_root),
        "selected_candidate": {
            "candidate_id": candidate_id,
            "midi_path": str(midi_path),
            "source_report_path": str(source_report_path),
            "metrics": {
                "note_count": int(candidate.get("note_count", 0) or 0),
                "unique_pitch_count": int(candidate.get("unique_pitch_count", 0) or 0),
                "dead_air_ratio": float(candidate.get("dead_air_ratio", 0.0) or 0.0),
            },
            "focused_solo_metrics": {
                "focused_note_count": int(candidate.get("focused_note_count", 0) or 0),
                "focused_unique_pitch_count": int(candidate.get("focused_unique_pitch_count", 0) or 0),
                "focused_adjacent_pitch_repeats": int(candidate.get("focused_adjacent_pitch_repeats", 0) or 0),
                "focused_duplicated_3_note_pitch_class_chunks": 0,
                "focused_max_interval": int(candidate.get("focused_max_interval", 0) or 0),
            },
        },
    }


def compact_source_result(source_candidate: dict[str, Any], fill_report: dict[str, Any]) -> dict[str, Any]:
    repair = _dict(fill_report.get("repair_summary"))
    return {
        "source_candidate_id": str(source_candidate.get("candidate_id") or ""),
        "source_run_id": str(source_candidate.get("source_run_id") or ""),
        "source_seed": int(source_candidate.get("source_seed", 0) or 0),
        "sample_index": int(source_candidate.get("sample_index", 0) or 0),
        "sample_seed": int(source_candidate.get("sample_seed", 0) or 0),
        "variant_count": int(fill_report.get("variant_count", 0) or 0),
        "qualified_variant_count": int(fill_report.get("qualified_variant_count", 0) or 0),
        "selected_candidate_id": str(repair.get("selected_candidate_id") or ""),
        "selected_fill_addition_count": int(repair.get("selected_fill_addition_count", 0) or 0),
        "qualified": bool(repair.get("qualified", False)),
        "duration_coverage_fill_improved": bool(repair.get("duration_coverage_fill_improved", False)),
        "remaining_flags": list(repair.get("remaining_flags") or []),
        "baseline_dead_air_ratio": float(repair.get("baseline_dead_air_ratio", 0.0) or 0.0),
        "selected_dead_air_ratio": float(repair.get("selected_dead_air_ratio", 0.0) or 0.0),
        "dead_air_delta_from_baseline": float(repair.get("dead_air_delta_from_baseline", 0.0) or 0.0),
        "selected_focused_note_count": int(repair.get("selected_focused_note_count", 0) or 0),
        "selected_focused_unique_pitch_count": int(repair.get("selected_focused_unique_pitch_count", 0) or 0),
        "selected_adjacent_pitch_repeats": int(repair.get("selected_adjacent_pitch_repeats", 0) or 0),
        "selected_max_interval": int(repair.get("selected_max_interval", 0) or 0),
        "claim_boundary": str(repair.get("claim_boundary") or ""),
    }


def build_broader_repeatability_report(
    *,
    next_decision: dict[str, Any],
    user_listening_consolidation: dict[str, Any],
    duration_fill_summary: dict[str, Any],
    distinct_sample_seed_sweep: dict[str, Any],
    output_dir: Path,
    generation_output_root: Path,
    max_source_candidates: int,
    min_source_candidates: int,
    min_qualified_source_candidates: int,
    fill_max_additions: Sequence[int],
    dead_air_threshold_sec: float,
    simultaneous_limit: int,
    min_unique_pitch_count: int,
    max_dead_air_ratio_exclusive: float,
    min_note_count: int,
    max_simultaneous_notes: int,
    max_duplicated_3_note_chunks: int,
    max_adjacent_pitch_repeats_exclusive: int,
    max_interval_exclusive: int,
) -> dict[str, Any]:
    decision = _dict(next_decision.get("decision"))
    if str(decision.get("next_boundary") or "") != "broader_repeatability_sweep":
        raise StageBDurationCoverageBroaderRepeatabilitySweepError("next decision must target broader repeatability")
    if not bool(decision.get("auto_progress_allowed", False)):
        raise StageBDurationCoverageBroaderRepeatabilitySweepError("next decision does not allow auto progress")
    if bool(decision.get("critical_user_input_required", True)):
        raise StageBDurationCoverageBroaderRepeatabilitySweepError("critical user input is required")

    consolidation_boundary = _dict(user_listening_consolidation.get("consolidated_claim_boundary"))
    if bool(consolidation_boundary.get("broad_model_quality_claimed", True)):
        raise StageBDurationCoverageBroaderRepeatabilitySweepError("broad model quality must not be claimed")

    anchor = current_anchor_summary(duration_fill_summary)
    if anchor["selected_candidate_id"] != str(user_listening_consolidation.get("candidate_id") or ""):
        raise StageBDurationCoverageBroaderRepeatabilitySweepError("current anchor candidate mismatch")

    source_candidates = qualified_distinct_candidates(distinct_sample_seed_sweep, limit=max_source_candidates)
    if len(source_candidates) < int(min_source_candidates):
        raise StageBDurationCoverageBroaderRepeatabilitySweepError(
            f"expected at least {int(min_source_candidates)} source candidates, got {len(source_candidates)}"
        )

    source_results: list[dict[str, Any]] = []
    for index, candidate in enumerate(source_candidates, start=1):
        source_output_dir = output_dir / "source_sweeps" / f"{index:02d}_{candidate['sample_seed']}"
        fill_report = build_duration_coverage_fill_report(
            previous_summary_from_candidate(candidate, generation_output_root=generation_output_root),
            output_dir=source_output_dir,
            fill_max_additions=fill_max_additions,
            dead_air_threshold_sec=float(dead_air_threshold_sec),
            simultaneous_limit=int(simultaneous_limit),
            min_unique_pitch_count=int(min_unique_pitch_count),
            max_dead_air_ratio_exclusive=float(max_dead_air_ratio_exclusive),
            min_note_count=int(min_note_count),
            max_simultaneous_notes=int(max_simultaneous_notes),
            max_duplicated_3_note_chunks=int(max_duplicated_3_note_chunks),
            max_adjacent_pitch_repeats_exclusive=int(max_adjacent_pitch_repeats_exclusive),
            max_interval_exclusive=int(max_interval_exclusive),
        )
        source_results.append(compact_source_result(candidate, fill_report))

    qualified_source_count = sum(1 for result in source_results if result["qualified"])
    dead_air_improved_source_count = sum(1 for result in source_results if result["duration_coverage_fill_improved"])
    total_variant_count = sum(result["variant_count"] for result in source_results)
    total_qualified_variant_count = sum(result["qualified_variant_count"] for result in source_results)
    distinct_sample_seed_count = len({result["sample_seed"] for result in source_results})

    if qualified_source_count < int(min_qualified_source_candidates):
        boundary = "insufficient_distinct_source_duration_fill_repeatability"
    elif dead_air_improved_source_count < qualified_source_count:
        boundary = "qualified_gate_repeatability_with_partial_dead_air_gain"
    else:
        boundary = "qualified_gate_repeatability_with_dead_air_gain"

    return {
        "schema_version": "stage_b_duration_coverage_fill_broader_repeatability_sweep_v1",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "source_schemas": {
            "next_decision": str(next_decision.get("schema_version") or ""),
            "user_listening_consolidation": str(user_listening_consolidation.get("schema_version") or ""),
            "duration_fill_summary": str(duration_fill_summary.get("schema_version") or ""),
            "distinct_sample_seed_sweep": str(distinct_sample_seed_sweep.get("schema_version") or ""),
        },
        "current_keep_anchor": anchor,
        "source_repeatability_results": source_results,
        "repeatability_summary": {
            "boundary": boundary,
            "source_candidate_count": int(len(source_results)),
            "distinct_sample_seed_count": int(distinct_sample_seed_count),
            "qualified_source_candidate_count": int(qualified_source_count),
            "dead_air_improved_source_candidate_count": int(dead_air_improved_source_count),
            "total_variant_count": int(total_variant_count),
            "total_qualified_variant_count": int(total_qualified_variant_count),
            "min_source_candidates": int(min_source_candidates),
            "min_qualified_source_candidates": int(min_qualified_source_candidates),
            "current_anchor_qualified": bool(anchor["qualified"]),
            "current_anchor_single_user_preference_claimed": bool(
                consolidation_boundary.get("single_user_human_audio_preference_claimed", False)
            ),
            "broad_model_quality_claimed": False,
        },
        "claim_boundary": {
            "boundary": boundary,
            "current_keep_anchor_supported_by_single_user_listening": True,
            "distinct_source_midi_gate_repeatability_claimed": qualified_source_count
            >= int(min_qualified_source_candidates),
            "uniform_dead_air_gain_claimed": dead_air_improved_source_count == qualified_source_count,
            "new_source_human_audio_preference_claimed": False,
            "multi_reviewer_preference_claimed": False,
            "broad_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
            "production_ready_improviser_claimed": False,
        },
        "proven": [
            "current_keep_anchor_midi_gate_and_single_user_preference",
            "distinct_source_duration_fill_midi_gate_repeatability"
            if qualified_source_count >= int(min_qualified_source_candidates)
            else "distinct_source_duration_fill_midi_gate_repeatability_not_met",
        ],
        "not_proven": [
            "uniform_dead_air_gain_across_distinct_sources",
            "new_source_human_audio_preference",
            "multi_reviewer_preference",
            "broad_trained_model_quality",
            "brad_style_adaptation",
            "production_ready_improviser",
        ],
        "next_recommended_issue": (
            "Stage B margin-recovered phrase/vocabulary duration coverage fill "
            "dead-air gain repeatability repair"
            if boundary == "qualified_gate_repeatability_with_partial_dead_air_gain"
            else "Stage B margin-recovered phrase/vocabulary duration coverage fill repeatability consolidation"
        ),
    }


def validate_broader_repeatability_report(
    report: dict[str, Any],
    *,
    expected_boundary: str | None,
    min_source_candidates: int,
    min_qualified_source_candidates: int,
    require_no_broad_quality_claim: bool,
) -> dict[str, Any]:
    summary = _dict(report.get("repeatability_summary"))
    claim = _dict(report.get("claim_boundary"))
    boundary = str(summary.get("boundary") or "")
    if expected_boundary and boundary != expected_boundary:
        raise StageBDurationCoverageBroaderRepeatabilitySweepError(
            f"expected boundary {expected_boundary}, got {boundary}"
        )
    if int(summary.get("source_candidate_count", 0) or 0) < int(min_source_candidates):
        raise StageBDurationCoverageBroaderRepeatabilitySweepError("not enough source candidates")
    if int(summary.get("qualified_source_candidate_count", 0) or 0) < int(min_qualified_source_candidates):
        raise StageBDurationCoverageBroaderRepeatabilitySweepError("not enough qualified source candidates")
    if require_no_broad_quality_claim:
        blocked = [
            "new_source_human_audio_preference_claimed",
            "multi_reviewer_preference_claimed",
            "broad_model_quality_claimed",
            "brad_style_adaptation_claimed",
            "production_ready_improviser_claimed",
        ]
        claimed = [name for name in blocked if bool(claim.get(name, True))]
        if claimed:
            raise StageBDurationCoverageBroaderRepeatabilitySweepError(f"unexpected broad claim: {claimed}")
    return {
        "boundary": boundary,
        "source_candidate_count": int(summary.get("source_candidate_count", 0) or 0),
        "distinct_sample_seed_count": int(summary.get("distinct_sample_seed_count", 0) or 0),
        "qualified_source_candidate_count": int(summary.get("qualified_source_candidate_count", 0) or 0),
        "dead_air_improved_source_candidate_count": int(
            summary.get("dead_air_improved_source_candidate_count", 0) or 0
        ),
        "total_variant_count": int(summary.get("total_variant_count", 0) or 0),
        "total_qualified_variant_count": int(summary.get("total_qualified_variant_count", 0) or 0),
        "broad_model_quality_claimed": bool(summary.get("broad_model_quality_claimed", True)),
        "next_recommended_issue": str(report.get("next_recommended_issue") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    summary = report["repeatability_summary"]
    anchor = report["current_keep_anchor"]
    lines = [
        "# Stage B Duration Coverage Fill Broader Repeatability Sweep",
        "",
        f"- boundary: `{summary['boundary']}`",
        f"- current keep anchor: `{anchor['selected_candidate_id']}`",
        f"- source candidates: `{summary['source_candidate_count']}`",
        f"- qualified source candidates: `{summary['qualified_source_candidate_count']}`",
        f"- dead-air improved source candidates: `{summary['dead_air_improved_source_candidate_count']}`",
        f"- total variants: `{summary['total_variant_count']}`",
        f"- qualified variants: `{summary['total_qualified_variant_count']}`",
        f"- broad model quality claimed: `{summary['broad_model_quality_claimed']}`",
        "",
        "## Current Anchor",
        "",
        f"- selected: `{anchor['selected_candidate_id']}`",
        f"- variants: `{anchor['qualified_variant_count']}/{anchor['variant_count']}`",
        f"- dead-air: `{anchor['baseline_dead_air_ratio']:.4f}` -> `{anchor['selected_dead_air_ratio']:.4f}`",
        f"- notes / unique: `{anchor['selected_focused_note_count']}` / `{anchor['selected_focused_unique_pitch_count']}`",
        "",
        "## Source Sweep",
        "",
        "| source | sample seed | selected | variants | qualified | dead-air | unique | adj repeat | max interval | improved |",
        "|---|---:|---|---:|:---:|---:|---:|---:|---:|:---:|",
    ]
    for result in report["source_repeatability_results"]:
        lines.append(
            "| `{source}` | {sample_seed} | `{selected}` | {qualified_variants}/{variants} | {qualified} | "
            "{baseline:.4f} -> {selected_dead:.4f} | {unique} | {adj} | {interval} | {improved} |".format(
                source=result["source_candidate_id"],
                sample_seed=int(result["sample_seed"]),
                selected=result["selected_candidate_id"],
                qualified_variants=int(result["qualified_variant_count"]),
                variants=int(result["variant_count"]),
                qualified=bool(result["qualified"]),
                baseline=float(result["baseline_dead_air_ratio"]),
                selected_dead=float(result["selected_dead_air_ratio"]),
                unique=int(result["selected_focused_unique_pitch_count"]),
                adj=int(result["selected_adjacent_pitch_repeats"]),
                interval=int(result["selected_max_interval"]),
                improved=bool(result["duration_coverage_fill_improved"]),
            )
        )
    lines.extend(["", "## Not Proven", ""])
    for item in report.get("not_proven", []):
        lines.append(f"- `{item}`")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize broader duration coverage fill repeatability")
    parser.add_argument(
        "--next_decision",
        type=str,
        default="outputs/stage_b_duration_coverage_fill_next_decision/"
        "harness_stage_b_duration_coverage_fill_next_decision/"
        "stage_b_duration_coverage_fill_next_decision.json",
    )
    parser.add_argument(
        "--user_listening_consolidation",
        type=str,
        default="outputs/stage_b_duration_coverage_fill_user_listening_review_consolidation/"
        "harness_stage_b_duration_coverage_fill_user_listening_review_consolidation/"
        "stage_b_duration_coverage_fill_user_listening_review_consolidation.json",
    )
    parser.add_argument(
        "--duration_fill_summary",
        type=str,
        default="outputs/stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_repair/"
        "harness_stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_repair/"
        "duration_coverage_fill_repair_summary.json",
    )
    parser.add_argument(
        "--distinct_sample_seed_sweep",
        type=str,
        default="outputs/stage_b_margin_recovered_phrase_vocabulary_distinct_sample_seed_sweep/"
        "harness_stage_b_margin_recovered_phrase_vocabulary_distinct_sample_seed_sweep/"
        "distinct_sample_seed_sweep_summary.json",
    )
    parser.add_argument("--generation_output_root", type=str, default="outputs/stage_b_generation_probe")
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/stage_b_duration_coverage_fill_broader_repeatability_sweep",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--max_source_candidates", type=int, default=2)
    parser.add_argument("--min_source_candidates", type=int, default=2)
    parser.add_argument("--min_qualified_source_candidates", type=int, default=2)
    parser.add_argument("--fill_max_additions", action="append", type=int, default=None)
    parser.add_argument("--dead_air_threshold_sec", type=float, default=0.18)
    parser.add_argument("--simultaneous_limit", type=int, default=1)
    parser.add_argument("--min_unique_pitch_count", type=int, default=7)
    parser.add_argument("--max_dead_air_ratio_exclusive", type=float, default=0.376)
    parser.add_argument("--min_note_count", type=int, default=12)
    parser.add_argument("--max_simultaneous_notes", type=int, default=1)
    parser.add_argument("--max_duplicated_3_note_chunks", type=int, default=0)
    parser.add_argument("--max_adjacent_pitch_repeats_exclusive", type=int, default=1)
    parser.add_argument("--max_interval_exclusive", type=int, default=12)
    parser.add_argument("--expected_boundary", type=str, default="")
    parser.add_argument("--require_no_broad_quality_claim", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    report = build_broader_repeatability_report(
        next_decision=read_json(Path(args.next_decision)),
        user_listening_consolidation=read_json(Path(args.user_listening_consolidation)),
        duration_fill_summary=read_json(Path(args.duration_fill_summary)),
        distinct_sample_seed_sweep=read_json(Path(args.distinct_sample_seed_sweep)),
        output_dir=output_dir,
        generation_output_root=Path(args.generation_output_root),
        max_source_candidates=int(args.max_source_candidates),
        min_source_candidates=int(args.min_source_candidates),
        min_qualified_source_candidates=int(args.min_qualified_source_candidates),
        fill_max_additions=args.fill_max_additions or list(DEFAULT_FILL_MAX_ADDITIONS),
        dead_air_threshold_sec=float(args.dead_air_threshold_sec),
        simultaneous_limit=int(args.simultaneous_limit),
        min_unique_pitch_count=int(args.min_unique_pitch_count),
        max_dead_air_ratio_exclusive=float(args.max_dead_air_ratio_exclusive),
        min_note_count=int(args.min_note_count),
        max_simultaneous_notes=int(args.max_simultaneous_notes),
        max_duplicated_3_note_chunks=int(args.max_duplicated_3_note_chunks),
        max_adjacent_pitch_repeats_exclusive=int(args.max_adjacent_pitch_repeats_exclusive),
        max_interval_exclusive=int(args.max_interval_exclusive),
    )
    summary = validate_broader_repeatability_report(
        report,
        expected_boundary=str(args.expected_boundary or ""),
        min_source_candidates=int(args.min_source_candidates),
        min_qualified_source_candidates=int(args.min_qualified_source_candidates),
        require_no_broad_quality_claim=bool(args.require_no_broad_quality_claim),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "stage_b_duration_coverage_fill_broader_repeatability_sweep.json"
    markdown_path = output_dir / "stage_b_duration_coverage_fill_broader_repeatability_sweep.md"
    write_json(report_path, report)
    write_json(output_dir / "stage_b_duration_coverage_fill_broader_repeatability_sweep_validation_summary.json", summary)
    markdown_path.write_text(markdown_report(report), encoding="utf-8")
    print(json.dumps({**summary, "report_path": str(report_path), "markdown_path": str(markdown_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
