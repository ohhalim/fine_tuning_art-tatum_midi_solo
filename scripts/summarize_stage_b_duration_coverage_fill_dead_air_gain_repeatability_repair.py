"""Repair duration/coverage repeatability selection by requiring dead-air gain."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from scripts.summarize_stage_b_duration_coverage_fill_broader_repeatability_sweep import (
    previous_summary_from_candidate,
    qualified_distinct_candidates,
)
from scripts.summarize_stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_repair import (
    DEFAULT_FILL_MAX_ADDITIONS,
    build_duration_coverage_fill_report,
)


class StageBDurationCoverageDeadAirGainRepeatabilityRepairError(ValueError):
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


def variant_dead_air(variant: dict[str, Any]) -> float:
    return float(_dict(variant.get("metrics")).get("dead_air_ratio", 1.0) or 1.0)


def variant_unique_pitch_count(variant: dict[str, Any]) -> int:
    return int(_dict(variant.get("focused_solo_metrics")).get("focused_unique_pitch_count", 0) or 0)


def variant_fill_addition_count(variant: dict[str, Any]) -> int:
    return int(_dict(variant.get("fill_repair")).get("fill_addition_count", 0) or 0)


def variant_qualified(variant: dict[str, Any]) -> bool:
    return bool(_dict(variant.get("duration_coverage_gate")).get("qualified", False))


def select_dead_air_gain_variant(fill_report: dict[str, Any]) -> dict[str, Any]:
    repair = _dict(fill_report.get("repair_summary"))
    baseline_dead_air = float(repair.get("baseline_dead_air_ratio", 1.0) or 1.0)
    variants = [dict(variant) for variant in _list(fill_report.get("variants")) if isinstance(variant, dict)]
    improved = [
        variant
        for variant in variants
        if variant_qualified(variant) and variant_dead_air(variant) < baseline_dead_air
    ]
    if not improved:
        fallback = _dict(fill_report.get("selected_candidate"))
        if not fallback:
            raise StageBDurationCoverageDeadAirGainRepeatabilityRepairError("no selectable variant found")
        return fallback
    improved.sort(
        key=lambda variant: (
            variant_fill_addition_count(variant),
            variant_dead_air(variant),
            -variant_unique_pitch_count(variant),
            str(variant.get("candidate_id") or ""),
        )
    )
    return improved[0]


def compact_dead_air_gain_result(source_candidate: dict[str, Any], fill_report: dict[str, Any]) -> dict[str, Any]:
    repair = _dict(fill_report.get("repair_summary"))
    selected = select_dead_air_gain_variant(fill_report)
    selected_metrics = _dict(selected.get("metrics"))
    selected_focused = _dict(selected.get("focused_solo_metrics"))
    baseline_dead_air = float(repair.get("baseline_dead_air_ratio", 1.0) or 1.0)
    selected_dead_air = variant_dead_air(selected)
    qualified_variants = [variant for variant in _list(fill_report.get("variants")) if variant_qualified(_dict(variant))]
    dead_air_gain_variants = [
        variant for variant in qualified_variants if variant_dead_air(_dict(variant)) < baseline_dead_air
    ]
    return {
        "source_candidate_id": str(source_candidate.get("candidate_id") or ""),
        "source_run_id": str(source_candidate.get("source_run_id") or ""),
        "source_seed": int(source_candidate.get("source_seed", 0) or 0),
        "sample_index": int(source_candidate.get("sample_index", 0) or 0),
        "sample_seed": int(source_candidate.get("sample_seed", 0) or 0),
        "variant_count": int(fill_report.get("variant_count", 0) or 0),
        "qualified_variant_count": int(fill_report.get("qualified_variant_count", 0) or 0),
        "dead_air_gain_variant_count": int(len(dead_air_gain_variants)),
        "selected_candidate_id": str(selected.get("candidate_id") or ""),
        "selected_midi_path": str(selected.get("midi_path") or ""),
        "selected_fill_addition_count": variant_fill_addition_count(selected),
        "qualified": variant_qualified(selected),
        "dead_air_gain_repaired": bool(selected_dead_air < baseline_dead_air and variant_qualified(selected)),
        "remaining_flags": list(_dict(selected.get("duration_coverage_gate")).get("flags") or []),
        "baseline_dead_air_ratio": float(baseline_dead_air),
        "selected_dead_air_ratio": float(selected_dead_air),
        "dead_air_delta_from_baseline": round(float(baseline_dead_air - selected_dead_air), 6),
        "selected_focused_note_count": int(selected_focused.get("focused_note_count", 0) or 0),
        "selected_focused_unique_pitch_count": int(
            selected_focused.get("focused_unique_pitch_count", 0) or selected_metrics.get("unique_pitch_count", 0) or 0
        ),
        "selected_adjacent_pitch_repeats": int(selected_focused.get("focused_adjacent_pitch_repeats", 0) or 0),
        "selected_duplicated_3_note_pitch_class_chunks": int(
            selected_focused.get("focused_duplicated_3_note_pitch_class_chunks", 0) or 0
        ),
        "selected_max_interval": int(selected_focused.get("focused_max_interval", 0) or 0),
        "claim_boundary": str(repair.get("claim_boundary") or ""),
    }


def build_dead_air_gain_repeatability_repair_report(
    *,
    broader_repeatability_sweep: dict[str, Any],
    distinct_sample_seed_sweep: dict[str, Any],
    output_dir: Path,
    generation_output_root: Path,
    max_source_candidates: int,
    min_source_candidates: int,
    min_dead_air_gain_source_candidates: int,
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
    previous_summary = _dict(broader_repeatability_sweep.get("repeatability_summary"))
    previous_boundary = str(previous_summary.get("boundary") or "")
    if previous_boundary != "qualified_gate_repeatability_with_partial_dead_air_gain":
        raise StageBDurationCoverageDeadAirGainRepeatabilityRepairError(
            f"expected partial dead-air boundary, got {previous_boundary}"
        )
    if bool(previous_summary.get("broad_model_quality_claimed", True)):
        raise StageBDurationCoverageDeadAirGainRepeatabilityRepairError("broad model quality must not be claimed")

    source_candidates = qualified_distinct_candidates(distinct_sample_seed_sweep, limit=max_source_candidates)
    if len(source_candidates) < int(min_source_candidates):
        raise StageBDurationCoverageDeadAirGainRepeatabilityRepairError(
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
        write_json(source_output_dir / "duration_coverage_fill_repair_summary.json", fill_report)
        source_results.append(compact_dead_air_gain_result(candidate, fill_report))

    dead_air_gain_source_count = sum(1 for result in source_results if result["dead_air_gain_repaired"])
    qualified_source_count = sum(1 for result in source_results if result["qualified"])
    total_variant_count = sum(result["variant_count"] for result in source_results)
    total_qualified_variant_count = sum(result["qualified_variant_count"] for result in source_results)
    total_dead_air_gain_variant_count = sum(result["dead_air_gain_variant_count"] for result in source_results)
    selected_fill_additions = sorted({result["selected_fill_addition_count"] for result in source_results})
    if dead_air_gain_source_count >= int(min_dead_air_gain_source_candidates):
        boundary = "qualified_gate_repeatability_with_dead_air_gain"
    else:
        boundary = "dead_air_gain_repeatability_not_repaired"

    return {
        "schema_version": "stage_b_duration_coverage_fill_dead_air_gain_repeatability_repair_v1",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "source_schemas": {
            "broader_repeatability_sweep": str(broader_repeatability_sweep.get("schema_version") or ""),
            "distinct_sample_seed_sweep": str(distinct_sample_seed_sweep.get("schema_version") or ""),
        },
        "previous_boundary": previous_boundary,
        "repair_selection_rule": "qualified_dead_air_gain_then_min_fill_additions",
        "source_repeatability_results": source_results,
        "repair_summary": {
            "boundary": boundary,
            "source_candidate_count": int(len(source_results)),
            "qualified_source_candidate_count": int(qualified_source_count),
            "dead_air_gain_source_candidate_count": int(dead_air_gain_source_count),
            "total_variant_count": int(total_variant_count),
            "total_qualified_variant_count": int(total_qualified_variant_count),
            "total_dead_air_gain_variant_count": int(total_dead_air_gain_variant_count),
            "selected_fill_additions": selected_fill_additions,
            "broad_model_quality_claimed": False,
        },
        "claim_boundary": {
            "boundary": boundary,
            "distinct_source_midi_gate_repeatability_claimed": qualified_source_count
            >= int(min_dead_air_gain_source_candidates),
            "selected_distinct_source_dead_air_gain_claimed": dead_air_gain_source_count
            >= int(min_dead_air_gain_source_candidates),
            "new_source_human_audio_preference_claimed": False,
            "multi_reviewer_preference_claimed": False,
            "broad_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
            "production_ready_improviser_claimed": False,
        },
        "proven": [
            "qualified_midi_gate_on_selected_distinct_sources",
            "dead_air_gain_on_selected_distinct_sources",
        ],
        "not_proven": [
            "new_source_human_audio_preference",
            "multi_reviewer_preference",
            "broad_trained_model_quality",
            "brad_style_adaptation",
            "production_ready_improviser",
        ],
        "next_recommended_issue": (
            "Stage B margin-recovered phrase/vocabulary duration coverage fill repeatability consolidation"
        ),
    }


def validate_dead_air_gain_repeatability_repair(
    report: dict[str, Any],
    *,
    expected_boundary: str | None,
    min_source_candidates: int,
    min_dead_air_gain_source_candidates: int,
    require_no_broad_quality_claim: bool,
) -> dict[str, Any]:
    summary = _dict(report.get("repair_summary"))
    claim = _dict(report.get("claim_boundary"))
    boundary = str(summary.get("boundary") or "")
    if expected_boundary and boundary != expected_boundary:
        raise StageBDurationCoverageDeadAirGainRepeatabilityRepairError(
            f"expected boundary {expected_boundary}, got {boundary}"
        )
    if int(summary.get("source_candidate_count", 0) or 0) < int(min_source_candidates):
        raise StageBDurationCoverageDeadAirGainRepeatabilityRepairError("not enough source candidates")
    if int(summary.get("dead_air_gain_source_candidate_count", 0) or 0) < int(
        min_dead_air_gain_source_candidates
    ):
        raise StageBDurationCoverageDeadAirGainRepeatabilityRepairError("not enough dead-air gain source candidates")
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
            raise StageBDurationCoverageDeadAirGainRepeatabilityRepairError(f"unexpected broad claim: {claimed}")
    return {
        "boundary": boundary,
        "source_candidate_count": int(summary.get("source_candidate_count", 0) or 0),
        "qualified_source_candidate_count": int(summary.get("qualified_source_candidate_count", 0) or 0),
        "dead_air_gain_source_candidate_count": int(summary.get("dead_air_gain_source_candidate_count", 0) or 0),
        "total_variant_count": int(summary.get("total_variant_count", 0) or 0),
        "total_qualified_variant_count": int(summary.get("total_qualified_variant_count", 0) or 0),
        "total_dead_air_gain_variant_count": int(summary.get("total_dead_air_gain_variant_count", 0) or 0),
        "selected_fill_additions": list(summary.get("selected_fill_additions") or []),
        "broad_model_quality_claimed": bool(summary.get("broad_model_quality_claimed", True)),
        "next_recommended_issue": str(report.get("next_recommended_issue") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    summary = report["repair_summary"]
    lines = [
        "# Stage B Duration Coverage Fill Dead-Air Gain Repeatability Repair",
        "",
        f"- previous boundary: `{report['previous_boundary']}`",
        f"- boundary: `{summary['boundary']}`",
        f"- selection rule: `{report['repair_selection_rule']}`",
        f"- source candidates: `{summary['source_candidate_count']}`",
        f"- qualified source candidates: `{summary['qualified_source_candidate_count']}`",
        f"- dead-air gain source candidates: `{summary['dead_air_gain_source_candidate_count']}`",
        f"- total variants: `{summary['total_variant_count']}`",
        f"- qualified variants: `{summary['total_qualified_variant_count']}`",
        f"- dead-air gain variants: `{summary['total_dead_air_gain_variant_count']}`",
        f"- selected fill additions: `{summary['selected_fill_additions']}`",
        f"- broad model quality claimed: `{summary['broad_model_quality_claimed']}`",
        "",
        "| source | sample seed | selected | variants | dead-air gain variants | dead-air | unique | adj repeat | max interval |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for result in report["source_repeatability_results"]:
        lines.append(
            "| `{source}` | {sample_seed} | `{selected}` | {qualified}/{variants} | {gain_variants} | "
            "{baseline:.4f} -> {selected_dead:.4f} | {unique} | {adj} | {interval} |".format(
                source=result["source_candidate_id"],
                sample_seed=int(result["sample_seed"]),
                selected=result["selected_candidate_id"],
                qualified=int(result["qualified_variant_count"]),
                variants=int(result["variant_count"]),
                gain_variants=int(result["dead_air_gain_variant_count"]),
                baseline=float(result["baseline_dead_air_ratio"]),
                selected_dead=float(result["selected_dead_air_ratio"]),
                unique=int(result["selected_focused_unique_pitch_count"]),
                adj=int(result["selected_adjacent_pitch_repeats"]),
                interval=int(result["selected_max_interval"]),
            )
        )
    lines.extend(["", "## Not Proven", ""])
    for item in report.get("not_proven", []):
        lines.append(f"- `{item}`")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Repair duration coverage fill dead-air gain repeatability")
    parser.add_argument(
        "--broader_repeatability_sweep",
        type=str,
        default="outputs/stage_b_duration_coverage_fill_broader_repeatability_sweep/"
        "harness_stage_b_duration_coverage_fill_broader_repeatability_sweep/"
        "stage_b_duration_coverage_fill_broader_repeatability_sweep.json",
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
        default="outputs/stage_b_duration_coverage_fill_dead_air_gain_repeatability_repair",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--max_source_candidates", type=int, default=2)
    parser.add_argument("--min_source_candidates", type=int, default=2)
    parser.add_argument("--min_dead_air_gain_source_candidates", type=int, default=2)
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
    report = build_dead_air_gain_repeatability_repair_report(
        broader_repeatability_sweep=read_json(Path(args.broader_repeatability_sweep)),
        distinct_sample_seed_sweep=read_json(Path(args.distinct_sample_seed_sweep)),
        output_dir=output_dir,
        generation_output_root=Path(args.generation_output_root),
        max_source_candidates=int(args.max_source_candidates),
        min_source_candidates=int(args.min_source_candidates),
        min_dead_air_gain_source_candidates=int(args.min_dead_air_gain_source_candidates),
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
    summary = validate_dead_air_gain_repeatability_repair(
        report,
        expected_boundary=str(args.expected_boundary or ""),
        min_source_candidates=int(args.min_source_candidates),
        min_dead_air_gain_source_candidates=int(args.min_dead_air_gain_source_candidates),
        require_no_broad_quality_claim=bool(args.require_no_broad_quality_claim),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "stage_b_duration_coverage_fill_dead_air_gain_repeatability_repair.json"
    markdown_path = output_dir / "stage_b_duration_coverage_fill_dead_air_gain_repeatability_repair.md"
    write_json(report_path, report)
    write_json(output_dir / "stage_b_duration_coverage_fill_dead_air_gain_repeatability_repair_validation_summary.json", summary)
    markdown_path.write_text(markdown_report(report), encoding="utf-8")
    print(json.dumps({**summary, "report_path": str(report_path), "markdown_path": str(markdown_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
