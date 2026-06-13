"""Sweep repaired best-of candidate pool safety gates without rendering audio."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from statistics import mean
from typing import Any

import pretty_midi

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from scripts.build_stage_b_midi_to_solo_bebop_language_best_of_package import (  # noqa: E402
    DEFAULT_OUTPUT_ROOT,
    apply_candidate_repairs,
    candidate_rows,
    filter_candidate_rows,
    package_paths,
    parse_globs,
    selection_sort_key,
)
from scripts.build_stage_b_midi_to_solo_bebop_language_package import (  # noqa: E402
    BebopLanguagePackageError,
    write_json,
    write_text,
)


def parse_float_grid(raw: str) -> list[float]:
    values = [float(item.strip()) for item in str(raw or "").split(",") if item.strip()]
    if not values:
        raise BebopLanguagePackageError("empty float grid")
    return values


def parse_int_grid(raw: str) -> list[int]:
    values = [int(item.strip()) for item in str(raw or "").split(",") if item.strip()]
    if not values:
        raise BebopLanguagePackageError("empty int grid")
    return values


def case_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(str(row["case_label"]) for row in rows)
    return dict(sorted(counts.items()))


def selectable_count(counts: dict[str, int], *, max_per_case: int) -> int:
    return sum(min(int(max_per_case), int(count)) for count in counts.values())


def metric_average(rows: list[dict[str, Any]], key: str) -> float:
    return mean(float(row["objective_metrics"][key]) for row in rows) if rows else 0.0


def summarize_rows(rows: list[dict[str, Any]], *, max_per_case: int) -> dict[str, Any]:
    counts = case_counts(rows)
    return {
        "candidate_count": len(rows),
        "case_counts": counts,
        "max_per_case": int(max_per_case),
        "selectable_count": selectable_count(counts, max_per_case=max_per_case),
        "avg_step_motion_ratio": metric_average(rows, "step_motion_ratio"),
        "avg_chromatic_step_ratio": metric_average(rows, "chromatic_step_ratio"),
        "avg_large_leap_ratio": metric_average(rows, "large_leap_ratio"),
        "avg_adjacent_repeat_ratio": metric_average(rows, "adjacent_repeat_ratio"),
        "avg_max_bar_pitch_class_jaccard": metric_average(rows, "max_bar_pitch_class_jaccard"),
        "avg_enclosure_proxy_ratio": metric_average(rows, "enclosure_proxy_ratio"),
    }


def filter_motion_rows(
    rows: list[dict[str, Any]],
    *,
    min_step_motion_ratio: float,
    min_chromatic_step_ratio: float,
    max_large_leap_ratio: float,
) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for row in rows:
        metrics = row["objective_metrics"]
        if float(metrics["step_motion_ratio"]) < float(min_step_motion_ratio):
            continue
        if float(metrics["chromatic_step_ratio"]) < float(min_chromatic_step_ratio):
            continue
        if float(metrics["large_leap_ratio"]) > float(max_large_leap_ratio):
            continue
        filtered.append(row)
    return filtered


def filter_guard_rows(
    rows: list[dict[str, Any]],
    *,
    max_gate_penalty: float,
    max_offbeat_non_chord_ratio: float,
    max_unresolved_offbeat_non_chord_ratio: float,
    max_dominant_altered_offbeat_ratio: float,
    max_adjacent_repeat_ratio: float,
    max_bar_pitch_class_jaccard: float,
) -> list[dict[str, Any]]:
    return filter_candidate_rows(
        rows,
        max_gate_penalty=max_gate_penalty,
        max_offbeat_non_chord_ratio=max_offbeat_non_chord_ratio,
        max_unresolved_offbeat_non_chord_ratio=max_unresolved_offbeat_non_chord_ratio,
        max_dominant_altered_offbeat_ratio=max_dominant_altered_offbeat_ratio,
        max_adjacent_repeat_ratio=max_adjacent_repeat_ratio,
        max_bar_pitch_class_jaccard=max_bar_pitch_class_jaccard,
    )


def build_guard_motion_configs(
    rows: list[dict[str, Any]],
    *,
    offbeat_values: list[float],
    bar_similarity_values: list[float],
    step_values: list[float],
    chromatic_values: list[float],
    large_leap_values: list[float],
    max_per_case_values: list[int],
    selected_count: int,
    max_gate_penalty: float,
    max_unresolved_offbeat_non_chord_ratio: float,
    max_dominant_altered_offbeat_ratio: float,
    max_adjacent_repeat_ratio: float,
) -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []
    for max_offbeat, max_bar_similarity in product(offbeat_values, bar_similarity_values):
        guard_rows = filter_guard_rows(
            rows,
            max_gate_penalty=max_gate_penalty,
            max_offbeat_non_chord_ratio=max_offbeat,
            max_unresolved_offbeat_non_chord_ratio=max_unresolved_offbeat_non_chord_ratio,
            max_dominant_altered_offbeat_ratio=max_dominant_altered_offbeat_ratio,
            max_adjacent_repeat_ratio=max_adjacent_repeat_ratio,
            max_bar_pitch_class_jaccard=max_bar_similarity,
        )
        for min_step, min_chromatic, max_large_leap in product(step_values, chromatic_values, large_leap_values):
            motion_rows = filter_motion_rows(
                guard_rows,
                min_step_motion_ratio=min_step,
                min_chromatic_step_ratio=min_chromatic,
                max_large_leap_ratio=max_large_leap,
            )
            counts = case_counts(motion_rows)
            config = {
                "max_offbeat_non_chord_ratio": float(max_offbeat),
                "max_bar_pitch_class_jaccard": float(max_bar_similarity),
                "min_step_motion_ratio": float(min_step),
                "min_chromatic_step_ratio": float(min_chromatic),
                "max_large_leap_ratio": float(max_large_leap),
                "candidate_count": len(motion_rows),
                "case_counts": counts,
                "guard_candidate_count": len(guard_rows),
                "feasible_for_selected_count": {},
            }
            for max_per_case in max_per_case_values:
                selectable = selectable_count(counts, max_per_case=max_per_case)
                config[f"selectable_max_per_case_{max_per_case}"] = selectable
                config["feasible_for_selected_count"][str(max_per_case)] = selectable >= int(selected_count)
            configs.append(config)
    configs.sort(
        key=lambda item: (
            -max(int(item.get(f"selectable_max_per_case_{value}", 0)) for value in max_per_case_values),
            -int(item["candidate_count"]),
            float(item["max_offbeat_non_chord_ratio"]),
            float(item["max_bar_pitch_class_jaccard"]),
            float(item["min_step_motion_ratio"]),
            float(item["min_chromatic_step_ratio"]),
            float(item["max_large_leap_ratio"]),
        )
    )
    return configs


def summarize_feasible_configs(
    configs: list[dict[str, Any]],
    *,
    baseline_max_offbeat_non_chord_ratio: float,
    baseline_max_bar_pitch_class_jaccard: float,
    max_per_case_values: list[int],
) -> dict[str, Any]:
    feasible = [
        item
        for item in configs
        if any(bool(value) for value in dict(item["feasible_for_selected_count"]).values())
    ]
    if not feasible:
        return {
            "feasible_config_count": 0,
            "min_feasible_max_offbeat_non_chord_ratio": None,
            "min_feasible_max_bar_pitch_class_jaccard": None,
            "stricter_offbeat_feasible": False,
            "stricter_bar_similarity_feasible": False,
            "feasible_by_max_per_case": {
                str(value): 0
                for value in max_per_case_values
            },
        }
    return {
        "feasible_config_count": len(feasible),
        "min_feasible_max_offbeat_non_chord_ratio": min(
            float(item["max_offbeat_non_chord_ratio"]) for item in feasible
        ),
        "min_feasible_max_bar_pitch_class_jaccard": min(
            float(item["max_bar_pitch_class_jaccard"]) for item in feasible
        ),
        "stricter_offbeat_feasible": any(
            float(item["max_offbeat_non_chord_ratio"]) < float(baseline_max_offbeat_non_chord_ratio)
            for item in feasible
        ),
        "stricter_bar_similarity_feasible": any(
            float(item["max_bar_pitch_class_jaccard"]) < float(baseline_max_bar_pitch_class_jaccard)
            for item in feasible
        ),
        "feasible_by_max_per_case": {
            str(value): sum(
                1
                for item in feasible
                if bool(dict(item["feasible_for_selected_count"]).get(str(value), False))
            )
            for value in max_per_case_values
        },
    }


def build_repaired_pool(args: argparse.Namespace) -> list[dict[str, Any]]:
    paths = package_paths(Path(args.source_root), parse_globs(str(args.package_globs)))
    rows = candidate_rows(
        paths=paths,
        target_chord_tone_ratio=float(args.target_chord_tone_ratio),
        target_offbeat_non_chord_ratio=float(args.target_offbeat_non_chord_ratio),
    )
    selection_rows = filter_candidate_rows(
        rows,
        max_gate_penalty=float(args.max_gate_penalty),
        max_offbeat_non_chord_ratio=float(args.max_offbeat_non_chord_ratio),
        max_unresolved_offbeat_non_chord_ratio=float(args.max_unresolved_offbeat_non_chord_ratio),
        max_dominant_altered_offbeat_ratio=float(args.max_dominant_altered_offbeat_ratio),
        max_adjacent_repeat_ratio=None,
        max_bar_pitch_class_jaccard=None,
    )
    repaired_rows: list[dict[str, Any]] = []
    for item in selection_rows:
        pm = pretty_midi.PrettyMIDI(str(Path(str(item["source_midi_path"]))))
        _pm, metrics, score, gate_penalty, *_ = apply_candidate_repairs(
            pm,
            item,
            bars=int(args.bars),
            bpm=float(args.bpm),
            target_chord_tone_ratio=float(args.target_chord_tone_ratio),
            target_offbeat_non_chord_ratio=float(args.target_offbeat_non_chord_ratio),
            repair_bar_similarity_enabled=True,
            repair_bar_similarity_iterations=int(args.repair_bar_similarity_iterations),
            repair_enclosure_density_enabled=True,
            repair_enclosure_density_iterations=int(args.repair_enclosure_density_iterations),
            max_enclosure_repair_offbeat_non_chord_ratio=float(args.max_enclosure_repair_offbeat_non_chord_ratio),
            repair_unresolved_offbeat_enabled=True,
            repair_unresolved_offbeat_iterations=int(args.repair_unresolved_offbeat_iterations),
            repair_adjacent_repeats_enabled=True,
            repair_adjacent_repeats_iterations=int(args.repair_adjacent_repeats_iterations),
            repair_large_leaps_enabled=True,
            repair_large_leaps_iterations=int(args.repair_large_leaps_iterations),
            min_large_leap_repair_enclosure_proxy_ratio=float(args.min_large_leap_repair_enclosure_proxy_ratio),
            repair_motion_balance_enabled=bool(args.repair_motion_balance),
            repair_motion_balance_iterations=int(args.repair_motion_balance_iterations),
            target_min_step_motion_ratio=float(args.target_min_step_motion_ratio),
            target_min_chromatic_step_ratio=float(args.target_min_chromatic_step_ratio),
            target_max_large_leap_ratio=float(args.target_max_large_leap_ratio),
            max_motion_balance_bar_pitch_class_jaccard=float(args.max_motion_balance_bar_pitch_class_jaccard),
        )
        repaired_rows.append(
            {
                **item,
                "objective_metrics": metrics,
                "score": float(score),
                "gate_penalty": float(gate_penalty),
            }
        )
    return sorted(
        repaired_rows,
        key=lambda row: selection_sort_key(row, selection_profile=str(args.selection_profile)),
    )


def markdown_report(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Bebop Language Safety Gate Feasibility Sweep",
        "",
        "## Summary",
        "",
        f"- repaired candidate pool: `{summary['repaired_pool_count']}`",
        f"- safety baseline candidate count: `{summary['safety_baseline']['candidate_count']}`",
        f"- safety baseline selectable max-per-case 2: `{summary['safety_baseline_selectable_max_per_case_2']}`",
        f"- safety baseline selectable max-per-case 3: `{summary['safety_baseline_selectable_max_per_case_3']}`",
        f"- selected count target: `{summary['selected_count_target']}`",
        f"- feasible guard/motion config count: `{summary['feasible_guard_motion_config_count']}`",
        f"- min feasible offbeat max: `{summary['feasible_config_summary']['min_feasible_max_offbeat_non_chord_ratio']}`",
        f"- min feasible bar similarity max: `{summary['feasible_config_summary']['min_feasible_max_bar_pitch_class_jaccard']}`",
        f"- stricter offbeat feasible: `{str(summary['feasible_config_summary']['stricter_offbeat_feasible']).lower()}`",
        f"- stricter bar similarity feasible: `{str(summary['feasible_config_summary']['stricter_bar_similarity_feasible']).lower()}`",
        f"- quality claim: `{str(report['quality_claimed']).lower()}`",
        "",
        "## Decision",
        "",
        f"- next boundary: `{report['decision']['next_boundary']}`",
        f"- representative replacement: `{str(report['decision']['representative_replacement']).lower()}`",
        "",
        "## Top Configs",
        "",
    ]
    for item in report["top_configs"][:8]:
        lines.append(
            "- "
            f"offbeat max `{item['max_offbeat_non_chord_ratio']:.4f}`, "
            f"bar sim max `{item['max_bar_pitch_class_jaccard']:.4f}`, "
            f"step `{item['min_step_motion_ratio']:.4f}`, "
            f"chromatic `{item['min_chromatic_step_ratio']:.4f}`, "
            f"large-leap `{item['max_large_leap_ratio']:.4f}`, "
            f"selectable max2 `{item['selectable_max_per_case_2']}`, "
            f"selectable max3 `{item['selectable_max_per_case_3']}`, "
            f"count `{item['candidate_count']}`"
        )
    lines.extend(["", "## Boundary", "", "- musical quality claim: `false`", "- listening preference claim: `false`"])
    return "\n".join(lines) + "\n"


def run_sweep(args: argparse.Namespace) -> dict[str, Any]:
    repaired_rows = build_repaired_pool(args)
    safety_rows = filter_guard_rows(
        repaired_rows,
        max_gate_penalty=float(args.max_gate_penalty),
        max_offbeat_non_chord_ratio=float(args.max_offbeat_non_chord_ratio),
        max_unresolved_offbeat_non_chord_ratio=float(args.max_unresolved_offbeat_non_chord_ratio),
        max_dominant_altered_offbeat_ratio=float(args.max_dominant_altered_offbeat_ratio),
        max_adjacent_repeat_ratio=float(args.max_adjacent_repeat_ratio),
        max_bar_pitch_class_jaccard=float(args.max_bar_pitch_class_jaccard),
    )
    max_per_case_values = parse_int_grid(str(args.max_per_case_values))
    offbeat_values = parse_float_grid(str(args.max_offbeat_non_chord_ratios))
    bar_similarity_values = parse_float_grid(str(args.max_bar_pitch_class_jaccards))
    step_values = parse_float_grid(str(args.min_step_motion_ratios))
    chromatic_values = parse_float_grid(str(args.min_chromatic_step_ratios))
    large_leap_values = parse_float_grid(str(args.max_large_leap_ratios))
    configs = build_guard_motion_configs(
        repaired_rows,
        offbeat_values=offbeat_values,
        bar_similarity_values=bar_similarity_values,
        step_values=step_values,
        chromatic_values=chromatic_values,
        large_leap_values=large_leap_values,
        max_per_case_values=max_per_case_values,
        selected_count=int(args.selected_count),
        max_gate_penalty=float(args.max_gate_penalty),
        max_unresolved_offbeat_non_chord_ratio=float(args.max_unresolved_offbeat_non_chord_ratio),
        max_dominant_altered_offbeat_ratio=float(args.max_dominant_altered_offbeat_ratio),
        max_adjacent_repeat_ratio=float(args.max_adjacent_repeat_ratio),
    )
    feasible_count = sum(
        1
        for item in configs
        if any(bool(value) for value in dict(item["feasible_for_selected_count"]).values())
    )
    feasible_summary = summarize_feasible_configs(
        configs,
        baseline_max_offbeat_non_chord_ratio=float(args.max_offbeat_non_chord_ratio),
        baseline_max_bar_pitch_class_jaccard=float(args.max_bar_pitch_class_jaccard),
        max_per_case_values=max_per_case_values,
    )
    report = {
        "schema_version": "stage_b_midi_to_solo_bebop_language_safety_gate_feasibility_sweep_v1",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "boundary": "stage_b_midi_to_solo_bebop_language_safety_gate_feasibility_sweep",
        "summary": {
            "repaired_pool_count": len(repaired_rows),
            "safety_baseline": summarize_rows(safety_rows, max_per_case=max(max_per_case_values)),
            "safety_baseline_selectable_max_per_case_2": selectable_count(case_counts(safety_rows), max_per_case=2),
            "safety_baseline_selectable_max_per_case_3": selectable_count(case_counts(safety_rows), max_per_case=3),
            "selected_count_target": int(args.selected_count),
            "feasible_guard_motion_config_count": feasible_count,
            "feasible_config_summary": feasible_summary,
            "motion_balance_repair": bool(args.repair_motion_balance),
        },
        "generation": {
            "bars": int(args.bars),
            "bpm": float(args.bpm),
            "target_chord_tone_ratio": float(args.target_chord_tone_ratio),
            "target_offbeat_non_chord_ratio": float(args.target_offbeat_non_chord_ratio),
            "max_gate_penalty": float(args.max_gate_penalty),
            "max_offbeat_non_chord_ratio": float(args.max_offbeat_non_chord_ratio),
            "max_unresolved_offbeat_non_chord_ratio": float(args.max_unresolved_offbeat_non_chord_ratio),
            "max_dominant_altered_offbeat_ratio": float(args.max_dominant_altered_offbeat_ratio),
            "max_adjacent_repeat_ratio": float(args.max_adjacent_repeat_ratio),
            "max_bar_pitch_class_jaccard": float(args.max_bar_pitch_class_jaccard),
            "max_per_case_values": max_per_case_values,
            "max_offbeat_non_chord_ratios": offbeat_values,
            "max_bar_pitch_class_jaccards": bar_similarity_values,
            "min_step_motion_ratios": step_values,
            "min_chromatic_step_ratios": chromatic_values,
            "max_large_leap_ratios": large_leap_values,
            "repair_motion_balance": bool(args.repair_motion_balance),
            "repair_motion_balance_iterations": int(args.repair_motion_balance_iterations),
            "target_min_step_motion_ratio": float(args.target_min_step_motion_ratio),
            "target_min_chromatic_step_ratio": float(args.target_min_chromatic_step_ratio),
            "target_max_large_leap_ratio": float(args.target_max_large_leap_ratio),
            "max_motion_balance_bar_pitch_class_jaccard": float(args.max_motion_balance_bar_pitch_class_jaccard),
            "selection_profile": str(args.selection_profile),
        },
        "top_configs": configs,
        "quality_claimed": False,
        "model_direct_claimed": False,
        "decision": {
            "current_boundary": "stage_b_midi_to_solo_bebop_language_safety_gate_feasibility_sweep",
            "next_boundary": (
                "motion_balance_guard_tightening_candidate_package"
                if feasible_count > 0
                else "targeted_generation_or_pool_expansion"
            ),
            "representative_replacement": False,
        },
    }
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run bebop-language safety gate feasibility sweep")
    parser.add_argument("--source_root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument(
        "--package_globs",
        default=(
            "manual_2026_06_13_bebop_language_*/bebop_language_package.json,"
            "parameter_sweep/manual_2026_06_13_bebop_language_param_sweep_v6_data_contour_resolution/config_*/bebop_language_package.json,"
            "parameter_sweep/manual_2026_06_13_bebop_language_param_sweep_v7_altered_balanced/config_*/bebop_language_package.json,"
            "parameter_sweep/manual_2026_06_13_bebop_language_param_sweep_v8_v22_micro/config_*/bebop_language_package.json,"
            "parameter_sweep/manual_2026_06_13_bebop_language_param_sweep_v9_strict_consonance/config_*/bebop_language_package.json,"
            "parameter_sweep/manual_2026_06_13_bebop_language_param_sweep_v10_interval_repeat_rank/config_*/bebop_language_package.json,"
            "parameter_sweep/manual_2026_06_13_bebop_language_param_sweep_v11_large_leap_pool/config_*/bebop_language_package.json"
        ),
    )
    parser.add_argument("--output_root", default=str(DEFAULT_OUTPUT_ROOT / "best_of_feasibility"))
    parser.add_argument("--run_id", default="")
    parser.add_argument("--bars", type=int, default=8)
    parser.add_argument("--bpm", type=float, default=124.0)
    parser.add_argument("--selected_count", type=int, default=8)
    parser.add_argument("--max_per_case_values", default="2,3")
    parser.add_argument("--target_chord_tone_ratio", type=float, default=0.78)
    parser.add_argument("--target_offbeat_non_chord_ratio", type=float, default=0.38)
    parser.add_argument("--max_gate_penalty", type=float, default=0.0)
    parser.add_argument("--max_offbeat_non_chord_ratio", type=float, default=0.40625)
    parser.add_argument("--max_offbeat_non_chord_ratios", default="0.3828125,0.390625,0.3984375,0.40625")
    parser.add_argument("--max_unresolved_offbeat_non_chord_ratio", type=float, default=0.03125)
    parser.add_argument("--max_dominant_altered_offbeat_ratio", type=float, default=0.25)
    parser.add_argument("--max_adjacent_repeat_ratio", type=float, default=0.0)
    parser.add_argument("--max_bar_pitch_class_jaccard", type=float, default=0.70)
    parser.add_argument("--max_bar_pitch_class_jaccards", default="0.625,0.65,0.675,0.70")
    parser.add_argument("--min_step_motion_ratios", default="0.36,0.38,0.40")
    parser.add_argument("--min_chromatic_step_ratios", default="0.18,0.20,0.22")
    parser.add_argument("--max_large_leap_ratios", default="0.06,0.08,0.10")
    parser.add_argument("--repair_bar_similarity_iterations", type=int, default=4)
    parser.add_argument("--repair_enclosure_density_iterations", type=int, default=8)
    parser.add_argument("--repair_unresolved_offbeat_iterations", type=int, default=4)
    parser.add_argument("--repair_adjacent_repeats_iterations", type=int, default=4)
    parser.add_argument("--repair_large_leaps_iterations", type=int, default=8)
    parser.add_argument("--min_large_leap_repair_enclosure_proxy_ratio", type=float, default=0.28125)
    parser.add_argument("--max_enclosure_repair_offbeat_non_chord_ratio", type=float, default=0.421875)
    parser.add_argument("--repair_motion_balance", action="store_true")
    parser.add_argument("--repair_motion_balance_iterations", type=int, default=12)
    parser.add_argument("--target_min_step_motion_ratio", type=float, default=0.40)
    parser.add_argument("--target_min_chromatic_step_ratio", type=float, default=0.22)
    parser.add_argument("--target_max_large_leap_ratio", type=float, default=0.055)
    parser.add_argument("--max_motion_balance_bar_pitch_class_jaccard", type=float, default=0.70)
    parser.add_argument(
        "--selection_profile",
        choices=["score", "bebop_language", "bebop_stepwise_chromatic"],
        default="bebop_stepwise_chromatic",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    run_id = str(args.run_id or datetime.now(timezone.utc).strftime("manual_%Y_%m_%d_bebop_language_safety_%H%M%S"))
    output_dir = Path(args.output_root) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    report = run_sweep(args)
    report["output_dir"] = str(output_dir)
    json_path = output_dir / "bebop_language_safety_gate_feasibility_sweep.json"
    md_path = output_dir / "bebop_language_safety_gate_feasibility_sweep.md"
    write_json(json_path, report)
    write_text(md_path, markdown_report(report))
    print(json.dumps({"report_path": str(json_path), "markdown_path": str(md_path), **report["summary"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
