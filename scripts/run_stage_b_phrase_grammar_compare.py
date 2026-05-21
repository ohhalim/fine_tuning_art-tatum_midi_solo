"""Compare Stage B phrase grammar constraints on one tiny checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from scripts.run_stage_b_pitch_mode_compare import (  # noqa: E402
    export_named_comparison_midis,
    export_review_candidates,
)
from scripts.run_stage_b_sampling_sweep import (  # noqa: E402
    probe_command,
    read_json,
    row_from_probe_report,
    run_command,
    write_json,
)

VALID_GRAMMAR_MODES = {"approach_baseline", "swing_motif_approach"}


def parse_grammar_modes(raw: str) -> list[str]:
    modes = [mode.strip().lower() for mode in raw.split(",") if mode.strip()]
    invalid = [mode for mode in modes if mode not in VALID_GRAMMAR_MODES]
    if invalid:
        raise ValueError(f"Unknown phrase grammar modes: {invalid}")
    return modes


def config_run_id(base_run_id: str, grammar_mode: str) -> str:
    return f"{base_run_id}_{grammar_mode}"


def best_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    return max(
        rows,
        key=lambda row: (
            int(row["strict_valid_sample_count"]),
            float(row["avg_syncopated_onset_ratio"]),
            float(row["avg_unique_bar_position_pattern_ratio"]),
            -float(row["avg_most_common_ioi_ratio"]),
            -float(row["avg_most_common_duration_ratio"]),
        ),
        default=None,
    )


def compact_row(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {
        "grammar_mode": row["grammar_mode"],
        "run_id": row["run_id"],
        "strict_valid_sample_count": int(row["strict_valid_sample_count"]),
        "valid_sample_count": int(row["valid_sample_count"]),
        "avg_root_tone_ratio": float(row["avg_root_tone_ratio"]),
        "avg_tension_ratio": float(row["avg_tension_ratio"]),
        "avg_approach_resolution_ratio": float(row["avg_approach_resolution_ratio"]),
        "avg_syncopated_onset_ratio": float(row["avg_syncopated_onset_ratio"]),
        "avg_unique_bar_position_pattern_ratio": float(row["avg_unique_bar_position_pattern_ratio"]),
        "avg_duration_diversity_ratio": float(row["avg_duration_diversity_ratio"]),
        "avg_most_common_duration_ratio": float(row["avg_most_common_duration_ratio"]),
        "avg_ioi_diversity_ratio": float(row["avg_ioi_diversity_ratio"]),
        "avg_most_common_ioi_ratio": float(row["avg_most_common_ioi_ratio"]),
    }


def build_phrase_grammar_summary(
    rows: list[dict[str, Any]],
    min_best_strict_valid_samples: int = 1,
) -> dict[str, Any]:
    mode_rows = {
        mode: [row for row in rows if row.get("grammar_mode") == mode]
        for mode in sorted(VALID_GRAMMAR_MODES)
    }
    best_baseline = best_row(mode_rows["approach_baseline"])
    best_swing = best_row(mode_rows["swing_motif_approach"])
    comparison_ready = bool(best_baseline and best_swing)
    passed_baseline = bool(
        best_baseline
        and int(best_baseline["strict_valid_sample_count"]) >= int(min_best_strict_valid_samples)
    )
    passed_swing = bool(
        best_swing
        and int(best_swing["strict_valid_sample_count"]) >= int(min_best_strict_valid_samples)
    )
    baseline_sync = float(best_baseline.get("avg_syncopated_onset_ratio", 0.0)) if best_baseline else 0.0
    swing_sync = float(best_swing.get("avg_syncopated_onset_ratio", 0.0)) if best_swing else 0.0
    baseline_ioi = float(best_baseline.get("avg_most_common_ioi_ratio", 0.0)) if best_baseline else 0.0
    swing_ioi = float(best_swing.get("avg_most_common_ioi_ratio", 0.0)) if best_swing else 0.0
    return {
        "config_count": int(len(rows)),
        "mode_counts": {mode: int(len(mode_rows[mode])) for mode in sorted(VALID_GRAMMAR_MODES)},
        "best_config": compact_row(best_row(rows)),
        "best_approach_baseline_config": compact_row(best_baseline),
        "best_swing_motif_approach_config": compact_row(best_swing),
        "min_best_strict_valid_samples": int(min_best_strict_valid_samples),
        "comparison_ready": comparison_ready,
        "passed_approach_baseline_gate": passed_baseline,
        "passed_swing_motif_approach_gate": passed_swing,
        "passed_compare_gate": bool(comparison_ready and passed_baseline and passed_swing),
        "syncopation_delta_swing_minus_baseline": float(swing_sync - baseline_sync),
        "most_common_ioi_delta_swing_minus_baseline": float(swing_ioi - baseline_ioi),
    }


def markdown_table(rows: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    lines = [
        "# Stage B Phrase Grammar Comparison",
        "",
        f"- passed compare gate: `{str(summary['passed_compare_gate']).lower()}`",
        f"- best baseline config: `{summary['best_approach_baseline_config']}`",
        f"- best swing/motif config: `{summary['best_swing_motif_approach_config']}`",
        f"- syncopation delta, swing minus baseline: `{summary['syncopation_delta_swing_minus_baseline']:.3f}`",
        f"- most-common IOI delta, swing minus baseline: `{summary['most_common_ioi_delta_swing_minus_baseline']:.3f}`",
        "",
        "| grammar | samples | strict | root | tension | resolved | sync | bar-var | dur-var | dur-rep | ioi-var | ioi-rep | report |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {grammar_mode} | {sample_count} | {strict_valid_sample_count} | "
            "{avg_root_tone_ratio:.3f} | {avg_tension_ratio:.3f} | "
            "{avg_approach_resolution_ratio:.3f} | {avg_syncopated_onset_ratio:.3f} | "
            "{avg_unique_bar_position_pattern_ratio:.3f} | {avg_duration_diversity_ratio:.3f} | "
            "{avg_most_common_duration_ratio:.3f} | {avg_ioi_diversity_ratio:.3f} | "
            "{avg_most_common_ioi_ratio:.3f} | `{report_path}` |".format(**row)
        )
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Stage B phrase grammar comparison")
    parser.add_argument("--output_root", type=str, default=str(ROOT_DIR / "outputs" / "stage_b_phrase_grammar_compare"))
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--issue_number", type=int, default=59)
    parser.add_argument("--grammar_modes", type=str, default="approach_baseline,swing_motif_approach")
    parser.add_argument("--note_groups_per_bar", type=int, default=8)
    parser.add_argument("--coverage_position_window", type=int, default=0)
    parser.add_argument("--chord_pitch_repeat_window", type=int, default=2)
    parser.add_argument("--top_k", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--max_files", type=int, default=2)
    parser.add_argument("--window_bars", type=int, default=8)
    parser.add_argument("--window_stride_bars", type=int, default=4)
    parser.add_argument("--min_window_target_notes", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_sequence", type=int, default=384)
    parser.add_argument("--num_samples", type=int, default=3)
    parser.add_argument("--bars", type=int, default=8)
    parser.add_argument("--max_simultaneous_notes", type=int, default=2)
    parser.add_argument("--min_valid_samples", type=int, default=1)
    parser.add_argument("--min_strict_valid_samples", type=int, default=1)
    parser.add_argument("--min_best_strict_valid_samples", type=int, default=1)
    parser.add_argument("--max_collapse_warning_sample_rate", type=float, default=0.34)
    parser.add_argument("--strict_min_unique_pitches", type=int, default=3)
    parser.add_argument("--strict_min_unique_positions", type=int, default=3)
    parser.add_argument("--strict_min_unique_position_pitch_pairs", type=int, default=4)
    parser.add_argument("--strict_max_repeated_position_pitch_pair_ratio", type=float, default=0.49)
    parser.add_argument("--strict_max_postprocess_removal_ratio", type=float, default=0.49)
    parser.add_argument("--require_all_grammar_samples", action="store_true")
    parser.add_argument("--review_output_root", type=str, default=str(ROOT_DIR / "outputs" / "stage_b_review_candidates"))
    parser.add_argument("--review_top_n", type=int, default=3)
    parser.add_argument("--copy_review_midi", action="store_true")
    parser.add_argument("--n_layers", type=int, default=1)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--dim_feedforward", type=int, default=128)
    parser.add_argument("--lora_r", type=int, default=4)
    parser.add_argument("--lora_alpha", type=int, default=8)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    compare_dir = Path(args.output_root) / run_id
    grammar_modes = parse_grammar_modes(args.grammar_modes)
    if not grammar_modes:
        raise ValueError("--grammar_modes must not be empty")

    command_results: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    review_exports: dict[str, dict[str, Any]] = {}
    checkpoint_dir: Path | None = None
    first_config = True

    for grammar_mode in grammar_modes:
        mode_run_id = config_run_id(run_id, grammar_mode)
        probe_args = argparse.Namespace(**vars(args))
        probe_args.output_root = args.output_root
        probe_args.constrained_note_groups_per_bar = int(args.note_groups_per_bar)
        probe_args.coverage_aware_positions = True
        probe_args.coverage_position_window = int(args.coverage_position_window)
        probe_args.chord_aware_pitches = True
        probe_args.chord_pitch_mode = "approach_tensions"
        probe_args.chord_pitch_repeat_window = int(args.chord_pitch_repeat_window)
        probe_args.jazz_rhythm_positions = grammar_mode == "swing_motif_approach"
        probe_args.jazz_duration_tokens = grammar_mode == "swing_motif_approach"
        probe_args.jazz_rhythm_profile = "swing_motif"
        cmd = probe_command(
            probe_args,
            run_id=mode_run_id,
            top_k=int(args.top_k),
            temperature=float(args.temperature),
            checkpoint_dir=checkpoint_dir,
            skip_prepare_train=not first_config,
        )
        cmd.extend(
            [
                "--window_bars",
                str(args.window_bars),
                "--window_stride_bars",
                str(args.window_stride_bars),
                "--min_window_target_notes",
                str(args.min_window_target_notes),
                "--bars",
                str(args.bars),
            ]
        )
        cmd.extend(["--coverage_aware_positions", "--coverage_position_window", str(args.coverage_position_window)])
        result = run_command(cmd)
        command_results.append(result)
        if result["returncode"] != 0:
            report = {
                "run_id": run_id,
                "failure_reason": f"probe command failed for grammar_mode={grammar_mode}",
                "command_results": command_results,
            }
            write_json(compare_dir / "phrase_grammar_compare_report.json", report)
            print(json.dumps(report, ensure_ascii=True, indent=2))
            return int(result["returncode"])

        report_path = Path(args.output_root) / mode_run_id / "report.json"
        probe_report = read_json(report_path)
        if checkpoint_dir is None:
            checkpoint_dir = Path(probe_report["checkpoint_dir"])
        first_config = False
        row = row_from_probe_report(int(args.top_k), float(args.temperature), mode_run_id, probe_report)
        row["grammar_mode"] = grammar_mode
        row["note_groups_per_bar"] = int(args.note_groups_per_bar)
        rows.append(row)

        review_output_dir = Path(args.review_output_root) / run_id / grammar_mode
        review_exports[grammar_mode] = export_review_candidates(
            report_path=report_path,
            output_dir=review_output_dir,
            top_n=int(args.review_top_n),
            copy_midi=bool(args.copy_review_midi),
        )

    rows = sorted(rows, key=lambda row: str(row["grammar_mode"]))
    summary = build_phrase_grammar_summary(
        rows,
        min_best_strict_valid_samples=args.min_best_strict_valid_samples,
    )
    report = {
        "run_id": run_id,
        "run_dir": str(compare_dir),
        "issue": int(args.issue_number),
        "grammar_modes": grammar_modes,
        "note_groups_per_bar": int(args.note_groups_per_bar),
        "top_k": int(args.top_k),
        "temperature": float(args.temperature),
        "summary": summary,
        "rows": rows,
        "review_exports": review_exports,
        "named_comparison_midis": export_named_comparison_midis(
            review_exports,
            output_dir=Path(args.review_output_root) / run_id,
        )
        if args.copy_review_midi
        else [],
        "command_results": command_results,
    }
    write_json(compare_dir / "phrase_grammar_compare_report.json", report)
    (compare_dir / "phrase_grammar_compare_report.md").write_text(markdown_table(rows, summary), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True, indent=2))
    return 0 if summary["passed_compare_gate"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
