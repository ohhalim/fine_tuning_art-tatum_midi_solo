"""Compare plain constrained and coverage-aware Stage B generation."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from scripts.run_stage_b_sampling_sweep import (  # noqa: E402
    parse_int_list,
    probe_command,
    read_json,
    row_from_probe_report,
    run_command,
    suffix_for_float,
    write_json,
)


def parse_modes(raw: str) -> list[str]:
    modes = [mode.strip().lower() for mode in raw.split(",") if mode.strip()]
    invalid = [mode for mode in modes if mode not in {"plain", "coverage"}]
    if invalid:
        raise ValueError(f"Unknown coverage sweep modes: {invalid}")
    return modes


def config_run_id(base_run_id: str, mode: str, note_groups_per_bar: int, top_k: int, temperature: float) -> str:
    return (
        f"{base_run_id}_{mode}_g{int(note_groups_per_bar)}"
        f"_k{int(top_k)}_t{suffix_for_float(float(temperature))}"
    )


def row_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        int(row["strict_valid_sample_count"]),
        int(row["valid_sample_count"]),
        float(row["avg_onset_coverage_ratio"]),
        float(row["avg_sustained_coverage_ratio"]),
        -float(row["collapse_warning_sample_rate"]),
        -int(row["max_longest_sustained_empty_run_steps"]),
    )


def best_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    return max(rows, key=row_sort_key, default=None)


def compact_config(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {
        "mode": row["mode"],
        "note_groups_per_bar": int(row["note_groups_per_bar"]),
        "top_k": int(row["top_k"]),
        "temperature": float(row["temperature"]),
        "run_id": row["run_id"],
        "strict_valid_sample_count": int(row["strict_valid_sample_count"]),
        "valid_sample_count": int(row["valid_sample_count"]),
        "avg_onset_coverage_ratio": float(row["avg_onset_coverage_ratio"]),
        "max_longest_sustained_empty_run_steps": int(row["max_longest_sustained_empty_run_steps"]),
    }


def build_ab_summary(
    rows: list[dict[str, Any]],
    min_best_strict_valid_samples: int = 1,
    max_collapse_warning_sample_rate: float = 0.34,
) -> dict[str, Any]:
    mode_rows = {
        "plain": [row for row in rows if row["mode"] == "plain"],
        "coverage": [row for row in rows if row["mode"] == "coverage"],
    }
    best = best_row(rows)
    best_plain = best_row(mode_rows["plain"])
    best_coverage = best_row(mode_rows["coverage"])
    passed_coverage = bool(
        best_coverage
        and int(best_coverage["strict_valid_sample_count"]) >= int(min_best_strict_valid_samples)
        and float(best_coverage["collapse_warning_sample_rate"]) <= float(max_collapse_warning_sample_rate)
    )
    passed_comparison = bool(best_plain is not None and best_coverage is not None)
    return {
        "config_count": int(len(rows)),
        "mode_counts": {mode: int(len(mode_rows[mode])) for mode in sorted(mode_rows)},
        "best_config": compact_config(best),
        "best_plain_config": compact_config(best_plain),
        "best_coverage_config": compact_config(best_coverage),
        "min_best_strict_valid_samples": int(min_best_strict_valid_samples),
        "max_collapse_warning_sample_rate": float(max_collapse_warning_sample_rate),
        "passed_coverage_gate": passed_coverage,
        "passed_comparison_gate": passed_comparison,
        "passed_ab_sweep_gate": bool(passed_coverage and passed_comparison),
    }


def markdown_table(rows: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    lines = [
        "# Stage B Coverage-Aware A/B Sweep",
        "",
        f"- passed A/B sweep gate: `{str(summary['passed_ab_sweep_gate']).lower()}`",
        f"- best plain config: `{summary['best_plain_config']}`",
        f"- best coverage config: `{summary['best_coverage_config']}`",
        "",
        "| mode | groups/bar | samples | grammar | valid | strict | onset | sustained | span | max empty | dead air | collapse | strict pass |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in rows:
        lines.append(
            "| {mode} | {note_groups_per_bar} | {sample_count} | {grammar_gate_sample_count} | "
            "{valid_sample_count} | {strict_valid_sample_count} | {avg_onset_coverage_ratio:.3f} | "
            "{avg_sustained_coverage_ratio:.3f} | {avg_position_span_ratio:.3f} | "
            "{max_longest_sustained_empty_run_steps} | {avg_dead_air_ratio:.3f} | "
            "{collapse_warning_sample_rate:.3f} | {passed_strict_review_gate} |".format(**row)
        )
    lines.append("")
    lines.append("## Failure Reasons")
    lines.append("")
    for row in rows:
        lines.append(
            f"- `{row['mode']}`, groups `{row['note_groups_per_bar']}`: "
            f"{row['diagnostic_failure_reasons']}"
        )
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Stage B coverage-aware A/B sweep")
    parser.add_argument("--output_root", type=str, default=str(ROOT_DIR / "outputs" / "stage_b_coverage_ab_sweep"))
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--issue_number", type=int, default=39)
    parser.add_argument("--modes", type=str, default="plain,coverage")
    parser.add_argument("--note_groups_per_bar_values", type=str, default="4,6,8")
    parser.add_argument("--coverage_position_window", type=int, default=0)
    parser.add_argument("--top_k", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--max_files", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_sequence", type=int, default=96)
    parser.add_argument("--num_samples", type=int, default=3)
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
    sweep_dir = Path(args.output_root) / run_id
    modes = parse_modes(args.modes)
    note_group_values = parse_int_list(args.note_groups_per_bar_values)
    if not modes or not note_group_values:
        raise ValueError("--modes and --note_groups_per_bar_values must not be empty")

    command_results: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    checkpoint_dir: Path | None = None
    first_config = True

    for note_groups_per_bar in note_group_values:
        for mode in modes:
            config_id = config_run_id(
                run_id,
                mode=mode,
                note_groups_per_bar=note_groups_per_bar,
                top_k=args.top_k,
                temperature=args.temperature,
            )
            probe_args = argparse.Namespace(**vars(args))
            probe_args.train_top_k = args.top_k
            probe_args.top_ks = str(args.top_k)
            probe_args.temperatures = str(args.temperature)
            probe_args.constrained_note_groups_per_bar = int(note_groups_per_bar)
            probe_args.coverage_aware_positions = mode == "coverage"
            probe_args.coverage_position_window = int(args.coverage_position_window)
            cmd = probe_command(
                probe_args,
                run_id=config_id,
                top_k=int(args.top_k),
                temperature=float(args.temperature),
                checkpoint_dir=checkpoint_dir,
                skip_prepare_train=not first_config,
            )
            if mode == "coverage":
                cmd.extend(["--coverage_aware_positions", "--coverage_position_window", str(args.coverage_position_window)])
            result = run_command(cmd)
            command_results.append(result)
            if result["returncode"] != 0:
                report = {
                    "run_id": run_id,
                    "failure_reason": f"probe command failed for mode={mode}, groups={note_groups_per_bar}",
                    "command_results": command_results,
                }
                write_json(sweep_dir / "ab_sweep_report.json", report)
                print(json.dumps(report, ensure_ascii=True, indent=2))
                return int(result["returncode"])

            report_path = Path(args.output_root) / config_id / "report.json"
            probe_report = read_json(report_path)
            if checkpoint_dir is None:
                checkpoint_dir = Path(probe_report["checkpoint_dir"])
            first_config = False
            row = row_from_probe_report(int(args.top_k), float(args.temperature), config_id, probe_report)
            row["mode"] = mode
            row["note_groups_per_bar"] = int(note_groups_per_bar)
            rows.append(row)

    rows = sorted(rows, key=lambda row: (int(row["note_groups_per_bar"]), row["mode"]))
    summary = build_ab_summary(
        rows,
        min_best_strict_valid_samples=args.min_best_strict_valid_samples,
        max_collapse_warning_sample_rate=args.max_collapse_warning_sample_rate,
    )
    report = {
        "run_id": run_id,
        "run_dir": str(sweep_dir),
        "issue": int(args.issue_number),
        "modes": modes,
        "note_groups_per_bar_values": note_group_values,
        "top_k": int(args.top_k),
        "temperature": float(args.temperature),
        "summary": summary,
        "rows": rows,
        "command_results": command_results,
    }
    write_json(sweep_dir / "ab_sweep_report.json", report)
    (sweep_dir / "ab_sweep_report.md").write_text(markdown_table(rows, summary), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True, indent=2))
    return 0 if summary["passed_ab_sweep_gate"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
