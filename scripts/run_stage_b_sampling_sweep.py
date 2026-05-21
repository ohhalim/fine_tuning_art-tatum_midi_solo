"""Run a small Stage B sampling sweep against one trained tiny checkpoint."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
PROBE_SCRIPT = ROOT_DIR / "scripts" / "run_stage_b_generation_probe.py"


def parse_int_list(raw: str) -> list[int]:
    return [int(value.strip()) for value in raw.split(",") if value.strip()]


def parse_float_list(raw: str) -> list[float]:
    return [float(value.strip()) for value in raw.split(",") if value.strip()]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def suffix_for_float(value: float) -> str:
    return str(value).replace("-", "m").replace(".", "p")


def run_command(cmd: list[str]) -> dict[str, Any]:
    completed = subprocess.run(cmd, cwd=ROOT_DIR, text=True, capture_output=True, check=False)
    return {
        "cmd": cmd,
        "returncode": int(completed.returncode),
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
    }


def probe_command(
    args: argparse.Namespace,
    run_id: str,
    top_k: int,
    temperature: float,
    checkpoint_dir: Path | None = None,
    skip_prepare_train: bool = False,
) -> list[str]:
    cmd = [
        sys.executable,
        str(PROBE_SCRIPT),
        "--run_id",
        run_id,
        "--output_root",
        str(Path(args.output_root)),
        "--issue_number",
        str(args.issue_number),
        "--max_files",
        str(args.max_files),
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(args.batch_size),
        "--max_sequence",
        str(args.max_sequence),
        "--num_samples",
        str(args.num_samples),
        "--generation_mode",
        "constrained",
        "--constrained_note_groups_per_bar",
        str(args.constrained_note_groups_per_bar),
        "--postprocess_overlap",
        "--max_simultaneous_notes",
        str(args.max_simultaneous_notes),
        "--temperature",
        str(temperature),
        "--top_k",
        str(top_k),
        "--min_valid_samples",
        str(args.min_valid_samples),
        "--min_strict_valid_samples",
        str(args.min_strict_valid_samples),
        "--max_collapse_warning_sample_rate",
        str(args.max_collapse_warning_sample_rate),
        "--strict_min_unique_pitches",
        str(args.strict_min_unique_pitches),
        "--strict_min_unique_positions",
        str(args.strict_min_unique_positions),
        "--strict_min_unique_position_pitch_pairs",
        str(args.strict_min_unique_position_pitch_pairs),
        "--strict_max_repeated_position_pitch_pair_ratio",
        str(args.strict_max_repeated_position_pitch_pair_ratio),
        "--strict_max_postprocess_removal_ratio",
        str(args.strict_max_postprocess_removal_ratio),
        "--n_layers",
        str(args.n_layers),
        "--num_heads",
        str(args.num_heads),
        "--d_model",
        str(args.d_model),
        "--dim_feedforward",
        str(args.dim_feedforward),
        "--lora_r",
        str(args.lora_r),
        "--lora_alpha",
        str(args.lora_alpha),
    ]
    if args.require_all_grammar_samples:
        cmd.append("--require_all_grammar_samples")
    if getattr(args, "chord_aware_pitches", False):
        cmd.append("--chord_aware_pitches")
        cmd.extend(["--chord_pitch_mode", str(getattr(args, "chord_pitch_mode", "tones_tensions"))])
        cmd.extend(["--chord_pitch_repeat_window", str(getattr(args, "chord_pitch_repeat_window", 2))])
    if skip_prepare_train:
        cmd.extend(["--skip_prepare", "--skip_train"])
    if checkpoint_dir is not None:
        cmd.extend(["--checkpoint_dir", str(checkpoint_dir)])
    return cmd


def row_from_probe_report(top_k: int, temperature: float, run_id: str, report: dict[str, Any]) -> dict[str, Any]:
    summary = report.get("summary", {})
    samples = report.get("samples", [])
    dead_air_ratios = [
        float(sample.get("metrics", {}).get("dead_air_ratio", 0.0) or 0.0)
        for sample in samples
        if isinstance(sample, dict)
    ]
    chord_tone_ratios = [
        float(sample.get("metrics", {}).get("chord_tone_ratio", 0.0) or 0.0)
        for sample in samples
        if isinstance(sample, dict)
    ]
    return {
        "run_id": run_id,
        "top_k": int(top_k),
        "temperature": float(temperature),
        "generation_mode": str(report.get("generation_mode", "")),
        "coverage_aware_positions": bool(report.get("coverage_aware_positions", False)),
        "coverage_position_window": int(report.get("coverage_position_window", 0) or 0),
        "chord_aware_pitches": bool(report.get("chord_aware_pitches", False)),
        "chord_pitch_mode": str(report.get("chord_pitch_mode", "")),
        "chord_pitch_repeat_window": int(report.get("chord_pitch_repeat_window", 0) or 0),
        "constrained_note_groups_per_bar": int(report.get("constrained_note_groups_per_bar", 0) or 0),
        "sample_count": int(summary.get("sample_count", 0)),
        "grammar_gate_sample_count": int(summary.get("grammar_gate_sample_count", 0)),
        "valid_sample_count": int(summary.get("valid_sample_count", 0)),
        "strict_valid_sample_count": int(summary.get("strict_valid_sample_count", 0)),
        "grammar_gate_sample_rate": float(summary.get("grammar_gate_sample_rate", 0.0)),
        "valid_sample_rate": float(summary.get("valid_sample_rate", 0.0)),
        "strict_valid_sample_rate": float(summary.get("strict_valid_sample_rate", 0.0)),
        "collapse_warning_sample_count": int(summary.get("collapse_warning_sample_count", 0)),
        "collapse_warning_sample_rate": float(summary.get("collapse_warning_sample_rate", 0.0)),
        "max_collapse_warning_sample_rate": float(summary.get("max_collapse_warning_sample_rate", 0.0)),
        "passed_collapse_rate_gate": bool(summary.get("passed_collapse_rate_gate", False)),
        "passed_strict_generation_gate": bool(summary.get("passed_strict_generation_gate", False)),
        "passed_strict_review_gate": bool(summary.get("passed_strict_review_gate", False)),
        "avg_repeated_position_pitch_pair_ratio": float(
            summary.get("avg_repeated_position_pitch_pair_ratio", 0.0)
        ),
        "max_repeated_position_pitch_pair_ratio": float(
            summary.get("max_repeated_position_pitch_pair_ratio", 0.0)
        ),
        "avg_postprocess_removal_ratio": float(summary.get("avg_postprocess_removal_ratio", 0.0)),
        "max_postprocess_removal_ratio": float(summary.get("max_postprocess_removal_ratio", 0.0)),
        "avg_onset_coverage_ratio": float(summary.get("avg_onset_coverage_ratio", 0.0)),
        "avg_sustained_coverage_ratio": float(summary.get("avg_sustained_coverage_ratio", 0.0)),
        "avg_position_span_ratio": float(summary.get("avg_position_span_ratio", 0.0)),
        "max_longest_sustained_empty_run_steps": int(
            summary.get("max_longest_sustained_empty_run_steps", 0) or 0
        ),
        "avg_dead_air_ratio": (
            float(sum(dead_air_ratios) / len(dead_air_ratios)) if dead_air_ratios else 0.0
        ),
        "max_dead_air_ratio": max(dead_air_ratios) if dead_air_ratios else 0.0,
        "avg_chord_tone_ratio": (
            float(sum(chord_tone_ratios) / len(chord_tone_ratios)) if chord_tone_ratios else 0.0
        ),
        "failure_reasons": summary.get("failure_reasons", {}),
        "diagnostic_failure_reasons": summary.get("diagnostic_failure_reasons", {}),
        "strict_failure_reasons": summary.get("strict_failure_reasons", {}),
        "report_path": str(Path(report["run_dir"]) / "report.json"),
    }


def build_sweep_summary(
    rows: list[dict[str, Any]],
    min_best_valid_samples: int = 1,
    min_best_strict_valid_samples: int = 1,
    max_collapse_warning_sample_rate: float = 0.34,
) -> dict[str, Any]:
    best_row = max(
        rows,
        key=lambda row: (
            row["strict_valid_sample_count"],
            row["valid_sample_count"],
            -row["collapse_warning_sample_rate"],
            row["valid_sample_rate"],
        ),
        default=None,
    )
    passed_basic = bool(best_row and int(best_row["valid_sample_count"]) >= int(min_best_valid_samples))
    passed_strict = bool(
        best_row
        and int(best_row["strict_valid_sample_count"]) >= int(min_best_strict_valid_samples)
        and float(best_row["collapse_warning_sample_rate"]) <= float(max_collapse_warning_sample_rate)
    )
    return {
        "config_count": int(len(rows)),
        "best_valid_sample_count": int(best_row["valid_sample_count"]) if best_row else 0,
        "best_valid_sample_rate": float(best_row["valid_sample_rate"]) if best_row else 0.0,
        "best_strict_valid_sample_count": int(best_row["strict_valid_sample_count"]) if best_row else 0,
        "best_strict_valid_sample_rate": float(best_row["strict_valid_sample_rate"]) if best_row else 0.0,
        "best_config": {
            "top_k": int(best_row["top_k"]),
            "temperature": float(best_row["temperature"]),
            "run_id": best_row["run_id"],
        }
        if best_row
        else None,
        "min_best_valid_samples": int(min_best_valid_samples),
        "min_best_strict_valid_samples": int(min_best_strict_valid_samples),
        "max_collapse_warning_sample_rate": float(max_collapse_warning_sample_rate),
        "passed_basic_sweep_gate": passed_basic,
        "passed_strict_sweep_gate": passed_strict,
        "passed_sweep_gate": passed_strict,
    }


def markdown_table(rows: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    lines = [
        "# Stage B Sampling Sweep",
        "",
        f"- passed sweep gate: `{str(summary['passed_sweep_gate']).lower()}`",
        f"- passed basic sweep gate: `{str(summary['passed_basic_sweep_gate']).lower()}`",
        f"- passed strict sweep gate: `{str(summary['passed_strict_sweep_gate']).lower()}`",
        f"- best config: `{summary['best_config']}`",
        "",
        "| top_k | temp | samples | grammar | valid | strict_valid | valid_rate | strict_rate | collapse_rate | strict_pass | avg_pair_repeat | max_remove |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {top_k} | {temperature:.3f} | {sample_count} | {grammar_gate_sample_count} | "
            "{valid_sample_count} | {strict_valid_sample_count} | {valid_sample_rate:.3f} | "
            "{strict_valid_sample_rate:.3f} | {collapse_warning_sample_rate:.3f} | "
            "{passed_strict_review_gate} | "
            "{avg_repeated_position_pitch_pair_ratio:.3f} | {max_postprocess_removal_ratio:.3f} |".format(**row)
        )
    lines.append("")
    lines.append("## Diagnostic Failures")
    lines.append("")
    for row in rows:
        lines.append(f"- `top_k={row['top_k']}`, `temperature={row['temperature']}`: {row['diagnostic_failure_reasons']}")
    lines.append("")
    lines.append("## Strict Gate Failures")
    lines.append("")
    for row in rows:
        lines.append(f"- `top_k={row['top_k']}`, `temperature={row['temperature']}`: {row['strict_failure_reasons']}")
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Stage B sampling sweep")
    parser.add_argument("--output_root", type=str, default=str(ROOT_DIR / "outputs" / "stage_b_sampling_sweep"))
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--issue_number", type=int, default=29)
    parser.add_argument("--top_ks", type=str, default="1,2")
    parser.add_argument("--temperatures", type=str, default="0.9")
    parser.add_argument("--train_top_k", type=int, default=2)
    parser.add_argument("--max_files", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_sequence", type=int, default=96)
    parser.add_argument("--num_samples", type=int, default=3)
    parser.add_argument("--constrained_note_groups_per_bar", type=int, default=4)
    parser.add_argument("--max_simultaneous_notes", type=int, default=2)
    parser.add_argument("--min_valid_samples", type=int, default=1)
    parser.add_argument("--min_best_valid_samples", type=int, default=1)
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
    top_ks = parse_int_list(args.top_ks)
    temperatures = parse_float_list(args.temperatures)
    if not top_ks or not temperatures:
        raise ValueError("--top_ks and --temperatures must not be empty")

    train_temperature = temperatures[0]
    train_config = (int(args.train_top_k), float(train_temperature))
    command_results: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    checkpoint_dir: Path | None = None
    completed_configs: set[tuple[int, float]] = set()

    for top_k in [args.train_top_k] + [value for value in top_ks if value != args.train_top_k]:
        for temperature in temperatures:
            config = (int(top_k), float(temperature))
            config_run_id = f"{run_id}_k{top_k}_t{suffix_for_float(temperature)}"
            skip_prepare_train = config != train_config
            cmd = probe_command(
                args,
                run_id=config_run_id,
                top_k=top_k,
                temperature=temperature,
                checkpoint_dir=checkpoint_dir,
                skip_prepare_train=skip_prepare_train,
            )
            result = run_command(cmd)
            command_results.append(result)
            if result["returncode"] != 0:
                report = {
                    "run_id": run_id,
                    "failure_reason": f"probe command failed for top_k={top_k}, temperature={temperature}",
                    "command_results": command_results,
                }
                write_json(sweep_dir / "sweep_report.json", report)
                print(json.dumps(report, ensure_ascii=True, indent=2))
                return int(result["returncode"])

            report_path = Path(args.output_root) / config_run_id / "report.json"
            probe_report = read_json(report_path)
            if checkpoint_dir is None:
                checkpoint_dir = Path(probe_report["checkpoint_dir"])
            if config not in completed_configs:
                rows.append(row_from_probe_report(top_k, temperature, config_run_id, probe_report))
                completed_configs.add(config)

    rows = sorted(rows, key=lambda row: (row["temperature"], row["top_k"]))
    summary = build_sweep_summary(
        rows,
        min_best_valid_samples=args.min_best_valid_samples,
        min_best_strict_valid_samples=args.min_best_strict_valid_samples,
        max_collapse_warning_sample_rate=args.max_collapse_warning_sample_rate,
    )
    report = {
        "run_id": run_id,
        "run_dir": str(sweep_dir),
        "issue": int(args.issue_number),
        "top_ks": top_ks,
        "temperatures": temperatures,
        "summary": summary,
        "rows": rows,
        "command_results": command_results,
    }
    write_json(sweep_dir / "sweep_report.json", report)
    (sweep_dir / "sweep_report.md").write_text(markdown_table(rows, summary), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True, indent=2))
    return 0 if summary["passed_sweep_gate"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
