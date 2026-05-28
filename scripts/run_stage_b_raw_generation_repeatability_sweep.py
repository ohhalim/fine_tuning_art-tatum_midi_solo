"""Run a broader Stage B raw generation repeatability sweep."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence


ROOT_DIR = Path(__file__).resolve().parents[1]
PROBE_SCRIPT = ROOT_DIR / "scripts" / "run_stage_b_generation_probe.py"


def parse_seed_values(raw: str) -> list[int]:
    seeds = [int(value.strip()) for value in raw.split(",") if value.strip()]
    if not seeds:
        raise ValueError("--seeds must include at least one seed")
    return seeds


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_command(cmd: list[str]) -> dict[str, Any]:
    completed = subprocess.run(cmd, cwd=ROOT_DIR, text=True, capture_output=True, check=False)
    return {
        "cmd": cmd,
        "returncode": int(completed.returncode),
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
    }


def probe_run_id(base_run_id: str, seed: int, max_files: int) -> str:
    return f"{base_run_id}_seed{int(seed)}_files{int(max_files)}"


def probe_command(args: argparse.Namespace, run_id: str, seed: int) -> list[str]:
    return [
        sys.executable,
        str(PROBE_SCRIPT),
        "--run_id",
        run_id,
        "--output_root",
        str(Path(args.probe_output_root)),
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
        "--seed",
        str(seed),
        "--temperature",
        str(args.temperature),
        "--top_k",
        str(args.top_k),
        "--postprocess_overlap",
        "--max_simultaneous_notes",
        str(args.max_simultaneous_notes),
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


def _sample_metric_values(report: dict[str, Any], key: str) -> list[float]:
    values: list[float] = []
    for sample in report.get("samples", []):
        if not isinstance(sample, dict):
            continue
        value = sample.get("metrics", {}).get(key)
        if value is not None:
            values.append(float(value))
    return values


def _grammar_values(report: dict[str, Any], key: str) -> list[int]:
    values: list[int] = []
    for sample in report.get("samples", []):
        if not isinstance(sample, dict):
            continue
        value = sample.get("grammar", {}).get(key)
        if value is not None:
            values.append(int(value))
    return values


def _collapse_values(report: dict[str, Any], key: str) -> list[float]:
    values: list[float] = []
    for sample in report.get("samples", []):
        if not isinstance(sample, dict):
            continue
        value = sample.get("collapse", {}).get(key)
        if value is not None:
            values.append(float(value))
    return values


def _range(values: Sequence[float | int]) -> dict[str, float | int | None]:
    if not values:
        return {"min": None, "max": None}
    return {"min": min(values), "max": max(values)}


def range_label(value: Any, digits: int = 3) -> str:
    if not isinstance(value, dict):
        return "-"
    minimum = value.get("min")
    maximum = value.get("max")
    if minimum is None or maximum is None:
        return "-"
    if isinstance(minimum, float) or isinstance(maximum, float):
        return f"{float(minimum):.{digits}f}-{float(maximum):.{digits}f}"
    return f"{minimum}-{maximum}"


def reason_label(value: Any) -> str:
    if not isinstance(value, dict) or not value:
        return "none"
    return ", ".join(f"{reason} ({count})" for reason, count in sorted(value.items()))


def row_from_report(
    seed: int,
    run_id: str,
    report_path: Path,
    report: dict[str, Any] | None,
    command_result: dict[str, Any],
) -> dict[str, Any]:
    if report is None:
        return {
            "seed": int(seed),
            "run_id": run_id,
            "report_path": str(report_path),
            "returncode": int(command_result["returncode"]),
            "sample_count": 0,
            "valid_sample_count": 0,
            "strict_valid_sample_count": 0,
            "grammar_gate_sample_count": 0,
            "passed_strict_review_gate": False,
            "failure_reasons": {"missing_report": 1},
            "strict_failure_reasons": {"missing_report": 1},
            "diagnostic_failure_reasons": {"missing_report": 1},
        }

    summary = report.get("summary", {})
    note_counts = _sample_metric_values(report, "note_count")
    unique_pitch_counts = _sample_metric_values(report, "unique_pitch_count")
    max_simultaneous_values = _sample_metric_values(report, "max_simultaneous_notes")
    phrase_coverage_values = _sample_metric_values(report, "phrase_coverage_ratio")
    complete_note_groups = _grammar_values(report, "complete_note_groups")
    invalid_token_counts = _grammar_values(report, "invalid_token_count")
    postprocess_removal_ratios = _collapse_values(report, "postprocess_removal_ratio")

    return {
        "seed": int(seed),
        "run_id": run_id,
        "report_path": str(report_path),
        "returncode": int(command_result["returncode"]),
        "input_file_count": int(report.get("dataset_summary", {}).get("input_file_count", 0) or 0),
        "train_samples": int(report.get("dataset_summary", {}).get("train_samples", 0) or 0),
        "val_samples": int(report.get("dataset_summary", {}).get("val_samples", 0) or 0),
        "sample_count": int(summary.get("sample_count", 0) or 0),
        "valid_sample_count": int(summary.get("valid_sample_count", 0) or 0),
        "strict_valid_sample_count": int(summary.get("strict_valid_sample_count", 0) or 0),
        "grammar_gate_sample_count": int(summary.get("grammar_gate_sample_count", 0) or 0),
        "valid_sample_rate": float(summary.get("valid_sample_rate", 0.0) or 0.0),
        "strict_valid_sample_rate": float(summary.get("strict_valid_sample_rate", 0.0) or 0.0),
        "grammar_gate_sample_rate": float(summary.get("grammar_gate_sample_rate", 0.0) or 0.0),
        "passed_strict_review_gate": bool(summary.get("passed_strict_review_gate", False)),
        "collapse_warning_sample_count": int(summary.get("collapse_warning_sample_count", 0) or 0),
        "collapse_warning_sample_rate": float(summary.get("collapse_warning_sample_rate", 0.0) or 0.0),
        "avg_postprocess_removal_ratio": float(summary.get("avg_postprocess_removal_ratio", 0.0) or 0.0),
        "max_postprocess_removal_ratio": float(summary.get("max_postprocess_removal_ratio", 0.0) or 0.0),
        "avg_repeated_position_pitch_pair_ratio": float(
            summary.get("avg_repeated_position_pitch_pair_ratio", 0.0) or 0.0
        ),
        "max_repeated_position_pitch_pair_ratio": float(
            summary.get("max_repeated_position_pitch_pair_ratio", 0.0) or 0.0
        ),
        "note_count_range": _range(note_counts),
        "unique_pitch_count_range": _range(unique_pitch_counts),
        "max_simultaneous_notes_range": _range(max_simultaneous_values),
        "phrase_coverage_ratio_range": _range(phrase_coverage_values),
        "complete_note_group_range": _range(complete_note_groups),
        "invalid_token_count_range": _range(invalid_token_counts),
        "postprocess_removal_ratio_range": _range(postprocess_removal_ratios),
        "failure_reasons": summary.get("failure_reasons", {}),
        "diagnostic_failure_reasons": summary.get("diagnostic_failure_reasons", {}),
        "strict_failure_reasons": summary.get("strict_failure_reasons", {}),
    }


def build_repeatability_summary(
    rows: Sequence[dict[str, Any]],
    min_seed_count: int,
    min_source_files: int,
    min_strict_samples_per_seed: int,
    min_overall_strict_rate: float,
    max_postprocess_removal_ratio: float,
) -> dict[str, Any]:
    total_samples = sum(int(row.get("sample_count", 0) or 0) for row in rows)
    total_valid = sum(int(row.get("valid_sample_count", 0) or 0) for row in rows)
    total_strict = sum(int(row.get("strict_valid_sample_count", 0) or 0) for row in rows)
    total_grammar = sum(int(row.get("grammar_gate_sample_count", 0) or 0) for row in rows)
    seed_values = [int(row["seed"]) for row in rows]
    input_file_counts = [int(row.get("input_file_count", 0) or 0) for row in rows]
    max_removal = max((float(row.get("max_postprocess_removal_ratio", 0.0) or 0.0) for row in rows), default=0.0)
    seeds_with_strict = [
        int(row["seed"])
        for row in rows
        if int(row.get("strict_valid_sample_count", 0) or 0) >= int(min_strict_samples_per_seed)
    ]
    failing_rows = [
        row
        for row in rows
        if int(row.get("strict_valid_sample_count", 0) or 0) < int(min_strict_samples_per_seed)
        or int(row.get("returncode", 0) or 0) != 0
    ]
    strict_rate = float(total_strict / total_samples) if total_samples else 0.0
    valid_rate = float(total_valid / total_samples) if total_samples else 0.0
    grammar_rate = float(total_grammar / total_samples) if total_samples else 0.0
    min_observed_input_files = min(input_file_counts) if input_file_counts else 0

    passed = bool(
        len(rows) >= int(min_seed_count)
        and min_observed_input_files >= int(min_source_files)
        and len(seeds_with_strict) == len(rows)
        and strict_rate >= float(min_overall_strict_rate)
        and max_removal <= float(max_postprocess_removal_ratio)
        and not any(int(row.get("returncode", 0) or 0) != 0 for row in rows)
    )

    return {
        "seed_count": int(len(rows)),
        "seed_values": seed_values,
        "min_seed_count": int(min_seed_count),
        "min_source_files": int(min_source_files),
        "min_observed_input_files": int(min_observed_input_files),
        "total_samples": int(total_samples),
        "total_valid_sample_count": int(total_valid),
        "total_strict_valid_sample_count": int(total_strict),
        "total_grammar_gate_sample_count": int(total_grammar),
        "valid_sample_rate": valid_rate,
        "strict_valid_sample_rate": strict_rate,
        "grammar_gate_sample_rate": grammar_rate,
        "min_strict_samples_per_seed": int(min_strict_samples_per_seed),
        "seeds_with_strict_sample": seeds_with_strict,
        "min_overall_strict_rate": float(min_overall_strict_rate),
        "max_postprocess_removal_ratio": float(max_removal),
        "allowed_max_postprocess_removal_ratio": float(max_postprocess_removal_ratio),
        "failing_seeds": [int(row["seed"]) for row in failing_rows],
        "passed_repeatability_gate": passed,
    }


def markdown_report(rows: Sequence[dict[str, Any]], summary: dict[str, Any]) -> str:
    lines = [
        "# Stage B Raw Generation Repeatability Sweep",
        "",
        f"- passed repeatability gate: `{str(summary['passed_repeatability_gate']).lower()}`",
        f"- seeds: `{summary['seed_values']}`",
        f"- source files per run: `{summary['min_observed_input_files']}`",
        f"- total samples: `{summary['total_samples']}`",
        f"- strict pass-rate: `{summary['strict_valid_sample_rate']:.3f}`",
        f"- grammar pass-rate: `{summary['grammar_gate_sample_rate']:.3f}`",
        f"- max postprocess removal ratio: `{summary['max_postprocess_removal_ratio']:.3f}`",
        "",
        "| seed | files | samples | grammar | valid | strict | notes | pitches | max simultaneous | phrase coverage | max removal | strict gate |",
        "|---:|---:|---:|---:|---:|---:|---|---|---|---|---:|:---:|",
    ]
    for row in rows:
        lines.append(
            "| {seed} | {input_file_count} | {sample_count} | {grammar_gate_sample_count} | "
            "{valid_sample_count} | {strict_valid_sample_count} | {note_range} | "
            "{pitch_range} | {simul_range} | {coverage_range} | {max_removal:.3f} | "
            "{strict_gate} |".format(
                seed=row["seed"],
                input_file_count=row.get("input_file_count", 0),
                sample_count=row.get("sample_count", 0),
                grammar_gate_sample_count=row.get("grammar_gate_sample_count", 0),
                valid_sample_count=row.get("valid_sample_count", 0),
                strict_valid_sample_count=row.get("strict_valid_sample_count", 0),
                note_range=range_label(row.get("note_count_range"), digits=0),
                pitch_range=range_label(row.get("unique_pitch_count_range"), digits=0),
                simul_range=range_label(row.get("max_simultaneous_notes_range"), digits=0),
                coverage_range=range_label(row.get("phrase_coverage_ratio_range"), digits=3),
                max_removal=float(row.get("max_postprocess_removal_ratio", 0.0) or 0.0),
                strict_gate=bool(row.get("passed_strict_review_gate", False)),
            )
        )
    lines.extend(["", "## Failure Reasons", ""])
    for row in rows:
        lines.append(
            f"- seed `{row['seed']}`: failure `{reason_label(row.get('failure_reasons'))}`, "
            f"strict `{reason_label(row.get('strict_failure_reasons'))}`"
        )
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Stage B raw generation repeatability sweep")
    parser.add_argument("--output_root", type=str, default=str(ROOT_DIR / "outputs" / "stage_b_raw_generation_repeatability"))
    parser.add_argument("--probe_output_root", type=str, default=str(ROOT_DIR / "outputs" / "stage_b_generation_probe"))
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--issue_number", type=int, default=224)
    parser.add_argument("--seeds", type=str, default="17,23,31")
    parser.add_argument("--max_files", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_sequence", type=int, default=96)
    parser.add_argument("--num_samples", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=4)
    parser.add_argument("--max_simultaneous_notes", type=int, default=2)
    parser.add_argument("--min_valid_samples", type=int, default=1)
    parser.add_argument("--min_strict_valid_samples", type=int, default=1)
    parser.add_argument("--min_seed_count", type=int, default=3)
    parser.add_argument("--min_source_files", type=int, default=2)
    parser.add_argument("--min_strict_samples_per_seed", type=int, default=1)
    parser.add_argument("--min_overall_strict_rate", type=float, default=0.67)
    parser.add_argument("--max_allowed_postprocess_removal_ratio", type=float, default=0.49)
    parser.add_argument("--max_collapse_warning_sample_rate", type=float, default=0.34)
    parser.add_argument("--strict_min_unique_pitches", type=int, default=3)
    parser.add_argument("--strict_min_unique_positions", type=int, default=3)
    parser.add_argument("--strict_min_unique_position_pitch_pairs", type=int, default=4)
    parser.add_argument("--strict_max_repeated_position_pitch_pair_ratio", type=float, default=0.49)
    parser.add_argument("--strict_max_postprocess_removal_ratio", type=float, default=0.49)
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
    seeds = parse_seed_values(args.seeds)
    rows: list[dict[str, Any]] = []
    command_results: list[dict[str, Any]] = []

    for seed in seeds:
        child_run_id = probe_run_id(run_id, seed=seed, max_files=args.max_files)
        cmd = probe_command(args, run_id=child_run_id, seed=seed)
        command_result = run_command(cmd)
        command_results.append(command_result)
        report_path = Path(args.probe_output_root) / child_run_id / "report.json"
        report = read_json(report_path) if report_path.exists() else None
        rows.append(
            row_from_report(
                seed=seed,
                run_id=child_run_id,
                report_path=report_path,
                report=report,
                command_result=command_result,
            )
        )

    summary = build_repeatability_summary(
        rows,
        min_seed_count=args.min_seed_count,
        min_source_files=args.min_source_files,
        min_strict_samples_per_seed=args.min_strict_samples_per_seed,
        min_overall_strict_rate=args.min_overall_strict_rate,
        max_postprocess_removal_ratio=args.max_allowed_postprocess_removal_ratio,
    )
    report = {
        "run_id": run_id,
        "issue": int(args.issue_number),
        "config": {
            "seeds": seeds,
            "max_files": int(args.max_files),
            "epochs": int(args.epochs),
            "num_samples": int(args.num_samples),
            "temperature": float(args.temperature),
            "top_k": int(args.top_k),
            "postprocess_overlap": True,
        },
        "summary": summary,
        "rows": rows,
        "command_results": command_results,
    }
    write_json(sweep_dir / "repeatability_summary.json", report)
    (sweep_dir / "repeatability_summary.md").write_text(
        markdown_report(rows, summary),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=True, indent=2))
    return 0 if summary["passed_repeatability_gate"] else 5


if __name__ == "__main__":
    raise SystemExit(main())
