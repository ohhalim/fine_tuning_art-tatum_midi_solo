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


def sample_float(sample: dict[str, Any], section: str, key: str, default: float = 0.0) -> float:
    value = sample.get(section, {}).get(key, default)
    return float(value if value is not None else default)


def sample_int(sample: dict[str, Any], section: str, key: str, default: int = 0) -> int:
    value = sample.get(section, {}).get(key, default)
    return int(value if value is not None else default)


def candidate_summary(seed: int, run_id: str, sample: dict[str, Any]) -> dict[str, Any]:
    return {
        "seed": int(seed),
        "run_id": run_id,
        "sample_index": int(sample.get("sample_index", 0) or 0),
        "midi_path": str(sample.get("midi_path", "")),
        "dead_air_ratio": sample_float(sample, "metrics", "dead_air_ratio"),
        "note_count": sample_int(sample, "metrics", "note_count"),
        "unique_pitch_count": sample_int(sample, "metrics", "unique_pitch_count"),
        "phrase_coverage_ratio": sample_float(sample, "metrics", "phrase_coverage_ratio"),
        "onset_coverage_ratio": sample_float(sample, "temporal_coverage", "onset_coverage_ratio"),
        "sustained_coverage_ratio": sample_float(sample, "temporal_coverage", "sustained_coverage_ratio"),
        "postprocess_removal_ratio": sample_float(sample, "collapse", "postprocess_removal_ratio"),
    }


def best_strict_candidate(seed: int, run_id: str, samples: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    strict_candidates = [sample for sample in samples if bool(sample.get("strict_valid", False))]
    if not strict_candidates:
        return None
    best = sorted(
        strict_candidates,
        key=lambda sample: (
            sample_float(sample, "metrics", "dead_air_ratio"),
            sample_float(sample, "collapse", "postprocess_removal_ratio"),
            -sample_int(sample, "metrics", "note_count"),
            int(sample.get("sample_index", 0) or 0),
        ),
    )[0]
    return candidate_summary(seed, run_id, best)


def row_from_report(
    seed: int,
    run_id: str,
    report_path: Path,
    report: dict[str, Any] | None,
    command_result: dict[str, Any],
    dead_air_gate: float,
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
            "dead_air_outlier_count": 0,
            "best_strict_candidate": None,
            "failure_reasons": {"missing_report": 1},
            "strict_failure_reasons": {"missing_report": 1},
            "diagnostic_failure_reasons": {"missing_report": 1},
        }

    summary = report.get("summary", {})
    samples = [sample for sample in report.get("samples", []) if isinstance(sample, dict)]
    note_counts = _sample_metric_values(report, "note_count")
    unique_pitch_counts = _sample_metric_values(report, "unique_pitch_count")
    max_simultaneous_values = _sample_metric_values(report, "max_simultaneous_notes")
    phrase_coverage_values = _sample_metric_values(report, "phrase_coverage_ratio")
    dead_air_ratios = _sample_metric_values(report, "dead_air_ratio")
    complete_note_groups = _grammar_values(report, "complete_note_groups")
    invalid_token_counts = _grammar_values(report, "invalid_token_count")
    postprocess_removal_ratios = _collapse_values(report, "postprocess_removal_ratio")
    dead_air_outlier_count = sum(1 for value in dead_air_ratios if float(value) >= float(dead_air_gate))

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
        "dead_air_ratio_range": _range(dead_air_ratios),
        "dead_air_outlier_count": int(dead_air_outlier_count),
        "dead_air_gate": float(dead_air_gate),
        "best_strict_candidate": best_strict_candidate(seed, run_id, samples),
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
    max_dead_air_outlier_rate: float,
    warning_min_strict_samples_per_seed: int = 2,
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
    total_dead_air_outliers = sum(int(row.get("dead_air_outlier_count", 0) or 0) for row in rows)
    dead_air_outlier_rate = float(total_dead_air_outliers / total_samples) if total_samples else 0.0
    seed_best_candidates = [
        row.get("best_strict_candidate")
        for row in rows
        if isinstance(row.get("best_strict_candidate"), dict)
    ]
    strict_margin_warning_rows = [
        {
            "seed": int(row["seed"]),
            "sample_count": int(row.get("sample_count", 0) or 0),
            "strict_valid_sample_count": int(row.get("strict_valid_sample_count", 0) or 0),
            "warning_min_strict_samples_per_seed": int(warning_min_strict_samples_per_seed),
            "strict_margin": int(
                int(row.get("strict_valid_sample_count", 0) or 0) - int(warning_min_strict_samples_per_seed)
            ),
        }
        for row in rows
        if int(row.get("strict_valid_sample_count", 0) or 0) < int(warning_min_strict_samples_per_seed)
    ]
    selected_best_candidate = (
        sorted(
            seed_best_candidates,
            key=lambda candidate: (
                float(candidate.get("dead_air_ratio", 0.0) or 0.0),
                float(candidate.get("postprocess_removal_ratio", 0.0) or 0.0),
                -int(candidate.get("note_count", 0) or 0),
                int(candidate.get("seed", 0) or 0),
                int(candidate.get("sample_index", 0) or 0),
            ),
        )[0]
        if seed_best_candidates
        else None
    )

    passed = bool(
        len(rows) >= int(min_seed_count)
        and min_observed_input_files >= int(min_source_files)
        and len(seeds_with_strict) == len(rows)
        and strict_rate >= float(min_overall_strict_rate)
        and max_removal <= float(max_postprocess_removal_ratio)
        and len(seed_best_candidates) == len(rows)
        and dead_air_outlier_rate <= float(max_dead_air_outlier_rate)
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
        "total_dead_air_outlier_count": int(total_dead_air_outliers),
        "dead_air_outlier_rate": dead_air_outlier_rate,
        "max_dead_air_outlier_rate": float(max_dead_air_outlier_rate),
        "warning_min_strict_samples_per_seed": int(warning_min_strict_samples_per_seed),
        "strict_margin_warning_seed_count": int(len(strict_margin_warning_rows)),
        "strict_margin_warning_seeds": [int(row["seed"]) for row in strict_margin_warning_rows],
        "strict_margin_warning_rows": strict_margin_warning_rows,
        "seed_best_candidates": seed_best_candidates,
        "selected_best_candidate": selected_best_candidate,
        "failing_seeds": [int(row["seed"]) for row in failing_rows],
        "passed_repeatability_gate": passed,
    }


def markdown_report(rows: Sequence[dict[str, Any]], summary: dict[str, Any]) -> str:
    selected = summary.get("selected_best_candidate") if isinstance(summary.get("selected_best_candidate"), dict) else None
    selected_label = (
        f"seed {selected['seed']} sample {selected['sample_index']} dead-air {float(selected['dead_air_ratio']):.3f}"
        if selected
        else "none"
    )
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
        f"- dead-air outlier rate: `{summary['dead_air_outlier_rate']:.3f}`",
        f"- warning min strict per seed: `{summary.get('warning_min_strict_samples_per_seed', 0)}`",
        f"- strict margin warning seeds: `{summary.get('strict_margin_warning_seeds', [])}`",
        f"- selected best candidate: `{selected_label}`",
        "",
        "| seed | files | samples | grammar | valid | strict | margin warning | dead-air outliers | best sample | best dead-air | notes | pitches | phrase coverage | max removal | strict gate |",
        "|---:|---:|---:|---:|---:|---:|:---:|---:|---:|---:|---|---|---|---:|:---:|",
    ]
    warning_seeds = set(int(seed) for seed in summary.get("strict_margin_warning_seeds", []))
    for row in rows:
        best = row.get("best_strict_candidate") if isinstance(row.get("best_strict_candidate"), dict) else None
        lines.append(
            "| {seed} | {input_file_count} | {sample_count} | {grammar_gate_sample_count} | "
            "{valid_sample_count} | {strict_valid_sample_count} | {margin_warning} | {dead_air_outlier_count} | "
            "{best_sample} | {best_dead_air} | {note_range} | {pitch_range} | {coverage_range} | {max_removal:.3f} | "
            "{strict_gate} |".format(
                seed=row["seed"],
                input_file_count=row.get("input_file_count", 0),
                sample_count=row.get("sample_count", 0),
                grammar_gate_sample_count=row.get("grammar_gate_sample_count", 0),
                valid_sample_count=row.get("valid_sample_count", 0),
                strict_valid_sample_count=row.get("strict_valid_sample_count", 0),
                margin_warning=int(row["seed"]) in warning_seeds,
                dead_air_outlier_count=row.get("dead_air_outlier_count", 0),
                best_sample=best.get("sample_index", "-") if best else "-",
                best_dead_air=f"{float(best.get('dead_air_ratio', 0.0)):.3f}" if best else "-",
                note_range=range_label(row.get("note_count_range"), digits=0),
                pitch_range=range_label(row.get("unique_pitch_count_range"), digits=0),
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
    parser.add_argument("--issue_number", type=int, default=0)
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
    parser.add_argument("--warning_min_strict_samples_per_seed", type=int, default=2)
    parser.add_argument("--max_allowed_postprocess_removal_ratio", type=float, default=0.49)
    parser.add_argument("--dead_air_gate", type=float, default=0.8)
    parser.add_argument("--max_dead_air_outlier_rate", type=float, default=0.25)
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
                dead_air_gate=args.dead_air_gate,
            )
        )

    summary = build_repeatability_summary(
        rows,
        min_seed_count=args.min_seed_count,
        min_source_files=args.min_source_files,
        min_strict_samples_per_seed=args.min_strict_samples_per_seed,
        min_overall_strict_rate=args.min_overall_strict_rate,
        max_postprocess_removal_ratio=args.max_allowed_postprocess_removal_ratio,
        max_dead_air_outlier_rate=args.max_dead_air_outlier_rate,
        warning_min_strict_samples_per_seed=args.warning_min_strict_samples_per_seed,
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
            "dead_air_gate": float(args.dead_air_gate),
            "max_dead_air_outlier_rate": float(args.max_dead_air_outlier_rate),
            "warning_min_strict_samples_per_seed": int(args.warning_min_strict_samples_per_seed),
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
