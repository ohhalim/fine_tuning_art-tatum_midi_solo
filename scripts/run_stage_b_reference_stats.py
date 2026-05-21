"""Build reference phrase statistics from real Stage B MIDI windows."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from scripts.run_stage_b_generation_probe import (  # noqa: E402
    analyze_stage_b_collapse,
    analyze_stage_b_phrase_contour,
    analyze_stage_b_rhythm_profile,
    analyze_stage_b_temporal_coverage,
    extract_stage_b_note_groups,
)


REFERENCE_METRIC_KEYS = [
    "note_group_count",
    "unique_pitch_count",
    "pitch_span",
    "repeated_pitch_ratio",
    "syncopated_onset_ratio",
    "unique_bar_position_pattern_ratio",
    "duration_diversity_ratio",
    "most_common_duration_ratio",
    "ioi_diversity_ratio",
    "most_common_ioi_ratio",
    "direction_change_ratio",
    "stepwise_motion_ratio",
    "leap_motion_ratio",
    "onset_coverage_ratio",
    "sustained_coverage_ratio",
]


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


def run_prepare_command(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    cmd = [
        sys.executable,
        "scripts/prepare_role_dataset.py",
        "--input_dir",
        str(args.input_dir),
        "--output_dir",
        str(output_dir),
        "--role",
        str(args.role),
        "--sequence_format",
        "stage_b_v1",
        "--stage_b_window_bars",
        str(args.window_bars),
        "--stage_b_window_stride_bars",
        str(args.window_stride_bars),
        "--stage_b_min_window_target_notes",
        str(args.min_window_target_notes),
        "--overwrite",
    ]
    if args.max_files is not None:
        cmd.extend(["--max_files", str(args.max_files)])
    return run_command(cmd)


def percentile(values: list[float], ratio: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * float(ratio)))))
    return float(ordered[index])


def compact_metric_stats(values: Iterable[float]) -> dict[str, float]:
    vals = [float(value) for value in values]
    if not vals:
        return {"count": 0.0, "mean": 0.0, "min": 0.0, "p25": 0.0, "p50": 0.0, "p75": 0.0, "max": 0.0}
    return {
        "count": float(len(vals)),
        "mean": float(mean(vals)),
        "min": float(min(vals)),
        "p25": percentile(vals, 0.25),
        "p50": percentile(vals, 0.50),
        "p75": percentile(vals, 0.75),
        "max": float(max(vals)),
    }


def analyze_token_record(tokens: list[int], bars: int) -> dict[str, Any]:
    groups = extract_stage_b_note_groups(tokens, primer_size=0)
    collapse = analyze_stage_b_collapse(tokens, primer_size=0)
    rhythm = analyze_stage_b_rhythm_profile(tokens, primer_size=0)
    contour = analyze_stage_b_phrase_contour(tokens, primer_size=0)
    temporal = analyze_stage_b_temporal_coverage(tokens, primer_size=0, bars=bars)
    pitches = [int(group["pitch"]) for group in groups]

    metrics = {
        "note_group_count": float(len(groups)),
        "unique_pitch_count": float(len(set(pitches))),
        "pitch_span": float(max(pitches) - min(pitches)) if pitches else 0.0,
        "repeated_pitch_ratio": float(collapse.get("repeated_pitch_ratio", 0.0) or 0.0),
        "syncopated_onset_ratio": float(rhythm.get("syncopated_onset_ratio", 0.0) or 0.0),
        "unique_bar_position_pattern_ratio": float(
            rhythm.get("unique_bar_position_pattern_ratio", 0.0) or 0.0
        ),
        "duration_diversity_ratio": float(rhythm.get("duration_diversity_ratio", 0.0) or 0.0),
        "most_common_duration_ratio": float(rhythm.get("most_common_duration_ratio", 0.0) or 0.0),
        "ioi_diversity_ratio": float(rhythm.get("ioi_diversity_ratio", 0.0) or 0.0),
        "most_common_ioi_ratio": float(rhythm.get("most_common_ioi_ratio", 0.0) or 0.0),
        "direction_change_ratio": float(contour.get("direction_change_ratio", 0.0) or 0.0),
        "stepwise_motion_ratio": float(contour.get("stepwise_motion_ratio", 0.0) or 0.0),
        "leap_motion_ratio": float(contour.get("leap_motion_ratio", 0.0) or 0.0),
        "onset_coverage_ratio": float(temporal.get("onset_coverage_ratio", 0.0) or 0.0),
        "sustained_coverage_ratio": float(temporal.get("sustained_coverage_ratio", 0.0) or 0.0),
    }
    return {
        "metrics": metrics,
        "rhythm_profile": rhythm,
        "phrase_contour": contour,
        "temporal_coverage": temporal,
        "collapse": collapse,
    }


def iter_token_files(tokenized_dir: Path) -> list[Path]:
    return sorted(tokenized_dir.glob("*/*.npy"))


def build_reference_rows(tokenized_dir: Path, bars: int, max_records: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, path in enumerate(iter_token_files(tokenized_dir), start=1):
        if max_records is not None and len(rows) >= int(max_records):
            break
        tokens = [int(token) for token in np.load(path).tolist()]
        if not tokens:
            continue
        analysis = analyze_token_record(tokens, bars=bars)
        rows.append(
            {
                "record_index": int(index),
                "relative_path": str(path.relative_to(tokenized_dir)),
                **analysis,
            }
        )
    return rows


def summarize_reference_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "record_count": int(len(rows)),
        "metric_keys": REFERENCE_METRIC_KEYS,
        "metrics": {
            key: compact_metric_stats(
                row.get("metrics", {})[key]
                for row in rows
                if key in row.get("metrics", {})
            )
            for key in REFERENCE_METRIC_KEYS
        },
    }


def generated_rows_from_report(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in report.get("rows", []):
        rows.append(
            {
                "label": str(row.get("grammar_mode") or row.get("run_id") or "generated"),
                "metrics": {
                    "syncopated_onset_ratio": float(row.get("avg_syncopated_onset_ratio", 0.0) or 0.0),
                    "unique_bar_position_pattern_ratio": float(
                        row.get("avg_unique_bar_position_pattern_ratio", 0.0) or 0.0
                    ),
                    "duration_diversity_ratio": float(row.get("avg_duration_diversity_ratio", 0.0) or 0.0),
                    "most_common_duration_ratio": float(row.get("avg_most_common_duration_ratio", 0.0) or 0.0),
                    "ioi_diversity_ratio": float(row.get("avg_ioi_diversity_ratio", 0.0) or 0.0),
                    "most_common_ioi_ratio": float(row.get("avg_most_common_ioi_ratio", 0.0) or 0.0),
                },
            }
        )
    return rows


def compare_generated_to_reference(
    generated_rows: list[dict[str, Any]],
    reference_summary: dict[str, Any],
) -> list[dict[str, Any]]:
    reference_metrics = reference_summary.get("metrics", {})
    comparisons: list[dict[str, Any]] = []
    for row in generated_rows:
        metrics = row.get("metrics", {})
        deltas: dict[str, float] = {}
        for key, value in metrics.items():
            if key not in reference_metrics:
                continue
            reference_mean = float(reference_metrics[key].get("mean", 0.0) or 0.0)
            deltas[key] = float(value) - reference_mean
        comparisons.append({"label": row.get("label"), "metrics": metrics, "delta_from_reference_mean": deltas})
    return comparisons


def markdown_report(report: dict[str, Any]) -> str:
    summary = report["reference_summary"]
    lines = [
        "# Stage B Reference Phrase Statistics",
        "",
        f"- record count: `{summary['record_count']}`",
        f"- input dir: `{report.get('input_dir')}`",
        f"- tokenized dir: `{report.get('tokenized_dir')}`",
        "",
        "| metric | mean | p25 | p50 | p75 | min | max |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key in REFERENCE_METRIC_KEYS:
        stats = summary["metrics"][key]
        lines.append(
            "| {key} | {mean:.3f} | {p25:.3f} | {p50:.3f} | {p75:.3f} | {min:.3f} | {max:.3f} |".format(
                key=key,
                **stats,
            )
        )
    if report.get("generated_comparison"):
        lines.extend(["", "## Generated Comparison", ""])
        for row in report["generated_comparison"]:
            lines.append(f"### {row['label']}")
            for key, delta in row["delta_from_reference_mean"].items():
                lines.append(f"- `{key}` delta from reference mean: `{delta:.3f}`")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Stage B real phrase reference statistics")
    parser.add_argument("--input_dir", type=str, default="./midi_dataset/midi/studio")
    parser.add_argument("--output_root", type=str, default=str(ROOT_DIR / "outputs" / "stage_b_reference_stats"))
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--role", type=str, default="lead")
    parser.add_argument("--max_files", type=int, default=4)
    parser.add_argument("--window_bars", type=int, default=8)
    parser.add_argument("--window_stride_bars", type=int, default=4)
    parser.add_argument("--min_window_target_notes", type=int, default=16)
    parser.add_argument("--max_records", type=int, default=64)
    parser.add_argument("--skip_prepare", action="store_true")
    parser.add_argument("--tokenized_dir", type=str, default=None)
    parser.add_argument("--generated_report", type=str, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_root) / run_id
    roles_dir = run_dir / "roles"
    tokenized_dir = Path(args.tokenized_dir) if args.tokenized_dir else roles_dir / args.role / "tokenized"

    report: dict[str, Any] = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "input_dir": str(args.input_dir),
        "tokenized_dir": str(tokenized_dir),
        "window_bars": int(args.window_bars),
        "window_stride_bars": int(args.window_stride_bars),
        "min_window_target_notes": int(args.min_window_target_notes),
        "max_files": args.max_files,
        "max_records": args.max_records,
    }

    if not args.skip_prepare:
        prepare_result = run_prepare_command(args, roles_dir)
        report["prepare_result"] = prepare_result
        if prepare_result["returncode"] != 0:
            write_json(run_dir / "reference_stats_report.json", report)
            print(json.dumps(report, ensure_ascii=True, indent=2))
            return int(prepare_result["returncode"])

    rows = build_reference_rows(tokenized_dir, bars=int(args.window_bars), max_records=args.max_records)
    report["reference_rows_head"] = rows[:8]
    report["reference_summary"] = summarize_reference_rows(rows)
    if not rows:
        report["failure_reason"] = "No Stage B tokenized records available for reference statistics"
        write_json(run_dir / "reference_stats_report.json", report)
        print(json.dumps(report, ensure_ascii=True, indent=2))
        return 2

    if args.generated_report:
        generated_report_path = Path(args.generated_report)
        if generated_report_path.exists():
            generated_rows = generated_rows_from_report(read_json(generated_report_path))
            report["generated_report"] = str(generated_report_path)
            report["generated_comparison"] = compare_generated_to_reference(
                generated_rows,
                report["reference_summary"],
            )
        else:
            report["generated_report_warning"] = f"missing generated report: {generated_report_path}"

    write_json(run_dir / "reference_stats_report.json", report)
    (run_dir / "reference_stats_report.md").write_text(markdown_report(report), encoding="utf-8")
    print(json.dumps(report["reference_summary"], ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
