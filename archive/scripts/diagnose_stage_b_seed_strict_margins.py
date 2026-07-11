"""Diagnose per-seed strict margin risk from a Stage B repeatability sweep."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence


ROOT_DIR = Path(__file__).resolve().parents[1]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _resolve_path(raw_path: str, source_path: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    source_parent = source_path.parent
    candidate = source_parent / path
    if candidate.exists():
        return candidate
    return ROOT_DIR / path


def _nested_float(source: dict[str, Any], section: str, key: str, default: float = 0.0) -> float:
    value = source.get(section, {}).get(key, default)
    return float(value if value is not None else default)


def _nested_int(source: dict[str, Any], section: str, key: str, default: int = 0) -> int:
    value = source.get(section, {}).get(key, default)
    return int(value if value is not None else default)


def _reason_text(sample: dict[str, Any]) -> str:
    values = [
        sample.get("failure_reason"),
        sample.get("diagnostic_failure_reason"),
    ]
    return " | ".join(str(value) for value in values if value)


def failure_tags(sample: dict[str, Any], dead_air_gate: float, min_unique_pitches: int) -> list[str]:
    tags: list[str] = []
    reason = _reason_text(sample).lower()
    dead_air_ratio = _nested_float(sample, "metrics", "dead_air_ratio")
    unique_pitch_count = _nested_int(sample, "metrics", "unique_pitch_count")
    grammar = sample.get("grammar", {}) if isinstance(sample.get("grammar"), dict) else {}
    postprocess_removal = _nested_float(sample, "collapse", "postprocess_removal_ratio")

    if not bool(sample.get("strict_valid", False)):
        tags.append("strict_invalid")
    if "dead-air" in reason or dead_air_ratio >= float(dead_air_gate):
        tags.append("dead_air")
    if "unique pitch" in reason or unique_pitch_count < int(min_unique_pitches):
        tags.append("unique_pitch")
    if not bool(sample.get("strict_valid", False)) and not bool(grammar.get("grammar_valid", True)):
        tags.append("grammar")
    if "postprocess" in reason:
        tags.append("postprocess")
    if postprocess_removal > 0.49:
        tags.append("postprocess_margin")
    return tags


def compact_sample(
    seed: int,
    sample: dict[str, Any],
    dead_air_gate: float,
    min_unique_pitches: int,
) -> dict[str, Any]:
    tags = failure_tags(sample, dead_air_gate=dead_air_gate, min_unique_pitches=min_unique_pitches)
    return {
        "seed": int(seed),
        "sample_index": int(sample.get("sample_index", 0) or 0),
        "valid": bool(sample.get("valid", False)),
        "strict_valid": bool(sample.get("strict_valid", False)),
        "grammar_valid": bool(sample.get("grammar", {}).get("grammar_valid", False)),
        "failure_reason": sample.get("failure_reason"),
        "diagnostic_failure_reason": sample.get("diagnostic_failure_reason"),
        "failure_tags": tags,
        "note_count": _nested_int(sample, "metrics", "note_count"),
        "unique_pitch_count": _nested_int(sample, "metrics", "unique_pitch_count"),
        "dead_air_ratio": _nested_float(sample, "metrics", "dead_air_ratio"),
        "phrase_coverage_ratio": _nested_float(sample, "metrics", "phrase_coverage_ratio"),
        "onset_coverage_ratio": _nested_float(sample, "temporal_coverage", "onset_coverage_ratio"),
        "sustained_coverage_ratio": _nested_float(sample, "temporal_coverage", "sustained_coverage_ratio"),
        "position_span_ratio": _nested_float(sample, "temporal_coverage", "position_span_ratio"),
        "head_empty_steps": _nested_int(sample, "temporal_coverage", "head_empty_steps"),
        "tail_empty_steps": _nested_int(sample, "temporal_coverage", "tail_empty_steps"),
        "postprocess_removal_ratio": _nested_float(sample, "collapse", "postprocess_removal_ratio"),
    }


def _tag_indices(samples: Sequence[dict[str, Any]], tag: str) -> list[int]:
    return [int(sample["sample_index"]) for sample in samples if tag in sample.get("failure_tags", [])]


def summarize_seed(
    row: dict[str, Any],
    samples: Sequence[dict[str, Any]],
    warning_min_strict_samples_per_seed: int,
) -> dict[str, Any]:
    strict_count = sum(1 for sample in samples if bool(sample.get("strict_valid", False)))
    sample_count = len(samples)
    tag_counts = Counter(tag for sample in samples for tag in sample.get("failure_tags", []))
    dead_air_indices = _tag_indices(samples, "dead_air")
    unique_pitch_indices = _tag_indices(samples, "unique_pitch")
    overlap_indices = [
        int(sample["sample_index"])
        for sample in samples
        if "dead_air" in sample.get("failure_tags", []) and "unique_pitch" in sample.get("failure_tags", [])
    ]
    best = row.get("best_strict_candidate") if isinstance(row.get("best_strict_candidate"), dict) else None
    return {
        "seed": int(row.get("seed", 0) or 0),
        "report_path": str(row.get("report_path", "")),
        "sample_count": int(sample_count),
        "strict_valid_sample_count": int(strict_count),
        "strict_margin": int(strict_count - int(warning_min_strict_samples_per_seed)),
        "margin_warning": bool(strict_count < int(warning_min_strict_samples_per_seed)),
        "dead_air_sample_indices": dead_air_indices,
        "unique_pitch_sample_indices": unique_pitch_indices,
        "dead_air_unique_pitch_overlap_indices": overlap_indices,
        "dead_air_unique_pitch_failures_separate": bool(dead_air_indices and unique_pitch_indices and not overlap_indices),
        "failure_tag_counts": dict(sorted(tag_counts.items())),
        "best_strict_candidate": best,
    }


def load_seed_samples(
    source_summary_path: Path,
    row: dict[str, Any],
    dead_air_gate: float,
    min_unique_pitches: int,
) -> list[dict[str, Any]]:
    report_path = _resolve_path(str(row.get("report_path", "")), source_summary_path)
    report = read_json(report_path)
    seed = int(row.get("seed", report.get("seed", 0)) or 0)
    return [
        compact_sample(seed, sample, dead_air_gate=dead_air_gate, min_unique_pitches=min_unique_pitches)
        for sample in report.get("samples", [])
        if isinstance(sample, dict)
    ]


def build_diagnostic_report(
    source_summary_path: Path,
    warning_min_strict_samples_per_seed: int,
    dead_air_gate: float,
    min_unique_pitches: int,
) -> dict[str, Any]:
    source = read_json(source_summary_path)
    rows = [row for row in source.get("rows", []) if isinstance(row, dict)]
    seed_reports: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []

    for row in rows:
        seed_samples = load_seed_samples(
            source_summary_path,
            row,
            dead_air_gate=dead_air_gate,
            min_unique_pitches=min_unique_pitches,
        )
        sample_rows.extend(seed_samples)
        seed_reports.append(
            summarize_seed(
                row,
                seed_samples,
                warning_min_strict_samples_per_seed=warning_min_strict_samples_per_seed,
            )
        )

    warning_seeds = [int(seed["seed"]) for seed in seed_reports if seed["margin_warning"]]
    separate_failure_seeds = [
        int(seed["seed"]) for seed in seed_reports if seed["dead_air_unique_pitch_failures_separate"]
    ]
    overlap_seeds = [
        int(seed["seed"]) for seed in seed_reports if seed["dead_air_unique_pitch_overlap_indices"]
    ]
    source_summary = source.get("summary", {}) if isinstance(source.get("summary"), dict) else {}

    return {
        "source_summary_path": str(source_summary_path),
        "source_run_id": str(source.get("run_id", "")),
        "source_issue": int(source.get("issue", 0) or 0),
        "hard_min_strict_samples_per_seed": int(source_summary.get("min_strict_samples_per_seed", 1) or 1),
        "warning_min_strict_samples_per_seed": int(warning_min_strict_samples_per_seed),
        "dead_air_gate": float(dead_air_gate),
        "min_unique_pitches": int(min_unique_pitches),
        "summary": {
            "seed_count": int(len(seed_reports)),
            "sample_count": int(len(sample_rows)),
            "margin_warning_seed_count": int(len(warning_seeds)),
            "margin_warning_seeds": warning_seeds,
            "dead_air_unique_pitch_overlap_seed_count": int(len(overlap_seeds)),
            "dead_air_unique_pitch_overlap_seeds": overlap_seeds,
            "dead_air_unique_pitch_separate_seed_count": int(len(separate_failure_seeds)),
            "dead_air_unique_pitch_separate_seeds": separate_failure_seeds,
        },
        "seeds": seed_reports,
        "samples": sorted(sample_rows, key=lambda sample: (int(sample["seed"]), int(sample["sample_index"]))),
    }


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _indices_label(values: Sequence[int]) -> str:
    return ", ".join(str(value) for value in values) if values else "-"


def markdown_report(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Stage B Seed Strict Margin Diagnostics",
        "",
        f"- source summary: `{report['source_summary_path']}`",
        f"- hard min strict per seed: `{report['hard_min_strict_samples_per_seed']}`",
        f"- warning min strict per seed: `{report['warning_min_strict_samples_per_seed']}`",
        f"- strict margin warning seeds: `{summary['margin_warning_seeds']}`",
        f"- dead-air + unique-pitch overlap seeds: `{summary['dead_air_unique_pitch_overlap_seeds']}`",
        f"- dead-air + unique-pitch separate seeds: `{summary['dead_air_unique_pitch_separate_seeds']}`",
        "",
        "| seed | samples | strict | margin warning | dead-air samples | unique-pitch samples | overlap samples | best strict sample | best dead-air | failure tags |",
        "|---:|---:|---:|:---:|---|---|---|---:|---:|---|",
    ]
    for seed in report["seeds"]:
        best = seed.get("best_strict_candidate") if isinstance(seed.get("best_strict_candidate"), dict) else None
        tag_label = ", ".join(
            f"{tag} {count}" for tag, count in seed.get("failure_tag_counts", {}).items()
        ) or "none"
        lines.append(
            "| {seed} | {sample_count} | {strict_valid_sample_count} | {margin_warning} | "
            "{dead_air_samples} | {unique_pitch_samples} | {overlap_samples} | {best_sample} | "
            "{best_dead_air} | {tags} |".format(
                seed=seed["seed"],
                sample_count=seed["sample_count"],
                strict_valid_sample_count=seed["strict_valid_sample_count"],
                margin_warning=seed["margin_warning"],
                dead_air_samples=_indices_label(seed["dead_air_sample_indices"]),
                unique_pitch_samples=_indices_label(seed["unique_pitch_sample_indices"]),
                overlap_samples=_indices_label(seed["dead_air_unique_pitch_overlap_indices"]),
                best_sample=best.get("sample_index", "-") if best else "-",
                best_dead_air=_fmt(float(best.get("dead_air_ratio", 0.0)), 3) if best else "-",
                tags=tag_label,
            )
        )

    warning_seeds = {int(seed["seed"]) for seed in report["seeds"] if seed["margin_warning"]}
    if warning_seeds:
        lines.extend(["", "## Margin Warning Samples", ""])
        lines.append(
            "| seed | sample | strict | notes | pitches | dead-air | phrase | onset | sustained | tail | removal | tags | reason |"
        )
        lines.append("|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|")
        for sample in report["samples"]:
            if int(sample["seed"]) not in warning_seeds:
                continue
            lines.append(
                "| {seed} | {sample_index} | {strict_valid} | {note_count} | {unique_pitch_count} | "
                "{dead_air_ratio} | {phrase_coverage_ratio} | {onset_coverage_ratio} | "
                "{sustained_coverage_ratio} | {tail_empty_steps} | {postprocess_removal_ratio} | "
                "{tags} | {reason} |".format(
                    seed=sample["seed"],
                    sample_index=sample["sample_index"],
                    strict_valid=sample["strict_valid"],
                    note_count=sample["note_count"],
                    unique_pitch_count=sample["unique_pitch_count"],
                    dead_air_ratio=_fmt(sample["dead_air_ratio"]),
                    phrase_coverage_ratio=_fmt(sample["phrase_coverage_ratio"]),
                    onset_coverage_ratio=_fmt(sample["onset_coverage_ratio"]),
                    sustained_coverage_ratio=_fmt(sample["sustained_coverage_ratio"]),
                    tail_empty_steps=sample["tail_empty_steps"],
                    postprocess_removal_ratio=_fmt(sample["postprocess_removal_ratio"]),
                    tags=", ".join(sample["failure_tags"]) or "none",
                    reason=sample.get("failure_reason") or "none",
                )
            )
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose Stage B seed strict margin risk")
    parser.add_argument("--summary_path", type=str, required=True)
    parser.add_argument(
        "--output_root",
        type=str,
        default=str(ROOT_DIR / "outputs" / "stage_b_seed_strict_margin_diagnostics"),
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--warning_min_strict_samples_per_seed", type=int, default=2)
    parser.add_argument("--dead_air_gate", type=float, default=0.8)
    parser.add_argument("--min_unique_pitches", type=int, default=3)
    parser.add_argument("--expected_margin_warning_seeds", type=str, default=None)
    return parser


def _parse_seed_list(raw: str | None) -> list[int] | None:
    if raw is None:
        return None
    return [int(value.strip()) for value in raw.split(",") if value.strip()]


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    report = build_diagnostic_report(
        source_summary_path=Path(args.summary_path),
        warning_min_strict_samples_per_seed=args.warning_min_strict_samples_per_seed,
        dead_air_gate=args.dead_air_gate,
        min_unique_pitches=args.min_unique_pitches,
    )
    write_json(output_dir / "seed_strict_margin_diagnostics.json", report)
    (output_dir / "seed_strict_margin_diagnostics.md").write_text(markdown_report(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True, indent=2))

    expected = _parse_seed_list(args.expected_margin_warning_seeds)
    if expected is not None and report["summary"]["margin_warning_seeds"] != expected:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
