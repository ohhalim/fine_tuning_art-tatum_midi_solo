"""Export review-ready candidate metrics from a Stage B repeatability summary."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence


ROOT_DIR = Path(__file__).resolve().parents[1]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _float(source: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = source.get(key, default)
    return float(value if value is not None else default)


def _int(source: dict[str, Any], key: str, default: int = 0) -> int:
    value = source.get(key, default)
    return int(value if value is not None else default)


def _row_by_seed(rows: Sequence[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {int(row.get("seed", 0) or 0): row for row in rows if isinstance(row, dict)}


def compact_candidate(
    candidate: dict[str, Any],
    *,
    selected_key: tuple[int, int] | None,
    row: dict[str, Any] | None,
) -> dict[str, Any]:
    seed = _int(candidate, "seed")
    sample_index = _int(candidate, "sample_index")
    row = row or {}
    return {
        "seed": seed,
        "sample_index": sample_index,
        "is_selected_best": bool(selected_key == (seed, sample_index)),
        "midi_path": str(candidate.get("midi_path", "")),
        "dead_air_ratio": _float(candidate, "dead_air_ratio"),
        "note_count": _int(candidate, "note_count"),
        "unique_pitch_count": _int(candidate, "unique_pitch_count"),
        "phrase_coverage_ratio": _float(candidate, "phrase_coverage_ratio"),
        "onset_coverage_ratio": _float(candidate, "onset_coverage_ratio"),
        "sustained_coverage_ratio": _float(candidate, "sustained_coverage_ratio"),
        "postprocess_removal_ratio": _float(candidate, "postprocess_removal_ratio"),
        "seed_sample_count": _int(row, "sample_count"),
        "seed_strict_valid_sample_count": _int(row, "strict_valid_sample_count"),
        "seed_dead_air_outlier_count": _int(row, "dead_air_outlier_count"),
        "seed_failure_reasons": row.get("failure_reasons", {}) if isinstance(row.get("failure_reasons"), dict) else {},
        "seed_strict_failure_reasons": (
            row.get("strict_failure_reasons", {}) if isinstance(row.get("strict_failure_reasons"), dict) else {}
        ),
    }


def rank_candidates(candidates: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = sorted(
        candidates,
        key=lambda candidate: (
            float(candidate["dead_air_ratio"]),
            float(candidate["postprocess_removal_ratio"]),
            -int(candidate["note_count"]),
            int(candidate["seed"]),
            int(candidate["sample_index"]),
        ),
    )
    return [dict(candidate, review_rank=index) for index, candidate in enumerate(ranked, start=1)]


def build_review_export(summary_path: Path) -> dict[str, Any]:
    source = read_json(summary_path)
    summary = source.get("summary", {}) if isinstance(source.get("summary"), dict) else {}
    rows = [row for row in source.get("rows", []) if isinstance(row, dict)]
    row_lookup = _row_by_seed(rows)
    selected = summary.get("selected_best_candidate")
    selected_key = None
    if isinstance(selected, dict):
        selected_key = (_int(selected, "seed"), _int(selected, "sample_index"))
    raw_candidates = [
        candidate
        for candidate in summary.get("seed_best_candidates", [])
        if isinstance(candidate, dict)
    ]
    candidates = [
        compact_candidate(
            candidate,
            selected_key=selected_key,
            row=row_lookup.get(_int(candidate, "seed")),
        )
        for candidate in raw_candidates
    ]
    ranked_candidates = rank_candidates(candidates)
    selected_rank = next(
        (int(candidate["review_rank"]) for candidate in ranked_candidates if candidate["is_selected_best"]),
        None,
    )

    return {
        "source_summary_path": str(summary_path),
        "source_run_id": str(source.get("run_id", "")),
        "source_issue": int(source.get("issue", 0) or 0),
        "candidate_count": int(len(ranked_candidates)),
        "selected_best_rank": selected_rank,
        "summary": {
            "passed_repeatability_gate": bool(summary.get("passed_repeatability_gate", False)),
            "total_samples": _int(summary, "total_samples"),
            "total_strict_valid_sample_count": _int(summary, "total_strict_valid_sample_count"),
            "strict_valid_sample_rate": _float(summary, "strict_valid_sample_rate"),
            "dead_air_outlier_rate": _float(summary, "dead_air_outlier_rate"),
            "strict_margin_warning_seeds": summary.get("strict_margin_warning_seeds", []),
        },
        "candidates": ranked_candidates,
    }


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _reason_label(value: Any) -> str:
    if not isinstance(value, dict) or not value:
        return "none"
    return ", ".join(f"{reason} ({count})" for reason, count in sorted(value.items()))


def markdown_report(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Stage B Margin-Recovered Candidate Review Export",
        "",
        f"- source summary: `{report['source_summary_path']}`",
        f"- source run: `{report['source_run_id']}`",
        f"- passed repeatability gate: `{str(summary['passed_repeatability_gate']).lower()}`",
        f"- total strict samples: `{summary['total_strict_valid_sample_count']}/{summary['total_samples']}`",
        f"- strict pass-rate: `{summary['strict_valid_sample_rate']:.3f}`",
        f"- dead-air outlier rate: `{summary['dead_air_outlier_rate']:.3f}`",
        f"- strict margin warning seeds: `{summary['strict_margin_warning_seeds']}`",
        f"- selected best rank: `{report['selected_best_rank']}`",
        "",
        "| rank | selected | seed | sample | seed strict | outliers | dead-air | notes | pitches | phrase | onset | sustained | removal | MIDI |",
        "|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for candidate in report["candidates"]:
        lines.append(
            "| {rank} | {selected} | {seed} | {sample} | {seed_strict}/{seed_samples} | "
            "{outliers} | {dead_air} | {notes} | {pitches} | {phrase} | {onset} | "
            "{sustained} | {removal} | `{midi}` |".format(
                rank=candidate["review_rank"],
                selected=candidate["is_selected_best"],
                seed=candidate["seed"],
                sample=candidate["sample_index"],
                seed_strict=candidate["seed_strict_valid_sample_count"],
                seed_samples=candidate["seed_sample_count"],
                outliers=candidate["seed_dead_air_outlier_count"],
                dead_air=_fmt(candidate["dead_air_ratio"]),
                notes=candidate["note_count"],
                pitches=candidate["unique_pitch_count"],
                phrase=_fmt(candidate["phrase_coverage_ratio"]),
                onset=_fmt(candidate["onset_coverage_ratio"]),
                sustained=_fmt(candidate["sustained_coverage_ratio"]),
                removal=_fmt(candidate["postprocess_removal_ratio"]),
                midi=candidate["midi_path"],
            )
        )
    lines.extend(["", "## Seed Failure Reasons", ""])
    for candidate in report["candidates"]:
        lines.append(
            f"- seed `{candidate['seed']}`: "
            f"failure `{_reason_label(candidate['seed_failure_reasons'])}`, "
            f"strict `{_reason_label(candidate['seed_strict_failure_reasons'])}`"
        )
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export Stage B margin-recovered candidate review metrics")
    parser.add_argument("--summary_path", type=str, required=True)
    parser.add_argument(
        "--output_root",
        type=str,
        default=str(ROOT_DIR / "outputs" / "stage_b_margin_recovered_review_export"),
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--expected_candidate_count", type=int, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    report = build_review_export(Path(args.summary_path))
    write_json(output_dir / "candidate_review_export.json", report)
    (output_dir / "candidate_review_export.md").write_text(markdown_report(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True, indent=2))

    if args.expected_candidate_count is not None and report["candidate_count"] != int(args.expected_candidate_count):
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
