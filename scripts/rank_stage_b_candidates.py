"""Rank Stage B generated MIDI candidates from an A/B sweep report."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def score_candidate(sample: dict[str, Any]) -> dict[str, Any]:
    metrics = sample.get("metrics", {})
    collapse = sample.get("collapse", {})
    temporal = sample.get("temporal_coverage", {})
    valid = bool(sample.get("valid"))
    strict_valid = bool(sample.get("strict_valid"))
    grammar_valid = bool(sample.get("grammar_gate_passed"))
    dead_air_ratio = _float(metrics.get("dead_air_ratio"))
    repetition_score = _float(metrics.get("repetition_score"))
    chord_tone_ratio = _float(metrics.get("chord_tone_ratio"))
    unique_pitch_count = _int(metrics.get("unique_pitch_count"))
    onset_coverage_ratio = _float(temporal.get("onset_coverage_ratio"))
    sustained_coverage_ratio = _float(temporal.get("sustained_coverage_ratio"))
    position_span_ratio = _float(temporal.get("position_span_ratio"))
    collapse_warning = bool(collapse.get("collapse_warning"))
    postprocess_removal_ratio = _float(collapse.get("postprocess_removal_ratio"))

    components = {
        "strict_valid_bonus": 40.0 if strict_valid else 0.0,
        "valid_bonus": 20.0 if valid else 0.0,
        "grammar_bonus": 10.0 if grammar_valid else 0.0,
        "onset_coverage_bonus": onset_coverage_ratio * 20.0,
        "sustained_coverage_bonus": sustained_coverage_ratio * 10.0,
        "position_span_bonus": position_span_ratio * 5.0,
        "chord_tone_bonus": chord_tone_ratio * 10.0,
        "pitch_diversity_bonus": min(unique_pitch_count, 6) * 2.0,
        "dead_air_penalty": dead_air_ratio * -20.0,
        "repetition_penalty": repetition_score * -8.0,
        "postprocess_penalty": postprocess_removal_ratio * -10.0,
        "collapse_warning_penalty": -25.0 if collapse_warning else 0.0,
    }
    score = round(sum(components.values()), 4)
    return {"score": score, "score_components": components}


def candidate_from_sample(row: dict[str, Any], sample: dict[str, Any], report_path: Path) -> dict[str, Any]:
    metrics = sample.get("metrics", {})
    temporal = sample.get("temporal_coverage", {})
    collapse = sample.get("collapse", {})
    scored = score_candidate(sample)
    return {
        "score": scored["score"],
        "score_components": scored["score_components"],
        "mode": row.get("mode"),
        "note_groups_per_bar": _int(row.get("note_groups_per_bar")),
        "run_id": row.get("run_id"),
        "sample_index": _int(sample.get("sample_index")),
        "midi_path": sample.get("midi_path"),
        "report_path": str(report_path),
        "valid": bool(sample.get("valid")),
        "strict_valid": bool(sample.get("strict_valid")),
        "grammar_gate_passed": bool(sample.get("grammar_gate_passed")),
        "failure_reason": sample.get("failure_reason"),
        "diagnostic_failure_reason": sample.get("diagnostic_failure_reason"),
        "note_count": _int(metrics.get("note_count")),
        "unique_pitch_count": _int(metrics.get("unique_pitch_count")),
        "dead_air_ratio": _float(metrics.get("dead_air_ratio")),
        "repetition_score": _float(metrics.get("repetition_score")),
        "chord_tone_ratio": _float(metrics.get("chord_tone_ratio")),
        "onset_coverage_ratio": _float(temporal.get("onset_coverage_ratio")),
        "sustained_coverage_ratio": _float(temporal.get("sustained_coverage_ratio")),
        "position_span_ratio": _float(temporal.get("position_span_ratio")),
        "longest_sustained_empty_run_steps": _int(temporal.get("longest_sustained_empty_run_steps")),
        "collapse_warning": bool(collapse.get("collapse_warning")),
        "postprocess_removal_ratio": _float(collapse.get("postprocess_removal_ratio")),
    }


def collect_candidates(ab_report: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    candidates: list[dict[str, Any]] = []
    warnings: list[str] = []
    for row in ab_report.get("rows", []):
        report_path = Path(str(row.get("report_path", "")))
        if not report_path.is_absolute():
            report_path = ROOT_DIR / report_path
        if not report_path.exists():
            warnings.append(f"missing report: {report_path}")
            continue
        probe_report = read_json(report_path)
        for sample in probe_report.get("samples", []):
            candidates.append(candidate_from_sample(row, sample, report_path))
    return candidates, warnings


def rank_candidates(candidates: list[dict[str, Any]], top_n: int) -> list[dict[str, Any]]:
    ranked = sorted(
        candidates,
        key=lambda candidate: (
            float(candidate["score"]),
            bool(candidate["strict_valid"]),
            bool(candidate["valid"]),
            float(candidate["onset_coverage_ratio"]),
            float(candidate["chord_tone_ratio"]),
            -float(candidate["dead_air_ratio"]),
        ),
        reverse=True,
    )
    top = ranked[: max(1, int(top_n))]
    for index, candidate in enumerate(top, start=1):
        candidate["rank"] = index
    return top


def build_summary(candidates: list[dict[str, Any]], top_candidates: list[dict[str, Any]]) -> dict[str, Any]:
    strict_candidates = [candidate for candidate in candidates if candidate["strict_valid"]]
    valid_candidates = [candidate for candidate in candidates if candidate["valid"]]
    top_strict = [candidate for candidate in top_candidates if candidate["strict_valid"]]
    return {
        "candidate_count": int(len(candidates)),
        "valid_candidate_count": int(len(valid_candidates)),
        "strict_candidate_count": int(len(strict_candidates)),
        "top_candidate_count": int(len(top_candidates)),
        "top_strict_candidate_count": int(len(top_strict)),
        "best_candidate": top_candidates[0] if top_candidates else None,
    }


def markdown_report(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Stage B Candidate Ranking",
        "",
        f"- candidate count: `{summary['candidate_count']}`",
        f"- valid candidates: `{summary['valid_candidate_count']}`",
        f"- strict candidates: `{summary['strict_candidate_count']}`",
        "",
        "| rank | score | mode | groups/bar | sample | strict | notes | onset | sustained | dead-air | chord-tone | MIDI |",
        "|---:|---:|---|---:|---:|:---:|---:|---:|---:|---:|---:|---|",
    ]
    for candidate in report["top_candidates"]:
        lines.append(
            "| {rank} | {score:.3f} | {mode} | {note_groups_per_bar} | {sample_index} | "
            "{strict_valid} | {note_count} | {onset_coverage_ratio:.3f} | "
            "{sustained_coverage_ratio:.3f} | {dead_air_ratio:.3f} | "
            "{chord_tone_ratio:.3f} | `{midi_path}` |".format(**candidate)
        )
    lines.append("")
    lines.append("## Scoring Note")
    lines.append("")
    lines.append("This score is a review-prioritization heuristic, not a musical-quality claim.")
    lines.append("It rewards strict validity, temporal coverage, chord-tone ratio, and pitch diversity.")
    lines.append("It penalizes dead-air, repetition, postprocess removal, and collapse warnings.")
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rank Stage B candidate MIDI samples from an A/B sweep")
    parser.add_argument(
        "--ab_sweep_report",
        type=str,
        default=str(
            ROOT_DIR
            / "outputs"
            / "stage_b_coverage_ab_sweep"
            / "harness_stage_b_coverage_ab_sweep"
            / "ab_sweep_report.json"
        ),
    )
    parser.add_argument("--output_root", type=str, default=str(ROOT_DIR / "outputs" / "stage_b_candidate_ranking"))
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--issue_number", type=int, default=41)
    parser.add_argument("--top_n", type=int, default=12)
    parser.add_argument("--min_top_strict_candidates", type=int, default=1)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_root) / run_id
    ab_report_path = Path(args.ab_sweep_report)
    ab_report = read_json(ab_report_path)
    candidates, warnings = collect_candidates(ab_report)
    top_candidates = rank_candidates(candidates, top_n=args.top_n)
    summary = build_summary(candidates, top_candidates)
    report = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "issue": int(args.issue_number),
        "ab_sweep_report": str(ab_report_path),
        "summary": summary,
        "warnings": warnings,
        "top_candidates": top_candidates,
    }
    write_json(run_dir / "candidate_rank_report.json", report)
    (run_dir / "candidate_rank_report.md").write_text(markdown_report(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True, indent=2))
    return 0 if int(summary["top_strict_candidate_count"]) >= int(args.min_top_strict_candidates) else 3


if __name__ == "__main__":
    raise SystemExit(main())
