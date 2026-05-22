"""Aggregate filled Stage B listening review notes into follow-up signals."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from scripts.build_listening_review_notes import ISSUE_VALUES, read_json, validate_review_notes, write_json


AGGREGATE_SCHEMA_VERSION = "stage_b_listening_review_aggregate_v1"
METRIC_KEYS = (
    "note_count",
    "unique_pitch_count",
    "chord_tone_ratio",
    "tension_ratio",
    "approach_ratio",
    "outside_ratio",
)


def _empty_counter(keys: set[str]) -> dict[str, int]:
    return {key: 0 for key in sorted(keys)}


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _metric_summary(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"count": len(candidates)}
    for key in METRIC_KEYS:
        values: list[float] = []
        for candidate in candidates:
            metrics = candidate.get("source_metrics", {})
            if key in metrics and metrics[key] is not None:
                values.append(float(metrics[key]))
        summary[f"avg_{key}"] = _mean(values)
    return summary


def _recommend_followups(
    reviewed_count: int,
    issue_counts: Counter[str],
    timing_counts: Counter[str],
    chord_fit_counts: Counter[str],
    phrase_quality_counts: Counter[str],
) -> list[dict[str, Any]]:
    if reviewed_count == 0:
        return [
            {
                "code": "collect_listening_reviews",
                "reason": "No reviewed candidates are present, so generation rules should not be changed from this artifact alone.",
                "count": 0,
            }
        ]

    recommendations: list[dict[str, Any]] = []

    def add(code: str, reason: str, count: int) -> None:
        if count > 0:
            recommendations.append({"code": code, "reason": reason, "count": int(count)})

    timing_problem_count = (
        issue_counts["bad_timing"]
        + timing_counts["too_stiff"]
        + timing_counts["too_loose"]
        + timing_counts["off_grid"]
    )
    safety_problem_count = issue_counts["too_safe"] + chord_fit_counts["too_safe"]
    phrase_problem_count = (
        issue_counts["too_scalar"]
        + issue_counts["too_mechanical"]
        + issue_counts["weak_phrase"]
        + phrase_quality_counts["exercise"]
        + phrase_quality_counts["fragment"]
    )
    chord_problem_count = issue_counts["bad_chord_fit"] + chord_fit_counts["too_outside"]

    add(
        "fix_timing_grid",
        "Reviewed candidates report timing or grid problems; keep straight-grid references before adding more swing looseness.",
        timing_problem_count,
    )
    add(
        "increase_tension_approach_vocabulary",
        "Reviewed candidates sound too safe; increase tension and approach-note vocabulary before broad training.",
        safety_problem_count,
    )
    add(
        "improve_phrase_vocabulary",
        "Reviewed candidates read as fragments, exercises, scalar motion, or mechanical phrases.",
        phrase_problem_count,
    )
    add(
        "tighten_chord_fit",
        "Reviewed candidates have chord-fit problems; revisit chord-aware pitch selection before style adaptation.",
        chord_problem_count,
    )
    add(
        "increase_motif_variation",
        "Reviewed candidates are too repetitive; strengthen motif variation and repetition gates.",
        issue_counts["too_repetitive"],
    )
    add(
        "increase_density_or_coverage",
        "Reviewed candidates are too sparse; increase phrase density or temporal coverage constraints.",
        issue_counts["too_sparse"],
    )

    recommendations.sort(key=lambda item: (-int(item["count"]), str(item["code"])))
    return recommendations


def aggregate_review_notes(payload: dict[str, Any]) -> dict[str, Any]:
    validation_summary = validate_review_notes(payload)
    candidates = payload["candidates"]

    phrase_quality_counts: Counter[str] = Counter()
    timing_counts: Counter[str] = Counter()
    chord_fit_counts: Counter[str] = Counter()
    decision_counts: Counter[str] = Counter()
    issue_counts: Counter[str] = Counter()
    objective_flag_counts: Counter[str] = Counter()
    objective_bucket_counts: Counter[str] = Counter()
    objective_reviewable_count = 0
    objective_candidates: list[dict[str, Any]] = []
    reviewed_candidates: list[dict[str, Any]] = []
    by_decision: dict[str, list[dict[str, Any]]] = {}

    for candidate in candidates:
        listening = candidate["listening"]
        status = str(listening["status"])
        phrase_quality = str(listening["phrase_quality"])
        timing = str(listening["timing"])
        chord_fit = str(listening["chord_fit"])
        decision = str(listening["decision"])

        phrase_quality_counts[phrase_quality] += 1
        timing_counts[timing] += 1
        chord_fit_counts[chord_fit] += 1
        decision_counts[decision] += 1
        by_decision.setdefault(decision, []).append(candidate)
        for issue in listening.get("issues", []):
            issue_counts[str(issue)] += 1

        objective_review = candidate.get("objective_review")
        if isinstance(objective_review, dict):
            objective_bucket = str(objective_review.get("objective_bucket", "problem"))
            objective_flags = [str(flag) for flag in objective_review.get("objective_flags", [])]
            objective_reviewable = bool(objective_review.get("objective_reviewable", False))
            objective_bucket_counts[objective_bucket] += 1
            objective_reviewable_count += int(objective_reviewable)
            for flag in objective_flags:
                objective_flag_counts[flag] += 1
            objective_candidates.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "objective_bucket": objective_bucket,
                    "objective_reviewable": objective_reviewable,
                    "objective_priority_score": int(objective_review.get("objective_priority_score", 0) or 0),
                    "objective_penalty": int(objective_review.get("objective_penalty", 0) or 0),
                    "objective_flags": objective_flags,
                }
            )

        if status == "reviewed":
            reviewed_candidates.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "phrase_quality": phrase_quality,
                    "timing": timing,
                    "chord_fit": chord_fit,
                    "issues": list(listening.get("issues", [])),
                    "decision": decision,
                    "notes": str(listening.get("notes", "")),
                    "source_metrics": candidate.get("source_metrics", {}),
                    "objective_review": candidate.get("objective_review", {}),
                }
            )

    reviewed_count = int(validation_summary["reviewed_count"])
    recommendations = _recommend_followups(
        reviewed_count=reviewed_count,
        issue_counts=issue_counts,
        timing_counts=timing_counts,
        chord_fit_counts=chord_fit_counts,
        phrase_quality_counts=phrase_quality_counts,
    )

    return {
        "schema_version": AGGREGATE_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "source_schema_version": payload.get("schema_version"),
        "source_review_notes": str(payload.get("source_review_notes", "")),
        "candidate_count": int(validation_summary["candidate_count"]),
        "reviewed_count": reviewed_count,
        "pending_count": int(validation_summary["pending_count"]),
        "has_reviewed_candidates": reviewed_count > 0,
        "decision_counts": dict(sorted(decision_counts.items())),
        "phrase_quality_counts": dict(sorted(phrase_quality_counts.items())),
        "timing_counts": dict(sorted(timing_counts.items())),
        "chord_fit_counts": dict(sorted(chord_fit_counts.items())),
        "issue_counts": {**_empty_counter(ISSUE_VALUES), **dict(sorted(issue_counts.items()))},
        "objective_review_candidate_count": int(len(objective_candidates)),
        "objective_reviewable_count": int(objective_reviewable_count),
        "objective_flag_counts": dict(sorted(objective_flag_counts.items())),
        "objective_bucket_counts": dict(sorted(objective_bucket_counts.items())),
        "objective_candidates_by_priority": sorted(
            objective_candidates,
            key=lambda item: (
                0 if item["objective_reviewable"] else 1,
                -int(item["objective_priority_score"]),
                int(item["objective_penalty"]),
                str(item["candidate_id"]),
            ),
        ),
        "source_metric_by_decision": {
            decision: _metric_summary(decision_candidates)
            for decision, decision_candidates in sorted(by_decision.items())
        },
        "reviewed_candidates": reviewed_candidates,
        "recommended_followups": recommendations,
    }


def markdown_summary(aggregate: dict[str, Any], output_path: Path) -> str:
    def append_count_table(title: str, counts: dict[str, int]) -> None:
        lines.extend(["", f"## {title}", "", "| item | count |", "|---|---:|"])
        rows = [(key, count) for key, count in counts.items() if count]
        if not rows:
            lines.append("| none | 0 |")
            return
        for key, count in rows:
            lines.append(f"| {key} | {count} |")

    lines = [
        "# Stage B Listening Review Aggregate",
        "",
        f"- output: `{output_path}`",
        f"- candidates: `{aggregate['candidate_count']}`",
        f"- reviewed: `{aggregate['reviewed_count']}`",
        f"- pending: `{aggregate['pending_count']}`",
        f"- has reviewed candidates: `{aggregate['has_reviewed_candidates']}`",
    ]
    append_count_table("Decision Counts", aggregate["decision_counts"])
    append_count_table("Phrase Quality Counts", aggregate["phrase_quality_counts"])
    append_count_table("Timing Counts", aggregate["timing_counts"])
    append_count_table("Chord Fit Counts", aggregate["chord_fit_counts"])
    append_count_table("Issue Counts", aggregate["issue_counts"])
    append_count_table("Objective Bucket Counts", aggregate["objective_bucket_counts"])
    append_count_table("Objective Flag Counts", aggregate["objective_flag_counts"])

    lines.extend(
        [
            "",
            "## Objective Review Priority",
            "",
            f"- objective candidates: `{aggregate['objective_review_candidate_count']}`",
            f"- objective reviewable: `{aggregate['objective_reviewable_count']}`",
            "",
            "| candidate | bucket | reviewable | priority | penalty | flags |",
            "|---|---|---:|---:|---:|---|",
        ]
    )
    for candidate in aggregate["objective_candidates_by_priority"]:
        flags = ", ".join(candidate["objective_flags"]) or "ok_objective"
        lines.append(
            f"| {candidate['candidate_id']} | {candidate['objective_bucket']} | "
            f"{str(candidate['objective_reviewable']).lower()} | {candidate['objective_priority_score']} | "
            f"{candidate['objective_penalty']} | {flags} |"
        )
    if not aggregate["objective_candidates_by_priority"]:
        lines.append("| none | none | false | 0 | 0 | none |")

    lines.extend(["", "## Recommended Followups", "", "| code | count | reason |", "|---|---:|---|"])
    for item in aggregate["recommended_followups"]:
        lines.append(f"| {item['code']} | {item['count']} | {item['reason']} |")
    return "\n".join(lines).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate filled Stage B listening review notes")
    parser.add_argument("--review_notes", type=str, required=True)
    parser.add_argument(
        "--output_root",
        type=str,
        default=str(ROOT_DIR / "outputs" / "stage_b_listening_review_aggregate"),
    )
    parser.add_argument("--run_id", type=str, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_root) / run_id
    notes_path = Path(args.review_notes)
    payload = read_json(notes_path)
    payload["source_review_notes"] = str(notes_path)
    aggregate = aggregate_review_notes(payload)
    output_path = run_dir / "listening_review_aggregate.json"
    write_json(output_path, aggregate)
    (run_dir / "listening_review_aggregate.md").write_text(markdown_summary(aggregate, output_path), encoding="utf-8")
    print(
        json.dumps(
            {
                "candidate_count": aggregate["candidate_count"],
                "reviewed_count": aggregate["reviewed_count"],
                "pending_count": aggregate["pending_count"],
                "has_reviewed_candidates": aggregate["has_reviewed_candidates"],
                "recommended_followups": aggregate["recommended_followups"],
                "aggregate_path": str(output_path),
            },
            ensure_ascii=True,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
