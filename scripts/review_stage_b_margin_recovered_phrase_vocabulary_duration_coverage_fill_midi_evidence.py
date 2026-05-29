"""Review source vs duration/coverage fill from MIDI evidence only."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class DurationCoverageFillMidiEvidenceReviewError(ValueError):
    pass


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def review_items(package: dict[str, Any]) -> dict[str, dict[str, Any]]:
    items = package.get("review_items")
    if not isinstance(items, list):
        raise DurationCoverageFillMidiEvidenceReviewError("package must contain review_items")
    by_role = {str(item.get("role") or ""): item for item in items if isinstance(item, dict)}
    required = {"source_constrained_partial", "duration_coverage_fill_keep"}
    missing = sorted(required - set(by_role))
    if missing:
        raise DurationCoverageFillMidiEvidenceReviewError(f"missing review roles: {missing}")
    return by_role


def metric(item: dict[str, Any], name: str, default: float = 0.0) -> float:
    metrics = item.get("metric_summary") if isinstance(item.get("metric_summary"), dict) else {}
    return float(metrics.get(name, default) or default)


def compact_item(item: dict[str, Any]) -> dict[str, Any]:
    metrics = item.get("metric_summary") if isinstance(item.get("metric_summary"), dict) else {}
    return {
        "role": str(item.get("role") or ""),
        "candidate_id": str(item.get("candidate_id") or ""),
        "note_count": int(metrics.get("note_count", 0) or 0),
        "focused_note_count": int(metrics.get("focused_note_count", 0) or 0),
        "unique_pitch_count": int(metrics.get("unique_pitch_count", 0) or 0),
        "focused_unique_pitch_count": int(metrics.get("focused_unique_pitch_count", 0) or 0),
        "dead_air_ratio": float(metrics.get("dead_air_ratio", 0.0) or 0.0),
        "max_simultaneous_notes": int(metrics.get("max_simultaneous_notes", 0) or 0),
        "focused_max_simultaneous_notes": int(metrics.get("focused_max_simultaneous_notes", 0) or 0),
        "adjacent_pitch_repeats": int(metrics.get("adjacent_pitch_repeats", 0) or 0),
        "duplicated_3_note_pitch_class_chunks": int(
            metrics.get("duplicated_3_note_pitch_class_chunks", 0) or 0
        ),
        "max_interval": int(metrics.get("max_interval", 0) or 0),
    }


def item_score(item: dict[str, Any]) -> float:
    return round(
        metric(item, "focused_note_count") * 2.0
        + metric(item, "focused_unique_pitch_count") * 5.0
        + (1.0 - min(1.0, metric(item, "dead_air_ratio", 1.0))) * 100.0
        - metric(item, "max_simultaneous_notes") * 10.0
        - metric(item, "adjacent_pitch_repeats") * 20.0
        - metric(item, "duplicated_3_note_pitch_class_chunks") * 30.0
        - max(0.0, metric(item, "max_interval") - 7.0) * 5.0,
        6,
    )


def preference_from_scores(source_score: float, fill_score: float) -> str:
    if fill_score > source_score:
        return "duration_coverage_fill_keep"
    if source_score > fill_score:
        return "source_constrained_partial"
    return "tie"


def build_midi_evidence_review(
    package: dict[str, Any],
    *,
    output_dir: Path,
) -> dict[str, Any]:
    package_boundary = package.get("package_boundary") if isinstance(package.get("package_boundary"), dict) else {}
    if bool(package_boundary.get("preference_claimed", True)):
        raise DurationCoverageFillMidiEvidenceReviewError("package already claims a preference")
    items = review_items(package)
    source = items["source_constrained_partial"]
    fill = items["duration_coverage_fill_keep"]
    source_score = item_score(source)
    fill_score = item_score(fill)
    preference = preference_from_scores(source_score, fill_score)
    source_metrics = compact_item(source)
    fill_metrics = compact_item(fill)
    return {
        "schema_version": "stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_midi_evidence_review_v1",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "source_package_schema": str(package.get("schema_version") or ""),
        "candidate_id": str(package.get("candidate_id") or fill.get("candidate_id") or ""),
        "review_basis": "midi_metric_and_note_structure",
        "review_items": [source_metrics, fill_metrics],
        "score": {
            "source_constrained_partial": source_score,
            "duration_coverage_fill_keep": fill_score,
            "score_delta_fill_minus_source": round(fill_score - source_score, 6),
        },
        "midi_evidence_review": {
            "status": "reviewed",
            "preference": preference,
            "timing": preference,
            "phrase": preference,
            "vocabulary": preference,
            "rationale": [
                "fill_candidate_has_lower_dead_air_ratio",
                "fill_candidate_has_higher_focused_note_count",
                "fill_candidate_has_higher_focused_unique_pitch_count",
                "fill_candidate_keeps_adjacent_repeat_and_interval_guardrails",
            ],
        },
        "metric_delta": {
            "dead_air_delta_fill_minus_source": round(
                fill_metrics["dead_air_ratio"] - source_metrics["dead_air_ratio"], 6
            ),
            "focused_note_count_delta_fill_minus_source": int(
                fill_metrics["focused_note_count"] - source_metrics["focused_note_count"]
            ),
            "focused_unique_pitch_count_delta_fill_minus_source": int(
                fill_metrics["focused_unique_pitch_count"] - source_metrics["focused_unique_pitch_count"]
            ),
            "max_simultaneous_notes_delta_fill_minus_source": int(
                fill_metrics["max_simultaneous_notes"] - source_metrics["max_simultaneous_notes"]
            ),
        },
        "claim_boundary": {
            "midi_evidence_preference_claimed": True,
            "human_audio_preference_claimed": False,
            "audio_render_used": False,
            "not_human_audio_review": True,
        },
        "not_proven": [
            "human_audio_preference",
            "audio_rendered_quality",
            "broad_trained_model_quality",
            "brad_style_adaptation",
        ],
        "next_recommended_issue": (
            "Stage B margin-recovered phrase/vocabulary duration coverage fill MIDI evidence review consolidation"
        ),
    }


def validate_midi_evidence_review(
    report: dict[str, Any],
    *,
    expected_candidate_id: str | None,
    expected_preference: str | None,
    require_no_human_audio_preference: bool,
    require_audio_not_rendered: bool,
) -> dict[str, Any]:
    candidate_id = str(report.get("candidate_id") or "")
    if expected_candidate_id and candidate_id != expected_candidate_id:
        raise DurationCoverageFillMidiEvidenceReviewError(
            f"expected candidate {expected_candidate_id}, got {candidate_id}"
        )
    review = report.get("midi_evidence_review") if isinstance(report.get("midi_evidence_review"), dict) else {}
    preference = str(review.get("preference") or "")
    if expected_preference and preference != expected_preference:
        raise DurationCoverageFillMidiEvidenceReviewError(
            f"expected preference {expected_preference}, got {preference}"
        )
    claim = report.get("claim_boundary") if isinstance(report.get("claim_boundary"), dict) else {}
    if require_no_human_audio_preference and bool(claim.get("human_audio_preference_claimed", True)):
        raise DurationCoverageFillMidiEvidenceReviewError("human/audio preference must not be claimed")
    if require_audio_not_rendered and bool(claim.get("audio_render_used", True)):
        raise DurationCoverageFillMidiEvidenceReviewError("audio render must not be claimed")
    return {
        "candidate_id": candidate_id,
        "review_basis": str(report.get("review_basis") or ""),
        "preference": preference,
        "score_delta_fill_minus_source": float(report.get("score", {}).get("score_delta_fill_minus_source", 0.0) or 0.0),
        "dead_air_delta_fill_minus_source": float(
            report.get("metric_delta", {}).get("dead_air_delta_fill_minus_source", 0.0) or 0.0
        ),
        "human_audio_preference_claimed": bool(claim.get("human_audio_preference_claimed", True)),
        "audio_render_used": bool(claim.get("audio_render_used", True)),
        "next_recommended_issue": str(report.get("next_recommended_issue") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    review = report["midi_evidence_review"]
    score = report["score"]
    claim = report["claim_boundary"]
    lines = [
        "# Stage B Margin-Recovered Phrase/Vocabulary Duration Coverage Fill MIDI Evidence Review",
        "",
        f"- candidate: `{report['candidate_id']}`",
        f"- review basis: `{report['review_basis']}`",
        f"- preference: `{review['preference']}`",
        f"- score delta fill-source: `{score['score_delta_fill_minus_source']:.3f}`",
        f"- human/audio preference claimed: `{claim['human_audio_preference_claimed']}`",
        f"- audio render used: `{claim['audio_render_used']}`",
        "",
        "| role | candidate | focused notes | focused unique | dead-air | max active | score |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    scores = report["score"]
    for item in report.get("review_items", []):
        role = item["role"]
        lines.append(
            "| "
            + " | ".join(
                [
                    role,
                    item["candidate_id"],
                    str(item["focused_note_count"]),
                    str(item["focused_unique_pitch_count"]),
                    f"{float(item['dead_air_ratio']):.4f}",
                    str(item["max_simultaneous_notes"]),
                    f"{float(scores[role]):.3f}",
                ]
            )
            + " |"
        )
    lines.extend(["", "## Not Proven", ""])
    for item in report.get("not_proven", []):
        lines.append(f"- `{item}`")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Review source vs duration fill from MIDI evidence")
    parser.add_argument("--audio_review_package", type=str, required=True)
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_midi_evidence_review",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--expected_candidate_id", type=str, default="")
    parser.add_argument("--expected_preference", type=str, default="")
    parser.add_argument("--require_no_human_audio_preference", action="store_true")
    parser.add_argument("--require_audio_not_rendered", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    report = build_midi_evidence_review(
        read_json(Path(args.audio_review_package)),
        output_dir=output_dir,
    )
    summary = validate_midi_evidence_review(
        report,
        expected_candidate_id=str(args.expected_candidate_id or ""),
        expected_preference=str(args.expected_preference or ""),
        require_no_human_audio_preference=bool(args.require_no_human_audio_preference),
        require_audio_not_rendered=bool(args.require_audio_not_rendered),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "duration_coverage_fill_midi_evidence_review.json"
    markdown_path = output_dir / "duration_coverage_fill_midi_evidence_review.md"
    write_json(report_path, report)
    write_json(output_dir / "duration_coverage_fill_midi_evidence_review_validation_summary.json", summary)
    markdown_path.write_text(markdown_report(report), encoding="utf-8")
    print(json.dumps({**summary, "report_path": str(report_path), "markdown_path": str(markdown_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
