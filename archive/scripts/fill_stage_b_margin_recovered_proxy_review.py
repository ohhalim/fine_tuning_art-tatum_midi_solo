"""Fill Stage B margin-recovered listening notes with MIDI-metric proxy review."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from scripts.build_stage_b_margin_recovered_listening_notes import validate_listening_notes, write_json  # noqa: E402


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _metrics(candidate: dict[str, Any]) -> dict[str, Any]:
    metrics = candidate.get("source_metrics")
    return metrics if isinstance(metrics, dict) else {}


def _metadata(candidate: dict[str, Any]) -> dict[str, Any]:
    metadata = candidate.get("review_metadata")
    return metadata if isinstance(metadata, dict) else {}


def proxy_score(candidate: dict[str, Any]) -> float:
    metrics = _metrics(candidate)
    metadata = _metadata(candidate)
    dead_air = float(metrics.get("dead_air_ratio", 0.0) or 0.0)
    phrase = float(metrics.get("phrase_coverage_ratio", 0.0) or 0.0)
    onset = float(metrics.get("onset_coverage_ratio", 0.0) or 0.0)
    sustained = float(metrics.get("sustained_coverage_ratio", 0.0) or 0.0)
    removal = float(metrics.get("postprocess_removal_ratio", 0.0) or 0.0)
    notes = int(metrics.get("note_count", 0) or 0)
    seed_outliers = int(metadata.get("seed_dead_air_outlier_count", 0) or 0)
    seed_strict = int(metadata.get("seed_strict_valid_sample_count", 0) or 0)
    seed_samples = int(metadata.get("seed_sample_count", 0) or 0)
    seed_margin = max(0, seed_strict - 2)
    seed_failure_penalty = 0.04 * seed_outliers + (0.05 if seed_samples and seed_strict < seed_samples else 0.0)
    return (
        phrase * 0.32
        + onset * 0.20
        + sustained * 0.20
        + min(notes, 24) / 24.0 * 0.12
        + seed_margin * 0.04
        - dead_air * 0.12
        - removal * 0.08
        - seed_failure_penalty
    )


def timing_label(metrics: dict[str, Any]) -> str:
    onset = float(metrics.get("onset_coverage_ratio", 0.0) or 0.0)
    sustained = float(metrics.get("sustained_coverage_ratio", 0.0) or 0.0)
    dead_air = float(metrics.get("dead_air_ratio", 0.0) or 0.0)
    if onset >= 0.50 and sustained >= 0.70 and dead_air <= 0.55:
        return "acceptable"
    if onset < 0.35 or sustained < 0.45:
        return "stiff"
    return "acceptable"


def phrase_label(metrics: dict[str, Any]) -> str:
    notes = int(metrics.get("note_count", 0) or 0)
    phrase = float(metrics.get("phrase_coverage_ratio", 0.0) or 0.0)
    if phrase >= 0.90 and notes >= 16:
        return "strong"
    if phrase >= 0.70 and notes >= 12:
        return "acceptable"
    if phrase >= 0.40:
        return "weak"
    return "broken"


def vocabulary_label(metrics: dict[str, Any]) -> str:
    notes = int(metrics.get("note_count", 0) or 0)
    pitches = int(metrics.get("unique_pitch_count", 0) or 0)
    phrase = float(metrics.get("phrase_coverage_ratio", 0.0) or 0.0)
    if notes >= 16 and pitches >= 4 and phrase >= 0.90:
        return "acceptable"
    if notes < 12 or pitches < 4:
        return "thin"
    return "too_safe"


def fill_candidate(candidate: dict[str, Any], keep_candidate_id: str) -> dict[str, Any]:
    updated = copy.deepcopy(candidate)
    candidate_id = str(updated.get("candidate_id", ""))
    metrics = _metrics(updated)
    score = proxy_score(updated)
    decision = "keep" if candidate_id == keep_candidate_id else "needs_followup"
    metadata = _metadata(updated)
    reasons: list[str] = [
        f"proxy_score={score:.3f}",
        f"dead_air={float(metrics.get('dead_air_ratio', 0.0) or 0.0):.3f}",
        f"phrase={float(metrics.get('phrase_coverage_ratio', 0.0) or 0.0):.3f}",
        f"onset={float(metrics.get('onset_coverage_ratio', 0.0) or 0.0):.3f}",
        f"sustained={float(metrics.get('sustained_coverage_ratio', 0.0) or 0.0):.3f}",
    ]
    if metadata.get("is_selected_best") and decision != "keep":
        reasons.append("dead-air selected best loses proxy review to richer coverage candidate")
    if int(metadata.get("seed_dead_air_outlier_count", 0) or 0):
        reasons.append("seed has remaining failure outliers")

    updated["proxy_review"] = {
        "method": "midi_metric_proxy_review",
        "not_human_listening": True,
        "score": score,
        "score_inputs": {
            "dead_air_ratio": float(metrics.get("dead_air_ratio", 0.0) or 0.0),
            "note_count": int(metrics.get("note_count", 0) or 0),
            "unique_pitch_count": int(metrics.get("unique_pitch_count", 0) or 0),
            "phrase_coverage_ratio": float(metrics.get("phrase_coverage_ratio", 0.0) or 0.0),
            "onset_coverage_ratio": float(metrics.get("onset_coverage_ratio", 0.0) or 0.0),
            "sustained_coverage_ratio": float(metrics.get("sustained_coverage_ratio", 0.0) or 0.0),
            "postprocess_removal_ratio": float(metrics.get("postprocess_removal_ratio", 0.0) or 0.0),
        },
        "reasons": reasons,
    }
    updated["listening"] = {
        "status": "reviewed",
        "timing": timing_label(metrics),
        "phrase": phrase_label(metrics),
        "jazz_vocabulary": vocabulary_label(metrics),
        "decision": decision,
        "notes": "; ".join(reasons),
    }
    return updated


def fill_proxy_review(notes: dict[str, Any]) -> dict[str, Any]:
    candidates = notes.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("notes must contain non-empty candidates")
    ranked = sorted(candidates, key=proxy_score, reverse=True)
    keep_candidate_id = str(ranked[0].get("candidate_id", ""))
    updated = copy.deepcopy(notes)
    updated["review_context"] = {
        **(updated.get("review_context", {}) if isinstance(updated.get("review_context"), dict) else {}),
        "proxy_review_filled": True,
        "proxy_review_method": "midi_metric_proxy_review",
        "not_human_listening": True,
    }
    updated["proxy_review_summary"] = {
        "keep_candidate_id": keep_candidate_id,
        "candidate_scores": [
            {"candidate_id": str(candidate.get("candidate_id", "")), "score": proxy_score(candidate)}
            for candidate in ranked
        ],
    }
    updated["candidates"] = [fill_candidate(candidate, keep_candidate_id) for candidate in candidates]
    return updated


def markdown_report(notes: dict[str, Any], summary: dict[str, Any]) -> str:
    proxy_summary = notes.get("proxy_review_summary", {}) if isinstance(notes.get("proxy_review_summary"), dict) else {}
    lines = [
        "# Stage B Margin-Recovered MIDI Proxy Review Fill",
        "",
        f"- source run: `{notes.get('source_run_id', '')}`",
        f"- reviewed count: `{summary['reviewed_count']}`",
        f"- keep count: `{summary['decision_counts']['keep']}`",
        f"- needs followup count: `{summary['decision_counts']['needs_followup']}`",
        f"- keep candidate: `{proxy_summary.get('keep_candidate_id', '')}`",
        "- review type: `midi_metric_proxy_review`",
        "- human listening proof: `false`",
        "",
        "| candidate | selected | score | timing | phrase | vocabulary | decision | notes |",
        "|---|:---:|---:|---|---|---|---|---|",
    ]
    for candidate in notes["candidates"]:
        metadata = _metadata(candidate)
        proxy = candidate.get("proxy_review", {}) if isinstance(candidate.get("proxy_review"), dict) else {}
        listening = candidate["listening"]
        lines.append(
            "| `{candidate_id}` | {selected} | {score:.3f} | {timing} | {phrase} | {vocab} | {decision} | {notes} |".format(
                candidate_id=candidate["candidate_id"],
                selected=metadata.get("is_selected_best", False),
                score=float(proxy.get("score", 0.0) or 0.0),
                timing=listening["timing"],
                phrase=listening["phrase"],
                vocab=listening["jazz_vocabulary"],
                decision=listening["decision"],
                notes=listening["notes"],
            )
        )
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fill Stage B margin-recovered MIDI proxy review notes")
    parser.add_argument("--review_notes", type=str, required=True)
    parser.add_argument(
        "--output_root",
        type=str,
        default=str(ROOT_DIR / "outputs" / "stage_b_margin_recovered_proxy_review"),
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--expected_keep_candidate_id", type=str, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or "harness_stage_b_margin_recovered_proxy_review"
    run_dir = Path(args.output_root) / run_id
    notes = read_json(Path(args.review_notes))
    filled = fill_proxy_review(notes)
    summary = validate_listening_notes(filled)
    write_json(run_dir / "listening_review_notes_proxy_filled.json", filled)
    write_json(run_dir / "listening_review_notes_proxy_summary.json", summary)
    (run_dir / "listening_review_notes_proxy_filled.md").write_text(markdown_report(filled, summary), encoding="utf-8")
    print(
        json.dumps(
            {
                **summary,
                "keep_candidate_id": filled["proxy_review_summary"]["keep_candidate_id"],
                "review_notes_path": str(run_dir / "listening_review_notes_proxy_filled.json"),
            },
            ensure_ascii=True,
            indent=2,
        )
    )
    if (
        args.expected_keep_candidate_id
        and filled["proxy_review_summary"]["keep_candidate_id"] != args.expected_keep_candidate_id
    ):
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
