"""Build focused listening notes for the margin-recovered timing/repetition candidate."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from scripts.build_focused_listening_review_notes import (  # noqa: E402
    build_focused_listening_review_notes,
    validate_focused_listening_review_notes,
)
from scripts.build_stage_b_margin_recovered_listening_notes import write_json  # noqa: E402
from scripts.build_stage_b_margin_recovered_timing_repetition_focused_package import read_json  # noqa: E402


class TimingRepetitionFocusedListeningNotesError(ValueError):
    pass


def decisions_by_candidate(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    candidates = report.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise TimingRepetitionFocusedListeningNotesError("focused context decision must contain candidates")
    rows: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        candidate_id = str(candidate.get("candidate_id") or "")
        if candidate_id:
            rows[candidate_id] = candidate
    return rows


def review_risks(metrics: dict[str, Any]) -> list[str]:
    risks = []
    if float(metrics.get("dead_air_ratio", 0.0) or 0.0) >= 0.35:
        risks.append("dead_air_ratio_remaining")
    if int(metrics.get("adjacent_pitch_repeats", 0) or 0) > 0:
        risks.append("adjacent_pitch_repeats")
    if int(metrics.get("max_interval", 0) or 0) >= 12:
        risks.append("wide_interval_review")
    return risks


def build_timing_repetition_focused_listening_notes(
    focused_package: dict[str, Any],
    focused_context_decision: dict[str, Any],
) -> dict[str, Any]:
    notes = build_focused_listening_review_notes(focused_package)
    notes["schema_version"] = "stage_b_margin_recovered_timing_repetition_focused_listening_notes_v1"
    notes["source_focused_context_decision"] = str(focused_context_decision.get("output_dir") or "")
    notes["review_context"]["focus"].extend(
        ["dead_air_repair_followup", "adjacent_pitch_repeats", "wide_interval_review"]
    )
    decisions = decisions_by_candidate(focused_context_decision)
    for candidate in notes["candidates"]:
        candidate_id = str(candidate.get("candidate_id") or "")
        decision = decisions.get(candidate_id)
        if not decision:
            raise TimingRepetitionFocusedListeningNotesError(
                f"missing focused context decision for {candidate_id}"
            )
        metrics = dict(decision.get("metrics") or {})
        context_summary = dict(decision.get("context_summary") or {})
        candidate["proxy_review"]["decision"] = str(decision.get("focused_context_decision") or "")
        candidate["proxy_review"]["decision_flags"] = list(decision.get("decision_flags") or [])
        candidate["focused_context_metrics"] = {
            "note_count": int(metrics.get("note_count", 0) or 0),
            "unique_pitch_count": int(metrics.get("unique_pitch_count", 0) or 0),
            "range": str(metrics.get("range") or ""),
            "phrase_span_beats": float(metrics.get("phrase_span_beats", 0.0) or 0.0),
            "dead_air_ratio": float(metrics.get("dead_air_ratio", 0.0) or 0.0),
            "onset_coverage_ratio": float(metrics.get("onset_coverage_ratio", 0.0) or 0.0),
            "sustained_coverage_ratio": float(metrics.get("sustained_coverage_ratio", 0.0) or 0.0),
            "adjacent_pitch_repeats": int(metrics.get("adjacent_pitch_repeats", 0) or 0),
            "duplicated_3_note_pitch_class_chunks": int(
                metrics.get("duplicated_3_note_pitch_class_chunks", 0) or 0
            ),
            "max_simultaneous_notes": int(metrics.get("max_simultaneous_notes", 0) or 0),
            "max_interval": int(metrics.get("max_interval", 0) or 0),
            "final_note": str(metrics.get("final_note") or ""),
            "final_chord": str(metrics.get("final_chord") or ""),
            "final_note_role": str(metrics.get("final_note_role") or ""),
        }
        candidate["focused_context_summary"] = {
            "has_chord_guide": bool(context_summary.get("has_chord_guide", False)),
            "has_bass_guide": bool(context_summary.get("has_bass_guide", False)),
            "has_solo_track": bool(context_summary.get("has_solo_track", False)),
        }
        candidate["review_risks"] = review_risks(metrics)
    return notes


def markdown_report(notes: dict[str, Any]) -> str:
    lines = [
        "# Stage B Timing/Repetition Focused Listening Notes",
        "",
        f"- candidate count: `{len(notes.get('candidates') or [])}`",
        f"- source focused package: `{notes.get('source_focused_review_package')}`",
        f"- source focused context decision: `{notes.get('source_focused_context_decision')}`",
        "",
        "| candidate | prior decision | listening decision | notes | unique | span | dead-air | adj repeat | max interval | final | risks |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for candidate in notes.get("candidates") or []:
        metrics = candidate.get("focused_context_metrics") if isinstance(candidate.get("focused_context_metrics"), dict) else {}
        listening = candidate.get("listening") if isinstance(candidate.get("listening"), dict) else {}
        proxy = candidate.get("proxy_review") if isinstance(candidate.get("proxy_review"), dict) else {}
        final = f"{metrics.get('final_note')} over {metrics.get('final_chord')} ({metrics.get('final_note_role')})"
        lines.append(
            "| "
            + " | ".join(
                [
                    str(candidate.get("candidate_id") or ""),
                    str(proxy.get("decision") or ""),
                    str(listening.get("decision") or ""),
                    str(metrics.get("note_count", 0)),
                    str(metrics.get("unique_pitch_count", 0)),
                    f"{float(metrics.get('phrase_span_beats', 0.0) or 0.0):.3f}",
                    f"{float(metrics.get('dead_air_ratio', 0.0) or 0.0):.3f}",
                    str(metrics.get("adjacent_pitch_repeats", 0)),
                    str(metrics.get("max_interval", 0)),
                    final,
                    ",".join(candidate.get("review_risks") or []) or "none",
                ]
            )
            + " |"
        )
    return "\n".join(lines).rstrip() + "\n"


def validate_notes(
    notes: dict[str, Any],
    *,
    expected_candidate_id: str | None,
    expected_prior_decision: str | None,
) -> dict[str, Any]:
    generic_notes = dict(notes)
    generic_notes["schema_version"] = "stage_b_focused_listening_review_notes_v1"
    summary = validate_focused_listening_review_notes(generic_notes)
    candidates = notes.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise TimingRepetitionFocusedListeningNotesError("notes must contain candidates")
    if expected_candidate_id:
        actual = [str(candidate.get("candidate_id") or "") for candidate in candidates]
        if actual != [expected_candidate_id]:
            raise TimingRepetitionFocusedListeningNotesError(
                f"expected candidate {expected_candidate_id}, got {actual}"
            )
    if expected_prior_decision:
        decisions = [str(candidate.get("proxy_review", {}).get("decision") or "") for candidate in candidates]
        if decisions != [expected_prior_decision]:
            raise TimingRepetitionFocusedListeningNotesError(
                f"expected prior decision {expected_prior_decision}, got {decisions}"
            )
    summary.update(
        {
            "candidate_ids": [str(candidate.get("candidate_id") or "") for candidate in candidates],
            "prior_decisions": [
                str(candidate.get("proxy_review", {}).get("decision") or "") for candidate in candidates
            ],
            "review_risks": {
                str(candidate.get("candidate_id") or ""): list(candidate.get("review_risks") or [])
                for candidate in candidates
            },
        }
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build timing/repetition focused listening notes")
    parser.add_argument("--focused_package", type=str, required=True)
    parser.add_argument("--focused_context_decision", type=str, required=True)
    parser.add_argument(
        "--output_root",
        type=str,
        default=str(ROOT_DIR / "outputs" / "stage_b_margin_recovered_timing_repetition_focused_listening_notes"),
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--expected_candidate_id", type=str, default="")
    parser.add_argument("--expected_prior_decision", type=str, default="")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    focused_package_path = Path(args.focused_package)
    focused_context_decision_path = Path(args.focused_context_decision)
    focused_package = read_json(focused_package_path)
    focused_package["output_dir"] = str(focused_package_path.parent)
    focused_context_decision = read_json(focused_context_decision_path)
    focused_context_decision["output_dir"] = str(focused_context_decision_path.parent)
    notes = build_timing_repetition_focused_listening_notes(focused_package, focused_context_decision)
    summary = validate_notes(
        notes,
        expected_candidate_id=str(args.expected_candidate_id or ""),
        expected_prior_decision=str(args.expected_prior_decision or ""),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    notes_path = output_dir / "focused_listening_review_notes_template.json"
    markdown_path = output_dir / "focused_listening_review_notes_template.md"
    write_json(notes_path, notes)
    write_json(output_dir / "focused_listening_review_notes_summary.json", summary)
    markdown_path.write_text(markdown_report(notes), encoding="utf-8")
    print(
        json.dumps(
            {**summary, "review_notes_path": str(notes_path), "markdown_path": str(markdown_path)},
            ensure_ascii=True,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
