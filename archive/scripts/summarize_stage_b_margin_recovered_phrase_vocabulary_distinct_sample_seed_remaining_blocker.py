"""Summarize remaining blockers for the distinct sample-seed phrase/vocabulary candidate."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class DistinctSampleSeedRemainingBlockerError(ValueError):
    pass


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def only_candidate(filled_notes: dict[str, Any]) -> dict[str, Any]:
    candidates = filled_notes.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 1:
        raise DistinctSampleSeedRemainingBlockerError("filled notes must contain exactly one candidate")
    candidate = candidates[0]
    if not isinstance(candidate, dict):
        raise DistinctSampleSeedRemainingBlockerError("candidate must be an object")
    return candidate


def remaining_blockers(candidate: dict[str, Any]) -> list[str]:
    metrics = candidate.get("focused_context_metrics") if isinstance(candidate.get("focused_context_metrics"), dict) else {}
    listening = candidate.get("listening") if isinstance(candidate.get("listening"), dict) else {}
    blockers: list[str] = []
    if str(listening.get("phrase_continuation") or "") != "acceptable":
        blockers.append("phrase_continuation_weak")
    if str(listening.get("jazz_vocabulary") or "") != "acceptable":
        blockers.append("jazz_vocabulary_thin")
    if float(metrics.get("phrase_span_beats", 0.0) or 0.0) < 7.0:
        blockers.append("short_phrase_span")
    if int(metrics.get("unique_pitch_count", 0) or 0) < 7:
        blockers.append("pitch_variety_floor")
    if int(metrics.get("adjacent_pitch_repeats", 0) or 0) > 0:
        blockers.append("adjacent_pitch_repeats")
    return blockers


def secondary_risks(candidate: dict[str, Any]) -> list[str]:
    metrics = candidate.get("focused_context_metrics") if isinstance(candidate.get("focused_context_metrics"), dict) else {}
    risks = []
    if float(metrics.get("dead_air_ratio", 0.0) or 0.0) >= 0.35:
        risks.append("dead_air_ratio_remaining")
    if str(metrics.get("final_note_role") or "") not in {"chord_tone", "tension"}:
        risks.append("final_landing_context_risk")
    if int(metrics.get("max_interval", 0) or 0) >= 12:
        risks.append("wide_interval_review")
    return risks


def compact_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    metadata = candidate.get("review_metadata") if isinstance(candidate.get("review_metadata"), dict) else {}
    metrics = candidate.get("focused_context_metrics") if isinstance(candidate.get("focused_context_metrics"), dict) else {}
    listening = candidate.get("listening") if isinstance(candidate.get("listening"), dict) else {}
    evidence = candidate.get("listening_fill_evidence") if isinstance(candidate.get("listening_fill_evidence"), dict) else {}
    return {
        "candidate_id": str(candidate.get("candidate_id") or ""),
        "source_run_id": str(metadata.get("source_run_id") or ""),
        "sample_index": int(metadata.get("sample_index", 0) or 0),
        "sample_seed": int(metadata.get("sample_seed", 0) or 0),
        "prior_decision": str(candidate.get("proxy_review", {}).get("decision") or ""),
        "final_decision": str(listening.get("decision") or ""),
        "timing": str(listening.get("timing") or ""),
        "chord_fit": str(listening.get("chord_fit") or ""),
        "phrase_continuation": str(listening.get("phrase_continuation") or ""),
        "landing": str(listening.get("landing") or ""),
        "jazz_vocabulary": str(listening.get("jazz_vocabulary") or ""),
        "note_count": int(metrics.get("note_count", 0) or 0),
        "unique_pitch_count": int(metrics.get("unique_pitch_count", 0) or 0),
        "range": str(metrics.get("range") or ""),
        "phrase_span_beats": float(metrics.get("phrase_span_beats", 0.0) or 0.0),
        "dead_air_ratio": float(metrics.get("dead_air_ratio", 0.0) or 0.0),
        "onset_coverage_ratio": float(metrics.get("onset_coverage_ratio", 0.0) or 0.0),
        "sustained_coverage_ratio": float(metrics.get("sustained_coverage_ratio", 0.0) or 0.0),
        "adjacent_pitch_repeats": int(metrics.get("adjacent_pitch_repeats", 0) or 0),
        "max_interval": int(metrics.get("max_interval", 0) or 0),
        "max_simultaneous_notes": int(metrics.get("max_simultaneous_notes", 0) or 0),
        "final_note": str(metrics.get("final_note") or ""),
        "final_chord": str(metrics.get("final_chord") or ""),
        "final_note_role": str(metrics.get("final_note_role") or ""),
        "not_human_audio_review": bool(evidence.get("not_human_audio_review", True)),
        "review_risks": list(candidate.get("review_risks") or []),
    }


def build_remaining_blocker_report(
    filled_notes: dict[str, Any],
    *,
    output_dir: Path,
) -> dict[str, Any]:
    candidate = only_candidate(filled_notes)
    compact = compact_candidate(candidate)
    blockers = remaining_blockers(candidate)
    risks = secondary_risks(candidate)
    keep_guardrails = {
        "prior_context_decision_keep": compact["prior_decision"] == "keep_for_focused_listening",
        "timing_acceptable": compact["timing"] == "acceptable",
        "landing_acceptable": compact["landing"] in {"strong", "acceptable"},
        "final_role_context_safe": compact["final_note_role"] in {"chord_tone", "tension"},
        "max_active_solo_line": compact["max_simultaneous_notes"] <= 1,
        "wide_interval_guardrail": compact["max_interval"] < 12,
    }
    return {
        "schema_version": "stage_b_margin_recovered_phrase_vocabulary_distinct_sample_seed_remaining_blocker_v1",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "source_filled_notes": str(filled_notes.get("output_dir") or ""),
        "candidate": compact,
        "remaining_blockers": blockers,
        "secondary_risks": risks,
        "keep_guardrails": keep_guardrails,
        "repair_target": {
            "boundary": "distinct_sample_seed_candidate_needs_phrase_vocabulary_repair",
            "action": "repair_phrase_span_pitch_variety_adjacent_repeat",
            "target_phrase_span_beats_min": 7.0,
            "target_unique_pitch_count_min": 7,
            "target_adjacent_pitch_repeats_max": 0,
            "preferred_dead_air_ratio_max": 0.35,
            "preserve_max_interval_below": 12,
            "preserve_max_simultaneous_notes_max": 1,
            "preserve_final_note_role": ["chord_tone", "tension"],
            "avoid_sample_seed": [85],
        },
        "claim_boundary": {
            "current": "distinct_sample_seed_context_keep_but_listening_fill_needs_followup",
            "not_proven": [
                "human_audio_preference",
                "broad_trained_model_quality",
                "brad_style_adaptation",
            ],
        },
        "next_recommended_issue": (
            "Stage B margin-recovered phrase/vocabulary distinct sample-seed remaining blocker repair sweep"
        ),
    }


def validate_remaining_blocker_report(
    report: dict[str, Any],
    *,
    expected_decision: str | None,
    require_remaining_blockers: bool,
) -> dict[str, Any]:
    candidate = report.get("candidate") if isinstance(report.get("candidate"), dict) else {}
    final_decision = str(candidate.get("final_decision") or "")
    if expected_decision and final_decision != expected_decision:
        raise DistinctSampleSeedRemainingBlockerError(f"expected decision {expected_decision}, got {final_decision}")
    blockers = list(report.get("remaining_blockers") or [])
    if require_remaining_blockers and not blockers:
        raise DistinctSampleSeedRemainingBlockerError("expected remaining blockers")
    return {
        "candidate_id": str(candidate.get("candidate_id") or ""),
        "sample_seed": int(candidate.get("sample_seed", 0) or 0),
        "final_decision": final_decision,
        "remaining_blockers": blockers,
        "secondary_risks": list(report.get("secondary_risks") or []),
        "repair_boundary": str(report.get("repair_target", {}).get("boundary") or ""),
        "next_recommended_issue": str(report.get("next_recommended_issue") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    candidate = report["candidate"]
    target = report["repair_target"]
    lines = [
        "# Stage B Margin-Recovered Phrase/Vocabulary Distinct Sample-Seed Remaining Blocker",
        "",
        f"- candidate: `{candidate['candidate_id']}`",
        f"- sample seed: `{candidate['sample_seed']}`",
        f"- final decision: `{candidate['final_decision']}`",
        f"- remaining blockers: `{report['remaining_blockers']}`",
        f"- secondary risks: `{report['secondary_risks']}`",
        f"- repair boundary: `{target['boundary']}`",
        "",
        "| metric | value | target |",
        "|---|---:|---:|",
        f"| phrase span beats | `{candidate['phrase_span_beats']:.3f}` | `>= {target['target_phrase_span_beats_min']:.1f}` |",
        f"| unique pitch count | `{candidate['unique_pitch_count']}` | `>= {target['target_unique_pitch_count_min']}` |",
        f"| adjacent pitch repeats | `{candidate['adjacent_pitch_repeats']}` | `<= {target['target_adjacent_pitch_repeats_max']}` |",
        f"| dead-air ratio | `{candidate['dead_air_ratio']:.3f}` | `<= {target['preferred_dead_air_ratio_max']:.2f}` |",
        f"| max interval | `{candidate['max_interval']}` | `< {target['preserve_max_interval_below']}` |",
        "",
        "This is a repair target summary, not a new generated-quality claim.",
    ]
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize distinct sample-seed remaining blockers")
    parser.add_argument("--filled_notes", type=str, required=True)
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/stage_b_margin_recovered_phrase_vocabulary_distinct_sample_seed_remaining_blocker",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--expected_decision", type=str, default="")
    parser.add_argument("--require_remaining_blockers", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    filled_notes_path = Path(args.filled_notes)
    filled_notes = read_json(filled_notes_path)
    filled_notes["output_dir"] = str(filled_notes_path.parent)
    report = build_remaining_blocker_report(filled_notes, output_dir=output_dir)
    summary = validate_remaining_blocker_report(
        report,
        expected_decision=str(args.expected_decision or ""),
        require_remaining_blockers=bool(args.require_remaining_blockers),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "remaining_blocker_summary.json"
    markdown_path = output_dir / "remaining_blocker_summary.md"
    write_json(report_path, report)
    write_json(output_dir / "remaining_blocker_validation_summary.json", summary)
    markdown_path.write_text(markdown_report(report), encoding="utf-8")
    print(json.dumps({**summary, "report_path": str(report_path), "markdown_path": str(markdown_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
