"""Build a human-listening comparison boundary for two phrase/vocabulary keep candidates."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class PhraseVocabularyHumanListeningComparisonError(ValueError):
    pass


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def keep_candidate(filled_notes: dict[str, Any], *, role: str) -> dict[str, Any]:
    candidates = filled_notes.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise PhraseVocabularyHumanListeningComparisonError(f"{role} filled notes must contain candidates")
    keep_candidates = [
        candidate
        for candidate in candidates
        if str(candidate.get("listening", {}).get("decision") or "") == "keep"
    ]
    if len(keep_candidates) != 1:
        raise PhraseVocabularyHumanListeningComparisonError(
            f"expected exactly one {role} keep candidate, got {len(keep_candidates)}"
        )
    return keep_candidates[0]


def note_signature(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    notes = candidate.get("objective_first_16_notes")
    if not isinstance(notes, list):
        return []
    signature = []
    for note in notes:
        if not isinstance(note, dict):
            continue
        signature.append(
            {
                "pitch": int(note.get("pitch", 0) or 0),
                "start_sec": round(float(note.get("start_sec", 0.0) or 0.0), 6),
                "duration_sec": round(
                    float(note.get("end_sec", 0.0) or 0.0) - float(note.get("start_sec", 0.0) or 0.0),
                    6,
                ),
                "velocity": int(note.get("velocity", 0) or 0),
            }
        )
    return signature


def metric_fingerprint(candidate: dict[str, Any]) -> dict[str, Any]:
    metrics = (
        candidate.get("focused_context_metrics")
        if isinstance(candidate.get("focused_context_metrics"), dict)
        else {}
    )
    return {
        "note_count": int(metrics.get("note_count", 0) or 0),
        "unique_pitch_count": int(metrics.get("unique_pitch_count", 0) or 0),
        "range": str(metrics.get("range") or ""),
        "phrase_span_beats": round(float(metrics.get("phrase_span_beats", 0.0) or 0.0), 3),
        "dead_air_ratio": round(float(metrics.get("dead_air_ratio", 0.0) or 0.0), 3),
        "sustained_coverage_ratio": round(float(metrics.get("sustained_coverage_ratio", 0.0) or 0.0), 3),
        "adjacent_pitch_repeats": int(metrics.get("adjacent_pitch_repeats", 0) or 0),
        "max_interval": int(metrics.get("max_interval", 0) or 0),
        "final_note": str(metrics.get("final_note") or ""),
        "final_chord": str(metrics.get("final_chord") or ""),
        "final_note_role": str(metrics.get("final_note_role") or ""),
    }


def compact_candidate(candidate: dict[str, Any], *, role: str) -> dict[str, Any]:
    metadata = candidate.get("review_metadata") if isinstance(candidate.get("review_metadata"), dict) else {}
    files = candidate.get("review_files") if isinstance(candidate.get("review_files"), dict) else {}
    listening = candidate.get("listening") if isinstance(candidate.get("listening"), dict) else {}
    evidence = (
        candidate.get("listening_fill_evidence")
        if isinstance(candidate.get("listening_fill_evidence"), dict)
        else {}
    )
    return {
        "role": role,
        "candidate_id": str(candidate.get("candidate_id") or ""),
        "source_run_id": str(metadata.get("source_run_id") or ""),
        "sample_index": int(metadata.get("sample_index", 0) or 0),
        "sample_seed": int(metadata.get("sample_seed", 0) or 0),
        "midi_path": str(files.get("midi_path") or ""),
        "context_midi_path": str(files.get("context_midi_path") or ""),
        "source_midi_path": str(files.get("source_midi_path") or ""),
        "prior_evidence_decision": str(listening.get("decision") or ""),
        "prior_timing": str(listening.get("timing") or ""),
        "prior_chord_fit": str(listening.get("chord_fit") or ""),
        "prior_phrase_continuation": str(listening.get("phrase_continuation") or ""),
        "prior_landing": str(listening.get("landing") or ""),
        "prior_jazz_vocabulary": str(listening.get("jazz_vocabulary") or ""),
        "metric_fingerprint": metric_fingerprint(candidate),
        "note_signature": note_signature(candidate),
        "review_risks": list(candidate.get("review_risks") or []),
        "not_human_audio_review": bool(evidence.get("not_human_audio_review", False)),
        "human_listening": {
            "status": "pending",
            "comparison_preference": "pending",
            "timing_preference": "pending",
            "phrase_preference": "pending",
            "vocabulary_preference": "pending",
            "audio_render_used": "",
            "reviewer": "",
            "notes": "",
        },
    }


def build_human_listening_comparison(
    two_candidate_keep: dict[str, Any],
    selected_filled_notes: dict[str, Any],
    peer_filled_notes: dict[str, Any],
    *,
    output_dir: Path,
) -> dict[str, Any]:
    selected = compact_candidate(keep_candidate(selected_filled_notes, role="selected"), role="selected")
    peer = compact_candidate(keep_candidate(peer_filled_notes, role="peer"), role="peer")
    keep_ids = {
        str(candidate.get("candidate_id") or "")
        for candidate in two_candidate_keep.get("keep_candidates", [])
        if isinstance(candidate, dict)
    }
    for candidate in (selected, peer):
        if candidate["candidate_id"] not in keep_ids:
            raise PhraseVocabularyHumanListeningComparisonError(
                f"candidate not found in two-candidate keep summary: {candidate['candidate_id']}"
            )

    note_sequence_match = selected["note_signature"] == peer["note_signature"]
    metric_fingerprint_match = selected["metric_fingerprint"] == peer["metric_fingerprint"]
    complete_signature = all(
        len(candidate["note_signature"]) >= candidate["metric_fingerprint"]["note_count"]
        for candidate in (selected, peer)
    )
    if note_sequence_match and complete_signature:
        comparison_boundary = "pending_human_review_same_midi_content"
        listenability = "not_meaningful_as_ab_if_same_render"
    else:
        comparison_boundary = "pending_human_review_distinct_midi_content"
        listenability = "requires_audio_render_or_human_listening"

    return {
        "schema_version": "stage_b_margin_recovered_phrase_vocabulary_human_listening_comparison_v1",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "source_two_candidate_keep_schema": str(two_candidate_keep.get("schema_version") or ""),
        "candidates": [selected, peer],
        "objective_comparison": {
            "note_sequence_match": note_sequence_match,
            "metric_fingerprint_match": metric_fingerprint_match,
            "complete_note_signature": complete_signature,
            "selected_note_signature_count": len(selected["note_signature"]),
            "peer_note_signature_count": len(peer["note_signature"]),
            "candidate_note_count": selected["metric_fingerprint"]["note_count"],
        },
        "human_listening_boundary": {
            "boundary": comparison_boundary,
            "listenability": listenability,
            "status": "pending",
            "preference_claimed": False,
            "not_human_audio_review": True,
            "requires_audio_render_or_human_review": True,
        },
        "not_proven": [
            "human_audio_preference",
            "audio_rendered_quality",
            "broad_trained_model_quality",
            "robust_repeatability",
            "brad_style_adaptation",
        ],
        "next_recommended_issue": "Stage B margin-recovered phrase/vocabulary duplicate-candidate source divergence audit"
        if note_sequence_match
        else "Stage B margin-recovered phrase/vocabulary human listening fill",
    }


def validate_human_listening_comparison(
    report: dict[str, Any],
    *,
    min_candidates: int,
    require_pending: bool,
    require_no_preference: bool,
    expect_note_sequence_match: bool | None,
) -> dict[str, Any]:
    candidates = report.get("candidates")
    if not isinstance(candidates, list):
        raise PhraseVocabularyHumanListeningComparisonError("report candidates must be a list")
    if len(candidates) < min_candidates:
        raise PhraseVocabularyHumanListeningComparisonError(f"candidate count {len(candidates)} < {min_candidates}")
    if require_pending:
        statuses = [str(candidate.get("human_listening", {}).get("status") or "") for candidate in candidates]
        if any(status != "pending" for status in statuses):
            raise PhraseVocabularyHumanListeningComparisonError(f"expected pending human statuses, got {statuses}")
    preference_claimed = bool(report.get("human_listening_boundary", {}).get("preference_claimed", True))
    if require_no_preference and preference_claimed:
        raise PhraseVocabularyHumanListeningComparisonError("human preference must not be claimed")
    note_sequence_match = bool(report.get("objective_comparison", {}).get("note_sequence_match", False))
    if expect_note_sequence_match is not None and note_sequence_match != expect_note_sequence_match:
        raise PhraseVocabularyHumanListeningComparisonError(
            f"note_sequence_match expected {expect_note_sequence_match}, got {note_sequence_match}"
        )
    return {
        "candidate_count": len(candidates),
        "human_statuses": [
            str(candidate.get("human_listening", {}).get("status") or "") for candidate in candidates
        ],
        "note_sequence_match": note_sequence_match,
        "metric_fingerprint_match": bool(
            report.get("objective_comparison", {}).get("metric_fingerprint_match", False)
        ),
        "boundary": str(report.get("human_listening_boundary", {}).get("boundary") or ""),
        "next_recommended_issue": str(report.get("next_recommended_issue") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    objective = report["objective_comparison"]
    boundary = report["human_listening_boundary"]
    lines = [
        "# Stage B Margin-Recovered Phrase/Vocabulary Human Listening Comparison",
        "",
        f"- candidates: `{len(report.get('candidates') or [])}`",
        f"- human status: `{boundary['status']}`",
        f"- boundary: `{boundary['boundary']}`",
        f"- note sequence match: `{objective['note_sequence_match']}`",
        f"- metric fingerprint match: `{objective['metric_fingerprint_match']}`",
        "",
        "This file prepares a human/audio review boundary. It does not claim a listening preference.",
        "",
        "| role | candidate | source | sample | prior decision | human status | notes | unique | dead-air | sustained | final landing | risk |",
        "|---|---|---|---:|---|---|---:|---:|---:|---:|---|---|",
    ]
    for candidate in report.get("candidates", []):
        metrics = candidate["metric_fingerprint"]
        final = f"{metrics['final_note']} over {metrics['final_chord']} ({metrics['final_note_role']})"
        risk = ",".join(candidate["review_risks"]) if candidate["review_risks"] else "none"
        lines.append(
            "| "
            + " | ".join(
                [
                    str(candidate["role"]),
                    str(candidate["candidate_id"]),
                    str(candidate["source_run_id"]),
                    str(candidate["sample_index"]),
                    str(candidate["prior_evidence_decision"]),
                    str(candidate["human_listening"]["status"]),
                    str(metrics["note_count"]),
                    str(metrics["unique_pitch_count"]),
                    f"{float(metrics['dead_air_ratio']):.3f}",
                    f"{float(metrics['sustained_coverage_ratio']):.3f}",
                    final,
                    risk,
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Review Boundary",
            "",
            f"- preference claimed: `{boundary['preference_claimed']}`",
            f"- listenability: `{boundary['listenability']}`",
            f"- next recommended issue: `{report['next_recommended_issue']}`",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build human listening comparison boundary")
    parser.add_argument("--two_candidate_keep", type=str, required=True)
    parser.add_argument("--selected_filled_notes", type=str, required=True)
    parser.add_argument("--peer_filled_notes", type=str, required=True)
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/stage_b_margin_recovered_phrase_vocabulary_human_listening_comparison",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--min_candidates", type=int, default=2)
    parser.add_argument("--require_pending", action="store_true")
    parser.add_argument("--require_no_preference", action="store_true")
    parser.add_argument("--expect_note_sequence_match", action="store_true")
    parser.add_argument("--expect_note_sequence_diff", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.expect_note_sequence_match and args.expect_note_sequence_diff:
        raise PhraseVocabularyHumanListeningComparisonError(
            "only one of --expect_note_sequence_match or --expect_note_sequence_diff is allowed"
        )
    expected_match: bool | None = None
    if args.expect_note_sequence_match:
        expected_match = True
    elif args.expect_note_sequence_diff:
        expected_match = False
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    report = build_human_listening_comparison(
        read_json(Path(args.two_candidate_keep)),
        read_json(Path(args.selected_filled_notes)),
        read_json(Path(args.peer_filled_notes)),
        output_dir=output_dir,
    )
    summary = validate_human_listening_comparison(
        report,
        min_candidates=int(args.min_candidates),
        require_pending=bool(args.require_pending),
        require_no_preference=bool(args.require_no_preference),
        expect_note_sequence_match=expected_match,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "human_listening_comparison_boundary.json"
    markdown_path = output_dir / "human_listening_comparison_boundary.md"
    write_json(report_path, report)
    write_json(output_dir / "human_listening_comparison_validation_summary.json", summary)
    markdown_path.write_text(markdown_report(report), encoding="utf-8")
    print(json.dumps({**summary, "report_path": str(report_path), "markdown_path": str(markdown_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
