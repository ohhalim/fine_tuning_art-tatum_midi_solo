"""Consolidate selected and peer phrase/vocabulary keep candidates."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class PhraseVocabularyTwoCandidateKeepError(ValueError):
    pass


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def keep_candidate(filled_notes: dict[str, Any], *, role: str) -> dict[str, Any]:
    candidates = filled_notes.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise PhraseVocabularyTwoCandidateKeepError(f"{role} filled notes must contain candidates")
    keep_candidates = [
        candidate
        for candidate in candidates
        if str(candidate.get("listening", {}).get("decision") or "") == "keep"
    ]
    if len(keep_candidates) != 1:
        raise PhraseVocabularyTwoCandidateKeepError(
            f"expected exactly one {role} keep candidate, got {len(keep_candidates)}"
        )
    return keep_candidates[0]


def compact_keep_candidate(candidate: dict[str, Any], *, role: str) -> dict[str, Any]:
    metadata = candidate.get("review_metadata") if isinstance(candidate.get("review_metadata"), dict) else {}
    listening = candidate.get("listening") if isinstance(candidate.get("listening"), dict) else {}
    focused = (
        candidate.get("focused_context_metrics")
        if isinstance(candidate.get("focused_context_metrics"), dict)
        else {}
    )
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
        "decision": str(listening.get("decision") or ""),
        "timing": str(listening.get("timing") or ""),
        "chord_fit": str(listening.get("chord_fit") or ""),
        "phrase_continuation": str(listening.get("phrase_continuation") or ""),
        "landing": str(listening.get("landing") or ""),
        "jazz_vocabulary": str(listening.get("jazz_vocabulary") or ""),
        "note_count": int(focused.get("note_count", 0) or 0),
        "unique_pitch_count": int(focused.get("unique_pitch_count", 0) or 0),
        "range": str(focused.get("range") or ""),
        "phrase_span_beats": float(focused.get("phrase_span_beats", 0.0) or 0.0),
        "max_active_notes": int(focused.get("max_simultaneous_notes", 0) or 0),
        "dead_air_ratio": float(focused.get("dead_air_ratio", 0.0) or 0.0),
        "sustained_coverage_ratio": float(focused.get("sustained_coverage_ratio", 0.0) or 0.0),
        "adjacent_pitch_repeats": int(focused.get("adjacent_pitch_repeats", 0) or 0),
        "max_interval": int(focused.get("max_interval", 0) or 0),
        "final_note": str(focused.get("final_note") or ""),
        "final_chord": str(focused.get("final_chord") or ""),
        "final_note_role": str(focused.get("final_note_role") or ""),
        "review_risks": list(candidate.get("review_risks") or []),
        "not_human_audio_review": bool(evidence.get("not_human_audio_review", False)),
    }


def build_two_candidate_keep_report(
    stability_summary: dict[str, Any],
    selected_filled_notes: dict[str, Any],
    peer_filled_notes: dict[str, Any],
    *,
    output_dir: Path,
) -> dict[str, Any]:
    selected_keep = compact_keep_candidate(keep_candidate(selected_filled_notes, role="selected"), role="selected")
    peer_keep = compact_keep_candidate(keep_candidate(peer_filled_notes, role="peer"), role="peer")

    stability_selected_id = str(stability_summary.get("selected_candidate", {}).get("candidate_id") or "")
    stability_peer_ids = {
        str(candidate.get("candidate_id") or "")
        for candidate in stability_summary.get("qualified_peers", [])
        if isinstance(candidate, dict)
    }
    if selected_keep["candidate_id"] != stability_selected_id:
        raise PhraseVocabularyTwoCandidateKeepError(
            f"selected keep does not match stability selected candidate: {selected_keep['candidate_id']}"
        )
    if peer_keep["candidate_id"] not in stability_peer_ids:
        raise PhraseVocabularyTwoCandidateKeepError(
            f"peer keep is not a qualified stability peer: {peer_keep['candidate_id']}"
        )

    keep_candidates = [selected_keep, peer_keep]
    qualified_candidate_count = int(stability_summary.get("qualified_candidate_count", 0) or 0)
    candidate_count = int(stability_summary.get("candidate_count", 0) or 0)
    qualified_rate = float(stability_summary.get("qualified_rate", 0.0) or 0.0)
    qualified_source_count = int(stability_summary.get("qualified_source_count", 0) or 0)
    not_human_audio_review = all(candidate["not_human_audio_review"] for candidate in keep_candidates)
    boundary = (
        "two_candidate_midi_context_keep_support"
        if len(keep_candidates) >= 2 and qualified_source_count >= 2
        else "insufficient_two_candidate_support"
    )

    return {
        "schema_version": "stage_b_margin_recovered_phrase_vocabulary_two_candidate_keep_v1",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "source_stability_schema": str(stability_summary.get("schema_version") or ""),
        "candidate_count": candidate_count,
        "qualified_candidate_count": qualified_candidate_count,
        "qualified_rate": round(qualified_rate, 6),
        "qualified_source_count": qualified_source_count,
        "keep_candidate_count": len(keep_candidates),
        "keep_candidates": keep_candidates,
        "evidence_boundary": {
            "boundary": boundary,
            "selected_and_peer_are_keep": all(candidate["decision"] == "keep" for candidate in keep_candidates),
            "two_source_support": qualified_source_count >= 2,
            "not_human_audio_review": not_human_audio_review,
            "claim": "two_candidate_midi_context_evidence_keep",
        },
        "not_proven": [
            "human_audio_preference",
            "broad_trained_model_quality",
            "robust_repeatability",
            "brad_style_adaptation",
            "production_ready_improviser",
        ],
        "next_recommended_issue": "Stage B margin-recovered phrase/vocabulary human listening comparison boundary",
    }


def validate_two_candidate_keep(
    report: dict[str, Any],
    *,
    min_keep_candidates: int,
    min_qualified_sources: int,
    max_qualified_rate: float | None,
    require_not_human_audio_review: bool,
) -> dict[str, Any]:
    keep_candidates = report.get("keep_candidates")
    if not isinstance(keep_candidates, list):
        raise PhraseVocabularyTwoCandidateKeepError("report keep_candidates must be a list")
    keep_count = int(report.get("keep_candidate_count", 0) or 0)
    if keep_count < min_keep_candidates:
        raise PhraseVocabularyTwoCandidateKeepError(f"keep_candidate_count {keep_count} < {min_keep_candidates}")
    non_keep = [candidate.get("candidate_id") for candidate in keep_candidates if candidate.get("decision") != "keep"]
    if non_keep:
        raise PhraseVocabularyTwoCandidateKeepError(f"non-keep candidates found: {non_keep}")
    source_count = int(report.get("qualified_source_count", 0) or 0)
    if source_count < min_qualified_sources:
        raise PhraseVocabularyTwoCandidateKeepError(
            f"qualified_source_count {source_count} < {min_qualified_sources}"
        )
    qualified_rate = float(report.get("qualified_rate", 0.0) or 0.0)
    if max_qualified_rate is not None and qualified_rate > max_qualified_rate:
        raise PhraseVocabularyTwoCandidateKeepError(
            f"qualified_rate {qualified_rate:.6f} > {max_qualified_rate:.6f}"
        )
    if require_not_human_audio_review and not bool(
        report.get("evidence_boundary", {}).get("not_human_audio_review", False)
    ):
        raise PhraseVocabularyTwoCandidateKeepError("expected not_human_audio_review boundary")
    return {
        "keep_candidate_count": keep_count,
        "qualified_candidate_count": int(report.get("qualified_candidate_count", 0) or 0),
        "qualified_source_count": source_count,
        "qualified_rate": qualified_rate,
        "boundary": str(report.get("evidence_boundary", {}).get("boundary") or ""),
        "next_recommended_issue": str(report.get("next_recommended_issue") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Stage B Margin-Recovered Phrase/Vocabulary Two-Candidate Keep",
        "",
        f"- keep candidates: `{report['keep_candidate_count']}`",
        f"- qualified candidates: `{report['qualified_candidate_count']}/{report['candidate_count']}`",
        f"- qualified rate: `{report['qualified_rate']}`",
        f"- qualified source count: `{report['qualified_source_count']}`",
        f"- boundary: `{report['evidence_boundary']['boundary']}`",
        "",
        "This is MIDI/context evidence support, not human audio proof or broad model quality.",
        "",
        "| role | candidate | source | sample | decision | timing | phrase | vocabulary | notes | unique | dead-air | sustained | adj repeat | max interval | final landing | risk |",
        "|---|---|---|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for candidate in report.get("keep_candidates", []):
        final_landing = f"{candidate['final_note']} over {candidate['final_chord']} ({candidate['final_note_role']})"
        risk = ",".join(candidate["review_risks"]) if candidate["review_risks"] else "none"
        lines.append(
            "| "
            + " | ".join(
                [
                    str(candidate["role"]),
                    str(candidate["candidate_id"]),
                    str(candidate["source_run_id"]),
                    str(candidate["sample_index"]),
                    str(candidate["decision"]),
                    str(candidate["timing"]),
                    str(candidate["phrase_continuation"]),
                    str(candidate["jazz_vocabulary"]),
                    str(candidate["note_count"]),
                    str(candidate["unique_pitch_count"]),
                    f"{float(candidate['dead_air_ratio']):.3f}",
                    f"{float(candidate['sustained_coverage_ratio']):.3f}",
                    str(candidate["adjacent_pitch_repeats"]),
                    str(candidate["max_interval"]),
                    final_landing,
                    risk,
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Not Proven",
            "",
        ]
    )
    for item in report.get("not_proven", []):
        lines.append(f"- `{item}`")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize two phrase/vocabulary keep candidates")
    parser.add_argument("--stability_summary", type=str, required=True)
    parser.add_argument("--selected_filled_notes", type=str, required=True)
    parser.add_argument("--peer_filled_notes", type=str, required=True)
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/stage_b_margin_recovered_phrase_vocabulary_two_candidate_keep",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--min_keep_candidates", type=int, default=2)
    parser.add_argument("--min_qualified_sources", type=int, default=2)
    parser.add_argument("--max_qualified_rate", type=float, default=None)
    parser.add_argument("--require_not_human_audio_review", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    report = build_two_candidate_keep_report(
        read_json(Path(args.stability_summary)),
        read_json(Path(args.selected_filled_notes)),
        read_json(Path(args.peer_filled_notes)),
        output_dir=output_dir,
    )
    summary = validate_two_candidate_keep(
        report,
        min_keep_candidates=int(args.min_keep_candidates),
        min_qualified_sources=int(args.min_qualified_sources),
        max_qualified_rate=args.max_qualified_rate,
        require_not_human_audio_review=bool(args.require_not_human_audio_review),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "two_candidate_keep_summary.json"
    markdown_path = output_dir / "two_candidate_keep_summary.md"
    write_json(report_path, report)
    write_json(output_dir / "two_candidate_keep_validation_summary.json", summary)
    markdown_path.write_text(markdown_report(report), encoding="utf-8")
    print(json.dumps({**summary, "report_path": str(report_path), "markdown_path": str(markdown_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
