"""Summarize stability of the margin-recovered phrase/vocabulary evidence keep candidate."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class PhraseVocabularyKeepStabilityError(ValueError):
    pass


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def candidates_by_id(candidates: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(candidate.get("candidate_id") or ""): candidate for candidate in candidates}


def compact_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    focused = candidate.get("focused_solo_metrics") if isinstance(candidate.get("focused_solo_metrics"), dict) else {}
    metrics = candidate.get("metrics") if isinstance(candidate.get("metrics"), dict) else {}
    gate = candidate.get("phrase_vocabulary_gate") if isinstance(candidate.get("phrase_vocabulary_gate"), dict) else {}
    return {
        "candidate_id": str(candidate.get("candidate_id") or ""),
        "source_run_id": str(candidate.get("source_run_id") or ""),
        "sample_index": int(candidate.get("sample_index", 0) or 0),
        "sample_seed": int(candidate.get("sample_seed", 0) or 0),
        "repair_rank": int(candidate.get("repair_rank", 0) or 0),
        "qualified": bool(gate.get("qualified", False)),
        "flags": list(gate.get("flags") or []),
        "focused_note_count": int(focused.get("focused_note_count", 0) or 0),
        "focused_unique_pitch_count": int(focused.get("focused_unique_pitch_count", 0) or 0),
        "focused_adjacent_pitch_repeats": int(focused.get("focused_adjacent_pitch_repeats", 0) or 0),
        "focused_max_interval": int(focused.get("focused_max_interval", 0) or 0),
        "focused_duplicated_3_note_chunks": int(
            focused.get("focused_duplicated_3_note_pitch_class_chunks", 0) or 0
        ),
        "dead_air_ratio": float(metrics.get("dead_air_ratio", 0.0) or 0.0),
    }


def filled_keep_candidate(filled_notes: dict[str, Any]) -> dict[str, Any]:
    candidates = filled_notes.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise PhraseVocabularyKeepStabilityError("filled notes must contain candidates")
    keep_candidates = [
        candidate
        for candidate in candidates
        if str(candidate.get("listening", {}).get("decision") or "") == "keep"
    ]
    if len(keep_candidates) != 1:
        raise PhraseVocabularyKeepStabilityError(f"expected exactly one filled keep candidate, got {len(keep_candidates)}")
    return keep_candidates[0]


def stability_boundary(*, selected_is_filled_keep: bool, qualified_count: int, qualified_source_count: int) -> str:
    if not selected_is_filled_keep:
        return "no_filled_keep"
    if qualified_count >= 2 and qualified_source_count >= 2:
        return "narrow_two_source_candidate_support"
    if qualified_count >= 2:
        return "same_source_peer_support"
    return "single_candidate_support_only"


def build_stability_report(
    repair_summary: dict[str, Any],
    filled_notes: dict[str, Any],
    *,
    output_dir: Path,
) -> dict[str, Any]:
    candidates = repair_summary.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise PhraseVocabularyKeepStabilityError("repair summary must contain candidates")
    filled_keep = filled_keep_candidate(filled_notes)
    selected_id = str(filled_keep.get("candidate_id") or "")
    by_id = candidates_by_id(candidates)
    selected = by_id.get(selected_id)
    if selected is None:
        raise PhraseVocabularyKeepStabilityError(f"filled keep candidate not found in repair summary: {selected_id}")
    qualified = [
        candidate
        for candidate in candidates
        if bool(candidate.get("phrase_vocabulary_gate", {}).get("qualified", False))
    ]
    qualified_sources = Counter(str(candidate.get("source_run_id") or "") for candidate in qualified)
    peers = [candidate for candidate in qualified if str(candidate.get("candidate_id") or "") != selected_id]
    candidate_count = int(repair_summary.get("candidate_count", len(candidates)) or len(candidates))
    qualified_count = int(len(qualified))
    source_count = int(len(qualified_sources))
    filled_listening = filled_keep.get("listening") if isinstance(filled_keep.get("listening"), dict) else {}
    selected_is_filled_keep = str(filled_listening.get("decision") or "") == "keep"
    return {
        "schema_version": "stage_b_margin_recovered_phrase_vocabulary_keep_stability_v1",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "source_repair_summary": str(repair_summary.get("output_dir") or ""),
        "source_filled_notes_schema": str(filled_notes.get("schema_version") or ""),
        "candidate_count": candidate_count,
        "qualified_candidate_count": qualified_count,
        "qualified_rate": round(qualified_count / max(1, candidate_count), 6),
        "qualified_source_count": source_count,
        "qualified_sources": dict(sorted(qualified_sources.items())),
        "selected_candidate": compact_candidate(selected),
        "filled_keep_decision": {
            "candidate_id": selected_id,
            "decision": str(filled_listening.get("decision") or ""),
            "timing": str(filled_listening.get("timing") or ""),
            "phrase_continuation": str(filled_listening.get("phrase_continuation") or ""),
            "jazz_vocabulary": str(filled_listening.get("jazz_vocabulary") or ""),
            "not_human_audio_review": bool(
                filled_keep.get("listening_fill_evidence", {}).get("not_human_audio_review", False)
            ),
        },
        "qualified_peer_count": int(len(peers)),
        "qualified_peers": [compact_candidate(candidate) for candidate in peers],
        "stability_summary": {
            "selected_is_filled_keep": bool(selected_is_filled_keep),
            "has_qualified_peer": bool(peers),
            "boundary": stability_boundary(
                selected_is_filled_keep=selected_is_filled_keep,
                qualified_count=qualified_count,
                qualified_source_count=source_count,
            ),
            "next_recommended_issue": (
                "Stage B margin-recovered phrase/vocabulary qualified peer focused context review"
                if peers
                else "Stage B margin-recovered phrase/vocabulary broader stability sweep"
            ),
        },
    }


def validate_stability(
    report: dict[str, Any],
    *,
    expected_selected_candidate_id: str | None,
    min_qualified_candidates: int,
    require_qualified_peer: bool,
) -> dict[str, Any]:
    selected_id = str(report.get("selected_candidate", {}).get("candidate_id") or "")
    if expected_selected_candidate_id and selected_id != expected_selected_candidate_id:
        raise PhraseVocabularyKeepStabilityError(
            f"expected selected candidate {expected_selected_candidate_id}, got {selected_id}"
        )
    qualified_count = int(report.get("qualified_candidate_count", 0) or 0)
    if qualified_count < int(min_qualified_candidates):
        raise PhraseVocabularyKeepStabilityError(
            f"qualified_candidate_count {qualified_count} < {min_qualified_candidates}"
        )
    peer_count = int(report.get("qualified_peer_count", 0) or 0)
    if require_qualified_peer and peer_count < 1:
        raise PhraseVocabularyKeepStabilityError("expected at least one qualified peer")
    if not bool(report.get("stability_summary", {}).get("selected_is_filled_keep", False)):
        raise PhraseVocabularyKeepStabilityError("selected candidate must be a filled keep")
    return {
        "selected_candidate_id": selected_id,
        "qualified_candidate_count": qualified_count,
        "qualified_peer_count": peer_count,
        "qualified_source_count": int(report.get("qualified_source_count", 0) or 0),
        "boundary": str(report.get("stability_summary", {}).get("boundary") or ""),
        "next_recommended_issue": str(report.get("stability_summary", {}).get("next_recommended_issue") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    selected = report["selected_candidate"]
    lines = [
        "# Stage B Margin-Recovered Phrase/Vocabulary Keep Stability",
        "",
        f"- candidate count: `{report['candidate_count']}`",
        f"- qualified candidates: `{report['qualified_candidate_count']}`",
        f"- qualified rate: `{report['qualified_rate']}`",
        f"- qualified sources: `{report['qualified_sources']}`",
        f"- selected candidate: `{selected['candidate_id']}`",
        f"- stability boundary: `{report['stability_summary']['boundary']}`",
        "",
        "This is a sweep stability comparison, not a broad model-quality claim.",
        "",
        "| candidate | source | rank | sample | notes | unique | dead-air | adj repeat | max interval | flags |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    rows = [selected] + list(report.get("qualified_peers") or [])
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["candidate_id"]),
                    str(row["source_run_id"]),
                    str(row["repair_rank"]),
                    str(row["sample_index"]),
                    str(row["focused_note_count"]),
                    str(row["focused_unique_pitch_count"]),
                    f"{float(row['dead_air_ratio']):.3f}",
                    str(row["focused_adjacent_pitch_repeats"]),
                    str(row["focused_max_interval"]),
                    ",".join(row["flags"]) if row["flags"] else "ok",
                ]
            )
            + " |"
        )
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize phrase/vocabulary keep stability")
    parser.add_argument("--repair_summary", type=str, required=True)
    parser.add_argument("--filled_notes", type=str, required=True)
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/stage_b_margin_recovered_phrase_vocabulary_keep_stability",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--expected_selected_candidate_id", type=str, default="")
    parser.add_argument("--min_qualified_candidates", type=int, default=1)
    parser.add_argument("--require_qualified_peer", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    report = build_stability_report(
        read_json(Path(args.repair_summary)),
        read_json(Path(args.filled_notes)),
        output_dir=output_dir,
    )
    summary = validate_stability(
        report,
        expected_selected_candidate_id=str(args.expected_selected_candidate_id or ""),
        min_qualified_candidates=int(args.min_qualified_candidates),
        require_qualified_peer=bool(args.require_qualified_peer),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "keep_stability_summary.json"
    markdown_path = output_dir / "keep_stability_summary.md"
    write_json(report_path, report)
    write_json(output_dir / "keep_stability_validation_summary.json", summary)
    markdown_path.write_text(markdown_report(report), encoding="utf-8")
    print(json.dumps({**summary, "report_path": str(report_path), "markdown_path": str(markdown_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
