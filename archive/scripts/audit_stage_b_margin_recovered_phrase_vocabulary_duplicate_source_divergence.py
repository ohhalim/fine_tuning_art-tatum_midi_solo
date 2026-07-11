"""Audit duplicate-output source divergence for phrase/vocabulary keep candidates."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class PhraseVocabularyDuplicateSourceDivergenceError(ValueError):
    pass


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def candidates_by_id(candidates: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(candidate.get("candidate_id") or ""): candidate for candidate in candidates}


def metric_fingerprint(candidate: dict[str, Any]) -> dict[str, Any]:
    metrics = candidate.get("metrics") if isinstance(candidate.get("metrics"), dict) else {}
    focused = candidate.get("focused_solo_metrics") if isinstance(candidate.get("focused_solo_metrics"), dict) else {}
    return {
        "note_count": int(metrics.get("note_count", 0) or 0),
        "unique_pitch_count": int(metrics.get("unique_pitch_count", 0) or 0),
        "dead_air_ratio": round(float(metrics.get("dead_air_ratio", 0.0) or 0.0), 6),
        "phrase_coverage_ratio": round(float(metrics.get("phrase_coverage_ratio", 0.0) or 0.0), 6),
        "focused_note_count": int(focused.get("focused_note_count", 0) or 0),
        "focused_unique_pitch_count": int(focused.get("focused_unique_pitch_count", 0) or 0),
        "focused_adjacent_pitch_repeats": int(focused.get("focused_adjacent_pitch_repeats", 0) or 0),
        "focused_max_interval": int(focused.get("focused_max_interval", 0) or 0),
        "focused_removed_note_count": int(focused.get("focused_postprocess_removed_note_count", 0) or 0),
    }


def compact_candidate(candidate: dict[str, Any], *, role: str) -> dict[str, Any]:
    source_request = candidate.get("source_request") if isinstance(candidate.get("source_request"), dict) else {}
    gate = candidate.get("phrase_vocabulary_gate") if isinstance(candidate.get("phrase_vocabulary_gate"), dict) else {}
    return {
        "role": role,
        "candidate_id": str(candidate.get("candidate_id") or ""),
        "source_run_id": str(candidate.get("source_run_id") or ""),
        "source_seed": int(source_request.get("seed", 0) or 0),
        "top_k": int(source_request.get("top_k", 0) or 0),
        "temperature": float(source_request.get("temperature", 0.0) or 0.0),
        "sample_count": int(source_request.get("sample_count", 0) or 0),
        "sample_index": int(candidate.get("sample_index", 0) or 0),
        "sample_seed": int(candidate.get("sample_seed", 0) or 0),
        "repair_rank": int(candidate.get("repair_rank", 0) or 0),
        "qualified": bool(gate.get("qualified", False)),
        "flags": list(gate.get("flags") or []),
        "metric_fingerprint": metric_fingerprint(candidate),
    }


def human_candidates_by_role(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    candidates = report.get("candidates")
    if not isinstance(candidates, list) or len(candidates) < 2:
        raise PhraseVocabularyDuplicateSourceDivergenceError("human comparison must contain two candidates")
    return {str(candidate.get("role") or ""): candidate for candidate in candidates if isinstance(candidate, dict)}


def build_duplicate_source_divergence_audit(
    repair_summary: dict[str, Any],
    human_comparison: dict[str, Any],
    *,
    output_dir: Path,
) -> dict[str, Any]:
    repair_candidates = repair_summary.get("candidates")
    if not isinstance(repair_candidates, list) or not repair_candidates:
        raise PhraseVocabularyDuplicateSourceDivergenceError("repair summary must contain candidates")
    human_by_role = human_candidates_by_role(human_comparison)
    selected_human = human_by_role.get("selected")
    peer_human = human_by_role.get("peer")
    if not selected_human or not peer_human:
        raise PhraseVocabularyDuplicateSourceDivergenceError("human comparison must contain selected and peer")

    by_id = candidates_by_id(repair_candidates)
    selected = by_id.get(str(selected_human.get("candidate_id") or ""))
    peer = by_id.get(str(peer_human.get("candidate_id") or ""))
    if selected is None or peer is None:
        raise PhraseVocabularyDuplicateSourceDivergenceError("human comparison candidates not found in repair summary")

    selected_row = compact_candidate(selected, role="selected")
    peer_row = compact_candidate(peer, role="peer")
    note_sequence_match = bool(human_comparison.get("objective_comparison", {}).get("note_sequence_match", False))
    metric_fingerprint_match = selected_row["metric_fingerprint"] == peer_row["metric_fingerprint"]
    shared_sample_seed = selected_row["sample_seed"] == peer_row["sample_seed"]
    source_seed_diff = selected_row["source_seed"] != peer_row["source_seed"]
    sample_index_diff = selected_row["sample_index"] != peer_row["sample_index"]

    if note_sequence_match and metric_fingerprint_match and shared_sample_seed:
        boundary = "shared_sample_seed_duplicate_output"
        claim_boundary = "two_source_qualified_but_not_two_distinct_outputs"
    elif note_sequence_match and metric_fingerprint_match:
        boundary = "duplicate_output_without_shared_sample_seed"
        claim_boundary = "two_source_qualified_but_output_diversity_unproven"
    else:
        boundary = "distinct_output_candidates"
        claim_boundary = "two_source_distinct_output_support"

    return {
        "schema_version": "stage_b_margin_recovered_phrase_vocabulary_duplicate_source_divergence_v1",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "candidate_count": int(repair_summary.get("candidate_count", len(repair_candidates)) or len(repair_candidates)),
        "qualified_candidate_count": int(repair_summary.get("qualified_candidate_count", 0) or 0),
        "audited_candidates": [selected_row, peer_row],
        "divergence_checks": {
            "source_seed_diff": source_seed_diff,
            "sample_index_diff": sample_index_diff,
            "shared_sample_seed": shared_sample_seed,
            "note_sequence_match": note_sequence_match,
            "metric_fingerprint_match": metric_fingerprint_match,
            "both_qualified": bool(selected_row["qualified"] and peer_row["qualified"]),
        },
        "divergence_boundary": {
            "boundary": boundary,
            "claim_boundary": claim_boundary,
            "source_diversity": "present" if source_seed_diff else "absent",
            "output_diversity": "absent" if note_sequence_match else "present",
        },
        "next_recommended_issue": "Stage B margin-recovered phrase/vocabulary sample-seed diversity repair",
    }


def validate_duplicate_source_divergence(
    report: dict[str, Any],
    *,
    require_shared_sample_seed: bool,
    require_duplicate_output: bool,
    expected_boundary: str | None,
) -> dict[str, Any]:
    checks = report.get("divergence_checks") if isinstance(report.get("divergence_checks"), dict) else {}
    boundary = str(report.get("divergence_boundary", {}).get("boundary") or "")
    if require_shared_sample_seed and not bool(checks.get("shared_sample_seed", False)):
        raise PhraseVocabularyDuplicateSourceDivergenceError("expected shared sample seed")
    if require_duplicate_output:
        if not bool(checks.get("note_sequence_match", False)):
            raise PhraseVocabularyDuplicateSourceDivergenceError("expected duplicate note sequence")
        if not bool(checks.get("metric_fingerprint_match", False)):
            raise PhraseVocabularyDuplicateSourceDivergenceError("expected duplicate metric fingerprint")
    if expected_boundary and boundary != expected_boundary:
        raise PhraseVocabularyDuplicateSourceDivergenceError(f"expected boundary {expected_boundary}, got {boundary}")
    return {
        "candidate_count": int(report.get("candidate_count", 0) or 0),
        "qualified_candidate_count": int(report.get("qualified_candidate_count", 0) or 0),
        "source_seed_diff": bool(checks.get("source_seed_diff", False)),
        "sample_index_diff": bool(checks.get("sample_index_diff", False)),
        "shared_sample_seed": bool(checks.get("shared_sample_seed", False)),
        "note_sequence_match": bool(checks.get("note_sequence_match", False)),
        "metric_fingerprint_match": bool(checks.get("metric_fingerprint_match", False)),
        "boundary": boundary,
        "claim_boundary": str(report.get("divergence_boundary", {}).get("claim_boundary") or ""),
        "next_recommended_issue": str(report.get("next_recommended_issue") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    checks = report["divergence_checks"]
    boundary = report["divergence_boundary"]
    lines = [
        "# Stage B Margin-Recovered Phrase/Vocabulary Duplicate Source Divergence",
        "",
        f"- candidate count: `{report['candidate_count']}`",
        f"- qualified candidates: `{report['qualified_candidate_count']}`",
        f"- source seed diff: `{checks['source_seed_diff']}`",
        f"- sample index diff: `{checks['sample_index_diff']}`",
        f"- shared sample seed: `{checks['shared_sample_seed']}`",
        f"- note sequence match: `{checks['note_sequence_match']}`",
        f"- metric fingerprint match: `{checks['metric_fingerprint_match']}`",
        f"- boundary: `{boundary['boundary']}`",
        f"- claim boundary: `{boundary['claim_boundary']}`",
        "",
        "| role | candidate | source seed | sample index | sample seed | rank | qualified | notes | unique | dead-air | focused max interval |",
        "|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|",
    ]
    for candidate in report.get("audited_candidates", []):
        metrics = candidate["metric_fingerprint"]
        lines.append(
            "| "
            + " | ".join(
                [
                    str(candidate["role"]),
                    str(candidate["candidate_id"]),
                    str(candidate["source_seed"]),
                    str(candidate["sample_index"]),
                    str(candidate["sample_seed"]),
                    str(candidate["repair_rank"]),
                    str(candidate["qualified"]),
                    str(metrics["note_count"]),
                    str(metrics["unique_pitch_count"]),
                    f"{float(metrics['dead_air_ratio']):.3f}",
                    str(metrics["focused_max_interval"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit duplicate source divergence for phrase/vocabulary keeps")
    parser.add_argument("--repair_summary", type=str, required=True)
    parser.add_argument("--human_comparison", type=str, required=True)
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/stage_b_margin_recovered_phrase_vocabulary_duplicate_source_divergence",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--require_shared_sample_seed", action="store_true")
    parser.add_argument("--require_duplicate_output", action="store_true")
    parser.add_argument("--expected_boundary", type=str, default="")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    report = build_duplicate_source_divergence_audit(
        read_json(Path(args.repair_summary)),
        read_json(Path(args.human_comparison)),
        output_dir=output_dir,
    )
    summary = validate_duplicate_source_divergence(
        report,
        require_shared_sample_seed=bool(args.require_shared_sample_seed),
        require_duplicate_output=bool(args.require_duplicate_output),
        expected_boundary=str(args.expected_boundary or ""),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "duplicate_source_divergence_audit.json"
    markdown_path = output_dir / "duplicate_source_divergence_audit.md"
    write_json(report_path, report)
    write_json(output_dir / "duplicate_source_divergence_validation_summary.json", summary)
    markdown_path.write_text(markdown_report(report), encoding="utf-8")
    print(json.dumps({**summary, "report_path": str(report_path), "markdown_path": str(markdown_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
