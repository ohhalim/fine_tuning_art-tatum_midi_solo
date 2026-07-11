"""Repair claim boundary for duplicate sample-seed phrase/vocabulary candidates."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class PhraseVocabularySampleSeedDiversityRepairError(ValueError):
    pass


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def qualified_candidates(repair_summary: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = repair_summary.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise PhraseVocabularySampleSeedDiversityRepairError("repair summary must contain candidates")
    return [
        candidate
        for candidate in candidates
        if bool(candidate.get("phrase_vocabulary_gate", {}).get("qualified", False))
    ]


def compact_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    source_request = candidate.get("source_request") if isinstance(candidate.get("source_request"), dict) else {}
    focused = candidate.get("focused_solo_metrics") if isinstance(candidate.get("focused_solo_metrics"), dict) else {}
    metrics = candidate.get("metrics") if isinstance(candidate.get("metrics"), dict) else {}
    return {
        "candidate_id": str(candidate.get("candidate_id") or ""),
        "source_run_id": str(candidate.get("source_run_id") or ""),
        "source_seed": int(source_request.get("seed", 0) or 0),
        "sample_index": int(candidate.get("sample_index", 0) or 0),
        "sample_seed": int(candidate.get("sample_seed", 0) or 0),
        "repair_rank": int(candidate.get("repair_rank", 0) or 0),
        "note_count": int(metrics.get("note_count", 0) or 0),
        "unique_pitch_count": int(metrics.get("unique_pitch_count", 0) or 0),
        "dead_air_ratio": float(metrics.get("dead_air_ratio", 0.0) or 0.0),
        "focused_note_count": int(focused.get("focused_note_count", 0) or 0),
        "focused_unique_pitch_count": int(focused.get("focused_unique_pitch_count", 0) or 0),
        "focused_adjacent_pitch_repeats": int(focused.get("focused_adjacent_pitch_repeats", 0) or 0),
        "focused_max_interval": int(focused.get("focused_max_interval", 0) or 0),
    }


def build_sample_seed_diversity_repair(
    repair_summary: dict[str, Any],
    duplicate_audit: dict[str, Any],
    *,
    output_dir: Path,
) -> dict[str, Any]:
    qualified = qualified_candidates(repair_summary)
    compact = [compact_candidate(candidate) for candidate in qualified]
    by_sample_seed: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for candidate in compact:
        by_sample_seed[int(candidate["sample_seed"])].append(candidate)

    sample_seed_counts = Counter(int(candidate["sample_seed"]) for candidate in compact)
    distinct_sample_seed_count = len(sample_seed_counts)
    duplicate_sample_seed_counts = {
        str(seed): count for seed, count in sorted(sample_seed_counts.items()) if count > 1
    }
    source_seed_count = len({int(candidate["source_seed"]) for candidate in compact})
    selected = compact[0] if compact else {}
    selected_sample_seed = int(selected.get("sample_seed", 0) or 0)
    distinct_peer_candidates = [
        candidate for candidate in compact[1:] if int(candidate["sample_seed"]) != selected_sample_seed
    ]
    duplicate_boundary = str(duplicate_audit.get("divergence_boundary", {}).get("boundary") or "")

    if distinct_peer_candidates:
        repaired_boundary = "two_distinct_sample_seed_output_support"
        action = "retain_distinct_peer_support"
    else:
        repaired_boundary = "single_distinct_sample_seed_keep_support"
        action = "demote_duplicate_peer_from_distinct_output_support"

    return {
        "schema_version": "stage_b_margin_recovered_phrase_vocabulary_sample_seed_diversity_repair_v1",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "candidate_count": int(repair_summary.get("candidate_count", 0) or 0),
        "qualified_candidate_count": len(compact),
        "qualified_source_seed_count": source_seed_count,
        "qualified_sample_seed_count": distinct_sample_seed_count,
        "qualified_sample_seed_counts": dict(sorted((str(seed), count) for seed, count in sample_seed_counts.items())),
        "duplicate_sample_seed_counts": duplicate_sample_seed_counts,
        "selected_candidate": selected,
        "distinct_peer_candidate_count": len(distinct_peer_candidates),
        "distinct_peer_candidates": distinct_peer_candidates,
        "source_duplicate_boundary": duplicate_boundary,
        "diversity_repair": {
            "boundary": repaired_boundary,
            "action": action,
            "count_source_support": source_seed_count >= 2,
            "count_distinct_output_support": len(distinct_peer_candidates) >= 1,
            "duplicate_sample_seed_peer_demoted": not bool(distinct_peer_candidates),
        },
        "claim_boundary": {
            "before": str(duplicate_audit.get("divergence_boundary", {}).get("claim_boundary") or ""),
            "after": (
                "single_distinct_sample_seed_keep_support_until_new_sampling"
                if not distinct_peer_candidates
                else "two_distinct_sample_seed_keep_support"
            ),
        },
        "next_recommended_issue": "Stage B margin-recovered phrase/vocabulary distinct sample-seed repair sweep",
    }


def validate_sample_seed_diversity_repair(
    report: dict[str, Any],
    *,
    expected_boundary: str | None,
    require_duplicate_demoted: bool,
) -> dict[str, Any]:
    boundary = str(report.get("diversity_repair", {}).get("boundary") or "")
    if expected_boundary and boundary != expected_boundary:
        raise PhraseVocabularySampleSeedDiversityRepairError(f"expected boundary {expected_boundary}, got {boundary}")
    if require_duplicate_demoted and not bool(
        report.get("diversity_repair", {}).get("duplicate_sample_seed_peer_demoted", False)
    ):
        raise PhraseVocabularySampleSeedDiversityRepairError("expected duplicate sample-seed peer demotion")
    return {
        "qualified_candidate_count": int(report.get("qualified_candidate_count", 0) or 0),
        "qualified_source_seed_count": int(report.get("qualified_source_seed_count", 0) or 0),
        "qualified_sample_seed_count": int(report.get("qualified_sample_seed_count", 0) or 0),
        "distinct_peer_candidate_count": int(report.get("distinct_peer_candidate_count", 0) or 0),
        "boundary": boundary,
        "claim_after": str(report.get("claim_boundary", {}).get("after") or ""),
        "next_recommended_issue": str(report.get("next_recommended_issue") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    repair = report["diversity_repair"]
    lines = [
        "# Stage B Margin-Recovered Phrase/Vocabulary Sample-Seed Diversity Repair",
        "",
        f"- qualified candidates: `{report['qualified_candidate_count']}`",
        f"- qualified source seeds: `{report['qualified_source_seed_count']}`",
        f"- qualified sample seeds: `{report['qualified_sample_seed_count']}`",
        f"- duplicate sample seed counts: `{report['duplicate_sample_seed_counts']}`",
        f"- distinct peer candidates: `{report['distinct_peer_candidate_count']}`",
        f"- boundary: `{repair['boundary']}`",
        f"- action: `{repair['action']}`",
        f"- claim after: `{report['claim_boundary']['after']}`",
        "",
        "| candidate | source seed | sample index | sample seed | rank | notes | unique | dead-air | focused unique | focused max interval |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    rows = [report["selected_candidate"]] + list(report.get("distinct_peer_candidates") or [])
    for candidate in rows:
        if not candidate:
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    str(candidate["candidate_id"]),
                    str(candidate["source_seed"]),
                    str(candidate["sample_index"]),
                    str(candidate["sample_seed"]),
                    str(candidate["repair_rank"]),
                    str(candidate["note_count"]),
                    str(candidate["unique_pitch_count"]),
                    f"{float(candidate['dead_air_ratio']):.3f}",
                    str(candidate["focused_unique_pitch_count"]),
                    str(candidate["focused_max_interval"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Repair sample-seed diversity boundary")
    parser.add_argument("--repair_summary", type=str, required=True)
    parser.add_argument("--duplicate_audit", type=str, required=True)
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/stage_b_margin_recovered_phrase_vocabulary_sample_seed_diversity_repair",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--expected_boundary", type=str, default="")
    parser.add_argument("--require_duplicate_demoted", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    report = build_sample_seed_diversity_repair(
        read_json(Path(args.repair_summary)),
        read_json(Path(args.duplicate_audit)),
        output_dir=output_dir,
    )
    summary = validate_sample_seed_diversity_repair(
        report,
        expected_boundary=str(args.expected_boundary or ""),
        require_duplicate_demoted=bool(args.require_duplicate_demoted),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "sample_seed_diversity_repair.json"
    markdown_path = output_dir / "sample_seed_diversity_repair.md"
    write_json(report_path, report)
    write_json(output_dir / "sample_seed_diversity_validation_summary.json", summary)
    markdown_path.write_text(markdown_report(report), encoding="utf-8")
    print(json.dumps({**summary, "report_path": str(report_path), "markdown_path": str(markdown_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
