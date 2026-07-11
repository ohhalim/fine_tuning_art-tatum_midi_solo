"""Summarize distinct sample-seed phrase/vocabulary repair sweep results."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class PhraseVocabularyDistinctSampleSeedSweepError(ValueError):
    pass


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def blocked_sample_seeds(sample_seed_repair: dict[str, Any]) -> set[int]:
    counts = sample_seed_repair.get("duplicate_sample_seed_counts")
    if isinstance(counts, dict) and counts:
        return {int(seed) for seed in counts.keys()}
    selected = sample_seed_repair.get("selected_candidate")
    if isinstance(selected, dict) and selected.get("sample_seed") is not None:
        return {int(selected["sample_seed"])}
    return set()


def compact_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    source_request = candidate.get("source_request") if isinstance(candidate.get("source_request"), dict) else {}
    metrics = candidate.get("metrics") if isinstance(candidate.get("metrics"), dict) else {}
    focused = candidate.get("focused_solo_metrics") if isinstance(candidate.get("focused_solo_metrics"), dict) else {}
    gate = candidate.get("phrase_vocabulary_gate") if isinstance(candidate.get("phrase_vocabulary_gate"), dict) else {}
    return {
        "candidate_id": str(candidate.get("candidate_id") or ""),
        "source_run_id": str(candidate.get("source_run_id") or ""),
        "source_seed": int(source_request.get("seed", 0) or 0),
        "sample_index": int(candidate.get("sample_index", 0) or 0),
        "sample_seed": int(candidate.get("sample_seed", 0) or 0),
        "repair_rank": int(candidate.get("repair_rank", 0) or 0),
        "qualified": bool(gate.get("qualified", False)),
        "flags": list(gate.get("flags") or []),
        "note_count": int(metrics.get("note_count", 0) or 0),
        "unique_pitch_count": int(metrics.get("unique_pitch_count", 0) or 0),
        "dead_air_ratio": float(metrics.get("dead_air_ratio", 0.0) or 0.0),
        "focused_note_count": int(focused.get("focused_note_count", 0) or 0),
        "focused_unique_pitch_count": int(focused.get("focused_unique_pitch_count", 0) or 0),
        "focused_adjacent_pitch_repeats": int(focused.get("focused_adjacent_pitch_repeats", 0) or 0),
        "focused_max_interval": int(focused.get("focused_max_interval", 0) or 0),
    }


def build_distinct_sample_seed_sweep_report(
    repair_summary: dict[str, Any],
    sample_seed_repair: dict[str, Any],
    *,
    output_dir: Path,
) -> dict[str, Any]:
    candidates = repair_summary.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise PhraseVocabularyDistinctSampleSeedSweepError("repair summary must contain candidates")
    blocked = blocked_sample_seeds(sample_seed_repair)
    compact = [compact_candidate(candidate) for candidate in candidates]
    qualified = [candidate for candidate in compact if bool(candidate["qualified"])]
    distinct_qualified = [candidate for candidate in qualified if int(candidate["sample_seed"]) not in blocked]
    sample_seed_counts = Counter(int(candidate["sample_seed"]) for candidate in qualified)

    if distinct_qualified:
        boundary = "distinct_sample_seed_qualified_candidate_found"
        selected = distinct_qualified[0]
        next_issue = "Stage B margin-recovered phrase/vocabulary distinct sample-seed focused context review"
    else:
        boundary = "no_distinct_sample_seed_qualified_candidate"
        selected = {}
        next_issue = "Stage B margin-recovered phrase/vocabulary sampling diversity parameter sweep"

    return {
        "schema_version": "stage_b_margin_recovered_phrase_vocabulary_distinct_sample_seed_sweep_v1",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "candidate_count": int(repair_summary.get("candidate_count", len(compact)) or len(compact)),
        "qualified_candidate_count": len(qualified),
        "blocked_sample_seeds": sorted(blocked),
        "qualified_sample_seed_counts": dict(sorted((str(seed), count) for seed, count in sample_seed_counts.items())),
        "distinct_sample_seed_qualified_count": len(distinct_qualified),
        "selected_distinct_candidate": selected,
        "top_candidates": compact[:10],
        "sweep_boundary": {
            "boundary": boundary,
            "has_distinct_sample_seed_qualified_candidate": bool(distinct_qualified),
            "blocked_duplicate_sample_seed_reused": any(
                int(candidate["sample_seed"]) in blocked for candidate in qualified
            ),
        },
        "next_recommended_issue": next_issue,
    }


def validate_distinct_sample_seed_sweep(
    report: dict[str, Any],
    *,
    min_candidates: int,
    expected_blocked_seed: int | None,
) -> dict[str, Any]:
    candidate_count = int(report.get("candidate_count", 0) or 0)
    if candidate_count < min_candidates:
        raise PhraseVocabularyDistinctSampleSeedSweepError(f"candidate_count {candidate_count} < {min_candidates}")
    blocked = [int(seed) for seed in report.get("blocked_sample_seeds", [])]
    if expected_blocked_seed is not None and int(expected_blocked_seed) not in blocked:
        raise PhraseVocabularyDistinctSampleSeedSweepError(f"expected blocked sample seed {expected_blocked_seed}")
    return {
        "candidate_count": candidate_count,
        "qualified_candidate_count": int(report.get("qualified_candidate_count", 0) or 0),
        "distinct_sample_seed_qualified_count": int(report.get("distinct_sample_seed_qualified_count", 0) or 0),
        "blocked_sample_seeds": blocked,
        "boundary": str(report.get("sweep_boundary", {}).get("boundary") or ""),
        "next_recommended_issue": str(report.get("next_recommended_issue") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    boundary = report["sweep_boundary"]
    lines = [
        "# Stage B Margin-Recovered Phrase/Vocabulary Distinct Sample-Seed Sweep",
        "",
        f"- candidate count: `{report['candidate_count']}`",
        f"- qualified candidates: `{report['qualified_candidate_count']}`",
        f"- blocked sample seeds: `{report['blocked_sample_seeds']}`",
        f"- distinct sample-seed qualified candidates: `{report['distinct_sample_seed_qualified_count']}`",
        f"- boundary: `{boundary['boundary']}`",
        "",
        "| rank | candidate | qualified | source seed | sample seed | notes | unique | dead-air | focused unique | focused max interval | flags |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for candidate in report.get("top_candidates", []):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(candidate["repair_rank"]),
                    str(candidate["candidate_id"]),
                    str(candidate["qualified"]),
                    str(candidate["source_seed"]),
                    str(candidate["sample_seed"]),
                    str(candidate["note_count"]),
                    str(candidate["unique_pitch_count"]),
                    f"{float(candidate['dead_air_ratio']):.3f}",
                    str(candidate["focused_unique_pitch_count"]),
                    str(candidate["focused_max_interval"]),
                    ",".join(candidate["flags"]) if candidate["flags"] else "ok",
                ]
            )
            + " |"
        )
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize distinct sample-seed repair sweep")
    parser.add_argument("--repair_summary", type=str, required=True)
    parser.add_argument("--sample_seed_repair", type=str, required=True)
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/stage_b_margin_recovered_phrase_vocabulary_distinct_sample_seed_sweep",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--min_candidates", type=int, default=1)
    parser.add_argument("--expected_blocked_seed", type=int, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    report = build_distinct_sample_seed_sweep_report(
        read_json(Path(args.repair_summary)),
        read_json(Path(args.sample_seed_repair)),
        output_dir=output_dir,
    )
    summary = validate_distinct_sample_seed_sweep(
        report,
        min_candidates=int(args.min_candidates),
        expected_blocked_seed=args.expected_blocked_seed,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "distinct_sample_seed_sweep_summary.json"
    markdown_path = output_dir / "distinct_sample_seed_sweep_summary.md"
    write_json(report_path, report)
    write_json(output_dir / "distinct_sample_seed_sweep_validation_summary.json", summary)
    markdown_path.write_text(markdown_report(report), encoding="utf-8")
    print(json.dumps({**summary, "report_path": str(report_path), "markdown_path": str(markdown_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
