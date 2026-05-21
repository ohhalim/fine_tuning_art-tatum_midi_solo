"""Export Stage B ranked MIDI candidates for manual listening review."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_RANKING_REPORT = (
    ROOT_DIR
    / "outputs"
    / "stage_b_candidate_ranking"
    / "harness_stage_b_chord_aware_probe"
    / "candidate_rank_report.json"
)
DEFAULT_OUTPUT_ROOT = ROOT_DIR / "outputs" / "stage_b_review_candidates"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def candidate_sort_key(candidate: dict[str, Any]) -> tuple[Any, ...]:
    return (
        bool(candidate.get("reviewable")),
        bool(candidate.get("strict_valid")),
        -len(candidate.get("review_flags", []) or []),
        _float(candidate.get("score")),
        _float(candidate.get("bar_chord_tone_ratio")),
        -_float(candidate.get("repeated_pitch_ratio")),
    )


def mode_from_probe_report(report: dict[str, Any]) -> str:
    coverage = bool(report.get("coverage_aware_positions", False))
    chord = bool(report.get("chord_aware_pitches", False))
    if coverage and chord:
        return "coverage_chord"
    if coverage:
        return "coverage"
    if chord:
        return "chord"
    return "plain"


def score_probe_sample(sample: dict[str, Any]) -> float:
    metrics = sample.get("metrics", {})
    collapse = sample.get("collapse", {})
    temporal = sample.get("temporal_coverage", {})
    return round(
        (30.0 if sample.get("strict_valid") else 0.0)
        + (15.0 if sample.get("valid") else 0.0)
        + (8.0 if sample.get("grammar_gate_passed") else 0.0)
        + _float(metrics.get("chord_tone_ratio")) * 30.0
        + _float(temporal.get("onset_coverage_ratio")) * 12.0
        + _float(temporal.get("sustained_coverage_ratio")) * 6.0
        + min(_int(metrics.get("unique_pitch_count")), 8) * 2.0
        - _float(metrics.get("dead_air_ratio")) * 10.0
        - _float(collapse.get("repeated_position_pitch_pair_ratio")) * 15.0
        - _float(collapse.get("repeated_pitch_ratio")) * 10.0,
        4,
    )


def candidates_from_generation_probe(report: dict[str, Any]) -> list[dict[str, Any]]:
    mode = mode_from_probe_report(report)
    note_groups_per_bar = _int(report.get("constrained_note_groups_per_bar"))
    candidates: list[dict[str, Any]] = []
    for sample in report.get("samples", []):
        if not isinstance(sample, dict):
            continue
        metrics = sample.get("metrics", {})
        collapse = sample.get("collapse", {})
        temporal = sample.get("temporal_coverage", {})
        review_flags: list[str] = []
        if not sample.get("strict_valid"):
            reason = sample.get("failure_reason") or sample.get("diagnostic_failure_reason") or "not_strict_valid"
            review_flags.append(str(reason))
        if collapse.get("collapse_warning"):
            review_flags.extend(str(reason) for reason in collapse.get("collapse_reasons", []))
        candidates.append(
            {
                "rank": _int(sample.get("sample_index")),
                "mode": mode,
                "note_groups_per_bar": note_groups_per_bar,
                "sample_index": _int(sample.get("sample_index")),
                "score": score_probe_sample(sample),
                "reviewable": bool(sample.get("strict_valid")) and not review_flags,
                "review_flags": review_flags,
                "midi_path": sample.get("midi_path"),
                "strict_valid": bool(sample.get("strict_valid")),
                "note_count": _int(metrics.get("note_count")),
                "unique_pitch_count": _int(metrics.get("unique_pitch_count")),
                "chord_tone_ratio": _float(metrics.get("chord_tone_ratio")),
                "bar_chord_tone_ratio": _float(metrics.get("chord_tone_ratio")),
                "min_bar_chord_tone_ratio": _float(metrics.get("chord_tone_ratio")),
                "dominant_pitch_ratio": _float(collapse.get("max_same_pitch_repeats"))
                / max(1, _int(collapse.get("note_group_count"))),
                "repeated_pitch_ratio": _float(collapse.get("repeated_pitch_ratio")),
                "onset_coverage_ratio": _float(temporal.get("onset_coverage_ratio")),
                "sustained_coverage_ratio": _float(temporal.get("sustained_coverage_ratio")),
            }
        )
    return candidates


def candidate_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    if "top_candidates" in report:
        return list(report.get("top_candidates", []))
    if "samples" in report:
        return candidates_from_generation_probe(report)
    return []


def select_review_candidates(
    report: dict[str, Any],
    top_n: int,
    mode: str | None = "coverage_chord",
    reviewable_only: bool = True,
) -> list[dict[str, Any]]:
    candidates = candidate_rows(report)
    if mode:
        candidates = [candidate for candidate in candidates if candidate.get("mode") == mode]
    if reviewable_only:
        candidates = [candidate for candidate in candidates if candidate.get("reviewable")]
    candidates = sorted(candidates, key=candidate_sort_key, reverse=True)
    return candidates[: max(1, int(top_n))]


def compact_candidate(candidate: dict[str, Any], rank: int) -> dict[str, Any]:
    return {
        "review_rank": int(rank),
        "source_rank": _int(candidate.get("rank")),
        "mode": candidate.get("mode"),
        "note_groups_per_bar": _int(candidate.get("note_groups_per_bar")),
        "sample_index": _int(candidate.get("sample_index")),
        "score": _float(candidate.get("score")),
        "reviewable": bool(candidate.get("reviewable")),
        "review_flags": list(candidate.get("review_flags", []) or []),
        "midi_path": candidate.get("midi_path"),
        "note_count": _int(candidate.get("note_count")),
        "unique_pitch_count": _int(candidate.get("unique_pitch_count")),
        "chord_tone_ratio": _float(candidate.get("chord_tone_ratio")),
        "bar_chord_tone_ratio": _float(candidate.get("bar_chord_tone_ratio")),
        "min_bar_chord_tone_ratio": _float(candidate.get("min_bar_chord_tone_ratio")),
        "dominant_pitch_ratio": _float(candidate.get("dominant_pitch_ratio")),
        "repeated_pitch_ratio": _float(candidate.get("repeated_pitch_ratio")),
        "onset_coverage_ratio": _float(candidate.get("onset_coverage_ratio")),
        "sustained_coverage_ratio": _float(candidate.get("sustained_coverage_ratio")),
    }


def copy_candidate_midi(candidate: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    source = candidate.get("midi_path")
    if not source:
        return {"copied": False, "reason": "missing midi_path"}
    source_path = Path(str(source))
    if not source_path.is_absolute():
        source_path = ROOT_DIR / source_path
    if not source_path.exists():
        return {"copied": False, "reason": f"missing source midi: {source_path}"}

    target_dir = output_dir / "midi"
    target_dir.mkdir(parents=True, exist_ok=True)
    target_name = (
        f"rank_{int(candidate['review_rank']):02d}_"
        f"{candidate['mode']}_g{int(candidate['note_groups_per_bar'])}_"
        f"s{int(candidate['sample_index'])}.mid"
    )
    target_path = target_dir / target_name
    shutil.copy2(source_path, target_path)
    return {
        "copied": True,
        "source_path": str(source_path),
        "review_midi_path": str(target_path),
        "review_midi_relative_path": str(target_path.relative_to(output_dir)),
    }


def markdown_report(manifest: dict[str, Any]) -> str:
    source_report = manifest.get("source_report") or manifest["source_ranking_report"]
    lines = [
        "# Stage B Candidate Listening Review",
        "",
        f"- source report: `{source_report}`",
        f"- generated at: `{manifest['generated_at']}`",
        f"- mode filter: `{manifest['mode_filter']}`",
        f"- reviewable only: `{str(manifest['reviewable_only']).lower()}`",
        f"- selected candidates: `{len(manifest['candidates'])}`",
        "",
        "| review | source rank | mode | groups/bar | sample | score | notes | pitches | chord | bar chord | min bar | dominant | repeat | midi |",
        "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for candidate in manifest["candidates"]:
        midi_path = candidate.get("review_midi_relative_path") or candidate.get("midi_path")
        table_candidate = dict(candidate)
        table_candidate["display_midi_path"] = midi_path
        lines.append(
            "| {review_rank} | {source_rank} | {mode} | {note_groups_per_bar} | {sample_index} | "
            "{score:.4f} | {note_count} | {unique_pitch_count} | {chord_tone_ratio:.3f} | "
            "{bar_chord_tone_ratio:.3f} | {min_bar_chord_tone_ratio:.3f} | "
            "{dominant_pitch_ratio:.3f} | {repeated_pitch_ratio:.3f} | `{display_midi_path}` |".format(
                **table_candidate
            )
        )
    lines.extend(
        [
            "",
            "## Review Checklist",
            "",
            "- solo-line shape",
            "- phrase contour",
            "- over-mechanical rhythm",
            "- excessive repeated-pitch dependence",
            "- excessive high-register bias",
            "- too-short fragment rather than phrase",
            "- chord-tone correctness sounding too constrained",
            "- one-note/two-note/chord-block/long-sustain failure",
        ]
    )
    return "\n".join(lines) + "\n"


def build_review_manifest(
    ranking_report_path: Path,
    output_dir: Path,
    top_n: int,
    mode: str | None,
    reviewable_only: bool,
    copy_midi: bool,
) -> dict[str, Any]:
    report = read_json(ranking_report_path)
    selected = select_review_candidates(
        report,
        top_n=top_n,
        mode=mode,
        reviewable_only=reviewable_only,
    )
    candidates = [compact_candidate(candidate, rank=index + 1) for index, candidate in enumerate(selected)]
    warnings: list[str] = []
    if copy_midi:
        for candidate in candidates:
            copy_result = copy_candidate_midi(candidate, output_dir)
            candidate.update(copy_result)
            if not copy_result.get("copied"):
                warnings.append(str(copy_result.get("reason", "copy failed")))

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "source_report": str(ranking_report_path),
        "source_ranking_report": str(ranking_report_path),
        "output_dir": str(output_dir),
        "mode_filter": mode,
        "reviewable_only": bool(reviewable_only),
        "top_n": int(top_n),
        "candidate_count": int(len(candidates)),
        "warnings": warnings,
        "candidates": candidates,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export Stage B candidates for manual listening review")
    parser.add_argument("--ranking_report", type=str, default=str(DEFAULT_RANKING_REPORT))
    parser.add_argument("--source_report", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--top_n", type=int, default=6)
    parser.add_argument("--mode", type=str, default="coverage_chord")
    parser.add_argument("--include_flagged", action="store_true")
    parser.add_argument("--copy_midi", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    ranking_report_path = Path(args.source_report or args.ranking_report)
    if not ranking_report_path.is_absolute():
        ranking_report_path = ROOT_DIR / ranking_report_path
    run_id = args.run_id or ranking_report_path.parent.name
    output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_ROOT / run_id
    if not output_dir.is_absolute():
        output_dir = ROOT_DIR / output_dir

    manifest = build_review_manifest(
        ranking_report_path=ranking_report_path,
        output_dir=output_dir,
        top_n=args.top_n,
        mode=args.mode or None,
        reviewable_only=not args.include_flagged,
        copy_midi=bool(args.copy_midi),
    )
    write_json(output_dir / "review_manifest.json", manifest)
    (output_dir / "review_candidates.md").write_text(markdown_report(manifest), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=True, indent=2))
    return 0 if manifest["candidate_count"] > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
