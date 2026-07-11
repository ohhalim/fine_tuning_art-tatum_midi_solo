"""Bridge generated candidate reports into the chord-labeled evaluator."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import pretty_midi

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from scripts.evaluate_chord_labeled_subset import (  # noqa: E402
    ManifestError,
    analyze_sample,
    summarize_samples,
    write_json,
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_path(base_path: Path, raw_path: str | None) -> Path | None:
    if not raw_path:
        return None
    path = Path(str(raw_path))
    if not path.is_absolute():
        path = base_path.parent / path
    return path


def sanitize_sample_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return cleaned.strip("_") or "candidate"


def expand_chords(chords: Sequence[str], bars: int) -> list[str]:
    if not chords:
        raise ManifestError("chord progression is required")
    return [str(chords[index % len(chords)]) for index in range(int(bars))]


def load_source_report(report_path: Path, report: dict[str, Any]) -> dict[str, Any] | None:
    source_path = resolve_path(report_path, report.get("source_report") or report.get("source_ranking_report"))
    if source_path and source_path.exists():
        return read_json(source_path)
    return None


def extract_request_context(report_path: Path, report: dict[str, Any], default_bpm: float) -> dict[str, Any]:
    source_report = load_source_report(report_path, report)
    source_request = source_report.get("request", {}) if source_report else {}
    request = report.get("request", {}) if isinstance(report.get("request"), dict) else {}

    chords = (
        report.get("chord_progression")
        or report.get("chords")
        or request.get("chord_progression")
        or source_request.get("chord_progression")
        or (source_report or {}).get("chords", [])
    )
    if not isinstance(chords, list) or not chords:
        raise ManifestError("candidate report does not include chord_progression/chords metadata")

    bars = report.get("bars") or request.get("bars") or source_request.get("bars") or len(chords)
    bpm = report.get("bpm") or request.get("bpm") or source_request.get("bpm") or default_bpm
    return {
        "chord_progression": [str(chord) for chord in chords],
        "bars": int(bars),
        "bpm": float(bpm),
    }


def collect_candidate_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = report.get("candidates")
    if isinstance(rows, list):
        return [row for row in rows if isinstance(row, dict)]
    rows = report.get("samples")
    if isinstance(rows, list):
        return [row for row in rows if isinstance(row, dict)]
    raise ManifestError("candidate report must contain candidates or samples")


def candidate_midi_path(row: dict[str, Any]) -> str | None:
    for key in ("review_midi_path", "midi_path", "source_path"):
        value = row.get(key)
        if value:
            return str(value)
    return None


def candidate_sample_id(index: int, row: dict[str, Any]) -> str:
    parts = [
        str(row.get("mode") or row.get("generation_mode") or "candidate"),
        f"rank_{row.get('review_rank') or row.get('rank') or index}",
        f"sample_{row.get('sample_index') or index}",
    ]
    return sanitize_sample_id("_".join(parts))


def build_eval_samples(
    report_path: Path,
    candidate_report: dict[str, Any],
    candidate_limit: int,
    default_bpm: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    context = extract_request_context(report_path, candidate_report, default_bpm=default_bpm)
    rows = collect_candidate_rows(candidate_report)[: max(1, int(candidate_limit))]
    samples: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        midi_path = candidate_midi_path(row)
        if not midi_path:
            continue
        resolved = resolve_path(report_path, midi_path)
        if not resolved or not resolved.exists():
            continue
        samples.append(
            {
                "sample_id": candidate_sample_id(index, row),
                "bar_count": context["bars"],
                "bpm": context["bpm"],
                "chords": expand_chords(context["chord_progression"], context["bars"]),
                "midi_path": str(resolved),
            }
        )
    if not samples:
        raise ManifestError("no generated candidates with readable MIDI paths")
    return samples, context


def build_report_from_candidate_report(
    report_path: Path,
    candidate_limit: int = 3,
    default_bpm: float = 124.0,
) -> dict[str, Any]:
    candidate_report = read_json(report_path)
    samples, context = build_eval_samples(
        report_path,
        candidate_report,
        candidate_limit=candidate_limit,
        default_bpm=default_bpm,
    )
    sample_reports = [analyze_sample(sample, manifest_path=report_path) for sample in samples]
    return {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "source_report_path": str(report_path),
        "chord_progression": context["chord_progression"],
        "expanded_chords": expand_chords(context["chord_progression"], context["bars"]),
        "bars": int(context["bars"]),
        "bpm": float(context["bpm"]),
        "samples": sample_reports,
        "summary": summarize_samples(sample_reports),
    }


def markdown_report(report: dict[str, Any]) -> str:
    summary = report["summary"]
    ratios = summary["role_ratios"]
    lines = [
        "# Generated Candidate Chord-Labeled Evaluation",
        "",
        f"- source report: `{report['source_report_path']}`",
        f"- chord progression: `{', '.join(report['chord_progression'])}`",
        f"- bars: `{report['bars']}`",
        f"- sample count: `{summary['sample_count']}`",
        f"- note count: `{summary['note_count']}`",
        f"- chord-tone ratio: `{ratios['chord_tone_ratio']:.3f}`",
        f"- tension ratio: `{ratios['tension_ratio']:.3f}`",
        f"- outside ratio: `{ratios['outside_ratio']:.3f}`",
        "",
        "## Candidates",
        "",
        "| sample | notes | unique pitches | chord-tone | tension | approach | outside |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for sample in report["samples"]:
        ratios = sample["role_ratios"]
        lines.append(
            "| {sample_id} | {note_count} | {unique_pitch_count} | {chord:.3f} | {tension:.3f} | {approach:.3f} | {outside:.3f} |".format(
                sample_id=sample["sample_id"],
                note_count=sample["note_count"],
                unique_pitch_count=sample["unique_pitch_count"],
                chord=ratios["chord_tone_ratio"],
                tension=ratios["tension_ratio"],
                approach=ratios["approach_ratio"],
                outside=ratios["outside_ratio"],
            )
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "- This evaluates generated candidates only when chord metadata is known.",
            "- It does not solve missing chord labels for real reference MIDI.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def chord_eval_review_append_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    ratios = summary["role_ratios"]
    lines = [
        "## Generated Chord Eval Summary",
        "",
        f"- source report: `{report['source_report_path']}`",
        f"- chord progression: `{', '.join(report['chord_progression'])}`",
        f"- evaluated candidates: `{summary['sample_count']}`",
        f"- note count: `{summary['note_count']}`",
        f"- chord-tone ratio: `{ratios['chord_tone_ratio']:.3f}`",
        f"- tension ratio: `{ratios['tension_ratio']:.3f}`",
        f"- approach ratio: `{ratios['approach_ratio']:.3f}`",
        f"- outside ratio: `{ratios['outside_ratio']:.3f}`",
        "",
        "| sample | chord-tone | tension | approach | outside |",
        "|---|---:|---:|---:|---:|",
    ]
    for sample in report["samples"]:
        sample_ratios = sample["role_ratios"]
        lines.append(
            "| {sample_id} | {chord:.3f} | {tension:.3f} | {approach:.3f} | {outside:.3f} |".format(
                sample_id=sample["sample_id"],
                chord=sample_ratios["chord_tone_ratio"],
                tension=sample_ratios["tension_ratio"],
                approach=sample_ratios["approach_ratio"],
                outside=sample_ratios["outside_ratio"],
            )
        )
    lines.extend(
        [
            "",
            "Boundary: this uses generated candidate chord metadata only; it does not label real reference MIDI.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def write_combined_review_markdown(review_markdown_path: Path, report: dict[str, Any], output_path: Path) -> None:
    if not review_markdown_path.exists():
        raise ManifestError(f"review markdown does not exist: {review_markdown_path}")
    original = review_markdown_path.read_text(encoding="utf-8").rstrip()
    combined = original + "\n\n---\n\n" + chord_eval_review_append_markdown(report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(combined, encoding="utf-8")


def write_tiny_fixture(run_dir: Path) -> Path:
    fixture_dir = run_dir / "fixture_generated_candidates"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    midi_path = fixture_dir / "tiny_generated_candidate.mid"

    midi = pretty_midi.PrettyMIDI(initial_tempo=124)
    instrument = pretty_midi.Instrument(program=0)
    step = 60.0 / 124.0 / 4.0
    notes = [
        (60, 0),
        (63, 4),
        (67, 8),
        (70, 12),
        (65, 16),
        (69, 20),
        (72, 24),
        (75, 28),
        (70, 32),
        (74, 36),
        (77, 40),
        (81, 44),
        (67, 48),
        (71, 52),
        (74, 56),
        (77, 60),
    ]
    for pitch, step_index in notes:
        start = step * step_index
        instrument.notes.append(pretty_midi.Note(velocity=84, pitch=pitch, start=start, end=start + step * 2))
    midi.instruments.append(instrument)
    midi.write(str(midi_path))

    report_path = run_dir / "fixture_review_manifest.json"
    write_json(
        report_path,
        {
            "chord_progression": ["Cm7", "F7", "Bbmaj7", "G7"],
            "bars": 4,
            "bpm": 124,
            "candidates": [
                {
                    "mode": "fixture_generated",
                    "review_rank": 1,
                    "sample_index": 1,
                    "review_midi_path": str(midi_path),
                }
            ],
        },
    )
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate generated candidates against known chord metadata")
    parser.add_argument("--candidate_report", type=str, default=None)
    parser.add_argument(
        "--output_root",
        type=str,
        default=str(ROOT_DIR / "outputs" / "stage_b_generated_chord_eval"),
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--candidate_limit", type=int, default=3)
    parser.add_argument("--default_bpm", type=float, default=124.0)
    parser.add_argument("--write_tiny_fixture", action="store_true")
    parser.add_argument("--review_markdown", type=str, default=None)
    parser.add_argument("--combined_review_markdown_name", type=str, default="review_candidates_with_chord_eval.md")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_root) / run_id
    if args.write_tiny_fixture:
        candidate_report = write_tiny_fixture(run_dir)
    elif args.candidate_report:
        candidate_report = Path(args.candidate_report)
    else:
        raise ManifestError("--candidate_report is required unless --write_tiny_fixture is set")

    report = build_report_from_candidate_report(
        Path(candidate_report),
        candidate_limit=args.candidate_limit,
        default_bpm=args.default_bpm,
    )
    write_json(run_dir / "generated_chord_eval_report.json", report)
    (run_dir / "generated_chord_eval_report.md").write_text(markdown_report(report), encoding="utf-8")
    combined_review_path: Path | None = None
    if args.review_markdown:
        combined_review_path = run_dir / str(args.combined_review_markdown_name)
        write_combined_review_markdown(Path(args.review_markdown), report=report, output_path=combined_review_path)
    print(
        json.dumps(
            {
                "sample_count": report["summary"]["sample_count"],
                "note_count": report["summary"]["note_count"],
                "chord_tone_ratio": report["summary"]["role_ratios"]["chord_tone_ratio"],
                "tension_ratio": report["summary"]["role_ratios"]["tension_ratio"],
                "outside_ratio": report["summary"]["role_ratios"]["outside_ratio"],
                "report_path": str(run_dir / "generated_chord_eval_report.md"),
                "combined_review_markdown_path": str(combined_review_path) if combined_review_path else None,
            },
            ensure_ascii=True,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
