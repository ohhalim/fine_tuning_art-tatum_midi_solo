"""Rank Stage B generated MIDI candidates from an A/B sweep report."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pretty_midi

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from scripts.stage_b_tokens import POSITIONS_PER_BAR, ROOT_TO_PC, parse_chord_symbol  # noqa: E402

QUALITY_INTERVALS = {
    "maj": {0, 4, 7},
    "maj7": {0, 4, 7, 11},
    "min": {0, 3, 7},
    "min7": {0, 3, 7, 10},
    "dom7": {0, 4, 7, 10},
    "dim": {0, 3, 6},
    "halfdim": {0, 3, 6, 10},
    "sus": {0, 5, 7},
    "unknown": {0, 4, 7},
}


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


def beats_per_bar(time_signature: str | None) -> int:
    if not time_signature:
        return 4
    numerator = str(time_signature).split("/", maxsplit=1)[0]
    return max(1, _int(numerator, default=4))


def chord_pitch_classes(chord: str | None) -> set[int]:
    root, quality = parse_chord_symbol(chord)
    root_pc = ROOT_TO_PC.get(root, 0)
    intervals = QUALITY_INTERVALS.get(quality, QUALITY_INTERVALS["unknown"])
    return {(root_pc + interval) % 12 for interval in intervals}


def analyze_midi_candidate(
    midi_path: str | None,
    request: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not midi_path:
        return {"analysis_available": False, "reason": "missing midi path"}
    path = Path(str(midi_path))
    if not path.is_absolute():
        path = ROOT_DIR / path
    if not path.exists():
        return {"analysis_available": False, "reason": f"missing midi file: {path}"}

    request = request or {}
    chords = [str(chord) for chord in request.get("chord_progression", []) if str(chord)]
    bpm = max(1e-6, _float(request.get("bpm"), default=124.0))
    bar_duration = beats_per_bar(request.get("time_signature")) * (60.0 / bpm)
    step_duration = bar_duration / int(POSITIONS_PER_BAR)

    pm = pretty_midi.PrettyMIDI(str(path))
    notes = sorted(
        [note for instrument in pm.instruments if not instrument.is_drum for note in instrument.notes],
        key=lambda note: (float(note.start), int(note.pitch), float(note.end)),
    )
    if not notes:
        return {"analysis_available": True, "note_count": 0, "review_flags": ["empty_midi"]}

    pitch_counts: dict[int, int] = {}
    per_bar_counts: dict[int, int] = {}
    per_bar_chord_hits: dict[int, int] = {}
    per_bar_positions: dict[int, list[int]] = {}
    chord_hits = 0

    for note in notes:
        pitch = int(note.pitch)
        pitch_counts[pitch] = pitch_counts.get(pitch, 0) + 1
        bar = max(0, int(float(note.start) // max(1e-6, bar_duration)))
        position = int(round((float(note.start) - (bar * bar_duration)) / max(1e-6, step_duration)))
        position = max(0, min(int(POSITIONS_PER_BAR) - 1, position))
        per_bar_counts[bar] = per_bar_counts.get(bar, 0) + 1
        per_bar_positions.setdefault(bar, []).append(position)
        if chords:
            pcs = chord_pitch_classes(chords[min(bar, len(chords) - 1)])
            if pitch % 12 in pcs:
                chord_hits += 1
                per_bar_chord_hits[bar] = per_bar_chord_hits.get(bar, 0) + 1

    note_count = len(notes)
    per_bar_ratios = {
        str(bar): (per_bar_chord_hits.get(bar, 0) / count if count else 0.0)
        for bar, count in sorted(per_bar_counts.items())
    }
    position_templates = [tuple(positions) for _, positions in sorted(per_bar_positions.items())]
    repeated_template_ratio = 0.0
    if len(position_templates) > 1:
        first_template = position_templates[0]
        repeated = sum(1 for template in position_templates[1:] if template == first_template)
        repeated_template_ratio = repeated / max(1, len(position_templates) - 1)

    dominant_pitch_count = max(pitch_counts.values()) if pitch_counts else 0
    unique_pitch_count = len(pitch_counts)
    repeated_pitch_ratio = 1.0 - (unique_pitch_count / note_count)
    chord_tone_ratio = chord_hits / note_count if chords else 0.0
    min_bar_chord_tone_ratio = min(per_bar_ratios.values()) if per_bar_ratios else chord_tone_ratio
    dominant_pitch_ratio = dominant_pitch_count / note_count if note_count else 0.0
    review_flags = review_flags_for_diagnostics(
        chord_tone_ratio=chord_tone_ratio,
        min_bar_chord_tone_ratio=min_bar_chord_tone_ratio,
        dominant_pitch_ratio=dominant_pitch_ratio,
        repeated_pitch_ratio=repeated_pitch_ratio,
        onset_template_repetition_ratio=repeated_template_ratio,
    )

    return {
        "analysis_available": True,
        "note_count": int(note_count),
        "unique_pitch_count": int(unique_pitch_count),
        "dominant_pitch": int(max(pitch_counts, key=pitch_counts.get)),
        "dominant_pitch_count": int(dominant_pitch_count),
        "dominant_pitch_ratio": dominant_pitch_ratio,
        "repeated_pitch_ratio": repeated_pitch_ratio,
        "bar_chord_tone_count": int(chord_hits),
        "bar_chord_tone_ratio": chord_tone_ratio,
        "min_bar_chord_tone_ratio": min_bar_chord_tone_ratio,
        "per_bar_chord_tone_ratio": per_bar_ratios,
        "onset_template_repetition_ratio": repeated_template_ratio,
        "review_flags": review_flags,
    }


def review_flags_for_diagnostics(
    chord_tone_ratio: float,
    min_bar_chord_tone_ratio: float,
    dominant_pitch_ratio: float,
    repeated_pitch_ratio: float,
    onset_template_repetition_ratio: float,
) -> list[str]:
    flags: list[str] = []
    if float(chord_tone_ratio) < 0.30:
        flags.append("low_chord_tone_ratio")
    if float(min_bar_chord_tone_ratio) < 0.20:
        flags.append("low_bar_chord_tone_ratio")
    if float(dominant_pitch_ratio) > 0.55:
        flags.append("dominant_pitch_repetition")
    if float(repeated_pitch_ratio) > 0.70:
        flags.append("low_pitch_variety")
    if float(onset_template_repetition_ratio) > 0.90:
        flags.append("repeated_onset_template")
    return flags


def score_candidate(sample: dict[str, Any]) -> dict[str, Any]:
    metrics = sample.get("metrics", {})
    collapse = sample.get("collapse", {})
    temporal = sample.get("temporal_coverage", {})
    harmonic = sample.get("harmonic_diagnostics", {})
    valid = bool(sample.get("valid"))
    strict_valid = bool(sample.get("strict_valid"))
    grammar_valid = bool(sample.get("grammar_gate_passed"))
    dead_air_ratio = _float(metrics.get("dead_air_ratio"))
    repetition_score = _float(metrics.get("repetition_score"))
    metric_chord_tone_ratio = _float(metrics.get("chord_tone_ratio"))
    bar_chord_tone_ratio = _float(harmonic.get("bar_chord_tone_ratio"), default=metric_chord_tone_ratio)
    chord_tone_ratio = min(metric_chord_tone_ratio, bar_chord_tone_ratio)
    min_bar_chord_tone_ratio = _float(harmonic.get("min_bar_chord_tone_ratio"), default=chord_tone_ratio)
    unique_pitch_count = _int(metrics.get("unique_pitch_count"))
    onset_coverage_ratio = _float(temporal.get("onset_coverage_ratio"))
    sustained_coverage_ratio = _float(temporal.get("sustained_coverage_ratio"))
    position_span_ratio = _float(temporal.get("position_span_ratio"))
    collapse_warning = bool(collapse.get("collapse_warning"))
    postprocess_removal_ratio = _float(collapse.get("postprocess_removal_ratio"))
    repeated_pitch_ratio = max(
        _float(collapse.get("repeated_pitch_ratio")),
        _float(harmonic.get("repeated_pitch_ratio")),
    )
    dominant_pitch_ratio = _float(harmonic.get("dominant_pitch_ratio"))
    onset_template_repetition_ratio = _float(harmonic.get("onset_template_repetition_ratio"))

    components = {
        "strict_valid_bonus": 30.0 if strict_valid else 0.0,
        "valid_bonus": 15.0 if valid else 0.0,
        "grammar_bonus": 8.0 if grammar_valid else 0.0,
        "onset_coverage_bonus": onset_coverage_ratio * 12.0,
        "sustained_coverage_bonus": sustained_coverage_ratio * 6.0,
        "position_span_bonus": position_span_ratio * 3.0,
        "chord_tone_bonus": chord_tone_ratio * 30.0,
        "min_bar_chord_tone_bonus": min_bar_chord_tone_ratio * 15.0,
        "pitch_diversity_bonus": min(unique_pitch_count, 6) * 2.0,
        "dead_air_penalty": dead_air_ratio * -15.0,
        "repetition_penalty": repetition_score * -8.0,
        "postprocess_penalty": postprocess_removal_ratio * -10.0,
        "collapse_warning_penalty": -25.0 if collapse_warning else 0.0,
        "low_chord_tone_penalty": -30.0
        if chord_tone_ratio < 0.25
        else (-15.0 if chord_tone_ratio < 0.40 else 0.0),
        "low_bar_chord_tone_penalty": -25.0
        if min_bar_chord_tone_ratio < 0.20
        else (-12.0 if min_bar_chord_tone_ratio < 0.30 else 0.0),
        "repeated_pitch_penalty": repeated_pitch_ratio * -20.0,
        "dominant_pitch_penalty": max(0.0, dominant_pitch_ratio - 0.45) * -40.0,
        "onset_template_penalty": onset_template_repetition_ratio * -12.0,
    }
    score = round(sum(components.values()), 4)
    return {"score": score, "score_components": components}


def candidate_from_sample(
    row: dict[str, Any],
    sample: dict[str, Any],
    report_path: Path,
    request: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metrics = sample.get("metrics", {})
    temporal = sample.get("temporal_coverage", {})
    collapse = sample.get("collapse", {})
    harmonic = analyze_midi_candidate(sample.get("midi_path"), request=request)
    sample_with_diagnostics = dict(sample)
    sample_with_diagnostics["harmonic_diagnostics"] = harmonic
    scored = score_candidate(sample_with_diagnostics)
    return {
        "score": scored["score"],
        "score_components": scored["score_components"],
        "review_flags": harmonic.get("review_flags", []),
        "reviewable": bool(sample.get("strict_valid")) and not harmonic.get("review_flags", []),
        "harmonic_diagnostics": harmonic,
        "mode": row.get("mode"),
        "note_groups_per_bar": _int(row.get("note_groups_per_bar")),
        "run_id": row.get("run_id"),
        "sample_index": _int(sample.get("sample_index")),
        "midi_path": sample.get("midi_path"),
        "report_path": str(report_path),
        "valid": bool(sample.get("valid")),
        "strict_valid": bool(sample.get("strict_valid")),
        "grammar_gate_passed": bool(sample.get("grammar_gate_passed")),
        "failure_reason": sample.get("failure_reason"),
        "diagnostic_failure_reason": sample.get("diagnostic_failure_reason"),
        "note_count": _int(metrics.get("note_count")),
        "unique_pitch_count": _int(metrics.get("unique_pitch_count")),
        "dead_air_ratio": _float(metrics.get("dead_air_ratio")),
        "repetition_score": _float(metrics.get("repetition_score")),
        "chord_tone_ratio": _float(metrics.get("chord_tone_ratio")),
        "bar_chord_tone_ratio": _float(harmonic.get("bar_chord_tone_ratio")),
        "min_bar_chord_tone_ratio": _float(harmonic.get("min_bar_chord_tone_ratio")),
        "dominant_pitch_ratio": _float(harmonic.get("dominant_pitch_ratio")),
        "repeated_pitch_ratio": max(
            _float(collapse.get("repeated_pitch_ratio")),
            _float(harmonic.get("repeated_pitch_ratio")),
        ),
        "onset_template_repetition_ratio": _float(harmonic.get("onset_template_repetition_ratio")),
        "onset_coverage_ratio": _float(temporal.get("onset_coverage_ratio")),
        "sustained_coverage_ratio": _float(temporal.get("sustained_coverage_ratio")),
        "position_span_ratio": _float(temporal.get("position_span_ratio")),
        "longest_sustained_empty_run_steps": _int(temporal.get("longest_sustained_empty_run_steps")),
        "collapse_warning": bool(collapse.get("collapse_warning")),
        "postprocess_removal_ratio": _float(collapse.get("postprocess_removal_ratio")),
    }


def collect_candidates(ab_report: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    candidates: list[dict[str, Any]] = []
    warnings: list[str] = []
    for row in ab_report.get("rows", []):
        report_path = Path(str(row.get("report_path", "")))
        if not report_path.is_absolute():
            report_path = ROOT_DIR / report_path
        if not report_path.exists():
            warnings.append(f"missing report: {report_path}")
            continue
        probe_report = read_json(report_path)
        request = probe_report.get("request", {})
        for sample in probe_report.get("samples", []):
            candidates.append(candidate_from_sample(row, sample, report_path, request=request))
    return candidates, warnings


def rank_candidates(candidates: list[dict[str, Any]], top_n: int) -> list[dict[str, Any]]:
    ranked = sorted(
        candidates,
        key=lambda candidate: (
            bool(candidate.get("reviewable", True)),
            bool(candidate["strict_valid"]),
            bool(candidate["valid"]),
            "low_chord_tone_ratio" not in candidate.get("review_flags", []),
            "low_bar_chord_tone_ratio" not in candidate.get("review_flags", []),
            float(candidate["score"]),
            float(candidate["onset_coverage_ratio"]),
            float(candidate.get("bar_chord_tone_ratio", candidate.get("chord_tone_ratio", 0.0))),
            -float(candidate["dead_air_ratio"]),
        ),
        reverse=True,
    )
    top = ranked[: max(1, int(top_n))]
    for index, candidate in enumerate(top, start=1):
        candidate["rank"] = index
    return top


def build_summary(candidates: list[dict[str, Any]], top_candidates: list[dict[str, Any]]) -> dict[str, Any]:
    strict_candidates = [candidate for candidate in candidates if candidate["strict_valid"]]
    valid_candidates = [candidate for candidate in candidates if candidate["valid"]]
    top_strict = [candidate for candidate in top_candidates if candidate["strict_valid"]]
    flagged_candidates = [candidate for candidate in candidates if candidate.get("review_flags")]
    viable_candidates = [candidate for candidate in candidates if candidate.get("reviewable")]
    top_viable = [candidate for candidate in top_candidates if candidate.get("reviewable")]
    return {
        "candidate_count": int(len(candidates)),
        "valid_candidate_count": int(len(valid_candidates)),
        "strict_candidate_count": int(len(strict_candidates)),
        "viable_candidate_count": int(len(viable_candidates)),
        "top_candidate_count": int(len(top_candidates)),
        "top_strict_candidate_count": int(len(top_strict)),
        "top_viable_candidate_count": int(len(top_viable)),
        "flagged_candidate_count": int(len(flagged_candidates)),
        "best_candidate": top_candidates[0] if top_candidates else None,
    }


def markdown_report(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Stage B Candidate Ranking",
        "",
        f"- candidate count: `{summary['candidate_count']}`",
        f"- valid candidates: `{summary['valid_candidate_count']}`",
        f"- strict candidates: `{summary['strict_candidate_count']}`",
        f"- viable candidates without review flags: `{summary['viable_candidate_count']}`",
        "",
        "| rank | score | mode | groups/bar | sample | strict | reviewable | notes | onset | sustained | dead-air | chord-tone | bar chord | flags | MIDI |",
        "|---:|---:|---|---:|---:|:---:|:---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for candidate in report["top_candidates"]:
        lines.append(
            "| {rank} | {score:.3f} | {mode} | {note_groups_per_bar} | {sample_index} | "
            "{strict_valid} | {reviewable} | {note_count} | {onset_coverage_ratio:.3f} | "
            "{sustained_coverage_ratio:.3f} | {dead_air_ratio:.3f} | "
            "{chord_tone_ratio:.3f} | {bar_chord_tone_ratio:.3f} | "
            "{review_flags} | `{midi_path}` |".format(**candidate)
        )
    lines.append("")
    lines.append("## Scoring Note")
    lines.append("")
    lines.append("This score is a review-prioritization heuristic, not a musical-quality claim.")
    lines.append("It rewards strict validity, temporal coverage, bar-aware chord-tone ratio, and pitch diversity.")
    lines.append("It penalizes dead-air, repetition, postprocess removal, collapse warnings, repeated pitch dominance, low per-bar chord tones, and repeated onset templates.")
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rank Stage B candidate MIDI samples from an A/B sweep")
    parser.add_argument(
        "--ab_sweep_report",
        type=str,
        default=str(
            ROOT_DIR
            / "outputs"
            / "stage_b_coverage_ab_sweep"
            / "harness_stage_b_coverage_ab_sweep"
            / "ab_sweep_report.json"
        ),
    )
    parser.add_argument("--output_root", type=str, default=str(ROOT_DIR / "outputs" / "stage_b_candidate_ranking"))
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--issue_number", type=int, default=41)
    parser.add_argument("--top_n", type=int, default=12)
    parser.add_argument("--min_top_strict_candidates", type=int, default=1)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_root) / run_id
    ab_report_path = Path(args.ab_sweep_report)
    ab_report = read_json(ab_report_path)
    candidates, warnings = collect_candidates(ab_report)
    top_candidates = rank_candidates(candidates, top_n=args.top_n)
    summary = build_summary(candidates, top_candidates)
    report = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "issue": int(args.issue_number),
        "ab_sweep_report": str(ab_report_path),
        "summary": summary,
        "warnings": warnings,
        "top_candidates": top_candidates,
    }
    write_json(run_dir / "candidate_rank_report.json", report)
    (run_dir / "candidate_rank_report.md").write_text(markdown_report(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True, indent=2))
    return 0 if int(summary["top_strict_candidate_count"]) >= int(args.min_top_strict_candidates) else 3


if __name__ == "__main__":
    raise SystemExit(main())
