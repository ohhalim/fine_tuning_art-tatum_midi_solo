"""Diagnose phrase-quality risks from model-direct MIDI note evidence."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pretty_midi

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from scripts.assess_stage_b_generic_base_readiness import read_json, write_json, write_text  # noqa: E402


class StageBMidiToSoloModelDirectPhraseQualityDiagnosticsError(ValueError):
    pass


SOURCE_BOUNDARY = "stage_b_midi_to_solo_model_direct_audio_evidence_consolidation"
BOUNDARY = "stage_b_midi_to_solo_model_direct_phrase_quality_diagnostics"
PITCH_REPAIR_BOUNDARY = "stage_b_midi_to_solo_model_direct_pitch_contour_repetition_repair"
TIMING_REPAIR_BOUNDARY = "stage_b_midi_to_solo_model_direct_timing_phrase_repair"
LISTENING_REVIEW_BOUNDARY = "stage_b_midi_to_solo_model_direct_listening_review_package"
SCHEMA_VERSION = "stage_b_midi_to_solo_model_direct_phrase_quality_diagnostics_v1"


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _bool_token(value: Any) -> str:
    return "true" if bool(value) else "false"


def load_midi_notes(path: str | Path) -> list[pretty_midi.Note]:
    midi = pretty_midi.PrettyMIDI(str(path))
    notes: list[pretty_midi.Note] = []
    for instrument in midi.instruments:
        if not instrument.is_drum:
            notes.extend(instrument.notes)
    return sorted(notes, key=lambda note: (float(note.start), int(note.pitch), float(note.end)))


def repeated_ngram_count(pitches: list[int], n: int) -> int:
    if len(pitches) < n * 2:
        return 0
    grams = [tuple(pitches[index : index + n]) for index in range(len(pitches) - n + 1)]
    counts = Counter(grams)
    return sum(count - 1 for count in counts.values() if count > 1)


def most_common_ratio(values: list[float | int], *, precision: int = 3) -> float:
    if not values:
        return 0.0
    rounded = [round(float(value), precision) for value in values]
    return max(Counter(rounded).values()) / len(rounded)


def note_metrics_for_path(path: str | Path, *, dead_air_threshold_seconds: float) -> dict[str, Any]:
    notes = load_midi_notes(path)
    if not notes:
        return {
            "midi_path": str(path),
            "note_count": 0,
            "diagnostic_flags": ["empty_midi"],
        }
    pitches = [int(note.pitch) for note in notes]
    starts = [float(note.start) for note in notes]
    durations = [max(0.0, float(note.end) - float(note.start)) for note in notes]
    intervals = [abs(pitches[index] - pitches[index - 1]) for index in range(1, len(pitches))]
    iois = [max(0.0, starts[index] - starts[index - 1]) for index in range(1, len(starts))]
    dead_air_events = sum(1 for gap in iois if gap >= float(dead_air_threshold_seconds))
    adjacent_pitch_repeats = sum(1 for index in range(1, len(pitches)) if pitches[index] == pitches[index - 1])
    large_intervals = [interval for interval in intervals if interval >= 12]
    max_interval = max(intervals) if intervals else 0
    pitch_min = min(pitches)
    pitch_max = max(pitches)
    duration_most_common_ratio = most_common_ratio(durations)
    ioi_most_common_ratio = most_common_ratio(iois)
    flags: list[str] = []
    if len(set(pitches)) < 8:
        flags.append("narrow_pitch_vocabulary")
    if adjacent_pitch_repeats > 0:
        flags.append("adjacent_pitch_repetition")
    if max_interval >= 12 or (len(large_intervals) / max(1, len(intervals))) >= 0.20:
        flags.append("wide_interval_contour")
    if duration_most_common_ratio >= 0.75:
        flags.append("duration_monotony")
    if ioi_most_common_ratio >= 0.75:
        flags.append("ioi_monotony")
    if dead_air_events / max(1, len(iois)) >= 0.35:
        flags.append("dead_air_gap")
    if pitch_max - pitch_min > 24:
        flags.append("wide_register_span")
    return {
        "midi_path": str(path),
        "note_count": len(notes),
        "unique_pitch_count": len(set(pitches)),
        "unique_pitch_class_count": len({pitch % 12 for pitch in pitches}),
        "pitch_min": pitch_min,
        "pitch_max": pitch_max,
        "pitch_span": pitch_max - pitch_min,
        "max_interval": max_interval,
        "avg_abs_interval": sum(intervals) / len(intervals) if intervals else 0.0,
        "large_interval_count": len(large_intervals),
        "large_interval_ratio": len(large_intervals) / max(1, len(intervals)),
        "adjacent_pitch_repeats": adjacent_pitch_repeats,
        "repeated_3gram_count": repeated_ngram_count(pitches, 3),
        "repeated_4gram_count": repeated_ngram_count(pitches, 4),
        "avg_duration_seconds": sum(durations) / len(durations),
        "max_duration_seconds": max(durations),
        "unique_duration_count": len({round(duration, 3) for duration in durations}),
        "duration_most_common_ratio": duration_most_common_ratio,
        "unique_ioi_count": len({round(ioi, 3) for ioi in iois}),
        "ioi_most_common_ratio": ioi_most_common_ratio,
        "dead_air_ratio": dead_air_events / max(1, len(iois)),
        "phrase_start_seconds": min(starts),
        "phrase_end_seconds": max(float(note.end) for note in notes),
        "diagnostic_flags": flags,
    }


def next_boundary_from_flags(candidates: list[dict[str, Any]]) -> str:
    all_flags = {flag for candidate in candidates for flag in _list(candidate.get("diagnostic_flags"))}
    if {"wide_interval_contour", "adjacent_pitch_repetition", "narrow_pitch_vocabulary"} & all_flags:
        return PITCH_REPAIR_BOUNDARY
    if {"duration_monotony", "ioi_monotony", "dead_air_gap"} & all_flags:
        return TIMING_REPAIR_BOUNDARY
    return LISTENING_REVIEW_BOUNDARY


def build_phrase_quality_diagnostics_report(
    *,
    evidence_report: dict[str, Any],
    output_dir: Path,
    issue_number: int,
    dead_air_threshold_seconds: float,
) -> dict[str, Any]:
    readiness = _dict(evidence_report.get("readiness"))
    objective = _dict(evidence_report.get("objective_evidence"))
    if str(evidence_report.get("boundary") or readiness.get("boundary") or "") != SOURCE_BOUNDARY:
        raise StageBMidiToSoloModelDirectPhraseQualityDiagnosticsError("audio evidence boundary required")
    if not bool(readiness.get("model_direct_midi_to_wav_technical_path_completed", False)):
        raise StageBMidiToSoloModelDirectPhraseQualityDiagnosticsError("model-direct technical path must be complete")
    if bool(readiness.get("model_direct_generation_quality_claimed", True)):
        raise StageBMidiToSoloModelDirectPhraseQualityDiagnosticsError("model quality must not be claimed upstream")
    midi_paths = [str(path) for path in _list(objective.get("midi_paths"))]
    if len(midi_paths) < 3:
        raise StageBMidiToSoloModelDirectPhraseQualityDiagnosticsError("at least 3 MIDI paths required")
    candidates = [
        {
            "rank": index,
            **note_metrics_for_path(path, dead_air_threshold_seconds=dead_air_threshold_seconds),
        }
        for index, path in enumerate(midi_paths[:3], start=1)
    ]
    next_boundary = next_boundary_from_flags(candidates)
    flag_counts = Counter(flag for candidate in candidates for flag in _list(candidate.get("diagnostic_flags")))
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "issue_number": int(issue_number),
        "boundary": BOUNDARY,
        "source_boundary": SOURCE_BOUNDARY,
        "diagnostic_config": {
            "dead_air_threshold_seconds": float(dead_air_threshold_seconds),
            "wide_interval_threshold_semitones": 12,
            "duration_monotony_ratio_threshold": 0.75,
            "ioi_monotony_ratio_threshold": 0.75,
            "dead_air_ratio_threshold": 0.35,
        },
        "candidate_diagnostics": candidates,
        "aggregate": {
            "candidate_count": len(candidates),
            "flag_counts": dict(sorted(flag_counts.items())),
            "max_interval_max": max((_int(candidate.get("max_interval")) for candidate in candidates), default=0),
            "adjacent_pitch_repeat_total": sum(
                _int(candidate.get("adjacent_pitch_repeats")) for candidate in candidates
            ),
            "max_duration_most_common_ratio": max(
                (_float(candidate.get("duration_most_common_ratio")) for candidate in candidates),
                default=0.0,
            ),
            "max_dead_air_ratio": max(
                (_float(candidate.get("dead_air_ratio")) for candidate in candidates),
                default=0.0,
            ),
        },
        "readiness": {
            "boundary": BOUNDARY,
            "phrase_quality_diagnostics_completed": True,
            "model_direct_midi_to_wav_technical_path_completed": True,
            "model_direct_generation_quality_claimed": False,
            "midi_to_solo_musical_quality_claimed": False,
            "human_audio_preference_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
        },
        "decision": {
            "current_boundary": BOUNDARY,
            "next_boundary": next_boundary,
            "auto_progress_allowed": True,
            "critical_user_input_required": False,
            "reason": "MIDI note diagnostics selected the next repair boundary without musical quality claim",
        },
        "not_proven": [
            "model_direct_generation_quality",
            "midi_to_solo_musical_quality",
            "human_audio_preference",
            "broad_trained_model_quality",
            "brad_style_adaptation",
        ],
        "next_recommended_issue": {
            PITCH_REPAIR_BOUNDARY: "Stage B MIDI-to-solo model-direct pitch contour repetition repair",
            TIMING_REPAIR_BOUNDARY: "Stage B MIDI-to-solo model-direct timing phrase repair",
            LISTENING_REVIEW_BOUNDARY: "Stage B MIDI-to-solo model-direct listening review package",
        }[next_boundary],
    }


def validate_phrase_quality_diagnostics_report(
    report: dict[str, Any],
    *,
    expected_boundary: str | None,
    require_diagnostics_completed: bool,
    require_no_quality_claim: bool,
    min_candidate_count: int,
) -> dict[str, Any]:
    boundary = str(report.get("boundary") or "")
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    aggregate = _dict(report.get("aggregate"))
    candidates = _list(report.get("candidate_diagnostics"))
    if expected_boundary and boundary != expected_boundary:
        raise StageBMidiToSoloModelDirectPhraseQualityDiagnosticsError(
            f"expected boundary {expected_boundary}, got {boundary}"
        )
    if require_diagnostics_completed and not bool(readiness.get("phrase_quality_diagnostics_completed", False)):
        raise StageBMidiToSoloModelDirectPhraseQualityDiagnosticsError("diagnostics must be completed")
    if len(candidates) < int(min_candidate_count):
        raise StageBMidiToSoloModelDirectPhraseQualityDiagnosticsError("candidate diagnostics below threshold")
    for candidate in candidates[: int(min_candidate_count)]:
        if not Path(str(_dict(candidate).get("midi_path") or "")).exists():
            raise StageBMidiToSoloModelDirectPhraseQualityDiagnosticsError("diagnostic MIDI path missing")
        if _int(_dict(candidate).get("note_count")) <= 0:
            raise StageBMidiToSoloModelDirectPhraseQualityDiagnosticsError("diagnostic note count required")
    if bool(decision.get("critical_user_input_required", True)):
        raise StageBMidiToSoloModelDirectPhraseQualityDiagnosticsError("critical user input should not be required")
    if require_no_quality_claim:
        blocked = [
            "model_direct_generation_quality_claimed",
            "midi_to_solo_musical_quality_claimed",
            "human_audio_preference_claimed",
            "broad_trained_model_quality_claimed",
            "brad_style_adaptation_claimed",
        ]
        claimed = [name for name in blocked if bool(readiness.get(name, True))]
        if claimed:
            raise StageBMidiToSoloModelDirectPhraseQualityDiagnosticsError(
                f"unexpected quality claim: {claimed}"
            )
    return {
        "boundary": boundary,
        "next_boundary": str(decision.get("next_boundary") or ""),
        "candidate_count": _int(aggregate.get("candidate_count")),
        "flag_counts": _dict(aggregate.get("flag_counts")),
        "max_interval_max": _int(aggregate.get("max_interval_max")),
        "adjacent_pitch_repeat_total": _int(aggregate.get("adjacent_pitch_repeat_total")),
        "max_duration_most_common_ratio": _float(aggregate.get("max_duration_most_common_ratio")),
        "max_dead_air_ratio": _float(aggregate.get("max_dead_air_ratio")),
        "model_direct_generation_quality_claimed": bool(
            readiness.get("model_direct_generation_quality_claimed", True)
        ),
        "human_audio_preference_claimed": bool(readiness.get("human_audio_preference_claimed", True)),
        "critical_user_input_required": bool(decision.get("critical_user_input_required", True)),
        "next_recommended_issue": str(report.get("next_recommended_issue") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    readiness = report["readiness"]
    decision = report["decision"]
    aggregate = report["aggregate"]
    lines = [
        "# Stage B MIDI-to-Solo Model-Direct Phrase Quality Diagnostics",
        "",
        "## Summary",
        "",
        f"- boundary: `{report['boundary']}`",
        f"- next boundary: `{decision['next_boundary']}`",
        f"- candidate count: `{aggregate['candidate_count']}`",
        f"- flag counts: `{aggregate['flag_counts']}`",
        f"- max interval max: `{aggregate['max_interval_max']}`",
        f"- adjacent pitch repeat total: `{aggregate['adjacent_pitch_repeat_total']}`",
        f"- max duration most-common ratio: `{aggregate['max_duration_most_common_ratio']}`",
        f"- max dead-air ratio: `{aggregate['max_dead_air_ratio']}`",
        f"- model-direct generation quality claimed: `{_bool_token(readiness['model_direct_generation_quality_claimed'])}`",
        f"- human/audio preference claimed: `{_bool_token(readiness['human_audio_preference_claimed'])}`",
        "",
        "## Candidate Diagnostics",
        "",
        "| rank | notes | unique pitch | range | max interval | adjacent repeats | duration ratio | dead-air | flags |",
        "|---:|---:|---:|---|---:|---:|---:|---:|---|",
    ]
    for candidate in report.get("candidate_diagnostics", []):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(candidate["rank"]),
                    str(candidate["note_count"]),
                    str(candidate["unique_pitch_count"]),
                    f"{candidate['pitch_min']}-{candidate['pitch_max']}",
                    str(candidate["max_interval"]),
                    str(candidate["adjacent_pitch_repeats"]),
                    f"{float(candidate['duration_most_common_ratio']):.3f}",
                    f"{float(candidate['dead_air_ratio']):.3f}",
                    ", ".join(candidate["diagnostic_flags"]) or "none",
                ]
            )
            + " |"
        )
    lines.extend(["", "## Not Proven", ""])
    for item in report.get("not_proven", []):
        lines.append(f"- `{item}`")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose model-direct phrase-quality risks")
    parser.add_argument("--audio_evidence_report", type=str, required=True)
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/stage_b_midi_to_solo_model_direct_phrase_quality_diagnostics",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--issue_number", type=int, default=505)
    parser.add_argument("--dead_air_threshold_seconds", type=float, default=0.5)
    parser.add_argument("--expected_boundary", type=str, default="")
    parser.add_argument("--min_candidate_count", type=int, default=3)
    parser.add_argument("--require_diagnostics_completed", action="store_true")
    parser.add_argument("--require_no_quality_claim", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    report = build_phrase_quality_diagnostics_report(
        evidence_report=read_json(Path(args.audio_evidence_report)),
        output_dir=output_dir,
        issue_number=int(args.issue_number),
        dead_air_threshold_seconds=float(args.dead_air_threshold_seconds),
    )
    summary = validate_phrase_quality_diagnostics_report(
        report,
        expected_boundary=str(args.expected_boundary or ""),
        require_diagnostics_completed=bool(args.require_diagnostics_completed),
        require_no_quality_claim=bool(args.require_no_quality_claim),
        min_candidate_count=int(args.min_candidate_count),
    )
    write_json(output_dir / "stage_b_midi_to_solo_model_direct_phrase_quality_diagnostics.json", report)
    write_json(
        output_dir / "stage_b_midi_to_solo_model_direct_phrase_quality_diagnostics_validation_summary.json",
        summary,
    )
    markdown = markdown_report(report)
    write_text(output_dir / "stage_b_midi_to_solo_model_direct_phrase_quality_diagnostics.md", markdown)
    if args.doc_path:
        write_text(Path(args.doc_path), markdown)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
