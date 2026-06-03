"""Analyze songlike melody rejection from model-direct MIDI note evidence."""

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

from scripts.assess_stage_b_generic_base_readiness import write_json, write_text  # noqa: E402
from scripts.fill_stage_b_midi_to_solo_model_direct_user_listening_review import (  # noqa: E402
    BOUNDARY as SOURCE_BOUNDARY,
)


class StageBMidiToSoloModelDirectSonglikeRejectionAnalysisError(ValueError):
    pass


BOUNDARY = "stage_b_midi_to_solo_model_direct_songlike_melody_rejection_analysis"
NEXT_BOUNDARY = "stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_decision"
SCHEMA_VERSION = "stage_b_midi_to_solo_model_direct_songlike_melody_rejection_analysis_v1"


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise StageBMidiToSoloModelDirectSonglikeRejectionAnalysisError(f"report missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


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


def most_common_ratio(values: list[float | int], *, precision: int = 3) -> float:
    if not values:
        return 0.0
    rounded = [round(float(value), precision) for value in values]
    return max(Counter(rounded).values()) / len(rounded)


def load_notes(path: str | Path) -> tuple[pretty_midi.PrettyMIDI, list[pretty_midi.Note]]:
    midi = pretty_midi.PrettyMIDI(str(path))
    notes: list[pretty_midi.Note] = []
    for instrument in midi.instruments:
        if not instrument.is_drum:
            notes.extend(instrument.notes)
    return midi, sorted(notes, key=lambda note: (float(note.start), int(note.pitch), float(note.end)))


def repeated_cycle(values: list[tuple[float, ...]], *, cycle_length: int) -> bool:
    if len(values) < cycle_length * 2:
        return False
    return values[:cycle_length] == values[cycle_length : cycle_length * 2]


def bar_patterns_for_notes(
    midi: pretty_midi.PrettyMIDI,
    notes: list[pretty_midi.Note],
) -> tuple[list[int], list[tuple[float, ...]]]:
    downbeats = [float(value) for value in midi.get_downbeats()]
    end_time = float(midi.get_end_time())
    if not downbeats:
        downbeats = [0.0]
    bar_length = downbeats[-1] - downbeats[-2] if len(downbeats) >= 2 else max(0.0001, end_time)
    while downbeats[-1] < end_time:
        downbeats.append(downbeats[-1] + bar_length)
    note_counts: list[int] = []
    onset_patterns: list[tuple[float, ...]] = []
    for index in range(len(downbeats) - 1):
        start = downbeats[index]
        end = downbeats[index + 1]
        bar_length = max(0.0001, end - start)
        bar_notes = [note for note in notes if start <= float(note.start) < end]
        note_counts.append(len(bar_notes))
        offsets = tuple(round((float(note.start) - start) / bar_length * 4.0, 3) for note in bar_notes)
        onset_patterns.append(offsets)
    return note_counts, onset_patterns


def rhythm_signature(notes: list[pretty_midi.Note]) -> tuple[tuple[float, ...], tuple[float, ...]]:
    starts = [round(float(note.start), 3) for note in notes]
    durations = [round(max(0.0, float(note.end) - float(note.start)), 3) for note in notes]
    iois = [round(starts[index] - starts[index - 1], 3) for index in range(1, len(starts))]
    return tuple(iois), tuple(durations)


def analyze_midi_candidate(path: str | Path, *, rank: int) -> dict[str, Any]:
    midi, notes = load_notes(path)
    if not notes:
        return {
            "rank": int(rank),
            "midi_path": str(path),
            "note_count": 0,
            "analysis_flags": ["empty_midi"],
        }
    pitches = [int(note.pitch) for note in notes]
    starts = [round(float(note.start), 3) for note in notes]
    durations = [round(max(0.0, float(note.end) - float(note.start)), 3) for note in notes]
    signed_intervals = [pitches[index] - pitches[index - 1] for index in range(1, len(pitches))]
    abs_intervals = [abs(value) for value in signed_intervals]
    iois = [round(starts[index] - starts[index - 1], 3) for index in range(1, len(starts))]
    note_counts_per_bar, onset_patterns = bar_patterns_for_notes(midi, notes)
    flags: list[str] = []
    max_abs_interval = max(abs_intervals) if abs_intervals else 0
    small_interval_ratio = sum(1 for value in abs_intervals if value <= 4) / max(1, len(abs_intervals))
    large_interval_ratio = sum(1 for value in abs_intervals if value >= 12) / max(1, len(abs_intervals))
    duration_most_common = most_common_ratio(durations)
    ioi_most_common = most_common_ratio(iois)
    note_count_bar_ratio = most_common_ratio(note_counts_per_bar, precision=0)
    if note_counts_per_bar and note_count_bar_ratio >= 0.95:
        flags.append("uniform_bar_density")
    if note_counts_per_bar and Counter(note_counts_per_bar).most_common(1)[0][0] == 4:
        flags.append("four_notes_per_bar_template")
    if duration_most_common >= 0.40:
        flags.append("duration_template_monotony")
    if ioi_most_common >= 0.40:
        flags.append("ioi_template_monotony")
    if max_abs_interval <= 9 and large_interval_ratio == 0.0:
        flags.append("safe_interval_cap_compression")
    if small_interval_ratio >= 0.55:
        flags.append("stepwise_contour_bias")
    if repeated_cycle(onset_patterns, cycle_length=4):
        flags.append("four_bar_rhythm_cycle_repeated")
    return {
        "rank": int(rank),
        "midi_path": str(path),
        "note_count": len(notes),
        "bar_count": len(note_counts_per_bar),
        "note_counts_per_bar": note_counts_per_bar,
        "most_common_note_count_per_bar": Counter(note_counts_per_bar).most_common(1)[0][0]
        if note_counts_per_bar
        else 0,
        "note_count_per_bar_most_common_ratio": note_count_bar_ratio,
        "unique_onset_pattern_count": len(set(onset_patterns)),
        "four_bar_rhythm_cycle_repeated": repeated_cycle(onset_patterns, cycle_length=4),
        "unique_pitch_count": len(set(pitches)),
        "pitch_span": max(pitches) - min(pitches),
        "max_abs_interval": max_abs_interval,
        "avg_abs_interval": sum(abs_intervals) / len(abs_intervals) if abs_intervals else 0.0,
        "small_interval_ratio_le4": small_interval_ratio,
        "large_interval_ratio_gte12": large_interval_ratio,
        "duration_most_common_ratio": duration_most_common,
        "ioi_most_common_ratio": ioi_most_common,
        "rhythm_signature": rhythm_signature(notes),
        "analysis_flags": flags,
    }


def validate_source_review_fill(report: dict[str, Any]) -> list[dict[str, Any]]:
    review = _dict(report.get("user_listening_review"))
    claim = _dict(report.get("claim_boundary"))
    if str(report.get("boundary") or "") != SOURCE_BOUNDARY:
        raise StageBMidiToSoloModelDirectSonglikeRejectionAnalysisError("user listening review fill boundary required")
    if str(review.get("overall_decision") or "") != "reject_all":
        raise StageBMidiToSoloModelDirectSonglikeRejectionAnalysisError("reject_all listening review required")
    if str(review.get("primary_failure") or "") != "songlike_melody_not_soloing":
        raise StageBMidiToSoloModelDirectSonglikeRejectionAnalysisError("songlike melody failure required")
    blocked_claims = [
        "human_audio_preference_claimed",
        "model_direct_candidate_keep_claimed",
        "midi_to_solo_musical_quality_claimed",
        "model_direct_generation_quality_claimed",
        "broad_trained_model_quality_claimed",
        "brad_style_adaptation_claimed",
    ]
    claimed = [name for name in blocked_claims if bool(claim.get(name, False))]
    if claimed:
        raise StageBMidiToSoloModelDirectSonglikeRejectionAnalysisError(f"unexpected source claim: {claimed}")
    candidates = [_dict(item) for item in _list(report.get("reviewed_candidates"))]
    if len(candidates) < 3:
        raise StageBMidiToSoloModelDirectSonglikeRejectionAnalysisError("at least 3 reviewed candidates required")
    return candidates


def build_songlike_rejection_analysis_report(
    review_fill_report: dict[str, Any],
    *,
    output_dir: Path,
) -> dict[str, Any]:
    source_candidates = validate_source_review_fill(review_fill_report)
    candidate_analyses = [
        analyze_midi_candidate(str(item.get("midi_path") or ""), rank=_int(item.get("rank")))
        for item in source_candidates[:3]
    ]
    rhythm_signatures = [tuple(item.get("rhythm_signature", ((), ()))) for item in candidate_analyses]
    shared_rhythm_signature_count = max(Counter(rhythm_signatures).values()) if rhythm_signatures else 0
    flag_counts = Counter(flag for item in candidate_analyses for flag in _list(item.get("analysis_flags")))
    key_failure_signals = [
        "uniform_bar_density",
        "four_notes_per_bar_template",
        "duration_template_monotony",
        "ioi_template_monotony",
        "safe_interval_cap_compression",
        "four_bar_rhythm_cycle_repeated",
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "boundary": BOUNDARY,
        "source_boundary": SOURCE_BOUNDARY,
        "source_review_summary": {
            "preferred_rank": _int(_dict(review_fill_report.get("user_listening_review")).get("preferred_rank")),
            "overall_decision": str(_dict(review_fill_report.get("user_listening_review")).get("overall_decision") or ""),
            "primary_failure": str(_dict(review_fill_report.get("user_listening_review")).get("primary_failure") or ""),
        },
        "candidate_analyses": candidate_analyses,
        "aggregate": {
            "candidate_count": len(candidate_analyses),
            "flag_counts": dict(sorted(flag_counts.items())),
            "uniform_bar_density_count": flag_counts.get("uniform_bar_density", 0),
            "four_notes_per_bar_template_count": flag_counts.get("four_notes_per_bar_template", 0),
            "duration_template_monotony_count": flag_counts.get("duration_template_monotony", 0),
            "ioi_template_monotony_count": flag_counts.get("ioi_template_monotony", 0),
            "safe_interval_cap_compression_count": flag_counts.get("safe_interval_cap_compression", 0),
            "four_bar_rhythm_cycle_repeated_count": flag_counts.get("four_bar_rhythm_cycle_repeated", 0),
            "shared_rhythm_signature_count": shared_rhythm_signature_count,
            "max_abs_interval_max": max((_int(item.get("max_abs_interval")) for item in candidate_analyses), default=0),
            "max_duration_most_common_ratio": max(
                (_float(item.get("duration_most_common_ratio")) for item in candidate_analyses),
                default=0.0,
            ),
            "max_ioi_most_common_ratio": max(
                (_float(item.get("ioi_most_common_ratio")) for item in candidate_analyses),
                default=0.0,
            ),
            "key_failure_signals": [signal for signal in key_failure_signals if flag_counts.get(signal, 0) > 0],
        },
        "readiness": {
            "boundary": BOUNDARY,
            "songlike_rejection_analysis_completed": True,
            "single_user_review_input_used": True,
            "midi_note_evidence_used": True,
            "human_audio_preference_claimed": False,
            "model_direct_candidate_keep_claimed": False,
            "model_direct_generation_quality_claimed": False,
            "midi_to_solo_musical_quality_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
        },
        "decision": {
            "current_boundary": BOUNDARY,
            "next_boundary": NEXT_BOUNDARY,
            "auto_progress_allowed": True,
            "critical_user_input_required": False,
            "reason": "songlike rejection is supported by repeated rhythm density and constrained interval vocabulary signals",
        },
        "not_proven": [
            "jazz_solo_musical_quality",
            "human_audio_keep_preference",
            "model_direct_generation_quality",
            "broad_trained_model_quality",
            "brad_style_adaptation",
        ],
        "next_recommended_issue": "Stage B MIDI-to-solo model-direct jazz phrase vocabulary repair decision",
    }


def validate_songlike_rejection_analysis_report(
    report: dict[str, Any],
    *,
    expected_boundary: str | None,
    expected_next_boundary: str | None,
    require_analysis_completed: bool,
    require_rejection_signals: bool,
    require_no_quality_claim: bool,
) -> dict[str, Any]:
    boundary = str(report.get("boundary") or "")
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    aggregate = _dict(report.get("aggregate"))
    if expected_boundary and boundary != expected_boundary:
        raise StageBMidiToSoloModelDirectSonglikeRejectionAnalysisError(
            f"expected boundary {expected_boundary}, got {boundary}"
        )
    if expected_next_boundary and str(decision.get("next_boundary") or "") != expected_next_boundary:
        raise StageBMidiToSoloModelDirectSonglikeRejectionAnalysisError("unexpected next boundary")
    if require_analysis_completed and not bool(readiness.get("songlike_rejection_analysis_completed", False)):
        raise StageBMidiToSoloModelDirectSonglikeRejectionAnalysisError("analysis completion required")
    if require_rejection_signals and not _list(aggregate.get("key_failure_signals")):
        raise StageBMidiToSoloModelDirectSonglikeRejectionAnalysisError("rejection signals required")
    if require_no_quality_claim:
        blocked = [
            "human_audio_preference_claimed",
            "model_direct_candidate_keep_claimed",
            "model_direct_generation_quality_claimed",
            "midi_to_solo_musical_quality_claimed",
            "broad_trained_model_quality_claimed",
            "brad_style_adaptation_claimed",
        ]
        claimed = [name for name in blocked if bool(readiness.get(name, True))]
        if claimed:
            raise StageBMidiToSoloModelDirectSonglikeRejectionAnalysisError(
                f"unexpected quality claim: {claimed}"
            )
    return {
        "boundary": boundary,
        "source_boundary": str(report.get("source_boundary") or ""),
        "candidate_count": _int(aggregate.get("candidate_count")),
        "uniform_bar_density_count": _int(aggregate.get("uniform_bar_density_count")),
        "four_notes_per_bar_template_count": _int(aggregate.get("four_notes_per_bar_template_count")),
        "duration_template_monotony_count": _int(aggregate.get("duration_template_monotony_count")),
        "ioi_template_monotony_count": _int(aggregate.get("ioi_template_monotony_count")),
        "safe_interval_cap_compression_count": _int(aggregate.get("safe_interval_cap_compression_count")),
        "four_bar_rhythm_cycle_repeated_count": _int(aggregate.get("four_bar_rhythm_cycle_repeated_count")),
        "shared_rhythm_signature_count": _int(aggregate.get("shared_rhythm_signature_count")),
        "key_failure_signals": _list(aggregate.get("key_failure_signals")),
        "max_abs_interval_max": _int(aggregate.get("max_abs_interval_max")),
        "human_audio_preference_claimed": bool(readiness.get("human_audio_preference_claimed", True)),
        "midi_to_solo_musical_quality_claimed": bool(
            readiness.get("midi_to_solo_musical_quality_claimed", True)
        ),
        "critical_user_input_required": bool(decision.get("critical_user_input_required", True)),
        "next_boundary": str(decision.get("next_boundary") or ""),
        "next_recommended_issue": str(report.get("next_recommended_issue") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    aggregate = report["aggregate"]
    readiness = report["readiness"]
    decision = report["decision"]
    lines = [
        "# Stage B MIDI-to-Solo Model-Direct Songlike Melody Rejection Analysis",
        "",
        "## Summary",
        "",
        f"- boundary: `{report['boundary']}`",
        f"- source boundary: `{report['source_boundary']}`",
        f"- next boundary: `{decision['next_boundary']}`",
        f"- candidate count: `{aggregate['candidate_count']}`",
        f"- uniform bar density count: `{aggregate['uniform_bar_density_count']}`",
        f"- four-notes-per-bar template count: `{aggregate['four_notes_per_bar_template_count']}`",
        f"- duration template monotony count: `{aggregate['duration_template_monotony_count']}`",
        f"- IOI template monotony count: `{aggregate['ioi_template_monotony_count']}`",
        f"- safe interval cap compression count: `{aggregate['safe_interval_cap_compression_count']}`",
        f"- four-bar rhythm cycle repeated count: `{aggregate['four_bar_rhythm_cycle_repeated_count']}`",
        f"- shared rhythm signature count: `{aggregate['shared_rhythm_signature_count']}`",
        f"- max abs interval max: `{aggregate['max_abs_interval_max']}`",
        f"- human/audio preference claimed: `{_bool_token(readiness['human_audio_preference_claimed'])}`",
        f"- MIDI-to-solo musical quality claimed: `{_bool_token(readiness['midi_to_solo_musical_quality_claimed'])}`",
        "",
        "## Candidate Signals",
        "",
        "| rank | notes | bars | notes/bar mode | max interval | duration ratio | IOI ratio | flags |",
        "|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in report.get("candidate_analyses", []):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(item["rank"]),
                    str(item["note_count"]),
                    str(item["bar_count"]),
                    str(item["most_common_note_count_per_bar"]),
                    str(item["max_abs_interval"]),
                    f"{float(item['duration_most_common_ratio']):.4f}",
                    f"{float(item['ioi_most_common_ratio']):.4f}",
                    ", ".join(f"`{flag}`" for flag in item.get("analysis_flags", [])),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Not Proven", ""])
    for item in report.get("not_proven", []):
        lines.append(f"- `{item}`")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze model-direct songlike melody rejection")
    parser.add_argument("--user_listening_review_fill", type=str, required=True)
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/stage_b_midi_to_solo_model_direct_songlike_melody_rejection_analysis",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--expected_boundary", type=str, default="")
    parser.add_argument("--expected_next_boundary", type=str, default="")
    parser.add_argument("--require_analysis_completed", action="store_true")
    parser.add_argument("--require_rejection_signals", action="store_true")
    parser.add_argument("--require_no_quality_claim", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    report = build_songlike_rejection_analysis_report(
        read_json(Path(args.user_listening_review_fill)),
        output_dir=output_dir,
    )
    summary = validate_songlike_rejection_analysis_report(
        report,
        expected_boundary=str(args.expected_boundary or ""),
        expected_next_boundary=str(args.expected_next_boundary or ""),
        require_analysis_completed=bool(args.require_analysis_completed),
        require_rejection_signals=bool(args.require_rejection_signals),
        require_no_quality_claim=bool(args.require_no_quality_claim),
    )
    write_json(output_dir / "stage_b_midi_to_solo_model_direct_songlike_melody_rejection_analysis.json", report)
    write_json(
        output_dir
        / "stage_b_midi_to_solo_model_direct_songlike_melody_rejection_analysis_validation_summary.json",
        summary,
    )
    markdown = markdown_report(report)
    write_text(output_dir / "stage_b_midi_to_solo_model_direct_songlike_melody_rejection_analysis.md", markdown)
    if args.doc_path:
        write_text(Path(args.doc_path), markdown)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
