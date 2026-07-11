"""Review repaired top8 Music Transformer solo candidates from objective MIDI evidence."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

import pretty_midi

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from inference.app.fallback import parse_chord  # noqa: E402
from scripts.assess_stage_b_generic_base_readiness import write_json, write_text  # noqa: E402


SCHEMA_VERSION = "music_transformer_solo_yield_repaired_top8_objective_failure_review_v1"
BOUNDARY = "music_transformer_solo_yield_repaired_top8_objective_failure_review"
NEXT_BOUNDARY = "music_transformer_solo_yield_chord_tone_landing_repair_sweep"
SELECTED_TARGET = "chord_tone_landing_repair"


class SoloYieldRepairedTop8ObjectiveFailureReviewError(ValueError):
    pass


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _bool_token(value: Any) -> str:
    return "true" if bool(value) else "false"


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SoloYieldRepairedTop8ObjectiveFailureReviewError(f"json not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _require_no_quality_claim(report: dict[str, Any]) -> None:
    readiness = _dict(report.get("readiness"))
    claimed = [
        key
        for key in ("musical_quality_claimed", "artist_style_claimed", "production_ready_claimed")
        if bool(readiness.get(key, False))
    ]
    if claimed:
        raise SoloYieldRepairedTop8ObjectiveFailureReviewError(
            f"unexpected quality claim: {claimed}"
        )


def _load_notes(midi_path: Path) -> list[pretty_midi.Note]:
    if not midi_path.exists():
        raise SoloYieldRepairedTop8ObjectiveFailureReviewError(f"MIDI file missing: {midi_path}")
    midi = pretty_midi.PrettyMIDI(str(midi_path))
    notes: list[pretty_midi.Note] = []
    for instrument in midi.instruments:
        if not instrument.is_drum:
            notes.extend(instrument.notes)
    return sorted(notes, key=lambda note: (float(note.start), int(note.pitch)))


def _chord_tone_pcs(chord: str) -> set[int]:
    root_pc, intervals = parse_chord(chord)
    return {(root_pc + interval) % 12 for interval in intervals}


def _chord_index_for_note(note: pretty_midi.Note, *, start: float, duration: float, chord_count: int) -> int:
    if chord_count <= 1 or duration <= 0:
        return 0
    ratio = max(0.0, min(0.999999, (float(note.start) - start) / duration))
    return min(chord_count - 1, int(ratio * chord_count))


def _direction_change_ratio(pitches: list[int]) -> float:
    if len(pitches) < 3:
        return 0.0
    signs: list[int] = []
    for left, right in zip(pitches, pitches[1:]):
        delta = int(right) - int(left)
        if delta > 0:
            signs.append(1)
        elif delta < 0:
            signs.append(-1)
    if len(signs) < 2:
        return 0.0
    changes = sum(1 for left, right in zip(signs, signs[1:]) if left != right)
    return changes / max(1, len(signs) - 1)


def midi_profile(candidate: dict[str, Any]) -> dict[str, Any]:
    midi_path = Path(str(candidate.get("review_midi_path") or ""))
    chords = [item.strip() for item in str(candidate.get("chords") or "").split(",") if item.strip()]
    if not chords:
        raise SoloYieldRepairedTop8ObjectiveFailureReviewError("candidate chords required")
    notes = _load_notes(midi_path)
    if not notes:
        return {
            "midi_note_count": 0,
            "midi_unique_pitch_count": 0,
            "midi_pitch_span": 0,
            "midi_max_abs_interval": 0,
            "midi_direction_change_ratio": 0.0,
            "midi_max_gap_seconds": 0.0,
            "midi_avg_gap_seconds": 0.0,
            "midi_chord_tone_ratio": 0.0,
            "midi_tension_ratio": 0.0,
            "final_landing_chord": chords[-1],
            "final_landing_pitch": None,
            "final_landing_is_chord_tone": False,
        }

    starts = [float(note.start) for note in notes]
    pitches = [int(note.pitch) for note in notes]
    phrase_start = min(starts)
    phrase_end = max(float(note.end) for note in notes)
    phrase_duration = max(1e-6, phrase_end - phrase_start)
    gaps = [max(0.0, starts[index] - starts[index - 1]) for index in range(1, len(starts))]
    intervals = [abs(pitches[index] - pitches[index - 1]) for index in range(1, len(pitches))]

    chord_tone_count = 0
    for note in notes:
        chord_index = _chord_index_for_note(
            note,
            start=phrase_start,
            duration=phrase_duration,
            chord_count=len(chords),
        )
        if int(note.pitch) % 12 in _chord_tone_pcs(chords[chord_index]):
            chord_tone_count += 1

    final_note = notes[-1]
    final_chord_index = _chord_index_for_note(
        final_note,
        start=phrase_start,
        duration=phrase_duration,
        chord_count=len(chords),
    )
    final_chord = chords[final_chord_index]
    final_is_chord_tone = int(final_note.pitch) % 12 in _chord_tone_pcs(final_chord)
    chord_tone_ratio = chord_tone_count / max(1, len(notes))

    return {
        "midi_note_count": len(notes),
        "midi_unique_pitch_count": len(set(pitches)),
        "midi_pitch_span": max(pitches) - min(pitches),
        "midi_max_abs_interval": max(intervals or [0]),
        "midi_direction_change_ratio": _direction_change_ratio(pitches),
        "midi_max_gap_seconds": max(gaps or [0.0]),
        "midi_avg_gap_seconds": mean(gaps) if gaps else 0.0,
        "midi_chord_tone_ratio": chord_tone_ratio,
        "midi_tension_ratio": 1.0 - chord_tone_ratio,
        "final_landing_chord": final_chord,
        "final_landing_pitch": int(final_note.pitch),
        "final_landing_is_chord_tone": final_is_chord_tone,
    }


def failure_labels(candidate: dict[str, Any], profile: dict[str, Any]) -> list[str]:
    labels: list[str] = []
    if not bool(profile.get("final_landing_is_chord_tone", False)):
        labels.append("final_landing_not_chord_tone")
    if _float(candidate.get("chord_tone_ratio")) < 0.50:
        labels.append("package_low_chord_tone_ratio")
    if _float(profile.get("midi_chord_tone_ratio")) < 0.50:
        labels.append("midi_low_chord_tone_ratio")
    if _float(candidate.get("dead_air_ratio")) >= 0.65:
        labels.append("dead_air_still_high")
    if _float(candidate.get("direction_change_ratio")) < 0.50:
        labels.append("weak_direction_change")
    if _int(candidate.get("note_count")) < 30:
        labels.append("low_note_count_for_4bar")
    if _int(profile.get("midi_max_abs_interval")) >= 8:
        labels.append("wide_interval_review")
    return labels


def _select_next_target(label_counts: Counter[str], candidate_count: int) -> tuple[str, str]:
    if label_counts.get("final_landing_not_chord_tone", 0) >= max(1, candidate_count // 2):
        return SELECTED_TARGET, NEXT_BOUNDARY
    if label_counts.get("package_low_chord_tone_ratio", 0) >= max(1, candidate_count // 2):
        return "chord_role_balance_repair", "music_transformer_solo_yield_chord_role_balance_repair_sweep"
    if label_counts.get("dead_air_still_high", 0) > 0:
        return "dead_air_aftercare", "music_transformer_solo_yield_dead_air_aftercare_sweep"
    return "listening_review_required", "music_transformer_solo_yield_listening_review"


def build_objective_failure_review(
    package_report: dict[str, Any],
    *,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if str(package_report.get("schema_version") or "") != "music_transformer_solo_yield_listening_package_v1":
        raise SoloYieldRepairedTop8ObjectiveFailureReviewError("listening package schema required")
    _require_no_quality_claim(package_report)
    candidates = [_dict(item) for item in _list(package_report.get("candidates"))]
    rows: list[dict[str, Any]] = []
    label_counts: Counter[str] = Counter()
    for candidate in candidates:
        profile = midi_profile(candidate)
        labels = failure_labels(candidate, profile)
        label_counts.update(labels)
        rows.append(
            {
                "review_index": _int(candidate.get("review_index")),
                "case_label": str(candidate.get("case_label") or ""),
                "chords": str(candidate.get("chords") or ""),
                "review_midi_path": str(candidate.get("review_midi_path") or ""),
                "review_wav_path": str(candidate.get("review_wav_path") or ""),
                "package_metrics": {
                    "note_count": _int(candidate.get("note_count")),
                    "unique_pitch_count": _int(candidate.get("unique_pitch_count")),
                    "dead_air_ratio": _float(candidate.get("dead_air_ratio")),
                    "direction_change_ratio": _float(candidate.get("direction_change_ratio")),
                    "syncopated_onset_ratio": _float(candidate.get("syncopated_onset_ratio")),
                    "chord_tone_ratio": _float(candidate.get("chord_tone_ratio")),
                    "tension_ratio": _float(candidate.get("tension_ratio")),
                    "score": _float(candidate.get("score")),
                },
                "midi_profile": profile,
                "failure_labels": labels,
            }
        )

    selected_target, next_boundary = _select_next_target(label_counts, len(rows))
    report = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "boundary": BOUNDARY,
        "source_package": {
            "schema_version": package_report.get("schema_version"),
            "output_dir": package_report.get("output_dir"),
            "candidate_count": _int(package_report.get("candidate_count")),
        },
        "candidate_count": len(rows),
        "candidates": rows,
        "aggregate": {
            "failure_label_counts": dict(sorted(label_counts.items())),
            "failed_candidate_count": sum(1 for row in rows if row["failure_labels"]),
            "candidate_without_failure_label_count": sum(1 for row in rows if not row["failure_labels"]),
            "final_landing_not_chord_tone_count": label_counts.get("final_landing_not_chord_tone", 0),
            "package_low_chord_tone_ratio_count": label_counts.get("package_low_chord_tone_ratio", 0),
            "midi_low_chord_tone_ratio_count": label_counts.get("midi_low_chord_tone_ratio", 0),
            "dead_air_still_high_count": label_counts.get("dead_air_still_high", 0),
        },
        "readiness": {
            "objective_failure_review_completed": True,
            "validated_listening_input_present": False,
            "musical_quality_claimed": False,
            "artist_style_claimed": False,
            "production_ready_claimed": False,
        },
        "decision": {
            "current_boundary": BOUNDARY,
            "selected_next_target": selected_target,
            "next_boundary": next_boundary,
            "critical_user_input_required": False,
            "reason": (
                "objective review found repeated final landing and chord-role failures; "
                "route to pitch-role repair without quality claim"
            ),
        },
        "not_proven": [
            "human_audio_preference",
            "stable_jazz_solo_quality",
            "artist_level_long_solo_generation",
            "production_ready_improviser",
        ],
    }
    write_json(output_dir / "objective_failure_review.json", report)
    write_json(output_dir / "objective_failure_review_summary.json", validate_report(report))
    write_text(output_dir / "objective_failure_review.md", markdown_report(report))
    return report


def validate_report(report: dict[str, Any], *, min_candidates: int = 1) -> dict[str, Any]:
    if str(report.get("schema_version") or "") != SCHEMA_VERSION:
        raise SoloYieldRepairedTop8ObjectiveFailureReviewError("schema version mismatch")
    readiness = _dict(report.get("readiness"))
    aggregate = _dict(report.get("aggregate"))
    decision = _dict(report.get("decision"))
    candidate_count = _int(report.get("candidate_count"))
    if candidate_count < int(min_candidates):
        raise SoloYieldRepairedTop8ObjectiveFailureReviewError("candidate count below requirement")
    if not bool(readiness.get("objective_failure_review_completed", False)):
        raise SoloYieldRepairedTop8ObjectiveFailureReviewError("objective failure review completion required")
    claimed = [
        key
        for key in ("musical_quality_claimed", "artist_style_claimed", "production_ready_claimed")
        if bool(readiness.get(key, True))
    ]
    if claimed:
        raise SoloYieldRepairedTop8ObjectiveFailureReviewError(f"unexpected quality claim: {claimed}")
    label_counts = _dict(aggregate.get("failure_label_counts"))
    return {
        "schema_version": str(report.get("schema_version") or ""),
        "candidate_count": candidate_count,
        "failed_candidate_count": _int(aggregate.get("failed_candidate_count")),
        "final_landing_not_chord_tone_count": _int(
            aggregate.get("final_landing_not_chord_tone_count")
        ),
        "package_low_chord_tone_ratio_count": _int(
            aggregate.get("package_low_chord_tone_ratio_count")
        ),
        "midi_low_chord_tone_ratio_count": _int(aggregate.get("midi_low_chord_tone_ratio_count")),
        "dead_air_still_high_count": _int(aggregate.get("dead_air_still_high_count")),
        "failure_label_type_count": len(label_counts),
        "selected_next_target": str(decision.get("selected_next_target") or ""),
        "next_boundary": str(decision.get("next_boundary") or ""),
        "musical_quality_claimed": bool(readiness.get("musical_quality_claimed", True)),
    }


def markdown_report(report: dict[str, Any]) -> str:
    aggregate = report["aggregate"]
    decision = report["decision"]
    readiness = report["readiness"]
    lines = [
        "# Music Transformer Solo Yield Repaired Top8 Objective Failure Review",
        "",
        "## Summary",
        "",
        f"- candidate count: `{report['candidate_count']}`",
        f"- failed candidate count: `{aggregate['failed_candidate_count']}`",
        f"- final landing not chord tone: `{aggregate['final_landing_not_chord_tone_count']}`",
        f"- package low chord-tone ratio: `{aggregate['package_low_chord_tone_ratio_count']}`",
        f"- MIDI low chord-tone ratio: `{aggregate['midi_low_chord_tone_ratio_count']}`",
        f"- dead-air still high: `{aggregate['dead_air_still_high_count']}`",
        f"- selected next target: `{decision['selected_next_target']}`",
        f"- next boundary: `{decision['next_boundary']}`",
        f"- musical quality claimed: `{_bool_token(readiness['musical_quality_claimed'])}`",
        "",
        "## Failure Label Counts",
        "",
    ]
    for label, count in sorted(aggregate["failure_label_counts"].items()):
        lines.append(f"- `{label}`: `{count}`")
    lines.extend(["", "## Candidates", ""])
    for row in report.get("candidates", []):
        metrics = row["package_metrics"]
        profile = row["midi_profile"]
        labels = ", ".join(f"`{label}`" for label in row["failure_labels"]) or "`none`"
        lines.extend(
            [
                f"- candidate `{row['review_index']}` / `{row['case_label']}`",
                f"  - labels: {labels}",
                f"  - package dead-air: `{float(metrics['dead_air_ratio']):.4f}`",
                f"  - package chord-tone ratio: `{float(metrics['chord_tone_ratio']):.4f}`",
                f"  - MIDI chord-tone ratio: `{float(profile['midi_chord_tone_ratio']):.4f}`",
                f"  - final landing: `{profile['final_landing_pitch']}` over `{profile['final_landing_chord']}`, chord-tone `{_bool_token(profile['final_landing_is_chord_tone'])}`",
            ]
        )
    lines.extend(["", "## Not Proven", ""])
    for item in report.get("not_proven", []):
        lines.append(f"- `{item}`")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Review repaired top8 objective failure labels")
    parser.add_argument(
        "--package_report",
        type=str,
        default=(
            "outputs/music_transformer_finetune_mvp/solo_yield_listening_review/"
            "issue_1250_4bar_repaired_top8_listening_package/listening_review_package.json"
        ),
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/music_transformer_finetune_mvp/solo_yield_objective_failure_review",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--min_candidates", type=int, default=8)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    report = build_objective_failure_review(
        read_json(Path(args.package_report)),
        output_dir=output_dir,
    )
    summary = validate_report(report, min_candidates=int(args.min_candidates))
    write_json(output_dir / "objective_failure_review_summary.json", summary)
    if args.doc_path:
        write_text(Path(args.doc_path), markdown_report(report))
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
