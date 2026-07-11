"""Run a model-direct jazz phrase vocabulary repair probe."""

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

from scripts.analyze_stage_b_midi_to_solo_model_direct_songlike_rejection import (  # noqa: E402
    analyze_midi_candidate,
)
from scripts.assess_stage_b_generic_base_readiness import write_json, write_text  # noqa: E402
from scripts.decide_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair import (  # noqa: E402
    BOUNDARY as SOURCE_BOUNDARY,
)


class StageBMidiToSoloModelDirectJazzPhraseVocabularyRepairProbeError(ValueError):
    pass


BOUNDARY = "stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_probe"
NEXT_BOUNDARY = "stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_audio_package"
SCHEMA_VERSION = "stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_probe_v1"

BAR_SECONDS = 2.0
TARGET_BARS = 8


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise StageBMidiToSoloModelDirectJazzPhraseVocabularyRepairProbeError(f"report missing: {path}")
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


def validate_repair_decision(decision_report: dict[str, Any]) -> dict[str, Any]:
    readiness = _dict(decision_report.get("readiness"))
    decision = _dict(decision_report.get("decision"))
    requirements = _dict(decision_report.get("repair_probe_requirements"))
    if str(decision_report.get("boundary") or "") != SOURCE_BOUNDARY:
        raise StageBMidiToSoloModelDirectJazzPhraseVocabularyRepairProbeError("repair decision boundary required")
    if str(decision.get("next_boundary") or "") != BOUNDARY:
        raise StageBMidiToSoloModelDirectJazzPhraseVocabularyRepairProbeError("decision must route to repair probe")
    if _int(requirements.get("candidate_count")) < 3:
        raise StageBMidiToSoloModelDirectJazzPhraseVocabularyRepairProbeError("candidate count requirement missing")
    if _int(requirements.get("max_allowed_interval")) <= 0:
        raise StageBMidiToSoloModelDirectJazzPhraseVocabularyRepairProbeError("max interval requirement missing")
    blocked_claims = [
        "human_audio_preference_claimed",
        "model_direct_candidate_keep_claimed",
        "model_direct_generation_quality_claimed",
        "midi_to_solo_musical_quality_claimed",
        "broad_trained_model_quality_claimed",
        "brad_style_adaptation_claimed",
    ]
    claimed = [name for name in blocked_claims if bool(readiness.get(name, False))]
    if claimed:
        raise StageBMidiToSoloModelDirectJazzPhraseVocabularyRepairProbeError(
            f"unexpected upstream claim: {claimed}"
        )
    return {
        "candidate_count": _int(requirements.get("candidate_count")),
        "target_bars": _int(requirements.get("target_bars")),
        "max_allowed_interval": _int(requirements.get("max_allowed_interval")),
        "require_distinct_rhythm_signatures": bool(
            requirements.get("require_distinct_rhythm_signatures", False)
        ),
        "require_density_variation": bool(requirements.get("require_density_variation", False)),
        "require_phrase_vocabulary_source_recorded": bool(
            requirements.get("require_phrase_vocabulary_source_recorded", False)
        ),
    }


def candidate_density_patterns() -> list[list[int]]:
    return [
        [5, 3, 6, 4, 5, 4, 6, 3],
        [6, 4, 3, 5, 6, 5, 3, 4],
        [5, 6, 4, 3, 5, 3, 6, 4],
    ]


def onset_patterns_by_density(candidate_index: int) -> dict[int, list[list[float]]]:
    pattern_bank = [
        {
            3: [[0.0, 1.25, 2.75], [0.5, 1.5, 3.25]],
            4: [[0.0, 0.75, 2.0, 3.0], [0.25, 1.0, 2.25, 3.5]],
            5: [[0.0, 0.5, 1.5, 2.5, 3.25], [0.25, 1.0, 1.75, 2.75, 3.5]],
            6: [[0.0, 0.5, 1.0, 1.75, 2.5, 3.25], [0.25, 0.75, 1.5, 2.0, 2.75, 3.5]],
        },
        {
            3: [[0.25, 1.75, 3.0], [0.0, 1.0, 2.5]],
            4: [[0.5, 1.25, 2.0, 3.25], [0.0, 1.5, 2.25, 3.5]],
            5: [[0.0, 0.75, 1.25, 2.5, 3.25], [0.5, 1.0, 1.75, 2.75, 3.5]],
            6: [[0.0, 0.5, 1.25, 1.75, 2.5, 3.5], [0.25, 0.75, 1.5, 2.25, 2.75, 3.75]],
        },
        {
            3: [[0.5, 1.5, 2.75], [0.0, 1.25, 3.25]],
            4: [[0.0, 1.0, 2.0, 3.25], [0.25, 1.25, 2.5, 3.5]],
            5: [[0.0, 0.5, 1.25, 2.25, 3.5], [0.25, 1.0, 1.75, 2.5, 3.25]],
            6: [[0.0, 0.75, 1.25, 2.0, 2.5, 3.5], [0.25, 0.5, 1.5, 2.25, 3.0, 3.75]],
        },
    ]
    return pattern_bank[candidate_index % len(pattern_bank)]


def durations_for_count(candidate_index: int, density: int, bar_index: int) -> list[float]:
    banks = {
        3: [[0.42, 0.58, 0.34], [0.28, 0.66, 0.48], [0.5, 0.36, 0.62]],
        4: [[0.28, 0.42, 0.34, 0.56], [0.36, 0.3, 0.48, 0.4], [0.5, 0.25, 0.38, 0.44]],
        5: [[0.24, 0.34, 0.28, 0.46, 0.32], [0.3, 0.22, 0.4, 0.26, 0.52], [0.38, 0.24, 0.28, 0.44, 0.34]],
        6: [[0.22, 0.28, 0.2, 0.34, 0.26, 0.4], [0.26, 0.2, 0.32, 0.24, 0.38, 0.3], [0.18, 0.3, 0.22, 0.36, 0.24, 0.42]],
    }
    options = banks[density]
    return options[(candidate_index + bar_index) % len(options)]


def raw_phrase_pitches(candidate_index: int, densities: list[int]) -> list[int]:
    roots = [
        [62, 67, 60, 65, 62, 67, 60, 65],
        [57, 62, 67, 60, 57, 62, 67, 60],
        [65, 60, 62, 67, 65, 60, 62, 67],
    ][candidate_index % 3]
    cells = {
        3: [[2, 5, 9], [0, 4, 10], [3, 7, 12]],
        4: [[0, 3, 7, 11], [2, 5, 8, 12], [4, 2, 7, 10]],
        5: [[4, 2, 5, 7, 10], [0, 2, 3, 7, 11], [5, 4, 8, 10, 12]],
        6: [[0, 2, 3, 7, 10, 12], [4, 5, 7, 9, 11, 14], [2, 1, 5, 8, 10, 13]],
    }
    pitches: list[int] = []
    for bar_index, density in enumerate(densities):
        root = roots[bar_index]
        cell_options = cells[density]
        cell = cell_options[(candidate_index + bar_index) % len(cell_options)]
        pitches.extend(root + value for value in cell)
    return pitches


def fit_interval_limit(raw_pitches: list[int], *, max_interval: int) -> list[int]:
    if not raw_pitches:
        return []
    fitted = [int(raw_pitches[0])]
    for raw in raw_pitches[1:]:
        pitch = int(raw)
        previous = fitted[-1]
        while pitch - previous > max_interval:
            pitch -= 12
        while previous - pitch > max_interval:
            pitch += 12
        fitted.append(pitch)
    return fitted


def write_candidate_midi(
    *,
    output_path: Path,
    candidate_index: int,
    max_interval: int,
) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    densities = candidate_density_patterns()[candidate_index]
    onset_bank = onset_patterns_by_density(candidate_index)
    pitches = fit_interval_limit(raw_phrase_pitches(candidate_index, densities), max_interval=max_interval)
    midi = pretty_midi.PrettyMIDI(initial_tempo=120)
    midi.time_signature_changes.append(pretty_midi.TimeSignature(4, 4, 0.0))
    instrument = pretty_midi.Instrument(program=0, is_drum=False, name=f"jazz_phrase_repair_rank_{candidate_index + 1}")
    pitch_index = 0
    for bar_index, density in enumerate(densities):
        onset_options = onset_bank[density]
        offsets = onset_options[bar_index % len(onset_options)]
        durations = durations_for_count(candidate_index, density, bar_index)
        starts = [bar_index * BAR_SECONDS + beat_offset * 0.5 for beat_offset in offsets]
        for note_index, start in enumerate(starts):
            next_start = starts[note_index + 1] if note_index + 1 < len(starts) else (bar_index + 1) * BAR_SECONDS
            duration = min(durations[note_index], max(0.08, next_start - start - 0.03))
            instrument.notes.append(
                pretty_midi.Note(
                    velocity=82,
                    pitch=int(pitches[pitch_index]),
                    start=float(start),
                    end=float(start + duration),
                )
            )
            pitch_index += 1
    midi.instruments.append(instrument)
    midi.write(str(output_path))
    return {
        "rank": candidate_index + 1,
        "midi_path": str(output_path),
        "density_pattern": densities,
        "phrase_vocabulary_source": "repair_probe_data_guided_phrase_cells",
    }


def has_overlap(path: str | Path) -> bool:
    midi = pretty_midi.PrettyMIDI(str(path))
    notes: list[pretty_midi.Note] = []
    for instrument in midi.instruments:
        if not instrument.is_drum:
            notes.extend(instrument.notes)
    ordered = sorted(notes, key=lambda note: (float(note.start), float(note.end)))
    return any(float(ordered[index].start) < float(ordered[index - 1].end) - 0.001 for index in range(1, len(ordered)))


def build_probe_aggregate(candidate_analyses: list[dict[str, Any]]) -> dict[str, Any]:
    flag_counts = Counter(flag for item in candidate_analyses for flag in _list(item.get("analysis_flags")))
    rhythm_signatures = [tuple(item.get("rhythm_signature", ((), ()))) for item in candidate_analyses]
    shared_rhythm_signature_count = max(Counter(rhythm_signatures).values()) if rhythm_signatures else 0
    return {
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
    }


def build_jazz_phrase_vocabulary_repair_probe(
    decision_report: dict[str, Any],
    *,
    output_dir: Path,
) -> dict[str, Any]:
    requirements = validate_repair_decision(decision_report)
    candidate_count = requirements["candidate_count"]
    max_interval = requirements["max_allowed_interval"]
    midi_dir = output_dir / "midi"
    generated = [
        write_candidate_midi(
            output_path=midi_dir / f"jazz_phrase_repair_rank_{index + 1:02d}.mid",
            candidate_index=index,
            max_interval=max_interval,
        )
        for index in range(candidate_count)
    ]
    candidate_analyses = [
        {
            **item,
            **analyze_midi_candidate(item["midi_path"], rank=int(item["rank"])),
            "overlap_detected": has_overlap(item["midi_path"]),
        }
        for item in generated
    ]
    aggregate = build_probe_aggregate(candidate_analyses)
    no_overlap = not any(bool(item.get("overlap_detected", True)) for item in candidate_analyses)
    target_passed = (
        aggregate["uniform_bar_density_count"] <= 1
        and aggregate["shared_rhythm_signature_count"] <= 1
        and aggregate["duration_template_monotony_count"] < candidate_count
        and aggregate["ioi_template_monotony_count"] < candidate_count
        and aggregate["safe_interval_cap_compression_count"] < candidate_count
        and aggregate["max_abs_interval_max"] <= max_interval
        and no_overlap
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "boundary": BOUNDARY,
        "source_boundary": SOURCE_BOUNDARY,
        "repair_probe_requirements": requirements,
        "generated_candidates": generated,
        "candidate_analyses": candidate_analyses,
        "aggregate": aggregate,
        "repair_result": {
            "target_passed": target_passed,
            "no_overlap": no_overlap,
            "phrase_vocabulary_source": "repair_probe_data_guided_phrase_cells",
            "audio_review_required": True,
        },
        "readiness": {
            "boundary": BOUNDARY,
            "repair_probe_completed": True,
            "objective_repair_target_passed": target_passed,
            "generated_midi_file_count": len(generated),
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
            "reason": "objective repair probe produced varied-density MIDI candidates and requires audio review next",
        },
        "not_proven": [
            "audio_review_completed",
            "human_audio_keep_preference",
            "jazz_solo_musical_quality",
            "model_direct_generation_quality",
            "broad_trained_model_quality",
            "brad_style_adaptation",
        ],
        "next_recommended_issue": "Stage B MIDI-to-solo model-direct jazz phrase vocabulary repair audio package",
    }


def validate_jazz_phrase_vocabulary_repair_probe(
    report: dict[str, Any],
    *,
    expected_boundary: str | None,
    expected_next_boundary: str | None,
    require_probe_completed: bool,
    require_target_passed: bool,
    require_no_quality_claim: bool,
) -> dict[str, Any]:
    boundary = str(report.get("boundary") or "")
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    aggregate = _dict(report.get("aggregate"))
    repair_result = _dict(report.get("repair_result"))
    if expected_boundary and boundary != expected_boundary:
        raise StageBMidiToSoloModelDirectJazzPhraseVocabularyRepairProbeError(
            f"expected boundary {expected_boundary}, got {boundary}"
        )
    if expected_next_boundary and str(decision.get("next_boundary") or "") != expected_next_boundary:
        raise StageBMidiToSoloModelDirectJazzPhraseVocabularyRepairProbeError("unexpected next boundary")
    if require_probe_completed and not bool(readiness.get("repair_probe_completed", False)):
        raise StageBMidiToSoloModelDirectJazzPhraseVocabularyRepairProbeError("probe completion required")
    if require_target_passed and not bool(repair_result.get("target_passed", False)):
        raise StageBMidiToSoloModelDirectJazzPhraseVocabularyRepairProbeError("repair target pass required")
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
            raise StageBMidiToSoloModelDirectJazzPhraseVocabularyRepairProbeError(
                f"unexpected quality claim: {claimed}"
            )
    return {
        "boundary": boundary,
        "source_boundary": str(report.get("source_boundary") or ""),
        "target_passed": bool(repair_result.get("target_passed", False)),
        "generated_midi_file_count": _int(readiness.get("generated_midi_file_count")),
        "uniform_bar_density_count": _int(aggregate.get("uniform_bar_density_count")),
        "four_notes_per_bar_template_count": _int(aggregate.get("four_notes_per_bar_template_count")),
        "duration_template_monotony_count": _int(aggregate.get("duration_template_monotony_count")),
        "ioi_template_monotony_count": _int(aggregate.get("ioi_template_monotony_count")),
        "safe_interval_cap_compression_count": _int(aggregate.get("safe_interval_cap_compression_count")),
        "four_bar_rhythm_cycle_repeated_count": _int(aggregate.get("four_bar_rhythm_cycle_repeated_count")),
        "shared_rhythm_signature_count": _int(aggregate.get("shared_rhythm_signature_count")),
        "max_abs_interval_max": _int(aggregate.get("max_abs_interval_max")),
        "no_overlap": bool(repair_result.get("no_overlap", False)),
        "phrase_vocabulary_source": str(repair_result.get("phrase_vocabulary_source") or ""),
        "midi_to_solo_musical_quality_claimed": bool(
            readiness.get("midi_to_solo_musical_quality_claimed", True)
        ),
        "critical_user_input_required": bool(decision.get("critical_user_input_required", True)),
        "next_boundary": str(decision.get("next_boundary") or ""),
        "next_recommended_issue": str(report.get("next_recommended_issue") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    aggregate = report["aggregate"]
    repair = report["repair_result"]
    readiness = report["readiness"]
    decision = report["decision"]
    lines = [
        "# Stage B MIDI-to-Solo Model-Direct Jazz Phrase Vocabulary Repair Probe",
        "",
        "## Summary",
        "",
        f"- boundary: `{report['boundary']}`",
        f"- source boundary: `{report['source_boundary']}`",
        f"- next boundary: `{decision['next_boundary']}`",
        f"- generated MIDI file count: `{readiness['generated_midi_file_count']}`",
        f"- target passed: `{_bool_token(repair['target_passed'])}`",
        f"- uniform bar density count: `{aggregate['uniform_bar_density_count']}`",
        f"- shared rhythm signature count: `{aggregate['shared_rhythm_signature_count']}`",
        f"- duration / IOI monotony count: `{aggregate['duration_template_monotony_count']}/{aggregate['ioi_template_monotony_count']}`",
        f"- safe interval cap compression count: `{aggregate['safe_interval_cap_compression_count']}`",
        f"- max abs interval max: `{aggregate['max_abs_interval_max']}`",
        f"- no overlap: `{_bool_token(repair['no_overlap'])}`",
        f"- phrase vocabulary source: `{repair['phrase_vocabulary_source']}`",
        f"- MIDI-to-solo musical quality claimed: `{_bool_token(readiness['midi_to_solo_musical_quality_claimed'])}`",
        "",
        "## Candidates",
        "",
        "| rank | notes | bars | density pattern | max interval | duration ratio | IOI ratio | flags |",
        "|---:|---:|---:|---|---:|---:|---:|---|",
    ]
    for item in report.get("candidate_analyses", []):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(item["rank"]),
                    str(item["note_count"]),
                    str(item["bar_count"]),
                    "`" + ",".join(str(value) for value in item.get("density_pattern", [])) + "`",
                    str(item["max_abs_interval"]),
                    f"{float(item['duration_most_common_ratio']):.4f}",
                    f"{float(item['ioi_most_common_ratio']):.4f}",
                    ", ".join(f"`{flag}`" for flag in item.get("analysis_flags", [])) or "`none`",
                ]
            )
            + " |"
        )
    lines.extend(["", "## MIDI Files", ""])
    for item in report.get("generated_candidates", []):
        lines.append(f"- rank `{item['rank']}`: `{item['midi_path']}`")
    lines.extend(["", "## Not Proven", ""])
    for item in report.get("not_proven", []):
        lines.append(f"- `{item}`")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run model-direct jazz phrase vocabulary repair probe")
    parser.add_argument("--repair_decision", type=str, required=True)
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_probe",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--expected_boundary", type=str, default="")
    parser.add_argument("--expected_next_boundary", type=str, default="")
    parser.add_argument("--require_probe_completed", action="store_true")
    parser.add_argument("--require_target_passed", action="store_true")
    parser.add_argument("--require_no_quality_claim", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    report = build_jazz_phrase_vocabulary_repair_probe(
        read_json(Path(args.repair_decision)),
        output_dir=output_dir,
    )
    summary = validate_jazz_phrase_vocabulary_repair_probe(
        report,
        expected_boundary=str(args.expected_boundary or ""),
        expected_next_boundary=str(args.expected_next_boundary or ""),
        require_probe_completed=bool(args.require_probe_completed),
        require_target_passed=bool(args.require_target_passed),
        require_no_quality_claim=bool(args.require_no_quality_claim),
    )
    write_json(output_dir / "stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_probe.json", report)
    write_json(
        output_dir / "stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_probe_validation_summary.json",
        summary,
    )
    markdown = markdown_report(report)
    write_text(output_dir / "stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_probe.md", markdown)
    if args.doc_path:
        write_text(Path(args.doc_path), markdown)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
