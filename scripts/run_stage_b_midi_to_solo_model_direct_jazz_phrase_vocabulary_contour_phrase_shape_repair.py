"""Run contour and phrase-shape repair for jazz phrase vocabulary MIDI candidates."""

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
from scripts.decide_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_objective_next import (  # noqa: E402
    BOUNDARY as SOURCE_BOUNDARY,
)
from scripts.run_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_probe import (  # noqa: E402
    BAR_SECONDS,
    candidate_density_patterns,
    durations_for_count,
    fit_interval_limit,
    has_overlap,
    onset_patterns_by_density,
)


class StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeRepairError(ValueError):
    pass


BOUNDARY = "stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_repair"
NEXT_BOUNDARY = "stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_audio_package"
SCHEMA_VERSION = "stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_repair_v1"


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeRepairError(
            f"report missing: {path}"
        )
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


def validate_objective_next_decision(report: dict[str, Any]) -> dict[str, Any]:
    boundary = _dict(report.get("objective_next_decision_boundary"))
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    targets = [str(item) for item in _list(report.get("selected_repair_targets"))]
    if str(boundary.get("boundary") or report.get("boundary") or "") != SOURCE_BOUNDARY:
        raise StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeRepairError(
            "objective-next decision boundary required"
        )
    if str(decision.get("next_boundary") or "") != BOUNDARY:
        raise StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeRepairError(
            "objective-next decision must route to contour phrase-shape repair"
        )
    if "reduce_stepwise_contour_bias" not in targets:
        raise StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeRepairError(
            "stepwise contour repair target required"
        )
    blocked_claims = [
        "human_audio_preference_claimed",
        "model_direct_generation_quality_claimed",
        "midi_to_solo_musical_quality_claimed",
        "broad_trained_model_quality_claimed",
        "brad_style_adaptation_claimed",
    ]
    claimed = [name for name in blocked_claims if bool(readiness.get(name, False))]
    if claimed:
        raise StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeRepairError(
            f"unexpected upstream claim: {claimed}"
        )
    objective = _dict(report.get("objective_summary"))
    return {
        "candidate_count": _int(objective.get("candidate_count")) or 3,
        "source_stepwise_contour_bias_count": _int(objective.get("stepwise_contour_bias_count")),
        "source_distinct_density_pattern_count": _int(objective.get("distinct_density_pattern_count")),
        "source_max_abs_interval_max": _int(objective.get("max_abs_interval_max")),
        "selected_repair_targets": targets,
    }


def contour_phrase_cells(candidate_index: int, density: int, bar_index: int) -> list[int]:
    banks = {
        3: [
            [0, 7, 2],
            [4, -3, 8],
            [7, 0, 11],
        ],
        4: [
            [0, 7, 2, 10],
            [5, -2, 9, 3],
            [7, 12, 5, 10],
        ],
        5: [
            [0, 7, 2, 10, 5],
            [4, -3, 8, 1, 11],
            [7, 12, 5, 9, 2],
        ],
        6: [
            [0, 7, 2, 10, 5, 12],
            [4, -3, 8, 1, 11, 6],
            [7, 12, 5, 9, 2, 10],
        ],
    }
    return banks[density][(candidate_index + bar_index) % len(banks[density])]


def contour_phrase_pitches(candidate_index: int, densities: list[int]) -> list[int]:
    roots = [
        [62, 65, 60, 67, 62, 65, 60, 67],
        [57, 64, 60, 67, 57, 64, 60, 67],
        [65, 60, 67, 62, 65, 60, 67, 62],
    ][candidate_index % 3]
    pitches: list[int] = []
    for bar_index, density in enumerate(densities):
        root = roots[bar_index]
        pitches.extend(root + value for value in contour_phrase_cells(candidate_index, density, bar_index))
    return pitches


def write_candidate_midi(
    *,
    output_path: Path,
    candidate_index: int,
    max_interval: int,
) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    densities = candidate_density_patterns()[candidate_index % len(candidate_density_patterns())]
    onset_bank = onset_patterns_by_density(candidate_index)
    pitches = fit_interval_limit(contour_phrase_pitches(candidate_index, densities), max_interval=max_interval)
    midi = pretty_midi.PrettyMIDI(initial_tempo=120)
    midi.time_signature_changes.append(pretty_midi.TimeSignature(4, 4, 0.0))
    instrument = pretty_midi.Instrument(
        program=0,
        is_drum=False,
        name=f"contour_phrase_shape_rank_{candidate_index + 1}",
    )
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
                    velocity=84,
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
        "phrase_vocabulary_source": "contour_phrase_shape_cells",
    }


def aggregate_candidate_analyses(candidate_analyses: list[dict[str, Any]]) -> dict[str, Any]:
    flag_counts = Counter(flag for item in candidate_analyses for flag in _list(item.get("analysis_flags")))
    rhythm_signatures = [tuple(item.get("rhythm_signature", ((), ()))) for item in candidate_analyses]
    shared_rhythm_signature_count = max(Counter(rhythm_signatures).values()) if rhythm_signatures else 0
    return {
        "candidate_count": len(candidate_analyses),
        "flag_counts": dict(sorted(flag_counts.items())),
        "stepwise_contour_bias_count": flag_counts.get("stepwise_contour_bias", 0),
        "uniform_bar_density_count": flag_counts.get("uniform_bar_density", 0),
        "four_notes_per_bar_template_count": flag_counts.get("four_notes_per_bar_template", 0),
        "duration_template_monotony_count": flag_counts.get("duration_template_monotony", 0),
        "ioi_template_monotony_count": flag_counts.get("ioi_template_monotony", 0),
        "safe_interval_cap_compression_count": flag_counts.get("safe_interval_cap_compression", 0),
        "four_bar_rhythm_cycle_repeated_count": flag_counts.get("four_bar_rhythm_cycle_repeated", 0),
        "shared_rhythm_signature_count": shared_rhythm_signature_count,
        "max_abs_interval_max": max((_int(item.get("max_abs_interval")) for item in candidate_analyses), default=0),
        "max_small_interval_ratio_le4": max(
            (_float(item.get("small_interval_ratio_le4")) for item in candidate_analyses),
            default=0.0,
        ),
    }


def build_contour_phrase_shape_repair_report(
    objective_next_report: dict[str, Any],
    *,
    output_dir: Path,
) -> dict[str, Any]:
    source = validate_objective_next_decision(objective_next_report)
    candidate_count = source["candidate_count"]
    max_interval = 12
    midi_dir = output_dir / "midi"
    generated = [
        write_candidate_midi(
            output_path=midi_dir / f"contour_phrase_shape_rank_{index + 1:02d}.mid",
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
    aggregate = aggregate_candidate_analyses(candidate_analyses)
    no_overlap = not any(bool(item.get("overlap_detected", True)) for item in candidate_analyses)
    stepwise_reduced = _int(aggregate.get("stepwise_contour_bias_count")) < _int(
        source.get("source_stepwise_contour_bias_count")
    )
    target_passed = (
        stepwise_reduced
        and _int(aggregate.get("max_abs_interval_max")) <= max_interval
        and _int(aggregate.get("shared_rhythm_signature_count")) <= 1
        and _int(aggregate.get("uniform_bar_density_count")) <= 1
        and no_overlap
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "boundary": BOUNDARY,
        "source_boundary": SOURCE_BOUNDARY,
        "source_objective_summary": source,
        "generated_candidates": generated,
        "candidate_analyses": candidate_analyses,
        "aggregate": aggregate,
        "repair_result": {
            "target_passed": target_passed,
            "stepwise_contour_bias_reduced": stepwise_reduced,
            "source_stepwise_contour_bias_count": source["source_stepwise_contour_bias_count"],
            "repaired_stepwise_contour_bias_count": _int(aggregate.get("stepwise_contour_bias_count")),
            "no_overlap": no_overlap,
            "phrase_vocabulary_source": "contour_phrase_shape_cells",
            "audio_review_required": True,
        },
        "readiness": {
            "boundary": BOUNDARY,
            "contour_phrase_shape_repair_completed": True,
            "objective_repair_target_passed": target_passed,
            "generated_midi_file_count": len(generated),
            "human_audio_preference_claimed": False,
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
            "reason": "contour phrase-shape candidates reduce stepwise contour bias and require audio package next",
        },
        "not_proven": [
            "audio_review_completed",
            "human_audio_preference",
            "model_direct_generation_quality",
            "midi_to_solo_musical_quality",
            "broad_trained_model_quality",
            "brad_style_adaptation",
        ],
        "next_recommended_issue": "Stage B MIDI-to-solo model-direct jazz phrase vocabulary contour phrase-shape audio package",
    }


def validate_contour_phrase_shape_repair_report(
    report: dict[str, Any],
    *,
    expected_boundary: str | None,
    expected_next_boundary: str | None,
    require_repair_completed: bool,
    require_target_passed: bool,
    require_stepwise_reduced: bool,
    require_no_quality_claim: bool,
) -> dict[str, Any]:
    boundary = str(report.get("boundary") or "")
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    aggregate = _dict(report.get("aggregate"))
    repair = _dict(report.get("repair_result"))
    if expected_boundary and boundary != expected_boundary:
        raise StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeRepairError(
            f"expected boundary {expected_boundary}, got {boundary}"
        )
    if expected_next_boundary and str(decision.get("next_boundary") or "") != expected_next_boundary:
        raise StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeRepairError(
            "unexpected next boundary"
        )
    if require_repair_completed and not bool(readiness.get("contour_phrase_shape_repair_completed", False)):
        raise StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeRepairError(
            "repair completion required"
        )
    if require_target_passed and not bool(repair.get("target_passed", False)):
        raise StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeRepairError(
            "repair target pass required"
        )
    if require_stepwise_reduced and not bool(repair.get("stepwise_contour_bias_reduced", False)):
        raise StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeRepairError(
            "stepwise contour bias reduction required"
        )
    if require_no_quality_claim:
        blocked = [
            "human_audio_preference_claimed",
            "model_direct_generation_quality_claimed",
            "midi_to_solo_musical_quality_claimed",
            "broad_trained_model_quality_claimed",
            "brad_style_adaptation_claimed",
        ]
        claimed = [name for name in blocked if bool(readiness.get(name, True))]
        if claimed:
            raise StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeRepairError(
                f"unexpected quality claim: {claimed}"
            )
    return {
        "boundary": boundary,
        "source_boundary": str(report.get("source_boundary") or ""),
        "target_passed": bool(repair.get("target_passed", False)),
        "generated_midi_file_count": _int(readiness.get("generated_midi_file_count")),
        "source_stepwise_contour_bias_count": _int(repair.get("source_stepwise_contour_bias_count")),
        "repaired_stepwise_contour_bias_count": _int(repair.get("repaired_stepwise_contour_bias_count")),
        "stepwise_contour_bias_reduced": bool(repair.get("stepwise_contour_bias_reduced", False)),
        "max_small_interval_ratio_le4": _float(aggregate.get("max_small_interval_ratio_le4")),
        "max_abs_interval_max": _int(aggregate.get("max_abs_interval_max")),
        "shared_rhythm_signature_count": _int(aggregate.get("shared_rhythm_signature_count")),
        "uniform_bar_density_count": _int(aggregate.get("uniform_bar_density_count")),
        "no_overlap": bool(repair.get("no_overlap", False)),
        "phrase_vocabulary_source": str(repair.get("phrase_vocabulary_source") or ""),
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
    repair = report["repair_result"]
    readiness = report["readiness"]
    decision = report["decision"]
    lines = [
        "# Stage B MIDI-to-Solo Model-Direct Jazz Phrase Vocabulary Contour Phrase-Shape Repair",
        "",
        "## Summary",
        "",
        f"- boundary: `{report['boundary']}`",
        f"- source boundary: `{report['source_boundary']}`",
        f"- next boundary: `{decision['next_boundary']}`",
        f"- generated MIDI file count: `{readiness['generated_midi_file_count']}`",
        f"- target passed: `{_bool_token(repair['target_passed'])}`",
        f"- stepwise contour bias: `{repair['source_stepwise_contour_bias_count']} -> {repair['repaired_stepwise_contour_bias_count']}`",
        f"- max small interval ratio <=4: `{float(aggregate['max_small_interval_ratio_le4']):.4f}`",
        f"- max abs interval max: `{aggregate['max_abs_interval_max']}`",
        f"- shared rhythm signature count: `{aggregate['shared_rhythm_signature_count']}`",
        f"- no overlap: `{_bool_token(repair['no_overlap'])}`",
        f"- phrase vocabulary source: `{repair['phrase_vocabulary_source']}`",
        f"- MIDI-to-solo musical quality claimed: `{_bool_token(readiness['midi_to_solo_musical_quality_claimed'])}`",
        "",
        "## Candidates",
        "",
        "| rank | notes | bars | density pattern | max interval | small interval ratio | flags |",
        "|---:|---:|---:|---|---:|---:|---|",
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
                    f"{float(item['small_interval_ratio_le4']):.4f}",
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
    parser = argparse.ArgumentParser(description="Run contour phrase-shape repair")
    parser.add_argument("--objective_next", type=str, required=True)
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_repair",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--expected_boundary", type=str, default="")
    parser.add_argument("--expected_next_boundary", type=str, default="")
    parser.add_argument("--require_repair_completed", action="store_true")
    parser.add_argument("--require_target_passed", action="store_true")
    parser.add_argument("--require_stepwise_reduced", action="store_true")
    parser.add_argument("--require_no_quality_claim", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    report = build_contour_phrase_shape_repair_report(
        read_json(Path(args.objective_next)),
        output_dir=output_dir,
    )
    summary = validate_contour_phrase_shape_repair_report(
        report,
        expected_boundary=str(args.expected_boundary or ""),
        expected_next_boundary=str(args.expected_next_boundary or ""),
        require_repair_completed=bool(args.require_repair_completed),
        require_target_passed=bool(args.require_target_passed),
        require_stepwise_reduced=bool(args.require_stepwise_reduced),
        require_no_quality_claim=bool(args.require_no_quality_claim),
    )
    write_json(output_dir / "stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_repair.json", report)
    write_json(
        output_dir
        / "stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_repair_validation_summary.json",
        summary,
    )
    markdown = markdown_report(report)
    write_text(output_dir / "stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_repair.md", markdown)
    if args.doc_path:
        write_text(Path(args.doc_path), markdown)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
