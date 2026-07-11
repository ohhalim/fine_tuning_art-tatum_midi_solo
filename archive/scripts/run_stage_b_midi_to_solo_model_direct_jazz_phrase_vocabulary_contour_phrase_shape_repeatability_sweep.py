"""Run objective-clean repeatability sweep for contour phrase-shape candidates."""

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
from scripts.decide_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_objective_next import (  # noqa: E402
    BOUNDARY as SOURCE_BOUNDARY,
)
from scripts.run_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_repair import (  # noqa: E402
    contour_phrase_cells,
    fit_interval_limit,
    has_overlap,
)
from scripts.run_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_probe import (  # noqa: E402
    BAR_SECONDS,
    durations_for_count,
    onset_patterns_by_density,
)


class StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeRepeatabilitySweepError(
    ValueError
):
    pass


BOUNDARY = (
    "stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_"
    "objective_clean_repeatability_sweep"
)
NEXT_BOUNDARY = (
    "stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_"
    "objective_clean_repeatability_consolidation"
)
FAILURE_NEXT_BOUNDARY = (
    "stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_"
    "repeatability_failure_repair_decision"
)
SCHEMA_VERSION = (
    "stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_"
    "repeatability_sweep_v1"
)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeRepeatabilitySweepError(
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


def validate_objective_next_report(report: dict[str, Any]) -> dict[str, Any]:
    boundary = _dict(report.get("objective_next_decision_boundary"))
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    objective = _dict(report.get("objective_summary"))
    if str(boundary.get("boundary") or report.get("boundary") or "") != SOURCE_BOUNDARY:
        raise StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeRepeatabilitySweepError(
            "contour phrase-shape objective-only decision report required"
        )
    if str(decision.get("next_boundary") or "") != BOUNDARY:
        raise StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeRepeatabilitySweepError(
            "objective-only decision must route to objective-clean repeatability sweep"
        )
    if not bool(readiness.get("objective_clean_candidate_boundary_supported", False)):
        raise StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeRepeatabilitySweepError(
            "objective-clean candidate boundary support required"
        )
    if bool(readiness.get("additional_repair_required", True)):
        raise StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeRepeatabilitySweepError(
            "additional repair should be false before repeatability sweep"
        )
    if _int(objective.get("current_analysis_flag_count")) != 0:
        raise StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeRepeatabilitySweepError(
            "source objective flags should be zero before repeatability sweep"
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
        raise StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeRepeatabilitySweepError(
            f"unexpected upstream claim: {claimed}"
        )
    return {
        "candidate_count": _int(objective.get("candidate_count")),
        "current_analysis_flag_count": _int(objective.get("current_analysis_flag_count")),
        "source_stepwise_contour_bias_count": _int(objective.get("source_stepwise_contour_bias_count")),
        "repaired_stepwise_contour_bias_count": _int(objective.get("repaired_stepwise_contour_bias_count")),
        "max_abs_interval_max": _int(objective.get("max_abs_interval_max")),
        "max_small_interval_ratio_le4": _float(objective.get("max_small_interval_ratio_le4")),
        "no_overlap": bool(objective.get("no_overlap", False)),
    }


def sweep_density_patterns() -> list[list[int]]:
    return [
        [5, 3, 6, 4, 5, 4, 6, 3],
        [6, 4, 3, 5, 6, 5, 3, 4],
        [5, 6, 4, 3, 5, 3, 6, 4],
        [5, 6, 3, 5, 4, 6, 5, 3],
        [6, 3, 5, 6, 4, 5, 6, 3],
        [3, 5, 6, 3, 5, 4, 3, 6],
    ]


def root_cycles() -> list[list[int]]:
    return [
        [62, 65, 60, 67, 62, 65, 60, 67],
        [57, 64, 60, 67, 57, 64, 60, 67],
        [65, 60, 67, 62, 65, 60, 67, 62],
        [60, 67, 62, 65, 60, 67, 62, 65],
        [64, 57, 65, 60, 64, 57, 65, 60],
        [67, 62, 65, 60, 67, 62, 65, 60],
    ]


def repeatability_pitches(seed: int, densities: list[int], *, max_interval: int) -> list[int]:
    roots = root_cycles()[seed % len(root_cycles())]
    pitches: list[int] = []
    for bar_index, density in enumerate(densities):
        root = roots[bar_index]
        cell = contour_phrase_cells(seed + (bar_index // 2), density, bar_index)
        if seed >= 3 and bar_index % 2:
            cell = list(reversed(cell))
        pitches.extend(root + value for value in cell)
    return fit_interval_limit(pitches, max_interval=max_interval)


def write_sweep_candidate_midi(
    *,
    output_path: Path,
    seed: int,
    max_interval: int,
) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    densities = sweep_density_patterns()[seed % len(sweep_density_patterns())]
    onset_bank = onset_patterns_by_density(seed)
    pitches = repeatability_pitches(seed, densities, max_interval=max_interval)
    midi = pretty_midi.PrettyMIDI(initial_tempo=120)
    midi.time_signature_changes.append(pretty_midi.TimeSignature(4, 4, 0.0))
    instrument = pretty_midi.Instrument(
        program=0,
        is_drum=False,
        name=f"contour_phrase_shape_repeatability_seed_{seed + 1}",
    )
    pitch_index = 0
    for bar_index, density in enumerate(densities):
        onset_options = onset_bank[density]
        offsets = onset_options[bar_index % len(onset_options)]
        durations = durations_for_count(seed, density, bar_index)
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
        "rank": seed + 1,
        "seed": seed,
        "midi_path": str(output_path),
        "density_pattern": densities,
        "phrase_vocabulary_source": "contour_phrase_shape_repeatability_cells",
    }


def aggregate_candidate_analyses(candidate_analyses: list[dict[str, Any]]) -> dict[str, Any]:
    flag_counts = Counter(flag for item in candidate_analyses for flag in _list(item.get("analysis_flags")))
    density_patterns = {tuple(_int(value) for value in _list(item.get("density_pattern"))) for item in candidate_analyses}
    clean_candidates = [
        item
        for item in candidate_analyses
        if not _list(item.get("analysis_flags")) and not bool(item.get("overlap_detected", True))
    ]
    return {
        "candidate_count": len(candidate_analyses),
        "qualified_candidate_count": len(clean_candidates),
        "objective_clean_pass_rate": len(clean_candidates) / max(1, len(candidate_analyses)),
        "flag_counts": dict(sorted(flag_counts.items())),
        "current_analysis_flag_count": sum(flag_counts.values()),
        "stepwise_contour_bias_count": flag_counts.get("stepwise_contour_bias", 0),
        "uniform_bar_density_count": flag_counts.get("uniform_bar_density", 0),
        "four_notes_per_bar_template_count": flag_counts.get("four_notes_per_bar_template", 0),
        "duration_template_monotony_count": flag_counts.get("duration_template_monotony", 0),
        "ioi_template_monotony_count": flag_counts.get("ioi_template_monotony", 0),
        "safe_interval_cap_compression_count": flag_counts.get("safe_interval_cap_compression", 0),
        "four_bar_rhythm_cycle_repeated_count": flag_counts.get("four_bar_rhythm_cycle_repeated", 0),
        "distinct_density_pattern_count": len(density_patterns),
        "max_abs_interval_max": max((_int(item.get("max_abs_interval")) for item in candidate_analyses), default=0),
        "max_small_interval_ratio_le4": max(
            (_float(item.get("small_interval_ratio_le4")) for item in candidate_analyses),
            default=0.0,
        ),
        "overlap_detected_count": sum(1 for item in candidate_analyses if bool(item.get("overlap_detected", False))),
    }


def build_repeatability_sweep_report(
    objective_next_report: dict[str, Any],
    *,
    output_dir: Path,
    sample_count: int,
    max_interval: int,
) -> dict[str, Any]:
    source = validate_objective_next_report(objective_next_report)
    if sample_count > len(sweep_density_patterns()):
        raise StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeRepeatabilitySweepError(
            "sample count exceeds configured sweep variants"
        )
    midi_dir = output_dir / "midi"
    generated = [
        write_sweep_candidate_midi(
            output_path=midi_dir / f"contour_phrase_shape_repeatability_seed_{seed + 1:02d}.mid",
            seed=seed,
            max_interval=max_interval,
        )
        for seed in range(sample_count)
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
    repeatability_passed = (
        _int(aggregate.get("qualified_candidate_count")) == sample_count
        and _int(aggregate.get("current_analysis_flag_count")) == 0
        and _int(aggregate.get("overlap_detected_count")) == 0
        and _int(aggregate.get("max_abs_interval_max")) <= max_interval
    )
    next_boundary = NEXT_BOUNDARY if repeatability_passed else FAILURE_NEXT_BOUNDARY
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
        "repeatability_result": {
            "repeatability_passed": repeatability_passed,
            "sample_count": sample_count,
            "qualified_candidate_count": _int(aggregate.get("qualified_candidate_count")),
            "objective_clean_pass_rate": _float(aggregate.get("objective_clean_pass_rate")),
            "max_interval": max_interval,
            "failure_next_boundary": FAILURE_NEXT_BOUNDARY,
        },
        "readiness": {
            "boundary": BOUNDARY,
            "objective_clean_repeatability_sweep_completed": True,
            "repeatability_passed": repeatability_passed,
            "generated_midi_file_count": len(generated),
            "human_audio_preference_claimed": False,
            "model_direct_generation_quality_claimed": False,
            "midi_to_solo_musical_quality_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
        },
        "decision": {
            "current_boundary": BOUNDARY,
            "next_boundary": next_boundary,
            "auto_progress_allowed": True,
            "critical_user_input_required": False,
            "reason": "objective-clean candidates repeated across distinct density/root variants",
        },
        "not_proven": [
            "listening_review_completed",
            "human_audio_preference",
            "model_direct_generation_quality",
            "midi_to_solo_musical_quality",
            "broad_trained_model_quality",
            "brad_style_adaptation",
        ],
        "next_recommended_issue": (
            "Stage B MIDI-to-solo model-direct jazz phrase vocabulary contour phrase-shape "
            "objective-clean repeatability consolidation"
            if repeatability_passed
            else "Stage B MIDI-to-solo model-direct jazz phrase vocabulary contour phrase-shape repeatability failure repair decision"
        ),
    }


def validate_repeatability_sweep_report(
    report: dict[str, Any],
    *,
    expected_boundary: str | None,
    expected_next_boundary: str | None,
    min_sample_count: int,
    min_qualified_count: int,
    require_repeatability_passed: bool,
    require_no_quality_claim: bool,
) -> dict[str, Any]:
    boundary = str(report.get("boundary") or "")
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    aggregate = _dict(report.get("aggregate"))
    repeatability = _dict(report.get("repeatability_result"))
    if expected_boundary and boundary != expected_boundary:
        raise StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeRepeatabilitySweepError(
            f"expected boundary {expected_boundary}, got {boundary}"
        )
    if expected_next_boundary and str(decision.get("next_boundary") or "") != expected_next_boundary:
        raise StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeRepeatabilitySweepError(
            "unexpected next boundary"
        )
    if _int(repeatability.get("sample_count")) < int(min_sample_count):
        raise StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeRepeatabilitySweepError(
            "sample count below requirement"
        )
    if _int(repeatability.get("qualified_candidate_count")) < int(min_qualified_count):
        raise StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeRepeatabilitySweepError(
            "qualified candidate count below requirement"
        )
    if require_repeatability_passed and not bool(repeatability.get("repeatability_passed", False)):
        raise StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeRepeatabilitySweepError(
            "repeatability pass required"
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
            raise StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeRepeatabilitySweepError(
                f"unexpected quality claim: {claimed}"
            )
    return {
        "boundary": boundary,
        "source_boundary": str(report.get("source_boundary") or ""),
        "sample_count": _int(repeatability.get("sample_count")),
        "generated_midi_file_count": _int(readiness.get("generated_midi_file_count")),
        "qualified_candidate_count": _int(repeatability.get("qualified_candidate_count")),
        "objective_clean_pass_rate": _float(repeatability.get("objective_clean_pass_rate")),
        "repeatability_passed": bool(repeatability.get("repeatability_passed", False)),
        "current_analysis_flag_count": _int(aggregate.get("current_analysis_flag_count")),
        "overlap_detected_count": _int(aggregate.get("overlap_detected_count")),
        "distinct_density_pattern_count": _int(aggregate.get("distinct_density_pattern_count")),
        "max_abs_interval_max": _int(aggregate.get("max_abs_interval_max")),
        "max_small_interval_ratio_le4": _float(aggregate.get("max_small_interval_ratio_le4")),
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
    repeatability = report["repeatability_result"]
    readiness = report["readiness"]
    decision = report["decision"]
    lines = [
        "# Stage B MIDI-to-Solo Model-Direct Jazz Phrase Vocabulary Contour Phrase-Shape Objective-Clean Repeatability Sweep",
        "",
        "## Summary",
        "",
        f"- boundary: `{report['boundary']}`",
        f"- source boundary: `{report['source_boundary']}`",
        f"- next boundary: `{decision['next_boundary']}`",
        f"- sample count: `{repeatability['sample_count']}`",
        f"- generated MIDI file count: `{readiness['generated_midi_file_count']}`",
        f"- qualified candidate count: `{repeatability['qualified_candidate_count']}`",
        f"- objective-clean pass rate: `{float(repeatability['objective_clean_pass_rate']):.4f}`",
        f"- repeatability passed: `{_bool_token(repeatability['repeatability_passed'])}`",
        f"- current analysis flag count: `{aggregate['current_analysis_flag_count']}`",
        f"- overlap detected count: `{aggregate['overlap_detected_count']}`",
        f"- distinct density pattern count: `{aggregate['distinct_density_pattern_count']}`",
        f"- max abs interval max: `{aggregate['max_abs_interval_max']}`",
        f"- max small interval ratio <=4: `{float(aggregate['max_small_interval_ratio_le4']):.4f}`",
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
    parser = argparse.ArgumentParser(description="Run contour phrase-shape objective-clean repeatability sweep")
    parser.add_argument("--objective_next", type=str, required=True)
    parser.add_argument(
        "--output_root",
        type=str,
        default=(
            "outputs/stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_"
            "contour_phrase_shape_repeatability_sweep"
        ),
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--expected_boundary", type=str, default="")
    parser.add_argument("--expected_next_boundary", type=str, default="")
    parser.add_argument("--sample_count", type=int, default=6)
    parser.add_argument("--max_interval", type=int, default=12)
    parser.add_argument("--min_sample_count", type=int, default=6)
    parser.add_argument("--min_qualified_count", type=int, default=6)
    parser.add_argument("--require_repeatability_passed", action="store_true")
    parser.add_argument("--require_no_quality_claim", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    report = build_repeatability_sweep_report(
        read_json(Path(args.objective_next)),
        output_dir=output_dir,
        sample_count=int(args.sample_count),
        max_interval=int(args.max_interval),
    )
    summary = validate_repeatability_sweep_report(
        report,
        expected_boundary=str(args.expected_boundary or ""),
        expected_next_boundary=str(args.expected_next_boundary or ""),
        min_sample_count=int(args.min_sample_count),
        min_qualified_count=int(args.min_qualified_count),
        require_repeatability_passed=bool(args.require_repeatability_passed),
        require_no_quality_claim=bool(args.require_no_quality_claim),
    )
    write_json(
        output_dir / "stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_repeatability_sweep.json",
        report,
    )
    write_json(
        output_dir
        / "stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_repeatability_sweep_validation_summary.json",
        summary,
    )
    markdown = markdown_report(report)
    write_text(
        output_dir / "stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_repeatability_sweep.md",
        markdown,
    )
    if args.doc_path:
        write_text(Path(args.doc_path), markdown)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
