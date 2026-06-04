"""Consolidate contour phrase-shape objective-clean repeatability evidence."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from scripts.assess_stage_b_generic_base_readiness import write_json, write_text  # noqa: E402
from scripts.run_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_repeatability_sweep import (  # noqa: E402
    BOUNDARY as SOURCE_BOUNDARY,
)


class StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeRepeatabilityConsolidationError(
    ValueError
):
    pass


BOUNDARY = (
    "stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_"
    "objective_clean_repeatability_consolidation"
)
NEXT_BOUNDARY = (
    "stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_"
    "repeatability_audio_review_package"
)
FAILURE_NEXT_BOUNDARY = (
    "stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_"
    "repeatability_failure_repair_decision"
)
SCHEMA_VERSION = (
    "stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_"
    "repeatability_consolidation_v1"
)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeRepeatabilityConsolidationError(
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


def validate_repeatability_sweep_report(report: dict[str, Any]) -> dict[str, Any]:
    if str(report.get("boundary") or "") != SOURCE_BOUNDARY:
        raise StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeRepeatabilityConsolidationError(
            "repeatability sweep report required"
        )
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    aggregate = _dict(report.get("aggregate"))
    repeatability = _dict(report.get("repeatability_result"))
    if str(decision.get("next_boundary") or "") != BOUNDARY:
        raise StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeRepeatabilityConsolidationError(
            "repeatability sweep must route to consolidation"
        )
    if not bool(repeatability.get("repeatability_passed", False)):
        raise StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeRepeatabilityConsolidationError(
            "repeatability pass required"
        )
    if _int(repeatability.get("qualified_candidate_count")) < _int(repeatability.get("sample_count")):
        raise StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeRepeatabilityConsolidationError(
            "all sweep samples should qualify for consolidation"
        )
    if _int(aggregate.get("current_analysis_flag_count")) != 0:
        raise StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeRepeatabilityConsolidationError(
            "analysis flag count should be zero"
        )
    if _int(aggregate.get("overlap_detected_count")) != 0:
        raise StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeRepeatabilityConsolidationError(
            "overlap count should be zero"
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
        raise StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeRepeatabilityConsolidationError(
            f"unexpected upstream claim: {claimed}"
        )
    candidates = [_dict(item) for item in _list(report.get("candidate_analyses")) if isinstance(item, dict)]
    generated = [_dict(item) for item in _list(report.get("generated_candidates")) if isinstance(item, dict)]
    if len(candidates) != _int(repeatability.get("sample_count")):
        raise StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeRepeatabilityConsolidationError(
            "candidate analysis count mismatch"
        )
    return {
        "sample_count": _int(repeatability.get("sample_count")),
        "generated_midi_file_count": _int(readiness.get("generated_midi_file_count")),
        "qualified_candidate_count": _int(repeatability.get("qualified_candidate_count")),
        "objective_clean_pass_rate": _float(repeatability.get("objective_clean_pass_rate")),
        "current_analysis_flag_count": _int(aggregate.get("current_analysis_flag_count")),
        "overlap_detected_count": _int(aggregate.get("overlap_detected_count")),
        "distinct_density_pattern_count": _int(aggregate.get("distinct_density_pattern_count")),
        "max_abs_interval_max": _int(aggregate.get("max_abs_interval_max")),
        "max_small_interval_ratio_le4": _float(aggregate.get("max_small_interval_ratio_le4")),
        "generated_midi_paths": [str(item.get("midi_path") or "") for item in generated],
        "candidate_ranks": [_int(item.get("rank")) for item in candidates],
    }


def build_repeatability_consolidation_report(
    repeatability_sweep: dict[str, Any],
    *,
    output_dir: Path,
) -> dict[str, Any]:
    evidence = validate_repeatability_sweep_report(repeatability_sweep)
    objective_repeatability_support = (
        _int(evidence.get("sample_count")) > 0
        and _int(evidence.get("qualified_candidate_count")) == _int(evidence.get("sample_count"))
        and _int(evidence.get("current_analysis_flag_count")) == 0
        and _int(evidence.get("overlap_detected_count")) == 0
    )
    next_boundary = NEXT_BOUNDARY if objective_repeatability_support else FAILURE_NEXT_BOUNDARY
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "boundary": BOUNDARY,
        "source_boundary": SOURCE_BOUNDARY,
        "evidence_summary": evidence,
        "consolidation_result": {
            "objective_repeatability_support": objective_repeatability_support,
            "additional_repair_required": not objective_repeatability_support,
            "audio_review_package_required": objective_repeatability_support,
            "support_scope": "objective_midi_only",
            "human_audio_preference_claimed": False,
            "midi_to_solo_musical_quality_claimed": False,
        },
        "readiness": {
            "boundary": BOUNDARY,
            "repeatability_consolidation_completed": True,
            "objective_repeatability_support": objective_repeatability_support,
            "audio_review_package_required": objective_repeatability_support,
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
            "reason": "objective-clean repeatability passed; route generated MIDI variants to audio review package",
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
            "Stage B MIDI-to-solo model-direct jazz phrase vocabulary contour phrase-shape repeatability audio review package"
            if objective_repeatability_support
            else "Stage B MIDI-to-solo model-direct jazz phrase vocabulary contour phrase-shape repeatability failure repair decision"
        ),
    }


def validate_repeatability_consolidation_report(
    report: dict[str, Any],
    *,
    expected_boundary: str | None,
    expected_next_boundary: str | None,
    min_sample_count: int,
    min_qualified_count: int,
    require_objective_support: bool,
    require_audio_review_required: bool,
    require_no_quality_claim: bool,
) -> dict[str, Any]:
    boundary = str(report.get("boundary") or "")
    evidence = _dict(report.get("evidence_summary"))
    result = _dict(report.get("consolidation_result"))
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    if expected_boundary and boundary != expected_boundary:
        raise StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeRepeatabilityConsolidationError(
            f"expected boundary {expected_boundary}, got {boundary}"
        )
    if expected_next_boundary and str(decision.get("next_boundary") or "") != expected_next_boundary:
        raise StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeRepeatabilityConsolidationError(
            "unexpected next boundary"
        )
    if _int(evidence.get("sample_count")) < int(min_sample_count):
        raise StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeRepeatabilityConsolidationError(
            "sample count below requirement"
        )
    if _int(evidence.get("qualified_candidate_count")) < int(min_qualified_count):
        raise StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeRepeatabilityConsolidationError(
            "qualified candidate count below requirement"
        )
    if require_objective_support and not bool(result.get("objective_repeatability_support", False)):
        raise StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeRepeatabilityConsolidationError(
            "objective repeatability support required"
        )
    if require_audio_review_required and not bool(result.get("audio_review_package_required", False)):
        raise StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeRepeatabilityConsolidationError(
            "audio review package requirement expected"
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
            raise StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeRepeatabilityConsolidationError(
                f"unexpected quality claim: {claimed}"
            )
    return {
        "boundary": boundary,
        "source_boundary": str(report.get("source_boundary") or ""),
        "sample_count": _int(evidence.get("sample_count")),
        "generated_midi_file_count": _int(evidence.get("generated_midi_file_count")),
        "qualified_candidate_count": _int(evidence.get("qualified_candidate_count")),
        "objective_clean_pass_rate": _float(evidence.get("objective_clean_pass_rate")),
        "current_analysis_flag_count": _int(evidence.get("current_analysis_flag_count")),
        "overlap_detected_count": _int(evidence.get("overlap_detected_count")),
        "distinct_density_pattern_count": _int(evidence.get("distinct_density_pattern_count")),
        "max_abs_interval_max": _int(evidence.get("max_abs_interval_max")),
        "max_small_interval_ratio_le4": _float(evidence.get("max_small_interval_ratio_le4")),
        "objective_repeatability_support": bool(result.get("objective_repeatability_support", False)),
        "additional_repair_required": bool(result.get("additional_repair_required", True)),
        "audio_review_package_required": bool(result.get("audio_review_package_required", False)),
        "human_audio_preference_claimed": bool(readiness.get("human_audio_preference_claimed", True)),
        "midi_to_solo_musical_quality_claimed": bool(
            readiness.get("midi_to_solo_musical_quality_claimed", True)
        ),
        "critical_user_input_required": bool(decision.get("critical_user_input_required", True)),
        "next_boundary": str(decision.get("next_boundary") or ""),
        "next_recommended_issue": str(report.get("next_recommended_issue") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    evidence = report["evidence_summary"]
    result = report["consolidation_result"]
    readiness = report["readiness"]
    decision = report["decision"]
    lines = [
        "# Stage B MIDI-to-Solo Model-Direct Jazz Phrase Vocabulary Contour Phrase-Shape Repeatability Consolidation",
        "",
        "## Summary",
        "",
        f"- boundary: `{report['boundary']}`",
        f"- source boundary: `{report['source_boundary']}`",
        f"- next boundary: `{decision['next_boundary']}`",
        f"- sample count: `{evidence['sample_count']}`",
        f"- generated MIDI file count: `{evidence['generated_midi_file_count']}`",
        f"- qualified candidate count: `{evidence['qualified_candidate_count']}`",
        f"- objective-clean pass rate: `{float(evidence['objective_clean_pass_rate']):.4f}`",
        f"- current analysis flag count: `{evidence['current_analysis_flag_count']}`",
        f"- overlap detected count: `{evidence['overlap_detected_count']}`",
        f"- distinct density pattern count: `{evidence['distinct_density_pattern_count']}`",
        f"- max abs interval max: `{evidence['max_abs_interval_max']}`",
        f"- max small interval ratio <=4: `{float(evidence['max_small_interval_ratio_le4']):.4f}`",
        f"- objective repeatability support: `{_bool_token(result['objective_repeatability_support'])}`",
        f"- audio review package required: `{_bool_token(result['audio_review_package_required'])}`",
        f"- MIDI-to-solo musical quality claimed: `{_bool_token(readiness['midi_to_solo_musical_quality_claimed'])}`",
        "",
        "## MIDI Files",
        "",
    ]
    for path in evidence.get("generated_midi_paths", []):
        lines.append(f"- `{path}`")
    lines.extend(["", "## Not Proven", ""])
    for item in report.get("not_proven", []):
        lines.append(f"- `{item}`")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Consolidate contour phrase-shape repeatability")
    parser.add_argument("--repeatability_sweep", type=str, required=True)
    parser.add_argument(
        "--output_root",
        type=str,
        default=(
            "outputs/stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_"
            "contour_phrase_shape_repeatability_consolidation"
        ),
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--expected_boundary", type=str, default="")
    parser.add_argument("--expected_next_boundary", type=str, default="")
    parser.add_argument("--min_sample_count", type=int, default=6)
    parser.add_argument("--min_qualified_count", type=int, default=6)
    parser.add_argument("--require_objective_support", action="store_true")
    parser.add_argument("--require_audio_review_required", action="store_true")
    parser.add_argument("--require_no_quality_claim", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    report = build_repeatability_consolidation_report(
        read_json(Path(args.repeatability_sweep)),
        output_dir=output_dir,
    )
    summary = validate_repeatability_consolidation_report(
        report,
        expected_boundary=str(args.expected_boundary or ""),
        expected_next_boundary=str(args.expected_next_boundary or ""),
        min_sample_count=int(args.min_sample_count),
        min_qualified_count=int(args.min_qualified_count),
        require_objective_support=bool(args.require_objective_support),
        require_audio_review_required=bool(args.require_audio_review_required),
        require_no_quality_claim=bool(args.require_no_quality_claim),
    )
    write_json(
        output_dir / "stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_repeatability_consolidation.json",
        report,
    )
    write_json(
        output_dir
        / "stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_repeatability_consolidation_validation_summary.json",
        summary,
    )
    markdown = markdown_report(report)
    write_text(
        output_dir / "stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_repeatability_consolidation.md",
        markdown,
    )
    if args.doc_path:
        write_text(Path(args.doc_path), markdown)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
