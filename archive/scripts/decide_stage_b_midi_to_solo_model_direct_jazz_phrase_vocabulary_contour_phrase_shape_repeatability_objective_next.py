"""Decide the objective-only next boundary after repeatability listening review setup."""

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
from scripts.build_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_repeatability_listening_review import (  # noqa: E402
    BOUNDARY as SOURCE_BOUNDARY,
)
from scripts.consolidate_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_repeatability import (  # noqa: E402
    BOUNDARY as REPEATABILITY_CONSOLIDATION_BOUNDARY,
)


class StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeRepeatabilityObjectiveNextError(
    ValueError
):
    pass


BOUNDARY = (
    "stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_"
    "repeatability_objective_only_next_decision"
)
FINAL_BOUNDARY = (
    "stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_"
    "repeatability_objective_path_complete"
)
NEXT_BOUNDARY = "stage_b_model_core_evidence_readme_refresh"
SCHEMA_VERSION = (
    "stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_"
    "repeatability_objective_next_v1"
)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeRepeatabilityObjectiveNextError(
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


def _count_or_len(summary: dict[str, Any], count_key: str, list_key: str) -> int:
    count = _int(summary.get(count_key))
    if count:
        return count
    return len(_list(summary.get(list_key)))


def validate_listening_review_boundary(report: dict[str, Any]) -> dict[str, Any]:
    boundary = _dict(report.get("listening_review_boundary"))
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    review_summary = _dict(report.get("review_input_summary"))
    if str(boundary.get("boundary") or report.get("boundary") or "") != SOURCE_BOUNDARY:
        raise StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeRepeatabilityObjectiveNextError(
            "repeatability listening review boundary required"
        )
    if str(decision.get("next_boundary") or "") != BOUNDARY:
        raise StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeRepeatabilityObjectiveNextError(
            "repeatability listening review must route to objective-only next decision"
        )
    if not bool(readiness.get("listening_review_boundary_prepared", False)):
        raise StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeRepeatabilityObjectiveNextError(
            "listening review boundary preparation required"
        )
    if bool(readiness.get("validated_review_input_present", True)):
        raise StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeRepeatabilityObjectiveNextError(
            "validated review input should be absent for objective-only decision"
        )
    blocked_claims = [
        "listening_review_completed",
        "human_audio_preference_claimed",
        "model_direct_generation_quality_claimed",
        "midi_to_solo_musical_quality_claimed",
        "broad_trained_model_quality_claimed",
        "brad_style_adaptation_claimed",
    ]
    claimed = [name for name in blocked_claims if bool(readiness.get(name, False))]
    if claimed:
        raise StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeRepeatabilityObjectiveNextError(
            f"unexpected upstream claim: {claimed}"
        )
    candidates = [_dict(item) for item in _list(report.get("review_candidates")) if isinstance(item, dict)]
    if len(candidates) < 1:
        raise StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeRepeatabilityObjectiveNextError(
            "repeatability review candidates required"
        )
    return {
        "candidate_count": len(candidates),
        "rendered_audio_file_count": _int(boundary.get("rendered_audio_file_count")),
        "review_input_template_written": bool(boundary.get("review_input_template_written", False)),
        "validated_review_input_present": bool(readiness.get("validated_review_input_present", True)),
        "preference_fill_allowed": bool(readiness.get("preference_fill_allowed", True)),
        "pending_status_field_count": _count_or_len(
            review_summary,
            "pending_status_field_count",
            "pending_status_fields",
        ),
        "pending_candidate_decision_count": _count_or_len(
            review_summary,
            "pending_candidate_decision_count",
            "pending_candidate_decisions",
        ),
        "pending_candidate_field_count": _count_or_len(
            review_summary,
            "pending_candidate_field_count",
            "pending_candidate_fields",
        ),
    }


def validate_repeatability_consolidation(report: dict[str, Any]) -> dict[str, Any]:
    if str(report.get("boundary") or "") != REPEATABILITY_CONSOLIDATION_BOUNDARY:
        raise StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeRepeatabilityObjectiveNextError(
            "repeatability consolidation boundary required"
        )
    evidence = _dict(report.get("evidence_summary"))
    result = _dict(report.get("consolidation_result"))
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    if not bool(result.get("objective_repeatability_support", False)):
        raise StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeRepeatabilityObjectiveNextError(
            "objective repeatability support required"
        )
    if bool(result.get("additional_repair_required", True)):
        raise StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeRepeatabilityObjectiveNextError(
            "additional repair should be false"
        )
    if not bool(result.get("audio_review_package_required", False)):
        raise StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeRepeatabilityObjectiveNextError(
            "audio review package should be required before listening boundary"
        )
    if _int(evidence.get("sample_count")) < 1:
        raise StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeRepeatabilityObjectiveNextError(
            "repeatability sample count required"
        )
    if _int(evidence.get("qualified_candidate_count")) != _int(evidence.get("sample_count")):
        raise StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeRepeatabilityObjectiveNextError(
            "all repeatability candidates should qualify"
        )
    if _int(evidence.get("current_analysis_flag_count")) != 0:
        raise StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeRepeatabilityObjectiveNextError(
            "repeatability analysis flags should be zero"
        )
    if _int(evidence.get("overlap_detected_count")) != 0:
        raise StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeRepeatabilityObjectiveNextError(
            "repeatability overlap count should be zero"
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
        raise StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeRepeatabilityObjectiveNextError(
            f"unexpected repeatability claim: {claimed}"
        )
    return {
        "source_boundary": str(report.get("source_boundary") or ""),
        "next_boundary": str(decision.get("next_boundary") or ""),
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
    }


def build_objective_next_decision_report(
    listening_review: dict[str, Any],
    repeatability_consolidation: dict[str, Any],
    *,
    output_dir: Path,
) -> dict[str, Any]:
    review = validate_listening_review_boundary(listening_review)
    repeatability = validate_repeatability_consolidation(repeatability_consolidation)
    objective_path_supported = (
        bool(repeatability.get("objective_repeatability_support", False))
        and _int(repeatability.get("sample_count")) == _int(repeatability.get("qualified_candidate_count"))
        and _int(repeatability.get("current_analysis_flag_count")) == 0
        and _int(repeatability.get("overlap_detected_count")) == 0
        and not bool(review.get("validated_review_input_present", True))
    )
    if not objective_path_supported:
        raise StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeRepeatabilityObjectiveNextError(
            "objective repeatability path support required"
        )
    boundary = {
        "boundary": BOUNDARY,
        "source_boundary": SOURCE_BOUNDARY,
        "repeatability_consolidation_boundary": REPEATABILITY_CONSOLIDATION_BOUNDARY,
        "final_boundary": FINAL_BOUNDARY,
        "objective_only_decision_completed": True,
        "objective_repeatability_path_supported": True,
        "selected_next_boundary": NEXT_BOUNDARY,
        "validated_review_input_present": False,
        "preference_fill_allowed": False,
        "human_audio_preference_claimed": False,
        "model_direct_generation_quality_claimed": False,
        "midi_to_solo_musical_quality_claimed": False,
        "broad_trained_model_quality_claimed": False,
        "brad_style_adaptation_claimed": False,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "boundary": BOUNDARY,
        "source_boundary": SOURCE_BOUNDARY,
        "final_boundary": FINAL_BOUNDARY,
        "review_boundary_summary": review,
        "objective_repeatability_summary": repeatability,
        "objective_next_decision_boundary": boundary,
        "readiness": {
            "boundary": BOUNDARY,
            "objective_only_decision_completed": True,
            "objective_repeatability_path_supported": True,
            "validated_review_input_present": False,
            "preference_fill_allowed": False,
            "human_audio_preference_claimed": False,
            "model_direct_generation_quality_claimed": False,
            "midi_to_solo_musical_quality_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
        },
        "decision": {
            "current_boundary": BOUNDARY,
            "final_boundary": FINAL_BOUNDARY,
            "next_boundary": NEXT_BOUNDARY,
            "auto_progress_allowed": True,
            "critical_user_input_required": False,
            "reason": (
                "repeatability objective evidence is supported; listening review input remains pending, "
                "so close the objective path and route evidence refresh without preference or quality claim"
            ),
        },
        "proven": [
            "repeatability_listening_review_input_template_prepared",
            "objective_repeatability_support_6_of_6",
            "human_audio_preference_claim_blocked_without_review_input",
        ],
        "not_proven": [
            "listening_review_completed",
            "human_audio_preference",
            "model_direct_generation_quality",
            "midi_to_solo_musical_quality",
            "broad_trained_model_quality",
            "brad_style_adaptation",
        ],
        "next_recommended_issue": "Stage B model-core evidence README refresh",
    }


def validate_objective_next_decision_report(
    report: dict[str, Any],
    *,
    expected_final_boundary: str | None,
    expected_next_boundary: str | None,
    min_sample_count: int,
    min_qualified_count: int,
    require_objective_support: bool,
    require_pending_review: bool,
    require_no_quality_claim: bool,
) -> dict[str, Any]:
    boundary = _dict(report.get("objective_next_decision_boundary"))
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    review = _dict(report.get("review_boundary_summary"))
    repeatability = _dict(report.get("objective_repeatability_summary"))
    if expected_final_boundary and str(decision.get("final_boundary") or "") != expected_final_boundary:
        raise StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeRepeatabilityObjectiveNextError(
            "unexpected final boundary"
        )
    if expected_next_boundary and str(decision.get("next_boundary") or "") != expected_next_boundary:
        raise StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeRepeatabilityObjectiveNextError(
            "unexpected next boundary"
        )
    if _int(repeatability.get("sample_count")) < int(min_sample_count):
        raise StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeRepeatabilityObjectiveNextError(
            "sample count below requirement"
        )
    if _int(repeatability.get("qualified_candidate_count")) < int(min_qualified_count):
        raise StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeRepeatabilityObjectiveNextError(
            "qualified candidate count below requirement"
        )
    if require_objective_support and not bool(readiness.get("objective_repeatability_path_supported", False)):
        raise StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeRepeatabilityObjectiveNextError(
            "objective repeatability path support required"
        )
    if require_pending_review and bool(readiness.get("validated_review_input_present", True)):
        raise StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeRepeatabilityObjectiveNextError(
            "review input should remain pending"
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
            raise StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeRepeatabilityObjectiveNextError(
                f"unexpected quality claim: {claimed}"
            )
    return {
        "boundary": str(boundary.get("boundary") or ""),
        "final_boundary": str(decision.get("final_boundary") or ""),
        "next_boundary": str(decision.get("next_boundary") or ""),
        "candidate_count": _int(review.get("candidate_count")),
        "rendered_audio_file_count": _int(review.get("rendered_audio_file_count")),
        "sample_count": _int(repeatability.get("sample_count")),
        "qualified_candidate_count": _int(repeatability.get("qualified_candidate_count")),
        "objective_clean_pass_rate": _float(repeatability.get("objective_clean_pass_rate")),
        "current_analysis_flag_count": _int(repeatability.get("current_analysis_flag_count")),
        "overlap_detected_count": _int(repeatability.get("overlap_detected_count")),
        "distinct_density_pattern_count": _int(repeatability.get("distinct_density_pattern_count")),
        "pending_status_field_count": _int(review.get("pending_status_field_count")),
        "pending_candidate_decision_count": _int(review.get("pending_candidate_decision_count")),
        "pending_candidate_field_count": _int(review.get("pending_candidate_field_count")),
        "objective_repeatability_path_supported": bool(
            readiness.get("objective_repeatability_path_supported", False)
        ),
        "validated_review_input_present": bool(readiness.get("validated_review_input_present", True)),
        "preference_fill_allowed": bool(readiness.get("preference_fill_allowed", True)),
        "human_audio_preference_claimed": bool(readiness.get("human_audio_preference_claimed", True)),
        "midi_to_solo_musical_quality_claimed": bool(
            readiness.get("midi_to_solo_musical_quality_claimed", True)
        ),
        "critical_user_input_required": bool(decision.get("critical_user_input_required", True)),
        "auto_progress_allowed": bool(decision.get("auto_progress_allowed", False)),
        "next_recommended_issue": str(report.get("next_recommended_issue") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    boundary = report["objective_next_decision_boundary"]
    decision = report["decision"]
    review = report["review_boundary_summary"]
    repeatability = report["objective_repeatability_summary"]
    lines = [
        "# Stage B MIDI-to-Solo Model-Direct Jazz Phrase Vocabulary Contour Phrase-Shape Repeatability Objective-Only Next Decision",
        "",
        "## Summary",
        "",
        f"- boundary: `{boundary['boundary']}`",
        f"- final boundary: `{decision['final_boundary']}`",
        f"- next boundary: `{decision['next_boundary']}`",
        f"- objective-only decision completed: `{_bool_token(boundary['objective_only_decision_completed'])}`",
        f"- objective repeatability path supported: `{_bool_token(boundary['objective_repeatability_path_supported'])}`",
        f"- candidate count: `{review['candidate_count']}`",
        f"- rendered audio file count: `{review['rendered_audio_file_count']}`",
        f"- sample count: `{repeatability['sample_count']}`",
        f"- qualified candidate count: `{repeatability['qualified_candidate_count']}`",
        f"- objective-clean pass rate: `{repeatability['objective_clean_pass_rate']:.4f}`",
        f"- current analysis flag count: `{repeatability['current_analysis_flag_count']}`",
        f"- overlap detected count: `{repeatability['overlap_detected_count']}`",
        f"- distinct density pattern count: `{repeatability['distinct_density_pattern_count']}`",
        f"- validated review input present: `{_bool_token(boundary['validated_review_input_present'])}`",
        f"- preference fill allowed: `{_bool_token(boundary['preference_fill_allowed'])}`",
        f"- human/audio preference claimed: `{_bool_token(boundary['human_audio_preference_claimed'])}`",
        f"- MIDI-to-solo musical quality claimed: `{_bool_token(boundary['midi_to_solo_musical_quality_claimed'])}`",
        "",
        "## Pending Review",
        "",
        f"- pending status fields: `{review['pending_status_field_count']}`",
        f"- pending candidate decisions: `{review['pending_candidate_decision_count']}`",
        f"- pending candidate fields: `{review['pending_candidate_field_count']}`",
        "",
        "## Proven",
        "",
    ]
    for item in report.get("proven", []):
        lines.append(f"- `{item}`")
    lines.extend(["", "## Not Proven", ""])
    for item in report.get("not_proven", []):
        lines.append(f"- `{item}`")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Decide repeatability objective-only next boundary")
    parser.add_argument(
        "--listening_review",
        type=str,
        default=(
            "outputs/stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_"
            "repeatability_listening_review/harness_stage_b_midi_to_solo_model_direct_jazz_phrase_"
            "vocabulary_contour_phrase_shape_repeatability_listening_review/"
            "stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_"
            "repeatability_listening_review.json"
        ),
    )
    parser.add_argument(
        "--repeatability_consolidation",
        type=str,
        default=(
            "outputs/stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_"
            "repeatability_consolidation/harness_stage_b_midi_to_solo_model_direct_jazz_phrase_"
            "vocabulary_contour_phrase_shape_repeatability_consolidation/"
            "stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_"
            "repeatability_consolidation.json"
        ),
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=(
            "outputs/stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_"
            "contour_phrase_shape_repeatability_objective_next"
        ),
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--expected_final_boundary", type=str, default="")
    parser.add_argument("--expected_next_boundary", type=str, default="")
    parser.add_argument("--min_sample_count", type=int, default=1)
    parser.add_argument("--min_qualified_count", type=int, default=1)
    parser.add_argument("--require_objective_support", action="store_true")
    parser.add_argument("--require_pending_review", action="store_true")
    parser.add_argument("--require_no_quality_claim", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    report = build_objective_next_decision_report(
        read_json(Path(args.listening_review)),
        read_json(Path(args.repeatability_consolidation)),
        output_dir=output_dir,
    )
    summary = validate_objective_next_decision_report(
        report,
        expected_final_boundary=str(args.expected_final_boundary or ""),
        expected_next_boundary=str(args.expected_next_boundary or ""),
        min_sample_count=int(args.min_sample_count),
        min_qualified_count=int(args.min_qualified_count),
        require_objective_support=bool(args.require_objective_support),
        require_pending_review=bool(args.require_pending_review),
        require_no_quality_claim=bool(args.require_no_quality_claim),
    )
    write_json(
        output_dir
        / "stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_repeatability_objective_next.json",
        report,
    )
    write_json(
        output_dir
        / "stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_repeatability_objective_next_validation_summary.json",
        summary,
    )
    markdown = markdown_report(report)
    write_text(
        output_dir
        / "stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_repeatability_objective_next.md",
        markdown,
    )
    if args.doc_path:
        write_text(Path(args.doc_path), markdown)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
