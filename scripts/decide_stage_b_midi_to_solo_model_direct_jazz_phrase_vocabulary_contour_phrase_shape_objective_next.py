"""Decide the objective-only next boundary after contour phrase-shape review setup."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from scripts.assess_stage_b_generic_base_readiness import write_json, write_text  # noqa: E402
from scripts.build_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_listening_review import (  # noqa: E402
    BOUNDARY as SOURCE_BOUNDARY,
)
from scripts.run_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_repair import (  # noqa: E402
    BOUNDARY as CONTOUR_REPAIR_BOUNDARY,
)


class StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeObjectiveNextError(ValueError):
    pass


BOUNDARY = (
    "stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_"
    "objective_only_next_decision"
)
NEXT_BOUNDARY = (
    "stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_"
    "objective_clean_repeatability_sweep"
)
SCHEMA_VERSION = (
    "stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_"
    "objective_next_v1"
)
NEXT_ACTIONS = [
    "preserve_objective_clean_candidate_boundary",
    "run_distinct_seed_repeatability_sweep",
    "keep_human_audio_preference_pending",
    "block_midi_to_solo_quality_claim",
]


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeObjectiveNextError(
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


def validate_listening_review_boundary(report: dict[str, Any]) -> list[dict[str, Any]]:
    boundary = _dict(report.get("listening_review_boundary"))
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    if str(boundary.get("boundary") or report.get("boundary") or "") != SOURCE_BOUNDARY:
        raise StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeObjectiveNextError(
            "contour phrase-shape listening review boundary required"
        )
    if str(decision.get("next_boundary") or "") != BOUNDARY:
        raise StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeObjectiveNextError(
            "listening review must route to contour phrase-shape objective-only decision"
        )
    if not bool(readiness.get("listening_review_boundary_prepared", False)):
        raise StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeObjectiveNextError(
            "listening review boundary preparation required"
        )
    if bool(readiness.get("validated_review_input_present", True)):
        raise StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeObjectiveNextError(
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
        raise StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeObjectiveNextError(
            f"unexpected upstream claim: {claimed}"
        )
    candidates = [_dict(item) for item in _list(report.get("review_candidates")) if isinstance(item, dict)]
    if len(candidates) < 1:
        raise StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeObjectiveNextError(
            "review candidates required"
        )
    return candidates


def validate_contour_repair_report(report: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if str(report.get("boundary") or "") != CONTOUR_REPAIR_BOUNDARY:
        raise StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeObjectiveNextError(
            "contour phrase-shape repair report required"
        )
    repair_result = _dict(report.get("repair_result"))
    aggregate = _dict(report.get("aggregate"))
    source = _dict(report.get("source_objective_summary"))
    readiness = _dict(report.get("readiness"))
    if not bool(repair_result.get("target_passed", False)):
        raise StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeObjectiveNextError(
            "contour phrase-shape repair target should pass before objective-clean decision"
        )
    if not bool(repair_result.get("stepwise_contour_bias_reduced", False)):
        raise StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeObjectiveNextError(
            "stepwise contour reduction evidence required"
        )
    if not bool(repair_result.get("no_overlap", False)):
        raise StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeObjectiveNextError(
            "no-overlap repair evidence required"
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
        raise StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeObjectiveNextError(
            f"unexpected repair claim: {claimed}"
        )
    return repair_result, aggregate, source


def objective_summary(
    candidates: list[dict[str, Any]],
    repair_result: dict[str, Any],
    aggregate: dict[str, Any],
    source: dict[str, Any],
) -> dict[str, Any]:
    flag_counter: Counter[str] = Counter()
    density_patterns: set[tuple[int, ...]] = set()
    note_counts: list[int] = []
    max_intervals: list[int] = []
    small_interval_ratios: list[float] = []
    for item in candidates:
        for flag in _list(item.get("analysis_flags")):
            flag_counter[str(flag)] += 1
        density_patterns.add(tuple(_int(value) for value in _list(item.get("density_pattern"))))
        note_counts.append(_int(item.get("note_count")))
        max_intervals.append(_int(item.get("max_abs_interval")))
        small_interval_ratios.append(_float(item.get("small_interval_ratio_le4")))
    current_analysis_flag_count = sum(flag_counter.values())
    candidate_count = len(candidates)
    repaired_stepwise_count = _int(aggregate.get("stepwise_contour_bias_count"))
    source_stepwise_count = _int(source.get("source_stepwise_contour_bias_count"))
    no_overlap = bool(repair_result.get("no_overlap", False))
    objective_clean_supported = (
        candidate_count > 0
        and current_analysis_flag_count == 0
        and repaired_stepwise_count == 0
        and no_overlap
        and bool(repair_result.get("target_passed", False))
    )
    return {
        "candidate_count": candidate_count,
        "flag_counts": dict(sorted(flag_counter.items())),
        "current_analysis_flag_count": current_analysis_flag_count,
        "source_stepwise_contour_bias_count": source_stepwise_count,
        "repaired_stepwise_contour_bias_count": repaired_stepwise_count,
        "stepwise_contour_bias_reduced": bool(repair_result.get("stepwise_contour_bias_reduced", False)),
        "objective_clean_candidate_boundary_supported": objective_clean_supported,
        "additional_repair_required": not objective_clean_supported,
        "distinct_density_pattern_count": len(density_patterns),
        "note_count_min": min(note_counts) if note_counts else 0,
        "note_count_max": max(note_counts) if note_counts else 0,
        "max_abs_interval_max": max(max_intervals) if max_intervals else 0,
        "max_small_interval_ratio_le4": max(small_interval_ratios) if small_interval_ratios else 0.0,
        "no_overlap": no_overlap,
        "shared_rhythm_signature_count": _int(aggregate.get("shared_rhythm_signature_count")),
    }


def build_objective_next_decision_report(
    listening_review: dict[str, Any],
    contour_repair: dict[str, Any],
    *,
    output_dir: Path,
) -> dict[str, Any]:
    candidates = validate_listening_review_boundary(listening_review)
    repair_result, aggregate, source = validate_contour_repair_report(contour_repair)
    objective = objective_summary(candidates, repair_result, aggregate, source)
    if bool(objective.get("additional_repair_required", True)):
        raise StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeObjectiveNextError(
            "objective-clean candidate boundary not supported"
        )
    boundary = {
        "boundary": BOUNDARY,
        "source_boundary": SOURCE_BOUNDARY,
        "repair_evidence_boundary": CONTOUR_REPAIR_BOUNDARY,
        "objective_only_decision_completed": True,
        "validated_review_input_present": False,
        "preference_fill_allowed": False,
        "selected_next_boundary": NEXT_BOUNDARY,
        "objective_clean_candidate_boundary_supported": True,
        "additional_repair_required": False,
        "source_stepwise_contour_bias_count": _int(objective.get("source_stepwise_contour_bias_count")),
        "repaired_stepwise_contour_bias_count": _int(objective.get("repaired_stepwise_contour_bias_count")),
        "current_analysis_flag_count": _int(objective.get("current_analysis_flag_count")),
        "candidate_count": _int(objective.get("candidate_count")),
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
        "repair_evidence_boundary": CONTOUR_REPAIR_BOUNDARY,
        "objective_summary": objective,
        "selected_next_actions": list(NEXT_ACTIONS),
        "objective_next_decision_boundary": boundary,
        "readiness": {
            "boundary": BOUNDARY,
            "objective_only_decision_completed": True,
            "objective_clean_candidate_boundary_supported": True,
            "additional_repair_required": False,
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
            "next_boundary": NEXT_BOUNDARY,
            "auto_progress_allowed": True,
            "critical_user_input_required": False,
            "reason": (
                "contour phrase-shape candidates have no objective analysis flags; "
                "human/audio review remains pending, so route to objective-clean repeatability sweep"
            ),
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
            "objective-clean repeatability sweep"
        ),
    }


def validate_objective_next_decision_report(
    report: dict[str, Any],
    *,
    expected_boundary: str | None,
    expected_next_boundary: str | None,
    require_objective_clean: bool,
    require_pending_review: bool,
    require_no_quality_claim: bool,
) -> dict[str, Any]:
    boundary = _dict(report.get("objective_next_decision_boundary"))
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    objective = _dict(report.get("objective_summary"))
    if expected_boundary and str(boundary.get("boundary") or "") != expected_boundary:
        raise StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeObjectiveNextError(
            f"expected boundary {expected_boundary}, got {boundary.get('boundary')}"
        )
    if expected_next_boundary and str(decision.get("next_boundary") or "") != expected_next_boundary:
        raise StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeObjectiveNextError(
            "unexpected next boundary"
        )
    if require_objective_clean:
        if not bool(readiness.get("objective_clean_candidate_boundary_supported", False)):
            raise StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeObjectiveNextError(
                "objective-clean boundary support required"
            )
        if bool(readiness.get("additional_repair_required", True)):
            raise StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeObjectiveNextError(
                "additional repair should be false"
            )
    if require_pending_review and bool(readiness.get("validated_review_input_present", True)):
        raise StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeObjectiveNextError(
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
            raise StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeObjectiveNextError(
                f"unexpected quality claim: {claimed}"
            )
    return {
        "boundary": str(boundary.get("boundary") or ""),
        "source_boundary": str(boundary.get("source_boundary") or ""),
        "repair_evidence_boundary": str(boundary.get("repair_evidence_boundary") or ""),
        "candidate_count": _int(objective.get("candidate_count")),
        "current_analysis_flag_count": _int(objective.get("current_analysis_flag_count")),
        "source_stepwise_contour_bias_count": _int(objective.get("source_stepwise_contour_bias_count")),
        "repaired_stepwise_contour_bias_count": _int(objective.get("repaired_stepwise_contour_bias_count")),
        "stepwise_contour_bias_reduced": bool(objective.get("stepwise_contour_bias_reduced", False)),
        "objective_clean_candidate_boundary_supported": bool(
            objective.get("objective_clean_candidate_boundary_supported", False)
        ),
        "additional_repair_required": bool(objective.get("additional_repair_required", True)),
        "distinct_density_pattern_count": _int(objective.get("distinct_density_pattern_count")),
        "max_abs_interval_max": _int(objective.get("max_abs_interval_max")),
        "max_small_interval_ratio_le4": _float(objective.get("max_small_interval_ratio_le4")),
        "no_overlap": bool(objective.get("no_overlap", False)),
        "selected_next_actions": [str(item) for item in _list(report.get("selected_next_actions"))],
        "validated_review_input_present": bool(readiness.get("validated_review_input_present", True)),
        "preference_fill_allowed": bool(readiness.get("preference_fill_allowed", True)),
        "human_audio_preference_claimed": bool(readiness.get("human_audio_preference_claimed", True)),
        "midi_to_solo_musical_quality_claimed": bool(
            readiness.get("midi_to_solo_musical_quality_claimed", True)
        ),
        "critical_user_input_required": bool(decision.get("critical_user_input_required", True)),
        "next_boundary": str(decision.get("next_boundary") or ""),
        "next_recommended_issue": str(report.get("next_recommended_issue") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    boundary = report["objective_next_decision_boundary"]
    decision = report["decision"]
    objective = report["objective_summary"]
    lines = [
        "# Stage B MIDI-to-Solo Model-Direct Jazz Phrase Vocabulary Contour Phrase-Shape Objective-Only Next Decision",
        "",
        "## Summary",
        "",
        f"- boundary: `{boundary['boundary']}`",
        f"- source boundary: `{boundary['source_boundary']}`",
        f"- repair evidence boundary: `{boundary['repair_evidence_boundary']}`",
        f"- next boundary: `{decision['next_boundary']}`",
        f"- objective-only decision completed: `{_bool_token(boundary['objective_only_decision_completed'])}`",
        f"- objective-clean candidate boundary supported: `{_bool_token(boundary['objective_clean_candidate_boundary_supported'])}`",
        f"- additional repair required: `{_bool_token(boundary['additional_repair_required'])}`",
        f"- candidate count: `{objective['candidate_count']}`",
        f"- current analysis flag count: `{objective['current_analysis_flag_count']}`",
        f"- source stepwise contour bias count: `{objective['source_stepwise_contour_bias_count']}`",
        f"- repaired stepwise contour bias count: `{objective['repaired_stepwise_contour_bias_count']}`",
        f"- max abs interval max: `{objective['max_abs_interval_max']}`",
        f"- max small interval ratio <=4: `{objective['max_small_interval_ratio_le4']:.4f}`",
        f"- no overlap: `{_bool_token(objective['no_overlap'])}`",
        f"- validated review input present: `{_bool_token(boundary['validated_review_input_present'])}`",
        f"- preference fill allowed: `{_bool_token(boundary['preference_fill_allowed'])}`",
        f"- human/audio preference claimed: `{_bool_token(boundary['human_audio_preference_claimed'])}`",
        f"- MIDI-to-solo musical quality claimed: `{_bool_token(boundary['midi_to_solo_musical_quality_claimed'])}`",
        "",
        "## Selected Next Actions",
        "",
    ]
    for action in report.get("selected_next_actions", []):
        lines.append(f"- `{action}`")
    lines.extend(["", "## Not Proven", ""])
    for item in report.get("not_proven", []):
        lines.append(f"- `{item}`")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Decide contour phrase-shape objective-only next boundary")
    parser.add_argument("--listening_review", type=str, required=True)
    parser.add_argument("--contour_repair", type=str, required=True)
    parser.add_argument(
        "--output_root",
        type=str,
        default=(
            "outputs/stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_"
            "contour_phrase_shape_objective_next"
        ),
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--expected_boundary", type=str, default="")
    parser.add_argument("--expected_next_boundary", type=str, default="")
    parser.add_argument("--require_objective_clean", action="store_true")
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
        read_json(Path(args.contour_repair)),
        output_dir=output_dir,
    )
    summary = validate_objective_next_decision_report(
        report,
        expected_boundary=str(args.expected_boundary or ""),
        expected_next_boundary=str(args.expected_next_boundary or ""),
        require_objective_clean=bool(args.require_objective_clean),
        require_pending_review=bool(args.require_pending_review),
        require_no_quality_claim=bool(args.require_no_quality_claim),
    )
    write_json(
        output_dir / "stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_objective_next.json",
        report,
    )
    write_json(
        output_dir
        / "stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_objective_next_validation_summary.json",
        summary,
    )
    markdown = markdown_report(report)
    write_text(
        output_dir / "stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_objective_next.md",
        markdown,
    )
    if args.doc_path:
        write_text(Path(args.doc_path), markdown)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
