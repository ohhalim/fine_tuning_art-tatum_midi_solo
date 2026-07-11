"""Decide the next objective-only repair after pending jazz phrase listening review."""

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
from scripts.build_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_listening_review import (  # noqa: E402
    BOUNDARY as SOURCE_BOUNDARY,
)


class StageBMidiToSoloModelDirectJazzPhraseVocabularyRepairObjectiveNextError(ValueError):
    pass


BOUNDARY = "stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_objective_only_next_decision"
NEXT_BOUNDARY = "stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_repair"
SCHEMA_VERSION = "stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_objective_next_v1"
REPAIR_TARGETS = [
    "reduce_stepwise_contour_bias",
    "add_phrase_shape_tension_release",
    "add_approach_enclosure_cells",
    "preserve_density_variation",
    "preserve_interval_guard",
    "preserve_no_quality_claim",
]


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise StageBMidiToSoloModelDirectJazzPhraseVocabularyRepairObjectiveNextError(f"report missing: {path}")
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
        raise StageBMidiToSoloModelDirectJazzPhraseVocabularyRepairObjectiveNextError(
            "listening review boundary required"
        )
    if str(decision.get("next_boundary") or "") != BOUNDARY:
        raise StageBMidiToSoloModelDirectJazzPhraseVocabularyRepairObjectiveNextError(
            "listening review must route to objective-only decision"
        )
    if not bool(readiness.get("listening_review_boundary_prepared", False)):
        raise StageBMidiToSoloModelDirectJazzPhraseVocabularyRepairObjectiveNextError(
            "listening review boundary preparation required"
        )
    if bool(readiness.get("validated_review_input_present", True)):
        raise StageBMidiToSoloModelDirectJazzPhraseVocabularyRepairObjectiveNextError(
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
        raise StageBMidiToSoloModelDirectJazzPhraseVocabularyRepairObjectiveNextError(
            f"unexpected upstream claim: {claimed}"
        )
    candidates = [_dict(item) for item in _list(report.get("review_candidates")) if isinstance(item, dict)]
    if len(candidates) < 1:
        raise StageBMidiToSoloModelDirectJazzPhraseVocabularyRepairObjectiveNextError(
            "review candidates required"
        )
    return candidates


def objective_summary(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    flag_counter: Counter[str] = Counter()
    density_patterns: set[tuple[int, ...]] = set()
    note_counts: list[int] = []
    max_intervals: list[int] = []
    duration_ratios: list[float] = []
    ioi_ratios: list[float] = []
    for item in candidates:
        for flag in _list(item.get("analysis_flags")):
            flag_counter[str(flag)] += 1
        density_patterns.add(tuple(_int(value) for value in _list(item.get("density_pattern"))))
        note_counts.append(_int(item.get("note_count")))
        max_intervals.append(_int(item.get("max_abs_interval")))
        duration_ratios.append(_float(item.get("duration_most_common_ratio")))
        ioi_ratios.append(_float(item.get("ioi_most_common_ratio")))
    candidate_count = len(candidates)
    return {
        "candidate_count": candidate_count,
        "flag_counts": dict(sorted(flag_counter.items())),
        "stepwise_contour_bias_count": int(flag_counter.get("stepwise_contour_bias", 0)),
        "all_candidates_stepwise_biased": int(flag_counter.get("stepwise_contour_bias", 0)) == candidate_count,
        "distinct_density_pattern_count": len(density_patterns),
        "fixed_density_count": 0 if len(density_patterns) == candidate_count else candidate_count - len(density_patterns),
        "note_count_min": min(note_counts) if note_counts else 0,
        "note_count_max": max(note_counts) if note_counts else 0,
        "max_abs_interval_max": max(max_intervals) if max_intervals else 0,
        "duration_most_common_ratio_max": max(duration_ratios) if duration_ratios else 0.0,
        "ioi_most_common_ratio_max": max(ioi_ratios) if ioi_ratios else 0.0,
    }


def build_objective_next_decision_report(
    listening_review: dict[str, Any],
    *,
    output_dir: Path,
) -> dict[str, Any]:
    candidates = validate_listening_review_boundary(listening_review)
    objective = objective_summary(candidates)
    selected_targets = list(REPAIR_TARGETS)
    boundary = {
        "boundary": BOUNDARY,
        "source_boundary": SOURCE_BOUNDARY,
        "objective_only_decision_completed": True,
        "validated_review_input_present": False,
        "preference_fill_allowed": False,
        "selected_next_boundary": NEXT_BOUNDARY,
        "selected_repair_target_count": len(selected_targets),
        "stepwise_contour_bias_count": _int(objective.get("stepwise_contour_bias_count")),
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
        "objective_summary": objective,
        "selected_repair_targets": selected_targets,
        "objective_next_decision_boundary": boundary,
        "readiness": {
            "boundary": BOUNDARY,
            "objective_only_decision_completed": True,
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
            "reason": "pending review blocks preference fill; objective flags route to contour and phrase-shape repair",
        },
        "not_proven": [
            "listening_review_completed",
            "human_audio_preference",
            "model_direct_generation_quality",
            "midi_to_solo_musical_quality",
            "broad_trained_model_quality",
            "brad_style_adaptation",
        ],
        "next_recommended_issue": "Stage B MIDI-to-solo model-direct jazz phrase vocabulary contour phrase-shape repair",
    }


def validate_objective_next_decision_report(
    report: dict[str, Any],
    *,
    expected_boundary: str | None,
    expected_next_boundary: str | None,
    min_repair_target_count: int,
    require_stepwise_target: bool,
    require_pending_review: bool,
    require_no_quality_claim: bool,
) -> dict[str, Any]:
    boundary = _dict(report.get("objective_next_decision_boundary"))
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    targets = [str(item) for item in _list(report.get("selected_repair_targets"))]
    if expected_boundary and str(boundary.get("boundary") or "") != expected_boundary:
        raise StageBMidiToSoloModelDirectJazzPhraseVocabularyRepairObjectiveNextError(
            f"expected boundary {expected_boundary}, got {boundary.get('boundary')}"
        )
    if expected_next_boundary and str(decision.get("next_boundary") or "") != expected_next_boundary:
        raise StageBMidiToSoloModelDirectJazzPhraseVocabularyRepairObjectiveNextError(
            "unexpected next boundary"
        )
    if len(targets) < int(min_repair_target_count):
        raise StageBMidiToSoloModelDirectJazzPhraseVocabularyRepairObjectiveNextError(
            "not enough selected repair targets"
        )
    if require_stepwise_target and "reduce_stepwise_contour_bias" not in targets:
        raise StageBMidiToSoloModelDirectJazzPhraseVocabularyRepairObjectiveNextError(
            "stepwise contour repair target required"
        )
    if require_pending_review and bool(readiness.get("validated_review_input_present", True)):
        raise StageBMidiToSoloModelDirectJazzPhraseVocabularyRepairObjectiveNextError(
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
            raise StageBMidiToSoloModelDirectJazzPhraseVocabularyRepairObjectiveNextError(
                f"unexpected quality claim: {claimed}"
            )
    objective = _dict(report.get("objective_summary"))
    return {
        "boundary": str(boundary.get("boundary") or ""),
        "source_boundary": str(boundary.get("source_boundary") or ""),
        "candidate_count": _int(objective.get("candidate_count")),
        "stepwise_contour_bias_count": _int(objective.get("stepwise_contour_bias_count")),
        "all_candidates_stepwise_biased": bool(objective.get("all_candidates_stepwise_biased", False)),
        "distinct_density_pattern_count": _int(objective.get("distinct_density_pattern_count")),
        "max_abs_interval_max": _int(objective.get("max_abs_interval_max")),
        "selected_repair_target_count": len(targets),
        "selected_repair_targets": targets,
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
        "# Stage B MIDI-to-Solo Model-Direct Jazz Phrase Vocabulary Repair Objective-Only Next Decision",
        "",
        "## Summary",
        "",
        f"- boundary: `{boundary['boundary']}`",
        f"- source boundary: `{boundary['source_boundary']}`",
        f"- next boundary: `{decision['next_boundary']}`",
        f"- objective-only decision completed: `{_bool_token(boundary['objective_only_decision_completed'])}`",
        f"- candidate count: `{objective['candidate_count']}`",
        f"- stepwise contour bias count: `{objective['stepwise_contour_bias_count']}`",
        f"- distinct density pattern count: `{objective['distinct_density_pattern_count']}`",
        f"- max abs interval max: `{objective['max_abs_interval_max']}`",
        f"- validated review input present: `{_bool_token(boundary['validated_review_input_present'])}`",
        f"- preference fill allowed: `{_bool_token(boundary['preference_fill_allowed'])}`",
        f"- human/audio preference claimed: `{_bool_token(boundary['human_audio_preference_claimed'])}`",
        f"- MIDI-to-solo musical quality claimed: `{_bool_token(boundary['midi_to_solo_musical_quality_claimed'])}`",
        "",
        "## Selected Repair Targets",
        "",
    ]
    for target in report.get("selected_repair_targets", []):
        lines.append(f"- `{target}`")
    lines.extend(["", "## Not Proven", ""])
    for item in report.get("not_proven", []):
        lines.append(f"- `{item}`")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Decide objective-only next repair")
    parser.add_argument("--listening_review", type=str, required=True)
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_objective_next",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--expected_boundary", type=str, default="")
    parser.add_argument("--expected_next_boundary", type=str, default="")
    parser.add_argument("--min_repair_target_count", type=int, default=6)
    parser.add_argument("--require_stepwise_target", action="store_true")
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
        output_dir=output_dir,
    )
    summary = validate_objective_next_decision_report(
        report,
        expected_boundary=str(args.expected_boundary or ""),
        expected_next_boundary=str(args.expected_next_boundary or ""),
        min_repair_target_count=int(args.min_repair_target_count),
        require_stepwise_target=bool(args.require_stepwise_target),
        require_pending_review=bool(args.require_pending_review),
        require_no_quality_claim=bool(args.require_no_quality_claim),
    )
    write_json(output_dir / "stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_objective_next.json", report)
    write_json(
        output_dir
        / "stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_objective_next_validation_summary.json",
        summary,
    )
    markdown = markdown_report(report)
    write_text(output_dir / "stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_objective_next.md", markdown)
    if args.doc_path:
        write_text(Path(args.doc_path), markdown)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
