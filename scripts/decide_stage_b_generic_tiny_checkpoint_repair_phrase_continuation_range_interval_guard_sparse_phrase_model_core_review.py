"""Decide model-core transition after sparse phrase repair rejection analysis."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardSparsePhraseModelCoreReviewError(
    ValueError
):
    pass


SOURCE_BOUNDARY = (
    "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_"
    "sparse_phrase_rejection_analysis"
)
BOUNDARY = (
    "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_"
    "sparse_phrase_model_core_review_decision"
)
NEXT_BOUNDARY = "stage_b_generic_model_core_training_data_plan"
EXPECTED_TARGET = "model_core_review_after_objective_proxy_gap"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _bool_token(value: Any) -> str:
    return "true" if bool(value) else "false"


def validate_sparse_rejection_analysis(report: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], list[Any]]:
    boundary = _dict(report.get("analysis_boundary"))
    rejection = _dict(report.get("rejection_analysis"))
    proxy_gap = _dict(rejection.get("objective_proxy_gap"))
    candidates = _list(report.get("candidates"))
    decision = _dict(report.get("decision"))

    if str(boundary.get("boundary") or "") != SOURCE_BOUNDARY:
        raise StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardSparsePhraseModelCoreReviewError(
            "sparse phrase rejection analysis boundary required"
        )
    if not bool(boundary.get("input_reject_all_verified", False)):
        raise StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardSparsePhraseModelCoreReviewError(
            "reject_all source verification required"
        )
    if not bool(boundary.get("objective_proxy_gap_recorded", False)):
        raise StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardSparsePhraseModelCoreReviewError(
            "objective proxy gap must be recorded"
        )
    if not bool(proxy_gap.get("recorded", False)):
        raise StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardSparsePhraseModelCoreReviewError(
            "objective proxy gap detail must be recorded"
        )
    if not _list(rejection.get("candidates_without_evidence_flags")):
        raise StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardSparsePhraseModelCoreReviewError(
            "at least one candidate without objective evidence flags required"
        )
    if str(rejection.get("primary_next_review_target") or "") != EXPECTED_TARGET:
        raise StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardSparsePhraseModelCoreReviewError(
            "model-core review target required"
        )
    if str(decision.get("next_boundary") or "") != BOUNDARY:
        raise StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardSparsePhraseModelCoreReviewError(
            "source report must route to model-core review decision"
        )
    if bool(boundary.get("musical_quality_claimed", True)) or bool(boundary.get("quality_cause_claimed", True)):
        raise StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardSparsePhraseModelCoreReviewError(
            "source report must not claim musical quality or quality cause"
        )
    if len(candidates) < 1:
        raise StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardSparsePhraseModelCoreReviewError(
            "candidate evidence required"
        )
    return boundary, rejection, candidates


def build_model_core_review_decision(
    sparse_rejection_analysis: dict[str, Any],
    *,
    output_dir: Path,
) -> dict[str, Any]:
    boundary, rejection, candidates = validate_sparse_rejection_analysis(sparse_rejection_analysis)
    without_flags = [int(rank) for rank in _list(rejection.get("candidates_without_evidence_flags"))]
    common_flags = [str(flag) for flag in _list(rejection.get("common_evidence_flags"))]
    return {
        "schema_version": (
            "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_"
            "sparse_phrase_model_core_review_decision_v1"
        ),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "source_schema": str(sparse_rejection_analysis.get("schema_version") or ""),
        "input_boundary": SOURCE_BOUNDARY,
        "methodology_evidence": {
            "single_user_reject_all_candidate_count": int(boundary.get("analyzed_candidate_count", len(candidates)) or 0),
            "common_objective_evidence_flags": common_flags,
            "candidate_without_objective_flag_count": len(without_flags),
            "candidates_without_objective_flags": without_flags,
            "objective_proxy_gap_recorded": True,
            "objective_proxy_gap_interpretation": (
                "objective_midi_proxy_not_sufficient_for_listening_acceptance"
            ),
            "constraint_repairs_already_checked": [
                "grammar_repair",
                "phrase_continuation",
                "range_interval_guard",
                "sparse_phrase_objective_gate",
            ],
        },
        "decision": {
            "current_boundary": BOUNDARY,
            "decision": "stop_constraint_postprocess_repair_loop",
            "continue_constraint_postprocess_repair_loop": False,
            "tiny_checkpoint_role": "diagnostic_only",
            "model_core_transition_required": True,
            "next_boundary": NEXT_BOUNDARY,
            "auto_progress_allowed": True,
            "critical_user_input_required": False,
            "reason": (
                "single-user listening rejected all sparse phrase candidates, and objective MIDI flags do not "
                "consistently explain the rejection; further rule repairs should not be treated as model-quality progress"
            ),
        },
        "claim_boundary": {
            "boundary": BOUNDARY,
            "constraint_repair_loop_stop_claimed": True,
            "tiny_checkpoint_diagnostic_only_claimed": True,
            "musical_quality_claimed": False,
            "quality_root_cause_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
            "production_ready_improviser_claimed": False,
        },
        "proven": [
            "single_user_reject_all_consumed",
            "objective_proxy_gap_recorded",
            "constraint_postprocess_repair_loop_not_selected_for_next_step",
            "tiny_checkpoint_marked_diagnostic_only",
        ],
        "not_proven": [
            "musical_quality",
            "quality_root_cause",
            "broad_trained_model_quality",
            "brad_style_adaptation",
            "production_ready_improviser",
        ],
        "next_recommended_issue": "Stage B generic model-core training data plan",
    }


def validate_model_core_review_decision(
    report: dict[str, Any],
    *,
    expected_boundary: str | None,
    expected_next_boundary: str | None,
    require_stop_repair_loop: bool,
    require_diagnostic_only: bool,
    require_no_quality_claim: bool,
) -> dict[str, Any]:
    decision = _dict(report.get("decision"))
    claim = _dict(report.get("claim_boundary"))
    evidence = _dict(report.get("methodology_evidence"))
    boundary = str(decision.get("current_boundary") or "")
    next_boundary = str(decision.get("next_boundary") or "")

    if expected_boundary and boundary != expected_boundary:
        raise StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardSparsePhraseModelCoreReviewError(
            f"expected boundary {expected_boundary}, got {boundary}"
        )
    if expected_next_boundary and next_boundary != expected_next_boundary:
        raise StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardSparsePhraseModelCoreReviewError(
            f"expected next boundary {expected_next_boundary}, got {next_boundary}"
        )
    if require_stop_repair_loop and bool(decision.get("continue_constraint_postprocess_repair_loop", True)):
        raise StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardSparsePhraseModelCoreReviewError(
            "constraint/postprocess repair loop must stop"
        )
    if require_diagnostic_only and str(decision.get("tiny_checkpoint_role") or "") != "diagnostic_only":
        raise StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardSparsePhraseModelCoreReviewError(
            "tiny checkpoint must be diagnostic-only"
        )
    if require_no_quality_claim:
        claimed = [
            bool(claim.get("musical_quality_claimed", True)),
            bool(claim.get("quality_root_cause_claimed", True)),
            bool(claim.get("broad_trained_model_quality_claimed", True)),
            bool(claim.get("brad_style_adaptation_claimed", True)),
            bool(claim.get("production_ready_improviser_claimed", True)),
        ]
        if any(claimed):
            raise StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardSparsePhraseModelCoreReviewError(
                "quality claims must not be set"
            )
    return {
        "boundary": boundary,
        "input_boundary": str(report.get("input_boundary") or ""),
        "decision": str(decision.get("decision") or ""),
        "continue_constraint_postprocess_repair_loop": bool(
            decision.get("continue_constraint_postprocess_repair_loop", True)
        ),
        "tiny_checkpoint_role": str(decision.get("tiny_checkpoint_role") or ""),
        "model_core_transition_required": bool(decision.get("model_core_transition_required", False)),
        "objective_proxy_gap_recorded": bool(evidence.get("objective_proxy_gap_recorded", False)),
        "candidate_without_objective_flag_count": int(
            evidence.get("candidate_without_objective_flag_count", 0) or 0
        ),
        "musical_quality_claimed": bool(claim.get("musical_quality_claimed", True)),
        "broad_trained_model_quality_claimed": bool(claim.get("broad_trained_model_quality_claimed", True)),
        "auto_progress_allowed": bool(decision.get("auto_progress_allowed", False)),
        "critical_user_input_required": bool(decision.get("critical_user_input_required", True)),
        "next_boundary": next_boundary,
        "next_recommended_issue": str(report.get("next_recommended_issue") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    evidence = report["methodology_evidence"]
    decision = report["decision"]
    claim = report["claim_boundary"]
    lines = [
        "# Stage B Generic Tiny Checkpoint Repair Phrase Continuation Range Interval Guard Sparse Phrase Model Core Review Decision",
        "",
        "## Summary",
        "",
        f"- boundary: `{decision['current_boundary']}`",
        f"- decision: `{decision['decision']}`",
        f"- continue constraint/postprocess repair loop: `{_bool_token(decision['continue_constraint_postprocess_repair_loop'])}`",
        f"- tiny checkpoint role: `{decision['tiny_checkpoint_role']}`",
        f"- model core transition required: `{_bool_token(decision['model_core_transition_required'])}`",
        f"- objective proxy gap recorded: `{_bool_token(evidence['objective_proxy_gap_recorded'])}`",
        f"- candidate without objective flags: `{evidence['candidate_without_objective_flag_count']}`",
        f"- musical quality claimed: `{_bool_token(claim['musical_quality_claimed'])}`",
        f"- broad trained model quality claimed: `{_bool_token(claim['broad_trained_model_quality_claimed'])}`",
        f"- next boundary: `{decision['next_boundary']}`",
        "",
        "## Evidence",
        "",
        f"- single-user reject_all candidate count: `{evidence['single_user_reject_all_candidate_count']}`",
        f"- common objective evidence flags: `{', '.join(evidence['common_objective_evidence_flags'])}`",
        f"- candidates without objective flags: `{', '.join(str(rank) for rank in evidence['candidates_without_objective_flags'])}`",
        "",
        "## Not Proven",
        "",
    ]
    for item in report.get("not_proven", []):
        lines.append(f"- `{item}`")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Decide model-core transition after sparse phrase rejection analysis")
    parser.add_argument(
        "--sparse_rejection_analysis",
        type=str,
        default="outputs/stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_"
        "sparse_phrase_rejection_analysis/"
        "harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_"
        "sparse_phrase_rejection_analysis/"
        "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_"
        "sparse_phrase_rejection_analysis.json",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=(
            "outputs/"
            "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_"
            "sparse_phrase_model_core_review_decision"
        ),
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--expected_boundary", type=str, default=BOUNDARY)
    parser.add_argument("--expected_next_boundary", type=str, default=NEXT_BOUNDARY)
    parser.add_argument("--require_stop_repair_loop", action="store_true")
    parser.add_argument("--require_diagnostic_only", action="store_true")
    parser.add_argument("--require_no_quality_claim", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    report = build_model_core_review_decision(
        read_json(Path(args.sparse_rejection_analysis)),
        output_dir=output_dir,
    )
    summary = validate_model_core_review_decision(
        report,
        expected_boundary=str(args.expected_boundary or ""),
        expected_next_boundary=str(args.expected_next_boundary or ""),
        require_stop_repair_loop=bool(args.require_stop_repair_loop),
        require_diagnostic_only=bool(args.require_diagnostic_only),
        require_no_quality_claim=bool(args.require_no_quality_claim),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        output_dir
        / "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_model_core_review_decision.json",
        report,
    )
    write_json(
        output_dir
        / "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_model_core_review_decision_validation_summary.json",
        summary,
    )
    markdown = markdown_report(report)
    write_text(
        output_dir
        / "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_model_core_review_decision.md",
        markdown,
    )
    if args.doc_path:
        write_text(Path(args.doc_path), markdown)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
