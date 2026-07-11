from __future__ import annotations

import unittest
from pathlib import Path

from scripts.decide_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_model_core_review import (
    BOUNDARY,
    NEXT_BOUNDARY,
    SOURCE_BOUNDARY,
    StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardSparsePhraseModelCoreReviewError,
    build_model_core_review_decision,
    validate_model_core_review_decision,
)


def sparse_rejection_analysis(*, quality_claimed: bool = False, proxy_gap: bool = True) -> dict:
    return {
        "schema_version": (
            "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_"
            "sparse_phrase_rejection_analysis_v1"
        ),
        "analysis_boundary": {
            "boundary": SOURCE_BOUNDARY,
            "input_reject_all_verified": True,
            "analyzed_candidate_count": 3,
            "human_audio_keep_claimed": False,
            "human_audio_preference_claimed": False,
            "musical_quality_claimed": quality_claimed,
            "quality_cause_claimed": False,
            "objective_proxy_gap_recorded": proxy_gap,
        },
        "rejection_analysis": {
            "candidate_count": 3,
            "evidence_flag_counts": {
                "compressed_pitch_vocabulary": 2,
                "repetitive_duration_profile": 2,
            },
            "common_evidence_flags": [],
            "candidates_without_evidence_flags": [1] if proxy_gap else [],
            "objective_proxy_gap": {
                "recorded": proxy_gap,
                "all_candidates_rejected_by_listening_review": True,
                "candidate_without_flag_count": 1 if proxy_gap else 0,
                "interpretation": "objective_midi_proxy_not_sufficient_for_listening_acceptance",
            },
            "primary_next_review_target": "model_core_review_after_objective_proxy_gap",
            "cause_claim": "not_claimed",
        },
        "candidates": [{"review_rank": 1}, {"review_rank": 2}, {"review_rank": 3}],
        "decision": {
            "next_boundary": BOUNDARY,
            "auto_progress_allowed": True,
            "critical_user_input_required": False,
        },
    }


class StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardSparsePhraseModelCoreReviewTest(
    unittest.TestCase
):
    def test_stops_repair_loop_and_routes_to_model_core_plan(self) -> None:
        report = build_model_core_review_decision(
            sparse_rejection_analysis(),
            output_dir=Path("outputs/model_core_review"),
        )
        summary = validate_model_core_review_decision(
            report,
            expected_boundary=BOUNDARY,
            expected_next_boundary=NEXT_BOUNDARY,
            require_stop_repair_loop=True,
            require_diagnostic_only=True,
            require_no_quality_claim=True,
        )

        self.assertEqual(summary["decision"], "stop_constraint_postprocess_repair_loop")
        self.assertFalse(summary["continue_constraint_postprocess_repair_loop"])
        self.assertEqual(summary["tiny_checkpoint_role"], "diagnostic_only")
        self.assertTrue(summary["model_core_transition_required"])
        self.assertTrue(summary["objective_proxy_gap_recorded"])
        self.assertEqual(summary["candidate_without_objective_flag_count"], 1)
        self.assertFalse(summary["musical_quality_claimed"])
        self.assertEqual(summary["next_boundary"], NEXT_BOUNDARY)

    def test_rejects_quality_claim_source(self) -> None:
        with self.assertRaises(
            StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardSparsePhraseModelCoreReviewError
        ):
            build_model_core_review_decision(
                sparse_rejection_analysis(quality_claimed=True),
                output_dir=Path("outputs/model_core_review"),
            )

    def test_rejects_absent_proxy_gap(self) -> None:
        with self.assertRaises(
            StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardSparsePhraseModelCoreReviewError
        ):
            build_model_core_review_decision(
                sparse_rejection_analysis(proxy_gap=False),
                output_dir=Path("outputs/model_core_review"),
            )


if __name__ == "__main__":
    unittest.main()
