from __future__ import annotations

import unittest
from pathlib import Path

from scripts.decide_stage_b_duration_coverage_outside_soloing_repair_final_decision import (
    StageBDurationCoverageOutsideSoloingRepairFinalDecisionError,
    build_final_decision,
    validate_final_decision,
)


def repeatability_consolidation(
    *,
    boundary: str = "outside_soloing_repair_objective_repeatability_support",
    review_input_present: bool = False,
    broad_claim: bool = False,
) -> dict:
    return {
        "schema_version": "stage_b_duration_coverage_fill_outside_soloing_repair_repeatability_consolidation_v1",
        "selected_source_objective_support": {
            "source_candidate_count": 2,
            "qualified_source_candidate_count": 2,
            "dead_air_preserved_source_candidate_count": 2,
        },
        "policy_repeatability_support": {
            "supported_repair_policy_count": 3,
            "total_variant_count": 6,
            "total_qualified_variant_count": 6,
            "selected_min_chord_tone_ratio": 1.0,
            "selected_max_non_chord_tone_run": 0,
            "selected_max_interval": 7,
        },
        "pending_user_review": {
            "boundary": "outside_soloing_repair_audio_review_pending",
            "review_input_present": review_input_present,
            "human_audio_preference_claimed": False,
        },
        "consolidated_claim_boundary": {
            "boundary": boundary,
            "objective_repair_repeatability_claimed": True,
            "human_audio_preference_claimed": False,
            "multi_reviewer_preference_claimed": False,
            "broad_model_quality_claimed": broad_claim,
            "brad_style_adaptation_claimed": False,
            "production_ready_improviser_claimed": False,
        },
    }


class StageBDurationCoverageOutsideSoloingRepairFinalDecisionTest(unittest.TestCase):
    def test_records_objective_path_complete_and_next_readme_boundary(self) -> None:
        report = build_final_decision(
            repeatability_consolidation(),
            output_dir=Path("outputs/final_decision"),
        )
        summary = validate_final_decision(
            report,
            expected_final_boundary="outside_soloing_repair_objective_path_complete",
            expected_next_boundary="stage_b_model_core_evidence_readme_refresh",
            require_auto_progress_allowed=True,
            require_no_critical_user_input=True,
            require_no_preference_claim=True,
            require_no_broad_quality_claim=True,
        )

        self.assertEqual(summary["input_boundary"], "outside_soloing_repair_objective_repeatability_support")
        self.assertEqual(summary["next_boundary"], "stage_b_model_core_evidence_readme_refresh")
        self.assertTrue(summary["auto_progress_allowed"])
        self.assertFalse(summary["critical_user_input_required"])
        self.assertFalse(summary["review_input_present"])
        self.assertFalse(summary["human_audio_preference_claimed"])
        self.assertFalse(summary["broad_model_quality_claimed"])

    def test_rejects_incomplete_repeatability_boundary(self) -> None:
        with self.assertRaises(StageBDurationCoverageOutsideSoloingRepairFinalDecisionError):
            build_final_decision(
                repeatability_consolidation(boundary="outside_soloing_repair_objective_repeatability_incomplete"),
                output_dir=Path("outputs/final_decision"),
            )

    def test_rejects_review_input_for_objective_only_final_decision(self) -> None:
        with self.assertRaises(StageBDurationCoverageOutsideSoloingRepairFinalDecisionError):
            build_final_decision(
                repeatability_consolidation(review_input_present=True),
                output_dir=Path("outputs/final_decision"),
            )

    def test_rejects_broad_quality_claim(self) -> None:
        with self.assertRaises(StageBDurationCoverageOutsideSoloingRepairFinalDecisionError):
            build_final_decision(
                repeatability_consolidation(broad_claim=True),
                output_dir=Path("outputs/final_decision"),
            )


if __name__ == "__main__":
    unittest.main()
