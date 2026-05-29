from __future__ import annotations

import unittest
from pathlib import Path

from scripts.summarize_stage_b_duration_coverage_fill_outside_soloing_repair_repeatability_consolidation import (
    StageBDurationCoverageOutsideSoloingRepairRepeatabilityConsolidationError,
    build_repeatability_consolidation_report,
    validate_repeatability_consolidation,
)


def objective_evidence(*, broad_claim: bool = False) -> dict:
    return {
        "schema_version": "stage_b_duration_coverage_fill_outside_soloing_repair_objective_evidence_consolidation_v1",
        "objective_evidence_summary": {
            "boundary": "outside_soloing_repair_objective_evidence_support",
            "source_candidate_count": 2,
            "qualified_source_candidate_count": 2,
            "dead_air_preserved_source_candidate_count": 2,
            "chord_tone_pass_source_candidate_count": 2,
            "non_chord_run_pass_source_candidate_count": 2,
            "interval_pass_source_candidate_count": 2,
            "selected_min_chord_tone_ratio": 1.0,
            "selected_max_non_chord_tone_run": 0,
            "selected_max_interval": 7,
            "human_audio_preference_claimed": False,
            "broad_model_quality_claimed": False,
        },
        "claim_boundary": {
            "boundary": "outside_soloing_repair_objective_evidence_support",
            "objective_midi_evidence_claimed": True,
            "human_audio_preference_claimed": False,
            "broad_model_quality_claimed": broad_claim,
        },
    }


def broader_repeatability(*, supported_policy_count: int = 3) -> dict:
    return {
        "schema_version": "stage_b_duration_coverage_fill_outside_soloing_repair_broader_repeatability_sweep_v1",
        "repeatability_summary": {
            "boundary": "outside_soloing_repair_policy_repeatability_support",
            "source_candidate_count": 2,
            "repair_policy_count": 3,
            "supported_repair_policy_count": supported_policy_count,
            "total_variant_count": 6,
            "total_qualified_variant_count": 6,
            "selected_min_chord_tone_ratio": 1.0,
            "selected_max_non_chord_tone_run": 0,
            "selected_max_interval": 7,
            "human_audio_preference_claimed": False,
            "broad_model_quality_claimed": False,
        },
        "claim_boundary": {
            "boundary": "outside_soloing_repair_policy_repeatability_support",
            "policy_repeatability_claimed": True,
            "human_audio_preference_claimed": False,
            "broad_model_quality_claimed": False,
        },
    }


def user_review_fill(*, review_input_present: bool = False, preference_claimed: bool = False) -> dict:
    return {
        "schema_version": "stage_b_duration_coverage_fill_outside_soloing_repair_user_listening_review_fill_v1",
        "review_input_present": review_input_present,
        "fill_status": "pending_review_input",
        "decision": {
            "objective_auto_progress_allowed": True,
            "critical_user_input_required": False,
            "next_boundary": "outside_soloing_repair_audio_review_pending",
        },
        "claim_boundary": {
            "boundary": "outside_soloing_repair_audio_review_pending",
            "human_audio_preference_claimed": preference_claimed,
            "pending_without_review_input": True,
            "objective_only_followup_allowed": True,
            "broad_model_quality_claimed": False,
        },
    }


class StageBDurationCoverageFillOutsideSoloingRepairRepeatabilityConsolidationTest(unittest.TestCase):
    def test_consolidates_objective_repeatability_support(self) -> None:
        report = build_repeatability_consolidation_report(
            objective_evidence=objective_evidence(),
            broader_repeatability=broader_repeatability(),
            user_review_fill=user_review_fill(),
            output_dir=Path("outputs/repeatability_consolidation"),
            min_source_candidates=2,
            min_policy_repeatability_count=3,
        )
        summary = validate_repeatability_consolidation(
            report,
            expected_boundary="outside_soloing_repair_objective_repeatability_support",
            min_source_candidates=2,
            min_policy_repeatability_count=3,
            require_pending_review_guard=True,
            require_no_preference_claim=True,
            require_no_broad_quality_claim=True,
        )

        self.assertTrue(summary["objective_repair_repeatability_claimed"])
        self.assertEqual(summary["objective_source_candidate_count"], 2)
        self.assertEqual(summary["supported_repair_policy_count"], 3)
        self.assertEqual(summary["total_qualified_variant_count"], 6)
        self.assertFalse(summary["review_input_present"])
        self.assertFalse(summary["human_audio_preference_claimed"])
        self.assertFalse(summary["broad_model_quality_claimed"])

    def test_rejects_preference_claim_without_review_boundary(self) -> None:
        with self.assertRaises(StageBDurationCoverageOutsideSoloingRepairRepeatabilityConsolidationError):
            build_repeatability_consolidation_report(
                objective_evidence=objective_evidence(),
                broader_repeatability=broader_repeatability(),
                user_review_fill=user_review_fill(preference_claimed=True),
                output_dir=Path("outputs/repeatability_consolidation"),
                min_source_candidates=2,
                min_policy_repeatability_count=3,
            )

    def test_rejects_present_review_input_for_objective_only_consolidation(self) -> None:
        with self.assertRaises(StageBDurationCoverageOutsideSoloingRepairRepeatabilityConsolidationError):
            build_repeatability_consolidation_report(
                objective_evidence=objective_evidence(),
                broader_repeatability=broader_repeatability(),
                user_review_fill=user_review_fill(review_input_present=True),
                output_dir=Path("outputs/repeatability_consolidation"),
                min_source_candidates=2,
                min_policy_repeatability_count=3,
            )

    def test_rejects_broad_quality_claim(self) -> None:
        with self.assertRaises(StageBDurationCoverageOutsideSoloingRepairRepeatabilityConsolidationError):
            build_repeatability_consolidation_report(
                objective_evidence=objective_evidence(broad_claim=True),
                broader_repeatability=broader_repeatability(),
                user_review_fill=user_review_fill(),
                output_dir=Path("outputs/repeatability_consolidation"),
                min_source_candidates=2,
                min_policy_repeatability_count=3,
            )

    def test_validation_rejects_insufficient_policy_support(self) -> None:
        report = build_repeatability_consolidation_report(
            objective_evidence=objective_evidence(),
            broader_repeatability=broader_repeatability(supported_policy_count=2),
            user_review_fill=user_review_fill(),
            output_dir=Path("outputs/repeatability_consolidation"),
            min_source_candidates=2,
            min_policy_repeatability_count=3,
        )

        with self.assertRaises(StageBDurationCoverageOutsideSoloingRepairRepeatabilityConsolidationError):
            validate_repeatability_consolidation(
                report,
                expected_boundary="outside_soloing_repair_objective_repeatability_support",
                min_source_candidates=2,
                min_policy_repeatability_count=3,
                require_pending_review_guard=True,
                require_no_preference_claim=True,
                require_no_broad_quality_claim=True,
            )


if __name__ == "__main__":
    unittest.main()
