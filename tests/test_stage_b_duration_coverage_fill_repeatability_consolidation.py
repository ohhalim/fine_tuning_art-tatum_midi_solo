from __future__ import annotations

import unittest
from pathlib import Path

from scripts.summarize_stage_b_duration_coverage_fill_repeatability_consolidation import (
    StageBDurationCoverageRepeatabilityConsolidationError,
    build_repeatability_consolidation_report,
    validate_repeatability_consolidation,
)


def user_listening_consolidation(*, broad_claim: bool = False) -> dict:
    return {
        "schema_version": "stage_b_duration_coverage_fill_user_listening_consolidation_v1",
        "candidate_id": "current_duration_fill_keep",
        "evidence_alignment": {
            "same_preferred_candidate": True,
            "rendered_audio_file_count": 2,
        },
        "consolidated_claim_boundary": {
            "preferred_candidate": "duration_coverage_fill_keep",
            "single_user_human_audio_preference_claimed": True,
            "broad_model_quality_claimed": broad_claim,
        },
    }


def dead_air_gain_repair(*, boundary: str = "qualified_gate_repeatability_with_dead_air_gain") -> dict:
    return {
        "schema_version": "stage_b_duration_coverage_fill_dead_air_gain_repeatability_repair_v1",
        "repair_summary": {
            "boundary": boundary,
            "source_candidate_count": 2,
            "qualified_source_candidate_count": 2,
            "dead_air_gain_source_candidate_count": 2,
            "total_variant_count": 8,
            "total_qualified_variant_count": 7,
            "total_dead_air_gain_variant_count": 6,
            "selected_fill_additions": [6],
            "broad_model_quality_claimed": False,
        },
        "claim_boundary": {
            "selected_distinct_source_dead_air_gain_claimed": True,
            "broad_model_quality_claimed": False,
        },
    }


class StageBDurationCoverageFillRepeatabilityConsolidationTest(unittest.TestCase):
    def test_consolidates_current_keep_and_distinct_source_repeatability(self) -> None:
        report = build_repeatability_consolidation_report(
            user_listening_consolidation=user_listening_consolidation(),
            dead_air_gain_repair=dead_air_gain_repair(),
            output_dir=Path("outputs/repeatability_consolidation"),
        )
        summary = validate_repeatability_consolidation(
            report,
            expected_boundary="current_keep_and_distinct_source_dead_air_gain_midi_support",
            require_no_broad_quality_claim=True,
        )

        self.assertTrue(summary["current_keep_single_user_preference_claimed"])
        self.assertTrue(summary["distinct_source_midi_gate_repeatability_claimed"])
        self.assertTrue(summary["distinct_source_dead_air_gain_claimed"])
        self.assertEqual(summary["source_candidate_count"], 2)
        self.assertEqual(summary["dead_air_gain_source_candidate_count"], 2)
        self.assertFalse(summary["broad_model_quality_claimed"])
        self.assertIn("new_source_human_audio_preference", report["not_proven"])

    def test_rejects_broad_quality_claim(self) -> None:
        with self.assertRaises(StageBDurationCoverageRepeatabilityConsolidationError):
            build_repeatability_consolidation_report(
                user_listening_consolidation=user_listening_consolidation(broad_claim=True),
                dead_air_gain_repair=dead_air_gain_repair(),
                output_dir=Path("outputs/repeatability_consolidation"),
            )

    def test_rejects_missing_dead_air_gain_boundary(self) -> None:
        with self.assertRaises(StageBDurationCoverageRepeatabilityConsolidationError):
            build_repeatability_consolidation_report(
                user_listening_consolidation=user_listening_consolidation(),
                dead_air_gain_repair=dead_air_gain_repair(boundary="dead_air_gain_repeatability_not_repaired"),
                output_dir=Path("outputs/repeatability_consolidation"),
            )


if __name__ == "__main__":
    unittest.main()
