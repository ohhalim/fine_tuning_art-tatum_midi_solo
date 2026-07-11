from __future__ import annotations

import unittest
from pathlib import Path

from scripts.decide_stage_b_duration_coverage_next_step import (
    StageBDurationCoverageNextDecisionError,
    build_next_decision,
    validate_next_decision,
)


def consolidation(*, same_preference: bool = True, broad_claim: bool = False) -> dict:
    return {
        "schema_version": "stage_b_duration_coverage_fill_user_listening_consolidation_v1",
        "candidate_id": "duration_fill_candidate",
        "evidence_alignment": {
            "same_preferred_candidate": same_preference,
            "rendered_audio_file_count": 2,
            "single_user_review": True,
        },
        "consolidated_claim_boundary": {
            "preferred_candidate": "duration_coverage_fill_keep",
            "single_user_human_audio_preference_claimed": True,
            "broad_model_quality_claimed": broad_claim,
        },
    }


class StageBDurationCoverageNextDecisionTest(unittest.TestCase):
    def test_selects_broader_repeatability_without_critical_user_input(self) -> None:
        report = build_next_decision(consolidation(), output_dir=Path("outputs/next_decision"))
        summary = validate_next_decision(
            report,
            expected_next_boundary="broader_repeatability_sweep",
            require_auto_progress_allowed=True,
            require_no_critical_user_input=True,
        )

        self.assertEqual(summary["preferred_candidate"], "duration_coverage_fill_keep")
        self.assertTrue(summary["auto_progress_allowed"])
        self.assertFalse(summary["critical_user_input_required"])
        self.assertIn("multi_seed_repeatability", report["not_proven"])

    def test_rejects_misaligned_evidence(self) -> None:
        with self.assertRaises(StageBDurationCoverageNextDecisionError):
            build_next_decision(consolidation(same_preference=False), output_dir=Path("outputs/next_decision"))

    def test_rejects_broad_quality_claim(self) -> None:
        with self.assertRaises(StageBDurationCoverageNextDecisionError):
            build_next_decision(consolidation(broad_claim=True), output_dir=Path("outputs/next_decision"))


if __name__ == "__main__":
    unittest.main()
