from __future__ import annotations

import unittest
from pathlib import Path

from scripts.decide_stage_b_duration_coverage_outside_soloing_repair import (
    StageBDurationCoverageOutsideSoloingRepairDecisionError,
    build_outside_soloing_repair_decision,
    validate_outside_soloing_repair_decision,
)


def user_listening_review(*, keep_claim: bool = False, boundary: str = "repeatability_audio_review_needs_followup") -> dict:
    return {
        "schema_version": "stage_b_duration_coverage_fill_repeatability_user_listening_review_fill_v1",
        "candidate_id": "duration_coverage_fill_repeatability_sources",
        "reviewed_audio_files": [{}, {}],
        "user_listening_review": {
            "overall_decision": "reject_all",
            "candidate_decision": "needs_followup",
            "timing": "outside_or_unclear",
            "phrase": "outside_or_unclear",
            "vocabulary": "outside_or_unclear",
            "assessment": "both candidates sound difficult and outside-soloing-like",
            "candidate_reviews": [
                {"decision": "needs_followup"},
                {"decision": "needs_followup"},
            ],
        },
        "claim_boundary": {
            "boundary": boundary,
            "repeatability_human_audio_keep_claimed": keep_claim,
            "broad_model_quality_claimed": False,
        },
    }


class StageBDurationCoverageOutsideSoloingRepairDecisionTest(unittest.TestCase):
    def test_selects_pitch_role_phrase_clarity_repair_boundary(self) -> None:
        report = build_outside_soloing_repair_decision(
            user_listening_review(),
            output_dir=Path("outputs/outside_soloing_decision"),
        )
        summary = validate_outside_soloing_repair_decision(
            report,
            expected_next_boundary="outside_soloing_pitch_role_phrase_clarity_repair",
            require_auto_progress_allowed=True,
            require_no_critical_user_input=True,
            require_no_broad_quality_claim=True,
        )

        self.assertEqual(summary["input_boundary"], "repeatability_audio_review_needs_followup")
        self.assertTrue(summary["auto_progress_allowed"])
        self.assertFalse(summary["critical_user_input_required"])
        self.assertFalse(summary["human_audio_keep_claimed"])
        self.assertGreaterEqual(summary["repair_target_count"], 5)
        self.assertIn("reduce_outside_sounding_pitch_choices", report["repair_targets"])

    def test_rejects_keep_claim(self) -> None:
        with self.assertRaises(StageBDurationCoverageOutsideSoloingRepairDecisionError):
            build_outside_soloing_repair_decision(
                user_listening_review(keep_claim=True),
                output_dir=Path("outputs/outside_soloing_decision"),
            )

    def test_rejects_unexpected_boundary(self) -> None:
        with self.assertRaises(StageBDurationCoverageOutsideSoloingRepairDecisionError):
            build_outside_soloing_repair_decision(
                user_listening_review(boundary="other_boundary"),
                output_dir=Path("outputs/outside_soloing_decision"),
            )


if __name__ == "__main__":
    unittest.main()
