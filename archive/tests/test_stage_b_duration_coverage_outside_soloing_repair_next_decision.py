from __future__ import annotations

import unittest
from pathlib import Path

from scripts.decide_stage_b_duration_coverage_outside_soloing_repair_next_step import (
    StageBDurationCoverageOutsideSoloingRepairNextDecisionError,
    build_outside_soloing_repair_next_decision,
    validate_outside_soloing_repair_next_decision,
)


def objective_evidence(*, boundary: str = "outside_soloing_repair_objective_evidence_support", broad_claim: bool = False) -> dict:
    return {
        "schema_version": "stage_b_duration_coverage_fill_outside_soloing_repair_objective_evidence_consolidation_v1",
        "objective_evidence_summary": {
            "boundary": boundary,
            "source_candidate_count": 2,
            "qualified_source_candidate_count": 2,
            "dead_air_preserved_source_candidate_count": 2,
            "chord_tone_pass_source_candidate_count": 2,
            "non_chord_run_pass_source_candidate_count": 2,
            "interval_pass_source_candidate_count": 2,
            "selected_min_chord_tone_ratio": 1.0,
            "selected_max_non_chord_tone_run": 0,
            "selected_max_interval": 7,
        },
        "claim_boundary": {
            "boundary": boundary,
            "human_audio_preference_claimed": False,
            "broad_model_quality_claimed": broad_claim,
        },
    }


class StageBDurationCoverageOutsideSoloingRepairNextDecisionTest(unittest.TestCase):
    def test_selects_broader_repeatability_sweep_boundary(self) -> None:
        report = build_outside_soloing_repair_next_decision(
            objective_evidence(),
            output_dir=Path("outputs/next_decision"),
        )
        summary = validate_outside_soloing_repair_next_decision(
            report,
            expected_next_boundary="outside_soloing_repair_broader_repeatability_sweep",
            require_auto_progress_allowed=True,
            require_no_critical_user_input=True,
            require_no_broad_quality_claim=True,
        )

        self.assertEqual(summary["input_boundary"], "outside_soloing_repair_objective_evidence_support")
        self.assertEqual(summary["next_boundary"], "outside_soloing_repair_broader_repeatability_sweep")
        self.assertTrue(summary["auto_progress_allowed"])
        self.assertFalse(summary["critical_user_input_required"])
        self.assertFalse(summary["human_audio_preference_claimed"])
        self.assertFalse(summary["broad_model_quality_claimed"])

    def test_rejects_incomplete_objective_boundary(self) -> None:
        with self.assertRaises(StageBDurationCoverageOutsideSoloingRepairNextDecisionError):
            build_outside_soloing_repair_next_decision(
                objective_evidence(boundary="outside_soloing_repair_objective_evidence_incomplete"),
                output_dir=Path("outputs/next_decision"),
            )

    def test_rejects_broad_quality_claim(self) -> None:
        with self.assertRaises(StageBDurationCoverageOutsideSoloingRepairNextDecisionError):
            build_outside_soloing_repair_next_decision(
                objective_evidence(broad_claim=True),
                output_dir=Path("outputs/next_decision"),
            )


if __name__ == "__main__":
    unittest.main()
