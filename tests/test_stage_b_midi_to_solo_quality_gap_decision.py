from __future__ import annotations

import unittest
from pathlib import Path

from scripts.audit_stage_b_midi_to_solo_mvp_completion import (
    BOUNDARY as MVP_COMPLETION_BOUNDARY,
)
from scripts.decide_stage_b_midi_to_solo_quality_gap import (
    BOUNDARY,
    NEXT_BOUNDARY,
    SELECTED_TARGET,
    StageBMidiToSoloQualityGapDecisionError,
    build_quality_gap_decision_report,
    validate_quality_gap_decision_report,
)


def mvp_completion_audit(
    *,
    generation_source: str = "context_conditioned_fallback",
    musical_complete: bool = False,
    quality_claim: bool = False,
    strict_count: int = 9,
) -> dict:
    return {
        "boundary": MVP_COMPLETION_BOUNDARY,
        "readiness": {
            "mvp_completion_audit_completed": True,
            "technical_model_core_mvp_completed": True,
            "quality_gap_decision_required": True,
            "human_audio_preference_claimed": False,
            "midi_to_solo_musical_quality_claimed": quality_claim,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
            "production_ready_claimed": False,
        },
        "completion_audit": {
            "technical_model_core_mvp_completed": True,
            "input_to_ranked_midi_completed": True,
            "input_to_rendered_wav_completed": True,
            "selected_scale_objective_repair_completed": True,
            "phrase_bank_cli_technical_path_completed": True,
            "readme_evidence_boundary_refreshed": True,
            "musical_quality_mvp_completed": musical_complete,
            "human_audio_preference_completed": False,
            "broad_trained_model_completed": False,
            "brad_style_adaptation_completed": False,
            "product_mvp_completed": False,
        },
        "current_evidence": {
            "generation_source": generation_source,
            "exported_candidate_count": 3,
            "rendered_audio_file_count": 3,
            "objective_sample_count": 9,
            "objective_strict_valid_sample_count": strict_count,
            "objective_dead_air_failure_count": 0,
            "objective_avg_postprocess_removal_ratio": 0.2176,
            "objective_target_avg_postprocess_removal_ratio": 0.3,
            "phrase_bank_cli_technical_path_ready": True,
            "cli_candidate_count": 3,
            "cli_rendered_audio_file_count": 3,
            "cli_input_context_bars": 228,
            "cli_preference_fill_allowed": False,
        },
        "decision": {
            "next_boundary": "stage_b_midi_to_solo_quality_gap_decision",
            "critical_user_input_required": False,
        },
    }


class StageBMidiToSoloQualityGapDecisionTest(unittest.TestCase):
    def test_selects_model_conditioned_input_path_alignment_for_fallback_path(self) -> None:
        report = build_quality_gap_decision_report(
            mvp_completion_audit=mvp_completion_audit(),
            output_dir=Path("outputs/quality_gap"),
            issue_number=618,
        )
        summary = validate_quality_gap_decision_report(
            report,
            expected_boundary=BOUNDARY,
            expected_next_boundary=NEXT_BOUNDARY,
            expected_target=SELECTED_TARGET,
            require_no_quality_claim=True,
        )

        self.assertEqual(summary["selected_target"], SELECTED_TARGET)
        self.assertEqual(summary["next_boundary"], NEXT_BOUNDARY)
        self.assertTrue(summary["fallback_path_active"])
        self.assertTrue(summary["model_conditioned_input_path_alignment_required"])
        self.assertFalse(summary["human_review_required_now"])
        self.assertTrue(summary["technical_model_core_mvp_completed"])
        self.assertTrue(summary["phrase_bank_cli_technical_path_completed"])
        self.assertEqual(summary["cli_candidate_count"], 3)
        self.assertEqual(summary["cli_rendered_audio_file_count"], 3)
        self.assertEqual(summary["cli_input_context_bars"], 228)
        self.assertFalse(summary["cli_preference_fill_allowed"])
        self.assertFalse(summary["musical_quality_mvp_completed"])
        self.assertFalse(summary["human_audio_preference_claimed"])
        self.assertFalse(summary["midi_to_solo_musical_quality_claimed"])

    def test_rejects_completed_musical_quality(self) -> None:
        with self.assertRaises(StageBMidiToSoloQualityGapDecisionError):
            build_quality_gap_decision_report(
                mvp_completion_audit=mvp_completion_audit(musical_complete=True),
                output_dir=Path("outputs/quality_gap"),
                issue_number=618,
            )

    def test_rejects_quality_claim(self) -> None:
        with self.assertRaises(StageBMidiToSoloQualityGapDecisionError):
            build_quality_gap_decision_report(
                mvp_completion_audit=mvp_completion_audit(quality_claim=True),
                output_dir=Path("outputs/quality_gap"),
                issue_number=618,
            )

    def test_rejects_objective_strict_shortfall(self) -> None:
        with self.assertRaises(StageBMidiToSoloQualityGapDecisionError):
            build_quality_gap_decision_report(
                mvp_completion_audit=mvp_completion_audit(strict_count=8),
                output_dir=Path("outputs/quality_gap"),
                issue_number=618,
            )

    def test_boundary_constants_are_stable(self) -> None:
        self.assertEqual(BOUNDARY, "stage_b_midi_to_solo_quality_gap_decision")
        self.assertEqual(
            NEXT_BOUNDARY,
            "stage_b_midi_to_solo_model_conditioned_input_path_quality_alignment",
        )
        self.assertEqual(SELECTED_TARGET, "model_conditioned_input_path_quality_alignment")


if __name__ == "__main__":
    unittest.main()
