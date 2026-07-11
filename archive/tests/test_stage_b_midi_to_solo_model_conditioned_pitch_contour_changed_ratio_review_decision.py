from __future__ import annotations

import unittest
from pathlib import Path

from scripts.decide_stage_b_midi_to_solo_model_conditioned_pitch_contour_changed_ratio_review import (
    BOUNDARY,
    NEXT_BOUNDARY,
    SELECTED_TARGET,
    StageBMidiToSoloPitchContourChangedRatioReviewDecisionError,
    build_changed_ratio_review_decision_report,
    validate_changed_ratio_review_decision_report,
)
from scripts.decide_stage_b_midi_to_solo_quality_gap import (
    BOUNDARY as QUALITY_GAP_BOUNDARY,
    PITCH_CONTOUR_CHANGED_RATIO_NEXT_BOUNDARY,
    PITCH_CONTOUR_CHANGED_RATIO_TARGET,
)


def quality_gap_decision(
    *,
    selected_target: str = PITCH_CONTOUR_CHANGED_RATIO_TARGET,
    next_boundary: str = PITCH_CONTOUR_CHANGED_RATIO_NEXT_BOUNDARY,
    changed_ratio_required: bool = True,
    alignment_required: bool = False,
    quality_claim: bool = False,
    interval: int = 11,
) -> dict:
    return {
        "boundary": QUALITY_GAP_BOUNDARY,
        "quality_gap": {
            "technical_model_core_mvp_completed": True,
            "phrase_bank_cli_technical_path_completed": True,
            "model_conditioned_pitch_contour_objective_completed": True,
            "musical_quality_mvp_completed": False,
            "human_audio_preference_completed": False,
            "product_mvp_completed": False,
            "fallback_path_active": True,
            "model_conditioned_input_path_alignment_required": alignment_required,
            "model_conditioned_pitch_contour_objective_path_ready": True,
            "pitch_contour_changed_ratio_review_required": changed_ratio_required,
            "human_review_required_now": False,
        },
        "mvp_completion_summary": {
            "technical_model_core_mvp_completed": True,
            "model_conditioned_pitch_contour_objective_completed": True,
            "model_conditioned_pitch_contour_objective_path_ready": True,
            "model_conditioned_pitch_contour_max_interval": interval,
            "model_conditioned_pitch_contour_max_interval_threshold": 12,
            "model_conditioned_pitch_contour_target_supported": interval <= 12,
            "model_conditioned_pitch_contour_pitch_changed_ratio_review_required": (
                changed_ratio_required
            ),
            "model_conditioned_pitch_contour_audio_review_required": True,
        },
        "selected_target": {
            "selected_target": selected_target,
            "selected_next_boundary": next_boundary,
            "fallback_path_active": True,
            "model_conditioned_pitch_contour_objective_path_ready": True,
            "pitch_contour_changed_ratio_review_required": changed_ratio_required,
            "human_review_required_now": False,
        },
        "readiness": {
            "quality_gap_decision_completed": True,
            "selected_target": selected_target,
            "next_boundary_selected": next_boundary,
            "human_review_required_now": False,
            "human_audio_preference_claimed": False,
            "midi_to_solo_musical_quality_claimed": quality_claim,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
            "production_ready_claimed": False,
        },
        "decision": {
            "next_boundary": next_boundary,
            "critical_user_input_required": False,
        },
    }


class StageBMidiToSoloPitchContourChangedRatioReviewDecisionTest(unittest.TestCase):
    def test_selects_lower_pitch_change_repair_probe_without_quality_claim(self) -> None:
        report = build_changed_ratio_review_decision_report(
            quality_gap_decision=quality_gap_decision(),
            output_dir=Path("outputs/changed_ratio"),
            issue_number=716,
            changed_ratio_review_threshold=0.5,
        )
        summary = validate_changed_ratio_review_decision_report(
            report,
            expected_boundary=BOUNDARY,
            expected_next_boundary=NEXT_BOUNDARY,
            expected_target=SELECTED_TARGET,
            require_repair_probe=True,
            require_no_quality_claim=True,
        )

        self.assertEqual(summary["selected_target"], SELECTED_TARGET)
        self.assertEqual(summary["next_boundary"], NEXT_BOUNDARY)
        self.assertTrue(summary["repair_probe_required"])
        self.assertTrue(summary["technical_model_core_mvp_completed"])
        self.assertTrue(summary["model_conditioned_pitch_contour_objective_completed"])
        self.assertFalse(summary["model_conditioned_input_path_alignment_required"])
        self.assertEqual(summary["max_interval"], 11)
        self.assertEqual(summary["max_interval_threshold"], 12)
        self.assertTrue(summary["pitch_contour_target_supported"])
        self.assertEqual(summary["changed_ratio_review_threshold"], 0.5)
        self.assertTrue(summary["changed_ratio_review_required"])
        self.assertTrue(summary["audio_review_required"])
        self.assertFalse(summary["human_audio_preference_claimed"])
        self.assertFalse(summary["midi_to_solo_musical_quality_claimed"])

    def test_rejects_wrong_selected_target(self) -> None:
        with self.assertRaises(StageBMidiToSoloPitchContourChangedRatioReviewDecisionError):
            build_changed_ratio_review_decision_report(
                quality_gap_decision=quality_gap_decision(selected_target="other_target"),
                output_dir=Path("outputs/changed_ratio"),
                issue_number=716,
                changed_ratio_review_threshold=0.5,
            )

    def test_rejects_alignment_required_loop(self) -> None:
        with self.assertRaises(StageBMidiToSoloPitchContourChangedRatioReviewDecisionError):
            build_changed_ratio_review_decision_report(
                quality_gap_decision=quality_gap_decision(alignment_required=True),
                output_dir=Path("outputs/changed_ratio"),
                issue_number=716,
                changed_ratio_review_threshold=0.5,
            )

    def test_rejects_missing_changed_ratio_requirement(self) -> None:
        with self.assertRaises(StageBMidiToSoloPitchContourChangedRatioReviewDecisionError):
            build_changed_ratio_review_decision_report(
                quality_gap_decision=quality_gap_decision(changed_ratio_required=False),
                output_dir=Path("outputs/changed_ratio"),
                issue_number=716,
                changed_ratio_review_threshold=0.5,
            )

    def test_rejects_interval_threshold_failure(self) -> None:
        with self.assertRaises(StageBMidiToSoloPitchContourChangedRatioReviewDecisionError):
            build_changed_ratio_review_decision_report(
                quality_gap_decision=quality_gap_decision(interval=13),
                output_dir=Path("outputs/changed_ratio"),
                issue_number=716,
                changed_ratio_review_threshold=0.5,
            )

    def test_rejects_quality_claim(self) -> None:
        with self.assertRaises(StageBMidiToSoloPitchContourChangedRatioReviewDecisionError):
            build_changed_ratio_review_decision_report(
                quality_gap_decision=quality_gap_decision(quality_claim=True),
                output_dir=Path("outputs/changed_ratio"),
                issue_number=716,
                changed_ratio_review_threshold=0.5,
            )

    def test_boundary_constants_are_stable(self) -> None:
        self.assertEqual(
            BOUNDARY,
            "stage_b_midi_to_solo_model_conditioned_pitch_contour_changed_ratio_review_decision",
        )
        self.assertEqual(
            NEXT_BOUNDARY,
            "stage_b_midi_to_solo_model_conditioned_pitch_contour_changed_ratio_repair_probe",
        )
        self.assertEqual(SELECTED_TARGET, "lower_pitch_change_ratio_repair_probe")


if __name__ == "__main__":
    unittest.main()
