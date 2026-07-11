from __future__ import annotations

import unittest
from pathlib import Path

from scripts.decide_stage_b_midi_to_solo_model_conditioned_input_path_quality_alignment import (
    BOUNDARY,
    NEXT_BOUNDARY,
    SELECTED_PROBE_TARGET,
    StageBMidiToSoloModelConditionedInputPathAlignmentError,
    build_alignment_report,
    validate_alignment_report,
)
from scripts.decide_stage_b_midi_to_solo_quality_gap import (
    BOUNDARY as QUALITY_GAP_BOUNDARY,
    NEXT_BOUNDARY as QUALITY_GAP_NEXT_BOUNDARY,
    SELECTED_TARGET,
)


def quality_gap_decision(
    *,
    selected_target: str = SELECTED_TARGET,
    fallback_active: bool = True,
    quality_claim: bool = False,
) -> dict:
    return {
        "boundary": QUALITY_GAP_BOUNDARY,
        "quality_gap": {
            "technical_model_core_mvp_completed": True,
            "phrase_bank_cli_technical_path_completed": True,
            "musical_quality_mvp_completed": False,
            "human_audio_preference_completed": False,
            "product_mvp_completed": False,
            "fallback_path_active": fallback_active,
            "model_conditioned_input_path_alignment_required": fallback_active,
            "human_review_required_now": False,
        },
        "mvp_completion_summary": {
            "phrase_bank_cli_technical_path_completed": True,
            "phrase_bank_cli_technical_path_ready": True,
            "cli_candidate_count": 3,
            "cli_rendered_audio_file_count": 3,
            "cli_input_context_bars": 228,
            "cli_preference_fill_allowed": False,
        },
        "selected_target": {
            "selected_target": selected_target,
            "selected_next_boundary": QUALITY_GAP_NEXT_BOUNDARY,
            "fallback_path_active": fallback_active,
            "human_review_required_now": False,
        },
        "readiness": {
            "quality_gap_decision_completed": True,
            "selected_target": selected_target,
            "next_boundary_selected": QUALITY_GAP_NEXT_BOUNDARY,
            "human_review_required_now": False,
            "human_audio_preference_claimed": False,
            "midi_to_solo_musical_quality_claimed": quality_claim,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
            "production_ready_claimed": False,
        },
        "decision": {
            "next_boundary": QUALITY_GAP_NEXT_BOUNDARY,
            "critical_user_input_required": False,
        },
    }


class StageBMidiToSoloModelConditionedInputPathAlignmentTest(unittest.TestCase):
    def test_selects_fallback_replacement_probe_without_quality_claim(self) -> None:
        report = build_alignment_report(
            quality_gap_decision=quality_gap_decision(),
            output_dir=Path("outputs/alignment"),
            issue_number=620,
        )
        summary = validate_alignment_report(
            report,
            expected_boundary=BOUNDARY,
            expected_next_boundary=NEXT_BOUNDARY,
            expected_probe_target=SELECTED_PROBE_TARGET,
            require_probe_required=True,
            require_no_quality_claim=True,
        )

        self.assertEqual(summary["selected_probe_target"], SELECTED_PROBE_TARGET)
        self.assertEqual(summary["next_boundary"], NEXT_BOUNDARY)
        self.assertFalse(summary["model_conditioned_input_path_aligned"])
        self.assertTrue(summary["fallback_replacement_probe_required"])
        self.assertTrue(summary["fallback_replacement_probe_required_from_decision"])
        self.assertTrue(summary["phrase_bank_cli_technical_path_completed"])
        self.assertEqual(summary["cli_candidate_count"], 3)
        self.assertEqual(summary["cli_rendered_audio_file_count"], 3)
        self.assertEqual(summary["cli_input_context_bars"], 228)
        self.assertFalse(summary["cli_preference_fill_allowed"])
        self.assertFalse(summary["human_review_required_now"])
        self.assertFalse(summary["human_audio_preference_claimed"])
        self.assertFalse(summary["midi_to_solo_musical_quality_claimed"])

    def test_rejects_wrong_selected_target(self) -> None:
        with self.assertRaises(StageBMidiToSoloModelConditionedInputPathAlignmentError):
            build_alignment_report(
                quality_gap_decision=quality_gap_decision(selected_target="listening_review_quality_gap"),
                output_dir=Path("outputs/alignment"),
                issue_number=620,
            )

    def test_rejects_inactive_fallback_path(self) -> None:
        with self.assertRaises(StageBMidiToSoloModelConditionedInputPathAlignmentError):
            build_alignment_report(
                quality_gap_decision=quality_gap_decision(fallback_active=False),
                output_dir=Path("outputs/alignment"),
                issue_number=620,
            )

    def test_rejects_quality_claim(self) -> None:
        with self.assertRaises(StageBMidiToSoloModelConditionedInputPathAlignmentError):
            build_alignment_report(
                quality_gap_decision=quality_gap_decision(quality_claim=True),
                output_dir=Path("outputs/alignment"),
                issue_number=620,
            )

    def test_boundary_constants_are_stable(self) -> None:
        self.assertEqual(BOUNDARY, "stage_b_midi_to_solo_model_conditioned_input_path_quality_alignment")
        self.assertEqual(NEXT_BOUNDARY, "stage_b_midi_to_solo_model_conditioned_input_path_probe")
        self.assertEqual(
            SELECTED_PROBE_TARGET,
            "replace_fallback_with_model_conditioned_input_path_probe",
        )


if __name__ == "__main__":
    unittest.main()
