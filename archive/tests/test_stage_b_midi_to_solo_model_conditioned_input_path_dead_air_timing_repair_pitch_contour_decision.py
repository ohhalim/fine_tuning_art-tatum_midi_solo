from __future__ import annotations

import unittest

from scripts.decide_stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_objective_next import (
    BOUNDARY as SOURCE_BOUNDARY,
    PITCH_CONTOUR_NEXT_BOUNDARY as SOURCE_NEXT_BOUNDARY,
)
from scripts.decide_stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_pitch_contour import (
    BOUNDARY,
    NEXT_BOUNDARY,
    StageBMidiToSoloModelConditionedInputPathDeadAirTimingRepairPitchContourError,
    build_pitch_contour_decision_report,
    validate_pitch_contour_decision_report,
)


def objective_next_source(
    *,
    max_interval: int = 62,
    quality_claim: bool = False,
    wide_interval_followup: bool = True,
) -> dict:
    next_boundary = (
        SOURCE_NEXT_BOUNDARY
        if wide_interval_followup
        else "stage_b_midi_to_solo_mvp_current_evidence_consolidation"
    )
    return {
        "boundary": SOURCE_BOUNDARY,
        "source_boundary": (
            "stage_b_midi_to_solo_model_conditioned_input_path_"
            "dead_air_timing_repair_audio_package"
        ),
        "objective_summary": {
            "rendered_audio_file_count": 3,
            "technical_wav_validation": True,
            "repaired_candidate_count": 3,
            "repaired_dead_air_max": 0.0,
            "max_added_note_ratio": 0.9166666666666666,
            "max_postprocess_removal_ratio": 0.0,
            "max_repaired_interval": max_interval,
            "max_interval_threshold": 12,
            "remaining_wide_interval_risk": wide_interval_followup,
            "wide_interval_followup_required": wide_interval_followup,
            "added_note_ratio_review_threshold": 0.75,
            "added_note_ratio_review_required": True,
        },
        "selected_next_target": {
            "target": (
                "wide_interval_pitch_contour_repair"
                if wide_interval_followup
                else "current_evidence_consolidation"
            ),
            "next_boundary": next_boundary,
        },
        "readiness": {
            "boundary": SOURCE_BOUNDARY,
            "objective_next_decision_completed": True,
            "technical_wav_validation": True,
            "dead_air_target_supported": True,
            "wide_interval_followup_required": wide_interval_followup,
            "current_evidence_consolidation_ready": not wide_interval_followup,
            "human_audio_preference_claimed": quality_claim,
            "audio_rendered_quality_claimed": False,
            "midi_to_solo_musical_quality_claimed": False,
            "model_checkpoint_generation_quality_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
            "production_ready_claimed": False,
        },
        "decision": {
            "current_boundary": SOURCE_BOUNDARY,
            "next_boundary": next_boundary,
            "auto_progress_allowed": True,
            "critical_user_input_required": False,
        },
        "next_recommended_issue": (
            "Stage B MIDI-to-solo model-conditioned input path "
            "dead-air timing repair pitch contour decision"
        ),
    }


class StageBMidiToSoloModelConditionedInputPathDeadAirTimingRepairPitchContourTest(
    unittest.TestCase
):
    def test_defines_pitch_contour_repair_probe_boundary(self) -> None:
        report = build_pitch_contour_decision_report(
            objective_next_report=objective_next_source(),
            output_dir="out",
            issue_number=696,
            target_max_interval=12,
            target_dead_air_max=0.35,
            min_repaired_candidate_count=3,
            max_simultaneous_notes=1,
            max_added_note_ratio_review_threshold=0.75,
        )
        summary = validate_pitch_contour_decision_report(
            report,
            expected_boundary=BOUNDARY,
            expected_next_boundary=NEXT_BOUNDARY,
            require_pitch_contour_decision=True,
            require_repair_probe=True,
            require_no_quality_claim=True,
        )

        self.assertTrue(summary["pitch_contour_decision_completed"])
        self.assertTrue(summary["repair_probe_required"])
        self.assertTrue(summary["technical_wav_validation"])
        self.assertTrue(summary["dead_air_target_supported"])
        self.assertEqual(summary["source_max_interval"], 62)
        self.assertEqual(summary["target_max_interval"], 12)
        self.assertEqual(summary["required_interval_reduction_min"], 50)
        self.assertTrue(summary["added_note_ratio_review_required"])
        self.assertFalse(summary["current_evidence_consolidation_ready"])
        self.assertFalse(summary["human_audio_preference_claimed"])

    def test_rejects_source_without_wide_interval_followup(self) -> None:
        with self.assertRaises(
            StageBMidiToSoloModelConditionedInputPathDeadAirTimingRepairPitchContourError
        ):
            build_pitch_contour_decision_report(
                objective_next_report=objective_next_source(
                    max_interval=9,
                    wide_interval_followup=False,
                ),
                output_dir="out",
                issue_number=696,
                target_max_interval=12,
                target_dead_air_max=0.35,
                min_repaired_candidate_count=3,
                max_simultaneous_notes=1,
                max_added_note_ratio_review_threshold=0.75,
            )

    def test_rejects_quality_claim(self) -> None:
        with self.assertRaises(
            StageBMidiToSoloModelConditionedInputPathDeadAirTimingRepairPitchContourError
        ):
            build_pitch_contour_decision_report(
                objective_next_report=objective_next_source(quality_claim=True),
                output_dir="out",
                issue_number=696,
                target_max_interval=12,
                target_dead_air_max=0.35,
                min_repaired_candidate_count=3,
                max_simultaneous_notes=1,
                max_added_note_ratio_review_threshold=0.75,
            )

    def test_boundary_constants_are_stable(self) -> None:
        self.assertEqual(
            BOUNDARY,
            "stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_pitch_contour_decision",
        )
        self.assertEqual(
            NEXT_BOUNDARY,
            "stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_pitch_contour_probe",
        )


if __name__ == "__main__":
    unittest.main()
