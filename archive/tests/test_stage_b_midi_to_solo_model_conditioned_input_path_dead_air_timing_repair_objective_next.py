from __future__ import annotations

import unittest

from scripts.decide_stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_objective_next import (
    BOUNDARY,
    CURRENT_EVIDENCE_NEXT_BOUNDARY,
    PITCH_CONTOUR_NEXT_BOUNDARY,
    StageBMidiToSoloModelConditionedInputPathDeadAirTimingRepairObjectiveNextError,
    build_objective_next_report,
    validate_objective_next_report,
)
from scripts.render_stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_audio import (
    BOUNDARY as SOURCE_BOUNDARY,
    NEXT_BOUNDARY as SOURCE_NEXT_BOUNDARY,
)


def audio_package(
    *,
    max_interval: int = 62,
    quality_claim: bool = False,
) -> dict:
    return {
        "summary": {
            "rendered_audio_file_count": 3,
            "technical_wav_validation": True,
            "repaired_candidate_count": 3,
            "repaired_dead_air_max": 0.0,
            "max_added_note_ratio": 0.9166666666666666,
            "max_postprocess_removal_ratio": 0.0,
            "max_repaired_interval": max_interval,
            "remaining_wide_interval_risk": max_interval >= 12,
        },
        "audio_render_boundary": {
            "boundary": SOURCE_BOUNDARY,
            "render_attempted": True,
            "rendered_audio_file_count": 3,
            "technical_wav_validation": True,
            "dead_air_timing_repair_audio_package_completed": True,
            "human_audio_preference_claimed": quality_claim,
            "audio_rendered_quality_claimed": False,
            "midi_to_solo_musical_quality_claimed": False,
            "model_checkpoint_generation_quality_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
            "production_ready_claimed": False,
        },
        "rendered_audio_files": [
            {
                "rank": index,
                "wav_file": {
                    "path": f"audio/rank_{index}.wav",
                    "exists": True,
                    "sample_rate": 44100,
                    "frame_count": 1000,
                    "size_bytes": 4044,
                },
            }
            for index in range(1, 4)
        ],
        "decision": {
            "next_boundary": SOURCE_NEXT_BOUNDARY,
            "critical_user_input_required": False,
        },
    }


class StageBMidiToSoloModelConditionedInputPathDeadAirTimingRepairObjectiveNextTest(
    unittest.TestCase
):
    def test_routes_to_pitch_contour_when_wide_interval_remains(self) -> None:
        report = build_objective_next_report(
            audio_package_report=audio_package(),
            output_dir="out",
            issue_number=694,
            expected_count=3,
            max_interval_threshold=12,
            max_added_note_ratio_review_threshold=0.75,
        )
        summary = validate_objective_next_report(
            report,
            expected_boundary=BOUNDARY,
            expected_next_boundary=PITCH_CONTOUR_NEXT_BOUNDARY,
            require_objective_decision=True,
            require_wide_interval_followup=True,
            require_no_quality_claim=True,
        )

        self.assertTrue(summary["objective_next_decision_completed"])
        self.assertTrue(summary["technical_wav_validation"])
        self.assertTrue(summary["dead_air_target_supported"])
        self.assertTrue(summary["wide_interval_followup_required"])
        self.assertFalse(summary["current_evidence_consolidation_ready"])
        self.assertTrue(summary["added_note_ratio_review_required"])
        self.assertEqual(summary["max_repaired_interval"], 62)
        self.assertFalse(summary["human_audio_preference_claimed"])

    def test_routes_to_current_evidence_without_wide_interval_risk(self) -> None:
        report = build_objective_next_report(
            audio_package_report=audio_package(max_interval=9),
            output_dir="out",
            issue_number=694,
            expected_count=3,
            max_interval_threshold=12,
            max_added_note_ratio_review_threshold=0.75,
        )
        summary = validate_objective_next_report(
            report,
            expected_boundary=BOUNDARY,
            expected_next_boundary=CURRENT_EVIDENCE_NEXT_BOUNDARY,
            require_objective_decision=True,
            require_wide_interval_followup=False,
            require_no_quality_claim=True,
        )

        self.assertFalse(summary["wide_interval_followup_required"])
        self.assertTrue(summary["current_evidence_consolidation_ready"])

    def test_rejects_audio_quality_claim(self) -> None:
        with self.assertRaises(
            StageBMidiToSoloModelConditionedInputPathDeadAirTimingRepairObjectiveNextError
        ):
            build_objective_next_report(
                audio_package_report=audio_package(quality_claim=True),
                output_dir="out",
                issue_number=694,
                expected_count=3,
                max_interval_threshold=12,
                max_added_note_ratio_review_threshold=0.75,
            )

    def test_boundary_constants_are_stable(self) -> None:
        self.assertEqual(
            BOUNDARY,
            "stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_objective_next_decision",
        )
        self.assertEqual(
            PITCH_CONTOUR_NEXT_BOUNDARY,
            "stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_pitch_contour_decision",
        )


if __name__ == "__main__":
    unittest.main()
