from __future__ import annotations

import unittest

from scripts.decide_stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_decision import (
    BOUNDARY,
    NEXT_BOUNDARY,
    StageBMidiToSoloModelConditionedInputPathDeadAirTimingRepairDecisionError,
    build_dead_air_timing_repair_decision_report,
    validate_dead_air_timing_repair_decision_report,
)
from scripts.decide_stage_b_midi_to_solo_model_conditioned_input_path_objective_next import (
    BOUNDARY as SOURCE_BOUNDARY,
    REPAIR_NEXT_BOUNDARY as SOURCE_NEXT_BOUNDARY,
)


def objective_next(
    *,
    repair_required: bool = True,
    quality_claim: bool = False,
) -> dict:
    return {
        "boundary": SOURCE_BOUNDARY,
        "objective_summary": {
            "candidate_count": 3,
            "exported_candidate_count": 3,
            "rendered_audio_file_count": 3,
            "technical_wav_validation": True,
            "dead_air_threshold": 0.5,
            "dead_air_failure_count": 3 if repair_required else 0,
            "all_candidates_dead_air_failure": repair_required,
            "dead_air_min": 0.6522 if repair_required else 0.2,
            "dead_air_max": 0.6522 if repair_required else 0.2,
            "best_note_count": 24,
            "best_unique_pitch_count": 20,
            "validated_review_input_present": False,
            "preference_fill_allowed": False,
        },
        "candidate_reviews": [
            {
                "rank": index,
                "dead_air_ratio": 0.6522,
            }
            for index in range(1, 4)
        ],
        "readiness": {
            "objective_only_next_decision_completed": True,
            "dead_air_timing_repair_required": repair_required,
            "current_evidence_consolidation_ready": not repair_required,
            "human_audio_preference_claimed": quality_claim,
            "midi_to_solo_musical_quality_claimed": False,
            "audio_rendered_quality_claimed": False,
            "model_checkpoint_generation_quality_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
            "production_ready_claimed": False,
        },
        "decision": {
            "next_boundary": SOURCE_NEXT_BOUNDARY,
            "critical_user_input_required": False,
        },
    }


class StageBMidiToSoloModelConditionedInputPathDeadAirTimingRepairDecisionTest(unittest.TestCase):
    def test_defines_dead_air_timing_repair_target(self) -> None:
        report = build_dead_air_timing_repair_decision_report(
            objective_next_report=objective_next(),
            output_dir="out",
            issue_number=688,
            target_dead_air_max=0.35,
            max_postprocess_removal_ratio=0.25,
        )
        summary = validate_dead_air_timing_repair_decision_report(
            report,
            expected_boundary=BOUNDARY,
            expected_next_boundary=NEXT_BOUNDARY,
            require_decision_completed=True,
            require_repair_probe=True,
            require_no_quality_claim=True,
        )

        self.assertTrue(summary["dead_air_timing_repair_decision_completed"])
        self.assertTrue(summary["repair_probe_required"])
        self.assertEqual(summary["selected_target"], "dead_air_timing_continuity")
        self.assertEqual(summary["source_dead_air_failure_count"], 3)
        self.assertGreater(summary["required_dead_air_gain_min"], 0.3)
        self.assertFalse(summary["human_audio_preference_claimed"])

    def test_rejects_source_without_repair_requirement(self) -> None:
        with self.assertRaises(
            StageBMidiToSoloModelConditionedInputPathDeadAirTimingRepairDecisionError
        ):
            build_dead_air_timing_repair_decision_report(
                objective_next_report=objective_next(repair_required=False),
                output_dir="out",
                issue_number=688,
                target_dead_air_max=0.35,
                max_postprocess_removal_ratio=0.25,
            )

    def test_rejects_upstream_quality_claim(self) -> None:
        with self.assertRaises(
            StageBMidiToSoloModelConditionedInputPathDeadAirTimingRepairDecisionError
        ):
            build_dead_air_timing_repair_decision_report(
                objective_next_report=objective_next(quality_claim=True),
                output_dir="out",
                issue_number=688,
                target_dead_air_max=0.35,
                max_postprocess_removal_ratio=0.25,
            )


if __name__ == "__main__":
    unittest.main()
