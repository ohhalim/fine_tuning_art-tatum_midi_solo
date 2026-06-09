from __future__ import annotations

import unittest

from scripts.decide_stage_b_midi_to_solo_model_conditioned_pitch_contour_changed_ratio_repair_objective_next import (
    BOUNDARY,
    CHANGED_RATIO_REPAIR_RETRY_NEXT_BOUNDARY,
    CURRENT_EVIDENCE_NEXT_BOUNDARY,
    StageBMidiToSoloPitchContourChangedRatioRepairObjectiveNextError,
    build_objective_next_report,
    validate_objective_next_report,
)
from scripts.guard_stage_b_midi_to_solo_model_conditioned_pitch_contour_changed_ratio_repair_listening_review_input import (
    BOUNDARY as SOURCE_BOUNDARY,
    OBJECTIVE_NEXT_BOUNDARY as SOURCE_NEXT_BOUNDARY,
)


SOURCE_SUMMARY = {
    "technical_wav_validation": True,
    "rendered_audio_file_count": 3,
    "max_repaired_interval": 12,
    "max_repaired_pitch_changed_ratio": 0.4348,
    "target_max_pitch_changed_ratio": 0.5,
    "audio_review_required": True,
}


def input_guard(
    *,
    preference_fill_allowed: bool = False,
    quality_claim: bool = False,
    source_summary: dict | None = None,
) -> dict:
    summary = source_summary if source_summary is not None else dict(SOURCE_SUMMARY)
    return {
        "boundary": SOURCE_BOUNDARY,
        "guard_result": {
            "validated_review_input_present": False,
            "preference_fill_allowed": preference_fill_allowed,
            "review_item_count": 3,
            "required_input_field_count": 4,
            "source_summary": dict(summary),
        },
        "readiness": {
            "listening_review_input_guard_completed": True,
            "human_audio_preference_claimed": quality_claim,
            "midi_to_solo_musical_quality_claimed": False,
            "audio_rendered_quality_claimed": False,
            "model_checkpoint_generation_quality_claimed": False,
            "model_direct_generation_quality_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
            "production_ready_claimed": False,
        },
        "decision": {
            "next_boundary": SOURCE_NEXT_BOUNDARY,
            "critical_user_input_required": False,
        },
    }


class StageBMidiToSoloPitchContourChangedRatioRepairObjectiveNextTest(
    unittest.TestCase
):
    def test_routes_to_current_evidence_when_changed_ratio_targets_pass(self) -> None:
        report = build_objective_next_report(
            input_guard_report=input_guard(),
            output_dir="out",
            issue_number=726,
            max_interval_threshold=12,
            max_pitch_changed_ratio=0.5,
        )
        summary = validate_objective_next_report(
            report,
            expected_boundary=BOUNDARY,
            expected_next_boundary=CURRENT_EVIDENCE_NEXT_BOUNDARY,
            require_objective_support=True,
            require_current_evidence_ready=True,
            require_no_quality_claim=True,
        )

        self.assertTrue(summary["objective_next_completed"])
        self.assertTrue(summary["technical_wav_validation"])
        self.assertEqual(summary["rendered_audio_file_count"], 3)
        self.assertEqual(summary["max_repaired_interval"], 12)
        self.assertTrue(summary["interval_target_supported"])
        self.assertAlmostEqual(summary["max_repaired_pitch_changed_ratio"], 0.4348)
        self.assertAlmostEqual(summary["target_max_pitch_changed_ratio"], 0.5)
        self.assertTrue(summary["changed_ratio_target_supported"])
        self.assertTrue(summary["changed_ratio_repair_objective_path_supported"])
        self.assertTrue(summary["current_evidence_consolidation_ready"])
        self.assertFalse(summary["preference_fill_allowed"])
        self.assertFalse(summary["human_audio_preference_claimed"])

    def test_routes_to_retry_when_changed_ratio_target_fails(self) -> None:
        broken_summary = dict(SOURCE_SUMMARY)
        broken_summary["max_repaired_pitch_changed_ratio"] = 0.6522
        report = build_objective_next_report(
            input_guard_report=input_guard(source_summary=broken_summary),
            output_dir="out",
            issue_number=726,
            max_interval_threshold=12,
            max_pitch_changed_ratio=0.5,
        )
        summary = validate_objective_next_report(
            report,
            expected_boundary=BOUNDARY,
            expected_next_boundary=CHANGED_RATIO_REPAIR_RETRY_NEXT_BOUNDARY,
            require_objective_support=False,
            require_current_evidence_ready=False,
            require_no_quality_claim=True,
        )

        self.assertFalse(summary["changed_ratio_target_supported"])
        self.assertFalse(summary["changed_ratio_repair_objective_path_supported"])
        self.assertFalse(summary["current_evidence_consolidation_ready"])

    def test_rejects_preference_fill_allowed(self) -> None:
        with self.assertRaises(StageBMidiToSoloPitchContourChangedRatioRepairObjectiveNextError):
            build_objective_next_report(
                input_guard_report=input_guard(preference_fill_allowed=True),
                output_dir="out",
                issue_number=726,
                max_interval_threshold=12,
                max_pitch_changed_ratio=0.5,
            )

    def test_rejects_upstream_quality_claim(self) -> None:
        with self.assertRaises(StageBMidiToSoloPitchContourChangedRatioRepairObjectiveNextError):
            build_objective_next_report(
                input_guard_report=input_guard(quality_claim=True),
                output_dir="out",
                issue_number=726,
                max_interval_threshold=12,
                max_pitch_changed_ratio=0.5,
            )


if __name__ == "__main__":
    unittest.main()
