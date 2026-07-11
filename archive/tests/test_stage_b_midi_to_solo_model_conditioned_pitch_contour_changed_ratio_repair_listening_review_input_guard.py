from __future__ import annotations

import unittest

from scripts.build_stage_b_midi_to_solo_model_conditioned_pitch_contour_changed_ratio_repair_listening_review_package import (
    BOUNDARY as SOURCE_BOUNDARY,
    NEXT_BOUNDARY as SOURCE_NEXT_BOUNDARY,
)
from scripts.guard_stage_b_midi_to_solo_model_conditioned_pitch_contour_changed_ratio_repair_listening_review_input import (
    BOUNDARY,
    FILL_BOUNDARY,
    OBJECTIVE_NEXT_BOUNDARY,
    StageBMidiToSoloPitchContourChangedRatioRepairListeningInputGuardError,
    build_listening_review_input_guard_report,
    validate_listening_review_input_guard_report,
)


SOURCE_SUMMARY = {
    "technical_wav_validation": True,
    "rendered_audio_file_count": 3,
    "max_repaired_interval": 12,
    "max_repaired_pitch_changed_ratio": 0.4348,
    "target_max_pitch_changed_ratio": 0.5,
    "audio_review_required": True,
}


def source_package(
    *,
    validated_review_input: bool = False,
    quality_claim: bool = False,
    source_summary: dict | None = None,
) -> dict:
    summary = source_summary if source_summary is not None else dict(SOURCE_SUMMARY)
    return {
        "boundary": SOURCE_BOUNDARY,
        "source_summary": dict(summary),
        "review_package": {
            "package_ready": True,
            "review_item_count": 3,
            "review_basis": "human_audio_listening_pending",
            "validated_review_input": validated_review_input,
            "required_input_fields": [
                "candidate_rank",
                "listening_status",
                "preference",
                "issue_notes",
            ],
        },
        "review_items": [
            {"rank": index, "wav_path": f"audio/rank_{index}.wav"}
            for index in range(1, 4)
        ],
        "readiness": {
            "boundary": SOURCE_BOUNDARY,
            "listening_review_package_ready": True,
            "review_item_count": 3,
            "validated_review_input": validated_review_input,
            "human_review_required_now": False,
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


class StageBMidiToSoloPitchContourChangedRatioRepairListeningInputGuardTest(
    unittest.TestCase
):
    def test_blocks_preference_fill_when_review_input_pending(self) -> None:
        report = build_listening_review_input_guard_report(
            source_package(),
            output_dir="out",
            issue_number=724,
        )
        summary = validate_listening_review_input_guard_report(
            report,
            expected_boundary=BOUNDARY,
            expected_next_boundary=OBJECTIVE_NEXT_BOUNDARY,
            require_guard_completed=True,
            require_preference_blocked=True,
            require_no_quality_claim=True,
        )

        self.assertFalse(summary["validated_review_input_present"])
        self.assertFalse(summary["preference_fill_allowed"])
        self.assertEqual(summary["review_item_count"], 3)
        self.assertEqual(summary["required_input_field_count"], 4)
        self.assertTrue(summary["technical_wav_validation"])
        self.assertEqual(summary["rendered_audio_file_count"], 3)
        self.assertEqual(summary["max_repaired_interval"], 12)
        self.assertAlmostEqual(summary["max_repaired_pitch_changed_ratio"], 0.4348)
        self.assertAlmostEqual(summary["target_max_pitch_changed_ratio"], 0.5)
        self.assertFalse(summary["human_audio_preference_claimed"])

    def test_routes_to_fill_when_validated_input_present(self) -> None:
        report = build_listening_review_input_guard_report(
            source_package(validated_review_input=True),
            output_dir="out",
            issue_number=724,
        )
        summary = validate_listening_review_input_guard_report(
            report,
            expected_boundary=BOUNDARY,
            expected_next_boundary=FILL_BOUNDARY,
            require_guard_completed=True,
            require_preference_blocked=False,
            require_no_quality_claim=True,
        )

        self.assertTrue(summary["validated_review_input_present"])
        self.assertTrue(summary["preference_fill_allowed"])

    def test_rejects_upstream_quality_claim(self) -> None:
        with self.assertRaises(
            StageBMidiToSoloPitchContourChangedRatioRepairListeningInputGuardError
        ):
            build_listening_review_input_guard_report(
                source_package(quality_claim=True),
                output_dir="out",
                issue_number=724,
            )

    def test_rejects_missing_technical_wav_validation(self) -> None:
        broken_summary = dict(SOURCE_SUMMARY)
        broken_summary["technical_wav_validation"] = False

        with self.assertRaises(
            StageBMidiToSoloPitchContourChangedRatioRepairListeningInputGuardError
        ):
            build_listening_review_input_guard_report(
                source_package(source_summary=broken_summary),
                output_dir="out",
                issue_number=724,
            )


if __name__ == "__main__":
    unittest.main()
