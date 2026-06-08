from __future__ import annotations

import unittest

from scripts.build_stage_b_midi_to_solo_model_conditioned_input_path_listening_review_package import (
    BOUNDARY as SOURCE_BOUNDARY,
    NEXT_BOUNDARY as SOURCE_NEXT_BOUNDARY,
)
from scripts.guard_stage_b_midi_to_solo_model_conditioned_input_path_listening_review_input import (
    BOUNDARY,
    FILL_BOUNDARY,
    OBJECTIVE_NEXT_BOUNDARY,
    StageBMidiToSoloModelConditionedInputPathListeningInputGuardError,
    build_listening_review_input_guard_report,
    validate_listening_review_input_guard_report,
)


SOURCE_EVIDENCE = {
    "phrase_bank_cli_technical_path_completed": True,
    "cli_candidate_count": 3,
    "cli_rendered_audio_file_count": 3,
    "cli_input_context_bars": 228,
    "cli_preference_fill_allowed": False,
}


def source_package(
    *,
    validated_review_input: bool = False,
    quality_claim: bool = False,
    source_evidence: dict | None = None,
) -> dict:
    evidence = source_evidence if source_evidence is not None else dict(SOURCE_EVIDENCE)
    return {
        "boundary": SOURCE_BOUNDARY,
        "replacement_source": {
            "source_evidence": dict(evidence),
        },
        "review_package": {
            "package_ready": True,
            "review_item_count": 3,
            "review_basis": "human_audio_listening_pending",
            "validated_review_input": validated_review_input,
            "source_evidence": dict(evidence),
            "required_input_fields": [
                "candidate_rank",
                "listening_status",
                "preference",
                "issue_notes",
            ],
        },
        "review_items": [
            {"rank": index, "wav_path": f"audio/rank_{index}.wav"} for index in range(1, 4)
        ],
        "readiness": {
            "listening_review_package_ready": True,
            "review_item_count": 3,
            "validated_review_input": validated_review_input,
            "human_review_required_now": False,
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


class StageBMidiToSoloModelConditionedInputPathListeningInputGuardTest(unittest.TestCase):
    def test_blocks_preference_fill_when_review_input_pending(self) -> None:
        report = build_listening_review_input_guard_report(
            source_package(),
            output_dir="out",
            issue_number=684,
        )
        summary = validate_listening_review_input_guard_report(
            report,
            expected_boundary=BOUNDARY,
            expected_next_boundary=OBJECTIVE_NEXT_BOUNDARY,
            require_guard_completed=True,
            require_pending_input=True,
            require_no_quality_claim=True,
        )

        self.assertFalse(summary["validated_review_input_present"])
        self.assertFalse(summary["preference_fill_allowed"])
        self.assertEqual(summary["review_item_count"], 3)
        self.assertEqual(summary["required_input_field_count"], 4)
        self.assertTrue(summary["phrase_bank_cli_technical_path_completed"])
        self.assertEqual(summary["cli_input_context_bars"], 228)
        self.assertFalse(summary["human_audio_preference_claimed"])

    def test_routes_to_fill_when_validated_input_present(self) -> None:
        report = build_listening_review_input_guard_report(
            source_package(validated_review_input=True),
            output_dir="out",
            issue_number=684,
        )
        summary = validate_listening_review_input_guard_report(
            report,
            expected_boundary=BOUNDARY,
            expected_next_boundary=FILL_BOUNDARY,
            require_guard_completed=True,
            require_pending_input=False,
            require_no_quality_claim=True,
        )

        self.assertTrue(summary["validated_review_input_present"])
        self.assertTrue(summary["preference_fill_allowed"])

    def test_rejects_upstream_quality_claim(self) -> None:
        with self.assertRaises(StageBMidiToSoloModelConditionedInputPathListeningInputGuardError):
            build_listening_review_input_guard_report(
                source_package(quality_claim=True),
                output_dir="out",
                issue_number=684,
            )

    def test_rejects_missing_cli_source_evidence(self) -> None:
        broken_evidence = dict(SOURCE_EVIDENCE)
        broken_evidence["phrase_bank_cli_technical_path_completed"] = False

        with self.assertRaises(StageBMidiToSoloModelConditionedInputPathListeningInputGuardError):
            build_listening_review_input_guard_report(
                source_package(source_evidence=broken_evidence),
                output_dir="out",
                issue_number=684,
            )


if __name__ == "__main__":
    unittest.main()
