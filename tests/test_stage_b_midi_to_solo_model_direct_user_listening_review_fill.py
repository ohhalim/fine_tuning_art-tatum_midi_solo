from __future__ import annotations

import unittest
from pathlib import Path

from scripts.fill_stage_b_midi_to_solo_model_direct_user_listening_review import (
    BOUNDARY,
    CLAIM_BOUNDARY,
    NEXT_BOUNDARY,
    StageBMidiToSoloModelDirectUserListeningReviewFillError,
    build_user_listening_review_fill,
    validate_user_listening_review_fill,
)


def listening_review_package(*, preference_claimed: bool = False) -> dict:
    return {
        "schema_version": "stage_b_midi_to_solo_model_direct_listening_review_package_v1",
        "listening_review_package_boundary": {
            "boundary": "stage_b_midi_to_solo_model_direct_listening_review_package",
            "source_boundary": "stage_b_midi_to_solo_model_direct_timing_phrase_repair",
            "candidate_count": 3,
            "midi_file_count": 3,
            "rendered_audio_file_count": 3,
            "technical_wav_validation": True,
            "review_input_template_written": True,
            "listening_review_completed": False,
            "human_audio_preference_claimed": preference_claimed,
            "model_direct_generation_quality_claimed": False,
            "midi_to_solo_musical_quality_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
        },
        "decision": {
            "next_boundary": "stage_b_midi_to_solo_model_direct_user_listening_review_fill",
            "critical_user_input_required": False,
        },
        "rendered_audio_files": [
            {
                "rank": 1,
                "sample_index": 1,
                "package_midi_path": "rank_01.mid",
                "source_note_count": 32,
                "source_unique_pitch_count": 13,
                "source_max_interval": 9,
                "source_dead_air_ratio": 0.2258,
                "source_diagnostic_flags": [],
                "wav_file": {
                    "path": "rank_01.wav",
                    "exists": True,
                    "duration_seconds": 19.030,
                    "sample_rate": 44100,
                    "sha256": "a" * 64,
                },
            },
            {
                "rank": 2,
                "sample_index": 2,
                "package_midi_path": "rank_02.mid",
                "source_note_count": 32,
                "source_unique_pitch_count": 15,
                "source_max_interval": 9,
                "source_dead_air_ratio": 0.2258,
                "source_diagnostic_flags": [],
                "wav_file": {
                    "path": "rank_02.wav",
                    "exists": True,
                    "duration_seconds": 19.001,
                    "sample_rate": 44100,
                    "sha256": "b" * 64,
                },
            },
            {
                "rank": 3,
                "sample_index": 3,
                "package_midi_path": "rank_03.mid",
                "source_note_count": 32,
                "source_unique_pitch_count": 14,
                "source_max_interval": 8,
                "source_dead_air_ratio": 0.2258,
                "source_diagnostic_flags": [],
                "wav_file": {
                    "path": "rank_03.wav",
                    "exists": True,
                    "duration_seconds": 18.926,
                    "sample_rate": 44100,
                    "sha256": "c" * 64,
                },
            },
        ],
    }


def input_guard_report() -> dict:
    return {
        "boundary": "stage_b_midi_to_solo_model_direct_user_listening_review_input_guard",
        "guard_result": {
            "validated_review_input_present": False,
            "preference_fill_allowed": False,
            "pending_status_field_count": 4,
            "pending_candidate_decision_count": 3,
            "pending_candidate_field_count": 9,
        },
        "readiness": {
            "human_audio_preference_claimed": False,
        },
    }


class StageBMidiToSoloModelDirectUserListeningReviewFillTest(unittest.TestCase):
    def test_records_rank_three_relative_best_reject_all_without_quality_claim(self) -> None:
        report = build_user_listening_review_fill(
            listening_review_package(),
            output_dir=Path("outputs/review_fill"),
            reviewer="user",
            preferred_rank=3,
            overall_decision="reject_all",
            candidate_decision="relative_best_needs_followup",
            timing="songlike_not_soloing",
            phrase="songlike_not_soloing",
            vocabulary="songlike_not_soloing",
            primary_failure="songlike_melody_not_soloing",
            assessment="rank 3 is relatively best, but all candidates sound like simple song melody rather than jazz soloing",
            notes="single user listening review of three rendered WAV files",
            guard_report=input_guard_report(),
        )
        summary = validate_user_listening_review_fill(
            report,
            expected_boundary=BOUNDARY,
            expected_next_boundary=NEXT_BOUNDARY,
            expected_overall_decision="reject_all",
            expected_preferred_rank=3,
            require_review_completed=True,
            require_no_keep_claim=True,
            require_no_quality_claim=True,
        )

        self.assertEqual(summary["claim_boundary"], CLAIM_BOUNDARY)
        self.assertEqual(summary["reviewed_candidate_count"], 3)
        self.assertEqual(summary["candidate_decision_for_preferred_rank"], "relative_best_needs_followup")
        self.assertEqual(summary["primary_failure"], "songlike_melody_not_soloing")
        self.assertFalse(summary["human_audio_preference_claimed"])
        self.assertFalse(summary["midi_to_solo_musical_quality_claimed"])
        candidate_reviews = report["user_listening_review"]["candidate_reviews"]
        self.assertTrue(candidate_reviews[2]["relative_best"])
        self.assertEqual(candidate_reviews[2]["musical_acceptance"], "reject")

    def test_rejects_source_package_with_existing_preference_claim(self) -> None:
        with self.assertRaises(StageBMidiToSoloModelDirectUserListeningReviewFillError):
            build_user_listening_review_fill(
                listening_review_package(preference_claimed=True),
                output_dir=Path("outputs/review_fill"),
                reviewer="user",
                preferred_rank=3,
                overall_decision="reject_all",
                candidate_decision="relative_best_needs_followup",
                timing="songlike_not_soloing",
                phrase="songlike_not_soloing",
                vocabulary="songlike_not_soloing",
                primary_failure="songlike_melody_not_soloing",
                assessment="all candidates need follow-up",
                notes="",
            )


if __name__ == "__main__":
    unittest.main()
