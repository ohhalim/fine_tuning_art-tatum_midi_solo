from __future__ import annotations

import unittest
from pathlib import Path

from scripts.summarize_stage_b_duration_coverage_user_listening_consolidation import (
    StageBDurationCoverageUserListeningConsolidationError,
    build_consolidation_report,
    validate_consolidation_report,
)


def midi_evidence_consolidation() -> dict:
    return {
        "schema_version": "stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_midi_evidence_consolidation_v1",
        "candidate_id": "duration_fill_candidate",
        "midi_evidence_summary": {
            "preference": "duration_coverage_fill_keep",
            "review_basis": "midi_metric_and_note_structure",
            "source_score": 91.857143,
            "fill_score": 171.588235,
            "score_delta_fill_minus_source": 79.731092,
            "dead_air_delta_fill_minus_source": -0.277311,
            "focused_note_count_delta_fill_minus_source": 6,
            "focused_unique_pitch_count_delta_fill_minus_source": 6,
            "max_simultaneous_notes_delta_fill_minus_source": -1,
        },
        "claim_boundary": {
            "boundary": "midi_evidence_preference_support",
            "midi_evidence_preference_claimed": True,
            "human_audio_preference_claimed": False,
        },
    }


def audio_render_attempt() -> dict:
    return {
        "schema_version": "stage_b_duration_coverage_fill_local_audio_render_attempt_v1",
        "candidate_id": "duration_fill_candidate",
        "audio_render_boundary": {
            "render_attempted": True,
            "technical_wav_validation": True,
            "rendered_audio_file_count": 2,
        },
        "rendered_audio_files": [
            {
                "role": "source_constrained_partial",
                "wav_file": {
                    "sample_rate": 44100,
                    "duration_seconds": 6.474,
                },
            },
            {
                "role": "duration_coverage_fill_keep",
                "wav_file": {
                    "sample_rate": 44100,
                    "duration_seconds": 6.474,
                },
            },
        ],
    }


def user_listening_review_fill(*, preference: str = "duration_coverage_fill_keep") -> dict:
    return {
        "schema_version": "stage_b_duration_coverage_fill_user_listening_review_fill_v1",
        "candidate_id": "duration_fill_candidate",
        "user_listening_review": {
            "review_basis": "user_listening_review_of_rendered_wav",
            "preference": preference,
            "timing": "duration_coverage_fill_keep",
            "phrase": "duration_coverage_fill_keep",
            "vocabulary": "duration_coverage_fill_keep",
            "source_assessment": "source sounds like random notes and is hard to understand",
            "fill_assessment": "fill sounds much more jazz-like as soloing",
            "notes": "single user listening review",
        },
        "claim_boundary": {
            "human_audio_preference_claimed": True,
            "single_user_review": True,
            "audio_rendered_quality_claimed": False,
            "broad_model_quality_claimed": False,
        },
    }


class StageBDurationCoverageFillUserListeningConsolidationTest(unittest.TestCase):
    def test_consolidates_aligned_midi_audio_and_user_evidence(self) -> None:
        report = build_consolidation_report(
            midi_evidence_consolidation(),
            audio_render_attempt(),
            user_listening_review_fill(),
            output_dir=Path("outputs/consolidation"),
        )
        summary = validate_consolidation_report(
            report,
            expected_boundary="midi_evidence_and_single_user_listening_support_duration_coverage_fill_keep",
            expected_preferred_candidate="duration_coverage_fill_keep",
            require_no_broad_quality_claim=True,
        )

        self.assertTrue(summary["same_preferred_candidate"])
        self.assertTrue(summary["single_user_review"])
        self.assertEqual(summary["rendered_audio_file_count"], 2)
        self.assertFalse(summary["broad_model_quality_claimed"])
        self.assertIn("multi_reviewer_preference", report["not_proven"])

    def test_rejects_mismatched_user_preference(self) -> None:
        with self.assertRaises(StageBDurationCoverageUserListeningConsolidationError):
            build_consolidation_report(
                midi_evidence_consolidation(),
                audio_render_attempt(),
                user_listening_review_fill(preference="source_constrained_partial"),
                output_dir=Path("outputs/consolidation"),
            )


if __name__ == "__main__":
    unittest.main()
