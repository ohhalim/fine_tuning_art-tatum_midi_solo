from __future__ import annotations

import unittest
from pathlib import Path

from scripts.summarize_stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_external_human_audio_boundary import (
    DurationCoverageFillExternalHumanAudioBoundaryError,
    build_external_boundary_report,
    validate_external_boundary,
)


def midi_evidence_consolidation(
    *,
    source_boundary: str = "midi_evidence_preference_support",
    human_audio_preference_claimed: bool = False,
) -> dict:
    return {
        "schema_version": "stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_midi_evidence_consolidation_v1",
        "candidate_id": "duration_fill_candidate",
        "midi_evidence_summary": {
            "preference": "duration_coverage_fill_keep",
            "source_score": 91.857143,
            "fill_score": 171.588235,
            "score_delta_fill_minus_source": 79.731092,
            "dead_air_delta_fill_minus_source": -0.277311,
            "focused_note_count_delta_fill_minus_source": 6,
            "focused_unique_pitch_count_delta_fill_minus_source": 6,
            "max_simultaneous_notes_delta_fill_minus_source": -1,
        },
        "claim_boundary": {
            "boundary": source_boundary,
            "midi_evidence_preference_claimed": True,
            "human_audio_preference_claimed": human_audio_preference_claimed,
            "audio_render_used": False,
        },
    }


class StageBMarginRecoveredPhraseVocabularyDurationCoverageFillExternalHumanAudioBoundaryTest(unittest.TestCase):
    def test_summarizes_external_boundary_without_human_audio_claim(self) -> None:
        report = build_external_boundary_report(
            midi_evidence_consolidation(),
            output_dir=Path("outputs/external_boundary"),
        )
        summary = validate_external_boundary(
            report,
            expected_candidate_id="duration_fill_candidate",
            expected_boundary="external_human_audio_review_required_for_human_preference_claim",
            require_no_human_audio_preference=True,
            require_pending_external_review=True,
        )

        self.assertEqual(summary["source_boundary"], "midi_evidence_preference_support")
        self.assertEqual(
            summary["external_boundary"],
            "external_human_audio_review_required_for_human_preference_claim",
        )
        self.assertEqual(summary["external_review_status"], "pending_external_review_input")
        self.assertEqual(summary["midi_evidence_preference"], "duration_coverage_fill_keep")
        self.assertFalse(summary["human_audio_preference_claimed"])
        self.assertGreater(summary["score_delta_fill_minus_source"], 0.0)
        self.assertIn("human_audio_preference", report["not_proven"])
        self.assertTrue(report["required_review_input"]["audio_render_used"]["required"])

    def test_rejects_source_human_audio_preference_claim(self) -> None:
        with self.assertRaises(DurationCoverageFillExternalHumanAudioBoundaryError):
            build_external_boundary_report(
                midi_evidence_consolidation(human_audio_preference_claimed=True),
                output_dir=Path("outputs/external_boundary"),
            )

    def test_rejects_unexpected_source_boundary(self) -> None:
        with self.assertRaises(DurationCoverageFillExternalHumanAudioBoundaryError):
            build_external_boundary_report(
                midi_evidence_consolidation(source_boundary="pending_human_audio_review"),
                output_dir=Path("outputs/external_boundary"),
            )


if __name__ == "__main__":
    unittest.main()
