from __future__ import annotations

import unittest
from pathlib import Path

from scripts.review_stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_midi_evidence import (
    DurationCoverageFillMidiEvidenceReviewError,
    build_midi_evidence_review,
    validate_midi_evidence_review,
)


def audio_review_package(*, preference_claimed: bool = False) -> dict:
    return {
        "schema_version": "stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_audio_review_package_v1",
        "candidate_id": "duration_fill_candidate",
        "package_boundary": {
            "preference_claimed": preference_claimed,
        },
        "review_items": [
            {
                "role": "source_constrained_partial",
                "candidate_id": "source_candidate",
                "metric_summary": {
                    "note_count": 15,
                    "focused_note_count": 12,
                    "unique_pitch_count": 10,
                    "focused_unique_pitch_count": 9,
                    "dead_air_ratio": 0.5714285714285714,
                    "max_simultaneous_notes": 2,
                    "focused_max_simultaneous_notes": 1,
                    "adjacent_pitch_repeats": 0,
                    "duplicated_3_note_pitch_class_chunks": 0,
                    "max_interval": 7,
                },
            },
            {
                "role": "duration_coverage_fill_keep",
                "candidate_id": "duration_fill_candidate",
                "metric_summary": {
                    "note_count": 18,
                    "focused_note_count": 18,
                    "unique_pitch_count": 15,
                    "focused_unique_pitch_count": 15,
                    "dead_air_ratio": 0.29411764705882354,
                    "max_simultaneous_notes": 1,
                    "focused_max_simultaneous_notes": 1,
                    "adjacent_pitch_repeats": 0,
                    "duplicated_3_note_pitch_class_chunks": 0,
                    "max_interval": 7,
                },
            },
        ],
    }


class StageBMarginRecoveredPhraseVocabularyDurationCoverageFillMidiEvidenceTest(unittest.TestCase):
    def test_prefers_duration_fill_from_midi_metrics_without_human_audio_claim(self) -> None:
        report = build_midi_evidence_review(
            audio_review_package(),
            output_dir=Path("outputs/midi_evidence"),
        )
        summary = validate_midi_evidence_review(
            report,
            expected_candidate_id="duration_fill_candidate",
            expected_preference="duration_coverage_fill_keep",
            require_no_human_audio_preference=True,
            require_audio_not_rendered=True,
        )

        self.assertEqual(summary["preference"], "duration_coverage_fill_keep")
        self.assertGreater(summary["score_delta_fill_minus_source"], 0.0)
        self.assertLess(summary["dead_air_delta_fill_minus_source"], 0.0)
        self.assertFalse(summary["human_audio_preference_claimed"])
        self.assertFalse(summary["audio_render_used"])
        self.assertIn("human_audio_preference", report["not_proven"])

    def test_rejects_package_with_existing_preference_claim(self) -> None:
        with self.assertRaises(DurationCoverageFillMidiEvidenceReviewError):
            build_midi_evidence_review(
                audio_review_package(preference_claimed=True),
                output_dir=Path("outputs/midi_evidence"),
            )


if __name__ == "__main__":
    unittest.main()
