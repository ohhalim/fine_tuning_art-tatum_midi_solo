from __future__ import annotations

import unittest
from pathlib import Path

from scripts.summarize_stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_midi_evidence_consolidation import (
    DurationCoverageFillMidiEvidenceConsolidationError,
    build_consolidation_report,
    validate_consolidation,
)


def midi_evidence_review(*, human_audio_preference_claimed: bool = False) -> dict:
    return {
        "schema_version": "stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_midi_evidence_review_v1",
        "candidate_id": "duration_fill_candidate",
        "review_basis": "midi_metric_and_note_structure",
        "score": {
            "source_constrained_partial": 91.857143,
            "duration_coverage_fill_keep": 171.588235,
            "score_delta_fill_minus_source": 79.731092,
        },
        "midi_evidence_review": {
            "status": "reviewed",
            "preference": "duration_coverage_fill_keep",
        },
        "metric_delta": {
            "dead_air_delta_fill_minus_source": -0.277311,
            "focused_note_count_delta_fill_minus_source": 6,
            "focused_unique_pitch_count_delta_fill_minus_source": 6,
            "max_simultaneous_notes_delta_fill_minus_source": -1,
        },
        "claim_boundary": {
            "midi_evidence_preference_claimed": True,
            "human_audio_preference_claimed": human_audio_preference_claimed,
            "audio_render_used": False,
            "not_human_audio_review": True,
        },
    }


class StageBMarginRecoveredPhraseVocabularyDurationCoverageFillMidiEvidenceConsolidationTest(unittest.TestCase):
    def test_consolidates_midi_evidence_boundary(self) -> None:
        report = build_consolidation_report(
            midi_evidence_review(),
            output_dir=Path("outputs/consolidation"),
        )
        summary = validate_consolidation(
            report,
            expected_candidate_id="duration_fill_candidate",
            expected_boundary="midi_evidence_preference_support",
            require_no_human_audio_preference=True,
        )

        self.assertEqual(summary["preference"], "duration_coverage_fill_keep")
        self.assertEqual(summary["boundary"], "midi_evidence_preference_support")
        self.assertGreater(summary["score_delta_fill_minus_source"], 0.0)
        self.assertLess(summary["dead_air_delta_fill_minus_source"], 0.0)
        self.assertFalse(summary["human_audio_preference_claimed"])
        self.assertIn("human_audio_preference", report["not_proven"])

    def test_rejects_human_audio_preference_claim(self) -> None:
        with self.assertRaises(DurationCoverageFillMidiEvidenceConsolidationError):
            build_consolidation_report(
                midi_evidence_review(human_audio_preference_claimed=True),
                output_dir=Path("outputs/consolidation"),
            )


if __name__ == "__main__":
    unittest.main()
