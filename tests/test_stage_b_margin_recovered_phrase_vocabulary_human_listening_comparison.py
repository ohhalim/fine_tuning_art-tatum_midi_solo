from __future__ import annotations

import unittest
from pathlib import Path

from scripts.build_stage_b_margin_recovered_phrase_vocabulary_human_listening_comparison import (
    PhraseVocabularyHumanListeningComparisonError,
    build_human_listening_comparison,
    validate_human_listening_comparison,
)


def filled_notes(candidate_id: str, *, source: str, pitch: int = 72) -> dict:
    return {
        "schema_version": "stage_b_margin_recovered_phrase_vocabulary_focused_listening_fill_v1",
        "candidates": [
            {
                "candidate_id": candidate_id,
                "review_metadata": {
                    "source_run_id": source,
                    "sample_index": 25,
                    "sample_seed": 85,
                },
                "review_files": {
                    "midi_path": f"outputs/{candidate_id}.mid",
                    "context_midi_path": f"outputs/{candidate_id}_context.mid",
                    "source_midi_path": f"outputs/{candidate_id}_source.mid",
                },
                "listening": {
                    "decision": "keep",
                    "timing": "acceptable",
                    "chord_fit": "strong",
                    "phrase_continuation": "acceptable",
                    "landing": "strong",
                    "jazz_vocabulary": "acceptable",
                },
                "focused_context_metrics": {
                    "note_count": 1,
                    "unique_pitch_count": 1,
                    "range": "C5-C5",
                    "phrase_span_beats": 1.0,
                    "dead_air_ratio": 0.0,
                    "sustained_coverage_ratio": 1.0,
                    "adjacent_pitch_repeats": 0,
                    "max_interval": 0,
                    "final_note": "C5",
                    "final_chord": "Fm7",
                    "final_note_role": "chord_tone",
                },
                "objective_first_16_notes": [
                    {
                        "pitch": pitch,
                        "start_sec": 0.0,
                        "end_sec": 0.25,
                        "velocity": 71,
                    }
                ],
                "review_risks": [],
                "listening_fill_evidence": {"not_human_audio_review": True},
            }
        ],
    }


class StageBMarginRecoveredPhraseVocabularyHumanListeningComparisonTest(unittest.TestCase):
    def test_builds_pending_boundary_for_same_note_sequence(self) -> None:
        two_candidate_keep = {
            "schema_version": "stage_b_margin_recovered_phrase_vocabulary_two_candidate_keep_v1",
            "keep_candidates": [
                {"candidate_id": "selected"},
                {"candidate_id": "peer"},
            ],
        }

        report = build_human_listening_comparison(
            two_candidate_keep,
            filled_notes("selected", source="seed43"),
            filled_notes("peer", source="seed61"),
            output_dir=Path("outputs/human_boundary"),
        )
        summary = validate_human_listening_comparison(
            report,
            min_candidates=2,
            require_pending=True,
            require_no_preference=True,
            expect_note_sequence_match=True,
        )

        self.assertEqual(summary["candidate_count"], 2)
        self.assertTrue(summary["note_sequence_match"])
        self.assertEqual(summary["boundary"], "pending_human_review_same_midi_content")
        self.assertEqual(report["candidates"][0]["human_listening"]["status"], "pending")
        self.assertFalse(report["human_listening_boundary"]["preference_claimed"])

    def test_rejects_unexpected_note_sequence_match(self) -> None:
        two_candidate_keep = {
            "keep_candidates": [
                {"candidate_id": "selected"},
                {"candidate_id": "peer"},
            ],
        }
        report = build_human_listening_comparison(
            two_candidate_keep,
            filled_notes("selected", source="seed43", pitch=72),
            filled_notes("peer", source="seed61", pitch=73),
            output_dir=Path("outputs/human_boundary"),
        )

        with self.assertRaises(PhraseVocabularyHumanListeningComparisonError):
            validate_human_listening_comparison(
                report,
                min_candidates=2,
                require_pending=True,
                require_no_preference=True,
                expect_note_sequence_match=True,
            )


if __name__ == "__main__":
    unittest.main()
