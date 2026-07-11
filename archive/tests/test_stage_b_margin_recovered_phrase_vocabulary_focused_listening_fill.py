from __future__ import annotations

import unittest

from scripts.fill_stage_b_margin_recovered_phrase_vocabulary_focused_listening_notes import (
    fill_notes,
    validate_fill,
)


class StageBMarginRecoveredPhraseVocabularyFocusedListeningFillTest(unittest.TestCase):
    def sample_notes(self) -> dict:
        return {
            "schema_version": "stage_b_margin_recovered_phrase_vocabulary_focused_listening_notes_v1",
            "candidates": [
                {
                    "candidate_id": "phrase_vocabulary_candidate",
                    "review_files": {
                        "midi_path": "outputs/phrase_vocab.mid",
                        "context_midi_path": "outputs/phrase_vocab_context.mid",
                    },
                    "proxy_review": {
                        "decision": "keep_for_focused_listening",
                    },
                    "focused_context_metrics": {
                        "note_count": 13,
                        "unique_pitch_count": 8,
                        "phrase_span_beats": 7.0,
                        "dead_air_ratio": 0.33333333333333337,
                        "onset_coverage_ratio": 0.5,
                        "sustained_coverage_ratio": 0.59375,
                        "adjacent_pitch_repeats": 0,
                        "duplicated_3_note_pitch_class_chunks": 0,
                        "max_simultaneous_notes": 1,
                        "max_interval": 7,
                        "final_note": "C5",
                        "final_chord": "Fm7",
                        "final_note_role": "chord_tone",
                    },
                    "review_risks": ["sustained_coverage_review"],
                    "listening": {
                        "status": "pending",
                        "timing": "pending",
                        "chord_fit": "pending",
                        "phrase_continuation": "pending",
                        "landing": "pending",
                        "jazz_vocabulary": "pending",
                        "decision": "pending",
                        "notes": "",
                    },
                }
            ],
        }

    def test_fill_marks_repaired_candidate_keep_with_sustained_coverage_evidence(self) -> None:
        filled = fill_notes(self.sample_notes())
        summary = validate_fill(
            filled,
            expected_candidate_id="phrase_vocabulary_candidate",
            expected_decision="keep",
        )
        candidate = filled["candidates"][0]

        self.assertEqual(summary["reviewed_count"], 1)
        self.assertEqual(summary["pending_count"], 0)
        self.assertEqual(candidate["listening"]["timing"], "acceptable")
        self.assertEqual(candidate["listening"]["chord_fit"], "strong")
        self.assertEqual(candidate["listening"]["phrase_continuation"], "acceptable")
        self.assertEqual(candidate["listening"]["landing"], "strong")
        self.assertEqual(candidate["listening"]["jazz_vocabulary"], "acceptable")
        self.assertEqual(candidate["listening"]["decision"], "keep")
        self.assertTrue(candidate["listening_fill_evidence"]["not_human_audio_review"])
        self.assertTrue(
            candidate["listening_fill_evidence"]["risk_interpretation"]["adjacent_repeat_blocker_repaired"]
        )
        self.assertTrue(
            candidate["listening_fill_evidence"]["risk_interpretation"]["wide_interval_blocker_repaired"]
        )


if __name__ == "__main__":
    unittest.main()
