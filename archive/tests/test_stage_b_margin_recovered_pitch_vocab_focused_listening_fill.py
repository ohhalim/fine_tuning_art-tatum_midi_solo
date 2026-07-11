from __future__ import annotations

import unittest

from scripts.fill_stage_b_margin_recovered_pitch_vocab_focused_listening_notes import (
    fill_notes,
    validate_fill,
)


class StageBMarginRecoveredPitchVocabFocusedListeningFillTest(unittest.TestCase):
    def sample_notes(self) -> dict:
        return {
            "schema_version": "stage_b_margin_recovered_pitch_vocab_focused_listening_notes_v1",
            "candidates": [
                {
                    "candidate_id": "pitch_vocab_candidate",
                    "review_files": {
                        "midi_path": "outputs/pitch.mid",
                        "context_midi_path": "outputs/pitch_context.mid",
                    },
                    "proxy_review": {
                        "decision": "keep_for_focused_listening",
                    },
                    "focused_context_metrics": {
                        "note_count": 13,
                        "unique_pitch_count": 6,
                        "phrase_span_beats": 6.25,
                        "dead_air_ratio": 0.4,
                        "adjacent_pitch_repeats": 3,
                        "duplicated_3_note_pitch_class_chunks": 0,
                        "max_simultaneous_notes": 1,
                        "final_note": "G#4",
                        "final_chord": "Fm7",
                        "final_note_role": "chord_tone",
                    },
                    "review_risks": ["dead_air_ratio_at_gate", "adjacent_pitch_repeats"],
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

    def test_fill_marks_candidate_as_needs_followup_with_evidence(self) -> None:
        filled = fill_notes(self.sample_notes())
        summary = validate_fill(
            filled,
            expected_candidate_id="pitch_vocab_candidate",
            expected_decision="needs_followup",
        )
        candidate = filled["candidates"][0]

        self.assertEqual(summary["reviewed_count"], 1)
        self.assertEqual(summary["pending_count"], 0)
        self.assertEqual(candidate["listening"]["timing"], "stiff")
        self.assertEqual(candidate["listening"]["chord_fit"], "strong")
        self.assertEqual(candidate["listening"]["decision"], "needs_followup")
        self.assertTrue(candidate["listening_fill_evidence"]["not_human_audio_review"])


if __name__ == "__main__":
    unittest.main()
