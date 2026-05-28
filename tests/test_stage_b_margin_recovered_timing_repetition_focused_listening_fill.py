from __future__ import annotations

import unittest

from scripts.fill_stage_b_margin_recovered_timing_repetition_focused_listening_notes import (
    fill_notes,
    validate_fill,
)


class StageBMarginRecoveredTimingRepetitionFocusedListeningFillTest(unittest.TestCase):
    def sample_notes(self) -> dict:
        return {
            "schema_version": "stage_b_margin_recovered_timing_repetition_focused_listening_notes_v1",
            "candidates": [
                {
                    "candidate_id": "timing_repetition_candidate",
                    "review_files": {
                        "midi_path": "outputs/timing.mid",
                        "context_midi_path": "outputs/timing_context.mid",
                    },
                    "proxy_review": {
                        "decision": "keep_for_focused_listening",
                    },
                    "focused_context_metrics": {
                        "note_count": 14,
                        "unique_pitch_count": 7,
                        "phrase_span_beats": 6.5,
                        "dead_air_ratio": 0.35294117647058826,
                        "adjacent_pitch_repeats": 2,
                        "duplicated_3_note_pitch_class_chunks": 0,
                        "max_simultaneous_notes": 1,
                        "max_interval": 16,
                        "final_note": "A#4",
                        "final_chord": "Fm7",
                        "final_note_role": "tension",
                    },
                    "review_risks": [
                        "dead_air_ratio_remaining",
                        "adjacent_pitch_repeats",
                        "wide_interval_review",
                    ],
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

    def test_fill_marks_timing_improved_but_followup_needed(self) -> None:
        filled = fill_notes(self.sample_notes())
        summary = validate_fill(
            filled,
            expected_candidate_id="timing_repetition_candidate",
            expected_decision="needs_followup",
        )
        candidate = filled["candidates"][0]

        self.assertEqual(summary["reviewed_count"], 1)
        self.assertEqual(summary["pending_count"], 0)
        self.assertEqual(candidate["listening"]["timing"], "acceptable")
        self.assertEqual(candidate["listening"]["chord_fit"], "acceptable")
        self.assertEqual(candidate["listening"]["phrase_continuation"], "weak")
        self.assertEqual(candidate["listening"]["landing"], "acceptable")
        self.assertEqual(candidate["listening"]["jazz_vocabulary"], "thin")
        self.assertEqual(candidate["listening"]["decision"], "needs_followup")
        self.assertTrue(candidate["listening_fill_evidence"]["improved_from_pitch_vocab_fill"]["timing"])
        self.assertTrue(candidate["listening_fill_evidence"]["not_human_audio_review"])


if __name__ == "__main__":
    unittest.main()
