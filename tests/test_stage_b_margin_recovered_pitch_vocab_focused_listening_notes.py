from __future__ import annotations

import unittest

from scripts.build_stage_b_margin_recovered_pitch_vocab_focused_listening_notes import (
    build_pitch_vocab_focused_listening_notes,
    validate_notes,
)


class StageBMarginRecoveredPitchVocabFocusedListeningNotesTest(unittest.TestCase):
    def sample_package(self) -> dict:
        return {
            "output_dir": "outputs/package",
            "candidates": [
                {
                    "candidate_id": "pitch_vocab_candidate",
                    "review_metadata": {"mode": "margin_recovered_pitch_vocab_sweep", "review_rank": 1},
                    "review_files": {
                        "midi_path": "outputs/package/midi/pitch.mid",
                        "context_midi_path": "outputs/package/context/pitch_context.mid",
                        "source_midi_path": "outputs/source/pitch.mid",
                    },
                    "source_metrics": {
                        "note_count": 13,
                        "unique_pitch_count": 6,
                        "dead_air_ratio": 0.4,
                    },
                    "listening": {
                        "decision": "pitch_vocab_qualified",
                        "phrase_quality": "pending_context",
                        "timing": "pending_context",
                        "chord_fit": "pending_context",
                        "issues": [],
                        "notes": "",
                    },
                    "objective_review": {"objective_flags": [], "objective_bucket": "clean"},
                    "objective_first_16_notes": [{"pitch": 68, "pitch_name": "G#4"}],
                }
            ],
        }

    def sample_decision(self) -> dict:
        return {
            "output_dir": "outputs/decision",
            "candidates": [
                {
                    "candidate_id": "pitch_vocab_candidate",
                    "focused_context_decision": "keep_for_focused_listening",
                    "decision_flags": [],
                    "metrics": {
                        "note_count": 13,
                        "unique_pitch_count": 6,
                        "range": "D#4-C5",
                        "phrase_span_beats": 6.25,
                        "dead_air_ratio": 0.4,
                        "onset_coverage_ratio": 0.5,
                        "sustained_coverage_ratio": 0.625,
                        "adjacent_pitch_repeats": 3,
                        "duplicated_3_note_pitch_class_chunks": 0,
                        "max_simultaneous_notes": 1,
                        "final_note": "G#4",
                        "final_chord": "Fm7",
                        "final_note_role": "chord_tone",
                    },
                    "context_summary": {
                        "has_chord_guide": True,
                        "has_bass_guide": True,
                        "has_solo_track": True,
                    },
                }
            ],
        }

    def test_builds_pending_notes_with_context_decision_and_risks(self) -> None:
        notes = build_pitch_vocab_focused_listening_notes(self.sample_package(), self.sample_decision())
        summary = validate_notes(
            notes,
            expected_candidate_id="pitch_vocab_candidate",
            expected_prior_decision="keep_for_focused_listening",
        )

        candidate = notes["candidates"][0]
        self.assertEqual(summary["candidate_count"], 1)
        self.assertEqual(summary["pending_count"], 1)
        self.assertEqual(candidate["proxy_review"]["decision"], "keep_for_focused_listening")
        self.assertEqual(candidate["focused_context_metrics"]["adjacent_pitch_repeats"], 3)
        self.assertIn("dead_air_ratio_at_gate", candidate["review_risks"])
        self.assertIn("adjacent_pitch_repeats", candidate["review_risks"])
        self.assertEqual(candidate["listening"]["decision"], "pending")


if __name__ == "__main__":
    unittest.main()
