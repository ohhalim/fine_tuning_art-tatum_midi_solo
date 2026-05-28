from __future__ import annotations

import unittest

from scripts.build_stage_b_margin_recovered_timing_repetition_focused_listening_notes import (
    build_timing_repetition_focused_listening_notes,
    validate_notes,
)


class StageBMarginRecoveredTimingRepetitionFocusedListeningNotesTest(unittest.TestCase):
    def sample_package(self) -> dict:
        return {
            "output_dir": "outputs/package",
            "candidates": [
                {
                    "candidate_id": "timing_repetition_candidate",
                    "review_metadata": {"mode": "margin_recovered_timing_repetition_repair", "review_rank": 1},
                    "review_files": {
                        "midi_path": "outputs/package/midi/timing.mid",
                        "context_midi_path": "outputs/package/context/timing_context.mid",
                        "source_midi_path": "outputs/source/timing.mid",
                    },
                    "source_metrics": {
                        "note_count": 14,
                        "unique_pitch_count": 7,
                        "dead_air_ratio": 0.35294117647058826,
                    },
                    "listening": {
                        "decision": "timing_repetition_qualified",
                        "phrase_quality": "pending_context",
                        "timing": "pending_context",
                        "chord_fit": "pending_context",
                        "issues": [],
                        "notes": "",
                    },
                    "objective_review": {"objective_flags": [], "objective_bucket": "clean"},
                    "objective_first_16_notes": [{"pitch": 70, "pitch_name": "A#4"}],
                }
            ],
        }

    def sample_decision(self) -> dict:
        return {
            "output_dir": "outputs/decision",
            "candidates": [
                {
                    "candidate_id": "timing_repetition_candidate",
                    "focused_context_decision": "keep_for_focused_listening",
                    "decision_flags": [],
                    "metrics": {
                        "note_count": 14,
                        "unique_pitch_count": 7,
                        "range": "C#4-G5",
                        "phrase_span_beats": 6.5,
                        "dead_air_ratio": 0.35294117647058826,
                        "onset_coverage_ratio": 0.5,
                        "sustained_coverage_ratio": 0.6875,
                        "adjacent_pitch_repeats": 2,
                        "duplicated_3_note_pitch_class_chunks": 0,
                        "max_simultaneous_notes": 1,
                        "max_interval": 16,
                        "final_note": "A#4",
                        "final_chord": "Fm7",
                        "final_note_role": "tension",
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
        notes = build_timing_repetition_focused_listening_notes(self.sample_package(), self.sample_decision())
        summary = validate_notes(
            notes,
            expected_candidate_id="timing_repetition_candidate",
            expected_prior_decision="keep_for_focused_listening",
        )

        candidate = notes["candidates"][0]
        self.assertEqual(summary["candidate_count"], 1)
        self.assertEqual(summary["pending_count"], 1)
        self.assertEqual(candidate["proxy_review"]["decision"], "keep_for_focused_listening")
        self.assertEqual(candidate["focused_context_metrics"]["adjacent_pitch_repeats"], 2)
        self.assertIn("dead_air_ratio_remaining", candidate["review_risks"])
        self.assertIn("adjacent_pitch_repeats", candidate["review_risks"])
        self.assertIn("wide_interval_review", candidate["review_risks"])
        self.assertEqual(candidate["listening"]["decision"], "pending")


if __name__ == "__main__":
    unittest.main()
