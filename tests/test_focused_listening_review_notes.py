from __future__ import annotations

import unittest

from scripts.build_focused_listening_review_notes import (
    FocusedListeningReviewNotesError,
    build_focused_listening_review_notes,
    validate_focused_listening_review_notes,
)


class FocusedListeningReviewNotesTest(unittest.TestCase):
    def sample_focused_package(self) -> dict:
        return {
            "output_dir": "outputs/stage_b_focused_review_package/harness_stage_b_register_safe_proxy_keep_focused_package",
            "candidates": [
                {
                    "candidate_id": "data_motif_rhythm_phrase_variation_rank_1_sample_3",
                    "review_metadata": {"mode": "data_motif_rhythm_phrase_variation", "review_rank": 1},
                    "review_files": {
                        "midi_path": "outputs/focused/midi/a.mid",
                        "context_midi_path": "outputs/focused/context/a_context.mid",
                        "source_midi_path": "outputs/source/a.mid",
                    },
                    "source_metrics": {
                        "note_count": 63,
                        "unique_pitch_count": 18,
                        "chord_tone_ratio": 0.0,
                        "tension_ratio": 0.349,
                        "outside_ratio": 0.0,
                        "root_tone_ratio": 0.032,
                        "dead_air_ratio": 0.629,
                        "syncopated_onset_ratio": 0.667,
                    },
                    "listening": {
                        "status": "reviewed",
                        "phrase_quality": "phrase",
                        "timing": "acceptable",
                        "chord_fit": "fits",
                        "issues": ["too_repetitive"],
                        "decision": "keep",
                        "notes": "Proxy keep only.",
                    },
                    "objective_review": {
                        "objective_flags": [],
                        "objective_bucket": "clean",
                    },
                    "objective_first_16_notes": [
                        {"start_beats": 0.0, "duration_beats": 0.25, "pitch": 70, "pitch_name": "A#4"}
                    ],
                }
            ],
        }

    def test_build_template_has_pending_real_listening_fields(self) -> None:
        notes = build_focused_listening_review_notes(self.sample_focused_package())
        candidate = notes["candidates"][0]

        self.assertEqual(notes["schema_version"], "stage_b_focused_listening_review_notes_v1")
        self.assertTrue(notes["review_context"]["real_listening_fields_are_separate_from_proxy_review"])
        self.assertEqual(candidate["proxy_review"]["decision"], "keep")
        self.assertEqual(candidate["listening"]["decision"], "pending")
        self.assertEqual(candidate["listening"]["jazz_vocabulary"], "pending")

    def test_build_template_carries_files_metrics_and_first_notes(self) -> None:
        notes = build_focused_listening_review_notes(self.sample_focused_package())
        candidate = notes["candidates"][0]

        self.assertEqual(candidate["review_files"]["midi_path"], "outputs/focused/midi/a.mid")
        self.assertEqual(candidate["review_files"]["context_midi_path"], "outputs/focused/context/a_context.mid")
        self.assertEqual(candidate["source_metrics"]["note_count"], 63)
        self.assertEqual(candidate["source_metrics"]["unique_pitch_count"], 18)
        self.assertEqual(candidate["source_metrics"]["tension_ratio"], 0.349)
        self.assertEqual(candidate["objective_first_16_notes"][0]["pitch_name"], "A#4")

    def test_validate_counts_pending_candidate(self) -> None:
        notes = build_focused_listening_review_notes(self.sample_focused_package())

        summary = validate_focused_listening_review_notes(notes)

        self.assertEqual(summary["candidate_count"], 1)
        self.assertEqual(summary["pending_count"], 1)
        self.assertEqual(summary["decision_counts"]["pending"], 1)

    def test_validate_accepts_reviewed_candidate(self) -> None:
        notes = build_focused_listening_review_notes(self.sample_focused_package())
        notes["candidates"][0]["listening"].update(
            {
                "status": "reviewed",
                "timing": "acceptable",
                "chord_fit": "acceptable",
                "phrase_continuation": "acceptable",
                "landing": "strong",
                "jazz_vocabulary": "thin",
                "decision": "needs_followup",
            }
        )

        summary = validate_focused_listening_review_notes(notes)

        self.assertEqual(summary["reviewed_count"], 1)
        self.assertEqual(summary["decision_counts"]["needs_followup"], 1)

    def test_validate_requires_context_midi_path(self) -> None:
        notes = build_focused_listening_review_notes(self.sample_focused_package())
        notes["candidates"][0]["review_files"]["context_midi_path"] = ""

        with self.assertRaises(FocusedListeningReviewNotesError):
            validate_focused_listening_review_notes(notes)


if __name__ == "__main__":
    unittest.main()
