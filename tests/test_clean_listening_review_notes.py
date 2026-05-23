from __future__ import annotations

import unittest

from scripts.build_clean_listening_review_notes import (
    CleanListeningReviewNotesError,
    build_clean_listening_review_notes,
    validate_clean_listening_review_notes,
)


class CleanListeningReviewNotesTest(unittest.TestCase):
    def sample_clean_package(self) -> dict:
        return {
            "output_dir": "outputs/stage_b_clean_review_package/harness_stage_b_clean_review_package",
            "candidates": [
                {
                    "candidate_id": "data_motif_phrase_recovery_rank_1_sample_1",
                    "review_midi_path": "outputs/a.mid",
                    "context_midi_path": "outputs/a_ctx.mid",
                    "chord_guide_path": "outputs/a_chord.mid",
                    "bass_root_guide_path": "outputs/a_bass.mid",
                    "metrics": {
                        "note_count": 63,
                        "unique_pitch_count": 19,
                        "chord_tone_ratio": 0.508,
                        "tension_ratio": 0.492,
                        "unresolved_large_leap_ratio": 0.0,
                    },
                }
            ],
        }

    def sample_diagnostics(self) -> dict:
        return {
            "output_dir": "outputs/stage_b_clean_context_diagnostics/harness_stage_b_clean_context_diagnostics",
            "candidates": [
                {
                    "candidate_id": "data_motif_phrase_recovery_rank_1_sample_1",
                    "solo_metrics": {
                        "note_count": 63,
                        "unique_pitch_count": 19,
                        "bar_coverage_ratio": 1.0,
                        "off_sixteenth_grid_ratio": 0.0,
                        "max_duration_beats": 1.0,
                        "max_simultaneous_notes": 1,
                    },
                }
            ],
        }

    def test_build_template_has_pending_fields(self) -> None:
        notes = build_clean_listening_review_notes(self.sample_clean_package(), self.sample_diagnostics())
        listening = notes["candidates"][0]["listening"]

        self.assertEqual(notes["schema_version"], "stage_b_clean_listening_review_notes_v1")
        self.assertEqual(listening["timing"], "pending")
        self.assertEqual(listening["phrase_continuation"], "pending")
        self.assertEqual(listening["landing"], "pending")
        self.assertEqual(listening["jazz_vocabulary"], "pending")

    def test_build_template_carries_review_paths_and_metrics(self) -> None:
        notes = build_clean_listening_review_notes(self.sample_clean_package(), self.sample_diagnostics())
        candidate = notes["candidates"][0]

        self.assertEqual(candidate["review_files"]["midi_path"], "outputs/a.mid")
        self.assertEqual(candidate["review_files"]["context_midi_path"], "outputs/a_ctx.mid")
        self.assertEqual(candidate["source_metrics"]["note_count"], 63)
        self.assertEqual(candidate["source_metrics"]["unique_pitch_count"], 19)
        self.assertEqual(candidate["source_metrics"]["chord_tone_ratio"], 0.508)
        self.assertEqual(candidate["source_metrics"]["bar_coverage_ratio"], 1.0)
        self.assertEqual(candidate["source_metrics"]["off_grid_ratio"], 0.0)

    def test_validate_counts_pending(self) -> None:
        notes = build_clean_listening_review_notes(self.sample_clean_package(), self.sample_diagnostics())

        summary = validate_clean_listening_review_notes(notes)

        self.assertEqual(summary["candidate_count"], 1)
        self.assertEqual(summary["pending_count"], 1)
        self.assertEqual(summary["decision_counts"]["pending"], 1)

    def test_validate_accepts_reviewed(self) -> None:
        notes = build_clean_listening_review_notes(self.sample_clean_package(), self.sample_diagnostics())
        notes["candidates"][0]["listening"].update(
            {
                "status": "reviewed",
                "timing": "acceptable",
                "chord_fit": "strong",
                "phrase_continuation": "acceptable",
                "landing": "strong",
                "jazz_vocabulary": "acceptable",
                "decision": "keep",
            }
        )

        summary = validate_clean_listening_review_notes(notes)

        self.assertEqual(summary["reviewed_count"], 1)
        self.assertEqual(summary["decision_counts"]["keep"], 1)

    def test_validate_rejects_invalid_enum(self) -> None:
        notes = build_clean_listening_review_notes(self.sample_clean_package(), self.sample_diagnostics())
        notes["candidates"][0]["listening"]["landing"] = "great"

        with self.assertRaises(CleanListeningReviewNotesError):
            validate_clean_listening_review_notes(notes)


if __name__ == "__main__":
    unittest.main()
