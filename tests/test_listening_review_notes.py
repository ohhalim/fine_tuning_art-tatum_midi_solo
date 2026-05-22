from __future__ import annotations

import unittest

from scripts.build_listening_review_notes import (
    ReviewNotesError,
    build_review_notes_from_review_manifest,
    build_review_notes_template,
    validate_review_notes,
)


class ListeningReviewNotesTest(unittest.TestCase):
    def sample_generated_report(self) -> dict:
        return {
            "source_report_path": "outputs/example/review_manifest.json",
            "samples": [
                {
                    "sample_id": "candidate_1",
                    "note_count": 32,
                    "unique_pitch_count": 12,
                    "role_ratios": {
                        "chord_tone_ratio": 0.75,
                        "tension_ratio": 0.125,
                        "approach_ratio": 0.125,
                        "outside_ratio": 0.0,
                    },
                }
            ],
        }

    def sample_review_manifest(self) -> dict:
        return {
            "chord_progression": ["Cm7", "F7", "Bbmaj7", "Ebmaj7"],
            "candidates": [
                {
                    "mode": "data_motif",
                    "review_rank": 1,
                    "sample_index": 1,
                    "sample_seed": 17,
                    "valid": True,
                    "strict_valid": True,
                    "review_midi_path": "named_midi/01_data_motif_rank_01_sample_01.mid",
                    "midi_path": "samples/data_motif/data_motif_sample_1.mid",
                    "context_midi_path": "context_midi/01_data_motif_rank_01_sample_01_with_context.mid",
                    "note_count": 63,
                    "unique_pitch_count": 24,
                    "dead_air_ratio": 0.56,
                    "syncopated_onset_ratio": 0.625,
                    "unique_bar_position_pattern_ratio": 1.0,
                    "duration_diversity_ratio": 0.0625,
                    "most_common_duration_ratio": 0.375,
                    "ioi_diversity_ratio": 0.079,
                    "most_common_ioi_ratio": 0.429,
                    "tension_ratio": 0.172,
                    "root_tone_ratio": 0.0,
                },
                {
                    "mode": "straight_grid",
                    "review_rank": 1,
                    "sample_index": 1,
                    "review_midi_path": "named_midi/04_straight_grid_rank_01_sample_01.mid",
                    "context_midi_path": "context_midi/04_straight_grid_rank_01_sample_01_with_context.mid",
                    "note_count": 64,
                    "unique_pitch_count": 26,
                },
            ],
        }

    def sample_objective_report(self) -> dict:
        return {
            "source_report_path": "outputs/objective/objective_midi_note_review.json",
            "candidates": [
                {
                    "candidate_id": "data_motif_rank_1_sample_1",
                    "objective_flags": ["overlap_polyphonic"],
                    "objective_penalty": 40,
                    "objective_priority_score": 60,
                    "objective_reviewable": False,
                    "objective_bucket": "problem",
                    "metrics": {
                        "max_active_notes": 2,
                        "polyphonic_tick_ratio": 0.25,
                        "off_sixteenth_grid_count": 0,
                        "stepwise_interval_ratio": 0.4,
                        "chromatic_interval_ratio": 0.1,
                        "chord_tone_ratio": 0.7,
                        "tension_ratio": 0.2,
                        "outside_ratio": 0.1,
                        "most_common_duration_ratio": 0.5,
                    },
                },
                {
                    "candidate_id": "straight_grid_rank_1_sample_1",
                    "objective_flags": ["chromatic_walk"],
                    "objective_penalty": 18,
                    "objective_priority_score": 82,
                    "objective_reviewable": True,
                    "objective_bucket": "warning",
                    "metrics": {
                        "max_active_notes": 1,
                        "polyphonic_tick_ratio": 0.0,
                        "off_sixteenth_grid_count": 0,
                        "stepwise_interval_ratio": 0.8,
                        "chromatic_interval_ratio": 0.4,
                        "chord_tone_ratio": 0.5,
                        "tension_ratio": 0.3,
                        "outside_ratio": 0.2,
                        "most_common_duration_ratio": 0.8,
                    },
                },
            ],
        }

    def test_build_review_notes_template_defaults_to_pending(self) -> None:
        notes = build_review_notes_template(self.sample_generated_report(), source_review_markdown="review.md")

        self.assertEqual(notes["schema_version"], "stage_b_listening_review_notes_v1")
        self.assertEqual(notes["candidates"][0]["candidate_id"], "candidate_1")
        self.assertEqual(notes["candidates"][0]["listening"]["status"], "pending")
        self.assertEqual(notes["candidates"][0]["source_metrics"]["chord_tone_ratio"], 0.75)

    def test_validate_review_notes_counts_pending_candidates(self) -> None:
        notes = build_review_notes_template(self.sample_generated_report())

        summary = validate_review_notes(notes)

        self.assertEqual(summary["candidate_count"], 1)
        self.assertEqual(summary["pending_count"], 1)
        self.assertEqual(summary["decision_counts"]["pending"], 1)

    def test_validate_review_notes_accepts_reviewed_candidate(self) -> None:
        notes = build_review_notes_template(self.sample_generated_report())
        notes["candidates"][0]["listening"].update(
            {
                "status": "reviewed",
                "phrase_quality": "exercise",
                "timing": "acceptable",
                "chord_fit": "too_safe",
                "issues": ["too_safe", "too_mechanical"],
                "decision": "needs_followup",
                "notes": "Sounds valid but too safe.",
            }
        )

        summary = validate_review_notes(notes)

        self.assertEqual(summary["reviewed_count"], 1)
        self.assertEqual(summary["decision_counts"]["needs_followup"], 1)

    def test_validate_review_notes_rejects_invalid_enum(self) -> None:
        notes = build_review_notes_template(self.sample_generated_report())
        notes["candidates"][0]["listening"]["timing"] = "swingy"

        with self.assertRaises(ReviewNotesError):
            validate_review_notes(notes)

    def test_validate_review_notes_rejects_duplicate_candidate_id(self) -> None:
        notes = build_review_notes_template(self.sample_generated_report())
        notes["candidates"].append(dict(notes["candidates"][0]))

        with self.assertRaises(ReviewNotesError):
            validate_review_notes(notes)

    def test_build_review_notes_from_review_manifest_includes_files_and_modes(self) -> None:
        notes = build_review_notes_from_review_manifest(
            self.sample_review_manifest(),
            source_review_markdown="review_candidates.md",
        )

        self.assertEqual(len(notes["candidates"]), 2)
        self.assertEqual(notes["candidates"][0]["candidate_id"], "data_motif_rank_1_sample_1")
        self.assertEqual(notes["candidates"][0]["review_metadata"]["mode"], "data_motif")
        self.assertEqual(
            notes["candidates"][0]["review_files"]["context_midi_path"],
            "context_midi/01_data_motif_rank_01_sample_01_with_context.mid",
        )
        self.assertEqual(notes["candidates"][0]["source_metrics"]["syncopated_onset_ratio"], 0.625)

    def test_build_review_notes_from_review_manifest_validates_pending_candidates(self) -> None:
        notes = build_review_notes_from_review_manifest(self.sample_review_manifest())

        summary = validate_review_notes(notes)

        self.assertEqual(summary["candidate_count"], 2)
        self.assertEqual(summary["pending_count"], 2)

    def test_build_review_notes_from_review_manifest_attaches_objective_review(self) -> None:
        notes = build_review_notes_from_review_manifest(
            self.sample_review_manifest(),
            objective_midi_review_report=self.sample_objective_report(),
        )

        objective_review = notes["candidates"][0]["objective_review"]
        summary = validate_review_notes(notes)

        self.assertEqual(notes["source_objective_midi_review_report"], "outputs/objective/objective_midi_note_review.json")
        self.assertEqual(objective_review["objective_bucket"], "problem")
        self.assertFalse(objective_review["objective_reviewable"])
        self.assertEqual(objective_review["metrics"]["max_active_notes"], 2)
        self.assertEqual(summary["candidate_count"], 2)


if __name__ == "__main__":
    unittest.main()
