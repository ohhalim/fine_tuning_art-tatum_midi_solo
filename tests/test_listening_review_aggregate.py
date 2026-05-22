from __future__ import annotations

import unittest
from pathlib import Path

from scripts.build_listening_review_notes import ReviewNotesError, build_review_notes_template
from scripts.summarize_listening_review_notes import aggregate_review_notes, markdown_summary


class ListeningReviewAggregateTest(unittest.TestCase):
    def sample_notes(self) -> dict:
        generated_report = {
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
                },
                {
                    "sample_id": "candidate_2",
                    "note_count": 24,
                    "unique_pitch_count": 8,
                    "role_ratios": {
                        "chord_tone_ratio": 0.875,
                        "tension_ratio": 0.0,
                        "approach_ratio": 0.125,
                        "outside_ratio": 0.0,
                    },
                },
            ],
        }
        return build_review_notes_template(generated_report)

    def test_pending_only_notes_do_not_create_generation_recommendation(self) -> None:
        aggregate = aggregate_review_notes(self.sample_notes())

        self.assertEqual(aggregate["reviewed_count"], 0)
        self.assertFalse(aggregate["has_reviewed_candidates"])
        self.assertEqual(aggregate["recommended_followups"][0]["code"], "collect_listening_reviews")

    def test_filled_notes_aggregate_issue_counts_and_followups(self) -> None:
        notes = self.sample_notes()
        notes["candidates"][0]["listening"].update(
            {
                "status": "reviewed",
                "phrase_quality": "exercise",
                "timing": "acceptable",
                "chord_fit": "too_safe",
                "issues": ["too_safe", "too_mechanical"],
                "decision": "needs_followup",
                "notes": "Valid but sounds like an exercise.",
            }
        )
        notes["candidates"][1]["listening"].update(
            {
                "status": "reviewed",
                "phrase_quality": "fragment",
                "timing": "off_grid",
                "chord_fit": "fits",
                "issues": ["bad_timing", "weak_phrase"],
                "decision": "reject",
                "notes": "Timing does not land cleanly.",
            }
        )

        aggregate = aggregate_review_notes(notes)
        followup_codes = [item["code"] for item in aggregate["recommended_followups"]]

        self.assertEqual(aggregate["reviewed_count"], 2)
        self.assertEqual(aggregate["issue_counts"]["too_safe"], 1)
        self.assertEqual(aggregate["issue_counts"]["bad_timing"], 1)
        self.assertIn("fix_timing_grid", followup_codes)
        self.assertIn("increase_tension_approach_vocabulary", followup_codes)
        self.assertIn("improve_phrase_vocabulary", followup_codes)

    def test_metric_summary_groups_by_decision(self) -> None:
        notes = self.sample_notes()
        notes["candidates"][0]["listening"].update(
            {
                "status": "reviewed",
                "phrase_quality": "phrase",
                "timing": "acceptable",
                "chord_fit": "fits",
                "decision": "keep",
            }
        )

        aggregate = aggregate_review_notes(notes)
        keep_metrics = aggregate["source_metric_by_decision"]["keep"]

        self.assertEqual(keep_metrics["count"], 1)
        self.assertEqual(keep_metrics["avg_note_count"], 32.0)
        self.assertEqual(keep_metrics["avg_chord_tone_ratio"], 0.75)

    def test_objective_review_fields_are_aggregated_for_priority(self) -> None:
        notes = self.sample_notes()
        notes["candidates"][0]["objective_review"] = {
            "objective_flags": ["overlap_polyphonic"],
            "objective_penalty": 40,
            "objective_priority_score": 60,
            "objective_reviewable": False,
            "objective_bucket": "problem",
            "metrics": {},
        }
        notes["candidates"][1]["objective_review"] = {
            "objective_flags": ["chromatic_walk"],
            "objective_penalty": 18,
            "objective_priority_score": 82,
            "objective_reviewable": True,
            "objective_bucket": "warning",
            "metrics": {},
        }

        aggregate = aggregate_review_notes(notes)
        priority_ids = [candidate["candidate_id"] for candidate in aggregate["objective_candidates_by_priority"]]

        self.assertEqual(aggregate["objective_review_candidate_count"], 2)
        self.assertEqual(aggregate["objective_reviewable_count"], 1)
        self.assertEqual(aggregate["objective_flag_counts"]["overlap_polyphonic"], 1)
        self.assertEqual(aggregate["objective_bucket_counts"]["warning"], 1)
        self.assertEqual(priority_ids[0], "candidate_2")

    def test_invalid_notes_are_rejected_before_aggregation(self) -> None:
        notes = self.sample_notes()
        notes["candidates"][0]["listening"]["decision"] = "maybe"

        with self.assertRaises(ReviewNotesError):
            aggregate_review_notes(notes)

    def test_markdown_summary_includes_review_dimension_tables(self) -> None:
        aggregate = aggregate_review_notes(self.sample_notes())

        markdown = markdown_summary(aggregate, Path("aggregate.json"))

        self.assertIn("## Phrase Quality Counts", markdown)
        self.assertIn("## Timing Counts", markdown)
        self.assertIn("## Chord Fit Counts", markdown)
        self.assertIn("| pending | 2 |", markdown)


if __name__ == "__main__":
    unittest.main()
