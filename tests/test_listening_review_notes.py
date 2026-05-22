from __future__ import annotations

import unittest

from scripts.build_listening_review_notes import (
    ReviewNotesError,
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


if __name__ == "__main__":
    unittest.main()
