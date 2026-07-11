from __future__ import annotations

import unittest

from scripts.build_stage_b_margin_recovered_listening_notes import (
    build_listening_notes,
    markdown_report,
    validate_listening_notes,
)


def candidate(rank: int, *, selected: bool = False) -> dict:
    return {
        "review_rank": rank,
        "is_selected_best": selected,
        "seed": 20 + rank,
        "sample_index": rank,
        "midi_path": f"sample_{rank}.mid",
        "dead_air_ratio": 0.3 + rank / 10,
        "note_count": 8 + rank,
        "unique_pitch_count": 4,
        "phrase_coverage_ratio": 0.5,
        "onset_coverage_ratio": 0.4,
        "sustained_coverage_ratio": 0.6,
        "postprocess_removal_ratio": 0.1,
        "seed_strict_valid_sample_count": 3,
        "seed_sample_count": 5,
        "seed_dead_air_outlier_count": 1,
    }


class StageBMarginRecoveredListeningNotesTest(unittest.TestCase):
    def test_build_and_validate_pending_notes(self) -> None:
        notes = build_listening_notes(
            {
                "source_summary_path": "summary.json",
                "source_run_id": "run",
                "summary": {"strict_valid_sample_rate": 0.8},
                "candidates": [candidate(1, selected=True), candidate(2), candidate(3)],
            }
        )
        summary = validate_listening_notes(notes)

        self.assertEqual(summary["candidate_count"], 3)
        self.assertEqual(summary["selected_best_count"], 1)
        self.assertEqual(summary["pending_count"], 3)
        self.assertEqual(notes["candidates"][0]["candidate_id"], "margin_recovered_rank_1_seed_21_sample_1")

    def test_validate_requires_exactly_one_selected_candidate(self) -> None:
        notes = build_listening_notes(
            {
                "source_summary_path": "summary.json",
                "source_run_id": "run",
                "candidates": [candidate(1), candidate(2)],
            }
        )

        with self.assertRaises(ValueError):
            validate_listening_notes(notes)

    def test_markdown_report_lists_pending_candidates(self) -> None:
        notes = build_listening_notes(
            {
                "source_summary_path": "summary.json",
                "source_run_id": "run",
                "candidates": [candidate(1, selected=True)],
            }
        )
        summary = validate_listening_notes(notes)

        markdown = markdown_report(notes, summary)

        self.assertIn("candidate count", markdown)
        self.assertIn("margin_recovered_rank_1_seed_21_sample_1", markdown)
        self.assertIn("pending", markdown)


if __name__ == "__main__":
    unittest.main()
