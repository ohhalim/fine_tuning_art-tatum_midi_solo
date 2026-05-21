from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.export_stage_b_review_candidates import (
    build_review_manifest,
    markdown_report,
    select_review_candidates,
)


def candidate(
    *,
    rank: int,
    mode: str = "coverage_chord",
    reviewable: bool = True,
    score: float = 10.0,
    midi_path: str = "sample.mid",
) -> dict:
    return {
        "rank": rank,
        "mode": mode,
        "note_groups_per_bar": 4,
        "sample_index": rank,
        "score": score,
        "reviewable": reviewable,
        "review_flags": [] if reviewable else ["low_chord_tone_ratio"],
        "midi_path": midi_path,
        "strict_valid": True,
        "note_count": 8,
        "unique_pitch_count": 5,
        "chord_tone_ratio": 0.75,
        "bar_chord_tone_ratio": 0.875,
        "min_bar_chord_tone_ratio": 0.8,
        "dominant_pitch_ratio": 0.3,
        "repeated_pitch_ratio": 0.25,
        "onset_coverage_ratio": 0.25,
        "sustained_coverage_ratio": 0.5,
    }


class StageBReviewExportTest(unittest.TestCase):
    def test_select_review_candidates_filters_mode_and_reviewable(self) -> None:
        report = {
            "top_candidates": [
                candidate(rank=1, mode="coverage", reviewable=True, score=99),
                candidate(rank=2, mode="coverage_chord", reviewable=False, score=98),
                candidate(rank=3, mode="coverage_chord", reviewable=True, score=97),
            ]
        }

        selected = select_review_candidates(report, top_n=2)

        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["rank"], 3)

    def test_build_review_manifest_copies_midi(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source = root / "source.mid"
            source.write_bytes(b"MThd")
            report_path = root / "candidate_rank_report.json"
            report_path.write_text(
                (
                    '{"top_candidates": ['
                    '{"rank": 1, "mode": "coverage_chord", "note_groups_per_bar": 4, '
                    '"sample_index": 2, "score": 96.6, "reviewable": true, '
                    '"review_flags": [], "midi_path": "'
                    + str(source)
                    + '", "strict_valid": true, "note_count": 8, '
                    '"unique_pitch_count": 6, "chord_tone_ratio": 0.75, '
                    '"bar_chord_tone_ratio": 0.875, "min_bar_chord_tone_ratio": 0.8, '
                    '"dominant_pitch_ratio": 0.375, "repeated_pitch_ratio": 0.25, '
                    '"onset_coverage_ratio": 0.25, "sustained_coverage_ratio": 0.53}'
                    "]}"
                ),
                encoding="utf-8",
            )

            manifest = build_review_manifest(
                ranking_report_path=report_path,
                output_dir=root / "review",
                top_n=3,
                mode="coverage_chord",
                reviewable_only=True,
                copy_midi=True,
            )

            self.assertEqual(manifest["candidate_count"], 1)
            self.assertTrue(manifest["candidates"][0]["copied"])
            self.assertTrue((root / "review" / manifest["candidates"][0]["review_midi_relative_path"]).exists())

    def test_markdown_report_contains_review_checklist(self) -> None:
        manifest = {
            "source_ranking_report": "report.json",
            "generated_at": "2026-05-21T00:00:00Z",
            "mode_filter": "coverage_chord",
            "reviewable_only": True,
            "candidates": [
                {
                    "review_rank": 1,
                    "source_rank": 1,
                    "mode": "coverage_chord",
                    "note_groups_per_bar": 4,
                    "sample_index": 2,
                    "score": 96.6,
                    "note_count": 8,
                    "unique_pitch_count": 6,
                    "chord_tone_ratio": 0.75,
                    "bar_chord_tone_ratio": 0.875,
                    "min_bar_chord_tone_ratio": 0.8,
                    "dominant_pitch_ratio": 0.375,
                    "repeated_pitch_ratio": 0.25,
                    "midi_path": "sample.mid",
                }
            ],
        }

        markdown = markdown_report(manifest)

        self.assertIn("coverage_chord", markdown)
        self.assertIn("Review Checklist", markdown)


if __name__ == "__main__":
    unittest.main()
