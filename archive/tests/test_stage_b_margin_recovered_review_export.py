from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.build_stage_b_margin_recovered_review_export import (
    build_review_export,
    markdown_report,
    write_json,
)


class StageBMarginRecoveredReviewExportTest(unittest.TestCase):
    def test_build_review_export_ranks_seed_best_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            summary_path = Path(tmp) / "repeatability_summary.json"
            write_json(
                summary_path,
                {
                    "run_id": "run",
                    "issue": 999,
                    "summary": {
                        "passed_repeatability_gate": True,
                        "total_samples": 15,
                        "total_strict_valid_sample_count": 12,
                        "strict_valid_sample_rate": 0.8,
                        "dead_air_outlier_rate": 0.133,
                        "strict_margin_warning_seeds": [],
                        "selected_best_candidate": {"seed": 23, "sample_index": 1},
                        "seed_best_candidates": [
                            {
                                "seed": 17,
                                "sample_index": 3,
                                "midi_path": "seed17.mid",
                                "dead_air_ratio": 0.5,
                                "note_count": 17,
                                "unique_pitch_count": 4,
                                "phrase_coverage_ratio": 1.0,
                                "onset_coverage_ratio": 0.594,
                                "sustained_coverage_ratio": 0.844,
                                "postprocess_removal_ratio": 0.227,
                            },
                            {
                                "seed": 23,
                                "sample_index": 1,
                                "midi_path": "seed23.mid",
                                "dead_air_ratio": 0.375,
                                "note_count": 9,
                                "unique_pitch_count": 4,
                                "phrase_coverage_ratio": 0.437,
                                "onset_coverage_ratio": 0.313,
                                "sustained_coverage_ratio": 0.438,
                                "postprocess_removal_ratio": 0.357,
                            },
                        ],
                    },
                    "rows": [
                        {
                            "seed": 17,
                            "sample_count": 5,
                            "strict_valid_sample_count": 3,
                            "dead_air_outlier_count": 1,
                            "failure_reasons": {"dead-air ratio too high": 1},
                            "strict_failure_reasons": {"midi_review_gate_failed": 1},
                        },
                        {
                            "seed": 23,
                            "sample_count": 5,
                            "strict_valid_sample_count": 4,
                            "dead_air_outlier_count": 1,
                            "failure_reasons": {},
                            "strict_failure_reasons": {},
                        },
                    ],
                },
            )

            report = build_review_export(summary_path)

        self.assertEqual(report["candidate_count"], 2)
        self.assertEqual(report["selected_best_rank"], 1)
        self.assertEqual(report["candidates"][0]["seed"], 23)
        self.assertTrue(report["candidates"][0]["is_selected_best"])
        self.assertEqual(report["candidates"][1]["seed_strict_valid_sample_count"], 3)

    def test_markdown_report_contains_metric_table(self) -> None:
        report = {
            "source_summary_path": "summary.json",
            "source_run_id": "run",
            "selected_best_rank": 1,
            "summary": {
                "passed_repeatability_gate": True,
                "total_samples": 15,
                "total_strict_valid_sample_count": 12,
                "strict_valid_sample_rate": 0.8,
                "dead_air_outlier_rate": 0.133,
                "strict_margin_warning_seeds": [],
            },
            "candidates": [
                {
                    "review_rank": 1,
                    "is_selected_best": True,
                    "seed": 23,
                    "sample_index": 1,
                    "seed_strict_valid_sample_count": 4,
                    "seed_sample_count": 5,
                    "seed_dead_air_outlier_count": 1,
                    "dead_air_ratio": 0.375,
                    "note_count": 9,
                    "unique_pitch_count": 4,
                    "phrase_coverage_ratio": 0.437,
                    "onset_coverage_ratio": 0.313,
                    "sustained_coverage_ratio": 0.438,
                    "postprocess_removal_ratio": 0.357,
                    "midi_path": "sample.mid",
                    "seed_failure_reasons": {},
                    "seed_strict_failure_reasons": {},
                }
            ],
        }

        markdown = markdown_report(report)

        self.assertIn("selected best rank", markdown)
        self.assertIn("| 1 | True | 23 | 1 | 4/5 |", markdown)
        self.assertIn("dead-air outlier rate", markdown)


if __name__ == "__main__":
    unittest.main()
