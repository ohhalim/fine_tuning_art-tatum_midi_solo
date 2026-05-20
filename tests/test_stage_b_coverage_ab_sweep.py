from __future__ import annotations

import unittest

from scripts.run_stage_b_coverage_ab_sweep import (
    build_ab_summary,
    config_run_id,
    markdown_table,
    parse_modes,
)


class StageBCoverageAbSweepTest(unittest.TestCase):
    def test_parse_modes_rejects_unknown_mode(self) -> None:
        with self.assertRaises(ValueError):
            parse_modes("plain,random")

    def test_config_run_id_is_stable(self) -> None:
        self.assertEqual(
            config_run_id("run", mode="coverage", note_groups_per_bar=6, top_k=2, temperature=0.9),
            "run_coverage_g6_k2_t0p9",
        )

    def test_build_ab_summary_requires_coverage_success_and_comparison(self) -> None:
        rows = [
            {
                "mode": "plain",
                "note_groups_per_bar": 4,
                "run_id": "plain",
                "top_k": 2,
                "temperature": 0.9,
                "valid_sample_count": 0,
                "strict_valid_sample_count": 0,
                "avg_onset_coverage_ratio": 0.167,
                "avg_sustained_coverage_ratio": 0.417,
                "collapse_warning_sample_rate": 0.0,
                "max_longest_sustained_empty_run_steps": 11,
            },
            {
                "mode": "coverage",
                "note_groups_per_bar": 4,
                "run_id": "coverage",
                "top_k": 2,
                "temperature": 0.9,
                "valid_sample_count": 3,
                "strict_valid_sample_count": 3,
                "avg_onset_coverage_ratio": 0.25,
                "avg_sustained_coverage_ratio": 0.427,
                "collapse_warning_sample_rate": 0.0,
                "max_longest_sustained_empty_run_steps": 6,
            },
        ]

        summary = build_ab_summary(rows, min_best_strict_valid_samples=1)

        self.assertTrue(summary["passed_ab_sweep_gate"])
        self.assertEqual(summary["best_coverage_config"]["strict_valid_sample_count"], 3)
        self.assertEqual(summary["best_plain_config"]["strict_valid_sample_count"], 0)

    def test_build_ab_summary_fails_without_plain_comparison(self) -> None:
        rows = [
            {
                "mode": "coverage",
                "note_groups_per_bar": 4,
                "run_id": "coverage",
                "top_k": 2,
                "temperature": 0.9,
                "valid_sample_count": 3,
                "strict_valid_sample_count": 3,
                "avg_onset_coverage_ratio": 0.25,
                "avg_sustained_coverage_ratio": 0.427,
                "collapse_warning_sample_rate": 0.0,
                "max_longest_sustained_empty_run_steps": 6,
            }
        ]

        summary = build_ab_summary(rows, min_best_strict_valid_samples=1)

        self.assertFalse(summary["passed_ab_sweep_gate"])
        self.assertFalse(summary["passed_comparison_gate"])

    def test_markdown_table_records_coverage_columns(self) -> None:
        rows = [
            {
                "mode": "coverage",
                "note_groups_per_bar": 4,
                "sample_count": 3,
                "grammar_gate_sample_count": 3,
                "valid_sample_count": 3,
                "strict_valid_sample_count": 3,
                "avg_onset_coverage_ratio": 0.25,
                "avg_sustained_coverage_ratio": 0.427,
                "avg_position_span_ratio": 0.813,
                "max_longest_sustained_empty_run_steps": 6,
                "avg_dead_air_ratio": 0.429,
                "collapse_warning_sample_rate": 0.0,
                "passed_strict_review_gate": True,
                "diagnostic_failure_reasons": {},
            }
        ]
        summary = {
            "passed_ab_sweep_gate": True,
            "best_plain_config": None,
            "best_coverage_config": {"mode": "coverage", "note_groups_per_bar": 4},
        }

        markdown = markdown_table(rows, summary)

        self.assertIn("groups/bar", markdown)
        self.assertIn("dead air", markdown)
        self.assertIn("coverage", markdown)


if __name__ == "__main__":
    unittest.main()
