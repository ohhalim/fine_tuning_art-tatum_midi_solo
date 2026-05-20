from __future__ import annotations

import unittest

from scripts.run_stage_b_sampling_sweep import build_sweep_summary, markdown_table


class StageBSamplingSweepTest(unittest.TestCase):
    def test_build_sweep_summary_selects_best_valid_config(self) -> None:
        rows = [
            {
                "run_id": "k1",
                "top_k": 1,
                "temperature": 0.9,
                "valid_sample_count": 0,
                "strict_valid_sample_count": 0,
                "valid_sample_rate": 0.0,
                "strict_valid_sample_rate": 0.0,
                "collapse_warning_sample_rate": 1.0,
            },
            {
                "run_id": "k2",
                "top_k": 2,
                "temperature": 0.9,
                "valid_sample_count": 1,
                "strict_valid_sample_count": 1,
                "valid_sample_rate": 0.333,
                "strict_valid_sample_rate": 0.333,
                "collapse_warning_sample_rate": 0.333,
            },
        ]

        summary = build_sweep_summary(rows, min_best_valid_samples=1)

        self.assertTrue(summary["passed_sweep_gate"])
        self.assertTrue(summary["passed_strict_sweep_gate"])
        self.assertEqual(summary["best_valid_sample_count"], 1)
        self.assertEqual(summary["best_strict_valid_sample_count"], 1)
        self.assertEqual(summary["best_config"], {"top_k": 2, "temperature": 0.9, "run_id": "k2"})

    def test_build_sweep_summary_fails_when_no_config_has_valid_samples(self) -> None:
        rows = [
            {
                "run_id": "k1",
                "top_k": 1,
                "temperature": 0.9,
                "valid_sample_count": 0,
                "strict_valid_sample_count": 0,
                "valid_sample_rate": 0.0,
                "strict_valid_sample_rate": 0.0,
                "collapse_warning_sample_rate": 1.0,
            },
            {
                "run_id": "k2",
                "top_k": 2,
                "temperature": 0.9,
                "valid_sample_count": 0,
                "strict_valid_sample_count": 0,
                "valid_sample_rate": 0.0,
                "strict_valid_sample_rate": 0.0,
                "collapse_warning_sample_rate": 0.667,
            },
        ]

        summary = build_sweep_summary(rows, min_best_valid_samples=1)

        self.assertFalse(summary["passed_sweep_gate"])
        self.assertFalse(summary["passed_strict_sweep_gate"])

    def test_build_sweep_summary_fails_when_collapse_rate_exceeds_cap(self) -> None:
        rows = [
            {
                "run_id": "k2",
                "top_k": 2,
                "temperature": 0.9,
                "valid_sample_count": 2,
                "strict_valid_sample_count": 1,
                "valid_sample_rate": 0.667,
                "strict_valid_sample_rate": 0.333,
                "collapse_warning_sample_rate": 0.667,
            }
        ]

        summary = build_sweep_summary(
            rows,
            min_best_valid_samples=1,
            min_best_strict_valid_samples=1,
            max_collapse_warning_sample_rate=0.34,
        )

        self.assertTrue(summary["passed_basic_sweep_gate"])
        self.assertFalse(summary["passed_strict_sweep_gate"])
        self.assertFalse(summary["passed_sweep_gate"])

    def test_markdown_table_records_collapse_columns(self) -> None:
        rows = [
            {
                "top_k": 1,
                "temperature": 0.9,
                "sample_count": 3,
                "grammar_gate_sample_count": 3,
                "valid_sample_count": 0,
                "strict_valid_sample_count": 0,
                "valid_sample_rate": 0.0,
                "strict_valid_sample_rate": 0.0,
                "collapse_warning_sample_rate": 1.0,
                "passed_strict_review_gate": False,
                "avg_repeated_position_pitch_pair_ratio": 0.75,
                "max_postprocess_removal_ratio": 0.5,
                "diagnostic_failure_reasons": {"collapse": 3},
                "strict_failure_reasons": {"midi_review_gate_failed": 3},
            }
        ]
        summary = {
            "passed_sweep_gate": False,
            "passed_basic_sweep_gate": False,
            "passed_strict_sweep_gate": False,
            "best_config": {"top_k": 1, "temperature": 0.9, "run_id": "k1"},
        }

        markdown = markdown_table(rows, summary)

        self.assertIn("collapse_rate", markdown)
        self.assertIn("strict_valid", markdown)
        self.assertIn("avg_pair_repeat", markdown)
        self.assertIn("Strict Gate Failures", markdown)
        self.assertIn("top_k=1", markdown)


if __name__ == "__main__":
    unittest.main()
