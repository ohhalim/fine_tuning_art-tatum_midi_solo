from __future__ import annotations

import unittest

from scripts.run_stage_b_raw_generation_repeatability_sweep import (
    build_repeatability_summary,
    best_strict_candidate,
    markdown_report,
    parse_seed_values,
    probe_run_id,
    range_label,
    reason_label,
)


class StageBRawGenerationRepeatabilityTest(unittest.TestCase):
    def test_parse_seed_values_rejects_empty_input(self) -> None:
        with self.assertRaises(ValueError):
            parse_seed_values("")

    def test_probe_run_id_is_stable(self) -> None:
        self.assertEqual(
            probe_run_id("run", seed=17, max_files=2),
            "run_seed17_files2",
        )

    def test_range_label_formats_compact_range(self) -> None:
        self.assertEqual(range_label({"min": 0.5, "max": 1.0}), "0.500-1.000")
        self.assertEqual(range_label({"min": 8, "max": 16}, digits=0), "8-16")

    def test_reason_label_formats_sorted_reason_counts(self) -> None:
        self.assertEqual(reason_label({}), "none")
        self.assertEqual(reason_label({"b": 2, "a": 1}), "a (1), b (2)")

    def test_build_repeatability_summary_requires_each_seed(self) -> None:
        rows = [
            {
                "seed": 17,
                "returncode": 0,
                "input_file_count": 2,
                "sample_count": 3,
                "valid_sample_count": 3,
                "strict_valid_sample_count": 3,
                "grammar_gate_sample_count": 3,
                "max_postprocess_removal_ratio": 0.2,
                "dead_air_outlier_count": 0,
                "best_strict_candidate": {"seed": 17, "sample_index": 1, "dead_air_ratio": 0.5},
            },
            {
                "seed": 23,
                "returncode": 0,
                "input_file_count": 2,
                "sample_count": 3,
                "valid_sample_count": 3,
                "strict_valid_sample_count": 0,
                "grammar_gate_sample_count": 3,
                "max_postprocess_removal_ratio": 0.2,
                "dead_air_outlier_count": 0,
                "best_strict_candidate": None,
            },
            {
                "seed": 31,
                "returncode": 0,
                "input_file_count": 2,
                "sample_count": 3,
                "valid_sample_count": 3,
                "strict_valid_sample_count": 3,
                "grammar_gate_sample_count": 3,
                "max_postprocess_removal_ratio": 0.2,
                "dead_air_outlier_count": 0,
                "best_strict_candidate": {"seed": 31, "sample_index": 2, "dead_air_ratio": 0.7},
            },
        ]

        summary = build_repeatability_summary(
            rows,
            min_seed_count=3,
            min_source_files=2,
            min_strict_samples_per_seed=1,
            min_overall_strict_rate=0.67,
            max_postprocess_removal_ratio=0.49,
            max_dead_air_outlier_rate=0.25,
        )

        self.assertFalse(summary["passed_repeatability_gate"])
        self.assertEqual(summary["failing_seeds"], [23])

    def test_best_strict_candidate_prefers_low_dead_air(self) -> None:
        candidate = best_strict_candidate(
            17,
            "run",
            [
                {
                    "sample_index": 1,
                    "strict_valid": True,
                    "metrics": {"dead_air_ratio": 0.7, "note_count": 12},
                    "collapse": {"postprocess_removal_ratio": 0.1},
                },
                {
                    "sample_index": 2,
                    "strict_valid": True,
                    "metrics": {"dead_air_ratio": 0.4, "note_count": 8},
                    "collapse": {"postprocess_removal_ratio": 0.2},
                },
                {
                    "sample_index": 3,
                    "strict_valid": False,
                    "metrics": {"dead_air_ratio": 0.1, "note_count": 16},
                    "collapse": {"postprocess_removal_ratio": 0.0},
                },
            ],
        )

        self.assertIsNotNone(candidate)
        self.assertEqual(candidate["sample_index"], 2)
        self.assertEqual(candidate["dead_air_ratio"], 0.4)

    def test_build_repeatability_summary_tracks_dead_air_candidate_gate(self) -> None:
        rows = [
            {
                "seed": 17,
                "returncode": 0,
                "input_file_count": 2,
                "sample_count": 3,
                "valid_sample_count": 3,
                "strict_valid_sample_count": 3,
                "grammar_gate_sample_count": 3,
                "max_postprocess_removal_ratio": 0.2,
                "dead_air_outlier_count": 1,
                "best_strict_candidate": {
                    "seed": 17,
                    "sample_index": 2,
                    "dead_air_ratio": 0.6,
                    "postprocess_removal_ratio": 0.1,
                    "note_count": 12,
                },
            },
            {
                "seed": 23,
                "returncode": 0,
                "input_file_count": 2,
                "sample_count": 3,
                "valid_sample_count": 3,
                "strict_valid_sample_count": 3,
                "grammar_gate_sample_count": 3,
                "max_postprocess_removal_ratio": 0.2,
                "dead_air_outlier_count": 0,
                "best_strict_candidate": {
                    "seed": 23,
                    "sample_index": 1,
                    "dead_air_ratio": 0.4,
                    "postprocess_removal_ratio": 0.2,
                    "note_count": 10,
                },
            },
        ]

        summary = build_repeatability_summary(
            rows,
            min_seed_count=2,
            min_source_files=2,
            min_strict_samples_per_seed=1,
            min_overall_strict_rate=0.67,
            max_postprocess_removal_ratio=0.49,
            max_dead_air_outlier_rate=0.25,
        )

        self.assertTrue(summary["passed_repeatability_gate"])
        self.assertEqual(summary["total_dead_air_outlier_count"], 1)
        self.assertAlmostEqual(summary["dead_air_outlier_rate"], 1 / 6)
        self.assertEqual(summary["selected_best_candidate"]["seed"], 23)

    def test_markdown_report_contains_gate_and_seed_rows(self) -> None:
        rows = [
            {
                "seed": 17,
                "input_file_count": 2,
                "sample_count": 3,
                "grammar_gate_sample_count": 3,
                "valid_sample_count": 3,
                "strict_valid_sample_count": 3,
                "note_count_range": {"min": 10, "max": 16},
                "unique_pitch_count_range": {"min": 4, "max": 6},
                "max_simultaneous_notes_range": {"min": 1, "max": 2},
                "phrase_coverage_ratio_range": {"min": 0.8, "max": 1.0},
                "max_postprocess_removal_ratio": 0.2,
                "dead_air_outlier_count": 0,
                "best_strict_candidate": {
                    "seed": 17,
                    "sample_index": 2,
                    "dead_air_ratio": 0.4,
                },
                "passed_strict_review_gate": True,
                "failure_reasons": {},
                "strict_failure_reasons": {},
            }
        ]
        summary = {
            "passed_repeatability_gate": True,
            "seed_values": [17],
            "min_observed_input_files": 2,
            "total_samples": 3,
            "strict_valid_sample_rate": 1.0,
            "grammar_gate_sample_rate": 1.0,
            "max_postprocess_removal_ratio": 0.2,
            "dead_air_outlier_rate": 0.0,
            "selected_best_candidate": {"seed": 17, "sample_index": 2, "dead_air_ratio": 0.4},
        }

        markdown = markdown_report(rows, summary)

        self.assertIn("passed repeatability gate", markdown)
        self.assertIn("| 17 |", markdown)
        self.assertIn("strict pass-rate", markdown)
        self.assertIn("selected best candidate", markdown)


if __name__ == "__main__":
    unittest.main()
