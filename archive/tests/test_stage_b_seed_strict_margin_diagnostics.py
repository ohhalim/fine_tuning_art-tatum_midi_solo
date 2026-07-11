from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.diagnose_stage_b_seed_strict_margins import (
    build_diagnostic_report,
    failure_tags,
    markdown_report,
    write_json,
)


def sample(
    index: int,
    *,
    strict: bool,
    reason: str | None,
    pitches: int,
    dead_air: float,
) -> dict:
    return {
        "sample_index": index,
        "valid": strict,
        "strict_valid": strict,
        "failure_reason": reason,
        "diagnostic_failure_reason": reason,
        "metrics": {
            "note_count": 12,
            "unique_pitch_count": pitches,
            "dead_air_ratio": dead_air,
            "phrase_coverage_ratio": 0.75,
        },
        "temporal_coverage": {
            "onset_coverage_ratio": 0.5,
            "sustained_coverage_ratio": 0.625,
            "position_span_ratio": 0.75,
            "head_empty_steps": 1,
            "tail_empty_steps": 0,
        },
        "collapse": {"postprocess_removal_ratio": 0.1},
        "grammar": {"grammar_valid": True},
    }


class StageBSeedStrictMarginDiagnosticsTest(unittest.TestCase):
    def test_failure_tags_split_dead_air_and_unique_pitch(self) -> None:
        dead_air_tags = failure_tags(
            sample(1, strict=False, reason="dead-air ratio too high: 0.857 >= 0.800", pitches=3, dead_air=0.857),
            dead_air_gate=0.8,
            min_unique_pitches=3,
        )
        pitch_tags = failure_tags(
            sample(2, strict=False, reason="unique pitch count too low: 2 < 3", pitches=2, dead_air=0.714),
            dead_air_gate=0.8,
            min_unique_pitches=3,
        )

        self.assertIn("dead_air", dead_air_tags)
        self.assertNotIn("unique_pitch", dead_air_tags)
        self.assertIn("unique_pitch", pitch_tags)
        self.assertNotIn("dead_air", pitch_tags)

    def test_build_diagnostic_report_marks_seed_margin_warning(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            seed17_report = root / "seed17_report.json"
            seed23_report = root / "seed23_report.json"
            summary_path = root / "repeatability_summary.json"

            write_json(
                seed17_report,
                {
                    "samples": [
                        sample(
                            1,
                            strict=False,
                            reason="dead-air ratio too high: 0.857 >= 0.800",
                            pitches=3,
                            dead_air=0.857,
                        ),
                        sample(
                            2,
                            strict=False,
                            reason="unique pitch count too low: 2 < 3",
                            pitches=2,
                            dead_air=0.714,
                        ),
                        sample(3, strict=True, reason=None, pitches=4, dead_air=0.5),
                    ]
                },
            )
            write_json(
                seed23_report,
                {
                    "samples": [
                        sample(1, strict=True, reason=None, pitches=4, dead_air=0.375),
                        sample(2, strict=True, reason=None, pitches=5, dead_air=0.5),
                        sample(3, strict=True, reason=None, pitches=4, dead_air=0.625),
                    ]
                },
            )
            write_json(
                summary_path,
                {
                    "run_id": "unit",
                    "issue": 999,
                    "summary": {"min_strict_samples_per_seed": 1},
                    "rows": [
                        {
                            "seed": 17,
                            "report_path": str(seed17_report),
                            "best_strict_candidate": {"sample_index": 3, "dead_air_ratio": 0.5},
                        },
                        {
                            "seed": 23,
                            "report_path": str(seed23_report),
                            "best_strict_candidate": {"sample_index": 1, "dead_air_ratio": 0.375},
                        },
                    ],
                },
            )

            report = build_diagnostic_report(
                summary_path,
                warning_min_strict_samples_per_seed=2,
                dead_air_gate=0.8,
                min_unique_pitches=3,
            )

        self.assertEqual(report["summary"]["margin_warning_seeds"], [17])
        self.assertEqual(report["summary"]["dead_air_unique_pitch_overlap_seeds"], [])
        self.assertEqual(report["summary"]["dead_air_unique_pitch_separate_seeds"], [17])
        seed17 = report["seeds"][0]
        self.assertTrue(seed17["margin_warning"])
        self.assertEqual(seed17["dead_air_sample_indices"], [1])
        self.assertEqual(seed17["unique_pitch_sample_indices"], [2])
        self.assertEqual(seed17["dead_air_unique_pitch_overlap_indices"], [])

    def test_markdown_report_lists_warning_samples(self) -> None:
        report = {
            "source_summary_path": "summary.json",
            "hard_min_strict_samples_per_seed": 1,
            "warning_min_strict_samples_per_seed": 2,
            "summary": {
                "margin_warning_seeds": [17],
                "dead_air_unique_pitch_overlap_seeds": [],
                "dead_air_unique_pitch_separate_seeds": [17],
            },
            "seeds": [
                {
                    "seed": 17,
                    "sample_count": 3,
                    "strict_valid_sample_count": 1,
                    "margin_warning": True,
                    "dead_air_sample_indices": [1],
                    "unique_pitch_sample_indices": [2],
                    "dead_air_unique_pitch_overlap_indices": [],
                    "failure_tag_counts": {"dead_air": 1, "strict_invalid": 2, "unique_pitch": 1},
                    "best_strict_candidate": {"sample_index": 3, "dead_air_ratio": 0.5},
                }
            ],
            "samples": [
                {
                    "seed": 17,
                    "sample_index": 1,
                    "strict_valid": False,
                    "note_count": 15,
                    "unique_pitch_count": 3,
                    "dead_air_ratio": 0.857,
                    "phrase_coverage_ratio": 1.0,
                    "onset_coverage_ratio": 0.3125,
                    "sustained_coverage_ratio": 0.84375,
                    "tail_empty_steps": 0,
                    "postprocess_removal_ratio": 0.25,
                    "failure_tags": ["strict_invalid", "dead_air"],
                    "failure_reason": "dead-air ratio too high",
                }
            ],
        }

        markdown = markdown_report(report)

        self.assertIn("strict margin warning seeds: `[17]`", markdown)
        self.assertIn("dead-air + unique-pitch separate seeds: `[17]`", markdown)
        self.assertIn("| 17 | 1 | False | 15 | 3 | 0.857", markdown)


if __name__ == "__main__":
    unittest.main()
