from __future__ import annotations

import unittest

from scripts.run_stage_b_pitch_mode_compare import (
    build_pitch_mode_summary,
    config_run_id,
    markdown_table,
    parse_pitch_modes,
)


def row(
    pitch_mode: str,
    *,
    strict: int,
    valid: int,
    root: float,
    tension: float,
) -> dict:
    return {
        "pitch_mode": pitch_mode,
        "run_id": f"run_{pitch_mode}",
        "sample_count": 3,
        "valid_sample_count": valid,
        "strict_valid_sample_count": strict,
        "avg_chord_tone_ratio": 1.0 - tension,
        "avg_root_tone_ratio": root,
        "avg_tension_ratio": tension,
        "avg_onset_coverage_ratio": 0.5,
        "avg_sustained_coverage_ratio": 0.75,
        "collapse_warning_sample_rate": 0.0,
        "passed_strict_review_gate": True,
        "diagnostic_failure_reasons": {},
        "report_path": f"outputs/{pitch_mode}/report.json",
    }


class StageBPitchModeCompareTest(unittest.TestCase):
    def test_parse_pitch_modes_rejects_unknown_mode(self) -> None:
        with self.assertRaises(ValueError):
            parse_pitch_modes("tones,chromatic")

    def test_parse_pitch_modes_accepts_expected_modes(self) -> None:
        self.assertEqual(parse_pitch_modes("tones,tones_tensions"), ["tones", "tones_tensions"])

    def test_config_run_id_is_stable(self) -> None:
        self.assertEqual(
            config_run_id("run", pitch_mode="tones_tensions", note_groups_per_bar=8, top_k=2, temperature=0.9),
            "run_coverage_chord_tones_tensions_g8_k2_t0p9",
        )

    def test_build_pitch_mode_summary_reports_deltas(self) -> None:
        rows = [
            row("tones", strict=3, valid=3, root=0.28, tension=0.0),
            row("tones_tensions", strict=3, valid=3, root=0.20, tension=0.22),
        ]

        summary = build_pitch_mode_summary(rows, min_best_strict_valid_samples=1)

        self.assertTrue(summary["passed_compare_gate"])
        self.assertAlmostEqual(summary["root_tone_ratio_delta_tensions_minus_tones"], -0.08)
        self.assertAlmostEqual(summary["tension_ratio_delta_tensions_minus_tones"], 0.22)
        self.assertEqual(summary["best_tones_tensions_config"]["avg_tension_ratio"], 0.22)

    def test_build_pitch_mode_summary_fails_without_tension_mode(self) -> None:
        summary = build_pitch_mode_summary([row("tones", strict=3, valid=3, root=0.28, tension=0.0)])

        self.assertFalse(summary["passed_compare_gate"])
        self.assertFalse(summary["comparison_ready"])

    def test_markdown_table_records_pitch_role_columns(self) -> None:
        rows = [
            row("tones", strict=3, valid=3, root=0.28, tension=0.0),
            row("tones_tensions", strict=3, valid=3, root=0.20, tension=0.22),
        ]
        summary = build_pitch_mode_summary(rows)

        markdown = markdown_table(rows, summary)

        self.assertIn("pitch mode", markdown)
        self.assertIn("root", markdown)
        self.assertIn("tension", markdown)
        self.assertIn("tones_tensions", markdown)


if __name__ == "__main__":
    unittest.main()
