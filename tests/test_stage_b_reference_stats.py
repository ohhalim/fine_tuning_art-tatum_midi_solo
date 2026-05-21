from __future__ import annotations

import unittest

from scripts.run_stage_b_reference_stats import (
    analyze_token_record,
    compare_generated_to_reference,
    generated_rows_from_report,
    summarize_reference_rows,
)
from scripts.stage_b_tokens import (
    TOKEN_END,
    chord_tokens,
    note_duration_token,
    note_pitch_token,
    note_velocity_token,
    position_token,
)


def token_phrase() -> list[int]:
    return [
        *chord_tokens("Cmaj7"),
        position_token(0),
        note_velocity_token(4),
        note_pitch_token(60),
        note_duration_token(2),
        position_token(3),
        note_velocity_token(4),
        note_pitch_token(62),
        note_duration_token(1),
        position_token(7),
        note_velocity_token(4),
        note_pitch_token(65),
        note_duration_token(3),
        position_token(11),
        note_velocity_token(4),
        note_pitch_token(64),
        note_duration_token(1),
        TOKEN_END,
    ]


class StageBReferenceStatsTest(unittest.TestCase):
    def test_analyze_token_record_reports_phrase_metrics(self) -> None:
        report = analyze_token_record(token_phrase(), bars=1)

        self.assertEqual(report["metrics"]["note_group_count"], 4.0)
        self.assertEqual(report["metrics"]["unique_pitch_count"], 4.0)
        self.assertGreater(report["metrics"]["syncopated_onset_ratio"], 0.5)
        self.assertGreater(report["metrics"]["direction_change_ratio"], 0.0)

    def test_summarize_reference_rows_builds_distribution(self) -> None:
        rows = [
            {"metrics": {"note_group_count": 4.0, "syncopated_onset_ratio": 0.5}},
            {"metrics": {"note_group_count": 8.0, "syncopated_onset_ratio": 0.75}},
        ]

        summary = summarize_reference_rows(rows)

        self.assertEqual(summary["record_count"], 2)
        self.assertAlmostEqual(summary["metrics"]["note_group_count"]["mean"], 6.0)
        self.assertAlmostEqual(summary["metrics"]["syncopated_onset_ratio"]["mean"], 0.625)

    def test_generated_rows_from_report_maps_comparable_rhythm_fields(self) -> None:
        generated = generated_rows_from_report(
            {
                "rows": [
                    {
                        "grammar_mode": "swing_motif_approach",
                        "avg_syncopated_onset_ratio": 0.75,
                        "avg_unique_bar_position_pattern_ratio": 0.5,
                        "avg_most_common_ioi_ratio": 0.47,
                    }
                ]
            }
        )

        self.assertEqual(generated[0]["label"], "swing_motif_approach")
        self.assertAlmostEqual(generated[0]["metrics"]["syncopated_onset_ratio"], 0.75)

    def test_compare_generated_to_reference_reports_deltas(self) -> None:
        comparisons = compare_generated_to_reference(
            [{"label": "generated", "metrics": {"syncopated_onset_ratio": 0.75}}],
            {"metrics": {"syncopated_onset_ratio": {"mean": 0.55}}},
        )

        self.assertAlmostEqual(
            comparisons[0]["delta_from_reference_mean"]["syncopated_onset_ratio"],
            0.20,
        )


if __name__ == "__main__":
    unittest.main()
