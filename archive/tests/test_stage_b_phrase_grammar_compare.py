from __future__ import annotations

import unittest

from scripts.run_stage_b_phrase_grammar_compare import (
    build_phrase_grammar_summary,
    config_run_id,
    markdown_table,
    parse_grammar_modes,
)


def row(
    grammar_mode: str,
    *,
    strict: int,
    sync: float,
    ioi_rep: float,
) -> dict:
    return {
        "grammar_mode": grammar_mode,
        "run_id": f"run_{grammar_mode}",
        "sample_count": 3,
        "valid_sample_count": strict,
        "strict_valid_sample_count": strict,
        "avg_root_tone_ratio": 0.1,
        "avg_tension_ratio": 0.2,
        "avg_approach_resolution_ratio": 0.8,
        "avg_syncopated_onset_ratio": sync,
        "avg_unique_bar_position_pattern_ratio": 0.5,
        "avg_duration_diversity_ratio": 0.1,
        "avg_most_common_duration_ratio": 0.5,
        "avg_ioi_diversity_ratio": 0.1,
        "avg_most_common_ioi_ratio": ioi_rep,
        "report_path": f"outputs/{grammar_mode}/report.json",
    }


class StageBPhraseGrammarCompareTest(unittest.TestCase):
    def test_parse_grammar_modes_rejects_unknown_mode(self) -> None:
        with self.assertRaises(ValueError):
            parse_grammar_modes("approach_baseline,random")

    def test_config_run_id_is_stable(self) -> None:
        self.assertEqual(
            config_run_id("run", "swing_motif_approach"),
            "run_swing_motif_approach",
        )

    def test_build_phrase_grammar_summary_reports_rhythm_delta(self) -> None:
        rows = [
            row("approach_baseline", strict=3, sync=0.50, ioi_rep=0.49),
            row("swing_motif_approach", strict=3, sync=0.75, ioi_rep=0.35),
        ]

        summary = build_phrase_grammar_summary(rows)

        self.assertTrue(summary["passed_compare_gate"])
        self.assertAlmostEqual(summary["syncopation_delta_swing_minus_baseline"], 0.25)
        self.assertAlmostEqual(summary["most_common_ioi_delta_swing_minus_baseline"], -0.14)

    def test_markdown_table_records_rhythm_columns(self) -> None:
        rows = [
            row("approach_baseline", strict=3, sync=0.50, ioi_rep=0.49),
            row("swing_motif_approach", strict=3, sync=0.75, ioi_rep=0.35),
        ]

        markdown = markdown_table(rows, build_phrase_grammar_summary(rows))

        self.assertIn("sync", markdown)
        self.assertIn("ioi-rep", markdown)
        self.assertIn("swing_motif_approach", markdown)


if __name__ == "__main__":
    unittest.main()
