from __future__ import annotations

import unittest
from pathlib import Path

from scripts.run_stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repair_repeatability_probe import (
    BOUNDARY,
    FAIL_NEXT_BOUNDARY,
    PASS_NEXT_BOUNDARY,
    SOURCE_BOUNDARY,
    StageBMidiToSoloControlledScaleCheckpointDeadAirRepairRepeatabilityProbeError,
    aggregate_seed_rows,
    build_repeatability_report,
    parse_seeds,
    validate_repeatability_report,
)


def seed_row(seed: int, *, strict_count: int = 3, failures: dict | None = None, collapse: int = 0) -> dict:
    diagnostic = failures or {}
    return {
        "seed": seed,
        "generation_command": {"returncode": 0},
        "sample_count": 3,
        "valid_sample_count": strict_count,
        "strict_valid_sample_count": strict_count,
        "grammar_gate_sample_count": 3,
        "diagnostic_failure_reasons": diagnostic,
        "strict_failure_reasons": diagnostic,
        "collapse_warning_sample_count": collapse,
        "avg_postprocess_removal_ratio": 0.333,
        "avg_onset_coverage_ratio": 0.57,
        "avg_sustained_coverage_ratio": 0.72,
        "passed_strict_review_gate": strict_count >= 1,
    }


def source_summary() -> dict:
    return {
        "source_sample_count": 3,
        "source_valid_sample_count": 3,
        "source_strict_valid_sample_count": 3,
        "source_grammar_gate_sample_count": 3,
        "source_avg_postprocess_removal_ratio": 0.333,
        "source_avg_onset_coverage_ratio": 0.5729,
        "source_avg_sustained_coverage_ratio": 0.7292,
        "temperature": 0.9,
        "top_k": 4,
        "max_sequence": 160,
        "constrained_note_groups_per_bar": 12,
        "coverage_position_window": 1,
        "chord_pitch_mode": "approach_tensions",
        "jazz_rhythm_profile": "swing_motif",
        "max_simultaneous_notes": 1,
    }


class Args:
    issue_number = 564
    seeds = "44,52,60"
    num_samples = 3


class StageBMidiToSoloControlledScaleCheckpointDeadAirRepairRepeatabilityProbeTest(
    unittest.TestCase
):
    def test_parse_seeds(self) -> None:
        self.assertEqual(parse_seeds("44, 52,60"), [44, 52, 60])

    def test_aggregate_seed_rows(self) -> None:
        aggregate = aggregate_seed_rows([seed_row(44), seed_row(52), seed_row(60)])

        self.assertEqual(aggregate["seed_count"], 3)
        self.assertEqual(aggregate["sample_count"], 9)
        self.assertEqual(aggregate["strict_valid_sample_count"], 9)
        self.assertTrue(aggregate["all_samples_strict_valid"])

    def test_builds_qualified_repeatability_report(self) -> None:
        report = build_repeatability_report(
            run_dir=Path("outputs/repeatability"),
            source_summary=source_summary(),
            seed_rows=[seed_row(44), seed_row(52), seed_row(60)],
            args=Args(),
        )
        summary = validate_repeatability_report(
            report,
            expected_boundary=BOUNDARY,
            expected_next_boundary=PASS_NEXT_BOUNDARY,
            require_completed=True,
            require_no_quality_claim=True,
        )

        self.assertTrue(summary["dead_air_repair_repeatability_target_qualified"])
        self.assertEqual(summary["strict_valid_sample_count"], 9)
        self.assertEqual(summary["next_boundary"], PASS_NEXT_BOUNDARY)

    def test_builds_partial_repeatability_report_without_failing_validation(self) -> None:
        report = build_repeatability_report(
            run_dir=Path("outputs/repeatability"),
            source_summary=source_summary(),
            seed_rows=[
                seed_row(44),
                seed_row(52),
                seed_row(
                    60,
                    strict_count=1,
                    failures={"dead-air ratio too high: 0.846 >= 0.800": 1},
                    collapse=1,
                ),
            ],
            args=Args(),
        )
        summary = validate_repeatability_report(
            report,
            expected_boundary=BOUNDARY,
            expected_next_boundary=FAIL_NEXT_BOUNDARY,
            require_completed=True,
            require_no_quality_claim=True,
        )

        self.assertFalse(summary["dead_air_repair_repeatability_target_qualified"])
        self.assertEqual(summary["strict_valid_sample_count"], 7)
        self.assertEqual(summary["collapse_warning_sample_count"], 1)
        self.assertIn("dead-air ratio too high: 0.846 >= 0.800", summary["diagnostic_failure_reasons"])
        self.assertEqual(summary["next_boundary"], FAIL_NEXT_BOUNDARY)

    def test_rejects_quality_claim(self) -> None:
        report = build_repeatability_report(
            run_dir=Path("outputs/repeatability"),
            source_summary=source_summary(),
            seed_rows=[seed_row(44), seed_row(52), seed_row(60)],
            args=Args(),
        )
        report["readiness"]["midi_to_solo_musical_quality_claimed"] = True
        with self.assertRaises(
            StageBMidiToSoloControlledScaleCheckpointDeadAirRepairRepeatabilityProbeError
        ):
            validate_repeatability_report(
                report,
                expected_boundary=BOUNDARY,
                expected_next_boundary=PASS_NEXT_BOUNDARY,
                require_completed=True,
                require_no_quality_claim=True,
            )

    def test_boundary_constants_are_stable(self) -> None:
        self.assertEqual(SOURCE_BOUNDARY, "stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repair_probe")
        self.assertEqual(BOUNDARY, "stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repair_repeatability_probe")


if __name__ == "__main__":
    unittest.main()
