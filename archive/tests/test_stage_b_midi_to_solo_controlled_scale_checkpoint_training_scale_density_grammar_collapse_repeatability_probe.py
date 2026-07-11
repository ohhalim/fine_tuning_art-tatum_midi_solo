from __future__ import annotations

import unittest
from pathlib import Path

from scripts.run_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_density_grammar_collapse_repeatability_probe import (
    BOUNDARY,
    DEAD_AIR_NEXT_BOUNDARY,
    FAIL_NEXT_BOUNDARY,
    PASS_NEXT_BOUNDARY,
    SOURCE_BOUNDARY,
    StageBMidiToSoloControlledScaleCheckpointTrainingScaleDensityGrammarCollapseRepeatabilityProbeError,
    aggregate_seed_rows,
    build_repeatability_report,
    parse_seeds,
    validate_repeatability_report,
)


def seed_row(seed: int, *, strict_count: int = 3, failures: dict | None = None, collapse: int = 0) -> dict:
    diagnostic = failures or {}
    dead_air_failures = sum(
        int(count)
        for reason, count in diagnostic.items()
        if str(reason).startswith("dead-air ratio too high:")
    )
    note_count_failures = sum(
        int(count)
        for reason, count in diagnostic.items()
        if str(reason).startswith("note count too low:")
    )
    return {
        "seed": seed,
        "generation_command": {"returncode": 0},
        "sample_count": 3,
        "valid_sample_count": strict_count,
        "strict_valid_sample_count": strict_count,
        "grammar_gate_sample_count": 3,
        "diagnostic_failure_reasons": diagnostic,
        "strict_failure_reasons": diagnostic,
        "note_count_failure_count": note_count_failures,
        "grammar_failure_count": int(diagnostic.get("grammar_gate_failed", 0)),
        "dead_air_failure_count": dead_air_failures,
        "collapse_warning_sample_count": collapse,
        "avg_postprocess_removal_ratio": 0.1875,
        "avg_onset_coverage_ratio": 0.46,
        "avg_sustained_coverage_ratio": 0.61,
        "passed_strict_review_gate": strict_count >= 1,
    }


def source_summary() -> dict:
    return {
        "source_sample_count": 3,
        "source_valid_sample_count": 1,
        "source_strict_valid_sample_count": 1,
        "source_grammar_gate_sample_count": 3,
        "source_note_count_failure_count": 0,
        "source_grammar_failure_count": 0,
        "source_dead_air_failure_count": 2,
        "source_collapse_warning_sample_count": 0,
        "source_avg_postprocess_removal_ratio": 0.1875,
        "source_avg_onset_coverage_ratio": 0.46875,
        "source_avg_sustained_coverage_ratio": 0.6146,
        "temperature": 0.9,
        "top_k": 4,
        "max_sequence": 160,
        "constrained_note_groups_per_bar": 8,
        "coverage_position_window": 1,
        "chord_pitch_mode": "approach_tensions",
        "jazz_rhythm_profile": "swing_motif",
        "max_simultaneous_notes": 1,
    }


class Args:
    issue_number = 588
    seeds = "47,52,60"
    num_samples = 3


class StageBMidiToSoloControlledScaleCheckpointTrainingScaleDensityGrammarCollapseRepeatabilityProbeTest(
    unittest.TestCase
):
    def test_parse_seeds(self) -> None:
        self.assertEqual(parse_seeds("47, 52,60"), [47, 52, 60])

    def test_aggregate_seed_rows(self) -> None:
        aggregate = aggregate_seed_rows([seed_row(47), seed_row(52), seed_row(60)])

        self.assertEqual(aggregate["seed_count"], 3)
        self.assertEqual(aggregate["sample_count"], 9)
        self.assertEqual(aggregate["strict_valid_sample_count"], 9)
        self.assertEqual(aggregate["note_count_failure_count"], 0)
        self.assertEqual(aggregate["grammar_failure_count"], 0)
        self.assertTrue(aggregate["all_samples_strict_valid"])

    def test_builds_qualified_repeatability_report(self) -> None:
        report = build_repeatability_report(
            run_dir=Path("outputs/repeatability"),
            source_summary=source_summary(),
            seed_rows=[seed_row(47), seed_row(52), seed_row(60)],
            args=Args(),
        )
        summary = validate_repeatability_report(
            report,
            expected_boundary=BOUNDARY,
            expected_next_boundary=PASS_NEXT_BOUNDARY,
            require_completed=True,
            require_no_quality_claim=True,
        )

        self.assertTrue(summary["density_grammar_collapse_repeatability_target_supported"])
        self.assertTrue(summary["strict_gate_stable"])
        self.assertEqual(summary["strict_valid_sample_count"], 9)
        self.assertEqual(summary["next_boundary"], PASS_NEXT_BOUNDARY)

    def test_builds_supported_repeatability_with_dead_air_remaining(self) -> None:
        report = build_repeatability_report(
            run_dir=Path("outputs/repeatability"),
            source_summary=source_summary(),
            seed_rows=[
                seed_row(47, strict_count=1, failures={"dead-air ratio too high: 0.833 >= 0.800": 2}),
                seed_row(52, strict_count=0, failures={"dead-air ratio too high: 0.846 >= 0.800": 3}),
                seed_row(
                    60,
                    strict_count=1,
                    failures={"dead-air ratio too high: 0.917 >= 0.800": 2},
                ),
            ],
            args=Args(),
        )
        summary = validate_repeatability_report(
            report,
            expected_boundary=BOUNDARY,
            expected_next_boundary=DEAD_AIR_NEXT_BOUNDARY,
            require_completed=True,
            require_no_quality_claim=True,
        )

        self.assertTrue(summary["density_grammar_collapse_repeatability_target_supported"])
        self.assertFalse(summary["strict_gate_stable"])
        self.assertTrue(summary["dead_air_remaining"])
        self.assertEqual(summary["strict_valid_sample_count"], 2)
        self.assertEqual(summary["dead_air_failure_count"], 7)
        self.assertEqual(summary["collapse_warning_sample_count"], 0)
        self.assertIn("dead-air ratio too high: 0.846 >= 0.800", summary["diagnostic_failure_reasons"])
        self.assertEqual(summary["next_boundary"], DEAD_AIR_NEXT_BOUNDARY)

    def test_builds_followup_when_density_target_regresses(self) -> None:
        report = build_repeatability_report(
            run_dir=Path("outputs/repeatability"),
            source_summary=source_summary(),
            seed_rows=[
                seed_row(47),
                seed_row(52, strict_count=0, failures={"note count too low: 4 < 6": 1}),
                seed_row(60),
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

        self.assertFalse(summary["density_grammar_collapse_repeatability_target_supported"])
        self.assertEqual(summary["note_count_failure_count"], 1)
        self.assertEqual(summary["next_boundary"], FAIL_NEXT_BOUNDARY)

    def test_rejects_quality_claim(self) -> None:
        report = build_repeatability_report(
            run_dir=Path("outputs/repeatability"),
            source_summary=source_summary(),
            seed_rows=[seed_row(47), seed_row(52), seed_row(60)],
            args=Args(),
        )
        report["readiness"]["midi_to_solo_musical_quality_claimed"] = True
        with self.assertRaises(
            StageBMidiToSoloControlledScaleCheckpointTrainingScaleDensityGrammarCollapseRepeatabilityProbeError
        ):
            validate_repeatability_report(
                report,
                expected_boundary=BOUNDARY,
                expected_next_boundary=PASS_NEXT_BOUNDARY,
                require_completed=True,
                require_no_quality_claim=True,
            )

    def test_boundary_constants_are_stable(self) -> None:
        self.assertEqual(
            SOURCE_BOUNDARY,
            "stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_density_grammar_collapse_repair_probe",
        )
        self.assertEqual(
            BOUNDARY,
            "stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_density_grammar_collapse_repeatability_probe",
        )


if __name__ == "__main__":
    unittest.main()
