from __future__ import annotations

import unittest
from pathlib import Path

from scripts.run_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_dead_air_repair_repeatability_probe import (
    BOUNDARY,
    FAIL_NEXT_BOUNDARY,
    PASS_NEXT_BOUNDARY,
    SOURCE_BOUNDARY,
    StageBMidiToSoloControlledScaleCheckpointTrainingScaleDeadAirRepairRepeatabilityProbeError,
    aggregate_seed_rows,
    build_repeatability_report,
    parse_seeds,
    validate_repair_probe,
    validate_repeatability_report,
)


def seed_row(
    seed: int,
    *,
    strict_count: int = 3,
    dead_air_failures: dict | None = None,
    collapse: int = 0,
) -> dict:
    diagnostic = dead_air_failures or {}
    return {
        "seed": seed,
        "generation_command": {"returncode": 0},
        "sample_count": 3,
        "valid_sample_count": strict_count,
        "strict_valid_sample_count": strict_count,
        "grammar_gate_sample_count": 3,
        "note_count_failure_count": 0,
        "grammar_failure_count": 0,
        "dead_air_failure_count": sum(diagnostic.values()),
        "diagnostic_failure_reasons": diagnostic,
        "strict_failure_reasons": {
            f"midi_review_gate_failed: {reason}": count
            for reason, count in diagnostic.items()
        },
        "collapse_warning_sample_count": collapse,
        "avg_postprocess_removal_ratio": 0.39,
        "avg_onset_coverage_ratio": 0.57,
        "avg_sustained_coverage_ratio": 0.71,
        "passed_strict_review_gate": strict_count >= 1,
    }


def repair_report(*, quality_claim: bool = False) -> dict:
    return {
        "boundary": SOURCE_BOUNDARY,
        "input": {
            "temperature": 0.9,
            "top_k": 4,
            "max_sequence": 160,
            "constrained_note_groups_per_bar": 12,
            "coverage_position_window": 1,
            "chord_pitch_mode": "approach_tensions",
            "jazz_rhythm_profile": "swing_motif",
            "max_simultaneous_notes": 1,
        },
        "repair_summary": {
            "sample_count": 3,
            "valid_sample_count": 3,
            "strict_valid_sample_count": 3,
            "grammar_gate_sample_count": 3,
            "note_count_failure_count": 0,
            "grammar_failure_count": 0,
            "dead_air_failure_count": 0,
            "collapse_warning_sample_count": 0,
            "avg_postprocess_removal_ratio": 0.3889,
            "avg_onset_coverage_ratio": 0.5729,
            "avg_sustained_coverage_ratio": 0.7083,
        },
        "readiness": {
            "boundary": SOURCE_BOUNDARY,
            "selected_scale_dead_air_target_qualified": True,
            "model_direct_generation_quality_claimed": False,
            "midi_to_solo_musical_quality_claimed": quality_claim,
            "human_audio_preference_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
        },
        "decision": {
            "next_boundary": BOUNDARY,
        },
    }


class Args:
    issue_number = 594
    seeds = "47,52,60"
    num_samples = 3


class StageBMidiToSoloControlledScaleCheckpointTrainingScaleDeadAirRepairRepeatabilityProbeTest(
    unittest.TestCase
):
    def test_parse_seeds(self) -> None:
        self.assertEqual(parse_seeds("47, 52,60"), [47, 52, 60])

    def test_validates_selected_scale_repair_probe(self) -> None:
        source = validate_repair_probe(repair_report())

        self.assertEqual(source["source_sample_count"], 3)
        self.assertEqual(source["source_strict_valid_sample_count"], 3)
        self.assertEqual(source["source_dead_air_failure_count"], 0)
        self.assertEqual(source["constrained_note_groups_per_bar"], 12)

    def test_aggregate_seed_rows_counts_dead_air(self) -> None:
        aggregate = aggregate_seed_rows(
            [
                seed_row(47),
                seed_row(
                    52,
                    strict_count=1,
                    dead_air_failures={"dead-air ratio too high: 0.833 >= 0.800": 2},
                    collapse=1,
                ),
                seed_row(60),
            ]
        )

        self.assertEqual(aggregate["seed_count"], 3)
        self.assertEqual(aggregate["sample_count"], 9)
        self.assertEqual(aggregate["strict_valid_sample_count"], 7)
        self.assertEqual(aggregate["dead_air_failure_count"], 2)
        self.assertEqual(aggregate["collapse_warning_sample_count"], 1)
        self.assertFalse(aggregate["all_samples_strict_valid"])

    def test_builds_qualified_repeatability_report(self) -> None:
        report = build_repeatability_report(
            run_dir=Path("outputs/repeatability"),
            source_summary=validate_repair_probe(repair_report()),
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

        self.assertTrue(
            summary["selected_scale_dead_air_repair_repeatability_target_qualified"]
        )
        self.assertEqual(summary["strict_valid_sample_count"], 9)
        self.assertEqual(summary["dead_air_failure_count"], 0)
        self.assertEqual(summary["next_boundary"], PASS_NEXT_BOUNDARY)

    def test_builds_partial_repeatability_report_without_failing_validation(self) -> None:
        report = build_repeatability_report(
            run_dir=Path("outputs/repeatability"),
            source_summary=validate_repair_probe(repair_report()),
            seed_rows=[
                seed_row(47),
                seed_row(
                    52,
                    strict_count=1,
                    dead_air_failures={
                        "dead-air ratio too high: 0.833 >= 0.800": 1,
                        "dead-air ratio too high: 1.000 >= 0.800; collapse=postprocess_removed_majority": 1,
                    },
                    collapse=1,
                ),
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

        self.assertFalse(
            summary["selected_scale_dead_air_repair_repeatability_target_qualified"]
        )
        self.assertEqual(summary["strict_valid_sample_count"], 7)
        self.assertEqual(summary["dead_air_failure_count"], 2)
        self.assertEqual(summary["collapse_warning_sample_count"], 1)
        self.assertEqual(summary["next_boundary"], FAIL_NEXT_BOUNDARY)

    def test_rejects_quality_claim(self) -> None:
        with self.assertRaises(
            StageBMidiToSoloControlledScaleCheckpointTrainingScaleDeadAirRepairRepeatabilityProbeError
        ):
            validate_repair_probe(repair_report(quality_claim=True))

    def test_boundary_constants_are_stable(self) -> None:
        self.assertEqual(
            SOURCE_BOUNDARY,
            "stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_dead_air_repair_probe",
        )
        self.assertEqual(
            BOUNDARY,
            "stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_dead_air_repair_repeatability_probe",
        )


if __name__ == "__main__":
    unittest.main()
