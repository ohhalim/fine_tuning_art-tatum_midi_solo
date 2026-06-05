from __future__ import annotations

import unittest
from pathlib import Path

from scripts.run_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_postprocess_removal_dead_air_repair_probe import (
    BOUNDARY,
    FAIL_NEXT_BOUNDARY,
    PASS_NEXT_BOUNDARY,
    SOURCE_BOUNDARY,
    StageBMidiToSoloControlledScaleCheckpointTrainingScalePostprocessRemovalDeadAirRepairProbeError,
    aggregate_seed_rows,
    build_repair_report,
    validate_followup_decision,
    validate_repair_report,
)


def followup_decision_report(*, quality_claim: bool = False, wrong_target: bool = False) -> dict:
    return {
        "schema_version": "stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_dead_air_repeatability_temperature_guard_followup_decision_v1",
        "boundary": SOURCE_BOUNDARY,
        "evidence": {
            "sample_count": 9,
            "valid_sample_count": 8,
            "strict_valid_sample_count": 8,
            "grammar_gate_sample_count": 9,
            "strict_sample_shortfall": 1,
            "note_count_failure_count": 0,
            "grammar_failure_count": 0,
            "dead_air_failure_count": 1,
            "collapse_warning_sample_count": 0,
            "avg_postprocess_removal_ratio": 0.3611,
            "avg_onset_coverage_ratio": 0.5764,
            "avg_sustained_coverage_ratio": 0.7222,
        },
        "decision": {
            "next_boundary": BOUNDARY,
            "selected_target": "wrong_target" if wrong_target else "postprocess_removal_dead_air_repair",
            "temperature_followup_selected": False,
            "postprocess_removal_repair_selected": True,
            "repair_config": {
                "source_temperature": 0.75,
                "top_k": 4,
                "seeds": [47, 52, 60],
                "num_samples": 3,
                "max_sequence": 160,
                "constrained_note_groups_per_bar": 12,
                "coverage_position_window": 1,
                "chord_pitch_mode": "approach_tensions",
                "jazz_rhythm_profile": "swing_motif",
                "max_simultaneous_notes": 1,
                "target_avg_postprocess_removal_ratio": 0.3,
                "target_dead_air_failure_count": 0,
            },
        },
        "claim_boundary": {
            "midi_to_solo_musical_quality_claimed": quality_claim,
            "human_audio_preference_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
            "production_ready_improviser_claimed": False,
        },
    }


def seed_row(
    seed: int,
    *,
    strict_count: int = 3,
    dead_air_failures: dict | None = None,
    collapse: int = 0,
    avg_removal: float = 0.2083,
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
        "avg_postprocess_removal_ratio": avg_removal,
        "max_postprocess_removal_ratio": avg_removal,
        "avg_onset_coverage_ratio": 0.7396,
        "avg_sustained_coverage_ratio": 0.7708,
        "passed_strict_review_gate": strict_count >= 1,
    }


class StageBMidiToSoloControlledScaleCheckpointTrainingScalePostprocessRemovalDeadAirRepairProbeTest(
    unittest.TestCase
):
    def test_validates_followup_decision(self) -> None:
        summary = validate_followup_decision(followup_decision_report())

        self.assertEqual(summary["source_dead_air_failure_count"], 1)
        self.assertEqual(summary["temperature"], 0.75)
        self.assertEqual(summary["top_k"], 4)
        self.assertEqual(summary["seeds"], [47, 52, 60])
        self.assertEqual(summary["target_avg_postprocess_removal_ratio"], 0.3)

    def test_aggregate_seed_rows_records_removal_and_dead_air(self) -> None:
        aggregate = aggregate_seed_rows(
            [
                seed_row(47),
                seed_row(52, strict_count=2, dead_air_failures={"dead-air ratio too high: 0.846 >= 0.800": 1}),
                seed_row(60),
            ]
        )

        self.assertEqual(aggregate["seed_count"], 3)
        self.assertEqual(aggregate["strict_valid_sample_count"], 8)
        self.assertEqual(aggregate["dead_air_failure_count"], 1)
        self.assertAlmostEqual(aggregate["avg_postprocess_removal_ratio"], 0.2083)
        self.assertFalse(aggregate["all_samples_strict_valid"])

    def test_builds_qualified_repair_report(self) -> None:
        report = build_repair_report(
            run_dir=Path("outputs/postprocess_repair"),
            source_summary=validate_followup_decision(followup_decision_report()),
            seed_rows=[seed_row(47), seed_row(52), seed_row(60)],
            issue_number=602,
        )
        summary = validate_repair_report(
            report,
            expected_boundary=BOUNDARY,
            expected_next_boundary=PASS_NEXT_BOUNDARY,
            require_completed=True,
            require_no_quality_claim=True,
        )

        self.assertTrue(summary["postprocess_removal_dead_air_repair_target_qualified"])
        self.assertEqual(summary["strict_valid_sample_count"], 9)
        self.assertEqual(summary["dead_air_failure_count"], 0)
        self.assertEqual(summary["dead_air_failure_delta"], -1)
        self.assertLess(summary["avg_postprocess_removal_ratio"], 0.3)
        self.assertTrue(summary["avoid_reused_positions"])

    def test_builds_partial_repair_report_when_dead_air_remains(self) -> None:
        report = build_repair_report(
            run_dir=Path("outputs/postprocess_repair"),
            source_summary=validate_followup_decision(followup_decision_report()),
            seed_rows=[
                seed_row(47),
                seed_row(52, strict_count=2, dead_air_failures={"dead-air ratio too high: 0.846 >= 0.800": 1}),
                seed_row(60),
            ],
            issue_number=602,
        )
        summary = validate_repair_report(
            report,
            expected_boundary=BOUNDARY,
            expected_next_boundary=FAIL_NEXT_BOUNDARY,
            require_completed=True,
            require_no_quality_claim=True,
        )

        self.assertFalse(summary["postprocess_removal_dead_air_repair_target_qualified"])
        self.assertEqual(summary["strict_valid_sample_count"], 8)
        self.assertEqual(summary["dead_air_failure_count"], 1)
        self.assertEqual(summary["next_boundary"], FAIL_NEXT_BOUNDARY)

    def test_builds_partial_repair_report_when_removal_target_missed(self) -> None:
        report = build_repair_report(
            run_dir=Path("outputs/postprocess_repair"),
            source_summary=validate_followup_decision(followup_decision_report()),
            seed_rows=[
                seed_row(47, avg_removal=0.35),
                seed_row(52, avg_removal=0.35),
                seed_row(60, avg_removal=0.35),
            ],
            issue_number=602,
        )
        summary = validate_repair_report(
            report,
            expected_boundary=BOUNDARY,
            expected_next_boundary=FAIL_NEXT_BOUNDARY,
            require_completed=True,
            require_no_quality_claim=True,
        )

        self.assertFalse(summary["postprocess_removal_dead_air_repair_target_qualified"])
        self.assertGreater(summary["avg_postprocess_removal_ratio"], 0.3)

    def test_rejects_wrong_followup_target(self) -> None:
        with self.assertRaises(
            StageBMidiToSoloControlledScaleCheckpointTrainingScalePostprocessRemovalDeadAirRepairProbeError
        ):
            validate_followup_decision(followup_decision_report(wrong_target=True))

    def test_rejects_quality_claim(self) -> None:
        with self.assertRaises(
            StageBMidiToSoloControlledScaleCheckpointTrainingScalePostprocessRemovalDeadAirRepairProbeError
        ):
            validate_followup_decision(followup_decision_report(quality_claim=True))


if __name__ == "__main__":
    unittest.main()
