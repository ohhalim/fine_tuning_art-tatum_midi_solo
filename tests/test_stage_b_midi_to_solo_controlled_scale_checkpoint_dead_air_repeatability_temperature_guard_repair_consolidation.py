from __future__ import annotations

import unittest
from pathlib import Path

from scripts.consolidate_stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repeatability_temperature_guard_repair import (
    BOUNDARY,
    NEXT_BOUNDARY,
    SOURCE_BOUNDARY,
    StageBMidiToSoloControlledScaleCheckpointTemperatureGuardRepairConsolidationError,
    build_consolidation_report,
    validate_consolidation_report,
)


def repair_probe_report(
    *,
    target_qualified: bool = True,
    strict_count: int = 9,
    dead_air_count: int = 0,
    collapse_count: int = 0,
    quality_claim: bool = False,
) -> dict:
    return {
        "schema_version": "stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repeatability_temperature_guard_repair_probe_v1",
        "boundary": SOURCE_BOUNDARY,
        "input": {
            "source_temperature": 0.9,
            "temperature": 0.75,
            "top_k": 4,
        },
        "seed_rows": [
            {"seed": seed, "generation_report_path": f"outputs/seed_{seed}/report.json"}
            for seed in [44, 52, 60]
        ],
        "aggregate": {
            "seed_count": 3,
            "seeds": [44, 52, 60],
            "sample_count": 9,
            "valid_sample_count": strict_count,
            "strict_valid_sample_count": strict_count,
            "grammar_gate_sample_count": 9,
            "collapse_warning_sample_count": collapse_count,
            "diagnostic_failure_reasons": (
                {} if dead_air_count == 0 else {"dead-air ratio too high: 0.812 >= 0.800": dead_air_count}
            ),
            "avg_postprocess_removal_ratio": 0.36574074074074076,
            "avg_onset_coverage_ratio": 0.5486111111111112,
            "avg_sustained_coverage_ratio": 0.7083333333333334,
            "all_seed_gate_passed": True,
            "all_samples_strict_valid": strict_count == 9,
        },
        "failure_summary": {
            "dead_air_failure_count": dead_air_count,
            "postprocess_collapse_failure_count": 0,
        },
        "comparison": {
            "strict_valid_sample_delta": strict_count - 7,
            "source_strict_sample_shortfall": 2,
            "repair_strict_sample_shortfall": 9 - strict_count,
            "source_dead_air_failure_count": 2,
            "repair_dead_air_failure_count": dead_air_count,
            "source_collapse_warning_sample_count": 1,
            "repair_collapse_warning_sample_count": collapse_count,
        },
        "readiness": {
            "boundary": SOURCE_BOUNDARY,
            "temperature_guard_repair_probe_completed": True,
            "temperature_guard_repair_target_qualified": target_qualified,
            "midi_to_solo_musical_quality_claimed": quality_claim,
            "human_audio_preference_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
        },
        "decision": {
            "current_boundary": SOURCE_BOUNDARY,
            "next_boundary": BOUNDARY,
            "critical_user_input_required": False,
        },
    }


class StageBMidiToSoloControlledScaleCheckpointDeadAirRepeatabilityTemperatureGuardRepairConsolidationTest(
    unittest.TestCase
):
    def test_consolidates_temperature_guard_support(self) -> None:
        report = build_consolidation_report(
            repair_probe_report(),
            output_dir=Path("outputs/temperature_guard_consolidation"),
        )
        summary = validate_consolidation_report(
            report,
            expected_boundary=BOUNDARY,
            expected_next_boundary=NEXT_BOUNDARY,
            min_sample_count=9,
            require_objective_support=True,
            require_audio_review_required=True,
            require_no_quality_claim=True,
        )

        self.assertEqual(summary["sample_count"], 9)
        self.assertEqual(summary["strict_valid_sample_count"], 9)
        self.assertEqual(summary["dead_air_failure_count"], 0)
        self.assertEqual(summary["collapse_warning_sample_count"], 0)
        self.assertEqual(summary["strict_valid_sample_delta"], 2)
        self.assertEqual(summary["repair_strict_sample_shortfall"], 0)
        self.assertTrue(summary["objective_temperature_guard_support"])
        self.assertTrue(summary["audio_review_package_required"])
        self.assertFalse(summary["additional_repair_required"])
        self.assertFalse(summary["human_audio_preference_claimed"])
        self.assertFalse(summary["midi_to_solo_musical_quality_claimed"])
        self.assertEqual(summary["next_boundary"], NEXT_BOUNDARY)

    def test_rejects_unqualified_probe(self) -> None:
        with self.assertRaises(
            StageBMidiToSoloControlledScaleCheckpointTemperatureGuardRepairConsolidationError
        ):
            build_consolidation_report(
                repair_probe_report(target_qualified=False),
                output_dir=Path("outputs/temperature_guard_consolidation"),
            )

    def test_rejects_remaining_dead_air(self) -> None:
        with self.assertRaises(
            StageBMidiToSoloControlledScaleCheckpointTemperatureGuardRepairConsolidationError
        ):
            build_consolidation_report(
                repair_probe_report(strict_count=8, dead_air_count=1),
                output_dir=Path("outputs/temperature_guard_consolidation"),
            )

    def test_rejects_remaining_collapse(self) -> None:
        with self.assertRaises(
            StageBMidiToSoloControlledScaleCheckpointTemperatureGuardRepairConsolidationError
        ):
            build_consolidation_report(
                repair_probe_report(collapse_count=1),
                output_dir=Path("outputs/temperature_guard_consolidation"),
            )

    def test_rejects_quality_claim(self) -> None:
        with self.assertRaises(
            StageBMidiToSoloControlledScaleCheckpointTemperatureGuardRepairConsolidationError
        ):
            build_consolidation_report(
                repair_probe_report(quality_claim=True),
                output_dir=Path("outputs/temperature_guard_consolidation"),
            )


if __name__ == "__main__":
    unittest.main()
