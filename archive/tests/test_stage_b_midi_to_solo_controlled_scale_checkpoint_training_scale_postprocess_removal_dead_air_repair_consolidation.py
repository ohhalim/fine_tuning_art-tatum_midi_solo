from __future__ import annotations

import unittest
from pathlib import Path

from scripts.consolidate_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_postprocess_removal_dead_air_repair import (
    BOUNDARY,
    NEXT_BOUNDARY,
    SOURCE_BOUNDARY,
    StageBMidiToSoloControlledScaleCheckpointTrainingScalePostprocessRemovalDeadAirRepairConsolidationError,
    build_consolidation_report,
    validate_consolidation_report,
)


def repair_probe_report(
    *,
    target_qualified: bool = True,
    strict_count: int = 9,
    dead_air_count: int = 0,
    collapse_count: int = 0,
    avg_removal: float = 0.2176,
    quality_claim: bool = False,
) -> dict:
    return {
        "schema_version": "stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_postprocess_removal_dead_air_repair_probe_v1",
        "boundary": SOURCE_BOUNDARY,
        "source_summary": {
            "source_strict_valid_sample_count": 8,
            "source_dead_air_failure_count": 1,
            "source_avg_postprocess_removal_ratio": 0.3611,
        },
        "input": {
            "temperature": 0.75,
            "top_k": 4,
            "avoid_reused_positions": True,
            "target_avg_postprocess_removal_ratio": 0.3,
        },
        "seed_rows": [
            {"seed": seed, "generation_report_path": f"outputs/seed_{seed}/report.json"}
            for seed in [47, 52, 60]
        ],
        "aggregate": {
            "seed_count": 3,
            "seeds": [47, 52, 60],
            "sample_count": 9,
            "valid_sample_count": strict_count,
            "strict_valid_sample_count": strict_count,
            "grammar_gate_sample_count": 9,
            "note_count_failure_count": 0,
            "grammar_failure_count": 0,
            "dead_air_failure_count": dead_air_count,
            "collapse_warning_sample_count": collapse_count,
            "diagnostic_failure_reasons": (
                {} if dead_air_count == 0 else {"dead-air ratio too high: 0.846 >= 0.800": dead_air_count}
            ),
            "avg_postprocess_removal_ratio": avg_removal,
            "max_postprocess_removal_ratio": 0.2917,
            "avg_onset_coverage_ratio": 0.7326,
            "avg_sustained_coverage_ratio": 0.7708,
            "all_seed_gate_passed": True,
            "all_samples_strict_valid": strict_count == 9,
        },
        "comparison": {
            "strict_valid_sample_delta": strict_count - 8,
            "dead_air_failure_delta": dead_air_count - 1,
            "postprocess_removal_delta": avg_removal - 0.3611,
            "onset_coverage_delta": 0.1563,
            "sustained_coverage_delta": 0.0486,
        },
        "readiness": {
            "boundary": SOURCE_BOUNDARY,
            "postprocess_removal_dead_air_repair_probe_completed": True,
            "postprocess_removal_dead_air_repair_target_qualified": target_qualified,
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


class StageBMidiToSoloControlledScaleCheckpointTrainingScalePostprocessRemovalDeadAirRepairConsolidationTest(
    unittest.TestCase
):
    def test_consolidates_objective_support(self) -> None:
        report = build_consolidation_report(
            repair_probe_report(),
            output_dir=Path("outputs/postprocess_consolidation"),
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
        self.assertEqual(summary["strict_valid_sample_delta"], 1)
        self.assertEqual(summary["dead_air_failure_delta"], -1)
        self.assertLess(summary["avg_postprocess_removal_ratio"], 0.3)
        self.assertTrue(summary["avoid_reused_positions"])
        self.assertTrue(summary["objective_midi_support"])
        self.assertTrue(summary["audio_review_package_required"])
        self.assertFalse(summary["additional_repair_required"])
        self.assertFalse(summary["human_audio_preference_claimed"])
        self.assertFalse(summary["midi_to_solo_musical_quality_claimed"])

    def test_rejects_unqualified_probe(self) -> None:
        with self.assertRaises(
            StageBMidiToSoloControlledScaleCheckpointTrainingScalePostprocessRemovalDeadAirRepairConsolidationError
        ):
            build_consolidation_report(
                repair_probe_report(target_qualified=False),
                output_dir=Path("outputs/postprocess_consolidation"),
            )

    def test_rejects_remaining_dead_air(self) -> None:
        with self.assertRaises(
            StageBMidiToSoloControlledScaleCheckpointTrainingScalePostprocessRemovalDeadAirRepairConsolidationError
        ):
            build_consolidation_report(
                repair_probe_report(strict_count=8, dead_air_count=1),
                output_dir=Path("outputs/postprocess_consolidation"),
            )

    def test_rejects_remaining_collapse(self) -> None:
        with self.assertRaises(
            StageBMidiToSoloControlledScaleCheckpointTrainingScalePostprocessRemovalDeadAirRepairConsolidationError
        ):
            build_consolidation_report(
                repair_probe_report(collapse_count=1),
                output_dir=Path("outputs/postprocess_consolidation"),
            )

    def test_rejects_quality_claim(self) -> None:
        with self.assertRaises(
            StageBMidiToSoloControlledScaleCheckpointTrainingScalePostprocessRemovalDeadAirRepairConsolidationError
        ):
            build_consolidation_report(
                repair_probe_report(quality_claim=True),
                output_dir=Path("outputs/postprocess_consolidation"),
            )


if __name__ == "__main__":
    unittest.main()
