from __future__ import annotations

import unittest
from pathlib import Path

from scripts.run_stage_b_generic_base_scale_checkpoint_generation_probe import (
    BOUNDARY as GENERIC_GENERATION_BOUNDARY,
)
from scripts.summarize_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_generation_probe import (
    BOUNDARY,
    FAIL_NEXT_BOUNDARY,
    PASS_NEXT_BOUNDARY,
    StageBMidiToSoloControlledScaleCheckpointTrainingScaleGenerationProbeError,
    build_generation_probe_summary,
    validate_generation_probe_summary,
)
from scripts.summarize_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_smoke import (
    BOUNDARY as SELECTED_TRAINING_BOUNDARY,
)


def selected_training(*, quality_claim: bool = False) -> dict:
    return {
        "boundary": SELECTED_TRAINING_BOUNDARY,
        "training_result": {
            "selected_train_records": 2048,
            "selected_val_records": 512,
            "max_sequence": 160,
            "best_validation_loss": 3.0396,
            "checkpoint_count": 1,
        },
        "readiness": {
            "checkpoint_generation_probe_ready": True,
            "midi_to_solo_musical_quality_claimed": quality_claim,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
        },
    }


def generation_probe(
    *,
    sample_count: int = 3,
    strict_count: int = 0,
    returncode: int = 0,
    broad_claim: bool = False,
) -> dict:
    return {
        "readiness": {
            "boundary": GENERIC_GENERATION_BOUNDARY,
            "generation_path_executable": returncode == 0 and sample_count > 0,
            "broad_trained_model_quality_claimed": broad_claim,
            "brad_style_adaptation_claimed": False,
            "human_audio_preference_claimed": False,
            "production_ready_improviser_claimed": False,
        },
        "decision": {
            "next_boundary": "stage_b_generic_base_scale_checkpoint_grammar_representation_decision",
        },
        "generation_command": {"returncode": returncode},
        "generation_summary": {
            "sample_count": sample_count,
            "valid_sample_count": strict_count,
            "strict_valid_sample_count": strict_count,
            "grammar_gate_sample_count": sample_count,
            "passed_generation_gate": strict_count > 0,
            "passed_grammar_gate": sample_count > 0,
            "passed_strict_review_gate": strict_count > 0,
            "collapse_warning_sample_count": 0,
            "collapse_warning_sample_rate": 0.0,
            "avg_onset_coverage_ratio": 0.5,
            "avg_sustained_coverage_ratio": 0.7,
            "max_longest_sustained_empty_run_steps": 2,
            "avg_postprocess_removal_ratio": 0.2,
            "max_postprocess_removal_ratio": 0.3,
            "failure_reasons": {},
            "strict_failure_reasons": {},
            "diagnostic_failure_reasons": {},
        },
    }


class StageBMidiToSoloControlledScaleCheckpointTrainingScaleGenerationProbeTest(
    unittest.TestCase
):
    def test_routes_failed_strict_gate_to_repair_decision(self) -> None:
        report = build_generation_probe_summary(
            selected_training(),
            generation_probe(strict_count=0),
            output_dir=Path("outputs/selected_generation_probe"),
        )
        summary = validate_generation_probe_summary(
            report,
            expected_boundary=BOUNDARY,
            min_sample_count=3,
            require_generation_executable=True,
            require_no_quality_claim=True,
        )

        self.assertEqual(summary["next_boundary"], FAIL_NEXT_BOUNDARY)
        self.assertEqual(summary["selected_train_records"], 2048)
        self.assertEqual(summary["best_validation_loss"], 3.0396)
        self.assertEqual(summary["sample_count"], 3)
        self.assertEqual(summary["strict_valid_sample_count"], 0)
        self.assertFalse(summary["raw_generation_quality_ready"])

    def test_routes_strict_gate_pass_to_repeatability(self) -> None:
        report = build_generation_probe_summary(
            selected_training(),
            generation_probe(strict_count=2),
            output_dir=Path("outputs/selected_generation_probe"),
        )

        self.assertEqual(report["decision"]["next_boundary"], PASS_NEXT_BOUNDARY)
        self.assertTrue(report["readiness"]["raw_generation_quality_ready"])

    def test_rejects_generation_failure(self) -> None:
        with self.assertRaises(
            StageBMidiToSoloControlledScaleCheckpointTrainingScaleGenerationProbeError
        ):
            build_generation_probe_summary(
                selected_training(),
                generation_probe(returncode=1),
                output_dir=Path("outputs/selected_generation_probe"),
            )

    def test_rejects_zero_samples(self) -> None:
        with self.assertRaises(
            StageBMidiToSoloControlledScaleCheckpointTrainingScaleGenerationProbeError
        ):
            build_generation_probe_summary(
                selected_training(),
                generation_probe(sample_count=0),
                output_dir=Path("outputs/selected_generation_probe"),
            )

    def test_rejects_quality_claim(self) -> None:
        with self.assertRaises(
            StageBMidiToSoloControlledScaleCheckpointTrainingScaleGenerationProbeError
        ):
            build_generation_probe_summary(
                selected_training(),
                generation_probe(broad_claim=True),
                output_dir=Path("outputs/selected_generation_probe"),
            )


if __name__ == "__main__":
    unittest.main()
