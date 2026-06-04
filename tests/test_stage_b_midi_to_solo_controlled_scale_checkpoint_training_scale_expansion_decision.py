from __future__ import annotations

import unittest
from pathlib import Path

from scripts.check_stage_b_midi_to_solo_training_resource_probe import (
    BOUNDARY as TRAINING_RESOURCE_BOUNDARY,
)
from scripts.decide_stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repeatability_temperature_guard_objective_next import (
    BOUNDARY as OBJECTIVE_BOUNDARY,
    FINAL_BOUNDARY as OBJECTIVE_FINAL_BOUNDARY,
)
from scripts.decide_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_expansion import (
    BOUNDARY,
    NEXT_BOUNDARY,
    StageBMidiToSoloControlledScaleCheckpointTrainingScaleExpansionError,
    build_training_scale_expansion_decision,
    validate_training_scale_expansion_decision,
)
from scripts.summarize_stage_b_midi_to_solo_controlled_training_scale_smoke import (
    BOUNDARY as CURRENT_TRAINING_BOUNDARY,
)


def training_resource(*, train_records: int = 154136, quality_claim: bool = False) -> dict:
    return {
        "boundary": TRAINING_RESOURCE_BOUNDARY,
        "full_window_resource": {
            "tokenized_train_files": train_records,
            "tokenized_val_files": 21845,
            "max_token_id": 544,
            "vocab_size": 547,
            "fits_vocab": True,
        },
        "readiness": {
            "midi_to_solo_training_resource_ready": True,
            "conditioned_generation_probe_ready": True,
            "midi_to_solo_mvp_claimed": False,
            "broad_training_executed": False,
            "broad_trained_model_quality_claimed": quality_claim,
            "brad_style_adaptation_claimed": False,
            "musical_quality_claimed": False,
        },
    }


def current_training(*, train_records: int = 512, quality_claim: bool = False) -> dict:
    return {
        "boundary": CURRENT_TRAINING_BOUNDARY,
        "training_result": {
            "selected_train_records": train_records,
            "selected_val_records": 128,
            "max_sequence": 160,
            "epochs": 1,
            "batch_size": 16,
            "lr": 0.0008,
            "training_returncode": 0,
            "best_validation_loss": 5.1061,
            "fits_vocab": True,
            "checkpoint_count": 1,
        },
        "readiness": {
            "controlled_training_scale_smoke_completed": True,
            "checkpoint_generation_probe_ready": True,
            "model_direct_generation_quality_claimed": False,
            "midi_to_solo_musical_quality_claimed": quality_claim,
            "human_audio_preference_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
        },
        "decision": {
            "next_boundary": "stage_b_midi_to_solo_controlled_scale_checkpoint_generation_probe",
            "critical_user_input_required": False,
        },
    }


def objective_path(*, quality_claim: bool = False, review_present: bool = False) -> dict:
    return {
        "boundary": OBJECTIVE_BOUNDARY,
        "decision": {
            "final_boundary": OBJECTIVE_FINAL_BOUNDARY,
            "next_boundary": BOUNDARY,
            "critical_user_input_required": False,
        },
        "temperature_guard_summary": {
            "sample_count": 9,
            "seed_count": 3,
            "strict_valid_sample_count": 9,
            "grammar_gate_sample_count": 9,
            "dead_air_failure_count": 0,
            "collapse_warning_sample_count": 0,
        },
        "review_boundary_summary": {
            "rendered_audio_file_count": 3,
            "validated_review_input_present": review_present,
            "pending_status_field_count": 4,
            "pending_candidate_decision_count": 3,
            "pending_candidate_field_count": 9,
        },
        "readiness": {
            "objective_temperature_guard_path_supported": True,
            "human_audio_preference_claimed": False,
            "midi_to_solo_musical_quality_claimed": quality_claim,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
        },
    }


class StageBMidiToSoloControlledScaleCheckpointTrainingScaleExpansionDecisionTest(
    unittest.TestCase
):
    def test_selects_next_bounded_training_scale(self) -> None:
        report = build_training_scale_expansion_decision(
            training_resource(),
            current_training(),
            objective_path(),
            output_dir=Path("outputs/controlled_training_scale_expansion"),
            target_train_records=2048,
            target_val_records=512,
        )
        summary = validate_training_scale_expansion_decision(
            report,
            expected_boundary=BOUNDARY,
            expected_next_boundary=NEXT_BOUNDARY,
            min_selected_train_records=2048,
            min_selected_val_records=512,
            require_scale_ready=True,
            require_no_quality_claim=True,
        )

        self.assertEqual(summary["selected_train_records"], 2048)
        self.assertEqual(summary["selected_val_records"], 512)
        self.assertEqual(summary["current_train_records"], 512)
        self.assertEqual(summary["current_val_records"], 128)
        self.assertEqual(summary["scale_multiplier_train"], 4.0)
        self.assertEqual(summary["scale_multiplier_val"], 4.0)
        self.assertEqual(summary["max_sequence"], 160)
        self.assertEqual(summary["current_best_validation_loss"], 5.1061)
        self.assertEqual(summary["objective_sample_count"], 9)
        self.assertEqual(summary["objective_strict_valid_sample_count"], 9)
        self.assertEqual(summary["rendered_audio_file_count"], 3)
        self.assertTrue(summary["controlled_training_scale_smoke_ready"])
        self.assertFalse(summary["cloud_or_gpu_spend_required"])
        self.assertFalse(summary["full_training_selected"])
        self.assertFalse(summary["critical_user_input_required"])

    def test_rejects_scale_not_larger_than_current(self) -> None:
        with self.assertRaises(
            StageBMidiToSoloControlledScaleCheckpointTrainingScaleExpansionError
        ):
            build_training_scale_expansion_decision(
                training_resource(),
                current_training(),
                objective_path(),
                output_dir=Path("outputs/controlled_training_scale_expansion"),
                target_train_records=512,
                target_val_records=128,
            )

    def test_rejects_small_training_resource(self) -> None:
        with self.assertRaises(
            StageBMidiToSoloControlledScaleCheckpointTrainingScaleExpansionError
        ):
            build_training_scale_expansion_decision(
                training_resource(train_records=1000),
                current_training(),
                objective_path(),
                output_dir=Path("outputs/controlled_training_scale_expansion"),
                target_train_records=2048,
                target_val_records=512,
            )

    def test_rejects_validated_review_input(self) -> None:
        with self.assertRaises(
            StageBMidiToSoloControlledScaleCheckpointTrainingScaleExpansionError
        ):
            build_training_scale_expansion_decision(
                training_resource(),
                current_training(),
                objective_path(review_present=True),
                output_dir=Path("outputs/controlled_training_scale_expansion"),
                target_train_records=2048,
                target_val_records=512,
            )

    def test_rejects_quality_claims(self) -> None:
        inputs = [
            (training_resource(quality_claim=True), current_training(), objective_path()),
            (training_resource(), current_training(quality_claim=True), objective_path()),
            (training_resource(), current_training(), objective_path(quality_claim=True)),
        ]
        for resource, training, objective in inputs:
            with self.assertRaises(
                StageBMidiToSoloControlledScaleCheckpointTrainingScaleExpansionError
            ):
                build_training_scale_expansion_decision(
                    resource,
                    training,
                    objective,
                    output_dir=Path("outputs/controlled_training_scale_expansion"),
                    target_train_records=2048,
                    target_val_records=512,
                )


if __name__ == "__main__":
    unittest.main()
