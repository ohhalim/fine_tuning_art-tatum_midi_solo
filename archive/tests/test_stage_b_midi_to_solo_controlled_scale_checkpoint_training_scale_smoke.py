from __future__ import annotations

import unittest
from pathlib import Path

from scripts.decide_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_expansion import (
    BOUNDARY as DECISION_BOUNDARY,
    NEXT_BOUNDARY as DECISION_NEXT_BOUNDARY,
)
from scripts.run_stage_b_generic_base_training_scale_smoke import (
    BOUNDARY as TRAINING_SMOKE_BOUNDARY,
)
from scripts.summarize_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_smoke import (
    BOUNDARY,
    NEXT_BOUNDARY,
    StageBMidiToSoloControlledScaleCheckpointTrainingScaleSmokeError,
    build_selected_scale_smoke_summary,
    validate_selected_scale_smoke_summary,
)


def decision_report(*, quality_claim: bool = False, full_training: bool = False) -> dict:
    return {
        "boundary": DECISION_BOUNDARY,
        "selected_scale_plan": {
            "selected_train_records": 2048,
            "selected_val_records": 512,
            "max_sequence": 160,
            "epochs": 1,
            "batch_size": 16,
            "lr": 0.0008,
            "seed": 47,
        },
        "readiness": {
            "controlled_training_scale_smoke_ready": True,
            "cloud_or_gpu_spend_required": False,
            "full_training_selected": full_training,
            "midi_to_solo_musical_quality_claimed": quality_claim,
            "human_audio_preference_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
        },
        "decision": {
            "next_boundary": DECISION_NEXT_BOUNDARY,
            "critical_user_input_required": False,
        },
    }


def training_smoke(
    *,
    train_records: int = 2048,
    val_records: int = 512,
    returncode: int = 0,
    checkpoint_count: int = 1,
    quality_claim: bool = False,
) -> dict:
    return {
        "boundary": TRAINING_SMOKE_BOUNDARY,
        "input": {
            "selected_train_records": train_records,
            "selected_val_records": val_records,
        },
        "training_config": {
            "max_sequence": 160,
            "epochs": 1,
            "batch_size": 16,
            "lr": 0.0008,
        },
        "training": {
            "returncode": returncode,
            "best_validation_loss": 4.8123 if returncode == 0 else None,
        },
        "token_stats": {
            "max_token_id": 544,
            "vocab_size": 547,
            "fits_vocab": True,
        },
        "artifacts": {
            "checkpoint_count": checkpoint_count,
            "checkpoint_files": ["outputs/checkpoints/checkpoint_epoch1.pt"],
            "lora_weights_exists": checkpoint_count > 0,
        },
        "readiness": {
            "training_scale_smoke_passed": returncode == 0 and checkpoint_count > 0,
            "full_generic_training_executed": False,
            "broad_trained_model_quality_claimed": quality_claim,
            "brad_style_adaptation_claimed": False,
        },
    }


class StageBMidiToSoloControlledScaleCheckpointTrainingScaleSmokeTest(unittest.TestCase):
    def test_summarizes_selected_scale_smoke(self) -> None:
        report = build_selected_scale_smoke_summary(
            decision_report(),
            training_smoke(),
            output_dir=Path("outputs/selected_scale_smoke"),
        )
        summary = validate_selected_scale_smoke_summary(
            report,
            expected_boundary=BOUNDARY,
            expected_next_boundary=NEXT_BOUNDARY,
            min_train_records=2048,
            min_val_records=512,
            require_checkpoint=True,
            require_no_quality_claim=True,
        )

        self.assertEqual(summary["selected_train_records"], 2048)
        self.assertEqual(summary["selected_val_records"], 512)
        self.assertEqual(summary["max_sequence"], 160)
        self.assertEqual(summary["training_returncode"], 0)
        self.assertEqual(summary["best_validation_loss"], 4.8123)
        self.assertEqual(summary["checkpoint_count"], 1)
        self.assertTrue(summary["fits_vocab"])
        self.assertTrue(summary["checkpoint_generation_probe_ready"])
        self.assertFalse(summary["critical_user_input_required"])
        self.assertFalse(summary["midi_to_solo_musical_quality_claimed"])

    def test_rejects_mismatched_train_records(self) -> None:
        with self.assertRaises(StageBMidiToSoloControlledScaleCheckpointTrainingScaleSmokeError):
            build_selected_scale_smoke_summary(
                decision_report(),
                training_smoke(train_records=1024),
                output_dir=Path("outputs/selected_scale_smoke"),
            )

    def test_rejects_failed_training(self) -> None:
        with self.assertRaises(StageBMidiToSoloControlledScaleCheckpointTrainingScaleSmokeError):
            build_selected_scale_smoke_summary(
                decision_report(),
                training_smoke(returncode=1),
                output_dir=Path("outputs/selected_scale_smoke"),
            )

    def test_rejects_missing_checkpoint(self) -> None:
        with self.assertRaises(StageBMidiToSoloControlledScaleCheckpointTrainingScaleSmokeError):
            build_selected_scale_smoke_summary(
                decision_report(),
                training_smoke(checkpoint_count=0),
                output_dir=Path("outputs/selected_scale_smoke"),
            )

    def test_rejects_quality_claims(self) -> None:
        with self.assertRaises(StageBMidiToSoloControlledScaleCheckpointTrainingScaleSmokeError):
            build_selected_scale_smoke_summary(
                decision_report(full_training=True),
                training_smoke(),
                output_dir=Path("outputs/selected_scale_smoke"),
            )

        with self.assertRaises(StageBMidiToSoloControlledScaleCheckpointTrainingScaleSmokeError):
            build_selected_scale_smoke_summary(
                decision_report(),
                training_smoke(quality_claim=True),
                output_dir=Path("outputs/selected_scale_smoke"),
            )


if __name__ == "__main__":
    unittest.main()
