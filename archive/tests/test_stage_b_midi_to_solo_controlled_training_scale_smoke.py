from __future__ import annotations

import unittest
from pathlib import Path

from scripts.decide_stage_b_midi_to_solo_training_scale_expansion import (
    BOUNDARY as DECISION_BOUNDARY,
    NEXT_BOUNDARY as DECISION_NEXT_BOUNDARY,
)
from scripts.run_stage_b_generic_base_training_scale_smoke import (
    BOUNDARY as TRAINING_SMOKE_BOUNDARY,
)
from scripts.summarize_stage_b_midi_to_solo_controlled_training_scale_smoke import (
    BOUNDARY,
    NEXT_BOUNDARY,
    StageBMidiToSoloControlledTrainingScaleSmokeSummaryError,
    build_controlled_training_scale_smoke_summary,
    validate_controlled_training_scale_smoke_summary,
)


def decision_report(*, spend_required: bool = False) -> dict:
    return {
        "boundary": DECISION_BOUNDARY,
        "selected_scale_plan": {
            "selected_train_records": 512,
            "selected_val_records": 128,
            "max_sequence": 160,
            "epochs": 1,
            "batch_size": 16,
            "lr": 8e-4,
            "seed": 43,
        },
        "readiness": {
            "controlled_training_scale_smoke_ready": True,
            "cloud_or_gpu_spend_required": spend_required,
        },
        "decision": {
            "next_boundary": DECISION_NEXT_BOUNDARY,
            "critical_user_input_required": False,
        },
    }


def training_smoke(
    *,
    train_records: int = 512,
    checkpoint_count: int = 1,
    quality_claim: bool = False,
) -> dict:
    return {
        "boundary": TRAINING_SMOKE_BOUNDARY,
        "input": {
            "selected_train_records": train_records,
            "selected_val_records": 128,
        },
        "training_config": {
            "max_sequence": 160,
            "epochs": 1,
            "batch_size": 16,
            "lr": 8e-4,
        },
        "training": {
            "returncode": 0,
            "best_validation_loss": 5.75,
        },
        "token_stats": {
            "max_token_id": 544,
            "vocab_size": 547,
            "fits_vocab": True,
        },
        "artifacts": {
            "checkpoint_count": checkpoint_count,
            "lora_weights_exists": True,
        },
        "readiness": {
            "training_scale_smoke_passed": True,
            "broad_trained_model_quality_claimed": quality_claim,
            "brad_style_adaptation_claimed": False,
        },
    }


class StageBMidiToSoloControlledTrainingScaleSmokeSummaryTest(unittest.TestCase):
    def test_summarizes_completed_controlled_scale_smoke_without_quality_claim(self) -> None:
        report = build_controlled_training_scale_smoke_summary(
            decision_report(),
            training_smoke(),
            output_dir=Path("outputs/controlled_training_scale_smoke"),
        )
        summary = validate_controlled_training_scale_smoke_summary(
            report,
            expected_boundary=BOUNDARY,
            expected_next_boundary=NEXT_BOUNDARY,
            min_train_records=512,
            min_val_records=128,
            require_checkpoint=True,
            require_no_quality_claim=True,
        )

        self.assertEqual(summary["selected_train_records"], 512)
        self.assertEqual(summary["selected_val_records"], 128)
        self.assertEqual(summary["max_sequence"], 160)
        self.assertEqual(summary["training_returncode"], 0)
        self.assertEqual(summary["best_validation_loss"], 5.75)
        self.assertEqual(summary["checkpoint_count"], 1)
        self.assertTrue(summary["fits_vocab"])
        self.assertTrue(summary["checkpoint_generation_probe_ready"])
        self.assertFalse(summary["critical_user_input_required"])
        self.assertFalse(summary["midi_to_solo_musical_quality_claimed"])

    def test_rejects_spend_required_decision(self) -> None:
        with self.assertRaises(StageBMidiToSoloControlledTrainingScaleSmokeSummaryError):
            build_controlled_training_scale_smoke_summary(
                decision_report(spend_required=True),
                training_smoke(),
                output_dir=Path("outputs/controlled_training_scale_smoke"),
            )

    def test_rejects_training_size_mismatch(self) -> None:
        with self.assertRaises(StageBMidiToSoloControlledTrainingScaleSmokeSummaryError):
            build_controlled_training_scale_smoke_summary(
                decision_report(),
                training_smoke(train_records=128),
                output_dir=Path("outputs/controlled_training_scale_smoke"),
            )

    def test_rejects_missing_checkpoint(self) -> None:
        with self.assertRaises(StageBMidiToSoloControlledTrainingScaleSmokeSummaryError):
            build_controlled_training_scale_smoke_summary(
                decision_report(),
                training_smoke(checkpoint_count=0),
                output_dir=Path("outputs/controlled_training_scale_smoke"),
            )

    def test_rejects_quality_claim(self) -> None:
        with self.assertRaises(StageBMidiToSoloControlledTrainingScaleSmokeSummaryError):
            build_controlled_training_scale_smoke_summary(
                decision_report(),
                training_smoke(quality_claim=True),
                output_dir=Path("outputs/controlled_training_scale_smoke"),
            )


if __name__ == "__main__":
    unittest.main()
