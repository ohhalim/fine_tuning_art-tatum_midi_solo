from __future__ import annotations

import unittest

from scripts.run_stage_b_generic_base_training_scale_smoke import (
    BOUNDARY,
    FULL_WINDOW_BOUNDARY,
    NEXT_BOUNDARY,
    StageBGenericBaseTrainingScaleSmokeError,
    validate_training_scale_smoke_report,
)


def scale_report(
    *,
    passed: bool = True,
    returncode: int = 0,
    train_records: int = 128,
    val_records: int = 32,
    fits_vocab: bool = True,
    checkpoint_count: int = 1,
    broad_claim: bool = False,
    brad_claim: bool = False,
) -> dict:
    return {
        "source_window_summary": {
            "source_tokenized_train_files": 154136,
            "source_tokenized_val_files": 21845,
        },
        "readiness": {
            "boundary": BOUNDARY,
            "training_scale_smoke_passed": passed,
            "generic_base_training_scale_smoked": passed,
            "generic_base_scale_checkpoint_generation_probe_ready": passed,
            "full_generic_training_executed": False,
            "broad_trained_model_quality_claimed": broad_claim,
            "brad_style_adaptation_claimed": brad_claim,
        },
        "decision": {
            "next_boundary": NEXT_BOUNDARY,
            "critical_user_input_required": False,
        },
        "input": {
            "selected_train_records": train_records,
            "selected_val_records": val_records,
        },
        "token_stats": {
            "max_token_id": 544,
            "vocab_size": 547,
            "fits_vocab": fits_vocab,
        },
        "training": {
            "returncode": returncode,
            "best_validation_loss": 5.678,
        },
        "artifacts": {
            "checkpoint_count": checkpoint_count,
            "lora_weights_exists": True,
        },
        "next_recommended_issue": "Stage B generic base scale checkpoint generation probe",
    }


class StageBGenericBaseTrainingScaleSmokeTest(unittest.TestCase):
    def test_accepts_passed_scale_smoke_without_quality_claim(self) -> None:
        summary = validate_training_scale_smoke_report(
            scale_report(),
            expected_boundary=BOUNDARY,
            expected_next_boundary=NEXT_BOUNDARY,
            require_training_scale_smoke_passed=True,
            require_no_broad_quality_claim=True,
            require_no_brad_style_claim=True,
            min_train_records=64,
            min_val_records=16,
        )

        self.assertTrue(summary["training_scale_smoke_passed"])
        self.assertTrue(summary["generic_base_scale_checkpoint_generation_probe_ready"])
        self.assertEqual(summary["selected_train_records"], 128)
        self.assertEqual(summary["selected_val_records"], 32)
        self.assertEqual(summary["source_tokenized_train_files"], 154136)
        self.assertEqual(summary["source_tokenized_val_files"], 21845)
        self.assertEqual(summary["best_validation_loss"], 5.678)
        self.assertFalse(summary["broad_trained_model_quality_claimed"])
        self.assertFalse(summary["brad_style_adaptation_claimed"])

    def test_rejects_training_failure(self) -> None:
        with self.assertRaises(StageBGenericBaseTrainingScaleSmokeError):
            validate_training_scale_smoke_report(
                scale_report(passed=False, returncode=1),
                expected_boundary=BOUNDARY,
                expected_next_boundary=NEXT_BOUNDARY,
                require_training_scale_smoke_passed=True,
                require_no_broad_quality_claim=True,
                require_no_brad_style_claim=True,
                min_train_records=64,
                min_val_records=16,
            )

    def test_rejects_tiny_sized_subset(self) -> None:
        with self.assertRaises(StageBGenericBaseTrainingScaleSmokeError):
            validate_training_scale_smoke_report(
                scale_report(train_records=32, val_records=8),
                expected_boundary=BOUNDARY,
                expected_next_boundary=NEXT_BOUNDARY,
                require_training_scale_smoke_passed=True,
                require_no_broad_quality_claim=True,
                require_no_brad_style_claim=True,
                min_train_records=64,
                min_val_records=16,
            )

    def test_rejects_missing_checkpoint(self) -> None:
        with self.assertRaises(StageBGenericBaseTrainingScaleSmokeError):
            validate_training_scale_smoke_report(
                scale_report(checkpoint_count=0),
                expected_boundary=BOUNDARY,
                expected_next_boundary=NEXT_BOUNDARY,
                require_training_scale_smoke_passed=True,
                require_no_broad_quality_claim=True,
                require_no_brad_style_claim=True,
                min_train_records=64,
                min_val_records=16,
            )

    def test_rejects_quality_claims(self) -> None:
        with self.assertRaises(StageBGenericBaseTrainingScaleSmokeError):
            validate_training_scale_smoke_report(
                scale_report(broad_claim=True),
                expected_boundary=BOUNDARY,
                expected_next_boundary=NEXT_BOUNDARY,
                require_training_scale_smoke_passed=True,
                require_no_broad_quality_claim=True,
                require_no_brad_style_claim=True,
                min_train_records=64,
                min_val_records=16,
            )

    def test_boundary_constants_are_stable(self) -> None:
        self.assertEqual(FULL_WINDOW_BOUNDARY, "stage_b_generic_full_manifest_window_preparation")
        self.assertEqual(BOUNDARY, "stage_b_generic_base_training_scale_smoke")
        self.assertEqual(NEXT_BOUNDARY, "stage_b_generic_base_scale_checkpoint_generation_probe")


if __name__ == "__main__":
    unittest.main()
