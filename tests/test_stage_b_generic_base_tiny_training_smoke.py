from __future__ import annotations

import unittest

from scripts.run_stage_b_generic_base_tiny_training_smoke import (
    StageBGenericBaseTinyTrainingSmokeError,
    parse_best_validation_loss,
    validate_training_smoke_report,
)


def training_report(*, returncode: int = 0, fits_vocab: bool = True, broad_claim: bool = False) -> dict:
    return {
        "readiness": {
            "boundary": "stage_b_generic_base_tiny_training_smoke",
            "tiny_training_smoke_passed": returncode == 0 and fits_vocab,
            "generic_base_training_path_smoked": returncode == 0 and fits_vocab,
            "broad_training_execution_ready": False,
            "broad_trained_model_quality_claimed": broad_claim,
            "brad_style_adaptation_claimed": False,
        },
        "decision": {
            "next_boundary": "stage_b_generic_tiny_checkpoint_generation_probe",
            "critical_user_input_required": False,
        },
        "training": {
            "returncode": returncode,
            "best_validation_loss": 1.234,
        },
        "token_stats": {
            "fits_vocab": fits_vocab,
        },
        "next_recommended_issue": "Stage B generic tiny checkpoint generation probe",
    }


class StageBGenericBaseTinyTrainingSmokeTest(unittest.TestCase):
    def test_parses_best_validation_loss(self) -> None:
        self.assertEqual(parse_best_validation_loss("Best validation loss: 1.2345"), 1.2345)

    def test_accepts_passed_training_smoke_without_quality_claim(self) -> None:
        summary = validate_training_smoke_report(
            training_report(),
            expected_boundary="stage_b_generic_base_tiny_training_smoke",
            expected_next_boundary="stage_b_generic_tiny_checkpoint_generation_probe",
            require_training_smoke_passed=True,
            require_no_broad_quality_claim=True,
            require_no_brad_style_claim=True,
        )

        self.assertTrue(summary["tiny_training_smoke_passed"])
        self.assertTrue(summary["generic_base_training_path_smoked"])
        self.assertEqual(summary["best_validation_loss"], 1.234)
        self.assertFalse(summary["broad_training_execution_ready"])
        self.assertFalse(summary["broad_trained_model_quality_claimed"])

    def test_rejects_training_failure(self) -> None:
        with self.assertRaises(StageBGenericBaseTinyTrainingSmokeError):
            validate_training_smoke_report(
                training_report(returncode=1),
                expected_boundary="stage_b_generic_base_tiny_training_smoke",
                expected_next_boundary="stage_b_generic_tiny_checkpoint_generation_probe",
                require_training_smoke_passed=True,
                require_no_broad_quality_claim=True,
                require_no_brad_style_claim=True,
            )

    def test_rejects_broad_quality_claim(self) -> None:
        with self.assertRaises(StageBGenericBaseTinyTrainingSmokeError):
            validate_training_smoke_report(
                training_report(broad_claim=True),
                expected_boundary="stage_b_generic_base_tiny_training_smoke",
                expected_next_boundary="stage_b_generic_tiny_checkpoint_generation_probe",
                require_training_smoke_passed=True,
                require_no_broad_quality_claim=True,
                require_no_brad_style_claim=True,
            )


if __name__ == "__main__":
    unittest.main()
