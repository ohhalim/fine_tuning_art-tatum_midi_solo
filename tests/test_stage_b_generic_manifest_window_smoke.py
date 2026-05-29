from __future__ import annotations

import unittest

from scripts.run_stage_b_generic_manifest_window_smoke import (
    StageBGenericManifestWindowSmokeError,
    validate_smoke_report,
)


def smoke_report(*, fits_vocab: bool = True, broad_claim: bool = False) -> dict:
    return {
        "readiness": {
            "boundary": "stage_b_generic_stage_b_window_prepare_smoke",
            "stage_b_window_prepare_smoke_ready": True,
            "generic_base_training_execution_ready": False,
            "broad_trained_model_quality_claimed": broad_claim,
            "brad_style_adaptation_claimed": False,
        },
        "decision": {
            "next_boundary": "stage_b_generic_base_tiny_training_smoke",
            "critical_user_input_required": False,
        },
        "token_stats": {
            "tokenized_train_files": 12,
            "tokenized_val_files": 4,
            "fits_vocab": fits_vocab,
        },
        "next_recommended_issue": "Stage B generic base tiny training smoke",
    }


class StageBGenericManifestWindowSmokeTest(unittest.TestCase):
    def test_accepts_ready_window_smoke_without_quality_claim(self) -> None:
        summary = validate_smoke_report(
            smoke_report(),
            expected_boundary="stage_b_generic_stage_b_window_prepare_smoke",
            expected_next_boundary="stage_b_generic_base_tiny_training_smoke",
            require_smoke_ready=True,
            require_no_broad_quality_claim=True,
            require_no_brad_style_claim=True,
        )

        self.assertTrue(summary["stage_b_window_prepare_smoke_ready"])
        self.assertEqual(summary["tokenized_train_files"], 12)
        self.assertEqual(summary["tokenized_val_files"], 4)
        self.assertTrue(summary["fits_vocab"])
        self.assertFalse(summary["generic_base_training_execution_ready"])
        self.assertFalse(summary["broad_trained_model_quality_claimed"])

    def test_rejects_broad_quality_claim(self) -> None:
        with self.assertRaises(StageBGenericManifestWindowSmokeError):
            validate_smoke_report(
                smoke_report(broad_claim=True),
                expected_boundary="stage_b_generic_stage_b_window_prepare_smoke",
                expected_next_boundary="stage_b_generic_base_tiny_training_smoke",
                require_smoke_ready=True,
                require_no_broad_quality_claim=True,
                require_no_brad_style_claim=True,
            )

    def test_rejects_vocab_overflow(self) -> None:
        with self.assertRaises(StageBGenericManifestWindowSmokeError):
            validate_smoke_report(
                smoke_report(fits_vocab=False),
                expected_boundary="stage_b_generic_stage_b_window_prepare_smoke",
                expected_next_boundary="stage_b_generic_base_tiny_training_smoke",
                require_smoke_ready=True,
                require_no_broad_quality_claim=True,
                require_no_brad_style_claim=True,
            )


if __name__ == "__main__":
    unittest.main()
