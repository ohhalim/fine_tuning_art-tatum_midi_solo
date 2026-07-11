from __future__ import annotations

import unittest

from scripts.run_stage_b_generic_full_manifest_window_preparation import (
    BOUNDARY,
    NEXT_BOUNDARY,
    StageBGenericFullManifestWindowPreparationError,
    validate_full_window_report,
    validate_training_data_plan,
)


def training_data_plan(*, quality_claimed: bool = False) -> dict:
    return {
        "claim_boundary": {
            "boundary": "stage_b_generic_model_core_training_data_plan",
            "training_data_plan_ready": True,
            "full_training_executed": False,
            "broad_trained_model_quality_claimed": quality_claimed,
        },
        "decision": {
            "next_boundary": BOUNDARY,
        },
        "training_data_plan": {
            "generic_train_file_count": 2433,
            "generic_val_file_count": 270,
            "window_parameters": {
                "window_bars": 2,
                "window_stride_bars": 2,
                "min_window_target_notes": 4,
            },
        },
    }


def full_window_report(
    *,
    ready: bool = True,
    train_records: int = 100,
    val_records: int = 10,
    fits_vocab: bool = True,
    quality_claimed: bool = False,
) -> dict:
    return {
        "readiness": {
            "boundary": BOUNDARY,
            "full_manifest_window_preparation_ready": ready,
            "generic_base_training_scale_smoke_ready": ready,
            "full_training_executed": False,
            "broad_trained_model_quality_claimed": quality_claimed,
            "brad_style_adaptation_claimed": False,
            "production_ready_improviser_claimed": False,
        },
        "decision": {
            "next_boundary": NEXT_BOUNDARY,
            "auto_progress_allowed": True,
            "critical_user_input_required": False,
        },
        "input": {
            "train_file_count": 2433,
            "val_file_count": 270,
        },
        "token_stats": {
            "tokenized_train_files": train_records,
            "tokenized_val_files": val_records,
            "max_token_id": 544,
            "vocab_size": 547,
            "fits_vocab": fits_vocab,
        },
        "next_recommended_issue": "Stage B generic base training scale smoke",
    }


class StageBGenericFullManifestWindowPreparationTest(unittest.TestCase):
    def test_validates_training_data_plan(self) -> None:
        plan = validate_training_data_plan(training_data_plan())

        self.assertEqual(plan["generic_train_file_count"], 2433)
        self.assertEqual(plan["generic_val_file_count"], 270)
        self.assertEqual(plan["window_bars"], 2)

    def test_rejects_quality_claim_in_plan(self) -> None:
        with self.assertRaises(StageBGenericFullManifestWindowPreparationError):
            validate_training_data_plan(training_data_plan(quality_claimed=True))

    def test_validates_ready_full_window_report(self) -> None:
        summary = validate_full_window_report(
            full_window_report(),
            expected_boundary=BOUNDARY,
            expected_next_boundary=NEXT_BOUNDARY,
            require_ready=True,
            require_no_training_claim=True,
            require_no_quality_claim=True,
            min_tokenized_train_files=1,
            min_tokenized_val_files=1,
        )

        self.assertTrue(summary["full_manifest_window_preparation_ready"])
        self.assertTrue(summary["generic_base_training_scale_smoke_ready"])
        self.assertFalse(summary["full_training_executed"])
        self.assertFalse(summary["broad_trained_model_quality_claimed"])
        self.assertEqual(summary["next_boundary"], NEXT_BOUNDARY)

    def test_rejects_vocab_overflow(self) -> None:
        with self.assertRaises(StageBGenericFullManifestWindowPreparationError):
            validate_full_window_report(
                full_window_report(fits_vocab=False),
                expected_boundary=BOUNDARY,
                expected_next_boundary=NEXT_BOUNDARY,
                require_ready=True,
                require_no_training_claim=True,
                require_no_quality_claim=True,
                min_tokenized_train_files=1,
                min_tokenized_val_files=1,
            )

    def test_rejects_missing_val_records(self) -> None:
        with self.assertRaises(StageBGenericFullManifestWindowPreparationError):
            validate_full_window_report(
                full_window_report(val_records=0),
                expected_boundary=BOUNDARY,
                expected_next_boundary=NEXT_BOUNDARY,
                require_ready=True,
                require_no_training_claim=True,
                require_no_quality_claim=True,
                min_tokenized_train_files=1,
                min_tokenized_val_files=1,
            )


if __name__ == "__main__":
    unittest.main()
