from __future__ import annotations

import unittest
from pathlib import Path

from scripts.plan_stage_b_generic_model_core_training_data import (
    NEXT_BOUNDARY,
    PLAN_BOUNDARY,
    StageBGenericModelCoreTrainingDataPlanError,
    build_training_data_plan,
    validate_training_data_plan,
)


def model_core_decision(*, continue_repair: bool = False) -> dict:
    return {
        "decision": {
            "current_boundary": (
                "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_"
                "sparse_phrase_model_core_review_decision"
            ),
            "decision": "stop_constraint_postprocess_repair_loop",
            "continue_constraint_postprocess_repair_loop": continue_repair,
            "tiny_checkpoint_role": "diagnostic_only",
            "model_core_transition_required": True,
            "next_boundary": PLAN_BOUNDARY,
        },
        "methodology_evidence": {
            "objective_proxy_gap_recorded": True,
            "candidate_without_objective_flag_count": 1,
        },
        "claim_boundary": {
            "musical_quality_claimed": False,
            "quality_root_cause_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
            "production_ready_improviser_claimed": False,
        },
    }


def manifest_contract(*, leak_count: int = 0) -> dict:
    return {
        "readiness": {
            "boundary": "stage_b_generic_base_manifest_contract",
            "manifest_contract_ready": True,
            "broad_training_execution_ready": False,
        },
        "split_counts": {
            "generic_jazz_train": 2433,
            "generic_jazz_val": 270,
            "brad_adaptation_train": 47,
            "brad_adaptation_val": 11,
            "brad_test_holdout": 14,
        },
        "guards": {
            "generic_brad_leak_count": leak_count,
            "brad_non_brad_leak_count": 0,
            "overlap_path_count": 0,
            "duplicate_exact_hash_group_count": 0,
            "duplicate_exact_file_count": 0,
        },
    }


def window_smoke(*, max_token_id: int = 544, vocab_size: int = 547) -> dict:
    return {
        "readiness": {
            "boundary": "stage_b_generic_stage_b_window_prepare_smoke",
            "stage_b_window_prepare_smoke_ready": True,
            "generic_base_training_execution_ready": False,
        },
        "input": {
            "selected_train_files": 6,
            "selected_val_files": 3,
            "window_bars": 2,
            "window_stride_bars": 1,
            "min_window_target_notes": 4,
        },
        "token_stats": {
            "files": 747,
            "non_empty_files": 747,
            "max_token_id": max_token_id,
            "vocab_size": vocab_size,
        },
        "dataset_summary": {
            "train_manifest": "generic_jazz_train.txt",
            "val_manifest": "generic_jazz_val.txt",
        },
    }


def tiny_training_smoke(*, returncode: int = 0) -> dict:
    return {
        "readiness": {
            "boundary": "stage_b_generic_base_tiny_training_smoke",
            "tiny_training_smoke_passed": True,
            "generic_base_training_path_smoked": True,
            "broad_training_execution_ready": False,
        },
        "input": {
            "selected_train_records": 32,
            "selected_val_records": 8,
        },
        "token_stats": {
            "files": 40,
            "non_empty_files": 40,
            "max_token_id": 544,
            "vocab_size": 547,
            "fits_vocab": True,
        },
        "training": {
            "returncode": returncode,
            "best_validation_loss": 6.1427,
        },
    }


class StageBGenericModelCoreTrainingDataPlanTest(unittest.TestCase):
    def test_builds_plan_from_ready_inputs_without_training_claim(self) -> None:
        report = build_training_data_plan(
            model_core_decision(),
            manifest_contract(),
            window_smoke(),
            tiny_training_smoke(),
            output_dir=Path("outputs/plan"),
        )
        summary = validate_training_data_plan(
            report,
            expected_boundary=PLAN_BOUNDARY,
            expected_next_boundary=NEXT_BOUNDARY,
            require_stop_repair_loop=True,
            require_no_quality_claim=True,
            min_generic_train_files=2000,
            min_generic_val_files=200,
        )

        self.assertEqual(summary["repair_loop_status"], "stopped")
        self.assertEqual(summary["tiny_checkpoint_role"], "diagnostic_only")
        self.assertEqual(summary["generic_train_file_count"], 2433)
        self.assertEqual(summary["generic_val_file_count"], 270)
        self.assertFalse(summary["full_training_executed"])
        self.assertFalse(summary["broad_trained_model_quality_claimed"])
        self.assertEqual(summary["next_boundary"], NEXT_BOUNDARY)

    def test_rejects_unstopped_repair_loop(self) -> None:
        with self.assertRaises(StageBGenericModelCoreTrainingDataPlanError):
            build_training_data_plan(
                model_core_decision(continue_repair=True),
                manifest_contract(),
                window_smoke(),
                tiny_training_smoke(),
                output_dir=Path("outputs/plan"),
            )

    def test_rejects_manifest_leak(self) -> None:
        with self.assertRaises(StageBGenericModelCoreTrainingDataPlanError):
            build_training_data_plan(
                model_core_decision(),
                manifest_contract(leak_count=1),
                window_smoke(),
                tiny_training_smoke(),
                output_dir=Path("outputs/plan"),
            )

    def test_rejects_vocab_overflow(self) -> None:
        with self.assertRaises(StageBGenericModelCoreTrainingDataPlanError):
            build_training_data_plan(
                model_core_decision(),
                manifest_contract(),
                window_smoke(max_token_id=547, vocab_size=547),
                tiny_training_smoke(),
                output_dir=Path("outputs/plan"),
            )

    def test_rejects_failed_training_smoke(self) -> None:
        with self.assertRaises(StageBGenericModelCoreTrainingDataPlanError):
            build_training_data_plan(
                model_core_decision(),
                manifest_contract(),
                window_smoke(),
                tiny_training_smoke(returncode=1),
                output_dir=Path("outputs/plan"),
            )


if __name__ == "__main__":
    unittest.main()
