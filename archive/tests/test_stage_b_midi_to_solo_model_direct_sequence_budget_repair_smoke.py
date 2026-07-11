from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.consolidate_stage_b_midi_to_solo_model_direct_sequence_budget_repair_smoke import (
    BOUNDARY,
    NEXT_BOUNDARY,
    StageBMidiToSoloModelDirectSequenceBudgetRepairSmokeError,
    build_sequence_budget_repair_smoke_report,
    validate_sequence_budget_repair_smoke_report,
)


def previous_repair_report(*, quality_claim: bool = False) -> dict:
    return {
        "boundary": "stage_b_midi_to_solo_model_direct_generation_repair",
        "direct_generation_contract": {
            "target_solo_bars": 8,
            "min_note_count": 24,
            "min_unique_pitch_count": 8,
            "max_simultaneous_notes": 1,
            "current_generation_source": "context_conditioned_fallback",
            "required_generation_source": "model_checkpoint_direct",
        },
        "sequence_budget_analysis": {
            "token_accounting": {
                "min_contract_tokens": 123,
            },
            "current_checkpoint": {
                "max_sequence": 96,
                "direct_note_capacity_under_budget": 17,
                "sequence_budget_sufficient_for_contract": False,
            },
        },
        "repair_scope": {
            "direct_repair_required": True,
            "recommended_max_sequence": 160,
        },
        "readiness": {
            "technical_mvp_preserved": True,
            "current_checkpoint_sequence_budget_sufficient": False,
            "model_direct_generation_quality_claimed": quality_claim,
            "midi_to_solo_musical_quality_claimed": False,
            "human_audio_preference_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
        },
        "decision": {
            "next_boundary": BOUNDARY,
            "critical_user_input_required": False,
        },
    }


def repaired_scale_smoke_report(*, max_sequence: int = 160, broad_claim: bool = False) -> dict:
    return {
        "readiness": {
            "boundary": "stage_b_generic_base_training_scale_smoke",
            "training_scale_smoke_passed": True,
            "generic_base_scale_checkpoint_generation_probe_ready": True,
            "full_generic_training_executed": False,
            "broad_trained_model_quality_claimed": broad_claim,
            "brad_style_adaptation_claimed": False,
        },
        "input": {
            "selected_train_records": 128,
            "selected_val_records": 32,
        },
        "training_config": {
            "max_sequence": max_sequence,
            "epochs": 1,
            "batch_size": 16,
        },
        "training": {
            "returncode": 0,
            "best_validation_loss": 5.8123,
        },
        "token_stats": {
            "max_token_id": 544,
            "vocab_size": 547,
            "fits_vocab": True,
        },
        "artifacts": {
            "checkpoint_count": 1,
            "lora_weights_exists": True,
            "checkpoint_files": ["checkpoint_epoch1.pt"],
        },
    }


class StageBMidiToSoloModelDirectSequenceBudgetRepairSmokeTest(unittest.TestCase):
    def test_consolidates_repaired_sequence_budget_without_quality_claim(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            report = build_sequence_budget_repair_smoke_report(
                previous_repair=previous_repair_report(),
                repaired_training_scale_smoke=repaired_scale_smoke_report(),
                output_dir=Path(temp_dir),
                issue_number=495,
            )
            summary = validate_sequence_budget_repair_smoke_report(
                report,
                expected_boundary=BOUNDARY,
                expected_next_boundary=NEXT_BOUNDARY,
                require_sequence_budget_sufficient=True,
                require_probe_ready=True,
                require_no_quality_claim=True,
            )

        self.assertEqual(summary["previous_max_sequence"], 96)
        self.assertEqual(summary["repaired_max_sequence"], 160)
        self.assertEqual(summary["minimum_contract_tokens"], 123)
        self.assertEqual(summary["previous_direct_note_capacity"], 17)
        self.assertEqual(summary["repaired_direct_note_capacity"], 33)
        self.assertEqual(summary["target_min_note_count"], 24)
        self.assertTrue(summary["sequence_budget_repaired"])
        self.assertTrue(summary["model_direct_8bar_generation_probe_ready"])
        self.assertFalse(summary["model_direct_generation_quality_claimed"])

    def test_rejects_insufficient_repaired_sequence_budget(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            report = build_sequence_budget_repair_smoke_report(
                previous_repair=previous_repair_report(),
                repaired_training_scale_smoke=repaired_scale_smoke_report(max_sequence=120),
                output_dir=Path(temp_dir),
                issue_number=495,
            )
            with self.assertRaises(StageBMidiToSoloModelDirectSequenceBudgetRepairSmokeError):
                validate_sequence_budget_repair_smoke_report(
                    report,
                    expected_boundary=BOUNDARY,
                    expected_next_boundary=NEXT_BOUNDARY,
                    require_sequence_budget_sufficient=True,
                    require_probe_ready=True,
                    require_no_quality_claim=True,
                )

    def test_rejects_previous_quality_claim(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            report = build_sequence_budget_repair_smoke_report(
                previous_repair=previous_repair_report(quality_claim=True),
                repaired_training_scale_smoke=repaired_scale_smoke_report(),
                output_dir=Path(temp_dir),
                issue_number=495,
            )
            with self.assertRaises(StageBMidiToSoloModelDirectSequenceBudgetRepairSmokeError):
                validate_sequence_budget_repair_smoke_report(
                    report,
                    expected_boundary=BOUNDARY,
                    expected_next_boundary=NEXT_BOUNDARY,
                    require_sequence_budget_sufficient=True,
                    require_probe_ready=True,
                    require_no_quality_claim=True,
                )

    def test_rejects_repaired_training_quality_claim(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            report = build_sequence_budget_repair_smoke_report(
                previous_repair=previous_repair_report(),
                repaired_training_scale_smoke=repaired_scale_smoke_report(broad_claim=True),
                output_dir=Path(temp_dir),
                issue_number=495,
            )
            with self.assertRaises(StageBMidiToSoloModelDirectSequenceBudgetRepairSmokeError):
                validate_sequence_budget_repair_smoke_report(
                    report,
                    expected_boundary=BOUNDARY,
                    expected_next_boundary=NEXT_BOUNDARY,
                    require_sequence_budget_sufficient=True,
                    require_probe_ready=True,
                    require_no_quality_claim=True,
                )

    def test_boundary_constants_are_stable(self) -> None:
        self.assertEqual(BOUNDARY, "stage_b_midi_to_solo_model_direct_sequence_budget_repair_smoke")
        self.assertEqual(NEXT_BOUNDARY, "stage_b_midi_to_solo_model_direct_8bar_generation_probe")


if __name__ == "__main__":
    unittest.main()
