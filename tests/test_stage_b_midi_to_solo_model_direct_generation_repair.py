from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.check_stage_b_midi_to_solo_model_direct_generation_repair import (
    BOUNDARY,
    NEXT_BOUNDARY,
    SCALE_SMOKE_BOUNDARY,
    StageBMidiToSoloModelDirectGenerationRepairError,
    build_model_direct_generation_repair_report,
    estimate_direct_stage_b_token_budget,
    validate_model_direct_generation_repair_report,
)


def mvp_execution_report(*, technical_mvp: bool = True, quality_claim: bool = False) -> dict:
    return {
        "boundary": "stage_b_midi_to_solo_mvp_execution_consolidation",
        "contract": {
            "target_solo_bars": 8,
            "min_note_count": 24,
            "min_unique_pitch_count": 8,
            "max_simultaneous_notes": 1,
        },
        "execution_path": {
            "generation_source": "context_conditioned_fallback",
            "exported_candidate_count": 3,
            "rendered_audio_file_count": 3,
        },
        "readiness": {
            "technical_execution_path_completed": technical_mvp,
            "input_to_ranked_midi_completed": technical_mvp,
            "input_to_rendered_audio_completed": technical_mvp,
            "midi_to_solo_technical_mvp_completed": technical_mvp,
            "midi_to_solo_musical_quality_claimed": quality_claim,
            "model_checkpoint_direct_generation_quality_claimed": quality_claim,
            "human_audio_preference_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
        },
        "decision": {
            "next_boundary": BOUNDARY,
            "critical_user_input_required": False,
        },
    }


def scale_smoke_report(*, max_sequence: int = 96, broad_claim: bool = False) -> dict:
    return {
        "readiness": {
            "boundary": SCALE_SMOKE_BOUNDARY,
            "training_scale_smoke_passed": True,
            "generic_base_scale_checkpoint_generation_probe_ready": True,
            "full_generic_training_executed": False,
            "broad_trained_model_quality_claimed": broad_claim,
            "brad_style_adaptation_claimed": False,
        },
        "decision": {"next_boundary": "stage_b_generic_base_scale_checkpoint_generation_probe"},
        "input": {
            "selected_train_records": 128,
            "selected_val_records": 32,
        },
        "training_config": {
            "max_sequence": max_sequence,
            "epochs": 1,
            "batch_size": 16,
        },
        "training": {"best_validation_loss": 5.9031},
        "token_stats": {
            "max_token_id": 544,
            "vocab_size": 547,
            "fits_vocab": True,
        },
        "artifacts": {"checkpoint_count": 1},
    }


class StageBMidiToSoloModelDirectGenerationRepairTest(unittest.TestCase):
    def test_estimates_current_sequence_gap_for_8bar_contract(self) -> None:
        budget = estimate_direct_stage_b_token_budget(
            target_bars=8,
            min_note_count=24,
            current_max_sequence=96,
        )

        self.assertEqual(budget["token_accounting"]["overhead_tokens"], 27)
        self.assertEqual(budget["token_accounting"]["min_contract_tokens"], 123)
        self.assertEqual(budget["current_checkpoint"]["direct_note_capacity_under_budget"], 17)
        self.assertFalse(budget["current_checkpoint"]["sequence_budget_sufficient_for_contract"])
        self.assertEqual(budget["repair_target"]["recommended_max_sequence"], 160)

    def test_builds_repair_boundary_without_quality_claim(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            report = build_model_direct_generation_repair_report(
                mvp_execution_report=mvp_execution_report(),
                training_scale_smoke=scale_smoke_report(),
                output_dir=Path(temp_dir),
                issue_number=493,
            )
            summary = validate_model_direct_generation_repair_report(
                report,
                expected_boundary=BOUNDARY,
                expected_next_boundary=NEXT_BOUNDARY,
                require_technical_mvp=True,
                require_sequence_budget_gap=True,
                require_no_quality_claim=True,
            )

        self.assertTrue(summary["technical_mvp_preserved"])
        self.assertEqual(summary["current_generation_source"], "context_conditioned_fallback")
        self.assertEqual(summary["required_generation_source"], "model_checkpoint_direct")
        self.assertEqual(summary["current_checkpoint_max_sequence"], 96)
        self.assertEqual(summary["minimum_contract_tokens"], 123)
        self.assertEqual(summary["direct_note_capacity_under_current_budget"], 17)
        self.assertTrue(summary["direct_repair_required"])
        self.assertFalse(summary["model_direct_generation_quality_claimed"])

    def test_rejects_missing_technical_mvp(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(StageBMidiToSoloModelDirectGenerationRepairError):
                build_model_direct_generation_repair_report(
                    mvp_execution_report=mvp_execution_report(technical_mvp=False),
                    training_scale_smoke=scale_smoke_report(),
                    output_dir=Path(temp_dir),
                    issue_number=493,
                )

    def test_rejects_upstream_quality_claim(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            report = build_model_direct_generation_repair_report(
                mvp_execution_report=mvp_execution_report(quality_claim=True),
                training_scale_smoke=scale_smoke_report(),
                output_dir=Path(temp_dir),
                issue_number=493,
            )
            with self.assertRaises(StageBMidiToSoloModelDirectGenerationRepairError):
                validate_model_direct_generation_repair_report(
                    report,
                    expected_boundary=BOUNDARY,
                    expected_next_boundary=NEXT_BOUNDARY,
                    require_technical_mvp=True,
                    require_sequence_budget_gap=True,
                    require_no_quality_claim=True,
                )

    def test_require_sequence_gap_rejects_sufficient_budget(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            report = build_model_direct_generation_repair_report(
                mvp_execution_report=mvp_execution_report(),
                training_scale_smoke=scale_smoke_report(max_sequence=160),
                output_dir=Path(temp_dir),
                issue_number=493,
            )
            with self.assertRaises(StageBMidiToSoloModelDirectGenerationRepairError):
                validate_model_direct_generation_repair_report(
                    report,
                    expected_boundary=BOUNDARY,
                    expected_next_boundary=NEXT_BOUNDARY,
                    require_technical_mvp=True,
                    require_sequence_budget_gap=True,
                    require_no_quality_claim=True,
                )

    def test_boundary_constants_are_stable(self) -> None:
        self.assertEqual(BOUNDARY, "stage_b_midi_to_solo_model_direct_generation_repair")
        self.assertEqual(NEXT_BOUNDARY, "stage_b_midi_to_solo_model_direct_sequence_budget_repair_smoke")
        self.assertEqual(SCALE_SMOKE_BOUNDARY, "stage_b_generic_base_training_scale_smoke")


if __name__ == "__main__":
    unittest.main()
