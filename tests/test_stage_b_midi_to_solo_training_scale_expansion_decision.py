from __future__ import annotations

import unittest
from pathlib import Path

from scripts.check_stage_b_midi_to_solo_training_resource_probe import (
    BOUNDARY as TRAINING_RESOURCE_BOUNDARY,
)
from scripts.consolidate_stage_b_midi_to_solo_model_direct_sequence_budget_repair_smoke import (
    BOUNDARY as SEQUENCE_BUDGET_BOUNDARY,
)
from scripts.decide_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_repeatability_objective_next import (
    BOUNDARY as OBJECTIVE_PATH_BOUNDARY,
    FINAL_BOUNDARY as OBJECTIVE_FINAL_BOUNDARY,
)
from scripts.decide_stage_b_midi_to_solo_training_scale_expansion import (
    BOUNDARY,
    NEXT_BOUNDARY,
    StageBMidiToSoloTrainingScaleExpansionDecisionError,
    build_training_scale_expansion_decision,
    validate_training_scale_expansion_decision,
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
        "scale_smoke_resource": {
            "selected_train_records": 128,
            "selected_val_records": 32,
            "best_validation_loss": 5.9031,
            "checkpoint_count": 1,
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


def sequence_budget(*, max_sequence: int = 160, quality_claim: bool = False) -> dict:
    return {
        "boundary": SEQUENCE_BUDGET_BOUNDARY,
        "repair_result": {
            "previous_max_sequence": 96,
            "repaired_max_sequence": max_sequence,
            "previous_direct_note_capacity": 17,
            "repaired_direct_note_capacity": 33 if max_sequence >= 160 else 20,
            "target_min_note_count": 24,
            "minimum_contract_tokens": 123,
            "sequence_budget_repaired": max_sequence >= 160,
        },
        "readiness": {
            "model_direct_8bar_generation_probe_ready": max_sequence >= 160,
            "model_direct_generation_quality_claimed": quality_claim,
            "midi_to_solo_musical_quality_claimed": False,
            "human_audio_preference_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
        },
    }


def objective_path(*, quality_claim: bool = False, review_present: bool = False) -> dict:
    return {
        "boundary": OBJECTIVE_PATH_BOUNDARY,
        "decision": {
            "final_boundary": OBJECTIVE_FINAL_BOUNDARY,
            "next_boundary": "stage_b_model_core_evidence_readme_refresh",
            "critical_user_input_required": False,
        },
        "objective_repeatability_summary": {
            "sample_count": 6,
            "qualified_candidate_count": 6,
            "objective_clean_pass_rate": 1.0,
            "current_analysis_flag_count": 0,
            "overlap_detected_count": 0,
        },
        "review_boundary_summary": {
            "rendered_audio_file_count": 6,
            "validated_review_input_present": review_present,
            "pending_status_field_count": 4,
            "pending_candidate_decision_count": 6,
            "pending_candidate_field_count": 18,
        },
        "readiness": {
            "objective_repeatability_path_supported": True,
            "human_audio_preference_claimed": False,
            "model_direct_generation_quality_claimed": quality_claim,
            "midi_to_solo_musical_quality_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
        },
    }


class StageBMidiToSoloTrainingScaleExpansionDecisionTest(unittest.TestCase):
    def test_selects_bounded_scale_smoke_after_objective_path_support(self) -> None:
        report = build_training_scale_expansion_decision(
            training_resource(),
            sequence_budget(),
            objective_path(),
            output_dir=Path("outputs/training_scale_expansion"),
            target_train_records=512,
            target_val_records=128,
        )
        summary = validate_training_scale_expansion_decision(
            report,
            expected_boundary=BOUNDARY,
            expected_next_boundary=NEXT_BOUNDARY,
            min_selected_train_records=512,
            min_selected_val_records=128,
            require_scale_ready=True,
            require_no_quality_claim=True,
        )

        self.assertEqual(summary["selected_train_records"], 512)
        self.assertEqual(summary["selected_val_records"], 128)
        self.assertEqual(summary["prior_train_records"], 128)
        self.assertEqual(summary["prior_val_records"], 32)
        self.assertEqual(summary["scale_multiplier_train"], 4.0)
        self.assertEqual(summary["scale_multiplier_val"], 4.0)
        self.assertEqual(summary["max_sequence"], 160)
        self.assertEqual(summary["objective_sample_count"], 6)
        self.assertEqual(summary["objective_qualified_candidate_count"], 6)
        self.assertEqual(summary["rendered_audio_file_count"], 6)
        self.assertTrue(summary["controlled_training_scale_smoke_ready"])
        self.assertFalse(summary["cloud_or_gpu_spend_required"])
        self.assertFalse(summary["critical_user_input_required"])
        self.assertFalse(summary["midi_to_solo_musical_quality_claimed"])

    def test_rejects_small_full_training_resource(self) -> None:
        with self.assertRaises(StageBMidiToSoloTrainingScaleExpansionDecisionError):
            build_training_scale_expansion_decision(
                training_resource(train_records=999),
                sequence_budget(),
                objective_path(),
                output_dir=Path("outputs/training_scale_expansion"),
                target_train_records=512,
                target_val_records=128,
            )

    def test_rejects_insufficient_sequence_budget(self) -> None:
        with self.assertRaises(StageBMidiToSoloTrainingScaleExpansionDecisionError):
            build_training_scale_expansion_decision(
                training_resource(),
                sequence_budget(max_sequence=120),
                objective_path(),
                output_dir=Path("outputs/training_scale_expansion"),
                target_train_records=512,
                target_val_records=128,
            )

    def test_rejects_validated_review_input_for_objective_path(self) -> None:
        with self.assertRaises(StageBMidiToSoloTrainingScaleExpansionDecisionError):
            build_training_scale_expansion_decision(
                training_resource(),
                sequence_budget(),
                objective_path(review_present=True),
                output_dir=Path("outputs/training_scale_expansion"),
                target_train_records=512,
                target_val_records=128,
            )

    def test_rejects_quality_claim(self) -> None:
        with self.assertRaises(StageBMidiToSoloTrainingScaleExpansionDecisionError):
            build_training_scale_expansion_decision(
                training_resource(quality_claim=True),
                sequence_budget(),
                objective_path(),
                output_dir=Path("outputs/training_scale_expansion"),
                target_train_records=512,
                target_val_records=128,
            )


if __name__ == "__main__":
    unittest.main()
