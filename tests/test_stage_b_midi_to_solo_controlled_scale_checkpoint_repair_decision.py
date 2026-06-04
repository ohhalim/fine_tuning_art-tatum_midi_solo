from __future__ import annotations

import unittest
from pathlib import Path

from scripts.decide_stage_b_midi_to_solo_controlled_scale_checkpoint_repair import (
    BOUNDARY,
    NEXT_BOUNDARY,
    StageBMidiToSoloControlledScaleCheckpointRepairDecisionError,
    build_repair_decision,
    validate_repair_decision,
)
from scripts.summarize_stage_b_midi_to_solo_controlled_scale_checkpoint_generation_probe import (
    BOUNDARY as SOURCE_BOUNDARY,
)


def generation_probe(
    *,
    next_boundary: str = BOUNDARY,
    note_count_failures: int = 3,
    collapse_count: int = 3,
    avg_postprocess: float = 0.809,
    quality_claim: bool = False,
) -> dict:
    return {
        "boundary": SOURCE_BOUNDARY,
        "training_source": {
            "selected_train_records": 512,
            "selected_val_records": 128,
            "max_sequence": 160,
            "best_validation_loss": 5.1061,
            "checkpoint_count": 1,
        },
        "generation_summary": {
            "command_returncode": 0,
            "sample_count": 3,
            "valid_sample_count": 0,
            "strict_valid_sample_count": 0,
            "grammar_gate_sample_count": 3,
            "passed_generation_gate": False,
            "passed_grammar_gate": True,
            "passed_strict_review_gate": False,
            "collapse_warning_sample_count": collapse_count,
            "collapse_warning_sample_rate": collapse_count / 3,
            "avg_onset_coverage_ratio": 0.0833,
            "avg_sustained_coverage_ratio": 0.1667,
            "max_longest_sustained_empty_run_steps": 32,
            "avg_postprocess_removal_ratio": avg_postprocess,
            "max_postprocess_removal_ratio": 0.864,
            "diagnostic_failure_reasons": {
                "note count too low: 3 < 6; collapse=postprocess_removed_majority": note_count_failures,
            },
            "strict_failure_reasons": {
                "postprocess removal ratio too high: 0.864 > 0.490": 1,
                "repeated position/pitch pair ratio too high: 0.500 > 0.490": 1,
            },
        },
        "readiness": {
            "generation_path_executable": True,
            "raw_generation_quality_ready": False,
            "model_direct_generation_quality_claimed": quality_claim,
            "midi_to_solo_musical_quality_claimed": False,
            "human_audio_preference_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
        },
        "decision": {
            "next_boundary": next_boundary,
            "critical_user_input_required": False,
        },
    }


class StageBMidiToSoloControlledScaleCheckpointRepairDecisionTest(unittest.TestCase):
    def test_selects_density_collapse_postprocess_repair_target(self) -> None:
        report = build_repair_decision(
            generation_probe(),
            output_dir=Path("outputs/repair_decision"),
        )
        summary = validate_repair_decision(
            report,
            expected_boundary=BOUNDARY,
            expected_next_boundary=NEXT_BOUNDARY,
            require_repair_target=True,
            require_no_quality_claim=True,
        )

        self.assertEqual(summary["sample_count"], 3)
        self.assertEqual(summary["valid_sample_count"], 0)
        self.assertEqual(summary["grammar_gate_sample_count"], 3)
        self.assertEqual(summary["note_count_failure_count"], 3)
        self.assertTrue(summary["all_samples_note_count_failed"])
        self.assertTrue(summary["collapse_across_all_samples"])
        self.assertTrue(summary["postprocess_removal_high"])
        self.assertTrue(summary["low_coverage_observed"])
        self.assertEqual(summary["selected_target"], "target_density_collapse_postprocess_repair")
        self.assertFalse(summary["postprocess_only_repair_selected"])
        self.assertFalse(summary["training_scale_change_selected"])
        self.assertFalse(summary["critical_user_input_required"])
        self.assertFalse(summary["midi_to_solo_musical_quality_claimed"])

    def test_rejects_unexpected_next_boundary(self) -> None:
        with self.assertRaises(StageBMidiToSoloControlledScaleCheckpointRepairDecisionError):
            build_repair_decision(
                generation_probe(next_boundary="stage_b_other_boundary"),
                output_dir=Path("outputs/repair_decision"),
            )

    def test_rejects_missing_note_count_evidence(self) -> None:
        with self.assertRaises(StageBMidiToSoloControlledScaleCheckpointRepairDecisionError):
            build_repair_decision(
                generation_probe(note_count_failures=0),
                output_dir=Path("outputs/repair_decision"),
            )

    def test_rejects_quality_claim(self) -> None:
        with self.assertRaises(StageBMidiToSoloControlledScaleCheckpointRepairDecisionError):
            build_repair_decision(
                generation_probe(quality_claim=True),
                output_dir=Path("outputs/repair_decision"),
            )

    def test_validation_rejects_non_collapse_target_evidence(self) -> None:
        report = build_repair_decision(
            generation_probe(collapse_count=2, avg_postprocess=0.6),
            output_dir=Path("outputs/repair_decision"),
        )
        with self.assertRaises(StageBMidiToSoloControlledScaleCheckpointRepairDecisionError):
            validate_repair_decision(
                report,
                expected_boundary=BOUNDARY,
                expected_next_boundary=NEXT_BOUNDARY,
                require_repair_target=True,
                require_no_quality_claim=True,
            )


if __name__ == "__main__":
    unittest.main()
