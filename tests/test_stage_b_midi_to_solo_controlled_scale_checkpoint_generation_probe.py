from __future__ import annotations

import unittest
from pathlib import Path

from scripts.run_stage_b_generic_base_scale_checkpoint_generation_probe import (
    BOUNDARY as GENERIC_GENERATION_BOUNDARY,
)
from scripts.summarize_stage_b_midi_to_solo_controlled_scale_checkpoint_generation_probe import (
    BOUNDARY,
    FAIL_NEXT_BOUNDARY,
    PASS_NEXT_BOUNDARY,
    StageBMidiToSoloControlledScaleCheckpointGenerationProbeSummaryError,
    build_controlled_generation_probe_summary,
    validate_controlled_generation_probe_summary,
)
from scripts.summarize_stage_b_midi_to_solo_controlled_training_scale_smoke import (
    BOUNDARY as CONTROLLED_TRAINING_BOUNDARY,
)


def controlled_training_report(*, checkpoint_count: int = 1) -> dict:
    return {
        "boundary": CONTROLLED_TRAINING_BOUNDARY,
        "training_result": {
            "selected_train_records": 512,
            "selected_val_records": 128,
            "max_sequence": 160,
            "best_validation_loss": 5.1061,
            "checkpoint_count": checkpoint_count,
        },
        "readiness": {
            "checkpoint_generation_probe_ready": True,
            "midi_to_solo_musical_quality_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
        },
    }


def generation_probe_report(
    *,
    sample_count: int = 3,
    strict_count: int = 0,
    command_returncode: int = 0,
    broad_claim: bool = False,
) -> dict:
    passed_strict = strict_count > 0
    return {
        "readiness": {
            "boundary": GENERIC_GENERATION_BOUNDARY,
            "generation_path_executable": command_returncode == 0 and sample_count > 0,
            "broad_trained_model_quality_claimed": broad_claim,
            "brad_style_adaptation_claimed": False,
            "human_audio_preference_claimed": False,
            "production_ready_improviser_claimed": False,
        },
        "decision": {
            "next_boundary": (
                "stage_b_generic_base_scale_checkpoint_repeatability_probe"
                if passed_strict
                else "stage_b_generic_base_scale_checkpoint_grammar_representation_decision"
            ),
            "critical_user_input_required": False,
        },
        "generation_command": {
            "returncode": command_returncode,
        },
        "generation_summary": {
            "sample_count": sample_count,
            "valid_sample_count": sample_count if sample_count else 0,
            "strict_valid_sample_count": strict_count,
            "grammar_gate_sample_count": sample_count if sample_count else 0,
            "passed_generation_gate": sample_count > 0,
            "passed_grammar_gate": sample_count > 0,
            "passed_strict_review_gate": passed_strict,
            "collapse_warning_sample_count": 0,
            "avg_onset_coverage_ratio": 0.75,
            "avg_sustained_coverage_ratio": 0.5,
            "max_longest_sustained_empty_run_steps": 8,
        },
    }


class StageBMidiToSoloControlledScaleCheckpointGenerationProbeTest(unittest.TestCase):
    def test_summarizes_generation_probe_without_quality_claim(self) -> None:
        report = build_controlled_generation_probe_summary(
            controlled_training_report(),
            generation_probe_report(sample_count=3, strict_count=0),
            output_dir=Path("outputs/controlled_scale_checkpoint_generation_probe"),
        )
        summary = validate_controlled_generation_probe_summary(
            report,
            expected_boundary=BOUNDARY,
            min_sample_count=1,
            require_generation_executable=True,
            require_no_quality_claim=True,
        )

        self.assertEqual(summary["selected_train_records"], 512)
        self.assertEqual(summary["selected_val_records"], 128)
        self.assertEqual(summary["sample_count"], 3)
        self.assertEqual(summary["valid_sample_count"], 3)
        self.assertEqual(summary["strict_valid_sample_count"], 0)
        self.assertEqual(summary["next_boundary"], FAIL_NEXT_BOUNDARY)
        self.assertFalse(summary["raw_generation_quality_ready"])
        self.assertFalse(summary["critical_user_input_required"])
        self.assertFalse(summary["midi_to_solo_musical_quality_claimed"])

    def test_routes_strict_gate_pass_to_repeatability(self) -> None:
        report = build_controlled_generation_probe_summary(
            controlled_training_report(),
            generation_probe_report(sample_count=3, strict_count=2),
            output_dir=Path("outputs/controlled_scale_checkpoint_generation_probe"),
        )

        self.assertEqual(report["decision"]["next_boundary"], PASS_NEXT_BOUNDARY)
        self.assertTrue(report["readiness"]["raw_generation_quality_ready"])

    def test_rejects_missing_training_checkpoint(self) -> None:
        with self.assertRaises(StageBMidiToSoloControlledScaleCheckpointGenerationProbeSummaryError):
            build_controlled_generation_probe_summary(
                controlled_training_report(checkpoint_count=0),
                generation_probe_report(),
                output_dir=Path("outputs/controlled_scale_checkpoint_generation_probe"),
            )

    def test_rejects_generation_command_failure(self) -> None:
        with self.assertRaises(StageBMidiToSoloControlledScaleCheckpointGenerationProbeSummaryError):
            build_controlled_generation_probe_summary(
                controlled_training_report(),
                generation_probe_report(command_returncode=1),
                output_dir=Path("outputs/controlled_scale_checkpoint_generation_probe"),
            )

    def test_rejects_broad_quality_claim(self) -> None:
        with self.assertRaises(StageBMidiToSoloControlledScaleCheckpointGenerationProbeSummaryError):
            build_controlled_generation_probe_summary(
                controlled_training_report(),
                generation_probe_report(broad_claim=True),
                output_dir=Path("outputs/controlled_scale_checkpoint_generation_probe"),
            )


if __name__ == "__main__":
    unittest.main()
