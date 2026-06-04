from __future__ import annotations

import unittest
from pathlib import Path

from scripts.build_stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repeatability_temperature_guard_listening_review import (
    BOUNDARY as LISTENING_BOUNDARY,
)
from scripts.consolidate_stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repeatability_temperature_guard_repair import (
    BOUNDARY as CONSOLIDATION_BOUNDARY,
)
from scripts.decide_stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repeatability_temperature_guard_objective_next import (
    FINAL_BOUNDARY,
    NEXT_BOUNDARY,
    StageBMidiToSoloControlledScaleCheckpointTemperatureGuardObjectiveNextError,
    build_objective_next_decision_report,
    validate_objective_next_decision_report,
)


def listening_review(*, validated_input: bool = False, quality_claim: bool = False) -> dict:
    return {
        "schema_version": "stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repeatability_temperature_guard_listening_review_v1",
        "boundary": LISTENING_BOUNDARY,
        "review_candidates": [
            {
                "rank": rank,
                "midi_path": f"outputs/temperature_guard/midi/rank_{rank:02d}.mid",
                "wav_path": f"outputs/temperature_guard/audio/rank_{rank:02d}.wav",
                "duration_seconds": 6.8,
                "sample_rate": 44100,
            }
            for rank in range(1, 4)
        ],
        "review_input_summary": {
            "validated_review_input_present": validated_input,
            "pending_status_fields": ["reviewer", "reviewed_at", "preferred_rank", "reject_all"],
            "pending_candidate_decisions": ["1", "2", "3"],
            "pending_candidate_fields": [
                "rank_1.musical_acceptance",
                "rank_1.issue_tags",
                "rank_1.short_note",
                "rank_2.musical_acceptance",
                "rank_2.issue_tags",
                "rank_2.short_note",
                "rank_3.musical_acceptance",
                "rank_3.issue_tags",
                "rank_3.short_note",
            ],
        },
        "listening_review_boundary": {
            "boundary": LISTENING_BOUNDARY,
            "candidate_count": 3,
            "rendered_audio_file_count": 3,
            "review_input_template_written": True,
            "validated_review_input_present": validated_input,
            "preference_fill_allowed": validated_input,
            "listening_review_completed": False,
            "human_audio_preference_claimed": False,
            "midi_to_solo_musical_quality_claimed": quality_claim,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
        },
        "readiness": {
            "boundary": LISTENING_BOUNDARY,
            "listening_review_boundary_prepared": True,
            "validated_review_input_present": validated_input,
            "preference_fill_allowed": validated_input,
            "listening_review_completed": False,
            "human_audio_preference_claimed": False,
            "midi_to_solo_musical_quality_claimed": quality_claim,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
        },
        "decision": {
            "next_boundary": "stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repeatability_temperature_guard_objective_only_next_decision",
            "critical_user_input_required": False,
        },
    }


def consolidation(
    *,
    sample_count: int = 9,
    strict_count: int = 9,
    dead_air_failure_count: int = 0,
    collapse_count: int = 0,
    quality_claim: bool = False,
) -> dict:
    return {
        "schema_version": "stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repeatability_temperature_guard_repair_consolidation_v1",
        "boundary": CONSOLIDATION_BOUNDARY,
        "evidence_summary": {
            "sample_count": sample_count,
            "seed_count": 3,
            "valid_sample_count": strict_count,
            "strict_valid_sample_count": strict_count,
            "grammar_gate_sample_count": sample_count,
            "collapse_warning_sample_count": collapse_count,
            "dead_air_failure_count": dead_air_failure_count,
            "postprocess_collapse_failure_count": 0,
            "strict_valid_sample_delta": 2,
            "source_strict_sample_shortfall": 2,
            "repair_strict_sample_shortfall": sample_count - strict_count,
            "source_dead_air_failure_count": 2,
            "repair_dead_air_failure_count": dead_air_failure_count,
            "source_temperature": 0.9,
            "temperature": 0.75,
            "top_k": 4,
            "avg_postprocess_removal_ratio": 0.3657,
            "avg_onset_coverage_ratio": 0.5486,
            "avg_sustained_coverage_ratio": 0.7083,
        },
        "consolidation_result": {
            "objective_temperature_guard_support": strict_count == sample_count
            and dead_air_failure_count == 0
            and collapse_count == 0,
            "additional_repair_required": strict_count != sample_count
            or dead_air_failure_count != 0
            or collapse_count != 0,
            "audio_review_package_required": True,
            "human_audio_preference_claimed": False,
            "midi_to_solo_musical_quality_claimed": False,
        },
        "readiness": {
            "boundary": CONSOLIDATION_BOUNDARY,
            "temperature_guard_repair_consolidation_completed": True,
            "objective_temperature_guard_support": strict_count == sample_count,
            "audio_review_package_required": True,
            "human_audio_preference_claimed": False,
            "midi_to_solo_musical_quality_claimed": quality_claim,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
        },
        "decision": {
            "next_boundary": "stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repeatability_temperature_guard_audio_review_package",
            "critical_user_input_required": False,
        },
    }


class StageBMidiToSoloControlledScaleCheckpointTemperatureGuardObjectiveNextTest(
    unittest.TestCase
):
    def test_closes_objective_path_and_routes_training_scale_decision(self) -> None:
        report = build_objective_next_decision_report(
            listening_review(),
            consolidation(),
            output_dir=Path("outputs/temperature_guard_objective_next"),
        )
        summary = validate_objective_next_decision_report(
            report,
            expected_final_boundary=FINAL_BOUNDARY,
            expected_next_boundary=NEXT_BOUNDARY,
            min_sample_count=9,
            min_candidate_count=3,
            require_objective_support=True,
            require_pending_review=True,
            require_no_quality_claim=True,
        )

        self.assertEqual(summary["candidate_count"], 3)
        self.assertEqual(summary["rendered_audio_file_count"], 3)
        self.assertEqual(summary["sample_count"], 9)
        self.assertEqual(summary["strict_valid_sample_count"], 9)
        self.assertEqual(summary["grammar_gate_sample_count"], 9)
        self.assertEqual(summary["dead_air_failure_count"], 0)
        self.assertEqual(summary["collapse_warning_sample_count"], 0)
        self.assertEqual(summary["source_temperature"], 0.9)
        self.assertEqual(summary["temperature"], 0.75)
        self.assertEqual(summary["top_k"], 4)
        self.assertTrue(summary["objective_temperature_guard_path_supported"])
        self.assertFalse(summary["validated_review_input_present"])
        self.assertFalse(summary["preference_fill_allowed"])
        self.assertFalse(summary["human_audio_preference_claimed"])
        self.assertFalse(summary["midi_to_solo_musical_quality_claimed"])
        self.assertEqual(summary["pending_status_field_count"], 4)
        self.assertEqual(summary["pending_candidate_decision_count"], 3)
        self.assertEqual(summary["pending_candidate_field_count"], 9)
        self.assertEqual(summary["next_boundary"], NEXT_BOUNDARY)

    def test_rejects_validated_review_input(self) -> None:
        with self.assertRaises(
            StageBMidiToSoloControlledScaleCheckpointTemperatureGuardObjectiveNextError
        ):
            build_objective_next_decision_report(
                listening_review(validated_input=True),
                consolidation(),
                output_dir=Path("outputs/temperature_guard_objective_next"),
            )

    def test_rejects_strict_shortfall(self) -> None:
        with self.assertRaises(
            StageBMidiToSoloControlledScaleCheckpointTemperatureGuardObjectiveNextError
        ):
            build_objective_next_decision_report(
                listening_review(),
                consolidation(strict_count=8),
                output_dir=Path("outputs/temperature_guard_objective_next"),
            )

    def test_rejects_dead_air_failure(self) -> None:
        with self.assertRaises(
            StageBMidiToSoloControlledScaleCheckpointTemperatureGuardObjectiveNextError
        ):
            build_objective_next_decision_report(
                listening_review(),
                consolidation(dead_air_failure_count=1),
                output_dir=Path("outputs/temperature_guard_objective_next"),
            )

    def test_rejects_quality_claim(self) -> None:
        with self.assertRaises(
            StageBMidiToSoloControlledScaleCheckpointTemperatureGuardObjectiveNextError
        ):
            build_objective_next_decision_report(
                listening_review(quality_claim=True),
                consolidation(),
                output_dir=Path("outputs/temperature_guard_objective_next"),
            )

        with self.assertRaises(
            StageBMidiToSoloControlledScaleCheckpointTemperatureGuardObjectiveNextError
        ):
            build_objective_next_decision_report(
                listening_review(),
                consolidation(quality_claim=True),
                output_dir=Path("outputs/temperature_guard_objective_next"),
            )


if __name__ == "__main__":
    unittest.main()
