from __future__ import annotations

import unittest
from pathlib import Path

from scripts.decide_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_objective_next import (
    BOUNDARY,
    NEXT_BOUNDARY,
    StageBMidiToSoloModelDirectJazzPhraseVocabularyRepairObjectiveNextError,
    build_objective_next_decision_report,
    validate_objective_next_decision_report,
)


def listening_review(*, validated_input: bool = False, quality_claim: bool = False) -> dict:
    return {
        "schema_version": "stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_listening_review_v1",
        "boundary": "stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_listening_review",
        "listening_review_boundary": {
            "boundary": "stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_listening_review",
            "source_boundary": "stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_audio_package",
            "candidate_count": 3,
            "rendered_audio_file_count": 3,
            "review_input_template_written": True,
            "validated_review_input_present": validated_input,
            "preference_fill_allowed": validated_input,
            "human_audio_preference_claimed": False,
            "midi_to_solo_musical_quality_claimed": False,
        },
        "readiness": {
            "boundary": "stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_listening_review",
            "listening_review_boundary_prepared": True,
            "validated_review_input_present": validated_input,
            "preference_fill_allowed": validated_input,
            "listening_review_completed": False,
            "human_audio_preference_claimed": False,
            "model_direct_generation_quality_claimed": quality_claim,
            "midi_to_solo_musical_quality_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
        },
        "decision": {
            "next_boundary": BOUNDARY,
            "critical_user_input_required": False,
        },
        "review_candidates": [
            {
                "rank": rank,
                "note_count": 36,
                "max_abs_interval": 10 + (rank % 2) * 2,
                "duration_most_common_ratio": 0.25 + (rank % 2) * 0.0833,
                "ioi_most_common_ratio": 0.3142 + (rank % 2) * 0.0285,
                "density_pattern": patterns[rank - 1],
                "analysis_flags": ["stepwise_contour_bias"],
            }
            for rank, patterns in [
                (1, [[5, 3, 6, 4, 5, 4, 6, 3], [6, 4, 3, 5, 6, 5, 3, 4], [5, 6, 4, 3, 5, 3, 6, 4]]),
                (2, [[5, 3, 6, 4, 5, 4, 6, 3], [6, 4, 3, 5, 6, 5, 3, 4], [5, 6, 4, 3, 5, 3, 6, 4]]),
                (3, [[5, 3, 6, 4, 5, 4, 6, 3], [6, 4, 3, 5, 6, 5, 3, 4], [5, 6, 4, 3, 5, 3, 6, 4]]),
            ]
        ],
    }


class StageBMidiToSoloModelDirectJazzPhraseVocabularyRepairObjectiveNextTest(unittest.TestCase):
    def test_routes_pending_review_to_contour_phrase_shape_repair(self) -> None:
        report = build_objective_next_decision_report(
            listening_review(),
            output_dir=Path("outputs/objective_next"),
        )
        summary = validate_objective_next_decision_report(
            report,
            expected_boundary=BOUNDARY,
            expected_next_boundary=NEXT_BOUNDARY,
            min_repair_target_count=6,
            require_stepwise_target=True,
            require_pending_review=True,
            require_no_quality_claim=True,
        )

        self.assertEqual(summary["candidate_count"], 3)
        self.assertEqual(summary["stepwise_contour_bias_count"], 3)
        self.assertTrue(summary["all_candidates_stepwise_biased"])
        self.assertEqual(summary["distinct_density_pattern_count"], 3)
        self.assertIn("reduce_stepwise_contour_bias", summary["selected_repair_targets"])
        self.assertFalse(summary["validated_review_input_present"])
        self.assertFalse(summary["preference_fill_allowed"])
        self.assertFalse(summary["human_audio_preference_claimed"])
        self.assertFalse(summary["midi_to_solo_musical_quality_claimed"])
        self.assertEqual(summary["next_boundary"], NEXT_BOUNDARY)

    def test_rejects_validated_review_input_for_objective_only_path(self) -> None:
        with self.assertRaises(StageBMidiToSoloModelDirectJazzPhraseVocabularyRepairObjectiveNextError):
            build_objective_next_decision_report(
                listening_review(validated_input=True),
                output_dir=Path("outputs/objective_next"),
            )

    def test_rejects_upstream_quality_claim(self) -> None:
        with self.assertRaises(StageBMidiToSoloModelDirectJazzPhraseVocabularyRepairObjectiveNextError):
            build_objective_next_decision_report(
                listening_review(quality_claim=True),
                output_dir=Path("outputs/objective_next"),
            )


if __name__ == "__main__":
    unittest.main()
