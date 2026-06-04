from __future__ import annotations

import unittest
from pathlib import Path

from scripts.decide_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_objective_next import (
    BOUNDARY,
    NEXT_BOUNDARY,
    StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeObjectiveNextError,
    build_objective_next_decision_report,
    validate_objective_next_decision_report,
)
from scripts.build_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_listening_review import (
    BOUNDARY as LISTENING_BOUNDARY,
)
from scripts.run_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_repair import (
    BOUNDARY as REPAIR_BOUNDARY,
)


def listening_review(*, validated_input: bool = False, quality_claim: bool = False, with_flag: bool = False) -> dict:
    flags = ["stepwise_contour_bias"] if with_flag else []
    return {
        "schema_version": "stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_listening_review_v1",
        "boundary": LISTENING_BOUNDARY,
        "listening_review_boundary": {
            "boundary": LISTENING_BOUNDARY,
            "source_boundary": "stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_audio_package",
            "candidate_count": 3,
            "rendered_audio_file_count": 3,
            "review_input_template_written": True,
            "validated_review_input_present": validated_input,
            "preference_fill_allowed": validated_input,
            "listening_review_completed": False,
            "human_audio_preference_claimed": False,
            "midi_to_solo_musical_quality_claimed": False,
        },
        "readiness": {
            "boundary": LISTENING_BOUNDARY,
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
                "rank": 1,
                "note_count": 36,
                "bar_count": 8,
                "max_abs_interval": 11,
                "small_interval_ratio_le4": 0.1714,
                "analysis_flags": flags,
                "density_pattern": [5, 3, 6, 4, 5, 4, 6, 3],
            },
            {
                "rank": 2,
                "note_count": 36,
                "bar_count": 8,
                "max_abs_interval": 11,
                "small_interval_ratio_le4": 0.1142,
                "analysis_flags": flags,
                "density_pattern": [6, 4, 3, 5, 6, 5, 3, 4],
            },
            {
                "rank": 3,
                "note_count": 36,
                "bar_count": 8,
                "max_abs_interval": 11,
                "small_interval_ratio_le4": 0.1428,
                "analysis_flags": flags,
                "density_pattern": [5, 6, 4, 3, 5, 3, 6, 4],
            },
        ],
    }


def contour_repair(*, target_passed: bool = True, no_overlap: bool = True, repaired_stepwise: int = 0) -> dict:
    return {
        "schema_version": "stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_repair_v1",
        "boundary": REPAIR_BOUNDARY,
        "source_objective_summary": {
            "candidate_count": 3,
            "source_stepwise_contour_bias_count": 3,
            "source_distinct_density_pattern_count": 3,
            "source_max_abs_interval_max": 12,
        },
        "aggregate": {
            "candidate_count": 3,
            "stepwise_contour_bias_count": repaired_stepwise,
            "shared_rhythm_signature_count": 1,
            "max_abs_interval_max": 11,
            "max_small_interval_ratio_le4": 0.1714,
        },
        "repair_result": {
            "target_passed": target_passed,
            "stepwise_contour_bias_reduced": repaired_stepwise == 0,
            "source_stepwise_contour_bias_count": 3,
            "repaired_stepwise_contour_bias_count": repaired_stepwise,
            "no_overlap": no_overlap,
            "phrase_vocabulary_source": "contour_phrase_shape_cells",
            "audio_review_required": True,
        },
        "readiness": {
            "boundary": REPAIR_BOUNDARY,
            "contour_phrase_shape_repair_completed": True,
            "objective_repair_target_passed": target_passed,
            "generated_midi_file_count": 3,
            "human_audio_preference_claimed": False,
            "model_direct_generation_quality_claimed": False,
            "midi_to_solo_musical_quality_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
        },
    }


class StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeObjectiveNextTest(unittest.TestCase):
    def test_routes_objective_clean_candidates_to_repeatability_sweep(self) -> None:
        report = build_objective_next_decision_report(
            listening_review(),
            contour_repair(),
            output_dir=Path("outputs/contour_objective_next"),
        )
        summary = validate_objective_next_decision_report(
            report,
            expected_boundary=BOUNDARY,
            expected_next_boundary=NEXT_BOUNDARY,
            require_objective_clean=True,
            require_pending_review=True,
            require_no_quality_claim=True,
        )

        self.assertEqual(summary["candidate_count"], 3)
        self.assertEqual(summary["current_analysis_flag_count"], 0)
        self.assertEqual(summary["source_stepwise_contour_bias_count"], 3)
        self.assertEqual(summary["repaired_stepwise_contour_bias_count"], 0)
        self.assertTrue(summary["stepwise_contour_bias_reduced"])
        self.assertTrue(summary["objective_clean_candidate_boundary_supported"])
        self.assertFalse(summary["additional_repair_required"])
        self.assertFalse(summary["validated_review_input_present"])
        self.assertFalse(summary["preference_fill_allowed"])
        self.assertFalse(summary["human_audio_preference_claimed"])
        self.assertFalse(summary["midi_to_solo_musical_quality_claimed"])
        self.assertIn("run_distinct_seed_repeatability_sweep", summary["selected_next_actions"])
        self.assertEqual(summary["next_boundary"], NEXT_BOUNDARY)

    def test_rejects_flagged_candidates_for_objective_clean_path(self) -> None:
        with self.assertRaises(StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeObjectiveNextError):
            build_objective_next_decision_report(
                listening_review(with_flag=True),
                contour_repair(),
                output_dir=Path("outputs/contour_objective_next"),
            )

    def test_rejects_validated_review_input_for_objective_only_path(self) -> None:
        with self.assertRaises(StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeObjectiveNextError):
            build_objective_next_decision_report(
                listening_review(validated_input=True),
                contour_repair(),
                output_dir=Path("outputs/contour_objective_next"),
            )

    def test_rejects_repair_without_no_overlap_evidence(self) -> None:
        with self.assertRaises(StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeObjectiveNextError):
            build_objective_next_decision_report(
                listening_review(),
                contour_repair(no_overlap=False),
                output_dir=Path("outputs/contour_objective_next"),
            )

    def test_rejects_upstream_quality_claim(self) -> None:
        with self.assertRaises(StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeObjectiveNextError):
            build_objective_next_decision_report(
                listening_review(quality_claim=True),
                contour_repair(),
                output_dir=Path("outputs/contour_objective_next"),
            )


if __name__ == "__main__":
    unittest.main()
