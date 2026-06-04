from __future__ import annotations

import unittest
from pathlib import Path

from scripts.build_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_repeatability_listening_review import (
    BOUNDARY as LISTENING_BOUNDARY,
)
from scripts.consolidate_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_repeatability import (
    BOUNDARY as CONSOLIDATION_BOUNDARY,
)
from scripts.decide_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_repeatability_objective_next import (
    FINAL_BOUNDARY,
    NEXT_BOUNDARY,
    StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeRepeatabilityObjectiveNextError,
    build_objective_next_decision_report,
    validate_objective_next_decision_report,
)


def listening_review(*, validated_input: bool = False, quality_claim: bool = False) -> dict:
    return {
        "schema_version": "stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_repeatability_listening_review_v1",
        "boundary": LISTENING_BOUNDARY,
        "source_boundary": "stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_repeatability_audio_review_package",
        "review_candidates": [
            {
                "rank": rank,
                "midi_path": f"outputs/repeatability/midi/seed_{rank:02d}.mid",
                "wav_path": f"outputs/repeatability/audio/seed_{rank:02d}.wav",
                "duration_seconds": 18.9,
                "sample_rate": 44100,
            }
            for rank in range(1, 7)
        ],
        "review_input_summary": {
            "validated_review_input_present": validated_input,
            "pending_status_field_count": 4,
            "pending_candidate_decision_count": 6,
            "pending_candidate_field_count": 18,
        },
        "listening_review_boundary": {
            "boundary": LISTENING_BOUNDARY,
            "source_boundary": "stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_repeatability_audio_review_package",
            "candidate_count": 6,
            "rendered_audio_file_count": 6,
            "review_input_template_written": True,
            "validated_review_input_present": validated_input,
            "preference_fill_allowed": validated_input,
            "listening_review_completed": False,
            "human_audio_preference_claimed": False,
            "model_direct_generation_quality_claimed": quality_claim,
            "midi_to_solo_musical_quality_claimed": False,
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
            "model_direct_generation_quality_claimed": quality_claim,
            "midi_to_solo_musical_quality_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
        },
        "decision": {
            "next_boundary": "stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_repeatability_objective_only_next_decision",
            "critical_user_input_required": False,
        },
    }


def repeatability_consolidation(
    *,
    qualified_count: int = 6,
    flag_count: int = 0,
    quality_claim: bool = False,
) -> dict:
    return {
        "schema_version": "stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_repeatability_consolidation_v1",
        "boundary": CONSOLIDATION_BOUNDARY,
        "source_boundary": "stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_objective_clean_repeatability_sweep",
        "evidence_summary": {
            "sample_count": 6,
            "generated_midi_file_count": 6,
            "qualified_candidate_count": qualified_count,
            "objective_clean_pass_rate": qualified_count / 6,
            "current_analysis_flag_count": flag_count,
            "overlap_detected_count": 0,
            "distinct_density_pattern_count": 6,
            "max_abs_interval_max": 12,
            "max_small_interval_ratio_le4": 0.1765,
        },
        "consolidation_result": {
            "objective_repeatability_support": qualified_count == 6 and flag_count == 0,
            "additional_repair_required": qualified_count != 6 or flag_count != 0,
            "audio_review_package_required": True,
            "human_audio_preference_claimed": False,
            "midi_to_solo_musical_quality_claimed": False,
        },
        "readiness": {
            "boundary": CONSOLIDATION_BOUNDARY,
            "repeatability_consolidation_completed": True,
            "objective_repeatability_support": qualified_count == 6 and flag_count == 0,
            "audio_review_package_required": True,
            "human_audio_preference_claimed": False,
            "model_direct_generation_quality_claimed": quality_claim,
            "midi_to_solo_musical_quality_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
        },
        "decision": {
            "next_boundary": "stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_repeatability_audio_review_package",
            "critical_user_input_required": False,
        },
    }


class StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeRepeatabilityObjectiveNextTest(
    unittest.TestCase
):
    def test_closes_repeatability_objective_path_without_quality_claim(self) -> None:
        report = build_objective_next_decision_report(
            listening_review(),
            repeatability_consolidation(),
            output_dir=Path("outputs/repeatability_objective_next"),
        )
        summary = validate_objective_next_decision_report(
            report,
            expected_final_boundary=FINAL_BOUNDARY,
            expected_next_boundary=NEXT_BOUNDARY,
            min_sample_count=6,
            min_qualified_count=6,
            require_objective_support=True,
            require_pending_review=True,
            require_no_quality_claim=True,
        )

        self.assertEqual(summary["candidate_count"], 6)
        self.assertEqual(summary["rendered_audio_file_count"], 6)
        self.assertEqual(summary["sample_count"], 6)
        self.assertEqual(summary["qualified_candidate_count"], 6)
        self.assertEqual(summary["objective_clean_pass_rate"], 1.0)
        self.assertEqual(summary["current_analysis_flag_count"], 0)
        self.assertEqual(summary["overlap_detected_count"], 0)
        self.assertTrue(summary["objective_repeatability_path_supported"])
        self.assertFalse(summary["validated_review_input_present"])
        self.assertFalse(summary["preference_fill_allowed"])
        self.assertFalse(summary["human_audio_preference_claimed"])
        self.assertFalse(summary["midi_to_solo_musical_quality_claimed"])
        self.assertEqual(summary["pending_status_field_count"], 4)
        self.assertEqual(summary["pending_candidate_decision_count"], 6)
        self.assertEqual(summary["pending_candidate_field_count"], 18)
        self.assertEqual(summary["next_boundary"], NEXT_BOUNDARY)

    def test_rejects_validated_review_input_for_objective_only_path(self) -> None:
        with self.assertRaises(
            StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeRepeatabilityObjectiveNextError
        ):
            build_objective_next_decision_report(
                listening_review(validated_input=True),
                repeatability_consolidation(),
                output_dir=Path("outputs/repeatability_objective_next"),
            )

    def test_rejects_failed_repeatability_support(self) -> None:
        with self.assertRaises(
            StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeRepeatabilityObjectiveNextError
        ):
            build_objective_next_decision_report(
                listening_review(),
                repeatability_consolidation(qualified_count=5),
                output_dir=Path("outputs/repeatability_objective_next"),
            )

    def test_rejects_repeatability_analysis_flags(self) -> None:
        with self.assertRaises(
            StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeRepeatabilityObjectiveNextError
        ):
            build_objective_next_decision_report(
                listening_review(),
                repeatability_consolidation(flag_count=1),
                output_dir=Path("outputs/repeatability_objective_next"),
            )

    def test_rejects_quality_claim(self) -> None:
        with self.assertRaises(
            StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeRepeatabilityObjectiveNextError
        ):
            build_objective_next_decision_report(
                listening_review(quality_claim=True),
                repeatability_consolidation(),
                output_dir=Path("outputs/repeatability_objective_next"),
            )


if __name__ == "__main__":
    unittest.main()
