from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.decide_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_objective_next import (
    BOUNDARY as OBJECTIVE_NEXT_BOUNDARY,
)
from scripts.run_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_repeatability_sweep import (
    BOUNDARY,
    NEXT_BOUNDARY,
    StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeRepeatabilitySweepError,
    build_repeatability_sweep_report,
    validate_repeatability_sweep_report,
)


def objective_next_report(*, objective_clean: bool = True, quality_claim: bool = False) -> dict:
    return {
        "schema_version": "stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_objective_next_v1",
        "boundary": OBJECTIVE_NEXT_BOUNDARY,
        "source_boundary": "stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_listening_review",
        "repair_evidence_boundary": "stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_repair",
        "objective_summary": {
            "candidate_count": 3,
            "current_analysis_flag_count": 0 if objective_clean else 1,
            "source_stepwise_contour_bias_count": 3,
            "repaired_stepwise_contour_bias_count": 0,
            "stepwise_contour_bias_reduced": True,
            "objective_clean_candidate_boundary_supported": objective_clean,
            "additional_repair_required": not objective_clean,
            "distinct_density_pattern_count": 3,
            "max_abs_interval_max": 11,
            "max_small_interval_ratio_le4": 0.1714,
            "no_overlap": True,
        },
        "objective_next_decision_boundary": {
            "boundary": OBJECTIVE_NEXT_BOUNDARY,
            "source_boundary": "stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_listening_review",
            "repair_evidence_boundary": "stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_repair",
            "objective_only_decision_completed": True,
            "validated_review_input_present": False,
            "preference_fill_allowed": False,
            "selected_next_boundary": BOUNDARY,
            "objective_clean_candidate_boundary_supported": objective_clean,
            "additional_repair_required": not objective_clean,
        },
        "readiness": {
            "boundary": OBJECTIVE_NEXT_BOUNDARY,
            "objective_only_decision_completed": True,
            "objective_clean_candidate_boundary_supported": objective_clean,
            "additional_repair_required": not objective_clean,
            "validated_review_input_present": False,
            "preference_fill_allowed": False,
            "human_audio_preference_claimed": False,
            "model_direct_generation_quality_claimed": quality_claim,
            "midi_to_solo_musical_quality_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
        },
        "decision": {
            "current_boundary": OBJECTIVE_NEXT_BOUNDARY,
            "next_boundary": BOUNDARY,
            "auto_progress_allowed": True,
            "critical_user_input_required": False,
        },
    }


class StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeRepeatabilitySweepTest(
    unittest.TestCase
):
    def test_generates_distinct_objective_clean_sweep_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            report = build_repeatability_sweep_report(
                objective_next_report(),
                output_dir=Path(temp_dir),
                sample_count=6,
                max_interval=12,
            )
        summary = validate_repeatability_sweep_report(
            report,
            expected_boundary=BOUNDARY,
            expected_next_boundary=NEXT_BOUNDARY,
            min_sample_count=6,
            min_qualified_count=6,
            require_repeatability_passed=True,
            require_no_quality_claim=True,
        )

        self.assertEqual(summary["sample_count"], 6)
        self.assertEqual(summary["generated_midi_file_count"], 6)
        self.assertEqual(summary["qualified_candidate_count"], 6)
        self.assertEqual(summary["objective_clean_pass_rate"], 1.0)
        self.assertEqual(summary["current_analysis_flag_count"], 0)
        self.assertEqual(summary["overlap_detected_count"], 0)
        self.assertGreaterEqual(summary["distinct_density_pattern_count"], 6)
        self.assertLessEqual(summary["max_abs_interval_max"], 12)
        self.assertTrue(summary["repeatability_passed"])
        self.assertFalse(summary["human_audio_preference_claimed"])
        self.assertFalse(summary["midi_to_solo_musical_quality_claimed"])
        self.assertEqual(summary["next_boundary"], NEXT_BOUNDARY)

    def test_rejects_non_objective_clean_source(self) -> None:
        with self.assertRaises(
            StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeRepeatabilitySweepError
        ):
            build_repeatability_sweep_report(
                objective_next_report(objective_clean=False),
                output_dir=Path("outputs/repeatability_sweep"),
                sample_count=6,
                max_interval=12,
            )

    def test_rejects_upstream_quality_claim(self) -> None:
        with self.assertRaises(
            StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeRepeatabilitySweepError
        ):
            build_repeatability_sweep_report(
                objective_next_report(quality_claim=True),
                output_dir=Path("outputs/repeatability_sweep"),
                sample_count=6,
                max_interval=12,
            )

    def test_rejects_sample_count_beyond_configured_variants(self) -> None:
        with self.assertRaises(
            StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeRepeatabilitySweepError
        ):
            build_repeatability_sweep_report(
                objective_next_report(),
                output_dir=Path("outputs/repeatability_sweep"),
                sample_count=7,
                max_interval=12,
            )


if __name__ == "__main__":
    unittest.main()
