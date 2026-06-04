from __future__ import annotations

import unittest
from pathlib import Path

from scripts.consolidate_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_repeatability import (
    BOUNDARY,
    NEXT_BOUNDARY,
    StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeRepeatabilityConsolidationError,
    build_repeatability_consolidation_report,
    validate_repeatability_consolidation_report,
)
from scripts.run_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_repeatability_sweep import (
    BOUNDARY as SWEEP_BOUNDARY,
)


def repeatability_sweep(
    *,
    passed: bool = True,
    qualified_count: int = 6,
    quality_claim: bool = False,
    flag_count: int = 0,
) -> dict:
    return {
        "schema_version": "stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_repeatability_sweep_v1",
        "boundary": SWEEP_BOUNDARY,
        "source_boundary": "stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_objective_only_next_decision",
        "generated_candidates": [
            {
                "rank": rank,
                "seed": rank - 1,
                "midi_path": f"outputs/repeatability/midi/seed_{rank:02d}.mid",
                "density_pattern": [5, 3, 6, 4, 5, 4, 6, 3],
            }
            for rank in range(1, 7)
        ],
        "candidate_analyses": [
            {
                "rank": rank,
                "note_count": 36,
                "bar_count": 8,
                "analysis_flags": [] if flag_count == 0 else ["ioi_template_monotony"],
                "overlap_detected": False,
            }
            for rank in range(1, 7)
        ],
        "aggregate": {
            "candidate_count": 6,
            "qualified_candidate_count": qualified_count,
            "objective_clean_pass_rate": qualified_count / 6,
            "flag_counts": {} if flag_count == 0 else {"ioi_template_monotony": flag_count},
            "current_analysis_flag_count": flag_count,
            "overlap_detected_count": 0,
            "distinct_density_pattern_count": 6,
            "max_abs_interval_max": 12,
            "max_small_interval_ratio_le4": 0.1765,
        },
        "repeatability_result": {
            "repeatability_passed": passed,
            "sample_count": 6,
            "qualified_candidate_count": qualified_count,
            "objective_clean_pass_rate": qualified_count / 6,
            "max_interval": 12,
            "failure_next_boundary": "stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_repeatability_failure_repair_decision",
        },
        "readiness": {
            "boundary": SWEEP_BOUNDARY,
            "objective_clean_repeatability_sweep_completed": True,
            "repeatability_passed": passed,
            "generated_midi_file_count": 6,
            "human_audio_preference_claimed": False,
            "model_direct_generation_quality_claimed": quality_claim,
            "midi_to_solo_musical_quality_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
        },
        "decision": {
            "current_boundary": SWEEP_BOUNDARY,
            "next_boundary": BOUNDARY,
            "auto_progress_allowed": True,
            "critical_user_input_required": False,
        },
    }


class StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeRepeatabilityConsolidationTest(
    unittest.TestCase
):
    def test_consolidates_passed_repeatability_support(self) -> None:
        report = build_repeatability_consolidation_report(
            repeatability_sweep(),
            output_dir=Path("outputs/repeatability_consolidation"),
        )
        summary = validate_repeatability_consolidation_report(
            report,
            expected_boundary=BOUNDARY,
            expected_next_boundary=NEXT_BOUNDARY,
            min_sample_count=6,
            min_qualified_count=6,
            require_objective_support=True,
            require_audio_review_required=True,
            require_no_quality_claim=True,
        )

        self.assertEqual(summary["sample_count"], 6)
        self.assertEqual(summary["qualified_candidate_count"], 6)
        self.assertEqual(summary["objective_clean_pass_rate"], 1.0)
        self.assertEqual(summary["current_analysis_flag_count"], 0)
        self.assertEqual(summary["overlap_detected_count"], 0)
        self.assertTrue(summary["objective_repeatability_support"])
        self.assertFalse(summary["additional_repair_required"])
        self.assertTrue(summary["audio_review_package_required"])
        self.assertFalse(summary["human_audio_preference_claimed"])
        self.assertFalse(summary["midi_to_solo_musical_quality_claimed"])
        self.assertEqual(summary["next_boundary"], NEXT_BOUNDARY)

    def test_rejects_failed_repeatability_sweep(self) -> None:
        with self.assertRaises(
            StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeRepeatabilityConsolidationError
        ):
            build_repeatability_consolidation_report(
                repeatability_sweep(passed=False, qualified_count=5),
                output_dir=Path("outputs/repeatability_consolidation"),
            )

    def test_rejects_flagged_repeatability_sweep(self) -> None:
        with self.assertRaises(
            StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeRepeatabilityConsolidationError
        ):
            build_repeatability_consolidation_report(
                repeatability_sweep(flag_count=1),
                output_dir=Path("outputs/repeatability_consolidation"),
            )

    def test_rejects_upstream_quality_claim(self) -> None:
        with self.assertRaises(
            StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeRepeatabilityConsolidationError
        ):
            build_repeatability_consolidation_report(
                repeatability_sweep(quality_claim=True),
                output_dir=Path("outputs/repeatability_consolidation"),
            )


if __name__ == "__main__":
    unittest.main()
