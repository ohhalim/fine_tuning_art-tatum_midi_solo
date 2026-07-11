from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.run_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_repair import (
    BOUNDARY,
    NEXT_BOUNDARY,
    StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeRepairError,
    build_contour_phrase_shape_repair_report,
    validate_contour_phrase_shape_repair_report,
)


def objective_next(*, quality_claim: bool = False, next_boundary: str = BOUNDARY) -> dict:
    return {
        "schema_version": "stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_objective_next_v1",
        "boundary": "stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_objective_only_next_decision",
        "objective_next_decision_boundary": {
            "boundary": "stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_objective_only_next_decision",
            "source_boundary": "stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_listening_review",
        },
        "objective_summary": {
            "candidate_count": 3,
            "stepwise_contour_bias_count": 3,
            "distinct_density_pattern_count": 3,
            "max_abs_interval_max": 12,
        },
        "selected_repair_targets": [
            "reduce_stepwise_contour_bias",
            "add_phrase_shape_tension_release",
            "add_approach_enclosure_cells",
            "preserve_density_variation",
            "preserve_interval_guard",
            "preserve_no_quality_claim",
        ],
        "readiness": {
            "human_audio_preference_claimed": False,
            "model_direct_generation_quality_claimed": quality_claim,
            "midi_to_solo_musical_quality_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
        },
        "decision": {
            "next_boundary": next_boundary,
        },
    }


class StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeRepairTest(unittest.TestCase):
    def test_generates_contour_repaired_candidates_without_quality_claim(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            report = build_contour_phrase_shape_repair_report(
                objective_next(),
                output_dir=Path(temp_dir) / "repair",
            )
            summary = validate_contour_phrase_shape_repair_report(
                report,
                expected_boundary=BOUNDARY,
                expected_next_boundary=NEXT_BOUNDARY,
                require_repair_completed=True,
                require_target_passed=True,
                require_stepwise_reduced=True,
                require_no_quality_claim=True,
            )

        self.assertTrue(summary["target_passed"])
        self.assertEqual(summary["generated_midi_file_count"], 3)
        self.assertEqual(summary["source_stepwise_contour_bias_count"], 3)
        self.assertLess(summary["repaired_stepwise_contour_bias_count"], 3)
        self.assertTrue(summary["stepwise_contour_bias_reduced"])
        self.assertLess(summary["max_small_interval_ratio_le4"], 0.55)
        self.assertLessEqual(summary["max_abs_interval_max"], 12)
        self.assertLessEqual(summary["shared_rhythm_signature_count"], 1)
        self.assertTrue(summary["no_overlap"])
        self.assertEqual(summary["phrase_vocabulary_source"], "contour_phrase_shape_cells")
        self.assertFalse(summary["midi_to_solo_musical_quality_claimed"])
        self.assertEqual(summary["next_boundary"], NEXT_BOUNDARY)

    def test_rejects_upstream_quality_claim(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeRepairError):
                build_contour_phrase_shape_repair_report(
                    objective_next(quality_claim=True),
                    output_dir=Path(temp_dir) / "repair",
                )

    def test_rejects_wrong_next_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeRepairError):
                build_contour_phrase_shape_repair_report(
                    objective_next(next_boundary="other_boundary"),
                    output_dir=Path(temp_dir) / "repair",
                )


if __name__ == "__main__":
    unittest.main()
