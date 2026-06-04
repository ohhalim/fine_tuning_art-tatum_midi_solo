from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.run_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_probe import (
    BOUNDARY,
    NEXT_BOUNDARY,
    StageBMidiToSoloModelDirectJazzPhraseVocabularyRepairProbeError,
    build_jazz_phrase_vocabulary_repair_probe,
    validate_jazz_phrase_vocabulary_repair_probe,
)


def repair_decision(*, quality_claim: bool = False, next_boundary: str = BOUNDARY) -> dict:
    return {
        "schema_version": "stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_decision_v1",
        "boundary": "stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_decision",
        "repair_probe_requirements": {
            "candidate_count": 3,
            "target_bars": 8,
            "require_distinct_rhythm_signatures": True,
            "require_density_variation": True,
            "require_phrase_vocabulary_source_recorded": True,
            "max_allowed_interval": 12,
            "require_no_overlap": True,
            "require_no_quality_claim": True,
            "requires_audio_render_after_probe": True,
        },
        "readiness": {
            "human_audio_preference_claimed": False,
            "model_direct_candidate_keep_claimed": False,
            "model_direct_generation_quality_claimed": quality_claim,
            "midi_to_solo_musical_quality_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
        },
        "decision": {
            "next_boundary": next_boundary,
        },
    }


class StageBMidiToSoloModelDirectJazzPhraseVocabularyRepairProbeTest(unittest.TestCase):
    def test_generates_repaired_candidates_without_quality_claim(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            report = build_jazz_phrase_vocabulary_repair_probe(
                repair_decision(),
                output_dir=Path(temp_dir) / "probe",
            )
            summary = validate_jazz_phrase_vocabulary_repair_probe(
                report,
                expected_boundary=BOUNDARY,
                expected_next_boundary=NEXT_BOUNDARY,
                require_probe_completed=True,
                require_target_passed=True,
                require_no_quality_claim=True,
            )

        self.assertTrue(summary["target_passed"])
        self.assertEqual(summary["generated_midi_file_count"], 3)
        self.assertLessEqual(summary["uniform_bar_density_count"], 1)
        self.assertLessEqual(summary["shared_rhythm_signature_count"], 1)
        self.assertLess(summary["duration_template_monotony_count"], 3)
        self.assertLess(summary["ioi_template_monotony_count"], 3)
        self.assertLess(summary["safe_interval_cap_compression_count"], 3)
        self.assertLessEqual(summary["max_abs_interval_max"], 12)
        self.assertTrue(summary["no_overlap"])
        self.assertEqual(summary["phrase_vocabulary_source"], "repair_probe_data_guided_phrase_cells")
        self.assertFalse(summary["midi_to_solo_musical_quality_claimed"])

    def test_rejects_upstream_quality_claim(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(StageBMidiToSoloModelDirectJazzPhraseVocabularyRepairProbeError):
                build_jazz_phrase_vocabulary_repair_probe(
                    repair_decision(quality_claim=True),
                    output_dir=Path(temp_dir) / "probe",
                )

    def test_rejects_wrong_next_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(StageBMidiToSoloModelDirectJazzPhraseVocabularyRepairProbeError):
                build_jazz_phrase_vocabulary_repair_probe(
                    repair_decision(next_boundary="other_boundary"),
                    output_dir=Path(temp_dir) / "probe",
                )


if __name__ == "__main__":
    unittest.main()
