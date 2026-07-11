from __future__ import annotations

import unittest
from pathlib import Path

from scripts.decide_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair import (
    BOUNDARY,
    NEXT_BOUNDARY,
    StageBMidiToSoloModelDirectJazzPhraseVocabularyRepairDecisionError,
    build_jazz_phrase_vocabulary_repair_decision,
    validate_jazz_phrase_vocabulary_repair_decision,
)


def songlike_analysis(*, quality_claim: bool = False, key_signals: bool = True) -> dict:
    return {
        "schema_version": "stage_b_midi_to_solo_model_direct_songlike_melody_rejection_analysis_v1",
        "boundary": "stage_b_midi_to_solo_model_direct_songlike_melody_rejection_analysis",
        "aggregate": {
            "candidate_count": 3,
            "uniform_bar_density_count": 3,
            "four_notes_per_bar_template_count": 3,
            "duration_template_monotony_count": 3,
            "ioi_template_monotony_count": 3,
            "safe_interval_cap_compression_count": 3,
            "four_bar_rhythm_cycle_repeated_count": 3,
            "shared_rhythm_signature_count": 3,
            "max_abs_interval_max": 9,
            "key_failure_signals": ["uniform_bar_density"] if key_signals else [],
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
            "next_boundary": "stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_decision",
        },
    }


class StageBMidiToSoloModelDirectJazzPhraseVocabularyRepairDecisionTest(unittest.TestCase):
    def test_selects_repair_probe_without_quality_claim(self) -> None:
        report = build_jazz_phrase_vocabulary_repair_decision(
            songlike_analysis(),
            output_dir=Path("outputs/repair_decision"),
        )
        summary = validate_jazz_phrase_vocabulary_repair_decision(
            report,
            expected_boundary=BOUNDARY,
            expected_next_boundary=NEXT_BOUNDARY,
            min_repair_target_count=6,
            require_auto_progress_allowed=True,
            require_no_quality_claim=True,
        )

        self.assertEqual(summary["repair_target_count"], 6)
        self.assertIn("break_uniform_bar_density", summary["repair_target_ids"])
        self.assertIn("replace_shared_rhythm_template", summary["repair_target_ids"])
        self.assertTrue(summary["require_distinct_rhythm_signatures"])
        self.assertEqual(summary["max_allowed_interval"], 12)
        self.assertTrue(summary["auto_progress_allowed"])
        self.assertFalse(summary["critical_user_input_required"])
        self.assertFalse(summary["human_audio_preference_claimed"])
        self.assertFalse(summary["midi_to_solo_musical_quality_claimed"])

    def test_rejects_missing_failure_signals(self) -> None:
        with self.assertRaises(StageBMidiToSoloModelDirectJazzPhraseVocabularyRepairDecisionError):
            build_jazz_phrase_vocabulary_repair_decision(
                songlike_analysis(key_signals=False),
                output_dir=Path("outputs/repair_decision"),
            )

    def test_rejects_upstream_quality_claim(self) -> None:
        with self.assertRaises(StageBMidiToSoloModelDirectJazzPhraseVocabularyRepairDecisionError):
            build_jazz_phrase_vocabulary_repair_decision(
                songlike_analysis(quality_claim=True),
                output_dir=Path("outputs/repair_decision"),
            )


if __name__ == "__main__":
    unittest.main()
