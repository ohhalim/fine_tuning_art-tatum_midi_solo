from __future__ import annotations

import unittest
from pathlib import Path

from scripts.summarize_stage_b_margin_recovered_phrase_vocabulary_distinct_sample_seed_remaining_blocker import (
    build_remaining_blocker_report,
    validate_remaining_blocker_report,
)


class StageBMarginRecoveredPhraseVocabularyDistinctSampleSeedRemainingBlockerTest(unittest.TestCase):
    def sample_filled_notes(self) -> dict:
        return {
            "output_dir": "outputs/fill",
            "candidates": [
                {
                    "candidate_id": "distinct_candidate",
                    "review_metadata": {
                        "source_run_id": "seed109_run",
                        "sample_index": 47,
                        "sample_seed": 155,
                    },
                    "proxy_review": {"decision": "keep_for_focused_listening"},
                    "focused_context_metrics": {
                        "note_count": 13,
                        "unique_pitch_count": 6,
                        "range": "A#4-D#5",
                        "phrase_span_beats": 6.75,
                        "dead_air_ratio": 0.375,
                        "onset_coverage_ratio": 0.5625,
                        "sustained_coverage_ratio": 0.78125,
                        "adjacent_pitch_repeats": 1,
                        "max_interval": 3,
                        "max_simultaneous_notes": 1,
                        "final_note": "D5",
                        "final_chord": "Fm7",
                        "final_note_role": "tension",
                    },
                    "review_risks": ["dead_air_ratio_remaining", "adjacent_pitch_repeats"],
                    "listening": {
                        "timing": "acceptable",
                        "chord_fit": "acceptable",
                        "phrase_continuation": "weak",
                        "landing": "acceptable",
                        "jazz_vocabulary": "thin",
                        "decision": "needs_followup",
                    },
                    "listening_fill_evidence": {
                        "not_human_audio_review": True,
                    },
                }
            ],
        }

    def test_builds_repair_target_from_needs_followup_fill(self) -> None:
        report = build_remaining_blocker_report(
            self.sample_filled_notes(),
            output_dir=Path("outputs/remaining"),
        )
        summary = validate_remaining_blocker_report(
            report,
            expected_decision="needs_followup",
            require_remaining_blockers=True,
        )

        self.assertEqual(summary["candidate_id"], "distinct_candidate")
        self.assertEqual(summary["sample_seed"], 155)
        self.assertEqual(summary["final_decision"], "needs_followup")
        self.assertIn("phrase_continuation_weak", summary["remaining_blockers"])
        self.assertIn("jazz_vocabulary_thin", summary["remaining_blockers"])
        self.assertIn("short_phrase_span", summary["remaining_blockers"])
        self.assertIn("pitch_variety_floor", summary["remaining_blockers"])
        self.assertIn("adjacent_pitch_repeats", summary["remaining_blockers"])
        self.assertIn("dead_air_ratio_remaining", summary["secondary_risks"])
        self.assertEqual(
            summary["repair_boundary"],
            "distinct_sample_seed_candidate_needs_phrase_vocabulary_repair",
        )
        self.assertTrue(report["keep_guardrails"]["wide_interval_guardrail"])
        self.assertEqual(report["repair_target"]["target_adjacent_pitch_repeats_max"], 0)


if __name__ == "__main__":
    unittest.main()
