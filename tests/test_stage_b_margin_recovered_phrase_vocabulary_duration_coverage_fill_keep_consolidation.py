from __future__ import annotations

import unittest
from pathlib import Path

from scripts.summarize_stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_keep_consolidation import (
    DurationCoverageFillKeepConsolidationError,
    build_keep_consolidation_report,
    validate_keep_consolidation,
)


def filled_notes(*, decision: str = "keep") -> dict:
    return {
        "schema_version": "stage_b_margin_recovered_phrase_vocabulary_focused_listening_fill_v1",
        "candidates": [
            {
                "candidate_id": "duration_fill_candidate",
                "review_metadata": {
                    "mode": "margin_recovered_phrase_vocabulary_duration_coverage_fill_repair",
                },
                "review_files": {
                    "midi_path": "outputs/solo.mid",
                    "context_midi_path": "outputs/context.mid",
                    "source_midi_path": "outputs/source.mid",
                },
                "listening": {
                    "status": "reviewed",
                    "timing": "acceptable",
                    "chord_fit": "strong",
                    "phrase_continuation": "acceptable",
                    "landing": "strong",
                    "jazz_vocabulary": "acceptable",
                    "decision": decision,
                },
                "focused_context_metrics": {
                    "note_count": 18,
                    "unique_pitch_count": 15,
                    "range": "D#4-G#5",
                    "phrase_span_beats": 7.0,
                    "dead_air_ratio": 0.29411764705882354,
                    "onset_coverage_ratio": 0.5625,
                    "sustained_coverage_ratio": 0.625,
                    "adjacent_pitch_repeats": 0,
                    "duplicated_3_note_pitch_class_chunks": 0,
                    "max_simultaneous_notes": 1,
                    "max_interval": 7,
                    "final_note": "F4",
                    "final_chord": "Fm7",
                    "final_note_role": "chord_tone",
                },
                "review_risks": [],
                "listening_fill_evidence": {
                    "not_human_audio_review": True,
                },
            }
        ],
    }


def duration_fill_summary() -> dict:
    return {
        "schema_version": "stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_repair_v1",
        "variant_count": 4,
        "qualified_variant_count": 2,
        "source_candidate": {
            "candidate_id": "source_candidate",
            "metrics": {
                "note_count": 15,
                "dead_air_ratio": 0.5714285714285714,
            },
            "focused_solo_metrics": {
                "focused_note_count": 12,
                "focused_unique_pitch_count": 9,
            },
        },
        "selected_candidate": {
            "candidate_id": "duration_fill_candidate",
            "metrics": {
                "note_count": 18,
                "dead_air_ratio": 0.29411764705882354,
            },
            "focused_solo_metrics": {
                "focused_note_count": 18,
                "focused_unique_pitch_count": 15,
                "focused_adjacent_pitch_repeats": 0,
                "focused_duplicated_3_note_pitch_class_chunks": 0,
                "focused_max_interval": 7,
            },
            "fill_repair": {
                "max_additions": 6,
                "fill_addition_count": 6,
            },
            "duration_coverage_gate": {
                "qualified": True,
                "flags": [],
            },
        },
        "repair_summary": {
            "selected_candidate_id": "duration_fill_candidate",
            "selected_fill_addition_count": 6,
            "qualified": True,
            "remaining_flags": [],
            "baseline_dead_air_ratio": 0.5714285714285714,
            "selected_dead_air_ratio": 0.29411764705882354,
            "dead_air_delta_from_baseline": 0.277311,
            "selected_adjacent_pitch_repeats": 0,
            "selected_duplicated_3_note_pitch_class_chunks": 0,
            "selected_max_interval": 7,
            "duration_coverage_fill_improved": True,
            "claim_boundary": "postprocess_duration_coverage_fill_candidate",
        },
    }


class StageBMarginRecoveredPhraseVocabularyDurationCoverageFillKeepConsolidationTest(unittest.TestCase):
    def test_builds_single_postprocess_keep_boundary(self) -> None:
        report = build_keep_consolidation_report(
            filled_notes(),
            duration_fill_summary(),
            output_dir=Path("outputs/keep_consolidation"),
        )
        summary = validate_keep_consolidation(
            report,
            expected_candidate_id="duration_fill_candidate",
            expected_boundary="single_postprocess_candidate_keep_support",
            require_not_human_audio_review=True,
            require_postprocess_claim_boundary="postprocess_duration_coverage_fill_candidate",
        )

        self.assertEqual(summary["decision"], "keep")
        self.assertEqual(summary["boundary"], "single_postprocess_candidate_keep_support")
        self.assertEqual(summary["fill_addition_count"], 6)
        self.assertEqual(summary["note_count"], 18)
        self.assertEqual(summary["unique_pitch_count"], 15)
        self.assertEqual(summary["adjacent_pitch_repeats"], 0)
        self.assertEqual(summary["max_interval"], 7)
        self.assertIn("human_audio_preference", report["not_proven"])

    def test_rejects_non_matching_duration_candidate(self) -> None:
        summary = duration_fill_summary()
        summary["repair_summary"]["selected_candidate_id"] = "other_candidate"

        with self.assertRaises(DurationCoverageFillKeepConsolidationError):
            build_keep_consolidation_report(
                filled_notes(),
                summary,
                output_dir=Path("outputs/keep_consolidation"),
            )

    def test_requires_keep_decision(self) -> None:
        with self.assertRaises(DurationCoverageFillKeepConsolidationError):
            build_keep_consolidation_report(
                filled_notes(decision="needs_followup"),
                duration_fill_summary(),
                output_dir=Path("outputs/keep_consolidation"),
            )


if __name__ == "__main__":
    unittest.main()
