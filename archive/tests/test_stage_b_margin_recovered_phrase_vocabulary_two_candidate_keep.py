from __future__ import annotations

import unittest
from pathlib import Path

from scripts.summarize_stage_b_margin_recovered_phrase_vocabulary_two_candidate_keep import (
    PhraseVocabularyTwoCandidateKeepError,
    build_two_candidate_keep_report,
    validate_two_candidate_keep,
)


def filled_candidate(candidate_id: str, *, source: str, decision: str = "keep") -> dict:
    return {
        "schema_version": "stage_b_margin_recovered_phrase_vocabulary_focused_listening_fill_v1",
        "candidates": [
            {
                "candidate_id": candidate_id,
                "review_metadata": {
                    "source_run_id": source,
                    "sample_index": 25,
                    "sample_seed": 85,
                },
                "listening": {
                    "decision": decision,
                    "timing": "acceptable",
                    "chord_fit": "strong",
                    "phrase_continuation": "acceptable",
                    "landing": "strong",
                    "jazz_vocabulary": "acceptable",
                },
                "focused_context_metrics": {
                    "note_count": 13,
                    "unique_pitch_count": 8,
                    "range": "G4-E5",
                    "phrase_span_beats": 7.0,
                    "max_simultaneous_notes": 1,
                    "dead_air_ratio": 0.3333333333333333,
                    "sustained_coverage_ratio": 0.59375,
                    "adjacent_pitch_repeats": 0,
                    "max_interval": 7,
                    "final_note": "C5",
                    "final_chord": "Fm7",
                    "final_note_role": "chord_tone",
                },
                "review_risks": ["sustained_coverage_review"],
                "listening_fill_evidence": {"not_human_audio_review": True},
            }
        ],
    }


class StageBMarginRecoveredPhraseVocabularyTwoCandidateKeepTest(unittest.TestCase):
    def test_builds_two_candidate_keep_report(self) -> None:
        stability = {
            "schema_version": "stage_b_margin_recovered_phrase_vocabulary_keep_stability_v1",
            "candidate_count": 96,
            "qualified_candidate_count": 2,
            "qualified_rate": 0.020833,
            "qualified_source_count": 2,
            "selected_candidate": {"candidate_id": "selected"},
            "qualified_peers": [{"candidate_id": "peer"}],
        }

        report = build_two_candidate_keep_report(
            stability,
            filled_candidate("selected", source="seed43"),
            filled_candidate("peer", source="seed61"),
            output_dir=Path("outputs/two_candidate"),
        )
        summary = validate_two_candidate_keep(
            report,
            min_keep_candidates=2,
            min_qualified_sources=2,
            max_qualified_rate=0.05,
            require_not_human_audio_review=True,
        )

        self.assertEqual(summary["keep_candidate_count"], 2)
        self.assertEqual(summary["qualified_candidate_count"], 2)
        self.assertEqual(summary["qualified_source_count"], 2)
        self.assertEqual(summary["boundary"], "two_candidate_midi_context_keep_support")
        self.assertEqual(report["keep_candidates"][0]["role"], "selected")
        self.assertEqual(report["keep_candidates"][1]["role"], "peer")

    def test_rejects_peer_without_keep_decision(self) -> None:
        stability = {
            "selected_candidate": {"candidate_id": "selected"},
            "qualified_peers": [{"candidate_id": "peer"}],
        }

        with self.assertRaises(PhraseVocabularyTwoCandidateKeepError):
            build_two_candidate_keep_report(
                stability,
                filled_candidate("selected", source="seed43"),
                filled_candidate("peer", source="seed61", decision="needs_followup"),
                output_dir=Path("outputs/two_candidate"),
            )


if __name__ == "__main__":
    unittest.main()
