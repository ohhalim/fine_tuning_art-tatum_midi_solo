from __future__ import annotations

import unittest
from pathlib import Path

from scripts.summarize_stage_b_margin_recovered_phrase_vocabulary_keep_stability import (
    build_stability_report,
    validate_stability,
)


def candidate(candidate_id: str, *, source: str, rank: int, qualified: bool) -> dict:
    return {
        "candidate_id": candidate_id,
        "source_run_id": source,
        "sample_index": rank + 10,
        "sample_seed": rank + 70,
        "repair_rank": rank,
        "metrics": {"dead_air_ratio": 0.33333333333333337},
        "focused_solo_metrics": {
            "focused_note_count": 13,
            "focused_unique_pitch_count": 8,
            "focused_adjacent_pitch_repeats": 0,
            "focused_max_interval": 7,
            "focused_duplicated_3_note_pitch_class_chunks": 0,
        },
        "phrase_vocabulary_gate": {"qualified": qualified, "flags": [] if qualified else ["low_pitch_variety"]},
    }


class StageBMarginRecoveredPhraseVocabularyKeepStabilityTest(unittest.TestCase):
    def test_builds_stability_report_with_qualified_peer(self) -> None:
        repair_summary = {
            "output_dir": "outputs/repair",
            "candidate_count": 3,
            "candidates": [
                candidate("selected", source="seed43", rank=1, qualified=True),
                candidate("peer", source="seed61", rank=2, qualified=True),
                candidate("rejected", source="seed43", rank=3, qualified=False),
            ],
        }
        filled_notes = {
            "schema_version": "stage_b_margin_recovered_phrase_vocabulary_focused_listening_fill_v1",
            "candidates": [
                {
                    "candidate_id": "selected",
                    "listening": {
                        "decision": "keep",
                        "timing": "acceptable",
                        "phrase_continuation": "acceptable",
                        "jazz_vocabulary": "acceptable",
                    },
                    "listening_fill_evidence": {"not_human_audio_review": True},
                }
            ],
        }
        report = build_stability_report(repair_summary, filled_notes, output_dir=Path("outputs/stability"))
        summary = validate_stability(
            report,
            expected_selected_candidate_id="selected",
            min_qualified_candidates=2,
            require_qualified_peer=True,
        )

        self.assertEqual(summary["qualified_candidate_count"], 2)
        self.assertEqual(summary["qualified_peer_count"], 1)
        self.assertEqual(summary["qualified_source_count"], 2)
        self.assertEqual(summary["boundary"], "narrow_two_source_candidate_support")
        self.assertTrue(report["stability_summary"]["has_qualified_peer"])


if __name__ == "__main__":
    unittest.main()
