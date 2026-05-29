from __future__ import annotations

import unittest
from pathlib import Path

from scripts.repair_stage_b_margin_recovered_phrase_vocabulary_sample_seed_diversity import (
    build_sample_seed_diversity_repair,
    validate_sample_seed_diversity_repair,
)


def qualified(candidate_id: str, *, source_seed: int, sample_seed: int, rank: int) -> dict:
    return {
        "candidate_id": candidate_id,
        "source_run_id": f"run_{source_seed}",
        "sample_index": rank + 20,
        "sample_seed": sample_seed,
        "repair_rank": rank,
        "source_request": {"seed": source_seed},
        "metrics": {
            "note_count": 16,
            "unique_pitch_count": 8,
            "dead_air_ratio": 0.3333333333333333,
        },
        "focused_solo_metrics": {
            "focused_note_count": 13,
            "focused_unique_pitch_count": 8,
            "focused_adjacent_pitch_repeats": 0,
            "focused_max_interval": 7,
        },
        "phrase_vocabulary_gate": {"qualified": True, "flags": []},
    }


class StageBMarginRecoveredPhraseVocabularySampleSeedDiversityTest(unittest.TestCase):
    def test_demotes_duplicate_sample_seed_peer(self) -> None:
        repair_summary = {
            "candidate_count": 96,
            "candidates": [
                qualified("selected", source_seed=43, sample_seed=85, rank=1),
                qualified("peer", source_seed=61, sample_seed=85, rank=2),
            ],
        }
        duplicate_audit = {
            "divergence_boundary": {
                "boundary": "shared_sample_seed_duplicate_output",
                "claim_boundary": "two_source_qualified_but_not_two_distinct_outputs",
            }
        }

        report = build_sample_seed_diversity_repair(
            repair_summary,
            duplicate_audit,
            output_dir=Path("outputs/sample_seed"),
        )
        summary = validate_sample_seed_diversity_repair(
            report,
            expected_boundary="single_distinct_sample_seed_keep_support",
            require_duplicate_demoted=True,
        )

        self.assertEqual(summary["qualified_candidate_count"], 2)
        self.assertEqual(summary["qualified_source_seed_count"], 2)
        self.assertEqual(summary["qualified_sample_seed_count"], 1)
        self.assertEqual(summary["distinct_peer_candidate_count"], 0)
        self.assertEqual(summary["claim_after"], "single_distinct_sample_seed_keep_support_until_new_sampling")


if __name__ == "__main__":
    unittest.main()
