from __future__ import annotations

import unittest
from pathlib import Path

from scripts.audit_stage_b_margin_recovered_phrase_vocabulary_duplicate_source_divergence import (
    build_duplicate_source_divergence_audit,
    validate_duplicate_source_divergence,
)


def repair_candidate(candidate_id: str, *, source_seed: int, sample_index: int, sample_seed: int) -> dict:
    return {
        "candidate_id": candidate_id,
        "source_run_id": f"run_{source_seed}",
        "sample_index": sample_index,
        "sample_seed": sample_seed,
        "repair_rank": 1,
        "source_request": {"seed": source_seed, "top_k": 7, "temperature": 0.82, "sample_count": 48},
        "metrics": {
            "note_count": 16,
            "unique_pitch_count": 8,
            "dead_air_ratio": 0.3333333333333333,
            "phrase_coverage_ratio": 0.875,
        },
        "focused_solo_metrics": {
            "focused_note_count": 13,
            "focused_unique_pitch_count": 8,
            "focused_adjacent_pitch_repeats": 0,
            "focused_max_interval": 7,
            "focused_postprocess_removed_note_count": 3,
        },
        "phrase_vocabulary_gate": {"qualified": True, "flags": []},
    }


class StageBMarginRecoveredPhraseVocabularyDuplicateSourceDivergenceTest(unittest.TestCase):
    def test_audits_shared_sample_seed_duplicate_output(self) -> None:
        repair_summary = {
            "candidate_count": 96,
            "qualified_candidate_count": 2,
            "candidates": [
                repair_candidate("selected", source_seed=43, sample_index=43, sample_seed=85),
                repair_candidate("peer", source_seed=61, sample_index=25, sample_seed=85),
            ],
        }
        human_comparison = {
            "objective_comparison": {"note_sequence_match": True},
            "candidates": [
                {"role": "selected", "candidate_id": "selected"},
                {"role": "peer", "candidate_id": "peer"},
            ],
        }

        report = build_duplicate_source_divergence_audit(
            repair_summary,
            human_comparison,
            output_dir=Path("outputs/divergence"),
        )
        summary = validate_duplicate_source_divergence(
            report,
            require_shared_sample_seed=True,
            require_duplicate_output=True,
            expected_boundary="shared_sample_seed_duplicate_output",
        )

        self.assertTrue(summary["source_seed_diff"])
        self.assertTrue(summary["sample_index_diff"])
        self.assertTrue(summary["shared_sample_seed"])
        self.assertEqual(summary["claim_boundary"], "two_source_qualified_but_not_two_distinct_outputs")


if __name__ == "__main__":
    unittest.main()
