from __future__ import annotations

import unittest
from pathlib import Path

from scripts.summarize_stage_b_margin_recovered_phrase_vocabulary_distinct_sample_seed_sweep import (
    build_distinct_sample_seed_sweep_report,
    validate_distinct_sample_seed_sweep,
)


def candidate(candidate_id: str, *, sample_seed: int, qualified: bool) -> dict:
    return {
        "candidate_id": candidate_id,
        "source_run_id": "run",
        "sample_index": 1,
        "sample_seed": sample_seed,
        "repair_rank": 1,
        "source_request": {"seed": sample_seed},
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
        "phrase_vocabulary_gate": {"qualified": qualified, "flags": [] if qualified else ["low_pitch_variety"]},
    }


class StageBMarginRecoveredPhraseVocabularyDistinctSampleSeedSweepTest(unittest.TestCase):
    def test_finds_distinct_sample_seed_candidate(self) -> None:
        repair_summary = {
            "candidate_count": 2,
            "candidates": [
                candidate("blocked", sample_seed=85, qualified=True),
                candidate("distinct", sample_seed=109, qualified=True),
            ],
        }
        sample_seed_repair = {"duplicate_sample_seed_counts": {"85": 2}}

        report = build_distinct_sample_seed_sweep_report(
            repair_summary,
            sample_seed_repair,
            output_dir=Path("outputs/distinct"),
        )
        summary = validate_distinct_sample_seed_sweep(
            report,
            min_candidates=2,
            expected_blocked_seed=85,
        )

        self.assertEqual(summary["distinct_sample_seed_qualified_count"], 1)
        self.assertEqual(summary["boundary"], "distinct_sample_seed_qualified_candidate_found")
        self.assertEqual(report["selected_distinct_candidate"]["candidate_id"], "distinct")

    def test_records_absent_distinct_candidate(self) -> None:
        repair_summary = {
            "candidate_count": 1,
            "candidates": [candidate("blocked", sample_seed=85, qualified=True)],
        }
        sample_seed_repair = {"duplicate_sample_seed_counts": {"85": 2}}

        report = build_distinct_sample_seed_sweep_report(
            repair_summary,
            sample_seed_repair,
            output_dir=Path("outputs/distinct"),
        )

        self.assertEqual(report["sweep_boundary"]["boundary"], "no_distinct_sample_seed_qualified_candidate")


if __name__ == "__main__":
    unittest.main()
