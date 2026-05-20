from __future__ import annotations

import unittest

from scripts.rank_stage_b_candidates import build_summary, rank_candidates, score_candidate


class StageBCandidateRankingTest(unittest.TestCase):
    def test_score_candidate_rewards_strict_temporal_and_chord_tone_quality(self) -> None:
        weak = {
            "valid": False,
            "strict_valid": False,
            "grammar_gate_passed": True,
            "metrics": {
                "dead_air_ratio": 0.8,
                "repetition_score": 0.5,
                "chord_tone_ratio": 0.2,
                "unique_pitch_count": 2,
            },
            "temporal_coverage": {
                "onset_coverage_ratio": 0.1,
                "sustained_coverage_ratio": 0.3,
                "position_span_ratio": 0.5,
            },
            "collapse": {"collapse_warning": True, "postprocess_removal_ratio": 0.5},
        }
        strong = {
            "valid": True,
            "strict_valid": True,
            "grammar_gate_passed": True,
            "metrics": {
                "dead_air_ratio": 0.4,
                "repetition_score": 0.1,
                "chord_tone_ratio": 0.5,
                "unique_pitch_count": 4,
            },
            "temporal_coverage": {
                "onset_coverage_ratio": 0.5,
                "sustained_coverage_ratio": 0.8,
                "position_span_ratio": 0.9,
            },
            "collapse": {"collapse_warning": False, "postprocess_removal_ratio": 0.0},
        }

        self.assertGreater(score_candidate(strong)["score"], score_candidate(weak)["score"])

    def test_rank_candidates_orders_by_score(self) -> None:
        candidates = [
            {
                "score": 10.0,
                "strict_valid": False,
                "valid": True,
                "onset_coverage_ratio": 0.2,
                "chord_tone_ratio": 0.4,
                "dead_air_ratio": 0.7,
            },
            {
                "score": 20.0,
                "strict_valid": True,
                "valid": True,
                "onset_coverage_ratio": 0.5,
                "chord_tone_ratio": 0.3,
                "dead_air_ratio": 0.4,
            },
        ]

        ranked = rank_candidates(candidates, top_n=2)

        self.assertEqual(ranked[0]["score"], 20.0)
        self.assertEqual(ranked[0]["rank"], 1)

    def test_build_summary_counts_strict_candidates(self) -> None:
        candidates = [
            {"valid": True, "strict_valid": True},
            {"valid": True, "strict_valid": False},
        ]
        top = [{"valid": True, "strict_valid": True}]

        summary = build_summary(candidates, top)

        self.assertEqual(summary["candidate_count"], 2)
        self.assertEqual(summary["valid_candidate_count"], 2)
        self.assertEqual(summary["strict_candidate_count"], 1)
        self.assertEqual(summary["top_strict_candidate_count"], 1)


if __name__ == "__main__":
    unittest.main()
