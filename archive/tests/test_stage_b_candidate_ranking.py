from __future__ import annotations

import unittest

from scripts.rank_stage_b_candidates import (
    build_summary,
    review_flags_for_diagnostics,
    rank_candidates,
    score_candidate,
)


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

    def test_score_candidate_penalizes_low_bar_chord_and_repeated_pitch(self) -> None:
        repetitive = {
            "valid": True,
            "strict_valid": True,
            "grammar_gate_passed": True,
            "metrics": {
                "dead_air_ratio": 0.45,
                "repetition_score": 0.1,
                "chord_tone_ratio": 0.31,
                "unique_pitch_count": 3,
            },
            "temporal_coverage": {
                "onset_coverage_ratio": 0.5,
                "sustained_coverage_ratio": 0.9,
                "position_span_ratio": 0.9,
            },
            "collapse": {"collapse_warning": False, "postprocess_removal_ratio": 0.0},
            "harmonic_diagnostics": {
                "bar_chord_tone_ratio": 0.18,
                "min_bar_chord_tone_ratio": 0.0,
                "dominant_pitch_ratio": 0.56,
                "repeated_pitch_ratio": 0.81,
                "onset_template_repetition_ratio": 1.0,
            },
        }
        balanced = {
            "valid": True,
            "strict_valid": True,
            "grammar_gate_passed": True,
            "metrics": {
                "dead_air_ratio": 0.45,
                "repetition_score": 0.1,
                "chord_tone_ratio": 0.5,
                "unique_pitch_count": 5,
            },
            "temporal_coverage": {
                "onset_coverage_ratio": 0.38,
                "sustained_coverage_ratio": 0.7,
                "position_span_ratio": 0.9,
            },
            "collapse": {"collapse_warning": False, "postprocess_removal_ratio": 0.0},
            "harmonic_diagnostics": {
                "bar_chord_tone_ratio": 0.5,
                "min_bar_chord_tone_ratio": 0.35,
                "dominant_pitch_ratio": 0.35,
                "repeated_pitch_ratio": 0.45,
                "onset_template_repetition_ratio": 0.0,
            },
        }

        self.assertLess(score_candidate(repetitive)["score"], score_candidate(balanced)["score"])

    def test_review_flags_mark_harmonic_and_template_failures(self) -> None:
        flags = review_flags_for_diagnostics(
            chord_tone_ratio=0.18,
            min_bar_chord_tone_ratio=0.0,
            dominant_pitch_ratio=0.56,
            repeated_pitch_ratio=0.81,
            onset_template_repetition_ratio=1.0,
        )

        self.assertIn("low_chord_tone_ratio", flags)
        self.assertIn("low_bar_chord_tone_ratio", flags)
        self.assertIn("dominant_pitch_repetition", flags)
        self.assertIn("low_pitch_variety", flags)
        self.assertIn("repeated_onset_template", flags)

    def test_build_summary_counts_strict_candidates(self) -> None:
        candidates = [
            {"valid": True, "strict_valid": True},
            {"valid": True, "strict_valid": False},
        ]
        top = [{"valid": True, "strict_valid": True, "reviewable": True}]

        summary = build_summary(candidates, top)

        self.assertEqual(summary["candidate_count"], 2)
        self.assertEqual(summary["valid_candidate_count"], 2)
        self.assertEqual(summary["strict_candidate_count"], 1)
        self.assertEqual(summary["top_strict_candidate_count"], 1)
        self.assertEqual(summary["top_viable_candidate_count"], 1)

    def test_rank_candidates_prefers_reviewable_candidate_over_flagged_score(self) -> None:
        candidates = [
            {
                "score": 30.0,
                "reviewable": False,
                "review_flags": ["low_chord_tone_ratio"],
                "strict_valid": True,
                "valid": True,
                "onset_coverage_ratio": 0.5,
                "bar_chord_tone_ratio": 0.1,
                "dead_air_ratio": 0.4,
            },
            {
                "score": 20.0,
                "reviewable": True,
                "review_flags": [],
                "strict_valid": True,
                "valid": True,
                "onset_coverage_ratio": 0.3,
                "bar_chord_tone_ratio": 0.5,
                "dead_air_ratio": 0.5,
            },
        ]

        ranked = rank_candidates(candidates, top_n=2)

        self.assertTrue(ranked[0]["reviewable"])


if __name__ == "__main__":
    unittest.main()
