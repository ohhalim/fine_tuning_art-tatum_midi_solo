from __future__ import annotations

import unittest

from scripts.fill_stage_b_margin_recovered_proxy_review import fill_proxy_review, markdown_report, proxy_score
from scripts.build_stage_b_margin_recovered_listening_notes import build_listening_notes, validate_listening_notes


def export_candidate(
    rank: int,
    *,
    selected: bool,
    seed: int,
    sample: int,
    dead_air: float,
    notes: int,
    phrase: float,
    seed_strict: int | None = None,
    seed_outliers: int | None = None,
) -> dict:
    rich = phrase > 0.8
    return {
        "review_rank": rank,
        "is_selected_best": selected,
        "seed": seed,
        "sample_index": sample,
        "midi_path": f"seed_{seed}_sample_{sample}.mid",
        "dead_air_ratio": dead_air,
        "note_count": notes,
        "unique_pitch_count": 4,
        "phrase_coverage_ratio": phrase,
        "onset_coverage_ratio": 0.5 if phrase > 0.8 else 0.3125,
        "sustained_coverage_ratio": 0.72 if phrase > 0.8 else 0.4375,
        "postprocess_removal_ratio": 0.1 if rich else 0.35,
        "seed_strict_valid_sample_count": seed_strict if seed_strict is not None else (5 if rich else 4),
        "seed_sample_count": 5,
        "seed_dead_air_outlier_count": seed_outliers if seed_outliers is not None else (0 if rich else 1),
    }


class StageBMarginRecoveredProxyReviewTest(unittest.TestCase):
    def test_proxy_score_prefers_richer_candidate_over_dead_air_only_candidate(self) -> None:
        selected_dead_air = build_listening_notes(
            {
                "source_summary_path": "summary.json",
                "source_run_id": "run",
                "candidates": [
                    export_candidate(1, selected=True, seed=23, sample=1, dead_air=0.375, notes=9, phrase=0.437),
                    export_candidate(2, selected=False, seed=31, sample=5, dead_air=0.444, notes=19, phrase=0.937),
                ],
            }
        )["candidates"]

        self.assertGreater(proxy_score(selected_dead_air[1]), proxy_score(selected_dead_air[0]))

    def test_fill_proxy_review_marks_one_keep_and_reviews_all(self) -> None:
        notes = build_listening_notes(
            {
                "source_summary_path": "summary.json",
                "source_run_id": "run",
                "candidates": [
                    export_candidate(1, selected=True, seed=23, sample=1, dead_air=0.375, notes=9, phrase=0.437),
                    export_candidate(2, selected=False, seed=31, sample=5, dead_air=0.444, notes=19, phrase=0.937),
                    export_candidate(
                        3,
                        selected=False,
                        seed=17,
                        sample=3,
                        dead_air=0.5,
                        notes=17,
                        phrase=1.0,
                        seed_strict=3,
                        seed_outliers=1,
                    ),
                ],
            }
        )

        filled = fill_proxy_review(notes)
        summary = validate_listening_notes(filled)

        self.assertEqual(summary["reviewed_count"], 3)
        self.assertEqual(summary["decision_counts"]["keep"], 1)
        self.assertEqual(filled["proxy_review_summary"]["keep_candidate_id"], "margin_recovered_rank_2_seed_31_sample_5")
        self.assertEqual(filled["candidates"][0]["listening"]["decision"], "needs_followup")
        self.assertIn("not_human_listening", filled["candidates"][1]["proxy_review"])

    def test_markdown_report_lists_proxy_review_decisions(self) -> None:
        notes = build_listening_notes(
            {
                "source_summary_path": "summary.json",
                "source_run_id": "run",
                "candidates": [
                    export_candidate(1, selected=True, seed=23, sample=1, dead_air=0.375, notes=9, phrase=0.437),
                    export_candidate(2, selected=False, seed=31, sample=5, dead_air=0.444, notes=19, phrase=0.937),
                ],
            }
        )
        filled = fill_proxy_review(notes)
        summary = validate_listening_notes(filled)

        markdown = markdown_report(filled, summary)

        self.assertIn("human listening proof: `false`", markdown)
        self.assertIn("margin_recovered_rank_2_seed_31_sample_5", markdown)
        self.assertIn("keep", markdown)


if __name__ == "__main__":
    unittest.main()
