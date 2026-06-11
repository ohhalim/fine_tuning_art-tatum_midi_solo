from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.build_music_transformer_solo_yield_objective_quality_rubric_baseline import (
    SoloYieldObjectiveQualityRubricBaselineError,
    build_rubric_report,
    label_candidate,
    validate_rubric_report,
)


def candidate(
    *,
    review_index: int = 1,
    case_label: str = "minor_backdoor",
    rank: int = 1,
    note_count: int = 30,
    dead_air_ratio: float = 0.72,
    direction_change_ratio: float = 0.62,
    syncopated_onset_ratio: float = 0.82,
    chord_tone_ratio: float = 0.44,
    tension_ratio: float = 0.24,
) -> dict:
    return {
        "review_index": review_index,
        "case_label": case_label,
        "rank": rank,
        "score": 233.5,
        "note_count": note_count,
        "dead_air_ratio": dead_air_ratio,
        "direction_change_ratio": direction_change_ratio,
        "syncopated_onset_ratio": syncopated_onset_ratio,
        "chord_tone_ratio": chord_tone_ratio,
        "tension_ratio": tension_ratio,
        "review_midi_path": f"midi/candidate_{review_index}.mid",
        "review_wav_path": f"audio/candidate_{review_index}.wav",
    }


def listening_package(*, quality_claim: bool = False) -> dict:
    return {
        "schema_version": "music_transformer_solo_yield_listening_package_v1",
        "output_dir": "outputs/listening_package",
        "candidate_count": 8,
        "candidates": [
            candidate(review_index=1, dead_air_ratio=0.72),
            candidate(review_index=2, dead_air_ratio=0.69),
            candidate(review_index=3, dead_air_ratio=0.65, direction_change_ratio=0.44),
            candidate(review_index=4, dead_air_ratio=0.64, tension_ratio=0.16),
            candidate(review_index=5, dead_air_ratio=0.66),
            candidate(review_index=6, dead_air_ratio=0.63),
            candidate(review_index=7, dead_air_ratio=0.62),
            candidate(review_index=8, dead_air_ratio=0.61),
        ],
        "readiness": {
            "musical_quality_claimed": quality_claim,
            "artist_style_claimed": False,
            "production_ready_claimed": False,
        },
    }


class MusicTransformerSoloYieldObjectiveQualityRubricBaselineTest(unittest.TestCase):
    def test_label_candidate_marks_major_and_watch_labels(self) -> None:
        labeled = label_candidate(
            candidate(
                note_count=27,
                dead_air_ratio=0.67,
                direction_change_ratio=0.45,
                tension_ratio=0.37,
            )
        )

        self.assertIn("low_note_density", labeled["major_labels"])
        self.assertIn("weak_direction_change", labeled["major_labels"])
        self.assertIn("dead_air_watch", labeled["watch_labels"])
        self.assertIn("tension_high_watch", labeled["watch_labels"])
        self.assertFalse(labeled["quality_proxy_pass"])

    def test_builds_rubric_and_selects_dead_air_target(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            report = build_rubric_report(
                listening_package=listening_package(),
                output_dir=Path(temp_dir) / "rubric",
            )
        summary = validate_rubric_report(
            report,
            min_candidate_count=8,
            expected_target="dead_air_density_balance_repair",
            require_no_quality_claim=True,
        )

        self.assertEqual(summary["candidate_count"], 8)
        self.assertEqual(summary["quality_proxy_fail_count"], 4)
        self.assertEqual(summary["major_label_counts"]["dead_air_high"], 2)
        self.assertEqual(summary["major_label_counts"]["weak_direction_change"], 1)
        self.assertEqual(summary["major_label_counts"]["low_tension_color"], 1)
        self.assertEqual(summary["selected_repair_target"], "dead_air_density_balance_repair")
        self.assertFalse(summary["critical_user_input_required"])
        self.assertFalse(summary["musical_quality_claimed"])

    def test_rejects_quality_claim_in_source_package(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(SoloYieldObjectiveQualityRubricBaselineError):
                build_rubric_report(
                    listening_package=listening_package(quality_claim=True),
                    output_dir=Path(temp_dir) / "rubric",
                )

    def test_validation_rejects_target_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            report = build_rubric_report(
                listening_package=listening_package(),
                output_dir=Path(temp_dir) / "rubric",
            )

        with self.assertRaises(SoloYieldObjectiveQualityRubricBaselineError):
            validate_rubric_report(
                report,
                min_candidate_count=8,
                expected_target="phrase_direction_balance_repair",
                require_no_quality_claim=True,
            )


if __name__ == "__main__":
    unittest.main()
