from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.decide_music_transformer_solo_yield_objective_next import (
    build_decision,
    validate_report,
)


def package_report() -> dict:
    return {
        "schema_version": "music_transformer_solo_yield_listening_package_v1",
        "output_dir": "outputs/package",
        "candidate_count": 3,
        "candidates": [
            {
                "review_index": 1,
                "case_label": "minor_backdoor",
                "score": 230.0,
                "note_count": 17,
                "dead_air_ratio": 0.5,
                "direction_change_ratio": 0.4,
                "syncopated_onset_ratio": 0.3,
                "review_wav_path": "outputs/package/audio/candidate_01.wav",
            },
            {
                "review_index": 2,
                "case_label": "dominant_cycle",
                "score": 232.0,
                "note_count": 18,
                "dead_air_ratio": 0.6,
                "direction_change_ratio": 0.5,
                "syncopated_onset_ratio": 0.2,
                "review_wav_path": "outputs/package/audio/candidate_02.wav",
            },
            {
                "review_index": 3,
                "case_label": "rhythm_turnaround",
                "score": 231.0,
                "note_count": 19,
                "dead_air_ratio": 0.4,
                "direction_change_ratio": 0.2,
                "syncopated_onset_ratio": 0.4,
                "review_wav_path": "outputs/package/audio/candidate_03.wav",
            },
        ],
    }


def guard_report(*, preference_fill_allowed: bool = False) -> dict:
    return {
        "schema_version": "music_transformer_solo_yield_listening_input_guard_v1",
        "output_dir": "outputs/guard",
        "input_validation": {
            "validated_listening_input_present": preference_fill_allowed,
        },
        "readiness": {
            "preference_fill_allowed": preference_fill_allowed,
        },
    }


class MusicTransformerSoloYieldObjectiveNextTest(unittest.TestCase):
    def test_pending_input_keeps_default_next_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            report = build_decision(
                package_report(),
                guard_report(),
                output_dir=Path(temp_dir) / "decision",
                top_n=2,
            )
        summary = validate_report(report, require_no_quality_claim=True)

        self.assertEqual(summary["candidate_count"], 3)
        self.assertEqual(summary["selected_objective_candidate_count"], 2)
        self.assertFalse(summary["validated_listening_input_present"])
        self.assertFalse(summary["preference_fill_allowed"])
        self.assertEqual(
            summary["next_boundary"],
            "music_transformer_solo_yield_larger_sample_repeatability_sweep",
        )

    def test_pending_input_accepts_explicit_next_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            report = build_decision(
                package_report(),
                guard_report(),
                output_dir=Path(temp_dir) / "decision",
                top_n=2,
                pending_next_boundary="music_transformer_solo_yield_4bar_phrase_expansion_probe",
                pending_reason="larger sample objective package complete; expand phrase length",
            )
        summary = validate_report(report, require_no_quality_claim=True)

        self.assertEqual(
            summary["next_boundary"],
            "music_transformer_solo_yield_4bar_phrase_expansion_probe",
        )
        self.assertEqual(
            report["decision"]["reason"],
            "larger sample objective package complete; expand phrase length",
        )

    def test_validated_input_routes_to_preference_fill(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            report = build_decision(
                package_report(),
                guard_report(preference_fill_allowed=True),
                output_dir=Path(temp_dir) / "decision",
                top_n=2,
                pending_next_boundary="music_transformer_solo_yield_4bar_phrase_expansion_probe",
            )
        summary = validate_report(report, require_no_quality_claim=True)

        self.assertTrue(summary["validated_listening_input_present"])
        self.assertTrue(summary["preference_fill_allowed"])
        self.assertEqual(summary["next_boundary"], "music_transformer_solo_yield_listening_review_fill")


if __name__ == "__main__":
    unittest.main()
