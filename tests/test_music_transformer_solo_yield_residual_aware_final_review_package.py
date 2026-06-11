from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.build_music_transformer_solo_yield_residual_aware_final_review_package import (
    SoloYieldResidualAwareFinalReviewPackageError,
    build_final_review_package,
    validate_final_review_package,
)
from scripts.render_stage_b_midi_to_solo_candidate_audio import sha256_file


def write_file(path: Path, content: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return sha256_file(path)


def source_package(temp_dir: Path, *, quality_claim: bool = False) -> dict:
    midi_path = temp_dir / "midi" / "candidate_01.mid"
    wav_path = temp_dir / "audio" / "candidate_01.wav"
    midi_sha = write_file(midi_path, b"midi")
    wav_sha = write_file(wav_path, b"wav")
    return {
        "schema_version": "music_transformer_solo_yield_rhythm_syncopation_balance_repair_package_v1",
        "output_dir": "outputs/source_package",
        "selected_candidates": [
            {
                "repair_index": 1,
                "case_label": "minor_backdoor",
                "sample_index": 9,
                "sample_seed": 1009,
                "repair_midi_path": str(midi_path),
                "repair_midi_sha256": midi_sha,
                "note_count": 32,
                "dead_air_ratio": 0.64,
                "direction_change_ratio": 0.50,
                "syncopated_onset_ratio": 0.86,
                "chord_tone_ratio": 0.44,
                "tension_ratio": 0.16,
            }
        ],
        "rendered_audio_files": [
            {
                "repair_index": 1,
                "wav_file": {
                    "path": str(wav_path),
                    "sha256": wav_sha,
                    "duration_seconds": 1.25,
                    "sample_rate": 44100,
                },
            }
        ],
        "readiness": {
            "musical_quality_claimed": quality_claim,
            "artist_style_claimed": False,
            "production_ready_claimed": False,
        },
    }


def rubric_report() -> dict:
    return {
        "schema_version": "music_transformer_solo_yield_objective_quality_rubric_baseline_v1",
        "output_dir": "outputs/rubric",
        "candidate_labels": [
            {
                "review_index": 1,
                "major_labels": ["low_tension_color"],
                "watch_labels": [],
                "quality_proxy_pass": False,
            }
        ],
        "aggregate": {
            "candidate_count": 1,
            "quality_proxy_pass_count": 0,
            "quality_proxy_fail_count": 1,
            "major_label_counts": {"low_tension_color": 1},
            "watch_label_counts": {},
        },
        "decision": {
            "selected_repair_target": "tension_color_balance_repair",
        },
        "readiness": {
            "musical_quality_claimed": False,
            "artist_style_claimed": False,
            "production_ready_claimed": False,
        },
    }


def feasibility_decision() -> dict:
    return {
        "schema_version": "music_transformer_solo_yield_residual_tension_target_decision_v1",
        "output_dir": "outputs/decision",
        "decision": {
            "tension_repeat_feasible": False,
            "tension_repeat_blocked_cases": ["minor_backdoor"],
        },
        "readiness": {
            "musical_quality_claimed": False,
            "artist_style_claimed": False,
            "production_ready_claimed": False,
        },
    }


class MusicTransformerSoloYieldResidualAwareFinalReviewPackageTest(unittest.TestCase):
    def test_builds_final_review_package_with_residual_context(self) -> None:
        with tempfile.TemporaryDirectory() as raw_temp:
            temp_dir = Path(raw_temp)
            report = build_final_review_package(
                source_package=source_package(temp_dir),
                rubric_report=rubric_report(),
                feasibility_decision=feasibility_decision(),
                output_dir=temp_dir / "final",
            )
        summary = validate_final_review_package(
            report,
            min_candidate_count=1,
            require_residual_context=True,
            require_no_quality_claim=True,
        )

        self.assertEqual(summary["candidate_count"], 1)
        self.assertEqual(summary["midi_count"], 1)
        self.assertEqual(summary["wav_count"], 1)
        self.assertEqual(summary["major_label_counts"]["low_tension_color"], 1)
        self.assertFalse(summary["tension_repeat_feasible"])
        self.assertFalse(summary["validated_listening_input_present"])
        self.assertFalse(summary["musical_quality_claimed"])
        self.assertEqual(
            report["review_input_template"]["schema_version"],
            "music_transformer_solo_yield_residual_aware_review_input_v1",
        )

    def test_rejects_checksum_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as raw_temp:
            temp_dir = Path(raw_temp)
            package = source_package(temp_dir)
            package["selected_candidates"][0]["repair_midi_sha256"] = "bad"

            with self.assertRaises(SoloYieldResidualAwareFinalReviewPackageError):
                build_final_review_package(
                    source_package=package,
                    rubric_report=rubric_report(),
                    feasibility_decision=feasibility_decision(),
                    output_dir=temp_dir / "final",
                )

    def test_rejects_quality_claim(self) -> None:
        with tempfile.TemporaryDirectory() as raw_temp:
            temp_dir = Path(raw_temp)

            with self.assertRaises(SoloYieldResidualAwareFinalReviewPackageError):
                build_final_review_package(
                    source_package=source_package(temp_dir, quality_claim=True),
                    rubric_report=rubric_report(),
                    feasibility_decision=feasibility_decision(),
                    output_dir=temp_dir / "final",
                )


if __name__ == "__main__":
    unittest.main()
