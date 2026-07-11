from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.render_stage_b_duration_coverage_fill_repeatability_audio_review_package import (
    StageBDurationCoverageRepeatabilityAudioReviewPackageError,
    build_repeatability_audio_review_package,
    validate_repeatability_audio_review_package,
)
from tests.test_stage_b_local_audio_render_attempt import fake_runner, write_midi


def repeatability_consolidation(*, broad_claim: bool = False) -> dict:
    return {
        "schema_version": "stage_b_duration_coverage_fill_repeatability_consolidation_v1",
        "consolidated_claim_boundary": {
            "boundary": "current_keep_and_distinct_source_dead_air_gain_midi_support",
            "broad_model_quality_claimed": broad_claim,
        },
    }


def dead_air_gain_repair(root: Path) -> dict:
    first = root / "sample_155.mid"
    second = root / "sample_131.mid"
    write_midi(first, 67)
    write_midi(second, 71)
    return {
        "schema_version": "stage_b_duration_coverage_fill_dead_air_gain_repeatability_repair_v1",
        "repair_summary": {
            "boundary": "qualified_gate_repeatability_with_dead_air_gain",
        },
        "source_repeatability_results": [
            {
                "source_candidate_id": "source_155",
                "selected_candidate_id": "source_155_fill",
                "selected_midi_path": str(first),
                "sample_seed": 155,
                "baseline_dead_air_ratio": 0.375,
                "selected_dead_air_ratio": 0.3333333333333333,
                "dead_air_delta_from_baseline": 0.041667,
                "selected_focused_note_count": 19,
                "selected_focused_unique_pitch_count": 12,
                "selected_adjacent_pitch_repeats": 0,
                "selected_max_interval": 6,
            },
            {
                "source_candidate_id": "source_131",
                "selected_candidate_id": "source_131_fill",
                "selected_midi_path": str(second),
                "sample_seed": 131,
                "baseline_dead_air_ratio": 0.375,
                "selected_dead_air_ratio": 0.35294117647058826,
                "dead_air_delta_from_baseline": 0.022059,
                "selected_focused_note_count": 18,
                "selected_focused_unique_pitch_count": 13,
                "selected_adjacent_pitch_repeats": 0,
                "selected_max_interval": 11,
            },
        ],
    }


class StageBDurationCoverageFillRepeatabilityAudioReviewPackageTest(unittest.TestCase):
    def test_renders_repeatability_review_wavs_without_quality_claim(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            renderer = root / "fluidsynth"
            soundfont = root / "piano.sf2"
            renderer.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            soundfont.write_bytes(b"sf2")
            report = build_repeatability_audio_review_package(
                repeatability_consolidation=repeatability_consolidation(),
                dead_air_gain_repair=dead_air_gain_repair(root),
                output_dir=root / "audio_review",
                renderer_path=str(renderer),
                soundfont_path=str(soundfont),
                sample_rate=44100,
                runner=fake_runner,
            )
            summary = validate_repeatability_audio_review_package(
                report,
                expected_file_count=2,
                expected_sample_rate=44100,
                require_no_quality_claim=True,
            )

            self.assertEqual(summary["status"], "ready_for_user_listening_review")
            self.assertEqual(summary["rendered_audio_file_count"], 2)
            self.assertTrue(summary["technical_wav_validation"])
            self.assertFalse(summary["audio_rendered_quality_claimed"])
            self.assertFalse(summary["human_audio_preference_claimed"])
            self.assertEqual(len(summary["wav_paths"]), 2)

    def test_rejects_broad_quality_claim(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with self.assertRaises(StageBDurationCoverageRepeatabilityAudioReviewPackageError):
                build_repeatability_audio_review_package(
                    repeatability_consolidation=repeatability_consolidation(broad_claim=True),
                    dead_air_gain_repair=dead_air_gain_repair(root),
                    output_dir=root / "audio_review",
                    renderer_path=str(root / "fluidsynth"),
                    soundfont_path=str(root / "piano.sf2"),
                    sample_rate=44100,
                    runner=fake_runner,
                )


if __name__ == "__main__":
    unittest.main()
