from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.render_stage_b_duration_coverage_fill_outside_soloing_repair_audio_review_package import (
    StageBDurationCoverageOutsideSoloingRepairAudioReviewPackageError,
    build_outside_soloing_repair_audio_review_package,
    validate_outside_soloing_repair_audio_review_package,
)
from tests.test_stage_b_local_audio_render_attempt import fake_runner, write_midi


def outside_soloing_repair_sweep(root: Path, *, broad_claim: bool = False) -> dict:
    first = root / "outside_155.mid"
    second = root / "outside_131.mid"
    write_midi(first, 67)
    write_midi(second, 71)
    return {
        "schema_version": "stage_b_duration_coverage_fill_outside_soloing_repair_sweep_v1",
        "repair_summary": {
            "boundary": "outside_soloing_pitch_role_repair_candidates",
            "source_candidate_count": 2,
            "repaired_source_candidate_count": 2,
            "dead_air_preserved_source_candidate_count": 2,
            "total_variant_count": 6,
            "total_qualified_variant_count": 6,
            "broad_model_quality_claimed": False,
        },
        "claim_boundary": {
            "boundary": "outside_soloing_pitch_role_repair_candidates",
            "human_audio_preference_claimed": False,
            "broad_model_quality_claimed": broad_claim,
        },
        "source_repair_results": [
            {
                "source_candidate_id": "source_155",
                "sample_seed": 155,
                "source_selected_dead_air_ratio": 0.3333333333333333,
                "selected_candidate": {
                    "candidate_id": "source_155_outside_repair",
                    "repair_policy": "contour_resolution",
                    "midi_path": str(first),
                    "outside_soloing_gate": {"qualified": True, "flags": []},
                    "metrics": {
                        "dead_air_ratio": 0.3333333333333333,
                        "chord_tone_ratio": 1.0,
                    },
                    "focused_solo_metrics": {
                        "focused_note_count": 18,
                        "focused_unique_pitch_count": 10,
                        "focused_max_interval": 7,
                    },
                    "pitch_role_metrics": {
                        "max_non_chord_tone_run": 0,
                    },
                },
            },
            {
                "source_candidate_id": "source_131",
                "sample_seed": 131,
                "source_selected_dead_air_ratio": 0.35294117647058826,
                "selected_candidate": {
                    "candidate_id": "source_131_outside_repair",
                    "repair_policy": "contour_resolution",
                    "midi_path": str(second),
                    "outside_soloing_gate": {"qualified": True, "flags": []},
                    "metrics": {
                        "dead_air_ratio": 0.35294117647058826,
                        "chord_tone_ratio": 1.0,
                    },
                    "focused_solo_metrics": {
                        "focused_note_count": 18,
                        "focused_unique_pitch_count": 9,
                        "focused_max_interval": 5,
                    },
                    "pitch_role_metrics": {
                        "max_non_chord_tone_run": 0,
                    },
                },
            },
        ],
    }


class StageBDurationCoverageFillOutsideSoloingRepairAudioReviewPackageTest(unittest.TestCase):
    def test_renders_repaired_candidates_without_quality_claim(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            renderer = root / "fluidsynth"
            soundfont = root / "piano.sf2"
            renderer.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            soundfont.write_bytes(b"sf2")
            report = build_outside_soloing_repair_audio_review_package(
                outside_soloing_repair_sweep=outside_soloing_repair_sweep(root),
                output_dir=root / "audio_review",
                renderer_path=str(renderer),
                soundfont_path=str(soundfont),
                sample_rate=44100,
                runner=fake_runner,
            )
            summary = validate_outside_soloing_repair_audio_review_package(
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
            with self.assertRaises(StageBDurationCoverageOutsideSoloingRepairAudioReviewPackageError):
                build_outside_soloing_repair_audio_review_package(
                    outside_soloing_repair_sweep=outside_soloing_repair_sweep(root, broad_claim=True),
                    output_dir=root / "audio_review",
                    renderer_path=str(root / "fluidsynth"),
                    soundfont_path=str(root / "piano.sf2"),
                    sample_rate=44100,
                    runner=fake_runner,
                )


if __name__ == "__main__":
    unittest.main()
