from __future__ import annotations

import stat
import tempfile
import unittest
from pathlib import Path

from scripts.build_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_audio_render_package import (
    StageBGenericTinyCheckpointRepairPhraseContinuationAudioRenderPackageError,
    build_audio_render_package,
    validate_audio_render_package,
)


def fake_midi(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"MThd\x00\x00\x00\x06\x00\x00\x00\x01\x00`MTrk\x00\x00\x00\x04\x00\xff/\x00")


def fake_executable(path: Path) -> None:
    path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def sweep_report(root: Path, *, target_passed: bool = True, quality_claimed: bool = False) -> dict:
    midi_path = root / "stage_b_sample_1.mid"
    fake_midi(midi_path)
    return {
        "schema_version": "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_sweep_v1",
        "run_dir": str(root / "sweep"),
        "readiness": {
            "boundary": "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_repair_sweep",
            "phrase_continuation_repair_target_passed": target_passed,
            "musical_quality_claimed": quality_claimed,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
        },
        "decision": {
            "next_boundary": "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_audio_render_package",
        },
        "phrase_continuation": {
            "target_passed": target_passed,
            "target_qualified_count": 1 if target_passed else 0,
            "candidate_count": 1,
            "ranked_candidates": [
                {
                    "sample_index": 1,
                    "sample_seed": 62,
                    "midi_path": str(midi_path),
                    "target_qualified": target_passed,
                    "note_count": 9,
                    "phrase_coverage_ratio": 0.90625,
                    "dead_air_ratio": 0.625,
                    "tail_empty_steps": 2,
                    "pitch_role_chord_tone_ratio": 0.5625,
                    "postprocess_removal_ratio": 0.4375,
                    "max_simultaneous_notes": 1,
                }
            ],
        },
    }


class StageBGenericTinyCheckpointRepairPhraseContinuationAudioRenderPackageTest(unittest.TestCase):
    def test_builds_ready_single_candidate_package_without_quality_claim(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            renderer = root / "fluidsynth"
            soundfont = root / "piano.sf2"
            fake_executable(renderer)
            soundfont.write_bytes(b"sf2")
            report = build_audio_render_package(
                sweep_report(root),
                output_dir=root / "render_package",
                requested_renderer="fluidsynth",
                soundfont_path=str(soundfont),
                renderer_paths={"fluidsynth": str(renderer), "timidity": ""},
            )
            summary = validate_audio_render_package(
                report,
                expected_boundary="stage_b_generic_tiny_checkpoint_repair_phrase_continuation_audio_render_package",
                expected_status="ready_for_local_render",
                min_planned_outputs=1,
                require_target_qualified=True,
                require_required_midi_exists=True,
                require_no_audio_claim=True,
            )

            self.assertEqual(summary["render_status"], "ready_for_local_render")
            self.assertEqual(summary["planned_audio_output_count"], 1)
            self.assertTrue(report["review_items"][0]["target_qualified"])
            self.assertTrue(report["planned_audio_outputs"][0]["render_command"])
            self.assertIn(str(soundfont), report["planned_audio_outputs"][0]["render_command"])
            self.assertFalse(summary["audio_rendered_quality_claimed"])
            self.assertFalse(summary["human_audio_preference_claimed"])

    def test_rejects_unqualified_sweep(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(StageBGenericTinyCheckpointRepairPhraseContinuationAudioRenderPackageError):
                build_audio_render_package(
                    sweep_report(Path(temp_dir), target_passed=False),
                    output_dir=Path(temp_dir) / "render_package",
                )

    def test_rejects_quality_claimed_source(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(StageBGenericTinyCheckpointRepairPhraseContinuationAudioRenderPackageError):
                build_audio_render_package(
                    sweep_report(Path(temp_dir), quality_claimed=True),
                    output_dir=Path(temp_dir) / "render_package",
                )


if __name__ == "__main__":
    unittest.main()
