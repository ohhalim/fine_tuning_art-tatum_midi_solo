from __future__ import annotations

import stat
import tempfile
import unittest
from pathlib import Path

from scripts.build_stage_b_generic_tiny_checkpoint_repair_audio_render_package import (
    StageBGenericTinyCheckpointRepairAudioRenderPackageError,
    build_audio_render_package,
    validate_audio_render_package,
)


def fake_midi(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"MThd\x00\x00\x00\x06\x00\x00\x00\x01\x00`MTrk\x00\x00\x00\x04\x00\xff/\x00")


def fake_executable(path: Path) -> None:
    path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def listening_fill_report(root: Path, *, quality_claimed: bool = False) -> dict:
    midi_a = root / "rank_01.mid"
    midi_b = root / "rank_02.mid"
    fake_midi(midi_a)
    fake_midi(midi_b)
    return {
        "schema_version": "stage_b_generic_tiny_checkpoint_repair_listening_fill_v1",
        "run_dir": str(root / "fill"),
        "review_input_present": False,
        "fill_status": "pending_review_input",
        "readiness": {
            "boundary": "stage_b_generic_tiny_checkpoint_repair_listening_fill",
            "human_review_filled": False,
            "musical_quality_claimed": quality_claimed,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
        },
        "decision": {
            "next_boundary": "stage_b_generic_tiny_checkpoint_repair_audio_render_package",
        },
        "listening_fill": {
            "status": "pending_review_input",
            "candidate_count": 2,
            "keep_count": 0,
            "candidate_reviews": [
                {
                    "review_rank": 1,
                    "sample_seed": 47,
                    "sample_index": 6,
                    "midi_path": str(midi_a),
                    "status": "pending_review_input",
                    "keep_decision": "pending",
                },
                {
                    "review_rank": 2,
                    "sample_seed": 45,
                    "sample_index": 4,
                    "midi_path": str(midi_b),
                    "status": "pending_review_input",
                    "keep_decision": "pending",
                },
            ],
        },
    }


class StageBGenericTinyCheckpointRepairAudioRenderPackageTest(unittest.TestCase):
    def test_records_ready_commands_without_audio_claim(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            renderer = root / "fluidsynth"
            soundfont = root / "piano.sf2"
            fake_executable(renderer)
            soundfont.write_bytes(b"sf2")
            report = build_audio_render_package(
                listening_fill_report(root),
                output_dir=root / "render_package",
                requested_renderer="fluidsynth",
                soundfont_path=str(soundfont),
                renderer_paths={"fluidsynth": str(renderer), "timidity": ""},
            )
            summary = validate_audio_render_package(
                report,
                expected_boundary="stage_b_generic_tiny_checkpoint_repair_audio_render_package",
                expected_status="ready_for_local_render",
                min_planned_outputs=2,
                require_required_midi_exists=True,
                require_no_audio_claim=True,
            )

            self.assertEqual(summary["render_status"], "ready_for_local_render")
            self.assertEqual(summary["planned_audio_output_count"], 2)
            self.assertTrue(report["planned_audio_outputs"][0]["render_command"])
            self.assertIn(str(soundfont), report["planned_audio_outputs"][0]["render_command"])
            self.assertFalse(summary["audio_rendered_quality_claimed"])
            self.assertFalse(summary["human_audio_preference_claimed"])

    def test_records_soundfont_missing_without_render_command(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            renderer = root / "fluidsynth"
            fake_executable(renderer)
            report = build_audio_render_package(
                listening_fill_report(root),
                output_dir=root / "render_package",
                requested_renderer="fluidsynth",
                soundfont_path=str(root / "missing.sf2"),
                renderer_paths={"fluidsynth": str(renderer), "timidity": ""},
            )
            summary = validate_audio_render_package(
                report,
                expected_boundary="stage_b_generic_tiny_checkpoint_repair_audio_render_package",
                expected_status="soundfont_missing",
                min_planned_outputs=2,
                require_required_midi_exists=True,
                require_no_audio_claim=True,
            )

            self.assertEqual(summary["render_status"], "soundfont_missing")
            self.assertFalse(report["planned_audio_outputs"][0]["render_command"])
            self.assertFalse(summary["auto_progress_allowed"])

    def test_rejects_quality_claimed_source(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(StageBGenericTinyCheckpointRepairAudioRenderPackageError):
                build_audio_render_package(
                    listening_fill_report(Path(temp_dir), quality_claimed=True),
                    output_dir=Path(temp_dir) / "render_package",
                )


if __name__ == "__main__":
    unittest.main()
