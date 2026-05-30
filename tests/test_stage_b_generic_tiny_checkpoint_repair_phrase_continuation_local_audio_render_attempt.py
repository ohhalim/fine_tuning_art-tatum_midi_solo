from __future__ import annotations

import subprocess
import tempfile
import unittest
import wave
from pathlib import Path
from typing import Sequence

from scripts.render_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_audio import (
    StageBGenericTinyCheckpointRepairPhraseContinuationAudioRenderError,
    build_audio_render_report,
    validate_audio_render_report,
)


def fake_midi(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"MThd\x00\x00\x00\x06\x00\x00\x00\x01\x00`MTrk\x00\x00\x00\x04\x00\xff/\x00")


def write_wav(path: Path, sample_rate: int = 44100) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"\x00\x00\x00\x00" * sample_rate)


def fake_runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    output_index = list(command).index("-F") + 1
    write_wav(Path(command[output_index]))
    return subprocess.CompletedProcess(list(command), 0, stdout="rendered", stderr="")


def local_audio_render_package(root: Path, *, status: str = "ready_for_local_render") -> dict:
    midi_path = root / "stage_b_sample_1.mid"
    fake_midi(midi_path)
    return {
        "schema_version": "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_audio_render_package_v1",
        "renderer_probe": {
            "selected_renderer": str(root / "fluidsynth"),
            "soundfont_path": str(root / "piano.sf2"),
        },
        "local_audio_render_boundary": {
            "boundary": "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_audio_render_package",
            "status": status,
            "render_attempted": False,
            "audio_rendered_quality_claimed": False,
            "human_audio_preference_claimed": False,
        },
        "review_items": [
            {
                "review_rank": 1,
                "sample_seed": 62,
                "sample_index": 1,
                "target_qualified": True,
                "midi_file": {"path": str(midi_path), "required": True},
            },
        ],
    }


class StageBGenericTinyCheckpointRepairPhraseContinuationLocalAudioRenderAttemptTest(unittest.TestCase):
    def test_renders_single_wav_without_quality_claim(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            renderer = root / "fluidsynth"
            soundfont = root / "piano.sf2"
            renderer.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            soundfont.write_bytes(b"sf2")
            report = build_audio_render_report(
                local_audio_render_package(root),
                output_dir=root / "rendered",
                renderer_path=str(renderer),
                soundfont_path=str(soundfont),
                sample_rate=44100,
                runner=fake_runner,
            )
            summary = validate_audio_render_report(
                report,
                expected_boundary="stage_b_generic_tiny_checkpoint_repair_phrase_continuation_local_audio_render_attempt",
                expected_file_count=1,
                expected_sample_rate=44100,
                require_no_quality_claim=True,
            )

            self.assertTrue(summary["render_attempted"])
            self.assertEqual(summary["rendered_audio_file_count"], 1)
            self.assertTrue(summary["technical_wav_validation"])
            self.assertFalse(summary["audio_rendered_quality_claimed"])
            self.assertFalse(summary["human_audio_preference_claimed"])
            self.assertTrue(summary["critical_user_input_required"])

    def test_rejects_not_ready_package(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            renderer = root / "fluidsynth"
            soundfont = root / "piano.sf2"
            renderer.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            soundfont.write_bytes(b"sf2")

            with self.assertRaises(StageBGenericTinyCheckpointRepairPhraseContinuationAudioRenderError):
                build_audio_render_report(
                    local_audio_render_package(root, status="soundfont_missing"),
                    output_dir=root / "rendered",
                    renderer_path=str(renderer),
                    soundfont_path=str(soundfont),
                    sample_rate=44100,
                    runner=fake_runner,
                )

    def test_rejects_unqualified_review_item(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            package = local_audio_render_package(root)
            package["review_items"][0]["target_qualified"] = False
            renderer = root / "fluidsynth"
            soundfont = root / "piano.sf2"
            renderer.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            soundfont.write_bytes(b"sf2")

            with self.assertRaises(StageBGenericTinyCheckpointRepairPhraseContinuationAudioRenderError):
                build_audio_render_report(
                    package,
                    output_dir=root / "rendered",
                    renderer_path=str(renderer),
                    soundfont_path=str(soundfont),
                    sample_rate=44100,
                    runner=fake_runner,
                )


if __name__ == "__main__":
    unittest.main()
