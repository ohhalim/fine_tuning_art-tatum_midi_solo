from __future__ import annotations

import subprocess
import unittest
import wave
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Sequence

import pretty_midi

from scripts.render_stage_b_duration_coverage_fill_audio import (
    StageBDurationCoverageFillAudioRenderError,
    build_audio_render_report,
    validate_audio_render_report,
)


def write_midi(path: Path, pitch: int) -> None:
    midi = pretty_midi.PrettyMIDI(initial_tempo=124)
    piano = pretty_midi.Instrument(program=0, is_drum=False, name="solo")
    piano.notes.append(pretty_midi.Note(velocity=76, pitch=pitch, start=0.0, end=0.2))
    midi.instruments.append(piano)
    path.parent.mkdir(parents=True, exist_ok=True)
    midi.write(str(path))


def write_wav(path: Path, sample_rate: int = 44100) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"\x00\x00\x00\x00" * sample_rate)


def local_audio_render_package(root: Path) -> dict:
    source_path = root / "source.mid"
    fill_path = root / "fill.mid"
    write_midi(source_path, 60)
    write_midi(fill_path, 64)
    return {
        "schema_version": "stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_local_audio_render_package_v1",
        "candidate_id": "duration_fill_candidate",
        "review_items": [
            {
                "role": "source_constrained_partial",
                "candidate_id": "source_candidate",
                "midi_file": {"path": str(source_path), "required": True},
            },
            {
                "role": "duration_coverage_fill_keep",
                "candidate_id": "duration_fill_candidate",
                "midi_file": {"path": str(fill_path), "required": True},
            },
        ],
    }


def fake_runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    wav_path = Path(command[3])
    write_wav(wav_path)
    return subprocess.CompletedProcess(list(command), 0, stdout="rendered", stderr="")


class StageBDurationCoverageFillAudioRenderAttemptTest(unittest.TestCase):
    def test_renders_two_wav_files_without_quality_claim(self) -> None:
        with TemporaryDirectory() as temp_dir:
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
                expected_file_count=2,
                expected_sample_rate=44100,
                require_no_quality_claim=True,
            )

            self.assertTrue(summary["render_attempted"])
            self.assertEqual(summary["rendered_audio_file_count"], 2)
            self.assertTrue(summary["technical_wav_validation"])
            self.assertFalse(summary["audio_rendered_quality_claimed"])
            self.assertFalse(summary["human_audio_preference_claimed"])
            self.assertEqual(len(summary["wav_paths"]), 2)

    def test_rejects_missing_soundfont(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            renderer = root / "fluidsynth"
            renderer.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")

            with self.assertRaises(StageBDurationCoverageFillAudioRenderError):
                build_audio_render_report(
                    local_audio_render_package(root),
                    output_dir=root / "rendered",
                    renderer_path=str(renderer),
                    soundfont_path=str(root / "missing.sf2"),
                    sample_rate=44100,
                    runner=fake_runner,
                )


if __name__ == "__main__":
    unittest.main()
