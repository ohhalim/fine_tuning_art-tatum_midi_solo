from __future__ import annotations

import subprocess
import tempfile
import unittest
import wave
from pathlib import Path
from typing import Sequence

from scripts.render_stage_b_midi_to_solo_candidate_audio import (
    BOUNDARY,
    StageBMidiToSoloCandidateAudioRenderError,
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


def source_report(root: Path, *, preference_claim: bool = False) -> dict:
    midi_paths = [root / f"rank_{index:02d}.mid" for index in range(1, 4)]
    for path in midi_paths:
        fake_midi(path)
    return {
        "boundary": "stage_b_midi_to_solo_conditioned_generation_probe",
        "readiness": {
            "ranked_midi_candidates_exported": True,
            "human_audio_preference_claimed": preference_claim,
        },
        "top_candidates": [
            {
                "rank": index,
                "seed": 486 + index,
                "export_midi_path": str(path),
                "score": 1.5 + index,
                "note_count": 60,
                "unique_pitch_count": 14,
            }
            for index, path in enumerate(midi_paths, start=1)
        ],
    }


class StageBMidiToSoloCandidateAudioRenderTest(unittest.TestCase):
    def test_renders_wav_files_without_quality_claim(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            renderer = root / "fluidsynth"
            soundfont = root / "piano.sf2"
            renderer.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            soundfont.write_bytes(b"sf2")
            report = build_audio_render_report(
                source_report(root),
                output_dir=root / "rendered",
                renderer_path=str(renderer),
                soundfont_path=str(soundfont),
                sample_rate=44100,
                expected_file_count=3,
                runner=fake_runner,
            )
            summary = validate_audio_render_report(
                report,
                expected_boundary=BOUNDARY,
                expected_file_count=3,
                expected_sample_rate=44100,
                require_no_quality_claim=True,
            )

            self.assertTrue(summary["render_attempted"])
            self.assertEqual(summary["rendered_audio_file_count"], 3)
            self.assertTrue(summary["technical_wav_validation"])
            self.assertFalse(summary["audio_rendered_quality_claimed"])
            self.assertFalse(summary["human_audio_preference_claimed"])
            self.assertFalse(summary["midi_to_solo_mvp_claimed"])
            self.assertFalse(summary["critical_user_input_required"])

    def test_rejects_missing_soundfont(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            renderer = root / "fluidsynth"
            renderer.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            with self.assertRaises(StageBMidiToSoloCandidateAudioRenderError):
                build_audio_render_report(
                    source_report(root),
                    output_dir=root / "rendered",
                    renderer_path=str(renderer),
                    soundfont_path=str(root / "missing.sf2"),
                    sample_rate=44100,
                    expected_file_count=3,
                    runner=fake_runner,
                )

    def test_rejects_upstream_human_preference_claim(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            renderer = root / "fluidsynth"
            soundfont = root / "piano.sf2"
            renderer.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            soundfont.write_bytes(b"sf2")
            with self.assertRaises(StageBMidiToSoloCandidateAudioRenderError):
                build_audio_render_report(
                    source_report(root, preference_claim=True),
                    output_dir=root / "rendered",
                    renderer_path=str(renderer),
                    soundfont_path=str(soundfont),
                    sample_rate=44100,
                    expected_file_count=3,
                    runner=fake_runner,
                )


if __name__ == "__main__":
    unittest.main()
