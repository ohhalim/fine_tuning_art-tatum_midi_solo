from __future__ import annotations

import subprocess
import tempfile
import unittest
import wave
from pathlib import Path
from typing import Sequence

from scripts.render_stage_b_midi_to_solo_model_direct_audio import (
    BOUNDARY,
    StageBMidiToSoloModelDirectAudioRenderError,
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


def source_report(root: Path, *, quality_claim: bool = False, strict_count: int = 3) -> dict:
    midi_paths = [root / f"stage_b_sample_{index}.mid" for index in range(1, 4)]
    for path in midi_paths:
        fake_midi(path)
    return {
        "boundary": "stage_b_midi_to_solo_model_direct_monophonic_overlap_repair",
        "readiness": {
            "direct_generation_review_gate_passed": True,
            "model_direct_generation_quality_claimed": quality_claim,
            "human_audio_preference_claimed": False,
        },
        "repaired_generation_summary": {
            "sample_count": 3,
            "valid_sample_count": 3,
            "strict_valid_sample_count": strict_count,
            "min_postprocess_note_count": 24,
            "midi_paths": [str(path) for path in midi_paths],
        },
    }


class StageBMidiToSoloModelDirectAudioRenderTest(unittest.TestCase):
    def test_renders_model_direct_wavs_without_quality_claim(self) -> None:
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

            self.assertEqual(summary["source_boundary"], "stage_b_midi_to_solo_model_direct_monophonic_overlap_repair")
            self.assertTrue(summary["render_attempted"])
            self.assertEqual(summary["rendered_audio_file_count"], 3)
            self.assertTrue(summary["technical_wav_validation"])
            self.assertFalse(summary["audio_rendered_quality_claimed"])
            self.assertFalse(summary["human_audio_preference_claimed"])
            self.assertFalse(summary["model_direct_generation_quality_claimed"])
            self.assertFalse(summary["critical_user_input_required"])
            self.assertEqual(summary["next_boundary"], "stage_b_midi_to_solo_model_direct_audio_evidence_consolidation")

    def test_rejects_unqualified_direct_source(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            renderer = root / "fluidsynth"
            soundfont = root / "piano.sf2"
            renderer.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            soundfont.write_bytes(b"sf2")
            with self.assertRaises(StageBMidiToSoloModelDirectAudioRenderError):
                build_audio_render_report(
                    source_report(root, strict_count=2),
                    output_dir=root / "rendered",
                    renderer_path=str(renderer),
                    soundfont_path=str(soundfont),
                    sample_rate=44100,
                    expected_file_count=3,
                    runner=fake_runner,
                )

    def test_rejects_upstream_model_quality_claim(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            renderer = root / "fluidsynth"
            soundfont = root / "piano.sf2"
            renderer.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            soundfont.write_bytes(b"sf2")
            with self.assertRaises(StageBMidiToSoloModelDirectAudioRenderError):
                build_audio_render_report(
                    source_report(root, quality_claim=True),
                    output_dir=root / "rendered",
                    renderer_path=str(renderer),
                    soundfont_path=str(soundfont),
                    sample_rate=44100,
                    expected_file_count=3,
                    runner=fake_runner,
                )


if __name__ == "__main__":
    unittest.main()
