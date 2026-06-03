from __future__ import annotations

import subprocess
import tempfile
import unittest
import wave
from pathlib import Path
from typing import Sequence

from scripts.build_stage_b_midi_to_solo_model_direct_listening_review_package import (
    BOUNDARY,
    StageBMidiToSoloModelDirectListeningReviewPackageError,
    build_listening_review_package_report,
    validate_listening_review_package_report,
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


def timing_repair_report(root: Path, *, quality_claim: bool = False, passed: bool = True) -> dict:
    midi_paths = [root / f"stage_b_sample_{index}.mid" for index in range(1, 4)]
    for path in midi_paths:
        fake_midi(path)
    return {
        "boundary": "stage_b_midi_to_solo_model_direct_timing_phrase_repair",
        "readiness": {
            "boundary": "stage_b_midi_to_solo_model_direct_timing_phrase_repair",
            "timing_phrase_repair_passed": passed,
            "model_direct_generation_quality_claimed": quality_claim,
            "human_audio_preference_claimed": False,
        },
        "decision": {
            "next_boundary": BOUNDARY,
            "critical_user_input_required": False,
        },
        "repair_result": {
            "previous_dead_air_flag_count": 3,
            "repaired_dead_air_flag_count": 0,
            "previous_max_dead_air_ratio": 0.6522,
            "repaired_max_dead_air_ratio": 0.2258,
        },
        "generation_summary": {
            "all_midi_paths_exist": True,
            "midi_paths": [str(path) for path in midi_paths],
            "strict_valid_sample_count": 3,
        },
        "repaired_diagnostics_summary": {
            "candidate_diagnostics": [
                {
                    "rank": index,
                    "note_count": 32,
                    "unique_pitch_count": 10 + index,
                    "max_interval": 9,
                    "dead_air_ratio": 0.2258,
                    "diagnostic_flags": [],
                }
                for index in range(1, 4)
            ]
        },
    }


class StageBMidiToSoloModelDirectListeningReviewPackageTest(unittest.TestCase):
    def test_builds_listening_review_package_without_quality_claim(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            renderer = root / "fluidsynth"
            soundfont = root / "piano.sf2"
            renderer.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            soundfont.write_bytes(b"sf2")
            report = build_listening_review_package_report(
                timing_repair_report(root),
                output_dir=root / "package",
                renderer_path=str(renderer),
                soundfont_path=str(soundfont),
                sample_rate=44100,
                expected_file_count=3,
                runner=fake_runner,
            )
            summary = validate_listening_review_package_report(
                report,
                expected_boundary=BOUNDARY,
                expected_file_count=3,
                expected_sample_rate=44100,
                require_no_quality_claim=True,
            )

            self.assertEqual(summary["source_boundary"], "stage_b_midi_to_solo_model_direct_timing_phrase_repair")
            self.assertEqual(summary["candidate_count"], 3)
            self.assertEqual(summary["midi_file_count"], 3)
            self.assertEqual(summary["rendered_audio_file_count"], 3)
            self.assertTrue(summary["technical_wav_validation"])
            self.assertTrue(summary["review_input_template_written"])
            self.assertFalse(summary["listening_review_completed"])
            self.assertFalse(summary["human_audio_preference_claimed"])
            self.assertFalse(summary["model_direct_generation_quality_claimed"])
            self.assertFalse(summary["critical_user_input_required"])
            self.assertEqual(summary["next_boundary"], "stage_b_midi_to_solo_model_direct_user_listening_review_fill")
            self.assertTrue(Path(summary["review_input_template_path"]).exists())

    def test_rejects_unpassed_timing_repair(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            renderer = root / "fluidsynth"
            soundfont = root / "piano.sf2"
            renderer.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            soundfont.write_bytes(b"sf2")
            with self.assertRaises(StageBMidiToSoloModelDirectListeningReviewPackageError):
                build_listening_review_package_report(
                    timing_repair_report(root, passed=False),
                    output_dir=root / "package",
                    renderer_path=str(renderer),
                    soundfont_path=str(soundfont),
                    sample_rate=44100,
                    expected_file_count=3,
                    runner=fake_runner,
                )

    def test_rejects_upstream_quality_claim(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            renderer = root / "fluidsynth"
            soundfont = root / "piano.sf2"
            renderer.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            soundfont.write_bytes(b"sf2")
            with self.assertRaises(StageBMidiToSoloModelDirectListeningReviewPackageError):
                build_listening_review_package_report(
                    timing_repair_report(root, quality_claim=True),
                    output_dir=root / "package",
                    renderer_path=str(renderer),
                    soundfont_path=str(soundfont),
                    sample_rate=44100,
                    expected_file_count=3,
                    runner=fake_runner,
                )


if __name__ == "__main__":
    unittest.main()
