from __future__ import annotations

import subprocess
import tempfile
import unittest
import wave
from pathlib import Path
from typing import Sequence

from scripts.build_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_audio_package import (
    BOUNDARY,
    NEXT_BOUNDARY,
    StageBMidiToSoloModelDirectJazzPhraseVocabularyRepairAudioPackageError,
    build_repair_audio_package_report,
    validate_repair_audio_package_report,
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


def repair_probe(root: Path, *, quality_claim: bool = False, target_passed: bool = True) -> dict:
    midi_paths = [root / f"jazz_phrase_repair_rank_{index:02d}.mid" for index in range(1, 4)]
    for path in midi_paths:
        fake_midi(path)
    return {
        "schema_version": "stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_probe_v1",
        "boundary": "stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_probe",
        "generated_candidates": [
            {
                "rank": index,
                "midi_path": str(path),
                "density_pattern": [5, 3, 6, 4, 5, 4, 6, 3],
                "phrase_vocabulary_source": "repair_probe_data_guided_phrase_cells",
            }
            for index, path in enumerate(midi_paths, start=1)
        ],
        "candidate_analyses": [
            {
                "rank": index,
                "note_count": 36,
                "bar_count": 8,
                "max_abs_interval": 10 + (index % 2),
                "duration_most_common_ratio": 0.25,
                "ioi_most_common_ratio": 0.31,
                "analysis_flags": [],
            }
            for index in range(1, 4)
        ],
        "repair_result": {
            "target_passed": target_passed,
            "no_overlap": True,
        },
        "readiness": {
            "boundary": "stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_probe",
            "repair_probe_completed": True,
            "objective_repair_target_passed": target_passed,
            "generated_midi_file_count": 3,
            "human_audio_preference_claimed": False,
            "model_direct_candidate_keep_claimed": False,
            "model_direct_generation_quality_claimed": quality_claim,
            "midi_to_solo_musical_quality_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
        },
        "decision": {
            "next_boundary": BOUNDARY,
        },
    }


class StageBMidiToSoloModelDirectJazzPhraseVocabularyRepairAudioPackageTest(unittest.TestCase):
    def test_builds_audio_package_without_quality_claim(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            renderer = root / "fluidsynth"
            soundfont = root / "piano.sf2"
            renderer.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            soundfont.write_bytes(b"sf2")
            report = build_repair_audio_package_report(
                repair_probe(root),
                output_dir=root / "audio_package",
                renderer_path=str(renderer),
                soundfont_path=str(soundfont),
                sample_rate=44100,
                expected_file_count=3,
                runner=fake_runner,
            )
            summary = validate_repair_audio_package_report(
                report,
                expected_boundary=BOUNDARY,
                expected_file_count=3,
                expected_sample_rate=44100,
                require_no_quality_claim=True,
            )

        self.assertEqual(summary["source_boundary"], "stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_probe")
        self.assertEqual(summary["candidate_count"], 3)
        self.assertEqual(summary["rendered_audio_file_count"], 3)
        self.assertTrue(summary["technical_wav_validation"])
        self.assertFalse(summary["listening_review_completed"])
        self.assertFalse(summary["human_audio_preference_claimed"])
        self.assertFalse(summary["midi_to_solo_musical_quality_claimed"])
        self.assertFalse(summary["critical_user_input_required"])
        self.assertEqual(summary["next_boundary"], NEXT_BOUNDARY)
        self.assertEqual(len(summary["wav_paths"]), 3)

    def test_rejects_unpassed_probe(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            renderer = root / "fluidsynth"
            soundfont = root / "piano.sf2"
            renderer.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            soundfont.write_bytes(b"sf2")
            with self.assertRaises(StageBMidiToSoloModelDirectJazzPhraseVocabularyRepairAudioPackageError):
                build_repair_audio_package_report(
                    repair_probe(root, target_passed=False),
                    output_dir=root / "audio_package",
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
            with self.assertRaises(StageBMidiToSoloModelDirectJazzPhraseVocabularyRepairAudioPackageError):
                build_repair_audio_package_report(
                    repair_probe(root, quality_claim=True),
                    output_dir=root / "audio_package",
                    renderer_path=str(renderer),
                    soundfont_path=str(soundfont),
                    sample_rate=44100,
                    expected_file_count=3,
                    runner=fake_runner,
                )


if __name__ == "__main__":
    unittest.main()
