from __future__ import annotations

import subprocess
import tempfile
import unittest
import wave
from pathlib import Path
from typing import Sequence

from scripts.build_stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repeatability_temperature_guard_audio_package import (
    BOUNDARY,
    NEXT_BOUNDARY,
    StageBMidiToSoloControlledScaleCheckpointTemperatureGuardAudioPackageError,
    build_audio_package_report,
    select_seed_candidate,
    validate_audio_package_report,
)
from scripts.consolidate_stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repeatability_temperature_guard_repair import (
    BOUNDARY as CONSOLIDATION_BOUNDARY,
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


def write_json(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def generation_report(root: Path, seed: int) -> Path:
    midi_a = root / f"seed_{seed}" / "sample_1.mid"
    midi_b = root / f"seed_{seed}" / "sample_2.mid"
    fake_midi(midi_a)
    fake_midi(midi_b)
    report = root / f"seed_{seed}" / "report.json"
    write_json(
        report,
        """
{
  "samples": [
    {
      "sample_index": 1,
      "sample_seed": SEED,
      "midi_path": "MIDI_A",
      "strict_valid": true,
      "grammar_gate_passed": true,
      "metrics": {"dead_air_ratio": 0.72, "unique_pitch_count": 9, "note_count": 14, "duration_sec": 3.7},
      "collapse": {"postprocess_removal_ratio": 0.4}
    },
    {
      "sample_index": 2,
      "sample_seed": SEED_PLUS,
      "midi_path": "MIDI_B",
      "strict_valid": true,
      "grammar_gate_passed": true,
      "metrics": {"dead_air_ratio": 0.53, "unique_pitch_count": 12, "note_count": 16, "duration_sec": 3.8},
      "collapse": {"postprocess_removal_ratio": 0.33}
    }
  ]
}
""".replace("SEED_PLUS", str(seed + 1))
        .replace("SEED", str(seed))
        .replace("MIDI_A", str(midi_a))
        .replace("MIDI_B", str(midi_b)),
    )
    return report


def consolidation(root: Path, *, support: bool = True, quality_claim: bool = False) -> dict:
    reports = [generation_report(root, seed) for seed in [44, 52, 60]]
    return {
        "schema_version": "stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repeatability_temperature_guard_repair_consolidation_v1",
        "boundary": CONSOLIDATION_BOUNDARY,
        "evidence_summary": {
            "seed_count": 3,
            "sample_count": 9,
            "valid_sample_count": 9,
            "strict_valid_sample_count": 9,
            "grammar_gate_sample_count": 9,
            "source_temperature": 0.9,
            "temperature": 0.75,
            "top_k": 4,
            "generation_report_paths": [str(path) for path in reports],
        },
        "consolidation_result": {
            "objective_temperature_guard_support": support,
            "audio_review_package_required": support,
            "additional_repair_required": not support,
            "human_audio_preference_claimed": False,
            "midi_to_solo_musical_quality_claimed": False,
        },
        "readiness": {
            "boundary": CONSOLIDATION_BOUNDARY,
            "temperature_guard_repair_consolidation_completed": True,
            "objective_temperature_guard_support": support,
            "audio_review_package_required": support,
            "human_audio_preference_claimed": False,
            "midi_to_solo_musical_quality_claimed": quality_claim,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
        },
        "decision": {
            "current_boundary": CONSOLIDATION_BOUNDARY,
            "next_boundary": BOUNDARY,
            "critical_user_input_required": False,
        },
    }


class StageBMidiToSoloControlledScaleCheckpointDeadAirRepeatabilityTemperatureGuardAudioPackageTest(
    unittest.TestCase
):
    def test_selects_seed_candidate_by_dead_air_then_postprocess(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = generation_report(Path(temp_dir), 44)
            candidate = select_seed_candidate(report_path)

        self.assertEqual(candidate["sample_index"], 2)

    def test_builds_audio_review_package_without_quality_claim(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            renderer = root / "fluidsynth"
            soundfont = root / "piano.sf2"
            renderer.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            soundfont.write_bytes(b"sf2")
            report = build_audio_package_report(
                consolidation(root),
                output_dir=root / "audio_package",
                renderer_path=str(renderer),
                soundfont_path=str(soundfont),
                sample_rate=44100,
                expected_file_count=3,
                runner=fake_runner,
            )
            summary = validate_audio_package_report(
                report,
                expected_boundary=BOUNDARY,
                expected_file_count=3,
                expected_sample_rate=44100,
                require_no_quality_claim=True,
            )

        self.assertEqual(summary["candidate_count"], 3)
        self.assertEqual(summary["rendered_audio_file_count"], 3)
        self.assertTrue(summary["technical_wav_validation"])
        self.assertFalse(summary["listening_review_completed"])
        self.assertFalse(summary["human_audio_preference_claimed"])
        self.assertFalse(summary["midi_to_solo_musical_quality_claimed"])
        self.assertFalse(summary["critical_user_input_required"])
        self.assertEqual(summary["next_boundary"], NEXT_BOUNDARY)
        self.assertEqual(len(summary["wav_paths"]), 3)

    def test_rejects_unsupported_consolidation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            renderer = root / "fluidsynth"
            soundfont = root / "piano.sf2"
            renderer.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            soundfont.write_bytes(b"sf2")
            with self.assertRaises(StageBMidiToSoloControlledScaleCheckpointTemperatureGuardAudioPackageError):
                build_audio_package_report(
                    consolidation(root, support=False),
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
            with self.assertRaises(StageBMidiToSoloControlledScaleCheckpointTemperatureGuardAudioPackageError):
                build_audio_package_report(
                    consolidation(root, quality_claim=True),
                    output_dir=root / "audio_package",
                    renderer_path=str(renderer),
                    soundfont_path=str(soundfont),
                    sample_rate=44100,
                    expected_file_count=3,
                    runner=fake_runner,
                )


if __name__ == "__main__":
    unittest.main()
