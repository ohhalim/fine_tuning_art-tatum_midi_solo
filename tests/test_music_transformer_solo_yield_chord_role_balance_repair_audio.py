from __future__ import annotations

import subprocess
import tempfile
import unittest
import wave
from pathlib import Path
from typing import Sequence

import pretty_midi

from scripts.render_music_transformer_solo_yield_chord_role_balance_repair_audio import (
    NEXT_BOUNDARY,
    SCHEMA_VERSION,
    SoloYieldChordRoleBalanceRepairAudioPackageError,
    build_audio_package,
    validate_report,
)
from scripts.run_music_transformer_solo_yield_chord_role_balance_repair_sweep import (
    SCHEMA_VERSION as CHORD_ROLE_REPAIR_SCHEMA_VERSION,
)


def write_midi(path: Path) -> None:
    midi = pretty_midi.PrettyMIDI(initial_tempo=124)
    instrument = pretty_midi.Instrument(program=0)
    instrument.notes.append(pretty_midi.Note(velocity=90, pitch=60, start=0.0, end=0.25))
    instrument.notes.append(pretty_midi.Note(velocity=90, pitch=64, start=0.5, end=0.75))
    midi.instruments.append(instrument)
    path.parent.mkdir(parents=True, exist_ok=True)
    midi.write(str(path))


def write_wav(path: Path, *, sample_rate: int = 44100) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"\x00\x00" * sample_rate)


def fake_runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    wav_path = Path(command[list(command).index("-F") + 1])
    sample_rate = int(command[list(command).index("-r") + 1])
    write_wav(wav_path, sample_rate=sample_rate)
    return subprocess.CompletedProcess(list(command), 0, stdout="rendered", stderr="")


def repair_sweep(root: Path, *, quality_claim: bool = False) -> dict:
    midi_paths = [root / f"candidate_{index}.mid" for index in range(1, 3)]
    for path in midi_paths:
        write_midi(path)
    return {
        "schema_version": CHORD_ROLE_REPAIR_SCHEMA_VERSION,
        "output_dir": str(root / "repair"),
        "candidate_repairs": [
            {
                "review_index": 1,
                "case_label": "minor_backdoor",
                "source_midi_path": str(midi_paths[0]),
                "repaired_midi_path": str(midi_paths[0]),
            },
            {
                "review_index": 2,
                "case_label": "dominant_cycle",
                "source_midi_path": str(midi_paths[1]),
                "repaired_midi_path": str(midi_paths[1]),
            },
        ],
        "aggregate": {
            "candidate_count": 2,
            "repaired_midi_count": 2,
            "target_supported": True,
            "low_chord_role_count_before": 1,
            "low_chord_role_count_after": 0,
            "changed_note_total": 1,
            "max_abs_pitch_shift": 2,
            "chord_tone_ratio_decrease_count": 0,
            "weak_direction_change_count_after": 0,
            "final_landing_not_chord_tone_count_after": 0,
            "wide_interval_review_count_before": 1,
            "wide_interval_review_count_after": 0,
        },
        "readiness": {
            "audio_rendered_quality_claimed": False,
            "musical_quality_claimed": quality_claim,
            "artist_style_claimed": False,
            "production_ready_claimed": False,
        },
    }


class MusicTransformerSoloYieldChordRoleBalanceRepairAudioTest(unittest.TestCase):
    def test_renders_chord_role_repaired_midi_wavs_without_quality_claim(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            renderer = root / "fluidsynth"
            soundfont = root / "soundfont.sf2"
            renderer.write_text("#!/bin/sh\n", encoding="utf-8")
            soundfont.write_bytes(b"sf2")
            report = build_audio_package(
                repair_sweep(root),
                output_dir=root / "audio",
                renderer_path=str(renderer),
                soundfont_path=str(soundfont),
                sample_rate=22050,
                runner=fake_runner,
            )
        summary = validate_report(report, min_wav_count=2)

        self.assertEqual(summary["schema_version"], SCHEMA_VERSION)
        self.assertEqual(summary["rendered_wav_count"], 2)
        self.assertTrue(summary["technical_wav_validation"])
        self.assertEqual(summary["sample_rate"], 22050)
        self.assertFalse(summary["audio_rendered_quality_claimed"])
        self.assertFalse(summary["musical_quality_claimed"])
        self.assertEqual(summary["next_boundary"], NEXT_BOUNDARY)

    def test_rejects_quality_claim(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            renderer = root / "fluidsynth"
            soundfont = root / "soundfont.sf2"
            renderer.write_text("#!/bin/sh\n", encoding="utf-8")
            soundfont.write_bytes(b"sf2")
            with self.assertRaises(SoloYieldChordRoleBalanceRepairAudioPackageError):
                build_audio_package(
                    repair_sweep(root, quality_claim=True),
                    output_dir=root / "audio",
                    renderer_path=str(renderer),
                    soundfont_path=str(soundfont),
                    sample_rate=22050,
                    runner=fake_runner,
                )

    def test_rejects_missing_repaired_midi(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            renderer = root / "fluidsynth"
            soundfont = root / "soundfont.sf2"
            renderer.write_text("#!/bin/sh\n", encoding="utf-8")
            soundfont.write_bytes(b"sf2")
            source = repair_sweep(root)
            Path(source["candidate_repairs"][0]["repaired_midi_path"]).unlink()
            with self.assertRaises(SoloYieldChordRoleBalanceRepairAudioPackageError):
                build_audio_package(
                    source,
                    output_dir=root / "audio",
                    renderer_path=str(renderer),
                    soundfont_path=str(soundfont),
                    sample_rate=22050,
                    runner=fake_runner,
                )


if __name__ == "__main__":
    unittest.main()
