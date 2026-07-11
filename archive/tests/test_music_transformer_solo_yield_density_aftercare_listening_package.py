from __future__ import annotations

import tempfile
import unittest
import wave
from pathlib import Path

import pretty_midi

from scripts.build_music_transformer_solo_yield_density_aftercare_listening_package import (
    INPUT_SCHEMA_VERSION,
    NEXT_BOUNDARY,
    SCHEMA_VERSION,
    SoloYieldDensityAftercareListeningPackageError,
    build_listening_package,
    validate_report,
)
from scripts.render_music_transformer_solo_yield_density_aftercare_audio import (
    SCHEMA_VERSION as AUDIO_PACKAGE_SCHEMA_VERSION,
)
from scripts.render_stage_b_midi_to_solo_candidate_audio import wav_meta


def write_midi(path: Path) -> None:
    midi = pretty_midi.PrettyMIDI(initial_tempo=124)
    instrument = pretty_midi.Instrument(program=0)
    instrument.notes.append(pretty_midi.Note(velocity=90, pitch=60, start=0.0, end=0.25))
    midi.instruments.append(instrument)
    path.parent.mkdir(parents=True, exist_ok=True)
    midi.write(str(path))


def write_wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(22050)
        handle.writeframes(b"\x00\x00" * 22050)


def audio_package(root: Path, *, quality_claim: bool = False) -> dict:
    midi_paths = [root / f"candidate_{index}.mid" for index in range(1, 3)]
    wav_paths = [root / f"candidate_{index}.wav" for index in range(1, 3)]
    for midi_path, wav_path in zip(midi_paths, wav_paths):
        write_midi(midi_path)
        write_wav(wav_path)
    return {
        "schema_version": AUDIO_PACKAGE_SCHEMA_VERSION,
        "output_dir": str(root / "audio_package"),
        "rendered_audio_files": [
            {
                "review_index": 1,
                "case_label": "minor_backdoor",
                "repaired_midi_path": str(midi_paths[0]),
                "wav_file": wav_meta(wav_paths[0]),
            },
            {
                "review_index": 2,
                "case_label": "dominant_cycle",
                "repaired_midi_path": str(midi_paths[1]),
                "wav_file": wav_meta(wav_paths[1]),
            },
        ],
        "aggregate": {
            "rendered_wav_count": 2,
            "technical_wav_validation": True,
        },
        "readiness": {
            "audio_rendered_quality_claimed": False,
            "musical_quality_claimed": quality_claim,
            "artist_style_claimed": False,
            "production_ready_claimed": False,
        },
    }


class MusicTransformerSoloYieldDensityAftercareListeningPackageTest(unittest.TestCase):
    def test_builds_listening_package_without_quality_claim(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            report = build_listening_package(
                audio_package(root),
                output_dir=root / "listening",
                min_candidates=2,
            )
        summary = validate_report(report, min_candidates=2)

        self.assertEqual(summary["schema_version"], SCHEMA_VERSION)
        self.assertEqual(summary["candidate_count"], 2)
        self.assertEqual(summary["candidate_midi_files_copied"], 2)
        self.assertEqual(summary["candidate_wav_files_copied"], 2)
        self.assertTrue(summary["review_input_template_written"])
        self.assertFalse(summary["validated_listening_input_present"])
        self.assertFalse(summary["preference_fill_allowed"])
        self.assertFalse(summary["musical_quality_claimed"])
        self.assertEqual(summary["next_boundary"], NEXT_BOUNDARY)
        self.assertEqual(report["review_input_template"]["schema_version"], INPUT_SCHEMA_VERSION)

    def test_rejects_quality_claim(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with self.assertRaises(SoloYieldDensityAftercareListeningPackageError):
                build_listening_package(
                    audio_package(root, quality_claim=True),
                    output_dir=root / "listening",
                    min_candidates=2,
                )

    def test_rejects_missing_wav(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = audio_package(root)
            Path(source["rendered_audio_files"][0]["wav_file"]["path"]).unlink()
            with self.assertRaises(SoloYieldDensityAftercareListeningPackageError):
                build_listening_package(source, output_dir=root / "listening", min_candidates=2)


if __name__ == "__main__":
    unittest.main()
