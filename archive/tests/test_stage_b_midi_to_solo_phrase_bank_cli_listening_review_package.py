from __future__ import annotations

import tempfile
import unittest
import wave
from pathlib import Path

import pretty_midi

from scripts.build_stage_b_midi_to_solo_phrase_bank_cli_listening_review_package import (
    BOUNDARY,
    NEXT_BOUNDARY,
    StageBMidiToSoloPhraseBankCliListeningPackageError,
    build_listening_review_package_report,
    validate_listening_review_package_report,
)
from scripts.render_stage_b_midi_to_solo_phrase_bank_cli_audio_smoke import (
    BOUNDARY as AUDIO_BOUNDARY,
    NEXT_BOUNDARY as AUDIO_NEXT_BOUNDARY,
)


def write_midi(path: Path) -> None:
    midi = pretty_midi.PrettyMIDI(initial_tempo=120)
    instrument = pretty_midi.Instrument(program=0, is_drum=False)
    instrument.notes.append(pretty_midi.Note(velocity=80, pitch=60, start=0.0, end=0.2))
    midi.instruments.append(instrument)
    midi.write(str(path))


def write_wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(44100)
        handle.writeframes(b"\x00\x00\x00\x00" * 44100)


def audio_report(tmp_path: Path, *, quality_claim: bool = False) -> dict:
    items = []
    for rank, seed in enumerate([635, 632, 638], start=1):
        midi_path = tmp_path / f"rank_{rank}.mid"
        wav_path = tmp_path / f"rank_{rank}.wav"
        write_midi(midi_path)
        write_wav(wav_path)
        items.append(
            {
                "rank": rank,
                "mode": "cli_user_input_smoke",
                "sample_index": rank,
                "sample_seed": seed,
                "source_midi_path": str(midi_path),
                "source_note_count": 96,
                "source_unique_pitch_count": 20,
                "source_dead_air_ratio": 0.2,
                "source_phrase_coverage_ratio": 1.0,
                "wav_file": {
                    "path": str(wav_path),
                    "exists": True,
                    "duration_seconds": 1.0,
                    "sample_rate": 44100,
                    "size_bytes": wav_path.stat().st_size,
                    "sha256": "abc",
                },
            }
        )
    return {
        "audio_render_boundary": {
            "boundary": AUDIO_BOUNDARY,
            "technical_wav_validation": True,
            "cli_user_input_audio_render_completed": True,
            "phrase_bank_ranked_audio_render_completed": True,
            "phrase_bank_listening_review_package_required": True,
            "human_audio_preference_claimed": quality_claim,
            "midi_to_solo_musical_quality_claimed": False,
            "audio_rendered_quality_claimed": False,
            "phrase_bank_musical_quality_claimed": False,
            "model_checkpoint_generation_quality_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
            "production_ready_claimed": False,
        },
        "decision": {
            "next_boundary": AUDIO_NEXT_BOUNDARY,
            "critical_user_input_required": False,
        },
        "rendered_audio_files": items,
    }


class StageBMidiToSoloPhraseBankCliListeningPackageTest(unittest.TestCase):
    def test_builds_pending_review_package(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report = build_listening_review_package_report(
                audio_render_report=audio_report(Path(tmp)),
                output_dir=Path(tmp) / "out",
                issue_number=658,
                expected_count=3,
            )
            summary = validate_listening_review_package_report(
                report,
                expected_boundary=BOUNDARY,
                expected_next_boundary=NEXT_BOUNDARY,
                expected_review_item_count=3,
                require_package_ready=True,
                require_no_quality_claim=True,
            )

            self.assertTrue(summary["listening_review_package_ready"])
            self.assertFalse(summary["validated_review_input"])
            self.assertEqual(summary["review_item_count"], 3)
            self.assertFalse(summary["human_audio_preference_claimed"])

    def test_rejects_upstream_quality_claim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(StageBMidiToSoloPhraseBankCliListeningPackageError):
                build_listening_review_package_report(
                    audio_render_report=audio_report(Path(tmp), quality_claim=True),
                    output_dir=Path(tmp) / "out",
                    issue_number=658,
                    expected_count=3,
                )

    def test_boundary_constants_are_stable(self) -> None:
        self.assertEqual(BOUNDARY, "stage_b_midi_to_solo_phrase_bank_cli_listening_review_package")
        self.assertEqual(NEXT_BOUNDARY, "stage_b_midi_to_solo_phrase_bank_cli_listening_review_input_guard")


if __name__ == "__main__":
    unittest.main()
