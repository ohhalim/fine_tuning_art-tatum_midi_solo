from __future__ import annotations

import subprocess
import tempfile
import unittest
import wave
from pathlib import Path
from typing import Sequence

import pretty_midi

from scripts.render_stage_b_midi_to_solo_phrase_bank_dead_air_density_repair_audio import (
    BOUNDARY,
    NEXT_BOUNDARY,
    StageBMidiToSoloPhraseBankDeadAirDensityRepairAudioError,
    build_audio_render_report,
    validate_audio_render_report,
)
from scripts.run_stage_b_midi_to_solo_phrase_bank_dead_air_density_repair_probe import (
    BOUNDARY as SOURCE_BOUNDARY,
    NEXT_BOUNDARY as SOURCE_NEXT_BOUNDARY,
)


def write_midi(path: Path) -> None:
    midi = pretty_midi.PrettyMIDI(initial_tempo=120)
    instrument = pretty_midi.Instrument(program=0, is_drum=False)
    instrument.notes.append(pretty_midi.Note(velocity=80, pitch=60, start=0.0, end=0.2))
    midi.instruments.append(instrument)
    midi.write(str(path))


def write_wav(path: Path, sample_rate: int = 44100) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"\x00\x00\x00\x00" * sample_rate)


def fake_runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    wav_path = Path(command[3])
    write_wav(wav_path, sample_rate=int(command[5]))
    return subprocess.CompletedProcess(list(command), 0, stdout="rendered", stderr="")


def repair_report(tmp_path: Path, *, quality_claim: bool = False) -> dict:
    midi_paths: list[Path] = []
    for rank in range(1, 4):
        path = tmp_path / f"rank_{rank}.mid"
        write_midi(path)
        midi_paths.append(path)
    return {
        "boundary": SOURCE_BOUNDARY,
        "summary": {
            "repair_probe_target_passed": True,
        },
        "readiness": {
            "dead_air_density_repair_probe_completed": True,
            "human_audio_preference_claimed": quality_claim,
            "midi_to_solo_musical_quality_claimed": False,
            "audio_rendered_quality_claimed": False,
            "phrase_bank_musical_quality_claimed": False,
            "model_checkpoint_generation_quality_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
        },
        "decision": {
            "next_boundary": SOURCE_NEXT_BOUNDARY,
            "critical_user_input_required": False,
        },
        "repaired_candidates": [
            {
                "rank": rank,
                "sample_seed": 630 + rank,
                "repaired_midi_path": str(path),
                "repaired_metrics": {
                    "note_count": 96,
                    "unique_pitch_count": 20,
                    "dead_air_ratio": 0.2,
                    "phrase_coverage_ratio": 1.0,
                },
                "repair_gate": {"qualified": True, "flags": []},
            }
            for rank, path in enumerate(midi_paths, start=1)
        ],
    }


class StageBMidiToSoloPhraseBankDeadAirDensityRepairAudioTest(unittest.TestCase):
    def test_renders_repaired_midi_to_wav_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            renderer = tmp_path / "fluidsynth"
            soundfont = tmp_path / "soundfont.sf2"
            renderer.write_text("", encoding="utf-8")
            soundfont.write_bytes(b"sf2")

            report = build_audio_render_report(
                repair_report(tmp_path),
                output_dir=tmp_path / "out",
                renderer_path=str(renderer),
                soundfont_path=str(soundfont),
                sample_rate=44100,
                expected_file_count=3,
                runner=fake_runner,
            )
            summary = validate_audio_render_report(
                report,
                expected_boundary=BOUNDARY,
                expected_next_boundary=NEXT_BOUNDARY,
                expected_file_count=3,
                expected_sample_rate=44100,
                require_repaired_audio_path=True,
                require_no_quality_claim=True,
            )

            self.assertEqual(summary["rendered_audio_file_count"], 3)
            self.assertTrue(summary["technical_wav_validation"])
            self.assertFalse(summary["human_audio_preference_claimed"])

    def test_rejects_upstream_quality_claim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            renderer = tmp_path / "fluidsynth"
            soundfont = tmp_path / "soundfont.sf2"
            renderer.write_text("", encoding="utf-8")
            soundfont.write_bytes(b"sf2")

            with self.assertRaises(StageBMidiToSoloPhraseBankDeadAirDensityRepairAudioError):
                build_audio_render_report(
                    repair_report(tmp_path, quality_claim=True),
                    output_dir=tmp_path / "out",
                    renderer_path=str(renderer),
                    soundfont_path=str(soundfont),
                    sample_rate=44100,
                    expected_file_count=3,
                    runner=fake_runner,
                )


if __name__ == "__main__":
    unittest.main()
