from __future__ import annotations

import subprocess
import tempfile
import unittest
import wave
from pathlib import Path
from typing import Sequence

import pretty_midi

from scripts.render_stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_tone_landing_repair_audio import (
    BOUNDARY,
    NEXT_BOUNDARY,
    StageBMidiToSoloChordToneLandingRepairAudioError,
    build_audio_render_report,
    validate_audio_render_report,
)
from scripts.run_stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_tone_landing_repair_sweep import (
    BOUNDARY as SOURCE_BOUNDARY,
    NEXT_BOUNDARY as SOURCE_NEXT_BOUNDARY,
    SELECTED_TARGET as SOURCE_SELECTED_TARGET,
)


def write_midi(path: Path, pitches: list[int]) -> None:
    midi = pretty_midi.PrettyMIDI(initial_tempo=124)
    midi.time_signature_changes.append(pretty_midi.TimeSignature(4, 4, 0.0))
    instrument = pretty_midi.Instrument(program=0)
    for index, pitch in enumerate(pitches):
        start = index * 0.45
        instrument.notes.append(
            pretty_midi.Note(velocity=90, pitch=int(pitch), start=start, end=start + 0.25)
        )
    midi.instruments.append(instrument)
    midi.write(str(path))


def write_wav(path: Path, *, sample_rate: int = 44100) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"\x00\x00" * (sample_rate // 10))


def fake_runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    wav_path = Path(str(command[list(command).index("-F") + 1]))
    sample_rate = int(command[list(command).index("-r") + 1])
    write_wav(wav_path, sample_rate=sample_rate)
    return subprocess.CompletedProcess(list(command), 0, stdout="rendered", stderr="")


def source_report(root: Path, *, quality_claim: bool = False) -> dict:
    repairs = []
    for index in range(1, 7):
        source_path = root / f"source_{index}.mid"
        repaired_path = root / f"chord_tone_landing_repair_{index}.mid"
        write_midi(source_path, [61, 66, 59, 64] * 8)
        write_midi(repaired_path, [60, 67, 58, 63] * 8)
        before_flags = ["weak_chord_tone_landing_risk"]
        if index >= 2:
            before_flags.insert(0, "outside_soloing_pitch_role_risk")
        after_flags = ["outside_soloing_pitch_role_risk"] if index in {3, 5} else []
        repairs.append(
            {
                "rank": index,
                "source_midi_path": str(source_path),
                "repaired_midi_path": str(repaired_path),
                "repair": {
                    "changed_note_count": 7,
                },
                "before": {
                    "chord_tone_ratio": 0.25,
                    "strong_beat_chord_tone_ratio": 0.0,
                    "cadence_landing_chord_tone": index == 4,
                    "cadence_landing_role": "approach",
                    "max_non_chord_tone_run": 5,
                    "bridge_flags": before_flags,
                },
                "after": {
                    "chord_tone_ratio": 0.6,
                    "strong_beat_chord_tone_ratio": 1.0,
                    "cadence_landing_chord_tone": True,
                    "cadence_landing_role": "guide",
                    "max_non_chord_tone_run": 2,
                    "bridge_flags": after_flags,
                },
            }
        )
    return {
        "boundary": SOURCE_BOUNDARY,
        "candidate_repairs": repairs,
        "aggregate": {
            "candidate_count": 6,
            "repaired_midi_count": 6,
            "changed_note_total": 42,
            "objective_outside_soloing_pitch_role_risk_count": 5,
            "weak_chord_tone_landing_risk_count_before": 6,
            "weak_chord_tone_landing_risk_count_after": 0,
            "weak_chord_tone_landing_risk_delta": 6,
            "outside_soloing_pitch_role_risk_count_before": 5,
            "outside_soloing_pitch_role_risk_count_after": 2,
            "outside_soloing_pitch_role_risk_delta": 3,
            "outside_soloing_repair_targeted": False,
            "outside_soloing_residual_risk_preserved": True,
            "final_landing_chord_tone_count_before": 1,
            "final_landing_chord_tone_count_after": 6,
            "target_supported": True,
        },
        "readiness": {
            "boundary": SOURCE_BOUNDARY,
            "chord_tone_landing_repair_sweep_completed": True,
            "candidate_count": 6,
            "repaired_midi_count": 6,
            "target_supported": True,
            "objective_outside_soloing_pitch_role_risk_count": 5,
            "outside_soloing_pitch_role_risk_count_before": 5,
            "outside_soloing_pitch_role_risk_count_after": 2,
            "outside_soloing_repair_targeted": False,
            "outside_soloing_residual_risk_preserved": True,
            "human_audio_preference_claimed": False,
            "midi_to_solo_musical_quality_claimed": quality_claim,
            "audio_rendered_quality_claimed": False,
            "model_checkpoint_generation_quality_claimed": False,
            "model_direct_generation_quality_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
            "production_ready_claimed": False,
        },
        "decision": {
            "current_boundary": SOURCE_BOUNDARY,
            "next_boundary": SOURCE_NEXT_BOUNDARY,
            "selected_target": SOURCE_SELECTED_TARGET,
            "critical_user_input_required": False,
        },
    }


class StageBMidiToSoloChordToneLandingRepairAudioTest(unittest.TestCase):
    def test_renders_chord_tone_landing_repair_wavs_without_quality_claim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            renderer = root / "fluidsynth"
            soundfont = root / "soundfont.sf2"
            renderer.write_text("#!/bin/sh\n", encoding="utf-8")
            soundfont.write_bytes(b"sf2")
            report = build_audio_render_report(
                source_report(root),
                output_dir=root / "audio_package",
                renderer_path=str(renderer),
                soundfont_path=str(soundfont),
                sample_rate=44100,
                expected_file_count=6,
                runner=fake_runner,
            )
            summary = validate_audio_render_report(
                report,
                expected_boundary=BOUNDARY,
                expected_next_boundary=NEXT_BOUNDARY,
                expected_file_count=6,
                expected_sample_rate=44100,
                require_audio_package_completed=True,
                require_no_quality_claim=True,
            )

            self.assertTrue(
                summary[
                    "songlike_melody_contour_phrase_rhythm_chord_tone_landing_repair_audio_package_completed"
                ]
            )
            self.assertEqual(summary["rendered_audio_file_count"], 6)
            self.assertTrue(summary["technical_wav_validation"])
            self.assertEqual(summary["weak_chord_tone_landing_risk_delta"], 6)
            self.assertEqual(summary["objective_outside_soloing_pitch_role_risk_count"], 5)
            self.assertEqual(summary["outside_soloing_pitch_role_risk_count_after"], 2)
            self.assertEqual(summary["outside_soloing_pitch_role_risk_delta"], 3)
            self.assertFalse(summary["outside_soloing_repair_targeted"])
            self.assertTrue(summary["outside_soloing_residual_risk_preserved"])
            self.assertEqual(summary["final_landing_chord_tone_count_after"], 6)
            self.assertTrue(summary["audio_review_required"])
            self.assertFalse(summary["human_audio_preference_claimed"])
            self.assertFalse(summary["midi_to_solo_musical_quality_claimed"])

    def test_rejects_source_quality_claim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            renderer = root / "fluidsynth"
            soundfont = root / "soundfont.sf2"
            renderer.write_text("#!/bin/sh\n", encoding="utf-8")
            soundfont.write_bytes(b"sf2")
            with self.assertRaises(StageBMidiToSoloChordToneLandingRepairAudioError):
                build_audio_render_report(
                    source_report(root, quality_claim=True),
                    output_dir=root / "audio_package",
                    renderer_path=str(renderer),
                    soundfont_path=str(soundfont),
                    sample_rate=44100,
                    expected_file_count=6,
                    runner=fake_runner,
                )

    def test_constants_are_stable(self) -> None:
        self.assertEqual(
            BOUNDARY,
            "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_tone_landing_repair_audio_package",
        )
        self.assertEqual(
            NEXT_BOUNDARY,
            "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_tone_landing_repair_listening_review_package",
        )


if __name__ == "__main__":
    unittest.main()
