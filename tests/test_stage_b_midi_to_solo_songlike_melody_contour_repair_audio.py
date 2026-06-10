from __future__ import annotations

import subprocess
import tempfile
import unittest
import wave
from pathlib import Path
from typing import Sequence

import pretty_midi

from scripts.render_stage_b_midi_to_solo_songlike_melody_contour_repair_audio import (
    BOUNDARY,
    NEXT_BOUNDARY,
    StageBMidiToSoloSonglikeMelodyContourRepairAudioError,
    build_audio_render_report,
    validate_audio_render_report,
)
from scripts.run_stage_b_midi_to_solo_songlike_melody_contour_repair_sweep import (
    BOUNDARY as SOURCE_BOUNDARY,
    NEXT_BOUNDARY as SOURCE_NEXT_BOUNDARY,
)


def write_midi(path: Path, pitches: list[int]) -> None:
    midi = pretty_midi.PrettyMIDI(initial_tempo=120)
    midi.time_signature_changes.append(pretty_midi.TimeSignature(4, 4, 0.0))
    instrument = pretty_midi.Instrument(program=0)
    for index, pitch in enumerate(pitches):
        start = index * 0.5
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
        midi_path = root / f"songlike_repair_{index}.mid"
        write_midi(midi_path, [60, 67, 62, 72, 65, 77] * 4)
        labels = ["phrase_shape_missing_tension_release"] if index <= 2 else []
        if index in {3, 4}:
            labels = ["rhythmic_monotony"]
        repairs.append(
            {
                "source": "unit_source",
                "rank": index,
                "source_rank": index,
                "source_midi_path": f"source_{index}.mid",
                "contour_repaired_midi_path": str(midi_path),
                "source_failure_labels": ["songlike_melody_not_soloing"],
                "density_pattern": [5, 3, 6, 4, 5, 4, 6, 3],
                "contour_repaired_labeling": {
                    "failure_labels": labels,
                    "not_evaluable_labels": [
                        "outside_soloing_without_context",
                        "weak_chord_tone_landing",
                    ],
                    "metrics": {
                        "note_count": 24,
                        "unique_pitch_count": 6,
                        "dead_air_ratio": 0.0,
                        "max_abs_interval": 12,
                        "max_simultaneous_notes": 1,
                    },
                },
            }
        )
    return {
        "boundary": SOURCE_BOUNDARY,
        "candidate_repairs": repairs,
        "aggregate": {
            "candidate_count": 6,
            "source_total_failure_label_count": 8,
            "repaired_total_failure_label_count": 4,
            "failure_label_delta": 4,
            "source_songlike_failure_count": 5,
            "repaired_songlike_failure_count": 0,
            "songlike_failure_delta": 5,
            "improved_candidate_count": 4,
            "technical_regression_count": 0,
            "source_outside_soloing_repair_evidence_ready": True,
            "objective_source_outside_soloing_repair_wav_count": 6,
            "objective_source_outside_soloing_repair_source_objective_pitch_role_risk_count": 5,
            "objective_source_outside_soloing_repair_source_pitch_role_risk_count_before": 5,
            "objective_source_outside_soloing_repair_source_pitch_role_risk_count_after": 2,
            "objective_source_outside_soloing_repair_source_pitch_role_risk_delta": 3,
            "objective_source_outside_soloing_repair_source_targeted": False,
            "objective_source_outside_soloing_repair_source_residual_risk_preserved": True,
            "objective_source_outside_soloing_repair_pitch_role_risk_count_after": 0,
            "objective_source_outside_soloing_repair_pitch_role_risk_delta": 2,
            "source_outside_soloing_repair_source_objective_pitch_role_risk_count": 5,
            "source_outside_soloing_repair_source_pitch_role_risk_count_before": 5,
            "source_outside_soloing_repair_source_pitch_role_risk_count_after": 2,
            "source_outside_soloing_repair_source_pitch_role_risk_delta": 3,
            "source_outside_soloing_repair_source_targeted": False,
            "source_outside_soloing_repair_source_residual_risk_preserved": True,
            "source_outside_soloing_repair_pitch_role_risk_count_after": 0,
            "source_outside_soloing_repair_pitch_role_risk_delta": 2,
            "source_outside_soloing_not_evaluable_count": 6,
            "repaired_outside_soloing_not_evaluable_count": 6,
            "repaired_not_evaluable_counts": {
                "outside_soloing_without_context": 6,
                "weak_chord_tone_landing": 6,
            },
            "repaired_failure_counts": {
                "phrase_shape_missing_tension_release": 2,
                "rhythmic_monotony": 2,
            },
            "target_supported": True,
        },
        "selected_next_target": {
            "selected_target": "songlike_melody_contour_repair_audio_package",
            "selected_next_boundary": SOURCE_NEXT_BOUNDARY,
        },
        "readiness": {
            "boundary": SOURCE_BOUNDARY,
            "songlike_melody_contour_repair_sweep_completed": True,
            "songlike_melody_contour_repair_target_supported": True,
            "candidate_count": 6,
            "failure_label_delta": 4,
            "songlike_failure_delta": 5,
            "technical_regression_count": 0,
            "audio_package_ready": True,
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
            "critical_user_input_required": False,
        },
    }


class StageBMidiToSoloSonglikeMelodyContourRepairAudioTest(unittest.TestCase):
    def test_renders_songlike_contour_repair_wavs_without_quality_claim(self) -> None:
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
                summary["songlike_melody_contour_repair_audio_package_completed"]
            )
            self.assertEqual(summary["rendered_audio_file_count"], 6)
            self.assertTrue(summary["technical_wav_validation"])
            self.assertEqual(summary["songlike_failure_delta"], 5)
            self.assertEqual(summary["technical_regression_count"], 0)
            self.assertTrue(summary["source_outside_soloing_repair_evidence_ready"])
            self.assertEqual(summary["objective_source_outside_soloing_repair_wav_count"], 6)
            self.assertEqual(
                summary[
                    "objective_source_outside_soloing_repair_source_objective_pitch_role_risk_count"
                ],
                5,
            )
            self.assertEqual(
                summary[
                    "objective_source_outside_soloing_repair_source_pitch_role_risk_count_before"
                ],
                5,
            )
            self.assertEqual(
                summary[
                    "objective_source_outside_soloing_repair_source_pitch_role_risk_count_after"
                ],
                2,
            )
            self.assertEqual(
                summary["objective_source_outside_soloing_repair_source_pitch_role_risk_delta"],
                3,
            )
            self.assertFalse(summary["objective_source_outside_soloing_repair_source_targeted"])
            self.assertTrue(
                summary["objective_source_outside_soloing_repair_source_residual_risk_preserved"]
            )
            self.assertEqual(
                summary["objective_source_outside_soloing_repair_pitch_role_risk_count_after"],
                0,
            )
            self.assertEqual(summary["objective_source_outside_soloing_repair_pitch_role_risk_delta"], 2)
            self.assertEqual(
                summary["source_outside_soloing_repair_source_objective_pitch_role_risk_count"],
                5,
            )
            self.assertEqual(
                summary["source_outside_soloing_repair_source_pitch_role_risk_count_before"],
                5,
            )
            self.assertEqual(
                summary["source_outside_soloing_repair_source_pitch_role_risk_count_after"],
                2,
            )
            self.assertEqual(summary["source_outside_soloing_repair_source_pitch_role_risk_delta"], 3)
            self.assertFalse(summary["source_outside_soloing_repair_source_targeted"])
            self.assertTrue(summary["source_outside_soloing_repair_source_residual_risk_preserved"])
            self.assertEqual(summary["source_outside_soloing_repair_pitch_role_risk_count_after"], 0)
            self.assertEqual(summary["source_outside_soloing_repair_pitch_role_risk_delta"], 2)
            self.assertEqual(summary["source_outside_soloing_not_evaluable_count"], 6)
            self.assertEqual(summary["repaired_outside_soloing_not_evaluable_count"], 6)
            self.assertEqual(
                summary["repaired_not_evaluable_counts"]["outside_soloing_without_context"],
                6,
            )
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
            with self.assertRaises(StageBMidiToSoloSonglikeMelodyContourRepairAudioError):
                build_audio_render_report(
                    source_report(root, quality_claim=True),
                    output_dir=root / "audio_package",
                    renderer_path=str(renderer),
                    soundfont_path=str(soundfont),
                    sample_rate=44100,
                    expected_file_count=6,
                    runner=fake_runner,
                )

    def test_rejects_missing_outside_soloing_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            renderer = root / "fluidsynth"
            soundfont = root / "soundfont.sf2"
            renderer.write_text("#!/bin/sh\n", encoding="utf-8")
            soundfont.write_bytes(b"sf2")
            source = source_report(root)
            source["aggregate"]["repaired_outside_soloing_not_evaluable_count"] = 0
            with self.assertRaises(StageBMidiToSoloSonglikeMelodyContourRepairAudioError):
                build_audio_render_report(
                    source,
                    output_dir=root / "audio_package",
                    renderer_path=str(renderer),
                    soundfont_path=str(soundfont),
                    sample_rate=44100,
                    expected_file_count=6,
                    runner=fake_runner,
                )

    def test_rejects_source_context_delta_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            renderer = root / "fluidsynth"
            soundfont = root / "soundfont.sf2"
            renderer.write_text("#!/bin/sh\n", encoding="utf-8")
            soundfont.write_bytes(b"sf2")
            source = source_report(root)
            source["aggregate"]["source_outside_soloing_repair_source_pitch_role_risk_delta"] = 9
            with self.assertRaises(StageBMidiToSoloSonglikeMelodyContourRepairAudioError):
                build_audio_render_report(
                    source,
                    output_dir=root / "audio_package",
                    renderer_path=str(renderer),
                    soundfont_path=str(soundfont),
                    sample_rate=44100,
                    expected_file_count=6,
                    runner=fake_runner,
                )

    def test_constants_are_stable(self) -> None:
        self.assertEqual(BOUNDARY, "stage_b_midi_to_solo_songlike_melody_contour_repair_audio_package")
        self.assertEqual(NEXT_BOUNDARY, "stage_b_midi_to_solo_songlike_melody_contour_repair_listening_review_package")


if __name__ == "__main__":
    unittest.main()
