from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pretty_midi

from scripts.decide_stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_tone_landing_repair_followup import (
    BOUNDARY as FOLLOWUP_BOUNDARY,
    NEXT_BOUNDARY as FOLLOWUP_NEXT_BOUNDARY,
    SELECTED_TARGET as FOLLOWUP_SELECTED_TARGET,
)
from scripts.run_stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_tone_landing_outside_soloing_repair_sweep import (
    BOUNDARY,
    NEXT_BOUNDARY,
    OUTSIDE_RISK_FLAG,
    SELECTED_TARGET,
    StageBMidiToSoloChordToneLandingOutsideSoloingRepairSweepError,
    build_outside_soloing_repair_sweep_report,
    validate_outside_soloing_repair_sweep_report,
)
from scripts.run_stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_tone_landing_repair_sweep import (
    BOUNDARY as CHORD_TONE_SWEEP_BOUNDARY,
)


CHORDS = ["Cm7", "Fm7", "Bb7", "Ebmaj7"]

SOURCE_CONTEXT = {
    "followup_objective_source_outside_soloing_source_pitch_role_risk_count_before": 5,
    "followup_objective_source_outside_soloing_source_pitch_role_risk_count_after": 2,
    "followup_objective_source_outside_soloing_source_pitch_role_risk_delta": 3,
    "followup_objective_source_outside_soloing_source_context_preserved": True,
    "followup_objective_source_outside_soloing_source_targeted": False,
    "followup_objective_source_outside_soloing_source_residual_risk_preserved": True,
    "followup_objective_source_outside_soloing_current_pitch_role_risk_count_after": 0,
    "followup_objective_source_outside_soloing_current_pitch_role_risk_delta": 2,
    "followup_repair_sweep_source_outside_soloing_source_pitch_role_risk_count_before": 5,
    "followup_repair_sweep_source_outside_soloing_source_pitch_role_risk_count_after": 2,
    "followup_repair_sweep_source_outside_soloing_source_pitch_role_risk_delta": 3,
    "followup_repair_sweep_source_outside_soloing_source_context_preserved": True,
    "followup_repair_sweep_source_outside_soloing_source_targeted": False,
    "followup_repair_sweep_source_outside_soloing_source_residual_risk_preserved": True,
    "followup_repair_sweep_source_outside_soloing_current_pitch_role_risk_count_after": 0,
    "followup_repair_sweep_source_outside_soloing_current_pitch_role_risk_delta": 2,
    "repair_sweep_source_outside_soloing_source_pitch_role_risk_count_before": 5,
    "repair_sweep_source_outside_soloing_source_pitch_role_risk_count_after": 2,
    "repair_sweep_source_outside_soloing_source_pitch_role_risk_delta": 3,
    "repair_sweep_source_outside_soloing_source_context_preserved": True,
    "repair_sweep_source_outside_soloing_source_targeted": False,
    "repair_sweep_source_outside_soloing_source_residual_risk_preserved": True,
    "repair_sweep_source_outside_soloing_current_pitch_role_risk_count_after": 0,
    "repair_sweep_source_outside_soloing_current_pitch_role_risk_delta": 2,
}


def _start(bar: int, position: int, *, bpm: float) -> float:
    seconds_per_beat = 60.0 / bpm
    return (bar * 4.0 + position / 4.0) * seconds_per_beat


def write_outside_run_midi(path: Path, *, bpm: float = 124.0) -> None:
    midi = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    midi.time_signature_changes.append(pretty_midi.TimeSignature(4, 4, 0.0))
    instrument = pretty_midi.Instrument(program=0)
    duration = 0.18 * (60.0 / bpm)
    for bar in range(8):
        for position, pitch in ((0, 60), (4, 63), (12, 70)):
            start = _start(bar, position, bpm=bpm)
            instrument.notes.append(
                pretty_midi.Note(
                    velocity=90,
                    pitch=pitch,
                    start=start,
                    end=start + duration,
                )
            )
    for position, pitch in ((5, 69), (6, 59), (7, 61), (9, 68)):
        start = _start(0, position, bpm=bpm)
        instrument.notes.append(
            pretty_midi.Note(velocity=90, pitch=pitch, start=start, end=start + duration)
        )
    midi.instruments.append(instrument)
    midi.write(str(path))


def followup_report(*, quality_claim: bool = False) -> dict:
    return {
        "boundary": FOLLOWUP_BOUNDARY,
        "selected_next_target": {
            "selected_target": FOLLOWUP_SELECTED_TARGET,
        },
        "readiness": {
            "followup_decision_completed": True,
            "outside_soloing_repair_selected": True,
            "weak_chord_tone_landing_resolved": True,
            "primary_remaining_risk_label": OUTSIDE_RISK_FLAG,
            "primary_remaining_risk_count": 2,
            "candidate_count": 6,
            "changed_note_total": 40,
            "weak_chord_tone_landing_risk_delta": 6,
            "objective_outside_soloing_pitch_role_risk_count": 5,
            "outside_soloing_pitch_role_risk_count_before": 5,
            "outside_soloing_pitch_role_risk_count_after": 2,
            "outside_soloing_pitch_role_risk_delta": 3,
            "outside_soloing_repair_targeted": False,
            "outside_soloing_residual_risk_preserved": True,
            **SOURCE_CONTEXT,
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
            "next_boundary": FOLLOWUP_NEXT_BOUNDARY,
            "critical_user_input_required": False,
        },
    }


def chord_tone_repair_sweep_report(midi_paths: list[Path]) -> dict:
    return {
        "boundary": CHORD_TONE_SWEEP_BOUNDARY,
        "context": {
            "chord_progression": CHORDS,
            "bpm": 124.0,
            "repair_policy": "strong_beat_and_final_note_nearest_chord_tone",
        },
        "candidate_repairs": [
            {
                "rank": index,
                "repaired_midi_path": str(path),
                "after": {
                    "chord_tone_ratio": 0.42,
                    "strong_beat_chord_tone_ratio": 1.0,
                    "cadence_landing_chord_tone": True,
                    "cadence_landing_role": "guide",
                    "max_non_chord_tone_run": 4,
                    "bridge_flags": [OUTSIDE_RISK_FLAG],
                },
            }
            for index, path in enumerate(midi_paths, start=1)
        ],
        "aggregate": {
            "candidate_count": 6,
            "repaired_midi_count": 6,
            "changed_note_total": 40,
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
            **SOURCE_CONTEXT,
        },
        "readiness": {
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
            "midi_to_solo_musical_quality_claimed": False,
            "audio_rendered_quality_claimed": False,
            "model_checkpoint_generation_quality_claimed": False,
            "model_direct_generation_quality_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
            "production_ready_claimed": False,
        },
        "decision": {
            "critical_user_input_required": False,
        },
    }


class StageBMidiToSoloChordToneLandingOutsideSoloingRepairSweepTest(unittest.TestCase):
    def test_repairs_outside_soloing_risk_and_routes_audio_package(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            midi_paths = []
            for index in range(6):
                midi_path = root / f"candidate_{index}.mid"
                write_outside_run_midi(midi_path)
                midi_paths.append(midi_path)

            report = build_outside_soloing_repair_sweep_report(
                followup_report=followup_report(),
                chord_tone_repair_sweep_report=chord_tone_repair_sweep_report(midi_paths),
                output_dir=root / "outside_repair",
                issue_number=1056,
            )
            summary = validate_outside_soloing_repair_sweep_report(
                report,
                expected_boundary=BOUNDARY,
                expected_next_boundary=NEXT_BOUNDARY,
                require_repair_completed=True,
                require_target_supported=True,
                require_no_quality_claim=True,
            )

            self.assertTrue(summary["outside_soloing_repair_sweep_completed"])
            self.assertEqual(summary["candidate_count"], 6)
            self.assertEqual(summary["repaired_midi_count"], 6)
            self.assertGreater(summary["changed_note_total"], 0)
            self.assertEqual(
                summary["source_objective_outside_soloing_pitch_role_risk_count"], 5
            )
            self.assertEqual(summary["source_outside_soloing_pitch_role_risk_count_before"], 5)
            self.assertEqual(summary["source_outside_soloing_pitch_role_risk_count_after"], 2)
            self.assertEqual(summary["source_outside_soloing_pitch_role_risk_delta"], 3)
            self.assertFalse(summary["source_outside_soloing_repair_targeted"])
            self.assertTrue(summary["source_outside_soloing_residual_risk_preserved"])
            self.assertEqual(
                summary[
                    "followup_objective_source_outside_soloing_source_pitch_role_risk_count_before"
                ],
                5,
            )
            self.assertEqual(
                summary[
                    "followup_objective_source_outside_soloing_source_pitch_role_risk_count_after"
                ],
                2,
            )
            self.assertEqual(
                summary[
                    "followup_objective_source_outside_soloing_current_pitch_role_risk_count_after"
                ],
                0,
            )
            self.assertTrue(
                summary[
                    "followup_objective_source_outside_soloing_source_context_preserved"
                ]
            )
            self.assertEqual(
                summary[
                    "followup_repair_sweep_source_outside_soloing_source_pitch_role_risk_delta"
                ],
                3,
            )
            self.assertTrue(
                summary[
                    "followup_repair_sweep_source_outside_soloing_source_context_preserved"
                ]
            )
            self.assertEqual(
                summary[
                    "repair_sweep_source_outside_soloing_current_pitch_role_risk_delta"
                ],
                2,
            )
            self.assertTrue(
                summary["repair_sweep_source_outside_soloing_source_context_preserved"]
            )
            self.assertGreater(summary["outside_soloing_pitch_role_risk_delta"], 0)
            self.assertEqual(summary["outside_soloing_pitch_role_risk_count_after"], 0)
            self.assertTrue(summary["outside_soloing_repair_targeted"])
            self.assertEqual(summary["weak_chord_tone_landing_risk_count_after"], 0)
            self.assertLess(
                summary["max_non_chord_tone_run_after"],
                summary["max_non_chord_tone_run_before"],
            )
            self.assertEqual(summary["selected_target"], SELECTED_TARGET)
            self.assertFalse(summary["human_audio_preference_claimed"])
            self.assertFalse(summary["midi_to_solo_musical_quality_claimed"])

    def test_rejects_followup_quality_claim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            midi_paths = []
            for index in range(6):
                midi_path = root / f"candidate_{index}.mid"
                write_outside_run_midi(midi_path)
                midi_paths.append(midi_path)

            with self.assertRaises(
                StageBMidiToSoloChordToneLandingOutsideSoloingRepairSweepError
            ):
                build_outside_soloing_repair_sweep_report(
                    followup_report=followup_report(quality_claim=True),
                    chord_tone_repair_sweep_report=chord_tone_repair_sweep_report(midi_paths),
                    output_dir=root / "outside_repair",
                    issue_number=1056,
                )

    def test_rejects_source_context_preservation_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            midi_paths = []
            for index in range(6):
                midi_path = root / f"candidate_{index}.mid"
                write_outside_run_midi(midi_path)
                midi_paths.append(midi_path)

            report = followup_report()
            report["readiness"][
                "followup_repair_sweep_source_outside_soloing_source_context_preserved"
            ] = False
            with self.assertRaises(
                StageBMidiToSoloChordToneLandingOutsideSoloingRepairSweepError
            ):
                build_outside_soloing_repair_sweep_report(
                    followup_report=report,
                    chord_tone_repair_sweep_report=chord_tone_repair_sweep_report(midi_paths),
                    output_dir=root / "outside_repair",
                    issue_number=1056,
                )

    def test_constants_are_stable(self) -> None:
        self.assertEqual(
            BOUNDARY,
            "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_tone_landing_outside_soloing_repair_sweep",
        )
        self.assertEqual(
            NEXT_BOUNDARY,
            "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_tone_landing_outside_soloing_repair_audio_package",
        )
        self.assertEqual(
            SELECTED_TARGET,
            "songlike_melody_contour_phrase_rhythm_chord_tone_landing_outside_soloing_repair_audio_package",
        )


if __name__ == "__main__":
    unittest.main()
