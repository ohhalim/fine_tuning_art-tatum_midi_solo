from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pretty_midi

from scripts.build_stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_context_pitch_role_bridge import (
    BOUNDARY as BRIDGE_BOUNDARY,
    BRIDGE_SOURCE_CONTEXT_PRESERVED_KEYS,
    SCHEMA_VERSION as BRIDGE_SCHEMA_VERSION,
)
from scripts.decide_stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_context_pitch_role_objective import (
    BOUNDARY as OBJECTIVE_BOUNDARY,
    NEXT_BOUNDARY as OBJECTIVE_NEXT_BOUNDARY,
    SCHEMA_VERSION as OBJECTIVE_SCHEMA_VERSION,
    SELECTED_TARGET as OBJECTIVE_SELECTED_TARGET,
)
from scripts.run_stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_tone_landing_repair_sweep import (
    BOUNDARY,
    NEXT_BOUNDARY,
    SCHEMA_VERSION,
    SELECTED_TARGET,
    StageBMidiToSoloChordToneLandingRepairSweepError,
    build_repair_sweep_report,
    validate_repair_sweep_report,
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


def write_weak_landing_midi(path: Path, *, bpm: float = 124.0) -> None:
    seconds_per_beat = 60.0 / bpm
    midi = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    instrument = pretty_midi.Instrument(program=0)
    weak_pitches = [61, 66, 59, 64]
    for bar in range(8):
        base_start = bar * 4.0 * seconds_per_beat
        for offset, pitch in enumerate(weak_pitches):
            start = base_start + offset * 1.0 * seconds_per_beat
            end = start + 0.35 * seconds_per_beat
            instrument.notes.append(
                pretty_midi.Note(velocity=90, pitch=pitch, start=start, end=end)
            )
    midi.instruments.append(instrument)
    midi.write(str(path))


def objective_decision_report(
    *,
    quality_claim: bool = False,
    outside_soloing_risk_count: int = 5,
) -> dict:
    return {
        "schema_version": OBJECTIVE_SCHEMA_VERSION,
        "boundary": OBJECTIVE_BOUNDARY,
        "source_schema_version": BRIDGE_SCHEMA_VERSION,
        "selected_next_target": {
            "selected_target": OBJECTIVE_SELECTED_TARGET,
        },
        "readiness": {
            "source_schema_version": BRIDGE_SCHEMA_VERSION,
            "pitch_role_objective_decision_completed": True,
            "candidate_count": 6,
            "weak_chord_tone_landing_risk_count": 6,
            "outside_soloing_pitch_role_risk_count": outside_soloing_risk_count,
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
            "next_boundary": OBJECTIVE_NEXT_BOUNDARY,
            "critical_user_input_required": False,
        },
    }


def bridge_report(midi_paths: list[Path], *, quality_claim: bool = False) -> dict:
    return {
        "schema_version": BRIDGE_SCHEMA_VERSION,
        "boundary": BRIDGE_BOUNDARY,
        "context": {
            "chord_progression": CHORDS,
            "bpm": 124.0,
        },
        "contextualized_candidates": [
            {
                "rank": index,
                "midi_path": str(path),
                "bridge_metrics": {
                    "bar_count": 8,
                    "chord_tone_ratio": 0.2,
                    "strong_beat_chord_tone_ratio": 0.0,
                    "cadence_landing_chord_tone": False,
                    "cadence_landing_role": "approach",
                    "max_non_chord_tone_run": 5,
                },
                "bridge_flags": [
                    "weak_chord_tone_landing_risk",
                    *(
                        ["outside_soloing_pitch_role_risk"]
                        if index <= 5
                        else []
                    ),
                ],
            }
            for index, path in enumerate(midi_paths, start=1)
        ],
        "readiness": {
            "candidate_count": 6,
            "not_evaluable_after_count": 0,
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
            "next_boundary": OBJECTIVE_BOUNDARY,
            "critical_user_input_required": False,
        },
    }


class StageBMidiToSoloChordToneLandingRepairSweepTest(unittest.TestCase):
    def test_repairs_final_landing_and_routes_audio_package(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            midi_paths = []
            for index in range(6):
                midi_path = root / f"candidate_{index}.mid"
                write_weak_landing_midi(midi_path)
                midi_paths.append(midi_path)

            report = build_repair_sweep_report(
                objective_decision_report=objective_decision_report(),
                bridge_report=bridge_report(midi_paths),
                output_dir=root / "repair",
                issue_number=1128,
            )
            summary = validate_repair_sweep_report(
                report,
                expected_boundary=BOUNDARY,
                expected_next_boundary=NEXT_BOUNDARY,
                require_repair_completed=True,
                require_target_supported=True,
                require_no_quality_claim=True,
            )

            self.assertEqual(report["schema_version"], SCHEMA_VERSION)
            self.assertEqual(report["source_schema_version"], OBJECTIVE_SCHEMA_VERSION)
            self.assertEqual(report["bridge_schema_version"], BRIDGE_SCHEMA_VERSION)
            self.assertEqual(summary["source_schema_version"], OBJECTIVE_SCHEMA_VERSION)
            self.assertEqual(summary["bridge_schema_version"], BRIDGE_SCHEMA_VERSION)
            self.assertTrue(summary["chord_tone_landing_repair_sweep_completed"])
            self.assertEqual(summary["candidate_count"], 6)
            self.assertEqual(summary["repaired_midi_count"], 6)
            self.assertGreater(summary["changed_note_total"], 0)
            self.assertEqual(summary["objective_outside_soloing_pitch_role_risk_count"], 5)
            self.assertGreater(summary["weak_chord_tone_landing_risk_delta"], 0)
            self.assertEqual(summary["outside_soloing_pitch_role_risk_count_before"], 5)
            self.assertFalse(summary["outside_soloing_repair_targeted"])
            self.assertTrue(summary["outside_soloing_residual_risk_preserved"])
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
                summary["followup_objective_source_outside_soloing_source_pitch_role_risk_delta"],
                3,
            )
            self.assertTrue(
                summary["followup_objective_source_outside_soloing_source_context_preserved"]
            )
            self.assertFalse(
                summary["followup_objective_source_outside_soloing_source_targeted"]
            )
            self.assertTrue(
                summary[
                    "followup_objective_source_outside_soloing_source_residual_risk_preserved"
                ]
            )
            self.assertEqual(
                summary[
                    "followup_objective_source_outside_soloing_current_pitch_role_risk_count_after"
                ],
                0,
            )
            self.assertTrue(
                summary["followup_repair_sweep_source_outside_soloing_source_context_preserved"]
            )
            self.assertTrue(
                summary["repair_sweep_source_outside_soloing_source_context_preserved"]
            )
            self.assertEqual(summary["final_landing_chord_tone_count_after"], 6)
            self.assertEqual(summary["selected_target"], SELECTED_TARGET)
            self.assertFalse(summary["human_audio_preference_claimed"])
            self.assertFalse(summary["midi_to_solo_musical_quality_claimed"])

    def test_rejects_objective_quality_claim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            midi_paths = []
            for index in range(6):
                midi_path = root / f"candidate_{index}.mid"
                write_weak_landing_midi(midi_path)
                midi_paths.append(midi_path)

            with self.assertRaises(StageBMidiToSoloChordToneLandingRepairSweepError):
                build_repair_sweep_report(
                    objective_decision_report=objective_decision_report(quality_claim=True),
                    bridge_report=bridge_report(midi_paths),
                    output_dir=root / "repair",
                    issue_number=1128,
                )

    def test_rejects_bridge_quality_claim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            midi_paths = []
            for index in range(6):
                midi_path = root / f"candidate_{index}.mid"
                write_weak_landing_midi(midi_path)
                midi_paths.append(midi_path)

            with self.assertRaises(StageBMidiToSoloChordToneLandingRepairSweepError):
                build_repair_sweep_report(
                    objective_decision_report=objective_decision_report(),
                    bridge_report=bridge_report(midi_paths, quality_claim=True),
                    output_dir=root / "repair",
                    issue_number=1128,
                )

    def test_rejects_outside_soloing_count_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            midi_paths = []
            for index in range(6):
                midi_path = root / f"candidate_{index}.mid"
                write_weak_landing_midi(midi_path)
                midi_paths.append(midi_path)

            with self.assertRaises(StageBMidiToSoloChordToneLandingRepairSweepError):
                build_repair_sweep_report(
                    objective_decision_report=objective_decision_report(
                        outside_soloing_risk_count=4
                    ),
                    bridge_report=bridge_report(midi_paths),
                    output_dir=root / "repair",
                    issue_number=1128,
                )

    def test_rejects_missing_source_context_field(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            midi_paths = []
            for index in range(6):
                midi_path = root / f"candidate_{index}.mid"
                write_weak_landing_midi(midi_path)
                midi_paths.append(midi_path)
            source = objective_decision_report()
            del source["readiness"][
                "followup_objective_source_outside_soloing_source_pitch_role_risk_delta"
            ]

            with self.assertRaises(StageBMidiToSoloChordToneLandingRepairSweepError):
                build_repair_sweep_report(
                    objective_decision_report=source,
                    bridge_report=bridge_report(midi_paths),
                    output_dir=root / "repair",
                    issue_number=1128,
                )

    def test_rejects_objective_bridge_source_context_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            midi_paths = []
            for index in range(6):
                midi_path = root / f"candidate_{index}.mid"
                write_weak_landing_midi(midi_path)
                midi_paths.append(midi_path)
            source = bridge_report(midi_paths)
            source["readiness"][
                "repair_sweep_source_outside_soloing_source_pitch_role_risk_count_after"
            ] = 1

            with self.assertRaises(StageBMidiToSoloChordToneLandingRepairSweepError):
                build_repair_sweep_report(
                    objective_decision_report=objective_decision_report(),
                    bridge_report=source,
                    output_dir=root / "repair",
                    issue_number=1128,
                )

    def test_rejects_source_context_preservation_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            midi_paths = []
            for index in range(6):
                midi_path = root / f"candidate_{index}.mid"
                write_weak_landing_midi(midi_path)
                midi_paths.append(midi_path)
            source = objective_decision_report()
            source["readiness"][
                "followup_objective_source_outside_soloing_source_context_preserved"
            ] = False

            with self.assertRaises(StageBMidiToSoloChordToneLandingRepairSweepError):
                build_repair_sweep_report(
                    objective_decision_report=source,
                    bridge_report=bridge_report(midi_paths),
                    output_dir=root / "repair",
                    issue_number=1128,
                )

    def test_rejects_objective_schema_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            midi_paths = []
            for index in range(6):
                midi_path = root / f"candidate_{index}.mid"
                write_weak_landing_midi(midi_path)
                midi_paths.append(midi_path)
            source = objective_decision_report()
            source["schema_version"] = (
                "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_context_pitch_role_objective_decision_v3"
            )

            with self.assertRaises(StageBMidiToSoloChordToneLandingRepairSweepError):
                build_repair_sweep_report(
                    objective_decision_report=source,
                    bridge_report=bridge_report(midi_paths),
                    output_dir=root / "repair",
                    issue_number=1128,
                )

    def test_rejects_bridge_schema_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            midi_paths = []
            for index in range(6):
                midi_path = root / f"candidate_{index}.mid"
                write_weak_landing_midi(midi_path)
                midi_paths.append(midi_path)
            source = bridge_report(midi_paths)
            source["schema_version"] = (
                "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_context_pitch_role_bridge_v3"
            )

            with self.assertRaises(StageBMidiToSoloChordToneLandingRepairSweepError):
                build_repair_sweep_report(
                    objective_decision_report=objective_decision_report(),
                    bridge_report=source,
                    output_dir=root / "repair",
                    issue_number=1128,
                )

    def test_rejects_report_preserved_flag_false(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            midi_paths = []
            for index in range(6):
                midi_path = root / f"candidate_{index}.mid"
                write_weak_landing_midi(midi_path)
                midi_paths.append(midi_path)
            report = build_repair_sweep_report(
                objective_decision_report=objective_decision_report(),
                bridge_report=bridge_report(midi_paths),
                output_dir=root / "repair",
                issue_number=1128,
            )
            report["readiness"][BRIDGE_SOURCE_CONTEXT_PRESERVED_KEYS[0]] = False

            with self.assertRaises(StageBMidiToSoloChordToneLandingRepairSweepError):
                validate_repair_sweep_report(
                    report,
                    expected_boundary=BOUNDARY,
                    expected_next_boundary=NEXT_BOUNDARY,
                    require_repair_completed=True,
                    require_target_supported=True,
                    require_no_quality_claim=True,
                )

    def test_constants_are_stable(self) -> None:
        self.assertEqual(
            BOUNDARY,
            "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_tone_landing_repair_sweep",
        )
        self.assertEqual(
            NEXT_BOUNDARY,
            "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_tone_landing_repair_audio_package",
        )
        self.assertEqual(
            SELECTED_TARGET,
            "songlike_melody_contour_phrase_rhythm_chord_tone_landing_repair_audio_package",
        )
        self.assertEqual(
            SCHEMA_VERSION,
            "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_tone_landing_repair_sweep_v4",
        )


if __name__ == "__main__":
    unittest.main()
