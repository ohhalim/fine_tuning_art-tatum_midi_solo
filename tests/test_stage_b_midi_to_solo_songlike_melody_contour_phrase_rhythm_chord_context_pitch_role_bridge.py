from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pretty_midi

from scripts.audit_stage_b_midi_to_solo_final_status import (
    OUTSIDE_SOLOING_REPAIR_OBJECTIVE_SCHEMA_VERSION,
)
from scripts.build_stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_context_pitch_role_bridge import (
    BOUNDARY,
    CONTEXT_LABELS,
    DEFAULT_CHORDS,
    EXPECTED_SOURCE_SCHEMA_VERSIONS,
    NEXT_BOUNDARY,
    BRIDGE_SOURCE_CONTEXT_PRESERVED_KEYS,
    SCHEMA_VERSION,
    SELECTED_TARGET,
    StageBMidiToSoloPhraseRhythmChordContextPitchRoleBridgeError,
    build_bridge_report,
    validate_bridge_report,
)
from scripts.decide_stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_repair_followup import (
    BOUNDARY as FOLLOWUP_BOUNDARY,
    EXPECTED_SOURCE_SCHEMA_VERSIONS as FOLLOWUP_SOURCE_SCHEMA_VERSIONS,
    NEXT_BOUNDARY as FOLLOWUP_NEXT_BOUNDARY,
    SCHEMA_VERSION as FOLLOWUP_SCHEMA_VERSION,
)
from scripts.run_stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_repair_sweep import (
    BOUNDARY as SWEEP_BOUNDARY,
    EXPECTED_SOURCE_SCHEMA_VERSIONS as SWEEP_SOURCE_SCHEMA_VERSIONS,
    NEXT_BOUNDARY as SWEEP_NEXT_BOUNDARY,
    SCHEMA_VERSION as SWEEP_SCHEMA_VERSION,
)


UPSTREAM_SOURCE_CONTEXT = {
    "followup_objective_source_outside_soloing_source_pitch_role_risk_count_before": 5,
    "followup_objective_source_outside_soloing_source_pitch_role_risk_count_after": 2,
    "followup_objective_source_outside_soloing_source_pitch_role_risk_delta": 3,
    "followup_objective_source_outside_soloing_source_targeted": False,
    "followup_objective_source_outside_soloing_source_residual_risk_preserved": True,
    "followup_objective_source_outside_soloing_source_context_preserved": True,
    "followup_objective_source_outside_soloing_current_pitch_role_risk_count_after": 0,
    "followup_objective_source_outside_soloing_current_pitch_role_risk_delta": 2,
    "followup_repair_sweep_source_outside_soloing_source_pitch_role_risk_count_before": 5,
    "followup_repair_sweep_source_outside_soloing_source_pitch_role_risk_count_after": 2,
    "followup_repair_sweep_source_outside_soloing_source_pitch_role_risk_delta": 3,
    "followup_repair_sweep_source_outside_soloing_source_targeted": False,
    "followup_repair_sweep_source_outside_soloing_source_residual_risk_preserved": True,
    "followup_repair_sweep_source_outside_soloing_source_context_preserved": True,
    "followup_repair_sweep_source_outside_soloing_current_pitch_role_risk_count_after": 0,
    "followup_repair_sweep_source_outside_soloing_current_pitch_role_risk_delta": 2,
    "repair_sweep_source_outside_soloing_source_pitch_role_risk_count_before": 5,
    "repair_sweep_source_outside_soloing_source_pitch_role_risk_count_after": 2,
    "repair_sweep_source_outside_soloing_source_pitch_role_risk_delta": 3,
    "repair_sweep_source_outside_soloing_source_targeted": False,
    "repair_sweep_source_outside_soloing_source_residual_risk_preserved": True,
    "repair_sweep_source_outside_soloing_source_context_preserved": True,
    "repair_sweep_source_outside_soloing_current_pitch_role_risk_count_after": 0,
    "repair_sweep_source_outside_soloing_current_pitch_role_risk_delta": 2,
}


def write_fixture_midi(path: Path, *, bpm: float = 124.0) -> None:
    seconds_per_beat = 60.0 / bpm
    midi = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    instrument = pretty_midi.Instrument(program=0)
    chord_pitches = [
        [60, 63, 67, 70],
        [65, 68, 72, 75],
        [58, 62, 65, 69],
        [63, 67, 70, 74],
    ]
    for bar in range(8):
        pitches = chord_pitches[bar % len(chord_pitches)]
        for offset, pitch in enumerate(pitches):
            start = (bar * 4.0 + offset * 0.5) * seconds_per_beat
            end = start + 0.35 * seconds_per_beat
            instrument.notes.append(
                pretty_midi.Note(velocity=90, pitch=pitch, start=start, end=end)
            )
    midi.instruments.append(instrument)
    midi.write(str(path))


def followup_report(*, quality_claim: bool = False) -> dict:
    return {
        "schema_version": FOLLOWUP_SCHEMA_VERSION,
        "boundary": FOLLOWUP_BOUNDARY,
        "source_schema_versions": dict(FOLLOWUP_SOURCE_SCHEMA_VERSIONS),
        "selected_next_target": {
            "selected_target": "songlike_melody_contour_phrase_rhythm_chord_context_pitch_role_bridge",
            "primary_remaining_failure_labels": ["rhythmic_monotony"],
            "primary_remaining_failure_count": 1,
            "context_target_labels": list(CONTEXT_LABELS),
            "context_not_evaluable_min_count": 6,
        },
        "repair_sweep_summary": {
            "remaining_failure_counts": {"rhythmic_monotony": 1},
            "not_evaluable_counts": {
                "outside_soloing_without_context": 6,
                "weak_chord_tone_landing": 6,
            },
        },
        "readiness": {
            "followup_decision_completed": True,
            "chord_context_pitch_role_bridge_selected": True,
            "candidate_count": 6,
            "source_total_failure_label_count": 4,
            "repaired_total_failure_label_count": 1,
            "failure_label_delta": 3,
            "source_phrase_rhythm_failure_count": 4,
            "repaired_phrase_rhythm_failure_count": 1,
            "phrase_rhythm_failure_delta": 3,
            "residual_rhythmic_monotony_count": 1,
            "context_not_evaluable_min_count": 6,
            "technical_regression_count": 0,
            "objective_source_outside_soloing_repair_evidence_ready": True,
            "objective_source_outside_soloing_repair_source_objective_pitch_role_risk_count": 5,
            "objective_source_outside_soloing_repair_source_pitch_role_risk_count_before": 5,
            "objective_source_outside_soloing_repair_source_pitch_role_risk_count_after": 2,
            "objective_source_outside_soloing_repair_source_pitch_role_risk_delta": 3,
            "objective_source_outside_soloing_repair_source_context_preserved": True,
            "objective_source_outside_soloing_repair_schema_context_preserved": True,
            "objective_source_outside_soloing_repair_objective_schema_version": (
                OUTSIDE_SOLOING_REPAIR_OBJECTIVE_SCHEMA_VERSION
            ),
            "objective_source_outside_soloing_repair_source_targeted": False,
            "objective_source_outside_soloing_repair_source_residual_risk_preserved": True,
            "objective_source_outside_soloing_repair_pitch_role_risk_count_after": 0,
            "objective_source_outside_soloing_repair_pitch_role_risk_delta": 2,
            "objective_source_outside_soloing_not_evaluable_count": 6,
            "objective_repaired_outside_soloing_not_evaluable_count": 6,
            "repair_sweep_source_outside_soloing_repair_evidence_ready": True,
            "repair_sweep_source_outside_soloing_repair_source_objective_pitch_role_risk_count": 5,
            "repair_sweep_source_outside_soloing_repair_source_pitch_role_risk_count_before": 5,
            "repair_sweep_source_outside_soloing_repair_source_pitch_role_risk_count_after": 2,
            "repair_sweep_source_outside_soloing_repair_source_pitch_role_risk_delta": 3,
            "repair_sweep_source_outside_soloing_repair_source_context_preserved": True,
            "repair_sweep_source_outside_soloing_repair_schema_context_preserved": True,
            "repair_sweep_source_outside_soloing_repair_objective_schema_version": (
                OUTSIDE_SOLOING_REPAIR_OBJECTIVE_SCHEMA_VERSION
            ),
            "repair_sweep_source_outside_soloing_repair_source_targeted": False,
            "repair_sweep_source_outside_soloing_repair_source_residual_risk_preserved": True,
            "repair_sweep_source_outside_soloing_repair_pitch_role_risk_count_after": 0,
            "repair_sweep_source_outside_soloing_repair_pitch_role_risk_delta": 2,
            "repair_sweep_source_outside_soloing_not_evaluable_count": 6,
            "repair_sweep_repaired_outside_soloing_not_evaluable_count": 6,
            **{f"objective_{key}": value for key, value in UPSTREAM_SOURCE_CONTEXT.items()},
            **UPSTREAM_SOURCE_CONTEXT,
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


def repair_sweep_report(midi_paths: list[Path], *, technical_regression_count: int = 0) -> dict:
    rows = []
    for index, path in enumerate(midi_paths, start=1):
        rows.append(
            {
                "rank": index,
                "phrase_rhythm_repaired_midi_path": str(path),
                "phrase_rhythm_repaired_labeling": {
                    "metrics": {
                        "bar_count": 8,
                        "grammar_valid": True,
                        "strict_valid": True,
                        "chord_context_available": False,
                    },
                    "failure_labels": ["rhythmic_monotony"] if index == 6 else [],
                    "not_evaluable_labels": list(CONTEXT_LABELS),
                },
            }
        )
    return {
        "schema_version": SWEEP_SCHEMA_VERSION,
        "boundary": SWEEP_BOUNDARY,
        "source_schema_versions": dict(SWEEP_SOURCE_SCHEMA_VERSIONS),
        "candidate_repairs": rows,
        "aggregate": {
            "candidate_count": len(rows),
            "target_supported": True,
            "source_total_failure_label_count": 4,
            "repaired_total_failure_label_count": 1,
            "failure_label_delta": 3,
            "source_phrase_rhythm_failure_count": 4,
            "repaired_phrase_rhythm_failure_count": 1,
            "phrase_rhythm_failure_delta": 3,
            "technical_regression_count": technical_regression_count,
            "source_outside_soloing_repair_evidence_ready": True,
            "objective_source_outside_soloing_repair_source_objective_pitch_role_risk_count": 5,
            "objective_source_outside_soloing_repair_source_pitch_role_risk_count_before": 5,
            "objective_source_outside_soloing_repair_source_pitch_role_risk_count_after": 2,
            "objective_source_outside_soloing_repair_source_pitch_role_risk_delta": 3,
            "objective_source_outside_soloing_repair_source_context_preserved": True,
            "objective_source_outside_soloing_repair_schema_context_preserved": True,
            "objective_source_outside_soloing_repair_objective_schema_version": (
                OUTSIDE_SOLOING_REPAIR_OBJECTIVE_SCHEMA_VERSION
            ),
            "objective_source_outside_soloing_repair_source_targeted": False,
            "objective_source_outside_soloing_repair_source_residual_risk_preserved": True,
            "objective_source_outside_soloing_repair_pitch_role_risk_count_after": 0,
            "objective_source_outside_soloing_repair_pitch_role_risk_delta": 2,
            **{f"objective_{key}": value for key, value in UPSTREAM_SOURCE_CONTEXT.items()},
            "source_outside_soloing_repair_source_objective_pitch_role_risk_count": 5,
            "source_outside_soloing_repair_source_pitch_role_risk_count_before": 5,
            "source_outside_soloing_repair_source_pitch_role_risk_count_after": 2,
            "source_outside_soloing_repair_source_pitch_role_risk_delta": 3,
            "source_outside_soloing_repair_source_context_preserved": True,
            "source_outside_soloing_repair_schema_context_preserved": True,
            "source_outside_soloing_repair_objective_schema_version": (
                OUTSIDE_SOLOING_REPAIR_OBJECTIVE_SCHEMA_VERSION
            ),
            "source_outside_soloing_repair_source_targeted": False,
            "source_outside_soloing_repair_source_residual_risk_preserved": True,
            "source_outside_soloing_repair_pitch_role_risk_count_after": 0,
            "source_outside_soloing_repair_pitch_role_risk_delta": 2,
            "source_outside_soloing_not_evaluable_count": 6,
            "repaired_outside_soloing_not_evaluable_count": 6,
            **UPSTREAM_SOURCE_CONTEXT,
            "repaired_not_evaluable_counts": {
                "outside_soloing_without_context": 6,
                "weak_chord_tone_landing": 6,
            },
        },
        "readiness": {
            "songlike_melody_contour_phrase_rhythm_repair_sweep_completed": True,
            "songlike_melody_contour_phrase_rhythm_repair_target_supported": True,
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
            "next_boundary": SWEEP_NEXT_BOUNDARY,
            "critical_user_input_required": False,
        },
    }


class StageBMidiToSoloPhraseRhythmChordContextPitchRoleBridgeTest(unittest.TestCase):
    def test_builds_bridge_and_clears_context_not_evaluable_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            midi_paths = []
            for index in range(6):
                midi_path = root / f"candidate_{index}.mid"
                write_fixture_midi(midi_path)
                midi_paths.append(midi_path)

            report = build_bridge_report(
                followup_report=followup_report(),
                repair_sweep_report=repair_sweep_report(midi_paths),
                chords=list(DEFAULT_CHORDS),
                bpm=124.0,
                output_dir=root / "bridge",
                issue_number=1208,
            )
            summary = validate_bridge_report(
                report,
                expected_boundary=BOUNDARY,
                expected_next_boundary=NEXT_BOUNDARY,
                require_bridge_completed=True,
                require_context_available=True,
                require_no_quality_claim=True,
            )

            self.assertTrue(summary["chord_context_pitch_role_bridge_completed"])
            self.assertEqual(report["schema_version"], SCHEMA_VERSION)
            self.assertEqual(report["issue_number"], 1208)
            self.assertEqual(summary["schema_version"], SCHEMA_VERSION)
            self.assertEqual(
                summary[
                    "source_songlike_melody_contour_phrase_rhythm_repair_followup_schema_version"
                ],
                FOLLOWUP_SCHEMA_VERSION,
            )
            self.assertEqual(
                summary[
                    "source_songlike_melody_contour_phrase_rhythm_repair_sweep_schema_version"
                ],
                SWEEP_SCHEMA_VERSION,
            )
            for key, expected in EXPECTED_SOURCE_SCHEMA_VERSIONS.items():
                self.assertEqual(report["source_schema_versions"][key], expected)
            self.assertEqual(summary["candidate_count"], 6)
            self.assertEqual(summary["chord_context_available_count"], 6)
            self.assertEqual(summary["pitch_role_metrics_defined_count"], 6)
            self.assertEqual(summary["not_evaluable_before_count"], 12)
            self.assertEqual(summary["not_evaluable_after_count"], 0)
            self.assertEqual(
                summary["followup_objective_source_outside_soloing_not_evaluable_count"],
                6,
            )
            self.assertEqual(
                summary["followup_objective_repaired_outside_soloing_not_evaluable_count"],
                6,
            )
            self.assertEqual(
                summary["followup_repair_sweep_source_outside_soloing_not_evaluable_count"],
                6,
            )
            self.assertEqual(
                summary["followup_repair_sweep_repaired_outside_soloing_not_evaluable_count"],
                6,
            )
            self.assertEqual(summary["repair_sweep_source_outside_soloing_not_evaluable_count"], 6)
            self.assertEqual(
                summary["repair_sweep_repaired_outside_soloing_not_evaluable_count"],
                6,
            )
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
            self.assertTrue(
                summary["followup_objective_source_outside_soloing_schema_context_preserved"]
            )
            self.assertEqual(
                summary["followup_objective_source_outside_soloing_objective_schema_version"],
                OUTSIDE_SOLOING_REPAIR_OBJECTIVE_SCHEMA_VERSION,
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
            self.assertEqual(
                summary["followup_objective_source_outside_soloing_current_pitch_role_risk_delta"],
                2,
            )
            self.assertEqual(
                summary[
                    "followup_repair_sweep_source_outside_soloing_source_pitch_role_risk_count_before"
                ],
                5,
            )
            self.assertTrue(
                summary["followup_repair_sweep_source_outside_soloing_source_context_preserved"]
            )
            self.assertTrue(
                summary["followup_repair_sweep_source_outside_soloing_schema_context_preserved"]
            )
            self.assertEqual(
                summary["followup_repair_sweep_source_outside_soloing_objective_schema_version"],
                OUTSIDE_SOLOING_REPAIR_OBJECTIVE_SCHEMA_VERSION,
            )
            self.assertEqual(
                summary[
                    "repair_sweep_source_outside_soloing_source_pitch_role_risk_count_after"
                ],
                2,
            )
            self.assertEqual(
                summary["repair_sweep_source_outside_soloing_source_pitch_role_risk_delta"],
                3,
            )
            self.assertTrue(
                summary["repair_sweep_source_outside_soloing_source_context_preserved"]
            )
            self.assertTrue(
                summary["repair_sweep_source_outside_soloing_schema_context_preserved"]
            )
            self.assertEqual(
                summary["repair_sweep_source_outside_soloing_objective_schema_version"],
                OUTSIDE_SOLOING_REPAIR_OBJECTIVE_SCHEMA_VERSION,
            )
            for key in BRIDGE_SOURCE_CONTEXT_PRESERVED_KEYS:
                self.assertTrue(summary[key])
            self.assertGreater(summary["min_chord_tone_ratio"], 0.0)
            self.assertEqual(summary["selected_target"], SELECTED_TARGET)
            self.assertFalse(summary["human_audio_preference_claimed"])
            self.assertFalse(summary["midi_to_solo_musical_quality_claimed"])

    def test_rejects_followup_quality_claim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            midi_paths = []
            for index in range(6):
                midi_path = root / f"candidate_{index}.mid"
                write_fixture_midi(midi_path)
                midi_paths.append(midi_path)

            with self.assertRaises(StageBMidiToSoloPhraseRhythmChordContextPitchRoleBridgeError):
                build_bridge_report(
                    followup_report=followup_report(quality_claim=True),
                    repair_sweep_report=repair_sweep_report(midi_paths),
                    chords=list(DEFAULT_CHORDS),
                    bpm=124.0,
                    output_dir=root / "bridge",
                    issue_number=1208,
                )

    def test_rejects_technical_regression(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            midi_paths = []
            for index in range(6):
                midi_path = root / f"candidate_{index}.mid"
                write_fixture_midi(midi_path)
                midi_paths.append(midi_path)

            with self.assertRaises(StageBMidiToSoloPhraseRhythmChordContextPitchRoleBridgeError):
                build_bridge_report(
                    followup_report=followup_report(),
                    repair_sweep_report=repair_sweep_report(
                        midi_paths, technical_regression_count=1
                    ),
                    chords=list(DEFAULT_CHORDS),
                    bpm=124.0,
                    output_dir=root / "bridge",
                    issue_number=1208,
                )

    def test_rejects_missing_followup_outside_soloing_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            midi_paths = []
            for index in range(6):
                midi_path = root / f"candidate_{index}.mid"
                write_fixture_midi(midi_path)
                midi_paths.append(midi_path)
            source = followup_report()
            del source["readiness"]["objective_source_outside_soloing_not_evaluable_count"]

            with self.assertRaises(StageBMidiToSoloPhraseRhythmChordContextPitchRoleBridgeError):
                build_bridge_report(
                    followup_report=source,
                    repair_sweep_report=repair_sweep_report(midi_paths),
                    chords=list(DEFAULT_CHORDS),
                    bpm=124.0,
                    output_dir=root / "bridge",
                    issue_number=1208,
                )

    def test_rejects_missing_repair_sweep_outside_soloing_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            midi_paths = []
            for index in range(6):
                midi_path = root / f"candidate_{index}.mid"
                write_fixture_midi(midi_path)
                midi_paths.append(midi_path)
            source = repair_sweep_report(midi_paths)
            del source["aggregate"]["source_outside_soloing_not_evaluable_count"]

            with self.assertRaises(StageBMidiToSoloPhraseRhythmChordContextPitchRoleBridgeError):
                build_bridge_report(
                    followup_report=followup_report(),
                    repair_sweep_report=source,
                    chords=list(DEFAULT_CHORDS),
                    bpm=124.0,
                    output_dir=root / "bridge",
                    issue_number=1208,
                )

    def test_rejects_followup_source_context_delta_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            midi_paths = []
            for index in range(6):
                midi_path = root / f"candidate_{index}.mid"
                write_fixture_midi(midi_path)
                midi_paths.append(midi_path)
            source = followup_report()
            source["readiness"][
                "objective_source_outside_soloing_repair_source_pitch_role_risk_delta"
            ] = 99

            with self.assertRaises(StageBMidiToSoloPhraseRhythmChordContextPitchRoleBridgeError):
                build_bridge_report(
                    followup_report=source,
                    repair_sweep_report=repair_sweep_report(midi_paths),
                    chords=list(DEFAULT_CHORDS),
                    bpm=124.0,
                    output_dir=root / "bridge",
                    issue_number=1208,
                )

    def test_rejects_followup_source_context_preservation_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            midi_paths = []
            for index in range(6):
                midi_path = root / f"candidate_{index}.mid"
                write_fixture_midi(midi_path)
                midi_paths.append(midi_path)
            source = followup_report()
            source["readiness"][
                "objective_source_outside_soloing_repair_source_context_preserved"
            ] = False

            with self.assertRaises(StageBMidiToSoloPhraseRhythmChordContextPitchRoleBridgeError):
                build_bridge_report(
                    followup_report=source,
                    repair_sweep_report=repair_sweep_report(midi_paths),
                    chords=list(DEFAULT_CHORDS),
                    bpm=124.0,
                    output_dir=root / "bridge",
                    issue_number=1208,
                )

    def test_rejects_repair_sweep_source_context_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            midi_paths = []
            for index in range(6):
                midi_path = root / f"candidate_{index}.mid"
                write_fixture_midi(midi_path)
                midi_paths.append(midi_path)
            source = repair_sweep_report(midi_paths)
            source["aggregate"][
                "source_outside_soloing_repair_source_pitch_role_risk_count_after"
            ] = 1
            source["aggregate"]["source_outside_soloing_repair_source_pitch_role_risk_delta"] = 4

            with self.assertRaises(StageBMidiToSoloPhraseRhythmChordContextPitchRoleBridgeError):
                build_bridge_report(
                    followup_report=followup_report(),
                    repair_sweep_report=source,
                    chords=list(DEFAULT_CHORDS),
                    bpm=124.0,
                    output_dir=root / "bridge",
                    issue_number=1208,
                )

    def test_rejects_repair_sweep_source_context_preservation_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            midi_paths = []
            for index in range(6):
                midi_path = root / f"candidate_{index}.mid"
                write_fixture_midi(midi_path)
                midi_paths.append(midi_path)
            source = repair_sweep_report(midi_paths)
            source["aggregate"]["source_outside_soloing_repair_source_context_preserved"] = False

            with self.assertRaises(StageBMidiToSoloPhraseRhythmChordContextPitchRoleBridgeError):
                build_bridge_report(
                    followup_report=followup_report(),
                    repair_sweep_report=source,
                    chords=list(DEFAULT_CHORDS),
                    bpm=124.0,
                    output_dir=root / "bridge",
                    issue_number=1208,
                )

    def test_rejects_report_preserved_flag_false(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            midi_paths = []
            for index in range(6):
                midi_path = root / f"candidate_{index}.mid"
                write_fixture_midi(midi_path)
                midi_paths.append(midi_path)

            report = build_bridge_report(
                followup_report=followup_report(),
                repair_sweep_report=repair_sweep_report(midi_paths),
                chords=list(DEFAULT_CHORDS),
                bpm=124.0,
                output_dir=root / "bridge",
                issue_number=1208,
            )
            report["readiness"][BRIDGE_SOURCE_CONTEXT_PRESERVED_KEYS[0]] = False

            with self.assertRaises(StageBMidiToSoloPhraseRhythmChordContextPitchRoleBridgeError):
                validate_bridge_report(
                    report,
                    expected_boundary=BOUNDARY,
                    expected_next_boundary=NEXT_BOUNDARY,
                    require_bridge_completed=True,
                    require_context_available=True,
                    require_no_quality_claim=True,
                )

    def test_rejects_followup_source_schema_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            midi_paths = []
            for index in range(6):
                midi_path = root / f"candidate_{index}.mid"
                write_fixture_midi(midi_path)
                midi_paths.append(midi_path)
            source = followup_report()
            source["source_schema_versions"][
                "songlike_melody_contour_phrase_rhythm_repair_objective_next"
            ] = "stale"

            with self.assertRaises(StageBMidiToSoloPhraseRhythmChordContextPitchRoleBridgeError):
                build_bridge_report(
                    followup_report=source,
                    repair_sweep_report=repair_sweep_report(midi_paths),
                    chords=list(DEFAULT_CHORDS),
                    bpm=124.0,
                    output_dir=root / "bridge",
                    issue_number=1208,
                )

    def test_rejects_repair_sweep_schema_context_false(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            midi_paths = []
            for index in range(6):
                midi_path = root / f"candidate_{index}.mid"
                write_fixture_midi(midi_path)
                midi_paths.append(midi_path)
            source = repair_sweep_report(midi_paths)
            source["aggregate"][
                "source_outside_soloing_repair_schema_context_preserved"
            ] = False

            with self.assertRaises(StageBMidiToSoloPhraseRhythmChordContextPitchRoleBridgeError):
                build_bridge_report(
                    followup_report=followup_report(),
                    repair_sweep_report=source,
                    chords=list(DEFAULT_CHORDS),
                    bpm=124.0,
                    output_dir=root / "bridge",
                    issue_number=1208,
                )

    def test_constants_are_stable(self) -> None:
        self.assertEqual(
            BOUNDARY,
            "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_context_pitch_role_bridge",
        )
        self.assertEqual(
            NEXT_BOUNDARY,
            "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_context_pitch_role_objective_decision",
        )
        self.assertEqual(
            SELECTED_TARGET,
            "songlike_melody_contour_phrase_rhythm_chord_context_pitch_role_objective_decision",
        )
        self.assertEqual(
            SCHEMA_VERSION,
            "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_context_pitch_role_bridge_v5",
        )


if __name__ == "__main__":
    unittest.main()
