from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pretty_midi

from scripts.decide_stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_decision import (
    BOUNDARY as DECISION_BOUNDARY,
    NEXT_BOUNDARY as DECISION_NEXT_BOUNDARY,
)
from scripts.export_stage_b_midi_to_solo_model_conditioned_input_path_candidates import (
    BOUNDARY as CANDIDATE_EXPORT_BOUNDARY,
)
from scripts.run_stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_probe import (
    BOUNDARY,
    NEXT_BOUNDARY,
    StageBMidiToSoloModelConditionedInputPathDeadAirTimingRepairProbeError,
    build_dead_air_timing_repair_probe_report,
    objective_metrics_for_path,
    repair_candidate_midi,
    validate_dead_air_timing_repair_probe_report,
)


def write_sparse_midi(path: Path, *, pitch_offset: int = 0) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    midi = pretty_midi.PrettyMIDI(initial_tempo=120)
    piano = pretty_midi.Instrument(program=0, is_drum=False, name="solo")
    starts = [
        0.0,
        0.125,
        1.0,
        2.0,
        2.125,
        3.0,
        4.0,
        4.125,
        5.0,
        6.0,
        6.125,
        7.0,
        8.0,
        8.125,
        9.0,
        10.0,
        10.125,
        11.0,
        12.0,
        12.125,
        13.0,
        14.0,
        14.125,
        15.0,
    ]
    pitches = [
        52,
        59,
        81,
        36,
        43,
        74,
        100,
        38,
        65,
        26,
        71,
        28,
        62,
        52,
        59,
        76,
        71,
        35,
        48,
        108,
        26,
        105,
        60,
        95,
    ]
    for start, pitch in zip(starts, pitches):
        piano.notes.append(
            pretty_midi.Note(
                velocity=84,
                pitch=int(pitch + pitch_offset),
                start=float(start),
                end=float(start + 0.125),
            )
        )
    midi.instruments.append(piano)
    midi.write(str(path))
    return str(path)


def repair_decision(*, quality_claim: bool = False) -> dict:
    return {
        "boundary": DECISION_BOUNDARY,
        "source_objective_summary": {
            "preference_fill_allowed": False,
        },
        "repair_target": {
            "selected_target": "dead_air_timing_continuity",
            "source_dead_air_failure_count": 3,
            "source_dead_air_max": 0.6521739130434783,
            "target_dead_air_max": 0.35,
            "required_dead_air_gain_min": 0.3021739130434783,
            "repair_probe_required": True,
        },
        "guardrails": {
            "min_note_count": 24,
            "min_unique_pitch_count": 8,
            "max_simultaneous_notes": 1,
            "max_postprocess_removal_ratio": 0.25,
            "require_preference_fill_blocked": True,
        },
        "readiness": {
            "dead_air_timing_repair_decision_completed": True,
            "human_audio_preference_claimed": quality_claim,
            "midi_to_solo_musical_quality_claimed": False,
            "audio_rendered_quality_claimed": False,
            "model_checkpoint_generation_quality_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
            "production_ready_claimed": False,
        },
        "decision": {
            "next_boundary": DECISION_NEXT_BOUNDARY,
            "critical_user_input_required": False,
        },
    }


def candidate_export(root: Path) -> dict:
    rows = []
    for index in range(1, 4):
        path = write_sparse_midi(root / f"midi/rank_{index:02d}.mid", pitch_offset=index - 1)
        rows.append(
            {
                "rank": index,
                "sample_index": index,
                "sample_seed": 690 + index,
                "export_midi_path": path,
                "contract_gate_passed": True,
                "note_count": 24,
                "unique_pitch_count": 19,
                "max_simultaneous_notes": 1,
                "dead_air_ratio": 0.6521739130434783,
            }
        )
    return {
        "boundary": CANDIDATE_EXPORT_BOUNDARY,
        "input_context": {
            "bpm": 120,
            "bars": 8,
            "chord_progression": ["Cmaj7", "F7", "G7", "Cmaj7"] * 2,
        },
        "top_candidates": rows,
        "readiness": {
            "model_conditioned_input_path_candidate_export_completed": True,
            "ranked_midi_candidates_exported": True,
            "model_conditioned_ranked_input_path_contract_matched": True,
            "human_audio_preference_claimed": False,
            "midi_to_solo_musical_quality_claimed": False,
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


class StageBMidiToSoloModelConditionedInputPathDeadAirTimingRepairProbeTest(
    unittest.TestCase
):
    def test_repair_candidate_reduces_dead_air_without_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = Path(write_sparse_midi(root / "source.mid"))
            repaired = root / "repaired.mid"
            stats = repair_candidate_midi(
                source,
                repaired,
                bpm=120,
                max_start_gap_seconds=0.49,
                min_note_duration_seconds=0.04,
                fill_note_duration_seconds=0.18,
                preferred_pitch_min=48,
                preferred_pitch_max=88,
            )
            before = objective_metrics_for_path(source, dead_air_threshold_seconds=0.5)
            after = objective_metrics_for_path(repaired, dead_air_threshold_seconds=0.5)

        self.assertEqual(before["note_count"], 24)
        self.assertGreater(stats["added_note_count"], 0)
        self.assertEqual(stats["removed_note_count"], 0)
        self.assertLess(after["dead_air_ratio"], before["dead_air_ratio"])
        self.assertEqual(after["dead_air_ratio"], 0.0)
        self.assertEqual(after["max_simultaneous_notes"], 1)

    def test_builds_repair_probe_and_routes_audio_package(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = build_dead_air_timing_repair_probe_report(
                repair_decision_report=repair_decision(),
                candidate_export_report=candidate_export(root),
                output_dir=root / "out",
                issue_number=690,
                min_repaired_candidates=3,
                dead_air_threshold_seconds=0.5,
                max_start_gap_seconds=0.49,
                fill_note_duration_seconds=0.18,
                min_note_duration_seconds=0.04,
                preferred_pitch_min=48,
                preferred_pitch_max=88,
            )
            summary = validate_dead_air_timing_repair_probe_report(
                report,
                expected_boundary=BOUNDARY,
                expected_next_boundary=NEXT_BOUNDARY,
                min_repaired_candidates=3,
                require_repair_passed=True,
                require_no_quality_claim=True,
            )

        self.assertTrue(summary["dead_air_timing_repair_probe_completed"])
        self.assertTrue(summary["dead_air_timing_repair_passed"])
        self.assertTrue(summary["dead_air_timing_audio_render_required"])
        self.assertEqual(summary["source_candidate_count"], 3)
        self.assertEqual(summary["repaired_candidate_count"], 3)
        self.assertEqual(summary["repaired_pass_count"], 3)
        self.assertGreater(summary["dead_air_gain_max"], 0.3)
        self.assertEqual(summary["repaired_dead_air_max"], 0.0)
        self.assertLessEqual(summary["max_repaired_simultaneous_notes"], 1)
        self.assertFalse(summary["human_audio_preference_claimed"])
        self.assertFalse(summary["midi_to_solo_musical_quality_claimed"])

    def test_rejects_repair_decision_quality_claim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(
                StageBMidiToSoloModelConditionedInputPathDeadAirTimingRepairProbeError
            ):
                build_dead_air_timing_repair_probe_report(
                    repair_decision_report=repair_decision(quality_claim=True),
                    candidate_export_report=candidate_export(root),
                    output_dir=root / "out",
                    issue_number=690,
                    min_repaired_candidates=3,
                    dead_air_threshold_seconds=0.5,
                    max_start_gap_seconds=0.49,
                    fill_note_duration_seconds=0.18,
                    min_note_duration_seconds=0.04,
                    preferred_pitch_min=48,
                    preferred_pitch_max=88,
                )

    def test_boundary_constants_are_stable(self) -> None:
        self.assertEqual(
            BOUNDARY,
            "stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_probe",
        )
        self.assertEqual(
            NEXT_BOUNDARY,
            "stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_audio_package",
        )


if __name__ == "__main__":
    unittest.main()
