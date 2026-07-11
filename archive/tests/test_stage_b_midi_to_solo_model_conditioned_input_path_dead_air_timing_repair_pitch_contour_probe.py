from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pretty_midi

from scripts.decide_stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_pitch_contour import (
    BOUNDARY as DECISION_BOUNDARY,
    NEXT_BOUNDARY as DECISION_NEXT_BOUNDARY,
)
from scripts.run_stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_pitch_contour_probe import (
    BOUNDARY,
    NEXT_BOUNDARY,
    StageBMidiToSoloModelConditionedInputPathDeadAirTimingRepairPitchContourProbeError,
    build_pitch_contour_probe_report,
    choose_contour_pitch,
    validate_pitch_contour_probe_report,
)
from scripts.run_stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_probe import (
    BOUNDARY as DEAD_AIR_REPAIR_BOUNDARY,
)


def write_wide_interval_midi(path: Path, pitches: list[int]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    midi = pretty_midi.PrettyMIDI(initial_tempo=120)
    piano = pretty_midi.Instrument(program=0, is_drum=False, name="solo")
    for index, pitch in enumerate(pitches):
        start = index * 0.25
        piano.notes.append(
            pretty_midi.Note(
                velocity=84,
                pitch=int(pitch),
                start=float(start),
                end=float(start + 0.18),
            )
        )
    midi.instruments.append(piano)
    midi.write(str(path))
    return str(path)


def pitch_contour_decision_source() -> dict:
    return {
        "boundary": DECISION_BOUNDARY,
        "source_boundary": (
            "stage_b_midi_to_solo_model_conditioned_input_path_"
            "dead_air_timing_repair_objective_next_decision"
        ),
        "source_objective_summary": {
            "rendered_audio_file_count": 3,
            "technical_wav_validation": True,
            "dead_air_target_supported": True,
            "repaired_dead_air_max": 0.0,
            "max_added_note_ratio": 0.9166666666666666,
            "added_note_ratio_review_required": True,
            "max_repaired_interval": 62,
            "remaining_wide_interval_risk": True,
            "wide_interval_followup_required": True,
            "current_evidence_consolidation_ready": False,
        },
        "selected_repair_target": {
            "target": "wide_interval_pitch_contour_repair",
            "primary_metric": "max_repaired_interval",
            "source_max_interval": 62,
            "target_max_interval": 12,
            "required_interval_reduction_min": 50,
            "repair_probe_boundary": DECISION_NEXT_BOUNDARY,
        },
        "repair_guardrails": {
            "preserve_dead_air_target": True,
            "source_repaired_dead_air_max": 0.0,
            "target_dead_air_max": 0.35,
            "min_repaired_candidate_count": 3,
            "max_simultaneous_notes": 1,
            "keep_note_count_and_unique_pitch_review": True,
            "max_added_note_ratio_review_threshold": 0.75,
            "source_max_added_note_ratio": 0.9166666666666666,
            "added_note_ratio_review_required": True,
        },
        "readiness": {
            "boundary": DECISION_BOUNDARY,
            "pitch_contour_decision_completed": True,
            "repair_probe_required": True,
            "technical_wav_validation": True,
            "dead_air_target_supported": True,
            "wide_interval_followup_required": True,
            "current_evidence_consolidation_ready": False,
            "human_audio_preference_claimed": False,
            "audio_rendered_quality_claimed": False,
            "midi_to_solo_musical_quality_claimed": False,
            "model_checkpoint_generation_quality_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
            "production_ready_claimed": False,
        },
        "decision": {
            "current_boundary": DECISION_BOUNDARY,
            "next_boundary": DECISION_NEXT_BOUNDARY,
            "auto_progress_allowed": True,
            "critical_user_input_required": False,
        },
    }


def dead_air_repair_probe_source(root: Path) -> dict:
    midi_paths = [
        write_wide_interval_midi(root / "rank_01_dead_air.mid", [48, 84, 50, 86, 52, 88]),
        write_wide_interval_midi(root / "rank_02_dead_air.mid", [55, 88, 57, 84, 59, 86]),
        write_wide_interval_midi(root / "rank_03_dead_air.mid", [60, 88, 62, 84, 64, 86]),
    ]
    return {
        "boundary": DEAD_AIR_REPAIR_BOUNDARY,
        "summary": {
            "repaired_candidate_count": 3,
            "repaired_pass_count": 3,
            "repaired_dead_air_max": 0.0,
            "max_added_note_ratio": 0.9166666666666666,
            "max_postprocess_removal_ratio": 0.0,
            "max_repaired_simultaneous_notes": 1,
            "max_repaired_interval": 62,
            "dead_air_timing_repair_passed": True,
        },
        "candidate_repairs": [
            {
                "rank": index,
                "sample_index": index,
                "sample_seed": 490 + index,
                "repaired_midi_path": path,
                "candidate_repair_passed": True,
            }
            for index, path in enumerate(midi_paths, start=1)
        ],
        "readiness": {
            "dead_air_timing_repair_passed": True,
            "human_audio_preference_claimed": False,
            "midi_to_solo_musical_quality_claimed": False,
            "audio_rendered_quality_claimed": False,
            "model_checkpoint_generation_quality_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
            "production_ready_claimed": False,
        },
        "decision": {
            "critical_user_input_required": False,
            "next_boundary": (
                "stage_b_midi_to_solo_model_conditioned_input_path_"
                "dead_air_timing_repair_audio_package"
            ),
        },
    }


class StageBMidiToSoloModelConditionedInputPathDeadAirTimingRepairPitchContourProbeTest(
    unittest.TestCase
):
    def test_choose_contour_pitch_preserves_pitch_class_with_small_interval(self) -> None:
        self.assertEqual(
            choose_contour_pitch(
                84,
                previous_pitch=48,
                preferred_pitch_min=48,
                preferred_pitch_max=88,
                max_adjacent_interval=12,
            ),
            48,
        )

    def test_repairs_wide_intervals_and_routes_audio_package(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            report = build_pitch_contour_probe_report(
                pitch_contour_decision_report=pitch_contour_decision_source(),
                dead_air_repair_probe_report=dead_air_repair_probe_source(root / "source"),
                output_dir=root / "out",
                issue_number=698,
                min_repaired_candidates=3,
                dead_air_threshold_seconds=0.5,
                preferred_pitch_min=48,
                preferred_pitch_max=88,
                max_adjacent_interval=12,
                min_unique_pitch_count=3,
            )
            summary = validate_pitch_contour_probe_report(
                report,
                expected_boundary=BOUNDARY,
                expected_next_boundary=NEXT_BOUNDARY,
                min_repaired_candidates=3,
                require_repair_passed=True,
                require_no_quality_claim=True,
            )

        self.assertTrue(summary["pitch_contour_repair_probe_completed"])
        self.assertTrue(summary["pitch_contour_repair_passed"])
        self.assertEqual(summary["repaired_pass_count"], 3)
        self.assertGreater(summary["source_max_interval"], 12)
        self.assertLessEqual(summary["repaired_max_interval"], 12)
        self.assertLessEqual(summary["repaired_dead_air_max"], 0.35)
        self.assertFalse(summary["human_audio_preference_claimed"])

    def test_rejects_quality_claim_in_decision_source(self) -> None:
        source = pitch_contour_decision_source()
        source["readiness"]["human_audio_preference_claimed"] = True
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with self.assertRaises(
                StageBMidiToSoloModelConditionedInputPathDeadAirTimingRepairPitchContourProbeError
            ):
                build_pitch_contour_probe_report(
                    pitch_contour_decision_report=source,
                    dead_air_repair_probe_report=dead_air_repair_probe_source(root / "source"),
                    output_dir=root / "out",
                    issue_number=698,
                    min_repaired_candidates=3,
                    dead_air_threshold_seconds=0.5,
                    preferred_pitch_min=48,
                    preferred_pitch_max=88,
                    max_adjacent_interval=12,
                    min_unique_pitch_count=4,
                )


if __name__ == "__main__":
    unittest.main()
