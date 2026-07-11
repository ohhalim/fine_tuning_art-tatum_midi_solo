from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pretty_midi

from scripts.decide_stage_b_midi_to_solo_model_conditioned_pitch_contour_changed_ratio_review import (
    BOUNDARY as DECISION_BOUNDARY,
    NEXT_BOUNDARY as DECISION_NEXT_BOUNDARY,
    SELECTED_TARGET as DECISION_TARGET,
)
from scripts.run_stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_pitch_contour_probe import (
    BOUNDARY as PITCH_CONTOUR_PROBE_BOUNDARY,
)
from scripts.run_stage_b_midi_to_solo_model_conditioned_pitch_contour_changed_ratio_repair_probe import (
    BOUNDARY,
    NEXT_BOUNDARY,
    StageBMidiToSoloPitchContourChangedRatioRepairProbeError,
    build_changed_ratio_repair_probe_report,
    minimum_change_contour_pitches,
    validate_changed_ratio_repair_probe_report,
)


def write_midi(path: Path, pitches: list[int]) -> str:
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


def changed_ratio_decision_source(*, quality_claim: bool = False) -> dict:
    return {
        "boundary": DECISION_BOUNDARY,
        "source_boundary": "stage_b_midi_to_solo_quality_gap_decision",
        "changed_ratio_review": {
            "technical_model_core_mvp_completed": True,
            "model_conditioned_pitch_contour_objective_completed": True,
            "fallback_path_active": True,
            "model_conditioned_input_path_alignment_required": False,
            "max_interval": 11,
            "max_interval_threshold": 12,
            "pitch_contour_target_supported": True,
            "changed_ratio_review_threshold": 0.5,
            "changed_ratio_review_required": True,
            "repair_probe_required": True,
            "audio_review_required": True,
            "human_audio_preference_claimed": False,
            "midi_to_solo_musical_quality_claimed": False,
        },
        "selected_target": {
            "selected_target": DECISION_TARGET,
            "selected_next_boundary": DECISION_NEXT_BOUNDARY,
        },
        "readiness": {
            "changed_ratio_review_decision_completed": True,
            "selected_target": DECISION_TARGET,
            "next_boundary_selected": DECISION_NEXT_BOUNDARY,
            "repair_probe_required": True,
            "human_audio_preference_claimed": quality_claim,
            "midi_to_solo_musical_quality_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
            "production_ready_claimed": False,
        },
        "decision": {
            "next_boundary": DECISION_NEXT_BOUNDARY,
            "critical_user_input_required": False,
        },
    }


def pitch_contour_probe_source(root: Path) -> dict:
    midi_paths = [
        write_midi(root / "rank_01_source.mid", [48, 84, 50, 86, 52, 88]),
        write_midi(root / "rank_02_source.mid", [55, 88, 57, 84, 59, 86]),
        write_midi(root / "rank_03_source.mid", [60, 88, 62, 84, 64, 86]),
    ]
    return {
        "boundary": PITCH_CONTOUR_PROBE_BOUNDARY,
        "summary": {
            "pitch_contour_repair_passed": True,
            "repaired_candidate_count": 3,
            "repaired_pass_count": 3,
            "max_pitch_changed_ratio": 0.7174,
            "repaired_max_interval": 11,
            "repaired_dead_air_max": 0.0,
        },
        "candidate_repairs": [
            {
                "rank": index,
                "sample_index": index,
                "sample_seed": 490 + index,
                "source_midi_path": path,
                "repaired_midi_path": path,
                "candidate_repair_passed": True,
            }
            for index, path in enumerate(midi_paths, start=1)
        ],
        "readiness": {
            "pitch_contour_repair_passed": True,
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


class StageBMidiToSoloPitchContourChangedRatioRepairProbeTest(unittest.TestCase):
    def test_minimum_change_contour_keeps_more_original_pitches(self) -> None:
        repaired = minimum_change_contour_pitches(
            [48, 84, 50, 86, 52, 88],
            preferred_pitch_min=48,
            preferred_pitch_max=88,
            max_adjacent_interval=12,
        )
        changed = sum(1 for before, after in zip([48, 84, 50, 86, 52, 88], repaired) if before != after)

        self.assertLessEqual(max(abs(repaired[index] - repaired[index - 1]) for index in range(1, len(repaired))), 12)
        self.assertLessEqual(changed / len(repaired), 0.5)

    def test_repairs_changed_ratio_and_routes_audio_package(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            report = build_changed_ratio_repair_probe_report(
                changed_ratio_decision=changed_ratio_decision_source(),
                pitch_contour_probe=pitch_contour_probe_source(root / "source"),
                output_dir=root / "out",
                issue_number=718,
                min_repaired_candidates=3,
                dead_air_threshold_seconds=0.5,
                preferred_pitch_min=48,
                preferred_pitch_max=88,
                max_adjacent_interval=12,
                max_pitch_changed_ratio=0.5,
                min_unique_pitch_count=3,
            )
            summary = validate_changed_ratio_repair_probe_report(
                report,
                expected_boundary=BOUNDARY,
                expected_next_boundary=NEXT_BOUNDARY,
                min_repaired_candidates=3,
                require_repair_passed=True,
                require_no_quality_claim=True,
            )

        self.assertTrue(summary["changed_ratio_repair_probe_completed"])
        self.assertTrue(summary["changed_ratio_repair_passed"])
        self.assertEqual(summary["repaired_pass_count"], 3)
        self.assertGreater(summary["source_max_pitch_changed_ratio"], 0.5)
        self.assertLessEqual(summary["repaired_max_pitch_changed_ratio"], 0.5)
        self.assertGreater(summary["pitch_changed_ratio_reduction"], 0.0)
        self.assertLessEqual(summary["repaired_max_interval"], 12)
        self.assertFalse(summary["human_audio_preference_claimed"])
        self.assertFalse(summary["midi_to_solo_musical_quality_claimed"])

    def test_rejects_quality_claim_in_decision_source(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with self.assertRaises(StageBMidiToSoloPitchContourChangedRatioRepairProbeError):
                build_changed_ratio_repair_probe_report(
                    changed_ratio_decision=changed_ratio_decision_source(quality_claim=True),
                    pitch_contour_probe=pitch_contour_probe_source(root / "source"),
                    output_dir=root / "out",
                    issue_number=718,
                    min_repaired_candidates=3,
                    dead_air_threshold_seconds=0.5,
                    preferred_pitch_min=48,
                    preferred_pitch_max=88,
                    max_adjacent_interval=12,
                    max_pitch_changed_ratio=0.5,
                    min_unique_pitch_count=3,
                )

    def test_boundary_constants_are_stable(self) -> None:
        self.assertEqual(
            BOUNDARY,
            "stage_b_midi_to_solo_model_conditioned_pitch_contour_changed_ratio_repair_probe",
        )
        self.assertEqual(
            NEXT_BOUNDARY,
            "stage_b_midi_to_solo_model_conditioned_pitch_contour_changed_ratio_repair_audio_package",
        )


if __name__ == "__main__":
    unittest.main()
