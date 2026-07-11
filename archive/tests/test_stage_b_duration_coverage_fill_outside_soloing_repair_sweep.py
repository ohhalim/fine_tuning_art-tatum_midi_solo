from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pretty_midi

from scripts.summarize_stage_b_duration_coverage_fill_outside_soloing_repair_sweep import (
    StageBDurationCoverageOutsideSoloingRepairSweepError,
    build_outside_soloing_repair_sweep_report,
    validate_outside_soloing_repair_sweep,
)


def outside_soloing_decision(*, boundary: str = "outside_soloing_pitch_role_phrase_clarity_repair") -> dict:
    return {
        "schema_version": "stage_b_duration_coverage_fill_outside_soloing_repair_decision_v1",
        "decision": {
            "next_boundary": boundary,
            "auto_progress_allowed": True,
            "critical_user_input_required": False,
        },
        "repair_targets": [
            "reduce_outside_sounding_pitch_choices",
            "increase_chord_tone_or_guide_tone_landing",
            "limit_non_chord_tone_run_length",
            "penalize_large_interval_after_fill",
            "prefer_phrase_contour_resolution_over_density",
        ],
        "selection_constraints": {
            "keep_dead_air_gain_gate": True,
            "keep_monophonic_gate": True,
            "require_audio_review_after_repair": True,
            "do_not_claim_broad_model_quality": True,
        },
        "claim_boundary": {
            "boundary": "outside_soloing_repair_decision",
            "human_audio_keep_claimed": False,
            "broad_model_quality_claimed": False,
        },
    }


def write_source_report(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "request": {
                    "bpm": 124,
                    "chord_progression": ["Cm7", "Fm7", "Bb7", "Ebmaj7"],
                    "bars": 2,
                    "density": "medium",
                    "energy": "mid",
                    "temperature": 0.82,
                    "top_k": 7,
                    "top_p": None,
                    "seed": 109,
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def write_outside_source_midi(path: Path, *, transpose: int = 0) -> None:
    midi = pretty_midi.PrettyMIDI(initial_tempo=124)
    piano = pretty_midi.Instrument(program=0, is_drum=False, name="outside_source")
    starts = [index * 0.16 for index in range(14)]
    pitches = [61, 75, 62, 76, 64, 78, 65, 79, 66, 80, 68, 82, 69, 83]
    for start, pitch in zip(starts, pitches):
        piano.notes.append(
            pretty_midi.Note(
                velocity=84,
                pitch=int(pitch + transpose),
                start=float(start),
                end=float(start + 0.09),
            )
        )
    midi.instruments.append(piano)
    path.parent.mkdir(parents=True, exist_ok=True)
    midi.write(str(path))


def dead_air_gain_repair(root: Path) -> dict:
    generation_root = root / "outputs" / "stage_b_generation_probe"
    write_source_report(generation_root / "run_a" / "report.json")
    write_source_report(generation_root / "run_b" / "report.json")
    midi_a = root / "selected" / "source_a.mid"
    midi_b = root / "selected" / "source_b.mid"
    write_outside_source_midi(midi_a, transpose=0)
    write_outside_source_midi(midi_b, transpose=1)
    return {
        "schema_version": "stage_b_duration_coverage_fill_dead_air_gain_repeatability_repair_v1",
        "repair_summary": {
            "boundary": "qualified_gate_repeatability_with_dead_air_gain",
            "source_candidate_count": 2,
            "dead_air_gain_source_candidate_count": 2,
            "broad_model_quality_claimed": False,
        },
        "claim_boundary": {
            "boundary": "qualified_gate_repeatability_with_dead_air_gain",
            "broad_model_quality_claimed": False,
        },
        "source_repeatability_results": [
            {
                "source_candidate_id": "source_a",
                "source_run_id": "run_a",
                "source_seed": 109,
                "sample_index": 1,
                "sample_seed": 155,
                "selected_candidate_id": "source_a_duration_fill",
                "selected_midi_path": str(midi_a),
                "baseline_dead_air_ratio": 0.375,
                "selected_dead_air_ratio": 0.0,
                "dead_air_gain_repaired": True,
                "selected_max_interval": 14,
            },
            {
                "source_candidate_id": "source_b",
                "source_run_id": "run_b",
                "source_seed": 109,
                "sample_index": 2,
                "sample_seed": 131,
                "selected_candidate_id": "source_b_duration_fill",
                "selected_midi_path": str(midi_b),
                "baseline_dead_air_ratio": 0.375,
                "selected_dead_air_ratio": 0.0,
                "dead_air_gain_repaired": True,
                "selected_max_interval": 14,
            },
        ],
    }


class StageBDurationCoverageFillOutsideSoloingRepairSweepTest(unittest.TestCase):
    def test_builds_pitch_role_repair_candidates(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            report = build_outside_soloing_repair_sweep_report(
                outside_soloing_decision=outside_soloing_decision(),
                dead_air_gain_repair=dead_air_gain_repair(root),
                output_dir=root / "outside_repair",
                generation_output_root=root / "outputs" / "stage_b_generation_probe",
                repair_policies=["chord_tone_snap", "guide_tone_landing", "contour_resolution"],
                min_source_candidates=2,
                min_repaired_source_candidates=2,
                min_chord_tone_ratio=0.72,
                max_non_chord_run_length=1,
                max_interval=7,
                min_unique_pitch_count=6,
                min_note_count=12,
                max_dead_air_ratio_exclusive=0.376,
                max_simultaneous_notes=1,
            )
            summary = validate_outside_soloing_repair_sweep(
                report,
                expected_boundary="outside_soloing_pitch_role_repair_candidates",
                min_source_candidates=2,
                min_repaired_source_candidates=2,
                require_no_broad_quality_claim=True,
            )

            self.assertEqual(summary["source_candidate_count"], 2)
            self.assertEqual(summary["repaired_source_candidate_count"], 2)
            self.assertEqual(summary["dead_air_preserved_source_candidate_count"], 2)
            self.assertEqual(summary["total_variant_count"], 6)
            self.assertFalse(summary["broad_model_quality_claimed"])
            self.assertGreaterEqual(summary["selected_min_chord_tone_ratio"], 0.72)
            self.assertLessEqual(summary["selected_max_non_chord_tone_run"], 1)
            self.assertLessEqual(summary["selected_max_interval"], 7)
            for result in report["source_repair_results"]:
                selected = result["selected_candidate"]
                self.assertTrue(selected["outside_soloing_gate"]["qualified"])
                self.assertFalse(selected["outside_soloing_gate"]["flags"])

    def test_rejects_unexpected_decision_boundary(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with self.assertRaises(StageBDurationCoverageOutsideSoloingRepairSweepError):
                build_outside_soloing_repair_sweep_report(
                    outside_soloing_decision=outside_soloing_decision(boundary="other_boundary"),
                    dead_air_gain_repair=dead_air_gain_repair(root),
                    output_dir=root / "outside_repair",
                    generation_output_root=root / "outputs" / "stage_b_generation_probe",
                    repair_policies=["chord_tone_snap"],
                    min_source_candidates=2,
                    min_repaired_source_candidates=2,
                    min_chord_tone_ratio=0.72,
                    max_non_chord_run_length=1,
                    max_interval=7,
                    min_unique_pitch_count=6,
                    min_note_count=12,
                    max_dead_air_ratio_exclusive=0.376,
                    max_simultaneous_notes=1,
                )


if __name__ == "__main__":
    unittest.main()
