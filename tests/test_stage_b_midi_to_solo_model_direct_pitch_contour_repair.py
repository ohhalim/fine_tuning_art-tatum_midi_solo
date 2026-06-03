from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path

import pretty_midi

from scripts.run_stage_b_midi_to_solo_model_direct_8bar_generation_probe import build_generation_command
from scripts.run_stage_b_midi_to_solo_model_direct_pitch_contour_repair import (
    BOUNDARY,
    TIMING_REPAIR_BOUNDARY,
    StageBMidiToSoloModelDirectPitchContourRepairError,
    build_pitch_contour_repair_report,
    validate_pitch_contour_repair_report,
)


def write_midi(path: Path, pitches: list[int]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    midi = pretty_midi.PrettyMIDI(initial_tempo=120)
    piano = pretty_midi.Instrument(program=0, is_drum=False, name="solo")
    for index, pitch in enumerate(pitches):
        start = index * 0.5
        piano.notes.append(pretty_midi.Note(velocity=84, pitch=pitch, start=start, end=start + 0.25))
    midi.instruments.append(piano)
    midi.write(str(path))
    return str(path)


def source_diagnostics() -> dict:
    return {
        "boundary": "stage_b_midi_to_solo_model_direct_phrase_quality_diagnostics",
        "readiness": {
            "boundary": "stage_b_midi_to_solo_model_direct_phrase_quality_diagnostics",
            "model_direct_generation_quality_claimed": False,
        },
        "aggregate": {
            "candidate_count": 3,
            "flag_counts": {
                "dead_air_gap": 3,
                "wide_interval_contour": 3,
                "wide_register_span": 3,
            },
            "max_interval_max": 82,
            "max_dead_air_ratio": 0.6522,
        },
        "decision": {
            "next_boundary": BOUNDARY,
            "critical_user_input_required": False,
        },
    }


def sequence_budget_repair() -> dict:
    return {
        "boundary": "stage_b_midi_to_solo_model_direct_sequence_budget_repair_smoke",
        "repair_result": {
            "previous_max_sequence": 96,
            "repaired_max_sequence": 160,
            "minimum_contract_tokens": 123,
            "repaired_direct_note_capacity": 33,
            "target_min_note_count": 24,
        },
        "readiness": {
            "model_direct_8bar_generation_probe_ready": True,
            "model_direct_generation_quality_claimed": False,
            "midi_to_solo_musical_quality_claimed": False,
            "human_audio_preference_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
        },
        "decision": {"next_boundary": "stage_b_midi_to_solo_model_direct_8bar_generation_probe"},
    }


def context_report() -> dict:
    return {
        "boundary": "stage_b_midi_to_solo_context_extraction_mvp",
        "summary": {
            "context_bars": 8,
            "context_event_count": 128,
            "unknown_chord_bar_count": 0,
            "low_confidence_bar_count": 4,
        },
        "context": {
            "bar_contexts": [
                {"bar_index": index, "tempo": 120.0, "chord_root": "C", "chord_quality": "maj7"}
                for index in range(8)
            ]
        },
        "readiness": {"context_extraction_completed": True},
    }


def scale_smoke(checkpoint_dir: Path) -> dict:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "checkpoint_epoch1.pt").write_bytes(b"stub")
    return {
        "checkpoint_dir": str(checkpoint_dir),
        "readiness": {
            "boundary": "stage_b_generic_base_training_scale_smoke",
            "training_scale_smoke_passed": True,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
        },
        "training_config": {"max_sequence": 160},
        "training": {"best_validation_loss": 6.1293},
        "artifacts": {"checkpoint_count": 1, "lora_weights_exists": True},
    }


def generation_report(root: Path) -> dict:
    midi_paths = [
        write_midi(root / "stage_b_sample_1.mid", [59, 67, 66, 72, 63, 67, 59, 60]),
        write_midi(root / "stage_b_sample_2.mid", [59, 60, 69, 72, 79, 77, 76, 67]),
        write_midi(root / "stage_b_sample_3.mid", [71, 76, 67, 60, 55, 62, 69, 77]),
    ]
    return {
        "summary": {
            "sample_count": 3,
            "valid_sample_count": 3,
            "strict_valid_sample_count": 3,
            "grammar_gate_sample_count": 3,
            "passed_generation_gate": True,
            "passed_grammar_gate": True,
            "passed_strict_review_gate": True,
            "collapse_warning_sample_count": 0,
            "collapse_warning_sample_rate": 0.0,
            "avg_postprocess_removal_ratio": 0.0,
        },
        "samples": [
            {
                "midi_path": path,
                "grammar": {"complete_note_groups": 24},
                "postprocess": {"before_note_count": 24, "after_note_count": 24},
                "metrics": {"note_count": 24},
            }
            for path in midi_paths
        ],
    }


class StageBMidiToSoloModelDirectPitchContourRepairTest(unittest.TestCase):
    def test_command_builder_passes_pitch_constraints(self) -> None:
        args = argparse.Namespace(
            issue_number=507,
            max_sequence=160,
            num_samples=3,
            seed=507,
            target_bars=8,
            note_groups_per_bar=3,
            chord_pitch_mode="tones_tensions",
            max_simultaneous_notes=1,
            min_valid_samples=1,
            min_strict_valid_samples=1,
            cap_duration_to_next_position=True,
            constrained_pitch_min=55,
            constrained_pitch_max=79,
            constrained_max_adjacent_interval=9,
        )
        command = build_generation_command(
            args=args,
            checkpoint_dir=Path("checkpoints"),
            generation_output_root=Path("out"),
            generation_run_id="repair",
            context_summary={"bpm": 120, "chord_progression": ["Cmaj7"] * 8},
        )

        self.assertIn("--constrained_pitch_min", command)
        self.assertIn("--constrained_pitch_max", command)
        self.assertIn("--constrained_max_adjacent_interval", command)
        self.assertIn("--cap_duration_to_next_position", command)

    def test_records_pitch_repair_and_routes_timing_followup(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            report = build_pitch_contour_repair_report(
                source_diagnostics=source_diagnostics(),
                sequence_budget_repair=sequence_budget_repair(),
                context_report=context_report(),
                repaired_training_scale_smoke=scale_smoke(root / "checkpoints"),
                generation_result={"returncode": 0, "cmd": [], "stdout_tail": "", "stderr_tail": ""},
                generation_report=generation_report(root / "samples"),
                generation_report_path=root / "generation" / "report.json",
                output_dir=root / "out",
                issue_number=507,
                target_bars=8,
                note_groups_per_bar=3,
                constrained_pitch_min=55,
                constrained_pitch_max=79,
                constrained_max_adjacent_interval=9,
                dead_air_threshold_seconds=0.5,
            )
            summary = validate_pitch_contour_repair_report(
                report,
                expected_boundary=BOUNDARY,
                expected_next_boundary=TIMING_REPAIR_BOUNDARY,
                require_repair_completed=True,
                require_pitch_repair_passed=True,
                require_no_quality_claim=True,
            )

        self.assertEqual(summary["strict_valid_sample_count"], 3)
        self.assertEqual(summary["previous_max_interval_max"], 82)
        self.assertLess(summary["repaired_max_interval_max"], 12)
        self.assertEqual(summary["repaired_wide_interval_flag_count"], 0)
        self.assertEqual(summary["repaired_wide_register_flag_count"], 0)
        self.assertTrue(summary["pitch_contour_repair_passed"])
        self.assertFalse(summary["model_direct_generation_quality_claimed"])
        self.assertFalse(summary["human_audio_preference_claimed"])

    def test_rejects_unrouted_source_diagnostics(self) -> None:
        source = source_diagnostics()
        source["decision"]["next_boundary"] = "other"
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with self.assertRaises(StageBMidiToSoloModelDirectPitchContourRepairError):
                build_pitch_contour_repair_report(
                    source_diagnostics=source,
                    sequence_budget_repair=sequence_budget_repair(),
                    context_report=context_report(),
                    repaired_training_scale_smoke=scale_smoke(root / "checkpoints"),
                    generation_result={"returncode": 0, "cmd": [], "stdout_tail": "", "stderr_tail": ""},
                    generation_report=generation_report(root / "samples"),
                    generation_report_path=root / "generation" / "report.json",
                    output_dir=root / "out",
                    issue_number=507,
                    target_bars=8,
                    note_groups_per_bar=3,
                    constrained_pitch_min=55,
                    constrained_pitch_max=79,
                    constrained_max_adjacent_interval=9,
                    dead_air_threshold_seconds=0.5,
                )


if __name__ == "__main__":
    unittest.main()
