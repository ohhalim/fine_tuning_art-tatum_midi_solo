from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path

import mido

from scripts.run_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sweep import (
    BOUNDARY,
    StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardSweepError,
    build_sweep_report,
    compact_candidate,
    validate_guard_decision,
    validate_sweep_report,
)


def write_midi(path: Path, pitches: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    midi = mido.MidiFile(ticks_per_beat=220)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    track.append(mido.Message("program_change", program=0, time=0))
    for index, pitch in enumerate(pitches):
        delta = 55 if index else 0
        track.append(mido.Message("note_on", note=pitch, velocity=64, time=delta))
        track.append(mido.Message("note_off", note=pitch, velocity=0, time=55))
    track.append(mido.MetaMessage("end_of_track", time=0))
    midi.save(path)


def decision_report(*, next_boundary: str = BOUNDARY) -> dict:
    return {
        "schema_version": "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_decision_v1",
        "readiness": {
            "boundary": "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_decision",
            "musical_quality_claimed": False,
        },
        "decision": {
            "next_boundary": next_boundary,
        },
        "guard_targets": {
            "max_pitch_span": 24,
            "max_abs_interval": 12,
            "max_large_interval_ratio": 0.35,
            "max_severe_interval_count": 0,
            "preferred_pitch_floor": 48,
            "preferred_pitch_ceiling": 84,
            "large_interval_threshold": 12,
            "severe_interval_threshold": 24,
        },
    }


def args() -> argparse.Namespace:
    return argparse.Namespace(
        issue_number=423,
        max_sequence=160,
        num_samples=2,
        seed=62,
        temperature=0.78,
        top_k=5,
        min_valid_samples=1,
        min_strict_valid_samples=1,
        note_groups_per_bar=8,
        chord_pitch_mode="tones_tensions",
        chord_pitch_repeat_window=2,
        interval_caps="9",
        max_simultaneous_notes=1,
        min_note_count=8,
        min_phrase_coverage_ratio=0.85,
        max_tail_empty_steps=2,
        max_postprocess_removal_ratio=0.49,
        max_severe_interval_count=0,
        min_target_qualified=1,
    )


def sample_row(path: Path, *, strict: bool = True, coverage: float = 0.9) -> dict:
    return {
        "sample_index": 1,
        "sample_seed": 70,
        "midi_path": str(path),
        "valid": True,
        "strict_valid": strict,
        "grammar_gate_passed": True,
        "metrics": {
            "note_count": 9,
            "phrase_coverage_ratio": coverage,
            "dead_air_ratio": 0.75,
            "max_simultaneous_notes": 1,
        },
        "temporal_coverage": {
            "tail_empty_steps": 0,
            "position_span_ratio": 1.0,
        },
        "collapse": {
            "postprocess_removal_ratio": 0.3125,
        },
        "pitch_roles": {
            "chord_tone_ratio": 0.75,
        },
    }


def generation_report(row: dict) -> dict:
    return {
        "summary": {
            "sample_count": 1,
            "valid_sample_count": 1,
            "strict_valid_sample_count": 1 if row["strict_valid"] else 0,
            "grammar_gate_sample_count": 1,
            "passed_generation_gate": True,
            "passed_strict_generation_gate": True,
            "collapse_warning_sample_rate": 0.0,
        },
        "samples": [row],
    }


class StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardSweepTest(unittest.TestCase):
    def test_compact_candidate_uses_actual_midi_note_guard(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            midi_path = Path(temp_dir) / "sample.mid"
            write_midi(midi_path, [53, 60, 67, 60, 58, 63, 67, 65, 62])

            candidate = compact_candidate(
                sample_row(midi_path),
                interval_cap=9,
                args=args(),
                targets=validate_guard_decision(decision_report()),
            )

            self.assertTrue(candidate["target_qualified"])
            self.assertEqual(candidate["midi_note_audit"]["pitch_span"], 14)
            self.assertEqual(candidate["midi_note_audit"]["max_abs_interval"], 7)
            self.assertEqual(candidate["target_failure_reasons"], [])

    def test_builds_sweep_report_with_target_candidate_without_quality_claim(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            midi_path = Path(temp_dir) / "sample.mid"
            write_midi(midi_path, [53, 60, 67, 60, 58, 63, 67, 65, 62])
            run_args = args()
            report = build_sweep_report(
                run_dir=Path(temp_dir) / "run",
                checkpoint_dir=Path(temp_dir) / "checkpoints",
                decision_report_path=Path(temp_dir) / "decision.json",
                decision_report=decision_report(),
                generation_runs=[
                    {
                        "interval_cap": 9,
                        "result": {"returncode": 0},
                        "report_path": str(Path(temp_dir) / "report.json"),
                        "report": generation_report(sample_row(midi_path)),
                    }
                ],
                args=run_args,
            )
            summary = validate_sweep_report(
                report,
                expected_boundary=BOUNDARY,
                min_generation_runs=1,
                min_candidate_count=1,
                min_target_qualified=1,
                require_no_quality_claim=True,
            )

            self.assertTrue(summary["target_passed"])
            self.assertEqual(summary["target_qualified_count"], 1)
            self.assertEqual(summary["top_interval_cap"], 9)
            self.assertFalse(summary["musical_quality_claimed"])
            self.assertFalse(summary["critical_user_input_required"])

    def test_records_failed_target_candidate_as_tuning_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            midi_path = Path(temp_dir) / "sample.mid"
            write_midi(midi_path, [48, 72, 51, 75, 53, 77, 55, 79])
            run_args = args()
            report = build_sweep_report(
                run_dir=Path(temp_dir) / "run",
                checkpoint_dir=Path(temp_dir) / "checkpoints",
                decision_report_path=Path(temp_dir) / "decision.json",
                decision_report=decision_report(),
                generation_runs=[
                    {
                        "interval_cap": 12,
                        "result": {"returncode": 0},
                        "report_path": str(Path(temp_dir) / "report.json"),
                        "report": generation_report(sample_row(midi_path)),
                    }
                ],
                args=run_args,
            )
            summary = validate_sweep_report(
                report,
                expected_boundary=BOUNDARY,
                min_generation_runs=1,
                min_candidate_count=1,
                min_target_qualified=0,
                require_no_quality_claim=True,
            )

            self.assertFalse(summary["target_passed"])
            self.assertEqual(summary["target_qualified_count"], 0)
            self.assertIn("sweep_tuning", summary["next_boundary"])
            top = report["range_interval_guard"]["ranked_candidates"][0]
            self.assertIn("max_interval_above_target", top["target_failure_reasons"])

    def test_rejects_unexpected_decision_boundary(self) -> None:
        with self.assertRaises(StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardSweepError):
            validate_guard_decision(decision_report(next_boundary="wrong_boundary"))


if __name__ == "__main__":
    unittest.main()
