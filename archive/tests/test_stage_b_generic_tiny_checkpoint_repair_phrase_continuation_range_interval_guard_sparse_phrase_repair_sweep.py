from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path

import mido

from scripts.run_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_repair_sweep import (
    BOUNDARY,
    SOURCE_BOUNDARY,
    StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardSparsePhraseRepairSweepError,
    build_sweep_report,
    validate_sparse_phrase_repair_decision,
    validate_sweep_report,
)


def write_test_midi(path: Path, notes: list[tuple[float, float, int]], ticks_per_beat: int = 120) -> None:
    midi = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    events: list[tuple[int, int, int]] = []
    for start_beats, duration_beats, pitch in notes:
        start = int(round(start_beats * ticks_per_beat))
        end = int(round((start_beats + duration_beats) * ticks_per_beat))
        events.append((start, 1, pitch))
        events.append((end, 0, pitch))
    events.sort(key=lambda item: (item[0], item[1]))
    cursor = 0
    for tick, on, pitch in events:
        delta = tick - cursor
        cursor = tick
        if on:
            track.append(mido.Message("note_on", note=pitch, velocity=80, time=delta))
        else:
            track.append(mido.Message("note_off", note=pitch, velocity=0, time=delta))
    midi.save(path)


def decision_report(*, quality_claimed: bool = False) -> dict:
    return {
        "schema_version": (
            "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_repair_decision_v1"
        ),
        "repair_decision": {
            "boundary": SOURCE_BOUNDARY,
            "primary_repair_target": "sparse_phrase_continuity_after_range_interval_guard",
            "target_thresholds": {
                "max_gap_ratio_to_window": 0.40,
                "max_internal_gap_beats": 0.75,
                "min_note_count": 10,
                "min_phrase_coverage_ratio": 0.90,
                "max_tail_empty_steps": 0,
                "max_abs_interval": 12,
            },
        },
        "readiness": {
            "human_audio_keep_claimed": False,
            "human_audio_preference_claimed": False,
            "musical_quality_claimed": quality_claimed,
            "quality_cause_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
        },
        "decision": {
            "next_boundary": BOUNDARY,
        },
        "observed_evidence": {
            "gap_ratio_max": 0.5312,
            "max_internal_gap_beats_max": 1.5,
        },
    }


def generation_sample(midi_path: Path) -> dict:
    return {
        "sample_index": 1,
        "sample_seed": 80,
        "midi_path": str(midi_path),
        "valid": True,
        "strict_valid": True,
        "grammar_gate_passed": True,
        "temporal_coverage": {
            "tail_empty_steps": 1,
        },
        "collapse": {
            "postprocess_removal_ratio": 0.20,
        },
        "metrics": {
            "note_count": 12,
            "phrase_coverage_ratio": 1.0,
            "max_simultaneous_notes": 1,
        },
    }


def args() -> argparse.Namespace:
    return argparse.Namespace(
        applied_max_tail_empty_steps=1,
        max_simultaneous_notes=1,
        max_postprocess_removal_ratio=0.49,
        max_pitch_span=24,
        max_large_interval_ratio=0.35,
        large_interval_threshold=12,
        severe_interval_threshold=24,
        phrase_window_beats=8.0,
        min_target_qualified=1,
        issue_number=435,
        num_samples=1,
        seed=80,
        temperature=0.74,
        top_k=5,
        note_groups_per_bar=10,
        interval_caps="9",
        coverage_aware_positions=True,
        coverage_position_window=0,
        pitch_min=48,
        pitch_max=84,
    )


class StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardSparsePhraseRepairSweepTest(
    unittest.TestCase
):
    def test_builds_sparse_phrase_sweep_report_with_target_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            midi_path = root / "candidate.mid"
            write_test_midi(
                midi_path,
                [(index * 0.5, 0.5, 60 + (index % 4) * 2) for index in range(12)],
            )
            generation_runs = [
                {
                    "interval_cap": 9,
                    "result": {"returncode": 0},
                    "report_path": str(root / "report.json"),
                    "report": {
                        "summary": {
                            "sample_count": 1,
                            "valid_sample_count": 1,
                            "strict_valid_sample_count": 1,
                            "grammar_gate_sample_count": 1,
                            "passed_generation_gate": True,
                            "passed_strict_generation_gate": True,
                            "collapse_warning_sample_rate": 0.0,
                        },
                        "samples": [generation_sample(midi_path)],
                    },
                }
            ]
            report = build_sweep_report(
                run_dir=root / "run",
                checkpoint_dir=root / "checkpoints",
                decision_report_path=root / "decision.json",
                decision_report=decision_report(),
                generation_runs=generation_runs,
                args=args(),
            )
            summary = validate_sweep_report(
                report,
                expected_boundary=BOUNDARY,
                min_generation_runs=1,
                min_candidate_count=1,
                min_target_qualified=1,
                require_gap_reduction=True,
                require_no_quality_claim=True,
            )

        self.assertTrue(summary["target_passed"])
        self.assertEqual(summary["target_qualified_count"], 1)
        self.assertTrue(summary["gap_ratio_reduced_vs_source_max"])
        self.assertLessEqual(summary["top_gap_ratio_to_window"], 0.40)
        self.assertFalse(summary["musical_quality_claimed"])

    def test_rejects_quality_claimed_decision(self) -> None:
        with self.assertRaises(
            StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardSparsePhraseRepairSweepError
        ):
            validate_sparse_phrase_repair_decision(decision_report(quality_claimed=True))


if __name__ == "__main__":
    unittest.main()
