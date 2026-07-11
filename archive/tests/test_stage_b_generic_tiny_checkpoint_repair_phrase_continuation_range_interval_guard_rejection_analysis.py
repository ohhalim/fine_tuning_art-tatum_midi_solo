from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import mido

from scripts.analyze_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_rejection import (
    BOUNDARY,
    NEXT_BOUNDARY,
    SOURCE_BOUNDARY,
    StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardRejectionAnalysisError,
    analyze_reviewed_candidate,
    build_rejection_analysis,
    validate_rejection_analysis,
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


def source_report(midi_paths: list[Path], *, keep_claimed: bool = False) -> dict:
    return {
        "schema_version": (
            "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_user_listening_review_v1"
        ),
        "reviewed_audio_files": [
            {
                "review_rank": index,
                "interval_cap": 9,
                "sample_seed": 70 + index,
                "sample_index": index,
                "source_midi_path": str(path),
                "wav_path": f"rank_{index}.wav",
                "duration_seconds": 7.0,
                "sample_rate": 44100,
                "sha256": str(index) * 64,
            }
            for index, path in enumerate(midi_paths, start=1)
        ],
        "user_listening_review": {
            "status": "reviewed",
            "overall_decision": "reject_all",
            "candidate_decision": "reject",
            "primary_failure": "subjective_not_musical",
        },
        "claim_boundary": {
            "boundary": SOURCE_BOUNDARY,
            "human_audio_reject_all_recorded": True,
            "human_audio_keep_claimed": keep_claimed,
            "human_audio_preference_claimed": False,
            "audio_rendered_quality_claimed": False,
            "musical_quality_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
        },
        "decision": {
            "next_boundary": BOUNDARY,
        },
    }


class StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardRejectionAnalysisTest(
    unittest.TestCase
):
    def test_analyzes_reject_all_candidates_without_quality_claim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            midi_paths = [root / f"candidate_{index}.mid" for index in range(3)]
            for path in midi_paths:
                write_test_midi(
                    path,
                    [
                        (0.0, 0.25, 60),
                        (1.5, 0.25, 60),
                        (3.0, 0.25, 67),
                        (4.5, 0.25, 60),
                        (6.0, 0.25, 67),
                    ],
                )

            report = build_rejection_analysis(
                source_report(midi_paths),
                output_dir=root / "out",
                expected_file_count=3,
                phrase_window_beats=8.0,
                sparse_gap_ratio=0.45,
                long_gap_beats=1.0,
            )
            summary = validate_rejection_analysis(
                report,
                expected_boundary=BOUNDARY,
                expected_candidate_count=3,
                require_reject_all_source=True,
                require_no_quality_claim=True,
                min_common_evidence_flags=1,
            )

        self.assertEqual(summary["boundary"], BOUNDARY)
        self.assertEqual(summary["candidate_count"], 3)
        self.assertEqual(summary["next_boundary"], NEXT_BOUNDARY)
        self.assertIn("high_dead_air_or_sparse_phrase", summary["common_evidence_flags"])
        self.assertEqual(
            summary["primary_next_repair_target"],
            "sparse_phrase_continuity_after_range_interval_guard",
        )
        self.assertFalse(summary["quality_cause_claimed"])
        self.assertTrue(summary["auto_progress_allowed"])

    def test_candidate_analysis_records_repetition_and_gap_flags(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "candidate.mid"
            write_test_midi(
                path,
                [
                    (0.0, 0.25, 60),
                    (1.5, 0.25, 60),
                    (3.0, 0.25, 67),
                    (4.5, 0.25, 60),
                    (6.0, 0.25, 67),
                ],
            )
            candidate = analyze_reviewed_candidate(
                {
                    "review_rank": 1,
                    "interval_cap": 9,
                    "sample_seed": 71,
                    "sample_index": 1,
                    "source_midi_path": str(path),
                    "wav_path": "candidate.wav",
                },
                phrase_window_beats=8.0,
                sparse_gap_ratio=0.45,
                long_gap_beats=1.0,
            )

        self.assertIn("high_dead_air_or_sparse_phrase", candidate["evidence_flags"])
        self.assertIn("long_internal_gap_present", candidate["evidence_flags"])
        self.assertIn("adjacent_pitch_repeat_present", candidate["evidence_flags"])
        self.assertIn("pitch_cell_repetition_present", candidate["evidence_flags"])

    def test_rejects_source_keep_claim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "candidate.mid"
            write_test_midi(path, [(0.0, 0.25, 60), (1.0, 0.25, 62)])

            with self.assertRaises(
                StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardRejectionAnalysisError
            ):
                build_rejection_analysis(
                    source_report([path], keep_claimed=True),
                    output_dir=Path(tmp) / "out",
                    expected_file_count=1,
                    phrase_window_beats=8.0,
                    sparse_gap_ratio=0.45,
                    long_gap_beats=1.0,
                )


if __name__ == "__main__":
    unittest.main()
