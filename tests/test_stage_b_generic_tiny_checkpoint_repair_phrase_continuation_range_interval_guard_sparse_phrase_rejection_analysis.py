from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import mido

from scripts.analyze_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_rejection import (
    BOUNDARY,
    NEXT_BOUNDARY,
    SOURCE_BOUNDARY,
    StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardSparsePhraseRejectionAnalysisError,
    build_sparse_phrase_rejection_analysis,
    validate_sparse_phrase_rejection_analysis,
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
            "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_"
            "sparse_phrase_user_listening_review_v1"
        ),
        "reviewed_audio_files": [
            {
                "review_rank": index,
                "interval_cap": 5,
                "sample_seed": 80 + index,
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


class StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardSparsePhraseRejectionAnalysisTest(
    unittest.TestCase
):
    def test_records_objective_proxy_gap_without_quality_claim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            midi_paths = [root / f"candidate_{index}.mid" for index in range(3)]
            write_test_midi(
                midi_paths[0],
                [
                    (0.0, 0.25, 60),
                    (0.25, 0.5, 62),
                    (0.75, 0.75, 64),
                    (1.5, 0.25, 65),
                    (1.75, 1.0, 67),
                    (2.75, 0.5, 69),
                    (3.25, 0.75, 71),
                    (4.0, 1.0, 72),
                    (5.0, 0.25, 70),
                    (5.25, 0.5, 68),
                    (5.75, 0.75, 66),
                    (6.5, 1.5, 63),
                ],
            )
            for path in midi_paths[1:]:
                write_test_midi(
                    path,
                    [
                        (0.0, 0.25, 60),
                        (0.5, 0.25, 60),
                        (1.0, 0.25, 60),
                        (1.5, 0.25, 62),
                        (2.0, 0.25, 62),
                    ],
                )

            report = build_sparse_phrase_rejection_analysis(
                source_report(midi_paths),
                output_dir=root / "out",
                expected_file_count=3,
                phrase_window_beats=8.0,
                sparse_gap_ratio=0.45,
                long_gap_beats=1.0,
            )
            summary = validate_sparse_phrase_rejection_analysis(
                report,
                expected_boundary=BOUNDARY,
                expected_candidate_count=3,
                require_reject_all_source=True,
                require_no_quality_claim=True,
                require_proxy_gap=True,
            )

        self.assertEqual(summary["boundary"], BOUNDARY)
        self.assertEqual(summary["candidate_count"], 3)
        self.assertEqual(summary["next_boundary"], NEXT_BOUNDARY)
        self.assertIn(1, summary["candidates_without_evidence_flags"])
        self.assertTrue(summary["objective_proxy_gap_recorded"])
        self.assertEqual(
            summary["primary_next_review_target"],
            "model_core_review_after_objective_proxy_gap",
        )
        self.assertFalse(summary["quality_cause_claimed"])
        self.assertTrue(summary["auto_progress_allowed"])

    def test_rejects_source_keep_claim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "candidate.mid"
            write_test_midi(path, [(0.0, 0.25, 60), (1.0, 0.25, 62), (2.0, 0.25, 64)])

            with self.assertRaises(
                StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardSparsePhraseRejectionAnalysisError
            ):
                build_sparse_phrase_rejection_analysis(
                    source_report([path], keep_claimed=True),
                    output_dir=Path(tmp) / "out",
                    expected_file_count=1,
                    phrase_window_beats=8.0,
                    sparse_gap_ratio=0.45,
                    long_gap_beats=1.0,
                )


if __name__ == "__main__":
    unittest.main()
