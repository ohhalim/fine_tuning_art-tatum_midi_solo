from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import mido

from scripts.fill_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_midi_note_failure_review import (
    StageBGenericTinyCheckpointRepairPhraseContinuationMidiNoteFailureReviewError,
    build_failure_review,
    validate_failure_review,
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


def audio_render_report(root: Path, *, quality_claimed: bool = False, pitches: list[int] | None = None) -> dict:
    midi_path = root / "stage_b_sample_1.mid"
    write_midi(midi_path, pitches or [38, 53, 29, 89, 29, 63, 60, 87, 53])
    return {
        "schema_version": "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_local_audio_render_attempt_v1",
        "audio_render_boundary": {
            "boundary": "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_local_audio_render_attempt",
            "render_attempted": True,
            "technical_wav_validation": True,
            "audio_rendered_quality_claimed": quality_claimed,
            "human_audio_preference_claimed": False,
            "musical_quality_claimed": False,
        },
        "rendered_audio_files": [
            {
                "review_rank": 1,
                "sample_seed": 62,
                "sample_index": 1,
                "source_midi_path": str(midi_path),
                "wav_file": {
                    "path": str(root / "rank_01.wav"),
                    "exists": True,
                    "duration_seconds": 9.326,
                    "sample_rate": 44100,
                    "sha256": "a" * 64,
                },
            }
        ],
    }


class StageBGenericTinyCheckpointRepairPhraseContinuationMidiNoteFailureReviewTest(unittest.TestCase):
    def test_records_midi_note_failure_without_keep_or_quality_claim(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            report = build_failure_review(
                audio_render_report(Path(temp_dir)),
                output_dir=Path(temp_dir) / "failure_review",
                reviewer="user",
                assessment="not musical",
                notes="MIDI note sequence shows random large jumps",
                max_pitch_span=24,
                max_abs_interval=12,
                max_large_interval_ratio=0.35,
                large_interval_threshold=12,
                severe_interval_threshold=24,
            )
            summary = validate_failure_review(
                report,
                expected_boundary="generic_tiny_checkpoint_repair_phrase_continuation_midi_note_failure_reject_all",
                expected_primary_failure="midi_note_random_large_leaps",
                expected_file_count=1,
                min_max_interval=24,
                require_no_keep_claim=True,
                require_no_quality_claim=True,
            )

            self.assertEqual(summary["overall_decision"], "reject_all")
            self.assertEqual(summary["max_abs_interval"], 60)
            self.assertEqual(summary["pitch_span"], 60)
            self.assertGreater(summary["large_interval_ratio"], 0.8)
            self.assertFalse(summary["human_audio_keep_claimed"])
            self.assertFalse(summary["musical_quality_claimed"])
            self.assertTrue(summary["auto_progress_allowed"])

    def test_rejects_source_quality_claim(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(StageBGenericTinyCheckpointRepairPhraseContinuationMidiNoteFailureReviewError):
                build_failure_review(
                    audio_render_report(Path(temp_dir), quality_claimed=True),
                    output_dir=Path(temp_dir) / "failure_review",
                    reviewer="user",
                    assessment="not musical",
                    notes="",
                    max_pitch_span=24,
                    max_abs_interval=12,
                    max_large_interval_ratio=0.35,
                    large_interval_threshold=12,
                    severe_interval_threshold=24,
                )

    def test_validation_rejects_non_failure_interval_profile(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            report = build_failure_review(
                audio_render_report(Path(temp_dir), pitches=[60, 62, 64, 65, 67, 69, 71, 72]),
                output_dir=Path(temp_dir) / "failure_review",
                reviewer="user",
                assessment="not musical",
                notes="",
                max_pitch_span=24,
                max_abs_interval=12,
                max_large_interval_ratio=0.35,
                large_interval_threshold=12,
                severe_interval_threshold=24,
            )

            with self.assertRaises(StageBGenericTinyCheckpointRepairPhraseContinuationMidiNoteFailureReviewError):
                validate_failure_review(
                    report,
                    expected_boundary="generic_tiny_checkpoint_repair_phrase_continuation_midi_note_failure_reject_all",
                    expected_primary_failure="midi_note_random_large_leaps",
                    expected_file_count=1,
                    min_max_interval=24,
                    require_no_keep_claim=True,
                    require_no_quality_claim=True,
                )


if __name__ == "__main__":
    unittest.main()
