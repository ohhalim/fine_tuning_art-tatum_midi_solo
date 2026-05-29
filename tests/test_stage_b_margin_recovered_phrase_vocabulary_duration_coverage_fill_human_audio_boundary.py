from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pretty_midi

from scripts.build_stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_human_audio_boundary import (
    DurationCoverageFillHumanAudioBoundaryError,
    build_human_audio_boundary,
    validate_human_audio_boundary,
)


def write_midi(path: Path, rows: list[tuple[int, float, float]]) -> None:
    midi = pretty_midi.PrettyMIDI(initial_tempo=124)
    piano = pretty_midi.Instrument(program=0, is_drum=False, name="solo")
    for pitch, start, end in rows:
        piano.notes.append(pretty_midi.Note(velocity=76, pitch=pitch, start=start, end=end))
    midi.instruments.append(piano)
    path.parent.mkdir(parents=True, exist_ok=True)
    midi.write(str(path))


def keep_consolidation(selected_path: Path) -> dict:
    return {
        "schema_version": "stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_keep_consolidation_v1",
        "candidate": {
            "candidate_id": "duration_fill_candidate",
            "decision": "keep",
            "context_midi_path": "outputs/context.mid",
        },
        "duration_coverage_repair": {
            "fill_addition_count": 1,
            "dead_air_delta_from_baseline": 0.25,
        },
        "evidence_boundary": {
            "boundary": "single_postprocess_candidate_keep_support",
        },
        "selected_path": str(selected_path),
    }


def duration_fill_summary(source_path: Path, selected_path: Path) -> dict:
    return {
        "schema_version": "stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_repair_v1",
        "source_candidate": {
            "candidate_id": "source_candidate",
            "midi_path": str(source_path),
            "metrics": {
                "note_count": 2,
                "unique_pitch_count": 2,
                "dead_air_ratio": 0.5,
                "max_simultaneous_notes": 1,
            },
            "focused_solo_metrics": {
                "focused_note_count": 2,
                "focused_unique_pitch_count": 2,
                "focused_adjacent_pitch_repeats": 0,
                "focused_duplicated_3_note_pitch_class_chunks": 0,
                "focused_max_simultaneous_notes": 1,
                "focused_max_interval": 2,
            },
        },
        "selected_candidate": {
            "candidate_id": "duration_fill_candidate",
            "midi_path": str(selected_path),
            "metrics": {
                "note_count": 3,
                "unique_pitch_count": 3,
                "dead_air_ratio": 0.25,
                "max_simultaneous_notes": 1,
            },
            "focused_solo_metrics": {
                "focused_note_count": 3,
                "focused_unique_pitch_count": 3,
                "focused_adjacent_pitch_repeats": 0,
                "focused_duplicated_3_note_pitch_class_chunks": 0,
                "focused_max_simultaneous_notes": 1,
                "focused_max_interval": 2,
            },
        },
    }


class StageBMarginRecoveredPhraseVocabularyDurationCoverageFillHumanAudioBoundaryTest(unittest.TestCase):
    def test_builds_pending_source_vs_fill_boundary(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_path = root / "source.mid"
            selected_path = root / "selected.mid"
            write_midi(source_path, [(60, 0.0, 0.2), (64, 0.5, 0.7)])
            write_midi(selected_path, [(60, 0.0, 0.2), (62, 0.25, 0.4), (64, 0.5, 0.7)])

            report = build_human_audio_boundary(
                keep_consolidation(selected_path),
                duration_fill_summary(source_path, selected_path),
                output_dir=root / "human_audio",
            )
            summary = validate_human_audio_boundary(
                report,
                expected_candidate_id="duration_fill_candidate",
                require_pending=True,
                require_no_preference=True,
                expect_distinct_midi_content=True,
            )

            self.assertEqual(summary["review_item_count"], 2)
            self.assertEqual(summary["human_status"], "pending")
            self.assertFalse(summary["preference_claimed"])
            self.assertFalse(summary["note_sequence_match"])
            self.assertEqual(summary["fill_addition_count"], 1)
            self.assertEqual(
                summary["boundary"],
                "pending_human_audio_review_source_vs_fill_distinct_midi_content",
            )

    def test_rejects_same_midi_when_distinct_required(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_path = root / "source.mid"
            selected_path = root / "selected.mid"
            rows = [(60, 0.0, 0.2), (64, 0.5, 0.7)]
            write_midi(source_path, rows)
            write_midi(selected_path, rows)

            report = build_human_audio_boundary(
                keep_consolidation(selected_path),
                duration_fill_summary(source_path, selected_path),
                output_dir=root / "human_audio",
            )

            with self.assertRaises(DurationCoverageFillHumanAudioBoundaryError):
                validate_human_audio_boundary(
                    report,
                    expected_candidate_id="duration_fill_candidate",
                    require_pending=True,
                    require_no_preference=True,
                    expect_distinct_midi_content=True,
                )


if __name__ == "__main__":
    unittest.main()
