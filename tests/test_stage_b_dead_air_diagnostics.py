from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pretty_midi

from scripts.diagnose_stage_b_dead_air_outliers import (
    build_summary,
    compact_sample,
    markdown_report,
    start_gap_rows,
)


class StageBDeadAirDiagnosticsTest(unittest.TestCase):
    def test_start_gap_rows_marks_dead_air_gaps(self) -> None:
        notes = [
            SimpleNamespace(start=0.0, end=0.1, pitch=60),
            SimpleNamespace(start=0.12, end=0.2, pitch=62),
            SimpleNamespace(start=0.4, end=0.5, pitch=64),
        ]

        gaps = start_gap_rows(notes, threshold_sec=0.18)

        self.assertEqual(len(gaps), 2)
        self.assertFalse(gaps[0]["is_dead_air_gap"])
        self.assertTrue(gaps[1]["is_dead_air_gap"])
        self.assertAlmostEqual(gaps[1]["start_gap_sec"], 0.28)

    def test_compact_sample_summarizes_outlier_midi(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            midi_path = Path(tmp) / "sample.mid"
            midi = pretty_midi.PrettyMIDI(initial_tempo=124)
            instrument = pretty_midi.Instrument(program=0, is_drum=False)
            instrument.notes.extend(
                [
                    pretty_midi.Note(velocity=80, pitch=60, start=0.0, end=0.1),
                    pretty_midi.Note(velocity=82, pitch=62, start=0.25, end=0.35),
                    pretty_midi.Note(velocity=84, pitch=64, start=0.5, end=0.6),
                ]
            )
            midi.instruments.append(instrument)
            midi.write(str(midi_path))

            row = compact_sample(
                {
                    "sample_index": 1,
                    "midi_path": str(midi_path),
                    "valid": False,
                    "strict_valid": False,
                    "failure_reason": "dead-air ratio too high: 1.000 >= 0.800",
                    "metrics": {
                        "note_count": 3,
                        "unique_pitch_count": 3,
                        "dead_air_ratio": 1.0,
                        "phrase_coverage_ratio": 0.5,
                    },
                    "temporal_coverage": {
                        "onset_coverage_ratio": 0.2,
                        "sustained_coverage_ratio": 0.3,
                        "position_span_ratio": 0.5,
                        "head_empty_steps": 2,
                        "tail_empty_steps": 8,
                        "longest_onset_empty_run_steps": 8,
                        "longest_sustained_empty_run_steps": 6,
                    },
                    "collapse": {
                        "postprocess_removal_ratio": 0.2,
                        "collapse_warning": False,
                        "repeated_position_pitch_pair_ratio": 0.0,
                    },
                    "grammar": {"grammar_valid": False, "invalid_token_count": 1},
                    "postprocess": {"removed_note_count": 1},
                    "generated_token_names_head": ["POSITION_0", "VELOCITY_3"],
                },
                threshold_sec=0.18,
                dead_air_gate=0.8,
            )

        self.assertTrue(row["dead_air_outlier"])
        self.assertEqual(row["dead_air_gap_count"], 2)
        self.assertEqual(row["postprocess_removed_note_count"], 1)
        self.assertEqual(row["note_rows_head"][0]["pitch"], 60)

    def test_markdown_report_lists_outlier_detail(self) -> None:
        sample = {
            "sample_index": 1,
            "valid": False,
            "strict_valid": False,
            "note_count": 8,
            "unique_pitch_count": 4,
            "dead_air_ratio": 0.857,
            "phrase_coverage_ratio": 0.469,
            "onset_coverage_ratio": 0.25,
            "sustained_coverage_ratio": 0.375,
            "position_span_ratio": 0.469,
            "head_empty_steps": 6,
            "tail_empty_steps": 11,
            "longest_sustained_empty_run_steps": 10,
            "postprocess_removed_note_count": 3,
            "max_start_gap_sec": 0.484,
            "failure_reason": "dead-air ratio too high",
            "dead_air_outlier": True,
            "dead_air_gap_count": 6,
            "gap_count": 7,
            "generated_token_names_head": ["NOTE_DURATION_3", "POSITION_4"],
            "dead_air_gaps": [
                {
                    "from_note": 1,
                    "to_note": 2,
                    "from_pitch": 60,
                    "to_pitch": 62,
                    "start_gap_sec": 0.25,
                    "silent_gap_sec": 0.15,
                }
            ],
        }
        report = {
            "source_report_path": "report.json",
            "dead_air_gate": 0.8,
            "dead_air_threshold_sec": 0.18,
            "summary": build_summary([sample]),
            "samples": [sample],
        }

        markdown = markdown_report(report)

        self.assertIn("outlier sample indices", markdown)
        self.assertIn("Sample 1", markdown)
        self.assertIn("dead-air gaps `6/7`", markdown)


if __name__ == "__main__":
    unittest.main()
