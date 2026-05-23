from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pretty_midi

from scripts.build_clean_context_diagnostics import (
    build_clean_context_diagnostics,
    context_summary,
    markdown_report,
)


def write_midi(path: Path, notes: list[tuple[int, float, float]], *, name: str = "Solo") -> None:
    midi = pretty_midi.PrettyMIDI(initial_tempo=120)
    instrument = pretty_midi.Instrument(program=0, is_drum=False, name=name)
    for pitch, start, end in notes:
        instrument.notes.append(pretty_midi.Note(velocity=80, pitch=pitch, start=start, end=end))
    midi.instruments.append(instrument)
    midi.write(str(path))


def write_context(path: Path) -> None:
    midi = pretty_midi.PrettyMIDI(initial_tempo=120)
    chord = pretty_midi.Instrument(program=4, is_drum=False, name="Chord Guide")
    bass = pretty_midi.Instrument(program=32, is_drum=False, name="Bass Root Guide")
    solo = pretty_midi.Instrument(program=0, is_drum=False, name="Solo - Test")
    chord.notes.append(pretty_midi.Note(velocity=50, pitch=60, start=0.0, end=2.0))
    bass.notes.append(pretty_midi.Note(velocity=50, pitch=36, start=0.0, end=2.0))
    solo.notes.append(pretty_midi.Note(velocity=80, pitch=72, start=0.0, end=0.25))
    midi.instruments.extend([chord, bass, solo])
    midi.write(str(path))


class CleanContextDiagnosticsTest(unittest.TestCase):
    def test_build_clean_context_diagnostics_reports_phrase_metrics(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            solo_path = tmp_path / "solo.mid"
            context_path = tmp_path / "context.mid"
            notes = [(60 + (index % 7), index * 0.25, index * 0.25 + 0.2) for index in range(32)]
            write_midi(solo_path, notes)
            write_context(context_path)

            report = build_clean_context_diagnostics(
                {
                    "output_dir": "outputs/clean",
                    "candidates": [
                        {
                            "candidate_id": "clean_rank_1",
                            "mode": "data_motif_phrase_recovery",
                            "review_rank": 1,
                            "sample_index": 1,
                            "review_midi_path": str(solo_path),
                            "context_midi_path": str(context_path),
                            "metrics": {"chord_tone_ratio": 0.5, "tension_ratio": 0.5},
                        }
                    ],
                },
                output_dir=tmp_path / "diagnostics",
            )

            self.assertEqual(report["candidate_count"], 1)
            candidate = report["candidates"][0]
            self.assertEqual(candidate["solo_metrics"]["note_count"], 32)
            self.assertEqual(candidate["solo_metrics"]["covered_bar_count"], 4)
            self.assertEqual(candidate["context_summary"]["has_chord_guide"], True)
            self.assertEqual(candidate["context_summary"]["has_bass_guide"], True)
            self.assertIn("timing", candidate["review_checklist"])

    def test_context_summary_handles_missing_context(self) -> None:
        summary = context_summary(Path("missing_context.mid"))

        self.assertFalse(summary["context_exists"])
        self.assertEqual(summary["instrument_count"], 0)

    def test_markdown_report_lists_decision_hints(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            solo_path = tmp_path / "solo.mid"
            context_path = tmp_path / "context.mid"
            write_midi(solo_path, [(60, 0.0, 0.25), (62, 0.5, 0.75), (64, 1.0, 1.25)])
            write_context(context_path)
            report = build_clean_context_diagnostics(
                {
                    "candidates": [
                        {
                            "candidate_id": "short_candidate",
                            "review_midi_path": str(solo_path),
                            "context_midi_path": str(context_path),
                        }
                    ],
                },
                output_dir=tmp_path / "diagnostics",
            )

        markdown = markdown_report(report)

        self.assertIn("short_candidate", markdown)
        self.assertIn("too_sparse_for_phrase_review", markdown)
        self.assertIn("Review Checklist", markdown)


if __name__ == "__main__":
    unittest.main()
