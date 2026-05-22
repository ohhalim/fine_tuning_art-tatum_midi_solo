from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pretty_midi

from scripts.evaluate_generated_candidate_chords import (
    ManifestError,
    build_report_from_candidate_report,
    expand_chords,
    write_tiny_fixture,
)


def write_tiny_midi(path: Path) -> None:
    midi = pretty_midi.PrettyMIDI(initial_tempo=120)
    instrument = pretty_midi.Instrument(program=0)
    instrument.notes.append(pretty_midi.Note(velocity=80, pitch=60, start=0.0, end=0.25))
    instrument.notes.append(pretty_midi.Note(velocity=80, pitch=64, start=0.5, end=0.75))
    instrument.notes.append(pretty_midi.Note(velocity=80, pitch=67, start=1.0, end=1.25))
    instrument.notes.append(pretty_midi.Note(velocity=80, pitch=71, start=1.5, end=1.75))
    midi.instruments.append(instrument)
    midi.write(str(path))


class GeneratedCandidateChordEvalTest(unittest.TestCase):
    def test_expand_chords_cycles_progression_to_bar_count(self) -> None:
        self.assertEqual(expand_chords(["Cm7", "F7"], 5), ["Cm7", "F7", "Cm7", "F7", "Cm7"])

    def test_build_report_reads_review_manifest_chords(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            midi_path = root / "candidate.mid"
            write_tiny_midi(midi_path)
            manifest_path = root / "review_manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "chord_progression": ["Cmaj7"],
                        "bars": 1,
                        "bpm": 120,
                        "candidates": [{"mode": "fixture", "review_rank": 1, "review_midi_path": str(midi_path)}],
                    }
                ),
                encoding="utf-8",
            )

            report = build_report_from_candidate_report(manifest_path)

            self.assertEqual(report["summary"]["sample_count"], 1)
            self.assertEqual(report["summary"]["note_count"], 4)
            self.assertGreater(report["summary"]["role_ratios"]["chord_tone_ratio"], 0.0)

    def test_build_report_falls_back_to_source_request(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            midi_path = root / "candidate.mid"
            write_tiny_midi(midi_path)
            source_path = root / "source_report.json"
            source_path.write_text(
                json.dumps({"request": {"chord_progression": ["Cmaj7"], "bars": 1, "bpm": 120}}),
                encoding="utf-8",
            )
            manifest_path = root / "review_manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "source_report": str(source_path),
                        "candidates": [{"mode": "fixture", "review_rank": 1, "midi_path": str(midi_path)}],
                    }
                ),
                encoding="utf-8",
            )

            report = build_report_from_candidate_report(manifest_path)

            self.assertEqual(report["chord_progression"], ["Cmaj7"])
            self.assertEqual(report["bars"], 1)

    def test_build_report_rejects_missing_chords(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            midi_path = root / "candidate.mid"
            write_tiny_midi(midi_path)
            manifest_path = root / "review_manifest.json"
            manifest_path.write_text(
                json.dumps({"candidates": [{"review_midi_path": str(midi_path)}]}),
                encoding="utf-8",
            )

            with self.assertRaises(ManifestError):
                build_report_from_candidate_report(manifest_path)

    def test_write_tiny_fixture_builds_evaluable_report(self) -> None:
        with TemporaryDirectory() as tmp:
            report_path = write_tiny_fixture(Path(tmp))

            report = build_report_from_candidate_report(report_path)

            self.assertEqual(report["summary"]["sample_count"], 1)
            self.assertGreater(report["summary"]["note_count"], 0)


if __name__ == "__main__":
    unittest.main()
