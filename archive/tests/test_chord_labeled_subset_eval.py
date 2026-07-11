from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pretty_midi

from scripts.evaluate_chord_labeled_subset import (
    ManifestError,
    analyze_sample,
    build_report,
    load_groups_from_midi,
    validate_manifest,
)


class ChordLabeledSubsetEvalTest(unittest.TestCase):
    def test_validate_manifest_accepts_inline_notes(self) -> None:
        samples = validate_manifest(
            {
                "schema_version": "stage_b_chord_labeled_eval_v1",
                "samples": [
                    {
                        "sample_id": "ok",
                        "bar_count": 1,
                        "chords": ["Cmaj7"],
                        "notes": [{"bar": 0, "position": 0, "pitch": 60}],
                    }
                ],
            }
        )

        self.assertEqual(samples[0]["sample_id"], "ok")

    def test_validate_manifest_rejects_chord_count_mismatch(self) -> None:
        with self.assertRaises(ManifestError):
            validate_manifest(
                {
                    "schema_version": "stage_b_chord_labeled_eval_v1",
                    "samples": [
                        {
                            "sample_id": "bad",
                            "bar_count": 2,
                            "chords": ["Cmaj7"],
                            "notes": [{"bar": 0, "position": 0, "pitch": 60}],
                        }
                    ],
                }
            )

    def test_analyze_sample_counts_pitch_roles(self) -> None:
        sample = validate_manifest(
            {
                "schema_version": "stage_b_chord_labeled_eval_v1",
                "samples": [
                    {
                        "sample_id": "roles",
                        "bar_count": 1,
                        "chords": ["Cmaj7"],
                        "notes": [
                            {"bar": 0, "position": 0, "pitch": 60},
                            {"bar": 0, "position": 4, "pitch": 64},
                            {"bar": 0, "position": 8, "pitch": 66},
                        ],
                    }
                ],
            }
        )[0]

        report = analyze_sample(sample, manifest_path=Path("manifest.json"))

        self.assertEqual(report["note_count"], 3)
        self.assertEqual(report["role_counts"]["root"], 1)
        self.assertGreater(report["role_counts"]["guide"], 0)
        self.assertGreater(report["role_counts"]["tension"], 0)

    def test_load_groups_from_midi_quantizes_notes(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            midi_path = root / "phrase.mid"
            midi = pretty_midi.PrettyMIDI(initial_tempo=120)
            instrument = pretty_midi.Instrument(program=0)
            instrument.notes.append(pretty_midi.Note(velocity=80, pitch=60, start=0.0, end=0.25))
            instrument.notes.append(pretty_midi.Note(velocity=80, pitch=64, start=0.5, end=0.75))
            midi.instruments.append(instrument)
            midi.write(str(midi_path))

            sample = {
                "sample_id": "midi",
                "bar_count": 1,
                "bpm": 120,
                "chords": ["Cmaj7"],
                "midi_path": "phrase.mid",
            }

            groups = load_groups_from_midi(sample, manifest_path=root / "manifest.json")

            self.assertEqual(groups[0]["position"], 0)
            self.assertEqual(groups[1]["position"], 4)

    def test_build_report_reads_fixture_manifest(self) -> None:
        report = build_report(Path("data/eval/stage_b_chord_labeled_tiny/manifest.json"))

        self.assertEqual(report["summary"]["sample_count"], 2)
        self.assertGreater(report["summary"]["note_count"], 0)
        self.assertGreater(report["summary"]["role_ratios"]["chord_tone_ratio"], 0.5)

    def test_build_report_rejects_unknown_chord(self) -> None:
        with TemporaryDirectory() as tmp:
            manifest_path = Path(tmp) / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "schema_version": "stage_b_chord_labeled_eval_v1",
                        "samples": [
                            {
                                "sample_id": "bad_chord",
                                "bar_count": 1,
                                "chords": ["not-a-chord"],
                                "notes": [{"bar": 0, "position": 0, "pitch": 60}],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaises(ManifestError):
                build_report(manifest_path)


if __name__ == "__main__":
    unittest.main()
