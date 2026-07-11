from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pretty_midi

from scripts.build_stage_b_margin_recovered_pitch_vocab_focused_package import (
    DEFAULT_DECISION,
    build_pitch_vocab_focused_package,
    validate_package,
)


def write_midi(path: Path) -> None:
    midi = pretty_midi.PrettyMIDI(initial_tempo=124)
    piano = pretty_midi.Instrument(program=0, is_drum=False, name="pitch_vocab")
    for index, pitch in enumerate([63, 65, 67, 68, 70, 72, 70, 68, 67, 65, 63, 65, 67]):
        start = index * 0.25
        piano.notes.append(pretty_midi.Note(velocity=84, pitch=pitch, start=start, end=start + 0.2))
    midi.instruments.append(piano)
    path.parent.mkdir(parents=True, exist_ok=True)
    midi.write(str(path))


def write_report(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "request": {
                    "chord_progression": ["Cm7", "Fm7", "Bb7", "Ebmaj7"],
                    "bpm": 124,
                    "bars": 2,
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def sweep_summary(root: Path) -> dict:
    sample_path = root / "generation" / "samples" / "stage_b_sample_4.mid"
    write_midi(sample_path)
    write_report(root / "generation" / "report.json")
    return {
        "output_dir": str(root / "sweep"),
        "sweep_summary": {
            "selected_candidate_id": "pitch_vocab_candidate",
            "dead_air_delta_from_previous": -0.105882,
            "focused_unique_pitch_delta_from_previous": 1,
        },
        "selected_candidate": {
            "candidate_id": "pitch_vocab_candidate",
            "sample_index": 4,
            "sample_seed": 20,
            "source_run_id": "test_pitch_vocab_run",
            "midi_path": str(sample_path),
            "metrics": {
                "note_count": 16,
                "unique_pitch_count": 6,
                "dead_air_ratio": 0.4,
            },
            "temporal_coverage": {
                "onset_coverage_ratio": 0.5,
                "sustained_coverage_ratio": 0.625,
            },
            "focused_solo_metrics": {
                "focused_note_count": 13,
                "focused_unique_pitch_count": 6,
                "focused_adjacent_pitch_repeats": 3,
                "focused_duplicated_3_note_pitch_class_chunks": 0,
            },
            "pitch_vocab_gate": {
                "qualified": True,
                "flags": [],
            },
        },
    }


class StageBMarginRecoveredPitchVocabFocusedPackageTest(unittest.TestCase):
    def test_builds_focused_package_from_selected_sweep_candidate(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            package = build_pitch_vocab_focused_package(
                sweep_summary(root),
                output_dir=root / "package",
                decision=DEFAULT_DECISION,
            )
            summary = validate_package(
                package,
                expected_candidate_id="pitch_vocab_candidate",
                min_candidates=1,
            )

            self.assertEqual(summary["candidate_count"], 1)
            self.assertEqual(summary["candidate_ids"], ["pitch_vocab_candidate"])
            candidate = package["candidates"][0]
            self.assertTrue(Path(candidate["review_files"]["midi_path"]).exists())
            self.assertTrue(Path(candidate["review_files"]["context_midi_path"]).exists())
            self.assertEqual(candidate["listening"]["decision"], DEFAULT_DECISION)


if __name__ == "__main__":
    unittest.main()
