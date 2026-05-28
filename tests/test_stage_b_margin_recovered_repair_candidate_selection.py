from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pretty_midi

from scripts.select_stage_b_margin_recovered_repair_candidate import (
    build_selection_report,
    validate_selection,
)


def write_midi(path: Path, pitches: list[int]) -> None:
    midi = pretty_midi.PrettyMIDI(initial_tempo=124)
    piano = pretty_midi.Instrument(program=0, is_drum=False, name="repair_candidate")
    for index, pitch in enumerate(pitches):
        start = index * 0.25
        piano.notes.append(pretty_midi.Note(velocity=84, pitch=pitch, start=start, end=start + 0.2))
    midi.instruments.append(piano)
    path.parent.mkdir(parents=True, exist_ok=True)
    midi.write(str(path))


def sample_report(root: Path) -> dict:
    first = root / "sample_1.mid"
    second = root / "sample_2.mid"
    write_midi(first, [60, 62, 64, 65])
    write_midi(second, [60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71])
    return {
        "run_id": "test_repair",
        "run_dir": str(root),
        "request": {"seed": 31},
        "samples": [
            {
                "sample_index": 1,
                "sample_seed": 31,
                "midi_path": str(first),
                "valid": True,
                "strict_valid": True,
                "grammar_gate_passed": True,
                "metrics": {"dead_air_ratio": 0.35, "phrase_coverage_ratio": 0.5},
                "temporal_coverage": {"onset_coverage_ratio": 0.3, "sustained_coverage_ratio": 0.4},
            },
            {
                "sample_index": 2,
                "sample_seed": 32,
                "midi_path": str(second),
                "valid": True,
                "strict_valid": True,
                "grammar_gate_passed": True,
                "metrics": {"dead_air_ratio": 0.25, "phrase_coverage_ratio": 0.8},
                "temporal_coverage": {"onset_coverage_ratio": 0.6, "sustained_coverage_ratio": 0.8},
            },
        ],
    }


class StageBMarginRecoveredRepairSelectionTest(unittest.TestCase):
    def test_selects_candidate_that_improves_dead_air_and_pitch_variety(self) -> None:
        with TemporaryDirectory() as temp_dir:
            report = build_selection_report(
                sample_report(Path(temp_dir)),
                output_dir=Path(temp_dir) / "selection",
                baseline_candidate_id="baseline",
                baseline_dead_air=0.44,
                baseline_unique_pitch_count=4,
            )
            summary = validate_selection(report, expected_sample_index=2, require_partial_repair=True)

            self.assertEqual(summary["selected_sample_index"], 2)
            self.assertGreater(summary["dead_air_delta"], 0.0)
            self.assertGreater(summary["focused_unique_pitch_delta"], 0)
            self.assertTrue(summary["focused_keep_ready"])


if __name__ == "__main__":
    unittest.main()
