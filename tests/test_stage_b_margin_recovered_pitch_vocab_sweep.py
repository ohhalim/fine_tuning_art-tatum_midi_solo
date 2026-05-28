from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pretty_midi

from scripts.summarize_stage_b_margin_recovered_pitch_vocab_sweep import (
    build_sweep_report,
    validate_sweep,
)


def write_midi(path: Path, pitches: list[int]) -> None:
    midi = pretty_midi.PrettyMIDI(initial_tempo=124)
    piano = pretty_midi.Instrument(program=0, is_drum=False, name="pitch_vocab_candidate")
    for index, pitch in enumerate(pitches):
        start = index * 0.25
        piano.notes.append(pretty_midi.Note(velocity=84, pitch=pitch, start=start, end=start + 0.2))
    midi.instruments.append(piano)
    path.parent.mkdir(parents=True, exist_ok=True)
    midi.write(str(path))


def write_report(path: Path, *, run_id: str, samples: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        __import__("json").dumps(
            {
                "run_id": run_id,
                "run_dir": str(path.parent),
                "request": {"seed": 17, "top_k": 5, "temperature": 0.9},
                "summary": {"sample_count": len(samples)},
                "samples": samples,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


class StageBMarginRecoveredPitchVocabSweepTest(unittest.TestCase):
    def test_prefers_qualified_candidate_over_higher_unique_sparse_candidate(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            sparse_midi = root / "sparse.mid"
            qualified_midi = root / "qualified.mid"
            write_midi(sparse_midi, [60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70])
            write_midi(qualified_midi, [60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72])
            report_path = root / "report.json"
            write_report(
                report_path,
                run_id="test_pitch_vocab_seed17_topk5",
                samples=[
                    {
                        "sample_index": 1,
                        "sample_seed": 20,
                        "midi_path": str(sparse_midi),
                        "valid": True,
                        "strict_valid": True,
                        "grammar_gate_passed": True,
                        "metrics": {"dead_air_ratio": 0.37, "phrase_coverage_ratio": 0.7},
                        "temporal_coverage": {"onset_coverage_ratio": 0.5, "sustained_coverage_ratio": 0.7},
                    },
                    {
                        "sample_index": 2,
                        "sample_seed": 21,
                        "midi_path": str(qualified_midi),
                        "valid": True,
                        "strict_valid": True,
                        "grammar_gate_passed": True,
                        "metrics": {"dead_air_ratio": 0.40, "phrase_coverage_ratio": 0.8},
                        "temporal_coverage": {"onset_coverage_ratio": 0.5, "sustained_coverage_ratio": 0.6},
                    },
                ],
            )

            report = build_sweep_report(
                [report_path],
                output_dir=root / "sweep",
                previous_candidate_id="previous",
                previous_dead_air=0.29411764705882354,
                previous_unique_pitch_count=5,
                min_unique_pitch_count=6,
                max_dead_air_ratio=0.40,
                min_note_count=12,
                max_simultaneous_notes=1,
                max_duplicated_3_note_chunks=0,
            )
            summary = validate_sweep(
                report,
                require_qualified=True,
                expected_source_run_id="test_pitch_vocab_seed17_topk5",
                expected_sample_index=2,
            )

            self.assertTrue(summary["qualified"])
            self.assertEqual(summary["qualified_candidate_count"], 1)
            self.assertEqual(summary["selected_sample_index"], 2)
            self.assertEqual(summary["focused_unique_pitch_delta_from_previous"], 8)


if __name__ == "__main__":
    unittest.main()
