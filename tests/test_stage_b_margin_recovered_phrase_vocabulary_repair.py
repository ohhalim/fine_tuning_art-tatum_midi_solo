from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pretty_midi

from scripts.summarize_stage_b_margin_recovered_phrase_vocabulary_repair import (
    build_repair_report,
    validate_repair,
)


def write_midi(path: Path, pitches: list[int]) -> None:
    midi = pretty_midi.PrettyMIDI(initial_tempo=124)
    piano = pretty_midi.Instrument(program=0, is_drum=False, name="phrase_vocab_candidate")
    for index, pitch in enumerate(pitches):
        start = index * 0.25
        piano.notes.append(pretty_midi.Note(velocity=84, pitch=pitch, start=start, end=start + 0.2))
    midi.instruments.append(piano)
    path.parent.mkdir(parents=True, exist_ok=True)
    midi.write(str(path))


def write_report(path: Path, *, run_id: str, seed: int, samples: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "run_dir": str(path.parent),
                "request": {"seed": seed, "top_k": 7, "temperature": 0.82},
                "summary": {"sample_count": len(samples)},
                "samples": samples,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


class StageBMarginRecoveredPhraseVocabularyRepairTest(unittest.TestCase):
    def test_prefers_adjacent_and_interval_repaired_candidate(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dead_air_edge_midi = root / "dead_air_edge.mid"
            wide_interval_midi = root / "wide_interval.mid"
            qualified_midi = root / "qualified.mid"
            write_midi(dead_air_edge_midi, [60, 62, 64, 65, 67, 69, 71, 72, 74, 76, 77, 79])
            write_midi(wide_interval_midi, [60, 76, 62, 64, 65, 67, 69, 71, 72, 74, 76, 77, 79])
            write_midi(qualified_midi, [60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71])
            report_path = root / "report.json"
            write_report(
                report_path,
                run_id="test_phrase_vocab_seed43_topk7",
                seed=43,
                samples=[
                    {
                        "sample_index": 1,
                        "sample_seed": 81,
                        "midi_path": str(dead_air_edge_midi),
                        "valid": True,
                        "strict_valid": True,
                        "grammar_gate_passed": True,
                        "metrics": {"dead_air_ratio": 0.40, "phrase_coverage_ratio": 0.8},
                        "temporal_coverage": {"onset_coverage_ratio": 0.5, "sustained_coverage_ratio": 0.7},
                    },
                    {
                        "sample_index": 2,
                        "sample_seed": 82,
                        "midi_path": str(wide_interval_midi),
                        "valid": True,
                        "strict_valid": True,
                        "grammar_gate_passed": True,
                        "metrics": {"dead_air_ratio": 0.33, "phrase_coverage_ratio": 0.8},
                        "temporal_coverage": {"onset_coverage_ratio": 0.5, "sustained_coverage_ratio": 0.7},
                    },
                    {
                        "sample_index": 3,
                        "sample_seed": 85,
                        "midi_path": str(qualified_midi),
                        "valid": True,
                        "strict_valid": True,
                        "grammar_gate_passed": True,
                        "metrics": {"dead_air_ratio": 0.33, "phrase_coverage_ratio": 0.8},
                        "temporal_coverage": {"onset_coverage_ratio": 0.5, "sustained_coverage_ratio": 0.7},
                    },
                ],
            )

            report = build_repair_report(
                [report_path],
                output_dir=root / "repair",
                previous_candidate_id="previous",
                previous_dead_air=0.35294117647058826,
                previous_unique_pitch_count=7,
                previous_note_count=14,
                previous_adjacent_pitch_repeats=2,
                previous_max_interval=16,
                min_unique_pitch_count=6,
                max_dead_air_ratio_exclusive=0.40,
                min_note_count=12,
                max_simultaneous_notes=1,
                max_duplicated_3_note_chunks=0,
                max_adjacent_pitch_repeats_exclusive=2,
                max_interval_exclusive=12,
            )
            summary = validate_repair(
                report,
                require_qualified=True,
                require_phrase_vocabulary_improvement=True,
                expected_source_run_id="test_phrase_vocab_seed43_topk7",
                expected_sample_index=3,
            )

            self.assertTrue(summary["qualified"])
            self.assertTrue(summary["phrase_vocabulary_improved"])
            self.assertEqual(summary["qualified_candidate_count"], 1)
            self.assertEqual(summary["selected_sample_index"], 3)
            self.assertEqual(summary["adjacent_pitch_repeat_delta_from_previous"], 2)
            self.assertGreater(summary["max_interval_delta_from_previous"], 0)


if __name__ == "__main__":
    unittest.main()
