from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pretty_midi

from scripts.summarize_stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_repair import (
    build_duration_coverage_fill_report,
    validate_duration_coverage_fill,
)


def write_source_midi(path: Path) -> None:
    midi = pretty_midi.PrettyMIDI(initial_tempo=124)
    piano = pretty_midi.Instrument(program=0, is_drum=False, name="coverage_adjacent_partial")
    rows = [
        (70, 0.120967, 0.483870),
        (77, 0.725805, 0.846772),
        (75, 0.967740, 1.088708),
        (79, 1.209675, 1.572578),
        (72, 1.572578, 1.693545),
        (65, 1.935480, 2.056448),
        (68, 2.298382, 2.419350),
        (67, 2.419350, 2.540318),
        (63, 2.903220, 3.024188),
        (70, 3.024188, 3.145155),
        (72, 3.266122, 3.387090),
        (65, 3.387090, 3.508058),
    ]
    for pitch, start, end in rows:
        piano.notes.append(pretty_midi.Note(velocity=84, pitch=pitch, start=start, end=end))
    midi.instruments.append(piano)
    path.parent.mkdir(parents=True, exist_ok=True)
    midi.write(str(path))


class StageBMarginRecoveredPhraseVocabularyDurationCoverageFillRepairTest(unittest.TestCase):
    def test_selects_minimal_qualified_duration_fill_candidate(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            midi_path = root / "partial.mid"
            source_report_path = root / "source_report.json"
            write_source_midi(midi_path)
            source_report_path.write_text(
                json.dumps(
                    {
                        "request": {
                            "bpm": 124,
                            "chord_progression": ["Cm7", "Fm7", "Bb7", "Ebmaj7"],
                            "bars": 2,
                            "density": "medium",
                            "energy": "mid",
                            "temperature": 0.82,
                            "top_k": 7,
                            "top_p": None,
                            "seed": 353,
                        }
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            previous_summary = {
                "output_dir": str(root / "previous"),
                "selected_candidate": {
                    "candidate_id": "margin_recovered_phrase_vocab_seed_353_topk_7_temp_082_n24_sample_3",
                    "midi_path": str(midi_path),
                    "source_report_path": str(source_report_path),
                    "metrics": {"dead_air_ratio": 0.5714285714285714},
                    "focused_solo_metrics": {
                        "focused_note_count": 12,
                        "focused_unique_pitch_count": 9,
                        "focused_adjacent_pitch_repeats": 0,
                        "focused_duplicated_3_note_pitch_class_chunks": 0,
                        "focused_max_interval": 7,
                    },
                },
            }

            report = build_duration_coverage_fill_report(
                previous_summary,
                output_dir=root / "fill",
                fill_max_additions=[4, 6, 8, 10],
                dead_air_threshold_sec=0.18,
                simultaneous_limit=1,
                min_unique_pitch_count=7,
                max_dead_air_ratio_exclusive=0.376,
                min_note_count=12,
                max_simultaneous_notes=1,
                max_duplicated_3_note_chunks=0,
                max_adjacent_pitch_repeats_exclusive=1,
                max_interval_exclusive=12,
            )
            summary = validate_duration_coverage_fill(
                report,
                require_qualified=True,
                require_dead_air_improvement=True,
                expected_fill_addition_count=6,
            )

            self.assertEqual(summary["qualified_variant_count"], 2)
            self.assertTrue(summary["qualified"])
            self.assertEqual(summary["selected_fill_addition_count"], 6)
            self.assertLess(summary["selected_dead_air_ratio"], 0.376)
            self.assertEqual(summary["selected_adjacent_pitch_repeats"], 0)
            self.assertEqual(summary["selected_duplicated_3_note_pitch_class_chunks"], 0)
            self.assertLess(summary["selected_max_interval"], 12)
            self.assertEqual(summary["claim_boundary"], "postprocess_duration_coverage_fill_candidate")


if __name__ == "__main__":
    unittest.main()
