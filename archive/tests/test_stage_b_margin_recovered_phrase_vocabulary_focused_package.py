from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pretty_midi

from scripts.build_stage_b_margin_recovered_phrase_vocabulary_focused_package import (
    DEFAULT_DECISION,
    build_phrase_vocabulary_focused_package,
    validate_package,
)


def write_midi(path: Path) -> None:
    midi = pretty_midi.PrettyMIDI(initial_tempo=124)
    piano = pretty_midi.Instrument(program=0, is_drum=False, name="phrase_vocabulary")
    for index, pitch in enumerate([67, 68, 70, 72, 74, 76, 74, 72, 71, 69, 68, 70, 72]):
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


def repair_summary(root: Path) -> dict:
    sample_path = root / "generation" / "samples" / "stage_b_sample_43.mid"
    peer_path = root / "generation" / "samples" / "stage_b_sample_25.mid"
    write_midi(sample_path)
    write_midi(peer_path)
    write_report(root / "generation" / "report.json")
    selected = {
        "candidate_id": "phrase_vocabulary_candidate",
        "sample_index": 43,
        "sample_seed": 85,
        "source_run_id": "test_phrase_vocabulary_run",
        "midi_path": str(sample_path),
        "metrics": {
            "note_count": 16,
            "unique_pitch_count": 8,
            "dead_air_ratio": 0.33333333333333337,
        },
        "temporal_coverage": {
            "onset_coverage_ratio": 0.5,
            "sustained_coverage_ratio": 0.59375,
        },
        "focused_solo_metrics": {
            "focused_note_count": 13,
            "focused_unique_pitch_count": 8,
            "focused_adjacent_pitch_repeats": 0,
            "focused_max_interval": 7,
            "focused_duplicated_3_note_pitch_class_chunks": 0,
        },
        "phrase_vocabulary_gate": {
            "qualified": True,
            "flags": [],
        },
    }
    peer = {**selected, "candidate_id": "phrase_vocabulary_peer", "sample_index": 25, "midi_path": str(peer_path)}
    return {
        "output_dir": str(root / "repair"),
        "repair_summary": {
            "selected_candidate_id": "phrase_vocabulary_candidate",
            "dead_air_delta_from_previous": 0.019608,
            "adjacent_pitch_repeat_delta_from_previous": 2,
            "max_interval_delta_from_previous": 9,
            "focused_unique_pitch_delta_from_previous": 1,
            "focused_note_delta_from_previous": -1,
        },
        "selected_candidate": selected,
        "candidates": [selected, peer],
    }


def duration_fill_summary(root: Path) -> dict:
    sample_path = root / "duration" / "midi" / "duration_fill.mid"
    write_midi(sample_path)
    source_report_path = root / "duration" / "source_report.json"
    write_report(source_report_path)
    selected = {
        "candidate_id": "duration_fill_candidate",
        "midi_path": str(sample_path),
        "metrics": {
            "note_count": 18,
            "unique_pitch_count": 15,
            "dead_air_ratio": 0.29411764705882354,
        },
        "focused_solo_metrics": {
            "focused_note_count": 18,
            "focused_unique_pitch_count": 15,
            "focused_adjacent_pitch_repeats": 0,
            "focused_max_interval": 7,
            "focused_duplicated_3_note_pitch_class_chunks": 0,
        },
        "duration_coverage_gate": {
            "qualified": True,
            "flags": [],
        },
    }
    return {
        "output_dir": str(root / "duration"),
        "source_candidate": {
            "source_report_path": str(source_report_path),
        },
        "repair_summary": {
            "selected_candidate_id": "duration_fill_candidate",
            "dead_air_delta_from_baseline": 0.277311,
            "selected_fill_addition_count": 6,
        },
        "selected_candidate": selected,
        "variants": [selected],
    }


class StageBMarginRecoveredPhraseVocabularyFocusedPackageTest(unittest.TestCase):
    def test_builds_focused_package_from_selected_repair_candidate(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            package = build_phrase_vocabulary_focused_package(
                repair_summary(root),
                output_dir=root / "package",
                decision=DEFAULT_DECISION,
            )
            summary = validate_package(
                package,
                expected_candidate_id="phrase_vocabulary_candidate",
                min_candidates=1,
            )

            self.assertEqual(summary["candidate_count"], 1)
            self.assertEqual(summary["candidate_ids"], ["phrase_vocabulary_candidate"])
            candidate = package["candidates"][0]
            self.assertTrue(Path(candidate["review_files"]["midi_path"]).exists())
            self.assertTrue(Path(candidate["review_files"]["context_midi_path"]).exists())
            self.assertEqual(candidate["listening"]["decision"], DEFAULT_DECISION)
            self.assertEqual(candidate["source_metrics"]["focused_max_interval"], 7)

    def test_builds_focused_package_from_explicit_peer_candidate(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            package = build_phrase_vocabulary_focused_package(
                repair_summary(root),
                output_dir=root / "package",
                decision=DEFAULT_DECISION,
                candidate_id="phrase_vocabulary_peer",
            )
            summary = validate_package(
                package,
                expected_candidate_id="phrase_vocabulary_peer",
                min_candidates=1,
            )

            self.assertEqual(summary["candidate_count"], 1)
            self.assertEqual(summary["candidate_ids"], ["phrase_vocabulary_peer"])
            candidate = package["candidates"][0]
            self.assertTrue(Path(candidate["review_files"]["midi_path"]).exists())
            self.assertTrue(Path(candidate["review_files"]["context_midi_path"]).exists())

    def test_builds_focused_package_from_duration_coverage_fill_candidate(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            package = build_phrase_vocabulary_focused_package(
                duration_fill_summary(root),
                output_dir=root / "package",
                decision="phrase_vocabulary_duration_coverage_fill_qualified",
            )
            summary = validate_package(
                package,
                expected_candidate_id="duration_fill_candidate",
                min_candidates=1,
            )

            self.assertEqual(summary["candidate_count"], 1)
            candidate = package["candidates"][0]
            self.assertTrue(Path(candidate["review_files"]["midi_path"]).exists())
            self.assertTrue(Path(candidate["review_files"]["context_midi_path"]).exists())
            self.assertEqual(
                candidate["listening"]["decision"],
                "phrase_vocabulary_duration_coverage_fill_qualified",
            )
            self.assertEqual(candidate["source_metrics"]["fill_addition_count"], 6)
            self.assertTrue(candidate["objective_review"]["qualified"])


if __name__ == "__main__":
    unittest.main()
