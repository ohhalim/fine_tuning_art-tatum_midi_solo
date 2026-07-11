from __future__ import annotations

import argparse
import unittest
from pathlib import Path

from scripts.audit_jazz_piano_dataset import parse_dataset_path, recommendation_for


class JazzDatasetAuditTest(unittest.TestCase):
    def test_parse_dataset_path_extracts_source_artist_album(self) -> None:
        path = Path("midi_dataset/midi/live/Brad Mehldau/Live in Tokyo/Track.midi")
        meta = parse_dataset_path(path, Path("midi_dataset/midi"))

        self.assertEqual(meta["source"], "live")
        self.assertEqual(meta["artist"], "Brad Mehldau")
        self.assertEqual(meta["album"], "Live in Tokyo")
        self.assertTrue(meta["is_brad_mehldau"])

    def test_recommendation_rejects_unreadable_first(self) -> None:
        args = argparse.Namespace(
            min_notes=24,
            min_duration_sec=5.0,
            max_duration_sec=1800.0,
            max_note_duration_ratio=0.85,
            min_piano_program_ratio=0.8,
        )

        recommendation = recommendation_for({"readable": False}, args)

        self.assertEqual(recommendation, "reject_unreadable")

    def test_recommendation_accepts_candidate(self) -> None:
        args = argparse.Namespace(
            min_notes=24,
            min_duration_sec=5.0,
            max_duration_sec=1800.0,
            max_note_duration_ratio=0.85,
            min_piano_program_ratio=0.8,
        )
        row = {
            "readable": True,
            "non_drum_note_count": 120,
            "duration_sec": 80.0,
            "pitch_out_of_piano_range": False,
            "long_sustain_suspect": False,
            "non_piano_program_suspect": False,
            "too_long": False,
        }

        self.assertEqual(recommendation_for(row, args), "candidate")


if __name__ == "__main__":
    unittest.main()
