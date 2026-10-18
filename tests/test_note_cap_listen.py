"""Note rebuild, divergence, window grid and excerpt statistics of the #1819 listening material (no model, no audio)."""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import note_cap_listen as nl  # noqa: E402


def row(i, kind, nid, pitch, due, block, vel=70):
    return {"index": i, "kind": kind, "pitch": pitch, "velocity": vel if kind == "on" else 0, "due_ms": due,
            "note_id": nid, "block": block}


class RebuildTest(unittest.TestCase):
    def test_planned_notes_pair_on_off_and_mark_open(self):
        snd = {"sent": [row(0, "on", 0, 60, 1000, 1), row(1, "on", 1, 64, 1000, 1), row(2, "off", 0, 60, 1200, 1),
                        row(3, "on", 2, 67, 1500, 2)]}
        ns = nl.planned_notes(snd)
        self.assertEqual([(n["pitch"], n["on_ms"], n["off_ms"], n["open"]) for n in ns],
                         [(60, 1000, 1200, False), (64, 1000, 1500, True), (67, 1500, 1510, True)])

    def test_first_divergence(self):
        a = nl.planned_notes({"sent": [row(0, "on", 0, 60, 1000, 1), row(1, "on", 1, 62, 2000, 2), row(2, "on", 2, 64, 3000, 3)]})
        b = nl.planned_notes({"sent": [row(0, "on", 0, 60, 1000, 1), row(1, "on", 1, 63, 2000, 2)]})
        self.assertEqual(nl.first_divergence(a, b), 2)
        self.assertIsNone(nl.first_divergence(a, a))


class GridTest(unittest.TestCase):
    def test_window_index_and_start(self):
        t0 = 7870
        self.assertEqual(nl.window_index(t0 + 468.75, t0), 0)
        self.assertEqual(nl.window_index(t0 + 468.75 + 937.5, t0), 1)
        self.assertEqual(nl.window_index(t0 + 400, t0), -1)
        self.assertEqual(nl.window_start(1, t0), 9280)          # 7870 + 468.75 + 937.5 = 9276.25 -> 9280

    def test_group_blocks(self):
        self.assertEqual(nl.group_blocks([16, 3, 4, 13, 5]), [(3, 5), (13, 13), (16, 16)])


class StatsTest(unittest.TestCase):
    def note(self, on, off, p=60):
        return {"pitch": p, "velocity": 70, "on_ms": on, "off_ms": off}

    def test_chord_rest_windows_and_window_start_onsets(self):
        t0 = 0
        ns = [self.note(500, 600, 60), self.note(510, 600, 64), self.note(520, 600, 67),   # one 3-note chord
              self.note(1410, 1500), self.note(2360, 2400)]                                    # 1410 = window 1 start (1406.25 -> 1410)
        st = nl.excerpt_stats(ns, 468.75, 2343.75 + 937.5, t0)
        self.assertEqual(st["max_chord"], 3)
        self.assertEqual(st["per_window"], {"0": 3, "1": 1, "2": 1})
        self.assertEqual(st["onsets_at_window_start"], [1410])
        self.assertAlmostEqual(st["longest_rest_ms"], 2343.75 + 937.5 - 2400, places=0)


if __name__ == "__main__":
    unittest.main()
