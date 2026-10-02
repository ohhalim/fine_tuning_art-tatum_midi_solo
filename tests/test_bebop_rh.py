"""Right-hand extraction and the bebop right-hand evaluation rules."""
from __future__ import annotations

import unittest

import pretty_midi

from scripts.bebop_rh_eval import judge, pooled, window_stats
from scripts.build_rh_dataset import right_hand, split_of


def n(p, s, e, v=80):
    return pretty_midi.Note(velocity=v, pitch=p, start=s, end=e)


class RightHandTest(unittest.TestCase):
    def test_left_hand_only_moments_become_rests(self) -> None:
        notes = [n(72, 0.0, 0.2), n(48, 0.0, 0.5), n(74, 0.2, 0.4),     # RH over LH chord
                 n(43, 1.0, 1.5), n(50, 1.01, 1.5),                     # LH alone: rest
                 n(76, 2.0, 2.3), n(52, 2.0, 2.3)]
        line = right_hand(notes)
        self.assertEqual([x.pitch for x in line], [72, 74, 76])
        self.assertTrue(all(a.end <= b.start for a, b in zip(line, line[1:])))

    def test_split_is_deterministic(self) -> None:
        self.assertEqual(split_of("abc"), split_of("abc"))
        self.assertIn(split_of("abc"), ("train", "val", "test"))


class WindowTest(unittest.TestCase):
    def test_rests_include_window_edges_and_split_phrases(self) -> None:
        w = window_stats([(0.0, 0.2), (0.2, 0.4), (1.0, 1.2), (1.2, 1.4)], 0.0, 4.0)
        self.assertEqual((w["notes"], w["rests"], w["phrases"]), (4, 2, [2, 2]))
        self.assertAlmostEqual(w["rest_time"], 0.6 + 2.6)

    def test_judge(self) -> None:
        real = pooled([{"notes": 16, "rest_time": 0.4, "rests": 1, "phrases": [16], "seconds": 4.0}])
        good = pooled([{"notes": 18, "rest_time": 0.4, "rests": 1, "phrases": [18], "seconds": 4.0}])
        base = pooled([{"notes": 50, "rest_time": 0.0, "rests": 0, "phrases": [50], "seconds": 4.0}])
        self.assertTrue(judge(real, good, base, 1.0)["pass"])
        self.assertFalse(judge(real, base, good, 1.0)["pass"])


if __name__ == "__main__":
    unittest.main()
