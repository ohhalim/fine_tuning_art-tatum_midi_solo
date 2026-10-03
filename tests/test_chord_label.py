"""Chord candidates for the label audit (inference/control/chord_label.py): positive / negative controls."""
from __future__ import annotations

import unittest

from inference.control.chord_label import candidates, classify

QUALITY_TONES = {"7": (4, 7, 10, 2, 9), "m7": (3, 7, 10, 2, 5), "maj7": (4, 7, 11, 2, 9), "m7b5": (3, 6, 10, 2, 5)}


def voicing(root, q, kind):
    third, fifth, seventh, ninth, _ = QUALITY_TONES[q]
    iv = {"root_3_7": [0, third, seventh] if q != "m7b5" else [0, third, fifth, seventh],
          "shell": [0, seventh, third + 12] if q != "m7b5" else [0, seventh, third + 12, fifth + 12],
          "rootless_a": [third, fifth, seventh, ninth + 12],
          "rootless_b": [seventh, ninth + 12, third + 12, fifth + 12]}[kind]
    return [(root + i) % 12 for i in iv], (root if kind in ("root_3_7", "shell") else None)


class ChordLabelTest(unittest.TestCase):
    def test_true_chord_is_always_a_candidate(self) -> None:
        for root in range(12):
            for q in QUALITY_TONES:
                for kind in ("root_3_7", "shell", "rootless_a", "rootless_b"):
                    pcs, _ = voicing(root, q, kind)
                    self.assertIn((root, q), candidates(pcs), (root, q, kind))

    def test_rooted_voicings_resolve_to_the_true_chord(self) -> None:
        for root in range(12):
            for q in QUALITY_TONES:
                for kind in ("root_3_7", "shell"):
                    pcs, bass = voicing(root, q, kind)
                    cls, label, _ = classify(pcs, bass)
                    self.assertIn(cls, ("clear", "bass_resolved"), (root, q, kind))
                    self.assertEqual(label, (root, q))

    def test_known_ambiguities_stay_ambiguous(self) -> None:
        # D-F-A-C: Dm7, F6, or rootless Bbmaj9
        self.assertEqual(sorted(candidates([2, 5, 9, 0])), [(2, "m7"), (5, "6"), (10, "maj7")])
        rootless_c7 = [4, 10, 2, 9]                                                        # E-Bb-D-A
        self.assertIn((6, "7"), candidates(rootless_c7))                                   # Gb7 (tritone sub)
        self.assertEqual(classify(rootless_c7, None)[0], "ambiguous")

    def test_window_union_hides_a_chord_change(self) -> None:
        # the hazard the audit measures: Dm7 then G7 in one window reads as one chord (Dm13), the G7 is lost
        dm7, g7 = {2, 5, 9, 0}, {7, 11, 5}
        self.assertEqual(candidates(dm7 | g7), [(2, "m7")])
        self.assertIn((7, "7"), candidates(g7))

    def test_too_few_tones_are_unknown_and_two_tones_stay_open(self) -> None:
        self.assertEqual(classify([0], 0)[0], "unknown")
        cls, label, _ = classify([0, 7], 0)                                                # open fifth: no single chord
        self.assertEqual((cls, label), ("ambiguous", None))


if __name__ == "__main__":
    unittest.main()
