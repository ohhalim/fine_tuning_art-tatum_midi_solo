"""Shared harmony guide (inference/control/harmony_contract.py, #1631)."""
from __future__ import annotations

import unittest

import pretty_midi

from inference.control.harmony_contract import (accompaniment_proxy, chord_pcs, guide_notes, guide_pitches,
                                                serialize, solo_window)
from scripts.style_distance import tokens_to_notes


def note(p, s, e, v=80):
    return pretty_midi.Note(velocity=v, pitch=p, start=s, end=e)


class GuideTest(unittest.TestCase):
    def test_every_chord_tone_once_with_the_root_in_the_bass(self) -> None:
        self.assertEqual(guide_pitches(*chord_pcs("G7")), [43, 50, 53, 59])      # G | D F B: the third is there
        self.assertEqual(guide_pitches(*chord_pcs("Dm7")), [38, 48, 53, 57])     # D | C F A
        self.assertNotEqual(guide_pitches(*chord_pcs("Dm7b5")), guide_pitches(*chord_pcs("Dm7")))
        self.assertIn(56, guide_pitches(*chord_pcs("Dm7b5")))                    # Ab, the flat fifth

    def test_guide_timing_matches_the_runtime_guide(self) -> None:
        ns = guide_notes(7, {7, 11, 2, 5}, 0.9375)
        self.assertTrue(all(n.start == 0.0 and abs(n.end - 0.797) < 1e-3 and n.velocity == 58 for n in ns))


class ProxyTest(unittest.TestCase):
    def test_low_notes_starting_in_or_sounding_at_the_window(self) -> None:
        notes = [note(36, 0.0, 2.0), note(47, 0.9, 1.1), note(64, 1.0, 1.2), note(52, 1.2, 1.4), note(40, 3.0, 3.5)]
        bass, pcs, stats = accompaniment_proxy(notes, 1.0, 2.0)
        self.assertEqual((bass, pcs), (0, {0, 11, 4}))           # C held, B sounding at 1.0, E starts in; 64 is solo
        self.assertFalse(stats["empty"])
        self.assertEqual(accompaniment_proxy(notes, 5.0, 6.0)[0], None)


class SerializeTest(unittest.TestCase):
    def test_guide_then_solo_on_its_own_timeline(self) -> None:
        guide = guide_notes(0, {0, 4, 7, 11}, 1.0)
        solo = solo_window([note(72, 1.1, 1.3), note(74, 1.5, 2.4), note(76, 2.5, 2.6)], 1.0, 2.0)
        toks, at = serialize(guide, solo, 1.0)
        self.assertEqual(sorted(n.pitch for n in tokens_to_notes(toks[:at])), guide_pitches(0, {0, 4, 7, 11}))
        body = tokens_to_notes(toks[at:])
        self.assertEqual([(n.pitch, round(n.start, 2), round(n.end, 2)) for n in body], [(72, 0.1, 0.3), (74, 0.5, 1.0)])


if __name__ == "__main__":
    unittest.main()
