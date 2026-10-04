"""Solo-aware guide revoicing for C1 (inference/control/comping.revoice_strike)."""
from __future__ import annotations

import unittest

from inference.control.comping import revoice_strike, solo_clashes


class RevoiceTest(unittest.TestCase):
    def test_root_melody_over_major_seventh_drops_the_seventh(self) -> None:
        # Fmaj7 guide A3 + E3 under a solo F5: E3-F5 is a compound minor 9th
        pitches, info = revoice_strike("Fmaj7", (52, 57), [(77, 0.0, 0.5)], 0.0, 0.5)
        self.assertEqual(info["status"], "changed")
        self.assertEqual(info["voicing"], ("3", "5"))
        self.assertEqual(pitches, (48, 57))                  # C3 + A3
        self.assertFalse(solo_clashes(pitches, [(77, 0.0, 0.5)], 0.0, 0.5))

    def test_dominant_keeps_its_tritone(self) -> None:
        pitches, info = revoice_strike("C7", (52, 58), [(65, 0.0, 0.5)], 0.0, 0.5)   # F over E3
        self.assertEqual((pitches, info["status"]), ((52, 58), "dominant_kept"))

    def test_no_clash_no_change_and_short_overlaps_ignored(self) -> None:
        self.assertEqual(revoice_strike("Fmaj7", (52, 57), [(76, 0.0, 0.5)], 0.0, 0.5)[1]["status"], "clear")
        self.assertEqual(revoice_strike("Fmaj7", (52, 57), [(77, 0.49, 0.6)], 0.0, 0.5)[1]["status"], "clear")

    def test_major_seventh_above_is_not_a_clash(self) -> None:
        self.assertFalse(solo_clashes((52,), [(75, 0.0, 0.5)], 0.0, 0.5))           # E3 under Eb5: major 7th family


if __name__ == "__main__":
    unittest.main()
