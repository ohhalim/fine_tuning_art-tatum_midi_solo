"""Reference clip features (issue #1762). Synthetic notes."""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from reference_clip_features import features  # noqa: E402


class FeatureTest(unittest.TestCase):
    def test_line_chord_and_register(self):
        line = [(60 + i % 2, 200.0 * i, 200.0 * i + 150) for i in range(10)]           # 10-note line, semitone steps
        chord = [(40, 5000.0, 6000.0), (44, 5010.0, 6000.0), (47, 5020.0, 6000.0)]     # one low 3-note chord
        f = features(line + chord + [(90, 7000.0, 7100.0)], 10.0)
        self.assertEqual(f["comping_per_s"], 0.1)
        self.assertEqual(f["lick_len_p90"], 10)
        self.assertEqual(f["semitone_step_share"], 1.0)
        self.assertEqual(f["low_share"], round(3 / 14, 4))
        self.assertEqual(f["high_share"], round(1 / 14, 4))


if __name__ == "__main__":
    unittest.main()
