"""Decoding-gap diagnosis rules (docs/experiments/DECODING_GAP_DIAG.md)."""
from __future__ import annotations

import unittest

from scripts.decoding_gap_diag import song_flag_rate, song_rates, summarize, verdict


def m(windows=10, distinct=10, interval=0, variant=0, exact=0, copy=False, valid=True, notes=20):
    return {"windows": windows, "distinct_windows": distinct, "interval": interval, "variant": variant,
            "exact": exact, "copy": copy, "valid": valid, "notes": notes}


def case(song, real, rolls):
    return {"song": song, "real": real, "rollouts": rolls}


class RatesTest(unittest.TestCase):
    def test_repetition_and_invalid_rows(self) -> None:
        rows = [("a", m(windows=10, distinct=6)), ("a", m(windows=10, distinct=10, valid=False))]
        self.assertAlmostEqual(song_rates(rows, "repetition")["a"], 0.4)
        self.assertEqual(song_flag_rate([("a", m(copy=True)), ("a", m())])["a"], 0.5)


class VerdictTest(unittest.TestCase):
    def cases(self, ir06, rep06=1, copy06=False):
        out = []
        for s in "abcd":
            out.append(case(s, m(interval=1, distinct=9),           # real IR 0.1, repetition 0.1
                            {"1.0": [m(interval=0)] * 2, "0.8": [m(interval=0)] * 2,
                             "0.6": [m(interval=ir06, distinct=10 - rep06, copy=copy06)] * 2}))
        return out

    def test_explains_when_ir_reaches_half_of_real_without_degeneration(self) -> None:
        v = verdict(summarize(self.cases(ir06=1)))
        self.assertEqual(v["label"], "decoding_explains")
        self.assertTrue(v["checks"]["0.6"]["all"])

    def test_mechanical_repetition_or_copying_does_not_count(self) -> None:
        self.assertEqual(verdict(summarize(self.cases(ir06=1, rep06=6)))["label"], "decoding_alone_insufficient")
        self.assertEqual(verdict(summarize(self.cases(ir06=1, copy06=True)))["label"], "decoding_alone_insufficient")

    def test_small_gain_is_insufficient(self) -> None:
        v = verdict(summarize(self.cases(ir06=0)))
        self.assertEqual(v["label"], "decoding_alone_insufficient")
        self.assertIn("not judged", v["note"])


if __name__ == "__main__":
    unittest.main()
