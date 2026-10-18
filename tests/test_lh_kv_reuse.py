"""Common-prefix accounting and cause attribution of #1793 (no model)."""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import lh_kv_reuse as kv  # noqa: E402

PN = lambda p, v=60: ("piano", p, v)   # noqa: E731
ON = lambda x: ("onset", x)             # noqa: E731
DU = lambda x: ("dur", x)               # noqa: E731


def req(k, toks, tags, label_start, keep=None, base=0.0, n=3):
    return {"k": k, "keep_from": keep, "base_ms": base, "win_labels": n, "boundary_labels": 0,
            "b": {"tokens": toks, "tags": tags, "label_start": label_start}, "a": None}


class KvReuseTest(unittest.TestCase):
    def test_common_prefix(self):
        self.assertEqual(kv.common_prefix([1, 2, 3, 4], [1, 2, 9, 4]), 2)
        self.assertEqual(kv.common_prefix([1, 2], [1, 2, 3]), 2)

    def test_causes(self):
        rh = [PN(72), ON(100), DU(50)]
        lh1 = [PN(40), ON(50), DU(200)]
        lh1_long = [PN(40), ON(50), DU(400)]
        lh2 = [PN(43), ON(80), DU(100)]
        tag = lambda seq: [g for part, g in seq for _ in part]  # noqa: E731
        P = lh1 + rh
        Pt = tag([(lh1, "lh"), (rh, "rh")])
        # insert: a new left-hand note before the right-hand note
        N = lh1 + lh2 + rh
        self.assertEqual(kv.cause(P, Pt, N, tag([(lh1, "lh"), (lh2, "lh"), (rh, "rh")]), kv.common_prefix(P, N)), "insert")
        # expired: the first left-hand note left the window
        N = rh
        self.assertEqual(kv.cause(P, Pt, N, tag([(rh, "rh")]), kv.common_prefix(P, N)), "expired")
        # held update: same note, longer duration
        N = lh1_long + rh
        self.assertEqual(kv.cause(P, Pt, N, tag([(lh1_long, "lh"), (rh, "rh")]), kv.common_prefix(P, N)), "held_update")
        # none: the new context is a prefix of the previous input
        self.assertEqual(kv.cause(P, Pt, lh1, tag([(lh1, "lh")]), 3), "none")

    def test_pair_rows_crop_is_full_reset(self):
        toks = [PN(72), ON(100), DU(50)]
        r1 = req(1, toks + [PN(74), ON(300), DU(50)], ["rh"] * 6, 3)
        r2 = req(2, toks + [PN(74), ON(300), DU(50)], ["rh"] * 6, 6)
        r3 = req(3, toks, ["rh"] * 3, 3, keep=4)
        rows = kv.pair_rows([r1, r2, r3], "b")
        self.assertEqual((rows[0]["P"], rows[0]["S"], rows[0]["cause"]), (6, 0, "none"))
        self.assertEqual((rows[1]["P"], rows[1]["cause"]), (0, "crop"))


if __name__ == "__main__":
    unittest.main()
