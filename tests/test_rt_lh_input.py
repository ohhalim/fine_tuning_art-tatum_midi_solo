"""Input rules, snapshot causality, crop reset and the deadline rule of #1797 (no MIDI, no model)."""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import rt_lh_input as li  # noqa: E402
from lh_context import Human, merge, window_notes  # noqa: E402
from lh_cache_replay import reuse_start  # noqa: E402


class InputRulesTest(unittest.TestCase):
    def test_retrigger_velocity0_unknown_off_cc64(self):
        st = li.LHState()
        st.feed("note_on", 100, 40, 70)
        st.feed("note_on", 300, 40, 75)              # re-struck while held: the first ends at 300
        st.feed("note_on", 500, 40, 0)               # velocity 0 = off for the second
        st.feed("note_off", 600, 43)                 # never started
        st.feed("control_change", 650, control=64)
        self.assertEqual([(n.on, n.off) for n in st.notes], [(100, 300), (300, 500)])
        self.assertEqual((st.counts["retrigger"], st.counts["velocity0_off"], st.counts["unknown_off"], st.counts["cc64_ignored"]),
                         (1, 1, 1, 1))

    def test_snapshot_only_onsets_before_s_and_held_cut(self):
        st = li.LHState()
        st.feed("note_on", 1000, 40, 70)             # held at s
        st.feed("note_on", 1500, 43, 70)
        st.feed("note_off", 1700, 43)
        st.feed("note_on", 2000, 47, 70)             # onset == s: not in this block
        notes = window_notes(Human(st.human()), 2000, 0)
        self.assertEqual([(n.pitch, n.on, n.off, n.held) for n in notes], [(40, 1000, 2000, True), (43, 1500, 1700, False)])

    def test_future_off_does_not_change_the_snapshot(self):
        a, b = li.LHState(), li.LHState()
        for st in (a, b):
            st.feed("note_on", 1000, 40, 70)
        b.feed("note_off", 2600, 40)                 # arrives after the snapshot time 2000
        na = window_notes(Human(a.human()), 2000, 0)
        nb = window_notes(Human(b.human()), 2000, 0)
        self.assertEqual([(n.on, n.off, n.held) for n in na], [(n.on, n.off, n.held) for n in nb])

    def test_same_onset_left_hand_goes_before_right_hand(self):
        ctx = [("prefix", "instrument", "piano"), "<S>", ("piano", 72, 60), ("onset", 1000), ("dur", 200)]
        st = li.LHState()
        st.feed("note_on", 1000, 40, 70)
        st.feed("note_off", 1100, 40)
        toks, tags = merge(ctx, 0.0, window_notes(Human(st.human()), 2000, 0))
        self.assertEqual(tags[2:8], ["lh"] * 3 + ["rh"] * 3)


class CropAndDeadlineTest(unittest.TestCase):
    def test_crop_resets_reuse_and_moves_base(self):
        crop = li.SessionCrop(cap=64, capacity=60 + 2 * 64 + 3, keep_target=20)
        toks = [("prefix", "instrument", "piano"), "<S>"] + ([("piano", 60, 60), ("onset", 10), ("dur", 10)] * 6 + ["<T>"]) * 4
        changed = crop.maybe_crop(toks, len(toks))
        self.assertTrue(changed)
        self.assertGreater(crop.base_ms, 0)
        self.assertEqual(reuse_start(valid_end=50, lcp=40, n_len=60, crop_changed=changed), 0)

    def test_no_crop_when_it_fits(self):
        crop = li.SessionCrop(cap=64, capacity=4096, keep_target=2048)
        self.assertFalse(crop.maybe_crop([("prefix", "instrument", "piano"), "<S>"], 2))

    def test_deadline_rule(self):
        self.assertFalse(li.is_late(100, 100))
        self.assertTrue(li.is_late(101, 100))


class FeederOrderTest(unittest.TestCase):
    def test_events_sorted_offs_first_at_same_time(self):
        import rt_lh_feeder as fd
        ev = fd.events([[40, 0, 500, 70], [43, 500, 900, 60]], offset_ms=1000)
        self.assertEqual(ev, [(1000, "note_on", 40, 70), (1500, "note_off", 40, 0), (1500, "note_on", 43, 60), (1900, "note_off", 43, 0)])


class InputIntegrityTest(unittest.TestCase):
    def test_b1_requires_every_message_sent_before_final_drain(self):
        from rt_lh_smoke_run import b1
        sent = [("note_on", 40), ("note_off", 40), ("note_on", 43)]
        rows = [{"send_ns": 1_000_000}, {"send_ns": 50_000_000}, {"send_ns": 99_500_000}]
        self.assertTrue(b1(sent[:2], sent, rows, final_drain_ns=100_000_000)["ok"])       # third is inside the 2 ms guard
        self.assertFalse(b1(sent[:1], sent, rows, final_drain_ns=100_000_000)["ok"])      # second was due and is missing
        self.assertFalse(b1([sent[1], sent[0]], sent, rows, final_drain_ns=100_000_000)["ok"])  # order changed


if __name__ == "__main__":
    unittest.main()
