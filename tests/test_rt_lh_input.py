"""Input rules, snapshot causality, crop reset and the deadline rule of #1797 (no MIDI, no model)."""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import rt_lh_input as li  # noqa: E402
from lh_context import W_MS, Human, merge, window_notes  # noqa: E402
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


class CropWithLeftHandTest(unittest.TestCase):
    """Issue #1803: the cut is chosen on the merged request, and a request that cannot fit stops."""

    def setUp(self):
        from crop_contract import HEADER
        self.rh = list(HEADER) + ["<T>"] + [x for i in range(60) for x in (("piano", 72, 60), ("onset", 10 * i), ("dur", 10))]

    def merged_len(self, n_lh, s_k=10000):
        human = Human([li.LiveNote(40 + i % 12, 5000 + 50 * i, 5000 + 50 * i + 20, 60) for i in range(n_lh)])
        return lambda sess, base: len(merge(sess, base, window_notes(human, s_k, max(s_k - 7500, base)))[0])

    def test_left_hand_that_cannot_fit_is_a_capacity_failure(self):
        from crop_contract import CapacityFailure
        for n_lh in (80, 30):                        # merged 422 and 272 (+ reserve 131) against capacity 400
            crop = li.SessionCrop(cap=64, capacity=400, keep_target=200)
            with self.assertRaises(CapacityFailure):
                crop.maybe_crop(self.rh, len(self.rh), self.merged_len(n_lh))

    def test_cut_moves_later_when_the_left_hand_needs_room(self):
        seg = [x for i in range(20) for x in (("piano", 72, 60), ("onset", 10 * i), ("dur", 10))]
        toks = [("prefix", "instrument", "piano"), "<S>"] + seg + ["<T>"] + seg + ["<T>"] + seg
        lh_after = {0: 100, 5000: 100, 10000: 0}      # left-hand tokens left in the window for each base
        crop = li.SessionCrop(cap=64, capacity=300, keep_target=130)
        self.assertTrue(crop.maybe_crop(toks, len(toks), lambda sess, base: len(sess) + lh_after[base]))
        # right hand alone keeps 123 <= 130 from the first <T> (base 5000), but merged that is 223 + reserve 131 > 300
        self.assertEqual(crop.base_ms, 10000)

    def test_right_hand_only_cut_would_not_fit_here(self):
        from crop_contract import choose_cut, needs_crop
        seg = [x for i in range(20) for x in (("piano", 72, 60), ("onset", 10 * i), ("dur", 10))]
        toks = [("prefix", "instrument", "piano"), "<S>"] + seg + ["<T>"] + seg + ["<T>"] + seg
        j = choose_cut(toks, len(toks), 64, 300, 130)
        self.assertEqual(toks[:j + 1].count("<T>"), 1)            # base 5000
        self.assertTrue(needs_crop(2 + len(toks) - (j + 1) + 100, 64, 300))


class UnderfillTest(unittest.TestCase):
    """Issue #1802: fill_block calls on_ready also when it ran out of samples before the horizon."""

    def test_block_problem_order(self):
        self.assertEqual(li.block_problem(101, 100, 0, 50), "deadline_miss")
        self.assertEqual(li.block_problem(99, 100, 40, 50), "horizon_underfill")
        self.assertIsNone(li.block_problem(99, 100, 50, 50))
        self.assertEqual(li.block_problem(10 ** 18, None, 40, 50), "horizon_underfill")   # cold block: no deadline

    def test_guard_commits_only_a_block_without_problem(self):
        calls = []
        self.assertEqual(li.guarded_commit(0, 1, 40, 50, lambda: calls.append(1)), "horizon_underfill")
        self.assertEqual(li.guarded_commit(2, 1, 50, 50, lambda: calls.append(1)), "deadline_miss")
        self.assertEqual(calls, [])
        self.assertIsNone(li.guarded_commit(0, 1, 50, 50, lambda: calls.append(1)))
        self.assertEqual(calls, [1])

    def test_underfilled_block_is_neither_committed_nor_sent(self):
        from aria_mlx_stream_loop import fill_block

        class Stop(Exception):
            pass

        committed = []

        def on_ready():                              # the producer's on_ready shape around the same guard
            if li.guarded_commit(0, 1, 0, 100, lambda: committed.append(1)):   # coverage stays 0, horizon 100
                raise Stop()

        with self.assertRaises(Stop):
            fill_block(lambda: None, lambda: 0, on_ready, 100, 200, 64)
        self.assertEqual(committed, [])


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


class LoopSpeedTest(unittest.TestCase):
    def test_only_warm_blocks_that_sampled_before_the_horizon(self):
        from rt_lh_smoke_run import loop_speed
        prod = {"blocks": [{"block": 1, "samples_to_ready": 10, "ready_s": 0.1},
                           {"block": 2, "samples_to_ready": 0, "ready_s": 0.0},          # horizon already covered
                           {"block": 3, "samples_to_ready": 20, "ready_s": 0.3},
                           {"block": 4, "samples_to_ready": None, "ready_s": None},       # capacity failure row
                           {"block": 5, "ready_s": 0.2}]}                                 # pre-#1813 row
        out = loop_speed(prod)
        self.assertEqual([r["block"] for r in out["blocks"]], [1, 3])
        self.assertEqual([r["ms_per_sample"] for r in out["blocks"]], [10.0, 15.0])
        self.assertEqual(out["median_ms_per_sample"], 15.0)


class RerunChecksFailClosedTest(unittest.TestCase):
    """#1807 judge must not pass a missing record, an EOF end or a mapping gap (Astra review M1)."""

    def base(self):
        import copy
        from rt_lh_smoke_run import BLOCKS, REQUIRED_A_CHECKS
        out = {"posthoc_before_horizon": {"notes_before_previous_horizon": [], "unmatched_on": 0, "rejected_late_on": []},
               "a_checks": {"checks": {k: {"ok": True} for k in REQUIRED_A_CHECKS}},
               "c2_safe_stop": {"pending_on_at_stop": 0, "port_closed": True, "stop_offs_failed": 0},
               "b1_input_integrity": {"ok": True}, "b2_causal": True}
        prod = {"abort": None, "blocks": [{"block": k, "len_N": 100} for k in range(1, BLOCKS)]}
        snd = {"end": "end"}
        return copy.deepcopy(out), prod, snd

    def judge(self, out, prod, snd):
        from rt_lh_smoke_run import rerun_checks
        return rerun_checks(out, prod, 4160, snd)

    def test_complete_normal_record_passes(self):
        self.assertTrue(all(self.judge(*self.base()).values()))

    def test_eof_end_without_abort_fails(self):
        out, prod, snd = self.base()
        snd["end"] = "eof"
        self.assertFalse(self.judge(out, prod, snd)["stop_is_recorded_abort_or_end"])

    def test_missing_warm_block_fails(self):
        out, prod, snd = self.base()
        prod["blocks"].pop()
        self.assertFalse(self.judge(out, prod, snd)["stop_is_recorded_abort_or_end"])

    def test_abort_must_match_the_sender_record(self):
        out, prod, snd = self.base()
        prod["abort"] = {"reason": "deadline_miss", "block": 2}
        prod["blocks"] = prod["blocks"][:2]
        snd.update(end="abort", abort={"reason": "deadline_miss", "block": 2})
        self.assertTrue(self.judge(out, prod, snd)["stop_is_recorded_abort_or_end"])
        snd["abort"] = {"reason": "deadline_miss", "block": 3}
        self.assertFalse(self.judge(out, prod, snd)["stop_is_recorded_abort_or_end"])
        snd.pop("abort")
        self.assertFalse(self.judge(out, prod, snd)["stop_is_recorded_abort_or_end"])

    def test_empty_or_partial_existing_checks_fail(self):
        out, prod, snd = self.base()
        out["a_checks"]["checks"] = {}
        self.assertFalse(self.judge(out, prod, snd)["existing_checks"])
        out, prod, snd = self.base()
        out["a_checks"]["checks"]["sender_exit"] = {}
        self.assertFalse(self.judge(out, prod, snd)["existing_checks"])
        out, prod, snd = self.base()
        del out["a_checks"]["checks"]["A2_capture_alternates_per_pitch"]
        self.assertFalse(self.judge(out, prod, snd)["existing_checks"])

    def test_unmatched_on_and_missing_rejection_record_fail(self):
        out, prod, snd = self.base()
        out["posthoc_before_horizon"]["unmatched_on"] = 99
        self.assertFalse(self.judge(out, prod, snd)["no_note_before_previous_horizon"])
        out, prod, snd = self.base()
        del out["posthoc_before_horizon"]["rejected_late_on"]
        self.assertFalse(self.judge(out, prod, snd)["no_late_intake_rejection"])

    def test_missing_stop_offs_record_and_no_forward_fail(self):
        out, prod, snd = self.base()
        out["c2_safe_stop"]["stop_offs_failed"] = None
        self.assertFalse(self.judge(out, prod, snd)["safe_stop"])
        out, prod, snd = self.base()
        for b in prod["blocks"]:
            b["len_N"] = None
        self.assertFalse(self.judge(out, prod, snd)["forwarded_blocks_fit_capacity"])

    def test_b1_not_applicable_counts_only_for_a_replay_run(self):
        """#1825: a replay run has no feeder; B1 is not applicable, never a pass, and only accepted when the producer
        says its left hand came from replay."""
        out, prod, snd = self.base()
        out["b1_input_integrity"] = {"ok": None, "not_applicable": True}
        self.assertFalse(self.judge(out, prod, snd)["existing_checks"])          # a live run cannot skip B1
        prod["lh_source"] = {"mode": "replay"}
        self.assertTrue(self.judge(out, prod, snd)["existing_checks"])
        out["b1_input_integrity"] = {"ok": None}                                 # neither a pass nor not applicable
        self.assertFalse(self.judge(out, prod, snd)["existing_checks"])


class LHReplayTest(unittest.TestCase):
    """#1825: the left hand read by music time equals a zero-jitter live feed drained at any time in [s_k, s_k + B)."""

    T0, OFF = 7870.0, 937.5
    # music time of note time t is T0 + OFF + t; block starts s_k = T0 + k * 937.5, so t = (k - 1) * 937.5 is s_k
    NOTES = [[47, 0.0, 1472.727, 50], [52, 415.909, 1472.727, 57], [51, 831.818, 1472.727, 52],
             [40, 500.0, 937.5, 60],                # off exactly at s_2
             [53, 1000.0, 1877.0, 60],              # off 2 ms after s_3: seen live only with late >= 2
             [50, 1500.0, 2400.0, 70], [50, 1900.0, 2600.0, 71],   # re-strike of a held pitch
             [45, 1875.0, 2812.5, 55],              # on exactly at s_3, off exactly at s_4
             [55, 2814.0, 3000.0, 60]]              # on 1.5 ms after s_4: seen live only with late >= 1.5

    def live(self, s, late):
        """Zero-jitter live state: every feeder event that arrived by the snapshot at music time s + late."""
        from rt_lh_feeder import events
        lh = li.LHState()
        for t, kind, pitch, vel in events(self.NOTES, self.OFF):
            if self.T0 + t <= s + late:
                lh.feed(kind, self.T0 + t, pitch, vel)
        return lh

    def test_replay_matches_zero_jitter_live_at_every_block(self):
        B = 937.5
        for late in (0.0, 0.5, 5.0, 400.0):
            replay, lh_r = li.LHReplay(self.NOTES, self.T0, self.OFF), li.LHState()
            for k in range(1, 6):
                s_k = self.T0 + k * B
                replay.feed_until(lh_r, s_k)
                start = max(s_k - W_MS, 0.0)
                want = window_notes(Human(self.live(s_k, late).human()), s_k, start)
                got = window_notes(Human(lh_r.human()), s_k, start)
                self.assertEqual(got, want, (late, k))

    def test_feed_is_deterministic_and_idempotent(self):
        a, b = li.LHReplay(self.NOTES, self.T0, self.OFF), li.LHReplay(self.NOTES, self.T0, self.OFF)
        la, lb = li.LHState(), li.LHState()
        for k in range(1, 6):
            s_k = self.T0 + k * 937.5
            fa = a.feed_until(la, s_k)
            self.assertEqual(a.feed_until(la, s_k), [])                         # nothing is fed twice
            self.assertEqual(fa, b.feed_until(lb, s_k))
            self.assertEqual([vars(n) for n in la.human()], [vars(n) for n in lb.human()])

    def test_live_arrival_jitter_can_change_tokens_replay_cannot(self):
        """Positive control for the #1825 cause: #1819 s3/s4 A, note 47 arrived 0.7 vs 4.4 ms late; cut at s_2 its
        length 937 vs 933 ms falls on 940 vs 930 ms tokens. Replay always uses the planned time."""
        from lh_context import note_tokens
        s_2 = self.T0 + 2 * 937.5

        def tokens(arrival_late):
            lh = li.LHState()
            lh.feed("note_on", self.T0 + self.OFF + arrival_late, 47, 50)
            return [note_tokens(n) for n in window_notes(Human(lh.human()), s_2, 0.0)]
        self.assertNotEqual(tokens(0.69), tokens(4.38))
        r1, r2, l1, l2 = li.LHReplay(self.NOTES[:1], self.T0), li.LHReplay(self.NOTES[:1], self.T0), li.LHState(), li.LHState()
        r1.feed_until(l1, s_2)
        r2.feed_until(l2, s_2)
        self.assertEqual([note_tokens(n) for n in window_notes(Human(l1.human()), s_2, 0.0)],
                         [note_tokens(n) for n in window_notes(Human(l2.human()), s_2, 0.0)])

    def test_events_are_the_feeder_events_on_the_music_clock(self):
        from rt_lh_feeder import events
        r = li.LHReplay(self.NOTES, self.T0, self.OFF)
        self.assertEqual(r.events, [(self.T0 + t, k, p, v) for t, k, p, v in events(self.NOTES, self.OFF)])


if __name__ == "__main__":
    unittest.main()
