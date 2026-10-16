"""Context crop + full-reset contract (issue #1758). Pure Python, synthetic token streams, small capacities."""
from __future__ import annotations

import os
import random
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, os.path.dirname(__file__))
from block_contract import block_start  # noqa: E402
from block_request import COLD, committing_request, payload  # noqa: E402
from aria_mlx_stream_loop import fill_block  # noqa: E402
from crop_contract import (CAP, CAPACITY, HEADER, CapacityFailure, absolute_times, choose_cut, needs_crop, reserve,  # noqa: E402
                           session_position_requests, session_requests, session_tokens, simulate_session_kv)
from stream_contract import TokenStream  # noqa: E402
from lh_role_proxy import Note  # noqa: E402
from test_block_request import aria_tokens  # noqa: E402

SMALL = {"cap": 4, "capacity": 120, "keep_target": 60}   # every synthetic segment stays under 120 - 8 tokens


def song(seed, n=140, span=60000):
    rng = random.Random(seed)
    notes = [(rng.randint(55, 95), 10 * rng.randint(0, span // 10), 10 * rng.randint(1, 120), 64) for _ in range(n)]
    notes += [(72, 9990, 100, 64), (76, 10010, 100, 64)]                       # both sides of a 5 s boundary
    notes += [(60, 20000, 100, 64), (64, 20000, 100, 64), (67, 20000, 100, 64)]   # a chord on one onset
    notes += [(84, 30000, 4500, 64)]                                            # a long note
    return notes


def humans(seed, span=60000):
    rng = random.Random(seed + 100)
    return [Note(i, rng.randint(28, 54), on, on + rng.choice([80, 600, 3000])) for i, on in
            enumerate(sorted(rng.uniform(0, span) for _ in range(60)))]


class DecisionTest(unittest.TestCase):
    def test_exact_capacity_boundary_and_one_more(self):
        self.assertEqual(reserve(4), 2 * 4 + 3)
        self.assertFalse(needs_crop(60 - 11, 4, 60))
        self.assertTrue(needs_crop(60 - 10, 4, 60))
        self.assertFalse(needs_crop(4096 - 131))
        self.assertTrue(needs_crop(4096 - 130))

    def test_cut_policy(self):
        n = ("piano", 60, 64), ("onset", 0), ("dur", 10)
        sess = list(HEADER) + list(n) * 4 + ["<T>"] + list(n) * 4 + ["<T>"] + list(n) * 2
        ce = len(sess)
        t1, t2 = sess.index("<T>"), ce - 1 - 6
        self.assertEqual(choose_cut(sess, ce, 4, 60, 30), t1)                     # earliest <T> under the target
        self.assertEqual(choose_cut(sess, ce, 4, 60, 10), t2)                     # only the later <T> is under 10
        self.assertEqual(choose_cut(sess, ce, 4, 19, 5), t2)                      # target missed, the latest safe <T>
        with self.assertRaises(CapacityFailure):
            choose_cut(sess, ce, 4, 18, 5)                                        # no <T> makes room
        with self.assertRaises(ValueError):
            choose_cut(sess, ce, 4, 40, 30)                                       # keep_target itself is not safe
        with self.assertRaises(CapacityFailure):
            choose_cut(list(HEADER) + list(n) * 30, 92, 4, 60, 30)                # no <T> at all

    def test_empty_kept_history_and_consecutive_boundaries(self):
        n = ("piano", 60, 64), ("onset", 0), ("dur", 10)
        sess = list(HEADER) + list(n) * 12 + ["<T>"]                               # history ends on a <T>
        self.assertEqual(choose_cut(sess, len(sess), 4, 60, 30), len(sess) - 1)   # kept history is empty
        sess2 = list(HEADER) + list(n) * 12 + ["<T>", "<T>", "<T>"]
        j = choose_cut(sess2, len(sess2), 4, 60, 30)
        self.assertEqual(sess2[j + 1:], ["<T>", "<T>"])                            # the earliest of the run is cut


class SessionTest(unittest.TestCase):
    def test_absolute_times_survive_repeated_crops(self):
        toks = aria_tokens(song(1))
        srows = session_requests(toks, 0.0, **SMALL)
        crops = [r for r in srows if r["crop"]]
        self.assertGreaterEqual(len(crops), 2)
        for r in srows:
            sess = session_tokens(toks, r)
            ctx = sess[:r["ctx_len"]]
            orig = absolute_times(list(toks[:r["ctx_end"]]), 0.0)
            kept = absolute_times(ctx, r["base_ms"])
            self.assertEqual(kept, orig[len(orig) - len(kept):], r["k"])          # same onset, pitch, dur
            if r["keep_from"] is not None:
                self.assertEqual(toks[r["keep_from"] - 1], "<T>")                  # cut right after a <T>
                self.assertEqual(r["base_ms"], toks[:r["keep_from"]].count("<T>") * 5000.0)   # cumulative offset

    def test_crop_happens_exactly_when_the_committed_length_needs_it(self):
        toks = aria_tokens(song(2))
        srows = session_requests(toks, 0.0, **SMALL)
        prev_keep = None
        for r in srows:
            before = r["ctx_end"] if prev_keep is None else len(HEADER) + r["ctx_end"] - prev_keep
            self.assertEqual(r["crop"] is not None, needs_crop(before, SMALL["cap"], SMALL["capacity"]), r["k"])
            self.assertFalse(needs_crop(r["ctx_len"], SMALL["cap"], SMALL["capacity"]), r["k"])   # always safe after
            prev_keep = r["keep_from"]

    def test_provenance_matches_the_sequential_session_replay(self):
        crops = 0
        for seed in range(4):
            toks, hum = aria_tokens(song(seed)), humans(seed)
            for t0 in (0.0, 234.375, 703.125):
                srows = session_requests(toks, t0, **SMALL)
                kv = simulate_session_kv(toks, t0, hum, **SMALL)
                self.assertEqual(sorted(kv), [r["k"] for r in srows])
                for r in srows:
                    want = [COLD if j < 0 else payload(hum, block_start(j, t0), j)
                            for j in session_position_requests(toks, srows, r["k"])[:r["ctx_len"] - 1]]
                    got, base, cut = kv[r["k"]]
                    self.assertEqual(got, want, (seed, t0, r["k"]))
                    self.assertEqual((base, cut), (r["base_ms"], r["crop"]["cut"] if r["crop"] else None))
                    crops += r["crop"] is not None
        self.assertGreater(crops, 10)

    def test_new_start_token_predicts_the_first_kept_token_with_its_old_condition(self):
        toks, hum = aria_tokens(song(5)), humans(5)
        srows = session_requests(toks, 0.0, **SMALL)
        rows = [{x: r[x] for x in ("k", "ctx_end", "win_end", "cont_end")} for r in srows]
        differs = 0
        for r in srows:
            if r["crop"] is None or r["ctx_len"] <= len(HEADER):
                continue
            tags = session_position_requests(toks, srows, r["k"])
            owner = committing_request(rows, r["keep_from"] + 1)[r["keep_from"]]
            self.assertEqual(tags[0], -1)                                           # prefix -> <S>: cold
            self.assertEqual(tags[1], owner)                                        # <S> -> first kept token
            self.assertEqual(tags[r["ctx_len"] - 1], r["k"])                        # p0 -> current request
            differs += payload(hum, block_start(owner, 0.0), owner) != payload(hum, block_start(r["k"], 0.0), r["k"])
        self.assertGreater(differs, 0)                                              # a past condition kept, not replaced

    def test_empty_kept_history_makes_the_start_token_p0(self):
        dense = [(60 + i % 30, 5000 + 40 * i, 30, 64) for i in range(20)]           # 60 tokens in one segment
        toks = aria_tokens([(60, 0, 50, 64)] + dense + [(70, 12000, 50, 64)])
        srows = session_requests(toks, 0.0, cap=4, capacity=77, keep_target=30)   # crop first when 67 tokens end on <T>(10 s)
        empty = [r for r in srows if r["crop"] and r["ctx_len"] == len(HEADER)]
        self.assertTrue(empty)
        for r in empty:
            self.assertEqual(session_position_requests(toks, srows, r["k"])[1], r["k"])

    def test_a_segment_too_long_for_the_capacity_fails_loudly(self):
        dense = [(60 + i % 30, 40 * i, 30, 64) for i in range(40)]                  # 120 tokens before any <T> > 120 - 8
        with self.assertRaises(CapacityFailure):
            session_requests(aria_tokens(dense + [(70, 9000, 50, 64)]), 0.0, **SMALL)

    def test_future_suffix_does_not_change_cut_offset_or_payload(self):
        rng = random.Random(7)
        base = song(7)
        toks = aria_tokens(base)
        srows = session_requests(toks, 0.0, **SMALL)
        for r in srows[1::3]:
            cut_ms = r["h_prev"]
            alt = aria_tokens([n for n in base if n[1] < cut_ms] +
                              [(rng.randint(55, 95), int(cut_ms) + 10 * rng.randint(1, 3000), 50, 64) for _ in range(60)])
            sa = session_requests(alt, 0.0, **SMALL)
            keys = ("ctx_end", "keep_from", "base_ms", "ctx_len", "crop")
            self.assertEqual([{x: s[x] for x in keys} for s in sa[:r["k"] + 1]],
                             [{x: s[x] for x in keys} for s in srows[:r["k"] + 1]], r["k"])

    def test_committed_notes_keep_their_scheduled_times_across_a_crop(self):
        """Scheduler note-offs are computed from absolute times once; a crop must not change any committed note's time."""
        toks = aria_tokens(song(3))
        srows = session_requests(toks, 0.0, **SMALL)
        for prev, r in zip(srows, srows[1:]):
            if r["crop"] is None:
                continue
            before = absolute_times(session_tokens(toks, prev)[:prev["ctx_len"]], prev["base_ms"])
            after = absolute_times(session_tokens(toks, r)[:r["ctx_len"]], r["base_ms"])
            self.assertTrue(set(after) <= set(before) | set(absolute_times(list(toks[:r["ctx_end"]]), 0.0)))
            self.assertEqual(after, absolute_times(list(toks[:r["ctx_end"]]), 0.0)[-len(after):] if after else [])



class RealCallSafetyTest(unittest.TestCase):
    """#1759 review H1: with the real fill_block over consecutive calls, a call that starts with needs_crop(committed)
    false never feeds a position >= capacity. The sampler is adversarial within the grammar."""

    @staticmethod
    def coverage_of(stream):
        st = stream.grammar_state()
        return st.segment * 5000 + st.last_onset if st.expect == "note" else -1

    @staticmethod
    def history(length):
        """A committed history of exactly `length` tokens: 3-token notes, padded with <D> (a no-op in the grammar)."""
        notes = (length - len(HEADER)) // 3
        return list(HEADER) + [("piano", 60, 64), ("onset", 0), ("dur", 10)] * notes + ["<D>"] * (length - len(HEADER) - 3 * notes)

    def test_review_counterexample_is_now_safe_and_the_old_bound_was_not(self):
        cap, capacity = CAP, CAPACITY
        for committed, crop_expected in ((capacity - reserve(cap), False), (capacity - reserve(cap) + 1, True)):
            s = TokenStream(self.history(committed))
            self.assertEqual(s.committed_end, committed)
            self.assertEqual(needs_crop(s.committed_end, cap, capacity), crop_expected)
            for t in (("piano", 61, 64), ("onset", 1200), ("dur", 10)):                # a boundary note past h
                s.append_sampled(t)
            fed = []

            def same_onset():
                exp = s.grammar_state().expect
                fed.append(len(s.tokens) - 1)
                s.append_sampled({"note": ("piano", 62, 64), "onset": ("onset", 1200), "dur": ("dur", 10)}[exp])
            fill_block(same_onset, lambda: self.coverage_of(s), lambda: s.commit_until(1000), 1000, 1937.5, cap)
            self.assertEqual(len(s.tokens) - s.committed_end, 3 + cap)                 # provisional 67
            fill_block(same_onset, lambda: self.coverage_of(s), lambda: s.commit_until(1937.5), 1937.5, 2875.0, cap)
            if not crop_expected:
                self.assertLess(max(fed), capacity)
                self.assertEqual(max(fed), committed + (3 + cap) + cap - 2)
        old_committed = capacity - 2 * cap                                             # run 1's bound let this through
        self.assertGreaterEqual(old_committed + (3 + cap) + cap - 2, capacity)

    def test_fuzz_consecutive_calls(self):
        beat, block = 468.75, 937.5
        worst, calls = float("-inf"), 0
        for seed in range(300):
            rng = random.Random(seed)
            cap = rng.choice([3, 4, 5, 7])
            capacity = rng.choice([60, 80, 120])
            s = TokenStream(list(HEADER))
            fed = []

            def sample():
                st = s.grammar_state()
                fed.append(len(s.tokens) - 1)
                if st.expect == "note":
                    s.append_sampled("<T>" if rng.random() < 0.1 else ("piano", rng.randint(55, 90), 64))
                elif st.expect == "onset":
                    step = rng.choice([0, 0, 0, 10, 470, 940, 1900])               # often the same onset (stall)
                    s.append_sampled(("onset", st.last_onset + step if st.last_onset + step <= 4990 else st.last_onset))
                else:
                    s.append_sampled(("dur", 10 * rng.randint(1, 50)))
            for k in range(400):
                h = (k + 1) * block + beat
                if needs_crop(s.committed_end, cap, capacity):
                    break                                                           # the crop path takes over here
                self.assertLessEqual(len(s.tokens) - s.committed_end, cap + 3, (seed, k))   # provisional bound
                if rng.random() < 0.2:
                    s.rollback_to_committed()                                       # a condition change
                fed.clear()
                fill_block(sample, lambda: self.coverage_of(s), lambda h=h: s.commit_until(h), h, h + block, cap)
                if fed:
                    self.assertLess(max(fed), capacity, (seed, k))
                    worst = max(worst, max(fed) - capacity)
                calls += 1
        self.assertGreater(calls, 3000)
        self.assertLess(worst, 0)


if __name__ == "__main__":
    unittest.main()
