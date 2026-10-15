"""Block-request training input contract (issue #1756). Pure Python, synthetic token streams."""
from __future__ import annotations

import os
import random
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from block_contract import BLOCK, block_start, horizon, snapshot  # noqa: E402
from block_request import (COLD, committing_request, loss_class, payload, position_requests,  # noqa: E402
                           primer_end, request_table, simulate_kv)
from lh_role_proxy import Note  # noqa: E402
from stream_contract import SEGMENT_MS, TokenStream  # noqa: E402

HEADER = [("prefix", "instrument", "piano"), "<S>"]


def aria_tokens(notes):
    """Aria abs tokens for (pitch, onset ms, dur ms, velocity) notes on a 10 ms grid: <T> at every 5 s boundary
    passed before a note, onsets relative to the segment, <E> at the end."""
    out, seg = list(HEADER), 0
    for p, on, dur, vel in sorted(notes, key=lambda n: (n[1], n[0])):
        while on >= (seg + 1) * SEGMENT_MS:
            out.append("<T>")
            seg += 1
        out += [("piano", p, vel), ("onset", on - seg * SEGMENT_MS), ("dur", dur)]
    return out + ["<E>"]


def human(*spec):
    return [Note(i, p, on, off) for i, (p, on, off) in enumerate(spec)]


def payloads_for(rows, k, humans):
    return [payload(humans, block_start(j, 0.0) if j >= 0 else 0.0, j) if j >= 0 else COLD for j in position_requests(rows, k)]


class FutureOnsetTest(unittest.TestCase):
    """Review Q2: the onset of the note being predicted must not choose the condition."""

    def test_next_target_onset_does_not_change_the_request_condition(self):
        humans = human((40, 1200, 3000))                                       # PC 4 starts at 1200
        base = [(60, 300, 100, 80), (62, 900, 100, 80)]
        early, late = aria_tokens(base + [(64, 2000, 100, 80)]), aria_tokens(base + [(64, 2400, 100, 80)])
        re, rl = request_table(early, 0.0), request_table(late, 0.0)
        self.assertEqual(payloads_for(re, 1, humans), payloads_for(rl, 1, humans))
        self.assertEqual(payload(humans, block_start(1, 0.0), 1), ((0,) * 12, 1))   # s_1 = 937.5: the 1200 note is not seen
        self.assertEqual(loss_class(rl, 1)[1], ["boundary"] * 3)               # late: window 1 empty, the 2400 note is its boundary
        self.assertEqual(loss_class(re, 1)[1], ["window"] * 3 + ["boundary"] * 0)
        self.assertEqual(payload(humans, block_start(2, 0.0), 2)[0][4], 1)     # request 2 (s = 1875) sees it
        # the rejected rule (snapshot chosen by the target onset) would give the two cases different conditions
        self.assertNotEqual(payload(humans, snapshot(2000, 0.0), 1), payload(humans, snapshot(2400, 0.0), 2))

    def test_any_target_suffix_keeps_every_condition_of_the_request(self):
        rng = random.Random(3)
        humans = human(*[(rng.randint(30, 54), on, on + rng.choice([80, 600, 2500])) for on in range(0, 12000, 350)])
        base = [(rng.randint(55, 90), 10 * rng.randint(0, 1200), 100, 64) for _ in range(40)]
        toks = aria_tokens(base)
        rows = request_table(toks, 0.0)
        for k in range(1, len(rows)):
            cut = rows[k]["h_prev"]
            kept = [n for n in base if n[1] < cut]
            alt = aria_tokens(kept + [(rng.randint(55, 90), int(cut) + 10 * rng.randint(1, 300), 50, 64) for _ in range(30)])
            ra = request_table(alt, 0.0)
            if k >= len(ra):
                continue
            p0 = rows[k]["ctx_end"] - 1
            self.assertEqual(alt[:ra[k]["ctx_end"]], toks[:rows[k]["ctx_end"]], k)
            pa, pb = payloads_for(ra, k, humans), payloads_for(rows, k, humans)
            self.assertEqual(pa[:p0], pb[:p0], k)                                   # context: same history, same payloads
            ck = payload(humans, block_start(k, 0.0), k)
            self.assertTrue(all(x == ck for x in pa[p0:] + pb[p0:]), k)             # every prediction position: c_k


class HumanHistoryTest(unittest.TestCase):
    def test_human_events_after_the_snapshot_are_invisible(self):
        s = block_start(3, 0.0)                                               # 2812.5
        before = human((40, 1000, 2900), (43, 2000, 2500))
        for later in (human((40, 1000, 4000), (43, 2000, 2500)),              # release after s moved
                      human((40, 1000, 2900), (43, 2000, 2500), (47, 2900, 3500)),   # note-on after s added
                      human((40, 1000, 2813), (43, 2000, 2500))):             # release still after s
            self.assertEqual(payload(later, s, 3), payload(before, s, 3))
        self.assertNotEqual(payload(human((40, 1000, 2800), (43, 2000, 2500)), s, 3), payload(before, s, 3))  # control


class AlignmentTest(unittest.TestCase):
    """Review: the first label of a request and the position that predicts it, for the three kinds of p0."""

    def check(self, toks, rows, k, p0_token):
        r = rows[k]
        p0 = r["ctx_end"] - 1
        self.assertEqual(toks[p0], p0_token)
        first, classes = loss_class(rows, k)
        self.assertEqual(first, p0 + 1)
        self.assertEqual(len(classes), r["cont_end"] - r["ctx_end"])
        self.assertEqual(position_requests(rows, k)[p0], k)                   # the first label is predicted under c_k
        self.assertEqual(len(position_requests(rows, k)), r["cont_end"] - 1)

    def test_bos_previous_dur_and_segment_boundary(self):
        toks = aria_tokens([(60, 0, 100, 80), (62, 4000, 100, 80), (64, 6000, 100, 80)])
        rows = request_table(toks, 0.0)
        self.check(toks, rows, 0, "<S>")                                       # nothing committed yet: BOS
        k_t = next(r["k"] for r in rows if r["ctx_end"] < r["win_end"] and toks[r["win_end"] - 1] == "<T>")
        self.assertEqual(toks[rows[k_t]["ctx_end"]:rows[k_t]["win_end"]], ["<T>"])   # a window holding only <T>
        self.check(toks, rows, k_t + 1, "<T>")
        k_d = next(r["k"] for r in rows if r["k"] > 0 and isinstance(toks[r["ctx_end"] - 1], tuple))
        self.assertEqual(toks[rows[k_d]["ctx_end"] - 1][0], "dur")
        self.check(toks, rows, k_d, toks[rows[k_d]["ctx_end"] - 1])


class EmptyColdTest(unittest.TestCase):
    def test_cold_warm_empty_and_empty_window(self):
        humans = human((40, 5000, 6000))
        self.assertEqual(payload(humans, block_start(0, 0.0), 0), COLD)
        self.assertEqual(payload(humans, block_start(2, 0.0), 2), ((0,) * 12, 1))   # observed, nothing sounding
        self.assertNotEqual(COLD, ((0,) * 12, 1))
        toks = aria_tokens([(60, 0, 100, 80), (62, 3000, 100, 80)])
        rows = request_table(toks, 0.0)
        empty = [r for r in rows if r["ctx_end"] == r["win_end"]]
        self.assertTrue(empty)
        for r in empty:
            self.assertTrue(r["boundary"])
            self.assertEqual(loss_class(rows, r["k"])[1], ["boundary"] * 3)

    def test_primer_is_context_with_the_cold_payload(self):
        toks = aria_tokens([(60, 100, 50, 80), (62, 600, 50, 80), (64, 1200, 50, 80)])
        t0 = 468.75
        self.assertEqual(primer_end(toks, t0), 2 + 3)                          # the 100 ms note only
        rows = request_table(toks, t0)
        self.assertEqual(rows[0]["ctx_end"], 5)
        self.assertEqual(committing_request(rows, 5), [-1] * 5)
        self.assertEqual(rows[0]["s"], t0 - 1000.0)


class HistoryTest(unittest.TestCase):
    def test_context_is_exactly_the_committed_model_stream(self):
        rng = random.Random(5)
        notes = [(rng.randint(55, 95), 10 * rng.randint(0, 2000), 10 * rng.randint(1, 60), 64) for _ in range(150)]
        toks = aria_tokens(notes)
        for t0 in (0.0, 234.375, 703.125):
            rows = request_table(toks, t0)
            p_end = primer_end(toks, t0)
            s = TokenStream(list(toks[:p_end]))
            for t in toks[p_end:-1]:
                s.append_sampled(t)
            self.assertEqual(rows[0]["ctx_end"], p_end)
            for r in rows:
                self.assertEqual(s.committed_end, r["ctx_end"])                      # context = committed before R_k
                s.commit_until(r["h"])
                self.assertEqual(s.committed_end, r["win_end"])
            windows = [i for r in rows for i in range(r["ctx_end"], r["win_end"])]
            self.assertEqual(windows, list(range(p_end, rows[-1]["win_end"])))   # every token once, in order
            self.assertEqual(rows[-1]["win_end"], len(toks) - 1)                 # all but <E>
            pitches = {t[1] for t in toks if isinstance(t, tuple) and t[0] == "piano"}
            self.assertTrue(all(p >= 55 for p in pitches))                       # model history only, no human notes

    def test_context_payloads_match_the_sequential_runtime_replay(self):
        for seed in range(4):
            rng = random.Random(seed)
            notes = [(rng.randint(55, 95), 10 * rng.randint(0, 1500), 10 * rng.randint(1, 80), 64) for _ in range(90)]
            humans = human(*[(rng.randint(28, 54), on, on + rng.choice([60, 400, 1500, 4000])) for on in
                             sorted(rng.uniform(0, 15000) for _ in range(40))])
            toks = aria_tokens(notes)
            for t0 in (0.0, 234.375, 468.75, 703.125):
                rows = request_table(toks, t0)
                kv = simulate_kv(toks, t0, humans)
                self.assertEqual(sorted(kv), [r["k"] for r in rows])
                for r in rows:
                    owner = position_requests(rows, r["k"])[:r["ctx_end"] - 1]
                    want = [COLD if j < 0 else payload(humans, block_start(j, t0), j) for j in owner]
                    self.assertEqual(kv[r["k"]], want, (seed, t0, r["k"]))

    def test_a_stream_cut_after_a_segment_boundary_commits_that_boundary(self):
        toks = aria_tokens([(60, 0, 100, 80), (62, 4000, 100, 80), (64, 6000, 100, 80)])
        rows = request_table(toks, 0.0)
        k = next(r["k"] for r in rows if toks[r["ctx_end"] - 1] == "<T>")       # a context that ends with <T>
        cut = rows[k]["ctx_end"]
        pre = request_table(list(toks[:cut]) + ["<E>"], 0.0)
        self.assertEqual(committing_request(pre, cut), committing_request(rows, cut))
        self.assertEqual(pre[-1]["win_end"], cut)
        self.assertFalse(pre[-1]["boundary"])
        kv = simulate_kv(list(toks[:cut]) + ["<E>"], 0.0, [])
        self.assertEqual(sorted(kv), [r["k"] for r in pre])

    def test_windows_follow_the_producer_horizons(self):
        toks = aria_tokens([(60 + i % 30, 10 * i * 37, 100, 64) for i in range(60)])
        rows = request_table(toks, 0.0)
        for r in rows:
            self.assertEqual((r["h_prev"], r["h"]), (0.0 if r["k"] == 0 else horizon(r["k"] - 1, 0.0), horizon(r["k"], 0.0)))
            self.assertEqual(r["h"] - r["h_prev"], BLOCK if r["k"] else BLOCK + 468.75)



class ExportSongTest(unittest.TestCase):
    def test_build_song_checks_are_zero_and_requests_carry_the_payload(self):
        from lh_request_export import build_song
        rng = random.Random(9)
        notes = [(rng.randint(55, 95), 10 * rng.randint(0, 2500), 10 * rng.randint(1, 80), 64) for _ in range(160)]
        humans = human(*[(rng.randint(28, 54), on, on + rng.choice([60, 400, 1500])) for on in
                         sorted(rng.uniform(0, 25000) for _ in range(60))])
        for idx in range(4):
            res = build_song(idx, aria_tokens(notes), humans)
            self.assertEqual(res["checks"], {"c1_replay_mismatch": 0, "c2_prefix_mismatch": 0,
                                             "c3_observed_only_mismatch": 0, "c4_low_pitch_tokens": 0})
            reqs = res["requests"]
            self.assertEqual(reqs[0]["available"], 0)
            self.assertTrue(all(r["available"] == 1 for r in reqs[1:]))
            self.assertTrue(any(any(r["pc12"]) for r in reqs))
            self.assertEqual(sum(r["window_tokens"] for r in reqs), len(aria_tokens(notes)) - 1 - res["primer_end"])

    def test_a_low_note_in_the_model_stream_is_counted(self):
        from lh_request_export import build_song
        res = build_song(0, aria_tokens([(60, 0, 100, 64), (40, 500, 100, 64)]), [])
        self.assertEqual(res["checks"]["c4_low_pitch_tokens"], 1)


if __name__ == "__main__":
    unittest.main()
