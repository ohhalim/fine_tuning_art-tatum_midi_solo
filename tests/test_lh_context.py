"""Observed left hand in the Aria context (issue #1772). Pure Python, synthetic token streams."""
from __future__ import annotations

import os
import random
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, os.path.dirname(__file__))
from crop_contract import session_requests  # noqa: E402
from lh_context import (END_SLACK_MS, ONSET_SLACK_MS, ContractViolation, Human, LHNote, build_request,  # noqa: E402
                        check_request, closest, grammar_ok, merge, note_tokens, q_dur, q_onset, q_velocity,
                        quantized_end, quantized_on, strip_lh, window_notes, DUR_GRID)
from lh_role_proxy import Note  # noqa: E402
from test_block_request import aria_tokens  # noqa: E402

SMALL = {"cap": 4, "capacity": 120, "keep_target": 60}


def rh_song(seed, n=120, span=40000):
    rng = random.Random(seed)
    notes = [(rng.randint(55, 95), 10 * rng.randint(0, span // 10), 10 * rng.randint(1, 80), 70) for _ in range(n)]
    return [t if not isinstance(t, list) else tuple(t) for t in aria_tokens(notes)]


def lh_song(seed, n=60, span=40000):
    rng = random.Random(seed)
    out = []
    for i in range(n):
        on = rng.uniform(0, span)
        out.append(Note(i, rng.randint(36, 54), on, on + rng.uniform(50, 2500), rng.randint(30, 110)))
    return out


def pool_of(songs):
    """songs: {name: (rh tokens, human notes)} -> donor pool {name: (Human, [(k, s), ...])}."""
    return {name: (Human(h), [(r["k"], r["s"]) for r in session_requests(tok, 0.0)]) for name, (tok, h) in songs.items()}


class QuantizeTest(unittest.TestCase):
    def test_closest_matches_a_linear_scan_away_from_ties(self):
        for n in range(-7, 5013, 3):
            if n % 10 == 5:
                continue
            want = min(DUR_GRID, key=lambda g: abs(g - n))
            self.assertEqual(closest(n, DUR_GRID), want, n)

    def test_rules(self):
        self.assertEqual(q_dur(0), 10)
        self.assertEqual(q_dur(3), 10)
        self.assertEqual(q_dur(7000), 5000)
        self.assertEqual(q_onset(4996), 4990)
        self.assertEqual(q_velocity(64), 60)
        self.assertEqual(q_velocity(3), 10)


class WindowTest(unittest.TestCase):
    def test_same_time_order(self):
        s = 1875.0
        h = [Note(0, 40, s, s + 500), Note(1, 41, 1000, s), Note(2, 42, 1000, s + 0.001), Note(3, 43, 999, 1500)]
        w = {n.pitch: n for n in window_notes(h, s, s - 7500)}
        self.assertNotIn(40, w)                     # onset == s_k is not observed yet
        self.assertFalse(w[41].held)                # off == s_k is released
        self.assertTrue(w[42].held)                 # off > s_k is held and cut
        self.assertEqual(w[42].raw_off, s)
        self.assertFalse(w[43].held)

    def test_window_start(self):
        h = [Note(0, 40, 99.0, 200), Note(1, 41, 100.0, 200)]
        self.assertEqual([n.pitch for n in window_notes(h, 1000.0, 100.0)], [41])


class TokenErrorTest(unittest.TestCase):
    """Astra review 2: quantization error is not a future leak; raw cut and token error are checked apart."""

    def test_two_and_a_half_ms_note(self):
        n = window_notes([Note(0, 40, 1000.0, 3000.0, 64)], 1002.5, 0.0)[0]
        self.assertEqual(n.raw_off, 1002.5)
        self.assertEqual(note_tokens(n)[2][2], ("dur", 10))
        self.assertEqual(quantized_end(n), 1010)                      # s + 7.5, inside the bound
        self.assertLessEqual(quantized_end(n), n.raw_off + END_SLACK_MS)

    def test_onset_and_min_dur_rounding_overlap(self):
        n = window_notes([Note(0, 40, 1004.6, 3000.0, 64)], 1006.0, 0.0)[0]   # onset 1005 -> 1010? dur 1 -> 10
        self.assertLessEqual(quantized_on(n), n.raw_on + ONSET_SLACK_MS)
        self.assertLessEqual(quantized_end(n), n.raw_off + END_SLACK_MS)
        self.assertGreater(quantized_end(n), n.raw_off)               # past s_k by rounding only

    def test_bound_holds_on_a_dense_grid(self):
        for on10 in range(0, 400):
            on = 1000 + on10 * 0.37
            for cut in (0.2, 2.5, 4.9, 5.0, 5.1, 9.7, 33.3):
                n = window_notes([Note(0, 40, on, on + 99999, 64)], on + cut, 0.0)[0]
                self.assertLessEqual(quantized_on(n), n.raw_on + ONSET_SLACK_MS)
                self.assertLessEqual(quantized_end(n), n.raw_off + END_SLACK_MS)


class MergeTest(unittest.TestCase):
    def setUp(self):
        self.ctx = [t if not isinstance(t, list) else tuple(t) for t in aria_tokens(
            [(60, 100, 50, 70), (64, 300, 50, 70), (67, 5200, 50, 70)])[:-1]]   # drop <E>

    def test_rh_untouched_and_ties_go_first(self):
        n = LHNote(40, 300, 400, False, 300.0, 400.0, 64)
        out, tags = merge(self.ctx, 0, [n])
        self.assertEqual(strip_lh(out, tags), self.ctx)
        self.assertTrue(grammar_ok(out))
        i = tags.index("lh")
        self.assertEqual(out[i], ("piano", 40, 60))
        self.assertEqual(out[i + 3], ("piano", 64, 70))              # same onset: human first

    def test_note_after_last_segment_raises(self):
        n = LHNote(40, 12000, 12100, False, 12000.0, 12100.0, 64)
        with self.assertRaises(ContractViolation):
            merge(self.ctx, 0, [n])

    def test_second_segment(self):
        n = LHNote(41, 5100, 5150, False, 5100.0, 5150.0, 64)
        out, tags = merge(self.ctx, 0, [n])
        self.assertTrue(grammar_ok(out))
        self.assertEqual(out[out.index("<T>") + 1], ("piano", 41, 60))


class BuildTest(unittest.TestCase):
    def setUp(self):
        self.songs = {s: (rh_song(10 + s), lh_song(20 + s)) for s in (1, 2, 3)}
        self.pool = pool_of(self.songs)

    def build(self, song, k, human=None, **kw):
        tok, h = self.songs[song]
        srows = session_requests(tok, 0.0, **{x: kw[x] for x in kw if x in SMALL})
        return build_request(tok, srows[k], Human(h if human is None else human), song, self.pool, **kw), srows

    def test_checks_zero_on_every_request(self):
        totals = {}
        for song in self.songs:
            tok, h = self.songs[song]
            for r in session_requests(tok, 0.0):
                b = build_request(tok, r, Human(h), song, self.pool)
                for key, val in check_request(b).items():
                    totals[key] = totals.get(key, 0) + val
        self.assertEqual(set(totals.values()), {0}, totals)

    def test_cold_request_has_no_human_notes(self):
        r, _ = self.build(1, 0)
        self.assertEqual(r["a"]["tokens"], r["b"]["tokens"])
        self.assertEqual(r["a"]["tokens"], r["c"]["tokens"])
        self.assertEqual(r["layer"], "empty")

    def test_donor_is_another_song_and_matches(self):
        seen = 0
        tok, h = self.songs[1]
        for srow in session_requests(tok, 0.0):
            r = build_request(tok, srow, Human(h), 1, self.pool)
            if isinstance(r["donor"], dict):
                seen += 1
                self.assertNotEqual(r["donor"]["song"], 1)
                self.assertTrue(r["eligible_c"])
                self.assertEqual(len(r["donor_notes"]), r["lh_notes"])
        self.assertGreater(seen, 0)

    def test_future_human_changes_do_not_change_inputs(self):
        """L1f: only onsets and offs after s_k change; a/b/c tokens, donor and crop must not."""
        tok, h = self.songs[2]
        for srow in session_requests(tok, 0.0)[1:12]:
            s = srow["s"]
            fut = [Note(n.nid, n.pitch, n.on, n.off + 1000 if n.off > s else n.off, n.vel) for n in h]
            fut += [Note(900 + i, 45, s + 10 * i, s + 10 * i + 300, 80) for i in range(5)]
            r1 = build_request(tok, srow, Human(h), 2, self.pool)
            r2 = build_request(tok, srow, Human(fut), 2, self.pool)
            for arm in "abc":
                self.assertEqual(r1[arm] and r1[arm]["tokens"], r2[arm] and r2[arm]["tokens"], (srow["k"], arm))
            self.assertEqual((r1["donor"], r1["keep_from"], r1["base_ms"]), (r2["donor"], r2["keep_from"], r2["base_ms"]))

    def test_target_future_does_not_change_context(self):
        tok, h = self.songs[3]
        srows = session_requests(tok, 0.0)
        srow = srows[5]
        alt = list(tok[:srow["cont_end"]]) + [("piano", 90, 70), ("onset", 4000), ("dur", 100)] + list(tok[srow["cont_end"]:])
        r1 = build_request(tok, srow, Human(h), 3, self.pool)
        r2 = build_request(alt, srow, Human(h), 3, self.pool)
        self.assertEqual(r1["b"]["tokens"], r2["b"]["tokens"])

    def test_missing_donor(self):
        tok, h = self.songs[1]
        srow = session_requests(tok, 0.0)[10]
        crowd = h + [Note(500 + i, 40, srow["s"] - 300 - i, srow["s"] - 200, 64) for i in range(40)]
        r = build_request(tok, srow, Human(crowd), 1, self.pool)
        self.assertEqual(r["donor"], "missing")
        self.assertIsNone(r["c"])
        self.assertFalse(r["eligible_c"])

    def test_crop_is_shared_and_labels_agree(self):
        songs = {s: (rh_song(40 + s, n=200), lh_song(50 + s, n=40)) for s in (1, 2)}
        pool = pool_of(songs)
        extra = 0
        for song, (tok, h) in songs.items():
            for srow in session_requests(tok, 0.0, **SMALL):
                r = build_request(tok, srow, Human(h), song, pool, **SMALL)
                self.assertEqual(set(check_request(r).values()), {0}, srow["k"])
                extra += r["extra_crop"] is not None
                a = r["a"]
                for arm in "bc":
                    x = r[arm]
                    if x is not None:
                        self.assertEqual(x["tokens"][x["label_start"]:], a["tokens"][a["label_start"]:])
                for n in r["own"]:
                    self.assertGreaterEqual(n.raw_on, r["base_ms"])
        self.assertGreater(extra, 0)


class FailurePathTest(unittest.TestCase):
    """#1773 review: a failing check must not leave the request 'ok' or the run exiting 0."""

    def test_injected_check_failure_marks_request_and_run(self):
        import json
        import tempfile
        from lh_context_build import contract_ok, finish, request_status
        songs = {s: (rh_song(10 + s), lh_song(20 + s)) for s in (1, 2)}
        pool = pool_of(songs)
        tok, h = songs[1]
        r = build_request(tok, session_requests(tok, 0.0)[5], Human(h), 1, pool)
        chk = dict(check_request(r))
        chk["L1f"] = 0
        self.assertEqual(request_status(chk), "ok")
        chk["L4"] = 1                                                    # the injection Astra used
        self.assertEqual(request_status(chk), "check_failed: L4")
        checks = {k: 0 for k in ("L1", "L1t", "L1f", "L2", "L3", "L4", "L5", "L6", "L7", "L8")}
        checks.update({k: checks[k] + v for k, v in chk.items()})
        counts = {"capacity_failure": 0, "contract_violation": 0}
        self.assertFalse(contract_ok(checks, counts))
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "sub", "report.json")
            self.assertEqual(finish({"contract_ok": contract_ok(checks, counts)}, path), 1)
            self.assertFalse(json.load(open(path))["contract_ok"])        # written before the nonzero exit
            checks["L4"] = 0
            self.assertEqual(finish({"contract_ok": contract_ok(checks, counts)}, path), 0)
            self.assertFalse(contract_ok({**checks, "L7": 1}, counts))     # the global L7 counts too


if __name__ == "__main__":
    unittest.main()
