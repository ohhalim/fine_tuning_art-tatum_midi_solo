"""The block snapshot contract against the real commit walk and the producer's horizons (issue #1754). Pure Python."""
from __future__ import annotations

import os
import random
import re
import sys
import unittest

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
from block_contract import (BEAT, BLOCK, PREBUFFER_LEAD_MS, WORST_LEAD_MS, block_of, block_start, horizon,  # noqa: E402
                            must_regenerate, snapshot)
from lh_role_proxy import INF, Note, assign_g3, online_condition, recent_condition  # noqa: E402
from stream_contract import SEGMENT_MS, TokenStream  # noqa: E402


def stream_from(onsets):
    """Tokens for one note per onset (ms on a 10 ms grid, sorted), with <T> at each 5 s segment."""
    toks, seg = [], 0
    for on in onsets:
        while on >= (seg + 1) * SEGMENT_MS:
            toks.append("<T>")
            seg += 1
        toks += [("piano", 60 + len(toks) % 24, 80), ("onset", on - seg * SEGMENT_MS), ("dur", 100)]
    return toks


class ConstantsTest(unittest.TestCase):
    def test_constants_match_the_runtime_sources(self):
        with open(os.path.join(HERE, "..", "scripts", "aria_mlx_stream_loop.py")) as f:
            loop = f.read()
        with open(os.path.join(HERE, "..", "scripts", "rt_producer.py")) as f:
            prod = f.read()
        self.assertIn(f"BEAT, BLOCK = {BEAT}, {BLOCK}", loop)
        self.assertRegex(prod, rf"LEAD_S, CAPACITY = {PREBUFFER_LEAD_MS / 1000}, ")
        self.assertIn("h0 = t0 + BLOCK + BEAT", prod)
        self.assertIn("now = t0 + k * BLOCK", prod)
        self.assertIn("h = now + BLOCK + BEAT", prod)


class CommitWalkTest(unittest.TestCase):
    def run_producer_horizons(self, onsets, t0, blocks):
        """Commit with the producer's horizon sequence and return {block: onsets committed in it}."""
        s = TokenStream([])
        for t in stream_from(onsets):
            s.append_sampled(t)
        out = {0: [n.on_ms for n in s.commit_until(t0 + BLOCK + BEAT)]}
        for k in range(1, blocks):
            out[k] = [n.on_ms for n in s.commit_until(t0 + k * BLOCK + BLOCK + BEAT)]
        return out

    def test_block_of_matches_commit_until_on_the_producer_horizons(self):
        rng = random.Random(0)
        for t0 in (0, 230, 470, 700, 1210):
            onsets = sorted({t0 + 10 * rng.randint(0, 3000) for _ in range(400)})
            got = self.run_producer_horizons(onsets, t0, 40)
            for k, ons in got.items():
                for on in ons:
                    self.assertEqual(block_of(on, t0), k, (t0, on))
            committed = [on for ons in got.values() for on in ons]
            self.assertEqual(committed, [on for on in onsets if block_of(on, t0) < 40])

    def test_onset_on_a_horizon_belongs_to_the_next_block(self):
        t0 = 0
        for k in range(0, 5):
            h = horizon(k, t0)
            self.assertEqual(block_of(h, t0), k + 1)
            self.assertEqual(block_of(h - 1e-6, t0), k)

    def test_lead_ranges(self):
        t0 = 350.0
        rng = random.Random(1)
        for _ in range(2000):
            t = t0 + rng.uniform(0, 60_000)
            k, d = block_of(t, t0), snapshot(t, t0)
            if k == 0:
                self.assertEqual(d, t0 - PREBUFFER_LEAD_MS)
                self.assertTrue(PREBUFFER_LEAD_MS <= t - d < PREBUFFER_LEAD_MS + WORST_LEAD_MS)
            else:
                self.assertTrue(BEAT <= t - d < WORST_LEAD_MS, (t, d))

    def test_primer_is_not_generated(self):
        self.assertIsNone(block_of(99.9, 100))
        self.assertIsNone(snapshot(99.9, 100))
        self.assertEqual(block_of(100, 100), 0)


class RegenerationRuleTest(unittest.TestCase):
    """The rollback rule compares the whole condition representation, not input events (#1755 review M1)."""

    def test_recent_window_expiry_needs_regeneration_without_any_new_event(self):
        notes = [Note(0, 40, 500, 600)]                       # g3 condition note, released long before block 2
        roles = assign_g3(notes)
        s1, s2 = block_start(1, 0), block_start(2, 0)         # 937.5, 1875
        events = [t for n in notes for t in (n.on, n.off, roles[n.nid].available_at) if s1 < t <= s2]
        self.assertEqual(events, [])                          # an event-only rule sees no change here

        def recent_ids(s):
            return frozenset(n.nid for n in recent_condition(notes, roles, s))
        self.assertEqual((recent_ids(s1), recent_ids(s2)), (frozenset({0}), frozenset()))
        self.assertTrue(must_regenerate(2, 0, recent_ids))

    def test_an_age_feature_changes_the_representation_of_the_same_held_note(self):
        notes = [Note(0, 40, 500, INF)]                       # still held: the sounding set stays {0}
        roles = assign_g3(notes)

        def ids(s):
            return frozenset(n.nid for n in online_condition(notes, roles, s))

        def ids_and_age(s):
            return frozenset((n.nid, int((s - n.on) // BEAT)) for n in online_condition(notes, roles, s))
        self.assertFalse(must_regenerate(2, 0, ids))
        self.assertTrue(must_regenerate(2, 0, ids_and_age))

    def test_block_zero_has_nothing_to_drop(self):
        self.assertFalse(must_regenerate(0, 0, lambda s: s))


class ColdBlockTest(unittest.TestCase):
    """Block 0 observes no later than t0 - 1000 ms. For the audited phases that is before 0 (#1755 review Q2)."""

    def test_cold_conditions_are_empty_for_the_audited_phases(self):
        rng = random.Random(2)
        notes = [Note(i, rng.randint(30, 90), on, on + rng.choice([50.0, 400.0, INF]))
                 for i, on in enumerate(sorted(rng.uniform(0, 3000) for _ in range(300)))]
        roles = assign_g3(notes)
        for t0 in (0.0, BLOCK / 4, BLOCK / 2, 3 * BLOCK / 4):
            self.assertLess(block_start(0, t0), 0)
            cold = [n for n in notes if block_of(n.on, t0) == 0]
            self.assertTrue(cold)
            for n in cold:
                d = snapshot(n.on, t0)
                self.assertEqual((online_condition(notes, roles, d), recent_condition(notes, roles, d)), ([], []))


if __name__ == "__main__":
    unittest.main()
