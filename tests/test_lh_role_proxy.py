"""Fixtures for the causal left-hand audit (issue #1750). Pure Python."""
from __future__ import annotations

import os
import random
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from lh_role_proxy import (LEAD_MS, RECENT_MS, RULES, Note, assign_g3, assign_grid300, assign_top50,  # noqa: E402
                           offline_condition, online_condition, pcs, prefix_view, recent_condition, replay)


def ids(cond):
    return sorted(n.nid for n in cond)


class FixtureTest(unittest.TestCase):
    def test_same_pitch_class_in_both_roles_is_not_a_shared_note(self):
        notes = [Note(0, 48, 0, 4000), Note(1, 72, 1000, 1200)]          # C3 held, C5 target
        roles = assign_g3(notes)
        cond = online_condition(notes, roles, 1000)
        self.assertEqual(ids(cond), [0])
        self.assertEqual(pcs(cond), frozenset({0}))                       # same PC as the target: allowed
        self.assertNotIn(1, ids(cond))

    def test_simultaneous_duplicate_notes_get_distinct_roles(self):
        notes = [Note(0, 64, 0, 500), Note(1, 64, 0, 500), Note(2, 48, 0, 500)]
        roles = assign_top50(notes)
        targets = [n for n, r in roles.items() if r.role == "target"]
        conds = [n for n, r in roles.items() if r.role == "cond"]
        self.assertEqual(len(targets), 1)
        self.assertFalse(set(targets) & set(conds))

    def test_rolled_chord_role_known_only_after_the_window_closes(self):
        notes = [Note(0, 48, 0, 1000), Note(1, 52, 20, 1000), Note(2, 64, 40, 1000), Note(3, 67, 60, 1000)]
        roles = assign_top50(notes)                                       # group [0, 20, 40], then [60]
        self.assertEqual(roles[2].role, "target")
        self.assertEqual(ids(online_condition(notes, roles, 45)), [])     # window still open at 45 ms
        self.assertEqual(ids(online_condition(notes, roles, 55)), [0, 1])

    def test_hand_crossing_follows_the_proxy_rule_not_the_hand(self):
        notes = [Note(0, 60, 0, 500), Note(1, 50, 0, 500)]                # "left hand" above G3, melody below
        roles = assign_g3(notes)
        self.assertEqual((roles[0].role, roles[1].role), ("target", "cond"))

    def test_unaccompanied_melody_has_empty_conditions(self):
        notes = [Note(i, 72 + i, i * 200, i * 200 + 150) for i in range(5)]
        for rule in RULES.values():
            roles = rule(notes)
            self.assertTrue(all(online_condition(notes, roles, n.on - LEAD_MS) == [] for n in notes if roles[n.nid].role == "target"))

    def test_sustained_condition_note_stays_until_released(self):
        notes = [Note(0, 40, 0, 4000)] + [Note(i, 72, i * 1000, i * 1000 + 200) for i in (1, 2, 3)] + [Note(4, 74, 5000, 5200)]
        roles = assign_g3(notes)
        for t in (1000, 2000, 3000):
            self.assertEqual(ids(online_condition(notes, roles, t - LEAD_MS)), [0])
        self.assertEqual(ids(online_condition(notes, roles, 5000 - LEAD_MS)), [])

    def test_grid_window_role_known_at_window_end(self):
        notes = [Note(0, 48, 10, 2000), Note(1, 52, 20, 2000), Note(2, 55, 30, 2000), Note(3, 72, 100, 2000)]
        roles = assign_grid300(notes)                                     # 4 onsets in [0, 300): top 72 target
        self.assertEqual(roles[3].role, "target")
        self.assertEqual(ids(online_condition(notes, roles, 299)), [])
        self.assertEqual(ids(online_condition(notes, roles, 300)), [0, 1, 2])

    def test_offline_shortcut_uses_notes_after_the_decision_time(self):
        notes = [Note(0, 48, 900, 2000), Note(1, 72, 1000, 1200)]         # chord 100 ms before the target
        roles = assign_g3(notes)
        d = 1000 - LEAD_MS
        self.assertEqual(ids(offline_condition(notes, roles, 1000)), [0])
        self.assertEqual(ids(online_condition(notes, roles, d)), [])      # 900 > d: not observable yet
        self.assertTrue(all(n.on > d for n in offline_condition(notes, roles, 1000)))


class RecentConditionTest(unittest.TestCase):
    def test_short_comping_hit_released_before_d_is_still_recent(self):
        notes = [Note(0, 48, 0, 120), Note(1, 72, 1000, 1200)]           # staccato chord tone, gone by d
        roles = assign_g3(notes)
        d = 1000 - LEAD_MS
        self.assertEqual(ids(online_condition(notes, roles, d)), [])
        self.assertEqual(ids(recent_condition(notes, roles, d)), [0])

    def test_recent_window_edges(self):
        notes = [Note(0, 48, 0, 50), Note(1, 50, 100, 150)]
        roles = assign_g3(notes)
        self.assertEqual(ids(recent_condition(notes, roles, RECENT_MS)), [0, 1])          # on = d - w: inside
        self.assertEqual(ids(recent_condition(notes, roles, RECENT_MS + 1)), [1])
        self.assertEqual(ids(recent_condition(notes, roles, 99)), [0])                    # 100 not started yet

    def test_recent_waits_for_the_window_to_close(self):
        notes = [Note(0, 48, 0, 30), Note(1, 52, 10, 30), Note(2, 64, 20, 30)]
        roles = assign_top50(notes)
        self.assertEqual(ids(recent_condition(notes, roles, 49)), [])
        self.assertEqual(ids(recent_condition(notes, roles, 50)), [0, 1])


class BoundaryTest(unittest.TestCase):
    """Window ends and releases exactly at d, checked on the batch functions and the replay alike."""

    def both(self, notes, name, d):
        roles = RULES[name](notes)
        (snd, rec), = replay(notes, name, [d])
        self.assertEqual(sorted(snd), ids(online_condition(notes, roles, d)))
        self.assertEqual(sorted(rec), ids(recent_condition(notes, roles, d)))
        return ids(online_condition(notes, roles, d))

    def test_top50_window_end(self):
        notes = [Note(0, 48, 1000, 3000), Note(1, 64, 1010, 3000)]            # window [1000, 1050)
        self.assertEqual(self.both(notes, "top50", 1049.999), [])
        self.assertEqual(self.both(notes, "top50", 1050), [0])
        self.assertEqual(self.both(notes, "top50", 1050.001), [0])

    def test_top50_note_at_the_window_end_starts_a_new_group(self):
        notes = [Note(0, 48, 1000, 3000), Note(1, 64, 1050, 3000)]
        roles = assign_top50(notes)
        self.assertEqual((roles[0].role, roles[1].role), ("excluded", "target"))   # each alone, 48 below G3
        self.assertEqual(self.both(notes, "top50", 1200), [])

    def test_grid300_window_end(self):
        notes = [Note(0, 45, 610, 3000), Note(1, 48, 620, 3000), Note(2, 67, 630, 3000)]   # window [600, 900)
        self.assertEqual(self.both(notes, "grid300", 899.999), [])
        self.assertEqual(self.both(notes, "grid300", 900), [0, 1])
        self.assertEqual(self.both(notes, "grid300", 900.001), [0, 1])

    def test_release_exactly_at_d_is_gone_and_after_d_is_sounding(self):
        notes = [Note(0, 48, 0, 500), Note(1, 50, 0, 500.001)]
        self.assertEqual(self.both(notes, "g3", 500), [1])                      # off == d: released; off > d: sounding
        self.assertEqual(self.both(notes, "g3", 500.001), [])


class ReplayTest(unittest.TestCase):
    def test_replay_matches_batch_on_fixtures(self):
        notes = [Note(0, 48, 0, 1000), Note(1, 52, 20, 1000), Note(2, 64, 40, 1000), Note(3, 67, 60, 1000),
                 Note(4, 40, 300, 310), Note(5, 72, 300, 900), Note(6, 72, 300, 900), Note(7, 50, 310, 310)]
        qs = [-10, 0, 45, 50, 55, 109, 110, 299, 300, 305, 310, 350, 600, 999, 1000, 2000]
        for name, rule in RULES.items():
            roles = rule(notes)
            got = replay(notes, name, qs)
            for d, (snd, rec) in zip(qs, got):
                self.assertEqual(sorted(snd), ids(online_condition(notes, roles, d)), (name, d))
                self.assertEqual(sorted(rec), ids(recent_condition(notes, roles, d)), (name, d))


class CausalityTest(unittest.TestCase):
    def random_notes(self, rng, n=200):
        out = []
        for i in range(n):
            on = rng.uniform(0, 20_000)
            out.append(Note(i, rng.randint(30, 90), on, on + rng.uniform(20, 3000)))
        return out

    def test_online_condition_from_prefix_only_equals_full(self):
        rng = random.Random(0)
        for name, rule in RULES.items():
            for trial in range(20):
                notes = self.random_notes(rng)
                full_roles = rule(notes)
                d = rng.uniform(1000, 19_000)
                pre = prefix_view(notes, d)
                self.assertEqual(ids(online_condition(notes, full_roles, d)), ids(online_condition(pre, rule(pre), d)), (name, trial))

    def test_changing_the_future_does_not_change_the_past_condition(self):
        rng = random.Random(1)
        for name, rule in RULES.items():
            for trial in range(20):
                notes = self.random_notes(rng)
                d = rng.uniform(1000, 19_000)
                past = [n for n in notes if n.on <= d]
                future = [Note(1000 + i, rng.randint(30, 90), d + rng.uniform(0.1, 3000), d + 5000) for i in range(50)]
                a = online_condition(notes, rule(notes), d)
                b = online_condition(past + future, rule(past + future), d)
                self.assertEqual(ids(a), ids(b), (name, trial))

    def test_replay_matches_batch_on_random_notes(self):
        rng = random.Random(3)
        for name, rule in RULES.items():
            for trial in range(10):
                notes = self.random_notes(rng)
                roles = rule(notes)
                qs = [rng.uniform(-500, 21_000) for _ in range(60)] + [n.on for n in notes[:20]] + [n.off for n in notes[:20]]
                for d, (snd, rec) in zip(qs, replay(notes, name, qs)):
                    self.assertEqual(sorted(snd), ids(online_condition(notes, roles, d)), (name, trial, d))
                    self.assertEqual(sorted(rec), ids(recent_condition(notes, roles, d)), (name, trial, d))

    def test_recent_condition_from_prefix_only_equals_full(self):
        rng = random.Random(4)
        for name, rule in RULES.items():
            for trial in range(20):
                notes = self.random_notes(rng)
                d = rng.uniform(1000, 19_000)
                pre = prefix_view(notes, d)
                self.assertEqual(ids(recent_condition(notes, rule(notes), d)), ids(recent_condition(pre, rule(pre), d)), (name, trial))

    def test_no_note_is_both_condition_and_target(self):
        rng = random.Random(2)
        for rule in RULES.values():
            roles = rule(self.random_notes(rng))
            self.assertFalse({k for k, r in roles.items() if r.role == "cond"} & {k for k, r in roles.items() if r.role == "target"})


if __name__ == "__main__":
    unittest.main()
