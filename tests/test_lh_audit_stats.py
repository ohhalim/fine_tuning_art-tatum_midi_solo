"""Checks for the causal left-hand audit statistics (issue #1750). Synthetic MIDI only."""
from __future__ import annotations

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import pretty_midi  # noqa: E402

from lh_audit_stats import (CONDS, LEADS, aggregate, audit_song, check_counts, future_release_drops,  # noqa: E402
                            leak_counts, load_notes, sha1_file, verdicts)
from lh_role_proxy import LEAD_MS, Note, assign_g3  # noqa: E402


def write_midi(path, notes):
    pm = pretty_midi.PrettyMIDI(initial_tempo=120)
    ins = pretty_midi.Instrument(0)
    for p, on, off in notes:
        ins.notes.append(pretty_midi.Note(velocity=70, pitch=p, start=on, end=off))
    pm.instruments.append(ins)
    pm.write(path)


def song(seed_shift=0.0):
    """Four bars: held and staccato left-hand chords, a rolled chord, a duplicated melody note, a 20 s pedal tone."""
    out = [(36, 0.0 + seed_shift, 20.0 + seed_shift)]                                   # very long low note
    for bar in range(8):
        b = bar * 2.0 + seed_shift
        out += [(48, b, b + 1.5), (52, b + 0.02, b + 1.5), (55, b + 0.04, b + 1.5)]       # rolled chord, held
        out += [(50, b + 1.0, b + 1.1), (53, b + 1.0, b + 1.1)]                           # staccato comp hit
        for k in range(8):
            out.append((67 + (k * 5) % 12, b + k * 0.25, b + k * 0.25 + 0.2))           # melody
        out.append((72, b + 0.5, b + 0.7))                                                # duplicate of a melody onset
        out.append((72, b + 0.5, b + 0.7))
    return out


class LeakCountTest(unittest.TestCase):
    def test_each_leak_kind_is_counted_on_its_own(self):
        notes = [Note(0, 48, 0, 2000), Note(1, 50, 900, 2000), Note(2, 72, 1000, 1200), Note(3, 72, 1000, 1200)]
        roles = assign_g3(notes)
        roles[1] = roles[1].__class__("cond", 1500)                                         # role known only at 1500
        t, d = notes[2], 1000 - LEAD_MS
        cond = [notes[0], notes[1], notes[3]]                                               # 3 is a target and a duplicate
        k1, k2, k3, dup = leak_counts(cond, d, roles, {2, 3}, t)
        self.assertEqual((k1, k2, k3, dup), (1, 2, 2, 1))

    def test_future_release_drop(self):
        notes = [Note(0, 48, 0, 800), Note(1, 50, 0, 1100), Note(2, 72, 1000, 1200)]
        roles = assign_g3(notes)
        self.assertEqual(future_release_drops(notes, roles, 1000 - LEAD_MS, 1000), 1)       # note 0 released in (d, t]


class AuditSongTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.a, cls.b = os.path.join(cls.tmp.name, "a.mid"), os.path.join(cls.tmp.name, "b.mid")
        write_midi(cls.a, song())
        write_midi(cls.b, song(0.13))
        cls.res = audit_song({"idx": 0, "stratum": "main", "path": cls.a, "sha1": sha1_file(cls.a), "donor_path": cls.b})

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_online_conditions_have_no_leak_and_match_replay_and_prefix(self):
        r = self.res
        self.assertIsNone(r["error"])
        for rule, x in r["rules"].items():
            for ln in LEADS:
                for c in CONDS:
                    st = x["online"][ln][c]
                    self.assertEqual((st["K1"], st["K2"], st["K3"], st["replay_mismatch"]), (0, 0, 0, 0), (rule, ln, c))
            inv = x["invariance"]
            self.assertGreater(inv["samples"], 0)
            self.assertEqual((inv["prefix_mismatch"], inv["prefix_finite_off_exposed"], inv["suffix_mismatch"]), (0, 0, 0), rule)

    def test_offline_shortcut_is_seen_using_future_information(self):
        off = self.res["rules"]["g3"]["offline"]["lead1beat"]
        self.assertGreater(off["K2"], 0)                                                    # staccato hit after d
        self.assertGreater(off["K4_future_release_drops"], 0)                               # staccato hit released before t
        self.assertGreater(self.res["rules"]["top50"]["offline"]["lead0"]["K3"], 0)         # window still open at t

    def test_recent_sees_the_staccato_hit_that_sounding_misses(self):
        g3 = self.res["rules"]["g3"]["online"]["lead1beat"]
        self.assertGreaterEqual(g3["recent"]["nonempty"], g3["sounding"]["nonempty"])

    def test_rows_are_one_per_target_and_rule(self):
        self.assertEqual(len(self.res["rows"]), sum(x["targets"] for x in self.res["rules"].values()))

    def test_report_tables_are_json_serializable(self):
        import copy
        import json
        r = copy.deepcopy(self.res)
        r.pop("rows")
        agg = aggregate([r], "main")
        agg["note_count_mismatch"] = 0
        agg["verdicts"] = verdicts(agg)
        json.dumps(agg)                                                                     # first run died here (#1750)
        notes, _ = load_notes(self.a)
        self.assertIs(type(notes[0].on), float)

    def test_sha1_mismatch_is_recorded_not_read(self):
        r = audit_song({"idx": 1, "stratum": "main", "path": self.a, "sha1": "0" * 40, "donor_path": self.b})
        self.assertEqual(r["error"], "sha1 mismatch")
        self.assertNotIn("rules", r)


class PositiveControlTest(unittest.TestCase):
    """A rule that claims its roles at the group start (peeking 50 ms ahead) must be caught."""

    def test_peeking_rule_is_caught_by_replay_and_by_a_different_future(self):
        import lh_audit_stats
        from lh_role_proxy import GROUP_MS, Role, assign_top50

        def peeking(notes):
            return {k: Role(r.role, r.available_at - GROUP_MS, r.reason) for k, r in assign_top50(notes).items()}
        with tempfile.TemporaryDirectory() as tmp:
            a, b = os.path.join(tmp, "a.mid"), os.path.join(tmp, "b.mid")
            write_midi(a, [(48, 0.0, 2.0), (64, 0.03, 0.5), (70, 1.0, 1.2)])          # 64 arrives inside 48's window
            write_midi(b, [(80, 0.0, 0.3)])                                            # a higher note right after d
            saved = lh_audit_stats.RULES
            lh_audit_stats.RULES = {"top50": peeking}
            try:
                r = audit_song({"idx": 0, "stratum": "main", "path": a, "sha1": None, "donor_path": b})
            finally:
                lh_audit_stats.RULES = saved
        x = r["rules"]["top50"]
        self.assertGreater(x["online"]["lead0"]["sounding"]["replay_mismatch"], 0)
        self.assertGreater(x["invariance"]["suffix_mismatch"], 0)


class VerdictTest(unittest.TestCase):
    def agg(self, nonempty, targets=25000, leak=0, songs_ok=10):
        st = {"K1": 0, "K2": leak, "K3": 0, "replay_mismatch": 0, "nonempty": nonempty, "targets": targets,
              "nonempty_ratio": round(nonempty / targets, 4) if targets else None}
        return {"songs": 10, "songs_ok": songs_ok, "note_count_mismatch": 0,
                "rules": {"g3": {"online": {ln: {c: dict(st) for c in CONDS} for ln in LEADS},
                                 "invariance": {"samples": 5, "prefix_mismatch": 0, "prefix_finite_off_exposed": 0, "suffix_mismatch": 0}}}}

    def verdict(self, *a, **k):
        return verdicts(self.agg(*a, **k))["g3"]["conditions"]["sounding"]["verdict"]

    def test_floor_is_judged_on_raw_counts_not_the_rounded_ratio(self):
        self.assertEqual(self.agg(2499)["rules"]["g3"]["online"]["lead1beat"]["sounding"]["nonempty_ratio"], 0.1)  # .09996 shown as .1
        self.assertEqual(self.verdict(2499), "constructible but almost always empty")       # just below
        self.assertEqual(self.verdict(2500), "causal pairs constructible")                  # exactly the floor
        self.assertEqual(self.verdict(2501), "causal pairs constructible")                  # just above
        self.assertEqual(self.verdict(0, targets=0), "constructible but almost always empty")

    def test_construction_failures(self):
        self.assertEqual(self.verdict(20000, leak=1), "construction failed")
        self.assertEqual(self.verdict(20000, songs_ok=9), "construction failed")

    def test_exploratory_counts_carry_no_verdict(self):
        cc = check_counts(self.agg(2500))["g3"]
        self.assertNotIn("verdict", cc)
        self.assertNotIn("construction_ok", cc)


if __name__ == "__main__":
    unittest.main()
