"""Rules of the #1782 fragment retrieval on synthetic songs (no data files)."""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import bebop_fragment_retrieval as bf  # noqa: E402

try:
    import pretty_midi
except ImportError:  # pragma: no cover
    pretty_midi = None

META = {"pianist": "test", "source": "synthetic", "sha1": "0" * 40}


def song(lh, rh_start=0.0, rh_n=20, rh_step=0.2, rh_dur=0.5):
    """[(id, note)]: a right-hand run (overlapping notes so top_notes truncates them) plus left-hand chords."""
    notes = [pretty_midi.Note(velocity=80, pitch=72 + (i % 5), start=rh_start + i * rh_step, end=rh_start + i * rh_step + rh_dur)
             for i in range(rh_n)]
    notes += [pretty_midi.Note(velocity=60, pitch=p, start=s, end=e) for s, e, ps in lh for p in ps]
    out = [(f"i0_n{j}", n) for j, n in enumerate(notes)]
    return sorted(out, key=lambda x: (x[1].start, x[1].pitch, x[0]))


@unittest.skipIf(pretty_midi is None, "pretty_midi not installed")
class RetrievalRulesTest(unittest.TestCase):
    def test_signed_k(self):
        self.assertEqual(bf.signed_k(7, 7), 0)
        self.assertEqual(bf.signed_k(2, 7), 5)
        self.assertEqual(bf.signed_k(0, 7), -5)
        self.assertEqual(bf.signed_k(1, 7), -6)                 # tie at 6 goes down
        self.assertEqual(bf.signed_k(8, 7), -1)

    def test_change_candidate_gm7_to_c7(self):
        # root + 3 + 7 voicings below G3 (bass = root): Gm7 then C7 struck with a right-hand onset at 2.0 s
        notes = song([(0.0, 1.95, (43, 46, 53)), (2.0, 4.0, (36, 40, 46))])
        got = [c for c in bf.song_candidates(notes, META) if c["target"] == "change"]
        self.assertEqual(len(got), 1)
        c = got[0]
        self.assertEqual((c["first"]["label"], c["second"]["label"]), ((7, "m7"), (0, "7")))
        self.assertEqual(c["k"], 0)
        self.assertAlmostEqual(c["tc_err"], 2.0 - 1.875, places=4)

    def test_hold_rejected_when_contradicted(self):
        fmaj7 = (0.0, 4.0, (41, 45, 52))                         # F A E
        self.assertTrue(any(c["target"] == "hold" for c in bf.song_candidates(song([fmaj7]), META)))
        contra = song([(0.0, 2.0, (41, 45, 52)), (2.0, 4.0, (41, 44, 51))])     # Fmaj7 then Fm7 inside the window
        self.assertFalse(any(c["target"] == "hold" for c in bf.song_candidates(contra, META)))

    def test_future_cluster_is_proxy_invalid(self):
        # left hand struck 50 ms after the phrase start: v1 takes it as "sounding", nothing actually sounds at t0 + 30 ms
        notes = song([(1.05, 4.5, (41, 45, 52))], rh_start=1.0)
        c = [c for c in bf.song_candidates(notes, META) if c["target"] == "hold"][0]
        self.assertEqual(c["slice_source"], "sounding")
        chk = bf.verify_t0(notes, c)
        self.assertEqual(chk["status"], "proxy_invalid")
        self.assertEqual(chk["t0_sounding_pcs"], [])

    def test_extract_restores_original_lengths(self):
        notes = song([(0.0, 4.0, (41, 45, 52))])
        c = [c for c in bf.song_candidates(notes, META) if c["target"] == "hold"][0]
        ex = bf.extract(notes, c)
        first = ex["selected"][0]
        self.assertEqual(first["end"] - first["start"], 0.5)     # original length kept
        self.assertAlmostEqual(first["trunc_end"], 0.2)          # top_notes cut it at the next onset
        self.assertGreater(ex["line_overlaps"], 0)               # overlaps are recorded, not removed
        self.assertTrue(all(r["pitch"] < bf.ACC_HIGH for r in ex["excluded"]))

    def test_t0_with_many_decimals_keeps_first_note(self):
        # 213.92045454... rounds up to 213.9205; the first phrase note must still be selected
        t0 = 213.92045454545453
        notes = song([(t0 - 0.2, t0 + 4.0, (41, 45, 52))], rh_start=t0)
        c = [c for c in bf.song_candidates(notes, META) if c["target"] == "hold"][0]
        self.assertGreater(c["t0"], t0)                          # the display value is rounded up
        ex = bf.extract(notes, c)
        self.assertEqual(ex["selected"][0]["start"], 0.0)
        self.assertEqual(len(ex["selected"]), sum(1 for i in range(20) if i * 0.2 < bf.WIN_S))

    def test_lick_filter_rejects_slow_melody_runs(self):
        fmaj7 = (0.0, 9.0, (41, 45, 52))
        slow = song([fmaj7], rh_n=12, rh_step=0.5, rh_dur=0.6)                # median IOI 0.5 s: melody-like
        self.assertTrue(any(c["target"] == "hold" for c in bf.song_candidates(slow, META)))
        self.assertFalse(any(c["target"] == "hold" for c in bf.song_candidates(slow, META, require_lick=True)))
        fast = song([fmaj7], rh_n=20, rh_step=0.2)                           # median IOI 0.2 s, 20 notes: a lick
        c = [c for c in bf.song_candidates(fast, META, require_lick=True) if c["target"] == "hold"][0]
        self.assertEqual(c["run"]["notes"], 20)
        self.assertEqual(bf.extract(fast, c)["selected_in_run"], len(bf.extract(fast, c)["selected"]))


if __name__ == "__main__":
    unittest.main()
