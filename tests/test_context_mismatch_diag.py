"""Boundary and state rules of the context-mismatch diagnosis (docs/experiments/CONTEXT_MISMATCH_DIAG.md)."""
from __future__ import annotations

import unittest

import pretty_midi

from scripts.context_mismatch_diag import (
    build_case,
    estimate_chord,
    extract_target,
    grammar_errors,
    is_vel,
    paired,
    sanitize,
    states,
    tail_exact,
    unrelated_chord,
    verdict,
)
from scripts.generate import encode_notes_simple


def song(n_notes=200, legato_every=0):
    notes = []
    for k in range(n_notes):
        s = k * 0.15
        e = s + (0.3 if legato_every and k % legato_every == 0 else 0.1)    # some notes overlap the next
        notes.append(pretty_midi.Note(velocity=60 + (k % 5) * 8, pitch=60 + (k * 7) % 24, start=s, end=e))
        if k % 4 == 0:
            notes.append(pretty_midi.Note(velocity=50, pitch=48 + (k % 12), start=s, end=s + 0.5))
    return encode_notes_simple(sorted(notes, key=lambda n: (n.start, n.pitch)))


class TargetTest(unittest.TestCase):
    def test_target_starts_with_velocity_on_a_silent_boundary_without_orphans(self) -> None:
        toks = song(legato_every=3)
        st = states(toks)
        found = 0
        for i in range(260, len(toks)):
            got = extract_target(toks, i, st=st)
            if got is None:
                continue
            target, _ = got
            found += 1
            self.assertFalse(st[i][0])                         # nothing sounding at the boundary
            self.assertTrue(is_vel(target[0]))                 # explicit velocity first
            self.assertEqual(grammar_errors(target), {"duplicate": 0, "orphan": 0})
        self.assertGreater(found, 0)

    def test_sounding_boundaries_are_refused(self) -> None:
        toks = song(legato_every=1)
        st = states(toks)
        for i in range(260, 400):
            if st[i][0]:
                self.assertIsNone(extract_target(toks, i, st=st))

    def test_sanitize_drops_orphans_and_closes_open_notes(self) -> None:
        self.assertEqual(sanitize([60 + 128, 356 + 15, 62, 256, 62 + 128, 64]), [371, 62, 256, 190, 64, 192])

    def test_tail_exact_matches_the_requested_length_or_gives_none(self) -> None:
        toks = song()
        t = tail_exact(toks, 30)
        self.assertEqual(len(t), 30)
        self.assertEqual(grammar_errors(t), {"duplicate": 0, "orphan": 0})


class CaseTest(unittest.TestCase):
    def test_every_primer_plus_target_is_grammatical_and_lengths_match(self) -> None:
        toks, other = song(legato_every=3), song(n_notes=180)
        st = states(toks)
        i = next(i for i in range(300, len(toks)) if extract_target(toks, i, st=st))
        target, _ = extract_target(toks, i, st=st)
        case = build_case(toks, i, target, other, 300)
        for name, p in case["primers"].items():
            self.assertEqual(grammar_errors(p + target), {"duplicate": 0, "orphan": 0}, name)
        if "C" in case["primers"] and "Ns" in case["primers"]:
            self.assertEqual(len(case["primers"]["Ns"]), len(case["primers"]["C"]))
        if "NX" in case["primers"]:
            self.assertEqual(len(case["primers"]["NX"]), len(case["primers"]["NC"]))

    def test_chord_estimate_and_unrelated_chord(self) -> None:
        notes = [pretty_midi.Note(velocity=80, pitch=p, start=0.1 * k, end=0.1 * k + 0.1)
                 for k, p in enumerate([60, 64, 67, 71, 72, 76])]
        target = encode_notes_simple(notes)
        self.assertEqual(estimate_chord(target), "Cmaj7")
        self.assertNotEqual(unrelated_chord("Cmaj7"), "Cmaj7")
        chromatic = encode_notes_simple([pretty_midi.Note(velocity=80, pitch=60 + k, start=0.1 * k, end=0.1 * k + 0.1)
                                         for k in range(12)])
        self.assertIsNone(estimate_chord(chromatic))


class VerdictTest(unittest.TestCase):
    def summ(self, mn, nsn, nxn, vr):
        f = lambda lo: {"ci95": [lo, lo + 0.1], "mean": lo + 0.05}
        return {"nll": {"M-N": f(mn), "Ns-N": f(nsn), "NX-N": f(nxn)}, "rollout_vr": {"N-Ns": f(vr)}}

    def test_mismatch_needs_all_three(self) -> None:
        self.assertEqual(verdict(self.summ(0.1, 0.2, -0.5, 0.01))["label"], "inference_mismatch_supported")
        self.assertEqual(verdict(self.summ(0.1, 0.2, -0.5, -0.01))["label"], "undecided")

    def test_no_prefix_effect_is_not_called_training_insufficiency(self) -> None:
        v = verdict(self.summ(-0.05, 0.2, 0.2, 0.01))
        self.assertEqual(v["label"], "prefix_effect_unconfirmed")
        self.assertIn("not judged", v["note"])

    def test_paired_means_per_song_first(self) -> None:
        cases = [{"song": "a", "nll": {"M": 2.0, "N": 1.0}}, {"song": "a", "nll": {"M": 4.0, "N": 1.0}},
                 {"song": "b", "nll": {"M": 1.0, "N": 1.0}}, {"song": "b", "nll": {"N": 1.0}}]
        r = paired(cases, "M", "N")
        self.assertEqual((r["songs"], r["positions"], r["mean"]), (2, 3, 1.0))


if __name__ == "__main__":
    unittest.main()
