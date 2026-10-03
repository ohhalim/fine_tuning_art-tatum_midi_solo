"""Runtime candidate ranker equals the offline one (inference/control/candidate_rank.py, #1637)."""
from __future__ import annotations

import random
import unittest

import pretty_midi

from inference.control.candidate_rank import pick, score
from scripts.candidate_select_eval import rank_pick
from scripts.candidate_select_eval import score as offline_score
from scripts.generate import encode_notes_simple


class EqualityTest(unittest.TestCase):
    def test_same_scores_and_pick_as_the_offline_check(self) -> None:
        rng = random.Random(0)
        for trial in range(50):
            pcs = {(rng.randrange(12) + i) % 12 for i in (0, 4, 7, 10)}
            solos = []
            for _ in range(3):
                t, solo = 0.0, []
                for _ in range(rng.randrange(0, 8)):
                    d = rng.choice([0.06, 0.12, 0.2])
                    solo.append((rng.randrange(55, 84), round(t, 2), round(t + d, 2)))
                    t += d + rng.choice([0.0, 0.05])
                solos.append([x for x in solo if x[1] < 0.9375])
            for so in solos:
                self.assertEqual(score(so, pcs), offline_score(so, pcs))
            valid = [rng.random() > 0.2 for _ in solos]
            expected = rank_pick([{"solo": so, "valid": v} for so, v in zip(solos, valid)], pcs)
            toks = [encode_notes_simple([pretty_midi.Note(velocity=80, pitch=p, start=s, end=e) for p, s, e in so])
                    for so in solos]
            got, _ = pick(toks, pcs, block_s=0.9375, valid=lambda t, m=dict(zip(map(tuple, toks), valid)): m[tuple(t)])
            self.assertEqual(got, expected, trial)


class RepeatWeightTest(unittest.TestCase):
    def test_repeated_notes_lower_the_score_only_with_a_weight(self) -> None:
        pcs = {0, 4, 7}
        rep = [(60, 0.0, 0.2), (60, 0.2, 0.4), (60, 0.4, 0.6), (64, 0.6, 0.8)]
        mov = [(60, 0.0, 0.2), (64, 0.2, 0.4), (67, 0.4, 0.6), (64, 0.6, 0.8)]
        self.assertEqual(score(rep, pcs), score(mov, pcs))                     # both all chord tones
        self.assertLess(score(rep, pcs, 0.5), score(mov, pcs, 0.5))
        self.assertAlmostEqual(score(rep, pcs) - score(rep, pcs, 0.5), 0.5 * 2 / 3)


class RuntimeFlagTest(unittest.TestCase):
    def test_candidates_need_the_sub_block_path_and_no_pattern_cache(self) -> None:
        from scripts import run_continuous_jazz
        with self.assertRaises(SystemExit):
            run_continuous_jazz.main(["--fallback-only", "--candidates", "3", "--pattern-cache",
                                      "--chord-primer", "--chord-blocks-per-bar", "2"])
        with self.assertRaises(SystemExit):
            run_continuous_jazz.main(["--fallback-only", "--candidates", "9"])


if __name__ == "__main__":
    unittest.main()
