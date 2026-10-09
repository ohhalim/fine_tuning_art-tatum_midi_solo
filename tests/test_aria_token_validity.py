"""Generated-token validity checker (scripts/aria_token_validity.py)."""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))

from scripts.aria_token_validity import check, segment_state


def n(p, onset, dur=100):
    return [("piano", p, 80), ("onset", onset), ("dur", dur)]


PROMPT = [("prefix", "instrument", "piano"), "<S>"] + n(60, 4800) + ["<T>"] + n(62, 200)


class ValidityTest(unittest.TestCase):
    def test_prompt_with_segment_reset_does_not_fake_a_reversal(self) -> None:
        self.assertEqual(segment_state(PROMPT), (200, "note"))
        r = check(PROMPT, n(64, 300) + n(65, 300))          # same onset allowed
        self.assertTrue(r["valid"])
        self.assertEqual(r["complete_notes"], 2)

    def test_reversal_inside_a_segment(self) -> None:
        r = check(PROMPT, n(64, 300) + n(65, 250))
        self.assertEqual((r["valid"], r["first_error"]["type"], r["first_error"]["index"]), (False, "onset_reversal", 4))
        self.assertEqual(r["notes_before_first_error"], 1)

    def test_t_resets_and_d_between_notes_are_valid(self) -> None:
        r = check(PROMPT, n(64, 4900) + ["<T>"] + n(65, 10) + ["<D>"] + n(67, 20))
        self.assertTrue(r["valid"])

    def test_d_inside_a_note_is_token_order(self) -> None:
        r = check(PROMPT, [("piano", 64, 80), "<D>", ("onset", 300), ("dur", 100)])
        self.assertEqual(r["first_error"]["type"], "token_order")

    def test_other_instrument_and_cascade_counting(self) -> None:
        r = check(PROMPT, [("organ", 64, 80), ("onset", 300), ("dur", 100)] + n(65, 400))
        self.assertEqual(r["first_error"]["type"], "non_piano_instrument")
        self.assertEqual(r["cascade_errors"], 2)            # the orphan onset and dur only
        self.assertEqual(r["complete_notes"], 1)

    def test_eos_stops_checking(self) -> None:
        r = check(PROMPT, n(64, 300) + ["<E>", ("dur", 5)])
        self.assertTrue(r["valid"] and r["ended_by_eos"])

    def test_cap_cut_tail_is_not_an_internal_error(self) -> None:
        r = check(PROMPT, n(64, 300) + [("piano", 65, 80), ("onset", 400)])
        self.assertTrue(r["valid"])
        self.assertEqual(r["tail"], "incomplete_tail")

    def test_prompt_ending_inside_a_note(self) -> None:
        r = check(PROMPT + [("piano", 70, 80)], [("onset", 250), ("dur", 50)])
        self.assertTrue(r["valid"])


if __name__ == "__main__":
    unittest.main()
