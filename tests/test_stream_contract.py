"""Regression for the real-time streaming contract (issue #1741). Pure Python, virtual clock."""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from stream_contract import GrammarState, Note, Scheduler, TokenStream  # noqa: E402

PREFIX = [("piano", 60, 80), ("onset", 0), ("dur", 400)]


def note(p, onset, dur, v=80):
    return [("piano", p, v), ("onset", onset), ("dur", dur)]


class TokenStreamTest(unittest.TestCase):
    def setUp(self):
        self.s = TokenStream(PREFIX)

    def test_positions_start_at_prefix_end(self):
        self.assertEqual((self.s.committed_end, self.s.kv_valid_length, self.s.next_write_pos), (3, 3, 3))

    def test_sampled_token_is_not_in_kv_until_written(self):
        self.s.append_sampled(("piano", 62, 80))
        self.assertEqual(len(self.s.tokens), self.s.kv_valid_length + 1)
        self.s.mark_written(4)
        self.assertEqual((self.s.kv_valid_length, self.s.next_write_pos), (4, 4))

    def test_bad_token_is_rejected_before_it_enters_the_stream(self):
        with self.assertRaises(ValueError):
            self.s.append_sampled(("onset", 10))
        self.assertEqual(len(self.s.tokens), 3)

    def test_commit_only_complete_notes_before_horizon(self):
        for t in note(62, 500, 300) + note(64, 900, 300) + [("piano", 65, 80), ("onset", 1200)]:
            self.s.append_sampled(t)
        notes = self.s.commit_until(10_000)
        self.assertEqual([(n.pitch, n.on_ms, n.off_ms) for n in notes], [(62, 500, 800), (64, 900, 1200)])
        self.assertEqual(self.s.committed_end, 9)          # the incomplete note (no dur yet) stays provisional

    def test_commit_stops_at_horizon(self):
        for t in note(62, 500, 300) + note(64, 900, 300):
            self.s.append_sampled(t)
        self.assertEqual([n.pitch for n in self.s.commit_until(900)], [62])
        self.assertEqual(self.s.committed_end, 6)

    def test_time_token_committed_only_when_horizon_reaches_next_segment(self):
        for t in ["<T>"] + note(62, 100, 200):
            self.s.append_sampled(t)
        self.assertEqual(self.s.commit_until(4_000), [])
        self.assertEqual(self.s.committed_end, 3)
        notes = self.s.commit_until(5_200)
        self.assertEqual([(n.on_ms, n.off_ms) for n in notes], [(5_100, 5_300)])

    def test_rollback_drops_provisional_and_restores_grammar_and_positions(self):
        for t in note(62, 500, 300):
            self.s.append_sampled(t)
        self.s.mark_written(6)
        self.s.commit_until(10_000)                        # committed_end = 6
        for t in note(64, 900, 300) + [("piano", 65, 80)]:
            self.s.append_sampled(t)
        self.s.mark_written(10)
        before = list(self.s.tokens[:6])
        self.assertEqual(self.s.rollback_to_committed(), 4)
        self.assertEqual(self.s.tokens, before)            # committed tokens unchanged
        self.assertEqual((self.s.kv_valid_length, self.s.next_write_pos), (6, 6))
        self.assertEqual(self.s.grammar_state(), GrammarState("note", 500, 0))
        self.s.append_sampled(("piano", 67, 70))           # generation restarts from the committed state

    def test_full_reset_requires_prefill_of_whole_committed_context(self):
        for t in note(62, 500, 300):
            self.s.append_sampled(t)
        self.s.mark_written(6)
        self.s.commit_until(10_000)
        self.s.append_sampled(("piano", 64, 80))
        self.s.request_full_reset()
        self.assertTrue(self.s.needs_full_prefill)
        self.assertEqual((self.s.kv_valid_length, self.s.next_write_pos, len(self.s.tokens)), (0, 0, 6))
        self.s.mark_written(6)
        self.assertFalse(self.s.needs_full_prefill)


class SchedulerTest(unittest.TestCase):
    def setUp(self):
        self.sch = Scheduler()

    def test_normal_on_off(self):
        self.sch.schedule([Note(0, 60, 80, 1_000, 1_500)])
        self.sch.tick(1_000)
        self.sch.tick(1_500)
        self.assertEqual(self.sch.sent, [(1_000, "on", 0, 60), (1_500, "off", 0, 60)])
        self.assertEqual(self.sch.sounding, {})

    def test_stop_before_a_scheduled_note_on(self):
        self.sch.schedule([Note(0, 60, 80, 10_000, 11_000)])
        self.sch.stop(9_000)
        self.sch.tick(10_000)
        self.sch.tick(11_000)
        self.assertEqual(self.sch.sent, [])                 # the future note never sounds

    def test_stop_right_after_note_on(self):
        self.sch.schedule([Note(0, 60, 80, 10_000, 11_000)])
        self.sch.tick(10_000)
        self.sch.stop(10_100)
        self.sch.tick(11_000)
        self.assertEqual(self.sch.sent, [(10_000, "on", 0, 60), (10_100, "off", 0, 60)])   # one off, not two

    def test_late_callback_of_old_generation_is_ignored_even_after_restart(self):
        self.sch.schedule([Note(0, 60, 80, 10_000, 11_000)])
        stale = list(self.sch.pending)                       # timers already armed by the OS
        self.sch.stop(9_000)
        self.sch.restart()
        self.sch.schedule([Note(1, 64, 80, 12_000, 12_500)])
        for e in stale:
            self.assertFalse(self.sch.deliver(e, e.time_ms))
        self.sch.tick(12_000)
        self.sch.tick(12_500)
        self.assertEqual(self.sch.sent, [(12_000, "on", 1, 64), (12_500, "off", 1, 64)])

    def test_stop_clears_pending_events(self):
        """Old-generation events are also rejected on delivery; this checks the first guard on its own."""
        self.sch.schedule([Note(0, 60, 80, 10_000, 11_000)])
        self.sch.stop(9_000)
        self.assertEqual(self.sch.pending, [])

    def test_no_new_events_while_stopped(self):
        self.sch.stop(0)
        self.assertEqual(self.sch.schedule([Note(0, 60, 80, 100, 200)]), 0)
        self.sch.tick(1_000)
        self.assertEqual(self.sch.sent, [])

    def test_rollback_does_not_touch_committed_note_offs(self):
        s = TokenStream(PREFIX)
        for t in note(62, 500, 1_000):
            s.append_sampled(t)
        self.sch.schedule(s.commit_until(10_000))
        self.sch.tick(500)
        for t in note(64, 900, 300):
            s.append_sampled(t)
        s.rollback_to_committed()                            # chord input arrives: provisional note dropped
        self.sch.tick(1_500)
        self.assertEqual(self.sch.sent, [(500, "on", 0, 62), (1_500, "off", 0, 62)])

    def test_same_pitch_overlap_offs_are_owned_per_note(self):
        self.sch.schedule([Note(0, 60, 80, 0, 1_000), Note(1, 60, 80, 500, 1_500)])
        for t in (0, 500, 1_000, 1_500):
            self.sch.tick(t)
        self.assertEqual([x[1:3] for x in self.sch.sent], [("on", 0), ("on", 1), ("off", 0), ("off", 1)])


if __name__ == "__main__":
    unittest.main()
