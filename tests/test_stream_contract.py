"""Regression for the real-time streaming contract (issue #1741). Pure Python, virtual clock."""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import random  # noqa: E402

from stream_contract import MIN_DUR_MS, GrammarState, Note, Scheduler, TokenStream, playable  # noqa: E402

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

    def test_real_prompt_header_is_accepted(self):
        s = TokenStream([("prefix", "instrument", "piano"), "<S>"] + PREFIX)
        self.assertEqual(s.grammar_state(), GrammarState("note", 0, 0))
        with self.assertRaises(ValueError):
            s.append_sampled(("onset", 10))                # the header does not relax the note grammar

    def test_tokens_after_eos_are_rejected(self):
        s = TokenStream([])
        s.append_sampled("<E>")
        with self.assertRaises(ValueError):
            s.append_sampled(("piano", 60, 80))
        self.assertEqual(s.commit_until(1_000), [])
        self.assertTrue(s.committed_ended)

    def test_provisional_eos_is_undone_by_rollback(self):
        for t in note(62, 500, 300):
            self.s.append_sampled(t)
        self.s.commit_until(10_000)
        self.s.append_sampled("<E>")                       # provisional end
        self.s.rollback_to_committed()
        self.assertFalse(self.s.committed_ended)
        self.s.append_sampled(("piano", 64, 80))           # generation can continue

    def test_committed_eos_keeps_the_stream_ended(self):
        self.s.append_sampled("<E>")
        self.s.commit_until(10_000)
        self.s.rollback_to_committed()
        self.assertTrue(self.s.committed_ended)
        with self.assertRaises(ValueError):
            self.s.append_sampled(("piano", 64, 80))


class SchedulerTest(unittest.TestCase):
    def setUp(self):
        self.sch = Scheduler()

    def test_normal_on_off(self):
        self.sch.schedule([Note(0, 60, 80, 1_000, 1_500)])
        self.sch.tick(1_000)
        self.sch.tick(1_500)
        self.assertEqual(self.sch.sent, [(1_000, "on", 0, 60), (1_500, "off", 0, 60)])
        self.assertEqual(self.sch.sounding, {})

    def test_clock_cannot_go_back(self):
        self.sch.tick(1_000)
        with self.assertRaises(ValueError):
            self.sch.tick(500)

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

    def test_callback_then_tick_sends_once(self):
        self.sch.schedule([Note(0, 60, 80, 100, 200)])
        on = self.sch.pending[0]
        self.assertTrue(self.sch.deliver(on, 100))
        self.sch.tick(100)
        self.sch.tick(200)
        self.assertEqual(self.sch.sent, [(100, "on", 0, 60), (200, "off", 0, 60)])

    def test_duplicate_callback_is_ignored(self):
        self.sch.schedule([Note(0, 60, 80, 100, 200)])
        on = self.sch.pending[0]
        self.assertTrue(self.sch.deliver(on, 100))
        self.assertFalse(self.sch.deliver(on, 101))

    def test_late_duplicate_on_after_off_is_ignored(self):
        self.sch.schedule([Note(0, 60, 80, 100, 200)])
        on = self.sch.pending[0]
        self.sch.tick(100)
        self.sch.tick(200)
        self.assertFalse(self.sch.deliver(on, 250))
        self.assertEqual(self.sch.sounding, {})
        self.assertEqual(len(self.sch.sent), 2)

    def test_same_pitch_overlap_offs_are_owned_per_note(self):
        """Logical-event contract only: MIDI 1.0 note-off has no note id (retrigger policy needed for a real synth)."""
        self.sch.schedule([Note(0, 60, 80, 0, 1_000), Note(1, 60, 80, 500, 1_500)])
        for t in (0, 500, 1_000, 1_500):
            self.sch.tick(t)
        self.assertEqual([x[1:3] for x in self.sch.sent], [("on", 0), ("on", 1), ("off", 0), ("off", 1)])



def midi_alternates(midi_out):
    """Per pitch the MIDI events go on, off, on, off ... and every pitch ends off."""
    state = {}
    for _, kind, p, _ in midi_out:
        if kind == "on":
            if state.get(p):
                return False
            state[p] = True
        else:
            if not state.get(p):
                return False
            state[p] = False
    return not any(state.values())


class MidiRetriggerTest(unittest.TestCase):
    def setUp(self):
        self.sch = Scheduler()

    def run_all(self, until):
        for t in range(0, until + 1, 50):
            self.sch.tick(t)

    def test_overlap_on_same_pitch_retriggers(self):
        self.sch.schedule([Note(0, 60, 80, 0, 1_000), Note(1, 60, 70, 500, 1_500)])
        self.run_all(1_500)
        self.assertEqual(self.sch.midi_out, [(0, "on", 60, 80), (500, "off", 60, 0), (500, "on", 60, 70), (1_500, "off", 60, 0)])
        self.assertTrue(midi_alternates(self.sch.midi_out))

    def test_same_time_off_then_on_is_not_a_retrigger(self):
        self.sch.schedule([Note(0, 60, 80, 0, 500), Note(1, 60, 80, 500, 1_000)])
        self.run_all(1_000)
        self.assertEqual([m[:3] for m in self.sch.midi_out], [(0, "on", 60), (500, "off", 60), (500, "on", 60), (1_000, "off", 60)])

    def test_stop_turns_off_each_owned_pitch_once(self):
        self.sch.schedule([Note(0, 60, 80, 0, 1_000), Note(1, 60, 80, 500, 1_500), Note(2, 64, 80, 0, 2_000)])
        self.run_all(700)
        self.sch.stop(700)
        self.run_all_after = [self.sch.tick(t) for t in range(750, 2_100, 50)]
        self.assertEqual(sorted(m[2] for m in self.sch.midi_out if m[0] == 700), [60, 64])
        self.assertTrue(midi_alternates(self.sch.midi_out))

    def test_late_old_generation_callback_sends_no_midi(self):
        self.sch.schedule([Note(0, 60, 80, 1_000, 2_000)])
        stale = list(self.sch.pending)
        self.sch.stop(500)
        self.sch.restart()
        for e in stale:
            self.sch.deliver(e, e.time_ms)
        self.assertEqual(self.sch.midi_out, [])

    def test_midi_due_keeps_scheduled_time_when_delivered_late(self):
        self.sch.schedule([Note(0, 60, 80, 100, 200), Note(1, 60, 80, 150, 300)])
        self.sch.tick(1_000)                               # everything delivered late, at once
        self.assertEqual([m[0] for m in self.sch.midi_out], [1_000] * 4)
        self.assertEqual(self.sch.midi_due, [100, 150, 150, 300])
        self.assertEqual(len(self.sch.midi_due), len(self.sch.midi_out))

    def test_random_overlaps_keep_midi_alternating(self):
        rng = random.Random(0)
        for trial in range(50):
            sch = Scheduler()
            notes = []
            for i in range(30):
                on = rng.randrange(0, 5_000, 10)
                notes.append(Note(i, rng.choice([60, 62, 64]), 80, on, on + rng.randrange(10, 2_000, 10)))
            sch.schedule(notes)
            for t in range(0, 7_001, 10):
                sch.tick(t)
            sch.stop(7_001)
            self.assertTrue(midi_alternates(sch.midi_out), trial)
            self.assertEqual(sum(1 for m in sch.midi_out if m[1] == "on"), len(notes))



class PlayableTest(unittest.TestCase):
    def test_velocity_zero_dropped_and_zero_length_extended(self):
        kept, dropped = playable([Note(0, 60, 0, 0, 100), Note(1, 62, 80, 200, 200), Note(2, 64, 70, 300, 400)])
        self.assertEqual([n.note_id for n in dropped], [0])
        self.assertEqual([(n.note_id, n.on_ms, n.off_ms) for n in kept], [(1, 200, 200 + MIN_DUR_MS), (2, 300, 400)])

    def test_schedule_rejects_unplayable_batch_as_a_whole(self):
        sch = Scheduler()
        for bad in (Note(1, 60, 80, 100, 100), Note(1, 60, 0, 0, 100), Note(1, 60, 128, 0, 100), Note(1, 128, 80, 0, 100)):
            with self.assertRaises(ValueError):
                sch.schedule([Note(0, 62, 80, 0, 50), bad])
            self.assertEqual(sch.pending, [])

    def test_zero_length_note_does_not_stay_on_after_playable(self):
        sch = Scheduler()
        kept, _ = playable([Note(0, 60, 80, 100, 100)])
        sch.schedule(kept)
        for t in range(0, 300, 10):
            sch.tick(t)
        self.assertEqual(sch.sounding, {})
        self.assertEqual([m[1] for m in sch.midi_out], ["on", "off"])


if __name__ == "__main__":
    unittest.main()
