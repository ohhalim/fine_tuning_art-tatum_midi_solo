"""Regression for the real-clock sender and the smoke analysis (issue #1747, Astra post-review). Fake port, fake clock."""
from __future__ import annotations

import json
import os
import queue
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from rt_sender import Sender, drain, reader  # noqa: E402
from rt_smoke_run import analyze  # noqa: E402


class FakePort:
    def __init__(self, fail=lambda m: False):
        self.fail, self.sent, self.tries, self.closed = fail, [], [], False

    def send(self, m):
        self.tries.append(m)
        if self.fail(m):
            raise OSError("send failed")
        self.sent.append(m)

    def close(self):
        self.closed = True


class Clock:
    def __init__(self):
        self.ns = 0

    def __call__(self):
        return self.ns


def start(s, clock):
    s.handle({"type": "start", "music_origin_ms": 0, "wall_origin_ns": 0})
    s.handle({"type": "notes", "block": 0, "notes": [[0, 60, 80, 0, 1000], [1, 64, 80, 0, 2000]]})
    clock.ns = int(10e6)
    s.step()                                         # both notes sounding


class SenderTest(unittest.TestCase):
    def test_off_failure_in_cleanup_still_closes_port_and_writes_record(self):
        clock = Clock()
        port = FakePort(fail=lambda m: m[0] == "note_off")
        s = Sender(port, clock)
        start(s, clock)
        log = s.cleanup()
        self.assertTrue(port.closed)
        self.assertTrue(log["port_closed"])
        self.assertEqual(log["stop_offs_failed"], 2)
        self.assertFalse(log["clean_stop"])
        self.assertEqual(sum(1 for m in port.tries if m[0] == "note_off"), 2)   # each off tried once, no retry loop

    def test_send_failure_during_run_is_not_retried_by_cleanup(self):
        clock = Clock()
        port = FakePort(fail=lambda m: m == ("note_on", 64, 80))
        s = Sender(port, clock)
        s.handle({"type": "start", "music_origin_ms": 0, "wall_origin_ns": 0})
        s.handle({"type": "notes", "block": 0, "notes": [[0, 60, 80, 0, 1000], [1, 64, 80, 0, 2000]]})
        clock.ns = int(10e6)
        with self.assertRaises(OSError):
            s.step()
        log = s.cleanup()
        self.assertEqual(port.tries.count(("note_on", 64, 80)), 1)
        self.assertTrue(port.closed)
        self.assertFalse(log["clean_stop"])

    def test_cleanup_cancels_unsent_note_on_and_sends_only_offs(self):
        clock = Clock()
        port = FakePort(fail=lambda m: m == ("note_on", 64, 80))
        s = Sender(port, clock)
        s.handle({"type": "start", "music_origin_ms": 0, "wall_origin_ns": 0})
        s.handle({"type": "notes", "block": 0, "notes": [[0, 60, 80, 0, 1000], [1, 64, 80, 0, 1000], [2, 67, 80, 0, 1000]]})
        clock.ns = int(10e6)
        with self.assertRaises(OSError):
            s.step()                                     # on60 sent, on64 failed, on67 not tried
        port.tries.clear()
        log = s.cleanup()
        self.assertEqual(port.tries, [("note_off", 60, 0), ("note_off", 64, 0), ("note_off", 67, 0)])
        self.assertEqual([c["pitch"] for c in log["cancelled_unsent_on"]], [67])
        self.assertFalse(log["clean_stop"])

    def test_close_failure_still_returns_record(self):
        clock = Clock()
        port = FakePort()

        def bad_close():
            raise OSError("close failed")
        port.close = bad_close
        s = Sender(port, clock)
        start(s, clock)
        log = s.cleanup()
        self.assertFalse(log["port_closed"])
        self.assertFalse(log["clean_stop"])

    def test_clean_stop_turns_off_sounding_notes(self):
        clock = Clock()
        port = FakePort()
        s = Sender(port, clock)
        start(s, clock)
        log = s.cleanup()
        self.assertTrue(log["clean_stop"])
        self.assertEqual(sorted(m[1] for m in port.sent if m[0] == "note_off"), [60, 64])
        self.assertEqual(len(log["expected_midi"]), 4)

    def test_note_already_late_at_intake_is_recorded_and_never_sent(self):
        clock = Clock()
        port = FakePort()
        s = Sender(port, clock)
        start(s, clock)                              # music now 10 ms
        clock.ns = int(300e6)
        s.handle({"type": "notes", "block": 1, "notes": [[2, 67, 80, 250, 900], [3, 69, 80, 400, 900]]})
        clock.ns = int(500e6)
        s.step()
        self.assertNotIn(("note_on", 67, 80), port.sent)
        self.assertIn(("note_on", 69, 80), port.sent)
        self.assertEqual([(r["note_id"], r["block"], r["on_ms"]) for r in s.log["rejected_late_on"]], [(2, 1, 250)])
        self.assertEqual(s.log["scheduled_notes"], 3)
        self.assertEqual({(r["note_id"], r["block"]) for r in s.log["sent"] if r["kind"] == "on"}, {(0, 0), (1, 0), (3, 1)})

    def test_abort_stops_before_any_new_note_and_cleanup_turns_off(self):
        clock = Clock()
        port = FakePort()
        s = Sender(port, clock)
        start(s, clock)                              # 60 and 64 sounding
        s.handle({"type": "notes", "block": 1, "notes": [[2, 67, 80, 500, 900]]})
        clock.ns = int(600e6)                        # 67 is due, but the abort arrives first
        done = s.process([{"type": "abort", "reason": "deadline_miss", "block": 1}])
        self.assertTrue(done)
        self.assertNotIn(("note_on", 67, 80), port.sent)
        log = s.cleanup()
        self.assertEqual(log["end"], "abort")
        self.assertEqual(log["abort"], {"reason": "deadline_miss", "block": 1})
        self.assertEqual(sorted(m[1] for m in port.sent if m[0] == "note_off"), [60, 64])
        self.assertEqual(log["pending_on_at_stop"], 1)  # 67 was still pending in the scheduler and is dropped by stop
        self.assertNotIn(("note_on", 67, 80), port.sent)

    def test_normal_end_waits_for_pending_and_ignores_the_following_eof(self):
        clock = Clock()
        s = Sender(FakePort(), clock)
        start(s, clock)
        s.handle({"type": "end"})
        s.handle({"type": "eof"})
        self.assertEqual(s.end, "end")
        self.assertFalse(s.done())                   # notes still pending
        clock.ns = int(2500e6)
        s.step()
        self.assertTrue(s.done())

    def test_reader_turns_a_broken_line_into_an_error_then_eof(self):
        q = queue.Queue()
        reader(['{"type": "start", "music_origin_ms": 0, "wall_origin_ns": 0}\n', '{"type": "no'], q)
        msgs = [q.get_nowait() for _ in range(q.qsize())]
        self.assertEqual([m["type"] for m in msgs], ["start", "reader_error", "eof"])

    def test_reader_error_stops_the_run_and_cleanup_turns_off(self):
        clock = Clock()
        port = FakePort()
        s = Sender(port, clock)
        start(s, clock)
        s.handle({"type": "reader_error", "error": "JSONDecodeError"})
        self.assertTrue(s.done())
        log = s.cleanup()
        self.assertEqual(log["error"], "JSONDecodeError")
        self.assertEqual(log["end"], "reader_error")
        self.assertEqual(sorted(m[1] for m in port.sent if m[0] == "note_off"), [60, 64])


class MainLoopPathTest(unittest.TestCase):
    """The same path as main(): reader -> queue -> drain -> Sender.process -> cleanup."""

    def run_lines(self, lines, clock_ns, port):
        clock = Clock()
        clock.ns = clock_ns
        s = Sender(port, clock)
        q = queue.Queue()
        reader(lines, q)
        done = s.process(drain(q))
        return s, done

    def lines(self, tail):
        return ['{"type": "start", "music_origin_ms": 0, "wall_origin_ns": 0}\n',
                '{"type": "notes", "block": 0, "notes": [[0, 60, 80, 10, 1000]]}\n'] + tail   # due at 10 ms, handed in on time

    def test_eof_with_a_due_note_starts_no_note(self):
        port = FakePort()
        s, done = self.run_lines(self.lines([]), int(10e6), port)       # note due, then EOF without end
        self.assertTrue(done)
        log = s.cleanup()
        self.assertEqual([m for m in port.sent if m[0] == "note_on"], [])
        self.assertEqual(log["end"], "eof")

    def test_broken_line_with_a_due_note_starts_no_note(self):
        port = FakePort()
        s, done = self.run_lines(self.lines(['{"type": "no']), int(10e6), port)
        self.assertTrue(done)
        s.cleanup()
        self.assertEqual([m for m in port.sent if m[0] == "note_on"], [])

    def test_eof_after_a_sounding_note_turns_it_off(self):
        clock = Clock()
        port = FakePort()
        s = Sender(port, clock)
        q = queue.Queue()
        reader(self.lines([]), q)
        msgs = drain(q)
        eof = msgs.pop()                                                  # deliver start + notes first, before the note is due
        self.assertFalse(s.process(msgs))
        clock.ns = int(10e6)
        self.assertFalse(s.process([]))                                   # note-on sent
        self.assertTrue(s.process([eof]))
        s.cleanup()
        self.assertEqual(port.sent, [("note_on", 60, 80), ("note_off", 60, 0)])

    def test_normal_end_drains_scheduled_notes(self):
        clock = Clock()
        port = FakePort()
        s = Sender(port, clock)
        q = queue.Queue()
        reader(self.lines(['{"type": "end"}\n']), q)
        self.assertFalse(s.process(drain(q)))                             # notes handed in before they are due, then end + eof
        clock.ns = int(10e6)
        self.assertFalse(s.process([]))                                   # note still playing
        clock.ns = int(1500e6)
        self.assertTrue(s.process([]))
        self.assertEqual(port.sent, [("note_on", 60, 80), ("note_off", 60, 0)])


class AnalyzeTest(unittest.TestCase):
    def write(self, d, sender, capture, extra_rows=()):
        json.dump(sender, open(os.path.join(d, "sender.json"), "w"))
        json.dump({"messages": capture}, open(os.path.join(d, "capture.json"), "w"))
        rows = [{"block": 0, "cold_s": 0.1}] + list(extra_rows)
        open(os.path.join(d, "producer_progress.jsonl"), "w").write("".join(json.dumps(r) + "\n" for r in rows))

    def sender_log(self, sent, expected, logical, scheduled):
        return {"sent": sent, "expected_midi": expected, "logical_sent": logical, "scheduled_notes": scheduled,
                "pending_on_at_stop": 0, "received_blocks": [], "end": "end", "error": None, "port_closed": True,
                "pending_at_stop": 0, "sounding_at_stop": 0, "dropped_velocity0": 0, "extended_zero_length": 0,
                "send_failures": [], "clean_stop": True, "origin": None}

    def test_a_pair_missing_from_both_sent_and_capture_is_detected(self):
        sent = [{"kind": "on", "pitch": 60, "velocity": 80, "scheduled_ns": 0, "send_return_ns": 1},
                {"kind": "off", "pitch": 60, "velocity": 0, "scheduled_ns": 10, "send_return_ns": 11}]
        expected = [["on", 60, 80, 0], ["on", 64, 80, 0], ["off", 60, 0, 10], ["off", 64, 0, 20]]
        capture = [{"capture_ns": 2, "bytes": [0x90, 60, 80]}, {"capture_ns": 12, "bytes": [0x80, 60, 0]}]
        logical = [[0, "on", 0, 60], [10, "off", 0, 60]]
        with tempfile.TemporaryDirectory() as d:
            self.write(d, self.sender_log(sent, expected, logical, 2), capture)
            r = analyze("t", d)
        self.assertTrue(r["checks"]["A1b_sent_equals_capture"]["ok"])
        self.assertFalse(r["checks"]["A1a_scheduler_expected_equals_sent"]["ok"])
        self.assertFalse(r["checks"]["A3_scheduled_notes_accounted"]["ok"])

    def test_block_stopped_before_generation_does_not_break_the_analysis(self):
        sent = [{"kind": "on", "pitch": 60, "velocity": 80, "scheduled_ns": 0, "send_return_ns": 1},
                {"kind": "off", "pitch": 60, "velocity": 0, "scheduled_ns": 10, "send_return_ns": 11}]
        capture = [{"capture_ns": 2, "bytes": [0x90, 60, 80]}, {"capture_ns": 12, "bytes": [0x80, 60, 0]}]
        rows = [{"block": 1, "late_start_ms": 0.1, "ready_margin_ms": 50.0, "ready_s": 0.2},
                {"block": 2, "problem": "capacity_failure", "detail": "no <T> fits"}]
        with tempfile.TemporaryDirectory() as d:
            self.write(d, self.sender_log(sent, [["on", 60, 80, 0], ["off", 60, 0, 10]], [[0, "on", 0, 60], [10, "off", 0, 60]], 1), capture, rows)
            r = analyze("t", d)
        self.assertEqual(r["producer"]["blocks_without_timing"], [2])
        self.assertEqual(r["producer"]["deadline_misses"], 0)

    def test_without_return_codes_exit_is_not_claimed(self):
        sent = [{"kind": "on", "pitch": 60, "velocity": 80, "scheduled_ns": 0, "send_return_ns": 1},
                {"kind": "off", "pitch": 60, "velocity": 0, "scheduled_ns": 10, "send_return_ns": 11}]
        capture = [{"capture_ns": 2, "bytes": [0x90, 60, 80]}, {"capture_ns": 12, "bytes": [0x80, 60, 0]}]
        with tempfile.TemporaryDirectory() as d:
            self.write(d, self.sender_log(sent, [["on", 60, 80, 0], ["off", 60, 0, 10]], [[0, "on", 0, 60], [10, "off", 0, 60]], 1), capture)
            r = analyze("t", d)
        self.assertIsNone(r["checks"]["sender_exit"]["ok"])
        self.assertTrue(r["checks"]["sender_exit"]["recorded_ok"])
        self.assertTrue(r["checks"]["A1a_scheduler_expected_equals_sent"]["ok"])


if __name__ == "__main__":
    unittest.main()
