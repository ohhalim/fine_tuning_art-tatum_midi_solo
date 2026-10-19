"""Pure logic of the live runner (issue #1826): port names and choice, session length, commands, click, summary."""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import rt_play as rp  # noqa: E402

GARBLED = "IAC 드라이버 AI In".encode("utf-8").decode("mac_roman")      # what rtmidi returns for a Korean IAC name


class PortTest(unittest.TestCase):
    NAMES = [GARBLED, "IAC 드라이버 버스 2".encode("utf-8").decode("mac_roman"), "Arturia KeyStep 37", "Arturia MiniLab"]

    def test_display_name_undoes_mac_roman(self):
        self.assertEqual(rp.display_name(GARBLED), "IAC 드라이버 AI In")
        self.assertEqual(rp.display_name("Arturia KeyStep 37"), "Arturia KeyStep 37")

    def test_pick_by_index_and_unique_part(self):
        self.assertEqual(rp.pick_port(self.NAMES, "3"), "Arturia KeyStep 37")
        self.assertEqual(rp.pick_port(self.NAMES, "keystep"), "Arturia KeyStep 37")
        self.assertEqual(rp.pick_port(self.NAMES, "AI In"), GARBLED)          # matched through the display name
        self.assertEqual(rp.pick_port(self.NAMES, GARBLED), GARBLED)          # or the raw name

    def test_ambiguous_missing_and_out_of_range(self):
        with self.assertRaises(ValueError):
            rp.pick_port(self.NAMES, "Arturia")
        with self.assertRaises(ValueError):
            rp.pick_port(self.NAMES, "Roland")
        with self.assertRaises(ValueError):
            rp.pick_port(self.NAMES, "9")
        with self.assertRaises(ValueError):
            rp.pick_port([], "1")


class SessionTest(unittest.TestCase):
    def test_blocks_and_lead(self):
        self.assertEqual(rp.blocks_for(1), 65)                # 60 000 / 937.5 = 64 blocks + cold block
        self.assertEqual(rp.lead_for(0), 1.0)                 # the producer default
        self.assertEqual(rp.lead_for(4), 2.125)               # 4 beats 1.875 s + 0.25 s
        self.assertEqual(round(rp.BPM), 128)

    def cfg(self, existing):
        return {"py": "py", "t3_py": "t3", "aria_repo": "aria", "ckpt": "ck", "data": "d", "blocks": 65, "lh_port": "KeyStep",
                "precision": "q8", "lead_s": 2.125, "out_port": "AI In", "out_existing": existing}

    def test_commands(self):
        files = {"origin": "o", "progress": "p", "producer": "r", "ready": "y", "sender": "s"}
        cmd = rp.producer_cmd(self.cfg(True), files)
        self.assertEqual(cmd[0], "t3")
        self.assertTrue(cmd[1].endswith("rt_lh_producer.py"))
        self.assertEqual(cmd[2:10], ["aria", "ck", "d", "65", "KeyStep", "o", "p", "r"])
        self.assertEqual(cmd[10:], ["--precision", "q8", "--lead-s", "2.125"])
        self.assertEqual(rp.sender_cmd(self.cfg(True), files)[-1], "--existing-port")
        self.assertNotIn("--existing-port", rp.sender_cmd(self.cfg(False), files))

    def test_note_cap_only_when_asked(self):
        files = {"origin": "o", "progress": "p", "producer": "r", "ready": "y", "sender": "s"}
        self.assertNotIn("--note-cap", rp.producer_cmd(self.cfg(True), files))
        cfg = dict(self.cfg(True), note_cap=12)
        self.assertEqual(rp.producer_cmd(cfg, files)[-2:], ["--note-cap", "12"])
        self.assertNotIn("--note-cap", rp.producer_cmd(dict(self.cfg(True), note_cap=0), files))   # explicit 0: off

    def test_negative_note_cap_is_refused_before_anything_starts(self):
        from unittest import mock
        with mock.patch.object(rp, "run") as run, mock.patch("sys.stderr"), self.assertRaises(SystemExit):
            rp.main(["--dry-run", "--note-cap", "-1"])
        run.assert_not_called()

    def test_click_times(self):
        t = rp.click_times(10_000_000_000, 4, 8)
        self.assertEqual(len(t), 12)
        self.assertEqual(t[4], (10_000_000_000, True))       # the origin is a bar start
        self.assertEqual(t[0], (int(10_000_000_000 - 4 * 468.75e6), True))
        self.assertEqual([a for _, a in t[4:]], [True, False, False, False, True, False, False, False])

    def test_fake_lh(self):
        notes = rp.fake_lh(6)
        self.assertEqual(len(notes), 16)                      # 4 blocks x 4 notes
        self.assertTrue(all(off > on for _, on, off, _ in notes))
        self.assertEqual(notes[4][1], 938)                    # second voicing at one block


class SummaryTest(unittest.TestCase):
    def test_summary_from_rows_and_records(self):
        rows = [{"block": 0, "cold_s": 1.0}, {"block": 1, "ready_margin_ms": 120.0}, {"block": 2, "ready_margin_ms": -5.0, "problem": "deadline_miss"}]
        sender = {"end": "abort", "sent": [{"kind": "on"}, {"kind": "off"}], "rejected_late_on": [], "pending_on_at_stop": 0,
                  "sounding_at_stop": 0, "stop_offs_failed": 0, "port_closed": True, "clean_stop": True}
        s = rp.summarize(rows, None, sender, {"messages": [1, 2]})
        self.assertEqual((s["blocks_timed"], s["deadline_misses"], s["margin_min_ms"]), (2, 1, -5.0))
        self.assertEqual(s["abort"], {"reason": "deadline_miss", "block": 2})
        self.assertEqual((s["notes_on_sent"], s["captured_messages"], s["sent_messages"]), (1, 2, 2))

    def test_summary_survives_missing_records(self):
        s = rp.summarize([], None, None)
        self.assertEqual(s["blocks_timed"], 0)
        self.assertIsNone(s["abort"])


class ShutdownOrderTest(unittest.TestCase):
    """Astra review of #1833: on a user stop the sender is signalled before a slow producer is waited for."""

    class Fake:
        def __init__(self, name, log, exits_on_signal=True):
            self.name, self.log, self.exits, self.done = name, log, exits_on_signal, False

        def poll(self):
            return 0 if self.done else None

        def send_signal(self, sig):
            self.log.append(("signal", self.name, sig))
            if self.exits:
                self.done = True

        def wait(self, timeout=None):
            self.log.append(("wait", self.name))
            if not self.done:
                import subprocess
                raise subprocess.TimeoutExpired(self.name, timeout)

        def terminate(self):
            self.log.append(("terminate", self.name))
            self.done = True

        def kill(self):
            self.done = True

    def test_user_stop_signals_sender_before_waiting_on_a_slow_producer(self):
        import signal
        import threading
        log = []
        procs = {"sender": self.Fake("sender", log), "producer": self.Fake("producer", log, exits_on_signal=False)}
        stop = threading.Event()
        rp.shutdown(procs, True, stop, 1)
        self.assertTrue(stop.is_set())
        first_wait = log.index(("wait", "producer"))
        self.assertLess(log.index(("signal", "sender", signal.SIGINT)), first_wait)
        self.assertEqual(sum(1 for e in log if e[:2] == ("signal", "sender")), 1)   # once: cleanup is not cut
        self.assertIn(("terminate", "producer"), log)

    def test_normal_end_lets_the_sender_drain_first(self):
        import threading
        log = []
        procs = {"sender": self.Fake("sender", log), "producer": self.Fake("producer", log)}
        procs["producer"].done = True                 # the producer already finished
        stop = threading.Event()
        rp.shutdown(procs, False, stop, 1)
        self.assertEqual(log[0], ("wait", "sender"))  # play-out wait before any signal to the sender


if __name__ == "__main__":
    unittest.main()
