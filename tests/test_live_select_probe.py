"""Probe maps selector messages to the block that consumed them and to adoption (Astra M3)."""
from __future__ import annotations

import unittest

from scripts.run_live_select_probe import adopted_blocks, switch_latencies


def report(per_bar, events, adopted=None, used_fallback=()):
    r = {"bpm": 120, "block_beats": 4, "run_completed": True,
         "adapter_swap": {"adapters": {"tatum": "t.pt", "mehldau": "m.pt"}, "per_bar": per_bar,
                          "control_events": events},
         "bars_detail": [{"bar_index": i, "source": "fallback_not_ready" if i in used_fallback else "model",
                          "used_fallback": i in used_fallback} for i in range(len(per_bar))]}
    if adopted is not None:
        r["adopted_blocks"] = adopted
    return r


class SwitchLatencyTest(unittest.TestCase):
    def test_noop_request_gets_no_latency(self) -> None:
        # Astra's case: asking again for the active adapter used to give -1000 ms.
        ev = [{"received_ms_from_start": 3000.0, "value": 0, "selected": "tatum", "previous": "tatum",
               "noop": True, "superseded": False, "consumed_block": 2}]
        row = switch_latencies(report(["tatum"] * 4, ev, adopted=[0, 1, 2, 3]))[0]
        self.assertEqual((row["status"], row["latency_ms"]), ("noop", None))

    def test_fallback_block_is_not_counted_as_applied(self) -> None:
        ev = [{"received_ms_from_start": 1000.0, "value": 1, "selected": "mehldau", "previous": "tatum",
               "noop": False, "superseded": False, "consumed_block": 2}]
        per_bar = ["tatum", "tatum", "mehldau", "mehldau"]
        row = switch_latencies(report(per_bar, ev, adopted=[0, 1, 3]))[0]
        self.assertEqual((row["status"], row["applied_bar"], row["latency_ms"]),
                         ("applied_after_fallback", 3, 5000.0))
        row = switch_latencies(report(per_bar, ev, adopted=[0, 1, 2, 3]))[0]
        self.assertEqual((row["status"], row["applied_bar"], row["latency_ms"]), ("applied", 2, 3000.0))

    def test_superseded_and_ignored(self) -> None:
        ev = [{"received_ms_from_start": 900.0, "value": 1, "selected": "mehldau", "previous": "tatum",
               "noop": False, "superseded": True, "consumed_block": 1},
              {"received_ms_from_start": 950.0, "value": 7, "selected": None, "previous": "mehldau",
               "noop": False, "superseded": False, "consumed_block": 1}]
        rows = switch_latencies(report(["tatum"] * 3, ev, adopted=[0, 1, 2]))
        self.assertEqual([r["status"] for r in rows], ["superseded", "ignored"])

    def test_legacy_reports_are_reconstructed_and_flagged(self) -> None:
        ev = [{"received_ms_from_start": 1000.0, "value": 1, "selected": "mehldau"},
              {"received_ms_from_start": 5000.0, "value": 1, "selected": "mehldau"}]   # repeat = noop
        rows = switch_latencies(report(["tatum", "tatum", "mehldau", "mehldau"], ev))
        self.assertEqual([(r["status"], r["mapping"], r["adoption_evidence"]) for r in rows],
                         [("applied", "legacy_reconstructed", "reconstructed_all_model"),
                          ("noop", "legacy_reconstructed", "reconstructed_all_model")])
        self.assertEqual(rows[0]["latency_ms"], 3000.0)
        adopted, how = adopted_blocks(report(["tatum"] * 3, [], used_fallback=(1,)))
        self.assertEqual((sorted(adopted), how), ([0, 2], "reconstructed_partial"))


if __name__ == "__main__":
    unittest.main()
