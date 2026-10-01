"""Port lookup for the FL Studio launcher (scripts/fl_live.py)."""
from __future__ import annotations

import unittest

from scripts.fl_live import find_ports


class FindPortsTest(unittest.TestCase):
    def test_picks_ai_in_as_source_and_the_other_iac_bus_as_destination(self) -> None:
        names = ["IAC ÎìúÎùºÏù¥Î≤Ñ AI In", "IAC ÎìúÎùºÏù¥Î≤Ñ Î≤ÑÏä§ 2"]
        self.assertEqual(find_ports(names, names), (names[0], names[1]))

    def test_missing_ports_stop_with_a_message(self) -> None:
        with self.assertRaises(SystemExit):
            find_ports([], [])


if __name__ == "__main__":
    unittest.main()
