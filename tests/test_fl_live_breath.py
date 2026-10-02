"""fl_live --breath passes --phrase-breath to the runtime only with --solo."""
from __future__ import annotations

import sys
import unittest
from unittest import mock

from scripts import fl_live


def command(*argv) -> list[str]:
    fake = mock.MagicMock()                                   # also absorbs the all-notes-off at exit
    fake.get_input_names.return_value = ["IAC Driver AI In"]
    fake.get_output_names.return_value = ["IAC Driver AI In", "IAC Driver 버스 2"]
    with mock.patch.dict(sys.modules, {"mido": fake}), mock.patch.object(fl_live.subprocess, "call",
                                                                          return_value=0) as call:
        fl_live.main(list(argv))
    return call.call_args.args[0]


class BreathTest(unittest.TestCase):
    def test_off_by_default(self) -> None:
        self.assertNotIn("--phrase-breath", command("--preset", "bebop"))

    def test_breath_with_solo(self) -> None:
        cmd = command("--preset", "bebop", "--breath", "24")
        self.assertEqual(cmd[cmd.index("--phrase-breath") + 1], "24")

    def test_no_breath_without_solo(self) -> None:
        self.assertNotIn("--phrase-breath", command("--preset", "bebop", "--breath", "24", "--no-solo"))


if __name__ == "__main__":
    unittest.main()
