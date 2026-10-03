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


class CompStyleTest(unittest.TestCase):
    def test_varied_is_the_fl_default_and_shell_can_be_chosen(self) -> None:
        cmd = command("--preset", "bebop")
        self.assertEqual(cmd[cmd.index("--comp-style") + 1], "varied")
        cmd = command("--preset", "bebop", "--comp-style", "shell")
        self.assertEqual(cmd[cmd.index("--comp-style") + 1], "shell")
        self.assertNotIn("--comp-style", command("--preset", "bebop", "--no-comp"))


class CandidatesTest(unittest.TestCase):
    def test_off_by_default_and_passed_when_set(self) -> None:
        self.assertNotIn("--candidates", command("--preset", "bebop"))
        cmd = command("--preset", "bebop", "--candidates", "2")
        self.assertEqual(cmd[cmd.index("--candidates") + 1], "2")


class CarryTest(unittest.TestCase):
    def test_off_by_default_and_after_the_guide_when_set(self) -> None:
        self.assertNotIn("--context-carry-tokens", command("--preset", "bebop"))
        cmd = command("--preset", "bebop", "--carry", "48")
        self.assertEqual(cmd[cmd.index("--context-carry-tokens") + 1], "48")
        self.assertEqual(cmd[cmd.index("--context-carry-position") + 1], "after")


if __name__ == "__main__":
    unittest.main()
