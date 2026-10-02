"""fl_live --phrase is opt-in and adds the G2 runtime flags."""
from __future__ import annotations

import unittest

from scripts.fl_live import PHRASE_ARGS


class PhraseTest(unittest.TestCase):
    def test_phrase_flags(self) -> None:
        for flag in ("--context-history", "--pattern-cache", "--temperature", "--start-budget-bars"):
            self.assertIn(flag, PHRASE_ARGS)
        self.assertEqual(PHRASE_ARGS[PHRASE_ARGS.index("--temperature") + 1], "0.6")


if __name__ == "__main__":
    unittest.main()
