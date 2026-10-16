"""Fixed song split rule (issue #1760 step B)."""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from cond_adapter_split import pick, title  # noqa: E402


class SplitTest(unittest.TestCase):
    def test_title_normalization(self):
        self.assertEqual(title("/x/Cherokee - Live At Maybeck.midi"), "cherokee")
        self.assertEqual(title("/x/Body And Soul (Take 2).mid"), "body and soul")

    def test_sha1_order_and_no_repeated_title(self):
        songs = [{"sha1": f"{i:02x}", "source": f"/x/Tune {i % 5} - live.mid"} for i in range(30)]
        tr, dev = pick(songs, 3, 2)
        self.assertEqual(tr + dev, [0, 1, 2, 3, 4])                        # sha1 order, titles 0..4 once each
        songs[1]["source"] = "/x/Tune 0 - other.mid"                       # song 1 now repeats song 0's title
        tr, dev = pick(songs, 3, 2)
        self.assertNotIn(1, tr + dev)
        self.assertEqual(len({title(songs[i]["source"]) for i in tr + dev}), 5)


if __name__ == "__main__":
    unittest.main()
