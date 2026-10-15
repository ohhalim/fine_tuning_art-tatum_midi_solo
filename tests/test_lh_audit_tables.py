"""Result tables come from the report only (#1750, #1754). Synthetic MIDI."""
from __future__ import annotations

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, os.path.dirname(__file__))
from lh_audit_stats import aggregate, audit_song, check_counts, decision_set, groups, verdicts  # noqa: E402
from lh_audit_tables import runtime_tables  # noqa: E402
from test_lh_audit_stats import song, write_midi  # noqa: E402


class RuntimeTablesTest(unittest.TestCase):
    def test_runtime_tables_render_from_a_report(self):
        names, judged = decision_set("runtime_block")
        with tempfile.TemporaryDirectory() as tmp:
            a, b = os.path.join(tmp, "a.mid"), os.path.join(tmp, "b.mid")
            write_midi(a, song())
            write_midi(b, song(0.13))
            res = audit_song({"idx": 0, "stratum": "main", "path": a, "sha1": None, "donor_path": b, "decisions": "runtime_block"})
        res.pop("rows")
        meta = {0: {"pianist": "Test Pianist", "rec": "studio"}}
        main = aggregate([res], "main", tuple(names))
        main["note_count_mismatch"] = 0
        for dn in judged:
            main[f"per_pianist_{dn}"] = groups([res], meta, "pianist", dn)
        main["verdicts"] = verdicts(main, judged)
        fable = aggregate([res], "afable", tuple(names))
        fable["checks"] = check_counts(fable)
        md = runtime_tables({"params": {"decision_set": "runtime_block"}, "strata": {"main": main, "afable": fable}})
        self.assertIn("| `g3` | sounding | – |", md)                       # no #1750 report given: comparison column empty
        self.assertIn("| Test Pianist | 1 |", md)
        self.assertEqual(md.count("| `top50` | block_p"), 4)
        self.assertIn("**cold(블록 0)와 warm (주대상)**", md)
        self.assertEqual(md.count("| `grid300` | recent | "), 2)                    # coverage row and cold/warm row


if __name__ == "__main__":
    unittest.main()
