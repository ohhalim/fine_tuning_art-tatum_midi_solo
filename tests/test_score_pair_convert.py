"""Score head converter (scripts/score_pair_convert.py) on an inline MusicXML snippet."""
from __future__ import annotations

import os
import tempfile
import unittest

from scripts.score_pair_convert import parse

XML = """<?xml version="1.0"?>
<score-partwise><part id="P1">
<measure number="1">
 <attributes><divisions>2</divisions><time><beats>4</beats><beat-type>4</beat-type></time></attributes>
 <harmony><root><root-step>C</root-step></root><kind>major-seventh</kind></harmony>
 <harmony><root><root-step>G</root-step></root><kind>dominant</kind><bass><bass-step>B</bass-step></bass><offset>4</offset></harmony>
 <note><pitch><step>E</step><octave>4</octave></pitch><duration>8</duration><tie type="start"/></note>
</measure>
<measure number="2">
 <harmony><root><root-step>D</root-step></root><kind>suspended-fourth</kind></harmony>
 <note><pitch><step>E</step><octave>4</octave></pitch><duration>8</duration><tie type="stop"/><tie type="start"/></note>
</measure>
<measure number="3">
 <harmony><kind>none</kind></harmony>
 <note><pitch><step>E</step><octave>4</octave></pitch><duration>2</duration><tie type="stop"/></note>
 <note><pitch><step>B</step><alter>-1</alter><octave>4</octave></pitch><duration>2</duration><time-modification><actual-notes>3</actual-notes><normal-notes>2</normal-notes></time-modification></note>
 <note><rest/><duration>4</duration></note>
</measure>
</part></score-partwise>"""


class ScorePairConvertTest(unittest.TestCase):
    def setUp(self) -> None:
        fd, self.path = tempfile.mkstemp(suffix=".xml")
        with os.fdopen(fd, "w") as f:
            f.write(XML)
        self.p = parse(self.path)

    def tearDown(self) -> None:
        os.remove(self.path)

    def test_tie_chain_across_three_measures_is_one_note(self) -> None:
        e = [n for n in self.p["notes"] if n["pitch"] == 64]
        self.assertEqual(len(e), 1)
        self.assertEqual((e[0]["onset"], e[0]["end"], e[0]["tied_parts"]), (0, 9, 3))

    def test_harmony_offset_slash_bass_no_chord_and_unmapped_kind(self) -> None:
        segs = [(float(s["onset"]), float(s["end"]), s["label"], s.get("quality"), s.get("bass")) for s in self.p["segments"]]
        self.assertEqual(segs, [(0, 2, "chord", "maj7", None), (2, 4, "chord", "7", "B"),
                                (4, 8, "chord", None, None), (8, 12, "no_chord", None, None)])
        self.assertIn("harmony kind 'suspended-fourth'", [u["item"] for u in self.p["unsupported"]])

    def test_measure_map_and_tuplet_duration(self) -> None:
        self.assertEqual([m["start_beat"] for m in self.p["measures"]], [0, 4, 8])
        bb = [n for n in self.p["notes"] if n["pitch"] == 70][0]
        self.assertEqual((float(bb["onset"]), float(bb["end"])), (9, 10))


if __name__ == "__main__":
    unittest.main()
