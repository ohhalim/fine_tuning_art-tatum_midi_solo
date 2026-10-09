#!/usr/bin/env python3
"""Independent check of the v2 dry-run sample rows against the source XML with music21
(docs/experiments/ARIA_COND_CONTRACT_V2.md). Does not use the converter or the v2 builder.
Run with music21 on PYTHONPATH: aria_cond_v2_xmlcheck.py <dry-run json>
"""
from __future__ import annotations

import json
import sys

import ast

from music21 import converter, harmony, note

M21_TONES = {"major": {0, 4, 7}, "minor": {0, 3, 7}, "dominant-seventh": {0, 4, 7, 10}, "major-seventh": {0, 4, 7, 11},
             "minor-seventh": {0, 3, 7, 10}, "half-diminished-seventh": {0, 3, 6, 10}, "diminished-seventh": {0, 3, 6, 9},
             "major-sixth": {0, 4, 7, 9}, "minor-sixth": {0, 3, 7, 9}, "dominant-ninth": {0, 4, 7, 10},
             "dominant-13th": {0, 4, 7, 10}, "minor-ninth": {0, 3, 7, 10}, "major-ninth": {0, 4, 7, 11}}


def main() -> None:
    path = sys.argv[1]
    with open(path) as f:
        report = json.load(f)
    for song, r in report.items():
        part = converter.parse(r["source_xml"]).parts[0]
        cs = sorted(part.flatten().getElementsByClass(harmony.ChordSymbol), key=lambda c: c.offset)
        notes = [(float(n.offset), n.pitch.midi) for n in part.stripTies().flatten().getElementsByClass(note.Note)]
        tol = 0.005 * r["bpm"] / 60.0 + 1e-6          # half of the 10 ms token step, in beats
        rows = [(float(c.offset), sorted((c.root().pitchClass + i) % 12 for i in M21_TONES[c.chordKind])) for c in cs]
        for s in r["samples"]:
            cur = next((p for i, (o, p) in enumerate(rows) if o <= s["beat"] and (i + 1 == len(rows) or rows[i + 1][0] > s["beat"])), None)
            later = [(o, p) for o, p in rows if o > s["beat"] and p != cur]
            nxt = later[0] if later else None
            delta = None if nxt is None else round((nxt[0] - s["beat"]) * 60.0 / r["bpm"], 4)
            s["music21"] = {"current": cur, "next": nxt[1] if nxt else None, "delta_sec": delta}
            s["match"] = cur == s["current"] and (nxt[1] if nxt else None) == s["next"] and (
                delta is None and s["delta_sec"] is None or abs(delta - s["delta_sec"]) < 1e-3)
            tk = ast.literal_eval(s["token"])
            if isinstance(tk, tuple) and tk[0] == "onset":
                pitch = ast.literal_eval(prev["token"])[1]
                s["onset_in_xml"] = any(abs(o - s["beat"]) <= tol and p == pitch for o, p in notes)
                s["match"] = s["match"] and s["onset_in_xml"]
            prev = s
        r["xml_check"] = {"tool": "music21", "samples": len(r["samples"]), "matched": sum(s["match"] for s in r["samples"]),
                          "onset_rows_found_in_xml": sum(1 for s in r["samples"] if s.get("onset_in_xml")),
                          "onset_rows": sum(1 for s in r["samples"] if "onset_in_xml" in s)}
        print(song, r["xml_check"])
    with open(path, "w") as f:
        json.dump(report, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
