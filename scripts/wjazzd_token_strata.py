#!/usr/bin/env python3
"""Token-exact v2 strata on a fixed WJazzD sample (docs/experiments/WJAZZD_AUDIT.md).

Sample fixed before the run: melid 1-10. Each solo is written as MIDI (velocity fixed, timing
from the database), tokenized with the Aria tokenizer, and for every note the chord at its actual
onset (time-based beat timeline) is compared with current / first next chord at the pitch token's
prefix time (contract v2). Also gives the score-level approximation for the same solos. Aggregates
only. Run with the Aria venv: wjazzd_token_strata.py <aria repo> <wjazzd.db> <out json>
"""
from __future__ import annotations

import bisect
import collections
import json
import os
import sqlite3
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from aria_cond_contract_v2 import features  # noqa: E402
from aria_t4b_sim import write_notes  # noqa: E402
from wjazzd_audit import parse_chord  # noqa: E402

SAMPLE = list(range(1, 11))


def key(s):
    root, q, *_ = parse_chord(s)
    return "N.C." if q == "N.C." else (root, q) if q else ("unmapped", s)


def main() -> None:
    aria_repo, db, out = sys.argv[1:4]
    sys.path.insert(0, aria_repo)
    from ariautils.midi import MidiDict
    from ariautils.tokenizer import AbsTokenizer
    tok = AbsTokenizer()
    c = sqlite3.connect(db)
    res = {}
    for melid in SAMPLE:
        notes = c.execute("select onset, pitch, duration from melody where melid=? order by eventid", (melid,)).fetchall()
        changes = c.execute("select onset, chord from beats where melid=? and chord!='' order by onset", (melid,)).fetchall()
        t0 = notes[0][0]
        end = max(o + d for o, _, d in notes) + 1.0
        plan = []
        for i, (t, s) in enumerate(changes):
            nxt = changes[i + 1][0] if i + 1 < len(changes) else end
            plan.append(((t - t0) * 1000.0, (nxt - t0) * 1000.0, key(s)))
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "x.mid")
            write_notes([(int(p), o - t0, o - t0 + d, 80) for o, p, d in notes], path)
            toks = tok.tokenize(MidiDict.from_midi(path))
        feats = features(toks, plan)
        exact = collections.Counter()
        for i, t in enumerate(toks[:-1]):
            if isinstance(t, tuple) and t[0] == "piano" and isinstance(toks[i + 1], tuple) and toks[i + 1][0] == "onset":
                onset_t = feats[i + 1]["t_ms"]
                seg = next((sg for sg in plan if sg[0] <= onset_t < sg[1]), None)
                on = None if seg is None else seg[2]
                exact["same_as_current" if on == feats[i]["current"] else "equals_first_next" if on == feats[i]["next"]
                      else "not_represented"] += 1
        # score-level approximation (previous note onset as prefix time), same definition as wjazzd_audit.py
        times = [p[0] for p in plan]
        merged = []
        for p in plan:
            if not merged or merged[-1][2] != p[2]:
                merged.append(p)
        mt = [p[0] for p in merged]
        approx = collections.Counter()
        prev = None
        for o, _, _ in notes:
            t = (o - t0) * 1000.0
            if prev is not None:
                j = bisect.bisect_right(mt, prev) - 1
                cur = merged[j][2] if j >= 0 else None
                nxt = merged[j + 1][2] if j + 1 < len(merged) else None
                k = bisect.bisect_right(times, t) - 1
                on = plan[k][2] if k >= 0 else None
                approx["same_as_current" if on == cur else "equals_first_next" if on == nxt else "not_represented"] += 1
            prev = t
        res[melid] = {"notes": len(notes), "tokens": len(toks), "token_exact": dict(exact), "approx": dict(approx)}
        print(melid, res[melid])
    tot = {k: collections.Counter() for k in ("token_exact", "approx")}
    for r in res.values():
        for k in tot:
            tot[k].update(r[k])
    with open(out, "w") as f:
        json.dump({"sample_melids": SAMPLE, "per_solo": res, "pooled": {k: dict(v) for k, v in tot.items()}}, f, indent=1)
    print({k: dict(v) for k, v in tot.items()})


if __name__ == "__main__":
    main()
