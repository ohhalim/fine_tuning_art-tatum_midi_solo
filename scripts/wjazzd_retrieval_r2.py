#!/usr/bin/env python3
"""Retrieval R2: the same baseline on windows with a real chord change, and an explicit MIDI export (issue #1720).

Step 1 (--fix-targets) writes the target manifest before any search: for melid 16, 347, 18, 25 the
first window that is eligible (exact chords, full 8 beats) and, after merging repeated chords, has at
least one chord change inside. A song without one is recorded as no_eligible_target.
Step 2 (--search) reads that manifest and runs the #1718 bank, matching, transposition, uniform time
scale and ranking unchanged. Export quantises to the writer's grid; notes that would get zero ticks
are excluded and reported (zero_tick_excluded); a candidate left with no note is invalid_export.
Run with the Aria venv:
  wjazzd_retrieval_r2.py --fix-targets <wjazzd.db> <manifest json>
  wjazzd_retrieval_r2.py --search <aria repo> <wjazzd.db> <pilot data_summary json> <manifest json> <out dir> <report json>
"""
from __future__ import annotations

import collections
import hashlib
import json
import os
import sqlite3
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from wjazzd_retrieval import TOP_K, build_windows, match, quantize_for_export  # noqa: E402

TARGETS = [16, 347, 18, 25]


def beats_of(c, m):
    return c.execute("select onset, bar, beat, chord from beats where melid=? order by onset", (m,)).fetchall()


def fix_targets(db, manifest_path):
    c = sqlite3.connect(db)
    out = {"rule": "first eligible window (exact chords, full 8 beats) with at least one chord change after merging repeats",
           "melids": TARGETS, "targets": []}
    for m in TARGETS:
        ws = [w for w in build_windows(beats_of(c, m), m) if w["eligible"] and len(w["rel"]) >= 2]
        if not ws:
            out["targets"].append({"melid": m, "status": "no_eligible_target"})
            continue
        w = ws[0]
        out["targets"].append({"melid": m, "status": "fixed", "bar": w["bar"], "start": w["start"], "end": w["end"],
                               "chords": w["chords"], "rel": w["rel"]})
    with open(manifest_path, "w") as f:
        json.dump(out, f, indent=1)
    with open(manifest_path, "rb") as f:
        print("manifest sha256", hashlib.sha256(f.read()).hexdigest()[:16])
    print(json.dumps([(t["melid"], t["status"], t.get("bar"), t.get("rel")) for t in out["targets"]]))


def search(aria_repo, db, summary_path, manifest_path, out_dir, report_path):
    sys.path.insert(0, aria_repo)
    from ariautils.midi import MidiDict
    from ariautils.tokenizer import AbsTokenizer
    from aria_t4b_sim import read_notes, write_notes
    from aria_token_validity import check as token_check
    tok = AbsTokenizer()
    os.makedirs(out_dir, exist_ok=True)
    c = sqlite3.connect(db)
    with open(manifest_path, "rb") as f:
        manifest_sha = hashlib.sha256(f.read()).hexdigest()
    with open(manifest_path) as f:
        manifest = json.load(f)
    with open(summary_path) as f:
        summ = json.load(f)
    train = [s["melid"] for s in summ["songs"] if s["split"] == "train"]

    def notes_of(m, start, end):
        return c.execute("select onset, pitch, duration from melody where melid=? and onset<? and onset+duration>? order by eventid",
                         (m, end, start)).fetchall()

    t0 = time.perf_counter()
    bank = [w for m in train for w in build_windows(beats_of(c, m), m) if w["eligible"]]
    by_rel = collections.defaultdict(list)
    for w in bank:
        by_rel[w["rel"]].append(w)
    index_s = time.perf_counter() - t0
    report = {"manifest_sha256": manifest_sha, "bank_windows": len(bank), "index_build_s": round(index_s, 3),
              "search_time_scope": "matching and placement only, index build excluded", "results": []}
    for t in manifest["targets"]:
        if t["status"] != "fixed":
            report["results"].append({"melid": t["melid"], "result": "no_eligible_target"})
            continue
        tgt = next(w for w in build_windows(beats_of(c, t["melid"]), t["melid"]) if w["bar"] == t["bar"])
        assert [list(x) for x in tgt["rel"]] == [list(x) for x in t["rel"]], "manifest and rebuilt window differ"
        ts = time.perf_counter()
        ranked, excluded = match(tgt, by_rel, notes_of)
        search_s = time.perf_counter() - ts
        entry = {"melid": t["melid"], "target_bar": t["bar"], "target_chords": t["chords"], "target_rel": t["rel"],
                 "target_seconds": round(tgt["end"] - tgt["start"], 4), "bank_rel_matches": len(by_rel.get(tgt["rel"], [])),
                 "candidates_after_exclusion": len(ranked), "excluded": dict(excluded), "search_s": round(search_s, 6), "candidates": []}
        exported = 0
        for r, cnd in enumerate(ranked[:TOP_K], 1):
            kept, zero, checks = quantize_for_export(cnd["notes"], tgt["end"] - tgt["start"])
            rec = {"rank": r, "source": "WJazzD DB melody rows", "source_melid": cnd["melid"], "source_bar": cnd["bar"],
                   "source_time_s": [round(cnd["start"], 3), round(cnd["end"], 3)], "transpose": cnd["transpose"],
                   "time_scale": round(cnd["scale"], 4), "adopted_notes": len(cnd["notes"]),
                   "carry_in_excluded": cnd["carry_in_excluded"], "cropped_at_end": cnd["cropped_at_end"],
                   "cropped_seconds": round(cnd["cropped_seconds"], 4), "zero_tick_excluded": len(zero), "export_checks": checks,
                   "boundary_errors_s": [{"beat": b["beat"], "error_s": round(b["error_s"], 4)} for b in cnd["boundary_errors"]],
                   "max_abs_boundary_error_s": round(max((abs(b["error_s"]) for b in cnd["boundary_errors"]), default=0.0), 4)}
            if not kept:
                rec["export"] = "invalid_export"
            else:
                base = os.path.join(out_dir, f"r2_target{t['melid']}_rank{r}_src{cnd['melid']}_bar{cnd['bar']}")
                write_notes([(p, on / 1000, off / 1000, 80) for _, p, on, off in kept], base + ".mid")
                back = sorted((p, round(s * 1000), round(e * 1000)) for p, s, e, _ in read_notes(base + ".mid"))
                rec["midi_roundtrip_exact"] = back == sorted((p, on, off) for _, p, on, off in kept)
                toks = tok.tokenize(MidiDict.from_midi(base + ".mid"))
                head = next(i for i, x in enumerate(toks) if x == "<S>") + 1
                rec["aria_token_layer_valid"] = token_check(toks[:head], toks[head:])["valid"]
                rec["export"] = "ok"
                exported += 1
                with open(base + "_sidecar.json", "w") as f:
                    json.dump({"source_rows": cnd["rows"], "kept": kept, "zero_tick_excluded": zero, "checks": checks}, f, indent=1, default=str)
            entry["candidates"].append(rec)
        entry["result"] = "no_match" if not ranked else ("returned" if exported else "invalid_export")
        entry["independent_sources"] = len({x["source_melid"] for x in entry["candidates"]})
        report["results"].append(entry)
        print(t["melid"], entry["result"], "rel matches", entry["bank_rel_matches"], "after excl", len(ranked),
              [(x["source_melid"], x["source_bar"], x["transpose"], x["time_scale"], x["max_abs_boundary_error_s"], x.get("export"), x["zero_tick_excluded"]) for x in entry["candidates"]])
    report["coverage"] = {"targets": len(manifest["targets"]),
                          "returned": sum(1 for r in report["results"] if r["result"] == "returned"),
                          "by_result": dict(collections.Counter(r["result"] for r in report["results"]))}
    with open(report_path, "w") as f:
        json.dump(report, f, indent=1, ensure_ascii=False, default=str)
    print(report["coverage"])


if __name__ == "__main__":
    if sys.argv[1] == "--fix-targets":
        fix_targets(*sys.argv[2:4])
    else:
        search(*sys.argv[2:8])
