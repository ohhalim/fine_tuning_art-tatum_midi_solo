#!/usr/bin/env python3
"""Local piano harmony label audit (docs/experiments/LABEL_AUDIT.md, U2).

For every right-hand phrase (run of >= 8 top-line notes) in data/bebop_rh, label the
chord from the accompaniment struck at the phrase start (one vertical slice), and
compare with the window union the existing proxy uses. Writes per-phrase records,
a summary with the preregistered verdict inputs, and note dumps for the manual check.
"""
from __future__ import annotations

import argparse
import bisect
import collections
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "music_transformer", ROOT / "scripts"):
    sys.path.insert(0, str(p))

ACC_HIGH = 64            # accompaniment: notes off the top line below E4
CLUSTER_S = 0.06
LOOKBACK_S, LOOKAHEAD_S = 0.1, 0.3
SLICE_OFFSET_S = 0.03
WINDOW_S = 0.9375
SPAN_CAP_S = 8.0
MAIN = ("m7", "7", "maj7")
MIN_PHRASES, MIN_SONGS, MIN_PIANISTS = 30, 10, 3
MANUAL_PER_QUALITY, MANUAL_SEED = 8, 0


def title(source: str) -> str:
    t = re.sub(r"\.midi?$", "", source.rsplit("/", 1)[1])
    return re.split(r" - |\(", t)[0].strip().lower()


def clusters(acc):
    """Onset groups of accompaniment notes: [(onset, [notes])]."""
    out = []
    for n in acc:
        if out and n.start - out[-1][0] <= CLUSTER_S:
            out[-1][1].append(n)
        else:
            out.append((n.start, [n]))
    return out


class Sounding:
    """Accompaniment notes sounding at a time, by bisection on onsets (notes sorted by start)."""

    def __init__(self, acc):
        self.acc = acc
        self.starts = [n.start for n in acc]
        self.longest = max((n.end - n.start for n in acc), default=0.0)

    def __call__(self, t):
        lo = bisect.bisect_left(self.starts, t - self.longest)
        hi = bisect.bisect_right(self.starts, t)
        return [n for n in self.acc[lo:hi] if n.end > t]


def slice_at(sounding, onset):
    notes = sounding(onset + SLICE_OFFSET_S)
    return {n.pitch % 12 for n in notes}, (min(notes, key=lambda n: n.pitch).pitch if notes else None), notes


def phrase_slice(sounding, onsets, t0):
    """The accompaniment cluster for a phrase starting at t0: (source, cluster index).

    The latest cluster struck before t0 + LOOKBACK_S whose slice still sounds at t0;
    otherwise the first cluster in [t0, t0 + LOOKAHEAD_S]."""
    k = bisect.bisect_right(onsets, t0 + LOOKBACK_S) - 1
    if k >= 0 and any(n.end > t0 for n in sounding(onsets[k] + SLICE_OFFSET_S)):
        return "sounding", k
    k = bisect.bisect_left(onsets, t0)
    if k < len(onsets) and onsets[k] <= t0 + LOOKAHEAD_S:
        return "next_cluster", k
    return "none", None


def audit_song(path: str):
    import pretty_midi
    from inference.control.chord_label import classify
    from inference.control.harmony_contract import SOLO_SPLIT
    from inference.control.lick_bank import MIN_NOTES, runs
    from inference.control.solo_line import top_notes

    pm = pretty_midi.PrettyMIDI(path)
    notes = sorted((n for i in pm.instruments for n in i.notes), key=lambda n: (n.start, n.pitch))
    line = [n for n in top_notes(notes) if n.pitch >= SOLO_SPLIT]
    in_line = {(round(n.start, 4), n.pitch) for n in line}
    acc = [n for n in notes if (round(n.start, 4), n.pitch) not in in_line and n.pitch < ACC_HIGH]
    groups = clusters(acc)
    onsets = [g[0] for g in groups]
    sounding = Sounding(acc)
    records = []
    for run in runs(line):
        if len(run) < MIN_NOTES:
            continue
        t0 = run[0].start
        src, k = phrase_slice(sounding, onsets, t0)
        rec = {"t0": round(t0, 3), "run_notes": len(run), "slice_source": src}
        if k is None:
            rec.update(cls="unknown", label=None, candidates=[], pcs=[], bass=None, cluster_onset=None)
        else:
            pcs, bass, _ = slice_at(sounding, groups[k][0])
            cls, label, cands = classify(pcs, bass)
            rec.update(cls=cls, label=label, candidates=cands, pcs=sorted(pcs), bass=bass,
                       cluster_onset=round(groups[k][0], 3))
            # how long the label holds: the first later slice that contradicts it (another
            # label, or candidates that exclude it); unknown slices do not end it
            span_end = None
            for on, _ in groups[k + 1:]:
                if on - t0 > SPAN_CAP_S or label is None:
                    break
                c2 = classify(*slice_at(sounding, on)[:2])
                if (c2[1] is not None and c2[1] != label) or (c2[0] == "ambiguous" and label not in c2[2]):
                    span_end = on
                    break
            span_end = span_end if span_end is not None else (t0 if label is None else t0 + SPAN_CAP_S)
            rec["span_s"] = round(max(0.0, span_end - t0), 3)
            rec["span_rh_notes"] = sum(1 for n in run if t0 <= n.start < span_end)
        union = [n for n in acc if n.start < t0 + WINDOW_S and (n.start >= t0 or n.end > t0 + 0.05)]
        ucls, ulabel, _ = classify({n.pitch % 12 for n in union},
                                   min(union, key=lambda n: n.pitch).pitch if union else None)
        rec.update(union_cls=ucls, union_label=ulabel)
        here = sounding((rec["cluster_onset"] or t0) + SLICE_OFFSET_S)
        rec["acc_55_63"] = sum(1 for n in here if 55 <= n.pitch < 64)
        rec["acc_sounding"] = len(here)
        records.append(rec)
    return records, acc


def dump_case(acc, rec) -> str:
    import pretty_midi
    from inference.control.chord_label import symbol
    on = rec["cluster_onset"]
    near = [n for n in acc if on - 0.5 <= n.start <= on + 0.6 or n.start <= on + SLICE_OFFSET_S < n.end]
    lines = [f"{rec['pianist']} | {rec['title']} | t0 {rec['t0']} | slice {rec['slice_source']} @ {on} | "
             f"label {symbol(rec['label'])} ({rec['cls']}) | candidates {[symbol(c) for c in rec['candidates']]} | "
             f"union {symbol(rec['union_label'])} ({rec['union_cls']})"]
    for n in sorted(near, key=lambda n: (n.start, n.pitch)):
        mark = "*" if n.start <= on + SLICE_OFFSET_S < n.end else " "
        lines.append(f"   {mark} {pretty_midi.note_number_to_name(n.pitch):4s} {n.start - on:+.3f} .. {n.end - on:+.3f}")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args(argv)
    manifest = json.loads((ROOT / "data/bebop_rh/manifest.json").read_text())
    songs = [s for s in manifest["songs"] if s.get("status") == "ok"]
    records, accs = [], {}
    for s in songs:
        recs, acc = audit_song(s["source"])
        accs[s["source"]] = acc
        for r in recs:
            r.update(source=s["source"], pianist=s["pianist"], split=s["split"], title=title(s["source"]))
        records += recs
    n = len(records)
    labelled = [r for r in records if r["cls"] in ("clear", "bass_resolved")]

    def per_quality(rows):
        out = {}
        for q in sorted({r["label"][1] for r in rows if r["label"]}):
            rs = [r for r in rows if r["label"] and r["label"][1] == q]
            out[q] = {"phrases": len(rs), "songs": len({r["title"] for r in rs}),
                      "pianists": len({r["pianist"] for r in rs}),
                      "by_split": dict(collections.Counter(r["split"] for r in rs))}
        return out

    def share(key):
        return {k: round(v / n, 4) for k, v in collections.Counter(r[key] for r in records).items()}

    coverage = per_quality(labelled)
    passes = {q: bool(coverage.get(q) and coverage[q]["phrases"] >= MIN_PHRASES and coverage[q]["songs"] >= MIN_SONGS
                      and coverage[q]["pianists"] >= MIN_PIANISTS) for q in MAIN}
    spans = sorted(r["span_s"] for r in labelled)
    span_notes = sorted(r["span_rh_notes"] for r in labelled)
    q = lambda xs, f: xs[min(len(xs) - 1, int(f * len(xs)))] if xs else None
    summary = {
        "phrases": n, "songs": len(songs), "titles": len({title(s["source"]) for s in songs}),
        "slice_class_share": share("cls"), "union_class_share": share("union_cls"),
        "slice_source_share": share("slice_source"),
        "coverage_clear_or_bass_resolved": coverage,
        "coverage_clear_only": per_quality([r for r in records if r["cls"] == "clear"]),
        "coverage_by_pianist": dict(collections.Counter(r["pianist"] for r in labelled)),
        "union_vs_slice": {
            "union_labelled_slice_not": sum(1 for r in records if r["union_cls"] in ("clear", "bass_resolved")
                                            and r["cls"] not in ("clear", "bass_resolved")),
            "both_labelled_disagree": sum(1 for r in labelled if r["union_cls"] in ("clear", "bass_resolved")
                                          and r["union_label"] != r["label"]),
            "both_labelled_agree": sum(1 for r in labelled if r["union_label"] == r["label"]),
        },
        "acc_55_63_share_of_sounding": round(sum(r["acc_55_63"] for r in records) / max(1, sum(r["acc_sounding"] for r in records)), 4),
        "span_s_p25_p50_p75": [q(spans, .25), q(spans, .5), q(spans, .75)],
        "span_rh_notes_p25_p50_p75": [q(span_notes, .25), q(span_notes, .5), q(span_notes, .75)],
        "coverage_passes": passes,
        "fields": {"root": "estimated (clear, bass_resolved) or unknown", "quality": "estimated or unknown",
                   "next_chord": "unknown", "beat": "unknown", "time_to_change": "unknown"},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "records.json").write_text(json.dumps(records) + "\n")
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    rng = random.Random(MANUAL_SEED)
    dumps = []
    for qual in MAIN:
        pool = [r for r in labelled if r["label"][1] == qual]
        for r in rng.sample(pool, min(MANUAL_PER_QUALITY, len(pool))):
            dumps.append(dump_case(accs[r["source"]], r))
    (args.output_dir / "manual_check.txt").write_text("\n\n".join(dumps) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
