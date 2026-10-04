#!/usr/bin/env python3
"""C1: same solo, guide comp revoiced only where it clashes (docs/experiments/C1_COMP_AB.md).

Reads the T1 clips the user rated (one track: solo >= C4, comp <= B3), revoices each comp
strike with ``comping.revoice_strike`` and checks that everything but the upper comp
pitches is unchanged. Writes the C1 MIDI, a per-strike report and before/after counts.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BPM = 128
BAR = 60.0 / BPM * 4
CHORDS = ["Gm7", "C7", "Fmaj7", "Fmaj7"]
SPLIT = 60          # solo >= C4, comp <= B3
UPPER_LOW = 48      # comp C3-B3: guide pair; below: bass root


def load(path):
    import pretty_midi
    pm = pretty_midi.PrettyMIDI(str(path))
    assert len(pm.instruments) == 1, "expected one track"
    notes = sorted(pm.instruments[0].notes, key=lambda n: (n.start, n.pitch))
    solo = [n for n in notes if n.pitch >= SPLIT]
    comp = [n for n in notes if n.pitch < SPLIT]
    return pm, solo, comp


def bar_of(pm, t: float, beats_per_bar: int = 4) -> int:
    """Bar index from the event's MIDI tick, not its read-back seconds: PrettyMIDI returns a
    strike written on a bar line ~20 us early, and a seconds tolerance would also pull a
    deliberate anticipation just before the line into the next bar (Astra on #1678)."""
    tempos = pm.get_tempo_changes()[1]
    assert len(tempos) == 1, "one tempo expected"
    return int(pm.time_to_tick(t)) // (pm.resolution * beats_per_bar)


def strikes(comp):
    groups = {}
    for n in comp:
        groups.setdefault(round(n.start, 6), []).append(n)
    return [groups[k] for k in sorted(groups)]


def counts(solo3, comp_notes):
    from inference.control.comping import solo_clashes
    n = dur = maj7 = 0
    for c in comp_notes:
        hits = solo_clashes([c.pitch], solo3, c.start, c.end)
        n += len(hits)
        dur += sum(h[2] for h in hits)
        maj7 += sum(1 for p, s, e in solo3 if min(e, c.end) - max(s, c.start) > 0.02 and p > c.pitch and (p - c.pitch) % 12 == 11)
    return {"minor9_pairs": n, "minor9_overlap_s": round(dur, 3), "major7_pairs": maj7}


def main(argv=None) -> int:
    import pretty_midi
    from inference.control.comping import revoice_strike

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clip", nargs=2, action="append", metavar=("TAG", "MIDI"), required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {}
    for tag, path in args.clip:
        pm, solo, comp = load(path)
        assert max((n.pitch for n in comp), default=0) < min(n.pitch for n in solo), "solo and comp ranges overlap"
        solo3 = [(n.pitch, n.start, n.end) for n in solo]
        new_comp, rows = [], []
        for strike in strikes(comp):
            bass = [n for n in strike if n.pitch < UPPER_LOW]
            upper = sorted((n for n in strike if n.pitch >= UPPER_LOW), key=lambda n: n.pitch)
            t0, t1 = strike[0].start, max(n.end for n in strike)
            assert all(abs(n.start - t0) < 1e-6 and abs(n.end - upper[0].end) < 1e-6 for n in upper), "upper notes differ in time"
            bar = bar_of(pm, t0)
            chord = CHORDS[bar % len(CHORDS)]
            pitches, info = revoice_strike(chord, tuple(n.pitch for n in upper), solo3, upper[0].start, upper[0].end)
            new_comp += [pretty_midi.Note(velocity=n.velocity, pitch=n.pitch, start=n.start, end=n.end) for n in bass]
            new_comp += [pretty_midi.Note(velocity=n.velocity, pitch=p, start=n.start, end=n.end) for n, p in zip(upper, pitches)]
            rows.append({"t0": round(t0, 3), "bar": bar + 1, "chord": chord,
                         "before": [n.pitch for n in upper], "after": list(pitches),
                         **{k: v for k, v in info.items() if k != "before"},
                         "clashes_before": [(p, q, round(d, 3)) for p, q, d in info["before"]]})
        # invariance: solo untouched; comp strikes identical except upper pitches
        key = lambda n: (round(n.start, 6), round(n.end, 6), n.velocity)
        assert sorted(map(key, comp)) == sorted(map(key, new_comp)), "comp timing/velocity changed"
        assert len(comp) == len(new_comp)
        out = pretty_midi.PrettyMIDI(initial_tempo=BPM)
        inst = pretty_midi.Instrument(program=pm.instruments[0].program)
        inst.notes = [pretty_midi.Note(velocity=n.velocity, pitch=n.pitch, start=n.start, end=n.end) for n in solo] + new_comp
        out.instruments.append(inst)
        out.write(str(args.output_dir / f"{tag}_c1.mid"))
        check = load(args.output_dir / f"{tag}_c1.mid")[1]
        assert [(n.pitch, round(n.start, 4), round(n.end, 4), n.velocity) for n in check] == \
               [(n.pitch, round(n.start, 4), round(n.end, 4), n.velocity) for n in solo], "solo changed"
        status = {}
        for r in rows:
            status[r["status"]] = status.get(r["status"], 0) + 1
        report[tag] = {"source": str(path), "strikes": len(rows), "status": status,
                       "before": counts(solo3, comp), "after": counts(solo3, new_comp),
                       "changes": [r for r in rows if r["status"] != "clear"]}
        print(tag, json.dumps({k: v for k, v in report[tag].items() if k != "changes"}))
        for r in report[tag]["changes"]:
            print("   bar", r["bar"], r["chord"], r["before"], "->", r["after"], r["status"],
                  r.get("voicing", ""), "omitted", r.get("omitted", ""), "solo fills", r.get("solo_fills_omitted", ""),
                  "clash", r["clashes_before"])
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
