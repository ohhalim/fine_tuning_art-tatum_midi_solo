#!/usr/bin/env python3
"""Phrase plan + harmonic realization, offline diagnosis (docs/experiments/PHRASE_REALIZER.md).

Same chords, same comp (guide: root + 3rd + 7th under the solo), same piano, four solos:
  raw        the current runtime solo (N1 + carry + avoid bias + repeat 0.5, #1662 confirm seeds)
  model_re   that solo's abstract phrase (rhythm, rests, interval contour) realized on the chords
  bebop_re   a real bebop right-hand line's abstract phrase realized (diagnostic oracle)
  tatum_re   a real Art Tatum top line's abstract phrase realized (diagnostic oracle)
Reports MIDI-readable defects against the comp, repetition / leaps, and how much of
the source contour survives. No success threshold replaces listening.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "music_transformer", ROOT / "scripts"):
    sys.path.insert(0, str(p))

BPM, BARS = 128, 16
PROGRESSIONS = {"iiVIF": ["Gm7", "C7", "Fmaj7", "Fmaj7"], "rhythmA": ["Bbmaj7", "G7", "Cm7", "F7"],
                "minorA": ["Bm7b5", "E7", "Am7", "Am7"]}
SOURCE_START_S = 20.0


def chord_fn(chords):
    from inference.app.fallback import parse_chord
    from inference.control.harmony_bias import SCALE

    bar = 60.0 / BPM * 4

    def at(t):
        root, iv = parse_chord(chords[int(t // bar) % len(chords)])
        return {(root + k) % 12 for k in iv}, {(root + k) % 12 for k in SCALE.get(tuple(iv), set(iv))}
    return at


def comp_notes(chords):
    from inference.control.comping import guide_half
    half = 60.0 / BPM * 2
    out, state = [], {}
    for block in range(BARS * 2):
        chord = chords[(block // 2) % len(chords)]
        notes, _ = guide_half(chord, block=block, bpm=BPM, state=state)
        out += [(p, block * half + s, block * half + e, v) for p, s, e, v in notes]
    return out


def top_line(path: str, seconds: float):
    import pretty_midi
    from inference.control.solo_line import top_notes
    pm = pretty_midi.PrettyMIDI(path)
    line = [n for n in top_notes([n for i in pm.instruments for n in i.notes]) if n.pitch >= 55]
    return [pretty_midi.Note(velocity=n.velocity, pitch=n.pitch, start=n.start - SOURCE_START_S,
                             end=n.end - SOURCE_START_S) for n in line
            if SOURCE_START_S <= n.start < SOURCE_START_S + seconds]


def raw_solo(tag: str):
    import pretty_midi
    from scripts.comp_source_check import tag_run
    d = sorted(glob.glob(str(ROOT / f"outputs/n1_bias_sweep/confirm/R0.5/bebop_{tag}_s44")))[0]
    r = json.loads(Path(d, "continuous_report.json").read_text())
    return [pretty_midi.Note(velocity=90, pitch=p, start=s, end=e) for p, s, e in sorted(tag_run(r)["solo"], key=lambda x: x[1])]


def defects(solo, comp, chord_at) -> dict:
    harsh = under = outside = rep = leap = steps = 0
    for i, (p, s, e) in enumerate(solo):
        ov = [q for q, qs, qe in comp if min(e, qe) - max(s, qs) > 0.02]
        harsh += any(abs(p - q) in (1, 11, 13) for q in ov)
        under += any(q >= p for q in ov)
        tones, scale = chord_at(s)
        if p % 12 not in scale:
            nxt = solo[i + 1] if i + 1 < len(solo) else None
            ok = nxt is not None and nxt[1] - e <= 0.3 and abs(nxt[0] - p) <= 2 and nxt[0] % 12 in chord_at(nxt[1])[0]
            outside += not ok
        if i and s - solo[i - 1][2] <= 0.3:
            steps += 1
            rep += p == solo[i - 1][0]
            leap += abs(p - solo[i - 1][0]) > 12
    n = max(1, len(solo))
    return {"notes": len(solo), "harsh": harsh / n, "under": under / n, "outside": outside / n,
            "repeat": rep / max(1, steps), "leap": leap / max(1, steps)}


def contour_kept(src_events, realized) -> float | None:
    agree = total = 0
    for ev, (a, b) in zip(src_events[1:], zip(realized, realized[1:])):
        if ev["interval"] is None:
            continue
        total += 1
        sign = lambda x: (x > 0) - (x < 0)
        agree += sign(ev["interval"]) == sign(b[0] - a[0])
    return agree / total if total else None


def main(argv=None) -> int:
    import pretty_midi
    from inference.control.realizer import abstract, realize

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args(argv)
    seconds = BARS * 4 * 60.0 / BPM
    manifest = json.loads((ROOT / "data/bebop_rh/manifest.json").read_text())
    val = [s["source"] for s in manifest["songs"] if s.get("split") == "val" and s.get("status") == "ok"]
    tatum = sorted(glob.glob("/Users/ohhalim/git_box/fine_tuning_art-tatum_midi_solo/midi_dataset/midi/studio/Art Tatum/**/*.mid*",
                             recursive=True))
    report = {}
    for k, (tag, chords) in enumerate(PROGRESSIONS.items()):
        at = chord_fn(chords)
        comp = comp_notes(chords)
        comp3 = [(p, s, e) for p, s, e, _ in comp]
        sources = {"raw": raw_solo(tag), "bebop": top_line(val[k * 7 % len(val)], seconds),
                   "tatum": top_line(tatum[k * 5 % len(tatum)], seconds)}
        variants = {"raw": [(n.pitch, n.start, n.end, n.velocity) for n in sources["raw"]]}
        kept = {}
        for name, src in (("model_re", sources["raw"]), ("bebop_re", sources["bebop"]), ("tatum_re", sources["tatum"])):
            ev = abstract(src)
            variants[name] = realize(ev, chord_at=at, bpm=BPM)
            kept[name] = contour_kept(ev, variants[name])
        report[tag] = {}
        for name, notes in variants.items():
            solo3 = [(p, s, e) for p, s, e, _ in notes]
            report[tag][name] = {**defects(solo3, comp3, at), "contour_kept": kept.get(name),
                                 "notes_per_s": len(notes) / seconds}
            pm = pretty_midi.PrettyMIDI(initial_tempo=BPM)
            si, ci = pretty_midi.Instrument(program=0, name="solo"), pretty_midi.Instrument(program=0, name="comp")
            si.notes = [pretty_midi.Note(velocity=int(v), pitch=int(p), start=float(s), end=float(e)) for p, s, e, v in notes]
            ci.notes = [pretty_midi.Note(velocity=int(v), pitch=int(p), start=float(s), end=float(e)) for p, s, e, v in comp]
            pm.instruments += [si, ci]
            args.output_dir.mkdir(parents=True, exist_ok=True)
            pm.write(str(args.output_dir / f"{tag}_{name}.mid"))
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    for tag, rows in report.items():
        print(tag)
        for name, m in rows.items():
            print(f"  {name:9s}", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
