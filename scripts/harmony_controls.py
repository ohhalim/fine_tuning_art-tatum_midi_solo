#!/usr/bin/env python3
"""Which harmony metric catches which error? Control response table (HARMONY_CONTRACT.md U3, #1631).

Real bebop windows (solo top line vs the accompaniment proxy of the same window),
each metric computed on the original and on paired controls:

  joint_transpose  solo and harmony moved together      -> metrics must not change
  harmony_swap     harmony from another song's window   -> fit should fall
  order_shuffle    solo pitches permuted, rhythm kept   -> only order-aware metrics should fall
  partial_shift    30% of solo notes moved one semitone -> moderate fall
  solo_shift       whole solo moved one semitone        -> stress control, not "always wrong"

Metrics (reported separately, never merged into one score):
  fit         duration-weighted share of solo notes whose pitch class is in the harmony
  clash       duration-weighted share a semitone from the nearest harmony pitch class
  resolution  share of non-harmony notes followed, within 2 notes and 0.4 s, by a harmony
              note at most 2 semitones away (step resolution or enclosure)
  rel_js      Jensen-Shannon distance of the (pitch class - bass) histogram to the train
              reference, pooled per condition
No beat grid exists in these transcriptions, so nothing is weighted by beat position.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "music_transformer", ROOT / "scripts"):
    sys.path.insert(0, str(p))

WINDOW_S = 0.9375
FIRST_S, STEP_S, MAX_PER_SONG = 2.0, 4.0, 20
MIN_SOLO, MIN_PCS = 3, 2
RESOLVE_NOTES, RESOLVE_S, RESOLVE_STEP = 2, 0.4, 2


def pc_dist(p: int, pcs) -> int:
    return min(min((p - q) % 12, (q - p) % 12) for q in pcs)


def window_metrics(solo, pcs) -> dict:
    """solo: list of (pitch, start, end); pcs: harmony pitch classes."""
    dur = [max(e - s, 0.01) for _, s, e in solo]
    tot = sum(dur)
    fit = sum(d for (p, _, _), d in zip(solo, dur) if p % 12 in pcs) / tot
    clash = sum(d for (p, _, _), d in zip(solo, dur) if pc_dist(p, pcs) == 1) / tot
    non, res = 0, 0
    for i, (p, s, e) in enumerate(solo):
        if p % 12 in pcs:
            continue
        non += 1
        for q, qs, _ in solo[i + 1:i + 1 + RESOLVE_NOTES]:
            if qs - s > RESOLVE_S:
                break
            if q % 12 in pcs and abs(q - p) <= RESOLVE_STEP:
                res += 1
                break
    return {"fit": fit, "clash": clash, "resolution": res / non if non else None, "non": non}


def rel_hist(solo, bass) -> list[float]:
    h = [0.0] * 12
    for p, s, e in solo:
        h[(p - bass) % 12] += max(e - s, 0.01)
    return h


def js_distance(a, b) -> float:
    sa, sb = sum(a), sum(b)
    a = [x / sa for x in a]
    b = [x / sb for x in b]
    m = [(x + y) / 2 for x, y in zip(a, b)]
    kl = lambda x, y: sum(xi * math.log(xi / yi) for xi, yi in zip(x, y) if xi > 0)
    return math.sqrt(max(0.0, (kl(a, m) + kl(b, m)) / 2))


def song_windows(path: str) -> list[dict]:
    import pretty_midi
    from inference.control.harmony_contract import SOLO_SPLIT, accompaniment_proxy, solo_window
    from inference.control.solo_line import top_notes

    pm = pretty_midi.PrettyMIDI(path)
    notes = sorted((n for i in pm.instruments for n in i.notes), key=lambda n: (n.start, n.pitch))
    line = [n for n in top_notes(notes) if n.pitch >= SOLO_SPLIT]
    out, t, end = [], FIRST_S, pm.get_end_time()
    while t + WINDOW_S <= end and len(out) < MAX_PER_SONG:
        bass, pcs, st = accompaniment_proxy(notes, t, t + WINDOW_S)
        solo = solo_window(line, t, t + WINDOW_S)
        if not st["empty"] and len(pcs) >= MIN_PCS and len(solo) >= MIN_SOLO:
            out.append({"bass": bass, "pcs": set(pcs), "solo": [(n.pitch, n.start, n.end) for n in solo]})
        t += STEP_S
    return out


def controls(w: dict, donor: dict, rng: random.Random) -> dict:
    solo, pcs, bass = w["solo"], w["pcs"], w["bass"]
    k = rng.randrange(1, 12)
    pitches = [p for p, _, _ in solo]
    perm = pitches[:]
    for _ in range(10):                                  # a permutation that changes the order
        rng.shuffle(perm)
        if perm != pitches or len(set(pitches)) == 1:
            break
    idx = set(rng.sample(range(len(solo)), max(1, round(0.3 * len(solo)))))
    return {
        "original": (solo, pcs, bass),
        "joint_transpose": ([(p + k, s, e) for p, s, e in solo], {(q + k) % 12 for q in pcs}, (bass + k) % 12),
        "harmony_swap": (solo, donor["pcs"], donor["bass"]),
        "order_shuffle": ([(q, s, e) for q, (_, s, e) in zip(perm, solo)], pcs, bass),
        "partial_shift": ([(p + rng.choice((-1, 1)) if i in idx else p, s, e) for i, (p, s, e) in enumerate(solo)],
                          pcs, bass),
        "solo_shift": ([(p + 1, s, e) for p, s, e in solo], pcs, bass),
    }


def table(windows_by_song: list[list[dict]], reference: list[float] | None, seed: int = 0) -> dict:
    rng = random.Random(seed)
    used = [ws for ws in windows_by_song if ws]
    rows = {}
    hists = {}
    for si, ws in enumerate(used):
        donor_song = used[(si + 7) % len(used)]
        for wi, w in enumerate(ws):
            for cond, (solo, pcs, bass) in controls(w, donor_song[wi % len(donor_song)], rng).items():
                m = window_metrics(solo, pcs)
                rows.setdefault(cond, []).append(m)
                h = hists.setdefault(cond, [0.0] * 12)
                for i, x in enumerate(rel_hist(solo, bass)):
                    h[i] += x
    orig = rows["original"]
    out = {}
    for cond, rs in rows.items():
        entry = {"windows": len(rs)}
        for metric in ("fit", "clash", "resolution"):
            pairs = [(r[metric], o[metric]) for r, o in zip(rs, orig) if r[metric] is not None and o[metric] is not None]
            vals = [a for a, _ in pairs]
            entry[metric] = sum(vals) / len(vals) if vals else None
            if cond != "original" and pairs:
                entry[f"{metric}_lower_than_original"] = sum(a < b - 1e-9 for a, b in pairs) / len(pairs)
                entry[f"{metric}_higher_than_original"] = sum(a > b + 1e-9 for a, b in pairs) / len(pairs)
        if reference is not None:
            entry["rel_js"] = js_distance(hists[cond], reference)
        out[cond] = entry
    return {"table": out, "original_hist": hists["original"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--train-songs", type=int, default=60, help="train songs used (first N by manifest order)")
    args = ap.parse_args(argv)
    manifest = json.loads((ROOT / "data/bebop_rh/manifest.json").read_text())
    ok = [s for s in manifest["songs"] if s.get("status") == "ok"]
    train = [song_windows(s["source"]) for s in [s for s in ok if s["split"] == "train"][:args.train_songs]]
    val = [song_windows(s["source"]) for s in ok if s["split"] == "val"]
    ref = table(train, None)["original_hist"]
    res = {"schema": "harmony_controls_v1", "window_s": WINDOW_S,
           "train": table(train, ref)["table"], "val": table(val, ref)["table"],
           "note": "train/val only; test untouched. val was seen in earlier exploration.",
           "musical_quality_verified": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(res, indent=2) + "\n")
    for split in ("train", "val"):
        print(split)
        for cond, e in res[split].items():
            print(f"  {cond:16s}", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in e.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
