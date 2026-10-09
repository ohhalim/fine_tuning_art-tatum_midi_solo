#!/usr/bin/env python3
"""WJazzD pilot data: song split, valid-note runs, chunks, masks, v2 features (docs/experiments/WJAZZD_PILOT.md).

Rules are fixed in the preregistration. Writes the chunk data locally (token ids, 29-dim condition
rows, loss masks, strata) and an aggregate summary for the repository. Run with the Aria venv:
wjazzd_pilot_data.py <aria repo> <wjazzd.db> <local out json> <summary json>
"""
from __future__ import annotations

import bisect
import collections
import json
import os
import random
import sqlite3
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
from aria_cond_contract_v2 import CAP_S, features  # noqa: E402
from aria_t4b_sim import write_notes  # noqa: E402
from inference.control.chord_label import QUALITIES  # noqa: E402
from wjazzd_audit import parse_chord  # noqa: E402

SEED, SPLITS = 20261009, (("train", 24), ("validation", 8), ("test", 8))
MAX_NOTES, MIN_NOTES, PREFIX_NOTES, MAX_TOKENS, VEL = 150, 32, 16, 512, 80


def status(s):
    if s is None:
        return "unknown"
    root, q, lossy, _, _ = parse_chord(s)
    if q == "N.C.":
        return "nc"
    if q is None:
        return "unmapped"
    return "lossy" if lossy else "exact"


def key(s):
    root, q, *_ = parse_chord(s)
    return "N.C." if q == "N.C." else (root, q) if q else ("unmapped", s)


def chroma(value):
    """value = (key, status) or None; family chord tones only."""
    if value is None or not isinstance(value[0], tuple) or value[0][0] == "unmapped":
        return [0.0] * 12
    root, q = value[0]
    pcs = {(root + i) % 12 for i in QUALITIES[q][0]}
    return [1.0 if k in pcs else 0.0 for k in range(12)]


def row(f):
    cur, nxt = f["current"], f["next"]
    return chroma(cur) + chroma(nxt) + [float(f["current_known"]), float(f["has_next"]), float(f["next_known"]),
                                        f["delta_norm"], float(f["delta_clamped"])]


def plan_for(c, melid, t0):
    """Chord plan of a whole solo in the token timeline of a chunk starting at t0 (seconds)."""
    notes_end = c.execute("select max(onset + duration) from melody where melid=?", (melid,)).fetchone()[0] + 1.0
    changes = c.execute("select onset, chord from beats where melid=? and chord!='' order by onset", (melid,)).fetchall()
    plan = []
    for j, (t, s) in enumerate(changes):
        tn = changes[j + 1][0] if j + 1 < len(changes) else notes_end
        plan.append(((t - t0) * 1000.0, (tn - t0) * 1000.0, (key(s), status(s))))
    return plan


def main() -> None:
    aria_repo, db, local_out, summary_out = sys.argv[1:5]
    sys.path.insert(0, aria_repo)
    from ariautils.midi import MidiDict
    from ariautils.tokenizer import AbsTokenizer
    tok = AbsTokenizer()
    c = sqlite3.connect(db)
    solos = c.execute("""select s.melid, s.compid, s.instrument, s.signature, coalesce(ci.template, ''), s.performer, s.title
                         from solo_info s left join composition_info ci on ci.compid = s.compid""").fetchall()
    elig = [r for r in solos if r[4] not in ("Blues", "I Got Rhythm") and r[3] == "4/4" and r[2] != "p"]
    by_comp = collections.defaultdict(list)
    for r in elig:
        by_comp[r[1]].append(r)
    rng = random.Random(SEED)
    comps = sorted(by_comp)
    rng.shuffle(comps)
    picks = [rng.choice(sorted(by_comp[k])) for k in comps]

    chunks, songs, skipped = [], [], collections.Counter()
    exclusions = collections.Counter()
    excl_by_suffix = collections.Counter()
    need = sum(n for _, n in SPLITS)
    for melid, compid, instr, sig, tmpl, performer, title in picks:
        if len(songs) == need:
            break
        notes = c.execute("select onset, pitch, duration, bar, beat from melody where melid=? order by eventid", (melid,)).fetchall()
        changes = c.execute("select onset, chord from beats where melid=? and chord!='' order by onset", (melid,)).fetchall()
        beat_chord, cur_c = {}, None
        for bar, beat, ch in c.execute("select bar, beat, chord from beats where melid=? order by onset", (melid,)):
            cur_c = ch or cur_c
            beat_chord[(bar, beat)] = cur_c
        times = [t for t, _ in changes]
        at = lambda t: changes[bisect.bisect_right(times, t) - 1][1] if bisect.bisect_right(times, t) > 0 else None

        def nxt_after(t, cur):
            j = bisect.bisect_right(times, t)
            for tt, s in changes[j:]:
                if cur is None or (key(s), status(s)) != (key(cur), status(cur)):
                    return s
            return None

        # valid notes (approximate prefix time = previous onset)
        valid = []
        prev = None
        for i, (onset, pitch, dur, bar, beat) in enumerate(notes):
            pt = onset if prev is None else prev
            cur, on = at(pt), at(onset)
            nx = nxt_after(pt, cur)
            reasons = [f"current_{status(cur)}"] if status(cur) != "exact" else []
            if nx is not None and status(nx) != "exact":
                reasons.append(f"next_{status(nx)}")
            if status(on) != "exact":
                reasons.append(f"onset_{status(on)}")
            valid.append(not reasons)
            for rsn in reasons[:1]:
                exclusions[rsn] += 1
                bad = cur if rsn.startswith("current") else nx if rsn.startswith("next") else on
                excl_by_suffix[(bad or "None")] += 1
            prev = onset
        runs, start = [], None
        for i, v in enumerate(valid + [False]):
            if v and start is None:
                start = i
            elif not v and start is not None:
                runs.append((start, i))
                start = None
        song_chunks = []
        for a, b in runs:
            for s0 in range(a, b, MAX_NOTES):
                s1 = min(b, s0 + MAX_NOTES)
                if s1 - s0 < MIN_NOTES:
                    skipped["chunk_short"] += 1
                    continue
                while True:
                    t0 = notes[s0][0]
                    seg = [(int(p), o - t0, o - t0 + d, VEL) for o, p, d, _, _ in notes[s0:s1]]
                    with tempfile.TemporaryDirectory() as d:
                        path = os.path.join(d, "x.mid")
                        write_notes(seg, path)
                        toks = tok.tokenize(MidiDict.from_midi(path))
                    if len(toks) <= MAX_TOKENS or s1 - s0 <= MIN_NOTES:
                        break
                    s1 -= 1
                if len(toks) > MAX_TOKENS:
                    skipped["chunk_too_long"] += 1
                    continue
                end = max(o + d for o, _, d, _, _ in notes) + 1.0
                plan = []
                for j, (t, s) in enumerate(changes):
                    tn = changes[j + 1][0] if j + 1 < len(changes) else end
                    plan.append(((t - t0) * 1000.0, (tn - t0) * 1000.0, (key(s), status(s))))
                feats = features(toks, plan)
                bad = [i for i, f in enumerate(feats) if f["current"] is None or f["current"][1] != "exact"
                       or (f["next"] is not None and f["next"][1] != "exact")]
                if bad:
                    skipped["chunk_condition_not_exact_after_tokenize"] += 1
                    continue
                piano_idx = [i for i, t in enumerate(toks) if isinstance(t, tuple) and t[0] == "piano"]
                first_target = piano_idx[PREFIX_NOTES]
                mask = [(k + 1) >= first_target and toks[k + 1] != "<E>" for k in range(len(toks) - 1)]
                strata = []
                for n_i, i in enumerate(piano_idx):
                    if i < first_target:
                        continue
                    onset_t = feats[i + 1]["t_ms"]
                    on = next((p[2] for p in plan if p[0] <= onset_t < p[1]), None)
                    cat = "current" if on == feats[i]["current"] else "boundary_first" if on == feats[i]["next"] else "not_represented"
                    o, _, _, bar, beat = notes[s0 + n_i]
                    metric = beat_chord.get((bar, beat))
                    strata.append({"pos": i, "stratum": cat,
                                   "anticipation": metric is not None and key(metric) != key(at(o))})
                song_chunks.append({"melid": melid, "t0": t0, "s0": s0, "notes": s1 - s0, "token_ids": tok.encode(toks),
                                    "cond": [row(f) for f in feats], "mask": mask, "first_target": first_target,
                                    "targets": int(sum(mask)), "pitch_targets": strata})
        if not song_chunks:
            skipped["song_without_valid_chunk"] += 1
            continue
        split = next(name for name, lim in ((n, sum(x for _, x in SPLITS[: k + 1])) for k, (n, _) in enumerate(SPLITS)) if len(songs) < lim)
        songs.append({"melid": melid, "compid": compid, "instrument": instr, "performer": performer, "title": title,
                      "split": split, "chunks": len(song_chunks), "notes": len(notes),
                      "valid_notes": sum(valid), "targets": sum(ch["targets"] for ch in song_chunks)})
        for ch in song_chunks:
            ch["split"] = split
        chunks += song_chunks
    with open(local_out, "w") as f:
        json.dump({"songs": songs, "chunks": chunks}, f)
    strata = collections.Counter((ch["split"], p["stratum"]) for ch in chunks for p in ch["pitch_targets"])
    antic = collections.Counter(ch["split"] for ch in chunks for p in ch["pitch_targets"] if p["anticipation"])
    summary = {"seed": SEED, "candidates": {"compositions": len(by_comp), "solos": len(elig)},
               "songs_taken": len(songs), "need": need, "skipped": dict(skipped),
               "songs": songs,
               "by_split": {s: {"songs": sum(1 for x in songs if x["split"] == s),
                                "chunks": sum(1 for ch in chunks if ch["split"] == s),
                                "targets": sum(ch["targets"] for ch in chunks if ch["split"] == s),
                                "pitch_targets": sum(len(ch["pitch_targets"]) for ch in chunks if ch["split"] == s),
                                "strata": {k[1]: v for k, v in strata.items() if k[0] == s},
                                "anticipation": antic.get(s, 0)} for s, _ in SPLITS},
               "note_exclusions_first_reason": dict(exclusions),
               "excluded_chord_strings_top": [[k, v] for k, v in excl_by_suffix.most_common(15)],
               "cap_s": CAP_S}
    with open(summary_out, "w") as f:
        json.dump(summary, f, indent=1, ensure_ascii=False)
    print(json.dumps({k: summary[k] for k in ("songs_taken", "need", "skipped", "by_split", "note_exclusions_first_reason")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
