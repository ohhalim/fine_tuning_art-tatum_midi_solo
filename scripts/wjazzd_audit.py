#!/usr/bin/env python3
"""WJazzD inspection before any training decision (docs/experiments/WJAZZD_AUDIT.md).

Read-only over the local wjazzd.db. Writes aggregate statistics only (no melodies, no chord
sequences of individual solos): provenance and license, inventory, chord-label mapping to the
project's quality families, chord–note alignment between the two chord sources of the database,
monophonic representation (the sections table turned out to equal the beat timeline for every
note, so the independent check is the note's own bar/beat annotation against the beat track),
and the v2 condition strata (chord at each note's onset vs current /
first next chord at the prefix time). Usage: wjazzd_audit.py <wjazzd.db> <out json>
"""
from __future__ import annotations

import bisect
import collections
import json
import re
import sqlite3
import statistics
import sys

ROOT = re.compile(r"^([A-G][b#]?)(.*)$")
NAMES = {l + a: (pc + d) % 12 for l, pc in zip("CDEFGAB", (0, 2, 4, 5, 7, 9, 11)) for a, d in (("", 0), ("#", 1), ("b", -1))}


def parse_chord(s: str):
    """WJazzD chord string -> (root pc, quality family or None, lossy, bass pc or None, note).
    Rules (fixed): '-' minor, 'j' major seventh, 'o' diminished, '+' augmented, 'sus' suspended,
    'm7b5' half-diminished, digits after the seventh/sixth are tensions (lossy), '/X' slash bass
    (lossy). No family for dim/aug triads, sus chords or aug-major: left unmapped, never filled."""
    if s == "NC":
        return None, "N.C.", False, None, "no chord"
    body, bass = (s.split("/", 1) + [None])[:2]
    m = ROOT.match(body)
    if not m:
        return None, None, False, None, f"unparsed {s!r}"
    root, suf = NAMES[m.group(1)], m.group(2)
    bass_pc = NAMES.get(bass) if bass else None
    if bass and bass_pc is None:
        return root, None, True, None, f"unparsed bass {bass!r}"
    lossy = bass is not None
    rules = [("m7b5", "m7b5"), ("-j7", "mMaj7"), ("-7", "m7"), ("-6", "m6"), ("-", "min"), ("j7", "maj7"),
             ("o7", "dim7"), ("+7", "7"), ("7", "7"), ("6", "6")]
    if suf.startswith("sus") or suf.startswith("+j") or suf in ("+", "o") or (suf.startswith("o") and not suf.startswith("o7")):
        return root, None, True, bass_pc, f"no family for {suf!r}"
    for prefix, q in rules:
        if suf.startswith(prefix):
            rest = suf[len(prefix):]
            if q == "min" and rest:                      # '-9' style with no seventh: not in the families
                return root, None, True, bass_pc, f"no family for {suf!r}"
            return root, q, lossy or bool(rest) or prefix == "+7", bass_pc, ""
    if suf == "":
        return root, "maj", lossy, bass_pc, ""
    if suf.startswith("9") or suf.startswith("69"):
        return root, None, True, bass_pc, f"no family for {suf!r}"
    return root, None, True, bass_pc, f"no family for {suf!r}"


def main() -> None:
    db, out = sys.argv[1:3]
    c = sqlite3.connect(db)
    info = c.execute("select name, creator, major, minor, release, created, license, status from db_info").fetchone()
    solos = c.execute("select melid, compid, instrument, style, signature, avgtempo, performer from solo_info").fetchall()
    comps = {r[0]: {"template": r[1], "form": r[2]} for r in c.execute("select compid, template, form from composition_info")}
    rep = {"db_info": dict(zip(["name", "creator", "major", "minor", "release", "created", "license", "status"], info)),
           "solos": len(solos)}
    rep["instruments"] = collections.Counter(s[2] for s in solos).most_common()
    rep["styles"] = collections.Counter(s[3] for s in solos).most_common()
    rep["signatures"] = collections.Counter(s[4] for s in solos).most_common()
    per_comp = collections.Counter(s[1] for s in solos)
    rep["compositions_with_solos"] = len(per_comp)
    rep["solos_per_composition"] = sorted(collections.Counter(per_comp.values()).items())
    rep["max_solos_one_composition"] = max(per_comp.values())
    tmpl = collections.Counter(comps[k]["template"] or "(none)" for k in per_comp)
    rep["templates_over_compositions"] = tmpl.most_common()
    rep["solos_in_template_groups"] = {t: sum(n for k, n in per_comp.items() if (comps[k]["template"] or "(none)") == t)
                                       for t, _ in tmpl.most_common(6)}

    # chord labels at change points
    changes = c.execute("select melid, onset, chord from beats where chord != '' order by melid, onset").fetchall()
    label_stats = collections.Counter()
    unmapped = collections.Counter()
    parsed = {}
    for _, _, s in changes:
        if s not in parsed:
            parsed[s] = parse_chord(s)
        root, q, lossy, bass, note = parsed[s]
        if q == "N.C.":
            label_stats["no_chord"] += 1
        elif q is None:
            label_stats["unmapped"] += 1
            unmapped[note] += 1
        elif lossy:
            label_stats["mapped_lossy"] += 1
        else:
            label_stats["mapped_exact"] += 1
    rep["chord_change_events"] = len(changes)
    rep["distinct_chord_strings"] = len(parsed)
    rep["chord_label_stats"] = dict(label_stats)
    rep["unmapped_top"] = unmapped.most_common(15)

    timeline = collections.defaultdict(list)
    for melid, onset, s in changes:
        timeline[melid].append((onset, s))

    def key(s):
        root, q, *_ = parsed[s]
        return "N.C." if q == "N.C." else (root, q) if q else ("unmapped", s)

    notes_all = collections.defaultdict(list)
    for melid, onset, pitch, dur, bar, beat in c.execute("select melid, onset, pitch, duration, bar, beat from melody order by melid, eventid"):
        notes_all[melid].append((onset, pitch, dur, bar, beat))
    beat_rows = collections.defaultdict(dict)
    beat_chord = collections.defaultdict(dict)
    for melid, onset, bar, beat, chord in c.execute("select melid, onset, bar, beat, chord from beats order by melid, onset"):
        beat_rows[melid][(bar, beat)] = onset
        if chord:
            beat_chord[melid]["_cur"] = chord
        beat_chord[melid][(bar, beat)] = beat_chord[melid].get("_cur")
    sections = collections.defaultdict(list)
    for melid, start, end, value in c.execute("select melid, start, end, value from sections where type='CHORD'"):
        sections[melid].append((start, end, value))

    align = collections.Counter()
    metric = collections.Counter()
    metric_lead = []
    disagree_offsets = []
    strata = collections.Counter()
    strata_by_instr = collections.defaultdict(collections.Counter)
    mono = collections.Counter()
    note_labels = collections.Counter()
    instr = {s[0]: s[2] for s in solos}
    for melid, notes in notes_all.items():
        tl = timeline.get(melid, [])
        times = [t for t, _ in tl]
        idx_chord = {}
        for start, end, value in sections.get(melid, []):
            for i in range(start, end + 1):
                idx_chord[i] = value

        def chord_at(t):
            j = bisect.bisect_right(times, t) - 1
            return tl[j][1] if j >= 0 else None

        merged = []
        for t, s in tl:
            if not merged or key(merged[-1][1]) != key(s):
                merged.append((t, s))
        mtimes = [t for t, _ in merged]

        def cur_next(t):
            j = bisect.bisect_right(mtimes, t) - 1
            cur = key(merged[j][1]) if j >= 0 else None
            nxt = key(merged[j + 1][1]) if j + 1 < len(merged) else None
            return cur, nxt

        prev_t = notes[0][0] if notes else 0.0
        for i, (onset, pitch, dur, bar, beat) in enumerate(notes):
            s_time = chord_at(onset)
            # independent source: the note's bar/beat annotation looked up in the beat track
            s_metric = beat_chord[melid].get((bar, beat))
            if (bar, beat) not in beat_rows[melid]:
                metric["annotated_beat_missing_in_beat_track"] += 1
            elif s_metric == s_time:
                metric["agree"] += 1
            else:
                metric["disagree"] += 1
                metric_lead.append(beat_rows[melid][(bar, beat)] - onset)
            note_labels["unlabeled" if s_time is None else "no_chord" if parsed[s_time][1] == "N.C." else
                        "unmapped" if parsed[s_time][1] is None else "lossy" if parsed[s_time][2] else "exact"] += 1
            s_sec = idx_chord.get(i)
            if s_sec is None:
                align["no_section_value"] += 1
            elif s_time is None:
                align["no_timeline_chord"] += 1
            elif s_sec == s_time:
                align["agree"] += 1
            else:
                align["disagree"] += 1
                j = bisect.bisect_right(times, onset)
                if j < len(times) and tl[j][1] == s_sec:
                    align["disagree_section_is_next_chord"] += 1
                    disagree_offsets.append(times[j] - onset)
            if i + 1 < len(notes):
                nxt_onset = notes[i + 1][0]                 # (onset, pitch, dur, bar, beat)
                if nxt_onset < onset + dur - 0.001:
                    mono["overlap_with_next"] += 1
                if abs(nxt_onset - onset) < 1e-6:
                    mono["same_onset_as_next"] += 1
            if dur <= 0:
                mono["nonpositive_duration"] += 1
            if pitch != int(pitch):
                mono["non_integer_pitch"] += 1
            mono["notes"] += 1
            # v2 strata: prefix time approximated by the previous note onset (first note: its own onset)
            if i > 0:
                cur, nxt = cur_next(prev_t)
                on = key(s_time) if s_time is not None else None
                cat = "same_as_current" if on == cur else "equals_first_next" if on == nxt else "not_represented"
                strata[cat] += 1
                strata_by_instr[instr[melid]][cat] += 1
            prev_t = onset
    rep["note_chord_labels"] = dict(note_labels)
    rep["alignment_timeline_vs_sections"] = dict(align)
    if disagree_offsets:
        rep["disagree_next_chord_lead_s"] = {"median": statistics.median(disagree_offsets),
                                             "p90": sorted(disagree_offsets)[int(0.9 * len(disagree_offsets))],
                                             "max": max(disagree_offsets)}
    rep["alignment_metrical_vs_time"] = dict(metric)
    if metric_lead:
        ml = sorted(metric_lead)
        rep["metrical_disagree_beat_minus_onset_s"] = {"min": ml[0], "median": statistics.median(ml), "max": ml[-1],
                                                       "note_before_its_beat": sum(1 for x in ml if x > 0)}
    rep["monophony"] = dict(mono)
    rep["v2_strata_approx"] = dict(strata)
    rep["v2_strata_by_instrument"] = {k: dict(v) for k, v in sorted(strata_by_instr.items())}
    rep["notes_with_metrical_position"] = c.execute("select count(*) from melody where bar is not null and beat is not null").fetchone()[0]
    rep["notes_with_loudness"] = c.execute("select count(*) from melody where loud_med is not null").fetchone()[0]
    with open(out, "w") as f:
        json.dump(rep, f, indent=1, ensure_ascii=False)
    print(json.dumps({k: rep[k] for k in ("solos", "chord_label_stats", "note_chord_labels", "alignment_timeline_vs_sections",
                                          "monophony", "v2_strata_approx")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
