#!/usr/bin/env python3
"""Training-entry audit of the local BebopNet score heads (docs/experiments/SCORE_PAIR_ENTRY.md).

Read-only, in memory: parses one representative file per song family with the score_pair
converter's parser, records file metadata (title, credits, rights, source URL), label loss,
cross-family duplicate content, and how often a rest skips more than one chord change, so that
the v2 condition (current + first next chord at the prefix time) does not contain the chord of
the next note's onset. Nothing is written next to the scores. Usage:
score_pair_entry_audit.py <xml dir> <out json>
"""
from __future__ import annotations

import collections
import glob
import hashlib
import json
import os
import re
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_pair_convert import parse  # noqa: E402

LOCAL_NOT_BEBOPNET = {"ours_iiVI_F"}          # untracked in bebopnet-code, written by music21 on 2026-10-03 (ours)
FAMILY_KIND = {"Despacito": "pop", "Dance_Monkey": "pop", "Never_Gonna_Give_You_Up": "pop", "Juice": "pop",
               "my_love": "pop", "simple_songs": "exercise", "Confirmation": "solo_transcription_by_title"}
NAMES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
NGRAM = 4
BASE_KINDS = {"major", "minor", "dominant", "major-seventh", "minor-seventh", "half-diminished",
              "diminished-seventh", "major-sixth", "minor-sixth"}


def family(path: str) -> str:
    b = os.path.basename(path)[:-4]
    b = re.sub(r"_short$", "", b)
    b = re.sub(r"_\d+$", "", b)
    return "Despacito" if b == "DespacitoS" else b


def meta(path: str) -> dict:
    r = ET.parse(path).getroot()
    return {"title": (r.findtext("work/work-title") or r.findtext("movement-title") or "").strip(),
            "credits": [(w.text or "").strip() for w in r.iter("credit-words")][:4],
            "rights": [(x.text or "").strip() for x in r.iter("rights")],
            "source": [(x.text or "").strip() for x in r.iter("source")],
            "software": [(x.text or "").strip() for x in r.iter("software")][:1],
            "encoding_date": [(x.text or "").strip() for x in r.iter("encoding-date")][:1]}


def chord_key(seg):
    if seg is None or seg["label"] == "unlabeled":
        return None
    return "N.C." if seg["label"] == "no_chord" else (seg["root"], seg["quality"])


def main() -> None:
    xml_dir, out = sys.argv[1:3]
    files = sorted(glob.glob(os.path.join(xml_dir, "**", "*.xml"), recursive=True), key=os.path.basename)
    fams = collections.OrderedDict()
    for f in files:
        fams.setdefault(family(f), []).append(f)
    rows, content, progressions = [], collections.defaultdict(list), {}
    pooled = collections.Counter()
    for fam, members in fams.items():
        rel = [os.path.relpath(m, xml_dir) for m in members]
        if fam in LOCAL_NOT_BEBOPNET:
            rows.append({"family": fam, "files": rel, "excluded": "local file, not from the BebopNet repo"})
            continue
        rep = next((m for m in members if "/short/" not in m and family(m) == os.path.basename(m)[:-4]), members[0])
        p = parse(rep)
        segs = p["segments"]
        chords = [s for s in segs if s["label"] == "chord"]
        lossy = [s for s in chords if s["degrees"] or s["kind"] not in BASE_KINDS or (s["bass"] not in (None, s["root"]))]
        unmapped = [s for s in chords if s["quality"] is None]
        notes = sorted(p["notes"], key=lambda n: n["onset"])
        at = lambda t: next((s for s in segs if s["onset"] <= t < s["end"]), None)
        cats = collections.Counter()
        prev_t = 0
        for n in notes:
            cur = chord_key(at(prev_t))
            later = [s for s in segs if s["onset"] > prev_t and chord_key(s) != cur]
            nxt = chord_key(later[0]) if later else None
            on = chord_key(at(n["onset"]))
            if on == cur:
                cats["same_as_current"] += 1
            elif on == nxt:
                cats["equals_first_next"] += 1
            else:
                cats["not_represented"] += 1
            prev_t = n["onset"]
        pooled.update(cats)
        seq = []
        for sg in chords:
            k = (NAMES.index(sg["root"]), sg["quality"])
            if not seq or seq[-1] != k:
                seq.append(k)
        # transposition-invariant chord n-grams: qualities plus root steps between neighbours
        grams = {tuple((seq[i + j][1], (seq[i + j + 1][0] - seq[i + j][0]) % 12) for j in range(NGRAM - 1)) + (seq[i + NGRAM - 1][1],)
                 for i in range(len(seq) - NGRAM + 1)}
        progressions[fam] = grams
        sig = hashlib.sha1(json.dumps([n["pitch"] for n in notes[:24]]).encode()).hexdigest()[:12]
        content[sig].append(fam)
        rows.append({"family": fam, "files": rel, "representative": os.path.relpath(rep, xml_dir),
                     "kind": FAMILY_KIND.get(fam, "jazz_head_by_title"), "meta": meta(rep),
                     "measures": len(p["measures"]), "irregular_measures": p["features"].get("irregular_measure", 0),
                     "notes": len(notes), "chord_segments": len(chords), "no_chord": p["features"].get("no_chord", 0),
                     "unlabeled_segments": sum(1 for s in segs if s["label"] == "unlabeled"),
                     "unmapped_kinds": sorted({s["kind"] for s in unmapped}), "lossy_segments": len(lossy),
                     "unsupported": len(p["unsupported"]), "unsupported_items": sorted({u["item"] for u in p["unsupported"]})[:6],
                     "repeat_or_ending": p["features"].get("repeat", 0) + p["features"].get("ending", 0),
                     "tempo_changes": p["features"].get("tempo_change", 0), "meter_changes": p["features"].get("time_signature_change", 0),
                     "next_note_chord": dict(cats), "content_sig": sig})
    dup = {k: v for k, v in content.items() if len(v) > 1}
    names = list(progressions)
    sims = sorted(((len(progressions[a] & progressions[b]) / len(progressions[a] | progressions[b]), a, b)
                   for i, a in enumerate(names) for b in names[i + 1:] if progressions[a] | progressions[b]), reverse=True)
    total = sum(pooled.values())
    report = {"xml_dir": xml_dir, "files": len(files), "families": len(fams),
              "bebopnet_families": sum(1 for r in rows if "excluded" not in r), "rows": rows,
              "cross_family_same_first24_pitches": dup,
              "progression_jaccard_top": [{"a": a, "b": b, "jaccard": round(j, 3)} for j, a, b in sims[:15]],
              "progression_pairs_jaccard_ge_0_3": sum(1 for j, _, _ in sims if j >= 0.3),
              "next_note_chord_pooled": dict(pooled), "not_represented_share": pooled["not_represented"] / total if total else None,
              "note": "prefix time approximated by the previous note onset (5 s <T> boundaries ignored)"}
    with open(out, "w") as f:
        json.dump(report, f, indent=1, ensure_ascii=False)
    print(len(files), "files", len(fams), "families", report["bebopnet_families"], "bebopnet", "dup", dup,
          "pooled", dict(pooled), "share", round(report["not_represented_share"], 4))


if __name__ == "__main__":
    main()
