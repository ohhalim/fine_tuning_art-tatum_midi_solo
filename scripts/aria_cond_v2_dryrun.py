#!/usr/bin/env python3
"""Read-only feature dry run of contract v2 on the score-head pilot (docs/experiments/ARIA_COND_CONTRACT_V2.md).

No model, no training: tokenizes each derived.mid with the Aria tokenizer, builds v2 features
from the score chord annotation, and writes counts plus fixed sample rows for an independent
check against the XML (aria_cond_v2_xmlcheck.py). Run with the Aria venv:
aria_cond_v2_dryrun.py <aria repo> <score pair root> <out json>
"""
from __future__ import annotations

import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from aria_cond_contract_v2 import FEATURE_DIM, NC, features, plan_from_beats, vector  # noqa: E402
from aria_cond_smoke import chord_pcs  # noqa: E402

SONGS = ["A_Foggy_Day", "All_The_Things_You_Are", "Billies_Bounce"]
SAMPLE_CHANGES = (5, 10, 15)     # fixed before the run: the 5th, 10th and 15th chord change in token time
# lossy annotation: degrees (tensions, alterations), extended kinds folded into a family, or a slash
# bass; the 12-dim family chroma keeps none of these


def main() -> None:
    aria_repo, root, out_path = sys.argv[1:4]
    sys.path.insert(0, aria_repo)
    from ariautils.midi import MidiDict
    from ariautils.tokenizer import AbsTokenizer
    tok = AbsTokenizer()
    report = {}
    for song in SONGS:
        with open(os.path.join(root, song, "conversion.json")) as f:
            conv = json.load(f)
        assert len(conv["tempos"]) == 1
        bpm = conv["tempos"][0]["bpm"]
        segs, lossy = [], []
        for c in conv["chords"]:
            if c["label"] == "chord" and c["quality"]:
                pcs = frozenset(chord_pcs(c["root"], c["quality"]))
            elif c["label"] == "no_chord":
                pcs = NC
            else:
                pcs = None
            segs.append((c["onset"], c["end"], pcs))
            lossy.append(bool(c.get("degrees")) or (c.get("bass") not in (None, c.get("root"))) or c.get("kind") not in (None, "none", "major", "minor", "dominant",
                                                                           "major-seventh", "minor-seventh", "half-diminished",
                                                                           "diminished-seventh", "major-sixth", "minor-sixth"))
        toks = tok.tokenize(MidiDict.from_midi(conv["derived_midi"]))
        first_onset_beat = min(n["onset_beat"] for n in conv["notes"])
        shift_ms = first_onset_beat * 60000.0 / bpm
        plan = plan_from_beats(segs, [(0, bpm)], shift_ms)
        feats = features(toks, plan)
        vecs = [vector(f) for f in feats]
        assert all(len(v) == FEATURE_DIM and all(math.isfinite(x) for x in v) for v in vecs)
        to_beat = lambda t_ms: (t_ms + shift_ms) * bpm / 60000.0
        seg_of = lambda beat: next((i for i, c in enumerate(conv["chords"]) if c["onset"] <= beat < c["end"]), None)
        # onset positions where the current chord differs from the previous onset position
        changes, prev = [], None
        for i, (t, f) in enumerate(zip(toks, feats)):
            if isinstance(t, tuple) and t[0] == "onset":
                if prev is not None and f["current"] != prev:
                    changes.append(i)
                prev = f["current"]
        samples = []
        for k in SAMPLE_CHANGES:
            if k <= len(changes):
                for i in (changes[k - 1] - 1, changes[k - 1]):       # the pitch token before, then the onset token
                    f, beat = feats[i], to_beat(feats[i]["t_ms"])
                    ci = seg_of(beat)
                    samples.append({"change": k, "position": i, "token": str(toks[i]), "t_ms": f["t_ms"], "beat": round(beat, 4),
                                    "measure": int(beat // 4) + 1, "beat_in_measure": round(beat % 4 + 1, 4),
                                    "current": None if f["current"] in (None, NC) else sorted(f["current"]),
                                    "current_label": None if ci is None else f"{conv['chords'][ci]['root']}{conv['chords'][ci]['quality']} (m{conv['chords'][ci]['measure']})",
                                    "next": None if f["next"] in (None, NC) else sorted(f["next"]),
                                    "delta_sec": None if f["delta_sec"] is None else round(f["delta_sec"], 4)})
        # token-exact: is the chord at each note's actual onset among (current, next) at its pitch position?
        cats = {"same_as_current": 0, "equals_first_next": 0, "not_represented": 0}
        for i, t in enumerate(toks[:-1]):
            if isinstance(t, tuple) and t[0] == "piano" and isinstance(toks[i + 1], tuple) and toks[i + 1][0] == "onset":
                onset_t = feats[i + 1]["t_ms"]
                seg = next((sg for sg in plan if sg[0] <= onset_t < sg[1]), None)
                on = None if seg is None else seg[2]
                key = "same_as_current" if on == feats[i]["current"] else "equals_first_next" if on == feats[i]["next"] else "not_represented"
                cats[key] += 1
        n = len(feats)
        report[song] = {"source_xml": conv["source"], "derived_midi": conv["derived_midi"], "bpm": bpm,
                        "leading_silence_ms": round(shift_ms, 3), "tokens": n,
                        "current_known": sum(f["current_known"] for f in feats), "unknown": sum(1 - f["current_known"] for f in feats),
                        "no_chord_positions": sum(f["current"] == NC for f in feats),
                        "has_next": sum(f["has_next"] for f in feats), "clamped": sum(f["delta_clamped"] for f in feats),
                        "delta_sec_max_unclamped": max((f["delta_sec"] for f in feats if f["delta_sec"] is not None and not f["delta_clamped"]), default=None),
                        "chord_changes_seen_at_onsets": len(changes), "next_note_chord_token_exact": cats,
                        "lossy_annotation_segments": sum(lossy), "of_segments": len(lossy),
                        "positions_on_lossy_segments": sum(1 for f in feats if f["current_known"]
                                                           and seg_of(to_beat(f["t_ms"])) is not None and lossy[seg_of(to_beat(f["t_ms"]))]),
                        "samples": samples}
        print(song, n, "unknown", report[song]["unknown"], "has_next", report[song]["has_next"], "clamped", report[song]["clamped"],
              "changes", len(changes), "lossy", sum(lossy), "/", len(lossy))
    with open(out_path, "w") as f:
        json.dump(report, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
