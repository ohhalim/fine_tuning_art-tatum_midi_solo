#!/usr/bin/env python3
"""T4a post-hoc decomposition (Astra #203), no new generation.

Generated notes in [t_c, t_c + 0.9375) are split into: exact re-strikes of an input chord pitch,
the same pitch class in another octave, and other notes; notes above the chord's top note are
reported separately (not a right-hand label). Per seed pair: d = [(P-Q)/n]_Y_P - [(P-Q)/n]_Y_Q,
for all window notes and for non-re-strike notes only. Also: history notes still sounding at t_c.
"""
import collections, json, sys
import pretty_midi

D = sys.argv[1] if len(sys.argv) > 1 else "/Users/ohhalim/git_box/t3_aria"
WIN = 0.9375
key = lambda n: (n.pitch, round(n.start, 2))


def main():
    plan = json.load(open(f"{D}/t4a/plan.json"))
    man = json.load(open(f"{D}/t4a/manifest.json"))
    rows = {}
    for r in man["runs"]:
        v = plan[r["history"]]
        root, t_c = v["root_pc"], v["t_c"]
        chord = v["inputs"][r["quality"]]["chord"]
        p_pcs, q_pcs = {(root + 4) % 12, (root + 11) % 12}, {(root + 3) % 12, (root + 10) % 12}
        inp_notes = [n for i in pretty_midi.PrettyMIDI(v["inputs"][r["quality"]]["path"]).instruments for n in i.notes]
        inp = collections.Counter(key(n) for n in inp_notes)
        out = sorted((n for i in pretty_midi.PrettyMIDI(f"{D}/t4a/{r['name']}.out.mid").instruments for n in i.notes), key=lambda n: (n.start, n.pitch))
        left, gen = inp.copy(), []
        for n in out:
            if left[key(n)] > 0:
                left[key(n)] -= 1
            else:
                gen.append(n)
        w = [n for n in gen if t_c - 1e-3 <= n.start < t_c + WIN]
        cats = collections.Counter()
        tone = collections.Counter()
        for n in w:
            c = "restrike" if n.pitch in chord else ("same_pc_other_octave" if n.pitch % 12 in {x % 12 for x in chord} else "other")
            cats[c] += 1
            t = "P" if n.pitch % 12 in p_pcs else ("Q" if n.pitch % 12 in q_pcs else None)
            if t:
                tone[(t, "restrike" if c == "restrike" else "non_restrike")] += 1
                if n.pitch > max(chord):
                    tone[(t, "above_chord")] += 1
        sounding_history = [n for n in inp_notes if n.start < t_c and n.end > t_c and n.pitch not in chord]
        rows[r["name"]] = {"history": r["history"], "quality": r["quality"], "seed": r["seed"], "window_notes": len(w),
                           "restrike": cats["restrike"], "same_pc_other_octave": cats["same_pc_other_octave"], "other": cats["other"],
                           "above_chord_top": sum(n.pitch > max(chord) for n in w),
                           "P_restrike": tone[("P", "restrike")], "P_non_restrike": tone[("P", "non_restrike")], "P_above": tone[("P", "above_chord")],
                           "Q_restrike": tone[("Q", "restrike")], "Q_non_restrike": tone[("Q", "non_restrike")], "Q_above": tone[("Q", "above_chord")],
                           "history_notes_sounding_at_chord": len(sounding_history),
                           "history_latest_end_after_chord_s": round(max((n.end for n in inp_notes if n.pitch not in chord or n.start < t_c - 1e-3), default=t_c) - t_c, 3)}
    pairs = []
    for h in plan:
        for s in (1, 2, 3, 4):
            a, b = rows[f"{h}_P_s{s}"], rows[f"{h}_Q_s{s}"]
            def frac(r, which):
                if which == "all":
                    p, q, n = r["P_restrike"] + r["P_non_restrike"], r["Q_restrike"] + r["Q_non_restrike"], r["window_notes"]
                else:
                    p, q, n = r["P_non_restrike"], r["Q_non_restrike"], r["window_notes"] - r["restrike"]
                return (p - q) / n if n else None
            fa, fb = frac(a, "all"), frac(b, "all")
            na, nb = frac(a, "nr"), frac(b, "nr")
            pairs.append({"history": h, "seed": s, "d_all": None if fa is None or fb is None else round(fa - fb, 3),
                          "d_non_restrike": None if na is None or nb is None else round(na - nb, 3)})
    json.dump({"runs": rows, "pairs": pairs}, open(f"{D}/t4a/decompose.json", "w"), indent=1)
    tot = collections.Counter()
    for r in rows.values():
        for k in ("window_notes", "restrike", "same_pc_other_octave", "other", "above_chord_top", "P_restrike", "P_non_restrike", "P_above", "Q_restrike", "Q_non_restrike", "Q_above"):
            tot[(r["quality"], k)] += r[k]
    for q in ("P", "Q"):
        print(q, {k: tot[(q, k)] for k in ("window_notes", "restrike", "same_pc_other_octave", "other", "above_chord_top", "P_restrike", "P_non_restrike", "P_above", "Q_restrike", "Q_non_restrike", "Q_above")})
    for h in plan:
        ps = [p for p in pairs if p["history"] == h]
        print(h, "d_all", [p["d_all"] for p in ps], "d_non_restrike", [p["d_non_restrike"] for p in ps])
    print("history notes sounding at chord onset (per input):", {h: rows[f"{h}_P_s1"]["history_notes_sounding_at_chord"] for h in plan},
          "latest history end after t_c:", {h: rows[f"{h}_P_s1"]["history_latest_end_after_chord_s"] for h in plan})


if __name__ == "__main__":
    main()
