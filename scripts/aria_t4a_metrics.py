#!/usr/bin/env python3
"""T4a metrics: chord-quality 2x2 on generated notes in [t_c, t_c + 0.9375), block timing."""
import collections, json, statistics, sys
import pretty_midi

D = sys.argv[1] if len(sys.argv) > 1 else "/Users/ohhalim/git_box/t3_aria"
WIN = 0.9375
key = lambda n: (n.pitch, round(n.start, 2))


def main():
    plan = json.load(open(f"{D}/t4a/plan.json"))
    man = json.load(open(f"{D}/t4a/manifest.json"))
    rows = []
    for r in man["runs"]:
        v = plan[r["history"]]
        root, t_c = v["root_pc"], v["t_c"]
        p_pcs, q_pcs = {(root + 4) % 12, (root + 11) % 12}, {(root + 3) % 12, (root + 10) % 12}
        inp = collections.Counter(key(n) for i in pretty_midi.PrettyMIDI(v["inputs"][r["quality"]]["path"]).instruments for n in i.notes)
        out = sorted((n for i in pretty_midi.PrettyMIDI(f"{D}/t4a/{r['name']}.out.mid").instruments for n in i.notes), key=lambda n: (n.start, n.pitch))
        left, gen = inp.copy(), []
        for n in out:
            if left[key(n)] > 0:
                left[key(n)] -= 1
            else:
                gen.append(n)
        unmatched_input = sum(left.values())
        w = [n for n in gen if t_c - 1e-3 <= n.start < t_c + WIN]
        rows.append({**{k: r[k] for k in ("name", "history", "quality", "seed", "total_s", "sample_s", "prompt_tokens", "generated_tokens", "eos", "cold")},
                     "unmatched_input_notes": unmatched_input, "gen_notes": len(gen), "window_notes": len(w),
                     "P_tones": sum(n.pitch % 12 in p_pcs for n in w), "Q_tones": sum(n.pitch % 12 in q_pcs for n in w),
                     "fills_window": bool(gen) and max(n.start for n in gen) >= t_c + WIN,
                     "first_gen_onset_after_chord": round(min(n.start for n in gen) - t_c, 3) if gen else None,
                     "gen_before_chord": sum(n.start < t_c - 1e-3 for n in gen)})
    json.dump(rows, open(f"{D}/t4a/metrics.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    by = {(r["history"], r["seed"], r["quality"]): r for r in rows}
    both = 0
    print("hist seed | Y_P: win P Q | Y_Q: win P Q | direction | fill P/Q")
    for h in plan:
        for s in (1, 2, 3, 4):
            a, b = by[(h, s, "P")], by[(h, s, "Q")]
            und = a["window_notes"] < 2 or b["window_notes"] < 2
            d = "undecidable" if und else ("both" if a["P_tones"] > a["Q_tones"] and b["Q_tones"] > b["P_tones"] else
                                           ("Y_P only" if a["P_tones"] > a["Q_tones"] else ("Y_Q only" if b["Q_tones"] > b["P_tones"] else "neither")))
            both += d == "both"
            print(f"{h} s{s} | {a['window_notes']:2d} {a['P_tones']} {a['Q_tones']} | {b['window_notes']:2d} {b['P_tones']} {b['Q_tones']} | {d:11s} | {a['fills_window']}/{b['fills_window']}")
    warm = [r["total_s"] for r in rows if not r["cold"]]
    print("pairs with both directions:", both, "/", len(plan) * 4)
    print("block total_s warm: p50", statistics.median(warm), "max", max(warm), "| cold", [r["total_s"] for r in rows if r["cold"]])
    print("unmatched input notes (should be 0):", sum(r["unmatched_input_notes"] for r in rows), "| gen notes before chord:", sum(r["gen_before_chord"] for r in rows), "| eos:", sum(r["eos"] for r in rows))


if __name__ == "__main__":
    main()
