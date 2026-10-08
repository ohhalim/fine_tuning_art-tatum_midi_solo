#!/usr/bin/env python3
"""Build the repeat / control contexts for docs/experiments/LOOP_TF.md (run in the main venv).

For each loop found by aria_loop_diag.py: prefix = output notes before the loop start t_L; unit U =
one period; R_n = prefix + n copies of U + candidate U; C_n = prefix + the n*P seconds before t_L
shifted forward by n*P + the same candidate. Writes <tag>_<cond><n>_{ctx,full}.mid and plan.json.
The exact code that produced loop_tf/plan.json was run inline on 2026-10-08; this file restates it.
"""
import collections, json, sys
import pretty_midi

D = sys.argv[1] if len(sys.argv) > 1 else "/Users/ohhalim/git_box/t3_aria"
LOOPS = {"P2_Harold_Mabern-2": ("P2_Harold_Mabern", 2), "P3_Kenny_Barron-1": ("P3_Kenny_Barron", 1), "TA-2": ("TA", 2), "TB-1": ("TB", 1)}
key = lambda n: (n.pitch, round(n.start, 2))


def events(notes, win=0.03):
    ev, cur = [], []
    for n in sorted(notes, key=lambda n: (n.start, n.pitch)):
        if cur and n.start - cur[0].start > win:
            ev.append((cur[0].start, tuple(sorted(m.pitch for m in cur))))
            cur = []
        cur.append(n)
    if cur:
        ev.append((cur[0].start, tuple(sorted(m.pitch for m in cur))))
    return ev


def longest_loop(ev, kmax=24):
    best = None
    for k in range(1, kmax + 1):
        run = 0
        for i in range(len(ev) - k):
            if ev[i][1] == ev[i + k][1]:
                run += 1
                start, end_i = i - run + 1, i + k
                dur = ev[end_i][0] - ev[start][0]
                if (run + k) / k >= 3 and dur >= 4.0 and (best is None or dur > best["dur"]):
                    best = {"k": k, "start_idx": start, "end_idx": end_i, "dur": dur}
            else:
                run = 0
    return best


def shifted(notes, dt):
    return [pretty_midi.Note(velocity=n.velocity, pitch=n.pitch, start=n.start + dt, end=n.end + dt) for n in notes]


def write(notes, path):
    pm = pretty_midi.PrettyMIDI(initial_tempo=120)
    inst = pretty_midi.Instrument(program=0)
    inst.notes = sorted(notes, key=lambda n: (n.start, n.pitch))
    pm.instruments.append(inst)
    pm.write(path)


def main():
    plan = {}
    for tag, (p, r) in LOOPS.items():
        dec = collections.Counter(key(n) for i in pretty_midi.PrettyMIDI(f"{D}/out/{p}_prompt_decoded.mid").instruments for n in i.notes)
        out = sorted((n for i in pretty_midi.PrettyMIDI(f"{D}/out/{p}/res_{r}.mid").instruments for n in i.notes), key=lambda n: (n.start, n.pitch))
        left, gen = dec.copy(), []
        for n in out:
            if left[key(n)] > 0:
                left[key(n)] -= 1
            else:
                gen.append(n)
        ev = events(gen)
        lp = longest_loop(ev)
        t_l = ev[lp["start_idx"]][0]
        period = ev[lp["start_idx"] + lp["k"]][0] - t_l
        prefix = [n for n in out if n.start < t_l - 1e-6]
        unit = [n for n in out if t_l - 1e-6 <= n.start < t_l + period - 1e-6]
        plan[tag] = {"t_L": round(t_l, 4), "P": round(period, 4), "k": lp["k"], "unit_notes": len(unit), "prefix_notes": len(prefix), "conditions": {}}
        for nrep in (0, 1, 2, 4):
            cand = shifted(unit, nrep * period)
            reps = [m for j in range(nrep) for m in shifted(unit, j * period)]
            filler = shifted([m for m in out if t_l - nrep * period - 1e-6 <= m.start < t_l - 1e-6], nrep * period)
            for cond, mid in (("R", reps), ("C", filler)):
                if nrep == 0 and cond == "C":
                    continue
                base = f"{D}/loop_tf/{tag}_{cond}{nrep}"
                write(prefix + mid, base + "_ctx.mid")
                write(prefix + mid + cand, base + "_full.mid")
                plan[tag]["conditions"][f"{cond}{nrep}"] = {"ctx": base + "_ctx.mid", "full": base + "_full.mid", "mid_notes": len(mid)}
    json.dump(plan, open(f"{D}/loop_tf/plan.json", "w"), indent=1)


if __name__ == "__main__":
    main()
