#!/usr/bin/env python3
"""T4a inputs (docs/experiments/T4A_CHORD_PROBE.md): fixed history + one human chord (maj7 or m7)."""
import json, sys
import pretty_midi

D = sys.argv[1] if len(sys.argv) > 1 else "/Users/ohhalim/git_box/t3_aria"
HIST = {"P2": "P2_Harold_Mabern", "P3": "P3_Kenny_Barron", "TA": "TA", "TB": "TB"}
QUAL = {"P": (4, 11), "Q": (3, 10)}          # maj7 vs m7: 3rd and 7th from the root
HALF_BAR = 0.9375


def main():
    plan = {}
    for tag, f in HIST.items():
        notes = [n for i in pretty_midi.PrettyMIDI(f"{D}/out/{f}_prompt_decoded.mid").instruments for n in i.notes]
        last = max(n.start for n in notes)
        root = min((n for n in notes if n.start >= last - 1.0), key=lambda n: n.pitch).pitch % 12
        t_c = last + 0.20
        plan[tag] = {"history": f, "first_onset": min(n.start for n in notes), "last_onset": round(last, 3), "t_c": round(t_c, 3), "root_pc": root, "inputs": {}}
        for q, (third, seventh) in QUAL.items():
            chord = [36 + root, 48 + (root + third) % 12, 48 + (root + seventh) % 12]
            pm = pretty_midi.PrettyMIDI(initial_tempo=120)
            inst = pretty_midi.Instrument(program=0)
            inst.notes = [pretty_midi.Note(velocity=n.velocity, pitch=n.pitch, start=n.start, end=n.end) for n in notes]
            inst.notes += [pretty_midi.Note(velocity=60, pitch=p, start=t_c, end=t_c + HALF_BAR) for p in chord]
            pm.instruments.append(inst)
            path = f"{D}/t4a/{tag}_{q}.mid"
            pm.write(path)
            plan[tag]["inputs"][q] = {"path": path, "chord": chord}
    json.dump(plan, open(f"{D}/t4a/plan.json", "w"), indent=1)
    for t, v in plan.items():
        print(t, {k: v[k] for k in ("first_onset", "last_onset", "t_c", "root_pc")}, {q: x["chord"] for q, x in v["inputs"].items()})


if __name__ == "__main__":
    main()
