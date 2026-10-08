#!/usr/bin/env python3
"""Metrics for docs/experiments/MINP_AB.md: first 16 s after generation start, per run."""
import collections, json, statistics, sys
import pretty_midi

D = sys.argv[1] if len(sys.argv) > 1 else "/Users/ohhalim/git_box/t3_aria"
DECODED = {"P1": "P1_pokey_jazz", "P2": "P2_Harold_Mabern", "P3": "P3_Kenny_Barron", "TA": "TA", "TB": "TB"}
WIN = 16.0
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


def stretches(ev, kmax=24, min_cycles=3):
    """Maximal pitch-set periodic stretches (start time, end time, period k)."""
    out = []
    for k in range(1, kmax + 1):
        run = 0
        for i in range(len(ev) - k):
            if ev[i][1] == ev[i + k][1]:
                run += 1
                last = i == len(ev) - k - 1 or ev[i + 1][1] != ev[i + 1 + k][1]
                if last and (run + k) / k >= min_cycles:
                    start = i - run + 1
                    out.append((ev[start][0], ev[i + k][0], k))
            else:
                run = 0
    return out


def union_len(iv):
    tot, cur = 0.0, None
    for a, b in sorted(iv):
        if cur is None or a > cur[1]:
            if cur:
                tot += cur[1] - cur[0]
            cur = [a, b]
        else:
            cur[1] = max(cur[1], b)
    if cur:
        tot += cur[1] - cur[0]
    return tot


def main():
    man = json.load(open(f"{D}/minp_ab/manifest.json"))
    rows = []
    for run in man["runs"]:
        dec = collections.Counter(key(n) for i in pretty_midi.PrettyMIDI(f"{D}/out/{DECODED[run['prompt']]}_prompt_decoded.mid").instruments for n in i.notes)
        out = sorted((n for i in pretty_midi.PrettyMIDI(f"{D}/minp_ab/{run['name']}.mid").instruments for n in i.notes), key=lambda n: (n.start, n.pitch))
        left, gen = dec.copy(), []
        for n in out:
            if left[key(n)] > 0:
                left[key(n)] -= 1
            else:
                gen.append(n)
        t0 = min(n.start for n in gen)
        w = [n for n in gen if n.start < t0 + WIN]
        ev = events(w)
        st = stretches(ev)
        occ = union_len([(a, b) for a, b, k in st if b - a >= 2.0]) / WIN
        loop4 = any(b - a >= 4.0 for a, b, k in st)
        collapse = []
        for j in range(int(WIN // 2)):
            ns = [n for n in w if t0 + 2 * j <= n.start < t0 + 2 * j + 2]
            if len(ns) >= 6 and all(n.pitch < 48 for n in ns):
                collapse.append(2 * j)
        last_onset = max(n.start for n in gen) - t0
        iois = [b[0] - a[0] for a, b in zip(ev, ev[1:])]
        rows.append({"name": run["name"], "prompt": run["prompt"], "min_p": run["min_p"], "seed": run["seed"], "stop": run["stop"],
                     "coverage_ok": last_onset >= WIN, "gen_last_onset_s": round(last_onset, 1), "notes_16s": len(w),
                     "repeat_occupancy": round(occ, 2), "loop_ge4s": loop4, "register_collapse_windows": collapse,
                     "distinct_pitches_16s": len({n.pitch for n in w}),
                     "median_event_gap_s": round(statistics.median(iois), 3) if iois else None,
                     "velocity_range": [min(n.velocity for n in w), max(n.velocity for n in w)] if w else None})
    json.dump(rows, open(f"{D}/minp_ab/metrics.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    by = {(r["prompt"], r["seed"], r["min_p"]): r for r in rows}
    print("prompt seed | A(.035) occ loop4 collapse cov | B(0) occ loop4 collapse cov | B<A")
    for p in DECODED:
        for s in (1, 2):
            a, b = by[(p, s, 0.035)], by[(p, s, 0.0)]
            print(f"{p} s{s} | {a['repeat_occupancy']:.2f} {a['loop_ge4s']!s:5} {len(a['register_collapse_windows'])} {a['coverage_ok']!s:5} | "
                  f"{b['repeat_occupancy']:.2f} {b['loop_ge4s']!s:5} {len(b['register_collapse_windows'])} {b['coverage_ok']!s:5} | {b['repeat_occupancy'] < a['repeat_occupancy']}"
                  f"  notes {a['notes_16s']}/{b['notes_16s']} pitches {a['distinct_pitches_16s']}/{b['distinct_pitches_16s']} gap {a['median_event_gap_s']}/{b['median_event_gap_s']} vel {a['velocity_range']}/{b['velocity_range']}")


if __name__ == "__main__":
    main()
