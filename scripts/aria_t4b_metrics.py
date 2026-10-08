#!/usr/bin/env python3
"""T4b metrics from t4b_main/manifest.json: per block chord following (re-strikes excluded), transitions, joins, timing."""
import collections, json, statistics, sys

D = sys.argv[1] if len(sys.argv) > 1 else "/Users/ohhalim/git_box/t3_aria"


def events(notes, win=0.03):
    ev, cur = [], []
    for n in sorted(notes, key=lambda n: (n[1], n[0])):
        if cur and n[1] - cur[0][1] > win:
            ev.append((cur[0][1], tuple(sorted(m[0] for m in cur))))
            cur = []
        cur.append(n)
    if cur:
        ev.append((cur[0][1], tuple(sorted(m[0] for m in cur))))
    return ev


def main():
    man = json.load(open(f"{D}/t4b_main/manifest.json"))
    rows = []
    for seq in man["sequences"]:
        root = seq["root_pc"]
        maj, mnr = {(root + 4) % 12, (root + 11) % 12}, {(root + 3) % 12, (root + 10) % 12}
        prev = None
        for b in seq["blocks"]:
            acc = b["accepted_notes"]
            chord = b["chord"]
            restrike = [n for n in acc if n[0] in chord]
            nr = [n for n in acc if n[0] not in chord]
            maj_nr = sum(n[0] % 12 in maj for n in nr)
            min_nr = sum(n[0] % 12 in mnr for n in nr)
            above = [n for n in nr if n[0] > max(chord)]
            follows = None
            if maj_nr + min_nr:
                follows = (maj_nr > min_nr) if b["quality"] == "maj7" else (min_nr > maj_nr)
            ev = events(acc)
            rep = sum(1 for a, c in zip(ev, ev[1:]) if a[1] == c[1])
            join = None
            if prev and prev["accepted_notes"] and acc:
                last_t = max(n[1] for n in prev["accepted_notes"])
                last_top = max(n[0] for n in prev["accepted_notes"] if abs(n[1] - last_t) < 0.03)
                first_t = min(n[1] for n in acc)
                first_top = max(n[0] for n in acc if abs(n[1] - first_t) < 0.03)
                join = {"gap_s": round(first_t - last_t, 3), "top_interval": first_top - last_top}
            rows.append({"history": seq["history"], "seed": seq["seed"], "j": b["j"], "quality": b["quality"],
                         "transition": None if prev is None else f"{prev['quality']}->{b['quality']}",
                         "accepted": len(acc), "restrike": len(restrike), "same_pc_other_octave": sum(n[0] % 12 in {c % 12 for c in chord} and n[0] not in chord for n in acc),
                         "maj7_tones_nr": maj_nr, "m7_tones_nr": min_nr,
                         "maj7_above": sum(n[0] % 12 in maj for n in above), "m7_above": sum(n[0] % 12 in mnr for n in above),
                         "follows": follows, "consecutive_same_event": rep, "join": join,
                         "cost_s": b["cost_s"], "miss": b["miss"], "lateness_s": b["lateness_s"],
                         "clipped": b["clipped_at_block_end"], "dropped_after_block": b["dropped_after_block"], "overlap": b["same_pitch_overlap"]})
            prev = b
    json.dump(rows, open(f"{D}/t4b_main/metrics.json", "w"), indent=1)
    f = [r for r in rows if r["follows"] is not None]
    print("blocks", len(rows), "decidable", len(f), "follows", sum(r["follows"] for r in f))
    for q in ("maj7", "m7"):
        g = [r for r in f if r["quality"] == q]
        print(" ", q, "follows", sum(r["follows"] for r in g), "/", len(g), "| maj7 tones", sum(r["maj7_tones_nr"] for r in rows if r["quality"] == q), "m7 tones", sum(r["m7_tones_nr"] for r in rows if r["quality"] == q),
              "| above maj7/m7", sum(r["maj7_above"] for r in rows if r["quality"] == q), "/", sum(r["m7_above"] for r in rows if r["quality"] == q))
    for t in ("maj7->m7", "m7->maj7"):
        g = [r for r in f if r["transition"] == t]
        print(" ", t, "follows", sum(r["follows"] for r in g), "/", len(g))
    for h in ("P2", "P3", "TA", "TB"):
        g = [r for r in f if r["history"] == h]
        print(" ", h, "follows", sum(r["follows"] for r in g), "/", len(g), "undecidable", sum(1 for r in rows if r["history"] == h and r["follows"] is None))
    tot = collections.Counter()
    for r in rows:
        for k in ("accepted", "restrike", "same_pc_other_octave", "clipped", "dropped_after_block", "overlap", "consecutive_same_event"):
            tot[k] += r[k]
    print("totals", dict(tot))
    cost = [r["cost_s"] for r in rows]
    print("cost p50", statistics.median(cost), "max", max(cost), "misses", sum(r["miss"] for r in rows), [(r["history"], r["seed"], r["j"], r["lateness_s"]) for r in rows if r["miss"]])
    joins = [r["join"] for r in rows if r["join"]]
    print("joins: gap p50", statistics.median(j["gap_s"] for j in joins), "|top interval| p50", statistics.median(abs(j["top_interval"]) for j in joins), ">12:", sum(abs(j["top_interval"]) > 12 for j in joins), "/", len(joins))


if __name__ == "__main__":
    main()
