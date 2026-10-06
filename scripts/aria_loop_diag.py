"""Repetition diagnosis on existing Aria outputs (Astra #197): no new generation.

Per output: generated part = output notes minus the decoded prompt tokens (pitch, onset multiset).
Events = notes grouped within 30 ms (sorted pitch tuples). A loop = the longest stretch where
event i equals event i+k (exact, period k <= 24 events) for >= 3 cycles and >= 4 s.
Reports loop start/duration/period/unit, register and density before vs during, and whether the
unit's events occur in the primer. Termination is inferred from the token budget (no logs kept).
"""
import collections, hashlib, json, sys
import pretty_midi

D = "/Users/ohhalim/git_box/t3_aria"
RUNS = [("T3", "P1_pokey_jazz"), ("T3", "P2_Harold_Mabern"), ("T3", "P3_Kenny_Barron"), ("T3c", "TA"), ("T3c", "TB")]
key = lambda n: (n.pitch, round(n.start, 2))
sha = lambda f: hashlib.sha1(open(f, "rb").read()).hexdigest()[:12]

def events(notes, win=0.03):
    ev, cur = [], []
    for n in sorted(notes, key=lambda n: (n.start, n.pitch)):
        if cur and n.start - cur[0].start > win:
            ev.append((cur[0].start, tuple(sorted(m.pitch for m in cur)))); cur = []
        cur.append(n)
    if cur: ev.append((cur[0].start, tuple(sorted(m.pitch for m in cur))))
    return ev

def longest_loop(ev, kmax=24):
    best = None
    for k in range(1, kmax + 1):
        run = 0
        for i in range(len(ev) - k):
            if ev[i][1] == ev[i + k][1]:
                run += 1
                start = i - run + 1
                end_i = i + k
                cycles = (run + k) / k
                dur = ev[end_i][0] - ev[start][0]
                if cycles >= 3 and dur >= 4.0 and (best is None or dur > best["dur"]):
                    best = {"k": k, "start_idx": start, "end_idx": end_i, "dur": dur, "cycles": cycles}
            else:
                run = 0
    return best

manifest = {"purpose": "loop diagnosis on existing Aria outputs (T3, T3c)", "aria_commit": "c2f67bc90b49335543c0249fe00fb84a9337faf8",
            "weights_sha256_16": "5e6b07ee2680f004", "sampler": {"temp": 0.98, "min_p": 0.035, "length": 1024, "prompt_s": 8, "variations": 2},
            "definitions": {"events": "notes within 30 ms grouped", "loop": "exact event repetition with period k<=24, >=3 cycles, >=4 s"},
            "inputs": {}}
rows = []
for exp, p in RUNS:
    dec_f = f"{D}/out/{p}_prompt_decoded.mid"
    dec = collections.Counter(key(n) for i in pretty_midi.PrettyMIDI(dec_f).instruments for n in i.notes)
    prompt_notes = [n for i in pretty_midi.PrettyMIDI(dec_f).instruments for n in i.notes]
    pev = events(prompt_notes)
    p_loop = longest_loop(pev)
    pev_set = collections.Counter(e[1] for e in pev)
    manifest["inputs"][p] = {"prompt_decoded": [dec_f, sha(dec_f)], "outputs": {}}
    for r in (1, 2):
        f = f"{D}/out/{p}/res_{r}.mid"
        manifest["inputs"][p]["outputs"][r] = [f, sha(f)]
        out = sorted((n for i in pretty_midi.PrettyMIDI(f).instruments for n in i.notes), key=lambda n: (n.start, n.pitch))
        left = dec.copy(); gen = []
        for n in out:
            if left[key(n)] > 0: left[key(n)] -= 1
            else: gen.append(n)
        t0 = min(n.start for n in gen)
        ev = events(gen)
        loop = longest_loop(ev)
        row = {"exp": exp, "clip": f"{p}-{r}", "gen_notes": len(gen), "gen_span_s": round(max(n.end for n in gen) - t0, 1),
               "termination": "length cap (≈1024 tokens; no EOS)" if len(gen) >= 330 else "early (EOS?)",
               "primer_loop": None if not p_loop else {"period_events": p_loop["k"], "dur_s": round(p_loop["dur"], 1)}}
        if loop:
            s_t = ev[loop["start_idx"]][0] - t0
            unit = [e[1] for e in ev[loop["start_idx"]:loop["start_idx"] + loop["k"]]]
            per_s = (ev[loop["end_idx"]][0] - ev[loop["start_idx"]][0]) / ((loop["end_idx"] - loop["start_idx"]) / loop["k"])
            before = [n for n in gen if n.start < ev[loop["start_idx"]][0]]
            during = [n for n in gen if ev[loop["start_idx"]][0] <= n.start <= ev[loop["end_idx"]][0]]
            mean = lambda xs: round(sum(n.pitch for n in xs) / len(xs), 1) if xs else None
            dens = lambda xs, d: round(len(xs) / d, 1) if xs and d > 0 else None
            row.update({"loop_start_s": round(s_t, 1), "loop_dur_s": round(loop["dur"], 1), "to_end": abs(ev[loop["end_idx"]][0] - ev[-1][0]) < 1.0,
                        "period_events": loop["k"], "period_s": round(per_s, 2), "cycles": round(loop["cycles"], 1),
                        "unit": [[pretty_midi.note_number_to_name(x) for x in e] for e in unit][:8],
                        "unit_events_in_primer": sum(1 for e in set(unit) if pev_set[e] > 0), "unit_distinct_events": len(set(unit)),
                        "mean_pitch_before_during": [mean(before), mean(during)],
                        "density_before_during": [dens(before, ev[loop["start_idx"]][0] - t0), dens(during, loop["dur"])]})
        rows.append(row)
json.dump(manifest, open(sys.argv[1] + "/manifest.json", "w"), indent=1, ensure_ascii=False)
json.dump(rows, open(sys.argv[1] + "/loops.json", "w"), indent=1, ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else str(o))
for r in rows:
    print(json.dumps(r, ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
