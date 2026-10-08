"""Repetition diagnosis on existing Aria outputs (Astra #197): no new generation.

Per output: generated part = output notes minus the decoded prompt tokens (pitch, onset multiset).
Events = notes grouped within 30 ms (sorted pitch tuples). A loop = the longest stretch where
the pitch set of event i equals that of event i+k (period k <= 24 events) for >= 3 cycles and
>= 4 s: a pitch-set sequence repetition after 30 ms grouping. Onset gaps, durations and
velocities are not compared; whether the rhythm repeats too is reported separately (Astra #198).
Reports loop start/duration/period/unit, register and density before vs during, whether it lasts
to the last generated onset, and how many unit events also occur (as single events, order and
rhythm ignored) in the primer. Termination was not logged: observed = unknown.
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
            "definitions": {"events": "notes within 30 ms grouped", "loop": "pitch-set sequence repetition after 30 ms grouping, period k<=24 events, >=3 cycles, >=4 s; rhythm not compared",
                            "to_last_onset": "loop end index == last generated event", "rhythm_repeat": "onset gaps of the loop span equal to those one period later within 30 ms"},
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
               "termination_observed": "unknown (not logged)",
               "length_cap_hypothesis": len(gen) >= 330,
               "primer_loop": None if not p_loop else {"period_events": p_loop["k"], "dur_s": round(p_loop["dur"], 1)}}
        if loop:
            s_t = ev[loop["start_idx"]][0] - t0
            unit = [e[1] for e in ev[loop["start_idx"]:loop["start_idx"] + loop["k"]]]
            per_s = (ev[loop["end_idx"]][0] - ev[loop["start_idx"]][0]) / ((loop["end_idx"] - loop["start_idx"]) / loop["k"])
            before = [n for n in gen if n.start < ev[loop["start_idx"]][0]]
            during = [n for n in gen if ev[loop["start_idx"]][0] <= n.start <= ev[loop["end_idx"]][0]]
            mean = lambda xs: round(sum(n.pitch for n in xs) / len(xs), 1) if xs else None
            dens = lambda xs, d: round(len(xs) / d, 1) if xs and d > 0 else None
            gaps = [ev[i + 1][0] - ev[i][0] for i in range(loop["start_idx"], loop["end_idx"] - loop["k"])]
            gaps_next = [ev[i + 1 + loop["k"]][0] - ev[i + loop["k"]][0] for i in range(loop["start_idx"], loop["end_idx"] - loop["k"])]
            rhythm = sum(1 for a, b in zip(gaps, gaps_next) if abs(a - b) <= 0.03) / max(1, len(gaps))
            row.update({"loop_start_s": round(s_t, 1), "loop_dur_s": round(loop["dur"], 1),
                        "to_last_onset": loop["end_idx"] == len(ev) - 1,
                        "events_after_loop": len(ev) - 1 - loop["end_idx"],
                        "seconds_after_loop": round(ev[-1][0] - ev[loop["end_idx"]][0], 2),
                        "rhythm_repeat_share": round(rhythm, 2),
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
