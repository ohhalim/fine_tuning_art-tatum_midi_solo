#!/usr/bin/env python3
"""T4b: sequential half-bar generation over a planned chord progression (run with the Aria venv).

Per (history, seed): blocks j = 0..3 at T_j = t_c + j*0.9375. Context for block j = history notes +
accepted generated notes of blocks < j + planned chords of blocks <= j (no future chord). Aria
continues 48 tokens; generated notes = output minus context (pitch, onset 10 ms multiset);
accepted = onset in [T_j, T_j + 0.9375), end clipped at the block end (end <= onset dropped); the raw
generated notes are kept separately. Planned chords stay input (source tag "guide").
Timing (Astra #209): one sequential worker. cost_j = context MIDI write + tokenize + sample +
detokenize + accept. Simulated clock in music time: actual_start_j = max(T_j - 0.9375,
actual_ready_{j-1}), actual_ready_j = actual_start_j + cost_j, miss = actual_ready_j > T_j - 0.05.
Block 0 starts at its nominal pre-roll T_0 - 0.9375. Blocks after a miss form a tentative chain.
"""
import argparse, collections, hashlib, json, os, sys, time

import mido

BLOCK, MARGIN = 0.9375, 0.05
TPB, TEMPO = 500, 500000            # 120 BPM, 500 ticks per beat: 1 tick = 1 ms, so the 10 ms token grid is exact
SOURCE_PRIORITY = {"guide": 0, "history": 1, "generated": 2}   # same pitch, same onset: lower wins in the model context


def grid_ms(t0_s: float, j: int) -> int:
    """Absolute block boundary q_j in integer ms on the 10 ms token grid: round half up of
    T0 + j * 0.9375 s (contract v2, Astra #210). Never accumulated, so no drift."""
    t_ms = round(t0_s * 1000) + j * 937.5
    return int((t_ms + 5) // 10) * 10


def accept(gen, q_ms: int, q_next_ms: int):
    """Generated notes [(pitch, start_s, end_s, vel)] with onset in [q_j, q_{j+1}) (ms), end clipped at
    q_{j+1}; an onset exactly on q_{j+1} belongs to the next block. Returns (accepted, clipped, lost_s)."""
    acc, clipped, lost = [], 0, 0.0
    for p, s, e, v in gen:
        s_ms = round(s * 1000)
        if q_ms <= s_ms < q_next_ms:
            end = min(e, q_next_ms / 1000)
            if e > q_next_ms / 1000:
                clipped += 1
                lost += e - q_next_ms / 1000
            if end > s:
                acc.append((p, s, end, v))
    return acc, clipped, lost


def normalize_context(tagged):
    """Model-context normalization for this writer/parser/tokenizer path (ariautils closes every open
    same-pitch note-on at one note-off). Input [(pitch, start, end, vel, source)]. Same pitch and
    same onset: the higher-priority source stays (guide > history > generated), the other is
    dropped from the context only. A same-pitch note sounding into a later onset ends there.
    end <= start is dropped. Returns (notes, log); originals are not modified."""
    log = collections.Counter()
    lost = collections.Counter()
    by_pitch = collections.defaultdict(list)
    for n in tagged:
        by_pitch[n[0]].append(list(n))
    out = []
    for pitch, notes in by_pitch.items():
        notes.sort(key=lambda n: (round(n[1] * 1000), SOURCE_PRIORITY[n[4]], -n[2]))
        kept = []
        for n in notes:
            if kept and round(kept[-1][1] * 1000) == round(n[1] * 1000):
                log[f"same_onset_drop:{kept[-1][4]}>{n[4]}"] += 1
                continue
            kept.append(n)
        for a, b in zip(kept, kept[1:]):
            if a[2] > b[1]:
                pair = f"{a[4]}<-{b[4]}"
                log[f"truncated:{pair}"] += 1
                lost[pair] += a[2] - b[1]
                a[2] = b[1]
        for n in kept:
            if n[2] > n[1]:
                out.append(tuple(n))
            else:
                log["dropped_zero_length"] += 1
    return sorted(out, key=lambda n: (n[1], n[0])), {"counts": dict(log), "lost_s": {k: round(v, 3) for k, v in lost.items()}}
QUAL = {"maj7": (4, 11), "m7": (3, 10)}
ORDER = {1: ["maj7", "m7", "maj7", "m7"], 2: ["m7", "maj7", "m7", "maj7"]}


def write_notes(notes, path):
    """notes: [(pitch, start, end, velocity, ...)] -> single-track MIDI, 1 tick = 1 ms."""
    ev = []
    for n in notes:
        p, s, e, v = n[:4]
        ev.append((round(s * 1000), 1, p, v))
        ev.append((round(e * 1000), 0, p, 0))
    ev.sort(key=lambda x: (x[0], x[1]))           # note-offs before note-ons at the same tick
    mid = mido.MidiFile(ticks_per_beat=TPB)
    tr = mido.MidiTrack()
    tr.append(mido.MetaMessage("set_tempo", tempo=TEMPO, time=0))
    last = 0
    for t, on, p, v in ev:
        tr.append(mido.Message("note_on" if on else "note_off", note=p, velocity=v if on else 0, time=t - last))
        last = t
    mid.tracks.append(tr)
    mid.save(path)


def read_notes(path):
    out, t, active = [], 0.0, collections.defaultdict(list)
    for msg in mido.MidiFile(path):
        t += msg.time
        if msg.type == "note_on" and msg.velocity > 0:
            active[msg.note].append((t, msg.velocity))
        elif msg.type in ("note_off", "note_on") and active[msg.note]:
            s, v = active[msg.note].pop(0)
            out.append((msg.note, s, t, v))
    return sorted(out, key=lambda n: (n[1], n[0]))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--d", required=True)
    ap.add_argument("--aria-dir", required=True)
    ap.add_argument("--histories", nargs="+", default=["P2", "P3", "TA", "TB"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sys.path.insert(0, args.aria_dir)
    import torch
    from ariautils.midi import MidiDict
    from ariautils.tokenizer import AbsTokenizer
    from aria.inference import get_inference_prompt
    from aria.inference.sample_mlx import sample_batch
    from aria.run import _load_inference_model_mlx

    os.makedirs(args.out, exist_ok=True)
    plan = json.load(open(f"{args.d}/t4a/plan.json"))
    tok = AbsTokenizer()
    t = time.time()
    model = _load_inference_model_mlx(f"{args.d}/ckpt/model-gen.safetensors", "medium", strict=False)
    man = {"load_s": round(time.time() - t, 2), "block": BLOCK, "budget_s": BLOCK - MARGIN, "end_policy": "min(raw_end, block end); end<=onset dropped",
           "temp": 0.98, "min_p": 0.0, "new_tokens": 48, "rng": "torch.manual_seed(1000*seed + j)", "sequences": []}
    man["contract"] = "v2 (Astra #210): absolute 10 ms grid q_j, 1 tick = 1 ms, context-only same-pitch normalization"
    key = lambda n: (n[0], round(n[1] * 1000))
    for h in args.histories:
        v = plan[h]
        hist = [n + ("history",) for n in read_notes(f"{args.d}/out/{v['history']}_prompt_decoded.mid")]
        root, t0 = v["root_pc"], v["t_c"]
        for seed in args.seeds:
            accepted, guides, blocks, prev_ready = [], [], [], None
            for j, q in enumerate(ORDER[seed]):
                T = t0 + j * BLOCK                                  # tempo clock (deadlines)
                qj, qn = grid_ms(t0, j), grid_ms(t0, j + 1)          # model grid (ms)
                third, seventh = QUAL[q]
                guides += [(p_, qj / 1000, qn / 1000, 60, "guide") for p_ in (36 + root, 48 + (root + third) % 12, 48 + (root + seventh) % 12)]
                tagged = hist + [n + ("generated",) for n in accepted] + guides
                ts = time.time()
                ctx, norm_log = normalize_context(tagged)
                path = os.path.join(args.out, f"{h}_s{seed}_b{j}_ctx.mid")
                write_notes(ctx, path)
                prompt = get_inference_prompt(MidiDict.from_midi(path), tok, 1e12)
                torch.manual_seed(1000 * seed + j)
                res = sample_batch(model=model, tokenizer=tok, prompt=prompt, num_variations=1, max_new_tokens=48,
                                   temp=0.98, force_end=False, top_p=None, min_p=0.0)[0]
                opath = os.path.join(args.out, f"{h}_s{seed}_b{j}_out.mid")
                tok.detokenize(res).to_midi().save(opath)
                out = read_notes(opath)
                left = collections.Counter(key(n) for n in ctx)
                gen = []
                for n in out:
                    if left[key(n)] > 0:
                        left[key(n)] -= 1
                    else:
                        gen.append(n)
                acc, clipped, lost = accept(gen, qj, qn)
                cost = time.time() - ts
                overlap = sum(1 for a in acc for b in acc if a is not b and a[0] == b[0] and a[1] < b[1] < a[2])
                accepted += acc
                # normalized-context round trip: (pitch, onset ms, end ms, velocity)
                try:
                    dpath = os.path.join(args.out, f"{h}_s{seed}_b{j}_ctx_decoded.mid")
                    tok.detokenize(prompt).to_midi().save(dpath)
                    dec = read_notes(dpath)
                    want = collections.Counter((n[0], round(n[1] * 1000), round(n[2] * 1000), n[3]) for n in ctx)
                    got = collections.Counter((n[0], round(n[1] * 1000), round(n[2] * 1000), n[3]) for n in dec)
                    ctx_rt = {"missing": sum((want - got).values()), "extra": sum((got - want).values()),
                              "missing_examples": [list(x) for x in list((want - got).elements())[:4]],
                              "extra_examples": [list(x) for x in list((got - want).elements())[:4]]}
                except Exception as exc:
                    ctx_rt = {"error": str(exc)}
                start_t = (T - BLOCK) if prev_ready is None else max(T - BLOCK, prev_ready)
                ready_at = start_t + cost
                prev_ready = ready_at
                json.dump([str(x) for x in res[len(prompt):]], open(os.path.join(args.out, f"{h}_s{seed}_b{j}_gen.tokens.json"), "w"))
                blocks.append({"j": j, "quality": q, "T": round(T, 4), "q_ms": qj, "q_next_ms": qn, "grid_error_ms": round(qj - T * 1000, 1),
                               "chord": [g[0] for g in guides[-3:]], "prompt_tokens": len(prompt), "normalization": norm_log,
                               "ctx_roundtrip": ctx_rt, "unmatched_ctx_in_output": sum(left.values()),
                               "gen_notes": len(gen), "accepted": len(acc),
                               "dropped_after_block": sum(1 for n in gen if round(n[1] * 1000) >= qn),
                               "before_block": sum(1 for n in gen if round(n[1] * 1000) < qj),
                               "clipped_at_block_end": clipped, "clipped_duration_lost_s": round(lost, 3), "same_pitch_overlap": overlap,
                               "generated_same_onset_as_guide_pitch": sum(1 for n in gen if any(round(n[1] * 1000) == round(g[1] * 1000) and n[0] == g[0] for g in guides)),
                               "first_gen_onset_rel_q_ms": (min(round(n[1] * 1000) for n in gen) - qj) if gen else None,
                               "eos": str(tok.eos_tok) in res[len(prompt):], "cost_s": round(cost, 3),
                               "sim_start_rel_T": round(start_t - T, 3), "sim_ready_rel_T": round(ready_at - T, 3),
                               "miss": ready_at > T - MARGIN, "lateness_s": round(max(0.0, ready_at - (T - MARGIN)), 3),
                               "raw_generated": [list(n) for n in gen], "accepted_notes": [list(n) for n in acc]})
                print(h, seed, j, q, "acc", len(acc), "cost", round(cost, 3), "ready_rel_T", blocks[-1]["sim_ready_rel_T"],
                      "first_rel_q_ms", blocks[-1]["first_gen_onset_rel_q_ms"], "before", blocks[-1]["before_block"],
                      "rt", {k: ctx_rt.get(k) for k in ("missing", "extra")}, "norm", norm_log["counts"], flush=True)
            man["sequences"].append({"history": h, "seed": seed, "root_pc": root, "t0": t0, "blocks": blocks})
    json.dump(man, open(os.path.join(args.out, "manifest.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
