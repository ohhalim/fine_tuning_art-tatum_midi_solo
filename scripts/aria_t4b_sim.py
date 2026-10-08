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
TPB, TEMPO = 480, 500000            # 120 BPM: 1 s = 960 ticks
QUAL = {"maj7": (4, 11), "m7": (3, 10)}
ORDER = {1: ["maj7", "m7", "maj7", "m7"], 2: ["m7", "maj7", "m7", "maj7"]}


def write_notes(notes, path):
    """notes: [(pitch, start, end, velocity)] -> single-track MIDI at 120 BPM."""
    ev = []
    for p, s, e, v in notes:
        ev.append((round(s * 960), 1, p, v))
        ev.append((round(e * 960), 0, p, 0))
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
    key = lambda n: (n[0], round(n[1], 2))
    for h in args.histories:
        v = plan[h]
        hist = read_notes(f"{args.d}/out/{v['history']}_prompt_decoded.mid")
        root, t0 = v["root_pc"], v["t_c"]
        for seed in args.seeds:
            accepted, guides, blocks, prev_ready = [], [], [], None
            for j, q in enumerate(ORDER[seed]):
                T = t0 + j * BLOCK
                third, seventh = QUAL[q]
                guides += [(36 + root, T, T + BLOCK, 60), (48 + (root + third) % 12, T, T + BLOCK, 60), (48 + (root + seventh) % 12, T, T + BLOCK, 60)]
                ctx = hist + accepted + guides
                ts = time.time()
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
                in_block = [n for n in gen if T - 1e-3 <= n[1] < T + BLOCK]
                acc = [(p, s, min(e, T + BLOCK), vv) for p, s, e, vv in in_block if min(e, T + BLOCK) > s]
                cost = time.time() - ts
                clipped = sum(1 for n in in_block if n[2] > T + BLOCK)
                lost = sum(max(0.0, n[2] - (T + BLOCK)) for n in in_block)
                # same-pitch overlap inside the accepted block (measured, not repaired)
                overlap = sum(1 for a in acc for b in acc if a is not b and a[0] == b[0] and a[1] < b[1] < a[2])
                accepted += acc
                # context preservation: history + accepted notes after the tokenizer round trip (pitch, onset, end within 10 ms, velocity)
                dec_ctx = None
                try:
                    dpath = os.path.join(args.out, f"{h}_s{seed}_b{j}_ctx_decoded.mid")
                    tok.detokenize(prompt).to_midi().save(dpath)
                    dec_ctx = read_notes(dpath)
                except Exception as exc:                  # recorded, not hidden
                    dec_ctx = str(exc)
                if isinstance(dec_ctx, list):
                    want = sorted((n[0], round(n[1], 2), round(n[2], 2), n[3]) for n in ctx)
                    got = sorted((n[0], round(n[1], 2), round(n[2], 2), n[3]) for n in dec_ctx)
                    cw, cg = collections.Counter(want), collections.Counter(got)
                    ctx_preserved = {"missing": sum((cw - cg).values()), "extra": sum((cg - cw).values())}
                else:
                    ctx_preserved = {"error": dec_ctx}
                start = (T - BLOCK) if prev_ready is None else max(T - BLOCK, prev_ready)
                ready_at = start + cost
                prev_ready = ready_at
                json.dump([str(x) for x in res[len(prompt):]], open(os.path.join(args.out, f"{h}_s{seed}_b{j}_gen.tokens.json"), "w"))
                blocks.append({"j": j, "quality": q, "T": round(T, 3), "chord": [g[0] for g in guides[-3:]],
                               "prompt_tokens": len(prompt), "ctx_preserved": ctx_preserved, "unmatched_ctx_in_output": sum(left.values()),
                               "gen_notes": len(gen), "accepted": len(acc), "dropped_after_block": sum(1 for n in gen if n[1] >= T + BLOCK),
                               "before_block": sum(1 for n in gen if n[1] < T - 1e-3), "clipped_at_block_end": clipped,
                               "clipped_duration_lost_s": round(lost, 3), "same_pitch_overlap": overlap,
                               "guide_pitch_also_generated_same_onset": sum(1 for n in gen if any(abs(n[1] - g[1]) < 0.01 and n[0] == g[0] for g in guides)),
                               "first_gen_onset_rel": round(min((n[1] for n in gen), default=float("nan")) - T, 3),
                               "eos": str(tok.eos_tok) in res[len(prompt):], "cost_s": round(cost, 3),
                               "sim_start_rel_T": round(start - T, 3), "sim_ready_rel_T": round(ready_at - T, 3),
                               "miss": ready_at > T - MARGIN, "lateness_s": round(max(0.0, ready_at - (T - MARGIN)), 3),
                               "raw_generated": [list(n) for n in gen], "accepted_notes": [list(n) for n in acc]})
                print(h, seed, j, q, "acc", len(acc), "cost", round(cost, 3), "ready_rel_T", blocks[-1]["sim_ready_rel_T"], "first", blocks[-1]["first_gen_onset_rel"], "ctx", ctx_preserved, flush=True)
            man["sequences"].append({"history": h, "seed": seed, "root_pc": root, "t0": t0, "blocks": blocks})
    json.dump(man, open(os.path.join(args.out, "manifest.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
