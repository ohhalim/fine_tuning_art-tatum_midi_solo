#!/usr/bin/env python3
"""Natural context vs chord primer on the same continuation (#1597).

docs/experiments/CONTEXT_MISMATCH_DIAG.md. For positions in unseen songs the
same real continuation (target, 2 s) is scored and continued under different
primers:

  N   real preceding 256 tokens           NC  N + guide of the estimated chord
  Ns  real preceding len(C) tokens         NX  N + guide of an unrelated chord (same length)
  C   chord guide (runtime-like)           M   another song's preceding 256 tokens

Measured: per-token NLL of the target (teacher forcing, leading velocity token
excluded) and, for N / Ns / C, free rollouts scored with the #1575 reappearance
measure against the real preceding 8 s. Only whether there is evidence for an
inference-input mismatch is judged; training insufficiency is not.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "music_transformer", ROOT / "music_transformer" / "third_party", ROOT / "scripts"):
    sys.path.insert(0, str(p))

NOTE_ON_END = 128
NOTE_OFF_END = 256
TS_START, TS_END = 256, 355
VEL_START, VEL_END = 356, 387
PREFIX = 256
TARGET_STEPS = 200          # 2 s
MAX_SEQ = 512               # the length the models were evaluated at
POSITIONS = 8
GUIDE_BPM = 128
GUIDE_SECONDS = 60.0 / GUIDE_BPM * 2
ESTIMATE_MIN_SHARE = 0.6
TEMPLATES = (("maj7", (0, 4, 7, 11)), ("7", (0, 4, 7, 10)), ("m7", (0, 3, 7, 10)),
             ("m7b5", (0, 3, 6, 10)), ("dim", (0, 3, 6, 9)))
PC = ("C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B")

SETS = {
    "dev": {"tatum": "data/tvm/holdout_tatum_val12/val", "base": "data/tvm/holdout_tatum_val12/val"},
    "eval": {"tatum": "data/tvm/holdout_tatum_fresh12/val", "base": "data/tvm/holdout_tatum_fresh12/val",
             "mehldau": "data/tvm/holdout_mehldau_val2/val"},
}
# Training files only; the sibling val/ folders are the selection sets (val12 is one of them).
TRAIN_DIRS = {"tatum": "data/tvm/tatum98/train", "mehldau": "data/tvm/mehldau16/train",
              "base": "data/jazz_full_notatum_nomehldau/train"}
SELECTION_DIRS = {"tatum": "data/tvm/tatum98/val", "mehldau": "data/tvm/mehldau16/val"}
CHECKPOINTS = {
    "tatum": "outputs/final_tatum/export/checkpoint_update518.pt",
    "base": "outputs/tvm/common_base/checkpoint_epoch8.pt",
    "mehldau": "outputs/clean_base/c2_export/checkpoint_update128.pt",
}


# ---- token state ------------------------------------------------------------------
def is_on(t):
    return 0 <= t < NOTE_ON_END


def is_off(t):
    return NOTE_ON_END <= t < NOTE_OFF_END


def is_ts(t):
    return TS_START <= t <= TS_END


def is_vel(t):
    return VEL_START <= t <= VEL_END


def grammar_errors(tokens) -> dict:
    """Duplicate note_on and orphan note_off counts of a token stream."""
    active, dup, orphan = set(), 0, 0
    for t in tokens:
        if is_on(t):
            dup += t in active
            active.add(t)
        elif is_off(t):
            p = t - NOTE_ON_END
            orphan += p not in active
            active.discard(p)
    return {"duplicate": dup, "orphan": orphan}


def states(tokens):
    """Per index: (active pitches before it, velocity token in force before it, steps before it)."""
    active, vel, steps, out = set(), None, 0, []
    for t in tokens:
        out.append((frozenset(active), vel, steps))
        if is_on(t):
            active.add(t)
        elif is_off(t):
            active.discard(t - NOTE_ON_END)
        elif is_ts(t):
            steps += t - TS_START + 1
        elif is_vel(t):
            vel = t
    return out


def sanitize(tokens) -> list[int]:
    """Drop note_offs whose note_on is outside the window and note_ons left open at the end."""
    active, kept = set(), []
    for t in tokens:
        if is_off(t):
            if t - NOTE_ON_END not in active:
                continue
            active.discard(t - NOTE_ON_END)
        elif is_on(t):
            if t in active:
                continue
            active.add(t)
        kept.append(t)
    if active:                                   # close at the end, no extra time
        kept += [p + NOTE_ON_END for p in sorted(active)]
    return kept


def strip_tail(tokens) -> list[int]:
    """Boundary rule: no time shift or velocity between primer and target."""
    t = list(tokens)
    while t and (is_ts(t[-1]) or is_vel(t[-1])):
        t.pop()
    return t


def tail(tokens, n) -> list[int]:
    from scripts.generate import truncate_tokens_preserving_velocity
    return sanitize(truncate_tokens_preserving_velocity(list(tokens), n)) if n > 0 else []


def tail_exact(tokens, n, slack: int = 12):
    """A sanitized tail of exactly ``n`` tokens, or None."""
    for m in range(n, n + slack + 1):
        t = tail(tokens, m)
        if len(t) == n:
            return t
    return None


# ---- target -------------------------------------------------------------------------
def extract_target(tokens, i, steps=TARGET_STEPS, st=None):
    """Target starting at index i: re-encoded notes of the next ``steps`` (10 ms), led by a velocity
    token. Requires no sounding note before i and a note_on at i. Returns (target, end index) or None."""
    import pretty_midi
    from scripts.generate import encode_notes_simple

    st = st if st is not None else states(tokens)
    active, vel, _ = st[i]
    if active or not is_on(tokens[i]) or vel is None:
        return None
    seg_notes, open_, cur, j = [], {}, 0, i
    v = vel - VEL_START
    while j < len(tokens) and cur < steps:
        t = tokens[j]
        if is_ts(t):
            cur += t - TS_START + 1
        elif is_vel(t):
            v = t - VEL_START
        elif is_on(t):
            if cur >= steps:
                break
            open_[t] = (cur, v)
        elif is_off(t) and t - NOTE_ON_END in open_:
            s, vv = open_.pop(t - NOTE_ON_END)
            seg_notes.append((s, min(cur, steps), t - NOTE_ON_END, vv))
        j += 1
    for p, (s, vv) in open_.items():
        seg_notes.append((s, min(cur, steps), p, vv))
    notes = [pretty_midi.Note(velocity=max(4, vv * 4), pitch=p, start=s / 100, end=max(e, s + 1) / 100)
             for s, e, p, vv in seg_notes if s < steps]
    if not notes:
        return None
    target = encode_notes_simple(sorted(notes, key=lambda n: (n.start, n.pitch)))
    if not target or not is_vel(target[0]):
        return None
    return target, j


def pick_positions(tokens, n=POSITIONS, prefix=PREFIX):
    """Up to n non-overlapping target starts after ``prefix`` tokens, spread over the song."""
    out, last_end = [], prefix
    span = len(tokens) - prefix
    st = states(tokens)
    if span <= 0:
        return out
    for k in range(n):
        want = prefix + int(span * k / n)
        i = max(want, last_end)
        while i < len(tokens):
            got = extract_target(tokens, i, st=st)
            if got is not None:
                out.append((i, got[0]))
                last_end = got[1]
                break
            i += 1
    return out


# ---- chords -------------------------------------------------------------------------
def estimate_chord(target):
    """Template chord covering the largest share of the target's note time, if that share is
    at least ESTIMATE_MIN_SHARE and root and third are present; else None."""
    from scripts.style_distance import tokens_to_notes

    notes = tokens_to_notes(target)
    if not notes:
        return None
    dur = {}
    for n in notes:
        dur[n.pitch % 12] = dur.get(n.pitch % 12, 0.0) + max(0.01, n.end - n.start)
    total = sum(dur.values())
    best, best_share = None, 0.0
    for name, tmpl in TEMPLATES:
        for root in range(12):
            pcs = {(root + k) % 12 for k in tmpl}
            if root not in dur or (root + tmpl[1]) % 12 not in dur:
                continue
            share = sum(v for p, v in dur.items() if p in pcs) / total
            if share > best_share + 1e-9:
                best, best_share = PC[root] + name, share
    return best if best_share >= ESTIMATE_MIN_SHARE else None


def chord_pcs(chord):
    from inference.app.fallback import parse_chord
    root, iv = parse_chord(chord)
    return {(root + k) % 12 for k in iv}


def unrelated_chord(chord):
    """Template chord sharing the fewest pitch classes with ``chord`` (deterministic)."""
    ref = chord_pcs(chord)
    best, best_common = None, 99
    for name, _ in TEMPLATES:
        for root in range(12):
            c = PC[root] + name
            common = len(chord_pcs(c) & ref)
            if common < best_common:
                best, best_common = c, common
    return best


def guide_tokens(chord):
    from inference.control.chord_primer import chord_guide_notes_for_duration
    from scripts.generate import encode_notes_simple, truncate_tokens_preserving_velocity

    notes = sorted(chord_guide_notes_for_duration(chord, bpm=GUIDE_BPM, seconds=GUIDE_SECONDS),
                   key=lambda n: (n.start, n.pitch))
    return truncate_tokens_preserving_velocity(encode_notes_simple(notes), 48)


# ---- one case -----------------------------------------------------------------------
def build_case(tokens, i, target, other_tokens, other_i):
    prefix = strip_tail(tokens[:i])
    case = {"i": i, "target_len": len(target), "status": {}}
    prim = {"N": tail(prefix, PREFIX), "M": tail(strip_tail(other_tokens[:other_i]), PREFIX)}
    chord = estimate_chord(target)
    case["chord"] = chord
    if chord is not None:
        g = guide_tokens(chord)
        x = guide_tokens(unrelated_chord(chord))
        case["unrelated"] = unrelated_chord(chord)
        prim["C"] = g
        ns = tail_exact(prefix, len(g))
        if ns is None:
            case["status"]["Ns"] = "length_mismatch"
        else:
            prim["Ns"] = ns
        prim["NC"] = prim["N"] + g
        if len(x) == len(g):
            prim["NX"] = prim["N"] + x
        else:
            case["status"]["NX"] = "length_mismatch"
    else:
        case["status"]["chord"] = "not_estimable"
    for k, p in list(prim.items()):
        if len(p) + len(target) > MAX_SEQ:
            case["status"][k] = "too_long"
            del prim[k]
        elif sum(grammar_errors(p + target).values()):
            case["status"][k] = "grammar"
            del prim[k]
    case["primers"] = prim
    return case


def target_nll(model, primer, target):
    import torch
    import torch.nn.functional as F

    seq = torch.tensor(list(primer) + list(target), dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        logits = model(seq[:, :-1])[0]
    y = seq[0, 1:]
    start = len(primer)                     # y index of target[1] is len(primer); target[0] is the velocity
    lp = F.log_softmax(logits[start:], dim=-1)
    ys = y[start:]
    return float(-lp.gather(1, ys.unsqueeze(1)).mean())


def top_line_from_tokens(tokens, origin_steps=0):
    from scripts.coherence_metrics import top_line
    from scripts.style_distance import tokens_to_notes
    return top_line([(n.start - origin_steps / 100, n.pitch) for n in tokens_to_notes(tokens)])


def session_line(tokens, i):
    """Top line of the real 8 s before index i, times relative to i."""
    st = states(tokens)
    steps_i = st[i][2] if i < len(st) else 0
    line = top_line_from_tokens(tokens[:i])
    return [(t - steps_i / 100, p) for t, p in line if -8.0 <= t - steps_i / 100 < 0]


def rollout_measures(gen_tokens, sline, chord):
    from scripts.coherence_metrics import top_line
    from scripts.motif_transfer_ab import reappearance
    from scripts.run_resident_model_probe import validate_generated_token_block
    from scripts.style_distance import tokens_to_notes

    notes = tokens_to_notes(gen_tokens)
    line = top_line([(n.start, n.pitch) for n in notes])
    r = reappearance(line, sline)
    ct = None
    if chord:
        pcs = chord_pcs(chord)
        ct = (sum(1 for n in notes if n.pitch % 12 in pcs), len(notes))
    return {**r, "notes": len(notes), "chord_tone": ct,
            "valid": bool(validate_generated_token_block(gen_tokens, lookahead_ms=TARGET_STEPS * 10,
                                                         allow_rest_bar=True)["valid"])}


# ---- aggregation -------------------------------------------------------------------
def bootstrap_ci(values, iters=4000, seed=0):
    import numpy as np
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return None
    if len(v) == 1:
        return [float(v[0]), float(v[0])]
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(iters, len(v)))].mean(1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def paired(cases, a, b, key="nll"):
    """Per-song mean of (a - b) over positions where both exist; then mean and song CI."""
    per_song = {}
    for c in cases:
        va, vb = c.get(key, {}).get(a), c.get(key, {}).get(b)
        if va is None or vb is None:
            continue
        per_song.setdefault(c["song"], []).append(va - vb)
    songs = [statistics.mean(v) for v in per_song.values()]
    return {"songs": len(songs), "positions": sum(len(v) for v in per_song.values()),
            "mean": statistics.mean(songs) if songs else None, "ci95": bootstrap_ci(songs)}


def rate(cases, cond, field):
    """Per-song ratio-of-sums of a reappearance field over rollouts, valid outputs only."""
    per_song = {}
    for c in cases:
        for r in c.get("rollouts", {}).get(cond, []):
            if not r["valid"]:
                continue
            s = per_song.setdefault(c["song"], [0, 0])
            s[0] += r[field]
            s[1] += r["windows"]
    return {song: (n / w if w else None) for song, (n, w) in per_song.items()}


def paired_rate(cases, a, b, field="variant"):
    ra, rb = rate(cases, a, field), rate(cases, b, field)
    diffs = [ra[s] - rb[s] for s in ra if s in rb and ra[s] is not None and rb[s] is not None]
    return {"songs": len(diffs), "mean": statistics.mean(diffs) if diffs else None, "ci95": bootstrap_ci(diffs)}


def verdict(summary) -> dict:
    """Fixed rule (plan): evidence for an inference-input mismatch, or not."""
    def pos(x):
        return x is not None and x["ci95"] is not None and x["ci95"][0] > 0
    prefix_effect = pos(summary["nll"]["M-N"])
    disturbance = pos(summary["nll"]["Ns-N"]) or pos(summary["nll"]["NX-N"])
    rollout = pos(summary["rollout_vr"]["N-Ns"])
    if prefix_effect and disturbance and rollout:
        label = "inference_mismatch_supported"
    elif not prefix_effect:
        label = "prefix_effect_unconfirmed"
    else:
        label = "undecided"
    return {"prefix_effect": prefix_effect, "short_or_insertion_disturbance": disturbance,
            "rollout_vr_N_over_Ns": rollout, "label": label,
            "note": "training insufficiency is not judged here; no result authorizes new training"}


def summarize(cases) -> dict:
    s = {"nll": {k: paired(cases, *k.split("-")) for k in ("M-N", "Ns-N", "NX-N", "NC-NX", "C-Ns", "NC-N")},
         "rollout_vr": {"N-Ns": paired_rate(cases, "N", "Ns"), "N-C": paired_rate(cases, "N", "C")},
         # descriptive only, not part of the verdict (VR is very sparse in rollouts)
         "rollout_ir": {"N-Ns": paired_rate(cases, "N", "Ns", "interval"),
                        "N-C": paired_rate(cases, "N", "C", "interval")}}
    statuses = {}
    for c in cases:
        for k, v in c["status"].items():
            statuses[f"{k}:{v}"] = statuses.get(f"{k}:{v}", 0) + 1
    s["statuses"] = statuses
    s["positions"] = len(cases)
    s["songs"] = len({c["song"] for c in cases})
    real = {}
    for c in cases:
        if c.get("real"):
            r = real.setdefault(c["song"], [0, 0])
            r[0] += c["real"]["variant"]
            r[1] += c["real"]["windows"]
    s["real_vr_by_song_mean"] = (statistics.mean([n / w for n, w in real.values() if w])
                                 if any(w for _, w in real.values()) else None)
    for cond in ("N", "Ns", "C"):
        for field, key in (("variant", "vr_mean"), ("interval", "ir_mean")):
            r = [v for v in rate(cases, cond, field).values() if v is not None]
            s.setdefault(key, {})[cond] = statistics.mean(r) if r else None
    s["real_ir_by_song_mean"] = None
    real_ir = {}
    for c in cases:
        if c.get("real"):
            r = real_ir.setdefault(c["song"], [0, 0])
            r[0] += c["real"]["interval"]
            r[1] += c["real"]["windows"]
    vals = [n / w for n, w in real_ir.values() if w]
    s["real_ir_by_song_mean"] = statistics.mean(vals) if vals else None
    return s


# ---- run ----------------------------------------------------------------------------
def file_hash(path):
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()


def run(set_name, models, out_dir, seeds=(1, 2), gen_tokens=320):
    import torch
    from scripts.generate import generate_once, load_model_with_lora
    from scripts.train_qlora import merge_lora_for_inference
    from scripts.validate_style_distance import load

    out_dir.mkdir(parents=True, exist_ok=True)
    report = {"schema": "context_mismatch_v1", "set": set_name, "style_verified": False,
              "musical_quality_verified": False, "models": {}}
    for m in models:
        song_dir = ROOT / SETS[set_name][m]
        files = sorted(song_dir.glob("*.npy"))
        def hashes(d):
            return {file_hash(f) for f in (ROOT / d).rglob("*.npy")} if d and (ROOT / d).exists() else None
        train_hashes, sel_hashes = hashes(TRAIN_DIRS.get(m)), hashes(SELECTION_DIRS.get(m))
        overlap = None if train_hashes is None else [f.name for f in files if file_hash(f) in train_hashes]
        sel_overlap = None if sel_hashes is None else [f.name for f in files if file_hash(f) in sel_hashes]
        songs = [(f.name, [int(t) for t in load(f)]) for f in files]
        ck = ROOT / CHECKPOINTS[m]
        model = load_model_with_lora(lora_path=str(ck.parent), checkpoint_path=str(ck),
                                     prefer_full_checkpoint=True, max_sequence=MAX_SEQ)
        merge_lora_for_inference(model)
        model.eval()
        cases, t0 = [], time.time()
        positions = {name: pick_positions(toks) for name, toks in songs}
        for si, (name, toks) in enumerate(songs):
            oname, otoks = songs[(si + 1) % len(songs)]
            opos = positions[oname]
            for pi, (i, target) in enumerate(positions[name]):
                other_i = opos[pi % len(opos)][0] if opos else PREFIX
                case = build_case(toks, i, target, otoks, other_i)
                case["song"] = name
                case["nll"] = {k: target_nll(model, p, target) for k, p in case["primers"].items()}
                sline = session_line(toks, i)
                from scripts.motif_transfer_ab import reappearance
                from scripts.coherence_metrics import top_line
                from scripts.style_distance import tokens_to_notes
                case["real"] = reappearance(top_line([(n.start, n.pitch) for n in tokens_to_notes(target)]), sline)
                case["rollouts"] = {}
                for cond in ("N", "Ns", "C"):
                    if cond not in case["primers"]:
                        continue
                    for seed in seeds:
                        torch.manual_seed(seed * 1000 + pi)
                        p = case["primers"][cond]
                        gen, _ = generate_once(model=model, primer=torch.tensor(p, dtype=torch.long),
                                               target_length=min(MAX_SEQ, len(p) + gen_tokens), strip_primer=True,
                                               temperature=1.0, top_k=32, top_p=0.95, grammar_mask=True,
                                               target_duration_seconds=TARGET_STEPS / 100, return_metadata=True,
                                               use_kv_cache=True)
                        case["rollouts"].setdefault(cond, []).append(
                            rollout_measures([int(t) for t in gen], sline, case["chord"]))
                case["primer_lens"] = {k: len(v) for k, v in case.pop("primers").items()}
                cases.append(case)
        summary = summarize(cases)
        summary["verdict"] = verdict(summary) if set_name == "eval" and m != "mehldau" else None
        report["models"][m] = {"checkpoint": str(ck), "songs": [n for n, _ in songs],
                               "train_overlap": overlap, "selection_overlap": sel_overlap,
                               "wall_s": round(time.time() - t0, 1),
                               "summary": summary}
        (out_dir / f"cases_{m}.json").write_text(json.dumps(cases) + "\n")
        print(m, json.dumps({"positions": summary["positions"], "M-N": summary["nll"]["M-N"]["mean"],
                             "Ns-N": summary["nll"]["Ns-N"]["mean"], "NX-N": summary["nll"]["NX-N"]["mean"],
                             "vr": summary["vr_mean"], "verdict": (summary["verdict"] or {}).get("label")}),
              flush=True)
        del model
    (out_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", choices=sorted(SETS), required=True)
    ap.add_argument("--models", default=None)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args(argv)
    models = args.models.split(",") if args.models else list(SETS[args.set])
    os.environ.setdefault("FORCE_CPU", "1")
    run(args.set, models, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
