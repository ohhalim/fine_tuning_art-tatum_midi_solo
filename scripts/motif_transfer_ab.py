#!/usr/bin/env python3
"""Does a motif from the session, put in the primer, come back varied? (#1575)

docs/experiments/MOTIF_TRANSFER.md (preregistered). Offline, fixed context: for
chosen half-bar blocks of an existing runtime session, the block is regenerated
three ways and only the new tokens are measured.

  A  [chord statement]                 the runtime primer
  B  [motif][chord statement]          latest 5-note top-line run of the two blocks before
  C  [shuffled motif][chord statement] same rhythm, velocities, first/last pitch, token count;
                                       interval order shuffled so no interval 3-gram is shared

Observations of past-pattern reappearance only: no quality, style or realtime
claim, and models are not compared with each other (different bases).
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import random
import statistics
import sys
import time
import zlib
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "music_transformer", ROOT / "music_transformer" / "third_party", ROOT / "scripts"):
    sys.path.insert(0, str(p))

from scripts.coherence_metrics import top_line  # noqa: E402

# ---- fixed design (docs/experiments/MOTIF_TRANSFER.md) -------------------------
MOTIF_NOTES = 5
MOTIF_IOI_S = (0.06, 0.6)
MOTIF_MAX_TOKENS = 24
CHORD_MAX_TOKENS = 48
SESSION_HORIZON_S = 8.0
WINDOW = 4
EXACT_TOL_S = 0.015
VARIANT_RATIO = (0.75, 1.25)
COPY_NOTES = 5
PITCH_RANGE = (21, 108)
GENERATION_TOKENS = 96
MAX_SEQUENCE = 192
BLOCKS = (6, 10, 14, 18, 22, 26, 30)
GATE_MIN_MEAN = 0.02
GATE_TOLERANCE = -0.05
GATE_COPY_CAP = 0.25

CHECKPOINTS = {
    "tatum": "outputs/final_tatum/export/checkpoint_update518.pt",
    "mehldau": "outputs/clean_base/c2_export/checkpoint_update128.pt",
    "base": "outputs/tvm/common_base/checkpoint_epoch8.pt",
}
_DEV_PROGS = {"iiVI": "ii_V_I_C", "blues": "blues_F", "minor": "minor_ii_V_i_C"}
_DEV_S43 = {"tatum": "outputs/ctx_hist/A_tatum_{tag}_s43", "mehldau": "outputs/ctx_hist/A_mehldau_{tag}_s43",
            "base": "outputs/motif_transfer/context/A_base_{tag}_s43"}
SETS = {
    # Development: contexts already seen in #1572 (seed 42 showcase, seed 43). Exploratory.
    "dev": {"generation_seeds": (1, 2), "min_positive_sessions": 5,
            "sessions": {m: [f"outputs/showcase_v1/runs/{d}_{m}" for d in _DEV_PROGS.values()]
                         + [_DEV_S43[m].format(tag=t) for t in _DEV_PROGS] for m in CHECKPOINTS}},
    # Evaluation: reserved, run only if every model passes dev with the candidate frozen.
    "eval": {"generation_seeds": (201, 202), "min_positive_sessions": 4,
             "sessions": {m: [f"outputs/motif_transfer/eval_context/{m}_{tag}_s{s}"
                              for tag in ("FiiVI", "GiiVIvi") for s in (61, 62)] for m in CHECKPOINTS}},
}


@dataclass(frozen=True)
class LineNote:
    onset: float      # onset of the cluster, as in coherence_metrics.top_line
    pitch: int        # its highest pitch
    end: float        # end of that highest note
    velocity: int | None


# ---- session material --------------------------------------------------------
def top_line_notes(notes, window_s: float = 0.05) -> list[LineNote]:
    """``notes``: (start, end, pitch, velocity). Same clusters and pitches as ``top_line``."""
    out, first, best = [], None, None
    for start, end, pitch, vel in sorted(notes, key=lambda n: (n[0], n[2])):
        if first is None or start - first > window_s:
            if first is not None:
                out.append(LineNote(first, best[0], best[1], best[2]))
            first, best = start, (pitch, end, vel)
        elif pitch > best[0]:
            best = (pitch, end, vel)
    if first is not None:
        out.append(LineNote(first, best[0], best[1], best[2]))
    return out


def session_notes(report: dict, midi_path: Path | None, tol_s: float = 0.005):
    """Played notes (absolute start, end, pitch, velocity) of a runtime report.

    Times come from ``played_bars`` (bar-aligned, as in coherence_metrics).
    ``played_bars`` has no velocity, so it is taken from ``played.mid``, which
    starts at the first dispatched note; unmatched notes get ``None``."""
    bar_s = 60.0 / report["bpm"] * report.get("beats_per_bar", 4)
    notes = sorted((i * bar_s + s, i * bar_s + e, int(p))
                   for i, b in enumerate(report["played_bars"]) for p, s, e in b["notes"])
    vel_of: dict[int, list[tuple[float, int]]] = {}
    if midi_path is not None and midi_path.exists() and notes:
        import pretty_midi

        origin = notes[0][0]
        for n in pretty_midi.PrettyMIDI(str(midi_path)).instruments[0].notes:
            vel_of.setdefault(int(n.pitch), []).append((origin + n.start, int(n.velocity)))
    out = []
    for start, end, pitch in notes:
        vel = next((v for t, v in vel_of.get(pitch, []) if abs(t - start) <= tol_s), None)
        out.append((start, end, pitch, vel))
    return out


def select_motif(line: list[LineNote], t_from: float, t_to: float, n: int = MOTIF_NOTES,
                 ioi=MOTIF_IOI_S) -> list[LineNote] | None:
    """Latest run of ``n`` consecutive top-line notes in [t_from, t_to) with every IOI in range."""
    inside = [x for x in line if t_from <= x.onset < t_to]
    for j in range(len(inside) - n, -1, -1):
        run = inside[j:j + n]
        if all(ioi[0] <= b.onset - a.onset <= ioi[1] for a, b in zip(run, run[1:])):
            return run
    return None


def motif_notes(run: list[LineNote], t_end: float):
    """The motif as a monophonic line from 0 s: each note ends by the next onset
    (last one by the block start ``t_end``) and lasts at least 20 ms."""
    import pretty_midi

    t0 = run[0].onset
    out = []
    for i, x in enumerate(run):
        limit = run[i + 1].onset if i + 1 < len(run) else t_end
        end = max(min(x.end, limit), x.onset + 0.02)
        out.append(pretty_midi.Note(velocity=int(x.velocity), pitch=int(x.pitch),
                                    start=round(x.onset - t0, 4), end=round(end - t0, 4)))
    return out


def intervals(pitches) -> tuple[int, ...]:
    return tuple(b - a for a, b in zip(pitches, pitches[1:]))


def grams(seq, n: int = 3) -> set:
    return {tuple(seq[i:i + n]) for i in range(len(seq) - n + 1)}


def shuffled_motif(notes, seed: int, pitch_range=PITCH_RANGE):
    """C's motif: same notes, interval order permuted.

    Every distinct permutation is tried in an order fixed by ``seed``; the first
    that differs from the original, shares no interval 3-gram with it and stays
    in the piano range wins. None when no permutation qualifies."""
    import pretty_midi

    pitches = [n.pitch for n in notes]
    iv = intervals(pitches)
    own = grams(iv)
    order = sorted(set(itertools.permutations(iv)))
    random.Random(seed).shuffle(order)
    for perm in order:
        if perm == iv or grams(perm) & own:
            continue
        new = list(itertools.accumulate(perm, initial=pitches[0]))
        if not all(pitch_range[0] <= p <= pitch_range[1] for p in new):
            continue
        return [pretty_midi.Note(velocity=n.velocity, pitch=p, start=n.start, end=n.end)
                for n, p in zip(notes, new)], perm
    return None


def case_seed(session: str, k: int) -> int:
    return zlib.crc32(f"{session}:{k}".encode())


# ---- metrics on new tokens only ------------------------------------------------
def windows(line, n: int = WINDOW):
    """(pitches, IOIs) of every run of ``n`` consecutive top-line notes."""
    out = []
    for i in range(len(line) - n + 1):
        seg = line[i:i + n]
        out.append((tuple(p for _, p in seg), tuple(b[0] - a[0] for a, b in zip(seg, seg[1:]))))
    return out


def degenerate(window) -> bool:
    """Same note repeated: every interval 0."""
    return all(i == 0 for i in intervals(window[0]))


def reappearance(gen_line, session_line, n: int = WINDOW) -> dict:
    """Session patterns (distinct interval n-grams) that reappear in the generated line.

    exact    same pitches, every IOI within 15 ms
    variant  same intervals (any transposition), every IOI ratio 0.75-1.25, not exact
    interval same intervals, any rhythm
    Each count is of distinct session patterns, so repeating one pattern adds at
    most 1 while the denominator (non-degenerate generated windows, repeats
    included) keeps growing. Degenerate windows are left out on both sides.
    A pattern found exactly is not also a variant (Astra review: one key could
    match one session window exactly and another as a variant)."""
    gen = [w for w in windows(gen_line, n) if not degenerate(w)]
    ses = [w for w in windows(session_line, n) if not degenerate(w)]
    exact, variant, interval = set(), set(), set()
    for gp, gd in gen:
        giv = intervals(gp)
        for sp, sd in ses:
            if intervals(sp) != giv:
                continue
            interval.add(giv)
            if gp == sp and all(abs(a - b) <= EXACT_TOL_S for a, b in zip(gd, sd)):
                exact.add(giv)
            elif all(VARIANT_RATIO[0] <= a / b <= VARIANT_RATIO[1] for a, b in zip(gd, sd)):
                variant.add(giv)
    variant -= exact
    return {"windows": len(gen), "distinct_windows": len({intervals(p) for p, _ in gen}),
            "exact": len(exact), "variant": len(variant), "interval": len(interval)}


def has_copy(gen_line, refs, n: int = COPY_NOTES) -> bool:
    """``n`` consecutive generated notes equal to ``n`` consecutive notes of a reference line
    (same pitches, IOIs within 15 ms)."""
    gen = windows(gen_line, n)
    for ref in refs:
        for rp, rd in windows(ref, n):
            for gp, gd in gen:
                if gp == rp and all(abs(a - b) <= EXACT_TOL_S for a, b in zip(gd, rd)):
                    return True
    return False


def chord_tone_counts(pitches, chord: str) -> tuple[int, int]:
    from inference.app.fallback import parse_chord

    root, iv = parse_chord(chord)
    pcs = {(root + k) % 12 for k in iv}
    pitches = list(pitches)
    return sum(1 for p in pitches if p % 12 in pcs), len(pitches)


# ---- aggregation and gate --------------------------------------------------------
def arm_rates(records) -> dict:
    """One arm of one session over one comparison's case list.

    Invalid outputs would be replaced by fallback at runtime, so the reappearance,
    copy and chord-tone measures use valid outputs only; validity keeps every
    generated case in its denominator. A rate whose denominator is 0 is None."""
    total = len(records)
    valid = [r for r in records if r["valid"]]
    win = sum(r["windows"] for r in valid)
    ct_hits = sum(r["chord_tone"][0] for r in valid)
    ct_tot = sum(r["chord_tone"][1] for r in valid)
    rate = lambda key: (sum(r[key] for r in valid) / win) if win else None
    return {"cases": total, "valid": len(valid), "valid_rate": len(valid) / total if total else None,
            "windows": win, "zero_window_valid": sum(1 for r in valid if r["windows"] == 0),
            "VR": rate("variant"), "XR": rate("exact"), "IR": rate("interval"),
            "repetition": (1 - sum(r["distinct_windows"] for r in valid) / win) if win else None,
            "chord_tone": ct_hits / ct_tot if ct_tot else None,
            "copies": sum(1 for r in valid if r["copy"]),
            "copy_rate": (sum(1 for r in valid if r["copy"]) / len(valid)) if valid else None}


COMPARISONS = {"BA": ("B", "A"), "BC": ("B", "C")}


def comparable(status: str, comparison: str) -> bool:
    """B-A needs a motif (B exists); B-C also needs a usable C."""
    if comparison == "BA":
        return status in ("ok", "c_unavailable", "token_mismatch")
    return status == "ok"


def summarize_sessions(cases: list[dict], records: list[dict]) -> dict:
    """Per model, per session, per comparison: both arms over the same case list."""
    by_case = {}
    for r in records:
        by_case.setdefault((r["case"], r["arm"]), []).append(r)
    out: dict = {}
    for c in cases:
        out.setdefault(c["model"], {}).setdefault(c["session"], {"statuses": {}, "cases": []})
        s = out[c["model"]][c["session"]]
        s["statuses"][c["status"]] = s["statuses"].get(c["status"], 0) + 1
        s["cases"].append(c)
    for model, sessions in out.items():
        for name, s in sessions.items():
            for comp, arms in COMPARISONS.items():
                ids = [c["id"] for c in s["cases"] if comparable(c["status"], comp)]
                s[comp] = {"case_count": len(ids)}
                for arm in arms:
                    s[comp][arm] = arm_rates([r for i in ids for r in by_case.get((i, arm), [])])
            del s["cases"]
    return out


def _diff(a, b):
    return None if a is None or b is None else a - b


def gate(sessions: dict, min_positive: int) -> dict:
    """One model. A session whose difference is undefined (no comparable case, or no
    window in an arm) counts as not positive; means use the defined sessions only,
    and none defined fails the criterion."""
    def paired(comp, arm_a, arm_b, key):
        return [_diff(s[comp][arm_a][key], s[comp][arm_b][key]) for s in sessions.values()]

    res = {"sessions": len(sessions)}
    for comp, (arm_a, arm_b) in COMPARISONS.items():
        d = paired(comp, arm_a, arm_b, "VR")
        defined = [x for x in d if x is not None]
        positive = sum(1 for x in defined if x > 0)
        mean = statistics.mean(defined) if defined else None
        res[f"VR_{comp}"] = {"diffs": d, "mean": mean, "positive": positive, "defined": len(defined),
                             "pass": mean is not None and mean >= GATE_MIN_MEAN and positive >= min_positive}
    for key in ("chord_tone", "valid_rate"):
        d = [x for x in paired("BA", "B", "A", key) if x is not None]
        mean = statistics.mean(d) if d else None
        res[f"{key}_BA"] = {"mean": mean, "defined": len(d), "pass": mean is not None and mean >= GATE_TOLERANCE}
    copies = sum(s["BA"]["B"]["copies"] for s in sessions.values())
    valid = sum(s["BA"]["B"]["valid"] for s in sessions.values())
    rate = copies / valid if valid else None
    res["copy_B"] = {"rate": rate, "copies": copies, "valid": valid,
                     "pass": rate is not None and rate <= GATE_COPY_CAP}
    res["pass"] = all(v["pass"] for k, v in res.items() if isinstance(v, dict))
    return res


# ---- generation ------------------------------------------------------------------
def prepare_case(model: str, session_dir: str, report: dict, notes, k: int) -> dict:
    """Everything about block ``k`` that does not depend on the generation seed.

    The top line is built only from notes that start before block ``k``: a 50 ms
    cluster opening just before the block could otherwise take its highest pitch
    from a note of block ``k`` itself (Astra review)."""
    from inference.control.chord_primer import chord_guide_notes_for_duration
    from scripts.generate import encode_notes_simple, truncate_tokens_preserving_velocity

    bpm = report["bpm"]
    block_s = 60.0 / bpm * 2
    chords = report["chords"]
    chord = chords[(k // 2) % len(chords)]
    t_k = k * block_s
    line = top_line_notes([n for n in notes if n[0] < t_k])
    chord_notes = sorted(chord_guide_notes_for_duration(chord, bpm=bpm, seconds=block_s),
                         key=lambda n: (n.start, n.pitch))
    case = {"id": f"{model}|{session_dir}|{k}", "model": model, "session": session_dir, "k": k,
            "chord": chord, "block_s": block_s,
            "chord_tokens": truncate_tokens_preserving_velocity(encode_notes_simple(chord_notes),
                                                                CHORD_MAX_TOKENS),
            "session_line": [(x.onset - t_k, x.pitch) for x in line
                             if t_k - SESSION_HORIZON_S <= x.onset < t_k]}
    run = select_motif(line, t_k - 2 * block_s, t_k)
    if run is None:
        return {**case, "status": "no_motif"}
    case["motif_source"] = [{"onset": round(x.onset, 4), "pitch": x.pitch, "end": round(x.end, 4),
                             "velocity": x.velocity} for x in run]
    if any(x.velocity is None for x in run):
        return {**case, "status": "velocity_unmatched"}
    b_notes = motif_notes(run, t_k)
    b_tokens = encode_notes_simple(b_notes)
    case["motif_B"] = [n.pitch for n in b_notes]
    case["motif_line_B"] = [(n.start, n.pitch) for n in b_notes]
    case["motif_tokens_B"] = b_tokens
    case["motif_chord_tone_B"] = chord_tone_counts(case["motif_B"], chord)
    if len(b_tokens) > MOTIF_MAX_TOKENS:
        return {**case, "status": "motif_too_long"}
    shuffled = shuffled_motif(b_notes, case_seed(session_dir, k))
    if shuffled is None:
        return {**case, "status": "c_unavailable"}
    c_notes, perm = shuffled
    c_tokens = encode_notes_simple(c_notes)
    case.update({"motif_C": [n.pitch for n in c_notes], "motif_line_C": [(n.start, n.pitch) for n in c_notes],
                 "motif_tokens_C": c_tokens, "c_intervals": list(perm),
                 "motif_chord_tone_C": chord_tone_counts([n.pitch for n in c_notes], chord)})
    if len(c_tokens) != len(b_tokens):
        return {**case, "status": "token_mismatch"}
    return {**case, "status": "ok"}


def primer_for(case: dict, arm: str) -> list[int]:
    if arm == "A":
        return list(case["chord_tokens"])
    return list(case[f"motif_tokens_{arm}"]) + list(case["chord_tokens"])


def arms_for(status: str) -> tuple[str, ...]:
    return {"ok": ("A", "B", "C"), "c_unavailable": ("A", "B"),
            "token_mismatch": ("A", "B")}.get(status, ("A",))


def measure(tokens, case: dict, arm: str) -> dict:
    from scripts.run_resident_model_probe import validate_generated_token_block
    from scripts.style_distance import tokens_to_notes

    notes = tokens_to_notes(tokens)
    line = top_line([(n.start, n.pitch) for n in notes])
    refs = [case["session_line"]] + ([case[f"motif_line_{arm}"]] if arm != "A" else [])
    valid = validate_generated_token_block(tokens, lookahead_ms=case["block_s"] * 1000,
                                           allow_rest_bar=True)["valid"]
    return {"valid": bool(valid), "notes": len(notes), "line": line,
            **reappearance(line, case["session_line"]),
            "copy": has_copy(line, refs),
            "chord_tone": chord_tone_counts([n.pitch for n in notes], case["chord"])}


def run(set_name: str, models, out_dir: Path, *, dry_run: bool = False) -> int:
    spec = SETS[set_name]
    out_dir.mkdir(parents=True, exist_ok=True)
    cases, records = [], []
    rec_path = out_dir / "records.jsonl"
    rec_file = None if dry_run else rec_path.open("w")
    started = time.time()
    for model_name in models:
        model = None
        if not dry_run:
            import torch
            from scripts.generate import generate_once, load_model_with_lora
            from scripts.train_qlora import merge_lora_for_inference

            ckpt = ROOT / CHECKPOINTS[model_name]
            model = load_model_with_lora(lora_path=str(ckpt.parent), checkpoint_path=str(ckpt),
                                         prefer_full_checkpoint=True, max_sequence=MAX_SEQUENCE)
            merge_lora_for_inference(model)
        for session_dir in spec["sessions"][model_name]:
            d = ROOT / session_dir
            report = json.loads((d / "continuous_report.json").read_text())
            notes = session_notes(report, d / "played.mid")
            n_blocks = report["bars"] * 2
            for k in BLOCKS:
                if k >= n_blocks:
                    continue
                case = prepare_case(model_name, session_dir, report, notes, k)
                cases.append(case)
                if dry_run:
                    continue
                for gen_seed in spec["generation_seeds"]:
                    for arm in arms_for(case["status"]):
                        primer = primer_for(case, arm)
                        torch.manual_seed(gen_seed * 1000 + k)
                        t0 = time.perf_counter()
                        tokens, _meta = generate_once(
                            model=model, primer=torch.tensor(primer, dtype=torch.long),
                            target_length=min(MAX_SEQUENCE, len(primer) + GENERATION_TOKENS),
                            strip_primer=True, temperature=1.0, top_k=32, top_p=0.95,
                            grammar_mask=True, target_duration_seconds=case["block_s"],
                            return_metadata=True, use_kv_cache=True)
                        ms = (time.perf_counter() - t0) * 1000
                        tokens = [int(t) for t in tokens]
                        rec = {"case": case["id"], "model": model_name, "session": session_dir, "k": k,
                               "seed": gen_seed, "arm": arm, "primer_len": len(primer),
                               "tokens": tokens, "generation_ms": round(ms, 1),
                               **measure(tokens, case, arm)}
                        records.append(rec)
                        rec_file.write(json.dumps(rec) + "\n")
                        rec_file.flush()
        del model
    if rec_file is not None:
        rec_file.close()
    statuses: dict = {}
    for c in cases:
        statuses.setdefault(c["model"], {}).setdefault(c["status"], 0)
        statuses[c["model"]][c["status"]] += 1
    (out_dir / "cases.json").write_text(json.dumps(cases, indent=1) + "\n")
    summary = {"schema": "motif_transfer_v1", "set": set_name, "exploratory": set_name == "dev",
               "style_verified": False, "musical_quality_verified": False,
               "generation_seeds": list(spec["generation_seeds"]), "statuses": statuses,
               "generations": len(records), "wall_s": round(time.time() - started, 1)}
    if not dry_run:
        sessions = summarize_sessions(cases, records)
        summary["sessions"] = sessions
        summary["gate"] = {m: gate(sessions[m], spec["min_positive_sessions"]) for m in sessions}
        ms = [r["generation_ms"] for r in records]
        summary["generation_ms"] = {"p50": statistics.median(ms), "max": max(ms)} if ms else None
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"statuses": statuses, "generations": len(records), "wall_s": summary["wall_s"]}))
    for m, g in summary.get("gate", {}).items():
        fmt = lambda v: "-" if v is None else f"{v:+.3f}"
        print(f"{m:8s} VR B-C {fmt(g['VR_BC']['mean'])} ({g['VR_BC']['positive']}/{g['sessions']}) "
              f"VR B-A {fmt(g['VR_BA']['mean'])} ({g['VR_BA']['positive']}/{g['sessions']}) "
              f"chord-tone {fmt(g['chord_tone_BA']['mean'])} valid {fmt(g['valid_rate_BA']['mean'])} "
              f"copy {fmt(g['copy_B']['rate'])} -> {'PASS' if g['pass'] else 'FAIL'}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", choices=sorted(SETS), required=True)
    ap.add_argument("--models", default="tatum,mehldau,base")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true", help="build the case list and statuses only, no model")
    args = ap.parse_args(argv)
    models = [m for m in args.models.split(",") if m]
    unknown = [m for m in models if m not in CHECKPOINTS]
    if unknown:
        ap.error(f"unknown model(s): {unknown}")
    os.environ.setdefault("FORCE_CPU", "1")
    return run(args.set, models, args.output_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
