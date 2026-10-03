#!/usr/bin/env python3
"""Runtime guide, old vs contract, on the same bebop checkpoint (docs/experiments/GUIDE_SWAP.md, #1635).

Offline, the runtime's default generation call for one half-bar block (guide only:
no input, history, comp or breath). For each chord and seed, the block is generated
after the old guide (chord_guide_notes_for_duration) and after the contract guide
(harmony_contract). Raw tokens, guides and validity are kept per sample.

2x2 response for fixed chord pairs (A, B): y_A after guide(A), y_B after guide(B), same
seed; diagonal response = [fit(y_A,A) - fit(y_A,B) + fit(y_B,B) - fit(y_B,A)] / 2,
scored against the chord symbols' pitch classes. A fit-proxy response, not a quality claim.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "music_transformer", ROOT / "music_transformer" / "third_party", ROOT / "scripts"):
    sys.path.insert(0, str(p))

BPM = 128
BLOCK_S = 60.0 / BPM * 2                         # half bar
GEN_TOKENS, MAX_SEQ = 96, 192                    # run_continuous_jazz defaults
CHORDS = ["Dm7", "G7", "Cmaj7", "F7", "Bb7", "Gm7", "C7", "Dm7b5", "Cm7"]
PAIRS = [("Dm7", "Dm7b5"), ("G7", "Gm7"), ("Cmaj7", "Cm7"), ("C7", "Cmaj7")]
SEEDS = range(40)
CHECKPOINT = "outputs/bebop_rh/export/checkpoint_update516.pt"


def guides(chord: str) -> dict:
    from inference.control.chord_primer import chord_guide_notes_for_duration
    from inference.control.harmony_contract import chord_pcs, guide_notes

    return {"old": chord_guide_notes_for_duration(chord, bpm=BPM, seconds=BLOCK_S),
            "new": guide_notes(*chord_pcs(chord), BLOCK_S)}


def solo_of(tokens):
    from inference.control.harmony_contract import SOLO_SPLIT
    from inference.control.solo_line import top_notes
    from scripts.style_distance import tokens_to_notes

    notes = tokens_to_notes(tokens)
    line = top_notes(notes)
    solo = [(n.pitch, n.start, min(n.end, BLOCK_S)) for n in line if n.pitch >= SOLO_SPLIT and n.start < BLOCK_S]
    low = sum(1 for n in notes if n.pitch < SOLO_SPLIT)
    return solo, low


def fit(solo, pcs) -> float | None:
    from scripts.harmony_controls import window_metrics
    return window_metrics(solo, pcs)["fit"] if solo else None


def diag(rows_a, rows_b, pcs_a, pcs_b) -> dict:
    """Per-seed diagonal response and same-output rate for one pair and one guide kind."""
    vals, same = [], 0
    for ra, rb in zip(rows_a, rows_b):
        same += int(ra["tokens"] == rb["tokens"])
        fa_a, fa_b, fb_b, fb_a = fit(ra["solo"], pcs_a), fit(ra["solo"], pcs_b), fit(rb["solo"], pcs_b), fit(rb["solo"], pcs_a)
        if None in (fa_a, fa_b, fb_b, fb_a):
            continue
        vals.append(((fa_a - fa_b) + (fb_b - fb_a)) / 2)
    rng = random.Random(0)
    boots = sorted(sum(rng.choice(vals) for _ in vals) / len(vals) for _ in range(2000)) if vals else []
    return {"n": len(vals), "mean": sum(vals) / len(vals) if vals else None,
            "ci95": [boots[50], boots[1949]] if boots else None, "same_output_rate": same / len(rows_a)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args(argv)
    os.environ.setdefault("FORCE_CPU", "1")
    import torch
    from inference.control.harmony_contract import chord_pcs
    from scripts.generate import encode_notes_simple, generate_once, load_model_with_lora
    from scripts.run_resident_model_probe import validate_generated_token_block
    from scripts.harmony_controls import window_metrics
    from scripts.train_qlora import merge_lora_for_inference

    model = load_model_with_lora(lora_path=str((ROOT / CHECKPOINT).parent), checkpoint_path=str(ROOT / CHECKPOINT),
                                 prefer_full_checkpoint=True, max_sequence=MAX_SEQ)
    merge_lora_for_inference(model)
    model.eval()
    samples = {}
    for chord in CHORDS:
        g = guides(chord)
        for kind, notes in g.items():
            primer = encode_notes_simple(sorted(notes, key=lambda n: (n.start, n.pitch)))
            rows = []
            for seed in SEEDS:
                torch.manual_seed(seed)
                gen, _ = generate_once(model=model, primer=torch.tensor(primer, dtype=torch.long),
                                       target_length=min(MAX_SEQ, len(primer) + GEN_TOKENS), strip_primer=True,
                                       temperature=1.0, top_k=32, top_p=0.95, grammar_mask=True,
                                       target_duration_seconds=BLOCK_S, return_metadata=True, use_kv_cache=True)
                gen = [int(t) for t in gen]
                solo, low = solo_of(gen)
                valid = bool(validate_generated_token_block(gen, lookahead_ms=BLOCK_S * 1000, allow_rest_bar=True)["valid"])
                rows.append({"seed": seed, "tokens": gen, "valid": valid, "solo": solo, "low_notes_removed": low})
            samples[(chord, kind)] = {"guide": sorted(n.pitch for n in notes), "rows": rows}
    # descriptive stats per guide kind, against each chord's own pitch classes
    desc = {}
    for kind in ("old", "new"):
        fits, clashes, thirds, n_notes, empty, invalid, low = [], [], [], 0, 0, 0, 0
        for chord in CHORDS:
            root, pcs = chord_pcs(chord)
            third = next(pc for pc in pcs if (pc - root) % 12 in (3, 4))
            for r in samples[(chord, kind)]["rows"]:
                invalid += int(not r["valid"])
                low += r["low_notes_removed"]
                if not r["solo"]:
                    empty += 1
                    continue
                m = window_metrics(r["solo"], pcs)
                fits.append(m["fit"])
                clashes.append(m["clash"])
                dur = sum(max(e - s, 0.01) for _, s, e in r["solo"])
                thirds.append(sum(max(e - s, 0.01) for p, s, e in r["solo"] if p % 12 == third) / dur)
                n_notes += len(r["solo"])
        total = len(CHORDS) * len(SEEDS)
        desc[kind] = {"samples": total, "invalid_rate": invalid / total, "empty_rate": empty / total,
                      "low_notes_removed_per_block": low / total, "notes_per_s": n_notes / (total * BLOCK_S),
                      "fit": sum(fits) / len(fits), "clash": sum(clashes) / len(clashes),
                      "third_share": sum(thirds) / len(thirds)}
    response = {}
    for a, b in PAIRS:
        pa, pb = chord_pcs(a)[1], chord_pcs(b)[1]
        response[f"{a}/{b}"] = {kind: diag(samples[(a, kind)]["rows"], samples[(b, kind)]["rows"], pa, pb)
                                for kind in ("old", "new")}
        response[f"{a}/{b}"]["guides"] = {kind: [samples[(a, kind)]["guide"], samples[(b, kind)]["guide"]]
                                          for kind in ("old", "new")}
    report = {"schema": "guide_swap_eval_v1", "checkpoint": CHECKPOINT, "block_s": BLOCK_S,
              "generation_tokens": GEN_TOKENS, "max_sequence": MAX_SEQ, "seeds": len(SEEDS),
              "descriptive": desc, "response_2x2": response, "musical_quality_verified": False}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output_dir / "samples.json").write_text(json.dumps(
        [{"chord": c, "guide_kind": k, **{kk: v for kk, v in s.items() if kk != "rows"}, "rows": s["rows"]}
         for (c, k), s in samples.items()]))
    print(json.dumps({"descriptive": desc, "response_2x2": {p: {k: v[k] for k in ("old", "new")}
                                                            for p, v in response.items()}}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
