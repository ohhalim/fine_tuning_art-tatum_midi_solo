#!/usr/bin/env python3
"""Pick one of N model candidates per block: original vs random vs fixed ranker (docs/experiments/CANDIDATE_SELECT.md).

Offline, the runtime's default block call (guide only) on held-out progressions.
Per block and seed, N=3 candidates from the same model and primer (seeds differ):
  A  candidate 0 (what the runtime plays today)
  B  one candidate picked at random (fixed RNG)
  C  the fixed ranker's pick: among valid candidates with >= MIN_NOTES solo notes,
     the highest fit - clash against the chord's pitch classes (ties -> lowest index);
     if none qualifies, candidate 0.
The ranker score improving is expected by construction and is not evidence of success;
the independent checks below are what the verdict uses. Raw candidates are kept.
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

from scripts.guide_swap_eval import BLOCK_S, BPM, GEN_TOKENS, MAX_SEQ, solo_of  # noqa: E402

PROGRESSIONS = {"iiVI_F": ["Gm7", "C7", "Fmaj7", "Fmaj7"], "rhythm_A": ["Bbmaj7", "G7", "Cm7", "F7"],
                "minor_A": ["Bm7b5", "E7", "Am7", "Am7"]}
SEEDS = range(40)
N = 3
MIN_NOTES = 3
SEED_STRIDE = 1000
CHECKPOINT = "outputs/bebop_rh/export/checkpoint_update516.pt"
BEAT_S = 60.0 / BPM


def score(solo, pcs) -> float | None:
    from scripts.harmony_controls import window_metrics
    if len(solo) < MIN_NOTES:
        return None
    m = window_metrics(solo, pcs)
    return m["fit"] - m["clash"]


def rank_pick(cands, pcs) -> int:
    best, best_i = None, 0
    for i, c in enumerate(cands):
        s = score(c["solo"], pcs) if c["valid"] else None
        if s is not None and (best is None or s > best):
            best, best_i = s, i
    return best_i


def independent(blocks, pcs_of) -> dict:
    """Checks the ranker does not optimise directly. blocks: list of (solo, chord)."""
    from scripts.harmony_controls import window_metrics
    n_notes = empty = one = 0
    same, steps, leaps, beat_clash, beat_n, grams = 0, 0, 0, 0, 0, []
    for solo, chord in blocks:
        pcs = pcs_of(chord)
        n_notes += len(solo)
        empty += int(not solo)
        one += int(len(solo) == 1)
        for (p, s, _), (q, _, _) in zip(solo, solo[1:]):
            steps += 1
            same += int(p == q)
            leaps += int(abs(q - p) > 7)
        for p, s, _ in solo:
            if min(abs(s - k * BEAT_S) for k in range(3)) <= 0.03:      # on a beat (0, 1, 2 within the block)
                beat_n += 1
                beat_clash += int(window_metrics([(p, 0.0, 0.1)], pcs)["clash"] > 0)
        grams.append(tuple(q - p for (p, _, _), (q, _, _) in zip(solo, solo[1:]))[:3])
    total = len(blocks)
    return {"blocks": total, "notes_per_s": n_notes / (total * BLOCK_S), "empty_rate": empty / total,
            "one_note_rate": one / total, "same_note_share": same / steps if steps else None,
            "leap_share": leaps / steps if steps else None,
            "on_beat_clash": beat_clash / beat_n if beat_n else None,
            "distinct_opening_intervals": len(set(grams)) / total}


def judge(a: dict, b: dict, c: dict, rank_gain: float) -> dict:
    ok = {
        "rank_gain_ge_0.05": rank_gain >= 0.05,
        "density_within_20pct_of_A": abs(c["notes_per_s"] - a["notes_per_s"]) <= 0.2 * a["notes_per_s"],
        "same_note_le_A_plus_0.05": c["same_note_share"] <= a["same_note_share"] + 0.05,
        "empty_le_A": c["empty_rate"] <= a["empty_rate"],
        "leaps_le_A_plus_0.05": c["leap_share"] <= a["leap_share"] + 0.05,
        "distinct_ge_0.9_B": c["distinct_opening_intervals"] >= 0.9 * b["distinct_opening_intervals"],
        "on_beat_clash_below_B": c["on_beat_clash"] < b["on_beat_clash"],
    }
    return {**ok, "pass": all(ok.values())}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args(argv)
    os.environ.setdefault("FORCE_CPU", "1")
    import torch
    from inference.control.chord_primer import chord_guide_notes_for_duration
    from inference.control.harmony_contract import chord_pcs
    from scripts.generate import encode_notes_simple, generate_once, load_model_with_lora
    from scripts.run_resident_model_probe import validate_generated_token_block
    from scripts.train_qlora import merge_lora_for_inference

    model = load_model_with_lora(lora_path=str((ROOT / CHECKPOINT).parent), checkpoint_path=str(ROOT / CHECKPOINT),
                                 prefer_full_checkpoint=True, max_sequence=MAX_SEQ)
    merge_lora_for_inference(model)
    model.eval()
    pcs_of = lambda c: chord_pcs(c)[1]
    rng = random.Random(0)
    raw, arms = [], {"A": [], "B": [], "C": []}
    gains = []
    for prog, chords in PROGRESSIONS.items():
        for chord in dict.fromkeys(chords):
            primer = encode_notes_simple(sorted(chord_guide_notes_for_duration(chord, bpm=BPM, seconds=BLOCK_S),
                                                key=lambda n: (n.start, n.pitch)))
            for seed in SEEDS:
                cands = []
                for k in range(N):
                    torch.manual_seed(seed + k * SEED_STRIDE)
                    gen, _ = generate_once(model=model, primer=torch.tensor(primer, dtype=torch.long),
                                           target_length=min(MAX_SEQ, len(primer) + GEN_TOKENS), strip_primer=True,
                                           temperature=1.0, top_k=32, top_p=0.95, grammar_mask=True,
                                           target_duration_seconds=BLOCK_S, return_metadata=True, use_kv_cache=True)
                    gen = [int(t) for t in gen]
                    solo, low = solo_of(gen)
                    valid = bool(validate_generated_token_block(gen, lookahead_ms=BLOCK_S * 1000,
                                                                allow_rest_bar=True)["valid"])
                    cands.append({"tokens": gen, "solo": solo, "valid": valid, "low_removed": low})
                pick_b, pick_c = rng.randrange(N), rank_pick(cands, pcs_of(chord))
                for arm, i in (("A", 0), ("B", pick_b), ("C", pick_c)):
                    arms[arm].append((cands[i]["solo"], chord))
                sb, sc = score(cands[pick_b]["solo"], pcs_of(chord)), score(cands[pick_c]["solo"], pcs_of(chord))
                if sb is not None and sc is not None:
                    gains.append(sc - sb)
                raw.append({"progression": prog, "chord": chord, "seed": seed, "pick_b": pick_b, "pick_c": pick_c,
                            "scores": [score(c["solo"], pcs_of(chord)) for c in cands], "candidates": cands})
    res = {arm: independent(blocks, pcs_of) for arm, blocks in arms.items()}
    rank_gain = sum(gains) / len(gains)
    report = {"schema": "candidate_select_eval_v1", "checkpoint": CHECKPOINT, "n": N, "seeds": len(SEEDS),
              "progressions": PROGRESSIONS, "rank_gain_vs_B": rank_gain,
              "c_picked_candidate0_rate": sum(r["pick_c"] == 0 for r in raw) / len(raw),
              "independent": res, "verdict": judge(res["A"], res["B"], res["C"], rank_gain),
              "musical_quality_verified": False}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output_dir / "raw.json").write_text(json.dumps(raw))
    print(json.dumps({k: report[k] for k in ("rank_gain_vs_B", "c_picked_candidate0_rate", "independent", "verdict")},
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
