#!/usr/bin/env python3
"""Bebop right-hand adapter vs the base on unseen right-hand lines (docs/experiments/BEBOP_RH_ADAPTER.md, #1618).

From test songs of ``data/bebop_rh``: natural 256-token right-hand context,
free 4 s continuation, compared with the real next 4 s on density, rests and
phrase length. Observations only: no style or quality claim.
"""
from __future__ import annotations

import argparse
import glob
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

WINDOW_S = 4.0
REST_S = 0.3
GEN_TOKENS = 256
MAX_SEQ = 512
SEEDS = (1, 2)
BASE = "outputs/tvm/common_base/checkpoint_epoch8.pt"


def window_stats(notes, start: float, length: float = WINDOW_S) -> dict:
    """Notes (start, end) of one window, times absolute; rests include silence at the window edges."""
    notes = sorted((s, e) for s, e in notes if start <= s < start + length)
    gaps, lens, cur = [], [], 0
    prev_end = start
    for s, e in notes:
        g = s - prev_end
        if g >= REST_S:
            gaps.append(g)
            if cur:
                lens.append(cur)
            cur = 0
        cur += 1
        prev_end = max(prev_end, e)
    tail_gap = start + length - prev_end
    if tail_gap >= REST_S:
        gaps.append(tail_gap)
    if cur:
        lens.append(cur)
    return {"notes": len(notes), "rest_time": sum(gaps), "rests": len(gaps), "phrases": lens, "seconds": length}


def pooled(rows) -> dict:
    secs = sum(r["seconds"] for r in rows)
    phrases = [x for r in rows for x in r["phrases"]]
    return {"notes_per_s": sum(r["notes"] for r in rows) / secs if secs else None,
            "rest_share": sum(r["rest_time"] for r in rows) / secs if secs else None,
            "rests_per_10s": sum(r["rests"] for r in rows) / secs * 10 if secs else None,
            "phrase_median": statistics.median(phrases) if phrases else None, "windows": len(rows)}


def judge(real: dict, adapter: dict, base: dict, valid_rate: float) -> dict:
    nps = adapter["notes_per_s"] / real["notes_per_s"] if real["notes_per_s"] else None
    ok = {
        "density_0.67_1.5x": nps is not None and 0.67 <= nps <= 1.5,
        "rest_half_of_real": adapter["rest_share"] is not None and adapter["rest_share"] >= 0.5 * real["rest_share"],
        "phrase_at_most_2x": adapter["phrase_median"] is not None and real["phrase_median"] is not None
        and adapter["phrase_median"] <= 2 * real["phrase_median"],
        "valid_0.9": valid_rate >= 0.9,
        "density_closer_than_base": all(x["notes_per_s"] for x in (adapter, base, real))
        and abs(math.log(adapter["notes_per_s"] / real["notes_per_s"]))
        < abs(math.log(base["notes_per_s"] / real["notes_per_s"])),
    }
    return {**ok, "density_ratio": nps, "pass": all(ok.values())}


def run(adapter_ckpt: str, out_dir: Path) -> dict:
    import torch
    from scripts.context_mismatch_diag import pick_positions, states, strip_tail, tail
    from scripts.generate import generate_once, load_model_with_lora
    from scripts.run_resident_model_probe import validate_generated_token_block
    from scripts.style_distance import tokens_to_notes
    from scripts.train_qlora import merge_lora_for_inference
    from inference.control.solo_line import top_notes
    from scripts.validate_style_distance import load

    out_dir.mkdir(parents=True, exist_ok=True)
    songs = [(f.name, [int(t) for t in load(f)]) for f in sorted((ROOT / "data/bebop_rh/test").glob("*.npy"))]
    positions = {n: pick_positions(t) for n, t in songs}
    real_rows, results = [], {}
    for name, toks in songs:
        notes = [(n.start, n.end) for n in tokens_to_notes(toks)]
        st = states(toks)
        for i, _ in positions[name]:
            real_rows.append(window_stats(notes, st[i][2] / 100))
    for label, ck in (("base", BASE), ("adapter", adapter_ckpt)):
        model = load_model_with_lora(lora_path=str((ROOT / ck).parent), checkpoint_path=str(ROOT / ck),
                                     prefer_full_checkpoint=True, max_sequence=MAX_SEQ)
        merge_lora_for_inference(model)
        model.eval()
        rows, valid, t0 = [], [], time.time()
        for name, toks in songs:
            for pi, (i, _) in enumerate(positions[name]):
                primer = tail(strip_tail(toks[:i]), 256)
                for seed in SEEDS:
                    torch.manual_seed(seed * 1000 + pi)
                    gen, _ = generate_once(model=model, primer=torch.tensor(primer, dtype=torch.long),
                                           target_length=min(MAX_SEQ, len(primer) + GEN_TOKENS), strip_primer=True,
                                           temperature=1.0, top_k=32, top_p=0.95, grammar_mask=True,
                                           target_duration_seconds=WINDOW_S, return_metadata=True, use_kv_cache=True)
                    gen = [int(x) for x in gen]
                    line = top_notes(tokens_to_notes(gen))
                    rows.append(window_stats([(n.start, n.end) for n in line], 0.0))
                    valid.append(bool(validate_generated_token_block(gen, lookahead_ms=WINDOW_S * 1000,
                                                                     allow_rest_bar=True)["valid"]))
        results[label] = {**pooled(rows), "valid_rate": sum(valid) / len(valid) if valid else None,
                          "wall_s": round(time.time() - t0, 1), "checkpoint": ck}
        del model
    real = pooled(real_rows)
    report = {"schema": "bebop_rh_eval_v1", "songs": len(songs), "positions": len(real_rows), "real": real,
              **results, "verdict": judge(real, results["adapter"], results["base"], results["adapter"]["valid_rate"]),
              "musical_quality_verified": False}
    (out_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=1))
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--adapter", default=None, help="exported checkpoint (default: newest in outputs/bebop_rh/export)")
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args(argv)
    adapter = args.adapter or sorted(glob.glob(str(ROOT / "outputs/bebop_rh/export/checkpoint_update*.pt")))[-1]
    adapter = str(Path(adapter).relative_to(ROOT)) if Path(adapter).is_absolute() else adapter
    os.environ.setdefault("FORCE_CPU", "1")
    run(adapter, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
