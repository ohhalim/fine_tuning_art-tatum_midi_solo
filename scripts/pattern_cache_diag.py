#!/usr/bin/env python3
"""Pattern-cache decoding on vs off, same positions and natural context (#1604).

docs/experiments/PATTERN_CACHE_DECODING.md. Rollouts at T 1.0 from the #1600
positions; the "on" arm adds ln 3 to the logit of note_on pitches that would
complete a recent top-line interval 3-gram (``scripts/pattern_cache.py``).
"""
from __future__ import annotations

import argparse
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

from scripts.context_mismatch_diag import (  # noqa: E402
    CHECKPOINTS, MAX_SEQ, PREFIX, SETS, TARGET_STEPS, bootstrap_ci, pick_positions, session_line, strip_tail, tail,
)
from scripts.decoding_gap_diag import GEN_TOKENS, SEEDS, mean_defined, measures, song_flag_rate, song_rates  # noqa: E402

BIAS = math.log(3.0)
ARMS = ("off", "on")


def arm_stats(cases, arm) -> dict:
    rows = [(c["song"], m) for c in cases for m in c["rollouts"][arm]]
    valid = [m["valid"] for _, m in rows]
    out = {k: mean_defined(song_rates(rows, k)) for k in ("interval", "variant", "exact", "repetition")}
    out["copy_rate"] = mean_defined(song_flag_rate(rows))
    out["valid_rate"] = sum(valid) / len(valid) if valid else None
    out["notes"] = statistics.mean(m["notes"] for _, m in rows) if rows else None
    fired = [m.get("cache_fired_share") for _, m in rows if m.get("cache_fired_share") is not None]
    out["cache_fired_share"] = statistics.mean(fired) if fired else None
    return out


def summarize(cases) -> dict:
    real_rows = [(c["song"], c["real"]) for c in cases]
    real = {k: mean_defined(song_rates(real_rows, k)) for k in ("interval", "variant", "exact", "repetition")}
    real["copy_rate"] = mean_defined(song_flag_rate(real_rows))
    ra = song_rates([(c["song"], m) for c in cases for m in c["rollouts"]["off"]], "interval")
    rb = song_rates([(c["song"], m) for c in cases for m in c["rollouts"]["on"]], "interval")
    diffs = [rb[s] - ra[s] for s in rb if s in ra and ra[s] is not None and rb[s] is not None]
    return {"real": real, "arms": {a: arm_stats(cases, a) for a in ARMS},
            "ir_on_minus_off": {"mean": statistics.mean(diffs) if diffs else None, "ci95": bootstrap_ci(diffs),
                                "songs": len(diffs)},
            "positions": len(cases), "songs": len({c["song"] for c in cases})}


def verdict(s) -> dict:
    real, on, d = s["real"], s["arms"]["on"], s["ir_on_minus_off"]
    ok = {
        "ir_half_of_real": on["interval"] is not None and real["interval"] is not None
        and on["interval"] >= 0.5 * real["interval"],
        "ir_above_off": bool(d["ci95"]) and d["ci95"][0] > 0,
        "repetition_guard": on["repetition"] is not None and real["repetition"] is not None
        and on["repetition"] <= real["repetition"] + 0.15,
        "copy_guard": on["copy_rate"] is not None and on["copy_rate"] <= max(0.25, 2 * (real["copy_rate"] or 0.0)),
        "valid_guard": on["valid_rate"] is not None and on["valid_rate"] >= 0.9,
    }
    return {**ok, "pass": all(ok.values()), "note": "no runtime default change; runtime opt-in and listening are separate"}


def run(set_name, models, out_dir):
    import torch
    from scripts.generate import generate_once, load_model_with_lora
    from scripts.pattern_cache import PatternCacheBias
    from scripts.run_resident_model_probe import validate_generated_token_block
    from scripts.train_qlora import merge_lora_for_inference
    from scripts.validate_style_distance import load

    out_dir.mkdir(parents=True, exist_ok=True)
    report = {"schema": "pattern_cache_v1", "set": set_name, "bias": BIAS, "musical_quality_verified": False,
              "models": {}}
    for m in models:
        files = sorted((ROOT / SETS[set_name][m]).glob("*.npy"))
        songs = [(f.name, [int(t) for t in load(f)]) for f in files]
        ck = ROOT / CHECKPOINTS[m]
        model = load_model_with_lora(lora_path=str(ck.parent), checkpoint_path=str(ck),
                                     prefer_full_checkpoint=True, max_sequence=MAX_SEQ)
        merge_lora_for_inference(model)
        model.eval()
        t0, cases = time.time(), []
        for name, toks in songs:
            for pi, (i, target) in enumerate(pick_positions(toks)):
                primer = tail(strip_tail(toks[:i]), PREFIX)
                sline = session_line(toks, i)
                case = {"song": name, "i": i, "real": measures(target, sline), "rollouts": {}}
                for arm in ARMS:
                    outs = []
                    for seed in SEEDS:
                        proc = PatternCacheBias(BIAS) if arm == "on" else None
                        torch.manual_seed(seed * 1000 + pi)
                        gen, _ = generate_once(model=model, primer=torch.tensor(primer, dtype=torch.long),
                                               target_length=min(MAX_SEQ, len(primer) + GEN_TOKENS),
                                               strip_primer=True, temperature=1.0, top_k=32, top_p=0.95,
                                               grammar_mask=True, target_duration_seconds=TARGET_STEPS / 100,
                                               return_metadata=True, use_kv_cache=True, logits_processor=proc)
                        gen = [int(x) for x in gen]
                        mm = measures(gen, sline)
                        mm["valid"] = bool(validate_generated_token_block(
                            gen, lookahead_ms=TARGET_STEPS * 10, allow_rest_bar=True)["valid"])
                        if proc is not None:
                            mm["cache_fired_share"] = proc.fired / proc.steps if proc.steps else 0.0
                        outs.append(mm)
                    case["rollouts"][arm] = outs
                cases.append(case)
        s = summarize(cases)
        s["verdict"] = verdict(s) if set_name == "eval" and m != "mehldau" else None
        report["models"][m] = {"checkpoint": str(ck), "songs": [n for n, _ in songs],
                               "wall_s": round(time.time() - t0, 1), "summary": s}
        (out_dir / f"cases_{m}.json").write_text(json.dumps(cases) + "\n")
        print(m, json.dumps({"real_ir": s["real"]["interval"], "off": s["arms"]["off"]["interval"],
                             "on": s["arms"]["on"]["interval"], "rep_on": s["arms"]["on"]["repetition"],
                             "fired": s["arms"]["on"]["cache_fired_share"],
                             "verdict": (s["verdict"] or {}).get("pass")}), flush=True)
        del model
    (out_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", choices=sorted(SETS), required=True)
    ap.add_argument("--models", default=None)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args(argv)
    os.environ.setdefault("FORCE_CPU", "1")
    run(args.set, args.models.split(",") if args.models else list(SETS[args.set]), args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
