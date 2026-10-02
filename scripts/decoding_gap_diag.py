#!/usr/bin/env python3
"""Does sharper decoding make rollouts reuse the preceding phrases? (#1600)

docs/experiments/DECODING_GAP_DIAG.md. Same positions and natural 256-token
context as the #1597 diagnosis (``context_mismatch_diag``); only the sampling
temperature changes (1.0 / 0.8 / 0.6, fixed). Rollouts and the real
continuation are scored with the same measures against the real preceding 8 s.
"""
from __future__ import annotations

import argparse
import json
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

TEMPERATURES = (1.0, 0.8, 0.6)
SEEDS = (1, 2)
GEN_TOKENS = 320


def measures(tokens, sline) -> dict:
    """IR/VR/XR, within-output repetition, 5-note copy of the preceding 8 s, note count."""
    from scripts.coherence_metrics import top_line
    from scripts.motif_transfer_ab import has_copy, reappearance
    from scripts.style_distance import tokens_to_notes

    notes = tokens_to_notes(tokens)
    line = top_line([(n.start, n.pitch) for n in notes])
    r = reappearance(line, sline)
    return {**r, "copy": has_copy(line, [sline]), "notes": len(notes)}


def song_rates(rows, field) -> dict:
    """Per song ratio of sums over valid rows: field / windows (repetition uses distinct_windows)."""
    acc = {}
    for song, m in rows:
        if not m.get("valid", True):
            continue
        a = acc.setdefault(song, [0, 0])
        a[0] += (m["windows"] - m["distinct_windows"]) if field == "repetition" else m[field]
        a[1] += m["windows"]
    return {s: (n / w if w else None) for s, (n, w) in acc.items()}


def song_flag_rate(rows, field="copy") -> dict:
    acc = {}
    for song, m in rows:
        if not m.get("valid", True):
            continue
        a = acc.setdefault(song, [0, 0])
        a[0] += bool(m[field])
        a[1] += 1
    return {s: n / t for s, (n, t) in acc.items() if t}


def mean_defined(d: dict):
    v = [x for x in d.values() if x is not None]
    return statistics.mean(v) if v else None


def summarize(cases) -> dict:
    real_rows = [(c["song"], c["real"]) for c in cases]
    out = {"real": {k: mean_defined(song_rates(real_rows, k)) for k in ("interval", "variant", "exact", "repetition")}}
    out["real"]["copy_rate"] = mean_defined(song_flag_rate(real_rows))
    per_t = {}
    for t in TEMPERATURES:
        rows = [(c["song"], m) for c in cases for m in c["rollouts"][str(t)]]
        valid = [m["valid"] for _, m in rows]
        per_t[str(t)] = {k: mean_defined(song_rates(rows, k)) for k in ("interval", "variant", "exact", "repetition")}
        per_t[str(t)]["copy_rate"] = mean_defined(song_flag_rate(rows))
        per_t[str(t)]["valid_rate"] = sum(valid) / len(valid) if valid else None
        per_t[str(t)]["notes"] = statistics.mean(m["notes"] for _, m in rows) if rows else None
        per_t[str(t)]["_ir_by_song"] = song_rates(rows, "interval")
    base = per_t["1.0"]["_ir_by_song"]
    for t in TEMPERATURES[1:]:
        cur = per_t[str(t)]["_ir_by_song"]
        diffs = [cur[s] - base[s] for s in cur if s in base and cur[s] is not None and base[s] is not None]
        per_t[str(t)]["ir_minus_t1"] = {"mean": statistics.mean(diffs) if diffs else None,
                                       "ci95": bootstrap_ci(diffs), "songs": len(diffs)}
    for t in TEMPERATURES:
        per_t[str(t)].pop("_ir_by_song")
    out["by_temperature"] = per_t
    out["positions"] = len(cases)
    out["songs"] = len({c["song"] for c in cases})
    return out


def verdict(s) -> dict:
    """Fixed rule (plan), thresholds from the real continuation."""
    real = s["real"]
    checks = {}
    for t in TEMPERATURES[1:]:
        b = s["by_temperature"][str(t)]
        d = b.get("ir_minus_t1") or {}
        ok = {
            "ir_half_of_real": b["interval"] is not None and real["interval"] is not None
            and b["interval"] >= 0.5 * real["interval"],
            "ir_above_t1": bool(d.get("ci95")) and d["ci95"][0] > 0,
            "repetition_guard": b["repetition"] is not None and real["repetition"] is not None
            and b["repetition"] <= real["repetition"] + 0.15,
            "copy_guard": b["copy_rate"] is not None
            and b["copy_rate"] <= max(0.25, 2 * (real["copy_rate"] or 0.0)),
            "valid_guard": b["valid_rate"] is not None and b["valid_rate"] >= 0.9,
        }
        checks[str(t)] = {**ok, "all": all(ok.values())}
    label = "decoding_explains" if any(c["all"] for c in checks.values()) else "decoding_alone_insufficient"
    return {"checks": checks, "label": label,
            "note": "training insufficiency is not judged; no result changes runtime defaults or authorizes training"}


def run(set_name, models, out_dir):
    import torch
    from scripts.generate import generate_once, load_model_with_lora
    from scripts.run_resident_model_probe import validate_generated_token_block
    from scripts.train_qlora import merge_lora_for_inference
    from scripts.validate_style_distance import load

    out_dir.mkdir(parents=True, exist_ok=True)
    report = {"schema": "decoding_gap_v1", "set": set_name, "temperatures": list(TEMPERATURES),
              "musical_quality_verified": False, "models": {}}
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
                for t in TEMPERATURES:
                    outs = []
                    for seed in SEEDS:
                        torch.manual_seed(seed * 1000 + pi)
                        gen, _ = generate_once(model=model, primer=torch.tensor(primer, dtype=torch.long),
                                               target_length=min(MAX_SEQ, len(primer) + GEN_TOKENS),
                                               strip_primer=True, temperature=t, top_k=32, top_p=0.95,
                                               grammar_mask=True, target_duration_seconds=TARGET_STEPS / 100,
                                               return_metadata=True, use_kv_cache=True)
                        gen = [int(x) for x in gen]
                        mm = measures(gen, sline)
                        mm["valid"] = bool(validate_generated_token_block(
                            gen, lookahead_ms=TARGET_STEPS * 10, allow_rest_bar=True)["valid"])
                        outs.append(mm)
                    case["rollouts"][str(t)] = outs
                cases.append(case)
        s = summarize(cases)
        s["verdict"] = verdict(s) if set_name == "eval" and m != "mehldau" else None
        report["models"][m] = {"checkpoint": str(ck), "songs": [n for n, _ in songs],
                               "wall_s": round(time.time() - t0, 1), "summary": s}
        (out_dir / f"cases_{m}.json").write_text(json.dumps(cases) + "\n")
        bt = s["by_temperature"]
        print(m, json.dumps({"real_ir": s["real"]["interval"],
                             "ir": {t: bt[t]["interval"] for t in bt},
                             "rep": {t: bt[t]["repetition"] for t in bt}, "real_rep": s["real"]["repetition"],
                             "verdict": (s["verdict"] or {}).get("label")}), flush=True)
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
