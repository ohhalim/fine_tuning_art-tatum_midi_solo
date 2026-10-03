#!/usr/bin/env python3
"""Free generation after the true harmony guide: guide adapter vs bebop (GUIDE_ADAPTER.md, #1631).

Val windows of HARMONY_CONTRACT U2. Each model generates one window (0.9375 s)
after the window's guide only (the runtime contract without history); the solo
is the generated top line at or above G3. Pooled metrics against the guide it
was given: rel_js to the train reference (the only U3-qualified gate), fit and
clash (group-level report), density; the same generations against a donor
guide show whether fit follows the given harmony.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "music_transformer", ROOT / "music_transformer" / "third_party", ROOT / "scripts"):
    sys.path.insert(0, str(p))

from scripts.harmony_condition_nll import SHUFFLE_OFFSET, WINDOW_S, song_windows  # noqa: E402
from scripts.harmony_controls import js_distance, rel_hist, window_metrics  # noqa: E402

SEEDS = (1, 2)
MODELS = {"bebop": "outputs/bebop_rh/export/checkpoint_update516.pt"}


def reference_hist(train_songs: int = 60) -> list[float]:
    from scripts.harmony_controls import song_windows as ctl_windows

    manifest = json.loads((ROOT / "data/bebop_rh/manifest.json").read_text())
    ref = [0.0] * 12
    for s in [s for s in manifest["songs"] if s.get("status") == "ok" and s["split"] == "train"][:train_songs]:
        for w in ctl_windows(s["source"]):
            for i, x in enumerate(rel_hist(w["solo"], w["bass"])):
                ref[i] += x
    return ref


def pooled(rows, ref) -> dict:
    """rows: (solo [(p,s,e)], pcs, bass) per generated window."""
    h = [0.0] * 12
    fits, clashes, notes = [], [], 0
    for solo, pcs, bass in rows:
        notes += len(solo)
        if not solo:
            continue
        for i, x in enumerate(rel_hist(solo, bass)):
            h[i] += x
        m = window_metrics(solo, pcs)
        fits.append(m["fit"])
        clashes.append(m["clash"])
    return {"windows": len(rows), "windows_with_notes": len(fits),
            "notes_per_s": notes / (len(rows) * WINDOW_S),
            "rel_js": js_distance(h, ref) if sum(h) else None,
            "fit": sum(fits) / len(fits) if fits else None,
            "clash": sum(clashes) / len(clashes) if clashes else None}


def judge(real: dict, bebop: dict, guide: dict) -> dict:
    ok = {
        "rel_js_below_bebop": guide["rel_js"] < bebop["rel_js"],
        "rel_js_at_most_0.10": guide["rel_js"] <= 0.10,
        "fit_above_bebop_by_0.03": guide["fit"] >= bebop["fit"] + 0.03,
        "density_0.67_1.5x_real": 0.67 * real["notes_per_s"] <= guide["notes_per_s"] <= 1.5 * real["notes_per_s"],
    }
    return {**ok, "pass": all(ok.values())}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--guide-checkpoint", required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args(argv)
    os.environ.setdefault("FORCE_CPU", "1")
    import torch
    from inference.control.harmony_contract import SOLO_SPLIT, guide_notes, serialize
    from inference.control.solo_line import top_notes
    from scripts.generate import generate_once, load_model_with_lora
    from scripts.style_distance import tokens_to_notes
    from scripts.train_qlora import merge_lora_for_inference

    manifest = json.loads((ROOT / "data/bebop_rh/manifest.json").read_text())
    songs = [s for s in manifest["songs"] if s.get("split") == "val" and s.get("status") == "ok"]
    used = [w for w in (song_windows(s["source"])[0] for s in songs) if w]
    ref = reference_hist()
    real_rows = [([(n.pitch, n.start, n.end) for n in w["solo"]], set(w["pcs"]), w["bass"]) for ws in used for w in ws]
    out = {"real": pooled(real_rows, ref)}
    for name, ck in {**MODELS, "guide": args.guide_checkpoint}.items():
        model = load_model_with_lora(lora_path=str((ROOT / ck).parent), checkpoint_path=str(ROOT / ck),
                                     prefer_full_checkpoint=True, max_sequence=512)
        merge_lora_for_inference(model)
        model.eval()
        given, donor = [], []
        for si, ws in enumerate(used):
            other = used[(si + SHUFFLE_OFFSET) % len(used)]
            for wi, w in enumerate(ws):
                o = other[wi % len(other)]
                head, _ = serialize(guide_notes(w["bass"], w["pcs"], WINDOW_S), [], 0.0)
                for seed in SEEDS:
                    torch.manual_seed(seed * 100003 + si * 101 + wi)
                    gen, _ = generate_once(model=model, primer=torch.tensor(head, dtype=torch.long),
                                           target_length=len(head) + 128, strip_primer=True, temperature=1.0,
                                           top_k=32, top_p=0.95, grammar_mask=True,
                                           target_duration_seconds=WINDOW_S, return_metadata=True,
                                           use_kv_cache=True)
                    line = [n for n in top_notes(tokens_to_notes([int(t) for t in gen]))
                            if n.pitch >= SOLO_SPLIT and n.start < WINDOW_S]
                    solo = [(n.pitch, n.start, min(n.end, WINDOW_S)) for n in line]
                    given.append((solo, set(w["pcs"]), w["bass"]))
                    donor.append((solo, set(o["pcs"]), o["bass"]))
        out[name] = {**pooled(given, ref), "fit_vs_donor_harmony": pooled(donor, ref)["fit"]}
        print(name, json.dumps(out[name]), flush=True)
        del model
    report = {"schema": "guide_gen_eval_v1", "models": out,
              "verdict": judge(out["real"], out["bebop"], out["guide"]), "musical_quality_verified": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["verdict"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
