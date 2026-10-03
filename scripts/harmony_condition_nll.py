#!/usr/bin/env python3
"""Do the existing models read a harmony guide? (docs/experiments/HARMONY_CONTRACT.md U2, #1631)

Val songs of data/bebop_rh, original two-hand MIDI. For each window the solo
line is scored after the true accompaniment-proxy guide, a guide from another
song, and no guide; paired NLL differences per model. No training.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "music_transformer", ROOT / "music_transformer" / "third_party", ROOT / "scripts"):
    sys.path.insert(0, str(p))

WINDOW_S = 0.9375
FIRST_S, STEP_S, MAX_PER_SONG = 2.0, 4.0, 20
MIN_SOLO, MIN_PCS = 3, 2
SHUFFLE_OFFSET = 7
MODELS = {"base": "outputs/tvm/common_base/checkpoint_epoch8.pt",
          "bebop": "outputs/bebop_rh/export/checkpoint_update516.pt",
          "tatum": "outputs/final_tatum/export/checkpoint_update518.pt",
          "mehldau": "outputs/clean_base/c2_export/checkpoint_update128.pt"}


def song_windows(path: str) -> tuple[list[dict], dict]:
    import pretty_midi
    from inference.control.harmony_contract import SOLO_SPLIT, accompaniment_proxy, solo_window
    from inference.control.solo_line import top_notes

    pm = pretty_midi.PrettyMIDI(path)
    notes = sorted((n for i in pm.instruments for n in i.notes), key=lambda n: (n.start, n.pitch))
    line = [n for n in top_notes(notes) if n.pitch >= SOLO_SPLIT]
    end = pm.get_end_time()
    out, stats = [], {"candidates": 0, "empty": 0, "few_pcs": 0, "few_solo": 0}
    t = FIRST_S
    while t + WINDOW_S <= end and len(out) < MAX_PER_SONG:
        stats["candidates"] += 1
        bass, pcs, st = accompaniment_proxy(notes, t, t + WINDOW_S)
        solo = solo_window(line, t, t + WINDOW_S)
        if st["empty"]:
            stats["empty"] += 1
        elif len(pcs) < MIN_PCS:
            stats["few_pcs"] += 1
        elif len(solo) < MIN_SOLO:
            stats["few_solo"] += 1
        else:
            out.append({"t": t, "bass": bass, "pcs": sorted(pcs), "solo": solo,
                        "onset_groups": st["onset_groups"]})
        t += STEP_S
    return out, stats


def matched_donor(used, si: int, w: dict):
    """A window from another song with the same pitch-class count (so the same guide token
    length) and a different guide, searched from song si + SHUFFLE_OFFSET onward (#1633 review M1)."""
    n = len(used)
    for k in range(SHUFFLE_OFFSET, SHUFFLE_OFFSET + n):
        sj = (si + k) % n
        if sj == si:
            continue
        for o in used[sj]:
            if len(o["pcs"]) == len(w["pcs"]) and (o["bass"], o["pcs"]) != (w["bass"], w["pcs"]):
                return o
    return None


def target_tokens(guide, solo) -> int:
    from inference.control.harmony_contract import serialize

    toks, at = serialize(guide, solo, WINDOW_S)
    return len(toks) - at - 1


def runtime_guide_stats() -> dict:
    """pc count / range of the runtime chord-symbol guides (the 7 progressions' chords) vs proxies."""
    from inference.control.harmony_contract import chord_pcs, guide_pitches

    chords = ["Dm7", "G7", "Cmaj7", "F7", "Bb7", "Gm7", "C7", "Dm7b5", "Cm7"]
    gp = [guide_pitches(*chord_pcs(c)) for c in chords]
    return {"chords": chords, "pcs": [len(g) for g in gp], "range": [min(map(min, gp)), max(map(max, gp))],
            "changes_per_window": 0}


def solo_nll(model, guide, solo) -> float:
    import torch
    import torch.nn.functional as F
    from inference.control.harmony_contract import serialize

    toks, at = serialize(guide, solo, WINDOW_S)
    x = torch.tensor([toks], dtype=torch.long)
    with torch.no_grad():
        logits = model(x)[0]
    # position i predicts token i+1; score solo tokens after the first one
    idx = list(range(at + 1, len(toks)))
    lp = F.log_softmax(logits[[i - 1 for i in idx]], dim=-1)
    return float(-lp[range(len(idx)), [toks[i] for i in idx]].mean())


def bootstrap_ci(per_song: list[list[float]], n: int = 2000, seed: int = 0) -> list[float]:
    rng = random.Random(seed)
    means = []
    for _ in range(n):
        pick = [per_song[rng.randrange(len(per_song))] for _ in per_song]
        vals = [v for s in pick for v in s]
        means.append(sum(vals) / len(vals))
    means.sort()
    return [means[int(0.025 * n)], means[int(0.975 * n) - 1]]


def judge(res: dict) -> dict:
    def reads(m):
        return res[m]["d_shuffled"]["ci95"][0] > 0
    b, base = res["bebop"]["d_shuffled"], res["base"]["d_shuffled"]
    go_reasons = []
    if not reads("bebop"):
        go_reasons.append("bebop does not read the guide")
    if b["mean"] < 0.5 * base["mean"]:
        go_reasons.append("bebop below half of base")
    return {"reads_guide": {m: reads(m) for m in res}, "retrain_go": bool(go_reasons), "go_reasons": go_reasons,
            "no_go_explanation_rejected": (not go_reasons) and b["mean"] >= base["mean"] and reads("bebop")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--matched", action="store_true",
                    help="shuffled donor with the same pitch-class count / guide length (review M1)")
    ap.add_argument("--add-model", action="append", default=[], metavar="NAME=CHECKPOINT",
                    help="score another checkpoint too (e.g. guide=outputs/bebop_guide/export/...)")
    args = ap.parse_args(argv)
    for spec in args.add_model:
        name, path = spec.split("=", 1)
        MODELS[name] = path
        args.models.append(name)
    os.environ.setdefault("FORCE_CPU", "1")
    from inference.control.harmony_contract import guide_notes
    from scripts.generate import load_model_with_lora
    from scripts.train_qlora import merge_lora_for_inference

    manifest = json.loads((ROOT / "data/bebop_rh/manifest.json").read_text())
    songs = [s for s in manifest["songs"] if s.get("split") == "val" and s.get("status") == "ok"]
    windows, extraction = [], {"candidates": 0, "empty": 0, "few_pcs": 0, "few_solo": 0}
    for s in songs:
        w, st = song_windows(s["source"])
        windows.append(w)
        for k in extraction:
            extraction[k] += st[k]
    used = [w for w in windows if w]
    flat = [x for w in used for x in w]
    extraction.update({"songs": len(songs), "songs_used": len(used), "windows": len(flat),
                       "pcs_hist": {k: sum(1 for x in flat if len(x["pcs"]) == k) for k in range(2, 13)},
                       "onset_groups_median": statistics.median(x["onset_groups"] for x in flat),
                       "proxy_range": [min(36 + x["bass"] for x in flat), 59],
                       "target_tokens": {"total": sum(target_tokens([], x["solo"]) for x in flat),
                                         "median": statistics.median(target_tokens([], x["solo"]) for x in flat)},
                       "runtime_guides": runtime_guide_stats()})
    res = {}
    for name in args.models:
        ck = ROOT / MODELS[name]
        model = load_model_with_lora(lora_path=str(ck.parent), checkpoint_path=str(ck),
                                     prefer_full_checkpoint=True, max_sequence=512)
        merge_lora_for_inference(model)
        model.eval()
        per_song = {"true": [], "shuffled": [], "absent": []}
        same_guide = unmatched = 0
        rows_out = []
        for si, ws in enumerate(used):
            other = used[(si + SHUFFLE_OFFSET) % len(used)]
            rows = {"true": [], "shuffled": [], "absent": []}
            for wi, w in enumerate(ws):
                o = matched_donor(used, si, w) if args.matched else other[wi % len(other)]
                if o is None:
                    unmatched += 1
                    continue
                same_guide += int((o["bass"], o["pcs"]) == (w["bass"], w["pcs"]))
                rows["true"].append(solo_nll(model, guide_notes(w["bass"], w["pcs"], WINDOW_S), w["solo"]))
                rows["shuffled"].append(solo_nll(model, guide_notes(o["bass"], o["pcs"], WINDOW_S), w["solo"]))
                rows["absent"].append(solo_nll(model, [], w["solo"]))
            for k in rows:
                per_song[k].append(rows[k])
            rows_out += [{"song": si, **{k: rows[k][wi] for k in rows}} for wi in range(len(rows["true"]))]
        out = {k: sum(map(sum, v)) / sum(map(len, v)) for k, v in per_song.items()}
        for cond in ("shuffled", "absent"):
            diffs = [d for d in ([a - b for a, b in zip(per_song[cond][i], per_song["true"][i])]
                                 for i in range(len(used))) if d]
            vals = [v for s in diffs for v in s]
            song_means = [sum(d) / len(d) for d in diffs]
            out[f"d_{cond}"] = {"mean": sum(vals) / len(vals), "ci95": bootstrap_ci(diffs),
                                "share_positive": sum(v > 0 for v in vals) / len(vals),
                                "song_macro_mean": sum(song_means) / len(song_means),
                                "songs_positive": sum(m > 0 for m in song_means)}
        out["donor_same_as_true"] = same_guide
        out["donor_unmatched_excluded"] = unmatched
        (args.output_dir / f"windows_{name}.json").parent.mkdir(parents=True, exist_ok=True)
        (args.output_dir / f"windows_{name}.json").write_text(json.dumps(rows_out))
        res[name] = out
        print(name, json.dumps(out), flush=True)
        del model
    report = {"schema": "harmony_condition_nll_v1", "extraction": extraction, "models": res,
              "verdict": judge(res) if {"bebop", "base"} <= set(res) else None, "musical_quality_verified": False}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"extraction": extraction, "verdict": report["verdict"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
