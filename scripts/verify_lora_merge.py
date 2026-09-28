#!/usr/bin/env python3
"""Criteria 1-2 of docs/experiments/LORA_MERGE.md on real checkpoints (KV cache on in both arms).

1. logits: merged vs unmerged model on several sequences (max abs diff < 1e-4)
2. tokens: generate_once (KV cache on) merged vs unmerged, chord-primer-like block
   settings, seeds 1-8 -> identical token lists. Also records wall time for both.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "music_transformer", ROOT / "music_transformer" / "third_party", ROOT / "scripts"):
    sys.path.insert(0, str(p))

MAIN = Path("/Users/ohhalim/git_box/fine_tuning_art-tatum_midi_solo")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", action="append", required=True, metavar="NAME=CHECKPOINT")
    ap.add_argument("--songs", type=Path, default=ROOT / "data/tvm/tatum16/train")
    ap.add_argument("--seeds", default="1,2,3,4,5,6,7,8")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args(argv)

    import numpy as np
    import torch
    from utilities.device import use_cuda
    use_cuda(False)
    from inference.control.chord_primer import chord_guide_notes_for_duration
    from scripts.generate import (encode_notes_simple, generate_once, load_model_with_lora,
                                  truncate_tokens_preserving_velocity)
    from scripts.validate_style_distance import load

    songs = sorted(args.songs.glob("*.npy"))[:3]
    seqs = [load(f)[s:s + L] for f in songs for s, L in ((0, 64), (500, 128), (1000, 192))]
    bpm, sub = 128, 240.0 / 128 / 2
    chord_primers = []
    for chord in ("Dm7", "G7", "Cmaj7", "A7"):
        notes = sorted(chord_guide_notes_for_duration(chord, bpm=bpm, seconds=sub),
                       key=lambda n: (n.start, n.pitch))
        chord_primers.append(truncate_tokens_preserving_velocity(encode_notes_simple(notes), 48))
    out = {"schema": "lora_merge_verify_v1", "models": {}}
    for spec in args.model:
        name, path = spec.split("=", 1)
        from scripts.train_qlora import merge_lora_for_inference

        def fresh():
            m = load_model_with_lora(lora_path=str(Path(path).parent), checkpoint_path=path,
                                     prefer_full_checkpoint=True)
            return m.eval()
        plain, merged_model = fresh(), fresh()
        merge_lora_for_inference(merged_model)
        maxdiff = 0.0
        with torch.no_grad():
            for seq in seqs:
                x = torch.tensor(np.asarray(seq, dtype=np.int64)).unsqueeze(0)
                maxdiff = max(maxdiff, (plain(x) - merged_model(x)).abs().max().item())
        identical, total, t_full, t_cache = 0, 0, 0.0, 0.0
        mismatches = []
        for seed in (int(s) for s in args.seeds.split(",") if s.strip()):
            for ci, cp in enumerate(chord_primers):
                primer = torch.tensor(cp, dtype=torch.long)
                res = {}
                for use in (False, True):
                    torch.manual_seed(seed * 100 + ci)
                    t0 = time.perf_counter()
                    toks, _ = generate_once(model=merged_model if use else plain, primer=primer,
                                            target_length=min(192, len(cp) + 96),
                                            strip_primer=True, temperature=1.0, top_k=32, top_p=0.95,
                                            grammar_mask=True, target_duration_seconds=sub,
                                            return_metadata=True, use_kv_cache=True)
                    dt = time.perf_counter() - t0
                    res[use] = [int(t) for t in toks]
                    if use:
                        t_cache += dt
                    else:
                        t_full += dt
                total += 1
                if res[False] == res[True]:
                    identical += 1
                else:
                    mismatches.append({"seed": seed, "chord": ci})
        out["models"][name] = {"logits_max_abs_diff": maxdiff, "logits_ok": maxdiff < 1e-4,
                               "blocks": total, "identical_blocks": identical,
                               "tokens_ok": identical == total, "mismatches": mismatches,
                               "wall_s_unmerged": t_full, "wall_s_merged": t_cache,
                               "speedup": t_full / t_cache if t_cache else None}
        print(name, json.dumps({k: v for k, v in out["models"][name].items() if k != "mismatches"}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
