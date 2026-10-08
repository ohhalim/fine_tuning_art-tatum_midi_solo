#!/usr/bin/env python3
"""min_p A/B generation for docs/experiments/MINP_AB.md (run with the Aria venv, outside this repo's venv).

Replicates `aria generate --backend mlx` (same prompt builder, model loader and sampler) with
batch 1, torch.manual_seed(seed) before each sample (the MLX sampler draws with torch.multinomial),
and saves the sampled tokens, the stop reason (EOS or length cap) and the MIDI.
"""
import argparse
import hashlib
import json
import os
import sys
import time


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--aria-dir", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--prompts", nargs="+", required=True, help="TAG=path.mid")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--min-p", type=float, nargs="+", default=[0.035, 0.0])
    ap.add_argument("--temp", type=float, default=0.98)
    ap.add_argument("--length", type=int, default=1024)
    ap.add_argument("--prompt-s", type=float, default=8.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sys.path.insert(0, args.aria_dir)

    import torch
    from ariautils.tokenizer import AbsTokenizer
    from aria.inference.sample_mlx import sample_batch
    from aria.run import _get_prompt, _load_inference_model_mlx

    os.makedirs(args.out, exist_ok=True)
    tok = AbsTokenizer()
    t_load = time.time()
    model = _load_inference_model_mlx(args.checkpoint, "medium", strict=False)
    load_s = time.time() - t_load
    sha = lambda f: hashlib.sha1(open(f, "rb").read()).hexdigest()[:12]
    manifest = {"aria_dir": args.aria_dir, "checkpoint": args.checkpoint, "temp": args.temp, "length": args.length,
                "prompt_s": args.prompt_s, "seeds": args.seeds, "min_p": args.min_p, "rng": "torch.manual_seed (torch.multinomial in sample_min_p)",
                "load_s": round(load_s, 2), "runs": []}
    for spec in args.prompts:
        tag, path = spec.split("=", 1)
        prompt = _get_prompt(path, prompt_duration_s=args.prompt_s)
        for mp in args.min_p:
            for seed in args.seeds:
                torch.manual_seed(seed)
                t0 = time.time()
                res = sample_batch(model=model, tokenizer=tok, prompt=prompt, num_variations=1,
                                   max_new_tokens=min(8096 - len(prompt), args.length), temp=args.temp,
                                   force_end=False, top_p=None, min_p=mp)[0]
                dt = time.time() - t0
                stop = "eos" if tok.eos_tok in res else "length_cap"
                name = f"{tag}_mp{mp}_s{seed}"
                tok.detokenize(res).to_midi().save(os.path.join(args.out, name + ".mid"))
                with open(os.path.join(args.out, name + ".tokens.json"), "w") as f:
                    json.dump([str(t) for t in res], f)
                manifest["runs"].append({"name": name, "prompt": tag, "prompt_path": path, "prompt_sha1": sha(path),
                                         "prompt_tokens": len(prompt), "min_p": mp, "seed": seed, "stop": stop,
                                         "generated_tokens": len(res) - len(prompt), "gen_seconds": round(dt, 2),
                                         "midi_sha1": sha(os.path.join(args.out, name + ".mid"))})
                print(name, stop, len(res) - len(prompt), round(dt, 1), flush=True)
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
