#!/usr/bin/env python3
"""T4a generation (run with the Aria venv): batch 1, min_p 0, 48 tokens after history + chord, timed end to end."""
import hashlib, json, os, sys, time


def main():
    D, aria_dir = sys.argv[1], sys.argv[2]
    sys.path.insert(0, aria_dir)
    import torch
    from ariautils.midi import MidiDict
    from ariautils.tokenizer import AbsTokenizer
    from aria.inference import get_inference_prompt
    from aria.inference.sample_mlx import sample_batch
    from aria.run import _load_inference_model_mlx

    sha = lambda f: hashlib.sha1(open(f, "rb").read()).hexdigest()[:12]
    plan = json.load(open(f"{D}/t4a/plan.json"))
    tok = AbsTokenizer()
    t = time.time()
    model = _load_inference_model_mlx(f"{D}/ckpt/model-gen.safetensors", "medium", strict=False)
    man = {"load_s": round(time.time() - t, 2), "temp": 0.98, "min_p": 0.0, "new_tokens": 48, "batch": 1,
           "rng": "torch.manual_seed", "runs": []}
    first = True
    for tag, v in plan.items():
        for q, inp in v["inputs"].items():
            for seed in (1, 2, 3, 4):
                torch.manual_seed(seed)
                t0 = time.time()
                prompt = get_inference_prompt(MidiDict.from_midi(inp["path"]), tok, 1e12)
                t1 = time.time()
                res = sample_batch(model=model, tokenizer=tok, prompt=prompt, num_variations=1, max_new_tokens=48,
                                   temp=0.98, force_end=False, top_p=None, min_p=0.0)[0]
                t2 = time.time()
                name = f"{tag}_{q}_s{seed}"
                tok.detokenize(res).to_midi().save(f"{D}/t4a/{name}.out.mid")
                t3 = time.time()
                json.dump([str(x) for x in res], open(f"{D}/t4a/{name}.tokens.json", "w"))
                man["runs"].append({"name": name, "history": tag, "quality": q, "seed": seed, "input_sha1": sha(inp["path"]),
                                    "prompt_tokens": len(prompt), "generated_tokens": len(res) - len(prompt),
                                    "eos": str(tok.eos_tok) in res[len(prompt):],
                                    "tokenize_s": round(t1 - t0, 3), "sample_s": round(t2 - t1, 3), "detok_s": round(t3 - t2, 3),
                                    "total_s": round(t3 - t0, 3), "cold": first})
                first = False
                print(name, man["runs"][-1]["total_s"], flush=True)
    json.dump(man, open(f"{D}/t4a/manifest.json", "w"), indent=1)


if __name__ == "__main__":
    main()
