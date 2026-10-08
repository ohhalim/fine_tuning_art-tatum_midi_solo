import json, sys
sys.path.insert(0, "/Users/ohhalim/git_box/t3_aria/aria")
import mlx.core as mx
from ariautils.midi import MidiDict
from ariautils.tokenizer import AbsTokenizer
from aria.inference import get_inference_prompt
from aria.run import _load_inference_model_mlx
D = "/Users/ohhalim/git_box/t3_aria"
tok = AbsTokenizer()
model = _load_inference_model_mlx(f"{D}/ckpt/model-gen.safetensors", "medium", strict=False)
plan = json.load(open(f"{D}/loop_tf/plan.json"))
TEMP, MIN_P = 0.98, 0.035
res = {}
for tag, v in plan.items():
    res[tag] = {}
    for cond, c in v["conditions"].items():
        ctx = get_inference_prompt(MidiDict.from_midi(c["ctx"]), tok, 1e12)
        full = get_inference_prompt(MidiDict.from_midi(c["full"]), tok, 1e12)
        if full[:len(ctx)] != ctx:
            res[tag][cond] = {"error": "context tokens are not a prefix of full tokens"}
            continue
        ids = tok.encode(full); L = len(ids); start = len(ctx)
        model.setup_cache(batch_size=1, max_seq_len=L + 1, dtype=mx.float32)
        logits = model(idxs=mx.array([ids], dtype=mx.int32), input_pos=mx.arange(0, L, dtype=mx.int32), offset=0, max_kv_pos=L - 1)[0]
        lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        p_s = mx.softmax(logits / TEMP, axis=-1)
        joint, keep, n = 0.0, 0, 0
        for j in range(start, L):
            t = ids[j]
            joint += lp[j - 1, t].item()
            keep += int(p_s[j - 1, t].item() >= MIN_P * mx.max(p_s[j - 1]).item())
            n += 1
        res[tag][cond] = {"ctx_tokens": start, "cand_tokens": n, "joint_logp": round(joint, 3), "mean_logp": round(joint / n, 3),
                          "min_p_survive": f"{keep}/{n}", "cand_tokens_preview": [str(x) for x in full[start:start + 6]]}
    print(tag, json.dumps(res[tag]))
json.dump(res, open(f"{D}/loop_tf/scores.json", "w"), indent=1)
