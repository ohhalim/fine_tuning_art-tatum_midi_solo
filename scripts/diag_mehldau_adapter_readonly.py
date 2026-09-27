"""Read-only diagnostic of the Mehldau LoRA runs. Never writes checkpoints.

Usage: .venv/bin/python scripts/diag_mehldau_adapter_readonly.py > readonly_diag.json
Measures changed tensors, LoRA update size, save/load identity and start-vs-adapted logits.
"""
import sys, json, math
from pathlib import Path
WT = Path("/Users/ohhalim/orca/workspaces/fine_tuning_art-tatum_midi_solo/즉흥연주재-설계")
M = Path("/Users/ohhalim/git_box/fine_tuning_art-tatum_midi_solo")
for p in [WT, WT / "music_transformer", WT / "scripts"]:
    sys.path.insert(0, str(p))
import numpy as np, torch, torch.nn.functional as F
from utilities.device import use_cuda
use_cuda(False)
torch.set_num_threads(8)
from scripts.generate import load_model_with_lora
from utilities.constants import TOKEN_PAD

START = {"from_base": M / "outputs/d0_experiment/armB_full2777/ckpt/checkpoint_epoch8.pt",
         "from_tatum": M / "outputs/d1_experiment/armD_lora/ckpt/checkpoint_epoch8.pt"}
def ck(p): return torch.load(p, map_location="cpu", weights_only=False)
out = {}

# 1) which tensors changed start -> epoch6/8, and effective out_proj delta size
for arm, sp in START.items():
    s0 = ck(sp)["model_state_dict"]
    for ep in (6, 8):
        s1 = ck(M / f"outputs/mehldau_lora/{arm}/checkpoint_epoch{ep}.pt")["model_state_dict"]
        changed = [k for k in s1 if k in s0 and not torch.equal(s0[k], s1[k])]
        missing = [k for k in s1 if k not in s0]
        rel = []
        for i in range(6):
            pre = f"transformer.encoder.layers.{i}.self_attn.out_proj."
            W = s0[pre + "original_layer.weight"]
            d0 = s0[pre + "lora_B"] @ s0[pre + "lora_A"] * 2.0
            d1 = s1[pre + "lora_B"] @ s1[pre + "lora_A"] * 2.0
            weff = W + d0
            rel.append({"layer": i, "base_W_norm": float(W.norm()),
                        "start_lora_delta_norm": float(d0.norm()),
                        "update_norm": float((d1 - d0).norm()),
                        "update_rel_to_Weff": float((d1 - d0).norm() / weff.norm()),
                        "lora_B_absmax_start": float(s0[pre + "lora_B"].abs().max()),
                        "lora_B_absmax_end": float(s1[pre + "lora_B"].abs().max())})
        out[f"{arm}_ep{ep}_param_delta"] = {"changed_keys": changed, "new_keys": missing, "per_layer": rel}

# 2) save/load identity: load_model_with_lora vs raw state dict
m = load_model_with_lora(lora_path=str(M / "outputs/mehldau_lora/from_tatum"),
                         checkpoint_path=str(M / "outputs/mehldau_lora/from_tatum/checkpoint_epoch8.pt"),
                         prefer_full_checkpoint=True, max_sequence=512)
raw = ck(M / "outputs/mehldau_lora/from_tatum/checkpoint_epoch8.pt")["model_state_dict"]
md = m.state_dict()
out["save_load"] = {"keys_equal": set(md) == set(raw),
                    "max_abs_diff": max(float((md[k].float() - raw[k].float()).abs().max()) for k in raw)}

# 3) logits: start vs adapted on train and val songs, no label smoothing, fixed crops
def load_split(split):
    return [np.load(p, allow_pickle=True).ravel().astype(np.int64)
            for p in sorted((M / "data/mehldau_full" / split).glob("*.npy"))]
def crops(seqs, L=512):
    for t in seqs:
        for s in range(0, max(1, len(t) - L), L):
            c = t[s:s + L + 1]
            if len(c) >= 2: yield c
def compare(ma, mb, seqs):
    ce_a = ce_b = kl = agree = n = 0.0; dl = 0.0
    with torch.no_grad():
        for c in crops(seqs):
            x = torch.tensor(c[:-1]).unsqueeze(0); y = torch.tensor(c[1:])
            la = ma(x)[0]; lb = mb(x)[0]
            k = y != TOKEN_PAD
            ce_a += F.cross_entropy(la[k], y[k], reduction="sum").item()
            ce_b += F.cross_entropy(lb[k], y[k], reduction="sum").item()
            pa = F.log_softmax(la[k], -1); pb = F.log_softmax(lb[k], -1)
            kl += (pa.exp() * (pa - pb)).sum().item()
            agree += (la[k].argmax(-1) == lb[k].argmax(-1)).sum().item()
            dl += (la[k] - lb[k]).abs().mean(-1).sum().item()
            n += int(k.sum())
    return {"tokens": int(n), "ce_start": ce_a / n, "ce_adapted": ce_b / n,
            "delta_ce": (ce_b - ce_a) / n, "kl_start_to_adapted": kl / n,
            "top1_agree": agree / n, "mean_abs_logit_delta": dl / n}
train, val = load_split("train"), load_split("val")
out["data"] = {"train_songs": len(train), "train_tokens": [len(t) for t in train],
               "val_tokens": [len(t) for t in val]}
def lm(p):
    return load_model_with_lora(lora_path=str(Path(p).parent), checkpoint_path=str(p),
                                prefer_full_checkpoint=True, max_sequence=512)
for arm, sp in START.items():
    ms = lm(sp)
    for ep in (8,):
        ma = lm(M / f"outputs/mehldau_lora/{arm}/checkpoint_epoch{ep}.pt")
        out[f"{arm}_ep{ep}_logits_train"] = compare(ms, ma, train)
        out[f"{arm}_ep{ep}_logits_val"] = compare(ms, ma, val)
print(json.dumps(out, indent=1))
