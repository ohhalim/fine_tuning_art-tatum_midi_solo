#!/usr/bin/env python3
"""Full-vocabulary probability mass at the first discriminating note (docs/experiments/ARIA_COND_32STEP.md).

Read-only: reloads the step-32 synthetic_diagnostic_only adapter (no optimizer, no training, no
generation) and splits the next-token distribution on the 4 held-out prefixes into
non-overlapping groups that must sum to 1. Run with the Aria venv:
aria_cond_mass.py <aria repo> <checkpoint> <32-step dir>
"""
from __future__ import annotations

import json
import math
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from aria_cond_contract import chroma_per_position  # noqa: E402
from aria_cond_smoke import chord_pcs, file_sha  # noqa: E402
from aria_cond_synth import CHORD, EVAL, VEL, example  # noqa: E402
from aria_t4b_sim import write_notes  # noqa: E402

PC_GROUPS = {"pc_E_B": {4, 11}, "pc_Eb_Bb": {3, 10}, "pc_C_G": {0, 7}}
CORRECT = {"P": "pc_E_B", "Q": "pc_Eb_Bb"}


def group_of(tok) -> str:
    if isinstance(tok, tuple):
        if tok[0] == "piano":
            pc = tok[1] % 12
            return next((g for g, s in PC_GROUPS.items() if pc in s), "pc_other")
        if tok[0] == "onset":
            return "onset"
        if tok[0] == "dur":
            return "dur"
        if tok[0] == "prefix":
            return "other"
        return "other_instrument"
    return {"<E>": "eos", "<T>": "time"}.get(tok, "other")


def main() -> None:
    aria_repo, ckpt, run_dir = sys.argv[1:4]
    sys.path.insert(0, aria_repo)
    import torch
    from safetensors.torch import load_file
    from ariautils.midi import MidiDict
    from ariautils.tokenizer import AbsTokenizer
    from aria.config import load_model_config
    from aria.model import ModelConfig, TransformerLM

    with open(os.path.join(run_dir, "result.json")) as f:
        prior = json.load(f)
    adapter_path = os.path.join(run_dir, "adapter_synthetic_diagnostic_only.pt")
    dev = torch.device("mps")
    tok = AbsTokenizer()
    cfg = ModelConfig(**load_model_config(name="medium"))
    cfg.set_vocab_size(tok.vocab_size)
    model = TransformerLM(cfg)
    model.load_state_dict(load_file(ckpt), strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model.to(dev)
    model.model.freqs_cis = None
    adapter = torch.nn.Linear(12, cfg.d_model, bias=False).to(dev)
    adapter.weight.data.copy_(torch.load(adapter_path)["weight"].to(dev))
    adapter.weight.requires_grad_(False)
    state = {"cond": None}

    def pre_hook(_mod, args, kwargs):
        if state["cond"] is None:
            return None
        return (args[0] + adapter(state["cond"]),) + args[1:], kwargs

    model.model.encode_layers[-1].register_forward_pre_hook(pre_hook, with_kwargs=True)
    vocab = [tok.decode([i])[0] for i in range(tok.vocab_size)]
    groups = [group_of(t) for t in vocab]
    names = ["pc_E_B", "pc_Eb_Bb", "pc_C_G", "pc_other", "eos", "time", "onset", "dur", "other_instrument", "other"]
    idx = {g: torch.tensor([i for i, x in enumerate(groups) if x == g], device=dev) for g in names}
    piano = torch.tensor([i for i, t in enumerate(vocab) if isinstance(t, tuple) and t[0] == "piano"], device=dev)
    rows, consistency = [], []
    for fam in EVAL:
        notes, cand = example(fam, "P")
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "x.mid")
            write_notes(notes, path)
            toks = tok.tokenize(MidiDict.from_midi(path))
        pos = next(i for i, x in enumerate(toks) if isinstance(x, tuple) and x[0] == "piano" and x[1] in cand.values()) - 1
        ids = torch.tensor([tok.encode(toks[: pos + 1])], device=dev)
        pitch_ids = {q: torch.tensor([i for i, t in enumerate(vocab) if isinstance(t, tuple) and t[:2] == ("piano", cand[q])], device=dev)
                     for q in "PQ"}
        for name, q in (("base", None), ("Cmaj7", "P"), ("Cm7", "Q"), ("zero", "zero")):
            if q is None:
                state["cond"] = None
            elif q == "zero":
                state["cond"] = torch.zeros(1, pos + 1, 12, device=dev)
            else:
                state["cond"] = torch.tensor([chroma_per_position(toks[: pos + 1], [(0, 10 ** 9, chord_pcs(*CHORD[q]))])],
                                             dtype=torch.float32, device=dev)
            with torch.no_grad():
                logits = model(ids)[0, pos].float()
            prob = torch.softmax(logits, -1)
            mass = {g: prob[idx[g]].sum().item() for g in names}
            piano_mass = prob[piano].sum().item()
            top = torch.topk(prob, 10)
            lp = torch.log(prob)
            tP, tQ = tok.encode([("piano", cand["P"], VEL)])[0], tok.encode([("piano", cand["Q"], VEL)])[0]
            row = {"family": fam, "condition": name, "candidates": cand, "mass_raw": mass, "sum_of_groups": sum(mass.values()),
                   "piano_mass": piano_mass,
                   "piano_conditional": {g: mass[g] / piano_mass for g in ("pc_E_B", "pc_Eb_Bb", "pc_C_G", "pc_other")},
                   "pitch_all_velocities_raw": {q: prob[pitch_ids[q]].sum().item() for q in "PQ"},
                   "L_vel80": (lp[tP] - lp[tQ]).item(),
                   "top10": [[str(vocab[i]), round(p, 5)] for p, i in zip(top.values.tolist(), top.indices.tolist())]}
            rows.append(row)
            if name in ("Cmaj7", "Cm7"):
                ref = next(r for r in prior["eval"]["step32"] if r["family"] == fam)["cond_" + name]["L"]
                consistency.append(abs(row["L_vel80"] - ref))
    worst = max(consistency)
    M = [r["mass_raw"][CORRECT["P" if r["condition"] == "Cmaj7" else "Q"]] for r in rows if r["condition"] in ("Cmaj7", "Cm7")]
    e_first = sum(M)
    p0 = math.prod(1 - m for m in M)
    first_pcs = [g["first_pitch"] for g in prior["generation"]]
    observed = sum(1 for g in prior["generation"] if g["first_pitch"] is not None and
                   g["first_pitch"] % 12 in PC_GROUPS[CORRECT["P" if g["condition"] == "Cmaj7" else "Q"]])
    out = {"adapter_sha256": file_sha(adapter_path), "adapter_use": "read-only diagnostic reload of synthetic_diagnostic_only",
           "consistency_max_abs_L_diff_vs_step32": worst, "groups": names, "rows": rows,
           "decision": {"E_first": e_first, "P0_no_correct_first_note_in_8": p0, "observed_correct_first_notes": observed,
                        "observed_first_pitches": first_pcs,
                        "result": "pc_mass_weak" if e_first < 1 else "pc_mass_present_sample_sparse"}}
    with open(os.path.join(run_dir, "mass.json"), "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(json.dumps({"consistency": worst, "sums": [round(r["sum_of_groups"], 6) for r in rows], **out["decision"]}))


if __name__ == "__main__":
    main()
