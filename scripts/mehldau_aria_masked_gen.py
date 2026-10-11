#!/usr/bin/env python3
"""Mehldau Aria base vs adapter under the same format-only grammar mask (issue #1731,
docs/experiments/MEHLDAU_ARIA_MASKED_GEN.md).

A new operating condition, not a repair of #1726: the #1726 verdict (development fail, test unopened) stays.
Fixed: frozen model-gen + lora_u128.pt from #1726, the 4 development prompts x seeds 1, 2 x {base, adapter},
512 new tokens, temperature 1, no min_p, stop at <E>, the #1726 sampler (one Exp(1) noise vector per step),
argmax over the tokens aria_grammar_mask allows. No reselection, retries or picking.

Modes (one per process):
  --precheck  input SHAs, sampler equivalence on 32 consecutive synthetic steps, mask non-empty in every
              state, a 24-token masked smoke that must equal the saved #1726 stream (a valid one), MIDI export.
  --generate  16 streams with a progress log; per step the pre-mask allowed mass and whether the mask changed
              the pick. Wall 3600 s; if hit, the partial result is recorded and not rerun with more budget.
  --render    MIDI of every stream (prompt + new, incomplete tail cut), same piano sound for all, 8 blind
              A/B pairs (same prompt and seed) shuffled with random.Random(2026); key outside the repo.

  mehldau_aria_masked_gen.py <aria repo> <checkpoint> <data json> <manifest json> <1726 run dir> <out dir> <report dir> --precheck|--generate|--render
"""
from __future__ import annotations

import collections
import json
import os
import random
import shutil
import signal
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from aria_cond_smoke import Stop, file_sha, pressure, rss  # noqa: E402
from aria_grammar_mask import GrammarMask, MaskStop, kind, masked_generate, run_streams  # noqa: E402
from aria_lora import AriaLoRA, input_mismatches  # noqa: E402
from aria_token_validity import check as token_check  # noqa: E402
from mehldau_aria_lora_run import longest_repeat, max_polyphony, note_events  # noqa: E402

RANK, ALPHA, SEED, TEMP = 16, 16, 0, 1.0
GEN_SEEDS, GEN_TOKENS, NGRAM = (1, 2), 512, 16
WALL = {"--precheck": 900, "--generate": 3600, "--render": 900}
SMOKE_TOKENS, SHUFFLE_SEED = 24, 2026
SF2 = os.path.expanduser("~/.local/share/soundfonts/generaluser-gs/v1.471.sf2")


def complete_prefix(tokens):
    """Tokens up to the last complete note: a stream cut by the cap inside a note drops that note."""
    expect, last_start = "note", None
    for i, t in enumerate(tokens):
        k = kind(t)
        if expect == "note" and k == "piano":
            expect, last_start = "onset", i
        elif expect == "onset" and k == "onset":
            expect = "dur"
        elif expect == "dur" and k == "dur":
            expect = "note"
    return tokens if expect == "note" else tokens[:last_start]


def main() -> None:
    aria_repo, ckpt, data_path, manifest_path, run_dir, out_dir, report_dir, mode = sys.argv[1:9]
    assert mode in WALL
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)
    sys.path.insert(0, aria_repo)
    report_path = os.path.join(report_dir, {"--precheck": "precheck.json", "--generate": "report.json", "--render": "render.json"}[mode])
    log = {"mode": mode, "checks": {}, "phases": {}, "result": None, "stopped_at": None}

    def check(name, ok, **detail):
        log["checks"][name] = {"ok": bool(ok), **detail}
        if not ok:
            raise Stop(f"check failed: {name}")

    def on_alarm(*_):
        raise Stop(f"wall limit {WALL[mode]} s")

    signal.signal(signal.SIGALRM, on_alarm)
    signal.alarm(WALL[mode])
    progress = open(os.path.join(out_dir, f"progress{mode.replace('--', '_')}.log"), "a")

    def say(msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        progress.write(line + "\n")
        progress.flush()

    try:
        import torch
        from ariautils.tokenizer import AbsTokenizer

        t0 = time.perf_counter()
        train = json.load(open(os.path.join(run_dir, "train_report.json")))
        lora_path, old_gen_path = os.path.join(run_dir, "lora_u128.pt"), os.path.join(run_dir, "generations.json")
        rec = {**{k: train["provenance"][k] for k in ("checkpoint_sha256", "data_sha256", "manifest_sha256")},
               "lora_file_sha256": train["lora_file_sha256"]}
        cur = {"checkpoint_sha256": file_sha(ckpt), "data_sha256": file_sha(data_path),
               "manifest_sha256": file_sha(manifest_path), "lora_file_sha256": file_sha(lora_path)}
        check("inputs_match_1726_train_report", not input_mismatches(rec, cur), mismatches=input_mismatches(rec, cur))
        log["inputs"] = {**cur, "unmasked_generations_sha256": file_sha(old_gen_path), "torch": torch.__version__}
        old = json.load(open(old_gen_path))
        data = json.load(open(data_path))
        prompts = data["prompts"]
        tok = AbsTokenizer()
        V = tok.vocab_size
        vocab = [tok.decode([i])[0] for i in range(V)]
        eos_id = tok.encode(["<E>"])[0]
        order = [(pi, s, on) for pi in range(len(prompts)) for s in GEN_SEEDS for on in (False, True)]   # #1726 order
        check("unmasked_streams_in_same_order", len(old) == len(order) and all(
            old[n]["prompt"] == prompts[pi]["path"] and old[n]["seed"] == s and old[n]["adapter"] == on
            for n, (pi, s, on) in enumerate(order)))

        if mode == "--render":
            rep = json.load(open(os.path.join(report_dir, "report.json")))
            check("generate_done_on_same_inputs", rep.get("result") == "generate_done" and rep["inputs"] == log["inputs"])
            gens = json.load(open(os.path.join(out_dir, "generations_masked.json")))["completed"]
            check("sixteen_completed_streams", len(gens) == 16)
            check("generations_file_matches_report", file_sha(os.path.join(out_dir, "generations_masked.json")) == rep["generations_masked_sha256"])
            check("fluidsynth_and_soundfont", shutil.which("fluidsynth") is not None and os.path.exists(SF2), soundfont=SF2)
            mid_dir, lis_dir = os.path.join(out_dir, "midi"), os.path.join(out_dir, "listening")
            check("output_dirs_empty", not (os.path.isdir(lis_dir) and os.listdir(lis_dir)))
            os.makedirs(mid_dir, exist_ok=True)
            os.makedirs(lis_dir, exist_ok=True)
            files = []
            for g in gens:
                toks = tok.decode(prompts[g["prompt_index"]]["ids"] + g["ids"])
                kept = complete_prefix([t for t in toks if t != "<E>"])
                md = tok.detokenize(kept)
                programs = sorted({m["data"] for m in md.instrument_msgs})
                name = f"p{g['prompt_index']}_s{g['seed']}_{'adapter' if g['adapter'] else 'base'}.mid"
                md.to_midi().save(os.path.join(mid_dir, name))
                files.append({"name": name, "tokens_cut": len(toks) - len(kept) - toks.count("<E>"), "programs": programs})
            check("all_piano_program", all(f["programs"] == [0] for f in files), programs=[f["programs"] for f in files])
            rng = random.Random(SHUFFLE_SEED)
            key = {"shuffle_seed": SHUFFLE_SEED, "soundfont": SF2, "pairs": []}
            for i, (pi, s) in enumerate((pi, s) for pi in range(len(prompts)) for s in GEN_SEEDS):
                src = [("base", f"p{pi}_s{s}_base.mid"), ("adapter", f"p{pi}_s{s}_adapter.mid")]
                rng.shuffle(src)
                entry = {"pair": i + 1, "prompt_index": pi, "seed": s}
                for label, (arm, name) in zip("AB", src):
                    subprocess.run(["fluidsynth", "-ni", "-q", "-F", os.path.join(lis_dir, f"pair{i + 1}_{label}.wav"), "-r", "44100",
                                    SF2, os.path.join(mid_dir, name)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    entry[label] = arm
                key["pairs"].append(entry)
            key_path = os.path.join(out_dir, "key.json")
            with open(key_path, "w") as f:
                json.dump(key, f, indent=1)
            sheet = ["# 블라인드 청취 — 답안지 (#1731)", "",
                     "`key.json`은 답을 다 쓴 뒤에 여세요. 각 쌍의 A/B는 같은 prompt(처음 8초)·seed에서, 같은 형식 마스크로 만든 두 모델의 생성입니다.",
                     "청취는 선택입니다. 듣지 않으면 선호는 미확인으로 남습니다.", "",
                     "| 쌍 | 더 멜다우 같은 쪽 (A/B/모름) | 음악적으로 더 나은 쪽 (A/B/같음) | 메모 |", "|---|---|---|---|"]
            sheet += [f"| {p['pair']} | | | |" for p in key["pairs"]]
            with open(os.path.join(lis_dir, "ANSWER_SHEET.md"), "w") as f:
                f.write("\n".join(sheet) + "\n")
            log["render"] = {"midi": files, "pairs": len(key["pairs"]), "key_sha256": file_sha(key_path),
                             "wav": sorted(x for x in os.listdir(lis_dir) if x.endswith(".wav"))}
            log["result"] = "render_done"
            return

        from safetensors.torch import load_file
        from aria.config import load_model_config
        from aria.model import ModelConfig, TransformerLM

        check("mps_fallback_off", not os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"))
        dev_ = torch.device("mps")
        cfg_dict = load_model_config(name="medium")
        cfg = ModelConfig(**cfg_dict)
        cfg.set_vocab_size(V)
        cfg.grad_checkpoint = True
        model = TransformerLM(cfg)
        model.load_state_dict(load_file(ckpt), strict=True)
        for p in model.parameters():
            p.requires_grad_(False)
        model.to(dev_)
        model.model.freqs_cis = None
        model.eval()
        lora = AriaLoRA(model, rank=RANK, alpha=ALPHA, seed=SEED, device=dev_)
        lora.load_state_dict(torch.load(lora_path))
        gm0 = GrammarMask(vocab)
        log["phases"]["load"] = {"wall_s": round(time.perf_counter() - t0, 3), **rss(), **pressure()}

        g1, g2, same = torch.Generator().manual_seed(2), torch.Generator().manual_seed(2), 0
        for k in range(32):
            p = torch.softmax(torch.randn(V, generator=torch.Generator().manual_seed(100 + k)) * 3, -1)
            same += torch.multinomial(p, 1, generator=g1).item() == int(torch.argmax(p / torch.empty(V).exponential_(1, generator=g2)))
        check("multinomial_is_exponential_race_over_steps", same == 32, matched=same, of=32)

        def generate(prompt_ids, seed, on, max_new, tag, buf):
            def next_probs(seq):
                with torch.no_grad():
                    return torch.softmax(model(torch.tensor([seq], device=dev_))[0, -1].float() / TEMP, -1).cpu()

            t = time.perf_counter()
            lora.on = on
            try:
                masked_generate(next_probs, tok.decode(list(prompt_ids)), prompt_ids, vocab, seed, max_new, eos_id, buf,
                                progress=lambda k: say(f"{tag} {k}/{max_new} tokens, {time.perf_counter() - t:.0f} s"))
            finally:
                lora.on = True

        old_by_key = {(o["prompt"], o["seed"], o["adapter"]): o for o in old}
        if mode == "--precheck":
            check("mask_non_empty_in_every_state", bool(gm0.note_ok.any() and gm0.dur_ok.any() and (gm0.onset_val >= gm0.onset_val.max()).any()),
                  between_notes=int(gm0.note_ok.sum()), dur=int(gm0.dur_ok.sum()), onset=int((gm0.onset_val >= 0).sum()),
                  max_onset=int(gm0.onset_val.max()))
            ref = old[0]
            check("smoke_reference_is_valid_unmasked_stream", ref["valid"] and not ref["adapter"])
            t = time.perf_counter()
            buf = {"ids": [], "pre_mask_allowed": [], "mask_changed": []}
            generate(prompts[0]["ids"], ref["seed"], False, SMOKE_TOKENS, "smoke", buf)
            new, pre_mass, changed = buf["ids"], buf["pre_mask_allowed"], buf["mask_changed"]
            torch.mps.synchronize()
            s_tok = (time.perf_counter() - t) / len(new)
            check("smoke_equals_saved_unmasked_stream", new == ref["ids"][:SMOKE_TOKENS] and not changed,
                  matched=sum(a == b for a, b in zip(new, ref["ids"])), of=SMOKE_TOKENS, min_pre_mask_allowed=min(pre_mass))
            md = tok.detokenize(complete_prefix(tok.decode(prompts[0]["ids"] + new)))
            check("midi_export", len(md.note_msgs) > 0, notes=len(md.note_msgs))
            log["projection"] = {"smoke_s_per_token_short_context": round(s_tok, 4),
                                 "estimate_basis": "#1726 generation phase 2357.5 s for the same 16 x 512 steps without the mask",
                                 "wall_limit_s": WALL["--generate"]}
            log["result"] = "precheck_done"
            return

        pre = json.load(open(os.path.join(report_dir, "precheck.json")))
        check("precheck_done_on_same_inputs", pre.get("result") == "precheck_done" and pre["inputs"] == log["inputs"])
        train_grams = set()
        for r in data["recordings"]:
            if r["split"] == "train":
                for w in r["windows"]:
                    ids = w["ids"]
                    train_grams.update(tuple(ids[i:i + NGRAM]) for i in range(len(ids) - NGRAM + 1))
        rows = []
        out_path = os.path.join(out_dir, "generations_masked.json")
        t_all = time.perf_counter()
        specs = [{"n": n, "prompt_index": pi, "prompt": prompts[pi]["path"], "seed": s, "adapter": on}
                 for n, (pi, s, on) in enumerate(order)]

        def tag_of(spec):
            return f"[{spec['n'] + 1}/16 p{spec['prompt_index']} s{spec['seed']} {'adapter' if spec['adapter'] else 'base'}]"

        def run_one(spec, buf):
            say(f"{tag_of(spec)} start")
            buf["_t"] = time.perf_counter()
            generate(prompts[spec["prompt_index"]]["ids"], spec["seed"], spec["adapter"], GEN_TOKENS, tag_of(spec), buf)

        def after(buf):
            wall = time.perf_counter() - buf.pop("_t")
            pi, s, on, new, pre_mass, changed = buf["prompt_index"], buf["seed"], buf["adapter"], buf["ids"], buf["pre_mask_allowed"], buf["mask_changed"]
            pt, nt = tok.decode(list(prompts[pi]["ids"])), tok.decode(new)
            v = token_check(pt, nt)
            ev, pev, new_ev = note_events(pt + nt), note_events(pt), note_events(nt)
            ps = [x[1] for x in nt if isinstance(x, tuple) and x[0] == "piano"]
            abs_new = ev[len(pev):]
            grams = [tuple(new[i:i + NGRAM]) for i in range(len(new) - NGRAM + 1)]
            o = old_by_key[(prompts[pi]["path"], s, on)]
            same_prefix = next((i for i, (a, b) in enumerate(zip(new, o["ids"])) if a != b), min(len(new), len(o["ids"])))
            rows.append({"prompt_index": pi, "prompt": prompts[pi]["path"], "seed": s, "adapter": on, "new_tokens": len(new),
                         "wall_s": round(wall, 1), "valid": v["valid"], "validity": v,
                         "mask_changed_steps": len(changed), "mask_changed_by_state": dict(collections.Counter(c["state"] for c in changed)),
                         "first_mask_change": changed[0]["step"] if changed else None,
                         "pre_mask_allowed_min": min(pre_mass), "steps_pre_mask_allowed_below_0.999": sum(a < 0.999 for a in pre_mass),
                         "pre_mask_disallowed_mean": sum(1 - a for a in pre_mass) / len(pre_mass),
                         "unmasked_1726_valid": o["valid"],
                         "unmasked_1726_first_error": o["validity"]["first_error"]["index"] if o["validity"]["first_error"] else None,
                         "same_tokens_as_unmasked_until": same_prefix, "identical_to_unmasked": new == o["ids"],
                         "notes": len(new_ev), "pitch_range": [min(ps), max(ps)] if ps else None,
                         "longest_repeat": longest_repeat(ps), "max_polyphony": max_polyphony(new_ev) if new_ev else 0,
                         "end_extension_seconds": round((max(e for _, e, _ in ev) - max((e for _, e, _ in pev), default=0)) / 1000, 3) if new_ev else 0.0,
                         "new_span_seconds": round((max(e for _, e, _ in abs_new) - min(o_ for o_, _, _ in abs_new)) / 1000, 3) if abs_new else 0.0,
                         "ended_by_eos": bool(new) and new[-1] == eos_id,
                         "train_16gram_hits": sum(1 for x in grams if x in train_grams), "16grams": len(grams)})
            say(f"{tag_of(buf)} done {len(new)} tokens {wall:.0f} s, valid={v['valid']}, mask changed {len(changed)} steps, total {time.perf_counter() - t_all:.0f} s")
            # a format-invalid masked stream is an implementation bug: stop now instead of spending the rest of the budget
            check(f"stream_{buf['n']}_format_valid", v["valid"], first_error=v["first_error"])

        def save(result):
            for b in result["completed"] + ([result["incomplete"]] if result["incomplete"] else []):
                b.pop("_t", None)
            with open(out_path, "w") as f:
                json.dump(result, f)
            log["generations_masked_sha256"] = file_sha(out_path)
            log["streams"] = rows
            log["incomplete"] = {k: v for k, v in result["incomplete"].items() if k not in ("ids", "pre_mask_allowed")} if result["incomplete"] else None
            log["phases"]["generate"] = {"wall_s": round(time.perf_counter() - t_all, 1), **rss(), **pressure()}

        run_streams(specs, run_one, after, save)
        check("mask_implementation_all_valid", len(rows) == 16 and all(r["valid"] for r in rows),
              invalid=[(r["prompt_index"], r["seed"], r["adapter"]) for r in rows if not r["valid"]])
        summary = {}
        for arm, on in (("base", False), ("adapter", True)):
            rs = [r for r in rows if r["adapter"] == on]
            summary[arm] = {"streams": len(rs), "streams_mask_changed": sum(r["mask_changed_steps"] > 0 for r in rs),
                            "mask_changed_steps_total": sum(r["mask_changed_steps"] for r in rs),
                            "mask_changed_by_state": dict(sum((collections.Counter(r["mask_changed_by_state"]) for r in rs), collections.Counter())),
                            "identical_to_unmasked": sum(r["identical_to_unmasked"] for r in rs),
                            "pre_mask_allowed_min": min(r["pre_mask_allowed_min"] for r in rs),
                            "pre_mask_disallowed_mean": sum(r["pre_mask_disallowed_mean"] for r in rs) / len(rs),
                            "train_16gram_hits": sum(r["train_16gram_hits"] for r in rs)}
        log["summary"] = summary
        log["result"] = "generate_done"
    except (Stop, MaskStop) as e:
        log["result"], log["stopped_at"] = "stopped", str(e)
    except RuntimeError as e:
        log["result"], log["stopped_at"] = "error", f"{type(e).__name__}: {e}"[:500]
    finally:
        signal.alarm(0)
        progress.close()
        with open(report_path, "w") as f:
            json.dump(log, f, ensure_ascii=False, indent=1)
        print(json.dumps({"result": log["result"], "stopped_at": log["stopped_at"],
                          "checks": {k: v["ok"] for k, v in log["checks"].items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
