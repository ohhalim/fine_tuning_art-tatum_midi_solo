#!/usr/bin/env python3
"""Aria base vs Jazz Pianist Style, small fixed comparison (issue #1774, docs/experiments/JPS_COMPARE.md).

16 cells fixed before running: primers 06/18 x seeds 1/2 x arms base / jps0 / jpsA / jpsB. Same canonical primer
ids, the #1726 sampler (temperature 1, one Exp(1) noise vector per step, argmax p/q), the aria_grammar_mask format
mask with <D>/<E> removed (continuous generation), full recompute every step (no KV cache), a fresh generator per
cell. Generation window: clip time [8,000, 28,000) ms, counted from the primer boundary; stop at the first new note
with onset >= 28,000 ms (dropped) or after 1,024 new tokens (incomplete tail dropped, "short"). No resampling.

Modes (one per process, Aria venv; the JPS repo is imported from <jps code>, nothing installed):
  --precheck  input SHAs, vocab, sampler equivalence, strict loads, logits differ across contexts, JPS trunk vs
              Aria diff (report only), same-seed determinism on 24 tokens per arm, MIDI export.
  --generate  smoke cell (06, 1, jpsA) first, then the budget gate, then the other 15 cells. Per-cell wall 900 s,
              RSS cap 24 GB, total 180 min. Thermal level logged between cells.
  --render    MIDI and AAC for every cell (generated part only, and with the primer); 12 blind pairs.
  --analyze   format validity, length, repetition, polyphony, reference-clip features, copy check.

  jps_compare.py <aria ckpt> <jps code> <jps ckpt dir> <primers json> <run dir> <report dir> --precheck|--generate|--render|--analyze
"""
from __future__ import annotations

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
from aria_grammar_mask import GrammarMask, MaskStop, kind, masked_pick  # noqa: E402

PRIMERS, SEEDS, ARMS = ("06", "18"), (1, 2), ("base", "jps0", "jpsA", "jpsB")
COND_ID = {"jpsA": 8, "jpsB": 2}                         # documented sorted(pijama12.csv) order: Hank Jones, Art Tatum
CA_LAYERS, CTX_LEN, N_ARTISTS = list(range(8, 16)), 4, 12
WIN_START_MS, WIN_END_MS, MAX_NEW = 8000, 28000, 1024
CELL_WALL_S, RSS_CAP, TOTAL_CAP_S, BUDGET_MARGIN = 900, 24 * 2 ** 30, 180 * 60, 1.25
SMOKE = ("06", 1, "jpsA")
CELLS = [(p, s, a) for p in PRIMERS for s in SEEDS for a in ARMS]
SHUFFLE_SEED = 1774
FIRST_SCREEN = [("06", 1, "base", "jpsA"), ("06", 1, "jpsA", "jpsB"), ("18", 1, "base", "jpsB"), ("18", 1, "jpsA", "jpsB")]
SF2 = os.path.expanduser("~/.local/share/soundfonts/generaluser-gs/v1.471.sf2")
FEATURES_BLOB = "4f8cddfff9e6152f6739b207c4d81e0988c98bc7"   # scripts/reference_clip_features.py of PR #1763
PIJAMA = "/Users/ohhalim/git_box/fine_tuning_art-tatum_midi_solo/midi_dataset/midi"
MODES = ("--precheck", "--generate", "--render", "--analyze")


def cell_name(p, s, a):
    return f"p{p}_s{s}_{a}"


def thermal_level():
    out = subprocess.run(["notifyutil", "-g", "com.apple.system.thermalpressurelevel"], capture_output=True, text=True).stdout
    try:
        return int(out.split()[-1])
    except (IndexError, ValueError):
        return -1


class ContinuousMask(GrammarMask):
    """The #1731 format mask without <D> and <E>: generation continues until the time window or the token cap."""

    def __init__(self, vocab):
        import torch
        super().__init__(vocab)
        self.note_ok = self.note_ok & ~torch.tensor([t in ("<D>", "<E>") for t in vocab])


def generate_cell(next_logits, prompt_tokens, prompt_ids, vocab, seed, buf, check_rss=None):
    """Fill buf in place (ids, stop, pre_mask_allowed, mask_changed) so a stop in the middle keeps what was made."""
    import math
    import torch
    gm = ContinuousMask(vocab)
    gm.start(prompt_tokens)
    seg = prompt_tokens.count("<T>")
    gen = torch.Generator().manual_seed(seed)
    seq = list(prompt_ids)
    for step in range(MAX_NEW):
        p = torch.softmax(next_logits(seq), -1).cpu()
        ok = gm.allowed()
        a = p[ok].sum().item()
        if not bool(ok.any()) or not math.isfinite(a) or a <= 0:
            raise MaskStop(f"step {step}: allowed set empty or allowed mass {a}")
        q = torch.empty(len(vocab)).exponential_(1, generator=gen)
        pick, free = masked_pick(p, ok, q)
        buf["pre_mask_allowed"].append(round(a, 6))
        if pick != free:
            buf["mask_changed"].append(step)
        t = vocab[pick]
        if t == "<T>" and (seg + 1) * 5000 >= WIN_END_MS:
            buf["stop"] = "window_end_by_T"
            return
        if kind(t) == "onset" and seg * 5000 + t[1] >= WIN_END_MS:
            buf["ids"].pop()                          # the note start of the note that falls outside the window
            buf["stop"] = "window_end"
            return
        if t == "<T>":
            seg += 1
        gm.advance(t)
        seq.append(pick)
        buf["ids"].append(pick)
        if check_rss and (step + 1) % 64 == 0:
            check_rss()
    buf["stop"] = "token_cap"


def complete_prefix(tokens):
    """Tokens up to the last complete note; a trailing note start or note start + onset is dropped."""
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


def note_list(tokens, seg0=0):
    """[(on_ms, off_ms, pitch, vel)] of complete notes, absolute ms; <T> advances 5 s."""
    seg, out, cur = seg0, [], None
    for t in tokens:
        if t == "<T>":
            seg += 1
        elif isinstance(t, tuple) and t[0] == "piano":
            cur = [t[1], t[2]]
        elif isinstance(t, tuple) and t[0] == "onset" and cur is not None and len(cur) == 2:
            cur.append(seg * 5000 + t[1])
        elif isinstance(t, tuple) and t[0] == "dur" and cur is not None and len(cur) == 3:
            out.append((cur[2], cur[2] + t[1], cur[0], cur[1]))
            cur = None
    return out


def write_midi(notes, path, shift_ms=0):
    """1 tick = 1 ms (tempo 1,000,000 us per beat, 1000 ticks per beat), piano program 0."""
    import mido
    ev = []
    for on, off, p, v in notes:
        ev.append((on - shift_ms, 1, p, v))
        ev.append((off - shift_ms, 0, p, 0))
    ev.sort(key=lambda e: (e[0], e[1]))          # offs before ons at the same tick
    mf = mido.MidiFile(ticks_per_beat=1000)
    tr = mido.MidiTrack()
    tr.append(mido.MetaMessage("set_tempo", tempo=1_000_000, time=0))
    tr.append(mido.Message("program_change", program=0, channel=0, time=0))
    now = 0
    for t, is_on, p, v in ev:
        t = max(0, int(round(t)))
        tr.append(mido.Message("note_on" if is_on else "note_off", note=p, velocity=v if is_on else 0, channel=0, time=t - now))
        now = t
    mf.tracks.append(tr)
    mf.save(path)


def main() -> None:
    aria_ckpt, jps_code, jps_dir, primers_path, run_dir, report_dir, mode = sys.argv[1:8]
    assert mode in MODES, mode
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)
    log = {"issue": 1774, "mode": mode, "checks": {}, "result": None, "stopped_at": None}
    report_path = os.path.join(report_dir, mode.strip("-") + ".json")
    progress = open(os.path.join(run_dir, f"progress_{mode.strip('-')}.log"), "a")

    def say(msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        progress.write(line + "\n")
        progress.flush()

    def check(name, ok, **detail):
        log["checks"][name] = {"ok": bool(ok), **detail}
        if not ok:
            raise Stop(f"check failed: {name}")

    try:
        primers = json.load(open(primers_path))["primers"]
        log["inputs"] = {"primers_json_sha256": file_sha(primers_path),
                         "primer_ids_sha256": {k: v["ids_sha256"] for k, v in primers.items()}}
        if mode == "--render":
            render(run_dir, report_dir, primers, log, check, say)
        elif mode == "--analyze":
            analyze(run_dir, report_dir, primers, log, check, say)
        else:
            models(aria_ckpt, jps_code, jps_dir, primers, run_dir, log, check, say, mode)
    except BaseException as e:   # noqa: BLE001 - every stop is recorded, then re-raised
        log["stopped_at"] = f"{type(e).__name__}: {e}"
        raise
    finally:
        with open(report_path, "w") as f:
            json.dump(log, f, ensure_ascii=False, indent=1)


def models(aria_ckpt, jps_code, jps_dir, primers, run_dir, log, check, say, mode):
    import torch
    from ariautils.tokenizer import AbsTokenizer
    from safetensors.torch import load_file
    sys.path.insert(0, jps_code)
    from aria.config import load_model_config
    from aria.model import ModelConfig, TransformerLM
    from llama_pijama.models import ArtistEmbedding, CrossAttentionTransformerLM

    check("mps_fallback_off", not os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"))
    dev = torch.device("mps")
    tok = AbsTokenizer()
    V = tok.vocab_size
    vocab = [tok.decode([i])[0] for i in range(V)]
    jps_cfg = json.load(open(os.path.join(jps_dir, "config.json")))
    log["inputs"].update({"aria_ckpt_sha256": file_sha(aria_ckpt),
                          "jps_model_sha256": file_sha(os.path.join(jps_dir, "model.safetensors")),
                          "jps_embeddings_sha256": file_sha(os.path.join(jps_dir, "artist_embeddings.safetensors")),
                          "jps_config": jps_cfg, "torch": torch.__version__})
    check("vocab_17727", V == 17727 and jps_cfg.get("vocab_size") == V and jps_cfg.get("num_artists") == N_ARTISTS,
          tokenizer=V, jps=jps_cfg.get("vocab_size"))
    for k, p in primers.items():
        ids = p["ids"]
        check(f"primer_{k}_roundtrip", [list(t) if isinstance(t, tuple) else t for t in tok.decode(ids)] == p["tokens"])

    t0 = time.perf_counter()

    def config():
        c = load_model_config(name="medium")
        c.setdefault("resid_dropout", 0.0)
        c["grad_checkpoint"] = False
        mc = ModelConfig(**c)
        mc.set_vocab_size(V)
        return mc

    base = TransformerLM(config())
    base.load_state_dict(load_file(aria_ckpt), strict=True)
    jps = CrossAttentionTransformerLM(config(), {"layers": CA_LAYERS, "dropout": 0.0, "gate_init": 0.1})
    jps.load_state_dict(load_file(os.path.join(jps_dir, "model.safetensors")), strict=True)
    emb = ArtistEmbedding(num_artists=N_ARTISTS, d_model=config().d_model, context_length=CTX_LEN, dropout=0.0)
    emb.load_state_dict(load_file(os.path.join(jps_dir, "artist_embeddings.safetensors")), strict=True)
    for m in (base, jps, emb):
        for prm in m.parameters():
            prm.requires_grad_(False)
        m.to(dev).eval()
    base.model.freqs_cis = None
    jps.model.freqs_cis = None
    log["checks"]["strict_loads"] = {"ok": True}
    with torch.no_grad():
        ctx = {}
        for arm, cid in COND_ID.items():
            ctx[arm] = emb(torch.tensor([cid], device=dev))
        c0, m0 = emb(torch.tensor([0], device=dev))
        ctx["jps0"] = (torch.zeros_like(c0), m0)        # author --mode ablation: context zeroed, mask kept
    log["load_s"] = round(time.perf_counter() - t0, 1)

    def next_logits_for(arm):
        def f(seq):
            with torch.no_grad():
                x = torch.tensor([seq], device=dev)
                out = base(x) if arm == "base" else jps(x, context=ctx[arm][0], context_mask=ctx[arm][1])
                return out[0, -1].float()
        return f

    def check_rss():
        if rss()["rss_now_bytes"] > RSS_CAP:
            raise Stop(f"rss over {RSS_CAP}")

    def tokens_of(p):
        return [tuple(t) if isinstance(t, list) else t for t in primers[p]["tokens"]]

    if mode == "--precheck":
        g1, g2, same = torch.Generator().manual_seed(2), torch.Generator().manual_seed(2), 0
        for k in range(32):
            pr = torch.softmax(torch.randn(V, generator=torch.Generator().manual_seed(100 + k)) * 3, -1)
            same += torch.multinomial(pr, 1, generator=g1).item() == int(torch.argmax(pr / torch.empty(V).exponential_(1, generator=g2)))
        check("multinomial_is_exponential_race", same == 32, matched=same)
        ids = primers["06"]["ids"]
        first = {arm: next_logits_for(arm)(ids).cpu() for arm in ARMS}
        d0a = (first["jps0"] - first["jpsA"]).abs().max().item()
        dab = (first["jpsA"] - first["jpsB"]).abs().max().item()
        dba = (first["base"] - first["jps0"]).abs().max().item()
        check("first_logits_differ_by_context", d0a > 1e-3 and dab > 1e-3, jps0_vs_jpsA=d0a, jpsA_vs_jpsB=dab, base_vs_jps0=dba)
        bs, js = base.state_dict(), jps.state_dict()
        trunk = {}
        for name, w in bs.items():
            if name in js and js[name].shape == w.shape:
                trunk[name] = ((js[name].float() - w.float()).norm() / (w.float().norm() + 1e-12)).item()
        log["jps_trunk_vs_aria"] = {"shared_tensors": len(trunk), "of_base": len(bs),
                                    "rel_diff_max": max(trunk.values()), "rel_diff_median": sorted(trunk.values())[len(trunk) // 2],
                                    "unchanged_tensors": sum(v == 0 for v in trunk.values()),
                                    "jps_only_tensors": len([n for n in js if n not in bs])}
        det = {}
        for arm in ARMS:
            outs = []
            for _ in range(2):
                f, gm, gen, seq, got = next_logits_for(arm), ContinuousMask(vocab), torch.Generator().manual_seed(1), list(ids), []
                gm.start(tokens_of("06"))
                for _step in range(24):
                    pick, _ = masked_pick(torch.softmax(f(seq), -1).cpu(), gm.allowed(), torch.empty(V).exponential_(1, generator=gen))
                    gm.advance(vocab[pick])
                    seq.append(pick)
                    got.append(pick)
                outs.append(got)
            det[arm] = outs[0] == outs[1]
            if arm == "jpsA":
                notes = note_list(complete_prefix(tokens_of("06") + [vocab[i] for i in outs[0]]))
                write_midi(notes, os.path.join(run_dir, "precheck_jpsA_24.mid"))
        check("same_seed_same_24_tokens", all(det.values()), per_arm=det)
        log["result"] = "precheck_ok"
        say(f"precheck ok, load {log['load_s']} s")
        return

    cells_dir = os.path.join(run_dir, "cells")
    os.makedirs(cells_dir, exist_ok=True)
    order = [SMOKE] + [c for c in CELLS if c != SMOKE]
    log["order"] = [cell_name(*c) for c in order]
    log["cells"] = {}
    used = 0.0

    class Alarm(Exception):
        pass

    def on_alarm(*_):
        raise Alarm(f"cell wall {CELL_WALL_S} s")

    signal.signal(signal.SIGALRM, on_alarm)
    for n, (p, s, arm) in enumerate(order):
        name = cell_name(p, s, arm)
        if n == 1:
            t1 = log["cells"][cell_name(*SMOKE)]["wall_s"]
            est = t1 * 15 * BUDGET_MARGIN
            log["budget"] = {"smoke_s": t1, "estimate_rest_s": round(est, 1), "cap_s": TOTAL_CAP_S, "ok": est <= TOTAL_CAP_S}
            if est > TOTAL_CAP_S:
                for c in order[1:]:
                    log["cells"][cell_name(*c)] = {"status": "not_run", "reason": "budget estimate over cap"}
                log["result"] = "stopped_budget_estimate"
                say(f"budget estimate {est:.0f} s > {TOTAL_CAP_S} s, stopping")
                return
        if used > TOTAL_CAP_S:
            for c in order[n:]:
                log["cells"][cell_name(*c)] = {"status": "not_run", "reason": "total cap reached"}
            log["result"] = "stopped_total_cap"
            say("total cap reached, stopping")
            return
        lvl, waited = thermal_level(), 0
        while lvl >= 2 and waited < 600:
            time.sleep(30)
            waited += 30
            lvl = thermal_level()
        buf = {"cell": name, "primer": p, "seed": s, "arm": arm, "ids": [], "pre_mask_allowed": [], "mask_changed": [],
               "stop": None, "thermal_before": lvl, "thermal_wait_s": waited}
        t = time.perf_counter()
        signal.alarm(CELL_WALL_S)
        status = "done"
        try:
            generate_cell(next_logits_for(arm), tokens_of(p), primers[p]["ids"], vocab, s, buf, check_rss)
        except Alarm as e:
            status, buf["stop"] = "timeout", str(e)
        except MaskStop as e:
            status, buf["stop"] = "mask_stop", str(e)
        finally:
            signal.alarm(0)
        buf["wall_s"] = round(time.perf_counter() - t, 1)
        used += buf["wall_s"]
        buf["status"] = status
        buf["new_tokens"] = len(buf["ids"])
        buf["mask_changed_n"] = len(buf["mask_changed"])
        buf["pre_mask_allowed_min"] = min(buf["pre_mask_allowed"]) if buf["pre_mask_allowed"] else None
        buf.update(rss())
        with open(os.path.join(cells_dir, name + ".json"), "w") as f:
            json.dump(buf, f)
        log["cells"][name] = {k: buf[k] for k in ("status", "stop", "new_tokens", "wall_s", "mask_changed_n",
                                                  "pre_mask_allowed_min", "thermal_before", "thermal_wait_s", "rss_max_bytes")}
        say(f"{name}: {status} {buf['stop']} {buf['new_tokens']} tokens {buf['wall_s']} s, thermal {lvl}, used {used:.0f} s")
    log["used_s"] = round(used, 1)
    log["result"] = "generate_done"


def load_cells(run_dir, primers, vocab_decode):
    out = {}
    for p, s, a in CELLS:
        path = os.path.join(run_dir, "cells", cell_name(p, s, a) + ".json")
        if not os.path.exists(path):
            out[(p, s, a)] = None
            continue
        c = json.load(open(path))
        prompt = [tuple(t) if isinstance(t, list) else t for t in primers[p]["tokens"]]
        gen = complete_prefix(vocab_decode(c["ids"]))
        c["prompt_notes"] = note_list(prompt)
        c["gen_notes"] = note_list(gen, seg0=prompt.count("<T>"))
        c["gen_tokens"] = gen
        out[(p, s, a)] = c
    return out


def render(run_dir, report_dir, primers, log, check, say):
    from ariautils.tokenizer import AbsTokenizer
    tok = AbsTokenizer()
    gen_rep = json.load(open(os.path.join(report_dir, "generate.json")))
    check("generate_done", gen_rep.get("result") == "generate_done", result=gen_rep.get("result"))
    check("fluidsynth_and_soundfont", shutil.which("fluidsynth") is not None and os.path.exists(SF2) and shutil.which("afconvert") is not None)
    cells = load_cells(run_dir, primers, tok.decode)
    check("sixteen_cells", all(c is not None and c["status"] == "done" for c in cells.values()),
          statuses={cell_name(*k): (c["status"] if c else None) for k, c in cells.items()})
    mid_dir, lis_dir = os.path.join(run_dir, "midi"), os.path.join(run_dir, "listening")
    check("listening_dir_empty", not (os.path.isdir(lis_dir) and os.listdir(lis_dir)))
    os.makedirs(mid_dir, exist_ok=True)
    os.makedirs(os.path.join(lis_dir, "with_primer"), exist_ok=True)
    files = {}
    for (p, s, a), c in cells.items():
        name = cell_name(p, s, a)
        shift = min([WIN_START_MS] + [n[0] for n in c["gen_notes"]])
        write_midi(c["gen_notes"], os.path.join(mid_dir, name + "_gen.mid"), shift_ms=shift)
        write_midi(c["prompt_notes"] + c["gen_notes"], os.path.join(mid_dir, name + "_full.mid"))
        files[name] = {"gen_notes": len(c["gen_notes"]), "shift_ms": shift}

    def to_aac(mid, out):
        wav = out[:-4] + ".wav"
        subprocess.run(["fluidsynth", "-ni", "-q", "-F", wav, "-r", "44100", SF2, mid], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(["afconvert", "-f", "m4af", "-d", "aac", wav, out], check=True)
        os.remove(wav)

    rng = random.Random(SHUFFLE_SEED)
    pairs, n = [], 0
    for p in PRIMERS:
        for s in SEEDS:
            for x, y in (("base", "jpsA"), ("base", "jpsB"), ("jpsA", "jpsB")):
                n += 1
                src = [x, y]
                rng.shuffle(src)
                entry = {"pair": n, "primer": p, "seed": s, "comparison": f"{x} vs {y}",
                         "first_screen": (p, s, x, y) in FIRST_SCREEN, "A": src[0], "B": src[1]}
                for label, arm in zip("AB", src):
                    name = cell_name(p, s, arm)
                    to_aac(os.path.join(mid_dir, name + "_gen.mid"), os.path.join(lis_dir, f"pair{n:02d}_{label}.m4a"))
                    to_aac(os.path.join(mid_dir, name + "_full.mid"), os.path.join(lis_dir, "with_primer", f"pair{n:02d}_{label}_full.m4a"))
                pairs.append(entry)
    key_path = os.path.join(run_dir, "key.json")
    with open(key_path, "w") as f:
        json.dump({"shuffle_seed": SHUFFLE_SEED, "soundfont": SF2, "pairs": pairs}, f, indent=1)
    first = [e["pair"] for e in pairs if e["first_screen"]]
    sheet = ["# 블라인드 청취 답안지 (#1774)", "",
             "각 쌍의 A/B는 같은 시작 프레이즈(기준 클립의 처음 8초)·같은 seed에서 서로 다른 모델/조건이 이어 친 20초입니다.",
             "`pairNN_A.m4a`·`pairNN_B.m4a`는 생성 부분만, `with_primer/`는 시작 프레이즈를 포함한 파일입니다.",
             f"먼저 들을 쌍: {', '.join(f'{x:02d}' for x in first)}. 나머지는 원할 때만 들어도 됩니다.",
             "답(`key.json`)은 이 폴더 밖에 있습니다. 다 쓴 뒤에 여세요. 청취는 선택이고, 안 들으면 결과는 미평가로 남습니다.", "",
             "평가 항목(각각 A / B / 같음 / 모름): ① 선율이 이어지고 변형되는가 ② 같은 패턴 반복이 거슬리지 않는가 ③ 컴핑이 솔로를 대신하지 않는가",
             "④ 저음·고음 사용이 의미 있는가 ⑤ 긴장음이 해결되는가 ⑥ 다시 쓰고 싶은가 ⑦ 스윙감이 거슬리는가(덜 거슬리는 쪽)", "",
             "| 쌍 | ① | ② | ③ | ④ | ⑤ | ⑥ | ⑦ | 메모 |", "|---|---|---|---|---|---|---|---|---|"]
    sheet += [f"| {e['pair']:02d}{' (먼저)' if e['first_screen'] else ''} | | | | | | | | |" for e in pairs]
    with open(os.path.join(lis_dir, "ANSWER_SHEET.md"), "w") as f:
        f.write("\n".join(sheet) + "\n")
    log["render"] = {"files": files, "pairs": len(pairs), "first_screen": first, "key_sha256": file_sha(key_path)}
    log["result"] = "render_done"
    say(f"render done: {len(pairs)} pairs")


def analyze(run_dir, report_dir, primers, log, check, say):
    import importlib.util
    import statistics
    import mido
    from ariautils.tokenizer import AbsTokenizer
    from aria_token_validity import check as token_check
    from mehldau_aria_lora_run import longest_repeat, max_polyphony
    tok = AbsTokenizer()
    feat_path = os.path.join(run_dir, "reference_clip_features.py")
    src = subprocess.run(["git", "cat-file", "-p", FEATURES_BLOB], capture_output=True, text=True, cwd=HERE).stdout
    check("features_blob", bool(src))
    open(feat_path, "w").write(src)
    spec = importlib.util.spec_from_file_location("reference_clip_features", feat_path)
    rcf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rcf)
    cells = load_cells(run_dir, primers, tok.decode)

    def pitch_seq(notes):
        return [n[2] for n in sorted(notes, key=lambda n: (n[0], n[2]))]

    def midi_pitches(path):
        t, ons, out = 0.0, {}, []
        for msg in mido.MidiFile(path):
            t += msg.time
            if msg.type == "note_on" and msg.velocity > 0:
                out.append((t, msg.note))
        return [p for _, p in sorted(out)]

    corpora = {}
    for artist in ("Art Tatum", "Hank Jones"):
        seqs = []
        for root, _, fs in os.walk(PIJAMA):
            if os.path.basename(os.path.dirname(root)) == artist or os.path.basename(root) == artist or f"/{artist}/" in root + "/":
                seqs += [midi_pitches(os.path.join(root, f)) for f in fs if f.endswith((".mid", ".midi"))]
        corpora[artist] = seqs
    clips = json.load(open("/Users/ohhalim/git_box/listening/reference_candidates/clips.json"))["clips"]
    for p in PRIMERS:
        srcp = next(c["source"] for c in clips if c["id"] == p)
        corpora[f"source_{p}"] = [midi_pitches(os.path.join(PIJAMA, srcp))]
    log["corpora"] = {k: {"files": len(v), "notes": sum(map(len, v))} for k, v in corpora.items()}

    def longest_match(gen, seqs, interval=False):
        if interval:
            gen = [b - a for a, b in zip(gen, gen[1:])]
            seqs = [[b - a for a, b in zip(s, s[1:])] for s in seqs]
        lo, hi = 0, len(gen)

        def hit(n):
            if n == 0:
                return True
            g = {tuple(gen[i:i + n]) for i in range(len(gen) - n + 1)}
            return any(tuple(s[i:i + n]) in g for s in seqs for i in range(len(s) - n + 1))
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if hit(mid):
                lo = mid
            else:
                hi = mid - 1
        return lo

    rows = {}
    for (p, s, a), c in cells.items():
        name = cell_name(p, s, a)
        if c is None:
            rows[name] = {"status": "missing"}
            continue
        prompt = [tuple(t) if isinstance(t, list) else t for t in primers[p]["tokens"]]
        v = token_check(prompt, tok.decode(c["ids"]))
        gn = c["gen_notes"]
        span_s = (WIN_END_MS - WIN_START_MS) / 1000
        reached = max([n[0] for n in gn], default=None)
        r = {"status": c["status"], "stop": c["stop"], "new_tokens": c["new_tokens"], "wall_s": c["wall_s"],
             "format_valid": v["valid"], "format_first_error": v["first_error"],
             "gen_notes": len(gn), "reached_ms": reached, "short": c["stop"] == "token_cap",
             "longest_repeat": longest_repeat(pitch_seq(gn)), "max_polyphony": max_polyphony([(n[0], n[1], n[2]) for n in gn]) if gn else 0,
             "mask_changed_n": c["mask_changed_n"], "pre_mask_allowed_min": c["pre_mask_allowed_min"]}
        if gn:
            r.update(rcf.features([(n[2], n[0], n[1]) for n in gn], span_s))
        ps = pitch_seq(gn)
        r["copy"] = {k: {"pitch": longest_match(ps, seqs), "interval": longest_match(ps, seqs, interval=True)} for k, seqs in corpora.items()}
        rows[name] = r
    log["cells"] = rows
    by_arm = {}
    for a in ARMS:
        rs = [rows[cell_name(p, s, a)] for p in PRIMERS for s in SEEDS if rows[cell_name(p, s, a)].get("gen_notes") is not None]
        by_arm[a] = {k: statistics.median(r[k] for r in rs) for k in ("gen_notes", "notes_per_s", "comping_per_s", "low_share",
                                                                     "high_share", "lick_len_p90", "longest_repeat", "max_polyphony")
                     if rs and all(k in r for r in rs)}
    log["median_by_arm"] = by_arm
    log["result"] = "analyze_done"
    say("analyze done")


if __name__ == "__main__":
    main()
