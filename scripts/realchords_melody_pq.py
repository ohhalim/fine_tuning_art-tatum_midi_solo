#!/usr/bin/env python3
"""ReaLchords melody RL over fixed chord progressions P/Q (issue #1780, docs/experiments/REALCHORDS_MELODY_PQ.md).

4 cells fixed before running: progressions P/Q x seeds 1/2. 8 bars, 4/4, 16 frames per bar, 128 BPM. Bar 1 is the
CMT head primer (8 eighths) with its chords; bars 2-8 sample one melody token per frame, then append that frame's
fixed chord token. Order [BOS, m1, c1, m2, c2, ...]: the melody at frame t sees chords up to t-1. Official online
invalid-token filter, temperature 1.0, torch.multinomial with a fresh generator per cell, full recompute, CPU float32.

Modes (one per process, realjam venv, PYTHONPATH=<realchords-pytorch>; nothing installed):
  --precheck  SHAs, vocab size, strict load, chord names, primer round trip, chord-alignment check, determinism.
  --generate  the 4 cells; per-cell wall 120 s, total 15 min, RSS 8 GB.
  --render    melody-only MIDI (velocity 90) and AAC (fluidsynth + GeneralUser GS), listening page.
  --analyze   CMT_CONTINUOUS.md bar scores and phrase metrics.

  realchords_melody_pq.py <realchords repo> <actor dir> <run dir> <report dir> --precheck|--generate|--render|--analyze
"""
from __future__ import annotations

import hashlib
import json
import os
import resource
import signal
import statistics
import subprocess
import sys
import time

PROG = {"P": ["Gm7", "C7", "Fmaj7", "Fmaj7"] * 2, "Q": ["Gm7b5", "C7", "Fm7", "Fm7"] * 2}
HEAD_BAR1 = [67, 69, 70, 72, 74, 72, 70, 69]                 # CMT t1_cmt_generate.py HEAD_BAR1, eighths over bar 1
FPB, BARS, BPM = 16, 8, 128
FRAME_S = 60 / BPM / 4                                       # 0.1171875 s per 16th
SEEDS = (1, 2)
CELLS = [("P", 1), ("Q", 1), ("P", 2), ("Q", 2)]
FIRST_LISTEN = [("P", 1), ("Q", 1)]
CELL_WALL_S, TOTAL_CAP_S, RSS_CAP = 120, 15 * 60, 8 * 2 ** 30
VELOCITY = 90
SHA = {"actor.pth": "c14130e6b4cacf3fbb506d32e35c55f9063e083d435951880c52171aef67846b",
       "args.yml": "cc81ffa7346e4925a1b496db6fccba3bab22f33d34bbc4e0af7ba35001ee45b2",
       "chord_names_augmented.json": "17db1594d39324492ee263e068da5fc6100015c34b7e82fbd734de2f7636cb2b"}
N_TOKENS = 5902
PCS = {"Gm7": {7, 10, 2, 5}, "C7": {0, 4, 7, 10}, "Fmaj7": {5, 9, 0, 4}, "Gm7b5": {7, 10, 1, 5}, "Fm7": {5, 8, 0, 3}}
UP3 = {"Gm7": "Bbm7", "C7": "Eb7", "Fmaj7": "Abmaj7", "Gm7b5": "Bbm7b5", "Fm7": "Abm7"}   # alignment check partner
SCORED = {3: "F", 4: "F", 5: "G", 7: "F", 8: "F"}           # bars (1-based) where P and Q differ, primer bar 1 excluded
P_TONES = {"F": {9, 4}, "G": {2}}                          # Fmaj7 A E / Gm7 D
Q_TONES = {"F": {8, 3}, "G": {1}}                          # Fm7 Ab Eb / Gm7b5 Db
SF2 = os.path.expanduser("~/.local/share/soundfonts/generaluser-gs/v1.471.sf2")
MODES = ("--precheck", "--generate", "--render", "--analyze")


class Stop(Exception):
    pass


def cell_name(prog, seed):
    return f"{prog}_s{seed}"


def file_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def rss_bytes():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss          # bytes on macOS


def thermal_level():
    out = subprocess.run(["notifyutil", "-g", "com.apple.system.thermalpressurelevel"], capture_output=True, text=True).stdout
    try:
        return int(out.split()[-1])
    except (IndexError, ValueError):
        return -1


def chord_frames(chords, fpb=FPB):
    """Token names per frame: CHORD_ON_x where the chord changes, CHORD_x otherwise (a repeated bar holds)."""
    out, prev = [], None
    for c in chords:
        for k in range(fpb):
            out.append(f"CHORD_ON_{c}" if (k == 0 and c != prev) else f"CHORD_{c}")
        prev = c
    return out


def primer_frames(pitches=HEAD_BAR1, step=2):
    """Eighths on the 16th grid: NOTE_ON_p then NOTE_p (hold)."""
    out = []
    for p in pitches:
        out += [f"NOTE_ON_{p}"] + [f"NOTE_{p}"] * (step - 1)
    return out


def frames_to_notes(names):
    """[(start_frame, end_frame, pitch)] from melody token names; SILENCE ends a note, a new onset starts one."""
    notes, cur = [], None
    for t, n in enumerate(names):
        if n.startswith("NOTE_ON_"):
            if cur:
                notes.append((cur[0], t, cur[1]))
            cur = (t, int(n[8:]))
        elif n.startswith("NOTE_") and cur and int(n[5:]) == cur[1]:
            continue
        else:                                   # SILENCE (or an orphan hold, which the filter forbids)
            if cur:
                notes.append((cur[0], t, cur[1]))
            cur = None
    if cur:
        notes.append((cur[0], len(names), cur[1]))
    return notes


def generate_cell(next_logits, filter_fn, bos, primer_ids, chord_ids, seed, buf, check=None):
    """Fill buf["melody"] (primer + sampled) in place so a stop keeps what was made."""
    import torch
    gen = torch.Generator().manual_seed(seed)
    seq = [bos]
    for m, c in zip(primer_ids, chord_ids):
        seq += [m, c]
    buf["melody"] = list(primer_ids)
    for t in range(len(primer_ids), len(chord_ids)):
        x = torch.tensor([seq])
        logits = filter_fn(next_logits(x).unsqueeze(0), x, t)[0]
        p = torch.softmax(logits.float(), -1)
        tok = int(torch.multinomial(p, 1, generator=gen))
        buf["melody"].append(tok)
        seq += [tok, chord_ids[t]]
        if check:
            check()
    return seq


def bar_scores(notes):
    """CMT_CONTINUOUS.md bar scoring on frame-time notes (start_frame, end_frame, pitch)."""
    bars = {}
    for b, key in SCORED.items():
        ns = [n for n in notes if (b - 1) * FPB <= n[0] < b * FPB]
        bars[b] = {"n": len(ns), "P": sum(n[2] % 12 in P_TONES[key] for n in ns),
                   "Q": sum(n[2] % 12 in Q_TONES[key] for n in ns),
                   "first16": sum(1 for n in ns if n[0] == (b - 1) * FPB)}
    return bars


def pair_diff(bars_p, bars_q):
    d = {}
    for b in SCORED:
        a, c = bars_p[b], bars_q[b]
        fa = (a["P"] - a["Q"]) / a["n"] if a["n"] else None
        fc = (c["P"] - c["Q"]) / c["n"] if c["n"] else None
        d[b] = None if fa is None or fc is None else round(fa - fc, 3)
    return d


def events(notes_s, win=0.03):
    """CMT events: (start_s, pitch tuple) groups of onsets within win."""
    ev, cur = [], []
    for n in sorted(notes_s):
        if cur and n[0] - cur[0][0] > win:
            ev.append((cur[0][0], tuple(sorted(m[2] for m in cur))))
            cur = []
        cur.append(n)
    if cur:
        ev.append((cur[0][0], tuple(sorted(m[2] for m in cur))))
    return ev


def longest_repeat(ev, kmax=24):
    """CMT longest_repeat (LOOP_DIAG): period <= kmax events, at least 3 repetitions; seconds."""
    best = 0.0
    for k in range(1, kmax + 1):
        run = 0
        for i in range(len(ev) - k):
            if ev[i][1] == ev[i + k][1]:
                run += 1
                if (run + k) / k >= 3:
                    best = max(best, ev[i + k][0] - ev[i - run + 1][0])
            else:
                run = 0
    return best


def phrase_metrics(notes):
    """Generated part (bars 2-8) in seconds at 128 BPM, CMT definitions."""
    gen = sorted((s * FRAME_S, e * FRAME_S, p) for s, e, p in notes if s >= FPB)
    if not gen:
        return {"gen_notes": 0}
    span0, span1 = gen[0][0], max(n[1] for n in gen)
    gaps = [b[0] - a[1] for a, b in zip(gen, gen[1:]) if b[0] - a[1] >= 0.3]
    durs = sorted(n[1] - n[0] for n in gen)
    return {"gen_notes": len(gen), "notes_per_s": round(len(gen) / (span1 - span0), 2),
            "rests_ge_0.3s": len(gaps), "rest_time_share": round(sum(gaps) / (span1 - span0), 3),
            "dur_median_s": round(statistics.median(durs), 3), "dur_p90_s": round(durs[int(0.9 * (len(durs) - 1))], 3),
            "dur_lt_0.1s_share": round(sum(d < 0.1 for d in durs) / len(durs), 3),
            "longest_repeat_s": round(longest_repeat(events(gen)), 2)}


def chord_tone_share(notes, chords):
    """Descriptive only: share of notes whose pitch class is in that bar's chord."""
    out = {}
    for b in range(2, BARS + 1):
        ns = [n for n in notes if (b - 1) * FPB <= n[0] < b * FPB]
        out[b] = round(sum(n[2] % 12 in PCS[chords[b - 1]] for n in ns) / len(ns), 3) if ns else None
    return out


def main() -> None:
    repo, actor_dir, run_dir, report_dir, mode = sys.argv[1:6]
    assert mode in MODES, mode
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)
    log = {"issue": 1780, "mode": mode, "checks": {}, "result": None, "stopped_at": None}
    report_path = os.path.join(report_dir, mode.strip("-") + ".json")

    def check(name, ok, **detail):
        log["checks"][name] = {"ok": bool(ok), **detail}
        if not ok:
            raise Stop(f"check failed: {name}")

    try:
        if mode == "--render":
            render(run_dir, report_dir, log, check)
        elif mode == "--analyze":
            analyze(run_dir, report_dir, log, check)
        else:
            models(repo, actor_dir, run_dir, log, check, mode)
    except BaseException as e:   # noqa: BLE001 - every stop is recorded, then re-raised
        log["stopped_at"] = f"{type(e).__name__}: {e}"
        raise
    finally:
        with open(report_path, "w") as f:
            json.dump(log, f, ensure_ascii=False, indent=1)


def models(repo, actor_dir, run_dir, log, check, mode):
    import torch
    sys.path.insert(0, repo)
    from realchords.dataset.hooktheory_tokenizer import HooktheoryTokenizer
    from realchords.model.gen_model import DecoderTransformer
    from realchords.model.sampling import filter_invalid_tokens_generate_online
    from realchords.utils.inference_utils import load_rl_model

    names_path = os.path.join(repo, "checkpoints", "chord_names_augmented.json")
    paths = {"actor.pth": os.path.join(actor_dir, "actor.pth"), "args.yml": os.path.join(actor_dir, "args.yml"),
             "chord_names_augmented.json": names_path}
    got = {k: file_sha(v) for k, v in paths.items()}
    log["inputs"] = {"sha256": got, "repo_head": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=repo).stdout.strip(),
                     "torch": torch.__version__}
    check("input_sha256", got == SHA, got=got)
    chord_names = json.load(open(names_path))
    tok = HooktheoryTokenizer(chord_names=chord_names)
    check("vocab_5902", tok.num_tokens == N_TOKENS, tokenizer=tok.num_tokens)
    check("chord_names_exact", all(c in chord_names for c in list(PCS) + list(UP3.values())))
    torch.set_num_threads(4)
    t0 = time.perf_counter()
    base = DecoderTransformer(dim=512, depth=8, heads=8, max_seq_len=514, dropout=0.1, num_tokens=tok.num_tokens, pad_value=tok.pad_token)
    model = load_rl_model(paths["actor.pth"], base, compile=False).float().eval()     # load_state_dict(strict=True)
    log["checks"]["strict_load"] = {"ok": True}
    log["load_s"] = round(time.perf_counter() - t0, 2)
    ids = tok.name_to_id
    primer_ids = [ids[n] for n in primer_frames()]
    filt = lambda logits, x, t: filter_invalid_tokens_generate_online(logits, x, t, model_part="melody", tokenizer=tok)  # noqa: E731

    def next_logits(x):
        with torch.no_grad():
            return model.decoder(x)[0, -1]

    if mode == "--precheck":
        back = frames_to_notes(primer_frames())
        check("primer_roundtrip", [(s, e, p) for s, e, p in back] == [(2 * i, 2 * i + 2, p) for i, p in enumerate(HEAD_BAR1)], notes=back)
        align = {}
        for c, up in UP3.items():
            mass = {}
            for ctx in (c, up):
                seq = [tok.bos_token]
                for k in range(8):
                    seq += [tok.silence_token, ids[f"CHORD_ON_{ctx}" if k == 0 else f"CHORD_{ctx}"]]
                x = torch.tensor([seq])
                p = torch.softmax(filt(next_logits(x).unsqueeze(0), x, 8)[0].float(), -1)
                mass[ctx] = float(sum(p[ids[f"NOTE_ON_{q}"]] for q in range(128) if q % 12 in PCS[c]))
            align[c] = {"own": round(mass[c], 4), "up3": round(mass[up], 4)}
        check("chord_alignment", all(v["own"] > v["up3"] for v in align.values()), per_chord=align)
        a, b = {}, {}
        cids = [ids[n] for n in chord_frames(PROG["P"])][:48]
        generate_cell(next_logits, filt, tok.bos_token, primer_ids, cids, 1, a)
        generate_cell(next_logits, filt, tok.bos_token, primer_ids, cids, 1, b)
        check("same_seed_same_tokens", a["melody"] == b["melody"], frames=len(a["melody"]))
        log["result"] = "precheck_ok"
        print("precheck ok", json.dumps(align))
        return

    cells_dir = os.path.join(run_dir, "cells")
    os.makedirs(cells_dir, exist_ok=True)
    log["cells"], used = {}, 0.0

    def on_alarm(*_):
        raise Stop(f"cell wall {CELL_WALL_S} s")

    def check_rss():
        if rss_bytes() > RSS_CAP:
            raise Stop(f"rss over {RSS_CAP}")

    signal.signal(signal.SIGALRM, on_alarm)
    for prog, seed in CELLS:
        name = cell_name(prog, seed)
        if used > TOTAL_CAP_S:
            log["cells"][name] = {"status": "not_run", "reason": "total cap"}
            continue
        chord_names_f = chord_frames(PROG[prog])
        buf = {"cell": name, "prog": prog, "seed": seed, "chords": chord_names_f, "melody": [], "thermal_before": thermal_level()}
        t = time.perf_counter()
        signal.alarm(CELL_WALL_S)
        status = "done"
        try:
            generate_cell(next_logits, filt, tok.bos_token, primer_ids, [ids[n] for n in chord_names_f], seed, buf, check_rss)
        except Stop as e:
            status, buf["stop"] = "failed", str(e)
        finally:
            signal.alarm(0)
        buf["wall_s"] = round(time.perf_counter() - t, 2)
        used += buf["wall_s"]
        buf["status"] = status
        buf["melody_names"] = [tok.id_to_name[i] for i in buf["melody"]]
        bad = [i for i, n in enumerate(buf["melody_names"]) if not (n == "SILENCE" or n.startswith("NOTE_"))]
        orphan = [i for i, n in enumerate(buf["melody_names"]) if n.startswith("NOTE_") and not n.startswith("NOTE_ON_")
                  and (i == 0 or buf["melody_names"][i - 1] not in (f"NOTE_ON_{n[5:]}", n))]
        buf["filter_violations"] = len(bad) + len(orphan)
        with open(os.path.join(cells_dir, name + ".json"), "w") as f:
            json.dump(buf, f)
        log["cells"][name] = {k: buf.get(k) for k in ("status", "stop", "wall_s", "thermal_before", "filter_violations")}
        print(name, status, buf["wall_s"], "s")
    log["used_s"] = round(used, 2)
    check("no_filter_violation", all(c.get("filter_violations", 0) == 0 for c in log["cells"].values()))
    log["result"] = "generate_done"


def load_cells(run_dir):
    out = {}
    for prog, seed in CELLS:
        path = os.path.join(run_dir, "cells", cell_name(prog, seed) + ".json")
        out[(prog, seed)] = json.load(open(path)) if os.path.exists(path) else None
    return out


def write_midi(notes, path):
    import pretty_midi
    pm = pretty_midi.PrettyMIDI(initial_tempo=BPM)
    inst = pretty_midi.Instrument(program=0, name="melody")
    inst.notes = [pretty_midi.Note(velocity=VELOCITY, pitch=p, start=s * FRAME_S, end=e * FRAME_S) for s, e, p in notes]
    pm.instruments.append(inst)
    pm.write(path)


def render(run_dir, report_dir, log, check):
    import shutil
    gen = json.load(open(os.path.join(report_dir, "generate.json")))
    check("generate_done", gen.get("result") == "generate_done", result=gen.get("result"))
    check("renderer", shutil.which("fluidsynth") is not None and shutil.which("afconvert") is not None and os.path.exists(SF2))
    cells = load_cells(run_dir)
    out_dir = os.path.join(run_dir, "listening")
    os.makedirs(out_dir, exist_ok=True)
    files = {}
    for (prog, seed), c in cells.items():
        name = cell_name(prog, seed)
        notes = frames_to_notes(c["melody_names"])
        overlaps = sum(1 for a, b in zip(notes, notes[1:]) if b[0] < a[1])
        mid = os.path.join(out_dir, name + ".mid")
        write_midi(notes, mid)
        wav, m4a = mid[:-4] + ".wav", mid[:-4] + ".m4a"
        subprocess.run(["fluidsynth", "-ni", "-q", "-F", wav, "-r", "44100", SF2, mid], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(["afconvert", "-f", "m4af", "-d", "aac", wav, m4a], check=True)
        os.remove(wav)
        files[name] = {"notes": len(notes), "overlaps": overlaps, "midi_sha256": file_sha(mid)}
    check("monophonic_no_overlap", all(v["overlaps"] == 0 for v in files.values()), files=files)
    log["render"] = {"files": files, "first_listen": [cell_name(*c) for c in FIRST_LISTEN], "velocity": VELOCITY, "bpm": BPM, "soundfont": SF2}
    log["result"] = "render_done"


def analyze(run_dir, report_dir, log, check):
    cells = load_cells(run_dir)
    check("four_cells", all(c is not None and c["status"] == "done" for c in cells.values()))
    rows = {}
    for (prog, seed), c in cells.items():
        notes = frames_to_notes(c["melody_names"])
        c7 = {b: sum(1 for n in notes if (b - 1) * FPB <= n[0] < b * FPB) for b in (2, 6)}
        rows[cell_name(prog, seed)] = {"bars": bar_scores(notes), "c7_bar_notes": c7, **phrase_metrics(notes),
                                       "chord_tone_share": chord_tone_share(notes, PROG[prog]), "wall_s": c["wall_s"]}
    log["cells"] = rows
    log["pairs"] = {s: pair_diff(rows[cell_name("P", s)]["bars"], rows[cell_name("Q", s)]["bars"]) for s in SEEDS}
    log["result"] = "analyze_done"


if __name__ == "__main__":
    main()
