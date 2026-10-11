#!/usr/bin/env python3
"""Fixed data for the Mehldau Aria LoRA pilot (issue #1726). Written before any training.

Recordings: every Brad Mehldau file in the PiJAMA `midi` tree (studio 18 + live 54). The `midi_kong`
file with the same relative path is the same recording in another transcription; it is recorded as a
group member and never used.

Groups (fixed rules): same normalised title -> one group; then any two recordings of different groups
whose transposition-invariant pitch-interval 16-grams overlap by containment >= 0.2 are merged too.
Titles and n-grams cannot find every duplicate (medleys, renamed tunes, loose quotes); that limit stays.
Split by group: sorted group keys, random.Random(0).shuffle, n_test = round(0.15 G), n_val =
round(0.15 G), test first, then val, rest train.

Windows: each recording is tokenised once with the unchanged AbsTokenizer (pedal extension on,
add_dim_tok False, add_eos_tok False), detokenised to notes, and cut into consecutive windows of at
most 1024 tokens. A window starts and ends on a note boundary and never splits notes sharing an onset;
the window's notes are rebuilt as a MidiDict with the first onset at 0 and tokenised again, so time is
re-normalised by the tokenizer itself. No window carries <E> or <D> (<D> is the tokenizer's
"ending soon" marker ~100 tokens before the end; in a cut window it would be a false end signal, and
whether a file ends at the true end of the performance is not checked). Velocity stays the Aria
quantised original. A remainder under 32 notes is dropped and counted. Loss targets are the tokens
after <S>.

Probe: pianists other than Mehldau and Tatum with >= 5 recordings whose normalised titles are not
Mehldau titles; random.Random(0).sample of 4 pianists, then 5 recordings each (sorted pools). Not used
for training. Interval-n-gram overlap with Mehldau is reported, not used to resample.
Evaluation uses at most the first 8 windows of each val / test / probe recording.
Generation prompts: the first 4 val groups (sorted key), first recording, notes with onset < 8 s.

  mehldau_aria_data.py <aria repo> <data json (outside repo)> <manifest json (repo)>
"""
from __future__ import annotations

import collections
import hashlib
import json
import math
import os
import random
import re
import sys
from pathlib import Path

PIJAMA = Path("/Users/ohhalim/git_box/fine_tuning_art-tatum_midi_solo/midi_dataset")
TARGET, EXCLUDE_PIANISTS = "Brad Mehldau", {"Brad Mehldau", "Art Tatum"}
SEED, MAX_TOKENS, MIN_NOTES, NGRAM, MERGE_AT = 0, 1024, 32, 16, 0.2
EVAL_WINDOWS, PROMPT_MS, N_PROMPTS = 8, 8000, 4
PROBE_PIANISTS, PROBE_PER_PIANIST, MIN_POOL = 4, 5, 5


def norm_title(name: str) -> str:
    s = Path(name).stem.lower()
    s = re.sub(r"\s+-\s+live\b.*$", "", s)        # "... - Live", "... - Live At <venue>, <date>"
    s = re.sub(r"\(.*?\)|\[.*?\]", "", s)
    s = re.sub(r"\blive\b", "", s)
    s = re.sub(r"[^a-z0-9 ]", "", s)
    return re.sub(r"\s+", " ", s).strip()


def recordings_of(pianist):
    out = []
    for kind in ("studio", "live"):
        root = PIJAMA / "midi" / kind / pianist
        if root.exists():
            for f in sorted(p for ext in (".mid", ".midi") for p in root.rglob(f"*{ext}")):
                rel = f.relative_to(PIJAMA / "midi")
                out.append({"path": str(rel), "kind": kind, "pianist": pianist, "album": f.parent.name,
                            "title": norm_title(f.name),
                            "kong": str(rel) if (PIJAMA / "midi_kong" / rel).exists() else None})
    return out


def interval_ngrams(notes):
    ps = [n["data"]["pitch"] for n in sorted(notes, key=lambda n: (n["data"]["start"], n["data"]["pitch"]))]
    iv = [b - a for a, b in zip(ps, ps[1:])]
    return {tuple(iv[i:i + NGRAM]) for i in range(len(iv) - NGRAM + 1)}


def containment(a, b):
    return len(a & b) / min(len(a), len(b)) if a and b else 0.0


class Encoder:
    def __init__(self, aria_repo):
        sys.path.insert(0, aria_repo)
        from ariautils.midi import MidiDict
        from ariautils.tokenizer import AbsTokenizer
        self.MidiDict, self.tok = MidiDict, AbsTokenizer()

    def tokens(self, md):
        return self.tok.tokenize(md, add_dim_tok=False, add_eos_tok=False)

    def full(self, rel):
        md = self.MidiDict.from_midi(str(PIJAMA / "midi" / rel))
        t = self.tokens(md)
        d = self.tok.detokenize(t)
        assert self.tokens(d) == t, f"tokenize(detokenize(t)) != t for {rel}"
        return t, d

    def sub(self, d, notes):
        t0 = notes[0]["data"]["start"]
        shifted = [{"type": "note", "channel": n["channel"], "tick": n["tick"] - t0,
                    "data": {**n["data"], "start": n["data"]["start"] - t0, "end": n["data"]["end"] - t0}} for n in notes]
        md = self.MidiDict(meta_msgs=[], tempo_msgs=d.tempo_msgs, pedal_msgs=[], instrument_msgs=d.instrument_msgs,
                           note_msgs=shifted, ticks_per_beat=d.ticks_per_beat, metadata={})
        return self.tokens(md), shifted


def windows_of(enc, d):
    notes = d.note_msgs
    n, out, s0, dropped = len(notes), [], 0, 0
    while s0 < n:
        s1 = min(n, s0 + (MAX_TOKENS - 2) // 3)
        while True:
            while s0 < s1 < n and notes[s1]["data"]["start"] == notes[s1 - 1]["data"]["start"]:
                s1 -= 1                                  # never split a chord onset
            toks, shifted = enc.sub(d, notes[s0:s1])
            if len(toks) <= MAX_TOKENS or s1 - s0 <= MIN_NOTES:
                break
            s1 -= max(1, (len(toks) - MAX_TOKENS) // 3)
        if s1 - s0 < MIN_NOTES or len(toks) > MAX_TOKENS:
            dropped += n - s0
            break
        start = notes[s0]["data"]["start"]
        back = enc.tok.detokenize(toks).note_msgs
        same = [(b["data"]["pitch"], b["data"]["start"], b["data"]["end"], b["data"]["velocity"]) for b in back] == \
               [(x["data"]["pitch"], x["data"]["start"], x["data"]["end"], x["data"]["velocity"]) for x in shifted]
        out.append({"note_range": [s0, s1], "start_ms": start, "n_tokens": len(toks),
                    "sounding_at_start": sum(1 for x in notes[:s0] if x["data"]["end"] > start),
                    "has_eos_or_dim": any(t in ("<E>", "<D>") for t in toks), "renormalised_roundtrip_exact": same,
                    "s_index": toks.index("<S>"), "ids": enc.tok.encode(toks)})
        s0 = s1
    return out, dropped


def sha_ids(ids):
    return hashlib.sha256(json.dumps(ids).encode()).hexdigest()


def main() -> None:
    aria_repo, data_path, manifest_path = sys.argv[1:4]
    enc = Encoder(aria_repo)
    recs = recordings_of(TARGET)
    grams, full_sha = {}, {}
    for r in recs:
        t, d = enc.full(r["path"])
        r["_d"] = d
        r["n_notes"] = len(d.note_msgs)
        r["duration_s"] = round(max(x["data"]["end"] for x in d.note_msgs) / 1000, 2)
        full_sha[r["path"]] = sha_ids(enc.tok.encode(t))
        grams[r["path"]] = interval_ngrams(d.note_msgs)

    # groups: title first, then n-gram containment merge (union-find)
    parent = {r["path"]: r["path"] for r in recs}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    by_title = collections.defaultdict(list)
    for r in recs:
        by_title[r["title"]].append(r["path"])
    for paths in by_title.values():
        for p in paths[1:]:
            parent[find(p)] = find(paths[0])
    overlaps = []
    for i, a in enumerate(recs):
        for b in recs[i + 1:]:
            c = containment(grams[a["path"]], grams[b["path"]])
            if c > 0:
                overlaps.append({"a": a["path"], "b": b["path"], "containment": round(c, 4), "same_title": a["title"] == b["title"]})
            if c >= MERGE_AT and find(a["path"]) != find(b["path"]):
                parent[find(b["path"])] = find(a["path"])
    members = collections.defaultdict(list)
    for r in recs:
        members[find(r["path"])].append(r)
    groups = {}
    for root, rs in members.items():
        key = sorted(x["title"] for x in rs)[0] + " | " + sorted(x["path"] for x in rs)[0]
        groups[key] = rs
    keys = sorted(groups)
    random.Random(SEED).shuffle(keys)
    g = len(keys)
    n_test, n_val = round(0.15 * g), round(0.15 * g)
    split_of = {k: ("test" if i < n_test else "val" if i < n_test + n_val else "train") for i, k in enumerate(keys)}
    for k, rs in groups.items():
        for r in rs:
            r["group"], r["split"] = k, split_of[k]

    # probe
    mehldau_titles = {r["title"] for r in recs}
    pools = {}
    for pianist in sorted(os.listdir(PIJAMA / "midi" / "studio") + os.listdir(PIJAMA / "midi" / "live")):
        if pianist in EXCLUDE_PIANISTS or pianist in pools:
            continue
        pool = [r for r in recordings_of(pianist) if r["title"] not in mehldau_titles]
        if len(pool) >= MIN_POOL:
            pools[pianist] = pool
    chosen = random.Random(SEED).sample(sorted(pools), PROBE_PIANISTS)
    probe = []
    for pianist in chosen:
        probe += random.Random(SEED).sample(pools[pianist], PROBE_PER_PIANIST)
    for r in probe:
        t, d = enc.full(r["path"])
        r["_d"], r["split"], r["group"] = d, "probe", None
        r["n_notes"], r["duration_s"] = len(d.note_msgs), round(max(x["data"]["end"] for x in d.note_msgs) / 1000, 2)
        full_sha[r["path"]] = sha_ids(enc.tok.encode(t))
        gp = interval_ngrams(d.note_msgs)
        r["max_containment_vs_mehldau"] = round(max(containment(gp, grams[m["path"]]) for m in recs), 4)

    data = {"recordings": []}
    for r in recs + probe:
        ws, dropped = windows_of(enc, r.pop("_d"))
        r["windows"], r["dropped_tail_notes"] = len(ws), dropped
        r["eval_windows"] = min(EVAL_WINDOWS, len(ws)) if r["split"] in ("val", "test", "probe") else 0
        r["windows_sha256"] = sha_ids([w["ids"] for w in ws])
        r["full_tokens_sha256"] = full_sha[r["path"]]
        r["window_checks"] = {"max_tokens": max(w["n_tokens"] for w in ws), "any_eos_or_dim": any(w["has_eos_or_dim"] for w in ws),
                              "all_roundtrip_exact": all(w["renormalised_roundtrip_exact"] for w in ws),
                              "sounding_at_start_total": sum(w["sounding_at_start"] for w in ws)}
        data["recordings"].append({"path": r["path"], "split": r["split"], "windows": ws})

    # generation prompts
    prompts = []
    val_keys = sorted(k for k in groups if split_of[k] == "val")[:N_PROMPTS]
    for k in val_keys:
        r = sorted(groups[k], key=lambda x: x["path"])[0]
        _, d = enc.full(r["path"])
        notes = [x for x in d.note_msgs if x["data"]["start"] < PROMPT_MS]
        toks, _ = enc.sub(d, notes)
        prompts.append({"path": r["path"], "group": k, "prompt_notes": len(notes), "prompt_tokens": len(toks),
                        "recording_duration_s": r["duration_s"], "available": r["duration_s"] >= PROMPT_MS / 1000 and len(notes) >= 8,
                        "ids": enc.tok.encode(toks)})
    data["prompts"] = prompts
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    with open(data_path, "w") as f:
        json.dump(data, f)
    with open(data_path, "rb") as f:
        data_sha = hashlib.sha256(f.read()).hexdigest()

    def counts(split):
        rs = [r for r in recs if r["split"] == split]
        return {"groups": len({r["group"] for r in rs}), "recordings": len(rs),
                "studio": sum(r["kind"] == "studio" for r in rs), "live": sum(r["kind"] == "live" for r in rs),
                "windows": sum(r["windows"] for r in rs), "eval_windows": sum(r["eval_windows"] for r in rs),
                "target_tokens": sum(w["n_tokens"] - w["s_index"] - 1 for x in data["recordings"] if x["split"] == split for w in x["windows"])}

    manifest = {"schema": "mehldau_aria_data_v1", "data_json": data_path, "data_sha256": data_sha,
                "rules": {"seed": SEED, "max_tokens": MAX_TOKENS, "min_notes": MIN_NOTES, "ngram": NGRAM, "merge_at": MERGE_AT,
                          "eval_windows_per_recording": EVAL_WINDOWS, "prompt_ms": PROMPT_MS,
                          "tokenize": "AbsTokenizer, pedal extension on, add_dim_tok False, add_eos_tok False",
                          "split": "groups sorted, Random(0).shuffle, test round(0.15G), val round(0.15G), rest train"},
                "groups_total": g, "split_counts": {s: counts(s) for s in ("train", "val", "test")},
                "probe": {"pianists": chosen, "recordings": len(probe), "windows": sum(r["windows"] for r in probe),
                          "eval_windows": sum(r["eval_windows"] for r in probe),
                          "max_containment_vs_mehldau": max(r["max_containment_vs_mehldau"] for r in probe)},
                "merged_by_ngram": [o for o in overlaps if o["containment"] >= MERGE_AT and not o["same_title"]],
                "top_overlaps": sorted(overlaps, key=lambda o: -o["containment"])[:15],
                "prompts": [{k: v for k, v in p.items() if k != "ids"} for p in prompts],
                "recordings": recs + probe}
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=1, ensure_ascii=False)
    print(json.dumps({k: manifest[k] for k in ("groups_total", "split_counts", "probe", "merged_by_ngram")}, ensure_ascii=False, indent=1))
    print("prompts", [(p["path"].split("/")[-1], p["prompt_tokens"], p["available"]) for p in manifest["prompts"]])
    print("data sha", data_sha[:16])


if __name__ == "__main__":
    main()
