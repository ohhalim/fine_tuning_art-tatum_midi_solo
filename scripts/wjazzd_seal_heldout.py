#!/usr/bin/env python3
"""Sealed WJazzD held-out set for a later single final evaluation (docs/experiments/ARIA_COND_CONTRACT_V2.md).

No model is loaded. Rules fixed before selection:
- candidates: compositions eligible under the pilot rules (template not Blues / I Got Rhythm, 4/4,
  not piano) that the pilot did not use for training, scoring or generation
- family separation: drop a candidate whose normalised title equals a pilot composition title or
  template, or whose template equals a pilot title or template; inside the sealed set keep only the
  first composition of any shared title or template
- order: sorted compids shuffled with random.Random(20261011); one solo per composition picked with
  the same generator; a composition is taken only if the pilot chunk rules give at least one valid
  chunk, otherwise it is skipped and recorded; never replaced because of any model score
- tokenisation is made reproducible by seeding Python's global random (used by the tokenizer for
  <D>) with 20261011 + melid right before each solo is built
Writes the sealed chunk data locally and a manifest with the selection, exclusions and hashes.
Run with the Aria venv: wjazzd_seal_heldout.py <aria repo> <wjazzd.db> <pilot data json> <local out json> <manifest json>
"""
from __future__ import annotations

import collections
import hashlib
import json
import os
import random
import sqlite3
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from wjazzd_pilot_data import song_chunks_for  # noqa: E402

SEED, TAKE = 20261011, 16


def norm(s):
    return " ".join((s or "").lower().split())


def sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def main() -> None:
    aria_repo, db, pilot_data, local_out, manifest_out = sys.argv[1:6]
    sys.path.insert(0, aria_repo)
    from ariautils.midi import MidiDict
    from ariautils.tokenizer import AbsTokenizer
    tok = AbsTokenizer()
    c = sqlite3.connect(db)
    with open(pilot_data) as f:
        pilot = json.load(f)
    pilot_comp = {s["compid"] for s in pilot["songs"]}
    comp = {r[0]: {"title": r[1], "template": r[2] or ""} for r in c.execute("select compid, title, template from composition_info")}
    solos = c.execute("select melid, compid, instrument, signature, performer, title from solo_info").fetchall()
    elig = [r for r in solos if comp[r[1]]["template"] not in ("Blues", "I Got Rhythm") and r[3] == "4/4" and r[2] != "p"]
    by_comp = collections.defaultdict(list)
    for r in elig:
        by_comp[r[1]].append(r)
    pilot_names = {norm(comp[k]["title"]) for k in pilot_comp} | {norm(comp[k]["template"]) for k in pilot_comp if comp[k]["template"]}
    excluded = collections.Counter()
    cands = []
    for k in sorted(by_comp):
        if k in pilot_comp:
            excluded["used_in_pilot"] += 1
        elif norm(comp[k]["title"]) in pilot_names or (comp[k]["template"] and norm(comp[k]["template"]) in pilot_names):
            excluded["title_or_template_shared_with_pilot"] += 1
        else:
            cands.append(k)
    rng = random.Random(SEED)
    rng.shuffle(cands)
    taken, skipped, seen_names = [], [], set()
    chunks = []
    for k in cands:
        if len(taken) == TAKE:
            break
        melid = rng.choice(sorted(r[0] for r in by_comp[k]))
        names = {norm(comp[k]["title"])} | ({norm(comp[k]["template"])} if comp[k]["template"] else set())
        if names & seen_names:
            skipped.append({"compid": k, "melid": melid, "reason": "title_or_template_shared_inside_sealed_set"})
            continue
        random.seed(SEED + melid)
        notes, valid, song_chunks = song_chunks_for(c, tok, MidiDict, melid, collections.Counter(), collections.Counter(), collections.Counter())
        if not song_chunks:
            skipped.append({"compid": k, "melid": melid, "reason": "no_valid_chunk"})
            continue
        seen_names |= names
        r = next(x for x in by_comp[k] if x[0] == melid)
        taken.append({"melid": melid, "compid": k, "title": comp[k]["title"], "instrument": r[2], "performer": r[4],
                      "chunks": len(song_chunks), "pitch_targets": sum(len(ch["pitch_targets"]) for ch in song_chunks)})
        for ch in song_chunks:
            ch["split"] = "sealed_heldout"
        chunks += song_chunks
    with open(local_out, "w") as f:
        json.dump({"songs": taken, "chunks": chunks}, f)
    sel = json.dumps(sorted([t["melid"], t["compid"]] for t in taken))
    manifest = {"purpose": "sealed held-out for one final evaluation after a development pass; never for training, development or tuning",
                "seed": SEED, "take": TAKE, "eligible_compositions": len(by_comp), "excluded": dict(excluded),
                "candidates_after_exclusion": len(cands), "taken": len(taken), "selected": taken, "skipped_in_order": skipped,
                "selection_sha256": hashlib.sha256(sel.encode()).hexdigest(),
                "sealed_data_sha256": sha(local_out), "pilot_data_sha256": sha(pilot_data),
                "builder_sha256": sha(os.path.join(HERE, "wjazzd_pilot_data.py")), "script_sha256": sha(os.path.abspath(__file__)),
                "not_in_scope": "Blues / I Got Rhythm compositions (75) stay an unused separate candidate layer; the 6 piano solos stay outside"}
    with open(manifest_out, "w") as f:
        json.dump(manifest, f, indent=1, ensure_ascii=False)
    print(json.dumps({k: manifest[k] for k in ("eligible_compositions", "excluded", "candidates_after_exclusion", "taken")}, ensure_ascii=False),
          "skipped", len(skipped), collections.Counter(s["reason"] for s in skipped))


if __name__ == "__main__":
    main()
