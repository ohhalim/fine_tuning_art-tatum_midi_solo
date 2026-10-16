#!/usr/bin/env python3
"""Canonical primers for the Aria base vs Jazz Pianist Style comparison (issue #1774, docs/experiments/JPS_COMPARE.md).

First 8 s of reference clips 06 and 18: AbsTokenizer with remove_preceding_silence=False, add_dim_tok=False,
add_eos_tok=False (the library defaults move the time origin and insert <D>), cut before the first note whose
absolute onset is >= 8,000 ms. Every arm receives exactly these token ids. Aria venv.

  jps_compare_primers.py <reference_candidates dir> <out json>
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

CLIPS = {"06": "Tigran Hamasyan — Looking For The Block", "18": "Barry Harris — All God's Chillun Got Rhythm"}
PRIMER_MS = 8000
TOKENIZE_KW = {"remove_preceding_silence": False, "add_dim_tok": False, "add_eos_tok": False}


def cut_at(tokens, end_ms):
    """Tokens before the first note whose absolute onset is >= end_ms. A note is (piano, onset, dur)."""
    seg = 0
    for i, t in enumerate(tokens):
        if t == "<T>":
            seg += 1
        elif isinstance(t, tuple) and t[0] == "onset" and seg * 5000 + t[1] >= end_ms:
            assert isinstance(tokens[i - 1], tuple) and tokens[i - 1][0] == "piano", tokens[i - 1]
            return tokens[: i - 1]
    return tokens


def main() -> None:
    clip_dir, out_path = sys.argv[1:3]
    from ariautils.midi import MidiDict
    from ariautils.tokenizer import AbsTokenizer
    tok = AbsTokenizer()
    out = {"issue": 1774, "primer_ms": PRIMER_MS, "tokenize": TOKENIZE_KW, "vocab_size": tok.vocab_size, "primers": {}}
    for cid, title in CLIPS.items():
        path = os.path.join(clip_dir, f"{cid}.mid")
        toks = cut_at(tok.tokenize(MidiDict.from_midi(path), **TOKENIZE_KW), PRIMER_MS)
        while toks and toks[-1] == "<T>":          # a trailing segment marker carries no note
            toks = toks[:-1]
        ids = tok.encode(toks)
        assert tok.decode(ids) == toks
        lib_default = tok.tokenize(MidiDict.from_midi(path))
        out["primers"][cid] = {
            "title": title, "midi": path, "midi_sha256": hashlib.sha256(open(path, "rb").read()).hexdigest(),
            "tokens": [list(t) if isinstance(t, tuple) else t for t in toks], "ids": ids, "n_tokens": len(ids),
            "notes": sum(isinstance(t, tuple) and t[0] == "piano" for t in toks), "segments": toks.count("<T>"),
            "ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
            "library_default_differs": lib_default[: len(toks)] != toks, "library_default_has_D": "<D>" in lib_default}
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    for cid, p in out["primers"].items():
        print(cid, p["n_tokens"], "tokens", p["notes"], "notes", p["segments"], "<T>", p["ids_sha256"][:16],
              "default differs", p["library_default_differs"], "default <D>", p["library_default_has_D"])


if __name__ == "__main__":
    main()
