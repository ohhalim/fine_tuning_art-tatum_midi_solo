"""Examples for the minimal condition-adapter run (issue #1760, docs/experiments/COND_ADAPTER_MIN.md). Pure Python.

One example is one block request in session coordinates (crop_contract, production settings): the session tokens up
to the end of the window, one 13-dim condition row per input position, and the window labels. Boundary tokens are not
labels (#1756 review Q2).
"""
from __future__ import annotations

import gzip
import json
import os
import random

from aria_mlx_cond import cond_rows, cond_vector
from block_contract import block_start
from block_request import payload
from crop_contract import CAP, HEADER, session_position_requests, session_requests
from lh_role_proxy import Note

SEED = 1760


def load_song(export_dir, idx):
    d = json.load(gzip.open(os.path.join(export_dir, "requests", f"{idx:03d}.json.gz")))
    nj = json.load(open(os.path.join(export_dir, "songs", f"{idx:03d}.notes.json")))
    assert (nj["idx"], nj["source"], nj["sha1"]) == (d["idx"], d["source"], d["sha1"]) == (idx, d["source"], d["sha1"])
    tokens = [tuple(t) if isinstance(t, list) else t for t in d["tokens"]]
    return tokens, d["token_ids"], [Note(nid, p, on, off) for nid, p, on, off in nj["human"]], d["t0"]


def song_examples(tokens, ids, humans, t0, song=None):
    """(eligible examples, counts). Requests with an empty window or window + boundary over CAP are left out."""
    srows = session_requests(tokens, t0)
    pays = {r["k"]: payload(humans, block_start(r["k"], t0), r["k"]) for r in srows}
    out, counts = [], {"requests": len(srows), "empty_window": 0, "over_cap": 0}
    for r in srows:
        win = r["win_end"] - r["ctx_end"]
        if win == 0:
            counts["empty_window"] += 1
            continue
        if win + r["cont_end"] - r["win_end"] > CAP:
            counts["over_cap"] += 1
            continue
        if r["keep_from"] is None:
            sess_ids, sess_toks = list(ids[:r["win_end"]]), list(tokens[:r["win_end"]])
        else:
            sess_ids = list(ids[:len(HEADER)]) + list(ids[r["keep_from"]:r["win_end"]])
            sess_toks = list(HEADER) + list(tokens[r["keep_from"]:r["win_end"]])
        tags = session_position_requests(tokens, srows, r["k"])[:len(sess_ids) - 1]
        out.append({"song": song, "k": r["k"], "ids": sess_ids, "rows": cond_rows(tags, pays), "first_label": r["ctx_len"],
                    "label_kinds": [token_kind(t) for t in sess_toks[r["ctx_len"]:]], "payload": pays[r["k"]],
                    "window_pcs": [t[1] % 12 for t in sess_toks[r["ctx_len"]:] if isinstance(t, tuple) and t[0] == "piano"]})
    return out, counts


def token_kind(t):
    if t == "<T>":
        return "T"
    return {"piano": "pitch", "onset": "onset", "dur": "dur"}.get(t[0] if isinstance(t, tuple) else t, "other")


def wrong_payload(pay):
    """A different PC set with the same size and the same `available`: the PCs moved by a tritone, or by one semitone
    when the set is tritone-symmetric. None for cold or warm-empty (no different set of the same size exists).
    Decided from the payload alone, never from labels or results."""
    pcs, available = pay
    on = [c for c in range(12) if pcs[c]]
    if not available or not on:
        return None
    for shift in (6, 1):
        moved = {(c + shift) % 12 for c in on}
        if moved != set(on):
            return tuple(int(c in moved) for c in range(12)), available
    return None


def with_current_condition(ex, pay):
    """The example's rows with every prediction position (p0 = first_label - 1 on) set to `pay`; context rows kept."""
    p0 = ex["first_label"] - 1
    return ex["rows"][:p0] + [cond_vector(pay)] * (len(ex["rows"]) - p0)


def pick_training(examples, n, seed=SEED):
    order = list(range(len(examples)))
    random.Random(seed).shuffle(order)
    return [examples[i] for i in order[:n]]


def pick_evenly(examples, n):
    if len(examples) <= n:
        return list(examples)
    return [examples[round(i * (len(examples) - 1) / (n - 1))] for i in range(n)]
