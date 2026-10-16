#!/usr/bin/env python3
"""Fixed song split for the minimal condition-adapter run (issue #1760, docs/experiments/COND_ADAPTER_MIN.md, step B).

Metadata only, decided before any training output: the manifest's train songs sorted by sha1; a song is skipped when its
normalized title (file name before " - " or "(", lower case) is already taken; the first 8 picks train the adapter, the
next 4 are the development probe. val/test are never opened. Song indexes are the manifest's train order (= #1756 export).

  cond_adapter_split.py <bebop_rh manifest> <out json>
"""
from __future__ import annotations

import json
import os
import re
import sys

N_TRAIN, N_DEV = 8, 4


def title(source):
    return re.sub(r"\s*[-–(].*$", "", os.path.splitext(os.path.basename(source))[0]).strip().lower()


def pick(train_songs, n_train=N_TRAIN, n_dev=N_DEV):
    order = sorted(range(len(train_songs)), key=lambda i: train_songs[i]["sha1"])
    taken, chosen = set(), []
    for i in order:
        t = title(train_songs[i]["source"])
        if t in taken:
            continue
        taken.add(t)
        chosen.append(i)
        if len(chosen) == n_train + n_dev:
            break
    return chosen[:n_train], chosen[n_train:]


def main() -> None:
    manifest_path, out_path = sys.argv[1:3]
    songs = [s for s in json.load(open(manifest_path))["songs"] if s["split"] == "train"]
    assert len(songs) == 189
    tr, dev = pick(songs)
    rec = lambda i: {"idx": i, "title": title(songs[i]["source"]), "pianist": songs[i]["pianist"], "sha1": songs[i]["sha1"]}
    out = {"issue": 1760, "rule": "train songs by sha1, skip taken titles, first 8 train / next 4 dev",
           "train": [rec(i) for i in tr], "dev": [rec(i) for i in dev]}
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
