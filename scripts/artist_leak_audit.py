#!/usr/bin/env python3
"""Which Mehldau / Tatum recordings were actually inside each training set? (issue #1722)

Every studio and live recording of both artists in the PiJAMA trees is encoded again with the
encoder that built data/jazz_full (archive/scripts/preprocess_jazz.py:process_midi_file) and its
token hash is looked up in the real training input folders. No manifest-based guess: a recording
counts as inside a set only when an identical token sequence is in that folder.

First check: the studio recordings must hash-match the existing artist sets (mehldau_full 18,
tatum_full 122). A miss there means the encoder did not reproduce and the audit is not trusted.

  artist_leak_audit.py <report json>
"""
from __future__ import annotations

import collections
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "music_transformer", ROOT / "music_transformer" / "third_party", ROOT / "archive" / "scripts"):
    sys.path.insert(0, str(p))

import numpy as np  # noqa: E402

MAIN = Path("/Users/ohhalim/git_box/fine_tuning_art-tatum_midi_solo")
PIJAMA = MAIN / "midi_dataset"
ARTISTS = ["Brad Mehldau", "Art Tatum"]
SETS = {
    "jazz_full": MAIN / "data/jazz_full",                                  # original base (armB)
    "jazz_full_nomehldau": ROOT / "data/jazz_full_nomehldau",              # clean base C0 (#1497)
    "jazz_full_notatum_nomehldau": ROOT / "data/jazz_full_notatum_nomehldau",
    "mehldau_full": MAIN / "data/mehldau_full",                            # Mehldau adapter train/val
    "tatum_full": ROOT / "data/tatum_full",                                # Tatum adapter train/val
}


def seq_hash(tokens) -> str:
    return hashlib.sha1(np.asarray(tokens, dtype=np.int64).ravel().tobytes()).hexdigest()


def norm_title(name: str) -> str:
    s = Path(name).stem.lower()
    s = re.sub(r"\(.*?\)|\[.*?\]", "", s)
    s = re.sub(r"\blive\b", "", s)
    s = re.sub(r"[^a-z0-9 ]", "", s)
    return re.sub(r"\s+", " ", s).strip()


def index_sets():
    idx, sizes = collections.defaultdict(list), {}
    for name, d in SETS.items():
        n = 0
        for split in ("train", "val"):
            for f in sorted((d / split).glob("*.npy")):
                idx[seq_hash(np.load(f))].append(f"{name}/{split}/{f.name}")
                n += 1
        sizes[name] = n
    return idx, sizes


def main() -> None:
    report_path = Path(sys.argv[1])
    from preprocess_jazz import process_midi_file
    idx, sizes = index_sets()
    folds = json.loads((ROOT / "data/mehldau_cv/folds.json").read_text())["held_out"]
    fold_of = {f"mehldau_full/train/{n}": int(k) for k, names in folds.items() for n in names}
    rows = []
    for tree in ("midi", "midi_kong"):
        for kind in ("studio", "live"):
            for artist in ARTISTS:
                adir = PIJAMA / tree / kind / artist
                if not adir.exists():
                    continue
                for f in sorted(p for ext in (".mid", ".midi") for p in adir.rglob(f"*{ext}")):
                    tok = process_midi_file(f)
                    rel = f.relative_to(adir)
                    row = {"artist": artist, "tree": tree, "kind": kind, "album": str(rel.parent), "file": rel.name,
                           "title": norm_title(rel.name)}
                    if tok is None:
                        row["encoded"] = False
                    else:
                        h = seq_hash(tok)
                        hits = idx.get(h, [])
                        row.update({"encoded": True, "tokens": int(len(tok)), "sha1": h, "in": hits,
                                    "mehldau_cv_held_out_fold": [fold_of[x] for x in hits if x in fold_of]})
                    rows.append(row)
                    print(tree, kind, artist, rel.name, row.get("in"), flush=True)

    def count(pred):
        return sum(1 for r in rows if pred(r))

    summary = []
    for tree in ("midi", "midi_kong"):
        for artist in ARTISTS:
            for kind in ("studio", "live"):
                sel = [r for r in rows if r["tree"] == tree and r["artist"] == artist and r["kind"] == kind]
                if not sel:
                    continue
                line = {"tree": tree, "artist": artist, "kind": kind, "recordings": len(sel),
                        "not_encoded": sum(1 for r in sel if not r["encoded"])}
                for name in SETS:
                    for split in ("train", "val"):
                        line[f"{name}/{split}"] = sum(1 for r in sel if any(x.startswith(f"{name}/{split}/") for x in r.get("in", [])))
                line["in_no_set"] = sum(1 for r in sel if r["encoded"] and not r["in"])
                summary.append(line)

    # same title recorded more than once (studio vs live, or two takes), main tree only
    by_title = collections.defaultdict(list)
    for r in rows:
        if r["tree"] == "midi":
            by_title[(r["artist"], r["title"])].append(f"{r['kind']}:{r['album']}/{r['file']}")
    repeated = {f"{a} | {t}": v for (a, t), v in by_title.items() if len(v) > 1}

    repro = {
        "mehldau_studio_in_mehldau_full": count(lambda r: r["tree"] == "midi" and r["artist"] == "Brad Mehldau" and r["kind"] == "studio"
                                                 and any(x.startswith("mehldau_full/") for x in r.get("in", []))),
        "mehldau_full_size": sizes["mehldau_full"],
        "tatum_studio_in_tatum_full": count(lambda r: r["tree"] == "midi" and r["artist"] == "Art Tatum"
                                            and any(x.startswith("tatum_full/") for x in r.get("in", []))),
        "tatum_full_size": sizes["tatum_full"],
    }
    report = {"encoder": "archive/scripts/preprocess_jazz.py:process_midi_file", "hash": "sha1 of int64 token array",
              "set_sizes": sizes, "set_paths": {k: str(v) for k, v in SETS.items()}, "reproduction_check": repro,
              "summary": summary, "same_title_recordings": repeated, "recordings": rows}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps(repro))
    for line in summary:
        print(json.dumps(line, ensure_ascii=False))


if __name__ == "__main__":
    main()
