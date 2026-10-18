#!/usr/bin/env python3
"""One real lick from reference clip 18, two-hand original vs top-line proxy (issue #1786, docs/experiments/CLIP18_LICK.md).

Not a model output. The first phrase starting inside the clip window that passes lick_bank.lick_from_run is taken
as is (no transposition, no time change). Writes the two-hand original and the top-line proxy from original note
IDs (original start/end/velocity), records the lick's implied tempo and the left-hand context (v1 labels, unknown
kept), and checks the MIDI round trip.

  clip18_lick.py <source midi> <run dir> <report json>
"""
from __future__ import annotations

import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.join(HERE, "..")):
    sys.path.insert(0, p)
import bebop_fragment_retrieval as bf  # noqa: E402

CLIP = (69.5, 99.5)                       # clips.json id 18: start_s, end_s


def pick_lick(notes, clip=CLIP):
    """(run as [(id, truncated note)], line, trunc, acc) for the first lick phrase starting inside the clip."""
    from inference.control.lick_bank import lick_from_run, runs
    line, trunc, acc, _ = bf.split_song(notes)
    id_of = {id(t): nid for nid, t in line}
    for run in runs([t for _, t in line]):
        if clip[0] <= run[0].start < clip[1] and lick_from_run(run) is not None:
            return [(id_of[id(t)], t) for t in run], line, trunc, acc
    return None, line, trunc, acc


def build(notes, run, trunc, acc):
    nb = bf.notes_by_id(notes)
    t0, t_last = run[0][1].start, run[-1][1].start
    ids = [nid for nid, _ in run]
    idset = set(ids)
    rel = lambda x: round(float(x) - t0, 4)  # noqa: E731
    melody = [{"id": nid, "pitch": nb[nid].pitch, "start": rel(nb[nid].start), "end": rel(nb[nid].end), "velocity": nb[nid].velocity,
               "trunc_end": rel(trunc[nid])} for nid in ids]
    span = [(nid, n) for nid, n in notes if t0 <= n.start <= t_last]
    held = [(nid, n) for nid, n in notes if n.start < t0 < n.end]
    twohand = [{"id": nid, "pitch": n.pitch, "start": rel(n.start), "end": rel(n.end), "velocity": n.velocity, "cut_start": False}
               for nid, n in span] + \
              [{"id": nid, "pitch": n.pitch, "start": 0.0, "end": rel(n.end), "velocity": n.velocity, "cut_start": True,
                "orig_start": rel(n.start)} for nid, n in held]
    excluded = [{"id": nid, "pitch": n.pitch, "start": rel(n.start)} for nid, n in span if nid not in idset]
    iois = [b[1].start - a[1].start for a, b in zip(run, run[1:])]
    med = statistics.median(iois)
    lab = bf.Labeler(acc)
    src, k, first = lab.phrase_label(t0)
    t_chk = t0 + bf.SLICE_OFFSET_S
    sounding = sorted({n.pitch % 12 for _, n in acc if n.start <= t_chk < n.end})
    clusters = []
    for kk, (on, _) in enumerate(lab.groups):
        if t0 <= on <= t_last:
            lb = lab.label_at_cluster(kk)
            clusters.append({"onset": rel(on), "cls": lb["cls"], "label": lb["label"], "pcs": lb["pcs"], "bass": lb["bass"]})
    poly = sum(1 for a, b in zip(melody, melody[1:]) if b["start"] < a["end"])
    context = {"t0_s": round(float(t0), 4), "lick_notes": len(run), "median_ioi_s": round(med, 4),
               "implied_tempo_bpm_if_ioi_is_eighth": round(60 / (2 * med), 1), "span_s": rel(t_last),
               "t0_slice_source": src, "t0_v1": None if first is None else {k2: first[k2] for k2 in ("cls", "label", "pcs", "bass")},
               "t0_sounding_pcs": sounding, "t0_v1_matches_sounding": None if first is None else first["pcs"] == sounding,
               "lh_clusters_in_span": clusters}
    return {"melody": melody, "twohand": twohand, "excluded": excluded, "melody_overlaps": poly, "context": context}


def main() -> None:
    src, run_dir, report_path = sys.argv[1:4]
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    notes = bf.load_notes(src)
    run, line, trunc, acc = pick_lick(notes)
    report = {"issue": 1786, "source": src, "source_sha256": bf.file_hash(src), "clip": CLIP}
    if run is None:
        report["result"] = "no_lick"
    else:
        out = build(notes, run, trunc, acc)
        files = [bf.write_midi(out["twohand"], os.path.join(run_dir, "A_twohand.mid")),
                 bf.write_midi(out["melody"], os.path.join(run_dir, "B_melody.mid"))]
        for f in files:
            bf.render(os.path.join(run_dir, f["file"]))
        with open(os.path.join(run_dir, "sidecar.json"), "w") as f:
            json.dump(out | {"files": files}, f, ensure_ascii=False, indent=1, default=str)
        report.update(result="done", context=out["context"], melody_notes=len(out["melody"]), twohand_notes=len(out["twohand"]),
                      cut_start_notes=sum(1 for r in out["twohand"] if r["cut_start"]), excluded_notes=len(out["excluded"]),
                      melody_overlaps=out["melody_overlaps"], files=files)
    with open(report_path, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=1, default=str)
    print(json.dumps({k: report.get(k) for k in ("result", "melody_notes", "twohand_notes")}))


if __name__ == "__main__":
    main()
