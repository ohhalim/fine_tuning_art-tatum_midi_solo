#!/usr/bin/env python3
"""Real bebop piano fragments for two fixed 3.75 s targets (issue #1782, docs/experiments/BEBOP_FRAGMENT_RETRIEVAL.md).

Not a model output. From the train songs of data/bebop_rh (13 bebop pianists, original MIDI re-read), propose
candidates with the LABEL_AUDIT v1 vertical-slice chord labels, take the deterministic top1 per target, check the
chosen ones against the notes actually sounding at t0, and write the fragment back from original note IDs
(original start/end/velocity). No time stretch, no pitch clamp, no reselection after a failed check.

  bebop_fragment_retrieval.py <bebop_rh manifest> <run dir> <report dir>
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    sys.path.insert(0, str(p))

WIN_S = 3.75                                   # onset selection window = 2 bars at 128 BPM (not source bars)
TARGET_TC_S, TC_TOL_S = 1.875, 0.469           # change target: chord change after 1.875 s, search tolerance 1 beat
TARGETS = {"change": {"first": (7, "m7"), "second_rel": (5, "7")},      # Gm7 -> C7 (second root = first + 5)
           "hold": {"first": (5, "maj7")}}                              # Fmaj7 held
OK_CLS = ("clear", "bass_resolved")
SLICE_OFFSET_S = 0.03
ACC_HIGH, SOLO_SPLIT, MIN_RUN = 64, 55, 8
SF2 = os.path.expanduser("~/.local/share/soundfonts/generaluser-gs/v1.471.sf2")


def signed_k(r, target_root):
    """Signed smallest shift from root r to target_root, in -6..+5 (a tie at 6 goes down)."""
    d = (target_root - r) % 12
    return d - 12 if d >= 6 else d


def load_notes(path):
    """[(id, note)] sorted by (start, pitch); id = i{instrument}_n{index in that instrument}."""
    import pretty_midi
    pm = pretty_midi.PrettyMIDI(path)
    out = [(f"i{i}_n{j}", n) for i, inst in enumerate(pm.instruments) for j, n in enumerate(inst.notes)]
    return sorted(out, key=lambda x: (x[1].start, x[1].pitch, x[0]))


def split_song(notes):
    """Top-line proxy (ids picked by top_notes, at or above G3), its truncated copies, and the accompaniment."""
    from inference.control.solo_line import top_notes
    by_key = {}
    for nid, n in notes:
        by_key.setdefault((n.start, n.pitch), []).append(nid)
    line, trunc, used, dup = [], {}, set(), 0
    for t in top_notes([n for _, n in notes]):
        if t.pitch < SOLO_SPLIT:
            continue
        ids = [i for i in by_key[(t.start, t.pitch)] if i not in used]
        dup += len(by_key[(t.start, t.pitch)]) > 1
        nid = ids[0]
        used.add(nid)
        line.append((nid, t))
        trunc[nid] = t.end
    acc = [(nid, n) for nid, n in notes if nid not in used and n.pitch < ACC_HIGH]
    return line, trunc, acc, dup


class Labeler:
    """LABEL_AUDIT v1 machinery over one song's accompaniment."""

    def __init__(self, acc):
        import label_audit as la
        from inference.control.chord_label import classify
        self.la, self.classify = la, classify
        self.acc = [n for _, n in acc]
        self.groups = la.clusters(self.acc)
        self.onsets = [g[0] for g in self.groups]
        self.sounding = la.Sounding(self.acc)

    def label_at_cluster(self, k):
        pcs, bass, _ = self.la.slice_at(self.sounding, self.groups[k][0])
        cls, label, cands = self.classify(pcs, bass)
        return {"cls": cls, "label": label, "cands": cands, "pcs": sorted(pcs), "bass": bass, "onset": self.groups[k][0]}

    def phrase_label(self, t0):
        src, k = self.la.phrase_slice(self.sounding, self.onsets, t0)
        return src, k, (None if k is None else self.label_at_cluster(k))

    def scan(self, k_from, label, t_end):
        """Clusters after index k_from with onset < t_end: first contradiction of label (v1 span rule) and unknown counts."""
        n, unknown, unknown_s, contra = 0, 0, 0.0, None
        for k in range(k_from + 1, len(self.groups)):
            on = self.groups[k][0]
            if on >= t_end:
                break
            n += 1
            lab = self.label_at_cluster(k)
            if (lab["label"] is not None and lab["label"] != label) or (lab["cls"] == "ambiguous" and label not in lab["cands"]):
                contra = (k, lab)
                break
            if lab["cls"] == "unknown":
                unknown += 1
                nxt = self.groups[k + 1][0] if k + 1 < len(self.groups) else t_end
                unknown_s += min(nxt, t_end) - on
        return {"slices": n, "unknown": unknown, "unknown_s": round(unknown_s, 3), "contra": contra}


def song_candidates(notes, meta, require_lick=False):
    """Candidates for both targets from one song (each a dict with a sort key). require_lick: R2, the run must be a lick."""
    import statistics
    from inference.control.lick_bank import lick_from_run, runs
    line, trunc, acc, dup = split_song(notes)
    lab = Labeler(acc)
    nb = notes_by_id(notes)
    out = []
    for run in runs([t for _, t in line]):
        if len(run) < MIN_RUN:
            continue
        if require_lick and lick_from_run(run) is None:
            continue
        t0 = run[0].start
        run_info = {"notes": len(run), "median_ioi_s": round(statistics.median(b.start - a.start for a, b in zip(run, run[1:])), 4),
                    "end_s": round(run[-1].end - t0, 4)}
        src, k, first = lab.phrase_label(t0)
        if first is None or first["cls"] not in OK_CLS:
            continue
        r, q = first["label"]
        for name, tg in TARGETS.items():
            if q != tg["first"][1]:
                continue
            kshift = signed_k(r, tg["first"][0])
            cand = {"target": name, "meta": meta, "run": run_info, "run_starts": [float(n.start) for n in run],
                    "t0": round(t0, 4), "t0_exact": float(t0), "slice_source": src, "k": kshift, "first": first,
                    "cluster_index": k, "dup_keys": dup}
            if name == "hold":
                s = lab.scan(k, first["label"], t0 + WIN_S)
                if s["contra"] is not None:
                    continue
                cand.update(scan=[s], n_clear=int(first["cls"] == "clear"), tc_err=0.0)
            else:
                s1 = lab.scan(k, first["label"], t0 + WIN_S)
                if s1["contra"] is None:
                    continue
                kc, second = s1["contra"]
                want = ((r + tg["second_rel"][0]) % 12, tg["second_rel"][1])
                err = second["onset"] - (t0 + TARGET_TC_S)
                if second["label"] != want or second["cls"] not in OK_CLS or abs(err) > TC_TOL_S:
                    continue
                s2 = lab.scan(kc, second["label"], t0 + WIN_S)
                if s2["contra"] is not None:
                    continue
                cand.update(second=second, tc=round(second["onset"], 4), tc_exact=float(second["onset"]), tc_target=round(t0 + TARGET_TC_S, 4), tc_err=round(err, 4),
                            scan=[s1 | {"contra": None}, s2], n_clear=int(first["cls"] == "clear") + int(second["cls"] == "clear"))
            sel = [(nid, t) for nid, t in line if t0 <= t.start < t0 + WIN_S]
            pitches = [nb[nid].pitch + kshift for nid, _ in sel]
            if not pitches or min(pitches) < 0 or max(pitches) > 127:
                continue
            cand["key"] = (-cand["n_clear"], abs(kshift), abs(cand["tc_err"]), meta["sha1"], t0)
            out.append(cand)
    return out


def notes_by_id(notes):
    return {nid: n for nid, n in notes}


def verify_t0(notes, cand):
    """Accompaniment pitch classes actually sounding (start <= t < end) at t0 + 30 ms (and tc + 30 ms for change),
    compared with the v1 slices. Exact times are used; rounded values are for display only. ok needs every check."""
    _, _, acc, _ = split_song(notes)

    def sounding(t):
        here = [(nid, n) for nid, n in acc if n.start <= t < n.end]
        return here, sorted({n.pitch % 12 for _, n in here})

    rows = lambda here: [{"id": nid, "pitch": n.pitch, "start": round(n.start, 4), "end": round(n.end, 4)} for nid, n in here]  # noqa: E731
    t = cand["t0_exact"] + SLICE_OFFSET_S
    here, pcs = sounding(t)
    out = {"t0_check_time": round(t, 4), "t0_sounding": rows(here), "t0_sounding_pcs": pcs, "v1_slice_pcs": cand["first"]["pcs"],
           "v1_slice_onset": round(cand["first"]["onset"], 4), "t0_match": pcs == cand["first"]["pcs"]}
    if cand["target"] == "change":
        t2 = cand["tc_exact"] + SLICE_OFFSET_S
        here2, pcs2 = sounding(t2)
        out.update(tc_check_time=round(t2, 4), tc_sounding=rows(here2), tc_sounding_pcs=pcs2, v1_second_pcs=cand["second"]["pcs"],
                   tc_match=pcs2 == cand["second"]["pcs"])
    out["status"] = "ok" if out["t0_match"] and out.get("tc_match", True) else "proxy_invalid"
    return out


def extract(notes, cand):
    """Selected top-line notes restored from original IDs, excluded notes, the two-hand reference."""
    line, trunc, acc, _ = split_song(notes)
    t0 = cand["t0_exact"]                       # exact: a rounded t0 can push the phrase start out of the window
    t1 = t0 + WIN_S
    sel_ids = [nid for nid, t in line if t0 <= t.start < t1]
    nb = notes_by_id(notes)
    sel = [(nid, nb[nid]) for nid in sel_ids]
    selset = set(sel_ids)
    excluded = [(nid, n) for nid, n in notes if t0 <= n.start < t1 and nid not in selset]
    ref = [(nid, n, n.start) for nid, n in notes if t0 <= n.start < t1] + \
          [(nid, n, n.start) for nid, n in notes if n.start < t0 < n.end]
    run_starts = set(cand.get("run_starts", []))
    in_run = sum(1 for _, n in sel if float(n.start) in run_starts)
    poly = sum(1 for (_, a), (_, b) in zip(sel, sel[1:]) if b.start < a.end)
    same = 0
    for i, (_, a) in enumerate(sel):
        same += sum(1 for _, b in sel[i + 1:] if b.pitch == a.pitch and b.start < a.end)
    rows = lambda xs: [{"id": nid, "pitch": n.pitch, "start": round(n.start - t0, 4), "end": round(n.end - t0, 4), "velocity": n.velocity} for nid, n in xs]  # noqa: E731
    return {"selected": [r | {"trunc_end": round(trunc[r["id"]] - t0, 4)} for r in rows(sel)],
            "excluded": rows(excluded),
            "twohand_ref": [{"id": nid, "pitch": n.pitch, "start": round(max(n.start, t0) - t0, 4), "end": round(n.end - t0, 4),
                             "velocity": n.velocity, "cut_start": bool(s < t0), "orig_start": round(s - t0, 4)} for nid, n, s in ref],
            "line_overlaps": poly, "same_pitch_overlaps": same, "selected_in_run": in_run,
            "file_len_s": round(max(n.end for _, n in sel) - t0, 4) if sel else 0.0}


def write_midi(rows, path, shift=0):
    import pretty_midi
    pm = pretty_midi.PrettyMIDI(initial_tempo=120)
    inst = pretty_midi.Instrument(program=0, name="fragment")
    inst.notes = [pretty_midi.Note(velocity=r["velocity"], pitch=r["pitch"] + shift, start=r["start"], end=r["end"]) for r in rows]
    pm.instruments.append(inst)
    pm.write(path)
    back = sorted((n.start, n.end, n.pitch, n.velocity) for n in pretty_midi.PrettyMIDI(path).instruments[0].notes)
    want = sorted((r["start"], r["end"], r["pitch"] + shift, r["velocity"]) for r in rows)
    errs = [max(abs(a[0] - b[0]), abs(a[1] - b[1])) for a, b in zip(want, back)]
    bad = sum(1 for a, b, e in zip(want, back, errs) if a[2:] != b[2:] or e > 0.001) + abs(len(want) - len(back))
    return {"file": os.path.basename(path), "notes": len(rows), "read_back": len(back), "roundtrip_diffs": bad,
            "max_time_err_ms": round(max(errs, default=0.0) * 1000, 3)}


def render(mid):
    wav, m4a = mid[:-4] + ".wav", mid[:-4] + ".m4a"
    subprocess.run(["fluidsynth", "-ni", "-q", "-F", wav, "-r", "44100", SF2, mid], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["afconvert", "-f", "m4af", "-d", "aac", wav, m4a], check=True)
    os.remove(wav)


def file_hash(path, algo="sha256"):
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> None:
    manifest_path, run_dir, report_dir = sys.argv[1:4]
    lick = "--lick" in sys.argv[4:]
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)
    man = json.load(open(manifest_path))
    songs = [s for s in man["songs"] if s["split"] == "train"]
    report = {"issue": 1782, "round": "R2 lick" if lick else "R1", "manifest_sha256": file_hash(manifest_path), "songs": len(songs), "skipped": [], "candidates": {}, "selected": {}}
    cands, loaded = [], {}
    for s in songs:
        if not os.path.exists(s["source"]) or file_hash(s["source"], "sha1") != s["sha1"]:
            report["skipped"].append({"source": s["source"], "reason": "missing or sha1 mismatch"})
            continue
        notes = load_notes(s["source"])
        meta = {"pianist": s["pianist"], "source": s["source"], "sha1": s["sha1"]}
        got = song_candidates(notes, meta, require_lick=lick)
        if got:
            loaded[s["sha1"]] = notes
        cands += got
    for name in TARGETS:
        mine = sorted((c for c in cands if c["target"] == name), key=lambda c: c["key"])
        report["candidates"][name] = {"n": len(mine), "songs": len({c["meta"]["sha1"] for c in mine}),
                                      "pianists": len({c["meta"]["pianist"] for c in mine})}
        if not mine:
            report["selected"][name] = {"status": "no_match"}
            continue
        c = mine[0]
        notes = loaded[c["meta"]["sha1"]]
        chk = verify_t0(notes, c)
        ex = extract(notes, c)
        sel = {k: v for k, v in c.items() if k not in ("key", "scan", "run_starts")}
        sel["scan"] = [{k: v for k, v in s.items() if k != "contra"} for s in c["scan"]]
        sel.update(check=chk, extract=ex, status=chk["status"], window_s=WIN_S)
        files = []
        if chk["status"] == "ok":
            base = os.path.join(run_dir, name)
            files.append(write_midi(ex["selected"], base + "_melody_orig.mid"))
            if c["k"] != 0:
                files.append(write_midi(ex["selected"], base + f"_melody_k{c['k']:+d}.mid", shift=c["k"]))
            files.append(write_midi(ex["twohand_ref"], base + "_twohand_ref.mid"))
            for f in files:
                render(os.path.join(run_dir, f["file"]))
        sel["files"] = files
        with open(os.path.join(run_dir, name + "_sidecar.json"), "w") as f:
            json.dump(sel, f, ensure_ascii=False, indent=1, default=str)
        report["selected"][name] = {k: sel[k] for k in ("status", "meta", "t0", "k", "slice_source", "tc_err", "n_clear")} | {
            "first": sel["first"], "second": sel.get("second"), "tc": sel.get("tc"), "tc_target": sel.get("tc_target"),
            "scan": sel["scan"], "check_status": chk["status"], "t0_match": chk["t0_match"], "tc_match": chk.get("tc_match"),
            "t0_sounding_pcs": chk["t0_sounding_pcs"], "v1_slice_pcs": chk["v1_slice_pcs"], "tc_sounding_pcs": chk.get("tc_sounding_pcs"),
            "selected_notes": len(ex["selected"]), "excluded_notes": len(ex["excluded"]), "line_overlaps": ex["line_overlaps"],
            "same_pitch_overlaps": ex["same_pitch_overlaps"], "file_len_s": ex["file_len_s"], "files": files,
            "run": c["run"], "selected_in_run": ex["selected_in_run"]}
    report["result"] = "done"
    with open(os.path.join(report_dir, "report_r2.json" if lick else "report.json"), "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=1, default=str)
    print(json.dumps({k: report["selected"][k].get("status") for k in TARGETS}), json.dumps(report["candidates"]))


if __name__ == "__main__":
    main()
