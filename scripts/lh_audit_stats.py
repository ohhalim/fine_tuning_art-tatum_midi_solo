#!/usr/bin/env python3
"""Causal left-hand audit statistics (issue #1750, docs/experiments/CAUSAL_LH_AUDIT.md). Runs once, after the
preregistration is committed and reviewed. No model, no training, two-hand MIDI only.

For each song and each role-proxy rule (scripts/lh_role_proxy.py) it builds, for every target note, the condition a
real-time system could see at the decision time d = target onset - lead, and counts separately:
  K1  condition notes whose id is a target id (or the target itself)
  K2  condition notes that start after d
  K3  condition notes whose role is not known yet at d (rule window still open)
  K4  (offline only) condition notes dropped because they are released in (d, target onset]: release info from after d
Every count keeps its denominator (targets, condition entries). The same targets are answered three more ways:
an independent event replay (all targets), prefix-only extraction and a different future suffix (sampled targets).
The offline shortcut (notes sounding at the target onset) is measured, not used.

  lh_audit_stats.py <bebop_rh manifest> <A Fable dir> <raw out dir> <report json> [--workers N] [--decisions SET]

Decision sets: "ideal" (default, #1750): d = onset - 0 and onset - 1 beat. "runtime_block" (#1754,
docs/experiments/BLOCK_SNAPSHOT_LH_AUDIT.md): d = the block snapshot of scripts/block_contract.py at four block phases,
plus a fixed-delay diagnostic at the largest warm gap (onset - 1406.25 ms). Onsets before t0 (the primer) are left out
of that decision. Targets of the cold block 0 are also counted on their own (cold vs warm, #1755 review Q2).

Main stratum: the manifest's train songs only (val/test are never opened). A Fable: separate exploratory stratum.
"""
from __future__ import annotations

import bisect
import collections
import csv
import gzip
import hashlib
import io
import json
import math
import multiprocessing
import os
import platform
import statistics
import subprocess
import sys
import time
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from block_contract import BEAT, BLOCK, PREBUFFER_LEAD_MS, WORST_LEAD_MS, block_of  # noqa: E402
from block_contract import snapshot as block_snapshot  # noqa: E402
from lh_role_proxy import (GRID_MIN, GRID_MS, GROUP_MS, INF, LEAD_MS, RECENT_MS, RULES, SPLIT_PITCH, Note,  # noqa: E402
                           offline_condition, online_condition, prefix_view, recent_condition, replay)

LEADS = {"lead0": 0.0, "lead1beat": LEAD_MS}   # per-note ideal decision time d = onset - L, not the runtime block snapshot
CONDS = ("sounding", "recent")
PREFIX_SAMPLES = 64              # targets per song and rule for the prefix / suffix checks (evenly spaced)
DONOR_SPAN_MS = 30_000.0         # the replacement future: the next song's first 30 s, starting 1 us after d
DONOR_ID_BASE = 10_000_000
NONEMPTY_FLOOR = Fraction(1, 10)  # sanity floor at lead1beat, judged on the raw counts (rounded ratios are display only)
DUP_MS = 1.0
FABLE_SONGS = 13                 # A Fable track count, registered; a different count stops the run
PHASES = (0.0, BLOCK / 4, BLOCK / 2, 3 * BLOCK / 4)   # t0 of the block grid inside a song (#1754)
# Every cold snapshot (t0 - 1000 ms) is then before 0, so with onsets >= 0 a cold target's condition is empty. The audit
# counts cold targets and cold nonempty conditions on their own; a nonempty cold condition breaks construction (review Q2).
assert max(PHASES) < PREBUFFER_LEAD_MS
CONSTRUCTIBLE = {"ideal": "causal pairs constructible",
                 "runtime_block": "causal pairs constructible under the proposed block contract"}   # rollback rule not built (Q3)


def decision_set(name):
    """({decision: onset -> d, or None when the onset is not generated}, decisions the verdict is judged on)."""
    if name == "ideal":
        return {"lead0": lambda t: t - 0.0, "lead1beat": lambda t: t - LEAD_MS}, ("lead1beat",)
    if name == "runtime_block":
        ds = {f"block_p{i}": (lambda t, p=p: block_snapshot(t, p)) for i, p in enumerate(PHASES)}
        ds["lead_worst"] = lambda t: t - WORST_LEAD_MS
        return ds, tuple(f"block_p{i}" for i in range(len(PHASES)))
    raise ValueError(name)


def cold_rule(ln):
    """For a block decision: onset -> True when the cold block 0 commits it. None for every other decision."""
    if not ln.startswith("block_p"):
        return None
    t0 = PHASES[int(ln[len("block_p"):])]
    return lambda t: block_of(t, t0) == 0


def list_midi(root):
    """MIDI files under root at any depth (PiJAMA keeps songs in <pianist>/<album>/)."""
    return sorted(os.path.join(d, f) for d, _, fs in os.walk(root) for f in fs if f.lower().endswith((".mid", ".midi")))


def sha1_file(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_notes(path):
    """Notes in ms from pretty_midi (tempo map applied). Ids follow (onset, pitch, offset, velocity) order.
    pretty_midi gives numpy float64 times; they become plain floats (same value) so every count stays a Python int."""
    import pretty_midi
    pm = pretty_midi.PrettyMIDI(path)
    raw = sorted((float(x.start) * 1000.0, int(x.pitch), float(x.end) * 1000.0, int(x.velocity))
                 for ins in pm.instruments if not ins.is_drum for x in ins.notes)
    info = {"instruments": len(pm.instruments), "drum_tracks": sum(1 for i in pm.instruments if i.is_drum),
            "pedal_cc64": sum(1 for i in pm.instruments for c in i.control_changes if c.number == 64),
            "nonpositive_length": sum(1 for on, _, off, _ in raw if off <= on)}
    return [Note(i, p, on, off, v) for i, (on, p, off, v) in enumerate(raw)], info


def pcmask(cond):
    m = 0
    for n in cond:
        m |= 1 << (n.pitch % 12)
    return m


def leak_counts(cond, d, roles, target_ids, target):
    """K1, K2, K3 for one condition list (see module docstring), plus exact duplicates of the target note."""
    k1 = sum(1 for n in cond if n.nid in target_ids or n.nid == target.nid)
    k2 = sum(1 for n in cond if n.on > d)
    k3 = sum(1 for n in cond if roles[n.nid].available_at > d)
    dup = sum(1 for n in cond if n.pitch == target.pitch and abs(n.on - target.on) < DUP_MS)
    return k1, k2, k3, dup


def future_release_drops(notes, roles, d, t_on):
    """Condition notes a real-time system sees as sounding at d that the offline shortcut drops because they are
    released in (d, t_on]."""
    return sum(1 for n in notes if roles[n.nid].role == "cond" and roles[n.nid].available_at <= d and n.on <= d < n.off <= t_on)


def sample_indices(n, k=PREFIX_SAMPLES):
    if n <= k:
        return list(range(n))
    return sorted({round(i * (n - 1) / (k - 1)) for i in range(k)})


def new_cond_stats():
    return {"targets": 0, "nonempty": 0, "entries": 0, "K1": 0, "K2": 0, "K3": 0, "dup_of_target": 0, "targets_with_leak": 0,
            "replay_mismatch": 0, "negative_d": 0, "sizes": [], "pcmasks": [], "age_from_available": [], "age_from_onset": []}


def audit_song(job):
    """One song, all rules. Returns aggregates and per-target rows. Pure function of the files it reads."""
    out = {"idx": job["idx"], "stratum": job["stratum"], "sha1_expected": job.get("sha1"), "error": None}
    try:
        out["sha1"] = sha1_file(job["path"])
        if job.get("sha1") and out["sha1"] != job["sha1"]:
            out["error"] = "sha1 mismatch"
            return out
        notes, info = load_notes(job["path"])
        donor, _ = load_notes(job["donor_path"])
    except Exception as e:  # recorded per song; the run goes on
        out["error"] = f"{type(e).__name__}: {e}"[:300]
        return out
    donor = [n for n in donor if n.on < DONOR_SPAN_MS]
    out.update(info)
    out["notes"] = len(notes)
    out["duration_s"] = round(max((n.off for n in notes), default=0.0) / 1000.0, 3)
    out["rules"], rows = {}, []
    set_name = job.get("decisions", "ideal")
    decisions, _ = decision_set(set_name)
    span = max(max((n.off - n.on for n in notes), default=0.0), RECENT_MS) + 1.0
    for rule_name, rule in RULES.items():
        roles = rule(notes)
        conds = sorted((n for n in notes if roles[n.nid].role == "cond"), key=lambda n: (n.on, n.nid))
        cons = [n.on for n in conds]

        def near(d, conds=conds, cons=cons):
            """Condition-role notes that can matter at d: started in [d - span, d]. span >= the longest note and the
            recent window, so the registered functions give the same answer on this slice as on all notes (the
            replay below, on all notes, checks it for every target)."""
            return conds[bisect.bisect_left(cons, d - span):bisect.bisect_right(cons, d)]
        targets = sorted((n for n in notes if roles[n.nid].role == "target"), key=lambda n: (n.on, n.nid))
        target_ids = {n.nid for n in targets}
        r = {"role_counts": collections.Counter(f"{x.role}:{x.reason}" if x.reason else x.role for x in roles.values()),
             "cond_role_delay": [roles[n.nid].available_at - n.on for n in notes if roles[n.nid].role == "cond"],
             "online": {ln: {c: new_cond_stats() for c in CONDS} for ln in decisions}, "offline": {}}
        cache, dmap = {}, {}
        for ln, dfun in decisions.items():
            for t in targets:
                d = dfun(t.on)
                if d is not None:
                    dmap[(ln, t.nid)] = d
            gen = [t for t in targets if (ln, t.nid) in dmap]
            ds = [dmap[(ln, t.nid)] for t in gen]
            rep = replay(notes, rule_name, ds)
            for t, d, (rs, rr) in zip(gen, ds, rep):
                cand = near(d)
                cs, cr = online_condition(cand, roles, d), recent_condition(cand, roles, d)
                cache[(ln, t.nid)] = (cs, cr)
                for c, cond, rid in (("sounding", cs, rs), ("recent", cr, rr)):
                    st = r["online"][ln][c]
                    k1, k2, k3, dup = leak_counts(cond, d, roles, target_ids, t)
                    st["targets"] += 1
                    st["negative_d"] += bool(d < 0)
                    st["nonempty"] += bool(cond)
                    st["entries"] += len(cond)
                    st["sizes"].append(len(cond))
                    if cond:
                        st["pcmasks"].append(pcmask(cond))
                    for k, v in (("K1", k1), ("K2", k2), ("K3", k3), ("dup_of_target", dup)):
                        st[k] += v
                    st["targets_with_leak"] += (k1 + k2 + k3) > 0
                    st["age_from_available"].extend(d - roles[n.nid].available_at for n in cond)
                    st["age_from_onset"].extend(d - n.on for n in cond)
                    st["replay_mismatch"] += frozenset(n.nid for n in cond) != rid
        for ln in decisions:
            off = {"targets": 0, "nonempty": 0, "entries": 0, "K1": 0, "K2": 0, "K3": 0, "dup_of_target": 0, "targets_with_leak": 0,
                   "K4_future_release_drops": 0, "targets_with_K4": 0, "offline_nonempty_online_sounding_empty": 0,
                   "offline_nonempty_online_recent_empty": 0, "jaccard_vs_sounding": [], "jaccard_vs_recent": [], "pcmasks": []}
            for t in targets:
                if (ln, t.nid) not in dmap:
                    continue
                d = dmap[(ln, t.nid)]
                co = offline_condition(near(t.on), roles, t.on)
                k1, k2, k3, dup = leak_counts(co, d, roles, target_ids, t)
                k4 = future_release_drops(near(d), roles, d, t.on)
                off["targets"] += 1
                off["nonempty"] += bool(co)
                off["entries"] += len(co)
                for k, v in (("K1", k1), ("K2", k2), ("K3", k3), ("dup_of_target", dup), ("K4_future_release_drops", k4)):
                    off[k] += v
                off["targets_with_leak"] += (k1 + k2 + k3) > 0
                off["targets_with_K4"] += k4 > 0
                if co:
                    off["pcmasks"].append(pcmask(co))
                cs, cr = cache[(ln, t.nid)]
                ido = {n.nid for n in co}
                for key, cond in (("sounding", cs), ("recent", cr)):
                    idc = {n.nid for n in cond}
                    if ido and not idc:
                        off[f"offline_nonempty_online_{key}_empty"] += 1
                    if ido or idc:
                        off[f"jaccard_vs_{key}"].append(len(ido & idc) / len(ido | idc))
                if set_name == "ideal" and ln == "lead1beat":
                    rows.append([job["stratum"], job["idx"], rule_name, t.nid, round(t.on, 3), t.pitch, len(cache[("lead0", t.nid)][0]),
                                 len(cs), len(cr), len(co), pcmask(cs), pcmask(cr), pcmask(co), k2, k3, k4])
            r["offline"][ln] = off
        if set_name != "ideal":
            r["primer_excluded"] = {ln: sum(1 for t in targets if (ln, t.nid) not in dmap) for ln in decisions}
            r["lead_ms"] = {ln: [t.on - dmap[(ln, t.nid)] for t in targets if (ln, t.nid) in dmap] for ln in decisions}
            r["cold"] = {}
            for ln in decisions:
                is_cold = cold_rule(ln)
                if is_cold is None:
                    continue
                cold = [t for t in targets if (ln, t.nid) in dmap and is_cold(t.on)]
                r["cold"][ln] = {c: {"targets": len(cold), "nonempty": sum(bool(cache[(ln, t.nid)][i]) for t in cold)}
                                 for i, c in enumerate(CONDS)}
            for t in targets:
                row = [job["stratum"], job["idx"], rule_name, t.nid, round(t.on, 3), t.pitch]
                for ln in decisions:
                    if (ln, t.nid) in dmap:
                        cs, cr = cache[(ln, t.nid)]
                        row += [round(dmap[(ln, t.nid)], 3), len(cs), len(cr), pcmask(cs), pcmask(cr)]
                    else:
                        row += ["", "", "", "", ""]
                rows.append(row)
        # prefix-only extraction and a replaced future, on evenly spaced targets
        chk = {"samples": 0, "prefix_mismatch": 0, "prefix_finite_off_exposed": 0, "suffix_mismatch": 0}
        for i in sample_indices(len(targets)):
            t = targets[i]
            for ln in decisions:
                if (ln, t.nid) not in dmap:
                    continue
                d = dmap[(ln, t.nid)]
                cs, cr = cache[(ln, t.nid)]
                full = (sorted(n.nid for n in cs), sorted(n.nid for n in cr))
                pre = prefix_view(notes, d)
                pr = rule(pre)
                ps = online_condition(pre, pr, d)
                chk["prefix_mismatch"] += (sorted(n.nid for n in ps), sorted(n.nid for n in recent_condition(pre, pr, d))) != full
                chk["prefix_finite_off_exposed"] += sum(1 for n in ps if n.off != INF)
                fut = [Note(DONOR_ID_BASE + n.nid, n.pitch, d + 0.001 + n.on, d + 0.001 + n.off, n.vel) for n in donor]
                alt = pre + fut
                ar = rule(alt)
                chk["suffix_mismatch"] += (sorted(n.nid for n in online_condition(alt, ar, d)),
                                           sorted(n.nid for n in recent_condition(alt, ar, d))) != full
                chk["samples"] += 1
        r["invariance"] = chk
        r["targets"] = len(targets)
        out["rules"][rule_name] = r
    out["rows"] = rows
    out["decisions"] = list(decisions)
    return out


def dist(xs):
    if not xs:
        return None
    s = sorted(xs)

    def q(p):
        return s[min(len(s) - 1, max(0, math.ceil(p * len(s)) - 1))]
    return {"n": len(s), "mean": round(statistics.fmean(s), 4), "min": round(s[0], 4), "p10": round(q(0.10), 4),
            "median": round(q(0.50), 4), "p90": round(q(0.90), 4), "max": round(s[-1], 4)}


def pc_diversity(masks):
    if not masks:
        return {"nonempty": 0}
    c = collections.Counter(masks)
    n = len(masks)
    ent = -sum(v / n * math.log2(v / n) for v in c.values())
    return {"nonempty": n, "distinct_pc_sets": len(c), "entropy_bits": round(ent, 3),
            "top5_share": round(sum(v for _, v in c.most_common(5)) / n, 4),
            "pc_set_size": dist([bin(m).count("1") for m in masks])}


def ratio(a, b):
    return round(a / b, 4) if b else None


def aggregate(results, label, names=tuple(LEADS)):
    """Pool per-song results of one stratum into the report tables."""
    ok = [r for r in results if r["error"] is None]
    agg = {"songs": len(results), "songs_ok": len(ok), "errors": [{"idx": r["idx"], "error": r["error"]} for r in results if r["error"]],
           "notes": sum(r["notes"] for r in ok), "pedal_cc64_total": sum(r["pedal_cc64"] for r in ok),
           "nonpositive_length_total": sum(r["nonpositive_length"] for r in ok), "rules": {}}
    for rule in RULES:
        rr = [r["rules"][rule] for r in ok]
        roles = collections.Counter()
        for x in rr:
            roles.update(x["role_counts"])
        tab = {"role_counts": dict(roles), "targets": sum(x["targets"] for x in rr),
               "target_share_of_notes": ratio(sum(x["targets"] for x in rr), agg["notes"]),
               "time_cond_role_delay_ms": dist([v for x in rr for v in x["cond_role_delay"]]), "online": {}, "offline": {}}
        for ln in names:
            tab["online"][ln] = {}
            for c in CONDS:
                sts = [x["online"][ln][c] for x in rr]
                tot = {k: sum(s[k] for s in sts) for k in ("targets", "nonempty", "entries", "K1", "K2", "K3", "dup_of_target",
                                                           "targets_with_leak", "replay_mismatch", "negative_d")}
                per_song = [ratio(s["nonempty"], s["targets"]) for s in sts if s["targets"]]
                tab["online"][ln][c] = {**tot, "nonempty_ratio": ratio(tot["nonempty"], tot["targets"]),
                                        "mean_size": ratio(tot["entries"], tot["targets"]),
                                        "size": dist([v for s in sts for v in s["sizes"]]),
                                        "pc_diversity": pc_diversity([m for s in sts for m in s["pcmasks"]]),
                                        "time_age_from_available_ms": dist([v for s in sts for v in s["age_from_available"]]),
                                        "time_age_from_onset_ms": dist([v for s in sts for v in s["age_from_onset"]]),
                                        "per_song_nonempty_ratio": dist(per_song),
                                        "songs_with_zero_nonempty": sum(1 for v in per_song if v == 0)}
            offs = [x["offline"][ln] for x in rr]
            keys = ("targets", "nonempty", "entries", "K1", "K2", "K3", "dup_of_target", "targets_with_leak", "K4_future_release_drops",
                    "targets_with_K4", "offline_nonempty_online_sounding_empty", "offline_nonempty_online_recent_empty")
            tot = {k: sum(o[k] for o in offs) for k in keys}
            tab["offline"][ln] = {**tot, "nonempty_ratio": ratio(tot["nonempty"], tot["targets"]),
                                  "entries_future_onset_share": ratio(tot["K2"], tot["entries"]),
                                  "entries_role_unknown_share": ratio(tot["K3"], tot["entries"]),
                                  "targets_with_leak_share": ratio(tot["targets_with_leak"], tot["targets"]),
                                  "targets_with_K4_share": ratio(tot["targets_with_K4"], tot["targets"]),
                                  "jaccard_vs_sounding": dist([v for o in offs for v in o["jaccard_vs_sounding"]]),
                                  "jaccard_vs_recent": dist([v for o in offs for v in o["jaccard_vs_recent"]]),
                                  "pc_diversity": pc_diversity([m for o in offs for m in o["pcmasks"]])}
        inv = {k: sum(x["invariance"][k] for x in rr) for k in ("samples", "prefix_mismatch", "prefix_finite_off_exposed", "suffix_mismatch")}
        tab["invariance"] = inv
        if rr and "primer_excluded" in rr[0]:
            tab["primer_excluded"] = {ln: sum(x["primer_excluded"][ln] for x in rr) for ln in names}
            tab["time_lead_ms"] = {ln: dist([v for x in rr for v in x["lead_ms"][ln]]) for ln in names}
        if rr and "cold" in rr[0]:
            tab["cold"], tab["warm"] = {}, {}
            for ln in rr[0]["cold"]:
                tab["cold"][ln], tab["warm"][ln] = {}, {}
                for c in CONDS:
                    ct = sum(x["cold"][ln][c]["targets"] for x in rr)
                    cn = sum(x["cold"][ln][c]["nonempty"] for x in rr)
                    on = tab["online"][ln][c]
                    tab["cold"][ln][c] = {"targets": ct, "nonempty": cn}
                    tab["warm"][ln][c] = {"targets": on["targets"] - ct, "nonempty": on["nonempty"] - cn,
                                          "nonempty_ratio": ratio(on["nonempty"] - cn, on["targets"] - ct)}
        agg["rules"][rule] = tab
    return agg


def groups(results, meta, key, dn="lead1beat"):
    """Pooled nonempty ratio at one decision per pianist / recording type."""
    out = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0, 0]))
    for r in results:
        if r["error"]:
            continue
        g = meta[r["idx"]][key]
        for rule in RULES:
            for c in CONDS:
                st = r["rules"][rule]["online"][dn][c]
                acc = out[g][f"{rule}/{c}"]
                acc[0] += st["targets"]
                acc[1] += st["nonempty"]
                acc[2] += 1
    return {g: {k: {"songs": v[2], "targets": v[0], "nonempty_ratio": ratio(v[1], v[0])} for k, v in d.items()} for g, d in out.items()}


def check_counts(agg):
    """The raw check numbers per rule. The exploratory stratum reports only these (no eligibility verdict)."""
    out = {}
    for rule, t in agg["rules"].items():
        on = [t["online"][ln][c] for ln in t["online"] for c in CONDS]
        out[rule] = {"online_leaks": {k: sum(s[k] for s in on) for k in ("K1", "K2", "K3")},
                     "replay_mismatch": sum(s["replay_mismatch"] for s in on), **t["invariance"]}
        if "cold" in t:
            out[rule]["cold_nonempty"] = sum(t["cold"][ln][c]["nonempty"] for ln in t["cold"] for c in CONDS)
    return out


def nondegenerate(nonempty, targets):
    return targets > 0 and Fraction(nonempty, targets) >= NONEMPTY_FLOOR


def verdicts(agg, judged=("lead1beat",), constructible=CONSTRUCTIBLE["ideal"]):
    """Registered decision (CAUSAL_LH_AUDIT.md, BLOCK_SNAPSHOT_LH_AUDIT.md), main stratum only: per rule,
    construction = data intact + online leaks 0 + replay, prefix and suffix checks 0 (all decisions) + no nonempty
    cold condition (block decisions only); per rule and condition, non-degenerate = pooled nonempty / targets >= floor
    on the raw counts at every judged decision."""
    data_ok = agg["songs_ok"] == agg["songs"] and agg.get("note_count_mismatch", 0) == 0
    out = {}
    for rule, cc in check_counts(agg).items():
        construct = (data_ok and not any(cc["online_leaks"].values()) and cc["replay_mismatch"] == 0
                     and cc["prefix_mismatch"] == 0 and cc["prefix_finite_off_exposed"] == 0 and cc["suffix_mismatch"] == 0
                     and cc.get("cold_nonempty", 0) == 0)
        out[rule] = {"data_ok": data_ok, **cc, "construction_ok": construct, "conditions": {}}
        for c in CONDS:
            row, nd = {}, True
            for dn in judged:
                st = agg["rules"][rule]["online"][dn][c]
                row.update({f"{dn}_nonempty": st["nonempty"], f"{dn}_targets": st["targets"],
                            f"{dn}_nonempty_ratio_display": st["nonempty_ratio"]})
                nd = nd and nondegenerate(st["nonempty"], st["targets"])
            out[rule]["conditions"][c] = {**row, "nondegenerate": nd,
                                          "verdict": (constructible if construct and nd else
                                                      "constructible but almost always empty" if construct else "construction failed")}
    return out


def write_rows(path, header, rows):
    """gzip CSV with no name and mtime 0 in the header, so the same rows always give the same file bytes."""
    with open(path, "wb") as fb, gzip.GzipFile(filename="", fileobj=fb, mode="wb", mtime=0) as gz, \
            io.TextIOWrapper(gz, newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for row in rows:
            w.writerow(row)


def git_head():
    try:
        return subprocess.run(["git", "-C", HERE, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


def main() -> None:
    args = sys.argv[1:]
    workers, set_name = 6, "ideal"
    if "--decisions" in args:
        i = args.index("--decisions")
        set_name = args[i + 1]
        del args[i:i + 2]
    decisions, judged = decision_set(set_name)
    names = tuple(decisions)
    if "--workers" in args:
        i = args.index("--workers")
        workers = int(args[i + 1])
        del args[i:i + 2]
    manifest_path, fable_dir, raw_dir, report_path = args[:4]
    t0 = time.time()
    manifest = json.load(open(manifest_path))
    train = [s for s in manifest["songs"] if s["split"] == "train"]
    assert len(train) == 189, len(train)
    fable = list_midi(fable_dir)
    assert len(fable) == FABLE_SONGS, (len(fable), fable_dir)   # run 2 silently read 0 songs from the album folder
    jobs, meta = [], {}
    for i, s in enumerate(train):
        rec = "live" if "/midi/live/" in s["source"] else "studio" if "/midi/studio/" in s["source"] else "other"
        meta[("main", i)] = {"pianist": s["pianist"], "rec": rec, "manifest_notes": s["all_notes"], "source": s["source"], "sha1": s["sha1"]}
        jobs.append({"idx": i, "stratum": "main", "path": s["source"], "sha1": s["sha1"], "donor_path": train[(i + 1) % len(train)]["source"],
                     "decisions": set_name})
    for i, p in enumerate(fable):
        meta[("afable", i)] = {"pianist": "Tigran Hamasyan", "rec": "studio", "manifest_notes": None, "source": p, "sha1": None}
        jobs.append({"idx": i, "stratum": "afable", "path": p, "sha1": None, "donor_path": fable[(i + 1) % len(fable)],
                     "decisions": set_name})
    with multiprocessing.get_context("spawn").Pool(workers) as pool:
        results = pool.map(audit_song, jobs, chunksize=1)
    os.makedirs(raw_dir, exist_ok=True)
    raw_path = os.path.join(raw_dir, "targets_lead1beat.csv.gz" if set_name == "ideal" else f"targets_{set_name}.csv.gz")
    if set_name == "ideal":
        header = ["stratum", "song_idx", "rule", "target_nid", "target_on_ms", "target_pitch", "n_sounding_lead0", "n_sounding",
                  "n_recent", "n_offline", "pcmask_sounding", "pcmask_recent", "pcmask_offline", "offline_K2_future_onset",
                  "offline_K3_role_unknown", "offline_K4_future_release_drops"]
    else:
        header = (["stratum", "song_idx", "rule", "target_nid", "target_on_ms", "target_pitch"]
                  + [f"{ln}_{k}" for ln in names for k in ("d_ms", "n_sounding", "n_recent", "pcmask_sounding", "pcmask_recent")])
    write_rows(raw_path, header, (row for r in results for row in r.pop("rows", [])))
    ideal = set_name == "ideal"
    report = {"issue": 1750 if ideal else 1754,
              "doc": "docs/experiments/CAUSAL_LH_AUDIT.md" if ideal else "docs/experiments/BLOCK_SNAPSHOT_LH_AUDIT.md", "quality_claimed": False,
              "run": {"git_head": git_head(), "lh_role_proxy_sha256": sha256_file(os.path.join(HERE, "lh_role_proxy.py")),
                      "stats_script_sha256": sha256_file(os.path.abspath(__file__)), "python": platform.python_version(),
                      "pretty_midi": __import__("pretty_midi").__version__, "workers": workers, "seconds": None,
                      "manifest": manifest_path, "manifest_sha256": sha256_file(manifest_path), "fable_dir": fable_dir,
                      "raw_targets": raw_path, "raw_targets_sha256": None},
              "params": {"split_pitch": SPLIT_PITCH, "group_ms": GROUP_MS, "grid_ms": GRID_MS, "grid_min": GRID_MIN, "leads_ms": LEADS,
                         "recent_ms": RECENT_MS, "prefix_samples": PREFIX_SAMPLES, "donor_span_ms": DONOR_SPAN_MS,
                         "nonempty_floor": str(NONEMPTY_FLOOR), "dup_ms": DUP_MS,
                         "lead_semantics": "per-note ideal decision time d = target onset - L (diagnostic); the runtime block "
                                           "snapshot (now = t0 + k*937.5) is not verified here (#1750 review M2)", "units": "ms from pretty_midi seconds x 1000 (tempo map applied)"},
              "strata": {}}
    if not ideal:
        report["run"]["block_contract_sha256"] = sha256_file(os.path.join(HERE, "block_contract.py"))
        report["params"].pop("leads_ms")
        report["params"]["decision_set"] = set_name
        report["params"]["decisions"] = {**{f"block_p{i}": f"block snapshot (block_contract.snapshot) with t0 = {p} ms" for i, p in enumerate(PHASES)},
                                         "lead_worst": f"onset - {WORST_LEAD_MS} ms: fixed-delay diagnostic at the largest gap of warm "
                                                       "blocks (k >= 1). Not a lower bound of any phase's coverage (conditions are "
                                                       "not monotone in d) and not a model of the cold block 0"}
        report["params"]["judged"] = list(judged)
        report["params"].update({"block_ms": BLOCK, "beat_ms": BEAT, "prebuffer_lead_ms": PREBUFFER_LEAD_MS, "worst_lead_ms": WORST_LEAD_MS})
        report["params"]["lead_semantics"] = ("proposed runtime block snapshot contract (#1754): block k >= 1 observes at t0 + k*B and "
                                              "commits onsets in [t0 + k*B + beat, t0 + (k+1)*B + beat); block 0 observes no later than "
                                              "t0 - 1000 ms (an upper bound); requires regeneration whenever the whole condition "
                                              "representation at a block start differs from the previous one (not implemented)")
    for st in ("main", "afable"):
        rs = [r for r in results if r["stratum"] == st]
        agg = aggregate(rs, st, names)
        if st == "main":
            agg["note_count_mismatch"] = sum(1 for r in rs if r["error"] is None and r["notes"] != meta[("main", r["idx"])]["manifest_notes"])
        for dn in judged:
            agg[f"per_pianist_{dn}"] = groups(rs, {r["idx"]: meta[(st, r["idx"])] for r in rs}, "pianist", dn)
            agg[f"per_recording_type_{dn}"] = groups(rs, {r["idx"]: meta[(st, r["idx"])] for r in rs}, "rec", dn)
        agg["songs_table"] = [{"idx": r["idx"], "sha1": r.get("sha1"), "pianist": meta[(st, r["idx"])]["pianist"],
                               "rec": meta[(st, r["idx"])]["rec"], "source": meta[(st, r["idx"])]["source"], "error": r["error"],
                               "notes": r.get("notes"), "duration_s": r.get("duration_s"),
                               **({rule: {"targets": r["rules"][rule]["targets"],
                                          **{f"{c}_{dn}_nonempty": ratio(r["rules"][rule]["online"][dn][c]["nonempty"],
                                                                         r["rules"][rule]["online"][dn][c]["targets"])
                                             for dn in judged for c in CONDS}}
                                   for rule in RULES} if r["error"] is None else {})} for r in rs]
        if st == "main":
            agg["verdicts"] = verdicts(agg, judged, CONSTRUCTIBLE[set_name])
        else:
            agg["checks"] = check_counts(agg)
            agg["note"] = ("exploratory stress stratum: check numbers and statistics only, no eligibility verdict; never pooled "
                           "with the main stratum, never used for reference selection or style validation")
        report["strata"][st] = agg
    report["run"]["raw_targets_sha256"] = sha256_file(raw_path)
    with gzip.open(raw_path, "rb") as f:
        report["run"]["raw_targets_content_sha256"] = hashlib.sha256(f.read()).hexdigest()
    report["run"]["seconds"] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    main_v = report["strata"]["main"]["verdicts"]
    print("main", json.dumps({rule: {c: v["conditions"][c]["verdict"] for c in CONDS} for rule, v in main_v.items()}, ensure_ascii=False))
    print("afable checks", json.dumps(report["strata"]["afable"]["checks"], ensure_ascii=False))


if __name__ == "__main__":
    main()
