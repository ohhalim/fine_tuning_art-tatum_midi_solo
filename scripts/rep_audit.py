#!/usr/bin/env python3
"""Representation audit (Astra #214): the same raw A Fable segments through the Aria tokenizer round trip
and through the CMT input path (top-voice proxy, then the fixed frame grid of CMT's preprocess.py with
velocity removed). Run with the Aria venv: rep_audit.py <aria repo> <out dir>. No training, no generation."""
import collections, glob, hashlib, json, os, statistics, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from aria_t4b_sim import read_notes, write_notes      # mido helpers, 1 tick = 1 ms

SRC = "/Users/ohhalim/git_box/fine_tuning_art-tatum_midi_solo/midi_dataset/midi/studio/Tigran Hamasyan/A Fable"
T0, DUR = 60.0, 8.0
CMT_FRAME = 1 / 8          # preprocess.py: frame_per_second = (16 / 4) * (120 / 60) = 8, a fixed seconds grid
CMT_PITCH_RANGE = 48       # preprocess.py: instance skipped if highest - 12 * (lowest // 12) >= 48
CMT_MAX_LEAP = 12          # preprocess.py: instance dropped if consecutive onsets differ by more than 12
ARIA_MAX_DUR = 5.0         # ariautils config.json: max_dur_ms 5000


def top_voice(notes, win=0.03):
    out, cur = [], []
    for n in sorted(notes, key=lambda n: (n[1], -n[0])):
        if cur and n[1] - cur[0][1] > win:
            out.append(max(cur, key=lambda x: x[0]))
            cur = []
        cur.append(n)
    if cur:
        out.append(max(cur, key=lambda x: x[0]))
    return out


def grid(tv, phase=0.0):
    frames, collide = collections.OrderedDict(), 0
    for p, s, e, v in tv:
        fr = round((s - phase) / CMT_FRAME)
        if fr in frames:
            collide += 1
            continue
        frames[fr] = (p, s, e)
    return frames, collide


def simultaneity_changes(pairs):
    """pairs: (original note, round-trip note). Count close pairs whose equal/unequal onset status flipped."""
    merged = split = 0
    for i in range(len(pairs)):
        for j in range(i + 1, len(pairs)):
            (a, a2), (b, b2) = pairs[i], pairs[j]
            if abs(a[1] - b[1]) > 0.02:
                continue
            same, same2 = round(a[1], 3) == round(b[1], 3), round(a2[1], 3) == round(b2[1], 3)
            merged += (not same) and same2
            split += same and not same2
    return merged, split


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]


def chords_in(notes, win=0.03):
    groups, cur = [], []
    for n in sorted(notes, key=lambda n: n[1]):
        if cur and n[1] - cur[0][1] > win:
            groups.append(cur)
            cur = []
        cur.append(n)
    if cur:
        groups.append(cur)
    return sum(1 for g in groups if len(g) >= 2), len(groups)


def main():
    sys.path.insert(0, sys.argv[1])
    from ariautils.midi import MidiDict
    from ariautils.tokenizer import AbsTokenizer
    from aria.inference import get_inference_prompt
    tok = AbsTokenizer()
    out_dir = sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for f in sorted(glob.glob(SRC + "/*.midi")):
        name = os.path.basename(f)[:-5]
        allnotes = read_notes(f)
        end = max(n[2] for n in allnotes)
        if end < T0 + DUR:
            rows.append({"segment": name, "excluded": f"too short ({end:.1f} s)"})
            continue
        seg = [(p, s - T0, e - T0, v) for p, s, e, v in allnotes if T0 <= s < T0 + DUR]   # ends not clipped
        spath = os.path.join(out_dir, f"{name}_60-68.mid")
        write_notes(seg, spath)
        seg = read_notes(spath)                                  # what the tokenizer will read (1 ms ticks)
        # --- Aria round trip
        toks = get_inference_prompt(MidiDict.from_midi(spath), tok, 1e12)
        dpath = os.path.join(out_dir, f"{name}_aria_rt.mid")
        tok.detokenize(toks).to_midi().save(dpath)
        rt = read_notes(dpath)
        shift = min(n[1] for n in seg) - min(n[1] for n in rt)  # Aria drops leading silence
        rt = [(p, s + shift, e + shift, v) for p, s, e, v in rt]
        pool = collections.defaultdict(list)
        for n in rt:
            pool[n[0]].append(n)
        on_err, end_err, vel_err, matched, pairs = [], [], [], 0, []
        for n in seg:
            cands = [m for m in pool[n[0]] if abs(m[1] - n[1]) <= 0.015]
            if cands:
                m = min(cands, key=lambda m: abs(m[1] - n[1]))
                pool[n[0]].remove(m)
                matched += 1
                pairs.append((n, m))
                on_err.append(abs(m[1] - n[1])); end_err.append(abs(m[2] - n[2])); vel_err.append(abs(m[3] - n[3]))
        long_notes = [i for i, (n, m) in enumerate(pairs) if n[2] - n[1] > ARIA_MAX_DUR]
        short_end = [abs(m[2] - n[2]) for n, m in pairs if n[2] - n[1] <= ARIA_MAX_DUR]
        merged, split = simultaneity_changes(pairs)
        ch_seg, ev_seg = chords_in(seg)
        ch_rt, ev_rt = chords_in(rt)
        # --- CMT input path: top voice, then 8 frames/s grid, velocity removed
        tv = top_voice(seg)
        frames, collide = grid(tv)
        sweep = [grid(tv, ph / 1000)[1] for ph in range(0, 125, 25)]
        kept = list(frames.values())
        leaps = sum(1 for a, b in zip(kept, kept[1:]) if abs(b[0] - a[0]) > CMT_MAX_LEAP)
        q_on_err = [abs(fr * CMT_FRAME - s) for fr, (p, s, e) in frames.items()]
        q_dur_err = [abs(max(1, round((e - s) / CMT_FRAME)) * CMT_FRAME - (e - s)) for fr, (p, s, e) in frames.items()]
        base = 12 * (min(p for p, _, _ in frames.values()) // 12)
        out_of_range = sum(1 for p, _, _ in frames.values() if p - base >= CMT_PITCH_RANGE)
        vels = [n[3] for n in seg]
        rows.append({"segment": name, "notes": len(seg), "chord_events": ch_seg, "events": ev_seg,
                     "src_sha16": sha(f), "segment_sha16": sha(spath),
                     "aria": {"tokens": len(toks), "notes_after": len(rt), "matched_15ms": matched,
                              "notes_over_5s": len(long_notes),
                              "end_err_ms_max_notes_le_5s": round(1000 * max(short_end), 1) if short_end else None,
                              "simultaneous_pairs_merged": merged, "simultaneous_pairs_split": split,
                              "onset_err_ms_max": round(1000 * max(on_err), 1) if on_err else None,
                              "end_err_ms_median": round(1000 * statistics.median(end_err), 1) if end_err else None,
                              "end_err_ms_max": round(1000 * max(end_err), 1) if end_err else None,
                              "velocity_err_max": max(vel_err) if vel_err else None,
                              "chord_events_after": ch_rt},
                     "cmt": {"top_voice_notes": len(tv), "top_voice_share": round(len(tv) / len(seg), 3),
                             "top_voice_below_C4": sum(1 for n in tv if n[0] < 60),
                             "after_grid": len(frames), "grid_collisions_dropped": collide,
                             "collisions_phase_sweep_min_max": [min(sweep), max(sweep)],
                             "leaps_over_12": leaps,
                             "top_voice_range": max(p for p, _, _ in kept) - min(p for p, _, _ in kept),
                             "onset_err_ms_median": round(1000 * statistics.median(q_on_err), 1),
                             "onset_err_ms_max": round(1000 * max(q_on_err), 1),
                             "dur_err_ms_median": round(1000 * statistics.median(q_dur_err), 1),
                             "out_of_48_pitch_window": out_of_range,
                             "velocity_sd_removed": round(statistics.pstdev(vels), 1),
                             "kept_share_of_all_notes": round(len(frames) / len(seg), 3)}})
        print(name, rows[-1]["notes"], "aria", rows[-1]["aria"]["matched_15ms"], "/", rows[-1]["notes"], "cmt kept", rows[-1]["cmt"]["after_grid"], flush=True)
    json.dump(rows, open(os.path.join(out_dir, "audit.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
