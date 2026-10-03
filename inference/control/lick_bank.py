"""Licks from real right-hand lines, kept in beats, realized on the chords (docs/experiments/LICK_BANK.md).

The phrase realizer on a real bebop phrase was "the most stable, but hardly any
soloing: melody-like lines, not licks" (user, 2026-10-03); its source was a ballad
section (2.4 notes/s). A lick here is a run of at least ``MIN_NOTES`` notes with no
gap over ``GAP_S`` and a steady pulse (median inter-onset interval within the
eighth-note range of fast playing). Its rhythm is stored in beats, taking the
run's median inter-onset interval as an eighth note, so a lick played at 220 BPM
keeps its shape at the runtime tempo. Pitches are re-chosen by ``realizer``.
"""
from __future__ import annotations

import random
import statistics

MIN_NOTES, MAX_NOTES = 8, 24
GAP_S = 0.25
IOI_RANGE = (0.09, 0.26)        # median IOI of a run counted as eighth notes (about 115-330 BPM)


def runs(line):
    """Consecutive notes with gaps <= GAP_S."""
    out, cur = [], []
    for n in sorted(line, key=lambda n: n.start):
        if cur and n.start - cur[-1].end > GAP_S:
            out.append(cur)
            cur = []
        cur.append(n)
    if cur:
        out.append(cur)
    return out


def lick_from_run(run) -> dict | None:
    if not MIN_NOTES <= len(run) <= MAX_NOTES:
        return None
    iois = [b.start - a.start for a, b in zip(run, run[1:])]
    med = statistics.median(iois)
    if not IOI_RANGE[0] <= med <= IOI_RANGE[1]:
        return None
    eighth = med
    t0 = run[0].start
    # onsets and lengths in eighth notes, rounded to sixteenth notes
    on = [round((n.start - t0) / eighth * 2) / 2 for n in run]
    if any(b <= a for a, b in zip(on, on[1:])):
        return None
    # legato inside the lick: each note lasts to the next onset (the last one its own length)
    dur = [max(0.5, on[k + 1] - on[k]) for k in range(len(on) - 1)] + [max(1.0, round((run[-1].end - run[-1].start) / eighth * 2) / 2)]
    return {"onset_8ths": on, "dur_8ths": dur, "intervals": [b.pitch - a.pitch for a, b in zip(run, run[1:])],
            "velocity": [n.velocity for n in run], "length_8ths": on[-1] + dur[-1]}


def build(lines) -> list[dict]:
    bank = []
    for line in lines:
        for run in runs(line):
            lick = lick_from_run(run)
            if lick:
                bank.append(lick)
    return bank


def plan(bank, *, bars: int, bpm: float, seed: int, rest_8ths=(1, 4)):
    """Abstract events (for ``realizer.realize``) filling ``bars`` of 4/4 with licks and rests.

    A lick that does not fit the remaining time is replaced by another that does
    (up to 50 draws), so the plan does not stop early and leave the end silent."""
    rng = random.Random(seed)
    eighth = 60.0 / bpm / 2
    total = bars * 8
    pos = rng.choice([0, 1])                 # start on the beat or the & of 1
    events = []
    while pos < total - 4:
        fitting = None
        for _ in range(50):
            lick = rng.choice(bank)
            if pos + lick["length_8ths"] <= total:
                fitting = lick
                break
        if fitting is None:
            break
        lick = fitting
        vel = lick["velocity"]
        lo, hi = min(vel), max(vel)
        for k, (on, d) in enumerate(zip(lick["onset_8ths"], lick["dur_8ths"])):
            # source velocities come from different recordings: rescale each lick to 62-84, +6 on the beat
            v = 62 + (22 * (vel[k] - lo) / (hi - lo) if hi > lo else 11) + (6 if (pos + on) % 2 == 0 else 0)
            events.append({"onset": (pos + on) * eighth, "dur": d * eighth * 0.92, "velocity": int(min(90, v)),
                           "interval": None if k == 0 else lick["intervals"][k - 1], "phrase_end": False})
        events[-1]["phrase_end"] = True
        pos += int(lick["length_8ths"] + 0.999) + rng.randint(*rest_8ths)
    return events


def swing(notes, bpm: float, ratio: float = 2 / 3, min_s: float = 0.12):
    """Swung onsets for straight-eighth notes [(pitch, onset, end, velocity)], a line or chords.

    Off-beat eighths move later and each note's end is recomputed from what follows it
    (#1669: mapping the ends too squeezed off-beat notes to ~0.07 s, "plinky"):
    * held over the next onset (longer than 1.2x the gap): its own end, swung, so a
      sustained comp note is not cut by another voice
    * legato into the next onset (at least 0.8x the gap): up to the next swung onset
    * otherwise its length, at least ``min_s``
    Notes sharing an onset (chord tones) do not shorten each other. No note sounds into
    the next onset of the same pitch: after a short interval the note stays short
    (#1674 capped at any later onset, which would cut held comp notes; #1669's final
    ``max`` let two same-pitch notes overlap by 42 ms)."""
    beat = 60.0 / bpm

    def at(t):
        b = t / beat
        whole = int(b + 1e-6)
        f = b - whole
        f = f / 0.5 * ratio if f < 0.5 else ratio + (f - 0.5) / 0.5 * (1 - ratio)
        return (whole + f) * beat

    out = []
    for p, s, e, v in notes:
        on, length = at(s), e - s
        later = [q for q in notes if q[1] > s + 1e-6]
        nxt = min((q[1] for q in later), default=None)
        same = min((q[1] for q in later if q[0] == p), default=None)
        if nxt is not None and length > (nxt - s) * 1.2:
            end = at(e)
        elif nxt is not None and length >= (nxt - s) * 0.8:
            end = at(nxt) - 0.01
        else:
            end = on + max(min_s, length)
        if same is not None:
            end = min(end, at(same) - 0.01)
        out.append((p, on, max(end, on + 0.01), v))
    return out
