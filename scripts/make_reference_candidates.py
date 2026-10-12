#!/usr/bin/env python3
"""Reference-clip candidates for the user to listen to and choose (issue #1735 follow-up).

Fixed rule, no picking of moments: for each listed PiJAMA MIDI file, 30 s starting at 25 % of the file's length.
Notes that start inside the window are kept (cut at the window end), the sustain pedal state at the window start is
carried in, everything is rendered with the same piano (GeneralUser GS, fluidsynth) and converted to AAC (afconvert).
Straight rendering: no swing is added. Audio stays on this Mac (transcriptions of commercial recordings).

  make_reference_candidates.py <out dir>
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import mido

PIJAMA = "/Users/ohhalim/git_box/fine_tuning_art-tatum_midi_solo/midi_dataset/midi"
SF2 = os.path.expanduser("~/.local/share/soundfonts/generaluser-gs/v1.471.sf2")
WINDOW_S, START_FRAC = 30.0, 0.25
TIGRAN = "studio/Tigran Hamasyan/A Fable"
CLIPS = (
    [("Tigran Hamasyan", f"{TIGRAN}/{t}.midi") for t in (
        "A Fable", "A Memory That Became A Dream", "Illusion", "Kakavik (The Little Partridge)", "Longing",
        "Looking For The Block", "Mother, Where Are You", "Rain Shadow", "Samsara", "Someday My Prince Will Come",
        "The Legend Of The Moon", "The Spinners", "What The Waves Brought")]
    + [("Brad Mehldau", f"live/Brad Mehldau/10 Years Solo Live/{t} - Live.midi") for t in ("Holland", "Jigsaw Falling into Place", "Knives Out", "Teardrop")]
    + [("Barry Harris", "live/Barry Harris/The Maybeck Recital Series, Vol. 12/All God's Chillun Got Rhythm - Live At Maybeck Recital Hall, Berkeley, CA  March 12, 1990.midi"),
       ("Tommy Flanagan", "studio/Tommy Flanagan/Solo Piano/Con Alma.midi"),
       ("Bill Charlap", "studio/Bill Charlap/Rolling Stone Original/A Foggy Day - Live -  Solo Piano.midi"),
       ("Art Tatum", "studio/Art Tatum/The Art Tatum Solo Masterpieces, Vol. 2/Come Rain Or Come Shine.midi"),
       ("Art Tatum", "studio/Art Tatum/The Art Tatum Solo Masterpieces, Vol. 2/Elegy.midi")]
)


def events_in_seconds(path):
    t, out = 0.0, []
    for msg in mido.MidiFile(path):
        t += msg.time
        if msg.type in ("note_on", "note_off") or (msg.type == "control_change" and msg.control == 64):
            out.append((t, msg))
    return out, t


def excerpt(path, out_mid):
    ev, total = events_in_seconds(path)
    start = total * START_FRAC
    end = min(total, start + WINDOW_S)
    pedal = 0
    for t, m in ev:
        if t >= start:
            break
        if m.type == "control_change":
            pedal = m.value
    keep = [(0.0, mido.Message("control_change", control=64, value=pedal))]
    sounding = set()
    for t, m in ev:
        if not (start <= t < end):
            continue
        if m.type == "note_on" and m.velocity > 0:
            sounding.add(m.note)
            keep.append((t - start, mido.Message("note_on", note=m.note, velocity=m.velocity)))
        elif m.type in ("note_on", "note_off"):
            if m.note in sounding:                      # only notes that started inside the window
                sounding.discard(m.note)
                keep.append((t - start, mido.Message("note_off", note=m.note, velocity=0)))
        else:
            keep.append((t - start, mido.Message("control_change", control=64, value=m.value)))
    dur = end - start
    keep += [(dur, mido.Message("note_off", note=n, velocity=0)) for n in sorted(sounding)]
    keep.append((dur, mido.Message("control_change", control=64, value=0)))
    mf = mido.MidiFile(ticks_per_beat=480)
    tr = mido.MidiTrack()
    mf.tracks.append(tr)
    tr.append(mido.MetaMessage("set_tempo", tempo=500000, time=0))
    tr.append(mido.Message("program_change", program=0, time=0))
    last = 0
    for t, m in sorted(keep, key=lambda x: x[0]):
        tick = int(round(t * 960))                      # 480 ticks per beat at 120 BPM = 960 ticks per second
        tr.append(m.copy(time=tick - last))
        last = tick
    mf.save(out_mid)
    notes = sum(1 for _, m in keep if m.type == "note_on")
    return {"start_s": round(start, 2), "end_s": round(end, 2), "total_s": round(total, 2), "notes": notes}


def main() -> None:
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    rows = []
    for i, (artist, rel) in enumerate(CLIPS, 1):
        base = f"{i:02d}"
        mid, wav, m4a = (os.path.join(out, f"{base}.{x}") for x in ("mid", "wav", "m4a"))
        info = excerpt(os.path.join(PIJAMA, rel), mid)
        subprocess.run(["fluidsynth", "-ni", "-q", "-F", wav, "-r", "44100", SF2, mid], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(["afconvert", "-f", "m4af", "-d", "aac", "-b", "160000", wav, m4a], check=True)
        os.remove(wav)
        rows.append({"id": base, "artist": artist, "title": os.path.splitext(os.path.basename(rel))[0], "source": rel,
                     "audio": f"{base}.m4a", **info})
        print(base, artist, rows[-1]["title"], info, flush=True)
    with open(os.path.join(out, "clips.json"), "w") as f:
        json.dump({"rule": f"{WINDOW_S:.0f} s from {START_FRAC:.0%} of each file, same piano, straight", "clips": rows}, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
