#!/usr/bin/env python3
"""Play the runtime from the computer keyboard, no extra packages (#1581).

docs/phase1/TYPING_KEYBOARD_FL.md. Opens a virtual MIDI output (default
``TypingKeyboard``) that the runtime reads with ``--input-port TypingKeyboard``.

Layout (tracker style):

  chords   Z S X D C V G B H N J M    = C3 C#3 D3 D#3 E3 F3 F#3 G3 G#3 A3 A#3 B3
  melody   Q 2 W 3 E R 5 T 6 Y 7 U I 9 O 0 P  = C4 ... E5

A terminal sees key presses but not releases, so:
  * chord keys pressed within ``--chord-gap`` seconds of each other form one
    chord, held until the next chord or Space (all chord notes stay below 60,
    the runtime's default --chord-split, so --live-chords can read them);
  * melody keys play a note of ``--melody-length`` seconds.
  [ / ] shift the melody octave, Space releases the chord, Esc or Ctrl-C quits.
"""
from __future__ import annotations

import argparse
import select
import sys
import termios
import time
import tty

CHORD_KEYS = "zsxdcvgbhnjm"
MELODY_KEYS = "q2w3er5t6y7ui9o0p"
CHORD_BASE = 48
MELODY_BASE = 60


class TypingKeyboard:
    """Key presses -> (time, message type, pitch) events, clock supplied by the caller."""

    def __init__(self, *, chord_gap: float = 0.15, melody_length: float = 0.25,
                 velocity: int = 80) -> None:
        self.chord_gap = chord_gap
        self.melody_length = melody_length
        self.velocity = velocity
        self.melody_shift = 0
        self.chord: list[int] = []
        self._last_chord_key = None
        self._pending_off: list[tuple[float, int]] = []      # (due, pitch)

    def press(self, key: str, now: float) -> list[tuple[str, int]]:
        key = key.lower()
        out: list[tuple[str, int]] = []
        if key in CHORD_KEYS:
            pitch = CHORD_BASE + CHORD_KEYS.index(key)
            new_gesture = self._last_chord_key is None or now - self._last_chord_key > self.chord_gap
            self._last_chord_key = now
            if new_gesture:
                out += [("note_off", p) for p in self.chord]
                self.chord = []
            if pitch not in self.chord:
                self.chord.append(pitch)
                out.append(("note_on", pitch))
        elif key in MELODY_KEYS:
            pitch = MELODY_BASE + self.melody_shift + MELODY_KEYS.index(key)
            if 0 <= pitch <= 127:
                # A repeat of a still-sounding note restarts it.
                if any(p == pitch for _, p in self._pending_off):
                    self._pending_off = [(d, p) for d, p in self._pending_off if p != pitch]
                    out.append(("note_off", pitch))
                out.append(("note_on", pitch))
                self._pending_off.append((now + self.melody_length, pitch))
        elif key == " ":
            out += [("note_off", p) for p in self.chord]
            self.chord = []
            self._last_chord_key = None
        elif key == "[":
            self.melody_shift = max(-24, self.melody_shift - 12)
        elif key == "]":
            self.melody_shift = min(24, self.melody_shift + 12)
        return out

    def due(self, now: float) -> list[tuple[str, int]]:
        ready = [p for d, p in self._pending_off if d <= now]
        self._pending_off = [(d, p) for d, p in self._pending_off if d > now]
        return [("note_off", p) for p in ready]

    def release_all(self) -> list[tuple[str, int]]:
        out = [("note_off", p) for p in self.chord] + [("note_off", p) for _, p in self._pending_off]
        self.chord, self._pending_off = [], []
        return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port-name", default="TypingKeyboard")
    ap.add_argument("--chord-gap", type=float, default=0.15)
    ap.add_argument("--melody-length", type=float, default=0.25)
    ap.add_argument("--velocity", type=int, default=80)
    args = ap.parse_args(argv)
    import mido

    kb = TypingKeyboard(chord_gap=args.chord_gap, melody_length=args.melody_length, velocity=args.velocity)
    if not sys.stdin.isatty():
        ap.error("run this in a terminal (it reads single key presses)")
    with mido.open_output(args.port_name, virtual=True) as port:
        def send(events):
            for kind, pitch in events:
                port.send(mido.Message(kind, note=pitch, velocity=args.velocity if kind == "note_on" else 0))

        print(f"virtual MIDI output '{args.port_name}' open. chords: Z-row (C3-B3), melody: Q-row (C4-). "
              "Space releases the chord, [ ] octave, Esc quits.", flush=True)
        fd = sys.stdin.fileno()
        saved = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while True:
                ready, _, _ = select.select([sys.stdin], [], [], 0.01)
                now = time.monotonic()
                send(kb.due(now))
                if ready:
                    ch = sys.stdin.read(1)
                    if ch in ("\x1b", "\x03"):
                        break
                    send(kb.press(ch, now))
        except KeyboardInterrupt:
            pass
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, saved)
            send(kb.release_all())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
