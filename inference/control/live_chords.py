"""Chords the player holds, read from the input snapshot (#1576).

docs/experiments/LIVE_CHORDS.md. The progression is otherwise fixed at launch
by ``--chords``; here the left hand names the chord for the next block.

Recognition is template matching on the pitch classes held below a split
point when the snapshot was taken. Names use the runtime's own vocabulary
(``inference.app.fallback.parse_chord``): triads map to the four-note chord
the chord guide would voice (C-E-G -> Cmaj7, A-C-E -> Am7, B-D-F -> Bm7b5).
"""
from __future__ import annotations

from dataclasses import dataclass, field

PC_NAMES = ("C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B")
# Order breaks ties: a major triad is Cmaj7 before C7, a minor triad m7 before m7b5.
TEMPLATES = (("maj7", (0, 4, 7, 11)), ("7", (0, 4, 7, 10)), ("m7", (0, 3, 7, 10)),
             ("m7b5", (0, 3, 6, 10)), ("dim", (0, 3, 6, 9)))


def recognize(pitches, *, min_pitch_classes: int = 3) -> str | None:
    """Chord symbol for the held ``pitches``, or None.

    Every held pitch class must belong to the chord, the root and third must
    be held, and at least ``min_pitch_classes`` distinct classes are needed.
    Among matches: most chord tones held, then root in the bass, then template order."""
    pitches = sorted(set(int(p) for p in pitches))
    pcs = {p % 12 for p in pitches}
    if len(pcs) < min_pitch_classes:
        return None
    bass = pitches[0] % 12
    best, best_key = None, None
    for order, (name, tmpl) in enumerate(TEMPLATES):
        for root in range(12):
            rel = {(p - root) % 12 for p in pcs}
            if not rel <= set(tmpl) or 0 not in rel or tmpl[1] not in rel:
                continue
            key = (len(rel), root == bass, -order)
            if best_key is None or key > best_key:
                best, best_key = PC_NAMES[root] + name, key
    return best


def held_notes(events, *, below: int):
    """(pitch, note_on received_ns) still held at the end of the snapshot, below ``below``."""
    held: dict[int, int] = {}
    for e in events:
        m = e.message
        if m.type == "note_on" and m.velocity > 0:
            if m.note < below:
                held[m.note] = e.received_ns
        elif m.type in ("note_off", "note_on"):
            held.pop(m.note, None)
    return held


@dataclass
class LiveChordTracker:
    """Per-block chord from the newest input snapshot.

    ``follow`` decides whether the block uses it; ``observe`` only records, so a
    control run sees the same recognition log. Releasing the keys keeps the last
    chord; before the first recognition the launch progression applies."""

    split: int = 60
    follow: bool = True
    current: str | None = None
    changes: list = field(default_factory=list)     # {"chord", "onset_ns", "block"}
    blocks: dict = field(default_factory=dict)      # block -> {"chord", "source", "recognized"}

    def update(self, events, block_index: int, default: str) -> str:
        held = held_notes(events, below=self.split)
        recognized = recognize(held) if held else None
        if recognized is not None and recognized != self.current:
            self.current = recognized
            self.changes.append({"chord": recognized, "onset_ns": max(held.values()), "block": block_index})
        if not self.follow or self.current is None:
            chord, source = default, "static"
        else:
            chord, source = self.current, ("live" if recognized is not None else "held")
        self.blocks[block_index] = {"chord": chord, "source": source, "recognized": recognized}
        return chord

    def chord_for(self, block_index: int, default: str) -> str:
        entry = self.blocks.get(block_index)
        return entry["chord"] if entry else default

    def report(self, block_start_ns, origin_ns: int, *, adopted=None, consumed_through=None,
               fallback_chord=None) -> dict:
        """``block_start_ns(b)``: session time of block b; times are ms from ``origin_ns``.

        Three stages are kept apart (Astra reviews, #1595):
          seen       the snapshot of block ``seen_in_block`` held the chord (recognition)
          generated  first block whose generation attempt used it (failed attempts included),
                     before the next change was seen
          adopted    first of those whose model block the scheduler took (``latency_ms``)
        ``adopted`` and ``consumed_through`` come from one ``adoption_snapshot()``.
        "Adopted" means handed to the scheduler (scheduled), not finished playing.
        A block at or below ``consumed_through`` that was not adopted had its
        fallback scheduled, with ``fallback_chord(b)``; a later block was never
        asked for, so its scheduled chord is unknown (None)."""
        def ms(b, onset):
            return round((block_start_ns(b) - onset) / 1e6, 1) if b is not None else None

        live = [b for b in sorted(self.blocks) if self.blocks[b]["source"] != "static"]
        changes = []
        for i, c in enumerate(self.changes):
            until = self.changes[i + 1]["block"] if i + 1 < len(self.changes) else None
            gen = [b for b in live if b >= c["block"] and (until is None or b < until)
                   and self.blocks[b]["chord"] == c["chord"]]
            first_gen = gen[0] if gen else None
            first_adopted = (next((b for b in gen if b in adopted), None) if adopted is not None else None)
            changes.append({"chord": c["chord"], "onset_ms": round((c["onset_ns"] - origin_ns) / 1e6, 1),
                            "seen_in_block": c["block"], "seen_latency_ms": ms(c["block"], c["onset_ns"]),
                            "first_generated_block": first_gen,
                            "generated_latency_ms": ms(first_gen, c["onset_ns"]),
                            "first_adopted_block": first_adopted,
                            "latency_ms": ms(first_adopted, c["onset_ns"])})
        blocks = []
        for b in sorted(self.blocks):
            entry = {"block": b, **self.blocks[b]}
            if adopted is not None:
                if b in adopted:
                    status, scheduled = "adopted", self.blocks[b]["chord"]
                elif consumed_through is not None and b <= consumed_through:
                    status, scheduled = "fallback", (fallback_chord(b) if fallback_chord else None)
                else:
                    status, scheduled = "not_consumed", None
                entry["schedule"] = status
                entry["scheduled_chord"] = scheduled
            blocks.append(entry)
        return {"follow": self.follow, "split": self.split, "adoption_known": adopted is not None,
                "changes": changes, "blocks": blocks}
