"""Real-time streaming contract as two small pure state machines (issue #1741, docs/experiments/ARIA_MLX_ROLLBACK.md).

No model, no MIDI port, virtual clock. Tokens are decoded Aria tokens: ("piano", pitch, velocity), ("onset", ms),
("dur", ms), "<T>" (5 s segment reset), "<D>", "<E>".

TokenStream (rules 1, 2, 3, 5, 6, 8 of the contract)
- positions, all end-exclusive token indexes:
  committed_end    tokens before it were handed to the scheduler
  kv_valid_length  KV holds tokens[:kv_valid_length]
  next_write_pos   where the next KV write goes (== kv_valid_length unless a full re-prefill is pending)
  A sampled token is appended to `tokens` before it is written to the KV, so len(tokens) can be kv_valid_length + 1.
- commit only complete notes (piano, onset, dur) whose absolute onset is before the horizon; a <T> is committed once
  the horizon reaches the next segment. Committed tokens are never changed by a rollback.
- rollback_to_committed() drops the provisional tokens, moves kv_valid_length / next_write_pos back to committed_end
  and restores the grammar state saved at committed_end.
- request_full_reset() (adapter swap, context trim) drops provisional tokens and marks the KV invalid: the whole
  committed context must be prefilled again.

Scheduler (rules 4 and 7)
- committed notes become note-on / note-off events tagged with the current generation and a unique event id;
  each event is consumed once, whether by tick() or by a timer callback. note-offs belong to the scheduler, not to the
  model, so a rollback never touches them. Ownership per note is a logical-event contract: MIDI 1.0 note-off carries
  no note id, so a channel/pitch retrigger policy is still needed before a real synth is connected.
- stop(): block new events, bump the generation (every pending event of an older generation is void), send
  note-off for every sounding note. restart(): accept events again with the new generation. Any callback that
  arrives with an older generation is ignored, also after a restart.
"""
from __future__ import annotations

from dataclasses import dataclass, field

SEGMENT_MS = 5000


def _kind(t):
    return t[0] if isinstance(t, tuple) else t


@dataclass(frozen=True)
class GrammarState:
    expect: str = "note"        # "note" | "onset" | "dur" | "end" (after <E>: nothing more is accepted)
    last_onset: int = 0         # last onset inside the current segment
    segment: int = 0            # number of <T> seen

    def advance(self, t):
        k = _kind(t)
        if self.expect == "end":
            raise ValueError(f"stream ended at <E>: {t}")
        if self.expect == "note":
            if k == "<T>":
                return GrammarState("note", 0, self.segment + 1)
            if k == "piano":
                return GrammarState("onset", self.last_onset, self.segment)
            if t in ("<D>", "<S>") or k == "prefix":    # sequence header tokens of a real Aria prompt
                return self
            if t == "<E>":
                return GrammarState("end", self.last_onset, self.segment)
            raise ValueError(f"not allowed between notes: {t}")
        if self.expect == "onset":
            if k != "onset" or t[1] < self.last_onset:
                raise ValueError(f"bad onset after a note start: {t}")
            return GrammarState("dur", t[1], self.segment)
        if k != "dur":
            raise ValueError(f"expected dur: {t}")
        return GrammarState("note", self.last_onset, self.segment)


@dataclass(frozen=True)
class Note:
    note_id: int
    pitch: int
    velocity: int
    on_ms: int
    off_ms: int


@dataclass
class TokenStream:
    prefix: list
    tokens: list = field(default_factory=list)
    committed_end: int = 0
    kv_valid_length: int = 0
    next_write_pos: int = 0
    needs_full_prefill: bool = False

    def __post_init__(self):
        st = GrammarState()
        for t in self.prefix:
            st = st.advance(t)
        self.tokens = list(self.prefix)
        self.committed_end = self.kv_valid_length = self.next_write_pos = len(self.prefix)
        self._state_at = {len(self.prefix): st}
        self._next_note_id = 0

    # model side -------------------------------------------------------------
    def append_sampled(self, t):
        """A sampled token: in `tokens`, not yet in the KV."""
        self.grammar_state().advance(t)          # raises if the token breaks the grammar
        self.tokens.append(t)

    def mark_written(self, upto):
        """The KV now holds tokens[:upto] (after a forward)."""
        if not self.next_write_pos <= upto <= len(self.tokens):
            raise ValueError(f"write up to {upto} from {self.next_write_pos} with {len(self.tokens)} tokens")
        self.kv_valid_length = self.next_write_pos = upto
        if upto >= self.committed_end:
            self.needs_full_prefill = False

    @property
    def committed_ended(self):
        """True once a committed <E> ends the stream; a provisional <E> is undone by a rollback."""
        return self._state_at[self.committed_end].expect == "end"

    def grammar_state(self, upto=None):
        upto = len(self.tokens) if upto is None else upto
        base = max(p for p in self._state_at if p <= upto)
        st = self._state_at[base]
        for t in self.tokens[base:upto]:
            st = st.advance(t)
        return st

    # runtime side -----------------------------------------------------------
    def commit_until(self, horizon_ms):
        """Commit complete notes with onset < horizon_ms (and <T> once the horizon reaches the next segment)."""
        out, i, st = [], self.committed_end, self._state_at[self.committed_end]
        while i < len(self.tokens):
            t = self.tokens[i]
            k = _kind(t)
            if k == "<T>":
                if (st.segment + 1) * SEGMENT_MS > horizon_ms:
                    break
                st, i = st.advance(t), i + 1
            elif k in ("<D>", "<E>"):
                st, i = st.advance(t), i + 1
            elif k == "piano":
                if i + 2 >= len(self.tokens):
                    break                                   # incomplete note: never split it
                on_t, dur_t = self.tokens[i + 1], self.tokens[i + 2]
                on_ms = st.segment * SEGMENT_MS + on_t[1]
                if on_ms >= horizon_ms:
                    break
                out.append(Note(self._next_note_id, t[1], t[2], on_ms, on_ms + dur_t[1]))
                self._next_note_id += 1
                for x in (t, on_t, dur_t):
                    st = st.advance(x)
                i += 3
            else:
                raise ValueError(f"commit walk hit {t} at {i}")
            self.committed_end = i
            self._state_at[i] = st
        return out

    def rollback_to_committed(self):
        """Drop provisional tokens; the KV and the grammar restart at committed_end."""
        dropped = len(self.tokens) - self.committed_end
        del self.tokens[self.committed_end:]
        self.kv_valid_length = min(self.kv_valid_length, self.committed_end)
        if not self.needs_full_prefill:
            self.next_write_pos = self.kv_valid_length
        return dropped

    def request_full_reset(self):
        """Adapter swap or context trim: drop provisional tokens and re-prefill the whole committed context."""
        self.rollback_to_committed()
        self.kv_valid_length = self.next_write_pos = 0
        self.needs_full_prefill = True


@dataclass(frozen=True)
class Event:
    time_ms: int
    kind: str                  # "on" | "off"
    note_id: int
    pitch: int
    velocity: int
    generation: int
    event_id: int


class Scheduler:
    def __init__(self):
        self.generation = 0
        self.accepting = True
        self.pending = []      # Events not yet due
        self.sounding = {}     # note_id -> pitch
        self.sent = []         # (time_ms, kind, note_id, pitch) actually sent to the synth
        self.consumed = set()  # event_ids already delivered: each event is consumed once, by tick or a callback
        self.last_tick_ms = float("-inf")
        self._next_event_id = 0

    def schedule(self, notes):
        if not self.accepting:
            return 0
        for n in notes:
            for time_ms, kind, vel in ((n.on_ms, "on", n.velocity), (n.off_ms, "off", 0)):
                self.pending.append(Event(time_ms, kind, n.note_id, n.pitch, vel, self.generation, self._next_event_id))
                self._next_event_id += 1
        self.pending.sort(key=lambda e: (e.time_ms, e.kind == "on"))   # offs before ons at the same time
        return len(notes)

    def deliver(self, event, now_ms):
        """A timer callback. Events of an older generation are ignored (late callbacks after stop/restart)."""
        if event.generation != self.generation or event.event_id in self.consumed:
            return False
        self.consumed.add(event.event_id)
        self.pending = [e for e in self.pending if e.event_id != event.event_id]
        if event.kind == "on":
            self.sounding[event.note_id] = event.pitch
        elif event.note_id in self.sounding:
            del self.sounding[event.note_id]
        else:
            return False       # the note was already turned off (e.g. by stop)
        self.sent.append((now_ms, event.kind, event.note_id, event.pitch))
        return True

    def tick(self, now_ms):
        if now_ms < self.last_tick_ms:
            raise ValueError(f"clock went back: tick({now_ms}) after tick({self.last_tick_ms})")
        self.last_tick_ms = now_ms
        due = [e for e in self.pending if e.time_ms <= now_ms]
        return [e for e in due if self.deliver(e, now_ms)]

    def stop(self, now_ms):
        self.accepting = False
        self.generation += 1
        self.pending = []
        for note_id, pitch in sorted(self.sounding.items()):
            self.sent.append((now_ms, "off", note_id, pitch))
        self.sounding.clear()

    def restart(self):
        self.accepting = True
