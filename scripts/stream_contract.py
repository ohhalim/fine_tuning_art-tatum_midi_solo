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
- commit floor (issue #1798): a commit_until(h) whose walk reached a token at or past h (a complete note with onset
  >= h, or a <T> into a segment starting after h) declares the music time before h final, so floor_ms becomes h.
  A token appended later may not put a note before the floor: inside the floor's segment the onset must be at least
  the floor rounded up to the 10 ms grid, and before that segment only <T> may follow a note. Without this, notes
  sampled after a rollback could land between the last committed note and h and reach the scheduler already late.
  A commit that ran out of tokens before h (the stream did not cover h) does not move the floor.
- request_full_reset() (adapter swap, context trim) drops provisional tokens and marks the KV invalid: the whole
  committed context must be prefilled again.

Scheduler (rules 4 and 7)
- committed notes become note-on / note-off events tagged with the current generation and a unique event id;
  each event is consumed once, whether by tick() or by a timer callback. note-offs belong to the scheduler, not to the
  model, so a rollback never touches them. Ownership per note is a logical-event contract: MIDI 1.0 note-off carries
  no note id. The MIDI stream (`midi_out`, one channel) therefore follows a retrigger policy (issue #1745): a
  note-on on a pitch that is already sounding first sends note-off for it and takes over the pitch; the scheduled
  note-off of the note that lost the pitch is not sent; stop() sends note-off for every owned pitch. Per pitch the
  MIDI events alternate on, off.
- stop(): block new events, bump the generation (every pending event of an older generation is void), send
  note-off for every sounding note. restart(): accept events again with the new generation. Any callback that
  arrives with an older generation is ignored, also after a restart.
"""
from __future__ import annotations

from dataclasses import dataclass, field

SEGMENT_MS = 5000
MIN_DUR_MS = 10           # the tokenizer's time step: a zero-length note gets this length (issue #1747)
ONSET_STEP_MS = 10        # onset grid of the Aria abs tokenizer


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


def playable(notes):
    """Notes the MIDI layer can send (issue #1747). Aria has dur 0 and velocity 0 tokens.
    - velocity 0 is dropped: a MIDI note-on with velocity 0 means note-off
    - dur 0 becomes MIN_DUR_MS; otherwise its note-off would sort before its note-on and leave the note sounding
    Returns (kept, dropped)."""
    kept, dropped = [], []
    for n in notes:
        if n.velocity <= 0:
            dropped.append(n)
        elif n.off_ms <= n.on_ms:
            kept.append(Note(n.note_id, n.pitch, n.velocity, n.on_ms, n.on_ms + MIN_DUR_MS))
        else:
            kept.append(n)
    return kept, dropped


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
        self.floor_ms = float("-inf")            # music time before it is final (last successful commit horizon)

    # model side -------------------------------------------------------------
    def onset_floor(self, st=None):
        """(reset_first, min_onset) the commit floor puts on the next tokens from grammar state st (default: the end).
        reset_first: the floor is in a later segment, so only <T> may follow a note. min_onset: the lowest onset
        allowed in the current segment (0 when the floor does not bind)."""
        st = self.grammar_state() if st is None else st
        if self.floor_ms == float("-inf"):
            return False, 0
        seg = int(self.floor_ms // SEGMENT_MS)
        rel = -(-(self.floor_ms - seg * SEGMENT_MS) // ONSET_STEP_MS) * ONSET_STEP_MS     # ceil to the onset grid
        if rel >= SEGMENT_MS:
            seg, rel = seg + 1, 0
        if st.segment < seg:
            return True, 0
        return False, int(rel) if st.segment == seg else 0

    def append_sampled(self, t):
        """A sampled token: in `tokens`, not yet in the KV."""
        st = self.grammar_state()
        nxt = st.advance(t)                      # raises if the token breaks the grammar
        reset_first, min_onset = self.onset_floor(st)
        if reset_first and _kind(t) == "piano":
            raise ValueError(f"note before the commit floor {self.floor_ms}: segment {st.segment} needs <T> first")
        if st.expect == "onset" and nxt.last_onset < min_onset:
            raise ValueError(f"onset {t} before the commit floor {self.floor_ms} (min onset {min_onset})")
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
        covered = False                                     # the walk met a token at or past the horizon
        while i < len(self.tokens):
            t = self.tokens[i]
            k = _kind(t)
            if k == "<T>":
                if (st.segment + 1) * SEGMENT_MS > horizon_ms:
                    covered = True
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
                    covered = True
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
        if covered:
            self.floor_ms = max(self.floor_ms, horizon_ms)
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
        self.midi_out = []     # (time_ms, "on"|"off", pitch, velocity): what goes to the synth on one channel
        self.midi_due = []     # the scheduled (event) time of each midi_out entry; midi_out holds the delivery time
        self.midi_note = []    # the note_id of each midi_out entry
        self.pitch_owner = {}  # pitch -> note_id that currently owns it in the MIDI stream
        self.rejected_late = []  # notes whose onset had already passed when they were handed in (issue #1798)

    INTAKE_GRACE_MS = 5.0      # a note handed in up to this late is still queued: the sender's 1 ms poll and the pipe

    def schedule(self, notes, now_ms=None):
        """Notes must be playable: 0 <= pitch <= 127, 1 <= velocity <= 127, off_ms > on_ms. A batch with any bad note
        is rejected as a whole before anything is queued (use playable() first).
        With now_ms, a note whose onset is more than INTAKE_GRACE_MS before now_ms is not queued at all (neither its
        on nor its off) and is recorded in rejected_late: it would only start late. This is the intake check; the
        delivery of queued events by tick() keeps its ordinary polling jitter. Returns the number of notes queued."""
        bad = [n for n in notes if not (0 <= n.pitch <= 127 and 1 <= n.velocity <= 127 and n.off_ms > n.on_ms)]
        if bad:
            raise ValueError(f"unplayable notes: {bad[:3]}")
        if not self.accepting:
            return 0
        if now_ms is not None:
            cut = now_ms - self.INTAKE_GRACE_MS
            self.rejected_late.extend((n, now_ms) for n in notes if n.on_ms < cut)
            notes = [n for n in notes if n.on_ms >= cut]
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
        self._to_midi(event, now_ms)
        return True

    def _to_midi(self, event, now_ms):
        p = event.pitch
        if event.kind == "on":
            if p in self.pitch_owner:                      # retrigger: end the sounding note on this pitch first
                self.midi_out.append((now_ms, "off", p, 0))
                self.midi_due.append(event.time_ms)
                self.midi_note.append(self.pitch_owner[p])
            self.midi_out.append((now_ms, "on", p, event.velocity))
            self.midi_due.append(event.time_ms)
            self.midi_note.append(event.note_id)
            self.pitch_owner[p] = event.note_id
        elif self.pitch_owner.get(p) == event.note_id:
            self.midi_out.append((now_ms, "off", p, 0))
            self.midi_due.append(event.time_ms)
            self.midi_note.append(event.note_id)
            del self.pitch_owner[p]
        # else: this note lost its pitch to a retrigger; its note-off is not sent

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
        for p in sorted(self.pitch_owner):
            self.midi_out.append((now_ms, "off", p, 0))
            self.midi_due.append(now_ms)                 # a stop flush is due when it happens
            self.midi_note.append(self.pitch_owner[p])
        self.pitch_owner.clear()

    def restart(self):
        self.accepting = True
