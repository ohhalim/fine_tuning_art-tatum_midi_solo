#!/usr/bin/env python3
"""Play bars continuously, generating each bar while the previous one plays.

``run_jazz_mvp.py`` generates every bar first and plays afterwards. This script
keeps that one intact and takes the other path: a background producer fills one
bar ahead while the scheduler dispatches the current bar. A bar that is not
ready in time is played from a prebuilt fallback and the late model result is
discarded.

Scope is deliberately narrow for now: fixed 4/4, one BPM for the whole run,
8-16 bars. Nothing here claims real-time co-performance; see the limits printed
at the end of a run.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from inference.app.fallback import build_fallback_midi
from inference.app.schemas import GenerationRequest
from inference.realtime.blocks import build_scheduled_midi_block
from inference.realtime.continuous import (
    BarBlockProducer,
    MidiInputSnapshotBuffer,
    input_events_to_notes,
    summarize_production,
)
from inference.realtime.scheduler import (
    DEADLINE_POLICY_RECORD_AND_CONTINUE,
    MonotonicBarClock,
    OneBarMidiScheduler,
)
from scripts.run_jazz_mvp import fit_window
from scripts.run_resident_model_probe import validate_generated_token_block

BEATS_PER_BAR = 4


def build_fallback_blocks(*, clock, bars, bpm, chords, seed, duration):
    """Prebuild every fallback up front so ``get`` never has to build one."""
    blocks = {}
    for index in range(bars):
        request = GenerationRequest(
            bpm=bpm, bars=1, chord_progression=[chords[index % len(chords)]], seed=seed + index
        )
        request.validate()
        blocks[index] = build_scheduled_midi_block(
            midi=fit_window(build_fallback_midi(request), duration),
            clock=clock, bar_index=index, block_id=f"fallback-{index}",
            source_context_id="prebuilt-fallback", context_version=0,
            adapter="fallback", fallback_used=True,
        )
    return blocks


def build_chord_live_primer(input_events, chord, *, bpm, base_primer,
                            primer_max_tokens=48, now_ns=None):
    """Opt-in primer that states the bar's harmony under the recent playing.

    Returns ``(primer, used_input, used_chord)``.

    Two things differ from ``build_live_primer`` and both are deliberate:

    * The harmony is carried as notes, because the checkpoint never saw a chord
      token. Counting the tokenized training sets shows zero control tokens in
      either the pretrain or the adaptation data, so a chord symbol in the
      prefix would be an embedding row with no gradient behind it.
    * No control prefix is added, for the same reason: ROLE_LEAD, TEMPO_*, BAR
      and COND_SEP are untrained here too. The default path still adds them;
      this one does not, which is a confound to keep in mind when comparing the
      two rather than a silent improvement.

    Nothing is filtered. Non-chord tones stay reachable.
    """
    import torch

    from inference.control.chord_primer import build_chord_primer

    notes = input_events_to_notes(input_events, end_ns=now_ns)
    tokens, used_chord = build_chord_primer(
        [chord] if chord else [], bpm=bpm, bars=1,
        melodic_notes=notes, primer_max_tokens=primer_max_tokens,
    )
    if not tokens:
        return base_primer, False, False
    return torch.tensor(tokens, dtype=torch.long), bool(notes), used_chord


def build_live_primer(input_events, *, base_primer, control_format, role, tempo_bpm,
                      primer_max_tokens=32, now_ns=None):
    """Condition the next bar on what the player just played.

    Returns ``(primer, used_input)``. Falls back to ``base_primer`` whenever the
    window holds nothing encodable, so a silent player still gets a bar.

    Note this inherits the D3 constraint: a MIDI primer carries texture, not
    just pitch, so the generated bar tracks the input's texture as well as its
    notes. That is wanted for co-performance but it is not chord conditioning.
    """
    import torch

    from scripts.control_tokens import build_control_primer, control_prefix_tokens
    from scripts.generate import encode_notes_simple, truncate_tokens_preserving_velocity

    notes = input_events_to_notes(input_events, end_ns=now_ns)
    if not notes:
        return base_primer, False
    tokens = encode_notes_simple(notes)
    if not tokens:
        return base_primer, False
    # Truncate here, preserving velocity state, instead of letting
    # build_control_primer tail-slice it away. Steady playing emits exactly one
    # velocity token, at the very start, so a plain tail slice loses it.
    # Derive the room build_control_primer will leave. Hardcoding it would fail
    # silently the day the control prefix changes length: the second truncation
    # would drop the velocity token again.
    prefix_budget = len(control_prefix_tokens(role=role, tempo_bpm=tempo_bpm)) + 1
    tokens = truncate_tokens_preserving_velocity(
        tokens, max(1, primer_max_tokens - prefix_budget)
    )
    primer = build_control_primer(
        tokens, role=role, tempo_bpm=tempo_bpm, append_sep_token=True,
        primer_max_tokens=primer_max_tokens,
    )
    if not primer:
        return base_primer, False
    return torch.tensor(primer, dtype=torch.long), True


def make_sub_block_builder(*, clock, duration, generate_sub, blocks_per_bar):
    """Producer callback that fills one bar as several sub-blocks.

    The scheduler is untouched: it still consumes one block per bar. The split
    happens inside this callback, which restates the harmony at each sub-block.
    Measured to be the lever that actually moves bar-chord alignment - a bar
    primed only at its downbeat gives the model no way to see the chord again
    partway through (docs/experiments/CHORD_PRIMER_AB.md §23).

    A sub-block that fails validation is skipped rather than failing the bar:
    losing half a bar is better than losing all of it. The bar fails only if
    every sub-block does.
    """
    import pretty_midi

    sub_duration = duration / blocks_per_bar

    def build(bar_index, input_events):
        notes, valid_subs = [], 0
        for sub in range(blocks_per_bar):
            tokens = generate_sub(bar_index, sub, input_events, sub_duration)
            valid = validate_generated_token_block(
                tokens, lookahead_ms=sub_duration * 1000, allow_rest_bar=True
            )
            if not valid["valid"]:
                continue
            valid_subs += 1
            from midi_processor.processor import decode_midi

            sub_midi = fit_window(decode_midi(tokens), sub_duration)
            offset = sub * sub_duration
            for instrument in sub_midi.instruments:
                for note in instrument.notes:
                    notes.append(pretty_midi.Note(
                        note.velocity, note.pitch,
                        note.start + offset, min(note.end + offset, duration)))
        if not valid_subs:
            raise ValueError(f"all {blocks_per_bar} sub-blocks failed validation")

        merged = pretty_midi.PrettyMIDI()
        instrument = pretty_midi.Instrument(program=0, name="lead")
        instrument.notes = [n for n in notes if n.end > n.start]
        merged.instruments = [instrument]
        return build_scheduled_midi_block(
            midi=fit_window(merged, duration),
            clock=clock, bar_index=bar_index, block_id=str(bar_index),
            source_context_id="continuous_sub_blocks", context_version=0,
            adapter="model", fallback_used=False,
            allow_empty=not instrument.notes,
        )

    return build


def make_block_builder(*, clock, duration, generate):
    """Return the producer callback. Runs on the producer thread, never the scheduler's."""

    def build(bar_index, input_events):
        # `chords` is not passed to the model: generation is not chord
        # conditioned yet. It only shapes the prebuilt fallbacks.
        tokens, _metadata = generate(bar_index, input_events)
        # A whole-bar rest is playable here: the scheduler simply waits it out.
        valid = validate_generated_token_block(
            tokens, lookahead_ms=duration * 1000, allow_rest_bar=True
        )
        if not valid["valid"]:
            raise ValueError(f"invalid model block: {valid}")
        from midi_processor.processor import decode_midi

        return build_scheduled_midi_block(
            midi=fit_window(decode_midi(tokens), duration),
            clock=clock, bar_index=bar_index, block_id=str(bar_index),
            source_context_id="continuous", context_version=0,
            adapter="model", fallback_used=False,
            allow_empty=bool(valid["rest_bar_accepted"]),
        )

    return build


def block_metrics(block, *, chord: str, adapter: str | None, input_events) -> dict:
    """Per-block observations taken on the producer thread right after a model
    block is built (docs/experiments/LIVE_METRICS.md). Not a quality measure."""
    from inference.app.fallback import parse_chord

    pitches = [e.message.note for e in block.events
               if e.message.type == "note_on" and e.message.velocity > 0]
    root, intervals = parse_chord(chord)
    pcs = {(root + i) % 12 for i in intervals}
    played_in = [e.message.note for e in input_events
                 if e.message.type == "note_on" and e.message.velocity > 0]
    return {
        "block": block.bar_index, "adapter": adapter, "chord": chord, "notes": len(pitches),
        "pitch_mean": round(sum(pitches) / len(pitches), 2) if pitches else None,
        "pitch_min": min(pitches) if pitches else None,
        "pitch_max": max(pitches) if pitches else None,
        "chord_tone_ratio": (round(sum((p % 12) in pcs for p in pitches) / len(pitches), 4)
                             if pitches else None),
        "input_notes": len(played_in),
        "input_pitch_mean": round(sum(played_in) / len(played_in), 2) if played_in else None,
    }


def block_voicings(block, *, window_s: float = 0.05):
    """Pitch-class sets of near-simultaneous onsets in a block, grouped exactly as
    D1's ``diversity_metrics.group_voicings`` (50 ms window)."""
    from scripts.diversity_metrics import group_voicings

    notes = [((e.target_ns - block.target_start_ns) / 1e9, e.message.note, 0.0, e.message.velocity)
             for e in block.events if e.message.type == "note_on" and e.message.velocity > 0]
    return group_voicings(notes, window_sec=window_s)


class VoicingPool:
    """Running unique voicings per adapter over a session (in-loop diversity)."""

    def __init__(self) -> None:
        self._pool: dict[str, list] = {}

    def add(self, adapter: str | None, voicings) -> dict:
        pool = self._pool.setdefault(adapter or "-", [])
        pool.extend(voicings)
        unique = len(set(pool))
        return {"voicings": len(voicings), "unique_voicings_so_far": unique,
                "distinct_voicing_ratio_so_far": round(unique / len(pool), 4) if pool else None}


class BlockMetricsRecorder:
    """Generation metrics for every model block; played metrics only for adopted ones.

    A block counts as adopted only when the producer's ``get`` actually returned
    the model block to the scheduler (``producer.adopted_blocks``). Ready-but-late
    blocks the scheduler replaced with a fallback, and blocks never fetched, stay
    in ``generation`` only and never reach the voicing pool (Astra M2). Whether
    an adopted block's events were all sent is a separate status, filled in from
    the scheduler records at the end (docs/experiments/LIVE_METRICS.md).
    """

    def __init__(self, *, producer_ref, chord_for_block, adapter_for_block, on_adopted=None):
        self._producer_ref = producer_ref          # () -> BarBlockProducer (set after creation)
        self._chord_for_block = chord_for_block
        self._adapter_for_block = adapter_for_block
        self._on_adopted = on_adopted              # callback(metrics) for live printing
        self.generation: dict[int, dict] = {}
        self.played: dict[int, dict] = {}
        self._pending: list[int] = []
        self._pool = VoicingPool()
        self._voicings: dict[int, list] = {}

    def on_generated(self, block, input_events) -> None:
        """Producer thread, right after a model block is built (before it is ready)."""
        b = block.bar_index
        m = block_metrics(block, chord=self._chord_for_block(b), adapter=self._adapter_for_block(b),
                          input_events=input_events)
        m["events"] = len(block.events)
        self.generation[b] = m
        self._voicings[b] = block_voicings(block)
        self._pending.append(b)
        self._settle()

    def _settle(self, final: bool = False) -> None:
        producer = self._producer_ref()
        if producer is None:
            return
        adopted, decided_up_to = producer.adoption_snapshot()
        still = []
        for b in sorted(self._pending):
            if b in adopted:
                m = dict(self.generation[b])
                m.update(self._pool.add(m["adapter"], self._voicings[b]))
                self.played[b] = m
                if self._on_adopted is not None:
                    self._on_adopted(m)
            elif not final and b > decided_up_to:
                still.append(b)                   # not asked for yet: undecided
        self._pending = still

    def finish(self, records) -> None:
        """After the session: settle everything and add the send status."""
        self._settle(final=True)
        sent: dict[int, int] = {}
        for r in records:
            sent[r.bar_index] = sent.get(r.bar_index, 0) + 1
        for b, m in self.generation.items():
            m["adopted"] = b in self.played
        for b, m in self.played.items():
            n = sent.get(b, 0)
            m["events_sent"] = n
            m["send_status"] = ("complete" if n == m["events"] else "partial" if n else "none")


def with_block_metrics(factory, *, record):
    """Wrap a builder factory so every model block also reports ``block_metrics``."""
    def make(*, clock, duration):
        build = factory(clock=clock, duration=duration)

        def build_and_measure(block_index, input_events):
            block = build(block_index, input_events)
            record(block, input_events)
            return block
        return build_and_measure
    return make


def _make_builder(*, clock, duration, generate, sub_builder):
    """Pick the producer callback: sub-blocks, one bar, or generation disabled."""
    if sub_builder is not None:
        return sub_builder(clock=clock, duration=duration)
    if generate is None:
        return None
    return make_block_builder(clock=clock, duration=duration, generate=generate)


def run_session(*, port, bars, bpm, chords, seed, generate, input_buffer=None,
                start_delay_seconds=2.5, spin_window_ms=5.0,
                clock=None, clock_ns=None, wait_until=None,
                deadline_policy=DEADLINE_POLICY_RECORD_AND_CONTINUE,
                sub_builder=None, fetch_margin_ms=None, start_budget_bars=None,
                beats_per_block=BEATS_PER_BAR, adaptive_start_safety=None, on_producer=None):
    """Play ``bars`` bars, producing one bar ahead.

    ``clock``/``clock_ns``/``wait_until`` exist so tests can drive the run off a
    fake clock instead of waiting out real bars.
    """
    duration = 60.0 / bpm * beats_per_block
    if clock is None:
        clock = MonotonicBarClock(
            bpm=bpm, beats_per_bar=beats_per_block,
            start_ns=time.perf_counter_ns() + round(start_delay_seconds * 1e9),
        )
    fallbacks = build_fallback_blocks(
        clock=clock, bars=bars, bpm=bpm, chords=chords, seed=seed, duration=duration
    )
    def start_not_before(block_index):
        """Latest start that still leaves the budget before the block's fetch.

        With ``adaptive_start_safety`` the budget grows to safety x the slowest
        of the last four generations whenever that exceeds the fixed fraction,
        so a loaded machine starts earlier instead of missing the fetch
        (docs/experiments/ADAPTIVE_BUDGET.md)."""
        budget = start_budget_bars * clock.bar_duration_ns
        if adaptive_start_safety:
            recent = [r.completed_ns - r.requested_ns for r in producer.records
                      if r.requested_ns is not None and r.completed_ns is not None][-4:]
            if recent:
                budget = max(budget, adaptive_start_safety * max(recent))
        budget = min(budget, clock.bar_duration_ns)
        return clock.bar_start_ns(block_index) - round(fetch_margin_ms * 1e6) - round(budget)

    producer = BarBlockProducer(
        # Lead of 2, not 1. The scheduler asks for bar 1 at bar 0's downbeat,
        # and the watermark only advances once it starts, so a lead of 1 leaves
        # bar 1 unable to begin until playback is already underway.
        bar_count=bars, fallback_blocks=fallbacks, clock=clock, input_buffer=input_buffer,
        max_lead_bars=2,
        # With a late fetch, steady state builds only the next bar, from the
        # newest input (docs/experiments/GENERATION_LEAD.md).
        steady_lead_bars=1 if fetch_margin_ms is not None else None,
        # Optional: hold each steady-state bar back so it starts only
        # `start_budget_bars` before its fetch (docs/experiments/START_BUDGET.md).
        start_not_before_ns=(
            None if start_budget_bars is None or fetch_margin_ms is None
            else lambda b: start_not_before(b)),
        build_block=_make_builder(clock=clock, duration=duration, generate=generate,
                                  sub_builder=sub_builder),
    )
    # A single OS hiccup must not end a performance. The scheduler probe uses
    # abort_on_first_miss to make timing failures loud; a live instrument wants
    # the miss recorded and the music continued.
    if on_producer is not None:
        on_producer(producer)
    scheduler_kwargs = {"sink": port, "clock": clock, "spin_window_ms": spin_window_ms,
                        "deadline_policy": deadline_policy, "fetch_margin_ms": fetch_margin_ms}
    if clock_ns is not None:
        scheduler_kwargs["clock_ns"] = clock_ns
    if wait_until is not None:
        scheduler_kwargs["wait_until"] = wait_until
    scheduler = OneBarMidiScheduler(**scheduler_kwargs)
    try:
        producer.start()
        # Cold start needs room for two bars, not one. The producer is a single
        # thread, and the scheduler asks for bar 1 at bar 0's downbeat, so bars
        # 0 and 1 must both be generated inside start_delay_seconds. Measured:
        # a run whose first two bars summed to 1036ms missed with a 1s delay.
        deadline = time.monotonic() + max(0.0, start_delay_seconds)
        for warmup_bar in (0, 1):
            if warmup_bar >= bars:
                break
            producer.wait_for_bar(warmup_bar, timeout=max(0.0, deadline - time.monotonic()))
        result = scheduler.run(blocks=producer, expected_bar_count=bars)
    finally:
        # Reset on completion, abort, exception and Ctrl-C alike.
        producer.close()
        try:
            port.reset()
        finally:
            port.panic()
    return result, producer


def played_bar_notes(result, clock, bars: int) -> list[dict]:
    """Dispatched notes per scheduler bar, timed from that bar's grid start.

    ``played.mid`` starts at the first note, so it cannot say where a bar
    begins; this keeps the bar alignment for per-bar analysis.
    """
    per_bar: list[list[list[float]]] = [[] for _ in range(bars)]
    open_notes: dict[int, tuple[int, int]] = {}
    for r in result.records:
        m = r.message
        if m.type == "note_on" and m.velocity > 0:
            open_notes[m.note] = (r.bar_index, r.target_ns)
        elif m.type in ("note_off", "note_on") and m.note in open_notes:
            bar, on_ns = open_notes.pop(m.note)
            if 0 <= bar < bars:
                start = (on_ns - clock.bar_start_ns(bar)) / 1e9
                end = (r.target_ns - clock.bar_start_ns(bar)) / 1e9
                per_bar[bar].append([int(m.note), round(start, 4), round(end, 4)])
    return [{"bar": b, "notes": sorted(n, key=lambda x: (x[1], x[0]))} for b, n in enumerate(per_bar)]


def merge_half_bars(half_bars: list[dict], *, half_seconds: float) -> list[dict]:
    """Join half-bar blocks back into bars, second half offset by half a bar."""
    out = []
    for b in range(len(half_bars) // 2):
        first, second = half_bars[2 * b]["notes"], half_bars[2 * b + 1]["notes"]
        notes = first + [[p, round(s + half_seconds, 4), round(e + half_seconds, 4)]
                         for p, s, e in second]
        out.append({"bar": b, "notes": sorted(notes, key=lambda x: (x[1], x[0]))})
    return out


def carry_tokens(previous, n: int) -> list[int]:
    """Last ``n`` tokens of the previous block with its velocity state kept (0 = off)."""
    if n <= 0 or not previous:
        return []
    from scripts.generate import truncate_tokens_preserving_velocity

    return truncate_tokens_preserving_velocity(previous, n)


def block_to_notes(block):
    """Notes of a scheduled block, timed from its start (unclosed notes end at the block end)."""
    import pretty_midi

    start_ns = block.target_start_ns
    end_s = ((block.target_end_ns - start_ns) / 1e9) if block.target_end_ns is not None else None
    open_notes, notes = {}, []
    for e in block.events:
        m = e.message
        t = (e.target_ns - start_ns) / 1e9
        if m.type == "note_on" and m.velocity > 0:
            open_notes[m.note] = (t, m.velocity)
        elif m.type in ("note_off", "note_on") and m.note in open_notes:
            t0, vel = open_notes.pop(m.note)
            # Keep the played duration, however short (Astra review): callers
            # quantise and clamp to the block, and decide what a zero length means.
            notes.append(pretty_midi.Note(velocity=vel, pitch=m.note, start=t0, end=max(t, t0)))
    for pitch, (t0, vel) in open_notes.items():
        notes.append(pretty_midi.Note(velocity=vel, pitch=pitch, start=t0,
                                      end=max(end_s if end_s is not None else t0, t0)))
    return sorted(notes, key=lambda n: (n.start, n.pitch))


class PlayedHistory:
    """Token history of the adopted/scheduled blocks, one block at a time.

    Astra reviews of #1572: a block enters only once the scheduler has asked for
    it: the model block if get() adopted it, otherwise the fallback scheduled in
    its place. This is what was handed to the scheduler, not a record of sends
    that finished: a block's future note-offs are included, and a cancelled run
    can stop mid-block.

    Timing: every note time is first placed on the absolute 10 ms session grid,
    and tokens encode differences of those grid times, so per-event rounding
    cannot accumulate (a note held to each block end used to add 2.5 ms per
    block). Each settled block ends exactly on its cumulative boundary
    (docs/experiments/CONTEXT_HISTORY.md).
    """

    def __init__(self, block_s: float, cap: int = 2048, horizon_s: float = 30.0):
        self.block_s = block_s
        self.cap = cap
        self.horizon_steps = int(round(horizon_s * 100))
        self.tokens: list[int] = []
        self.next_block = 0
        self.boundary_step = 0            # end of the last settled block, in 10 ms steps
        self._notes: list[tuple[int, int, int, int]] = []    # (start, end, pitch, velocity) steps
        self._last_end: dict[int, int] = {}                  # latest end tick per pitch

    def settle(self, *, adopted, watermark: int, generated: dict, fallback_for) -> None:
        if watermark < self.next_block:
            return
        for i in range(self.next_block, watermark + 1):
            block = generated.get(i) if i in adopted else fallback_for(i)
            base_s = i * self.block_s
            lo = int(round(i * self.block_s * 100))            # this block's quantised edges
            hi = int(round((i + 1) * self.block_s * 100))
            for n in (block_to_notes(block) if block is not None else []):
                start = min(max(int(round((base_s + n.start) * 100)), lo), hi)
                end = min(max(int(round((base_s + n.end) * 100)), lo), hi)
                if end <= start:
                    # Zero length on the 10 ms grid (e.g. a 2.5 ms note at the block
                    # end). Policy: move the onset one tick earlier when that stays
                    # inside the block and does not overlap the same pitch; else drop.
                    # Rounding keeps order, so only this move can create an overlap,
                    # and notes arrive in onset order: the pitch's last end suffices.
                    start, end = end - 1, end
                    if start < lo or start < self._last_end.get(n.pitch, -1):
                        continue
                self._notes.append((start, end, n.pitch, n.velocity))
                self._last_end[n.pitch] = max(self._last_end.get(n.pitch, -1), end)
            self.boundary_step = hi
            self.next_block = i + 1
        self._rebuild()

    def _rebuild(self) -> None:
        import pretty_midi
        from scripts.generate import encode_notes_simple, truncate_tokens_preserving_velocity

        low = self.boundary_step - self.horizon_steps
        self._notes = [x for x in self._notes if x[1] > low]
        if not self._notes:
            self.tokens = []
            return
        origin = min(x[0] for x in self._notes)
        notes = [pretty_midi.Note(velocity=v, pitch=p, start=(s - origin) / 100, end=(e - origin) / 100)
                 for s, e, p, v in self._notes]
        toks = encode_notes_simple(sorted(notes, key=lambda n: (n.start, n.pitch)))
        remaining = self.boundary_step - max(max(x[1] for x in self._notes), max(x[0] for x in self._notes))
        while remaining > 0:
            step = min(remaining, 100)
            toks.append(255 + step)
            remaining -= step
        self.tokens = truncate_tokens_preserving_velocity(toks, self.cap)

    def span_steps(self) -> int:
        """Time covered by the (uncapped) note window, first note to the last boundary."""
        if not self._notes:
            return 0
        return self.boundary_step - min(x[0] for x in self._notes)


def summarize_lateness_ms(lateness_ns) -> dict:
    """Distribution of scheduler dispatch lateness over every attempt, not just the tail."""
    values = sorted(ns / 1e6 for ns in lateness_ns)
    if not values:
        return {"count": 0}

    def pct(q):
        return values[min(len(values) - 1, int(round(q * (len(values) - 1))))]

    return {"count": len(values), "p50": pct(0.50), "p95": pct(0.95), "p99": pct(0.99),
            "maximum": values[-1],
            "over_5ms": sum(v > 5.0 for v in values), "over_10ms": sum(v > 10.0 for v in values)}


def summarize_deadline_misses(misses, producer_records) -> list[dict]:
    """Each scheduler miss, and whether the producer was generating at that moment.

    Both clocks are ``time.perf_counter_ns``. ``producer_busy`` is true when the
    late window [target, dispatch start] overlaps any bar's generation interval.
    """
    intervals = [(r.requested_ns, r.completed_ns) for r in producer_records
                 if getattr(r, "requested_ns", None) is not None
                 and getattr(r, "completed_ns", None) is not None]
    out = []
    for m in misses:
        busy = any(start <= m.dispatch_started_ns and end >= m.target_ns
                   for start, end in intervals)
        out.append({"bar_index": m.bar_index, "sequence_index": m.sequence_index,
                    "lateness_ms": m.lateness_ns / 1e6, "message_type": m.message_type,
                    "is_bar_start": m.is_bar_start, "is_catch_up": m.is_catch_up,
                    "target_ns": m.target_ns, "producer_busy": busy})
    return out


def summarize_capture(result, captured, *, drain_completed):
    """Compare what the scheduler intended to send against what a separate
    CoreMIDI input actually observed.

    Kept apart from every producer metric: this measures the output path only
    (scheduled instant -> capture), never model latency.
    """
    # The safe reset/panic sent after the run emits control_change traffic that
    # the scheduler never recorded. Compare note events only, or every one of
    # those messages reads as a spurious duplicate.
    sent = [r for r in result.records if r.message.type in ("note_on", "note_off")]
    notes = [
        (received_ns, message)
        for received_ns, message in captured
        if message.type in ("note_on", "note_off")
    ]
    latencies_ms = [
        (received_ns - sent[i].target_ns) / 1e6
        for i, (received_ns, _message) in enumerate(notes)
        if i < len(sent)
    ]
    order_mismatch_count = sum(
        1
        for i, (_received_ns, message) in enumerate(notes)
        if i < len(sent) and message.note != sent[i].message.note
    )
    ordered = sorted(latencies_ms)
    return {
        "capture_observed": bool(notes),
        "drain_completed": drain_completed,
        "sent_note_event_count": len(sent),
        "captured_note_event_count": len(notes),
        "captured_total_message_count": len(captured),
        "event_loss_count": max(0, len(sent) - len(notes)),
        "duplicate_output_count": max(0, len(notes) - len(sent)),
        "order_mismatch_count": order_mismatch_count,
        "scheduled_to_capture_ms": (
            {
                "p50": ordered[len(ordered) // 2],
                "maximum": ordered[-1],
                "sample_count": len(ordered),
            }
            if ordered
            else None
        ),
    }


def write_played_midi(result, path, *, bpm):
    """Write what the scheduler actually dispatched, so a run can be listened to.

    Built from the scheduler's own records rather than the generated blocks, so
    a bar served from fallback appears exactly as it was played.
    """
    import pretty_midi

    records = [r for r in result.records if r.message.type in ("note_on", "note_off")]
    if not records:
        return None
    origin_ns = records[0].target_ns
    midi = pretty_midi.PrettyMIDI(initial_tempo=float(bpm))
    instrument = pretty_midi.Instrument(program=0, name="Continuous lead")
    open_notes: dict[int, tuple[float, int]] = {}
    for record in records:
        seconds = (record.target_ns - origin_ns) / 1e9
        note = record.message.note
        if record.message.type == "note_on" and record.message.velocity > 0:
            open_notes[note] = (seconds, record.message.velocity)
        else:
            started = open_notes.pop(note, None)
            if started is not None and seconds > started[0]:
                instrument.notes.append(
                    pretty_midi.Note(velocity=started[1], pitch=note,
                                     start=started[0], end=seconds)
                )
    midi.instruments = [instrument]
    midi.write(str(path))
    return len(instrument.notes)


def build_report(result, producer, *, bars, bpm, capture=None, spin_window_ms=None):
    return {
        "schema": "continuous_jazz_session_v1",
        "bpm": bpm,
        "bars": bars,
        "beats_per_bar": BEATS_PER_BAR,
        "generation_mode": "one_bar_lookahead_background_producer",
        "run_completed": result.run_completed,
        "completed_bars": result.completed_bar_count,
        "queue_underrun_count": result.queue_underrun_count,
        "scheduler_dispatch_deadline_miss_count": result.scheduler_dispatch_deadline_miss_count,
        "send_failure_count": result.send_failure_count,
        "deadline_policy": result.deadline_policy,
        "accepted_sends": len(result.records),
        "production": summarize_production(producer.records),
        "bars_detail": [vars(r) for r in producer.records],
        # Separate from any producer timing: this is the scheduler's own lateness.
        "dispatch_attempt_lateness_ms": sorted(
            ns / 1e6 for ns in result.dispatch_attempt_lateness_ns
        )[-5:],
        "spin_window_ms": spin_window_ms,
        "deadline_miss_detail": summarize_deadline_misses(
            result.scheduler_dispatch_deadline_misses, producer.records),
        "dispatch_attempt_lateness_summary_ms": summarize_lateness_ms(
            result.dispatch_attempt_lateness_ns),
        "capture": capture,
        "realtime_coperformance_verified": False,
        "musical_quality_verified": False,
        "model_chord_conditioning": False,
        "external_keyboard_verified": False,
        "daw_audio_verified": False,
        "output_capture_observed": bool(capture and capture["capture_observed"]),
    }


def _resolve_capture_name(mido_module, virtual_port):
    """CoreMIDI may expose a virtual source under a client-prefixed name."""
    names = mido_module.get_input_names()
    if virtual_port in names:
        return virtual_port
    matches = [n for n in names if virtual_port in n]
    if not matches:
        raise RuntimeError(f"virtual port {virtual_port!r} not visible as an input: {names}")
    return matches[0]


def _fetch_margin(text: str) -> float | None:
    return None if text.lower() in ("off", "none") else float(text)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--conditioning-midi", type=Path)
    parser.add_argument("--fallback-only", action="store_true")
    parser.add_argument("--bars", type=int, default=8)
    parser.add_argument("--bpm", type=int, default=128)
    parser.add_argument("--chords", default="Dm7,G7,Cmaj7,A7")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--port", help="Exact existing MIDI output name")
    parser.add_argument("--input-port", help="Exact existing MIDI input name")
    parser.add_argument("--virtual-port", default="ContinuousJazz",
                        help="Name for a virtual output port when --port is absent")
    parser.add_argument("--capture", action="store_true",
                        help="Open an independent CoreMIDI input on the virtual port "
                             "and report scheduled-instant to capture latency")
    parser.add_argument("--drain-seconds", type=float, default=2.0)
    parser.add_argument("--generation-tokens", type=int, default=96,
                        help="New tokens allowed per bar, on top of the primer. "
                             "An absolute cap shrinks the budget whenever the primer "
                             "grows, which shows up as duration underfill.")
    parser.add_argument("--max-sequence", type=int, default=192)
    parser.add_argument("--chord-blocks-per-bar", type=int, default=1,
                        help="with --chord-primer: restate the chord this many times "
                             "per bar. 2 is the measured recommendation; 4 exceeds the "
                             "per-block latency budget (see the experiment doc)")
    parser.add_argument("--chord-primer", action="store_true",
                        help="opt-in: state each bar's chord as notes in the primer. "
                             "Note-based steering, not learned chord conditioning; "
                             "see docs/experiments/CHORD_PRIMER_AB.md")
    parser.add_argument("--kv-cache", action=argparse.BooleanOptionalAction, default=True,
                        help="KV-cached generation: identical tokens, about half the generation "
                             "time (docs/experiments/KV_CACHE.md); --no-kv-cache for the old path")
    parser.add_argument("--merge-lora", action=argparse.BooleanOptionalAction, default=True,
                        help="fold LoRA deltas into the base weights before playing: identical "
                             "tokens, about 50%% lower p50 on the Tatum final adapter "
                             "(docs/experiments/LORA_MERGE.md); --no-merge-lora for the old path")
    parser.add_argument("--half-bar-blocks", action="store_true",
                        help="with --chord-primer --chord-blocks-per-bar 2: hand each half bar to "
                             "the scheduler as its own block, so late fetch, start budget, input "
                             "and adapter choice work per half bar (docs/experiments/HALF_BAR_BLOCKS.md)")
    parser.add_argument("--live-metrics", action="store_true",
                        help="print one line of block metrics (adapter, notes, pitch, chord-tone "
                             "ratio, input notes) as each block is generated; they are always "
                             "saved in the report as block_metrics (docs/experiments/LIVE_METRICS.md)")
    parser.add_argument("--block-metrics", action=argparse.BooleanOptionalAction, default=True,
                        help="record per-block metrics in the report (on by default); "
                             "--no-block-metrics turns the in-loop measurement off")
    parser.add_argument("--reserve-chord-tokens", action=argparse.BooleanOptionalAction,
                        default=False,
                        help="experimental, off by default: with --chord-primer and live input, keep "
                             "the full chord statement right before generation and fill the rest of "
                             "the 48-token primer with the newest input. Raises chord-tone ratio under "
                             "dense input but weakened register following (docs/experiments/"
                             "RESERVE_CHORD_TOKENS.md, INPUT_REGISTER_FOLLOW.md)")
    parser.add_argument("--adapter-name", default="primary",
                        help="name of the --checkpoint adapter in --adapter-schedule")
    parser.add_argument("--swap-adapter", action="append", default=[], metavar="NAME=CHECKPOINT",
                        help="another adapter on the same base, swapped in at bar starts "
                             "(docs/experiments/ADAPTER_SWAP.md); needs --adapter-schedule")
    parser.add_argument("--adapter-schedule", default=None,
                        help='bars per adapter, cycled: e.g. "tatum:4,mehldau:4"')
    parser.add_argument("--adapter-control", default=None, metavar="program|cc:N",
                        help="choose the adapter live from --input-port: Program Change p picks "
                             "the p-th adapter (--checkpoint first, then --swap-adapter order), or "
                             "CC N splits 0-127 across them; applied at the next bar generated "
                             "(docs/experiments/ADAPTER_LIVE_SELECT.md)")
    parser.add_argument("--allow-different-bases", action="store_true",
                        help="let --swap-adapter checkpoints come from different bases; a swap "
                             "then copies the whole model instead of the LoRA-touched weights")
    parser.add_argument("--context-carry-tokens", type=int, default=0,
                        help="with --chord-primer: prepend the last N tokens of the previous "
                             "valid block to each block's primer (0 = off, the previous behaviour)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="sampling temperature (1.0 = previous behaviour). Lower values made "
                             "continuous Tatum output reuse motifs more (docs/experiments/COHERENCE_GOAL.md)")
    parser.add_argument("--live-chords", choices=["off", "observe", "follow"], default="off",
                        help="read the chord held below --chord-split from --input-port: observe "
                             "records it, follow also uses it as the next block's chord statement "
                             "(docs/experiments/LIVE_CHORDS.md)")
    parser.add_argument("--chord-split", type=int, default=60,
                        help="--live-chords: only notes below this pitch name the chord (128 = all)")
    parser.add_argument("--ignore-echo-ms", type=float, default=0.0,
                        help="drop input notes that repeat a note this run sent within N ms "
                             "(a DAW forwarding the AI channel back to its MIDI Out); 0 = off")
    parser.add_argument("--context-history", action="store_true",
                        help="with --context-carry-tokens: carry the last N tokens of everything "
                             "played so far (several blocks), not only the previous block "
                             "(docs/experiments/CONTEXT_HISTORY.md)")
    parser.add_argument("--context-carry-position", choices=["before", "after"], default="before",
                        help="where the carried tail goes: before the chord statement (default) "
                             "or after it, right before generation")
    parser.add_argument("--thread-qos", default="user-interactive",
                        choices=["default", "user-initiated", "user-interactive"],
                        help="macOS QoS class for the scheduler thread "
                             "(docs/experiments/RUNTIME_STALL_CAUSE.md)")
    parser.add_argument("--stall-trace", action="store_true",
                        help="record GC pauses and in-/out-of-process heartbeat gaps and "
                             "attribute dispatches >10 ms late (docs/experiments/RUNTIME_STALL_CAUSE.md)")
    parser.add_argument("--fetch-margin-ms", type=_fetch_margin, default=50.0,
                        help="ask the producer for each bar only this long before its downbeat "
                             "and keep one bar of lead, so input reaches the output two bars "
                             "sooner (docs/experiments/GENERATION_LEAD.md); 'off' restores the "
                             "old one-bar-ahead fetch with two bars of lead")
    parser.add_argument("--start-budget-bars", default="auto",
                        help="with late fetch: start generating each bar only this fraction of "
                             "a bar before it is fetched, so it hears more recent input "
                             "(docs/experiments/START_BUDGET.md). 'auto' = 0.5 with late fetch, "
                             "nothing without it; 'off' starts right after the previous fetch")
    parser.add_argument("--adaptive-start-safety", default="off",
                        help="with a start budget: widen it to this multiple of the slowest of "
                             "the last four generations when that is larger, e.g. 1.5 "
                             "(docs/experiments/ADAPTIVE_BUDGET.md); 'off' keeps it fixed")
    parser.add_argument("--spin-window-ms", type=float, default=5.0,
                        help="scheduler busy-spin before each event. The wait before it "
                             "can oversleep by a few ms; a longer spin absorbs that but "
                             "holds the GIL longer (docs/experiments/TATUM_REALTIME_TEMPO.md)")
    args = parser.parse_args(argv)

    chords = [c.strip() for c in args.chords.split(",") if c.strip()]
    # 16 was the first build's scope (ace0787f); longer runs are for playing (#1581).
    if not 8 <= args.bars <= 256:
        parser.error("bars must be between 8 and 256")
    if not 40 <= args.bpm <= 240 or not chords:
        parser.error("require 40..240 BPM and nonempty chords")
    if args.generation_tokens < 1 or args.max_sequence < args.generation_tokens:
        parser.error("require generation_tokens >= 1 and max_sequence >= generation_tokens")
    if not 0.0 <= args.spin_window_ms <= 50.0:
        parser.error("spin_window_ms must be between 0 and 50")
    if args.fetch_margin_ms is not None and not 0.0 <= args.fetch_margin_ms <= 500.0:
        parser.error("fetch_margin_ms must be between 0 and 500")
    safety = str(args.adaptive_start_safety).lower()
    if safety in ("off", "none"):
        args.adaptive_start_safety = None
    else:
        try:
            args.adaptive_start_safety = float(safety)
        except ValueError:
            parser.error("--adaptive-start-safety takes off or a number")
        if not 1.0 <= args.adaptive_start_safety <= 4.0:
            parser.error("adaptive_start_safety must be between 1 and 4")
    budget = str(args.start_budget_bars).lower()
    if budget == "auto":
        args.start_budget_bars = 0.5 if args.fetch_margin_ms is not None else None
    elif budget in ("off", "none"):
        args.start_budget_bars = None
    else:
        try:
            args.start_budget_bars = float(budget)
        except ValueError:
            parser.error("--start-budget-bars takes auto, off or a number")
        if args.fetch_margin_ms is None:
            parser.error("--start-budget-bars needs the late fetch (drop --fetch-margin-ms off)")
        if not 0.1 <= args.start_budget_bars <= 0.95:
            parser.error("start_budget_bars must be between 0.1 and 0.95")
    if not 1 <= args.chord_blocks_per_bar <= 4:
        parser.error("chord_blocks_per_bar must be between 1 and 4")
    if args.context_carry_tokens < 0:
        parser.error("context_carry_tokens must be >= 0")
    if args.context_carry_tokens and not args.chord_primer:
        parser.error("--context-carry-tokens needs --chord-primer")
    if not 0.1 <= args.temperature <= 2.0:
        parser.error("temperature must be between 0.1 and 2.0")
    if args.context_history and not args.context_carry_tokens:
        parser.error("--context-history needs --context-carry-tokens")
    if args.context_history and not args.half_bar_blocks:
        # Each history entry is one scheduler block whose adoption is known;
        # other paths would silently ignore the option (Astra review).
        parser.error("--context-history needs --half-bar-blocks")
    if args.context_carry_tokens and args.chord_blocks_per_bar < 2:
        parser.error("--context-carry-tokens needs --chord-blocks-per-bar 2 (sub-block path)")
    if args.chord_primer and 48 + args.context_carry_tokens + args.generation_tokens > args.max_sequence:
        parser.error("chord primer (48) + context carry + generation tokens exceeds --max-sequence")
    if args.chord_blocks_per_bar > 1 and not args.chord_primer:
        parser.error("--chord-blocks-per-bar needs --chord-primer")
    if args.half_bar_blocks and not (args.chord_primer and args.chord_blocks_per_bar == 2):
        parser.error("--half-bar-blocks needs --chord-primer --chord-blocks-per-bar 2")
    if args.live_chords != "off" and not args.input_port:
        parser.error("--live-chords reads the player's input; give --input-port")
    if args.live_chords != "off" and not args.half_bar_blocks:
        # One chord decision per scheduler block, made with that block's snapshot.
        parser.error("--live-chords needs --half-bar-blocks")
    if not 1 <= args.chord_split <= 128:
        parser.error("chord_split must be between 1 and 128 (128 = every held note)")
    if not 0 <= args.ignore_echo_ms <= 500:
        parser.error("ignore_echo_ms must be between 0 and 500")
    if args.ignore_echo_ms and not args.input_port:
        parser.error("--ignore-echo-ms filters --input-port; give --input-port")
    if args.live_metrics and not args.block_metrics:
        parser.error("--live-metrics needs --block-metrics")
    if args.half_bar_blocks and args.fallback_only:
        parser.error("--half-bar-blocks needs a model; drop --fallback-only")
    swap_specs = {}
    for spec in args.swap_adapter:
        name, sep, path = spec.partition("=")
        if not sep or not name or not path:
            parser.error("--swap-adapter takes NAME=CHECKPOINT")
        if name in swap_specs or name == args.adapter_name:
            parser.error(f"duplicate adapter name {name!r}")
        swap_specs[name] = Path(path)
    if swap_specs and not (args.adapter_schedule or args.adapter_control):
        parser.error("--swap-adapter needs --adapter-schedule or --adapter-control")
    if (args.adapter_schedule or args.adapter_control) and not swap_specs:
        parser.error("--adapter-schedule/--adapter-control need --swap-adapter")
    if args.adapter_schedule and args.adapter_control:
        parser.error("use either --adapter-schedule or --adapter-control")
    if args.adapter_control and not args.input_port:
        parser.error("--adapter-control reads the player's input; give --input-port")
    if swap_specs and not args.merge_lora:
        parser.error("adapter swapping works on merged weights; drop --no-merge-lora")
    if swap_specs and args.fallback_only:
        parser.error("--swap-adapter needs a model; drop --fallback-only")

    generate = None
    live_chords = None
    if args.live_chords != "off":
        from inference.control.live_chords import LiveChordTracker
        live_chords = LiveChordTracker(split=args.chord_split, follow=args.live_chords == "follow")
    live_primer_bars: list[bool] = []
    chord_primer_bars: list[bool] = []
    if not args.fallback_only:
        if not args.checkpoint or not args.conditioning_midi:
            parser.error("provide --checkpoint and --conditioning-midi, or --fallback-only")
        import torch
        from scripts.generate import build_primer, generate_once, load_model_with_lora

        model = load_model_with_lora(
            lora_path=str(args.checkpoint.parent), checkpoint_path=str(args.checkpoint),
            prefer_full_checkpoint=True, max_sequence=args.max_sequence,
        )
        if args.merge_lora:
            from scripts.train_qlora import merge_lora_for_inference
            merge_lora_for_inference(model)
        bank = schedule = live_selector = None
        if swap_specs:
            from scripts.adapter_bank import AdapterBank, parse_schedule
            models = {args.adapter_name: model}
            for name, path in swap_specs.items():
                other = load_model_with_lora(
                    lora_path=str(path.parent), checkpoint_path=str(path),
                    prefer_full_checkpoint=True, max_sequence=args.max_sequence,
                )
                merge_lora_for_inference(other)
                models[name] = other
            bank = AdapterBank(models, allow_different_bases=args.allow_different_bases)
            del models
            if args.adapter_schedule:
                schedule = parse_schedule(args.adapter_schedule, bank.names)
            else:
                from scripts.adapter_bank import LiveAdapterSelector
                live_selector = LiveAdapterSelector(bank.names, args.adapter_control)
        adapter_per_bar: dict[int, str] = {}

        def select_adapter(block_index, input_events=(), bar_index=None):
            """Swap at block starts only, on the producer thread, between generations.

            A block is a bar, or a half bar with --half-bar-blocks; a schedule
            still counts whole bars."""
            if bank is None:
                return
            if live_selector is not None:
                name = live_selector.update(input_events, block_index=block_index)
            else:
                from scripts.adapter_bank import adapter_for_bar
                name = adapter_for_bar(schedule, block_index if bar_index is None else bar_index)
            bank.select(name)
            adapter_per_bar[block_index] = name
        base_primer = build_primer(
            conditioning_midi=str(args.conditioning_midi), primer_max_tokens=32,
            append_sep_token=True, control_format="control_v1", role="lead",
            tempo_bpm=args.bpm,
        )

        context_carry = {"tokens": []}
        played_history = PlayedHistory(60.0 / args.bpm * 2)     # half-bar blocks only
        generated_blocks: dict = {}

        def generate_sub(bar_index, sub_index, input_events, sub_duration, block_index=None):
            """One sub-block: harmony restated, then continue.

            ``block_index`` is set when the sub-block is its own scheduler block
            (--half-bar-blocks): then every sub-block takes the newest input and
            adapter choice, not only the downbeat one. Seeds stay per (bar, sub),
            so with no input the tokens match the one-block-per-bar path."""
            from inference.control.chord_primer import chord_guide_notes_for_duration
            from scripts.generate import (
                encode_notes_simple,
                truncate_tokens_preserving_velocity,
            )

            fresh_block = sub_index == 0 or block_index is not None
            if fresh_block:
                select_adapter(bar_index if block_index is None else block_index,
                               input_events, bar_index=bar_index)
            chord = chords[bar_index % len(chords)]
            if live_chords is not None:
                chord = live_chords.update(input_events, block_index, default=chord)
            notes = list(chord_guide_notes_for_duration(chord, bpm=args.bpm,
                                                        seconds=sub_duration))
            # Only the downbeat sub-block folds in what the player just did; a
            # later sub-block would be conditioning on input it already used.
            # A half-bar block has its own, newer snapshot, so it folds its input in.
            played = input_events_to_notes(input_events) if fresh_block else []
            if args.reserve_chord_tokens and played:
                # Dense input used to push the chord statement out of the 48-token
                # window (docs/experiments/RESERVE_CHORD_TOKENS.md): keep the whole
                # chord statement next to generation and give the input what is left.
                chord_tokens = truncate_tokens_preserving_velocity(encode_notes_simple(
                    sorted(notes, key=lambda note: (note.start, note.pitch))), 48)
                input_tokens = truncate_tokens_preserving_velocity(encode_notes_simple(
                    sorted(played, key=lambda note: (note.start, note.pitch))),
                    max(0, 48 - len(chord_tokens)))
                notes = notes + played
                tokens = list(input_tokens) + list(chord_tokens)
            else:
                notes.extend(played)
                notes.sort(key=lambda note: (note.start, note.pitch))
                tokens = truncate_tokens_preserving_velocity(encode_notes_simple(notes), 48)
            # Opt-in context carry: the tail of the previous valid block goes in
            # front of this block's chord statement, so the model continues the
            # line instead of starting every half bar from scratch.
            if args.context_history:
                # Only blocks the scheduler already asked for: adopted model
                # blocks, or the fallbacks that played instead (Astra review).
                producer_now = producer_box.get("producer")
                if producer_now is not None:
                    adopted_now, watermark_now = producer_now.adoption_snapshot()
                    played_history.settle(adopted=adopted_now, watermark=watermark_now,
                                          generated=generated_blocks,
                                          fallback_for=producer_now.fallback_for)
                carried = carry_tokens(played_history.tokens, args.context_carry_tokens)
            else:
                carried = carry_tokens(context_carry["tokens"], args.context_carry_tokens)
            ordered = (tokens + carried) if args.context_carry_position == "after" else (carried + tokens)
            primer = torch.tensor(ordered or [60], dtype=torch.long)
            chord_primer_bars.append(bool(notes))
            torch.manual_seed(args.seed + bar_index * 13 + sub_index * 977)
            tokens_out, _meta = generate_once(
                model=model, primer=primer,
                target_length=min(args.max_sequence, len(primer) + args.generation_tokens),
                strip_primer=True, temperature=args.temperature, top_k=32, top_p=0.95,
                grammar_mask=True, target_duration_seconds=sub_duration,
                return_metadata=True, use_kv_cache=args.kv_cache,
            )
            if args.context_carry_tokens > 0 and validate_generated_token_block(
                    tokens_out, lookahead_ms=sub_duration * 1000, allow_rest_bar=True)["valid"]:
                if not args.context_history:
                    context_carry["tokens"] = [int(t) for t in tokens_out]
            return tokens_out

        def generate(bar_index, input_events):
            select_adapter(bar_index, input_events)
            if args.chord_primer:
                primer, used_input, used_chord = build_chord_live_primer(
                    input_events, chords[bar_index % len(chords)], bpm=args.bpm,
                    base_primer=base_primer,
                )
                chord_primer_bars.append(used_chord)
            else:
                primer, used_input = build_live_primer(
                    input_events, base_primer=base_primer,
                    control_format="control_v1", role="lead", tempo_bpm=args.bpm,
                )
            live_primer_bars.append(used_input)
            torch.manual_seed(args.seed + bar_index)
            # Budget the new tokens, not the total: a longer primer must not
            # silently eat the room the bar needs to be filled.
            target_length = min(args.max_sequence, len(primer) + args.generation_tokens)
            return generate_once(
                model=model, primer=primer, target_length=target_length, strip_primer=True,
                temperature=args.temperature, top_k=32, top_p=0.95, grammar_mask=True,
                target_duration_seconds=240.0 / args.bpm, return_metadata=True,
                use_kv_cache=args.kv_cache,
            )
    # --fallback-only leaves `generate` as None: a deliberate mode, so the
    # producer records it as fallback_disabled rather than a generation error.

    import mido

    input_buffer = MidiInputSnapshotBuffer()
    echo_guard = None
    if args.ignore_echo_ms:
        from inference.realtime.echo import EchoGuard
        echo_guard = EchoGuard(args.ignore_echo_ms)
    input_port = None
    if args.input_port:
        input_port = mido.open_input(
            args.input_port,
            callback=(lambda m: input_buffer.handle(m)) if echo_guard is None
            else (lambda m: None if echo_guard.is_echo(m) else input_buffer.handle(m)),
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "continuous_report.json"
    opener = (
        (lambda: mido.open_output(args.port)) if args.port
        else (lambda: mido.open_output(args.virtual_port, virtual=True))
    )
    if args.capture and args.port:
        parser.error("--capture applies to the virtual output port, so omit --port")

    report = None
    captured: list[tuple[int, object]] = []
    capture_input = None
    try:
        with opener() as port:
            if echo_guard is not None:
                port = echo_guard.wrap(port)
            if args.capture:
                # Independent consumer: a separate CoreMIDI input, not the sink.
                capture_input = mido.open_input(
                    _resolve_capture_name(mido, args.virtual_port),
                    callback=lambda m: captured.append((time.perf_counter_ns(), m)),
                )
            sub_builder = None
            blocks = args.bars * 2 if args.half_bar_blocks else args.bars
            if args.half_bar_blocks:
                def sub_builder(*, clock, duration):
                    return make_sub_block_builder(
                        clock=clock, duration=duration, blocks_per_bar=1,
                        generate_sub=lambda h, _sub, events, d: generate_sub(
                            h // 2, h % 2, events, d, block_index=h))
            elif args.chord_blocks_per_bar > 1:
                def sub_builder(*, clock, duration):
                    return make_sub_block_builder(
                        clock=clock, duration=duration, generate_sub=generate_sub,
                        blocks_per_bar=args.chord_blocks_per_bar)
            recorder = None
            producer_box: dict = {}
            if not args.fallback_only and args.context_history:
                sub_builder = with_block_metrics(
                    sub_builder, record=lambda block, _ev: generated_blocks.__setitem__(block.bar_index, block))
            if not args.fallback_only and args.block_metrics:
                def print_adopted(m):
                    print(f"block {m['block']:3d} {m['adapter'] or '-':>8} notes {m['notes']:3d} "
                          f"pitch {m['pitch_mean'] if m['pitch_mean'] is not None else '-':>6} "
                          f"chord-tone {m['chord_tone_ratio'] if m['chord_tone_ratio'] is not None else '-':>6} "
                          f"input {m['input_notes']} voicings {m['voicings']} "
                          f"unique-so-far {m['unique_voicings_so_far']}", flush=True)

                recorder = BlockMetricsRecorder(
                    producer_ref=lambda: producer_box.get("producer"),
                    chord_for_block=lambda b: (
                        live_chords.chord_for(b, chords[(b // 2) % len(chords)]) if live_chords is not None
                        else chords[(b // 2 if args.half_bar_blocks else b) % len(chords)]),
                    adapter_for_block=lambda b: adapter_per_bar.get(b, None if bank else args.adapter_name),
                    on_adopted=print_adopted if args.live_metrics else None)
                base_factory = sub_builder or (
                    lambda *, clock, duration: make_block_builder(
                        clock=clock, duration=duration, generate=generate))
                sub_builder = with_block_metrics(base_factory, record=recorder.on_generated)
            qos_result = None
            if args.thread_qos != "default":
                # The scheduler runs on this (main) thread inside run_session.
                from inference.realtime.thread_qos import set_current_thread_qos
                qos_result = set_current_thread_qos(args.thread_qos)
            tracer = None
            if args.stall_trace:
                from inference.realtime.stall_trace import StallTracer
                tracer = StallTracer().start()
            try:
                result, producer = run_session(
                    port=port, bars=blocks, bpm=args.bpm, seed=args.seed,
                    chords=([chords[(h // 2) % len(chords)] for h in range(2 * len(chords))]
                            if args.half_bar_blocks else chords),
                    beats_per_block=2 if args.half_bar_blocks else BEATS_PER_BAR,
                    generate=generate, input_buffer=input_buffer, sub_builder=sub_builder,
                    spin_window_ms=args.spin_window_ms, fetch_margin_ms=args.fetch_margin_ms,
                    start_budget_bars=args.start_budget_bars,
                    adaptive_start_safety=args.adaptive_start_safety,
                    on_producer=lambda p: producer_box.__setitem__("producer", p),
                )
            finally:
                if tracer is not None:
                    tracer.stop()
            drain_completed = False
            if args.capture:
                # Let in-flight packets land before tearing the port down.
                deadline = time.monotonic() + args.drain_seconds
                last = -1
                while time.monotonic() < deadline and last != len(captured):
                    last = len(captured)
                    time.sleep(0.25)
                drain_completed = last == len(captured)
            capture_summary = (
                summarize_capture(result, list(captured), drain_completed=drain_completed)
                if args.capture
                else None
            )
            report = build_report(result, producer, bars=args.bars, bpm=args.bpm,
                                  spin_window_ms=args.spin_window_ms,
                                  capture=capture_summary)
            if args.half_bar_blocks:
                # Scheduler and producer counts are per half-bar block here.
                report["block_beats"] = 2
                report["blocks"] = blocks
                report["completed_bars"] = result.completed_bar_count // 2
            if tracer is not None:
                from inference.realtime.stall_trace import classify_late_events, summarize_trace
                producer_intervals = [
                    (r.requested_ns, r.completed_ns) for r in producer.records
                    if getattr(r, "requested_ns", None) is not None
                    and getattr(r, "completed_ns", None) is not None]
                late = classify_late_events(
                    result.records, threshold_ns=10_000_000,
                    gc_intervals=tracer.gc_intervals, inproc_gaps=tracer.inproc_gaps,
                    external_gaps=tracer.external_gaps, producer_intervals=producer_intervals)
                report["stall_trace"] = summarize_trace(tracer, late)
            report["thread_qos"] = qos_result or {"requested": "default", "applied": False,
                                                  "error": None}
            report["input_events_received"] = input_buffer.received_count
            report["live_primer_bar_count"] = (
                sum(1 for x in live_primer_bars if x) if not args.fallback_only else 0
            )
            report["chord_primer_enabled"] = bool(args.chord_primer)
            report["chord_blocks_per_bar"] = args.chord_blocks_per_bar
            report["kv_cache"] = bool(args.kv_cache)
            report["merge_lora"] = bool(args.merge_lora)
            # Blocks whose model block get() actually handed to the scheduler.
            report["adopted_blocks"] = sorted(producer.adopted_blocks)
            report["fetch_margin_ms"] = args.fetch_margin_ms
            if echo_guard is not None:
                report["echo_guard"] = {"window_ms": args.ignore_echo_ms, "dropped": echo_guard.dropped}
            if live_chords is not None and producer.clock is not None:
                report["live_chords"] = live_chords.report(producer.clock.bar_start_ns,
                                                           producer.clock.bar_start_ns(0))
            report["start_budget_bars"] = args.start_budget_bars
            report["adaptive_start_safety"] = args.adaptive_start_safety
            if not args.fallback_only and bank is not None:
                swaps = sorted(bank.swap_ms)
                report["adapter_swap"] = {
                    "adapters": {args.adapter_name: str(args.checkpoint),
                                 **{n: str(p) for n, p in swap_specs.items()}},
                    "schedule": args.adapter_schedule,
                    "control": args.adapter_control,
                    "per_bar": [adapter_per_bar.get(i) for i in range(blocks)],
                    "shared_base": bank.shared_base,
                    "swap_keys": len(bank.swap_keys),
                    "swaps": len(swaps),
                    "swap_ms_p50": swaps[len(swaps) // 2] if swaps else None,
                    "swap_ms_max": swaps[-1] if swaps else None,
                }
                clock_for_events = getattr(producer, "_clock", None)
                if live_selector is not None and clock_for_events is not None:
                    report["adapter_swap"]["control_events"] = [
                        {"received_ms_from_start": (e["received_ns"] - clock_for_events.start_ns) / 1e6,
                         "value": e["value"], "selected": e["selected"], "previous": e["previous"],
                         "noop": e["noop"], "superseded": e["superseded"],
                         "consumed_block": e["consumed_block"]}
                        for e in live_selector.events]
            report["chords"] = chords
            report["context_carry_tokens"] = args.context_carry_tokens
            report["context_carry_position"] = args.context_carry_position
            report["context_history"] = bool(args.context_history)
            report["temperature"] = args.temperature
            report["chord_primer_bar_count"] = sum(1 for x in chord_primer_bars if x)
            # Note-based steering only. The model has no chord token, and no
            # human has judged whether the result sounds harmonically right.
            report["learned_chord_conditioning"] = False
            report["chord_following_verified"] = False
            args.output_dir.mkdir(parents=True, exist_ok=True)
            clock = getattr(producer, "_clock", None)
            if clock is not None:
                played = played_bar_notes(result, clock, blocks)
                if args.half_bar_blocks:
                    played = merge_half_bars(played, half_seconds=120.0 / args.bpm)
                report["played_bars"] = played
            # block_metrics: only blocks whose model block get() returned to the
            # scheduler (played; send_status says whether all events went out).
            # generation_metrics: every model block built, adopted or not.
            if recorder is not None:
                recorder.finish(result.records)
                report["block_metrics"] = [recorder.played.get(i) for i in range(blocks)]
                report["generation_metrics"] = [recorder.generation.get(i) for i in range(blocks)]
            else:
                report["block_metrics"] = report["generation_metrics"] = None
            report["played_note_count"] = write_played_midi(
                result, args.output_dir / "played.mid", bpm=args.bpm
            )
    finally:
        from inference.realtime.transport import close_mido_input

        for opened in (capture_input, input_port):
            if opened is not None:
                close_mido_input(opened)
        if report is not None:
            report_path.write_text(json.dumps(report, indent=2, default=str) + "\n")

    print(json.dumps(report["production"], indent=2))
    print(f"\nreport: {report_path}")
    print(f"played MIDI: {args.output_dir / 'played.mid'}")
    print("NOT verified: external keyboard, DAW audio, musical quality, chord conditioning.")
    return 0 if report["run_completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
