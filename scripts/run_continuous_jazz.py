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


def _make_builder(*, clock, duration, generate, sub_builder):
    """Pick the producer callback: sub-blocks, one bar, or generation disabled."""
    if sub_builder is not None:
        return sub_builder(clock=clock, duration=duration)
    if generate is None:
        return None
    return make_block_builder(clock=clock, duration=duration, generate=generate)


def run_session(*, port, bars, bpm, chords, seed, generate, input_buffer=None,
                start_delay_seconds=2.5, spin_window_ms=1.0,
                clock=None, clock_ns=None, wait_until=None,
                deadline_policy=DEADLINE_POLICY_RECORD_AND_CONTINUE,
                sub_builder=None):
    """Play ``bars`` bars, producing one bar ahead.

    ``clock``/``clock_ns``/``wait_until`` exist so tests can drive the run off a
    fake clock instead of waiting out real bars.
    """
    duration = 240.0 / bpm
    if clock is None:
        clock = MonotonicBarClock(
            bpm=bpm, beats_per_bar=BEATS_PER_BAR,
            start_ns=time.perf_counter_ns() + round(start_delay_seconds * 1e9),
        )
    fallbacks = build_fallback_blocks(
        clock=clock, bars=bars, bpm=bpm, chords=chords, seed=seed, duration=duration
    )
    producer = BarBlockProducer(
        # Lead of 2, not 1. The scheduler asks for bar 1 at bar 0's downbeat,
        # and the watermark only advances once it starts, so a lead of 1 leaves
        # bar 1 unable to begin until playback is already underway.
        bar_count=bars, fallback_blocks=fallbacks, clock=clock, input_buffer=input_buffer,
        max_lead_bars=2,
        build_block=_make_builder(clock=clock, duration=duration, generate=generate,
                                  sub_builder=sub_builder),
    )
    # A single OS hiccup must not end a performance. The scheduler probe uses
    # abort_on_first_miss to make timing failures loud; a live instrument wants
    # the miss recorded and the music continued.
    scheduler_kwargs = {"sink": port, "clock": clock, "spin_window_ms": spin_window_ms,
                        "deadline_policy": deadline_policy}
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


def build_report(result, producer, *, bars, bpm, capture=None):
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
    args = parser.parse_args(argv)

    chords = [c.strip() for c in args.chords.split(",") if c.strip()]
    if not 8 <= args.bars <= 16:
        parser.error("this path is limited to 8..16 bars for now")
    if not 40 <= args.bpm <= 240 or not chords:
        parser.error("require 40..240 BPM and nonempty chords")
    if args.generation_tokens < 1 or args.max_sequence < args.generation_tokens:
        parser.error("require generation_tokens >= 1 and max_sequence >= generation_tokens")
    if not 1 <= args.chord_blocks_per_bar <= 4:
        parser.error("chord_blocks_per_bar must be between 1 and 4")
    if args.chord_blocks_per_bar > 1 and not args.chord_primer:
        parser.error("--chord-blocks-per-bar needs --chord-primer")

    generate = None
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
        base_primer = build_primer(
            conditioning_midi=str(args.conditioning_midi), primer_max_tokens=32,
            append_sep_token=True, control_format="control_v1", role="lead",
            tempo_bpm=args.bpm,
        )

        def generate_sub(bar_index, sub_index, input_events, sub_duration):
            """One sub-block: harmony restated, then continue."""
            from inference.control.chord_primer import chord_guide_notes_for_duration
            from scripts.generate import (
                encode_notes_simple,
                truncate_tokens_preserving_velocity,
            )

            chord = chords[bar_index % len(chords)]
            notes = list(chord_guide_notes_for_duration(chord, bpm=args.bpm,
                                                        seconds=sub_duration))
            # Only the downbeat sub-block folds in what the player just did; a
            # later sub-block would be conditioning on input it already used.
            if sub_index == 0:
                notes.extend(input_events_to_notes(input_events))
            notes.sort(key=lambda note: (note.start, note.pitch))
            tokens = truncate_tokens_preserving_velocity(encode_notes_simple(notes), 48)
            primer = torch.tensor(tokens or [60], dtype=torch.long)
            chord_primer_bars.append(bool(notes))
            torch.manual_seed(args.seed + bar_index * 13 + sub_index * 977)
            tokens_out, _meta = generate_once(
                model=model, primer=primer,
                target_length=min(args.max_sequence, len(primer) + args.generation_tokens),
                strip_primer=True, temperature=1.0, top_k=32, top_p=0.95,
                grammar_mask=True, target_duration_seconds=sub_duration,
                return_metadata=True,
            )
            return tokens_out

        def generate(bar_index, input_events):
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
                temperature=1.0, top_k=32, top_p=0.95, grammar_mask=True,
                target_duration_seconds=240.0 / args.bpm, return_metadata=True,
            )
    # --fallback-only leaves `generate` as None: a deliberate mode, so the
    # producer records it as fallback_disabled rather than a generation error.

    import mido

    input_buffer = MidiInputSnapshotBuffer()
    input_port = None
    if args.input_port:
        input_port = mido.open_input(
            args.input_port, callback=lambda m: input_buffer.handle(m)
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
            if args.capture:
                # Independent consumer: a separate CoreMIDI input, not the sink.
                capture_input = mido.open_input(
                    _resolve_capture_name(mido, args.virtual_port),
                    callback=lambda m: captured.append((time.perf_counter_ns(), m)),
                )
            sub_builder = None
            if args.chord_blocks_per_bar > 1:
                def sub_builder(*, clock, duration):
                    return make_sub_block_builder(
                        clock=clock, duration=duration, generate_sub=generate_sub,
                        blocks_per_bar=args.chord_blocks_per_bar)
            result, producer = run_session(
                port=port, bars=args.bars, bpm=args.bpm, chords=chords, seed=args.seed,
                generate=generate, input_buffer=input_buffer, sub_builder=sub_builder,
            )
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
                                  capture=capture_summary)
            report["input_events_received"] = input_buffer.received_count
            report["live_primer_bar_count"] = (
                sum(1 for x in live_primer_bars if x) if not args.fallback_only else 0
            )
            report["chord_primer_enabled"] = bool(args.chord_primer)
            report["chord_blocks_per_bar"] = args.chord_blocks_per_bar
            report["chord_primer_bar_count"] = sum(1 for x in chord_primer_bars if x)
            # Note-based steering only. The model has no chord token, and no
            # human has judged whether the result sounds harmonically right.
            report["learned_chord_conditioning"] = False
            report["chord_following_verified"] = False
            args.output_dir.mkdir(parents=True, exist_ok=True)
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
