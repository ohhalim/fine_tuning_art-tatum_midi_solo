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
    summarize_production,
)
from inference.realtime.scheduler import MonotonicBarClock, OneBarMidiScheduler
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


def make_block_builder(*, clock, duration, generate):
    """Return the producer callback. Runs on the producer thread, never the scheduler's."""

    def build(bar_index, input_events):
        # `chords` is not passed to the model: generation is not chord
        # conditioned yet. It only shapes the prebuilt fallbacks.
        tokens, _metadata = generate(bar_index, input_events)
        valid = validate_generated_token_block(tokens, lookahead_ms=duration * 1000)
        if not valid["valid"]:
            raise ValueError(f"invalid model block: {valid}")
        from midi_processor.processor import decode_midi

        return build_scheduled_midi_block(
            midi=fit_window(decode_midi(tokens), duration),
            clock=clock, bar_index=bar_index, block_id=str(bar_index),
            source_context_id="continuous", context_version=0,
            adapter="model", fallback_used=False,
        )

    return build


def run_session(*, port, bars, bpm, chords, seed, generate, input_buffer=None,
                start_delay_seconds=1.0, spin_window_ms=1.0,
                clock=None, clock_ns=None, wait_until=None):
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
        bar_count=bars, fallback_blocks=fallbacks, clock=clock, input_buffer=input_buffer,
        build_block=make_block_builder(clock=clock, duration=duration, generate=generate),
    )
    scheduler_kwargs = {"sink": port, "clock": clock, "spin_window_ms": spin_window_ms}
    if clock_ns is not None:
        scheduler_kwargs["clock_ns"] = clock_ns
    if wait_until is not None:
        scheduler_kwargs["wait_until"] = wait_until
    scheduler = OneBarMidiScheduler(**scheduler_kwargs)
    try:
        producer.start()
        # Give bar 0 the lead time it would get from a bar of playback.
        producer.wait_for_bar(0, timeout=max(0.0, start_delay_seconds))
        result = scheduler.run(blocks=producer, expected_bar_count=bars)
    finally:
        # Reset on completion, abort, exception and Ctrl-C alike.
        producer.close()
        try:
            port.reset()
        finally:
            port.panic()
    return result, producer


def build_report(result, producer, *, bars, bpm):
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
        "accepted_sends": len(result.records),
        "production": summarize_production(producer.records),
        "bars_detail": [vars(r) for r in producer.records],
        # Separate from any producer timing: this is the scheduler's own lateness.
        "dispatch_attempt_lateness_ms": sorted(
            ns / 1e6 for ns in result.dispatch_attempt_lateness_ns
        )[-5:],
        "realtime_coperformance_verified": False,
        "musical_quality_verified": False,
        "model_chord_conditioning": False,
        "external_keyboard_verified": False,
        "daw_audio_verified": False,
        "output_capture_observed": False,
    }


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
    args = parser.parse_args(argv)

    chords = [c.strip() for c in args.chords.split(",") if c.strip()]
    if not 8 <= args.bars <= 16:
        parser.error("this path is limited to 8..16 bars for now")
    if not 40 <= args.bpm <= 240 or not chords:
        parser.error("require 40..240 BPM and nonempty chords")

    generate = None
    if not args.fallback_only:
        if not args.checkpoint or not args.conditioning_midi:
            parser.error("provide --checkpoint and --conditioning-midi, or --fallback-only")
        import torch
        from scripts.generate import build_primer, generate_once, load_model_with_lora

        model = load_model_with_lora(
            lora_path=str(args.checkpoint.parent), checkpoint_path=str(args.checkpoint),
            prefer_full_checkpoint=True, max_sequence=80,
        )
        base_primer = build_primer(
            conditioning_midi=str(args.conditioning_midi), primer_max_tokens=32,
            append_sep_token=True, control_format="control_v1", role="lead",
            tempo_bpm=args.bpm,
        )

        def generate(bar_index, _input_events):
            # Input events are captured and timestamped, but folding them into
            # the primer is not implemented yet; the primer stays fixed.
            torch.manual_seed(args.seed + bar_index)
            return generate_once(
                model=model, primer=base_primer, target_length=80, strip_primer=True,
                temperature=1.0, top_k=32, top_p=0.95, grammar_mask=True,
                target_duration_seconds=240.0 / args.bpm, return_metadata=True,
            )
    else:
        def generate(bar_index, _input_events):
            raise RuntimeError("fallback-only mode: no model configured")

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
    report = None
    try:
        with opener() as port:
            result, producer = run_session(
                port=port, bars=args.bars, bpm=args.bpm, chords=chords, seed=args.seed,
                generate=generate, input_buffer=input_buffer,
            )
            report = build_report(result, producer, bars=args.bars, bpm=args.bpm)
            report["input_events_received"] = input_buffer.received_count
    finally:
        if input_port is not None:
            from inference.realtime.transport import close_mido_input

            close_mido_input(input_port)
        if report is not None:
            report_path.write_text(json.dumps(report, indent=2, default=str) + "\n")

    print(json.dumps(report["production"], indent=2))
    print(f"\nreport: {report_path}")
    print("NOT verified: external keyboard, DAW audio, musical quality, chord conditioning.")
    return 0 if report["run_completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
