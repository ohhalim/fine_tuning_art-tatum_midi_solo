#!/usr/bin/env python3
"""Generate a preloaded jazz MIDI phrase; optionally play its lead via MIDI."""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pretty_midi
from inference.app.fallback import build_fallback_midi, parse_chord
from inference.app.schemas import GenerationRequest
from inference.realtime.blocks import build_scheduled_midi_block
from inference.realtime.scheduler import MonotonicBarClock, OneBarMidiScheduler
from scripts.run_resident_model_probe import validate_generated_token_block


def fit_window(midi, duration):
    """Clip quantization overshoot; preserve intentional silence."""
    result = copy.deepcopy(midi)
    for instrument in result.instruments:
        for note in instrument.notes:
            if not all(math.isfinite(t) for t in (note.start, note.end)):
                raise ValueError("non-finite note time")
            note.end = min(note.end, duration)
        instrument.notes = [n for n in instrument.notes if n.start < n.end]
    return result


def prepare_phrase(*, generate, bpm, bars, chords, seed):
    duration = 240.0 / bpm
    clock = MonotonicBarClock(bpm=bpm, beats_per_bar=4, start_ns=0)
    phrase = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    lead = pretty_midi.Instrument(program=0, name="Generated lead")
    backing = pretty_midi.Instrument(program=0, name="Chord reference")
    midis, details = [], []
    for index in range(bars):
        started = time.perf_counter_ns()
        chord = chords[index % len(chords)]
        request = GenerationRequest(bpm=bpm, bars=1, chord_progression=[chord], seed=seed + index)
        request.validate()
        fallback_reason = None
        metadata = None
        if generate is None:
            midi = build_fallback_midi(request)
            source = "fallback_only"
        else:
            # Unexpected inference failures abort; they are not hidden as fallback.
            tokens, metadata = generate(index)
            valid = validate_generated_token_block(tokens, lookahead_ms=duration * 1000)
            if not valid["valid"]:
                fallback_reason = valid
                midi = build_fallback_midi(request)
                source = "invalid_model_fallback"
            else:
                from midi_processor.processor import decode_midi
                midi = decode_midi(tokens)
                source = "model"
        def schedulable(candidate, candidate_source):
            build_scheduled_midi_block(
                midi=candidate, clock=clock, bar_index=index, block_id=str(index),
                source_context_id="fixed-primer", context_version=0,
                adapter=candidate_source, fallback_used=candidate_source != "model",
            )

        midi = fit_window(midi, duration)
        try:
            schedulable(midi, source)
        except ValueError as exc:
            # A model block can satisfy token validation and still be unschedulable
            # (e.g. a decoded velocity of 0). Treat it like any other invalid model
            # block instead of aborting. A fallback block failing here is a real bug.
            if source != "model":
                raise
            fallback_reason = {
                "valid": False,
                "unschedulable_model_block": f"{type(exc).__name__}: {exc}",
            }
            source = "unschedulable_model_fallback"
            midi = fit_window(build_fallback_midi(request), duration)
            schedulable(midi, source)
        midis.append(midi)
        # Generation is autoregressive, so wall time tracks emitted token count.
        # Record it: a latency budget is only checkable against the token budget.
        details.append(dict(bar=index, chord=chord, source=source,
                            invalid_model_validation=fallback_reason,
                            ready_ms=(time.perf_counter_ns() - started) / 1e6,
                            generation=metadata))
        for inst in midi.instruments:
            for note in inst.notes:
                lead.notes.append(pretty_midi.Note(note.velocity, note.pitch,
                                  note.start + index * duration, note.end + index * duration))
        root, intervals = parse_chord(chord)
        for interval in intervals:
            backing.notes.append(pretty_midi.Note(45, 48 + root + interval,
                                 index * duration, (index + 1) * duration))
        print(f"bar {index + 1}/{bars}: {source}", flush=True)
    phrase.instruments = [lead, backing]
    return phrase, midis, details


def play_phrase(midis, bpm, port, details=None):
    clock = MonotonicBarClock(bpm=bpm, beats_per_bar=4,
                             start_ns=time.perf_counter_ns() + 1_000_000_000)
    blocks, sequence = {}, 0
    try:
        for index, midi in enumerate(midis):
            source = details[index]["source"] if details else "unknown"
            block = build_scheduled_midi_block(
                midi=midi, clock=clock, bar_index=index, block_id=str(index),
                source_context_id="preloaded", context_version=0,
                adapter=source, fallback_used=source != "model", sequence_start_index=sequence,
            )
            blocks[index] = block
            sequence += len(block.events)
        result = OneBarMidiScheduler(sink=port, clock=clock, spin_window_ms=1).run(
            blocks=blocks, expected_bar_count=len(blocks))
        return dict(run_completed=result.run_completed,
                    completed_bars=result.completed_bar_count,
                    accepted_sends=len(result.records),
                    dispatch_misses=result.scheduler_dispatch_deadline_miss_count,
                    send_failures=result.send_failure_count,
                    output_capture_observed=False)
    finally:
        # Reset on normal completion, scheduler abort, exception, and Ctrl-C.
        try:
            port.reset()
        finally:
            port.panic()


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
    parser.add_argument("--port", help="Exact existing MIDI output name; plays lead only")
    args = parser.parse_args(argv)
    chords = [c.strip() for c in args.chords.split(",") if c.strip()]
    if not 1 <= args.bars <= 64 or not 40 <= args.bpm <= 240 or not chords:
        parser.error("require 1..64 bars, 40..240 BPM, and nonempty chords")
    generate = None
    if not args.fallback_only:
        if not args.checkpoint or not args.conditioning_midi:
            parser.error("provide --checkpoint and --conditioning-midi, or --fallback-only")
        import torch
        from scripts.generate import load_model_with_lora, build_primer, generate_once
        model = load_model_with_lora(lora_path=str(args.checkpoint.parent),
                    checkpoint_path=str(args.checkpoint), prefer_full_checkpoint=True, max_sequence=80)
        primer = build_primer(conditioning_midi=str(args.conditioning_midi),
                    primer_max_tokens=32, append_sep_token=True, control_format="control_v1",
                    role="lead", tempo_bpm=args.bpm)
        def generate(index):
            torch.manual_seed(args.seed + index)
            return generate_once(model=model, primer=primer, target_length=80,
                    strip_primer=True, temperature=1.0, top_k=32, top_p=0.95,
                    grammar_mask=True, target_duration_seconds=240.0 / args.bpm,
                    return_metadata=True)
    phrase, midis, details = prepare_phrase(generate=generate, bpm=args.bpm,
                        bars=args.bars, chords=chords, seed=args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    phrase.write(str(args.output_dir / "with_chords.mid"))
    phrase.instruments = phrase.instruments[:1]
    phrase.write(str(args.output_dir / "lead.mid"))
    report = dict(schema="jazz_preloaded_mvp_v1", bpm=args.bpm, bars=args.bars,
                  target_duration_seconds=args.bars * 240.0 / args.bpm,
                  generation_mode="fixed_primer_independent_bars", bars_detail=details,
                  realtime_generation_verified=False, musical_quality_verified=False,
                  model_chord_conditioning=False, playback=None)
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    if args.port:
        import mido
        try:
            with mido.open_output(args.port) as port:
                report["playback"] = play_phrase(midis, args.bpm, port, details)
                report["playback"]["reset_completed"] = True
        except BaseException as exc:
            report["playback_error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            report_path.write_text(json.dumps(report, indent=2) + "\n")
        if not report["playback"]["run_completed"]:
            return 1
    print(f"MIDI and report: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
