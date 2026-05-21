"""Compare hand-written swing grammar with data-derived motif baseline generation."""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from inference.app.schemas import GenerationRequest  # noqa: E402
from scripts.run_stage_b_generation_probe import (  # noqa: E402
    build_probe_summary,
    build_stage_b_primer,
    chord_aware_pitch_tokens,
    decode_stage_b_midi,
    jazz_rhythm_duration_tokens,
    jazz_rhythm_position_tokens,
    parse_chords,
    pitch_from_token,
    postprocess_stage_b_midi,
    sample_report,
)
from scripts.stage_b_tokens import (  # noqa: E402
    MAX_DURATION_STEPS,
    PIANO_PITCH_MAX,
    PIANO_PITCH_MIN,
    POSITIONS_PER_BAR,
    TOKEN_BAR,
    TOKEN_END,
    chord_tokens,
    note_duration_token,
    note_pitch_token,
    note_velocity_token,
    pitch_from_token as stage_b_pitch_from_token,
    position_token,
)


VALID_BASELINE_MODES = {"hand_written_swing", "data_motif"}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def run_command(cmd: list[str]) -> dict[str, Any]:
    completed = subprocess.run(cmd, cwd=ROOT_DIR, text=True, capture_output=True, check=False)
    return {
        "cmd": cmd,
        "returncode": int(completed.returncode),
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
    }


def run_template_extraction(args: argparse.Namespace, run_dir: Path) -> tuple[Path, dict[str, Any]]:
    template_run_id = f"{args.run_id}_templates"
    output_root = run_dir / "templates"
    cmd = [
        sys.executable,
        "scripts/run_stage_b_motif_template_extraction.py",
        "--run_id",
        template_run_id,
        "--output_root",
        str(output_root),
        "--input_dir",
        str(args.input_dir),
        "--max_files",
        str(args.max_files),
        "--window_bars",
        str(args.window_bars),
        "--window_stride_bars",
        str(args.window_stride_bars),
        "--min_window_target_notes",
        str(args.min_window_target_notes),
        "--motif_length",
        str(args.motif_length),
        "--max_bar_span",
        str(args.max_bar_span),
        "--max_records",
        str(args.max_records),
        "--top_n",
        str(args.template_top_n),
    ]
    result = run_command(cmd)
    report_path = output_root / template_run_id / "motif_template_report.json"
    return report_path, result


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_baseline_modes(raw: str) -> list[str]:
    modes = [mode.strip().lower() for mode in raw.split(",") if mode.strip()]
    invalid = [mode for mode in modes if mode not in VALID_BASELINE_MODES]
    if invalid:
        raise ValueError(f"Unknown baseline modes: {invalid}")
    return modes


def weighted_choice(rows: Sequence[dict[str, Any]], rng: random.Random, index: int) -> dict[str, Any]:
    if not rows:
        raise ValueError("template rows must not be empty")
    weights = [max(1, int(row.get("count", 1))) for row in rows]
    if index < len(rows):
        # Cycle through the highest-ranked rows first so tiny harness runs cover more than one template.
        return rows[index]
    return rng.choices(list(rows), weights=weights, k=1)[0]


def strictly_increasing_positions(raw_positions: Sequence[int], minimum: int, maximum: int) -> list[int]:
    result: list[int] = []
    previous = int(minimum) - 1
    for raw in raw_positions:
        value = max(int(minimum), min(int(maximum), int(raw)))
        if value <= previous:
            value = previous + 1
        if value > int(maximum):
            break
        result.append(value)
        previous = value
    return result


def normalize_position_deltas(
    position_deltas: Sequence[int],
    *,
    slot_start: int,
    slot_size: int,
) -> list[int]:
    deltas = [max(0, int(delta)) for delta in position_deltas]
    if not deltas:
        return []
    local_max = max(deltas)
    slot_size = max(1, int(slot_size))
    slot_end = min(int(POSITIONS_PER_BAR) - 1, int(slot_start) + slot_size - 1)
    if local_max <= 0:
        raw_positions = [int(slot_start) + index for index in range(len(deltas))]
    else:
        raw_positions = [
            int(slot_start) + int(round(delta * (slot_size - 1) / local_max))
            for delta in deltas
        ]
    return strictly_increasing_positions(raw_positions, int(slot_start), slot_end)


def duration_tokens_from_steps(duration_steps: Sequence[int], target_count: int) -> list[int]:
    steps = [max(1, min(int(MAX_DURATION_STEPS), int(step))) for step in duration_steps]
    if not steps:
        steps = [2]
    while len(steps) < int(target_count):
        steps.append(steps[-1])
    return [note_duration_token(step) for step in steps[: int(target_count)]]


def fit_duration_tokens_to_positions(
    positions: Sequence[int],
    duration_steps: Sequence[int],
    max_tail_duration: int = 4,
) -> list[int]:
    raw_steps = [max(1, min(int(MAX_DURATION_STEPS), int(step))) for step in duration_steps]
    if not raw_steps:
        raw_steps = [2]
    while len(raw_steps) < len(positions):
        raw_steps.append(raw_steps[-1])

    fitted: list[int] = []
    for index, position in enumerate(positions):
        if index + 1 < len(positions):
            max_duration = max(1, int(positions[index + 1]) - int(position))
        else:
            max_duration = max(1, min(int(max_tail_duration), int(POSITIONS_PER_BAR) - int(position)))
        fitted.append(max(1, min(raw_steps[index], max_duration)))
    return [note_duration_token(step) for step in fitted]


def nearest_allowed_pitch_token(
    target_pitch: int,
    allowed_tokens: Sequence[int],
    recent_pitches: Sequence[int] | None = None,
) -> int:
    if not allowed_tokens:
        return note_pitch_token(max(int(PIANO_PITCH_MIN), min(int(PIANO_PITCH_MAX), int(target_pitch))))
    blocked = set(int(pitch) for pitch in (recent_pitches or [])[-2:])
    candidates = [int(token) for token in allowed_tokens if stage_b_pitch_from_token(token) not in blocked]
    if not candidates:
        candidates = [int(token) for token in allowed_tokens]
    return min(
        candidates,
        key=lambda token: (
            abs(stage_b_pitch_from_token(token) - int(target_pitch)),
            abs(stage_b_pitch_from_token(token) - 67),
            stage_b_pitch_from_token(token),
        ),
    )


def base_pitch_token_for_chord(chord: str | None, rng: random.Random, recent_pitches: Sequence[int]) -> int:
    allowed = chord_aware_pitch_tokens(
        chord,
        pitch_mode="tones_tensions",
        recent_pitches=recent_pitches,
        repeat_window=2,
    )
    target = 64 + rng.choice([-5, -2, 0, 2, 5])
    return nearest_allowed_pitch_token(target, allowed, recent_pitches)


def pitch_tokens_from_contour(
    chord: str | None,
    pitch_intervals: Sequence[int],
    *,
    rng: random.Random,
    recent_pitches: list[int],
    group_offset: int,
) -> list[int]:
    start_token = base_pitch_token_for_chord(chord, rng, recent_pitches)
    start_pitch = pitch_from_token(start_token)
    tokens: list[int] = []
    intervals = [int(interval) for interval in pitch_intervals] or [0]
    for index, interval in enumerate(intervals):
        target_pitch = start_pitch + int(interval)
        allowed = chord_aware_pitch_tokens(
            chord,
            pitch_mode="approach_tensions",
            recent_pitches=recent_pitches,
            repeat_window=2,
            group_index=group_offset + index,
        )
        token = nearest_allowed_pitch_token(target_pitch, allowed, recent_pitches)
        tokens.append(token)
        recent_pitches.append(pitch_from_token(token))
    return tokens


def hand_written_swing_tokens(
    *,
    primer_tokens: Sequence[int],
    chords: Sequence[str],
    bars: int,
    note_groups_per_bar: int,
    seed: int,
) -> list[int]:
    rng = random.Random(int(seed))
    tokens = [int(token) for token in primer_tokens]
    recent_pitches: list[int] = []
    contour = [0, 2, 5, 3, 7, 5, 9, 7]
    for bar_index in range(max(1, int(bars))):
        chord = chords[bar_index % len(chords)] if chords else None
        if bar_index > 0:
            tokens.append(TOKEN_BAR)
            tokens.extend(chord_tokens(chord))
        base_token = base_pitch_token_for_chord(chord, rng, recent_pitches)
        base_pitch = pitch_from_token(base_token)
        for group_index in range(max(1, int(note_groups_per_bar))):
            position_candidates = jazz_rhythm_position_tokens(
                bar_index=bar_index,
                group_index=group_index,
                note_groups_per_bar=note_groups_per_bar,
                profile="swing_motif",
            )
            duration_candidates = jazz_rhythm_duration_tokens(
                bar_index=bar_index,
                group_index=group_index,
                note_groups_per_bar=note_groups_per_bar,
                profile="swing_motif",
            )
            target_pitch = base_pitch + contour[group_index % len(contour)]
            pitch_candidates = chord_aware_pitch_tokens(
                chord,
                pitch_mode="approach_tensions",
                recent_pitches=recent_pitches,
                repeat_window=2,
                group_index=group_index,
            )
            pitch_token_value = nearest_allowed_pitch_token(target_pitch, pitch_candidates, recent_pitches)
            recent_pitches.append(pitch_from_token(pitch_token_value))
            tokens.extend(
                [
                    position_candidates[len(position_candidates) // 2],
                    note_velocity_token(4),
                    pitch_token_value,
                    duration_candidates[len(duration_candidates) // 2],
                ]
            )
    tokens.append(TOKEN_END)
    return tokens


def data_motif_tokens(
    *,
    primer_tokens: Sequence[int],
    chords: Sequence[str],
    bars: int,
    note_groups_per_bar: int,
    template_report: dict[str, Any],
    seed: int,
) -> list[int]:
    rng = random.Random(int(seed))
    summary = template_report["summary"]
    rhythm_rows = summary["top_rhythm_templates"]
    contour_rows = summary["top_contour_templates"]
    tokens = [int(token) for token in primer_tokens]
    recent_pitches: list[int] = []
    motif_length = 4
    motifs_per_bar = max(1, int(round(max(1, int(note_groups_per_bar)) / motif_length)))
    slot_size = max(1, int(POSITIONS_PER_BAR) // motifs_per_bar)

    for bar_index in range(max(1, int(bars))):
        chord = chords[bar_index % len(chords)] if chords else None
        if bar_index > 0:
            tokens.append(TOKEN_BAR)
            tokens.extend(chord_tokens(chord))
        emitted_in_bar = 0
        for motif_index in range(motifs_per_bar):
            row_index = bar_index * motifs_per_bar + motif_index
            rhythm = weighted_choice(rhythm_rows, rng, row_index)["key"]
            contour = weighted_choice(contour_rows, rng, row_index + int(seed))["key"]
            slot_start = min(int(POSITIONS_PER_BAR) - 1, motif_index * slot_size)
            positions = normalize_position_deltas(
                rhythm["position_deltas"],
                slot_start=slot_start,
                slot_size=slot_size,
            )
            durations = fit_duration_tokens_to_positions(positions, rhythm["duration_steps"])
            pitch_tokens = pitch_tokens_from_contour(
                chord,
                contour["pitch_intervals"],
                rng=rng,
                recent_pitches=recent_pitches,
                group_offset=emitted_in_bar,
            )
            for position, duration_token, pitch_token_value in zip(positions, durations, pitch_tokens):
                if emitted_in_bar >= int(note_groups_per_bar):
                    break
                tokens.extend(
                    [
                        position_token(position),
                        note_velocity_token(4),
                        pitch_token_value,
                        duration_token,
                    ]
                )
                emitted_in_bar += 1
    tokens.append(TOKEN_END)
    return tokens


def generated_tokens_for_mode(
    mode: str,
    *,
    primer_tokens: Sequence[int],
    chords: Sequence[str],
    bars: int,
    note_groups_per_bar: int,
    template_report: dict[str, Any],
    seed: int,
) -> list[int]:
    if mode == "hand_written_swing":
        return hand_written_swing_tokens(
            primer_tokens=primer_tokens,
            chords=chords,
            bars=bars,
            note_groups_per_bar=note_groups_per_bar,
            seed=seed,
        )
    if mode == "data_motif":
        return data_motif_tokens(
            primer_tokens=primer_tokens,
            chords=chords,
            bars=bars,
            note_groups_per_bar=note_groups_per_bar,
            template_report=template_report,
            seed=seed,
        )
    raise ValueError(f"unknown baseline mode: {mode}")


def compact_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_count": int(summary["sample_count"]),
        "valid_sample_count": int(summary["valid_sample_count"]),
        "strict_valid_sample_count": int(summary["strict_valid_sample_count"]),
        "avg_syncopated_onset_ratio": float(summary["avg_syncopated_onset_ratio"]),
        "avg_unique_bar_position_pattern_ratio": float(summary["avg_unique_bar_position_pattern_ratio"]),
        "avg_duration_diversity_ratio": float(summary["avg_duration_diversity_ratio"]),
        "avg_most_common_duration_ratio": float(summary["avg_most_common_duration_ratio"]),
        "avg_ioi_diversity_ratio": float(summary["avg_ioi_diversity_ratio"]),
        "avg_most_common_ioi_ratio": float(summary["avg_most_common_ioi_ratio"]),
        "avg_tension_ratio": float(summary["avg_tension_ratio"]),
        "avg_root_tone_ratio": float(summary["avg_root_tone_ratio"]),
        "passed_strict_review_gate": bool(summary["passed_strict_review_gate"]),
    }


def build_compare_summary(
    mode_summaries: dict[str, dict[str, Any]],
    min_strict_valid_samples: int,
) -> dict[str, Any]:
    hand = mode_summaries.get("hand_written_swing")
    data = mode_summaries.get("data_motif")
    ready = bool(hand and data)
    data_passed = bool(data and int(data["strict_valid_sample_count"]) >= int(min_strict_valid_samples))
    hand_passed = bool(hand and int(hand["strict_valid_sample_count"]) >= int(min_strict_valid_samples))
    def delta(metric: str) -> float:
        return float(data.get(metric, 0.0) - hand.get(metric, 0.0)) if ready and data and hand else 0.0

    return {
        "comparison_ready": ready,
        "passed_hand_written_swing_gate": hand_passed,
        "passed_data_motif_gate": data_passed,
        "passed_compare_gate": bool(ready and hand_passed and data_passed),
        "duration_diversity_delta_data_minus_hand": delta("avg_duration_diversity_ratio"),
        "ioi_diversity_delta_data_minus_hand": delta("avg_ioi_diversity_ratio"),
        "bar_pattern_delta_data_minus_hand": delta("avg_unique_bar_position_pattern_ratio"),
        "syncopation_delta_data_minus_hand": delta("avg_syncopated_onset_ratio"),
        "mode_summaries": {mode: compact_summary(summary) for mode, summary in sorted(mode_summaries.items())},
    }


def markdown_report(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Stage B Data Motif Generation Compare",
        "",
        f"- passed compare gate: `{str(summary['passed_compare_gate']).lower()}`",
        f"- duration diversity delta, data minus hand: `{summary['duration_diversity_delta_data_minus_hand']:.3f}`",
        f"- IOI diversity delta, data minus hand: `{summary['ioi_diversity_delta_data_minus_hand']:.3f}`",
        f"- bar-pattern delta, data minus hand: `{summary['bar_pattern_delta_data_minus_hand']:.3f}`",
        f"- syncopation delta, data minus hand: `{summary['syncopation_delta_data_minus_hand']:.3f}`",
        "",
        "| mode | samples | strict | sync | bar-var | dur-var | dur-rep | ioi-var | ioi-rep | tension | root |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode, row in sorted(summary["mode_summaries"].items()):
        lines.append(
            "| {mode} | {sample_count} | {strict_valid_sample_count} | "
            "{avg_syncopated_onset_ratio:.3f} | {avg_unique_bar_position_pattern_ratio:.3f} | "
            "{avg_duration_diversity_ratio:.3f} | {avg_most_common_duration_ratio:.3f} | "
            "{avg_ioi_diversity_ratio:.3f} | {avg_most_common_ioi_ratio:.3f} | "
            "{avg_tension_ratio:.3f} | {avg_root_tone_ratio:.3f} |".format(mode=mode, **row)
        )
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Stage B data-derived motif baseline generation compare")
    parser.add_argument("--output_root", type=str, default=str(ROOT_DIR / "outputs" / "stage_b_data_motif_compare"))
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--input_dir", type=str, default="./midi_dataset/midi/studio")
    parser.add_argument("--issue_number", type=int, default=65)
    parser.add_argument("--baseline_modes", type=str, default="hand_written_swing,data_motif")
    parser.add_argument("--max_files", type=int, default=4)
    parser.add_argument("--window_bars", type=int, default=8)
    parser.add_argument("--window_stride_bars", type=int, default=4)
    parser.add_argument("--min_window_target_notes", type=int, default=16)
    parser.add_argument("--motif_length", type=int, default=4)
    parser.add_argument("--max_bar_span", type=int, default=2)
    parser.add_argument("--max_records", type=int, default=64)
    parser.add_argument("--template_top_n", type=int, default=32)
    parser.add_argument("--bpm", type=int, default=124)
    parser.add_argument("--bars", type=int, default=8)
    parser.add_argument("--chords", type=str, default="Cm7,Fm7,Bb7,Ebmaj7")
    parser.add_argument("--density", type=str, default="medium")
    parser.add_argument("--energy", type=str, default="mid")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=2)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--num_samples", type=int, default=3)
    parser.add_argument("--note_groups_per_bar", type=int, default=8)
    parser.add_argument("--max_sequence", type=int, default=384)
    parser.add_argument("--max_simultaneous_notes", type=int, default=2)
    parser.add_argument("--min_strict_valid_samples", type=int, default=1)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_root) / args.run_id
    samples_dir = run_dir / "samples"
    modes = parse_baseline_modes(args.baseline_modes)
    chords = parse_chords(args.chords)
    request = GenerationRequest(
        bpm=int(args.bpm),
        chord_progression=chords,
        bars=int(args.bars),
        density=args.density,
        energy=args.energy,
        temperature=float(args.temperature),
        top_k=int(args.top_k),
        seed=int(args.seed),
    )
    request.validate()

    template_report_path, template_command = run_template_extraction(args, run_dir)
    report: dict[str, Any] = {
        "run_id": args.run_id,
        "run_dir": str(run_dir),
        "issue": int(args.issue_number),
        "baseline_modes": modes,
        "template_report_path": str(template_report_path),
        "template_command": template_command,
        "chords": chords,
        "bars": int(args.bars),
        "note_groups_per_bar": int(args.note_groups_per_bar),
        "samples": {},
    }
    if template_command["returncode"] != 0:
        report["failure_reason"] = "motif template extraction failed"
        write_json(run_dir / "data_motif_compare_report.json", report)
        print(json.dumps(report, ensure_ascii=True, indent=2))
        return int(template_command["returncode"])

    template_report = read_json(template_report_path)
    primer_tokens = build_stage_b_primer(chords, args.bpm)
    mode_summaries: dict[str, dict[str, Any]] = {}
    for mode in modes:
        rows: list[dict[str, Any]] = []
        for index in range(1, int(args.num_samples) + 1):
            sample_seed = int(args.seed) + index - 1
            tokens = generated_tokens_for_mode(
                mode,
                primer_tokens=primer_tokens,
                chords=chords,
                bars=int(args.bars),
                note_groups_per_bar=int(args.note_groups_per_bar),
                template_report=template_report,
                seed=sample_seed,
            )[: int(args.max_sequence)]
            midi_path = samples_dir / mode / f"{mode}_sample_{index}.mid"
            midi_path.parent.mkdir(parents=True, exist_ok=True)
            midi = decode_stage_b_midi(tokens, tempo_bpm=args.bpm)
            postprocess_report = postprocess_stage_b_midi(
                midi,
                simultaneous_limit=int(args.max_simultaneous_notes),
            )
            midi.write(str(midi_path))
            rows.append(
                sample_report(
                    sample_index=index,
                    sample_seed=sample_seed,
                    tokens=tokens,
                    primer_size=len(primer_tokens),
                    target_length=int(args.max_sequence),
                    midi_path=midi_path,
                    request=request,
                    postprocess_report=postprocess_report,
                )
            )
        summary = build_probe_summary(
            rows,
            min_valid_samples=1,
            min_strict_valid_samples=int(args.min_strict_valid_samples),
            require_all_grammar_samples=True,
        )
        report["samples"][mode] = rows
        mode_summaries[mode] = summary

    report["summary"] = build_compare_summary(
        mode_summaries,
        min_strict_valid_samples=int(args.min_strict_valid_samples),
    )
    write_json(run_dir / "data_motif_compare_report.json", report)
    (run_dir / "data_motif_compare_report.md").write_text(markdown_report(report), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=True, indent=2))
    return 0 if report["summary"]["passed_compare_gate"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
