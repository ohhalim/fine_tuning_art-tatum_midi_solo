"""Build reference phrase statistics from real Stage B MIDI windows."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from scripts.run_stage_b_generation_probe import (  # noqa: E402
    analyze_stage_b_collapse,
    analyze_stage_b_phrase_contour,
    analyze_stage_b_rhythm_profile,
    analyze_stage_b_temporal_coverage,
    chord_approach_pitch_classes,
    chord_pitch_classes,
    chord_root_pitch_class,
    extract_stage_b_note_groups,
)
from scripts.stage_b_tokens import (  # noqa: E402
    CHORD_QUALITIES,
    CHORD_ROOTS,
    ROOT_TO_PC,
    TOKEN_BAR,
    TOKEN_CHORD_QUALITY_START,
    TOKEN_CHORD_ROOT_START,
    TOKEN_END,
    parse_chord_symbol,
)


REFERENCE_METRIC_KEYS = [
    "note_group_count",
    "unique_pitch_count",
    "pitch_span",
    "repeated_pitch_ratio",
    "syncopated_onset_ratio",
    "unique_bar_position_pattern_ratio",
    "duration_diversity_ratio",
    "most_common_duration_ratio",
    "ioi_diversity_ratio",
    "most_common_ioi_ratio",
    "direction_change_ratio",
    "stepwise_motion_ratio",
    "leap_motion_ratio",
    "onset_coverage_ratio",
    "sustained_coverage_ratio",
]

PITCH_ROLE_KEYS = ["root", "guide", "chord", "tension", "approach", "outside", "unknown_chord"]
PITCH_ROLE_RATIO_KEYS = [
    "root_tone_ratio",
    "guide_tone_ratio",
    "chord_tone_ratio",
    "non_root_chord_tone_ratio",
    "tension_ratio",
    "approach_ratio",
    "outside_ratio",
    "unknown_chord_ratio",
]

QUALITY_SUFFIX = {
    "maj": "",
    "maj7": "maj7",
    "min": "m",
    "min7": "m7",
    "dom7": "7",
    "dim": "dim",
    "halfdim": "m7b5",
    "sus": "sus",
    "unknown": "",
}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_command(cmd: list[str]) -> dict[str, Any]:
    completed = subprocess.run(cmd, cwd=ROOT_DIR, text=True, capture_output=True, check=False)
    return {
        "cmd": cmd,
        "returncode": int(completed.returncode),
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
    }


def run_prepare_command(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    cmd = [
        sys.executable,
        "scripts/prepare_role_dataset.py",
        "--input_dir",
        str(args.input_dir),
        "--output_dir",
        str(output_dir),
        "--role",
        str(args.role),
        "--sequence_format",
        "stage_b_v1",
        "--stage_b_window_bars",
        str(args.window_bars),
        "--stage_b_window_stride_bars",
        str(args.window_stride_bars),
        "--stage_b_min_window_target_notes",
        str(args.min_window_target_notes),
        "--overwrite",
    ]
    if args.max_files is not None:
        cmd.extend(["--max_files", str(args.max_files)])
    return run_command(cmd)


def percentile(values: list[float], ratio: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * float(ratio)))))
    return float(ordered[index])


def compact_metric_stats(values: Iterable[float]) -> dict[str, float]:
    vals = [float(value) for value in values]
    if not vals:
        return {"count": 0.0, "mean": 0.0, "min": 0.0, "p25": 0.0, "p50": 0.0, "p75": 0.0, "max": 0.0}
    return {
        "count": float(len(vals)),
        "mean": float(mean(vals)),
        "min": float(min(vals)),
        "p25": percentile(vals, 0.25),
        "p50": percentile(vals, 0.50),
        "p75": percentile(vals, 0.75),
        "max": float(max(vals)),
    }


def chord_symbol_from_parts(root: str | None, quality: str | None) -> str | None:
    if not root or root == "N":
        return None
    normalized_quality = quality if quality in QUALITY_SUFFIX else "unknown"
    return f"{root}{QUALITY_SUFFIX[normalized_quality]}"


def extract_bar_chords(tokens: list[int]) -> dict[int, str | None]:
    bar_index = -1
    pending_root: str | None = None
    pending_quality: str | None = None
    bar_chords: dict[int, str | None] = {}

    for raw_token in tokens:
        token = int(raw_token)
        if token == TOKEN_END:
            break
        if token == TOKEN_BAR:
            if bar_index >= 0:
                bar_chords[bar_index] = chord_symbol_from_parts(pending_root, pending_quality)
            bar_index += 1
            pending_root = None
            pending_quality = None
            continue
        if TOKEN_CHORD_ROOT_START <= token < TOKEN_CHORD_ROOT_START + len(CHORD_ROOTS):
            pending_root = CHORD_ROOTS[token - TOKEN_CHORD_ROOT_START]
            continue
        if TOKEN_CHORD_QUALITY_START <= token < TOKEN_CHORD_QUALITY_START + len(CHORD_QUALITIES):
            pending_quality = CHORD_QUALITIES[token - TOKEN_CHORD_QUALITY_START]

    if bar_index >= 0:
        bar_chords[bar_index] = chord_symbol_from_parts(pending_root, pending_quality)
    return bar_chords


def guide_tone_pitch_classes(chord: str | None) -> set[int]:
    root, quality = parse_chord_symbol(chord)
    root_pc = ROOT_TO_PC.get(root)
    if root_pc is None:
        return set()
    guide_intervals_by_quality = {
        "maj": {4, 7},
        "maj7": {4, 11},
        "min": {3, 7},
        "min7": {3, 10},
        "dom7": {4, 10},
        "dim": {3, 6},
        "halfdim": {3, 10},
        "sus": {5, 10},
        "unknown": {4, 7},
    }
    intervals = guide_intervals_by_quality.get(quality, guide_intervals_by_quality["unknown"])
    return {(int(root_pc) + int(interval)) % 12 for interval in intervals}


def position_bucket(position: int) -> str:
    value = int(position)
    if value % 4 == 0:
        return "strong"
    if value % 2 == 0:
        return "eighth"
    return "offgrid"


def pitch_role_for_group(group: dict[str, int], chord: str | None) -> str:
    if not chord:
        return "unknown_chord"
    pitch_class = int(group["pitch"]) % 12
    root_pc = chord_root_pitch_class(chord)
    guide_pcs = guide_tone_pitch_classes(chord)
    chord_pcs = chord_pitch_classes(chord, pitch_mode="tones")
    tension_pcs = chord_pitch_classes(chord, pitch_mode="tones_tensions") - chord_pcs
    approach_pcs = chord_approach_pitch_classes(chord)

    if root_pc is not None and pitch_class == root_pc:
        return "root"
    if pitch_class in guide_pcs:
        return "guide"
    if pitch_class in chord_pcs:
        return "chord"
    if pitch_class in tension_pcs:
        return "tension"
    if pitch_class in approach_pcs:
        return "approach"
    return "outside"


def _empty_role_counts() -> dict[str, int]:
    return {role: 0 for role in PITCH_ROLE_KEYS}


def _role_ratios(counts: dict[str, int]) -> dict[str, float]:
    total = sum(int(counts.get(role, 0)) for role in PITCH_ROLE_KEYS)
    if total <= 0:
        return {role: 0.0 for role in PITCH_ROLE_KEYS}
    return {role: float(counts.get(role, 0) / total) for role in PITCH_ROLE_KEYS}


def pitch_role_ratio_metrics(role_counts: dict[str, int]) -> dict[str, float]:
    total = sum(int(role_counts.get(role, 0)) for role in PITCH_ROLE_KEYS)
    if total <= 0:
        return {key: 0.0 for key in PITCH_ROLE_RATIO_KEYS}
    root = int(role_counts.get("root", 0))
    guide = int(role_counts.get("guide", 0))
    chord = int(role_counts.get("chord", 0))
    tension = int(role_counts.get("tension", 0))
    approach = int(role_counts.get("approach", 0))
    outside = int(role_counts.get("outside", 0))
    unknown = int(role_counts.get("unknown_chord", 0))
    chord_total = root + guide + chord
    return {
        "root_tone_ratio": float(root / total),
        "guide_tone_ratio": float(guide / total),
        "chord_tone_ratio": float(chord_total / total),
        "non_root_chord_tone_ratio": float((guide + chord) / total),
        "tension_ratio": float(tension / total),
        "approach_ratio": float(approach / total),
        "outside_ratio": float(outside / total),
        "unknown_chord_ratio": float(unknown / total),
    }


def analyze_stage_b_pitch_role_landings(tokens: list[int]) -> dict[str, Any]:
    groups = extract_stage_b_note_groups(tokens, primer_size=0)
    bar_chords = extract_bar_chords(tokens)
    role_counts = _empty_role_counts()
    bucket_counts: dict[str, dict[str, int]] = {
        "strong": _empty_role_counts(),
        "eighth": _empty_role_counts(),
        "offgrid": _empty_role_counts(),
    }
    position_mod4_counts: dict[str, dict[str, int]] = {str(index): _empty_role_counts() for index in range(4)}
    known_chord_notes = 0

    for group in groups:
        group_bar = int(group["bar"])
        chord = bar_chords.get(group_bar)
        if chord is None and (group_bar - 1) in bar_chords:
            chord = bar_chords.get(group_bar - 1)
        role = pitch_role_for_group(group, chord)
        role_counts[role] += 1
        bucket = position_bucket(int(group["position"]))
        bucket_counts[bucket][role] += 1
        position_mod4_counts[str(int(group["position"]) % 4)][role] += 1
        if chord:
            known_chord_notes += 1

    return {
        "note_group_count": int(len(groups)),
        "known_chord_note_count": int(known_chord_notes),
        "known_chord_note_ratio": float(known_chord_notes / len(groups)) if groups else 0.0,
        "bar_chords": {str(bar): chord for bar, chord in sorted(bar_chords.items())},
        "role_counts": role_counts,
        "role_ratios": _role_ratios(role_counts),
        "cumulative_ratios": pitch_role_ratio_metrics(role_counts),
        "bucket_counts": bucket_counts,
        "bucket_ratios": {bucket: _role_ratios(counts) for bucket, counts in bucket_counts.items()},
        "position_mod4_counts": position_mod4_counts,
        "position_mod4_ratios": {
            bucket: _role_ratios(counts)
            for bucket, counts in position_mod4_counts.items()
        },
    }


def analyze_token_record(tokens: list[int], bars: int) -> dict[str, Any]:
    groups = extract_stage_b_note_groups(tokens, primer_size=0)
    collapse = analyze_stage_b_collapse(tokens, primer_size=0)
    rhythm = analyze_stage_b_rhythm_profile(tokens, primer_size=0)
    contour = analyze_stage_b_phrase_contour(tokens, primer_size=0)
    temporal = analyze_stage_b_temporal_coverage(tokens, primer_size=0, bars=bars)
    pitch_role_landings = analyze_stage_b_pitch_role_landings(tokens)
    pitches = [int(group["pitch"]) for group in groups]

    metrics = {
        "note_group_count": float(len(groups)),
        "unique_pitch_count": float(len(set(pitches))),
        "pitch_span": float(max(pitches) - min(pitches)) if pitches else 0.0,
        "repeated_pitch_ratio": float(collapse.get("repeated_pitch_ratio", 0.0) or 0.0),
        "syncopated_onset_ratio": float(rhythm.get("syncopated_onset_ratio", 0.0) or 0.0),
        "unique_bar_position_pattern_ratio": float(
            rhythm.get("unique_bar_position_pattern_ratio", 0.0) or 0.0
        ),
        "duration_diversity_ratio": float(rhythm.get("duration_diversity_ratio", 0.0) or 0.0),
        "most_common_duration_ratio": float(rhythm.get("most_common_duration_ratio", 0.0) or 0.0),
        "ioi_diversity_ratio": float(rhythm.get("ioi_diversity_ratio", 0.0) or 0.0),
        "most_common_ioi_ratio": float(rhythm.get("most_common_ioi_ratio", 0.0) or 0.0),
        "direction_change_ratio": float(contour.get("direction_change_ratio", 0.0) or 0.0),
        "stepwise_motion_ratio": float(contour.get("stepwise_motion_ratio", 0.0) or 0.0),
        "leap_motion_ratio": float(contour.get("leap_motion_ratio", 0.0) or 0.0),
        "onset_coverage_ratio": float(temporal.get("onset_coverage_ratio", 0.0) or 0.0),
        "sustained_coverage_ratio": float(temporal.get("sustained_coverage_ratio", 0.0) or 0.0),
    }
    return {
        "metrics": metrics,
        "rhythm_profile": rhythm,
        "phrase_contour": contour,
        "temporal_coverage": temporal,
        "collapse": collapse,
        "pitch_role_landings": pitch_role_landings,
    }


def iter_token_files(tokenized_dir: Path) -> list[Path]:
    return sorted(tokenized_dir.glob("*/*.npy"))


def build_reference_rows(tokenized_dir: Path, bars: int, max_records: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, path in enumerate(iter_token_files(tokenized_dir), start=1):
        if max_records is not None and len(rows) >= int(max_records):
            break
        tokens = [int(token) for token in np.load(path).tolist()]
        if not tokens:
            continue
        analysis = analyze_token_record(tokens, bars=bars)
        rows.append(
            {
                "record_index": int(index),
                "relative_path": str(path.relative_to(tokenized_dir)),
                **analysis,
            }
        )
    return rows


def summarize_reference_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "record_count": int(len(rows)),
        "metric_keys": REFERENCE_METRIC_KEYS,
        "metrics": {
            key: compact_metric_stats(
                row.get("metrics", {})[key]
                for row in rows
                if key in row.get("metrics", {})
            )
            for key in REFERENCE_METRIC_KEYS
        },
        "pitch_role_landing": summarize_pitch_role_landings(rows),
    }


def merge_role_counts(rows: list[dict[str, Any]], key_path: tuple[str, ...]) -> dict[str, int]:
    counts = _empty_role_counts()
    for row in rows:
        value: Any = row
        for key in key_path:
            value = value.get(key, {}) if isinstance(value, dict) else {}
        if not isinstance(value, dict):
            continue
        for role in PITCH_ROLE_KEYS:
            counts[role] += int(value.get(role, 0) or 0)
    return counts


def summarize_pitch_role_landings(rows: list[dict[str, Any]]) -> dict[str, Any]:
    role_counts = merge_role_counts(rows, ("pitch_role_landings", "role_counts"))
    bucket_counts = {
        bucket: merge_role_counts(rows, ("pitch_role_landings", "bucket_counts", bucket))
        for bucket in ("strong", "eighth", "offgrid")
    }
    position_mod4_counts = {
        str(index): merge_role_counts(rows, ("pitch_role_landings", "position_mod4_counts", str(index)))
        for index in range(4)
    }
    note_count = sum(int(role_counts.get(role, 0)) for role in PITCH_ROLE_KEYS)
    known_count = sum(
        int(row.get("pitch_role_landings", {}).get("known_chord_note_count", 0) or 0)
        for row in rows
    )
    return {
        "note_group_count": int(note_count),
        "known_chord_note_count": int(known_count),
        "known_chord_note_ratio": float(known_count / note_count) if note_count else 0.0,
        "role_counts": role_counts,
        "role_ratios": _role_ratios(role_counts),
        "cumulative_ratios": pitch_role_ratio_metrics(role_counts),
        "bucket_counts": bucket_counts,
        "bucket_ratios": {bucket: _role_ratios(counts) for bucket, counts in bucket_counts.items()},
        "position_mod4_counts": position_mod4_counts,
        "position_mod4_ratios": {
            bucket: _role_ratios(counts)
            for bucket, counts in position_mod4_counts.items()
        },
    }


def _mean(values: list[float]) -> float:
    return float(mean(values)) if values else 0.0


def generated_rows_from_samples(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for mode, samples in sorted(report.get("samples", {}).items()):
        if not isinstance(samples, list) or not samples:
            continue
        rhythm_rows = [sample.get("rhythm_profile", {}) for sample in samples]
        pitch_rows = [sample.get("pitch_roles", {}) for sample in samples]
        rows.append(
            {
                "label": str(mode),
                "metrics": {
                    "syncopated_onset_ratio": _mean(
                        [float(row.get("syncopated_onset_ratio", 0.0) or 0.0) for row in rhythm_rows]
                    ),
                    "unique_bar_position_pattern_ratio": _mean(
                        [float(row.get("unique_bar_position_pattern_ratio", 0.0) or 0.0) for row in rhythm_rows]
                    ),
                    "duration_diversity_ratio": _mean(
                        [float(row.get("duration_diversity_ratio", 0.0) or 0.0) for row in rhythm_rows]
                    ),
                    "most_common_duration_ratio": _mean(
                        [float(row.get("most_common_duration_ratio", 0.0) or 0.0) for row in rhythm_rows]
                    ),
                    "ioi_diversity_ratio": _mean(
                        [float(row.get("ioi_diversity_ratio", 0.0) or 0.0) for row in rhythm_rows]
                    ),
                    "most_common_ioi_ratio": _mean(
                        [float(row.get("most_common_ioi_ratio", 0.0) or 0.0) for row in rhythm_rows]
                    ),
                    "root_tone_ratio": _mean(
                        [float(row.get("root_tone_ratio", 0.0) or 0.0) for row in pitch_rows]
                    ),
                    "chord_tone_ratio": _mean(
                        [float(row.get("chord_tone_ratio", 0.0) or 0.0) for row in pitch_rows]
                    ),
                    "non_root_chord_tone_ratio": _mean(
                        [float(row.get("non_root_chord_tone_ratio", 0.0) or 0.0) for row in pitch_rows]
                    ),
                    "tension_ratio": _mean(
                        [float(row.get("tension_ratio", 0.0) or 0.0) for row in pitch_rows]
                    ),
                    "non_chord_tone_ratio": _mean(
                        [float(row.get("non_chord_tone_ratio", 0.0) or 0.0) for row in pitch_rows]
                    ),
                },
            }
        )
    return rows


def generated_rows_from_report(report: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(report.get("samples"), dict):
        return generated_rows_from_samples(report)
    rows: list[dict[str, Any]] = []
    for row in report.get("rows", []):
        rows.append(
            {
                "label": str(row.get("grammar_mode") or row.get("run_id") or "generated"),
                "metrics": {
                    "syncopated_onset_ratio": float(row.get("avg_syncopated_onset_ratio", 0.0) or 0.0),
                    "unique_bar_position_pattern_ratio": float(
                        row.get("avg_unique_bar_position_pattern_ratio", 0.0) or 0.0
                    ),
                    "duration_diversity_ratio": float(row.get("avg_duration_diversity_ratio", 0.0) or 0.0),
                    "most_common_duration_ratio": float(row.get("avg_most_common_duration_ratio", 0.0) or 0.0),
                    "ioi_diversity_ratio": float(row.get("avg_ioi_diversity_ratio", 0.0) or 0.0),
                    "most_common_ioi_ratio": float(row.get("avg_most_common_ioi_ratio", 0.0) or 0.0),
                },
            }
        )
    return rows


def compare_generated_to_reference(
    generated_rows: list[dict[str, Any]],
    reference_summary: dict[str, Any],
    include_pitch_roles: bool | None = None,
) -> list[dict[str, Any]]:
    reference_metrics = dict(reference_summary.get("metrics", {}))
    pitch_summary = reference_summary.get("pitch_role_landing", {})
    if include_pitch_roles is None:
        include_pitch_roles = float(pitch_summary.get("known_chord_note_ratio", 0.0) or 0.0) >= 0.5
    if include_pitch_roles:
        pitch_reference_metrics = pitch_summary.get("cumulative_ratios", {})
        for key, value in pitch_reference_metrics.items():
            reference_metrics[key] = {"mean": float(value)}
    comparisons: list[dict[str, Any]] = []
    for row in generated_rows:
        metrics = row.get("metrics", {})
        deltas: dict[str, float] = {}
        for key, value in metrics.items():
            if key not in reference_metrics:
                continue
            reference_mean = float(reference_metrics[key].get("mean", 0.0) or 0.0)
            deltas[key] = float(value) - reference_mean
        comparisons.append({"label": row.get("label"), "metrics": metrics, "delta_from_reference_mean": deltas})
    return comparisons


def markdown_report(report: dict[str, Any]) -> str:
    summary = report["reference_summary"]
    lines = [
        "# Stage B Reference Phrase Statistics",
        "",
        f"- record count: `{summary['record_count']}`",
        f"- input dir: `{report.get('input_dir')}`",
        f"- tokenized dir: `{report.get('tokenized_dir')}`",
        "",
        "| metric | mean | p25 | p50 | p75 | min | max |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key in REFERENCE_METRIC_KEYS:
        stats = summary["metrics"][key]
        lines.append(
            "| {key} | {mean:.3f} | {p25:.3f} | {p50:.3f} | {p75:.3f} | {min:.3f} | {max:.3f} |".format(
                key=key,
                **stats,
            )
        )
    pitch_summary = summary.get("pitch_role_landing", {})
    if pitch_summary:
        lines.extend(
            [
                "",
                "## Pitch Role Landing",
                "",
                f"- known chord note ratio: `{pitch_summary.get('known_chord_note_ratio', 0.0):.3f}`",
                "",
                "| role | ratio | count |",
                "|---|---:|---:|",
            ]
        )
        role_ratios = pitch_summary.get("role_ratios", {})
        role_counts = pitch_summary.get("role_counts", {})
        for role in PITCH_ROLE_KEYS:
            lines.append(f"| {role} | {float(role_ratios.get(role, 0.0)):.3f} | {int(role_counts.get(role, 0))} |")
        lines.extend(["", "### Strong Beat Role Ratios", "", "| role | ratio |", "|---|---:|"])
        strong_ratios = pitch_summary.get("bucket_ratios", {}).get("strong", {})
        for role in PITCH_ROLE_KEYS:
            lines.append(f"| {role} | {float(strong_ratios.get(role, 0.0)):.3f} |")
    if report.get("pitch_role_reference_warning"):
        lines.extend(
            [
                "",
                "## Pitch Role Reference Warning",
                "",
                f"- `{report['pitch_role_reference_warning']}`",
                f"- required known chord note ratio: `{report.get('min_pitch_role_known_chord_ratio', 0.5):.3f}`",
                f"- actual known chord note ratio: `{pitch_summary.get('known_chord_note_ratio', 0.0):.3f}`",
                "- generated pitch-role deltas are intentionally omitted until chord annotations exist.",
            ]
        )
    if report.get("generated_comparison"):
        lines.extend(["", "## Generated Comparison", ""])
        for row in report["generated_comparison"]:
            lines.append(f"### {row['label']}")
            for key, delta in row["delta_from_reference_mean"].items():
                lines.append(f"- `{key}` delta from reference mean: `{delta:.3f}`")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Stage B real phrase reference statistics")
    parser.add_argument("--input_dir", type=str, default="./midi_dataset/midi/studio")
    parser.add_argument("--output_root", type=str, default=str(ROOT_DIR / "outputs" / "stage_b_reference_stats"))
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--role", type=str, default="lead")
    parser.add_argument("--max_files", type=int, default=4)
    parser.add_argument("--window_bars", type=int, default=8)
    parser.add_argument("--window_stride_bars", type=int, default=4)
    parser.add_argument("--min_window_target_notes", type=int, default=16)
    parser.add_argument("--max_records", type=int, default=64)
    parser.add_argument("--skip_prepare", action="store_true")
    parser.add_argument("--tokenized_dir", type=str, default=None)
    parser.add_argument("--generated_report", type=str, default=None)
    parser.add_argument("--min_pitch_role_known_chord_ratio", type=float, default=0.5)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_root) / run_id
    roles_dir = run_dir / "roles"
    tokenized_dir = Path(args.tokenized_dir) if args.tokenized_dir else roles_dir / args.role / "tokenized"

    report: dict[str, Any] = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "input_dir": str(args.input_dir),
        "tokenized_dir": str(tokenized_dir),
        "window_bars": int(args.window_bars),
        "window_stride_bars": int(args.window_stride_bars),
        "min_window_target_notes": int(args.min_window_target_notes),
        "max_files": args.max_files,
        "max_records": args.max_records,
    }

    if not args.skip_prepare:
        prepare_result = run_prepare_command(args, roles_dir)
        report["prepare_result"] = prepare_result
        if prepare_result["returncode"] != 0:
            write_json(run_dir / "reference_stats_report.json", report)
            print(json.dumps(report, ensure_ascii=True, indent=2))
            return int(prepare_result["returncode"])

    rows = build_reference_rows(tokenized_dir, bars=int(args.window_bars), max_records=args.max_records)
    report["reference_rows_head"] = rows[:8]
    report["reference_summary"] = summarize_reference_rows(rows)
    if not rows:
        report["failure_reason"] = "No Stage B tokenized records available for reference statistics"
        write_json(run_dir / "reference_stats_report.json", report)
        print(json.dumps(report, ensure_ascii=True, indent=2))
        return 2

    if args.generated_report:
        generated_report_path = Path(args.generated_report)
        if generated_report_path.exists():
            generated_rows = generated_rows_from_report(read_json(generated_report_path))
            report["generated_report"] = str(generated_report_path)
            known_chord_ratio = float(
                report["reference_summary"]
                .get("pitch_role_landing", {})
                .get("known_chord_note_ratio", 0.0)
                or 0.0
            )
            pitch_role_ready = known_chord_ratio >= float(args.min_pitch_role_known_chord_ratio)
            report["pitch_role_reference_ready"] = bool(pitch_role_ready)
            report["min_pitch_role_known_chord_ratio"] = float(args.min_pitch_role_known_chord_ratio)
            if not pitch_role_ready:
                report["pitch_role_reference_warning"] = (
                    "Reference tokenized records do not have enough chord annotations for pitch-role comparison"
                )
            report["generated_comparison"] = compare_generated_to_reference(
                generated_rows,
                report["reference_summary"],
                include_pitch_roles=pitch_role_ready,
            )
        else:
            report["generated_report_warning"] = f"missing generated report: {generated_report_path}"

    write_json(run_dir / "reference_stats_report.json", report)
    (run_dir / "reference_stats_report.md").write_text(markdown_report(report), encoding="utf-8")
    print(json.dumps(report["reference_summary"], ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
