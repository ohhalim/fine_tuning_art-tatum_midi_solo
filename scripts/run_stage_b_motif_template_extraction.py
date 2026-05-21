"""Extract data-derived motif templates from real Stage B phrase windows."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from scripts.run_stage_b_generation_probe import extract_stage_b_note_groups  # noqa: E402
from scripts.stage_b_tokens import POSITIONS_PER_BAR  # noqa: E402

STRONG_POSITIONS = {0, 4, 8, 12}


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


def absolute_position(group: dict[str, int]) -> int:
    return int(group["bar"]) * int(POSITIONS_PER_BAR) + int(group["position"])


def _sign(value: int) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def motif_from_groups(
    groups: Sequence[dict[str, int]],
    *,
    source_record: str,
    start_index: int,
) -> dict[str, Any]:
    abs_positions = [absolute_position(group) for group in groups]
    pitches = [int(group["pitch"]) for group in groups]
    durations = [int(group["duration_steps"]) for group in groups]
    first_abs = abs_positions[0]
    first_pitch = pitches[0]
    position_deltas = [int(position - first_abs) for position in abs_positions]
    ioi_steps = [int(abs_positions[index + 1] - abs_positions[index]) for index in range(len(abs_positions) - 1)]
    pitch_intervals = [int(pitch - first_pitch) for pitch in pitches]
    melodic_intervals = [int(pitches[index + 1] - pitches[index]) for index in range(len(pitches) - 1)]
    interval_signs = [_sign(interval) for interval in melodic_intervals if interval != 0]
    direction_changes = sum(1 for index in range(1, len(interval_signs)) if interval_signs[index] != interval_signs[index - 1])
    start_bar = int(groups[0]["bar"])
    end_bar = int(groups[-1]["bar"])
    positions = [int(group["position"]) for group in groups]
    syncopated_count = sum(1 for position in positions if position not in STRONG_POSITIONS)

    return {
        "source_record": source_record,
        "start_index": int(start_index),
        "length": int(len(groups)),
        "start_bar": start_bar,
        "start_position": int(groups[0]["position"]),
        "bar_span": int(end_bar - start_bar + 1),
        "position_deltas": position_deltas,
        "ioi_steps": ioi_steps,
        "duration_steps": durations,
        "pitch_intervals": pitch_intervals,
        "melodic_intervals": melodic_intervals,
        "interval_signs": interval_signs,
        "pitch_span": int(max(pitches) - min(pitches)) if pitches else 0,
        "syncopated_onset_ratio": float(syncopated_count / len(groups)) if groups else 0.0,
        "direction_change_ratio": float(direction_changes / max(1, len(interval_signs) - 1)) if interval_signs else 0.0,
        "rhythm_key": {
            "position_deltas": position_deltas,
            "duration_steps": durations,
        },
        "contour_key": {
            "pitch_intervals": pitch_intervals,
            "melodic_intervals": melodic_intervals,
        },
        "full_key": {
            "position_deltas": position_deltas,
            "duration_steps": durations,
            "pitch_intervals": pitch_intervals,
        },
    }


def stable_key(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def extract_motif_templates_from_tokens(
    tokens: Sequence[int],
    *,
    source_record: str,
    motif_length: int,
    max_bar_span: int,
    require_strictly_increasing_onsets: bool = True,
) -> list[dict[str, Any]]:
    groups = sorted(
        extract_stage_b_note_groups(tokens, primer_size=0),
        key=lambda group: (int(group["bar"]), int(group["position"]), int(group["pitch"])),
    )
    length = max(2, int(motif_length))
    motifs: list[dict[str, Any]] = []
    if len(groups) < length:
        return motifs
    for start_index in range(0, len(groups) - length + 1):
        segment = groups[start_index : start_index + length]
        bar_span = int(segment[-1]["bar"]) - int(segment[0]["bar"]) + 1
        if bar_span > int(max_bar_span):
            continue
        if require_strictly_increasing_onsets:
            abs_positions = [absolute_position(group) for group in segment]
            if any(abs_positions[index + 1] <= abs_positions[index] for index in range(len(abs_positions) - 1)):
                continue
        motifs.append(motif_from_groups(segment, source_record=source_record, start_index=start_index))
    return motifs


def iter_token_files(tokenized_dir: Path) -> list[Path]:
    return sorted(tokenized_dir.glob("*/*.npy"))


def load_motif_rows(
    tokenized_dir: Path,
    *,
    motif_length: int,
    max_bar_span: int,
    max_records: int | None,
    require_strictly_increasing_onsets: bool = True,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in iter_token_files(tokenized_dir):
        if max_records is not None and len({row["source_record"] for row in rows}) >= int(max_records):
            break
        tokens = [int(token) for token in np.load(path).tolist()]
        if not tokens:
            continue
        rows.extend(
            extract_motif_templates_from_tokens(
                tokens,
                source_record=str(path.relative_to(tokenized_dir)),
                motif_length=motif_length,
                max_bar_span=max_bar_span,
                require_strictly_increasing_onsets=require_strictly_increasing_onsets,
            )
        )
    return rows


def compact_counter(
    counter: Counter[str],
    *,
    examples_by_key: dict[str, list[dict[str, Any]]],
    total_count: int,
    top_n: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rank, (key, count) in enumerate(counter.most_common(int(top_n)), start=1):
        examples = examples_by_key.get(key, [])
        decoded_key = json.loads(key)
        rows.append(
            {
                "rank": int(rank),
                "count": int(count),
                "support_ratio": float(count / total_count) if total_count else 0.0,
                "key": decoded_key,
                "examples": [
                    {
                        "source_record": example["source_record"],
                        "start_index": int(example["start_index"]),
                        "start_bar": int(example["start_bar"]),
                        "start_position": int(example["start_position"]),
                    }
                    for example in examples[:3]
                ],
            }
        )
    return rows


def summarize_motif_templates(motif_rows: list[dict[str, Any]], top_n: int = 20) -> dict[str, Any]:
    rhythm_counter: Counter[str] = Counter()
    contour_counter: Counter[str] = Counter()
    full_counter: Counter[str] = Counter()
    examples_by_rhythm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    examples_by_contour: dict[str, list[dict[str, Any]]] = defaultdict(list)
    examples_by_full: dict[str, list[dict[str, Any]]] = defaultdict(list)
    source_records = {str(row["source_record"]) for row in motif_rows}

    for row in motif_rows:
        rhythm_key = stable_key(row["rhythm_key"])
        contour_key = stable_key(row["contour_key"])
        full_key = stable_key(row["full_key"])
        rhythm_counter[rhythm_key] += 1
        contour_counter[contour_key] += 1
        full_counter[full_key] += 1
        examples_by_rhythm[rhythm_key].append(row)
        examples_by_contour[contour_key].append(row)
        examples_by_full[full_key].append(row)

    total = int(len(motif_rows))
    return {
        "source_record_count": int(len(source_records)),
        "motif_count": total,
        "unique_rhythm_template_count": int(len(rhythm_counter)),
        "unique_contour_template_count": int(len(contour_counter)),
        "unique_full_template_count": int(len(full_counter)),
        "top_rhythm_template_support_ratio": (
            float(rhythm_counter.most_common(1)[0][1] / total) if total and rhythm_counter else 0.0
        ),
        "top_contour_template_support_ratio": (
            float(contour_counter.most_common(1)[0][1] / total) if total and contour_counter else 0.0
        ),
        "top_full_template_support_ratio": (
            float(full_counter.most_common(1)[0][1] / total) if total and full_counter else 0.0
        ),
        "top_rhythm_templates": compact_counter(
            rhythm_counter,
            examples_by_key=examples_by_rhythm,
            total_count=total,
            top_n=top_n,
        ),
        "top_contour_templates": compact_counter(
            contour_counter,
            examples_by_key=examples_by_contour,
            total_count=total,
            top_n=top_n,
        ),
        "top_full_templates": compact_counter(
            full_counter,
            examples_by_key=examples_by_full,
            total_count=total,
            top_n=top_n,
        ),
    }


def format_key_value(value: Any) -> str:
    if isinstance(value, dict):
        return "; ".join(f"{key}={val}" for key, val in value.items())
    return str(value)


def markdown_template_table(title: str, rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        f"## {title}",
        "",
        "| rank | count | support | key | example |",
        "|---:|---:|---:|---|---|",
    ]
    for row in rows:
        example = row["examples"][0] if row["examples"] else {}
        example_text = (
            f"{example.get('source_record')}@{example.get('start_index')}"
            if example
            else ""
        )
        lines.append(
            f"| {row['rank']} | {row['count']} | {row['support_ratio']:.3f} | "
            f"`{format_key_value(row['key'])}` | `{example_text}` |"
        )
    return lines


def markdown_report(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Stage B Motif Template Extraction",
        "",
        f"- source records: `{summary['source_record_count']}`",
        f"- motif count: `{summary['motif_count']}`",
        f"- unique rhythm templates: `{summary['unique_rhythm_template_count']}`",
        f"- unique contour templates: `{summary['unique_contour_template_count']}`",
        f"- unique full templates: `{summary['unique_full_template_count']}`",
        f"- top rhythm support: `{summary['top_rhythm_template_support_ratio']:.3f}`",
        f"- top contour support: `{summary['top_contour_template_support_ratio']:.3f}`",
        f"- top full support: `{summary['top_full_template_support_ratio']:.3f}`",
        "",
    ]
    lines.extend(markdown_template_table("Top Rhythm Templates", summary["top_rhythm_templates"]))
    lines.append("")
    lines.extend(markdown_template_table("Top Contour Templates", summary["top_contour_templates"]))
    lines.append("")
    lines.extend(markdown_template_table("Top Full Motif Templates", summary["top_full_templates"]))
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract Stage B data-derived phrase motif templates")
    parser.add_argument("--input_dir", type=str, default="./midi_dataset/midi/studio")
    parser.add_argument("--output_root", type=str, default=str(ROOT_DIR / "outputs" / "stage_b_motif_templates"))
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--role", type=str, default="lead")
    parser.add_argument("--max_files", type=int, default=4)
    parser.add_argument("--window_bars", type=int, default=8)
    parser.add_argument("--window_stride_bars", type=int, default=4)
    parser.add_argument("--min_window_target_notes", type=int, default=16)
    parser.add_argument("--motif_length", type=int, default=4)
    parser.add_argument("--max_bar_span", type=int, default=2)
    parser.add_argument("--max_records", type=int, default=64)
    parser.add_argument("--top_n", type=int, default=20)
    parser.add_argument("--allow_same_onset_motifs", action="store_true")
    parser.add_argument("--skip_prepare", action="store_true")
    parser.add_argument("--tokenized_dir", type=str, default=None)
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
        "motif_length": int(args.motif_length),
        "max_bar_span": int(args.max_bar_span),
        "max_files": args.max_files,
        "max_records": args.max_records,
        "top_n": int(args.top_n),
        "require_strictly_increasing_onsets": not bool(args.allow_same_onset_motifs),
    }

    if not args.skip_prepare:
        prepare_result = run_prepare_command(args, roles_dir)
        report["prepare_result"] = prepare_result
        if prepare_result["returncode"] != 0:
            write_json(run_dir / "motif_template_report.json", report)
            print(json.dumps(report, ensure_ascii=True, indent=2))
            return int(prepare_result["returncode"])

    motif_rows = load_motif_rows(
        tokenized_dir,
        motif_length=int(args.motif_length),
        max_bar_span=int(args.max_bar_span),
        max_records=args.max_records,
        require_strictly_increasing_onsets=not bool(args.allow_same_onset_motifs),
    )
    report["motif_rows_head"] = motif_rows[:8]
    report["summary"] = summarize_motif_templates(motif_rows, top_n=int(args.top_n))
    if not motif_rows:
        report["failure_reason"] = "No motif templates extracted from Stage B tokenized records"
        write_json(run_dir / "motif_template_report.json", report)
        print(json.dumps(report, ensure_ascii=True, indent=2))
        return 2

    write_json(run_dir / "motif_template_report.json", report)
    (run_dir / "motif_template_report.md").write_text(markdown_report(report), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
