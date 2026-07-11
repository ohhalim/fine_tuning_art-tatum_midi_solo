"""Compare Stage B chord-aware pitch modes on one tiny checkpoint."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from scripts.export_stage_b_review_candidates import (  # noqa: E402
    build_review_manifest,
    markdown_report as review_markdown_report,
    write_json as write_review_json,
)
from scripts.run_stage_b_sampling_sweep import (  # noqa: E402
    probe_command,
    read_json,
    row_from_probe_report,
    run_command,
    suffix_for_float,
    write_json,
)

VALID_PITCH_MODES = {"tones", "tones_tensions", "approach_tensions"}


def parse_pitch_modes(raw: str) -> list[str]:
    modes = [mode.strip().lower() for mode in raw.split(",") if mode.strip()]
    invalid = [mode for mode in modes if mode not in VALID_PITCH_MODES]
    if invalid:
        raise ValueError(f"Unknown pitch modes: {invalid}")
    return modes


def config_run_id(
    base_run_id: str,
    pitch_mode: str,
    note_groups_per_bar: int,
    top_k: int,
    temperature: float,
) -> str:
    return (
        f"{base_run_id}_coverage_chord_{pitch_mode}_g{int(note_groups_per_bar)}"
        f"_k{int(top_k)}_t{suffix_for_float(float(temperature))}"
    )


def best_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    return max(
        rows,
        key=lambda row: (
            int(row["strict_valid_sample_count"]),
            int(row["valid_sample_count"]),
            float(row["avg_tension_ratio"]),
            -float(row["avg_root_tone_ratio"]),
            -float(row["collapse_warning_sample_rate"]),
        ),
        default=None,
    )


def compact_row(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {
        "pitch_mode": row["pitch_mode"],
        "run_id": row["run_id"],
        "strict_valid_sample_count": int(row["strict_valid_sample_count"]),
        "valid_sample_count": int(row["valid_sample_count"]),
        "avg_chord_tone_ratio": float(row["avg_chord_tone_ratio"]),
        "avg_root_tone_ratio": float(row["avg_root_tone_ratio"]),
        "avg_tension_ratio": float(row["avg_tension_ratio"]),
        "avg_approach_candidate_ratio": float(row.get("avg_approach_candidate_ratio", 0.0)),
        "avg_approach_resolution_ratio": float(row.get("avg_approach_resolution_ratio", 0.0)),
        "avg_onset_coverage_ratio": float(row["avg_onset_coverage_ratio"]),
        "avg_sustained_coverage_ratio": float(row["avg_sustained_coverage_ratio"]),
    }


def build_pitch_mode_summary(
    rows: list[dict[str, Any]],
    min_best_strict_valid_samples: int = 1,
) -> dict[str, Any]:
    mode_rows = {mode: [row for row in rows if row.get("pitch_mode") == mode] for mode in sorted(VALID_PITCH_MODES)}
    best_tones = best_row(mode_rows["tones"])
    best_tensions = best_row(mode_rows["tones_tensions"])
    best_approach = best_row(mode_rows["approach_tensions"])
    best = best_row(rows)
    comparison_ready = bool(best_tones and best_tensions)
    tones_root = float(best_tones.get("avg_root_tone_ratio", 0.0)) if best_tones else 0.0
    tensions_root = float(best_tensions.get("avg_root_tone_ratio", 0.0)) if best_tensions else 0.0
    tones_tension = float(best_tones.get("avg_tension_ratio", 0.0)) if best_tones else 0.0
    tensions_tension = float(best_tensions.get("avg_tension_ratio", 0.0)) if best_tensions else 0.0
    passed_tones = bool(
        best_tones
        and int(best_tones["strict_valid_sample_count"]) >= int(min_best_strict_valid_samples)
    )
    passed_tensions = bool(
        best_tensions
        and int(best_tensions["strict_valid_sample_count"]) >= int(min_best_strict_valid_samples)
    )
    passed_approach = bool(
        best_approach
        and int(best_approach["strict_valid_sample_count"]) >= int(min_best_strict_valid_samples)
    )
    return {
        "config_count": int(len(rows)),
        "mode_counts": {mode: int(len(mode_rows[mode])) for mode in sorted(VALID_PITCH_MODES)},
        "best_config": compact_row(best),
        "best_tones_config": compact_row(best_tones),
        "best_tones_tensions_config": compact_row(best_tensions),
        "best_approach_tensions_config": compact_row(best_approach),
        "min_best_strict_valid_samples": int(min_best_strict_valid_samples),
        "comparison_ready": comparison_ready,
        "passed_tones_gate": passed_tones,
        "passed_tones_tensions_gate": passed_tensions,
        "passed_approach_tensions_gate": passed_approach,
        "passed_compare_gate": bool(
            comparison_ready
            and passed_tones
            and passed_tensions
            and (not mode_rows["approach_tensions"] or passed_approach)
        ),
        "root_tone_ratio_delta_tensions_minus_tones": float(tensions_root - tones_root),
        "tension_ratio_delta_tensions_minus_tones": float(tensions_tension - tones_tension),
    }


def markdown_table(rows: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    lines = [
        "# Stage B Pitch Mode Comparison",
        "",
        f"- passed compare gate: `{str(summary['passed_compare_gate']).lower()}`",
        f"- best tones config: `{summary['best_tones_config']}`",
        f"- best tones+tensions config: `{summary['best_tones_tensions_config']}`",
        f"- best approach+tensions config: `{summary['best_approach_tensions_config']}`",
        f"- root delta, tensions minus tones: `{summary['root_tone_ratio_delta_tensions_minus_tones']:.3f}`",
        f"- tension delta, tensions minus tones: `{summary['tension_ratio_delta_tensions_minus_tones']:.3f}`",
        "",
        "| pitch mode | samples | valid | strict | chord | root | tension | approach | resolved | onset | sustained | collapse | strict pass | report |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {pitch_mode} | {sample_count} | {valid_sample_count} | {strict_valid_sample_count} | "
            "{avg_chord_tone_ratio:.3f} | {avg_root_tone_ratio:.3f} | {avg_tension_ratio:.3f} | "
            "{avg_approach_candidate_ratio:.3f} | {avg_approach_resolution_ratio:.3f} | "
            "{avg_onset_coverage_ratio:.3f} | {avg_sustained_coverage_ratio:.3f} | "
            "{collapse_warning_sample_rate:.3f} | {passed_strict_review_gate} | `{report_path}` |".format(**row)
        )
    lines.append("")
    lines.append("## Diagnostic Failures")
    lines.append("")
    for row in rows:
        lines.append(f"- `{row['pitch_mode']}`: {row['diagnostic_failure_reasons']}")
    return "\n".join(lines) + "\n"


def export_review_candidates(
    report_path: Path,
    output_dir: Path,
    top_n: int,
    copy_midi: bool,
) -> dict[str, Any]:
    manifest = build_review_manifest(
        ranking_report_path=report_path,
        output_dir=output_dir,
        top_n=top_n,
        mode="coverage_chord",
        reviewable_only=True,
        copy_midi=copy_midi,
    )
    write_review_json(output_dir / "review_manifest.json", manifest)
    (output_dir / "review_candidates.md").write_text(review_markdown_report(manifest), encoding="utf-8")
    return manifest


def export_named_comparison_midis(review_exports: dict[str, dict[str, Any]], output_dir: Path) -> list[dict[str, Any]]:
    named_dir = output_dir / "compare_named_midi"
    named_dir.mkdir(parents=True, exist_ok=True)
    copied: list[dict[str, Any]] = []
    for mode_index, mode in enumerate(sorted(review_exports), start=1):
        manifest = review_exports[mode]
        for candidate in manifest.get("candidates", []):
            source = candidate.get("review_midi_path") or candidate.get("midi_path")
            if not source:
                continue
            source_path = Path(str(source))
            if not source_path.is_absolute():
                source_path = ROOT_DIR / source_path
            if not source_path.exists():
                continue
            target_name = (
                f"{mode_index:02d}_{mode}_rank_{int(candidate['review_rank']):02d}_"
                f"sample_{int(candidate['sample_index'])}.mid"
            )
            target_path = named_dir / target_name
            shutil.copy2(source_path, target_path)
            copied.append(
                {
                    "mode": mode,
                    "review_rank": int(candidate["review_rank"]),
                    "source_path": str(source_path),
                    "named_midi_path": str(target_path),
                }
            )
    return copied


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Stage B tones vs tones_tensions comparison")
    parser.add_argument("--output_root", type=str, default=str(ROOT_DIR / "outputs" / "stage_b_pitch_mode_compare"))
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--issue_number", type=int, default=55)
    parser.add_argument("--pitch_modes", type=str, default="tones,tones_tensions")
    parser.add_argument("--note_groups_per_bar", type=int, default=8)
    parser.add_argument("--coverage_position_window", type=int, default=0)
    parser.add_argument("--chord_pitch_repeat_window", type=int, default=2)
    parser.add_argument("--top_k", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--max_files", type=int, default=2)
    parser.add_argument("--window_bars", type=int, default=4)
    parser.add_argument("--window_stride_bars", type=int, default=2)
    parser.add_argument("--min_window_target_notes", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_sequence", type=int, default=192)
    parser.add_argument("--num_samples", type=int, default=3)
    parser.add_argument("--bars", type=int, default=4)
    parser.add_argument("--max_simultaneous_notes", type=int, default=2)
    parser.add_argument("--min_valid_samples", type=int, default=1)
    parser.add_argument("--min_strict_valid_samples", type=int, default=1)
    parser.add_argument("--min_best_strict_valid_samples", type=int, default=1)
    parser.add_argument("--max_collapse_warning_sample_rate", type=float, default=0.34)
    parser.add_argument("--strict_min_unique_pitches", type=int, default=3)
    parser.add_argument("--strict_min_unique_positions", type=int, default=3)
    parser.add_argument("--strict_min_unique_position_pitch_pairs", type=int, default=4)
    parser.add_argument("--strict_max_repeated_position_pitch_pair_ratio", type=float, default=0.49)
    parser.add_argument("--strict_max_postprocess_removal_ratio", type=float, default=0.49)
    parser.add_argument("--require_all_grammar_samples", action="store_true")
    parser.add_argument("--review_output_root", type=str, default=str(ROOT_DIR / "outputs" / "stage_b_review_candidates"))
    parser.add_argument("--review_top_n", type=int, default=3)
    parser.add_argument("--copy_review_midi", action="store_true")
    parser.add_argument("--n_layers", type=int, default=1)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--dim_feedforward", type=int, default=128)
    parser.add_argument("--lora_r", type=int, default=4)
    parser.add_argument("--lora_alpha", type=int, default=8)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    compare_dir = Path(args.output_root) / run_id
    pitch_modes = parse_pitch_modes(args.pitch_modes)
    if not pitch_modes:
        raise ValueError("--pitch_modes must not be empty")

    command_results: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    review_exports: dict[str, dict[str, Any]] = {}
    checkpoint_dir: Path | None = None
    first_config = True

    for pitch_mode in pitch_modes:
        mode_run_id = config_run_id(
            run_id,
            pitch_mode=pitch_mode,
            note_groups_per_bar=args.note_groups_per_bar,
            top_k=args.top_k,
            temperature=args.temperature,
        )
        probe_args = argparse.Namespace(**vars(args))
        probe_args.output_root = args.output_root
        probe_args.constrained_note_groups_per_bar = int(args.note_groups_per_bar)
        probe_args.coverage_aware_positions = True
        probe_args.coverage_position_window = int(args.coverage_position_window)
        probe_args.chord_aware_pitches = True
        probe_args.chord_pitch_mode = pitch_mode
        probe_args.chord_pitch_repeat_window = int(args.chord_pitch_repeat_window)
        cmd = probe_command(
            probe_args,
            run_id=mode_run_id,
            top_k=int(args.top_k),
            temperature=float(args.temperature),
            checkpoint_dir=checkpoint_dir,
            skip_prepare_train=not first_config,
        )
        cmd.extend(
            [
                "--window_bars",
                str(args.window_bars),
                "--window_stride_bars",
                str(args.window_stride_bars),
                "--min_window_target_notes",
                str(args.min_window_target_notes),
                "--bars",
                str(args.bars),
            ]
        )
        cmd.extend(["--coverage_aware_positions", "--coverage_position_window", str(args.coverage_position_window)])
        result = run_command(cmd)
        command_results.append(result)
        if result["returncode"] != 0:
            report = {
                "run_id": run_id,
                "failure_reason": f"probe command failed for pitch_mode={pitch_mode}",
                "command_results": command_results,
            }
            write_json(compare_dir / "pitch_mode_compare_report.json", report)
            print(json.dumps(report, ensure_ascii=True, indent=2))
            return int(result["returncode"])

        report_path = Path(args.output_root) / mode_run_id / "report.json"
        probe_report = read_json(report_path)
        if checkpoint_dir is None:
            checkpoint_dir = Path(probe_report["checkpoint_dir"])
        first_config = False
        row = row_from_probe_report(int(args.top_k), float(args.temperature), mode_run_id, probe_report)
        row["pitch_mode"] = pitch_mode
        row["note_groups_per_bar"] = int(args.note_groups_per_bar)
        rows.append(row)

        review_output_dir = Path(args.review_output_root) / run_id / pitch_mode
        review_exports[pitch_mode] = export_review_candidates(
            report_path=report_path,
            output_dir=review_output_dir,
            top_n=int(args.review_top_n),
            copy_midi=bool(args.copy_review_midi),
        )

    rows = sorted(rows, key=lambda row: str(row["pitch_mode"]))
    summary = build_pitch_mode_summary(
        rows,
        min_best_strict_valid_samples=args.min_best_strict_valid_samples,
    )
    report = {
        "run_id": run_id,
        "run_dir": str(compare_dir),
        "issue": int(args.issue_number),
        "pitch_modes": pitch_modes,
        "note_groups_per_bar": int(args.note_groups_per_bar),
        "top_k": int(args.top_k),
        "temperature": float(args.temperature),
        "summary": summary,
        "rows": rows,
        "review_exports": review_exports,
        "named_comparison_midis": export_named_comparison_midis(
            review_exports,
            output_dir=Path(args.review_output_root) / run_id,
        )
        if args.copy_review_midi
        else [],
        "command_results": command_results,
    }
    write_json(compare_dir / "pitch_mode_compare_report.json", report)
    (compare_dir / "pitch_mode_compare_report.md").write_text(markdown_table(rows, summary), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True, indent=2))
    return 0 if summary["passed_compare_gate"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
