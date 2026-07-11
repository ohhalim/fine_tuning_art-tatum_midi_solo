"""Run dead-air repair sweep for Music Transformer solo-yield failure cases."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from scripts.assess_stage_b_generic_base_readiness import write_json, write_text  # noqa: E402


SCHEMA_VERSION = "music_transformer_solo_yield_dead_air_repair_sweep_v1"
DEFAULT_CHECKPOINT_DIR = (
    "outputs/stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_smoke/"
    "harness_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_smoke/"
    "training_smoke/controlled_2048_512_maxseq160/checkpoints"
)
DEFAULT_VARIANTS = [
    "fill_n8=fill:8",
    "fill_n9=fill:9",
    "fill_n10=fill:10",
]


class DeadAirRepairSweepError(ValueError):
    pass


@dataclass(frozen=True)
class RepairVariant:
    label: str
    duration_mode: str
    note_groups_per_bar: int


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _bool_token(value: Any) -> str:
    return "true" if bool(value) else "false"


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise DeadAirRepairSweepError(f"json not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def avg(values: Sequence[float]) -> float:
    return float(mean(values)) if values else 0.0


def parse_variant(raw: str) -> RepairVariant:
    if "=" in raw:
        label, value = raw.split("=", 1)
    else:
        label, value = raw, raw
    if ":" not in value:
        raise DeadAirRepairSweepError(f"variant must use mode:note_groups format: {raw}")
    duration_mode, note_groups = value.split(":", 1)
    duration_mode = duration_mode.strip()
    if duration_mode not in {"cap", "fill"}:
        raise DeadAirRepairSweepError(f"unsupported duration mode: {duration_mode}")
    return RepairVariant(
        label=label.strip().replace(" ", "_"),
        duration_mode=duration_mode,
        note_groups_per_bar=int(note_groups),
    )


def run_command(command: Sequence[str]) -> dict[str, Any]:
    completed = subprocess.run(
        list(command),
        cwd=ROOT_DIR,
        text=True,
        capture_output=True,
        check=False,
    )
    result = {
        "command": list(command),
        "returncode": int(completed.returncode),
        "stdout_tail": (completed.stdout or "")[-2000:],
        "stderr_tail": (completed.stderr or "")[-2000:],
    }
    if completed.returncode != 0:
        raise DeadAirRepairSweepError(
            "command failed: "
            + " ".join(command)
            + "\nstdout:\n"
            + result["stdout_tail"]
            + "\nstderr:\n"
            + result["stderr_tail"]
        )
    return result


def selected_failure_case(failure_review: dict[str, Any], label: str) -> dict[str, Any]:
    cases = [_dict(row) for row in _list(failure_review.get("case_reviews"))]
    if label:
        cases = [row for row in cases if str(row.get("label")) == label]
    if not cases:
        raise DeadAirRepairSweepError("no failure case available")
    return cases[0]


def probe_command(
    *,
    output_root: Path,
    run_id: str,
    checkpoint_dir: Path,
    case: dict[str, Any],
    variant: RepairVariant,
    sample_count: int,
    bpm: int,
    bars: int,
    density: str,
    temperature: float,
    top_k: int,
    pitch_min: int,
    pitch_max: int,
    max_adjacent_interval: int,
    max_sequence: int,
) -> list[str]:
    command = [
        sys.executable,
        "scripts/run_stage_b_generation_probe.py",
        "--output_root",
        str(output_root),
        "--run_id",
        run_id,
        "--checkpoint_dir",
        str(checkpoint_dir),
        "--skip_prepare",
        "--skip_train",
        "--generation_mode",
        "constrained",
        "--num_samples",
        str(sample_count),
        "--seed",
        str(_int(case.get("seed"))),
        "--bpm",
        str(bpm),
        "--bars",
        str(bars),
        "--chords",
        str(case.get("chords")),
        "--density",
        density,
        "--temperature",
        str(temperature),
        "--top_k",
        str(top_k),
        "--max_sequence",
        str(max_sequence),
        "--constrained_note_groups_per_bar",
        str(variant.note_groups_per_bar),
        "--coverage_aware_positions",
        "--coverage_position_window",
        "1",
        "--chord_aware_pitches",
        "--chord_pitch_mode",
        "approach_tensions",
        "--chord_pitch_repeat_window",
        "2",
        "--constrained_pitch_min",
        str(pitch_min),
        "--constrained_pitch_max",
        str(pitch_max),
        "--constrained_max_adjacent_interval",
        str(max_adjacent_interval),
        "--jazz_rhythm_positions",
        "--jazz_duration_tokens",
        "--jazz_rhythm_profile",
        "swing_motif",
        "--avoid_reused_positions",
        "--postprocess_overlap",
        "--max_simultaneous_notes",
        "1",
    ]
    if variant.duration_mode == "fill":
        command.append("--fill_duration_to_next_position")
    else:
        command.append("--cap_duration_to_next_position")
    return command


def package_command(
    *,
    probe_report: Path,
    output_root: Path,
    run_id: str,
    top_n: int,
) -> list[str]:
    return [
        sys.executable,
        "scripts/build_music_transformer_solo_yield_package.py",
        "--probe_report",
        str(probe_report),
        "--output_root",
        str(output_root),
        "--run_id",
        run_id,
        "--top_n",
        str(top_n),
    ]


def summarize_probe(report_path: Path) -> dict[str, Any]:
    report = read_json(report_path)
    summary = _dict(report.get("summary"))
    samples = [_dict(row) for row in _list(report.get("samples"))]
    strict_samples = [row for row in samples if bool(row.get("strict_valid", False))]
    invalid_samples = [row for row in samples if not bool(row.get("strict_valid", False))]
    dead_air_fail_count = sum(
        1
        for row in invalid_samples
        if str(row.get("failure_reason") or "").startswith("dead-air ratio too high")
    )
    return {
        "report_path": str(report_path),
        "sample_count": _int(summary.get("sample_count")),
        "valid_sample_count": _int(summary.get("valid_sample_count")),
        "strict_valid_sample_count": _int(summary.get("strict_valid_sample_count")),
        "grammar_gate_sample_count": _int(summary.get("grammar_gate_sample_count")),
        "collapse_warning_sample_count": _int(summary.get("collapse_warning_sample_count")),
        "dead_air_fail_count": dead_air_fail_count,
        "failure_reasons": _dict(summary.get("failure_reasons")),
        "strict_yield_rate": _float(summary.get("strict_valid_sample_rate")),
        "grammar_yield_rate": _float(summary.get("grammar_gate_sample_rate")),
        "avg_dead_air_ratio": avg([_float(_dict(row.get("metrics")).get("dead_air_ratio")) for row in samples]),
        "strict_avg_dead_air_ratio": avg(
            [_float(_dict(row.get("metrics")).get("dead_air_ratio")) for row in strict_samples]
        ),
        "invalid_avg_dead_air_ratio": avg(
            [_float(_dict(row.get("metrics")).get("dead_air_ratio")) for row in invalid_samples]
        ),
        "avg_postprocess_removal_ratio": _float(summary.get("avg_postprocess_removal_ratio")),
        "max_postprocess_removal_ratio": _float(summary.get("max_postprocess_removal_ratio")),
        "avg_onset_coverage_ratio": _float(summary.get("avg_onset_coverage_ratio")),
        "avg_sustained_coverage_ratio": _float(summary.get("avg_sustained_coverage_ratio")),
        "avg_duration_diversity_ratio": _float(summary.get("avg_duration_diversity_ratio")),
        "avg_most_common_duration_ratio": _float(summary.get("avg_most_common_duration_ratio")),
    }


def summarize_package(package_path: Path) -> dict[str, Any]:
    if not package_path.exists():
        return {}
    report = read_json(package_path)
    summary = _dict(report.get("yield_summary"))
    return {
        "package_report_path": str(package_path),
        "selected_candidate_count": _int(summary.get("selected_candidate_count")),
        "rendered_audio_file_count": _int(summary.get("rendered_audio_file_count")),
        "top_candidate_paths": [
            str(_dict(row).get("midi_path"))
            for row in _list(report.get("top_candidates"))
            if _dict(row).get("midi_path")
        ],
        "wav_paths": [
            str(_dict(_dict(row).get("wav_file")).get("path"))
            for row in _list(report.get("rendered_audio_files"))
            if _dict(_dict(row).get("wav_file")).get("path")
        ],
    }


def run_variant(
    *,
    output_dir: Path,
    checkpoint_dir: Path,
    case: dict[str, Any],
    variant: RepairVariant,
    sample_count: int,
    top_n: int,
    bpm: int,
    bars: int,
    density: str,
    temperature: float,
    top_k: int,
    pitch_min: int,
    pitch_max: int,
    max_adjacent_interval: int,
    max_sequence: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    case_label = str(case.get("label"))
    run_id = f"{case_label}_{variant.label}"
    probe_root = output_dir / "probes"
    package_root = output_dir / "packages"
    probe_report_path = probe_root / run_id / "report.json"
    package_report_path = package_root / run_id / "solo_yield_package.json"
    commands = [
        {
            "kind": "probe",
            "variant": variant.label,
            **run_command(
                probe_command(
                    output_root=probe_root,
                    run_id=run_id,
                    checkpoint_dir=checkpoint_dir,
                    case=case,
                    variant=variant,
                    sample_count=sample_count,
                    bpm=bpm,
                    bars=bars,
                    density=density,
                    temperature=temperature,
                    top_k=top_k,
                    pitch_min=pitch_min,
                    pitch_max=pitch_max,
                    max_adjacent_interval=max_adjacent_interval,
                    max_sequence=max_sequence,
                )
            ),
        },
        {
            "kind": "package",
            "variant": variant.label,
            **run_command(
                package_command(
                    probe_report=probe_report_path,
                    output_root=package_root,
                    run_id=run_id,
                    top_n=top_n,
                )
            ),
        },
    ]
    summary = summarize_probe(probe_report_path)
    summary.update(
        {
            "label": variant.label,
            "duration_mode": variant.duration_mode,
            "note_groups_per_bar": int(variant.note_groups_per_bar),
            "package": summarize_package(package_report_path),
        }
    )
    return summary, commands


def select_best_variant(variants: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not variants:
        raise DeadAirRepairSweepError("no repair variants generated")
    return sorted(
        variants,
        key=lambda row: (
            _int(row.get("strict_valid_sample_count")),
            _int(row.get("valid_sample_count")),
            _int(row.get("grammar_gate_sample_count")),
            -_int(row.get("dead_air_fail_count")),
            -_float(row.get("avg_dead_air_ratio")),
            _int(row.get("note_groups_per_bar")),
        ),
        reverse=True,
    )[0]


def build_repair_report(
    failure_review: dict[str, Any],
    *,
    output_dir: Path,
    checkpoint_dir: Path,
    case_label: str,
    variants: Sequence[RepairVariant],
    sample_count: int,
    top_n: int,
    bpm: int,
    bars: int,
    density: str,
    temperature: float,
    top_k: int,
    pitch_min: int,
    pitch_max: int,
    max_adjacent_interval: int,
    max_sequence: int,
) -> dict[str, Any]:
    if output_dir.exists():
        raise DeadAirRepairSweepError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    case = selected_failure_case(failure_review, case_label)
    source = {
        "label": str(case.get("label")),
        "chords": str(case.get("chords")),
        "seed": _int(case.get("seed")),
        "sample_count": _int(case.get("sample_count")),
        "strict_valid_sample_count": _int(case.get("strict_valid_sample_count")),
        "valid_sample_count": _int(case.get("valid_sample_count")),
        "grammar_gate_sample_count": _int(case.get("grammar_gate_sample_count")),
        "dead_air_fail_count": _int(case.get("dead_air_fail_count")),
        "strict_yield_rate": _float(case.get("strict_yield_rate")),
        "selected_repair_target": str(case.get("repair_target")),
    }
    command_results: list[dict[str, Any]] = []
    variant_rows: list[dict[str, Any]] = []
    for variant in variants:
        row, commands = run_variant(
            output_dir=output_dir,
            checkpoint_dir=checkpoint_dir,
            case=case,
            variant=variant,
            sample_count=sample_count,
            top_n=top_n,
            bpm=bpm,
            bars=bars,
            density=density,
            temperature=temperature,
            top_k=top_k,
            pitch_min=pitch_min,
            pitch_max=pitch_max,
            max_adjacent_interval=max_adjacent_interval,
            max_sequence=max_sequence,
        )
        variant_rows.append(row)
        command_results.extend(commands)
    best = select_best_variant(variant_rows)
    strict_delta = _int(best.get("strict_valid_sample_count")) - _int(source.get("strict_valid_sample_count"))
    dead_air_delta = _int(source.get("dead_air_fail_count")) - _int(best.get("dead_air_fail_count"))
    repair_improved = bool(strict_delta > 0 and dead_air_delta > 0)
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "source_failure_review": {
            "schema_version": failure_review.get("schema_version"),
            "output_dir": failure_review.get("output_dir"),
            "selected_repair_target": _dict(failure_review.get("decision")).get("selected_repair_target"),
            "next_boundary": _dict(failure_review.get("decision")).get("next_boundary"),
        },
        "source_case": source,
        "variants": variant_rows,
        "best_variant": best,
        "comparison": {
            "source_strict_valid_sample_count": _int(source.get("strict_valid_sample_count")),
            "best_strict_valid_sample_count": _int(best.get("strict_valid_sample_count")),
            "strict_valid_sample_delta": strict_delta,
            "source_dead_air_fail_count": _int(source.get("dead_air_fail_count")),
            "best_dead_air_fail_count": _int(best.get("dead_air_fail_count")),
            "dead_air_fail_delta": dead_air_delta,
            "repair_improved": repair_improved,
            "best_variant_all_strict_valid": _int(best.get("strict_valid_sample_count")) == int(sample_count),
        },
        "commands": command_results,
        "decision": {
            "current_boundary": "music_transformer_solo_yield_dead_air_repair_sweep",
            "next_boundary": "music_transformer_solo_yield_repaired_progression_retry_sweep",
            "selected_variant": best["label"],
            "critical_user_input_required": False,
            "reason": "best variant reduced dead-air failures and improved strict candidate yield",
        },
        "readiness": {
            "dead_air_repair_sweep_completed": True,
            "repair_improved": repair_improved,
            "music_transformer_checkpoint_generation_used": True,
            "constrained_decoding_used": True,
            "technical_wav_render_completed": bool(_int(_dict(best.get("package")).get("rendered_audio_file_count"))),
            "musical_quality_claimed": False,
            "artist_style_claimed": False,
            "production_ready_claimed": False,
        },
        "not_proven": [
            "human_audio_preference",
            "stable_jazz_solo_quality",
            "full_progression_retry_sweep",
            "artist_level_long_solo_generation",
        ],
    }


def validate_report(
    report: dict[str, Any],
    *,
    require_improvement: bool,
    require_best_all_strict: bool,
    require_no_quality_claim: bool,
) -> dict[str, Any]:
    comparison = _dict(report.get("comparison"))
    readiness = _dict(report.get("readiness"))
    best = _dict(report.get("best_variant"))
    if require_improvement and not bool(comparison.get("repair_improved", False)):
        raise DeadAirRepairSweepError("repair improvement required")
    if require_best_all_strict and not bool(comparison.get("best_variant_all_strict_valid", False)):
        raise DeadAirRepairSweepError("best variant all-strict gate required")
    if require_no_quality_claim:
        claimed = [
            key
            for key in ("musical_quality_claimed", "artist_style_claimed", "production_ready_claimed")
            if bool(readiness.get(key, True))
        ]
        if claimed:
            raise DeadAirRepairSweepError(f"unexpected quality claim: {claimed}")
    return {
        "schema_version": str(report.get("schema_version")),
        "source_strict_valid_sample_count": _int(comparison.get("source_strict_valid_sample_count")),
        "best_strict_valid_sample_count": _int(comparison.get("best_strict_valid_sample_count")),
        "strict_valid_sample_delta": _int(comparison.get("strict_valid_sample_delta")),
        "source_dead_air_fail_count": _int(comparison.get("source_dead_air_fail_count")),
        "best_dead_air_fail_count": _int(comparison.get("best_dead_air_fail_count")),
        "dead_air_fail_delta": _int(comparison.get("dead_air_fail_delta")),
        "repair_improved": bool(comparison.get("repair_improved", False)),
        "best_variant_all_strict_valid": bool(comparison.get("best_variant_all_strict_valid", False)),
        "selected_variant": str(best.get("label") or ""),
        "selected_duration_mode": str(best.get("duration_mode") or ""),
        "selected_note_groups_per_bar": _int(best.get("note_groups_per_bar")),
        "rendered_audio_file_count": _int(_dict(best.get("package")).get("rendered_audio_file_count")),
        "musical_quality_claimed": bool(readiness.get("musical_quality_claimed", True)),
        "next_boundary": str(_dict(report.get("decision")).get("next_boundary") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    source = report["source_case"]
    comparison = report["comparison"]
    best = report["best_variant"]
    decision = report["decision"]
    readiness = report["readiness"]
    lines = [
        "# Stage B MIDI-to-Solo Dead-Air Repair Sweep",
        "",
        "## Summary",
        "",
        f"- source case: `{source['label']}`",
        f"- chords: `{source['chords']}`",
        f"- source strict: `{comparison['source_strict_valid_sample_count']}` / `{source['sample_count']}`",
        f"- best strict: `{comparison['best_strict_valid_sample_count']}` / `{source['sample_count']}`",
        f"- strict delta: `{comparison['strict_valid_sample_delta']}`",
        f"- source dead-air fails: `{comparison['source_dead_air_fail_count']}`",
        f"- best dead-air fails: `{comparison['best_dead_air_fail_count']}`",
        f"- selected variant: `{best['label']}`",
        f"- selected duration mode: `{best['duration_mode']}`",
        f"- selected note groups per bar: `{best['note_groups_per_bar']}`",
        f"- rendered WAV files: `{_dict(best.get('package')).get('rendered_audio_file_count', 0)}`",
        f"- repair improved: `{_bool_token(comparison['repair_improved'])}`",
        f"- musical quality claimed: `{_bool_token(readiness['musical_quality_claimed'])}`",
        f"- next boundary: `{decision['next_boundary']}`",
        "",
        "## Variants",
        "",
        "| variant | duration | groups/bar | strict | grammar | dead-air fails | avg dead-air | avg removal | WAV |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report.get("variants", []):
        package = _dict(row.get("package"))
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['label']}`",
                    f"`{row['duration_mode']}`",
                    str(row["note_groups_per_bar"]),
                    f"{row['strict_valid_sample_count']}/{row['sample_count']}",
                    f"{row['grammar_gate_sample_count']}/{row['sample_count']}",
                    str(row["dead_air_fail_count"]),
                    f"{float(row['avg_dead_air_ratio']):.4f}",
                    f"{float(row['avg_postprocess_removal_ratio']):.4f}",
                    str(package.get("rendered_audio_file_count", 0)),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Best Variant WAV Files", ""])
    wav_paths = _list(_dict(best.get("package")).get("wav_paths"))
    if wav_paths:
        for path in wav_paths:
            lines.append(f"- `{path}`")
    else:
        lines.append("- `none`")
    lines.extend(["", "## Not Proven", ""])
    for item in report.get("not_proven", []):
        lines.append(f"- `{item}`")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run dead-air repair sweep")
    parser.add_argument("--failure_review", type=str, required=True)
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/music_transformer_finetune_mvp/solo_yield_dead_air_repair_sweep",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--case_label", type=str, default="")
    parser.add_argument("--variant", action="append", default=[])
    parser.add_argument("--sample_count", type=int, default=6)
    parser.add_argument("--top_n", type=int, default=2)
    parser.add_argument("--bpm", type=int, default=124)
    parser.add_argument("--bars", type=int, default=2)
    parser.add_argument("--density", type=str, default="medium")
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--pitch_min", type=int, default=55)
    parser.add_argument("--pitch_max", type=int, default=84)
    parser.add_argument("--max_adjacent_interval", type=int, default=7)
    parser.add_argument("--max_sequence", type=int, default=160)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--require_improvement", action="store_true")
    parser.add_argument("--require_best_all_strict", action="store_true")
    parser.add_argument("--require_no_quality_claim", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    variants = [parse_variant(raw) for raw in (args.variant or DEFAULT_VARIANTS)]
    output_dir = Path(args.output_root) / run_id
    report = build_repair_report(
        read_json(Path(args.failure_review)),
        output_dir=output_dir,
        checkpoint_dir=Path(args.checkpoint_dir),
        case_label=str(args.case_label or ""),
        variants=variants,
        sample_count=int(args.sample_count),
        top_n=int(args.top_n),
        bpm=int(args.bpm),
        bars=int(args.bars),
        density=str(args.density),
        temperature=float(args.temperature),
        top_k=int(args.top_k),
        pitch_min=int(args.pitch_min),
        pitch_max=int(args.pitch_max),
        max_adjacent_interval=int(args.max_adjacent_interval),
        max_sequence=int(args.max_sequence),
    )
    summary = validate_report(
        report,
        require_improvement=bool(args.require_improvement),
        require_best_all_strict=bool(args.require_best_all_strict),
        require_no_quality_claim=bool(args.require_no_quality_claim),
    )
    write_json(output_dir / "solo_yield_dead_air_repair_sweep.json", report)
    write_json(output_dir / "solo_yield_dead_air_repair_sweep_summary.json", summary)
    markdown = markdown_report(report)
    write_text(output_dir / "solo_yield_dead_air_repair_sweep.md", markdown)
    if args.doc_path:
        write_text(Path(args.doc_path), markdown)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
