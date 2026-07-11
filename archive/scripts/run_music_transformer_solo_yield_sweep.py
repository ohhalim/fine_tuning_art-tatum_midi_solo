"""Run chord-progression yield sweep for Music Transformer solo candidates."""

from __future__ import annotations

import argparse
import json
import re
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


SCHEMA_VERSION = "music_transformer_solo_yield_sweep_v1"
DEFAULT_CHECKPOINT_DIR = (
    "outputs/stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_smoke/"
    "harness_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_smoke/"
    "training_smoke/controlled_2048_512_maxseq160/checkpoints"
)
DEFAULT_PROGRESSIONS = [
    "minor_backdoor=Cm7,F7,Bbmaj7,Ebmaj7",
    "major_ii_v_turnaround=Dm7,G7,Cmaj7,A7",
    "dominant_cycle=Em7,A7,Dmaj7,G7",
    "rhythm_turnaround=Bbmaj7,G7,Cm7,F7",
]


class SoloYieldSweepError(ValueError):
    pass


@dataclass(frozen=True)
class ProgressionCase:
    label: str
    chords: str


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


def slugify(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", value.strip()).strip("_").lower()
    return cleaned or "case"


def parse_progression(raw: str, index: int) -> ProgressionCase:
    if "=" in raw:
        label, chords = raw.split("=", 1)
    elif ":" in raw:
        label, chords = raw.split(":", 1)
    else:
        label, chords = f"case_{index:02d}", raw
    chord_list = [item.strip() for item in chords.split(",") if item.strip()]
    if not chord_list:
        raise SoloYieldSweepError(f"empty chord progression: {raw}")
    return ProgressionCase(label=slugify(label), chords=",".join(chord_list))


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SoloYieldSweepError(f"json not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


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
        raise SoloYieldSweepError(
            "command failed: "
            + " ".join(command)
            + "\nstdout:\n"
            + result["stdout_tail"]
            + "\nstderr:\n"
            + result["stderr_tail"]
        )
    return result


def probe_command(
    *,
    output_root: Path,
    run_id: str,
    checkpoint_dir: Path,
    case: ProgressionCase,
    seed: int,
    sample_count: int,
    bpm: int,
    bars: int,
    density: str,
    duration_mode: str,
    temperature: float,
    top_k: int,
    note_groups_per_bar: int,
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
        str(seed),
        "--bpm",
        str(bpm),
        "--bars",
        str(bars),
        "--chords",
        case.chords,
        "--density",
        density,
        "--temperature",
        str(temperature),
        "--top_k",
        str(top_k),
        "--max_sequence",
        str(max_sequence),
        "--constrained_note_groups_per_bar",
        str(note_groups_per_bar),
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
    if duration_mode == "fill":
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


def summarize_case(
    *,
    case: ProgressionCase,
    seed: int,
    probe_report_path: Path,
    package_report_path: Path | None,
    min_case_strict_yield_rate: float,
) -> dict[str, Any]:
    probe_report = read_json(probe_report_path)
    summary = _dict(probe_report.get("summary"))
    sample_count = _int(summary.get("sample_count"))
    valid_count = _int(summary.get("valid_sample_count"))
    strict_count = _int(summary.get("strict_valid_sample_count"))
    grammar_count = _int(summary.get("grammar_gate_sample_count"))
    collapse_count = _int(summary.get("collapse_warning_sample_count"))
    samples = _list(probe_report.get("samples"))
    valid_samples = [row for row in samples if bool(_dict(row).get("valid", False))]
    strict_samples = [row for row in samples if bool(_dict(row).get("strict_valid", False))]
    note_counts = [_int(_dict(row).get("metrics", {}).get("note_count")) for row in strict_samples]
    dead_air_values = [_float(_dict(row).get("metrics", {}).get("dead_air_ratio")) for row in strict_samples]
    strict_rate = float(strict_count / sample_count) if sample_count else 0.0

    package_summary: dict[str, Any] = {}
    if package_report_path and package_report_path.exists():
        package_report = read_json(package_report_path)
        package_summary = {
            "package_report_path": str(package_report_path),
            "selected_candidate_count": _int(
                _dict(package_report.get("yield_summary")).get("selected_candidate_count")
            ),
            "rendered_audio_file_count": _int(
                _dict(package_report.get("yield_summary")).get("rendered_audio_file_count")
            ),
            "top_candidate_paths": [
                str(_dict(item).get("midi_path"))
                for item in _list(package_report.get("top_candidates"))
                if _dict(item).get("midi_path")
            ],
            "wav_paths": [
                str(_dict(_dict(item).get("wav_file")).get("path"))
                for item in _list(package_report.get("rendered_audio_files"))
                if _dict(_dict(item).get("wav_file")).get("path")
            ],
        }

    return {
        "label": case.label,
        "chords": case.chords,
        "seed": int(seed),
        "probe_report_path": str(probe_report_path),
        "sample_count": sample_count,
        "valid_sample_count": valid_count,
        "strict_valid_sample_count": strict_count,
        "grammar_gate_sample_count": grammar_count,
        "collapse_warning_sample_count": collapse_count,
        "valid_yield_rate": float(valid_count / sample_count) if sample_count else 0.0,
        "strict_yield_rate": strict_rate,
        "grammar_yield_rate": float(grammar_count / sample_count) if sample_count else 0.0,
        "passed_generation_gate": bool(probe_report.get("passed_generation_gate", False)),
        "passed_strict_review_gate": bool(probe_report.get("passed_strict_review_gate", False)),
        "case_yield_floor_passed": bool(strict_rate >= min_case_strict_yield_rate),
        "valid_candidate_count": len(valid_samples),
        "strict_candidate_count": len(strict_samples),
        "strict_avg_note_count": float(mean(note_counts)) if note_counts else 0.0,
        "strict_avg_dead_air_ratio": float(mean(dead_air_values)) if dead_air_values else 0.0,
        "package": package_summary,
    }


def aggregate_cases(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    sample_count = sum(_int(row.get("sample_count")) for row in rows)
    valid_count = sum(_int(row.get("valid_sample_count")) for row in rows)
    strict_count = sum(_int(row.get("strict_valid_sample_count")) for row in rows)
    grammar_count = sum(_int(row.get("grammar_gate_sample_count")) for row in rows)
    rendered_count = sum(_int(_dict(row.get("package")).get("rendered_audio_file_count")) for row in rows)
    selected_count = sum(_int(_dict(row.get("package")).get("selected_candidate_count")) for row in rows)
    strict_rates = [_float(row.get("strict_yield_rate")) for row in rows]
    return {
        "case_count": len(rows),
        "sample_count": sample_count,
        "valid_sample_count": valid_count,
        "strict_valid_sample_count": strict_count,
        "grammar_gate_sample_count": grammar_count,
        "valid_yield_rate": float(valid_count / sample_count) if sample_count else 0.0,
        "strict_yield_rate": float(strict_count / sample_count) if sample_count else 0.0,
        "grammar_yield_rate": float(grammar_count / sample_count) if sample_count else 0.0,
        "min_case_strict_yield_rate": min(strict_rates, default=0.0),
        "max_case_strict_yield_rate": max(strict_rates, default=0.0),
        "avg_case_strict_yield_rate": float(mean(strict_rates)) if strict_rates else 0.0,
        "selected_candidate_count": selected_count,
        "rendered_audio_file_count": rendered_count,
        "case_yield_floor_passed_count": sum(1 for row in rows if bool(row.get("case_yield_floor_passed", False))),
        "failing_cases": [
            {
                "label": row["label"],
                "chords": row["chords"],
                "strict_yield_rate": row["strict_yield_rate"],
                "strict_valid_sample_count": row["strict_valid_sample_count"],
                "sample_count": row["sample_count"],
            }
            for row in rows
            if not bool(row.get("case_yield_floor_passed", False))
        ],
    }


def build_sweep_report(
    *,
    output_dir: Path,
    cases: Sequence[ProgressionCase],
    checkpoint_dir: Path,
    sample_count: int,
    top_n: int,
    seed_start: int,
    seed_stride: int,
    bpm: int,
    bars: int,
    density: str,
    duration_mode: str,
    temperature: float,
    top_k: int,
    note_groups_per_bar: int,
    pitch_min: int,
    pitch_max: int,
    max_adjacent_interval: int,
    max_sequence: int,
    min_total_strict_yield_rate: float,
    min_case_strict_yield_rate: float,
    package_candidates: bool,
) -> dict[str, Any]:
    if output_dir.exists():
        raise SoloYieldSweepError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    probe_root = output_dir / "probes"
    package_root = output_dir / "packages"
    case_rows: list[dict[str, Any]] = []
    command_results: list[dict[str, Any]] = []

    for index, case in enumerate(cases, start=1):
        seed = int(seed_start + (index - 1) * seed_stride)
        case_run_id = f"{index:02d}_{case.label}_seed_{seed}"
        probe_report_path = probe_root / case_run_id / "report.json"
        command_results.append(
            {
                "kind": "probe",
                "case": case.label,
                **run_command(
                    probe_command(
                        output_root=probe_root,
                        run_id=case_run_id,
                        checkpoint_dir=checkpoint_dir,
                        case=case,
                        seed=seed,
                        sample_count=sample_count,
                        bpm=bpm,
                        bars=bars,
                        density=density,
                        duration_mode=duration_mode,
                        temperature=temperature,
                        top_k=top_k,
                        note_groups_per_bar=note_groups_per_bar,
                        pitch_min=pitch_min,
                        pitch_max=pitch_max,
                        max_adjacent_interval=max_adjacent_interval,
                        max_sequence=max_sequence,
                    )
                ),
            }
        )
        package_report_path: Path | None = None
        if package_candidates:
            command_results.append(
                {
                    "kind": "package",
                    "case": case.label,
                    **run_command(
                        package_command(
                            probe_report=probe_report_path,
                            output_root=package_root,
                            run_id=case_run_id,
                            top_n=top_n,
                        )
                    ),
                }
            )
            package_report_path = package_root / case_run_id / "solo_yield_package.json"
        case_rows.append(
            summarize_case(
                case=case,
                seed=seed,
                probe_report_path=probe_report_path,
                package_report_path=package_report_path,
                min_case_strict_yield_rate=min_case_strict_yield_rate,
            )
        )

    aggregate = aggregate_cases(case_rows)
    total_floor_passed = _float(aggregate.get("strict_yield_rate")) >= float(min_total_strict_yield_rate)
    all_case_floor_passed = _int(aggregate.get("case_yield_floor_passed_count")) == len(case_rows)
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "request": {
            "sample_count_per_case": int(sample_count),
            "top_n_per_case": int(top_n),
            "seed_start": int(seed_start),
            "seed_stride": int(seed_stride),
            "bpm": int(bpm),
            "bars": int(bars),
            "density": density,
            "duration_mode": duration_mode,
            "temperature": float(temperature),
            "top_k": int(top_k),
            "note_groups_per_bar": int(note_groups_per_bar),
            "pitch_min": int(pitch_min),
            "pitch_max": int(pitch_max),
            "max_adjacent_interval": int(max_adjacent_interval),
            "max_sequence": int(max_sequence),
            "package_candidates": bool(package_candidates),
        },
        "thresholds": {
            "min_total_strict_yield_rate": float(min_total_strict_yield_rate),
            "min_case_strict_yield_rate": float(min_case_strict_yield_rate),
        },
        "cases": case_rows,
        "aggregate": aggregate,
        "commands": command_results,
        "readiness": {
            "music_transformer_checkpoint_generation_used": True,
            "constrained_decoding_used": True,
            "multi_progression_yield_sweep_completed": True,
            "total_yield_floor_passed": total_floor_passed,
            "all_case_yield_floor_passed": all_case_floor_passed,
            "ranked_midi_candidates_exported": bool(_int(aggregate.get("selected_candidate_count")) > 0),
            "technical_wav_render_completed": bool(_int(aggregate.get("rendered_audio_file_count")) > 0),
            "musical_quality_claimed": False,
            "artist_style_claimed": False,
            "production_ready_claimed": False,
        },
        "decision": {
            "current_boundary": "music_transformer_solo_yield_sweep",
            "next_boundary": (
                "music_transformer_solo_yield_failure_case_review"
                if aggregate["failing_cases"]
                else "music_transformer_solo_yield_candidate_listening_review"
            ),
            "critical_user_input_required": False,
            "reason": (
                "strict yield floor failed in one or more chord progressions"
                if aggregate["failing_cases"]
                else "strict yield floor passed across configured chord progressions"
            ),
        },
        "not_proven": [
            "human_audio_preference",
            "stable_jazz_solo_quality",
            "artist_level_long_solo_generation",
            "production_ready_improviser",
        ],
    }


def validate_sweep_report(
    report: dict[str, Any],
    *,
    min_cases: int,
    require_total_floor: bool,
    require_all_case_floor: bool,
    require_no_quality_claim: bool,
) -> dict[str, Any]:
    readiness = _dict(report.get("readiness"))
    aggregate = _dict(report.get("aggregate"))
    if _int(aggregate.get("case_count")) < int(min_cases):
        raise SoloYieldSweepError("case count below requirement")
    if require_total_floor and not bool(readiness.get("total_yield_floor_passed", False)):
        raise SoloYieldSweepError("total strict yield floor required")
    if require_all_case_floor and not bool(readiness.get("all_case_yield_floor_passed", False)):
        raise SoloYieldSweepError("all case strict yield floor required")
    if require_no_quality_claim:
        claimed = [
            key
            for key in ("musical_quality_claimed", "artist_style_claimed", "production_ready_claimed")
            if bool(readiness.get(key, True))
        ]
        if claimed:
            raise SoloYieldSweepError(f"unexpected quality claim: {claimed}")
    return {
        "schema_version": str(report.get("schema_version")),
        "case_count": _int(aggregate.get("case_count")),
        "sample_count": _int(aggregate.get("sample_count")),
        "valid_sample_count": _int(aggregate.get("valid_sample_count")),
        "strict_valid_sample_count": _int(aggregate.get("strict_valid_sample_count")),
        "grammar_gate_sample_count": _int(aggregate.get("grammar_gate_sample_count")),
        "valid_yield_rate": _float(aggregate.get("valid_yield_rate")),
        "strict_yield_rate": _float(aggregate.get("strict_yield_rate")),
        "grammar_yield_rate": _float(aggregate.get("grammar_yield_rate")),
        "min_case_strict_yield_rate": _float(aggregate.get("min_case_strict_yield_rate")),
        "selected_candidate_count": _int(aggregate.get("selected_candidate_count")),
        "rendered_audio_file_count": _int(aggregate.get("rendered_audio_file_count")),
        "total_yield_floor_passed": bool(readiness.get("total_yield_floor_passed", False)),
        "all_case_yield_floor_passed": bool(readiness.get("all_case_yield_floor_passed", False)),
        "musical_quality_claimed": bool(readiness.get("musical_quality_claimed", True)),
        "next_boundary": str(_dict(report.get("decision")).get("next_boundary") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    aggregate = report["aggregate"]
    readiness = report["readiness"]
    decision = report["decision"]
    lines = [
        "# Stage B MIDI-to-Solo Chord Progression Yield Sweep",
        "",
        "## Summary",
        "",
        f"- checkpoint generation used: `{_bool_token(readiness['music_transformer_checkpoint_generation_used'])}`",
        f"- constrained decoding used: `{_bool_token(readiness['constrained_decoding_used'])}`",
        f"- case count: `{aggregate['case_count']}`",
        f"- sample count: `{aggregate['sample_count']}`",
        f"- duration mode: `{report['request']['duration_mode']}`",
        f"- valid yield: `{aggregate['valid_sample_count']}` / `{aggregate['sample_count']}`",
        f"- strict yield: `{aggregate['strict_valid_sample_count']}` / `{aggregate['sample_count']}`",
        f"- grammar yield: `{aggregate['grammar_gate_sample_count']}` / `{aggregate['sample_count']}`",
        f"- strict yield rate: `{float(aggregate['strict_yield_rate']):.4f}`",
        f"- min case strict yield rate: `{float(aggregate['min_case_strict_yield_rate']):.4f}`",
        f"- selected MIDI candidates: `{aggregate['selected_candidate_count']}`",
        f"- rendered WAV files: `{aggregate['rendered_audio_file_count']}`",
        f"- total yield floor passed: `{_bool_token(readiness['total_yield_floor_passed'])}`",
        f"- all case yield floor passed: `{_bool_token(readiness['all_case_yield_floor_passed'])}`",
        f"- musical quality claimed: `{_bool_token(readiness['musical_quality_claimed'])}`",
        f"- next boundary: `{decision['next_boundary']}`",
        "",
        "## Cases",
        "",
        "| case | chords | seed | strict | valid | grammar | strict rate | avg notes | avg dead air | package |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in report.get("cases", []):
        package = _dict(row.get("package"))
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['label']}`",
                    f"`{row['chords']}`",
                    str(row["seed"]),
                    f"{row['strict_valid_sample_count']}/{row['sample_count']}",
                    f"{row['valid_sample_count']}/{row['sample_count']}",
                    f"{row['grammar_gate_sample_count']}/{row['sample_count']}",
                    f"{float(row['strict_yield_rate']):.4f}",
                    f"{float(row['strict_avg_note_count']):.2f}",
                    f"{float(row['strict_avg_dead_air_ratio']):.4f}",
                    f"`{package.get('package_report_path', '')}`" if package else "`none`",
                ]
            )
            + " |"
        )
    lines.extend(["", "## Failing Cases", ""])
    if aggregate.get("failing_cases"):
        for row in aggregate["failing_cases"]:
            lines.append(
                f"- `{row['label']}`: strict `{row['strict_valid_sample_count']}` / `{row['sample_count']}`, "
                f"rate `{float(row['strict_yield_rate']):.4f}`"
            )
    else:
        lines.append("- `none`")
    lines.extend(["", "## WAV Files", ""])
    wav_paths = [
        path
        for row in report.get("cases", [])
        for path in _list(_dict(row.get("package")).get("wav_paths"))
    ]
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
    parser = argparse.ArgumentParser(description="Run Music Transformer solo-yield sweep")
    parser.add_argument("--output_root", type=str, default="outputs/music_transformer_finetune_mvp/solo_yield_sweep")
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--progression", action="append", default=[])
    parser.add_argument("--sample_count", type=int, default=6)
    parser.add_argument("--top_n", type=int, default=2)
    parser.add_argument("--seed_start", type=int, default=900)
    parser.add_argument("--seed_stride", type=int, default=37)
    parser.add_argument("--bpm", type=int, default=124)
    parser.add_argument("--bars", type=int, default=2)
    parser.add_argument("--density", type=str, default="medium")
    parser.add_argument("--duration_mode", choices=("cap", "fill"), default="cap")
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--note_groups_per_bar", type=int, default=8)
    parser.add_argument("--pitch_min", type=int, default=55)
    parser.add_argument("--pitch_max", type=int, default=84)
    parser.add_argument("--max_adjacent_interval", type=int, default=7)
    parser.add_argument("--max_sequence", type=int, default=160)
    parser.add_argument("--min_total_strict_yield_rate", type=float, default=0.7)
    parser.add_argument("--min_case_strict_yield_rate", type=float, default=0.5)
    parser.add_argument("--min_cases", type=int, default=1)
    parser.add_argument("--skip_package", action="store_true")
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--require_total_floor", action="store_true")
    parser.add_argument("--require_all_case_floor", action="store_true")
    parser.add_argument("--require_no_quality_claim", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    raw_progressions = args.progression or DEFAULT_PROGRESSIONS
    cases = [parse_progression(raw, index) for index, raw in enumerate(raw_progressions, start=1)]
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    report = build_sweep_report(
        output_dir=output_dir,
        cases=cases,
        checkpoint_dir=Path(args.checkpoint_dir),
        sample_count=int(args.sample_count),
        top_n=int(args.top_n),
        seed_start=int(args.seed_start),
        seed_stride=int(args.seed_stride),
        bpm=int(args.bpm),
        bars=int(args.bars),
        density=str(args.density),
        duration_mode=str(args.duration_mode),
        temperature=float(args.temperature),
        top_k=int(args.top_k),
        note_groups_per_bar=int(args.note_groups_per_bar),
        pitch_min=int(args.pitch_min),
        pitch_max=int(args.pitch_max),
        max_adjacent_interval=int(args.max_adjacent_interval),
        max_sequence=int(args.max_sequence),
        min_total_strict_yield_rate=float(args.min_total_strict_yield_rate),
        min_case_strict_yield_rate=float(args.min_case_strict_yield_rate),
        package_candidates=not bool(args.skip_package),
    )
    summary = validate_sweep_report(
        report,
        min_cases=int(args.min_cases),
        require_total_floor=bool(args.require_total_floor),
        require_all_case_floor=bool(args.require_all_case_floor),
        require_no_quality_claim=bool(args.require_no_quality_claim),
    )
    write_json(output_dir / "solo_yield_sweep_report.json", report)
    write_json(output_dir / "solo_yield_sweep_summary.json", summary)
    markdown = markdown_report(report)
    write_text(output_dir / "solo_yield_sweep_report.md", markdown)
    if args.doc_path:
        write_text(Path(args.doc_path), markdown)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
