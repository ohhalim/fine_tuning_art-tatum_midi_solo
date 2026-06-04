"""Run repeatability probe for controlled checkpoint dead-air repair."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from scripts.assess_stage_b_generic_base_readiness import read_json, write_json, write_text  # noqa: E402


class StageBMidiToSoloControlledScaleCheckpointDeadAirRepairRepeatabilityProbeError(
    ValueError
):
    pass


SOURCE_BOUNDARY = "stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repair_probe"
BOUNDARY = "stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repair_repeatability_probe"
PASS_NEXT_BOUNDARY = "stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repair_repeatability_consolidation"
FAIL_NEXT_BOUNDARY = "stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repeatability_temperature_guard_decision"
SCHEMA_VERSION = "stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repair_repeatability_probe_v1"


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


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


def parse_seeds(raw: str) -> list[int]:
    seeds: list[int] = []
    for part in str(raw).split(","):
        stripped = part.strip()
        if stripped:
            seeds.append(int(stripped))
    if not seeds:
        raise StageBMidiToSoloControlledScaleCheckpointDeadAirRepairRepeatabilityProbeError(
            "seed list required"
        )
    return seeds


def merge_counts(target: dict[str, int], source: dict[str, Any]) -> None:
    for key, value in source.items():
        target[str(key)] = _int(target.get(str(key))) + _int(value)


def run_command(command: Sequence[str]) -> dict[str, Any]:
    completed = subprocess.run(
        list(command),
        cwd=str(ROOT_DIR),
        check=False,
        text=True,
        capture_output=True,
    )
    return {
        "cmd": list(command),
        "returncode": int(completed.returncode),
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
    }


def validate_repair_probe(report: dict[str, Any]) -> dict[str, Any]:
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    repair = _dict(report.get("repair_summary"))
    input_config = _dict(report.get("input"))
    if str(report.get("boundary") or readiness.get("boundary") or "") != SOURCE_BOUNDARY:
        raise StageBMidiToSoloControlledScaleCheckpointDeadAirRepairRepeatabilityProbeError(
            "dead-air repair boundary required"
        )
    if str(decision.get("next_boundary") or "") != BOUNDARY:
        raise StageBMidiToSoloControlledScaleCheckpointDeadAirRepairRepeatabilityProbeError(
            "dead-air repair must route to repeatability probe"
        )
    if not bool(readiness.get("dead_air_target_qualified", False)):
        raise StageBMidiToSoloControlledScaleCheckpointDeadAirRepairRepeatabilityProbeError(
            "dead-air repair target support required"
        )
    if bool(readiness.get("midi_to_solo_musical_quality_claimed", True)):
        raise StageBMidiToSoloControlledScaleCheckpointDeadAirRepairRepeatabilityProbeError(
            "musical quality must not be claimed"
        )
    if _int(repair.get("dead_air_failure_count")) != 0:
        raise StageBMidiToSoloControlledScaleCheckpointDeadAirRepairRepeatabilityProbeError(
            "source dead-air failure must be removed"
        )
    return {
        "source_sample_count": _int(repair.get("sample_count")),
        "source_valid_sample_count": _int(repair.get("valid_sample_count")),
        "source_strict_valid_sample_count": _int(repair.get("strict_valid_sample_count")),
        "source_grammar_gate_sample_count": _int(repair.get("grammar_gate_sample_count")),
        "source_avg_postprocess_removal_ratio": _float(
            repair.get("avg_postprocess_removal_ratio")
        ),
        "source_avg_onset_coverage_ratio": _float(repair.get("avg_onset_coverage_ratio")),
        "source_avg_sustained_coverage_ratio": _float(
            repair.get("avg_sustained_coverage_ratio")
        ),
        "temperature": _float(input_config.get("temperature")),
        "top_k": _int(input_config.get("top_k")),
        "max_sequence": _int(input_config.get("max_sequence")),
        "constrained_note_groups_per_bar": _int(
            input_config.get("constrained_note_groups_per_bar")
        ),
        "coverage_position_window": _int(input_config.get("coverage_position_window")),
        "chord_pitch_mode": str(input_config.get("chord_pitch_mode") or "approach_tensions"),
        "jazz_rhythm_profile": str(input_config.get("jazz_rhythm_profile") or "swing_motif"),
        "max_simultaneous_notes": _int(input_config.get("max_simultaneous_notes")),
    }


def build_generation_command(
    args: argparse.Namespace,
    *,
    seed: int,
    checkpoint_dir: Path,
    output_root: Path,
    run_id: str,
    repair_config: dict[str, Any],
) -> list[str]:
    return [
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
        "--issue_number",
        str(args.issue_number),
        "--max_sequence",
        str(repair_config["max_sequence"] or args.max_sequence),
        "--num_samples",
        str(args.num_samples),
        "--seed",
        str(seed),
        "--temperature",
        str(repair_config["temperature"] or args.temperature),
        "--top_k",
        str(repair_config["top_k"] or args.top_k),
        "--generation_mode",
        "constrained",
        "--constrained_note_groups_per_bar",
        str(repair_config["constrained_note_groups_per_bar"]),
        "--coverage_aware_positions",
        "--coverage_position_window",
        str(repair_config["coverage_position_window"]),
        "--chord_aware_pitches",
        "--chord_pitch_mode",
        str(repair_config["chord_pitch_mode"]),
        "--chord_pitch_repeat_window",
        str(args.chord_pitch_repeat_window),
        "--jazz_rhythm_positions",
        "--jazz_rhythm_profile",
        str(repair_config["jazz_rhythm_profile"]),
        "--jazz_duration_tokens",
        "--cap_duration_to_next_position",
        "--fill_duration_to_next_position",
        "--postprocess_overlap",
        "--max_simultaneous_notes",
        str(repair_config["max_simultaneous_notes"] or args.max_simultaneous_notes),
        "--min_valid_samples",
        str(args.min_valid_samples),
        "--min_strict_valid_samples",
        str(args.min_strict_valid_samples),
    ]


def run_seed_probe(
    args: argparse.Namespace,
    *,
    seed: int,
    checkpoint_dir: Path,
    output_root: Path,
    repair_config: dict[str, Any],
) -> dict[str, Any]:
    run_id = f"seed_{seed}"
    command_result = run_command(
        build_generation_command(
            args,
            seed=seed,
            checkpoint_dir=checkpoint_dir,
            output_root=output_root,
            run_id=run_id,
            repair_config=repair_config,
        )
    )
    report_path = output_root / run_id / "report.json"
    generation_report = read_json(report_path) if report_path.exists() else {}
    summary = _dict(generation_report.get("summary"))
    return {
        "seed": int(seed),
        "generation_report_path": str(report_path),
        "generation_command": command_result,
        "sample_count": _int(summary.get("sample_count")),
        "valid_sample_count": _int(summary.get("valid_sample_count")),
        "strict_valid_sample_count": _int(summary.get("strict_valid_sample_count")),
        "grammar_gate_sample_count": _int(summary.get("grammar_gate_sample_count")),
        "diagnostic_failure_reasons": _dict(summary.get("diagnostic_failure_reasons")),
        "strict_failure_reasons": _dict(summary.get("strict_failure_reasons")),
        "collapse_warning_sample_count": _int(summary.get("collapse_warning_sample_count")),
        "avg_postprocess_removal_ratio": _float(summary.get("avg_postprocess_removal_ratio")),
        "avg_onset_coverage_ratio": _float(summary.get("avg_onset_coverage_ratio")),
        "avg_sustained_coverage_ratio": _float(summary.get("avg_sustained_coverage_ratio")),
        "passed_strict_review_gate": bool(generation_report.get("passed_strict_review_gate", False)),
    }


def aggregate_seed_rows(seed_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    total_samples = sum(_int(row.get("sample_count")) for row in seed_rows)
    total_valid = sum(_int(row.get("valid_sample_count")) for row in seed_rows)
    total_strict = sum(_int(row.get("strict_valid_sample_count")) for row in seed_rows)
    total_grammar = sum(_int(row.get("grammar_gate_sample_count")) for row in seed_rows)
    diagnostics: dict[str, int] = {}
    strict_failures: dict[str, int] = {}
    for row in seed_rows:
        merge_counts(diagnostics, _dict(row.get("diagnostic_failure_reasons")))
        merge_counts(strict_failures, _dict(row.get("strict_failure_reasons")))
    weighted_postprocess = sum(
        _float(row.get("avg_postprocess_removal_ratio")) * _int(row.get("sample_count"))
        for row in seed_rows
    )
    weighted_onset = sum(
        _float(row.get("avg_onset_coverage_ratio")) * _int(row.get("sample_count"))
        for row in seed_rows
    )
    weighted_sustained = sum(
        _float(row.get("avg_sustained_coverage_ratio")) * _int(row.get("sample_count"))
        for row in seed_rows
    )
    return {
        "seed_count": len(seed_rows),
        "seeds": [int(row["seed"]) for row in seed_rows],
        "sample_count": int(total_samples),
        "valid_sample_count": int(total_valid),
        "strict_valid_sample_count": int(total_strict),
        "grammar_gate_sample_count": int(total_grammar),
        "valid_sample_rate": float(total_valid / total_samples) if total_samples else 0.0,
        "strict_valid_sample_rate": float(total_strict / total_samples) if total_samples else 0.0,
        "grammar_gate_sample_rate": float(total_grammar / total_samples) if total_samples else 0.0,
        "diagnostic_failure_reasons": diagnostics,
        "strict_failure_reasons": strict_failures,
        "collapse_warning_sample_count": sum(
            _int(row.get("collapse_warning_sample_count")) for row in seed_rows
        ),
        "avg_postprocess_removal_ratio": (
            float(weighted_postprocess / total_samples) if total_samples else 0.0
        ),
        "avg_onset_coverage_ratio": float(weighted_onset / total_samples) if total_samples else 0.0,
        "avg_sustained_coverage_ratio": (
            float(weighted_sustained / total_samples) if total_samples else 0.0
        ),
        "all_seed_commands_succeeded": all(
            _int(_dict(row.get("generation_command")).get("returncode")) == 0 for row in seed_rows
        ),
        "all_seed_gate_passed": all(bool(row.get("passed_strict_review_gate", False)) for row in seed_rows),
        "all_samples_strict_valid": bool(total_samples > 0 and total_strict == total_samples),
    }


def build_repeatability_report(
    *,
    run_dir: Path,
    source_summary: dict[str, Any],
    seed_rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    aggregate = aggregate_seed_rows(seed_rows)
    target_qualified = (
        bool(aggregate["all_seed_commands_succeeded"])
        and bool(aggregate["all_samples_strict_valid"])
        and _int(aggregate["collapse_warning_sample_count"]) == 0
        and not _dict(aggregate.get("diagnostic_failure_reasons"))
    )
    next_boundary = PASS_NEXT_BOUNDARY if target_qualified else FAIL_NEXT_BOUNDARY
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "run_dir": str(run_dir),
        "boundary": BOUNDARY,
        "source_boundary": SOURCE_BOUNDARY,
        "source_summary": source_summary,
        "input": {
            "issue_number": int(args.issue_number),
            "seeds": parse_seeds(args.seeds),
            "num_samples": int(args.num_samples),
        },
        "seed_rows": seed_rows,
        "aggregate": aggregate,
        "comparison": {
            "source_sample_count": int(source_summary["source_sample_count"]),
            "repeatability_sample_count": int(aggregate["sample_count"]),
            "strict_valid_sample_delta": _int(aggregate["strict_valid_sample_count"])
            - _int(source_summary["source_strict_valid_sample_count"]),
            "postprocess_removal_delta": _float(aggregate["avg_postprocess_removal_ratio"])
            - _float(source_summary["source_avg_postprocess_removal_ratio"]),
            "target_qualified": bool(target_qualified),
        },
        "readiness": {
            "boundary": BOUNDARY,
            "dead_air_repair_repeatability_probe_completed": True,
            "dead_air_repair_repeatability_target_qualified": bool(target_qualified),
            "midi_to_solo_musical_quality_claimed": False,
            "human_audio_preference_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
        },
        "decision": {
            "current_boundary": BOUNDARY,
            "next_boundary": next_boundary,
            "auto_progress_allowed": True,
            "critical_user_input_required": False,
            "reason": (
                "configured repeatability sweep records whether the single-seed dead-air repair "
                "holds across seeds without changing the quality claim boundary"
            ),
        },
        "not_proven": [
            "midi_to_solo_musical_quality",
            "human_audio_preference",
            "broad_trained_model_quality",
            "brad_style_adaptation",
        ],
        "next_recommended_issue": (
            "Stage B MIDI-to-solo controlled checkpoint dead-air repair repeatability consolidation"
            if target_qualified
            else "Stage B MIDI-to-solo controlled checkpoint dead-air repeatability temperature guard decision"
        ),
    }


def validate_repeatability_report(
    report: dict[str, Any],
    *,
    expected_boundary: str | None,
    expected_next_boundary: str | None,
    require_completed: bool,
    require_no_quality_claim: bool,
) -> dict[str, Any]:
    boundary = str(report.get("boundary") or "")
    decision = _dict(report.get("decision"))
    readiness = _dict(report.get("readiness"))
    aggregate = _dict(report.get("aggregate"))
    if expected_boundary and boundary != expected_boundary:
        raise StageBMidiToSoloControlledScaleCheckpointDeadAirRepairRepeatabilityProbeError(
            "unexpected boundary"
        )
    if expected_next_boundary and str(decision.get("next_boundary") or "") != expected_next_boundary:
        raise StageBMidiToSoloControlledScaleCheckpointDeadAirRepairRepeatabilityProbeError(
            "unexpected next boundary"
        )
    if require_completed and not bool(readiness.get("dead_air_repair_repeatability_probe_completed", False)):
        raise StageBMidiToSoloControlledScaleCheckpointDeadAirRepairRepeatabilityProbeError(
            "repeatability probe completion required"
        )
    if _int(aggregate.get("sample_count")) <= 0:
        raise StageBMidiToSoloControlledScaleCheckpointDeadAirRepairRepeatabilityProbeError(
            "repeatability samples required"
        )
    if require_no_quality_claim:
        claimed = [
            bool(readiness.get("midi_to_solo_musical_quality_claimed", True)),
            bool(readiness.get("human_audio_preference_claimed", True)),
            bool(readiness.get("broad_trained_model_quality_claimed", True)),
            bool(readiness.get("brad_style_adaptation_claimed", True)),
        ]
        if any(claimed):
            raise StageBMidiToSoloControlledScaleCheckpointDeadAirRepairRepeatabilityProbeError(
                "quality claims must remain false"
            )
    return {
        "boundary": boundary,
        "next_boundary": str(decision.get("next_boundary") or ""),
        "seed_count": _int(aggregate.get("seed_count")),
        "sample_count": _int(aggregate.get("sample_count")),
        "valid_sample_count": _int(aggregate.get("valid_sample_count")),
        "strict_valid_sample_count": _int(aggregate.get("strict_valid_sample_count")),
        "grammar_gate_sample_count": _int(aggregate.get("grammar_gate_sample_count")),
        "collapse_warning_sample_count": _int(aggregate.get("collapse_warning_sample_count")),
        "diagnostic_failure_reasons": _dict(aggregate.get("diagnostic_failure_reasons")),
        "all_seed_gate_passed": bool(aggregate.get("all_seed_gate_passed", False)),
        "all_samples_strict_valid": bool(aggregate.get("all_samples_strict_valid", False)),
        "dead_air_repair_repeatability_target_qualified": bool(
            readiness.get("dead_air_repair_repeatability_target_qualified", False)
        ),
        "critical_user_input_required": bool(decision.get("critical_user_input_required", True)),
        "midi_to_solo_musical_quality_claimed": bool(
            readiness.get("midi_to_solo_musical_quality_claimed", True)
        ),
        "next_recommended_issue": str(report.get("next_recommended_issue") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    aggregate = report["aggregate"]
    readiness = report["readiness"]
    decision = report["decision"]
    comparison = report["comparison"]
    lines = [
        "# Stage B MIDI-to-Solo Controlled Scale Checkpoint Dead-Air Repair Repeatability Probe",
        "",
        "## Summary",
        "",
        f"- boundary: `{report['boundary']}`",
        f"- next boundary: `{decision['next_boundary']}`",
        f"- seed count: `{aggregate['seed_count']}`",
        f"- sample count: `{aggregate['sample_count']}`",
        f"- dead-air repair repeatability target qualified: `{_bool_token(readiness['dead_air_repair_repeatability_target_qualified'])}`",
        f"- MIDI-to-solo musical quality claimed: `{_bool_token(readiness['midi_to_solo_musical_quality_claimed'])}`",
        "",
        "## Aggregate",
        "",
        f"- valid / strict / grammar: `{aggregate['valid_sample_count']}` / `{aggregate['strict_valid_sample_count']}` / `{aggregate['grammar_gate_sample_count']}`",
        f"- all seed gate passed: `{_bool_token(aggregate['all_seed_gate_passed'])}`",
        f"- all samples strict valid: `{_bool_token(aggregate['all_samples_strict_valid'])}`",
        f"- collapse warning sample count: `{aggregate['collapse_warning_sample_count']}`",
        f"- avg postprocess removal ratio: `{aggregate['avg_postprocess_removal_ratio']}`",
        f"- avg onset / sustained coverage ratio: `{aggregate['avg_onset_coverage_ratio']}` / `{aggregate['avg_sustained_coverage_ratio']}`",
        "",
        "## Delta",
        "",
        f"- source / repeatability sample count: `{comparison['source_sample_count']}` / `{comparison['repeatability_sample_count']}`",
        f"- strict valid sample delta: `{comparison['strict_valid_sample_delta']}`",
        f"- postprocess removal delta: `{comparison['postprocess_removal_delta']}`",
        "",
        "## Seed Rows",
        "",
    ]
    for row in report["seed_rows"]:
        lines.append(
            "- seed `{seed}`: valid/strict/grammar `{valid}`/`{strict}`/`{grammar}`, collapse `{collapse}`".format(
                seed=row["seed"],
                valid=row["valid_sample_count"],
                strict=row["strict_valid_sample_count"],
                grammar=row["grammar_gate_sample_count"],
                collapse=row["collapse_warning_sample_count"],
            )
        )
    lines.extend(["", "## Failure Reasons", ""])
    if aggregate["diagnostic_failure_reasons"]:
        for reason, count in aggregate["diagnostic_failure_reasons"].items():
            lines.append(f"- `{reason}`: `{count}`")
    else:
        lines.append("- none")
    lines.extend(["", "## Not Proven", ""])
    for item in report["not_proven"]:
        lines.append(f"- `{item}`")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run controlled dead-air repair repeatability probe")
    parser.add_argument(
        "--repair_report",
        type=str,
        default="outputs/stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repair_probe/"
        "harness_stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repair_probe/"
        "stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repair_probe.json",
    )
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repair_repeatability_probe",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--issue_number", type=int, default=564)
    parser.add_argument("--seeds", type=str, default="44,52,60")
    parser.add_argument("--num_samples", type=int, default=3)
    parser.add_argument("--max_sequence", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=4)
    parser.add_argument("--chord_pitch_repeat_window", type=int, default=2)
    parser.add_argument("--max_simultaneous_notes", type=int, default=1)
    parser.add_argument("--min_valid_samples", type=int, default=1)
    parser.add_argument("--min_strict_valid_samples", type=int, default=1)
    parser.add_argument("--expected_boundary", type=str, default="")
    parser.add_argument("--expected_next_boundary", type=str, default="")
    parser.add_argument("--require_completed", action="store_true")
    parser.add_argument("--require_no_quality_claim", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    source_summary = validate_repair_probe(read_json(Path(args.repair_report)))
    checkpoint_dir = Path(args.checkpoint_dir)
    if not (checkpoint_dir / "checkpoint_epoch1.pt").exists():
        raise StageBMidiToSoloControlledScaleCheckpointDeadAirRepairRepeatabilityProbeError(
            "checkpoint_epoch1.pt required"
        )
    probe_output_root = run_dir / "generation_probe"
    seed_rows = [
        run_seed_probe(
            args,
            seed=seed,
            checkpoint_dir=checkpoint_dir,
            output_root=probe_output_root,
            repair_config=source_summary,
        )
        for seed in parse_seeds(args.seeds)
    ]
    report = build_repeatability_report(
        run_dir=run_dir,
        source_summary=source_summary,
        seed_rows=seed_rows,
        args=args,
    )
    summary = validate_repeatability_report(
        report,
        expected_boundary=str(args.expected_boundary or ""),
        expected_next_boundary=str(args.expected_next_boundary or ""),
        require_completed=bool(args.require_completed),
        require_no_quality_claim=bool(args.require_no_quality_claim),
    )
    write_json(
        run_dir
        / "stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repair_repeatability_probe.json",
        report,
    )
    write_json(
        run_dir
        / "stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repair_repeatability_probe_validation_summary.json",
        summary,
    )
    markdown = markdown_report(report)
    write_text(
        run_dir
        / "stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repair_repeatability_probe.md",
        markdown,
    )
    if args.doc_path:
        write_text(Path(args.doc_path), markdown)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
