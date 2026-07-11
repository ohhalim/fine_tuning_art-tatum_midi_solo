"""Run the Stage B MIDI-to-solo model-direct monophonic overlap repair."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from scripts.assess_stage_b_generic_base_readiness import write_json, write_text  # noqa: E402
from scripts.run_stage_b_midi_to_solo_model_direct_8bar_generation_probe import (  # noqa: E402
    BOUNDARY as PREVIOUS_BOUNDARY,
    PASS_NEXT_BOUNDARY,
    build_generation_command,
    read_json,
    run_command,
    summarize_context,
    summarize_generation,
    summarize_repaired_scale_smoke,
    summarize_sequence_budget_repair,
)


class StageBMidiToSoloModelDirectMonophonicOverlapRepairError(ValueError):
    pass


BOUNDARY = "stage_b_midi_to_solo_model_direct_monophonic_overlap_repair"
FAIL_NEXT_BOUNDARY = "stage_b_midi_to_solo_model_direct_timing_density_duration_repair"
SCHEMA_VERSION = "stage_b_midi_to_solo_model_direct_monophonic_overlap_repair_v1"


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _bool_token(value: Any) -> str:
    return "true" if bool(value) else "false"


def summarize_previous_direct_probe(report: dict[str, Any]) -> dict[str, Any]:
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    generation = _dict(report.get("generation_summary"))
    boundary = str(report.get("boundary") or readiness.get("boundary") or "")
    if boundary != PREVIOUS_BOUNDARY:
        raise StageBMidiToSoloModelDirectMonophonicOverlapRepairError("previous direct probe boundary required")
    if str(decision.get("next_boundary") or "") != BOUNDARY:
        raise StageBMidiToSoloModelDirectMonophonicOverlapRepairError("previous direct probe must route to overlap repair")
    if not bool(readiness.get("direct_generated_midi_written", False)):
        raise StageBMidiToSoloModelDirectMonophonicOverlapRepairError("previous direct MIDI evidence required")
    if bool(readiness.get("direct_generation_review_gate_passed", True)):
        raise StageBMidiToSoloModelDirectMonophonicOverlapRepairError("previous review gate should be failed")
    return {
        "boundary": boundary,
        "sample_count": _int(generation.get("sample_count")),
        "valid_sample_count": _int(generation.get("valid_sample_count")),
        "strict_valid_sample_count": _int(generation.get("strict_valid_sample_count")),
        "grammar_gate_sample_count": _int(generation.get("grammar_gate_sample_count")),
        "min_pre_postprocess_note_groups": _int(generation.get("min_pre_postprocess_note_groups")),
        "min_postprocess_note_count": _int(generation.get("min_postprocess_note_count")),
        "max_postprocess_note_count": _int(generation.get("max_postprocess_note_count")),
        "avg_postprocess_removal_ratio": _float(generation.get("avg_postprocess_removal_ratio")),
        "collapse_warning_sample_rate": _float(generation.get("collapse_warning_sample_rate")),
        "direct_generation_review_gate_passed": bool(
            readiness.get("direct_generation_review_gate_passed", False)
        ),
        "model_direct_generation_quality_claimed": bool(
            readiness.get("model_direct_generation_quality_claimed", True)
        ),
        "midi_to_solo_musical_quality_claimed": bool(
            readiness.get("midi_to_solo_musical_quality_claimed", True)
        ),
        "human_audio_preference_claimed": bool(readiness.get("human_audio_preference_claimed", True)),
        "broad_trained_model_quality_claimed": bool(
            readiness.get("broad_trained_model_quality_claimed", True)
        ),
        "brad_style_adaptation_claimed": bool(readiness.get("brad_style_adaptation_claimed", True)),
    }


def build_monophonic_overlap_repair_report(
    *,
    previous_direct_probe: dict[str, Any],
    sequence_budget_repair: dict[str, Any],
    context_report: dict[str, Any],
    repaired_training_scale_smoke: dict[str, Any],
    generation_result: dict[str, Any],
    generation_report: dict[str, Any],
    generation_report_path: Path,
    output_dir: Path,
    issue_number: int,
    target_bars: int,
    note_groups_per_bar: int,
) -> dict[str, Any]:
    previous = summarize_previous_direct_probe(previous_direct_probe)
    sequence = summarize_sequence_budget_repair(sequence_budget_repair)
    context = summarize_context(context_report, target_bars=int(target_bars))
    scale = summarize_repaired_scale_smoke(repaired_training_scale_smoke)
    repaired = summarize_generation(generation_report)
    command_succeeded = _int(generation_result.get("returncode")) == 0
    generated_midi_written = bool(command_succeeded and repaired["sample_count"] > 0 and repaired["all_midi_paths_exist"])
    review_gate_passed = bool(repaired["passed_generation_gate"] and repaired["passed_strict_review_gate"])
    removal_reduced = repaired["avg_postprocess_removal_ratio"] < previous["avg_postprocess_removal_ratio"]
    next_boundary = PASS_NEXT_BOUNDARY if review_gate_passed else FAIL_NEXT_BOUNDARY
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "issue_number": int(issue_number),
        "boundary": BOUNDARY,
        "source_boundaries": {
            "previous_direct_probe": previous["boundary"],
            "sequence_budget_repair": sequence["boundary"],
            "context": context["boundary"],
            "repaired_training_scale_smoke": scale["boundary"],
        },
        "previous_direct_probe_summary": previous,
        "sequence_budget_summary": sequence,
        "context_summary": context,
        "repaired_scale_smoke_summary": scale,
        "repair_config": {
            "generation_source": "model_checkpoint_direct_constrained",
            "target_bars": int(target_bars),
            "note_groups_per_bar": int(note_groups_per_bar),
            "max_sequence": int(scale["max_sequence"]),
            "cap_duration_to_next_position": True,
        },
        "generation_report_path": str(generation_report_path),
        "generation_command": generation_result,
        "repaired_generation_summary": repaired,
        "repair_result": {
            "previous_valid_sample_count": previous["valid_sample_count"],
            "repaired_valid_sample_count": repaired["valid_sample_count"],
            "previous_strict_valid_sample_count": previous["strict_valid_sample_count"],
            "repaired_strict_valid_sample_count": repaired["strict_valid_sample_count"],
            "previous_avg_postprocess_removal_ratio": previous["avg_postprocess_removal_ratio"],
            "repaired_avg_postprocess_removal_ratio": repaired["avg_postprocess_removal_ratio"],
            "previous_collapse_warning_sample_rate": previous["collapse_warning_sample_rate"],
            "repaired_collapse_warning_sample_rate": repaired["collapse_warning_sample_rate"],
            "previous_min_postprocess_note_count": previous["min_postprocess_note_count"],
            "repaired_min_postprocess_note_count": repaired["min_postprocess_note_count"],
            "postprocess_removal_reduced": bool(removal_reduced),
            "review_gate_repaired": bool(review_gate_passed),
        },
        "readiness": {
            "boundary": BOUNDARY,
            "monophonic_overlap_repair_completed": bool(command_succeeded),
            "direct_generated_midi_written": bool(generated_midi_written),
            "postprocess_removal_reduced": bool(removal_reduced),
            "direct_generation_review_gate_passed": bool(review_gate_passed),
            "model_direct_generation_quality_claimed": False,
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
                "duration cap removed monophonic overlap blocker; direct model output can enter audio render"
            )
            if review_gate_passed
            else "duration cap reduced overlap but review gate remains blocked",
        },
        "not_proven": [
            "model_checkpoint_direct_8bar_generation_quality",
            "human_audio_preference",
            "broad_trained_model_quality",
            "brad_style_adaptation",
        ],
        "next_recommended_issue": (
            "Stage B MIDI-to-solo model-direct audio render package"
            if review_gate_passed
            else "Stage B MIDI-to-solo model-direct timing density duration repair"
        ),
    }


def validate_monophonic_overlap_repair_report(
    report: dict[str, Any],
    *,
    expected_boundary: str | None,
    expected_next_boundary: str | None,
    require_repair_completed: bool,
    require_review_gate_repaired: bool,
    require_no_quality_claim: bool,
) -> dict[str, Any]:
    boundary = str(report.get("boundary") or "")
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    result = _dict(report.get("repair_result"))
    previous = _dict(report.get("previous_direct_probe_summary"))
    repaired = _dict(report.get("repaired_generation_summary"))
    command = _dict(report.get("generation_command"))
    if expected_boundary and boundary != expected_boundary:
        raise StageBMidiToSoloModelDirectMonophonicOverlapRepairError(
            f"expected boundary {expected_boundary}, got {boundary}"
        )
    if expected_next_boundary and str(decision.get("next_boundary") or "") != expected_next_boundary:
        raise StageBMidiToSoloModelDirectMonophonicOverlapRepairError("unexpected next boundary")
    if require_repair_completed and not bool(readiness.get("monophonic_overlap_repair_completed", False)):
        raise StageBMidiToSoloModelDirectMonophonicOverlapRepairError("repair command must complete")
    if _int(command.get("returncode")) != 0:
        raise StageBMidiToSoloModelDirectMonophonicOverlapRepairError("repair generation command must succeed")
    if not bool(readiness.get("direct_generated_midi_written", False)):
        raise StageBMidiToSoloModelDirectMonophonicOverlapRepairError("generated MIDI files required")
    if not bool(readiness.get("postprocess_removal_reduced", False)):
        raise StageBMidiToSoloModelDirectMonophonicOverlapRepairError("postprocess removal should be reduced")
    if require_review_gate_repaired and not bool(readiness.get("direct_generation_review_gate_passed", False)):
        raise StageBMidiToSoloModelDirectMonophonicOverlapRepairError("review gate should be repaired")
    if bool(decision.get("critical_user_input_required", True)):
        raise StageBMidiToSoloModelDirectMonophonicOverlapRepairError("critical user input should not be required")
    if require_no_quality_claim:
        blocked = [
            "model_direct_generation_quality_claimed",
            "midi_to_solo_musical_quality_claimed",
            "human_audio_preference_claimed",
            "broad_trained_model_quality_claimed",
            "brad_style_adaptation_claimed",
        ]
        claimed = [name for name in blocked if bool(readiness.get(name, True))]
        if claimed:
            raise StageBMidiToSoloModelDirectMonophonicOverlapRepairError(f"unexpected quality claim: {claimed}")
        upstream_claims = [
            bool(previous.get("model_direct_generation_quality_claimed", True)),
            bool(previous.get("midi_to_solo_musical_quality_claimed", True)),
            bool(previous.get("human_audio_preference_claimed", True)),
            bool(previous.get("broad_trained_model_quality_claimed", True)),
            bool(previous.get("brad_style_adaptation_claimed", True)),
        ]
        if any(upstream_claims):
            raise StageBMidiToSoloModelDirectMonophonicOverlapRepairError(
                "upstream quality claims must remain false"
            )
    return {
        "boundary": boundary,
        "next_boundary": str(decision.get("next_boundary") or ""),
        "cap_duration_to_next_position": bool(_dict(report.get("repair_config")).get("cap_duration_to_next_position", False)),
        "sample_count": _int(repaired.get("sample_count")),
        "previous_valid_sample_count": _int(result.get("previous_valid_sample_count")),
        "repaired_valid_sample_count": _int(result.get("repaired_valid_sample_count")),
        "previous_strict_valid_sample_count": _int(result.get("previous_strict_valid_sample_count")),
        "repaired_strict_valid_sample_count": _int(result.get("repaired_strict_valid_sample_count")),
        "previous_avg_postprocess_removal_ratio": _float(result.get("previous_avg_postprocess_removal_ratio")),
        "repaired_avg_postprocess_removal_ratio": _float(result.get("repaired_avg_postprocess_removal_ratio")),
        "previous_collapse_warning_sample_rate": _float(result.get("previous_collapse_warning_sample_rate")),
        "repaired_collapse_warning_sample_rate": _float(result.get("repaired_collapse_warning_sample_rate")),
        "previous_min_postprocess_note_count": _int(result.get("previous_min_postprocess_note_count")),
        "repaired_min_postprocess_note_count": _int(result.get("repaired_min_postprocess_note_count")),
        "postprocess_removal_reduced": bool(readiness.get("postprocess_removal_reduced", False)),
        "direct_generation_review_gate_passed": bool(readiness.get("direct_generation_review_gate_passed", False)),
        "model_direct_generation_quality_claimed": bool(
            readiness.get("model_direct_generation_quality_claimed", True)
        ),
        "critical_user_input_required": bool(decision.get("critical_user_input_required", True)),
        "next_recommended_issue": str(report.get("next_recommended_issue") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    readiness = report["readiness"]
    decision = report["decision"]
    result = report["repair_result"]
    generation = report["repaired_generation_summary"]
    lines = [
        "# Stage B MIDI-to-Solo Model-Direct Monophonic Overlap Repair",
        "",
        "## Summary",
        "",
        f"- boundary: `{report['boundary']}`",
        f"- next boundary: `{decision['next_boundary']}`",
        f"- cap duration to next position: `{_bool_token(report['repair_config']['cap_duration_to_next_position'])}`",
        f"- postprocess removal reduced: `{_bool_token(readiness['postprocess_removal_reduced'])}`",
        f"- direct generation review gate passed: `{_bool_token(readiness['direct_generation_review_gate_passed'])}`",
        f"- model-direct generation quality claimed: `{_bool_token(readiness['model_direct_generation_quality_claimed'])}`",
        "",
        "## Before / After",
        "",
        f"- valid sample count: `{result['previous_valid_sample_count']}` -> `{result['repaired_valid_sample_count']}`",
        f"- strict valid sample count: `{result['previous_strict_valid_sample_count']}` -> `{result['repaired_strict_valid_sample_count']}`",
        f"- avg postprocess removal ratio: `{result['previous_avg_postprocess_removal_ratio']}` -> `{result['repaired_avg_postprocess_removal_ratio']}`",
        f"- collapse warning sample rate: `{result['previous_collapse_warning_sample_rate']}` -> `{result['repaired_collapse_warning_sample_rate']}`",
        f"- min postprocess note count: `{result['previous_min_postprocess_note_count']}` -> `{result['repaired_min_postprocess_note_count']}`",
        "",
        "## Repaired Generation",
        "",
        f"- sample count: `{generation['sample_count']}`",
        f"- grammar gate sample count: `{generation['grammar_gate_sample_count']}`",
        f"- valid sample count: `{generation['valid_sample_count']}`",
        f"- strict valid sample count: `{generation['strict_valid_sample_count']}`",
        f"- min postprocess note count: `{generation['min_postprocess_note_count']}`",
        f"- max postprocess note count: `{generation['max_postprocess_note_count']}`",
        "",
        "## MIDI Paths",
        "",
    ]
    for path in generation["midi_paths"]:
        lines.append(f"- `{path}`")
    lines.extend(["", "## Not Proven", ""])
    for item in report.get("not_proven", []):
        lines.append(f"- `{item}`")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MIDI-to-solo model-direct monophonic overlap repair")
    parser.add_argument("--previous_direct_probe", type=str, required=True)
    parser.add_argument("--sequence_budget_repair", type=str, required=True)
    parser.add_argument("--context_report", type=str, required=True)
    parser.add_argument("--repaired_training_scale_smoke", type=str, required=True)
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/stage_b_midi_to_solo_model_direct_monophonic_overlap_repair",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--issue_number", type=int, default=499)
    parser.add_argument("--target_bars", type=int, default=8)
    parser.add_argument("--note_groups_per_bar", type=int, default=3)
    parser.add_argument("--num_samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=497)
    parser.add_argument("--max_sequence", type=int, default=160)
    parser.add_argument("--chord_pitch_mode", type=str, default="tones_tensions")
    parser.add_argument("--max_simultaneous_notes", type=int, default=1)
    parser.add_argument("--min_valid_samples", type=int, default=1)
    parser.add_argument("--min_strict_valid_samples", type=int, default=1)
    parser.add_argument("--cap_duration_to_next_position", action="store_true", default=True)
    parser.add_argument("--expected_boundary", type=str, default="")
    parser.add_argument("--expected_next_boundary", type=str, default="")
    parser.add_argument("--require_repair_completed", action="store_true")
    parser.add_argument("--require_review_gate_repaired", action="store_true")
    parser.add_argument("--require_no_quality_claim", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    previous_direct_probe = read_json(Path(args.previous_direct_probe))
    sequence_budget_repair = read_json(Path(args.sequence_budget_repair))
    context_report = read_json(Path(args.context_report))
    repaired_training_scale_smoke = read_json(Path(args.repaired_training_scale_smoke))
    context_summary = summarize_context(context_report, target_bars=int(args.target_bars))
    scale_summary = summarize_repaired_scale_smoke(repaired_training_scale_smoke)
    generation_output_root = output_dir / "generation_probe"
    generation_run_id = "monophonic_overlap_repair"
    generation_result = run_command(
        build_generation_command(
            args=args,
            checkpoint_dir=Path(scale_summary["checkpoint_dir"]),
            generation_output_root=generation_output_root,
            generation_run_id=generation_run_id,
            context_summary=context_summary,
        )
    )
    generation_report_path = generation_output_root / generation_run_id / "report.json"
    generation_report = read_json(generation_report_path) if generation_report_path.exists() else {}
    report = build_monophonic_overlap_repair_report(
        previous_direct_probe=previous_direct_probe,
        sequence_budget_repair=sequence_budget_repair,
        context_report=context_report,
        repaired_training_scale_smoke=repaired_training_scale_smoke,
        generation_result=generation_result,
        generation_report=generation_report,
        generation_report_path=generation_report_path,
        output_dir=output_dir,
        issue_number=int(args.issue_number),
        target_bars=int(args.target_bars),
        note_groups_per_bar=int(args.note_groups_per_bar),
    )
    summary = validate_monophonic_overlap_repair_report(
        report,
        expected_boundary=str(args.expected_boundary or ""),
        expected_next_boundary=str(args.expected_next_boundary or ""),
        require_repair_completed=bool(args.require_repair_completed),
        require_review_gate_repaired=bool(args.require_review_gate_repaired),
        require_no_quality_claim=bool(args.require_no_quality_claim),
    )
    write_json(output_dir / "stage_b_midi_to_solo_model_direct_monophonic_overlap_repair.json", report)
    write_json(output_dir / "stage_b_midi_to_solo_model_direct_monophonic_overlap_repair_validation_summary.json", summary)
    markdown = markdown_report(report)
    write_text(output_dir / "stage_b_midi_to_solo_model_direct_monophonic_overlap_repair.md", markdown)
    if args.doc_path:
        write_text(Path(args.doc_path), markdown)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
