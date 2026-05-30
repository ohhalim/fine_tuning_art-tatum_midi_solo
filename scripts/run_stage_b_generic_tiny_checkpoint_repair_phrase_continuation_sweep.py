"""Run phrase-continuation repair sweep for the generic tiny checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from scripts.assess_stage_b_generic_base_readiness import read_json, write_json, write_text  # noqa: E402
from scripts.run_stage_b_generic_tiny_checkpoint_generation_probe import (  # noqa: E402
    _bool_token,
    _dict,
    _float,
    _int,
    run_command,
)


class StageBGenericTinyCheckpointRepairPhraseContinuationSweepError(ValueError):
    pass


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def build_generation_command(
    args: argparse.Namespace,
    *,
    checkpoint_dir: Path,
    output_root: Path,
    run_id: str,
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
        str(args.max_sequence),
        "--num_samples",
        str(args.num_samples),
        "--seed",
        str(args.seed),
        "--temperature",
        str(args.temperature),
        "--top_k",
        str(args.top_k),
        "--min_valid_samples",
        str(args.min_valid_samples),
        "--min_strict_valid_samples",
        str(args.min_strict_valid_samples),
        "--generation_mode",
        "constrained",
        "--constrained_note_groups_per_bar",
        str(args.note_groups_per_bar),
        "--jazz_duration_tokens",
        "--chord_aware_pitches",
        "--chord_pitch_mode",
        "tones_tensions",
        "--chord_pitch_repeat_window",
        str(args.chord_pitch_repeat_window),
        "--postprocess_overlap",
        "--max_simultaneous_notes",
        str(args.max_simultaneous_notes),
        "--require_all_grammar_samples",
    ]


def target_failure_reasons(row: dict[str, Any], *, args: argparse.Namespace) -> list[str]:
    temporal = _dict(row.get("temporal_coverage"))
    pitch_roles = _dict(row.get("pitch_roles"))
    collapse = _dict(row.get("collapse"))
    metrics = _dict(row.get("metrics"))
    reasons: list[str] = []
    if not bool(row.get("grammar_gate_passed", False)):
        reasons.append("grammar_gate_failed")
    if not bool(row.get("strict_valid", False)):
        reasons.append("strict_valid_failed")
    if _int(metrics.get("note_count")) < int(args.min_note_count):
        reasons.append("note_count_below_target")
    if _float(metrics.get("phrase_coverage_ratio")) < float(args.min_phrase_coverage_ratio):
        reasons.append("phrase_coverage_below_target")
    if _int(temporal.get("tail_empty_steps")) > int(args.max_tail_empty_steps):
        reasons.append("tail_empty_above_target")
    if _float(pitch_roles.get("chord_tone_ratio")) < float(args.min_pitch_role_chord_tone_ratio):
        reasons.append("pitch_role_chord_tone_below_target")
    if _int(metrics.get("max_simultaneous_notes")) > int(args.max_simultaneous_notes):
        reasons.append("max_simultaneous_notes_above_target")
    if _float(collapse.get("postprocess_removal_ratio")) > float(args.max_postprocess_removal_ratio):
        reasons.append("postprocess_removal_above_target")
    return reasons


def compact_candidate(row: dict[str, Any], *, args: argparse.Namespace) -> dict[str, Any]:
    temporal = _dict(row.get("temporal_coverage"))
    pitch_roles = _dict(row.get("pitch_roles"))
    collapse = _dict(row.get("collapse"))
    metrics = _dict(row.get("metrics"))
    reasons = target_failure_reasons(row, args=args)
    return {
        "sample_index": _int(row.get("sample_index")),
        "sample_seed": _int(row.get("sample_seed")),
        "midi_path": str(row.get("midi_path") or ""),
        "valid": bool(row.get("valid", False)),
        "strict_valid": bool(row.get("strict_valid", False)),
        "grammar_gate_passed": bool(row.get("grammar_gate_passed", False)),
        "target_qualified": not reasons,
        "target_failure_reasons": reasons,
        "note_count": _int(metrics.get("note_count")),
        "phrase_coverage_ratio": _float(metrics.get("phrase_coverage_ratio")),
        "dead_air_ratio": _float(metrics.get("dead_air_ratio")),
        "tail_empty_steps": _int(temporal.get("tail_empty_steps")),
        "position_span_ratio": _float(temporal.get("position_span_ratio")),
        "pitch_role_chord_tone_ratio": _float(pitch_roles.get("chord_tone_ratio")),
        "metrics_chord_tone_ratio": _float(metrics.get("chord_tone_ratio")),
        "postprocess_removal_ratio": _float(collapse.get("postprocess_removal_ratio")),
        "max_simultaneous_notes": _int(metrics.get("max_simultaneous_notes")),
    }


def sort_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        candidates,
        key=lambda row: (
            not bool(row.get("target_qualified", False)),
            -_float(row.get("pitch_role_chord_tone_ratio")),
            -_float(row.get("phrase_coverage_ratio")),
            _float(row.get("dead_air_ratio")),
            _int(row.get("sample_index")),
        ),
    )


def build_sweep_report(
    *,
    run_dir: Path,
    checkpoint_dir: Path,
    generation_result: dict[str, Any],
    generation_report_path: Path,
    generation_report: dict[str, Any],
    decision_report_path: Path,
    decision_report: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    decision = _dict(decision_report.get("decision"))
    if str(decision.get("next_boundary") or "") != "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_repair_sweep":
        raise StageBGenericTinyCheckpointRepairPhraseContinuationSweepError("unexpected phrase-continuation decision boundary")
    summary = _dict(generation_report.get("summary"))
    candidates = [compact_candidate(row, args=args) for row in _list(generation_report.get("samples")) if isinstance(row, dict)]
    ranked = sort_candidates(candidates)
    target_qualified = [row for row in ranked if bool(row.get("target_qualified", False))]
    target_passed = bool(
        _int(generation_result.get("returncode")) == 0
        and len(target_qualified) >= int(args.min_target_qualified)
        and _int(summary.get("grammar_gate_sample_count")) == _int(summary.get("sample_count"))
    )
    return {
        "schema_version": "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_sweep_v1",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "run_dir": str(run_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "decision_report_path": str(decision_report_path),
        "generation_report_path": str(generation_report_path),
        "input": {
            "issue_number": int(args.issue_number),
            "num_samples": int(args.num_samples),
            "seed": int(args.seed),
            "temperature": float(args.temperature),
            "top_k": int(args.top_k),
            "note_groups_per_bar": int(args.note_groups_per_bar),
            "chord_aware_pitches": True,
            "max_simultaneous_notes": int(args.max_simultaneous_notes),
            "min_note_count": int(args.min_note_count),
            "min_phrase_coverage_ratio": float(args.min_phrase_coverage_ratio),
            "max_tail_empty_steps": int(args.max_tail_empty_steps),
            "min_pitch_role_chord_tone_ratio": float(args.min_pitch_role_chord_tone_ratio),
            "max_postprocess_removal_ratio": float(args.max_postprocess_removal_ratio),
            "min_target_qualified": int(args.min_target_qualified),
        },
        "generation": {
            "command": generation_result,
            "summary": {
                "sample_count": _int(summary.get("sample_count")),
                "valid_sample_count": _int(summary.get("valid_sample_count")),
                "strict_valid_sample_count": _int(summary.get("strict_valid_sample_count")),
                "grammar_gate_sample_count": _int(summary.get("grammar_gate_sample_count")),
                "collapse_warning_sample_rate": _float(summary.get("collapse_warning_sample_rate")),
                "failure_reasons": _dict(summary.get("failure_reasons")),
                "strict_failure_reasons": _dict(summary.get("strict_failure_reasons")),
            },
        },
        "phrase_continuation": {
            "target_passed": target_passed,
            "target_qualified_count": len(target_qualified),
            "candidate_count": len(candidates),
            "ranked_candidates": ranked,
        },
        "readiness": {
            "boundary": "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_repair_sweep",
            "phrase_continuation_repair_target_passed": target_passed,
            "musical_quality_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
        },
        "decision": {
            "current_boundary": "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_repair_sweep",
            "next_boundary": (
                "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_audio_render_package"
                if target_passed
                else "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_sweep_tuning"
            ),
            "auto_progress_allowed": True,
            "critical_user_input_required": False,
            "reason": "target qualification is objective-only; listening quality remains unclaimed",
        },
        "not_proven": [
            "audio_rendered_quality",
            "human_audio_keep",
            "musical_quality",
            "broad_trained_model_quality",
            "brad_style_adaptation",
        ],
        "next_recommended_issue": (
            "Stage B generic tiny checkpoint repair phrase continuation audio render package"
            if target_passed
            else "Stage B generic tiny checkpoint repair phrase continuation sweep tuning"
        ),
    }


def validate_sweep_report(
    report: dict[str, Any],
    *,
    expected_boundary: str | None,
    min_target_qualified: int,
    require_no_quality_claim: bool,
) -> dict[str, Any]:
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    phrase = _dict(report.get("phrase_continuation"))
    boundary = str(readiness.get("boundary") or "")
    if expected_boundary and boundary != expected_boundary:
        raise StageBGenericTinyCheckpointRepairPhraseContinuationSweepError(
            f"expected boundary {expected_boundary}, got {boundary}"
        )
    if _int(phrase.get("target_qualified_count")) < min_target_qualified:
        raise StageBGenericTinyCheckpointRepairPhraseContinuationSweepError("target-qualified candidate count below target")
    if require_no_quality_claim:
        claimed = [
            bool(readiness.get("musical_quality_claimed", True)),
            bool(readiness.get("broad_trained_model_quality_claimed", True)),
            bool(readiness.get("brad_style_adaptation_claimed", True)),
        ]
        if any(claimed):
            raise StageBGenericTinyCheckpointRepairPhraseContinuationSweepError("quality claims must not be set")
    return {
        "boundary": boundary,
        "next_boundary": str(decision.get("next_boundary") or ""),
        "target_passed": bool(phrase.get("target_passed", False)),
        "target_qualified_count": _int(phrase.get("target_qualified_count")),
        "candidate_count": _int(phrase.get("candidate_count")),
        "musical_quality_claimed": bool(readiness.get("musical_quality_claimed", True)),
        "broad_trained_model_quality_claimed": bool(readiness.get("broad_trained_model_quality_claimed", True)),
        "brad_style_adaptation_claimed": bool(readiness.get("brad_style_adaptation_claimed", True)),
        "critical_user_input_required": bool(decision.get("critical_user_input_required", True)),
        "next_recommended_issue": str(report.get("next_recommended_issue") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    readiness = report["readiness"]
    decision = report["decision"]
    generation = report["generation"]["summary"]
    phrase = report["phrase_continuation"]
    lines = [
        "# Stage B Generic Tiny Checkpoint Repair Phrase Continuation Sweep",
        "",
        "## Summary",
        "",
        f"- boundary: `{readiness['boundary']}`",
        f"- next boundary: `{decision['next_boundary']}`",
        f"- target passed: `{_bool_token(phrase['target_passed'])}`",
        f"- target qualified count: `{phrase['target_qualified_count']}`",
        f"- candidate count: `{phrase['candidate_count']}`",
        f"- musical quality claimed: `{_bool_token(readiness['musical_quality_claimed'])}`",
        "",
        "## Generation",
        "",
        f"- sample count: `{generation['sample_count']}`",
        f"- valid / strict / grammar: `{generation['valid_sample_count']}/{generation['strict_valid_sample_count']}/{generation['grammar_gate_sample_count']}`",
        f"- collapse warning sample rate: `{generation['collapse_warning_sample_rate']}`",
        "",
        "## Top Candidates",
        "",
    ]
    for candidate in phrase.get("ranked_candidates", [])[:5]:
        lines.append(
            "- "
            f"sample `{candidate['sample_index']}` seed `{candidate['sample_seed']}` "
            f"target `{_bool_token(candidate['target_qualified'])}` "
            f"note_count `{candidate['note_count']}` "
            f"coverage `{candidate['phrase_coverage_ratio']}` "
            f"tail_empty `{candidate['tail_empty_steps']}` "
            f"chord_role `{candidate['pitch_role_chord_tone_ratio']}` "
            f"midi `{candidate['midi_path']}`"
        )
    lines.extend(["", "## Not Proven", ""])
    for item in report.get("not_proven", []):
        lines.append(f"- `{item}`")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run generic tiny checkpoint phrase continuation repair sweep")
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument(
        "--decision_report",
        type=str,
        default="outputs/stage_b_generic_tiny_checkpoint_repair_phrase_continuation_decision/"
        "harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_decision/"
        "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_decision.json",
    )
    parser.add_argument("--output_root", type=str, default="outputs/stage_b_generic_tiny_checkpoint_repair_phrase_continuation_sweep")
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--issue_number", type=int, default=413)
    parser.add_argument("--max_sequence", type=int, default=160)
    parser.add_argument("--num_samples", type=int, default=6)
    parser.add_argument("--seed", type=int, default=62)
    parser.add_argument("--temperature", type=float, default=0.78)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--min_valid_samples", type=int, default=1)
    parser.add_argument("--min_strict_valid_samples", type=int, default=1)
    parser.add_argument("--note_groups_per_bar", type=int, default=8)
    parser.add_argument("--chord_pitch_repeat_window", type=int, default=2)
    parser.add_argument("--max_simultaneous_notes", type=int, default=1)
    parser.add_argument("--min_note_count", type=int, default=8)
    parser.add_argument("--min_phrase_coverage_ratio", type=float, default=0.85)
    parser.add_argument("--max_tail_empty_steps", type=int, default=2)
    parser.add_argument("--min_pitch_role_chord_tone_ratio", type=float, default=0.5)
    parser.add_argument("--max_postprocess_removal_ratio", type=float, default=0.49)
    parser.add_argument("--min_target_qualified", type=int, default=1)
    parser.add_argument("--expected_boundary", type=str, default="")
    parser.add_argument("--require_no_quality_claim", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_root) / run_id
    checkpoint_dir = Path(args.checkpoint_dir)
    if not (checkpoint_dir / "checkpoint_epoch1.pt").exists():
        raise StageBGenericTinyCheckpointRepairPhraseContinuationSweepError("checkpoint required")
    decision_report_path = Path(args.decision_report)
    if not decision_report_path.exists():
        raise StageBGenericTinyCheckpointRepairPhraseContinuationSweepError("phrase-continuation decision report required")
    probe_output_root = run_dir / "generation_probe"
    probe_run_id = "phrase_continuation_chord_aware"
    command = build_generation_command(
        args,
        checkpoint_dir=checkpoint_dir,
        output_root=probe_output_root,
        run_id=probe_run_id,
    )
    generation_result = run_command(command)
    generation_report_path = probe_output_root / probe_run_id / "report.json"
    if not generation_report_path.exists():
        raise StageBGenericTinyCheckpointRepairPhraseContinuationSweepError("generation report not produced")
    report = build_sweep_report(
        run_dir=run_dir,
        checkpoint_dir=checkpoint_dir,
        generation_result=generation_result,
        generation_report_path=generation_report_path,
        generation_report=read_json(generation_report_path),
        decision_report_path=decision_report_path,
        decision_report=read_json(decision_report_path),
        args=args,
    )
    summary = validate_sweep_report(
        report,
        expected_boundary=str(args.expected_boundary or ""),
        min_target_qualified=int(args.min_target_qualified),
        require_no_quality_claim=bool(args.require_no_quality_claim),
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_sweep.json", report)
    write_json(run_dir / "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_sweep_validation_summary.json", summary)
    markdown = markdown_report(report)
    write_text(run_dir / "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_sweep.md", markdown)
    if args.doc_path:
        write_text(Path(args.doc_path), markdown)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
