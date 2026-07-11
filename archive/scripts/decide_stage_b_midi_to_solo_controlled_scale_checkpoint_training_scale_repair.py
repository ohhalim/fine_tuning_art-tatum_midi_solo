"""Decide the next repair target after selected-scale checkpoint generation failure."""

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
from scripts.summarize_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_generation_probe import (  # noqa: E402
    BOUNDARY as SOURCE_BOUNDARY,
)


class StageBMidiToSoloControlledScaleCheckpointTrainingScaleRepairDecisionError(
    ValueError
):
    pass


BOUNDARY = "stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_repair_decision"
NEXT_BOUNDARY = (
    "stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_"
    "density_grammar_collapse_repair_probe"
)
SCHEMA_VERSION = (
    "stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_repair_decision_v1"
)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleRepairDecisionError(
            f"report missing: {path}"
        )
    return json.loads(path.read_text(encoding="utf-8"))


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


def _count_prefixed(reasons: dict[str, Any], prefix: str) -> int:
    total = 0
    for reason, count in reasons.items():
        if str(reason).startswith(prefix):
            total += _int(count)
    return total


def validate_generation_probe(report: dict[str, Any]) -> dict[str, Any]:
    if str(report.get("boundary") or "") != SOURCE_BOUNDARY:
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleRepairDecisionError(
            "selected-scale checkpoint generation probe boundary required"
        )
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    training = _dict(report.get("training_source"))
    generation = _dict(report.get("generation_summary"))
    if str(decision.get("next_boundary") or "") != BOUNDARY:
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleRepairDecisionError(
            "generation probe must route to selected-scale repair decision"
        )
    if not bool(readiness.get("generation_path_executable", False)):
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleRepairDecisionError(
            "generation path must be executable"
        )
    if bool(readiness.get("raw_generation_quality_ready", True)):
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleRepairDecisionError(
            "raw generation quality must not be ready"
        )
    if bool(generation.get("passed_strict_review_gate", True)):
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleRepairDecisionError(
            "strict review gate should fail"
        )
    blocked_claims = [
        "model_direct_generation_quality_claimed",
        "midi_to_solo_musical_quality_claimed",
        "human_audio_preference_claimed",
        "broad_trained_model_quality_claimed",
        "brad_style_adaptation_claimed",
    ]
    claimed = [name for name in blocked_claims if bool(readiness.get(name, False))]
    if claimed:
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleRepairDecisionError(
            f"unexpected quality claim: {claimed}"
        )

    diagnostic = _dict(generation.get("diagnostic_failure_reasons"))
    strict = _dict(generation.get("strict_failure_reasons"))
    note_count_failure_count = _count_prefixed(diagnostic, "note count too low:")
    postprocess_failure_count = _count_prefixed(
        strict, "postprocess removal ratio too high:"
    )
    grammar_failure_count = _count_prefixed(strict, "grammar_gate_failed")
    repeated_pair_failure_count = _count_prefixed(
        strict, "repeated position/pitch pair ratio too high:"
    )
    sample_count = _int(generation.get("sample_count"))
    grammar_gate_sample_count = _int(generation.get("grammar_gate_sample_count"))
    if sample_count <= 0:
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleRepairDecisionError(
            "sample count required"
        )
    if note_count_failure_count <= 0:
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleRepairDecisionError(
            "note-count failure evidence required"
        )
    if grammar_gate_sample_count <= 0:
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleRepairDecisionError(
            "grammar gate evidence required"
        )
    if grammar_gate_sample_count >= sample_count and grammar_failure_count <= 0:
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleRepairDecisionError(
            "partial grammar failure evidence required"
        )

    return {
        "selected_train_records": _int(training.get("selected_train_records")),
        "selected_val_records": _int(training.get("selected_val_records")),
        "max_sequence": _int(training.get("max_sequence")),
        "best_validation_loss": _float(training.get("best_validation_loss")),
        "checkpoint_count": _int(training.get("checkpoint_count")),
        "command_returncode": _int(generation.get("command_returncode")),
        "sample_count": sample_count,
        "valid_sample_count": _int(generation.get("valid_sample_count")),
        "strict_valid_sample_count": _int(generation.get("strict_valid_sample_count")),
        "grammar_gate_sample_count": grammar_gate_sample_count,
        "passed_generation_gate": bool(generation.get("passed_generation_gate", False)),
        "passed_grammar_gate": bool(generation.get("passed_grammar_gate", False)),
        "passed_strict_review_gate": bool(generation.get("passed_strict_review_gate", False)),
        "collapse_warning_sample_count": _int(generation.get("collapse_warning_sample_count")),
        "collapse_warning_sample_rate": _float(generation.get("collapse_warning_sample_rate")),
        "avg_onset_coverage_ratio": _float(generation.get("avg_onset_coverage_ratio")),
        "avg_sustained_coverage_ratio": _float(
            generation.get("avg_sustained_coverage_ratio")
        ),
        "max_longest_sustained_empty_run_steps": _int(
            generation.get("max_longest_sustained_empty_run_steps")
        ),
        "avg_postprocess_removal_ratio": _float(
            generation.get("avg_postprocess_removal_ratio")
        ),
        "max_postprocess_removal_ratio": _float(
            generation.get("max_postprocess_removal_ratio")
        ),
        "note_count_failure_count": note_count_failure_count,
        "grammar_failure_count": grammar_failure_count,
        "postprocess_failure_count": postprocess_failure_count,
        "repeated_pair_failure_count": repeated_pair_failure_count,
        "diagnostic_failure_reasons": diagnostic,
        "strict_failure_reasons": strict,
    }


def build_repair_decision(
    generation_probe: dict[str, Any],
    *,
    output_dir: Path,
) -> dict[str, Any]:
    evidence = validate_generation_probe(generation_probe)
    all_samples_note_count_failed = (
        evidence["note_count_failure_count"] == evidence["sample_count"]
    )
    collapse_across_all_samples = (
        evidence["collapse_warning_sample_count"] == evidence["sample_count"]
    )
    partial_grammar_failure = (
        evidence["grammar_gate_sample_count"] < evidence["sample_count"]
        or evidence["grammar_failure_count"] > 0
    )
    postprocess_removal_high = evidence["avg_postprocess_removal_ratio"] > 0.49
    low_coverage = (
        evidence["avg_onset_coverage_ratio"] < 0.125
        and evidence["avg_sustained_coverage_ratio"] < 0.25
    )
    repair_targets = [
        "increase_note_density_before_postprocess",
        "repair_partial_generation_grammar_loss",
        "reduce_postprocess_removed_majority",
        "improve_onset_sustained_coverage",
        "preserve_selected_scale_checkpoint_evidence",
        "preserve_no_quality_claim",
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "boundary": BOUNDARY,
        "source_boundary": SOURCE_BOUNDARY,
        "evidence": {
            **evidence,
            "all_samples_note_count_failed": all_samples_note_count_failed,
            "collapse_across_all_samples": collapse_across_all_samples,
            "partial_grammar_failure": partial_grammar_failure,
            "postprocess_removal_high": postprocess_removal_high,
            "low_coverage_observed": low_coverage,
        },
        "repair_decision": {
            "current_boundary": BOUNDARY,
            "selected_target": "target_density_grammar_collapse_postprocess_repair",
            "repair_targets": repair_targets,
            "postprocess_only_repair_selected": False,
            "audio_review_selected": False,
            "additional_training_scale_selected": False,
            "next_boundary": NEXT_BOUNDARY,
            "auto_progress_allowed": True,
            "critical_user_input_required": False,
            "reason": (
                "selected-scale checkpoint reduced validation loss, but generation still failed "
                "strict MIDI review through note-count collapse, partial grammar loss, and high postprocess removal"
            ),
        },
        "claim_boundary": {
            "boundary": BOUNDARY,
            "repair_target_selected": True,
            "quality_root_cause_claimed": False,
            "model_direct_generation_quality_claimed": False,
            "midi_to_solo_musical_quality_claimed": False,
            "human_audio_preference_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
        },
        "not_proven": [
            "repair_probe_result",
            "quality_root_cause",
            "midi_to_solo_musical_quality",
            "human_audio_preference",
            "broad_trained_model_quality",
            "brad_style_adaptation",
        ],
        "next_recommended_issue": (
            "Stage B MIDI-to-solo controlled scale checkpoint training scale "
            "density grammar collapse repair probe"
        ),
    }


def validate_repair_decision(
    report: dict[str, Any],
    *,
    expected_boundary: str | None,
    expected_next_boundary: str | None,
    require_repair_target: bool,
    require_no_quality_claim: bool,
) -> dict[str, Any]:
    boundary = str(report.get("boundary") or "")
    evidence = _dict(report.get("evidence"))
    decision = _dict(report.get("repair_decision"))
    claim = _dict(report.get("claim_boundary"))
    if expected_boundary and boundary != expected_boundary:
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleRepairDecisionError(
            "unexpected boundary"
        )
    if expected_next_boundary and str(decision.get("next_boundary") or "") != expected_next_boundary:
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleRepairDecisionError(
            "unexpected next boundary"
        )
    if (
        require_repair_target
        and str(decision.get("selected_target") or "")
        != "target_density_grammar_collapse_postprocess_repair"
    ):
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleRepairDecisionError(
            "density/grammar/collapse repair target required"
        )
    if not bool(evidence.get("all_samples_note_count_failed", False)):
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleRepairDecisionError(
            "all-sample note-count failure required"
        )
    if not bool(evidence.get("collapse_across_all_samples", False)):
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleRepairDecisionError(
            "all-sample collapse warning required"
        )
    if not bool(evidence.get("partial_grammar_failure", False)):
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleRepairDecisionError(
            "partial grammar failure required"
        )
    if not bool(evidence.get("postprocess_removal_high", False)):
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleRepairDecisionError(
            "postprocess removal evidence required"
        )
    if require_no_quality_claim:
        blocked = [
            "quality_root_cause_claimed",
            "model_direct_generation_quality_claimed",
            "midi_to_solo_musical_quality_claimed",
            "human_audio_preference_claimed",
            "broad_trained_model_quality_claimed",
            "brad_style_adaptation_claimed",
        ]
        claimed = [name for name in blocked if bool(claim.get(name, True))]
        if claimed:
            raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleRepairDecisionError(
                f"unexpected quality claim: {claimed}"
            )
    return {
        "boundary": boundary,
        "source_boundary": str(report.get("source_boundary") or ""),
        "next_boundary": str(decision.get("next_boundary") or ""),
        "selected_target": str(decision.get("selected_target") or ""),
        "sample_count": _int(evidence.get("sample_count")),
        "valid_sample_count": _int(evidence.get("valid_sample_count")),
        "strict_valid_sample_count": _int(evidence.get("strict_valid_sample_count")),
        "grammar_gate_sample_count": _int(evidence.get("grammar_gate_sample_count")),
        "note_count_failure_count": _int(evidence.get("note_count_failure_count")),
        "grammar_failure_count": _int(evidence.get("grammar_failure_count")),
        "collapse_warning_sample_count": _int(
            evidence.get("collapse_warning_sample_count")
        ),
        "avg_postprocess_removal_ratio": _float(
            evidence.get("avg_postprocess_removal_ratio")
        ),
        "max_postprocess_removal_ratio": _float(
            evidence.get("max_postprocess_removal_ratio")
        ),
        "all_samples_note_count_failed": bool(
            evidence.get("all_samples_note_count_failed", False)
        ),
        "collapse_across_all_samples": bool(
            evidence.get("collapse_across_all_samples", False)
        ),
        "partial_grammar_failure": bool(evidence.get("partial_grammar_failure", False)),
        "postprocess_removal_high": bool(evidence.get("postprocess_removal_high", False)),
        "low_coverage_observed": bool(evidence.get("low_coverage_observed", False)),
        "postprocess_only_repair_selected": bool(
            decision.get("postprocess_only_repair_selected", True)
        ),
        "additional_training_scale_selected": bool(
            decision.get("additional_training_scale_selected", True)
        ),
        "critical_user_input_required": bool(
            decision.get("critical_user_input_required", True)
        ),
        "midi_to_solo_musical_quality_claimed": bool(
            claim.get("midi_to_solo_musical_quality_claimed", True)
        ),
        "next_recommended_issue": str(report.get("next_recommended_issue") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    evidence = report["evidence"]
    decision = report["repair_decision"]
    claim = report["claim_boundary"]
    lines = [
        "# Stage B MIDI-to-Solo Controlled Scale Checkpoint Training Scale Repair Decision",
        "",
        "## Summary",
        "",
        f"- boundary: `{report['boundary']}`",
        f"- next boundary: `{decision['next_boundary']}`",
        f"- selected target: `{decision['selected_target']}`",
        f"- postprocess-only repair selected: `{_bool_token(decision['postprocess_only_repair_selected'])}`",
        f"- audio review selected: `{_bool_token(decision['audio_review_selected'])}`",
        f"- additional training scale selected: `{_bool_token(decision['additional_training_scale_selected'])}`",
        f"- MIDI-to-solo musical quality claimed: `{_bool_token(claim['midi_to_solo_musical_quality_claimed'])}`",
        "",
        "## Evidence",
        "",
        f"- train / val records: `{evidence['selected_train_records']}` / `{evidence['selected_val_records']}`",
        f"- best validation loss: `{evidence['best_validation_loss']}`",
        f"- sample count: `{evidence['sample_count']}`",
        f"- valid / strict / grammar: `{evidence['valid_sample_count']}` / `{evidence['strict_valid_sample_count']}` / `{evidence['grammar_gate_sample_count']}`",
        f"- note count failure count: `{evidence['note_count_failure_count']}`",
        f"- grammar failure count: `{evidence['grammar_failure_count']}`",
        f"- collapse warning sample count / rate: `{evidence['collapse_warning_sample_count']}` / `{evidence['collapse_warning_sample_rate']}`",
        f"- avg onset / sustained coverage ratio: `{evidence['avg_onset_coverage_ratio']}` / `{evidence['avg_sustained_coverage_ratio']}`",
        f"- avg / max postprocess removal ratio: `{evidence['avg_postprocess_removal_ratio']}` / `{evidence['max_postprocess_removal_ratio']}`",
        "",
        "## Repair Targets",
        "",
    ]
    for target in decision["repair_targets"]:
        lines.append(f"- `{target}`")
    lines.extend(["", "## Failure Reasons", ""])
    for reason, count in evidence["diagnostic_failure_reasons"].items():
        lines.append(f"- `{reason}`: `{count}`")
    lines.extend(["", "## Strict Failure Reasons", ""])
    for reason, count in evidence["strict_failure_reasons"].items():
        lines.append(f"- `{reason}`: `{count}`")
    lines.extend(["", "## Not Proven", ""])
    for item in report["not_proven"]:
        lines.append(f"- `{item}`")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Decide selected-scale checkpoint repair target"
    )
    parser.add_argument(
        "--generation_probe",
        type=str,
        default=(
            "outputs/stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_generation_probe/"
            "harness_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_generation_probe/"
            "stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_generation_probe.json"
        ),
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=(
            "outputs/stage_b_midi_to_solo_controlled_scale_checkpoint_"
            "training_scale_repair_decision"
        ),
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--expected_boundary", type=str, default="")
    parser.add_argument("--expected_next_boundary", type=str, default="")
    parser.add_argument("--require_repair_target", action="store_true")
    parser.add_argument("--require_no_quality_claim", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    report = build_repair_decision(
        read_json(Path(args.generation_probe)),
        output_dir=output_dir,
    )
    summary = validate_repair_decision(
        report,
        expected_boundary=str(args.expected_boundary or ""),
        expected_next_boundary=str(args.expected_next_boundary or ""),
        require_repair_target=bool(args.require_repair_target),
        require_no_quality_claim=bool(args.require_no_quality_claim),
    )
    write_json(
        output_dir
        / "stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_repair_decision.json",
        report,
    )
    write_json(
        output_dir
        / "stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_repair_decision_validation_summary.json",
        summary,
    )
    markdown = markdown_report(report)
    write_text(
        output_dir
        / "stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_repair_decision.md",
        markdown,
    )
    if args.doc_path:
        write_text(Path(args.doc_path), markdown)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
