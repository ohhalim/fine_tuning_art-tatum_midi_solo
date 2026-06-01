"""Decide the next repair target after scale-checkpoint raw generation failure."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from scripts.assess_stage_b_generic_base_readiness import read_json, write_json, write_text


class StageBGenericBaseScaleCheckpointGrammarRepresentationDecisionError(ValueError):
    pass


SOURCE_BOUNDARY = "stage_b_generic_base_scale_checkpoint_generation_probe"
BOUNDARY = "stage_b_generic_base_scale_checkpoint_grammar_representation_decision"
NEXT_BOUNDARY = "stage_b_generic_base_scale_checkpoint_density_coverage_repair_probe"


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


def validate_generation_probe(report: dict[str, Any]) -> dict[str, Any]:
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    generation = _dict(report.get("generation_summary"))
    training = _dict(report.get("training_scale_summary"))
    boundary = str(readiness.get("boundary") or "")
    next_boundary = str(decision.get("next_boundary") or "")
    if boundary != SOURCE_BOUNDARY:
        raise StageBGenericBaseScaleCheckpointGrammarRepresentationDecisionError(
            "scale checkpoint generation probe boundary required"
        )
    if next_boundary != BOUNDARY:
        raise StageBGenericBaseScaleCheckpointGrammarRepresentationDecisionError(
            "generation probe must route to grammar/representation decision"
        )
    if not bool(readiness.get("generation_path_executable", False)):
        raise StageBGenericBaseScaleCheckpointGrammarRepresentationDecisionError("generation path must be executable")
    if _int(generation.get("sample_count")) <= 0:
        raise StageBGenericBaseScaleCheckpointGrammarRepresentationDecisionError("generated sample count required")
    if bool(generation.get("passed_strict_review_gate", True)):
        raise StageBGenericBaseScaleCheckpointGrammarRepresentationDecisionError(
            "decision expects strict review gate failure"
        )
    if bool(readiness.get("raw_generation_quality_ready", True)):
        raise StageBGenericBaseScaleCheckpointGrammarRepresentationDecisionError(
            "raw generation quality must not be ready"
        )
    if bool(readiness.get("broad_trained_model_quality_claimed", True)):
        raise StageBGenericBaseScaleCheckpointGrammarRepresentationDecisionError(
            "broad trained-model quality must not be claimed"
        )
    if bool(readiness.get("brad_style_adaptation_claimed", True)):
        raise StageBGenericBaseScaleCheckpointGrammarRepresentationDecisionError(
            "Brad style adaptation must not be claimed"
        )
    diagnostic = _dict(generation.get("diagnostic_failure_reasons"))
    note_count_failures = sum(
        count
        for reason, count in diagnostic.items()
        if str(reason).startswith("note count too low:")
    )
    if note_count_failures <= 0:
        raise StageBGenericBaseScaleCheckpointGrammarRepresentationDecisionError(
            "note-count failure evidence required"
        )
    return {
        "source_tokenized_train_files": _int(training.get("source_tokenized_train_files")),
        "source_tokenized_val_files": _int(training.get("source_tokenized_val_files")),
        "selected_train_records": _int(training.get("selected_train_records")),
        "selected_val_records": _int(training.get("selected_val_records")),
        "best_validation_loss": training.get("best_validation_loss"),
        "checkpoint_count": _int(training.get("checkpoint_count")),
        "sample_count": _int(generation.get("sample_count")),
        "valid_sample_count": _int(generation.get("valid_sample_count")),
        "strict_valid_sample_count": _int(generation.get("strict_valid_sample_count")),
        "grammar_gate_sample_count": _int(generation.get("grammar_gate_sample_count")),
        "collapse_warning_sample_rate": _float(generation.get("collapse_warning_sample_rate")),
        "avg_onset_coverage_ratio": _float(generation.get("avg_onset_coverage_ratio")),
        "avg_sustained_coverage_ratio": _float(generation.get("avg_sustained_coverage_ratio")),
        "max_longest_sustained_empty_run_steps": _int(
            generation.get("max_longest_sustained_empty_run_steps")
        ),
        "diagnostic_failure_reasons": diagnostic,
        "note_count_failure_count": int(note_count_failures),
    }


def build_decision_report(
    generation_probe: dict[str, Any],
    *,
    output_dir: Path,
) -> dict[str, Any]:
    evidence = validate_generation_probe(generation_probe)
    all_samples_note_count_failed = evidence["note_count_failure_count"] == evidence["sample_count"]
    low_coverage = (
        evidence["avg_onset_coverage_ratio"] < 0.125
        and evidence["avg_sustained_coverage_ratio"] < 0.25
    )
    selected_target = "target_density_coverage_repair"
    return {
        "schema_version": "stage_b_generic_base_scale_checkpoint_grammar_representation_decision_v1",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "source_schema": str(generation_probe.get("schema_version") or ""),
        "input_boundary": SOURCE_BOUNDARY,
        "evidence": {
            **evidence,
            "all_samples_note_count_failed": all_samples_note_count_failed,
            "low_coverage_observed": low_coverage,
            "collapse_warning_not_primary": evidence["collapse_warning_sample_rate"] == 0.0,
        },
        "decision": {
            "current_boundary": BOUNDARY,
            "decision": "select_density_coverage_repair_probe",
            "selected_target": selected_target,
            "postprocess_only_repair_selected": False,
            "audio_review_selected": False,
            "next_boundary": NEXT_BOUNDARY,
            "auto_progress_allowed": True,
            "critical_user_input_required": False,
            "reason": (
                "scale checkpoint raw generation produced MIDI files but all samples failed before review "
                "because note count and coverage were too low"
            ),
        },
        "claim_boundary": {
            "boundary": BOUNDARY,
            "raw_generation_failure_classified": True,
            "quality_root_cause_claimed": False,
            "musical_quality_claimed": False,
            "human_audio_preference_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
            "production_ready_improviser_claimed": False,
        },
        "proven": [
            "scale_checkpoint_generation_path_executable",
            "raw_generation_gate_failed",
            "note_count_failure_evidence_recorded",
            "low_coverage_evidence_recorded",
            "next_repair_target_selected",
        ],
        "not_proven": [
            "quality_root_cause",
            "musical_quality",
            "human_audio_preference",
            "broad_trained_model_quality",
            "brad_style_adaptation",
        ],
        "next_recommended_issue": "Stage B generic base scale checkpoint density coverage repair probe",
    }


def validate_decision_report(
    report: dict[str, Any],
    *,
    expected_boundary: str | None,
    expected_next_boundary: str | None,
    require_density_coverage_target: bool,
    require_no_quality_claim: bool,
) -> dict[str, Any]:
    decision = _dict(report.get("decision"))
    claim = _dict(report.get("claim_boundary"))
    evidence = _dict(report.get("evidence"))
    boundary = str(decision.get("current_boundary") or "")
    next_boundary = str(decision.get("next_boundary") or "")
    if expected_boundary and boundary != expected_boundary:
        raise StageBGenericBaseScaleCheckpointGrammarRepresentationDecisionError(
            f"expected boundary {expected_boundary}, got {boundary}"
        )
    if expected_next_boundary and next_boundary != expected_next_boundary:
        raise StageBGenericBaseScaleCheckpointGrammarRepresentationDecisionError(
            f"expected next boundary {expected_next_boundary}, got {next_boundary}"
        )
    if require_density_coverage_target and str(decision.get("selected_target") or "") != "target_density_coverage_repair":
        raise StageBGenericBaseScaleCheckpointGrammarRepresentationDecisionError(
            "density/coverage repair target required"
        )
    if not bool(evidence.get("all_samples_note_count_failed", False)):
        raise StageBGenericBaseScaleCheckpointGrammarRepresentationDecisionError(
            "all-sample note-count failure evidence required"
        )
    if not bool(evidence.get("low_coverage_observed", False)):
        raise StageBGenericBaseScaleCheckpointGrammarRepresentationDecisionError(
            "low coverage evidence required"
        )
    if require_no_quality_claim:
        claimed = [
            bool(claim.get("quality_root_cause_claimed", True)),
            bool(claim.get("musical_quality_claimed", True)),
            bool(claim.get("human_audio_preference_claimed", True)),
            bool(claim.get("broad_trained_model_quality_claimed", True)),
            bool(claim.get("brad_style_adaptation_claimed", True)),
            bool(claim.get("production_ready_improviser_claimed", True)),
        ]
        if any(claimed):
            raise StageBGenericBaseScaleCheckpointGrammarRepresentationDecisionError(
                "quality claims must remain false"
            )
    return {
        "boundary": boundary,
        "input_boundary": str(report.get("input_boundary") or ""),
        "decision": str(decision.get("decision") or ""),
        "selected_target": str(decision.get("selected_target") or ""),
        "postprocess_only_repair_selected": bool(decision.get("postprocess_only_repair_selected", True)),
        "audio_review_selected": bool(decision.get("audio_review_selected", True)),
        "sample_count": _int(evidence.get("sample_count")),
        "valid_sample_count": _int(evidence.get("valid_sample_count")),
        "strict_valid_sample_count": _int(evidence.get("strict_valid_sample_count")),
        "grammar_gate_sample_count": _int(evidence.get("grammar_gate_sample_count")),
        "note_count_failure_count": _int(evidence.get("note_count_failure_count")),
        "all_samples_note_count_failed": bool(evidence.get("all_samples_note_count_failed", False)),
        "low_coverage_observed": bool(evidence.get("low_coverage_observed", False)),
        "collapse_warning_not_primary": bool(evidence.get("collapse_warning_not_primary", False)),
        "quality_root_cause_claimed": bool(claim.get("quality_root_cause_claimed", True)),
        "broad_trained_model_quality_claimed": bool(claim.get("broad_trained_model_quality_claimed", True)),
        "brad_style_adaptation_claimed": bool(claim.get("brad_style_adaptation_claimed", True)),
        "auto_progress_allowed": bool(decision.get("auto_progress_allowed", False)),
        "critical_user_input_required": bool(decision.get("critical_user_input_required", True)),
        "next_boundary": next_boundary,
        "next_recommended_issue": str(report.get("next_recommended_issue") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    evidence = report["evidence"]
    decision = report["decision"]
    claim = report["claim_boundary"]
    lines = [
        "# Stage B Generic Base Scale Checkpoint Grammar Representation Decision",
        "",
        "## Summary",
        "",
        f"- boundary: `{decision['current_boundary']}`",
        f"- decision: `{decision['decision']}`",
        f"- selected target: `{decision['selected_target']}`",
        f"- postprocess-only repair selected: `{_bool_token(decision['postprocess_only_repair_selected'])}`",
        f"- audio review selected: `{_bool_token(decision['audio_review_selected'])}`",
        f"- quality root cause claimed: `{_bool_token(claim['quality_root_cause_claimed'])}`",
        f"- broad trained-model quality claimed: `{_bool_token(claim['broad_trained_model_quality_claimed'])}`",
        f"- Brad style adaptation claimed: `{_bool_token(claim['brad_style_adaptation_claimed'])}`",
        f"- next boundary: `{decision['next_boundary']}`",
        "",
        "## Evidence",
        "",
        f"- sample count: `{evidence['sample_count']}`",
        (
            "- valid / strict / grammar gate sample count: "
            f"`{evidence['valid_sample_count']}` / `{evidence['strict_valid_sample_count']}` / "
            f"`{evidence['grammar_gate_sample_count']}`"
        ),
        f"- note count failure count: `{evidence['note_count_failure_count']}`",
        f"- all samples note-count failed: `{_bool_token(evidence['all_samples_note_count_failed'])}`",
        f"- avg onset / sustained coverage ratio: `{evidence['avg_onset_coverage_ratio']}` / `{evidence['avg_sustained_coverage_ratio']}`",
        f"- max longest sustained empty run steps: `{evidence['max_longest_sustained_empty_run_steps']}`",
        f"- collapse warning not primary: `{_bool_token(evidence['collapse_warning_not_primary'])}`",
        "",
        "## Failure Reasons",
        "",
    ]
    for reason, count in evidence["diagnostic_failure_reasons"].items():
        lines.append(f"- `{reason}`: `{count}`")
    lines.extend(["", "## Not Proven", ""])
    for item in report["not_proven"]:
        lines.append(f"- `{item}`")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Decide Stage B generic base scale checkpoint grammar/representation target"
    )
    parser.add_argument(
        "--generation_probe",
        type=str,
        default="outputs/stage_b_generic_base_scale_checkpoint_generation_probe/"
        "harness_stage_b_generic_base_scale_checkpoint_generation_probe/"
        "stage_b_generic_base_scale_checkpoint_generation_probe.json",
    )
    parser.add_argument("--output_root", type=str, default="outputs/stage_b_generic_base_scale_checkpoint_grammar_representation_decision")
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--expected_boundary", type=str, default="")
    parser.add_argument("--expected_next_boundary", type=str, default="")
    parser.add_argument("--require_density_coverage_target", action="store_true")
    parser.add_argument("--require_no_quality_claim", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    generation_probe = read_json(Path(args.generation_probe))
    report = build_decision_report(generation_probe, output_dir=output_dir)
    summary = validate_decision_report(
        report,
        expected_boundary=str(args.expected_boundary or ""),
        expected_next_boundary=str(args.expected_next_boundary or ""),
        require_density_coverage_target=bool(args.require_density_coverage_target),
        require_no_quality_claim=bool(args.require_no_quality_claim),
    )
    write_json(output_dir / "stage_b_generic_base_scale_checkpoint_grammar_representation_decision.json", report)
    write_json(
        output_dir / "stage_b_generic_base_scale_checkpoint_grammar_representation_decision_validation_summary.json",
        summary,
    )
    markdown = markdown_report(report)
    write_text(output_dir / "stage_b_generic_base_scale_checkpoint_grammar_representation_decision.md", markdown)
    if args.doc_path:
        write_text(Path(args.doc_path), markdown)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
