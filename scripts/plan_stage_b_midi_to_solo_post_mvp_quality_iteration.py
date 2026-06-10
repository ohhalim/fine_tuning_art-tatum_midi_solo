"""Plan the first post-MVP MIDI-to-solo musical quality iteration."""

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
from scripts.audit_stage_b_midi_to_solo_final_status import (  # noqa: E402
    BOUNDARY as FINAL_STATUS_BOUNDARY,
    NEXT_BOUNDARY as FINAL_STATUS_NEXT_BOUNDARY,
    StageBMidiToSoloFinalStatusAuditError,
    validate_final_status_audit_report,
)


class StageBMidiToSoloPostMvpQualityIterationPlanError(ValueError):
    pass


BOUNDARY = "stage_b_midi_to_solo_post_mvp_quality_iteration_plan"
NEXT_BOUNDARY = "stage_b_midi_to_solo_quality_rubric_baseline"
SELECTED_TARGET = "quality_rubric_baseline"
SCHEMA_VERSION = "stage_b_midi_to_solo_post_mvp_quality_iteration_plan_v1"

QUALITY_CLAIM_KEYS = [
    "human_audio_preference_claimed",
    "midi_to_solo_musical_quality_claimed",
    "musical_quality_claimed",
    "phrase_bank_musical_quality_claimed",
    "audio_rendered_quality_claimed",
    "model_checkpoint_generation_quality_claimed",
    "model_direct_generation_quality_claimed",
    "broad_trained_model_quality_claimed",
    "brad_style_adaptation_claimed",
    "production_ready_claimed",
]


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _bool_token(value: Any) -> str:
    return "true" if bool(value) else "false"


def _require_no_quality_claim(container: dict[str, Any], *, label: str) -> None:
    claimed = [name for name in QUALITY_CLAIM_KEYS if bool(container.get(name, False))]
    if claimed:
        raise StageBMidiToSoloPostMvpQualityIterationPlanError(
            f"unexpected quality claim in {label}: {claimed}"
        )


def validate_final_status_source(report: dict[str, Any]) -> dict[str, Any]:
    if str(report.get("boundary") or "") != FINAL_STATUS_BOUNDARY:
        raise StageBMidiToSoloPostMvpQualityIterationPlanError(
            "final status audit boundary required"
        )
    try:
        summary = validate_final_status_audit_report(
            report,
            expected_boundary=FINAL_STATUS_BOUNDARY,
            expected_next_boundary=FINAL_STATUS_NEXT_BOUNDARY,
            require_technical_mvp_complete=True,
            require_readme_reflected=True,
            require_no_quality_claim=True,
        )
    except StageBMidiToSoloFinalStatusAuditError as exc:
        raise StageBMidiToSoloPostMvpQualityIterationPlanError(str(exc)) from exc

    final_status = _dict(report.get("final_status"))
    readiness = _dict(report.get("readiness"))
    _require_no_quality_claim(final_status, label="final status")
    _require_no_quality_claim(readiness, label="readiness")

    if not bool(summary.get("technical_mvp_complete", False)):
        raise StageBMidiToSoloPostMvpQualityIterationPlanError(
            "technical MVP completion required"
        )
    if not bool(summary.get("technical_mvp_ready_for_local_review", False)):
        raise StageBMidiToSoloPostMvpQualityIterationPlanError(
            "local review readiness required"
        )
    if not bool(summary.get("readme_final_evidence_reflected", False)):
        raise StageBMidiToSoloPostMvpQualityIterationPlanError(
            "README final evidence reflection required"
        )
    if _int(summary.get("cli_candidate_count")) < 3:
        raise StageBMidiToSoloPostMvpQualityIterationPlanError("CLI candidate count below 3")
    if _int(summary.get("changed_ratio_repair_wav_count")) < 3:
        raise StageBMidiToSoloPostMvpQualityIterationPlanError(
            "changed-ratio repair WAV count below 3"
        )
    if not bool(summary.get("outside_soloing_repair_evidence_ready", False)):
        raise StageBMidiToSoloPostMvpQualityIterationPlanError(
            "outside-soloing repair evidence readiness required"
        )
    if _int(summary.get("outside_soloing_repair_wav_count")) < 6:
        raise StageBMidiToSoloPostMvpQualityIterationPlanError(
            "outside-soloing repair WAV count below 6"
        )
    if _int(summary.get("outside_soloing_repair_pitch_role_risk_count_after")) != 0:
        raise StageBMidiToSoloPostMvpQualityIterationPlanError(
            "outside-soloing residual pitch-role risk should be zero"
        )
    if not bool(summary.get("listening_review_quality_gap_open", False)):
        raise StageBMidiToSoloPostMvpQualityIterationPlanError(
            "listening review quality gap should remain open"
        )
    if bool(summary.get("raw_artifact_upload_required", True)):
        raise StageBMidiToSoloPostMvpQualityIterationPlanError(
            "raw artifact upload should not be required"
        )
    if bool(summary.get("critical_user_input_required", True)):
        raise StageBMidiToSoloPostMvpQualityIterationPlanError(
            "critical user input should not be required"
        )
    return summary


def build_post_mvp_quality_iteration_plan_report(
    *,
    final_status_audit: dict[str, Any],
    output_dir: Path,
    issue_number: int,
) -> dict[str, Any]:
    source = validate_final_status_source(final_status_audit)
    ordered_work = [
        {
            "target": "quality_rubric_baseline",
            "objective": "define MIDI-evidence quality rubric before another repair sweep",
            "reason": "technical MVP is complete; musical-quality failure boundary is still unscored",
            "required_input": "current ranked MIDI/WAV evidence and existing objective metrics",
            "claim_boundary": "no musical quality or preference claim",
        },
        {
            "target": "candidate_failure_labeling",
            "objective": "label current candidates against sparse-output, songlike-melody, outside-soloing, monotony, and phrase-shape failures",
            "reason": "previous listening feedback rejected musical quality while technical gates passed",
            "required_input": "MIDI notes, timing grid, pitch role, cadence, and rendered WAV metadata",
            "claim_boundary": "objective labels only",
        },
        {
            "target": "targeted_quality_repair_sweep",
            "objective": "run repair/generation sweep against the highest-count failure labels",
            "reason": "avoid unbounded parameter changes after technical MVP completion",
            "required_input": "rubric baseline report and labeled candidate failures",
            "claim_boundary": "repair evidence, not final quality",
        },
        {
            "target": "audio_review_package",
            "objective": "render selected repaired candidates for listening comparison",
            "reason": "quality preference remains unclaimed until reviewed audio input exists",
            "required_input": "ranked repaired MIDI candidates",
            "claim_boundary": "technical WAV validation only",
        },
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "issue_number": int(issue_number),
        "boundary": BOUNDARY,
        "source_boundary": FINAL_STATUS_BOUNDARY,
        "source_summary": source,
        "post_mvp_status": {
            "technical_mvp_complete": bool(source["technical_mvp_complete"]),
            "local_review_ready": bool(source["technical_mvp_ready_for_local_review"]),
            "readme_final_evidence_reflected": bool(source["readme_final_evidence_reflected"]),
            "cli_candidate_count": _int(source["cli_candidate_count"]),
            "changed_ratio_repair_wav_count": _int(source["changed_ratio_repair_wav_count"]),
            "outside_soloing_repair_evidence_ready": bool(
                source["outside_soloing_repair_evidence_ready"]
            ),
            "outside_soloing_repair_wav_count": _int(
                source["outside_soloing_repair_wav_count"]
            ),
            "outside_soloing_repair_changed_note_total": _int(
                source["outside_soloing_repair_changed_note_total"]
            ),
            "outside_soloing_repair_pitch_role_risk_count_after": _int(
                source["outside_soloing_repair_pitch_role_risk_count_after"]
            ),
            "outside_soloing_repair_pitch_role_risk_delta": _int(
                source["outside_soloing_repair_pitch_role_risk_delta"]
            ),
            "listening_review_quality_gap_open": bool(
                source["listening_review_quality_gap_open"]
            ),
            "raw_artifact_upload_required": False,
            "human_audio_preference_claimed": False,
            "midi_to_solo_musical_quality_claimed": False,
        },
        "selected_next_target": {
            "selected_target": SELECTED_TARGET,
            "selected_next_boundary": NEXT_BOUNDARY,
            "reason": "technical MVP is complete; the next blocker is an explicit musical-quality rubric and failure taxonomy",
        },
        "ordered_work": ordered_work,
        "quality_failure_taxonomy_seed": [
            "sparse_or_empty_output",
            "songlike_melody_not_soloing",
            "outside_soloing_without_context",
            "rhythmic_monotony",
            "phrase_shape_missing_tension_release",
            "weak_chord_tone_landing",
            "dead_air_or_density_gap",
        ],
        "readiness": {
            "boundary": BOUNDARY,
            "post_mvp_quality_iteration_plan_completed": True,
            "selected_target": SELECTED_TARGET,
            "technical_mvp_complete": True,
            "quality_rubric_required": True,
            "candidate_failure_labeling_required": True,
            "targeted_quality_repair_sweep_required": True,
            "audio_review_package_required": True,
            "human_audio_preference_claimed": False,
            "midi_to_solo_musical_quality_claimed": False,
            "audio_rendered_quality_claimed": False,
            "model_checkpoint_generation_quality_claimed": False,
            "model_direct_generation_quality_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
            "production_ready_claimed": False,
        },
        "decision": {
            "current_boundary": BOUNDARY,
            "next_boundary": NEXT_BOUNDARY,
            "auto_progress_allowed": True,
            "critical_user_input_required": False,
            "reason": "quality iteration can continue from MIDI evidence without claiming listening preference",
        },
        "not_proven": [
            "quality_rubric_baseline_completed",
            "candidate_failure_labels_completed",
            "targeted_quality_repair_completed",
            "listening_review_completed",
            "human_audio_preference",
            "midi_to_solo_musical_quality",
            "broad_trained_model_quality",
        ],
        "next_recommended_issue": "Stage B MIDI-to-solo quality rubric baseline",
    }


def validate_post_mvp_quality_iteration_plan_report(
    report: dict[str, Any],
    *,
    expected_boundary: str | None,
    expected_next_boundary: str | None,
    expected_target: str | None,
    require_quality_rubric: bool,
    require_no_quality_claim: bool,
) -> dict[str, Any]:
    boundary = str(report.get("boundary") or "")
    decision = _dict(report.get("decision"))
    readiness = _dict(report.get("readiness"))
    selected = _dict(report.get("selected_next_target"))
    post_mvp_status = _dict(report.get("post_mvp_status"))
    ordered_work = report.get("ordered_work")
    if expected_boundary and boundary != expected_boundary:
        raise StageBMidiToSoloPostMvpQualityIterationPlanError(
            f"expected boundary {expected_boundary}, got {boundary}"
        )
    if expected_next_boundary and str(decision.get("next_boundary") or "") != expected_next_boundary:
        raise StageBMidiToSoloPostMvpQualityIterationPlanError("unexpected next boundary")
    if expected_target and str(selected.get("selected_target") or "") != expected_target:
        raise StageBMidiToSoloPostMvpQualityIterationPlanError("unexpected selected target")
    if not bool(readiness.get("post_mvp_quality_iteration_plan_completed", False)):
        raise StageBMidiToSoloPostMvpQualityIterationPlanError(
            "post-MVP quality iteration plan completion required"
        )
    if not bool(post_mvp_status.get("technical_mvp_complete", False)):
        raise StageBMidiToSoloPostMvpQualityIterationPlanError("technical MVP completion required")
    if require_quality_rubric and not bool(readiness.get("quality_rubric_required", False)):
        raise StageBMidiToSoloPostMvpQualityIterationPlanError("quality rubric requirement expected")
    if not isinstance(ordered_work, list) or len(ordered_work) < 4:
        raise StageBMidiToSoloPostMvpQualityIterationPlanError("ordered work must include at least 4 steps")
    if str(ordered_work[0].get("target") if isinstance(ordered_work[0], dict) else "") != SELECTED_TARGET:
        raise StageBMidiToSoloPostMvpQualityIterationPlanError("quality rubric must be first ordered target")
    if bool(decision.get("critical_user_input_required", True)):
        raise StageBMidiToSoloPostMvpQualityIterationPlanError(
            "critical user input should not be required"
        )
    if require_no_quality_claim:
        _require_no_quality_claim(readiness, label="post-MVP quality iteration readiness")
        _require_no_quality_claim(post_mvp_status, label="post-MVP status")
    return {
        "boundary": boundary,
        "next_boundary": str(decision.get("next_boundary") or ""),
        "selected_target": str(selected.get("selected_target") or ""),
        "post_mvp_quality_iteration_plan_completed": bool(
            readiness.get("post_mvp_quality_iteration_plan_completed", False)
        ),
        "technical_mvp_complete": bool(post_mvp_status.get("technical_mvp_complete", False)),
        "local_review_ready": bool(post_mvp_status.get("local_review_ready", False)),
        "quality_rubric_required": bool(readiness.get("quality_rubric_required", False)),
        "outside_soloing_repair_evidence_ready": bool(
            post_mvp_status.get("outside_soloing_repair_evidence_ready", False)
        ),
        "outside_soloing_repair_wav_count": _int(
            post_mvp_status.get("outside_soloing_repair_wav_count")
        ),
        "outside_soloing_repair_changed_note_total": _int(
            post_mvp_status.get("outside_soloing_repair_changed_note_total")
        ),
        "outside_soloing_repair_pitch_role_risk_count_after": _int(
            post_mvp_status.get("outside_soloing_repair_pitch_role_risk_count_after")
        ),
        "outside_soloing_repair_pitch_role_risk_delta": _int(
            post_mvp_status.get("outside_soloing_repair_pitch_role_risk_delta")
        ),
        "candidate_failure_labeling_required": bool(
            readiness.get("candidate_failure_labeling_required", False)
        ),
        "targeted_quality_repair_sweep_required": bool(
            readiness.get("targeted_quality_repair_sweep_required", False)
        ),
        "audio_review_package_required": bool(
            readiness.get("audio_review_package_required", False)
        ),
        "ordered_work_count": len(ordered_work),
        "quality_failure_taxonomy_seed_count": len(
            report.get("quality_failure_taxonomy_seed")
            if isinstance(report.get("quality_failure_taxonomy_seed"), list)
            else []
        ),
        "human_audio_preference_claimed": bool(
            readiness.get("human_audio_preference_claimed", True)
        ),
        "midi_to_solo_musical_quality_claimed": bool(
            readiness.get("midi_to_solo_musical_quality_claimed", True)
        ),
        "critical_user_input_required": bool(decision.get("critical_user_input_required", True)),
        "next_recommended_issue": str(report.get("next_recommended_issue") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    status = report["post_mvp_status"]
    selected = report["selected_next_target"]
    readiness = report["readiness"]
    lines = [
        "# Stage B MIDI-to-Solo Post-MVP Quality Iteration Plan",
        "",
        "## Summary",
        "",
        f"- boundary: `{report['boundary']}`",
        f"- source boundary: `{report['source_boundary']}`",
        f"- next boundary: `{selected['selected_next_boundary']}`",
        f"- selected target: `{selected['selected_target']}`",
        f"- technical MVP complete: `{_bool_token(status['technical_mvp_complete'])}`",
        f"- local review ready: `{_bool_token(status['local_review_ready'])}`",
        f"- outside-soloing repair evidence ready: `{_bool_token(status['outside_soloing_repair_evidence_ready'])}`",
        f"- outside-soloing repair WAV count: `{status['outside_soloing_repair_wav_count']}`",
        f"- outside-soloing pitch-role risk after / delta: `{status['outside_soloing_repair_pitch_role_risk_count_after']}` / `{status['outside_soloing_repair_pitch_role_risk_delta']}`",
        "",
        "## Required Work",
        "",
    ]
    for item in report["ordered_work"]:
        lines.append(f"- `{item['target']}`: {item['objective']}")
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            f"- human/audio preference claimed: `{_bool_token(readiness['human_audio_preference_claimed'])}`",
            f"- MIDI-to-solo musical quality claimed: `{_bool_token(readiness['midi_to_solo_musical_quality_claimed'])}`",
            f"- broad trained model quality claimed: `{_bool_token(readiness['broad_trained_model_quality_claimed'])}`",
            "",
            "## Next",
            "",
            f"- `{report['next_recommended_issue']}`",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plan post-MVP MIDI-to-solo quality iteration")
    parser.add_argument("--final_status_audit", type=str, required=True)
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/stage_b_midi_to_solo_post_mvp_quality_iteration_plan",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--issue_number", type=int, default=744)
    parser.add_argument("--expected_boundary", type=str, default="")
    parser.add_argument("--expected_next_boundary", type=str, default="")
    parser.add_argument("--expected_target", type=str, default="")
    parser.add_argument("--require_quality_rubric", action="store_true")
    parser.add_argument("--require_no_quality_claim", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    report = build_post_mvp_quality_iteration_plan_report(
        final_status_audit=read_json(Path(args.final_status_audit)),
        output_dir=output_dir,
        issue_number=int(args.issue_number),
    )
    summary = validate_post_mvp_quality_iteration_plan_report(
        report,
        expected_boundary=str(args.expected_boundary or ""),
        expected_next_boundary=str(args.expected_next_boundary or ""),
        expected_target=str(args.expected_target or ""),
        require_quality_rubric=bool(args.require_quality_rubric),
        require_no_quality_claim=bool(args.require_no_quality_claim),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        output_dir / "stage_b_midi_to_solo_post_mvp_quality_iteration_plan.json",
        report,
    )
    write_json(
        output_dir / "stage_b_midi_to_solo_post_mvp_quality_iteration_plan_validation_summary.json",
        summary,
    )
    markdown = markdown_report(report)
    write_text(output_dir / "stage_b_midi_to_solo_post_mvp_quality_iteration_plan.md", markdown)
    if args.doc_path:
        write_text(Path(args.doc_path), markdown)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
