"""Select the next repair target from chord-context pitch-role bridge evidence."""

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
from scripts.build_stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_context_pitch_role_bridge import (  # noqa: E402
    BOUNDARY as BRIDGE_BOUNDARY,
    BRIDGE_REQUIRED_SOURCE_CONTEXT_KEYS,
    NEXT_BOUNDARY as BRIDGE_NEXT_BOUNDARY,
)


class StageBMidiToSoloPitchRoleObjectiveDecisionError(ValueError):
    pass


BOUNDARY = (
    "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_context_pitch_role_objective_decision"
)
NEXT_BOUNDARY = (
    "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_tone_landing_repair_sweep"
)
SELECTED_TARGET = "songlike_melody_contour_phrase_rhythm_chord_tone_landing_repair_sweep"
SCHEMA_VERSION = "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_context_pitch_role_objective_decision_v3"
PRIMARY_RISK_LABEL = "weak_chord_tone_landing_risk"
SECONDARY_RISK_LABEL = "outside_soloing_pitch_role_risk"
QUALITY_CLAIM_KEYS = [
    "human_audio_preference_claimed",
    "midi_to_solo_musical_quality_claimed",
    "musical_quality_claimed",
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


def _float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _bool_token(value: Any) -> str:
    return "true" if bool(value) else "false"


def _require_no_quality_claim(container: dict[str, Any], *, label: str) -> None:
    claimed = [name for name in QUALITY_CLAIM_KEYS if bool(container.get(name, False))]
    if claimed:
        raise StageBMidiToSoloPitchRoleObjectiveDecisionError(
            f"unexpected quality claim in {label}: {claimed}"
        )


def validate_bridge_source(report: dict[str, Any]) -> dict[str, Any]:
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    summary = _dict(report.get("summary"))
    flags = _dict(summary.get("bridge_flag_counts"))
    if str(report.get("boundary") or "") != BRIDGE_BOUNDARY:
        raise StageBMidiToSoloPitchRoleObjectiveDecisionError(
            "chord-context pitch-role bridge boundary required"
        )
    if str(decision.get("next_boundary") or "") != BRIDGE_NEXT_BOUNDARY:
        raise StageBMidiToSoloPitchRoleObjectiveDecisionError(
            "bridge must route to pitch-role objective decision"
        )
    if not bool(readiness.get("chord_context_pitch_role_bridge_completed", False)):
        raise StageBMidiToSoloPitchRoleObjectiveDecisionError(
            "bridge completion required"
        )
    candidate_count = _int(readiness.get("candidate_count"))
    if candidate_count < 6:
        raise StageBMidiToSoloPitchRoleObjectiveDecisionError(
            "candidate count below 6"
        )
    if _int(readiness.get("chord_context_available_count")) != candidate_count:
        raise StageBMidiToSoloPitchRoleObjectiveDecisionError(
            "all candidates must have chord context"
        )
    if _int(readiness.get("pitch_role_metrics_defined_count")) != candidate_count:
        raise StageBMidiToSoloPitchRoleObjectiveDecisionError(
            "all candidates must have pitch-role metrics"
        )
    if _int(readiness.get("not_evaluable_after_count")) != 0:
        raise StageBMidiToSoloPitchRoleObjectiveDecisionError(
            "not-evaluable labels must be cleared before objective decision"
        )
    if _int(readiness.get("followup_objective_source_outside_soloing_not_evaluable_count")) <= 0:
        raise StageBMidiToSoloPitchRoleObjectiveDecisionError(
            "follow-up objective outside-soloing not-evaluable count should be preserved"
        )
    if _int(readiness.get("followup_repair_sweep_source_outside_soloing_not_evaluable_count")) <= 0:
        raise StageBMidiToSoloPitchRoleObjectiveDecisionError(
            "follow-up repair sweep outside-soloing not-evaluable count should be preserved"
        )
    if _int(readiness.get("repair_sweep_source_outside_soloing_not_evaluable_count")) <= 0:
        raise StageBMidiToSoloPitchRoleObjectiveDecisionError(
            "bridge repair sweep outside-soloing not-evaluable count should be preserved"
        )
    if bool(decision.get("critical_user_input_required", True)):
        raise StageBMidiToSoloPitchRoleObjectiveDecisionError(
            "critical user input should not be required"
        )
    _require_no_quality_claim(readiness, label="bridge readiness")
    for key in BRIDGE_REQUIRED_SOURCE_CONTEXT_KEYS:
        if key not in readiness:
            raise StageBMidiToSoloPitchRoleObjectiveDecisionError(
                f"bridge source-context field required: {key}"
            )
    for key in (
        "followup_objective_source_outside_soloing_source_context_preserved",
        "followup_repair_sweep_source_outside_soloing_source_context_preserved",
        "repair_sweep_source_outside_soloing_source_context_preserved",
    ):
        if not bool(readiness.get(key, False)):
            raise StageBMidiToSoloPitchRoleObjectiveDecisionError(
                f"source outside-soloing context should be preserved: {key}"
            )
    for key in (
        "followup_objective_source_outside_soloing_source_targeted",
        "followup_repair_sweep_source_outside_soloing_source_targeted",
        "repair_sweep_source_outside_soloing_source_targeted",
    ):
        if bool(readiness.get(key, True)):
            raise StageBMidiToSoloPitchRoleObjectiveDecisionError(
                f"source outside-soloing target should remain false: {key}"
            )
    for key in (
        "followup_objective_source_outside_soloing_source_residual_risk_preserved",
        "followup_repair_sweep_source_outside_soloing_source_residual_risk_preserved",
        "repair_sweep_source_outside_soloing_source_residual_risk_preserved",
    ):
        if not bool(readiness.get(key, False)):
            raise StageBMidiToSoloPitchRoleObjectiveDecisionError(
                f"source outside-soloing residual risk should be preserved: {key}"
            )
    for key in (
        "followup_objective_source_outside_soloing_current_pitch_role_risk_count_after",
        "followup_repair_sweep_source_outside_soloing_current_pitch_role_risk_count_after",
        "repair_sweep_source_outside_soloing_current_pitch_role_risk_count_after",
    ):
        if _int(readiness.get(key)) != 0:
            raise StageBMidiToSoloPitchRoleObjectiveDecisionError(
                f"current outside-soloing risk should remain resolved: {key}"
            )
    return {
        "boundary": BRIDGE_BOUNDARY,
        "candidate_count": candidate_count,
        "not_evaluable_before_count": _int(readiness.get("not_evaluable_before_count")),
        "not_evaluable_after_count": _int(readiness.get("not_evaluable_after_count")),
        "weak_chord_tone_landing_risk_count": _int(flags.get(PRIMARY_RISK_LABEL)),
        "outside_soloing_pitch_role_risk_count": _int(flags.get(SECONDARY_RISK_LABEL)),
        "followup_objective_source_outside_soloing_not_evaluable_count": _int(
            readiness.get("followup_objective_source_outside_soloing_not_evaluable_count")
        ),
        "followup_objective_repaired_outside_soloing_not_evaluable_count": _int(
            readiness.get("followup_objective_repaired_outside_soloing_not_evaluable_count")
        ),
        "followup_repair_sweep_source_outside_soloing_not_evaluable_count": _int(
            readiness.get("followup_repair_sweep_source_outside_soloing_not_evaluable_count")
        ),
        "followup_repair_sweep_repaired_outside_soloing_not_evaluable_count": _int(
            readiness.get("followup_repair_sweep_repaired_outside_soloing_not_evaluable_count")
        ),
        "repair_sweep_source_outside_soloing_not_evaluable_count": _int(
            readiness.get("repair_sweep_source_outside_soloing_not_evaluable_count")
        ),
        "repair_sweep_repaired_outside_soloing_not_evaluable_count": _int(
            readiness.get("repair_sweep_repaired_outside_soloing_not_evaluable_count")
        ),
        "min_chord_tone_ratio": _float(summary.get("min_chord_tone_ratio")),
        "max_outside_ratio": _float(summary.get("max_outside_ratio")),
        "max_non_chord_tone_run": _int(summary.get("max_non_chord_tone_run")),
        **{key: readiness.get(key) for key in BRIDGE_REQUIRED_SOURCE_CONTEXT_KEYS},
    }


def build_objective_decision_report(
    *,
    bridge_report: dict[str, Any],
    output_dir: Path,
    issue_number: int,
) -> dict[str, Any]:
    bridge = validate_bridge_source(bridge_report)
    weak_count = _int(bridge["weak_chord_tone_landing_risk_count"])
    outside_count = _int(bridge["outside_soloing_pitch_role_risk_count"])
    if weak_count <= 0 and outside_count <= 0:
        selected_target = "songlike_melody_contour_phrase_rhythm_pitch_role_audio_review_package"
        next_boundary = (
            "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_pitch_role_audio_review_package"
        )
        primary_label = "no_pitch_role_repair_required"
    elif weak_count >= outside_count:
        selected_target = SELECTED_TARGET
        next_boundary = NEXT_BOUNDARY
        primary_label = PRIMARY_RISK_LABEL
    else:
        selected_target = "songlike_melody_contour_phrase_rhythm_outside_soloing_pitch_role_repair_sweep"
        next_boundary = (
            "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_outside_soloing_pitch_role_repair_sweep"
        )
        primary_label = SECONDARY_RISK_LABEL
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "issue_number": int(issue_number),
        "boundary": BOUNDARY,
        "source_boundary": bridge["boundary"],
        "objective_summary": bridge,
        "selected_next_target": {
            "selected_target": selected_target,
            "selected_next_boundary": next_boundary,
            "primary_risk_label": primary_label,
            "weak_chord_tone_landing_risk_count": weak_count,
            "outside_soloing_pitch_role_risk_count": outside_count,
            "reason": "largest pitch-role risk count selected as next repair target",
        },
        "readiness": {
            "boundary": BOUNDARY,
            "pitch_role_objective_decision_completed": True,
            "candidate_count": _int(bridge["candidate_count"]),
            "primary_risk_label": primary_label,
            "weak_chord_tone_landing_risk_count": weak_count,
            "outside_soloing_pitch_role_risk_count": outside_count,
            "followup_objective_source_outside_soloing_not_evaluable_count": _int(
                bridge["followup_objective_source_outside_soloing_not_evaluable_count"]
            ),
            "followup_objective_repaired_outside_soloing_not_evaluable_count": _int(
                bridge["followup_objective_repaired_outside_soloing_not_evaluable_count"]
            ),
            "followup_repair_sweep_source_outside_soloing_not_evaluable_count": _int(
                bridge["followup_repair_sweep_source_outside_soloing_not_evaluable_count"]
            ),
            "followup_repair_sweep_repaired_outside_soloing_not_evaluable_count": _int(
                bridge["followup_repair_sweep_repaired_outside_soloing_not_evaluable_count"]
            ),
            "repair_sweep_source_outside_soloing_not_evaluable_count": _int(
                bridge["repair_sweep_source_outside_soloing_not_evaluable_count"]
            ),
            "repair_sweep_repaired_outside_soloing_not_evaluable_count": _int(
                bridge["repair_sweep_repaired_outside_soloing_not_evaluable_count"]
            ),
            **{key: bridge[key] for key in BRIDGE_REQUIRED_SOURCE_CONTEXT_KEYS},
            "not_evaluable_after_count": _int(bridge["not_evaluable_after_count"]),
            "human_audio_preference_claimed": False,
            "audio_rendered_quality_claimed": False,
            "midi_to_solo_musical_quality_claimed": False,
            "model_checkpoint_generation_quality_claimed": False,
            "model_direct_generation_quality_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
            "production_ready_claimed": False,
        },
        "decision": {
            "current_boundary": BOUNDARY,
            "next_boundary": next_boundary,
            "selected_target": selected_target,
            "auto_progress_allowed": True,
            "critical_user_input_required": False,
            "reason": "pitch-role objective evidence selected next repair target without quality claim",
        },
        "not_proven": [
            "human_audio_preference",
            "midi_to_solo_musical_quality",
            "audio_rendered_quality",
            "broad_trained_model_quality",
            "brad_style_adaptation",
        ],
        "next_recommended_issue": (
            "Stage B MIDI-to-solo songlike melody contour phrase/rhythm chord-tone landing repair sweep source-context refresh"
            if next_boundary == NEXT_BOUNDARY
            else "Stage B MIDI-to-solo songlike melody contour phrase/rhythm outside-soloing pitch-role repair sweep source-context refresh"
        ),
    }


def validate_objective_decision_report(
    report: dict[str, Any],
    *,
    expected_boundary: str | None,
    expected_next_boundary: str | None,
    expected_target: str | None,
    require_objective_decision: bool,
    require_no_quality_claim: bool,
) -> dict[str, Any]:
    boundary = str(report.get("boundary") or "")
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    selected = _dict(report.get("selected_next_target"))
    if expected_boundary and boundary != expected_boundary:
        raise StageBMidiToSoloPitchRoleObjectiveDecisionError(
            f"expected boundary {expected_boundary}, got {boundary}"
        )
    if expected_next_boundary and str(decision.get("next_boundary") or "") != expected_next_boundary:
        raise StageBMidiToSoloPitchRoleObjectiveDecisionError(
            "unexpected next boundary"
        )
    if expected_target and str(selected.get("selected_target") or "") != expected_target:
        raise StageBMidiToSoloPitchRoleObjectiveDecisionError(
            "unexpected selected target"
        )
    if require_objective_decision and not bool(
        readiness.get("pitch_role_objective_decision_completed", False)
    ):
        raise StageBMidiToSoloPitchRoleObjectiveDecisionError(
            "objective decision completion required"
        )
    if _int(readiness.get("candidate_count")) <= 0:
        raise StageBMidiToSoloPitchRoleObjectiveDecisionError(
            "candidate count required"
        )
    if _int(readiness.get("not_evaluable_after_count")) != 0:
        raise StageBMidiToSoloPitchRoleObjectiveDecisionError(
            "not-evaluable labels must remain cleared"
        )
    if bool(decision.get("critical_user_input_required", True)):
        raise StageBMidiToSoloPitchRoleObjectiveDecisionError(
            "critical user input should not be required"
        )
    if require_no_quality_claim:
        _require_no_quality_claim(readiness, label="objective decision readiness")
    for key in BRIDGE_REQUIRED_SOURCE_CONTEXT_KEYS:
        if key not in readiness:
            raise StageBMidiToSoloPitchRoleObjectiveDecisionError(
                f"objective source-context field required: {key}"
            )
    return {
        "boundary": boundary,
        "source_boundary": str(report.get("source_boundary") or ""),
        "next_boundary": str(decision.get("next_boundary") or ""),
        "selected_target": str(decision.get("selected_target") or ""),
        "pitch_role_objective_decision_completed": bool(
            readiness.get("pitch_role_objective_decision_completed", False)
        ),
        "candidate_count": _int(readiness.get("candidate_count")),
        "primary_risk_label": str(readiness.get("primary_risk_label") or ""),
        "weak_chord_tone_landing_risk_count": _int(
            readiness.get("weak_chord_tone_landing_risk_count")
        ),
        "outside_soloing_pitch_role_risk_count": _int(
            readiness.get("outside_soloing_pitch_role_risk_count")
        ),
        "not_evaluable_after_count": _int(readiness.get("not_evaluable_after_count")),
        "followup_objective_source_outside_soloing_not_evaluable_count": _int(
            readiness.get("followup_objective_source_outside_soloing_not_evaluable_count")
        ),
        "followup_objective_repaired_outside_soloing_not_evaluable_count": _int(
            readiness.get("followup_objective_repaired_outside_soloing_not_evaluable_count")
        ),
        "followup_repair_sweep_source_outside_soloing_not_evaluable_count": _int(
            readiness.get("followup_repair_sweep_source_outside_soloing_not_evaluable_count")
        ),
        "followup_repair_sweep_repaired_outside_soloing_not_evaluable_count": _int(
            readiness.get("followup_repair_sweep_repaired_outside_soloing_not_evaluable_count")
        ),
        "repair_sweep_source_outside_soloing_not_evaluable_count": _int(
            readiness.get("repair_sweep_source_outside_soloing_not_evaluable_count")
        ),
        "repair_sweep_repaired_outside_soloing_not_evaluable_count": _int(
            readiness.get("repair_sweep_repaired_outside_soloing_not_evaluable_count")
        ),
        **{key: readiness.get(key) for key in BRIDGE_REQUIRED_SOURCE_CONTEXT_KEYS},
        "human_audio_preference_claimed": bool(
            readiness.get("human_audio_preference_claimed", True)
        ),
        "midi_to_solo_musical_quality_claimed": bool(
            readiness.get("midi_to_solo_musical_quality_claimed", True)
        ),
        "critical_user_input_required": bool(
            decision.get("critical_user_input_required", True)
        ),
        "next_recommended_issue": str(report.get("next_recommended_issue") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    summary = report["objective_summary"]
    selected = report["selected_next_target"]
    readiness = report["readiness"]
    decision = report["decision"]
    lines = [
        "# Stage B MIDI-to-Solo Pitch-Role Objective Decision",
        "",
        "## Summary",
        "",
        f"- boundary: `{report['boundary']}`",
        f"- source boundary: `{report['source_boundary']}`",
        f"- next boundary: `{decision['next_boundary']}`",
        f"- selected target: `{decision['selected_target']}`",
        f"- primary risk label: `{selected['primary_risk_label']}`",
        f"- candidate count: `{summary['candidate_count']}`",
        f"- not evaluable count: `{summary['not_evaluable_before_count']} -> {summary['not_evaluable_after_count']}`",
        f"- follow-up objective source/repaired outside-soloing not evaluable count: `{summary['followup_objective_source_outside_soloing_not_evaluable_count']}/{summary['followup_objective_repaired_outside_soloing_not_evaluable_count']}`",
        f"- follow-up repair sweep source/repaired outside-soloing not evaluable count: `{summary['followup_repair_sweep_source_outside_soloing_not_evaluable_count']}/{summary['followup_repair_sweep_repaired_outside_soloing_not_evaluable_count']}`",
        f"- bridge repair sweep source/repaired outside-soloing not evaluable count: `{summary['repair_sweep_source_outside_soloing_not_evaluable_count']}/{summary['repair_sweep_repaired_outside_soloing_not_evaluable_count']}`",
        f"- follow-up objective source outside-soloing source pitch-role risk: `{summary['followup_objective_source_outside_soloing_source_pitch_role_risk_count_before']} -> {summary['followup_objective_source_outside_soloing_source_pitch_role_risk_count_after']}`",
        f"- follow-up objective source outside-soloing source pitch-role risk delta: `{summary['followup_objective_source_outside_soloing_source_pitch_role_risk_delta']}`",
        f"- follow-up objective source outside-soloing source context preserved: `{_bool_token(summary['followup_objective_source_outside_soloing_source_context_preserved'])}`",
        f"- follow-up objective source outside-soloing source targeted: `{_bool_token(summary['followup_objective_source_outside_soloing_source_targeted'])}`",
        f"- follow-up objective source outside-soloing source residual risk preserved: `{_bool_token(summary['followup_objective_source_outside_soloing_source_residual_risk_preserved'])}`",
        f"- follow-up objective source outside-soloing current repair pitch-role risk after/delta: `{summary['followup_objective_source_outside_soloing_current_pitch_role_risk_count_after']} / {summary['followup_objective_source_outside_soloing_current_pitch_role_risk_delta']}`",
        f"- follow-up repair sweep source outside-soloing source pitch-role risk: `{summary['followup_repair_sweep_source_outside_soloing_source_pitch_role_risk_count_before']} -> {summary['followup_repair_sweep_source_outside_soloing_source_pitch_role_risk_count_after']}`",
        f"- follow-up repair sweep source outside-soloing source pitch-role risk delta: `{summary['followup_repair_sweep_source_outside_soloing_source_pitch_role_risk_delta']}`",
        f"- follow-up repair sweep source outside-soloing source context preserved: `{_bool_token(summary['followup_repair_sweep_source_outside_soloing_source_context_preserved'])}`",
        f"- follow-up repair sweep source outside-soloing source targeted: `{_bool_token(summary['followup_repair_sweep_source_outside_soloing_source_targeted'])}`",
        f"- follow-up repair sweep source outside-soloing source residual risk preserved: `{_bool_token(summary['followup_repair_sweep_source_outside_soloing_source_residual_risk_preserved'])}`",
        f"- follow-up repair sweep source outside-soloing current repair pitch-role risk after/delta: `{summary['followup_repair_sweep_source_outside_soloing_current_pitch_role_risk_count_after']} / {summary['followup_repair_sweep_source_outside_soloing_current_pitch_role_risk_delta']}`",
        f"- bridge repair sweep source outside-soloing source pitch-role risk: `{summary['repair_sweep_source_outside_soloing_source_pitch_role_risk_count_before']} -> {summary['repair_sweep_source_outside_soloing_source_pitch_role_risk_count_after']}`",
        f"- bridge repair sweep source outside-soloing source pitch-role risk delta: `{summary['repair_sweep_source_outside_soloing_source_pitch_role_risk_delta']}`",
        f"- bridge repair sweep source outside-soloing source context preserved: `{_bool_token(summary['repair_sweep_source_outside_soloing_source_context_preserved'])}`",
        f"- bridge repair sweep source outside-soloing source targeted: `{_bool_token(summary['repair_sweep_source_outside_soloing_source_targeted'])}`",
        f"- bridge repair sweep source outside-soloing source residual risk preserved: `{_bool_token(summary['repair_sweep_source_outside_soloing_source_residual_risk_preserved'])}`",
        f"- bridge repair sweep source outside-soloing current repair pitch-role risk after/delta: `{summary['repair_sweep_source_outside_soloing_current_pitch_role_risk_count_after']} / {summary['repair_sweep_source_outside_soloing_current_pitch_role_risk_delta']}`",
        f"- weak chord-tone landing risk count: `{summary['weak_chord_tone_landing_risk_count']}`",
        f"- outside-soloing pitch-role risk count: `{summary['outside_soloing_pitch_role_risk_count']}`",
        f"- min chord-tone ratio: `{summary['min_chord_tone_ratio']:.3f}`",
        f"- max outside ratio: `{summary['max_outside_ratio']:.3f}`",
        f"- max non-chord run: `{summary['max_non_chord_tone_run']}`",
        f"- human/audio preference claimed: `{_bool_token(readiness['human_audio_preference_claimed'])}`",
        f"- MIDI-to-solo musical quality claimed: `{_bool_token(readiness['midi_to_solo_musical_quality_claimed'])}`",
        "",
        "## Decision",
        "",
        f"- reason: `{selected['reason']}`",
        f"- auto progress allowed: `{_bool_token(decision['auto_progress_allowed'])}`",
        f"- critical user input required: `{_bool_token(decision['critical_user_input_required'])}`",
        f"- next recommended issue: `{report['next_recommended_issue']}`",
        "",
        "## Claim Boundary",
        "",
    ]
    for item in report["not_proven"]:
        lines.append(f"- `{item}`")
    lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Decide next target from chord-context pitch-role bridge evidence"
    )
    parser.add_argument("--bridge_report", type=str, required=True)
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_context_pitch_role_objective_decision",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--issue_number", type=int, default=1042)
    parser.add_argument("--expected_boundary", type=str, default="")
    parser.add_argument("--expected_next_boundary", type=str, default="")
    parser.add_argument("--expected_target", type=str, default="")
    parser.add_argument("--require_objective_decision", action="store_true")
    parser.add_argument("--require_no_quality_claim", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    report = build_objective_decision_report(
        bridge_report=read_json(Path(args.bridge_report)),
        output_dir=output_dir,
        issue_number=int(args.issue_number),
    )
    summary = validate_objective_decision_report(
        report,
        expected_boundary=str(args.expected_boundary or ""),
        expected_next_boundary=str(args.expected_next_boundary or ""),
        expected_target=str(args.expected_target or ""),
        require_objective_decision=bool(args.require_objective_decision),
        require_no_quality_claim=bool(args.require_no_quality_claim),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        output_dir
        / "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_context_pitch_role_objective_decision.json",
        report,
    )
    write_json(
        output_dir
        / "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_context_pitch_role_objective_decision_validation_summary.json",
        summary,
    )
    markdown = markdown_report(report)
    write_text(
        output_dir
        / "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_context_pitch_role_objective_decision.md",
        markdown,
    )
    if args.doc_path:
        write_text(Path(args.doc_path), markdown)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
