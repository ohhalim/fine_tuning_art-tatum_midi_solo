"""Build the MIDI-to-solo quality rubric baseline for post-MVP iteration."""

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
from scripts.plan_stage_b_midi_to_solo_post_mvp_quality_iteration import (  # noqa: E402
    BOUNDARY as POST_MVP_PLAN_BOUNDARY,
    NEXT_BOUNDARY as POST_MVP_PLAN_NEXT_BOUNDARY,
    SELECTED_TARGET as POST_MVP_SELECTED_TARGET,
    StageBMidiToSoloPostMvpQualityIterationPlanError,
    validate_post_mvp_quality_iteration_plan_report,
)


class StageBMidiToSoloQualityRubricBaselineError(ValueError):
    pass


BOUNDARY = "stage_b_midi_to_solo_quality_rubric_baseline"
NEXT_BOUNDARY = "stage_b_midi_to_solo_candidate_failure_labeling"
SELECTED_TARGET = "candidate_failure_labeling"
SCHEMA_VERSION = "stage_b_midi_to_solo_quality_rubric_baseline_v1"

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


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


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
        raise StageBMidiToSoloQualityRubricBaselineError(
            f"unexpected quality claim in {label}: {claimed}"
        )


def validate_post_mvp_quality_iteration_plan_source(report: dict[str, Any]) -> dict[str, Any]:
    if str(report.get("boundary") or "") != POST_MVP_PLAN_BOUNDARY:
        raise StageBMidiToSoloQualityRubricBaselineError(
            "post-MVP quality iteration plan boundary required"
        )
    try:
        summary = validate_post_mvp_quality_iteration_plan_report(
            report,
            expected_boundary=POST_MVP_PLAN_BOUNDARY,
            expected_next_boundary=POST_MVP_PLAN_NEXT_BOUNDARY,
            expected_target=POST_MVP_SELECTED_TARGET,
            require_quality_rubric=True,
            require_no_quality_claim=True,
        )
    except StageBMidiToSoloPostMvpQualityIterationPlanError as exc:
        raise StageBMidiToSoloQualityRubricBaselineError(str(exc)) from exc

    readiness = _dict(report.get("readiness"))
    selected = _dict(report.get("selected_next_target"))
    _require_no_quality_claim(readiness, label="post-MVP readiness")
    if str(selected.get("selected_target") or "") != POST_MVP_SELECTED_TARGET:
        raise StageBMidiToSoloQualityRubricBaselineError("quality rubric target required")
    if not bool(summary.get("quality_rubric_required", False)):
        raise StageBMidiToSoloQualityRubricBaselineError("quality rubric requirement missing")
    if not bool(summary.get("candidate_failure_labeling_required", False)):
        raise StageBMidiToSoloQualityRubricBaselineError(
            "candidate failure labeling requirement missing"
        )
    if _int(summary.get("ordered_work_count")) < 4:
        raise StageBMidiToSoloQualityRubricBaselineError("ordered work count below 4")
    if _int(summary.get("quality_failure_taxonomy_seed_count")) < 7:
        raise StageBMidiToSoloQualityRubricBaselineError("taxonomy seed count below 7")
    return summary


def build_rubric_items() -> list[dict[str, Any]]:
    return [
        {
            "id": "sparse_or_empty_output",
            "metric_keys": ["note_count", "active_bar_count", "note_count_per_bar"],
            "failure_rule": "note_count < 12 or active_bar_count < 2",
            "threshold": {"min_note_count": 12, "min_active_bar_count": 2},
            "source_reason": "one-note/two-note or short fragment output cannot be treated as solo-line evidence",
        },
        {
            "id": "dead_air_or_density_gap",
            "metric_keys": ["dead_air_ratio", "max_gap_beats", "empty_bar_count"],
            "failure_rule": "dead_air_ratio > 0.35 or empty_bar_count > 0",
            "threshold": {"max_dead_air_ratio": 0.35, "max_empty_bar_count": 0},
            "source_reason": "technical candidates can pass file generation while still leaving audible gaps",
        },
        {
            "id": "rhythmic_monotony",
            "metric_keys": [
                "duration_most_common_ratio",
                "ioi_most_common_ratio",
                "note_count_per_bar_most_common_ratio",
                "unique_onset_pattern_count",
            ],
            "failure_rule": "duration_most_common_ratio >= 0.40 or ioi_most_common_ratio >= 0.40 or note_count_per_bar_most_common_ratio >= 0.95",
            "threshold": {
                "max_duration_most_common_ratio": 0.40,
                "max_ioi_most_common_ratio": 0.40,
                "max_note_count_per_bar_most_common_ratio": 0.95,
            },
            "source_reason": "previous rejected candidates showed songlike repeated rhythm templates",
        },
        {
            "id": "songlike_melody_not_soloing",
            "metric_keys": [
                "four_notes_per_bar_template",
                "four_bar_rhythm_cycle_repeated",
                "shared_rhythm_signature_count",
                "small_interval_ratio_le4",
            ],
            "failure_rule": "four_notes_per_bar_template or four_bar_rhythm_cycle_repeated or shared_rhythm_signature_count >= 3",
            "threshold": {
                "max_shared_rhythm_signature_count": 2,
                "max_small_interval_ratio_le4": 0.55,
            },
            "source_reason": "user listening rejection identified melody-like output instead of soloing",
        },
        {
            "id": "outside_soloing_without_context",
            "metric_keys": [
                "non_chord_tone_ratio",
                "avoid_tone_landing_count",
                "outside_pitch_run_length",
                "chord_context_available",
            ],
            "failure_rule": "outside_pitch_run_length >= 4 or avoid_tone_landing_count > 0 when chord_context_available",
            "threshold": {"max_outside_pitch_run_length": 3, "max_avoid_tone_landing_count": 0},
            "source_reason": "outside notes require chord-context support before being kept",
        },
        {
            "id": "weak_chord_tone_landing",
            "metric_keys": [
                "cadence_landing_chord_tone",
                "strong_beat_chord_tone_ratio",
                "last_note_chord_role",
            ],
            "failure_rule": "cadence_landing_chord_tone == false or strong_beat_chord_tone_ratio < 0.40",
            "threshold": {"min_strong_beat_chord_tone_ratio": 0.40},
            "source_reason": "jazz-line evidence should expose at least minimal chord-role anchoring",
        },
        {
            "id": "phrase_shape_missing_tension_release",
            "metric_keys": [
                "contour_turn_count",
                "large_interval_ratio_gte12",
                "phrase_peak_position_ratio",
                "cadence_resolution_present",
            ],
            "failure_rule": "contour_turn_count == 0 or cadence_resolution_present == false",
            "threshold": {
                "min_contour_turn_count": 1,
                "max_large_interval_ratio_gte12": 0.20,
                "phrase_peak_position_ratio_range": [0.25, 0.85],
            },
            "source_reason": "objective-safe output can still lack a phrase arc",
        },
        {
            "id": "technical_gate_regression",
            "metric_keys": ["overlap_count", "max_simultaneous_notes", "grammar_valid", "strict_valid"],
            "failure_rule": "grammar_valid == false or strict_valid == false or max_simultaneous_notes > 1",
            "threshold": {"max_simultaneous_notes": 1},
            "source_reason": "quality iteration must not regress the completed technical MVP gates",
        },
    ]


def build_quality_rubric_baseline_report(
    *,
    post_mvp_quality_plan: dict[str, Any],
    output_dir: Path,
    issue_number: int,
) -> dict[str, Any]:
    source = validate_post_mvp_quality_iteration_plan_source(post_mvp_quality_plan)
    rubric_items = build_rubric_items()
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "issue_number": int(issue_number),
        "boundary": BOUNDARY,
        "source_boundary": POST_MVP_PLAN_BOUNDARY,
        "source_summary": source,
        "rubric_baseline": {
            "rubric_item_count": len(rubric_items),
            "required_metric_group_count": len(
                {metric for item in rubric_items for metric in _list(item.get("metric_keys"))}
            ),
            "candidate_failure_labeling_ready": True,
            "human_audio_preference_claimed": False,
            "midi_to_solo_musical_quality_claimed": False,
        },
        "rubric_items": rubric_items,
        "selected_next_target": {
            "selected_target": SELECTED_TARGET,
            "selected_next_boundary": NEXT_BOUNDARY,
            "reason": "rubric baseline is ready; current candidates can be labeled before repair sweep",
        },
        "readiness": {
            "boundary": BOUNDARY,
            "quality_rubric_baseline_completed": True,
            "candidate_failure_labeling_ready": True,
            "rubric_item_count": len(rubric_items),
            "selected_target": SELECTED_TARGET,
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
            "reason": "candidate failure labeling can run from MIDI evidence and rubric thresholds",
        },
        "not_proven": [
            "candidate_failure_labels_completed",
            "targeted_quality_repair_completed",
            "human_audio_preference",
            "midi_to_solo_musical_quality",
            "broad_trained_model_quality",
        ],
        "next_recommended_issue": "Stage B MIDI-to-solo candidate failure labeling",
    }


def validate_quality_rubric_baseline_report(
    report: dict[str, Any],
    *,
    expected_boundary: str | None,
    expected_next_boundary: str | None,
    expected_target: str | None,
    min_rubric_item_count: int,
    require_candidate_labeling_ready: bool,
    require_no_quality_claim: bool,
) -> dict[str, Any]:
    boundary = str(report.get("boundary") or "")
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    selected = _dict(report.get("selected_next_target"))
    baseline = _dict(report.get("rubric_baseline"))
    rubric_items = _list(report.get("rubric_items"))
    if expected_boundary and boundary != expected_boundary:
        raise StageBMidiToSoloQualityRubricBaselineError(
            f"expected boundary {expected_boundary}, got {boundary}"
        )
    if expected_next_boundary and str(decision.get("next_boundary") or "") != expected_next_boundary:
        raise StageBMidiToSoloQualityRubricBaselineError("unexpected next boundary")
    if expected_target and str(selected.get("selected_target") or "") != expected_target:
        raise StageBMidiToSoloQualityRubricBaselineError("unexpected selected target")
    if not bool(readiness.get("quality_rubric_baseline_completed", False)):
        raise StageBMidiToSoloQualityRubricBaselineError("rubric baseline completion required")
    if len(rubric_items) < int(min_rubric_item_count):
        raise StageBMidiToSoloQualityRubricBaselineError("rubric item count below minimum")
    if _int(readiness.get("rubric_item_count")) != len(rubric_items):
        raise StageBMidiToSoloQualityRubricBaselineError("rubric item count mismatch")
    rubric_ids = [str(item.get("id") or "") for item in rubric_items if isinstance(item, dict)]
    required_ids = {
        "sparse_or_empty_output",
        "dead_air_or_density_gap",
        "rhythmic_monotony",
        "songlike_melody_not_soloing",
        "outside_soloing_without_context",
        "weak_chord_tone_landing",
        "phrase_shape_missing_tension_release",
        "technical_gate_regression",
    }
    missing_ids = sorted(required_ids - set(rubric_ids))
    if missing_ids:
        raise StageBMidiToSoloQualityRubricBaselineError(f"rubric ids missing: {missing_ids}")
    if require_candidate_labeling_ready and not bool(
        baseline.get("candidate_failure_labeling_ready", False)
    ):
        raise StageBMidiToSoloQualityRubricBaselineError("candidate failure labeling readiness required")
    if bool(decision.get("critical_user_input_required", True)):
        raise StageBMidiToSoloQualityRubricBaselineError("critical user input should not be required")
    if require_no_quality_claim:
        _require_no_quality_claim(readiness, label="rubric readiness")
        _require_no_quality_claim(baseline, label="rubric baseline")
    return {
        "boundary": boundary,
        "next_boundary": str(decision.get("next_boundary") or ""),
        "selected_target": str(selected.get("selected_target") or ""),
        "quality_rubric_baseline_completed": bool(
            readiness.get("quality_rubric_baseline_completed", False)
        ),
        "candidate_failure_labeling_ready": bool(
            baseline.get("candidate_failure_labeling_ready", False)
        ),
        "rubric_item_count": len(rubric_items),
        "required_metric_group_count": _int(baseline.get("required_metric_group_count")),
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
    baseline = report["rubric_baseline"]
    selected = report["selected_next_target"]
    readiness = report["readiness"]
    lines = [
        "# Stage B MIDI-to-Solo Quality Rubric Baseline",
        "",
        "## Summary",
        "",
        f"- boundary: `{report['boundary']}`",
        f"- source boundary: `{report['source_boundary']}`",
        f"- next boundary: `{selected['selected_next_boundary']}`",
        f"- selected target: `{selected['selected_target']}`",
        f"- rubric item count: `{baseline['rubric_item_count']}`",
        f"- candidate failure labeling ready: `{_bool_token(baseline['candidate_failure_labeling_ready'])}`",
        "",
        "## Rubric Items",
        "",
    ]
    for item in report["rubric_items"]:
        lines.append(f"- `{item['id']}`: {item['failure_rule']}")
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
    parser = argparse.ArgumentParser(description="Build MIDI-to-solo quality rubric baseline")
    parser.add_argument("--post_mvp_quality_plan", type=str, required=True)
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/stage_b_midi_to_solo_quality_rubric_baseline",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--issue_number", type=int, default=746)
    parser.add_argument("--expected_boundary", type=str, default="")
    parser.add_argument("--expected_next_boundary", type=str, default="")
    parser.add_argument("--expected_target", type=str, default="")
    parser.add_argument("--min_rubric_item_count", type=int, default=8)
    parser.add_argument("--require_candidate_labeling_ready", action="store_true")
    parser.add_argument("--require_no_quality_claim", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    report = build_quality_rubric_baseline_report(
        post_mvp_quality_plan=read_json(Path(args.post_mvp_quality_plan)),
        output_dir=output_dir,
        issue_number=int(args.issue_number),
    )
    summary = validate_quality_rubric_baseline_report(
        report,
        expected_boundary=str(args.expected_boundary or ""),
        expected_next_boundary=str(args.expected_next_boundary or ""),
        expected_target=str(args.expected_target or ""),
        min_rubric_item_count=int(args.min_rubric_item_count),
        require_candidate_labeling_ready=bool(args.require_candidate_labeling_ready),
        require_no_quality_claim=bool(args.require_no_quality_claim),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "stage_b_midi_to_solo_quality_rubric_baseline.json", report)
    write_json(
        output_dir / "stage_b_midi_to_solo_quality_rubric_baseline_validation_summary.json",
        summary,
    )
    markdown = markdown_report(report)
    write_text(output_dir / "stage_b_midi_to_solo_quality_rubric_baseline.md", markdown)
    if args.doc_path:
        write_text(Path(args.doc_path), markdown)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
