"""Decide next repair target after chord-role-balance repair from objective evidence."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from scripts.assess_stage_b_generic_base_readiness import write_json, write_text  # noqa: E402
from scripts.guard_music_transformer_solo_yield_chord_role_balance_listening_input import (  # noqa: E402
    SCHEMA_VERSION as INPUT_GUARD_SCHEMA_VERSION,
)
from scripts.run_music_transformer_solo_yield_chord_role_balance_repair_sweep import (  # noqa: E402
    SCHEMA_VERSION as REPAIR_SWEEP_SCHEMA_VERSION,
)


SCHEMA_VERSION = "music_transformer_solo_yield_chord_role_balance_objective_next_v1"
BOUNDARY = "music_transformer_solo_yield_chord_role_balance_repair_objective_only_next_decision"
NEXT_BOUNDARY_DENSITY_AFTERCARE = "music_transformer_solo_yield_density_aftercare_sweep"
NEXT_BOUNDARY_INTERVAL_CONTOUR = "music_transformer_solo_yield_interval_contour_aftercare_sweep"
NEXT_BOUNDARY_CHORD_ROLE_BALANCE = "music_transformer_solo_yield_chord_role_balance_repair_sweep"
NEXT_BOUNDARY_LISTENING_REVIEW = "music_transformer_solo_yield_chord_role_balance_repair_listening_review"
NEXT_BOUNDARY_LISTENING_REVIEW_FILL = (
    "music_transformer_solo_yield_chord_role_balance_repair_listening_review_fill"
)


class SoloYieldChordRoleBalanceObjectiveNextError(ValueError):
    pass


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
        raise SoloYieldChordRoleBalanceObjectiveNextError(f"json not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _require_no_quality_claim(readiness: dict[str, Any]) -> None:
    claimed = [
        key
        for key in (
            "audio_rendered_quality_claimed",
            "musical_quality_claimed",
            "artist_style_claimed",
            "production_ready_claimed",
        )
        if bool(readiness.get(key, False))
    ]
    if claimed:
        raise SoloYieldChordRoleBalanceObjectiveNextError(f"unexpected quality claim: {claimed}")


def validate_guard_report(guard_report: dict[str, Any]) -> dict[str, Any]:
    if str(guard_report.get("schema_version") or "") != INPUT_GUARD_SCHEMA_VERSION:
        raise SoloYieldChordRoleBalanceObjectiveNextError("input guard schema required")
    readiness = _dict(guard_report.get("readiness"))
    _require_no_quality_claim(readiness)
    input_validation = _dict(guard_report.get("input_validation"))
    return {
        "validated_listening_input_present": bool(
            input_validation.get("validated_listening_input_present", False)
        ),
        "preference_fill_allowed": bool(readiness.get("preference_fill_allowed", False)),
        "objective_only_next_decision_required": bool(
            readiness.get("objective_only_next_decision_required", False)
        ),
        "pending_candidate_field_count": _int(
            input_validation.get("pending_candidate_field_count")
        ),
    }


def validate_repair_sweep(repair_sweep: dict[str, Any]) -> list[dict[str, Any]]:
    if str(repair_sweep.get("schema_version") or "") != REPAIR_SWEEP_SCHEMA_VERSION:
        raise SoloYieldChordRoleBalanceObjectiveNextError("repair sweep schema required")
    readiness = _dict(repair_sweep.get("readiness"))
    _require_no_quality_claim(readiness)
    aggregate = _dict(repair_sweep.get("aggregate"))
    if not bool(aggregate.get("target_supported", False)):
        raise SoloYieldChordRoleBalanceObjectiveNextError("completed chord role balance repair required")
    if _int(aggregate.get("low_chord_role_count_after")) != 0:
        raise SoloYieldChordRoleBalanceObjectiveNextError("low chord-role residual risk must be zero")
    if _int(aggregate.get("weak_direction_change_count_after")) != 0:
        raise SoloYieldChordRoleBalanceObjectiveNextError("weak direction residual risk must be zero")
    if _int(aggregate.get("final_landing_not_chord_tone_count_after")) != 0:
        raise SoloYieldChordRoleBalanceObjectiveNextError("final landing residual risk must be zero")
    rows = [_dict(row) for row in _list(repair_sweep.get("candidate_repairs"))]
    if not rows:
        raise SoloYieldChordRoleBalanceObjectiveNextError("repair rows required")
    return rows


def residual_labels(after_profile: dict[str, Any]) -> list[str]:
    labels: list[str] = []
    if not bool(after_profile.get("final_landing_is_chord_tone", False)):
        labels.append("final_landing_not_chord_tone")
    if _float(after_profile.get("midi_chord_tone_ratio")) < 0.50:
        labels.append("midi_low_chord_tone_ratio")
    if _float(after_profile.get("midi_max_gap_seconds")) >= 0.65:
        labels.append("dead_air_aftercare")
    if _float(after_profile.get("midi_direction_change_ratio")) < 0.50:
        labels.append("weak_direction_change")
    if _int(after_profile.get("midi_note_count")) < 30:
        labels.append("low_note_count_for_4bar")
    if _int(after_profile.get("midi_max_abs_interval")) >= 8:
        labels.append("wide_interval_review")
    return labels


def build_residual_rows(repair_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in repair_rows:
        after = _dict(row.get("after_profile"))
        labels = residual_labels(after)
        rows.append(
            {
                "review_index": _int(row.get("review_index")),
                "case_label": str(row.get("case_label") or ""),
                "repaired_midi_path": str(row.get("repaired_midi_path") or ""),
                "source_wav_path": str(row.get("source_wav_path") or ""),
                "after_profile": after,
                "residual_labels": labels,
            }
        )
    return rows


def aggregate_residuals(rows: list[dict[str, Any]]) -> dict[str, Any]:
    label_counts: Counter[str] = Counter()
    for row in rows:
        label_counts.update(str(label) for label in _list(row.get("residual_labels")))
    chord_tone = [_float(_dict(row.get("after_profile")).get("midi_chord_tone_ratio")) for row in rows]
    max_gap = [_float(_dict(row.get("after_profile")).get("midi_max_gap_seconds")) for row in rows]
    direction = [
        _float(_dict(row.get("after_profile")).get("midi_direction_change_ratio")) for row in rows
    ]
    note_counts = [_int(_dict(row.get("after_profile")).get("midi_note_count")) for row in rows]
    intervals = [_int(_dict(row.get("after_profile")).get("midi_max_abs_interval")) for row in rows]
    return {
        "candidate_count": len(rows),
        "residual_label_counts": dict(sorted(label_counts.items())),
        "final_landing_not_chord_tone_count": label_counts.get("final_landing_not_chord_tone", 0),
        "midi_low_chord_tone_ratio_count": label_counts.get("midi_low_chord_tone_ratio", 0),
        "dead_air_aftercare_count": label_counts.get("dead_air_aftercare", 0),
        "weak_direction_change_count": label_counts.get("weak_direction_change", 0),
        "low_note_count_for_4bar_count": label_counts.get("low_note_count_for_4bar", 0),
        "wide_interval_review_count": label_counts.get("wide_interval_review", 0),
        "midi_chord_tone_ratio_min": min(chord_tone, default=0.0),
        "midi_chord_tone_ratio_max": max(chord_tone, default=0.0),
        "midi_chord_tone_ratio_avg": float(mean(chord_tone)) if chord_tone else 0.0,
        "midi_max_gap_seconds_max": max(max_gap, default=0.0),
        "midi_direction_change_ratio_min": min(direction, default=0.0),
        "midi_direction_change_ratio_avg": float(mean(direction)) if direction else 0.0,
        "midi_note_count_min": min(note_counts, default=0),
        "midi_note_count_max": max(note_counts, default=0),
        "midi_max_abs_interval_max": max(intervals, default=0),
    }


def select_next_target(aggregate: dict[str, Any]) -> tuple[str, str, str]:
    low_chord = _int(aggregate.get("midi_low_chord_tone_ratio_count"))
    low_note = _int(aggregate.get("low_note_count_for_4bar_count"))
    wide_interval = _int(aggregate.get("wide_interval_review_count"))
    dead_air = _int(aggregate.get("dead_air_aftercare_count"))
    if low_chord > 0:
        return (
            "chord_role_balance_repair",
            NEXT_BOUNDARY_CHORD_ROLE_BALANCE,
            "low MIDI chord-tone ratio remains after chord role balance repair",
        )
    if low_note > 0 and low_note >= max(wide_interval, dead_air):
        return (
            "density_aftercare",
            NEXT_BOUNDARY_DENSITY_AFTERCARE,
            "low note-count candidates are the largest remaining objective residual",
        )
    if wide_interval > 0:
        return (
            "interval_contour_aftercare",
            NEXT_BOUNDARY_INTERVAL_CONTOUR,
            "wide interval review count remains after chord role balance repair",
        )
    return (
        "listening_review_required",
        NEXT_BOUNDARY_LISTENING_REVIEW,
        "objective residual risk below repair threshold; listening review required",
    )


def build_decision(
    repair_sweep: dict[str, Any],
    guard_report: dict[str, Any],
    *,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    guard_validation = validate_guard_report(guard_report)
    repair_rows = validate_repair_sweep(repair_sweep)
    residual_rows = build_residual_rows(repair_rows)
    aggregate = aggregate_residuals(residual_rows)
    if guard_validation["preference_fill_allowed"]:
        selected_target = "listening_review_fill"
        next_boundary = NEXT_BOUNDARY_LISTENING_REVIEW_FILL
        reason = "validated listening input present"
    else:
        selected_target, next_boundary, reason = select_next_target(aggregate)
    report = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "boundary": BOUNDARY,
        "source_repair_sweep": {
            "schema_version": repair_sweep.get("schema_version"),
            "output_dir": repair_sweep.get("output_dir"),
            "candidate_count": _int(_dict(repair_sweep.get("aggregate")).get("candidate_count")),
            "low_chord_role_count_after": _int(
                _dict(repair_sweep.get("aggregate")).get("low_chord_role_count_after")
            ),
            "weak_direction_change_count_after": _int(
                _dict(repair_sweep.get("aggregate")).get("weak_direction_change_count_after")
            ),
            "final_landing_not_chord_tone_count_after": _int(
                _dict(repair_sweep.get("aggregate")).get("final_landing_not_chord_tone_count_after")
            ),
        },
        "source_guard": {
            "schema_version": guard_report.get("schema_version"),
            "output_dir": guard_report.get("output_dir"),
            "validated_listening_input_present": guard_validation[
                "validated_listening_input_present"
            ],
            "preference_fill_allowed": guard_validation["preference_fill_allowed"],
            "pending_candidate_field_count": guard_validation["pending_candidate_field_count"],
        },
        "candidate_residuals": residual_rows,
        "aggregate": aggregate,
        "readiness": {
            "objective_only_next_decision_completed": True,
            "validated_listening_input_present": guard_validation[
                "validated_listening_input_present"
            ],
            "preference_fill_allowed": guard_validation["preference_fill_allowed"],
            "audio_rendered_quality_claimed": False,
            "musical_quality_claimed": False,
            "artist_style_claimed": False,
            "production_ready_claimed": False,
        },
        "decision": {
            "current_boundary": BOUNDARY,
            "selected_next_target": selected_target,
            "next_boundary": next_boundary,
            "critical_user_input_required": False,
            "reason": reason,
        },
        "not_proven": [
            "human_audio_preference",
            "stable_jazz_solo_quality",
            "artist_level_long_solo_generation",
            "production_ready_improviser",
        ],
    }
    write_json(output_dir / "chord_role_balance_objective_next_decision.json", report)
    write_json(
        output_dir / "chord_role_balance_objective_next_decision_summary.json",
        validate_report(report, require_no_quality_claim=True),
    )
    write_text(output_dir / "chord_role_balance_objective_next_decision.md", markdown_report(report))
    return report


def validate_report(report: dict[str, Any], *, require_no_quality_claim: bool) -> dict[str, Any]:
    if str(report.get("schema_version") or "") != SCHEMA_VERSION:
        raise SoloYieldChordRoleBalanceObjectiveNextError("schema version mismatch")
    readiness = _dict(report.get("readiness"))
    if require_no_quality_claim:
        _require_no_quality_claim(readiness)
    aggregate = _dict(report.get("aggregate"))
    decision = _dict(report.get("decision"))
    return {
        "schema_version": str(report.get("schema_version") or ""),
        "boundary": str(report.get("boundary") or ""),
        "candidate_count": _int(aggregate.get("candidate_count")),
        "midi_low_chord_tone_ratio_count": _int(
            aggregate.get("midi_low_chord_tone_ratio_count")
        ),
        "low_note_count_for_4bar_count": _int(
            aggregate.get("low_note_count_for_4bar_count")
        ),
        "wide_interval_review_count": _int(aggregate.get("wide_interval_review_count")),
        "dead_air_aftercare_count": _int(aggregate.get("dead_air_aftercare_count")),
        "weak_direction_change_count": _int(aggregate.get("weak_direction_change_count")),
        "final_landing_not_chord_tone_count": _int(
            aggregate.get("final_landing_not_chord_tone_count")
        ),
        "validated_listening_input_present": bool(
            readiness.get("validated_listening_input_present", True)
        ),
        "preference_fill_allowed": bool(readiness.get("preference_fill_allowed", True)),
        "selected_next_target": str(decision.get("selected_next_target") or ""),
        "next_boundary": str(decision.get("next_boundary") or ""),
        "musical_quality_claimed": bool(readiness.get("musical_quality_claimed", True)),
    }


def markdown_report(report: dict[str, Any]) -> str:
    aggregate = report["aggregate"]
    readiness = report["readiness"]
    decision = report["decision"]
    lines = [
        "# Music Transformer Solo Yield Chord Role Balance Repair Objective-Only Next Decision",
        "",
        "## Summary",
        "",
        f"- boundary: `{report['boundary']}`",
        f"- candidate count: `{aggregate['candidate_count']}`",
        f"- MIDI low chord-tone ratio: `{aggregate['midi_low_chord_tone_ratio_count']}`",
        f"- low note count for 4bar: `{aggregate['low_note_count_for_4bar_count']}`",
        f"- wide interval review: `{aggregate['wide_interval_review_count']}`",
        f"- dead-air aftercare: `{aggregate['dead_air_aftercare_count']}`",
        f"- weak direction-change: `{aggregate['weak_direction_change_count']}`",
        f"- final landing not chord-tone: `{aggregate['final_landing_not_chord_tone_count']}`",
        f"- direction-change min/avg: `{float(aggregate['midi_direction_change_ratio_min']):.4f}` / `{float(aggregate['midi_direction_change_ratio_avg']):.4f}`",
        f"- chord-tone ratio min/avg/max: `{float(aggregate['midi_chord_tone_ratio_min']):.4f}` / `{float(aggregate['midi_chord_tone_ratio_avg']):.4f}` / `{float(aggregate['midi_chord_tone_ratio_max']):.4f}`",
        f"- note count min/max: `{aggregate['midi_note_count_min']}` / `{aggregate['midi_note_count_max']}`",
        f"- max interval: `{aggregate['midi_max_abs_interval_max']}`",
        f"- validated listening input present: `{_bool_token(readiness['validated_listening_input_present'])}`",
        f"- preference fill allowed: `{_bool_token(readiness['preference_fill_allowed'])}`",
        f"- selected next target: `{decision['selected_next_target']}`",
        f"- next boundary: `{decision['next_boundary']}`",
        f"- musical quality claimed: `{_bool_token(readiness['musical_quality_claimed'])}`",
        "",
        "## Residual Label Counts",
        "",
    ]
    for label, count in sorted(_dict(aggregate.get("residual_label_counts")).items()):
        lines.append(f"- `{label}`: `{count}`")
    lines.extend(["", "## Candidates", ""])
    for row in report.get("candidate_residuals", []):
        profile = row["after_profile"]
        labels = ", ".join(f"`{label}`" for label in row["residual_labels"]) or "`none`"
        lines.extend(
            [
                f"- candidate `{row['review_index']}` / `{row['case_label']}`",
                f"  - labels: {labels}",
                f"  - chord-tone ratio: `{float(profile['midi_chord_tone_ratio']):.4f}`",
                f"  - direction-change ratio: `{float(profile['midi_direction_change_ratio']):.4f}`",
                f"  - max gap seconds: `{float(profile['midi_max_gap_seconds']):.4f}`",
                f"  - note count: `{profile['midi_note_count']}`",
                f"  - max interval: `{profile['midi_max_abs_interval']}`",
            ]
        )
    lines.extend(["", "## Not Proven", ""])
    for item in report.get("not_proven", []):
        lines.append(f"- `{item}`")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Decide objective next target after chord role balance repair")
    parser.add_argument(
        "--repair_sweep_report",
        type=str,
        default=(
            "outputs/music_transformer_finetune_mvp/solo_yield_chord_role_balance_repair/"
            "issue_1284_chord_role_balance_repair_sweep/chord_role_balance_repair_sweep.json"
        ),
    )
    parser.add_argument(
        "--guard_report",
        type=str,
        default=(
            "outputs/music_transformer_finetune_mvp/"
            "solo_yield_chord_role_balance_repair_listening_input_guard/"
            "issue_1290_chord_role_balance_input_guard/"
            "chord_role_balance_repair_listening_input_guard.json"
        ),
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/music_transformer_finetune_mvp/solo_yield_chord_role_balance_repair_objective_next",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--require_no_quality_claim", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    report = build_decision(
        read_json(Path(args.repair_sweep_report)),
        read_json(Path(args.guard_report)),
        output_dir=output_dir,
    )
    summary = validate_report(report, require_no_quality_claim=bool(args.require_no_quality_claim))
    if args.doc_path:
        write_text(Path(args.doc_path), markdown_report(report))
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
