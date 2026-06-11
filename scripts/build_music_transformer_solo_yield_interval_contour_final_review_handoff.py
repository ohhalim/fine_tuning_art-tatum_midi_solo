"""Build final review handoff for interval-contour-aftercare solo candidates."""

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
from scripts.build_music_transformer_solo_yield_interval_contour_aftercare_listening_package import (  # noqa: E402
    INPUT_SCHEMA_VERSION,
    SCHEMA_VERSION as LISTENING_PACKAGE_SCHEMA_VERSION,
)
from scripts.decide_music_transformer_solo_yield_interval_contour_aftercare_objective_next import (  # noqa: E402
    SCHEMA_VERSION as OBJECTIVE_DECISION_SCHEMA_VERSION,
)
from scripts.render_music_transformer_solo_yield_interval_contour_aftercare_audio import (  # noqa: E402
    SCHEMA_VERSION as AUDIO_PACKAGE_SCHEMA_VERSION,
)


SCHEMA_VERSION = "music_transformer_solo_yield_interval_contour_final_review_handoff_v1"
BOUNDARY = "music_transformer_solo_yield_interval_contour_final_review_handoff"
NEXT_BOUNDARY = "music_transformer_solo_yield_interval_contour_aftercare_listening_review"


class SoloYieldIntervalContourFinalReviewHandoffError(ValueError):
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
        raise SoloYieldIntervalContourFinalReviewHandoffError(f"json not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _require_no_quality_claim(report: dict[str, Any], *, label: str) -> None:
    readiness = _dict(report.get("readiness"))
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
        raise SoloYieldIntervalContourFinalReviewHandoffError(
            f"unexpected quality claim in {label}: {claimed}"
        )


def _validate_listening_package(report: dict[str, Any]) -> list[dict[str, Any]]:
    if str(report.get("schema_version") or "") != LISTENING_PACKAGE_SCHEMA_VERSION:
        raise SoloYieldIntervalContourFinalReviewHandoffError("listening package schema required")
    _require_no_quality_claim(report, label="listening_package")
    readiness = _dict(report.get("readiness"))
    if not bool(readiness.get("listening_package_ready", False)):
        raise SoloYieldIntervalContourFinalReviewHandoffError("listening package ready flag required")
    candidates = [_dict(row) for row in _list(report.get("candidates"))]
    if not candidates:
        raise SoloYieldIntervalContourFinalReviewHandoffError("listening candidates required")
    if _int(readiness.get("candidate_midi_files_copied")) != len(candidates):
        raise SoloYieldIntervalContourFinalReviewHandoffError("candidate MIDI count mismatch")
    if _int(readiness.get("candidate_wav_files_copied")) != len(candidates):
        raise SoloYieldIntervalContourFinalReviewHandoffError("candidate WAV count mismatch")
    template = _dict(report.get("review_input_template"))
    if str(template.get("schema_version") or "") != INPUT_SCHEMA_VERSION:
        raise SoloYieldIntervalContourFinalReviewHandoffError("review input template schema required")
    if str(template.get("review_status") or "") != "pending":
        raise SoloYieldIntervalContourFinalReviewHandoffError("pending review input template required")
    return candidates


def _validate_audio_package(report: dict[str, Any], *, expected_count: int) -> list[dict[str, Any]]:
    if str(report.get("schema_version") or "") != AUDIO_PACKAGE_SCHEMA_VERSION:
        raise SoloYieldIntervalContourFinalReviewHandoffError("audio package schema required")
    _require_no_quality_claim(report, label="audio_package")
    aggregate = _dict(report.get("aggregate"))
    if not bool(aggregate.get("technical_wav_validation", False)):
        raise SoloYieldIntervalContourFinalReviewHandoffError("technical WAV validation required")
    if _int(aggregate.get("rendered_wav_count")) < int(expected_count):
        raise SoloYieldIntervalContourFinalReviewHandoffError("rendered WAV count below candidate count")
    return [_dict(row) for row in _list(report.get("rendered_audio_files"))]


def _validate_objective_decision(report: dict[str, Any], *, expected_count: int) -> list[dict[str, Any]]:
    if str(report.get("schema_version") or "") != OBJECTIVE_DECISION_SCHEMA_VERSION:
        raise SoloYieldIntervalContourFinalReviewHandoffError("objective decision schema required")
    _require_no_quality_claim(report, label="objective_decision")
    aggregate = _dict(report.get("aggregate"))
    if _int(aggregate.get("candidate_count")) != int(expected_count):
        raise SoloYieldIntervalContourFinalReviewHandoffError("objective candidate count mismatch")
    residual_keys = (
        "final_landing_not_chord_tone_count",
        "midi_low_chord_tone_ratio_count",
        "dead_air_aftercare_count",
        "weak_direction_change_count",
        "low_note_count_for_4bar_count",
        "wide_interval_review_count",
    )
    residual_total = sum(_int(aggregate.get(key)) for key in residual_keys)
    if residual_total != 0:
        raise SoloYieldIntervalContourFinalReviewHandoffError(
            f"objective residual risk must be zero: {residual_total}"
        )
    decision = _dict(report.get("decision"))
    if str(decision.get("selected_next_target") or "") != "listening_review_required":
        raise SoloYieldIntervalContourFinalReviewHandoffError("listening review decision required")
    return [_dict(row) for row in _list(report.get("candidate_residuals"))]


def _path_exists(path_value: Any) -> bool:
    path = Path(str(path_value or ""))
    return bool(str(path)) and path.exists()


def build_candidate_handoff(
    candidates: list[dict[str, Any]],
    audio_rows: list[dict[str, Any]],
    objective_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    audio_by_index = {_int(row.get("review_index")): row for row in audio_rows}
    objective_by_index = {_int(row.get("review_index")): row for row in objective_rows}
    handoff_rows: list[dict[str, Any]] = []
    for row in candidates:
        review_index = _int(row.get("review_index"))
        objective = objective_by_index.get(review_index)
        audio = audio_by_index.get(review_index)
        if objective is None:
            raise SoloYieldIntervalContourFinalReviewHandoffError(
                f"objective residual row missing: {review_index}"
            )
        if audio is None:
            raise SoloYieldIntervalContourFinalReviewHandoffError(
                f"audio row missing: {review_index}"
            )
        review_midi_path = str(row.get("review_midi_path") or "")
        review_wav_path = str(row.get("review_wav_path") or "")
        if not _path_exists(review_midi_path):
            raise SoloYieldIntervalContourFinalReviewHandoffError(
                f"review MIDI missing: {review_midi_path}"
            )
        if not _path_exists(review_wav_path):
            raise SoloYieldIntervalContourFinalReviewHandoffError(
                f"review WAV missing: {review_wav_path}"
            )
        profile = _dict(objective.get("after_profile"))
        residual_labels = [str(label) for label in _list(objective.get("residual_labels"))]
        if residual_labels:
            raise SoloYieldIntervalContourFinalReviewHandoffError(
                f"unexpected residual labels for candidate {review_index}: {residual_labels}"
            )
        wav_file = _dict(row.get("review_wav_file"))
        handoff_rows.append(
            {
                "review_index": review_index,
                "case_label": str(row.get("case_label") or ""),
                "review_midi_path": review_midi_path,
                "review_wav_path": review_wav_path,
                "review_midi_sha256": str(row.get("review_midi_sha256") or ""),
                "review_wav_sha256": str(wav_file.get("sha256") or ""),
                "duration_seconds": _float(wav_file.get("duration_seconds")),
                "sample_rate": _int(wav_file.get("sample_rate")),
                "objective_profile": {
                    "midi_note_count": _int(profile.get("midi_note_count")),
                    "midi_chord_tone_ratio": _float(profile.get("midi_chord_tone_ratio")),
                    "midi_max_gap_seconds": _float(profile.get("midi_max_gap_seconds")),
                    "midi_direction_change_ratio": _float(
                        profile.get("midi_direction_change_ratio")
                    ),
                    "midi_max_abs_interval": _int(profile.get("midi_max_abs_interval")),
                    "final_landing_chord": str(profile.get("final_landing_chord") or ""),
                    "final_landing_is_chord_tone": bool(
                        profile.get("final_landing_is_chord_tone", False)
                    ),
                },
                "residual_labels": residual_labels,
            }
        )
    return handoff_rows


def build_handoff_package(
    listening_package: dict[str, Any],
    audio_package: dict[str, Any],
    objective_decision: dict[str, Any],
    *,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = _validate_listening_package(listening_package)
    audio_rows = _validate_audio_package(audio_package, expected_count=len(candidates))
    objective_rows = _validate_objective_decision(objective_decision, expected_count=len(candidates))
    handoff_rows = build_candidate_handoff(candidates, audio_rows, objective_rows)
    objective_aggregate = _dict(objective_decision.get("aggregate"))
    audio_aggregate = _dict(audio_package.get("aggregate"))
    report = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "boundary": BOUNDARY,
        "source_reports": {
            "listening_package": {
                "schema_version": listening_package.get("schema_version"),
                "output_dir": listening_package.get("output_dir"),
                "candidate_count": _int(listening_package.get("candidate_count")),
            },
            "audio_package": {
                "schema_version": audio_package.get("schema_version"),
                "output_dir": audio_package.get("output_dir"),
                "rendered_wav_count": _int(audio_aggregate.get("rendered_wav_count")),
                "technical_wav_validation": bool(
                    audio_aggregate.get("technical_wav_validation", False)
                ),
            },
            "objective_decision": {
                "schema_version": objective_decision.get("schema_version"),
                "output_dir": objective_decision.get("output_dir"),
                "selected_next_target": _dict(objective_decision.get("decision")).get(
                    "selected_next_target"
                ),
            },
        },
        "candidate_handoff": handoff_rows,
        "aggregate": {
            "candidate_count": len(handoff_rows),
            "midi_count": len(handoff_rows),
            "wav_count": len(handoff_rows),
            "technical_wav_validation": bool(audio_aggregate.get("technical_wav_validation", False)),
            "objective_residual_label_count": sum(
                len(_list(row.get("residual_labels"))) for row in handoff_rows
            ),
            "midi_chord_tone_ratio_min": _float(
                objective_aggregate.get("midi_chord_tone_ratio_min")
            ),
            "midi_chord_tone_ratio_avg": _float(
                objective_aggregate.get("midi_chord_tone_ratio_avg")
            ),
            "midi_note_count_min": _int(objective_aggregate.get("midi_note_count_min")),
            "midi_note_count_max": _int(objective_aggregate.get("midi_note_count_max")),
            "midi_max_gap_seconds_max": _float(
                objective_aggregate.get("midi_max_gap_seconds_max")
            ),
            "midi_max_abs_interval_max": _int(
                objective_aggregate.get("midi_max_abs_interval_max")
            ),
        },
        "readiness": {
            "final_review_handoff_ready": bool(handoff_rows),
            "validated_listening_input_present": False,
            "preference_fill_allowed": False,
            "audio_rendered_quality_claimed": False,
            "musical_quality_claimed": False,
            "artist_style_claimed": False,
            "production_ready_claimed": False,
        },
        "decision": {
            "current_boundary": BOUNDARY,
            "selected_next_target": "manual_listening_review_pending",
            "next_boundary": NEXT_BOUNDARY,
            "critical_user_input_required": False,
            "reason": "latest MIDI/WAV review candidates are packaged; listening preference input remains pending",
        },
        "not_proven": [
            "human_audio_preference",
            "stable_jazz_solo_quality",
            "artist_level_long_solo_generation",
            "production_ready_improviser",
        ],
    }
    write_json(output_dir / "interval_contour_final_review_handoff.json", report)
    write_json(
        output_dir / "interval_contour_final_review_handoff_summary.json",
        validate_report(report, require_no_quality_claim=True),
    )
    write_text(output_dir / "interval_contour_final_review_handoff.md", markdown_report(report))
    return report


def validate_report(report: dict[str, Any], *, require_no_quality_claim: bool) -> dict[str, Any]:
    if str(report.get("schema_version") or "") != SCHEMA_VERSION:
        raise SoloYieldIntervalContourFinalReviewHandoffError("schema version mismatch")
    readiness = _dict(report.get("readiness"))
    if require_no_quality_claim:
        _require_no_quality_claim(report, label="handoff")
    aggregate = _dict(report.get("aggregate"))
    decision = _dict(report.get("decision"))
    return {
        "schema_version": str(report.get("schema_version") or ""),
        "boundary": str(report.get("boundary") or ""),
        "candidate_count": _int(aggregate.get("candidate_count")),
        "midi_count": _int(aggregate.get("midi_count")),
        "wav_count": _int(aggregate.get("wav_count")),
        "technical_wav_validation": bool(aggregate.get("technical_wav_validation", False)),
        "objective_residual_label_count": _int(aggregate.get("objective_residual_label_count")),
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
        "# Music Transformer Solo Yield Interval Contour Final Review Handoff",
        "",
        "## Summary",
        "",
        f"- boundary: `{report['boundary']}`",
        f"- candidate count: `{aggregate['candidate_count']}`",
        f"- MIDI/WAV count: `{aggregate['midi_count']}` / `{aggregate['wav_count']}`",
        f"- technical WAV validation: `{_bool_token(aggregate['technical_wav_validation'])}`",
        f"- objective residual label count: `{aggregate['objective_residual_label_count']}`",
        f"- chord-tone ratio min/avg: `{float(aggregate['midi_chord_tone_ratio_min']):.4f}` / `{float(aggregate['midi_chord_tone_ratio_avg']):.4f}`",
        f"- note count min/max: `{aggregate['midi_note_count_min']}` / `{aggregate['midi_note_count_max']}`",
        f"- max gap seconds: `{float(aggregate['midi_max_gap_seconds_max']):.4f}`",
        f"- max interval: `{aggregate['midi_max_abs_interval_max']}`",
        f"- validated listening input present: `{_bool_token(readiness['validated_listening_input_present'])}`",
        f"- preference fill allowed: `{_bool_token(readiness['preference_fill_allowed'])}`",
        f"- selected next target: `{decision['selected_next_target']}`",
        f"- next boundary: `{decision['next_boundary']}`",
        f"- musical quality claimed: `{_bool_token(readiness['musical_quality_claimed'])}`",
        "",
        "## Candidate Files",
        "",
    ]
    for row in report.get("candidate_handoff", []):
        profile = row["objective_profile"]
        lines.extend(
            [
                f"- candidate `{row['review_index']}` / `{row['case_label']}`",
                f"  - MIDI: `{row['review_midi_path']}`",
                f"  - WAV: `{row['review_wav_path']}`",
                f"  - note count: `{profile['midi_note_count']}`",
                f"  - chord-tone ratio: `{float(profile['midi_chord_tone_ratio']):.4f}`",
                f"  - max gap seconds: `{float(profile['midi_max_gap_seconds']):.4f}`",
                f"  - max interval: `{profile['midi_max_abs_interval']}`",
                f"  - residual labels: `none`",
            ]
        )
    lines.extend(["", "## Not Proven", ""])
    for item in report.get("not_proven", []):
        lines.append(f"- `{item}`")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build interval contour final review handoff")
    parser.add_argument(
        "--listening_package_report",
        type=str,
        default=(
            "outputs/music_transformer_finetune_mvp/"
            "solo_yield_interval_contour_aftercare_listening_review/"
            "issue_1308_interval_contour_listening_package/listening_review_package.json"
        ),
    )
    parser.add_argument(
        "--audio_package_report",
        type=str,
        default=(
            "outputs/music_transformer_finetune_mvp/solo_yield_interval_contour_aftercare_audio/"
            "issue_1306_interval_contour_audio_package/interval_contour_aftercare_audio_package.json"
        ),
    )
    parser.add_argument(
        "--objective_decision_report",
        type=str,
        default=(
            "outputs/music_transformer_finetune_mvp/"
            "solo_yield_interval_contour_aftercare_objective_next/"
            "issue_1312_interval_contour_objective_next/"
            "interval_contour_aftercare_objective_next_decision.json"
        ),
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/music_transformer_finetune_mvp/solo_yield_interval_contour_final_handoff",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--require_no_quality_claim", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    report = build_handoff_package(
        read_json(Path(args.listening_package_report)),
        read_json(Path(args.audio_package_report)),
        read_json(Path(args.objective_decision_report)),
        output_dir=output_dir,
    )
    summary = validate_report(report, require_no_quality_claim=bool(args.require_no_quality_claim))
    if args.doc_path:
        write_text(Path(args.doc_path), markdown_report(report))
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
