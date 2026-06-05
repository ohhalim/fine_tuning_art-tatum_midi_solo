"""Select the next phrase-bank MIDI-to-solo step from objective evidence only."""

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
from scripts.guard_stage_b_midi_to_solo_phrase_bank_listening_review_input import (  # noqa: E402
    BOUNDARY as INPUT_GUARD_BOUNDARY,
    OBJECTIVE_NEXT_BOUNDARY as SOURCE_NEXT_BOUNDARY,
)
from scripts.render_stage_b_midi_to_solo_phrase_bank_audio import (  # noqa: E402
    BOUNDARY as AUDIO_RENDER_BOUNDARY,
)
from scripts.run_stage_b_midi_to_solo_phrase_bank_retrieval_baseline import (  # noqa: E402
    BOUNDARY as PHRASE_BANK_BOUNDARY,
)


class StageBMidiToSoloPhraseBankObjectiveNextError(ValueError):
    pass


BOUNDARY = "stage_b_midi_to_solo_phrase_bank_objective_only_next_decision"
NEXT_BOUNDARY = "stage_b_midi_to_solo_phrase_bank_dead_air_density_repair_probe"
SCHEMA_VERSION = "stage_b_midi_to_solo_phrase_bank_objective_only_next_decision_v1"

QUALITY_CLAIM_KEYS = [
    "human_audio_preference_claimed",
    "midi_to_solo_musical_quality_claimed",
    "musical_quality_claimed",
    "audio_rendered_quality_claimed",
    "phrase_bank_musical_quality_claimed",
    "model_checkpoint_generation_quality_claimed",
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
        raise StageBMidiToSoloPhraseBankObjectiveNextError(f"unexpected quality claim in {label}: {claimed}")


def validate_input_guard_report(report: dict[str, Any]) -> dict[str, Any]:
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    guard = _dict(report.get("guard_result"))
    if str(report.get("boundary") or "") != INPUT_GUARD_BOUNDARY:
        raise StageBMidiToSoloPhraseBankObjectiveNextError("phrase-bank input guard boundary required")
    if str(decision.get("next_boundary") or "") != SOURCE_NEXT_BOUNDARY:
        raise StageBMidiToSoloPhraseBankObjectiveNextError("input guard must route to objective-only next decision")
    if not bool(readiness.get("listening_review_input_guard_completed", False)):
        raise StageBMidiToSoloPhraseBankObjectiveNextError("input guard completion required")
    if bool(guard.get("validated_review_input_present", True)):
        raise StageBMidiToSoloPhraseBankObjectiveNextError("objective-only decision requires pending review input")
    if bool(guard.get("preference_fill_allowed", True)):
        raise StageBMidiToSoloPhraseBankObjectiveNextError("preference fill must remain blocked")
    if bool(decision.get("critical_user_input_required", True)):
        raise StageBMidiToSoloPhraseBankObjectiveNextError("critical user input should not be required")
    _require_no_quality_claim(readiness, label="input guard readiness")
    return {
        "boundary": INPUT_GUARD_BOUNDARY,
        "validated_review_input_present": bool(guard.get("validated_review_input_present", False)),
        "preference_fill_allowed": bool(guard.get("preference_fill_allowed", False)),
        "review_item_count": _int(guard.get("review_item_count")),
    }


def validate_phrase_bank_report(report: dict[str, Any]) -> dict[str, Any]:
    readiness = _dict(report.get("readiness"))
    summary = _dict(report.get("summary"))
    if str(report.get("boundary") or "") != PHRASE_BANK_BOUNDARY:
        raise StageBMidiToSoloPhraseBankObjectiveNextError("phrase-bank retrieval boundary required")
    if not bool(readiness.get("phrase_bank_retrieval_baseline_completed", False)):
        raise StageBMidiToSoloPhraseBankObjectiveNextError("phrase-bank retrieval completion required")
    if _int(summary.get("exported_candidate_count")) <= 0:
        raise StageBMidiToSoloPhraseBankObjectiveNextError("exported phrase-bank candidates required")
    _require_no_quality_claim(readiness, label="phrase-bank readiness")
    candidates = [_dict(item) for item in _list(report.get("top_candidates"))]
    if not candidates:
        raise StageBMidiToSoloPhraseBankObjectiveNextError("top phrase-bank candidates required")
    return {
        "boundary": PHRASE_BANK_BOUNDARY,
        "summary": summary,
        "objective_gate": _dict(report.get("objective_gate")),
        "top_candidates": candidates,
    }


def validate_audio_render_report(report: dict[str, Any]) -> dict[str, Any]:
    boundary = _dict(report.get("audio_render_boundary"))
    if str(report.get("source_boundary") or "") != PHRASE_BANK_BOUNDARY:
        raise StageBMidiToSoloPhraseBankObjectiveNextError("audio render source must be phrase-bank retrieval")
    if str(boundary.get("boundary") or "") != AUDIO_RENDER_BOUNDARY:
        raise StageBMidiToSoloPhraseBankObjectiveNextError("phrase-bank audio render boundary required")
    if not bool(boundary.get("technical_wav_validation", False)):
        raise StageBMidiToSoloPhraseBankObjectiveNextError("technical WAV validation required")
    if _int(boundary.get("rendered_audio_file_count")) <= 0:
        raise StageBMidiToSoloPhraseBankObjectiveNextError("rendered WAV files required")
    _require_no_quality_claim(boundary, label="audio render boundary")
    rendered = [_dict(item) for item in _list(report.get("rendered_audio_files"))]
    if not rendered:
        raise StageBMidiToSoloPhraseBankObjectiveNextError("rendered audio items required")
    return {
        "boundary": AUDIO_RENDER_BOUNDARY,
        "rendered_audio_files": rendered,
        "technical_wav_validation": bool(boundary.get("technical_wav_validation", False)),
    }


def same_bar_density_pattern(candidate: dict[str, Any]) -> bool:
    per_bar = _dict(_dict(candidate.get("collapse")).get("per_bar_note_counts"))
    values = [_int(value) for value in per_bar.values()]
    return bool(values) and len(set(values)) == 1


def candidate_risk_flags(
    candidate: dict[str, Any],
    *,
    dead_air_review_max: float,
    min_rhythm_diversity: float,
    min_approach_resolution: float,
    max_pitch_reuse_ratio: float,
) -> list[str]:
    metrics = _dict(candidate.get("metrics"))
    rhythm = _dict(candidate.get("rhythm_profile"))
    approach = _dict(candidate.get("approach_resolution"))
    contour = _dict(candidate.get("phrase_contour"))
    collapse = _dict(candidate.get("collapse"))
    flags: list[str] = []
    if _float(metrics.get("dead_air_ratio")) >= float(dead_air_review_max):
        flags.append("dead_air_ratio_above_review_threshold")
    if same_bar_density_pattern(candidate):
        flags.append("uniform_bar_note_density")
    if _float(rhythm.get("duration_diversity_ratio")) < float(min_rhythm_diversity):
        flags.append("low_duration_diversity")
    if _float(rhythm.get("ioi_diversity_ratio")) < float(min_rhythm_diversity):
        flags.append("low_ioi_diversity")
    if _float(approach.get("approach_resolution_ratio")) < float(min_approach_resolution):
        flags.append("low_approach_resolution")
    if _float(collapse.get("repeated_pitch_ratio")) > float(max_pitch_reuse_ratio):
        flags.append("high_pitch_reuse_ratio")
    if _float(contour.get("leap_motion_ratio")) <= 0.0:
        flags.append("no_leap_motion")
    return flags


def build_candidate_reviews(
    *,
    phrase_bank: dict[str, Any],
    audio_render: dict[str, Any],
    dead_air_review_max: float,
    min_rhythm_diversity: float,
    min_approach_resolution: float,
    max_pitch_reuse_ratio: float,
) -> list[dict[str, Any]]:
    rendered_by_seed = {
        _int(item.get("sample_seed")): item for item in _list(audio_render.get("rendered_audio_files"))
    }
    reviews: list[dict[str, Any]] = []
    for rank, candidate in enumerate(_list(phrase_bank.get("top_candidates")), start=1):
        seed = _int(candidate.get("sample_seed"))
        metrics = _dict(candidate.get("metrics"))
        rhythm = _dict(candidate.get("rhythm_profile"))
        approach = _dict(candidate.get("approach_resolution"))
        contour = _dict(candidate.get("phrase_contour"))
        collapse = _dict(candidate.get("collapse"))
        audio_item = _dict(rendered_by_seed.get(seed))
        wav = _dict(audio_item.get("wav_file"))
        flags = candidate_risk_flags(
            candidate,
            dead_air_review_max=dead_air_review_max,
            min_rhythm_diversity=min_rhythm_diversity,
            min_approach_resolution=min_approach_resolution,
            max_pitch_reuse_ratio=max_pitch_reuse_ratio,
        )
        reviews.append(
            {
                "rank": rank,
                "sample_seed": seed,
                "mode": str(audio_item.get("mode") or ""),
                "midi_path": str(audio_item.get("source_midi_path") or candidate.get("midi_path") or ""),
                "wav_path": str(wav.get("path") or ""),
                "objective_metrics": {
                    "note_count": _int(metrics.get("note_count")),
                    "unique_pitch_count": _int(metrics.get("unique_pitch_count")),
                    "max_simultaneous_notes": _int(metrics.get("max_simultaneous_notes")),
                    "dead_air_ratio": _float(metrics.get("dead_air_ratio")),
                    "phrase_coverage_ratio": _float(metrics.get("phrase_coverage_ratio")),
                    "note_density": _float(metrics.get("note_density")),
                    "duration_seconds": _float(wav.get("duration_seconds")),
                    "duration_diversity_ratio": _float(rhythm.get("duration_diversity_ratio")),
                    "ioi_diversity_ratio": _float(rhythm.get("ioi_diversity_ratio")),
                    "repeated_pitch_ratio": _float(collapse.get("repeated_pitch_ratio")),
                    "max_same_pitch_repeats": _int(collapse.get("max_same_pitch_repeats")),
                    "leap_motion_ratio": _float(contour.get("leap_motion_ratio")),
                    "approach_resolution_ratio": _float(approach.get("approach_resolution_ratio")),
                },
                "risk_flags": flags,
                "objective_keep_candidate": not flags,
                "repair_required": bool(flags),
            }
        )
    return reviews


def build_objective_next_report(
    *,
    input_guard_report: dict[str, Any],
    phrase_bank_report: dict[str, Any],
    audio_render_report: dict[str, Any],
    output_dir: Path,
    issue_number: int,
    dead_air_review_max: float,
    min_rhythm_diversity: float,
    min_approach_resolution: float,
    max_pitch_reuse_ratio: float,
) -> dict[str, Any]:
    input_guard = validate_input_guard_report(input_guard_report)
    phrase_bank = validate_phrase_bank_report(phrase_bank_report)
    audio_render = validate_audio_render_report(audio_render_report)
    candidate_reviews = build_candidate_reviews(
        phrase_bank=phrase_bank,
        audio_render=audio_render,
        dead_air_review_max=dead_air_review_max,
        min_rhythm_diversity=min_rhythm_diversity,
        min_approach_resolution=min_approach_resolution,
        max_pitch_reuse_ratio=max_pitch_reuse_ratio,
    )
    keep_count = sum(1 for item in candidate_reviews if bool(item["objective_keep_candidate"]))
    repair_count = sum(1 for item in candidate_reviews if bool(item["repair_required"]))
    dead_air_values = [
        _float(_dict(item.get("objective_metrics")).get("dead_air_ratio")) for item in candidate_reviews
    ]
    next_boundary = NEXT_BOUNDARY if repair_count else "stage_b_midi_to_solo_phrase_bank_cli_mvp_package"
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "issue_number": int(issue_number),
        "boundary": BOUNDARY,
        "source_boundaries": {
            "input_guard": input_guard["boundary"],
            "phrase_bank": phrase_bank["boundary"],
            "audio_render": audio_render["boundary"],
        },
        "review_policy": {
            "basis": "objective_midi_and_wav_metadata_only",
            "dead_air_review_max": float(dead_air_review_max),
            "min_rhythm_diversity": float(min_rhythm_diversity),
            "min_approach_resolution": float(min_approach_resolution),
            "max_pitch_reuse_ratio": float(max_pitch_reuse_ratio),
            "human_audio_preference_allowed": False,
        },
        "objective_summary": {
            "candidate_count": len(candidate_reviews),
            "objective_keep_candidate_count": keep_count,
            "repair_required_candidate_count": repair_count,
            "all_candidates_require_repair": repair_count == len(candidate_reviews),
            "max_dead_air_ratio": max(dead_air_values) if dead_air_values else 0.0,
            "min_dead_air_ratio": min(dead_air_values) if dead_air_values else 0.0,
            "technical_wav_validation": bool(audio_render["technical_wav_validation"]),
            "validated_review_input_present": bool(input_guard["validated_review_input_present"]),
            "preference_fill_allowed": bool(input_guard["preference_fill_allowed"]),
        },
        "candidate_reviews": candidate_reviews,
        "readiness": {
            "boundary": BOUNDARY,
            "objective_only_next_decision_completed": True,
            "objective_keep_candidate_available": keep_count > 0,
            "phrase_bank_repair_required": repair_count > 0,
            "human_audio_preference_claimed": False,
            "midi_to_solo_musical_quality_claimed": False,
            "audio_rendered_quality_claimed": False,
            "phrase_bank_musical_quality_claimed": False,
            "model_checkpoint_generation_quality_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
        },
        "decision": {
            "current_boundary": BOUNDARY,
            "next_boundary": next_boundary,
            "auto_progress_allowed": True,
            "critical_user_input_required": False,
            "reason": (
                "phrase-bank candidates require objective repair before CLI MVP packaging"
                if repair_count
                else "phrase-bank objective candidate available for CLI MVP packaging"
            ),
        },
        "not_proven": [
            "human_audio_preference",
            "midi_to_solo_musical_quality",
            "phrase_bank_musical_quality",
            "broad_trained_model_quality",
            "brad_style_adaptation",
        ],
        "next_recommended_issue": (
            "Stage B MIDI-to-solo phrase-bank dead-air density repair probe"
            if repair_count
            else "Stage B MIDI-to-solo phrase-bank CLI MVP package"
        ),
    }


def validate_objective_next_report(
    report: dict[str, Any],
    *,
    expected_boundary: str | None,
    expected_next_boundary: str | None,
    require_objective_decision: bool,
    require_repair_required: bool,
    require_no_quality_claim: bool,
) -> dict[str, Any]:
    boundary = str(report.get("boundary") or "")
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    summary = _dict(report.get("objective_summary"))
    if expected_boundary and boundary != expected_boundary:
        raise StageBMidiToSoloPhraseBankObjectiveNextError(
            f"expected boundary {expected_boundary}, got {boundary}"
        )
    if expected_next_boundary and str(decision.get("next_boundary") or "") != expected_next_boundary:
        raise StageBMidiToSoloPhraseBankObjectiveNextError("unexpected next boundary")
    if require_objective_decision and not bool(readiness.get("objective_only_next_decision_completed", False)):
        raise StageBMidiToSoloPhraseBankObjectiveNextError("objective-only decision completion required")
    if require_repair_required and not bool(readiness.get("phrase_bank_repair_required", False)):
        raise StageBMidiToSoloPhraseBankObjectiveNextError("phrase-bank repair requirement expected")
    if bool(decision.get("critical_user_input_required", True)):
        raise StageBMidiToSoloPhraseBankObjectiveNextError("critical user input should not be required")
    if bool(summary.get("preference_fill_allowed", True)):
        raise StageBMidiToSoloPhraseBankObjectiveNextError("preference fill must remain blocked")
    if require_no_quality_claim:
        _require_no_quality_claim(readiness, label="objective-next readiness")
    return {
        "boundary": boundary,
        "next_boundary": str(decision.get("next_boundary") or ""),
        "candidate_count": _int(summary.get("candidate_count")),
        "objective_keep_candidate_count": _int(summary.get("objective_keep_candidate_count")),
        "repair_required_candidate_count": _int(summary.get("repair_required_candidate_count")),
        "all_candidates_require_repair": bool(summary.get("all_candidates_require_repair", False)),
        "max_dead_air_ratio": _float(summary.get("max_dead_air_ratio")),
        "human_audio_preference_claimed": bool(readiness.get("human_audio_preference_claimed", True)),
        "midi_to_solo_musical_quality_claimed": bool(
            readiness.get("midi_to_solo_musical_quality_claimed", True)
        ),
        "critical_user_input_required": bool(decision.get("critical_user_input_required", True)),
        "next_recommended_issue": str(report.get("next_recommended_issue") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    summary = report["objective_summary"]
    readiness = report["readiness"]
    decision = report["decision"]
    policy = report["review_policy"]
    lines = [
        "# Stage B MIDI-to-Solo Phrase-Bank Objective-Only Next Decision",
        "",
        "## Summary",
        "",
        f"- boundary: `{report['boundary']}`",
        f"- next boundary: `{decision['next_boundary']}`",
        f"- review basis: `{policy['basis']}`",
        f"- candidate count: `{summary['candidate_count']}`",
        f"- objective keep candidate count: `{summary['objective_keep_candidate_count']}`",
        f"- repair required candidate count: `{summary['repair_required_candidate_count']}`",
        f"- all candidates require repair: `{_bool_token(summary['all_candidates_require_repair'])}`",
        f"- dead-air range: `{summary['min_dead_air_ratio']:.4f} - {summary['max_dead_air_ratio']:.4f}`",
        f"- preference fill allowed: `{_bool_token(summary['preference_fill_allowed'])}`",
        f"- human/audio preference claimed: `{_bool_token(readiness['human_audio_preference_claimed'])}`",
        f"- MIDI-to-solo musical quality claimed: `{_bool_token(readiness['midi_to_solo_musical_quality_claimed'])}`",
        "",
        "## Candidate Objective Review",
        "",
    ]
    for item in report["candidate_reviews"]:
        metrics = item["objective_metrics"]
        lines.extend(
            [
                f"### Rank {item['rank']}",
                "",
                f"- seed: `{item['sample_seed']}`",
                f"- notes / unique pitches / max simultaneous: `{metrics['note_count']} / {metrics['unique_pitch_count']} / {metrics['max_simultaneous_notes']}`",
                f"- dead-air / phrase coverage: `{metrics['dead_air_ratio']:.4f} / {metrics['phrase_coverage_ratio']:.4f}`",
                f"- duration diversity / IOI diversity: `{metrics['duration_diversity_ratio']:.4f} / {metrics['ioi_diversity_ratio']:.4f}`",
                f"- approach resolution / repeated pitch ratio: `{metrics['approach_resolution_ratio']:.4f} / {metrics['repeated_pitch_ratio']:.4f}`",
                f"- repair required: `{_bool_token(item['repair_required'])}`",
                f"- risk flags: `{', '.join(item['risk_flags'])}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Decision",
            "",
            f"- auto progress allowed: `{_bool_token(decision['auto_progress_allowed'])}`",
            f"- critical user input required: `{_bool_token(decision['critical_user_input_required'])}`",
            f"- reason: `{decision['reason']}`",
            f"- next recommended issue: `{report['next_recommended_issue']}`",
            "",
            "## Not Proven",
            "",
        ]
    )
    for item in report["not_proven"]:
        lines.append(f"- `{item}`")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_id", default="harness_stage_b_midi_to_solo_phrase_bank_objective_next")
    parser.add_argument("--output_root", type=Path, default=Path("outputs/stage_b_midi_to_solo_phrase_bank_objective_only_next_decision"))
    parser.add_argument("--input_guard_report", type=Path, required=True)
    parser.add_argument("--phrase_bank_report", type=Path, required=True)
    parser.add_argument("--audio_render_report", type=Path, required=True)
    parser.add_argument("--doc_path", type=Path)
    parser.add_argument("--issue_number", type=int, default=640)
    parser.add_argument("--dead_air_review_max", type=float, default=0.45)
    parser.add_argument("--min_rhythm_diversity", type=float, default=0.12)
    parser.add_argument("--min_approach_resolution", type=float, default=0.40)
    parser.add_argument("--max_pitch_reuse_ratio", type=float, default=0.60)
    parser.add_argument("--expected_boundary")
    parser.add_argument("--expected_next_boundary")
    parser.add_argument("--require_objective_decision", action="store_true")
    parser.add_argument("--require_repair_required", action="store_true")
    parser.add_argument("--require_no_quality_claim", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_root / args.run_id
    report = build_objective_next_report(
        input_guard_report=read_json(args.input_guard_report),
        phrase_bank_report=read_json(args.phrase_bank_report),
        audio_render_report=read_json(args.audio_render_report),
        output_dir=output_dir,
        issue_number=args.issue_number,
        dead_air_review_max=args.dead_air_review_max,
        min_rhythm_diversity=args.min_rhythm_diversity,
        min_approach_resolution=args.min_approach_resolution,
        max_pitch_reuse_ratio=args.max_pitch_reuse_ratio,
    )
    summary = validate_objective_next_report(
        report,
        expected_boundary=args.expected_boundary,
        expected_next_boundary=args.expected_next_boundary,
        require_objective_decision=args.require_objective_decision,
        require_repair_required=args.require_repair_required,
        require_no_quality_claim=args.require_no_quality_claim,
    )
    json_path = output_dir / "stage_b_midi_to_solo_phrase_bank_objective_only_next_decision.json"
    md_path = output_dir / "stage_b_midi_to_solo_phrase_bank_objective_only_next_decision.md"
    validation_path = output_dir / "stage_b_midi_to_solo_phrase_bank_objective_only_next_decision_validation_summary.json"
    write_json(json_path, report)
    write_text(md_path, markdown_report(report))
    write_json(validation_path, summary)
    if args.doc_path:
        write_text(args.doc_path, markdown_report(report))
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
