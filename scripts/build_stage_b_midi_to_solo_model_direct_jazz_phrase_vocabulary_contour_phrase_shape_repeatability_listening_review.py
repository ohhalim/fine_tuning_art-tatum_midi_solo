"""Build pending listening review boundary for repeatability WAV candidates."""

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
from scripts.build_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_repeatability_audio_package import (  # noqa: E402
    BOUNDARY as SOURCE_BOUNDARY,
)
from scripts.guard_stage_b_midi_to_solo_model_direct_user_listening_review_input import (  # noqa: E402
    parse_review_input,
)


class StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeRepeatabilityListeningReviewError(
    ValueError
):
    pass


BOUNDARY = (
    "stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_"
    "repeatability_listening_review"
)
NEXT_BOUNDARY = (
    "stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_"
    "repeatability_objective_only_next_decision"
)
SCHEMA_VERSION = (
    "stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_"
    "repeatability_listening_review_v1"
)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeRepeatabilityListeningReviewError(
            f"report missing: {path}"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


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


def compact_rendered_candidate(item: dict[str, Any]) -> dict[str, Any]:
    wav_file = _dict(item.get("wav_file"))
    wav_path = Path(str(wav_file.get("path") or ""))
    if not bool(wav_file.get("exists", False)) or not wav_path.exists():
        raise StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeRepeatabilityListeningReviewError(
            f"missing WAV for rank {item.get('rank')}"
        )
    return {
        "rank": _int(item.get("rank")),
        "midi_path": str(item.get("midi_path") or ""),
        "midi_sha256": str(item.get("midi_sha256") or ""),
        "wav_path": str(wav_path),
        "wav_sha256": str(wav_file.get("sha256") or ""),
        "duration_seconds": _float(wav_file.get("duration_seconds")),
        "sample_rate": _int(wav_file.get("sample_rate")),
        "size_bytes": _int(wav_file.get("size_bytes")),
    }


def validate_audio_package(audio_package: dict[str, Any], expected_count: int) -> list[dict[str, Any]]:
    boundary = _dict(audio_package.get("audio_package_boundary"))
    decision = _dict(audio_package.get("decision"))
    if str(boundary.get("boundary") or "") != SOURCE_BOUNDARY:
        raise StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeRepeatabilityListeningReviewError(
            "repeatability audio package boundary required"
        )
    if str(decision.get("next_boundary") or "") != BOUNDARY:
        raise StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeRepeatabilityListeningReviewError(
            "audio package must route to repeatability listening review"
        )
    if not bool(boundary.get("technical_wav_validation", False)):
        raise StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeRepeatabilityListeningReviewError(
            "technical WAV validation required"
        )
    blocked_claims = [
        "listening_review_completed",
        "audio_rendered_quality_claimed",
        "human_audio_preference_claimed",
        "model_direct_generation_quality_claimed",
        "midi_to_solo_musical_quality_claimed",
        "broad_trained_model_quality_claimed",
        "brad_style_adaptation_claimed",
    ]
    claimed = [name for name in blocked_claims if bool(boundary.get(name, False))]
    if claimed:
        raise StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeRepeatabilityListeningReviewError(
            f"unexpected upstream claim: {claimed}"
        )
    candidates = [
        compact_rendered_candidate(item)
        for item in _list(audio_package.get("rendered_audio_files"))
        if isinstance(item, dict)
    ]
    if len(candidates) != int(expected_count):
        raise StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeRepeatabilityListeningReviewError(
            "unexpected rendered candidate count"
        )
    return candidates


def listening_review_template(candidates: list[dict[str, Any]]) -> str:
    lines = [
        "# Stage B MIDI-to-Solo Contour Phrase-Shape Repeatability Listening Review Input",
        "",
        "## Review Status",
        "",
        "- reviewer: `pending`",
        "- reviewed_at: `pending`",
        "- preferred_rank: `pending`",
        "- reject_all: `pending`",
        "- broad_model_quality_claim_allowed: `false`",
        "",
        "## Candidates",
        "",
        "| rank | midi | wav | duration | sample rate | size | sha256 | decision |",
        "|---:|---|---|---:|---:|---:|---|---|",
    ]
    for item in candidates:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(item["rank"]),
                    item["midi_path"],
                    item["wav_path"],
                    f"{float(item['duration_seconds']):.3f}",
                    str(item["sample_rate"]),
                    str(item["size_bytes"]),
                    str(item["wav_sha256"][:12]),
                    "`pending`",
                ]
            )
            + " |"
        )
    lines.extend(["", "## Per-Candidate Notes", ""])
    for item in candidates:
        lines.extend(
            [
                f"### Rank {int(item['rank'])}",
                "",
                "- musical_acceptance: `pending`",
                "- issue_tags: `pending`",
                "- short_note: `pending`",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def build_listening_review_report(
    audio_package: dict[str, Any],
    *,
    output_dir: Path,
    expected_file_count: int,
) -> dict[str, Any]:
    candidates = validate_audio_package(audio_package, expected_count=expected_file_count)
    review_input_path = output_dir / "review" / "contour_phrase_shape_repeatability_listening_review_input.md"
    write_text(review_input_path, listening_review_template(candidates))
    parsed = parse_review_input(review_input_path.read_text(encoding="utf-8"))
    validated_review_input = bool(parsed.get("validated_review_input_present", False))
    boundary = {
        "boundary": BOUNDARY,
        "source_boundary": SOURCE_BOUNDARY,
        "candidate_count": len(candidates),
        "rendered_audio_file_count": len(candidates),
        "review_input_template_written": True,
        "validated_review_input_present": validated_review_input,
        "preference_fill_allowed": validated_review_input,
        "listening_review_completed": False,
        "human_audio_preference_claimed": False,
        "model_direct_generation_quality_claimed": False,
        "midi_to_solo_musical_quality_claimed": False,
        "broad_trained_model_quality_claimed": False,
        "brad_style_adaptation_claimed": False,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "boundary": BOUNDARY,
        "source_boundary": SOURCE_BOUNDARY,
        "review_candidates": candidates,
        "review_input_template_path": str(review_input_path),
        "review_input_summary": parsed,
        "listening_review_boundary": boundary,
        "readiness": {
            "boundary": BOUNDARY,
            "listening_review_boundary_prepared": True,
            "validated_review_input_present": validated_review_input,
            "preference_fill_allowed": validated_review_input,
            "listening_review_completed": False,
            "human_audio_preference_claimed": False,
            "model_direct_generation_quality_claimed": False,
            "midi_to_solo_musical_quality_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
        },
        "decision": {
            "current_boundary": BOUNDARY,
            "next_boundary": NEXT_BOUNDARY,
            "auto_progress_allowed": True,
            "critical_user_input_required": False,
            "reason": "repeatability listening review input remains pending; route to objective-only next decision without preference claim",
        },
        "not_proven": [
            "listening_review_completed",
            "human_audio_preference",
            "model_direct_generation_quality",
            "midi_to_solo_musical_quality",
            "broad_trained_model_quality",
            "brad_style_adaptation",
        ],
        "next_recommended_issue": (
            "Stage B MIDI-to-solo model-direct jazz phrase vocabulary contour phrase-shape repeatability objective-only next decision"
        ),
    }


def validate_listening_review_report(
    report: dict[str, Any],
    *,
    expected_boundary: str | None,
    expected_next_boundary: str | None,
    expected_file_count: int,
    require_pending_review: bool,
    require_no_quality_claim: bool,
) -> dict[str, Any]:
    boundary = _dict(report.get("listening_review_boundary"))
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    if expected_boundary and str(boundary.get("boundary") or "") != expected_boundary:
        raise StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeRepeatabilityListeningReviewError(
            f"expected boundary {expected_boundary}, got {boundary.get('boundary')}"
        )
    if expected_next_boundary and str(decision.get("next_boundary") or "") != expected_next_boundary:
        raise StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeRepeatabilityListeningReviewError(
            "unexpected next boundary"
        )
    if _int(boundary.get("candidate_count")) != int(expected_file_count):
        raise StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeRepeatabilityListeningReviewError(
            "unexpected candidate count"
        )
    if not Path(str(report.get("review_input_template_path") or "")).exists():
        raise StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeRepeatabilityListeningReviewError(
            "review input template missing"
        )
    if require_pending_review and bool(boundary.get("validated_review_input_present", True)):
        raise StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeRepeatabilityListeningReviewError(
            "review input should remain pending"
        )
    if require_no_quality_claim:
        blocked = [
            "listening_review_completed",
            "human_audio_preference_claimed",
            "model_direct_generation_quality_claimed",
            "midi_to_solo_musical_quality_claimed",
            "broad_trained_model_quality_claimed",
            "brad_style_adaptation_claimed",
        ]
        claimed = [name for name in blocked if bool(readiness.get(name, True))]
        if claimed:
            raise StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeRepeatabilityListeningReviewError(
                f"unexpected quality claim: {claimed}"
            )
    review_summary = _dict(report.get("review_input_summary"))
    return {
        "boundary": str(boundary.get("boundary") or ""),
        "source_boundary": str(boundary.get("source_boundary") or ""),
        "candidate_count": _int(boundary.get("candidate_count")),
        "rendered_audio_file_count": _int(boundary.get("rendered_audio_file_count")),
        "review_input_template_written": bool(boundary.get("review_input_template_written", False)),
        "validated_review_input_present": bool(boundary.get("validated_review_input_present", True)),
        "preference_fill_allowed": bool(boundary.get("preference_fill_allowed", True)),
        "pending_status_field_count": len(_list(review_summary.get("pending_status_fields"))),
        "pending_candidate_decision_count": len(_list(review_summary.get("pending_candidate_decisions"))),
        "pending_candidate_field_count": len(_list(review_summary.get("pending_candidate_fields"))),
        "listening_review_completed": bool(readiness.get("listening_review_completed", True)),
        "human_audio_preference_claimed": bool(readiness.get("human_audio_preference_claimed", True)),
        "midi_to_solo_musical_quality_claimed": bool(
            readiness.get("midi_to_solo_musical_quality_claimed", True)
        ),
        "critical_user_input_required": bool(decision.get("critical_user_input_required", True)),
        "review_input_template_path": str(report.get("review_input_template_path") or ""),
        "next_boundary": str(decision.get("next_boundary") or ""),
        "next_recommended_issue": str(report.get("next_recommended_issue") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    boundary = report["listening_review_boundary"]
    decision = report["decision"]
    summary = report["review_input_summary"]
    lines = [
        "# Stage B MIDI-to-Solo Model-Direct Jazz Phrase Vocabulary Contour Phrase-Shape Repeatability Listening Review",
        "",
        "## Summary",
        "",
        f"- boundary: `{boundary['boundary']}`",
        f"- source boundary: `{boundary['source_boundary']}`",
        f"- next boundary: `{decision['next_boundary']}`",
        f"- candidate count: `{boundary['candidate_count']}`",
        f"- rendered audio file count: `{boundary['rendered_audio_file_count']}`",
        f"- review input template written: `{_bool_token(boundary['review_input_template_written'])}`",
        f"- validated review input present: `{_bool_token(boundary['validated_review_input_present'])}`",
        f"- preference fill allowed: `{_bool_token(boundary['preference_fill_allowed'])}`",
        f"- listening review completed: `{_bool_token(boundary['listening_review_completed'])}`",
        f"- human/audio preference claimed: `{_bool_token(boundary['human_audio_preference_claimed'])}`",
        f"- MIDI-to-solo musical quality claimed: `{_bool_token(boundary['midi_to_solo_musical_quality_claimed'])}`",
        "",
        "## Pending Fields",
        "",
        f"- pending status fields: `{len(_list(summary.get('pending_status_fields')))}`",
        f"- pending candidate decisions: `{len(_list(summary.get('pending_candidate_decisions')))}`",
        f"- pending candidate fields: `{len(_list(summary.get('pending_candidate_fields')))}`",
        "",
        "## Review Input",
        "",
        f"- template: `{report['review_input_template_path']}`",
        "",
        "## Not Proven",
        "",
    ]
    for item in report.get("not_proven", []):
        lines.append(f"- `{item}`")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build repeatability listening review boundary")
    parser.add_argument("--audio_package", type=str, required=True)
    parser.add_argument(
        "--output_root",
        type=str,
        default=(
            "outputs/stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_"
            "contour_phrase_shape_repeatability_listening_review"
        ),
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--expected_boundary", type=str, default="")
    parser.add_argument("--expected_next_boundary", type=str, default="")
    parser.add_argument("--expected_file_count", type=int, default=6)
    parser.add_argument("--require_pending_review", action="store_true")
    parser.add_argument("--require_no_quality_claim", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    report = build_listening_review_report(
        read_json(Path(args.audio_package)),
        output_dir=output_dir,
        expected_file_count=int(args.expected_file_count),
    )
    summary = validate_listening_review_report(
        report,
        expected_boundary=str(args.expected_boundary or ""),
        expected_next_boundary=str(args.expected_next_boundary or ""),
        expected_file_count=int(args.expected_file_count),
        require_pending_review=bool(args.require_pending_review),
        require_no_quality_claim=bool(args.require_no_quality_claim),
    )
    write_json(
        output_dir / "stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_repeatability_listening_review.json",
        report,
    )
    write_json(
        output_dir
        / "stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_repeatability_listening_review_validation_summary.json",
        summary,
    )
    markdown = markdown_report(report)
    write_text(
        output_dir / "stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_repeatability_listening_review.md",
        markdown,
    )
    if args.doc_path:
        write_text(Path(args.doc_path), markdown)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
