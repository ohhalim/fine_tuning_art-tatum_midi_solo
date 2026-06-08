"""Build a listening review package for phrase-bank CLI audio smoke output."""

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
from scripts.render_stage_b_midi_to_solo_phrase_bank_cli_audio_smoke import (  # noqa: E402
    BOUNDARY as AUDIO_RENDER_BOUNDARY,
    NEXT_BOUNDARY as AUDIO_RENDER_NEXT_BOUNDARY,
)


class StageBMidiToSoloPhraseBankCliListeningPackageError(ValueError):
    pass


BOUNDARY = "stage_b_midi_to_solo_phrase_bank_cli_listening_review_package"
NEXT_BOUNDARY = "stage_b_midi_to_solo_phrase_bank_cli_listening_review_input_guard"
SCHEMA_VERSION = "stage_b_midi_to_solo_phrase_bank_cli_listening_review_package_v1"

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


def _path_exists(path_text: str) -> bool:
    return bool(path_text and Path(path_text).exists())


def _require_no_quality_claim(container: dict[str, Any], *, label: str) -> None:
    claimed = [name for name in QUALITY_CLAIM_KEYS if bool(container.get(name, False))]
    if claimed:
        raise StageBMidiToSoloPhraseBankCliListeningPackageError(
            f"unexpected quality claim in {label}: {claimed}"
        )


def validate_audio_render_report(report: dict[str, Any], *, expected_count: int) -> list[dict[str, Any]]:
    boundary = _dict(report.get("audio_render_boundary"))
    decision = _dict(report.get("decision"))
    if str(boundary.get("boundary") or "") != AUDIO_RENDER_BOUNDARY:
        raise StageBMidiToSoloPhraseBankCliListeningPackageError("CLI audio render smoke boundary required")
    if str(decision.get("next_boundary") or "") != AUDIO_RENDER_NEXT_BOUNDARY:
        raise StageBMidiToSoloPhraseBankCliListeningPackageError("audio smoke must route to listening package")
    required_true = [
        "technical_wav_validation",
        "cli_user_input_audio_render_completed",
        "phrase_bank_ranked_audio_render_completed",
        "phrase_bank_listening_review_package_required",
    ]
    missing = [name for name in required_true if not bool(boundary.get(name, False))]
    if missing:
        raise StageBMidiToSoloPhraseBankCliListeningPackageError(
            f"missing audio render readiness: {missing}"
        )
    if bool(decision.get("critical_user_input_required", True)):
        raise StageBMidiToSoloPhraseBankCliListeningPackageError("critical user input should not be required")
    _require_no_quality_claim(boundary, label="audio render boundary")
    items = [_dict(item) for item in _list(report.get("rendered_audio_files"))]
    if len(items) < int(expected_count):
        raise StageBMidiToSoloPhraseBankCliListeningPackageError("rendered audio file count below expected")
    review_items: list[dict[str, Any]] = []
    for item in items[: int(expected_count)]:
        wav = _dict(item.get("wav_file"))
        wav_path = str(wav.get("path") or "")
        midi_path = str(item.get("source_midi_path") or "")
        if not _path_exists(wav_path) or not _path_exists(midi_path):
            raise StageBMidiToSoloPhraseBankCliListeningPackageError("review item MIDI/WAV artifact required")
        review_items.append(
            {
                "rank": _int(item.get("rank")),
                "mode": str(item.get("mode") or ""),
                "sample_index": _int(item.get("sample_index")),
                "sample_seed": _int(item.get("sample_seed")),
                "midi_path": midi_path,
                "wav_path": wav_path,
                "duration_seconds": _float(wav.get("duration_seconds")),
                "sample_rate": _int(wav.get("sample_rate")),
                "size_bytes": _int(wav.get("size_bytes")),
                "sha256": str(wav.get("sha256") or ""),
                "source_note_count": _int(item.get("source_note_count")),
                "source_unique_pitch_count": _int(item.get("source_unique_pitch_count")),
                "source_dead_air_ratio": _float(item.get("source_dead_air_ratio")),
                "source_phrase_coverage_ratio": _float(item.get("source_phrase_coverage_ratio")),
                "review_status": "pending",
            }
        )
    return review_items


def build_listening_review_package_report(
    *,
    audio_render_report: dict[str, Any],
    output_dir: Path,
    issue_number: int,
    expected_count: int,
) -> dict[str, Any]:
    review_items = validate_audio_render_report(audio_render_report, expected_count=expected_count)
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "issue_number": int(issue_number),
        "boundary": BOUNDARY,
        "source_boundaries": {
            "audio_render": AUDIO_RENDER_BOUNDARY,
        },
        "review_package": {
            "package_ready": True,
            "review_item_count": int(len(review_items)),
            "review_basis": "human_audio_listening_pending",
            "validated_review_input": False,
            "required_input_fields": [
                "candidate_rank",
                "listening_status",
                "preference",
                "issue_notes",
            ],
        },
        "review_items": review_items,
        "readiness": {
            "boundary": BOUNDARY,
            "listening_review_package_ready": True,
            "review_item_count": int(len(review_items)),
            "validated_review_input": False,
            "human_review_required_now": False,
            "human_audio_preference_claimed": False,
            "midi_to_solo_musical_quality_claimed": False,
            "audio_rendered_quality_claimed": False,
            "phrase_bank_musical_quality_claimed": False,
            "model_checkpoint_generation_quality_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
            "production_ready_claimed": False,
        },
        "decision": {
            "current_boundary": BOUNDARY,
            "next_boundary": NEXT_BOUNDARY,
            "auto_progress_allowed": True,
            "critical_user_input_required": False,
            "reason": "CLI WAV/MIDI review items are packaged; preference remains pending until validated listening input",
        },
        "not_proven": [
            "human_audio_preference",
            "midi_to_solo_musical_quality",
            "audio_rendered_quality",
            "phrase_bank_musical_quality",
            "broad_trained_model_quality",
            "brad_style_adaptation",
            "production_ready",
        ],
        "next_recommended_issue": "Stage B MIDI-to-solo phrase-bank CLI listening review input guard",
    }


def validate_listening_review_package_report(
    report: dict[str, Any],
    *,
    expected_boundary: str | None,
    expected_next_boundary: str | None,
    expected_review_item_count: int,
    require_package_ready: bool,
    require_no_quality_claim: bool,
) -> dict[str, Any]:
    boundary = str(report.get("boundary") or "")
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    if expected_boundary and boundary != expected_boundary:
        raise StageBMidiToSoloPhraseBankCliListeningPackageError(
            f"expected boundary {expected_boundary}, got {boundary}"
        )
    if expected_next_boundary and str(decision.get("next_boundary") or "") != expected_next_boundary:
        raise StageBMidiToSoloPhraseBankCliListeningPackageError("unexpected next boundary")
    if require_package_ready and not bool(readiness.get("listening_review_package_ready", False)):
        raise StageBMidiToSoloPhraseBankCliListeningPackageError("listening review package should be ready")
    if _int(readiness.get("review_item_count")) != int(expected_review_item_count):
        raise StageBMidiToSoloPhraseBankCliListeningPackageError("review item count mismatch")
    if bool(readiness.get("validated_review_input", True)):
        raise StageBMidiToSoloPhraseBankCliListeningPackageError("review input should remain pending")
    if bool(decision.get("critical_user_input_required", True)):
        raise StageBMidiToSoloPhraseBankCliListeningPackageError("critical user input should not be required")
    if require_no_quality_claim:
        _require_no_quality_claim(readiness, label="listening package readiness")
    return {
        "boundary": boundary,
        "next_boundary": str(decision.get("next_boundary") or ""),
        "listening_review_package_ready": bool(readiness.get("listening_review_package_ready", False)),
        "review_item_count": _int(readiness.get("review_item_count")),
        "validated_review_input": bool(readiness.get("validated_review_input", True)),
        "human_review_required_now": bool(readiness.get("human_review_required_now", True)),
        "human_audio_preference_claimed": bool(readiness.get("human_audio_preference_claimed", True)),
        "midi_to_solo_musical_quality_claimed": bool(
            readiness.get("midi_to_solo_musical_quality_claimed", True)
        ),
        "critical_user_input_required": bool(decision.get("critical_user_input_required", True)),
        "wav_paths": [str(_dict(item).get("wav_path") or "") for item in _list(report.get("review_items"))],
        "next_recommended_issue": str(report.get("next_recommended_issue") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    readiness = report["readiness"]
    decision = report["decision"]
    package = report["review_package"]
    lines = [
        "# Stage B MIDI-to-Solo Phrase-Bank CLI Listening Review Package",
        "",
        "## Summary",
        "",
        f"- boundary: `{report['boundary']}`",
        f"- next boundary: `{decision['next_boundary']}`",
        f"- package ready: `{_bool_token(package['package_ready'])}`",
        f"- review item count: `{package['review_item_count']}`",
        f"- validated review input: `{_bool_token(package['validated_review_input'])}`",
        f"- human/audio preference claimed: `{_bool_token(readiness['human_audio_preference_claimed'])}`",
        f"- MIDI-to-solo musical quality claimed: `{_bool_token(readiness['midi_to_solo_musical_quality_claimed'])}`",
        "",
        "## Review Items",
        "",
    ]
    for item in report["review_items"]:
        lines.extend(
            [
                f"### Rank {item['rank']}",
                "",
                f"- seed: `{item['sample_seed']}`",
                f"- MIDI: `{item['midi_path']}`",
                f"- WAV: `{item['wav_path']}`",
                f"- duration / sample rate / sha256: `{item['duration_seconds']:.3f}s / {item['sample_rate']} / {item['sha256'][:12]}`",
                f"- notes / unique pitches / dead-air: `{item['source_note_count']} / {item['source_unique_pitch_count']} / {item['source_dead_air_ratio']:.4f}`",
                f"- review status: `{item['review_status']}`",
                "",
            ]
        )
    lines.extend(["## Required Review Input", ""])
    for field in package["required_input_fields"]:
        lines.append(f"- `{field}`")
    lines.extend(["", "## Claim Boundary", ""])
    for item in report["not_proven"]:
        lines.append(f"- `{item}`")
    lines.extend(["", "## Next", "", f"- `{report['next_recommended_issue']}`", ""])
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build CLI listening review package from rendered WAV files")
    parser.add_argument("--audio_render_report", type=str, required=True)
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/stage_b_midi_to_solo_phrase_bank_cli_listening_review_package",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--issue_number", type=int, default=658)
    parser.add_argument("--expected_boundary", type=str, default="")
    parser.add_argument("--expected_next_boundary", type=str, default="")
    parser.add_argument("--expected_review_item_count", type=int, default=3)
    parser.add_argument("--require_package_ready", action="store_true")
    parser.add_argument("--require_no_quality_claim", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    report = build_listening_review_package_report(
        audio_render_report=read_json(Path(args.audio_render_report)),
        output_dir=output_dir,
        issue_number=int(args.issue_number),
        expected_count=int(args.expected_review_item_count),
    )
    summary = validate_listening_review_package_report(
        report,
        expected_boundary=str(args.expected_boundary or ""),
        expected_next_boundary=str(args.expected_next_boundary or ""),
        expected_review_item_count=int(args.expected_review_item_count),
        require_package_ready=bool(args.require_package_ready),
        require_no_quality_claim=bool(args.require_no_quality_claim),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "stage_b_midi_to_solo_phrase_bank_cli_listening_review_package.json", report)
    write_json(
        output_dir / "stage_b_midi_to_solo_phrase_bank_cli_listening_review_package_validation_summary.json",
        summary,
    )
    markdown = markdown_report(report)
    write_text(output_dir / "stage_b_midi_to_solo_phrase_bank_cli_listening_review_package.md", markdown)
    if args.doc_path:
        write_text(Path(args.doc_path), markdown)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
