"""Consolidate model-conditioned MIDI-to-solo input-path replacement evidence."""

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
from scripts.export_stage_b_midi_to_solo_model_conditioned_input_path_candidates import (  # noqa: E402
    BOUNDARY as CANDIDATE_EXPORT_BOUNDARY,
)
from scripts.render_stage_b_midi_to_solo_model_conditioned_input_path_audio import (  # noqa: E402
    BOUNDARY as AUDIO_RENDER_BOUNDARY,
    NEXT_BOUNDARY as AUDIO_RENDER_NEXT_BOUNDARY,
)


class StageBMidiToSoloModelConditionedInputPathReplacementConsolidationError(ValueError):
    pass


BOUNDARY = "stage_b_midi_to_solo_model_conditioned_input_path_replacement_consolidation"
NEXT_BOUNDARY = "stage_b_midi_to_solo_model_conditioned_input_path_listening_review_package"
SCHEMA_VERSION = "stage_b_midi_to_solo_model_conditioned_input_path_replacement_consolidation_v1"

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
        raise StageBMidiToSoloModelConditionedInputPathReplacementConsolidationError(
            f"unexpected quality claim in {label}: {claimed}"
        )


def validate_candidate_export(report: dict[str, Any], *, expected_count: int) -> dict[str, Any]:
    readiness = _dict(report.get("readiness"))
    summary = _dict(report.get("summary"))
    if str(report.get("boundary") or readiness.get("boundary") or "") != CANDIDATE_EXPORT_BOUNDARY:
        raise StageBMidiToSoloModelConditionedInputPathReplacementConsolidationError(
            "candidate export boundary required"
        )
    required_true = [
        "model_conditioned_input_path_candidate_export_completed",
        "ranked_midi_candidates_exported",
        "model_conditioned_ranked_input_path_contract_matched",
        "fallback_replacement_candidate_export_ready",
    ]
    missing = [name for name in required_true if not bool(readiness.get(name, False))]
    if missing:
        raise StageBMidiToSoloModelConditionedInputPathReplacementConsolidationError(
            f"missing candidate export readiness: {missing}"
        )
    if _int(summary.get("exported_candidate_count")) < int(expected_count):
        raise StageBMidiToSoloModelConditionedInputPathReplacementConsolidationError(
            "exported candidate count below expected"
        )
    top = [_dict(item) for item in _list(report.get("top_candidates"))[: int(expected_count)]]
    midi_paths = [str(item.get("export_midi_path") or "") for item in top]
    if len(midi_paths) < int(expected_count) or not all(_path_exists(path) for path in midi_paths):
        raise StageBMidiToSoloModelConditionedInputPathReplacementConsolidationError(
            "ranked MIDI artifacts required"
        )
    _require_no_quality_claim(readiness, label="candidate export readiness")
    return {
        "boundary": CANDIDATE_EXPORT_BOUNDARY,
        "ranked_midi_export_ready": True,
        "exported_candidate_count": _int(summary.get("exported_candidate_count")),
        "best_note_count": _int(summary.get("best_note_count")),
        "best_unique_pitch_count": _int(summary.get("best_unique_pitch_count")),
        "best_max_simultaneous_notes": _int(summary.get("best_max_simultaneous_notes")),
        "best_dead_air_ratio": _float(summary.get("best_dead_air_ratio")),
        "midi_paths": midi_paths,
    }


def validate_audio_render(report: dict[str, Any], *, expected_count: int) -> dict[str, Any]:
    boundary = _dict(report.get("audio_render_boundary"))
    decision = _dict(report.get("decision"))
    if str(boundary.get("boundary") or "") != AUDIO_RENDER_BOUNDARY:
        raise StageBMidiToSoloModelConditionedInputPathReplacementConsolidationError(
            "audio render boundary required"
        )
    if str(decision.get("next_boundary") or "") != AUDIO_RENDER_NEXT_BOUNDARY:
        raise StageBMidiToSoloModelConditionedInputPathReplacementConsolidationError(
            "audio render report must route to replacement consolidation"
        )
    required_true = [
        "render_attempted",
        "technical_wav_validation",
        "model_conditioned_ranked_audio_render_completed",
        "fallback_replacement_candidate_export_ready",
        "fallback_replacement_technical_path_ready",
        "fallback_replacement_ready",
    ]
    missing = [name for name in required_true if not bool(boundary.get(name, False))]
    if missing:
        raise StageBMidiToSoloModelConditionedInputPathReplacementConsolidationError(
            f"missing audio render readiness: {missing}"
        )
    files = [_dict(item) for item in _list(report.get("rendered_audio_files"))[: int(expected_count)]]
    if len(files) < int(expected_count):
        raise StageBMidiToSoloModelConditionedInputPathReplacementConsolidationError(
            "rendered file count below expected"
        )
    wav_paths: list[str] = []
    source_midi_paths: list[str] = []
    durations: list[float] = []
    for item in files:
        wav_file = _dict(item.get("wav_file"))
        wav_path = str(wav_file.get("path") or "")
        source_midi_path = str(item.get("source_midi_path") or "")
        if not _path_exists(wav_path):
            raise StageBMidiToSoloModelConditionedInputPathReplacementConsolidationError(
                "rendered WAV artifact required"
            )
        if _int(wav_file.get("sample_rate")) != 44100:
            raise StageBMidiToSoloModelConditionedInputPathReplacementConsolidationError(
                "unexpected WAV sample rate"
            )
        if _int(wav_file.get("frame_count")) <= 0:
            raise StageBMidiToSoloModelConditionedInputPathReplacementConsolidationError("empty WAV")
        wav_paths.append(wav_path)
        source_midi_paths.append(source_midi_path)
        durations.append(_float(wav_file.get("duration_seconds")))
    _require_no_quality_claim(boundary, label="audio render boundary")
    return {
        "boundary": AUDIO_RENDER_BOUNDARY,
        "ranked_audio_render_ready": True,
        "rendered_audio_file_count": _int(boundary.get("rendered_audio_file_count")),
        "technical_wav_validation": True,
        "fallback_replacement_technical_path_ready": True,
        "wav_duration_min_seconds": min(durations) if durations else 0.0,
        "wav_duration_max_seconds": max(durations) if durations else 0.0,
        "wav_paths": wav_paths,
        "source_midi_paths": source_midi_paths,
    }


def build_replacement_consolidation_report(
    *,
    candidate_export_report: dict[str, Any],
    audio_render_report: dict[str, Any],
    output_dir: Path,
    issue_number: int,
    expected_count: int,
) -> dict[str, Any]:
    candidate = validate_candidate_export(candidate_export_report, expected_count=expected_count)
    audio = validate_audio_render(audio_render_report, expected_count=expected_count)
    midi_audio_paths_match = candidate["midi_paths"] == audio["source_midi_paths"]
    if not midi_audio_paths_match:
        raise StageBMidiToSoloModelConditionedInputPathReplacementConsolidationError(
            "ranked MIDI export paths must match audio render source paths"
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "issue_number": int(issue_number),
        "boundary": BOUNDARY,
        "source_boundaries": {
            "candidate_export": CANDIDATE_EXPORT_BOUNDARY,
            "audio_render": AUDIO_RENDER_BOUNDARY,
        },
        "candidate_export": candidate,
        "audio_render": audio,
        "replacement_consolidation": {
            "model_conditioned_input_to_ranked_midi_completed": True,
            "model_conditioned_input_to_ranked_wav_completed": True,
            "ranked_midi_audio_paths_matched": True,
            "fallback_replacement_technical_path_ready": True,
            "fallback_replacement_ready": True,
            "musical_quality_completed": False,
            "human_audio_preference_completed": False,
        },
        "readiness": {
            "boundary": BOUNDARY,
            "model_conditioned_input_path_replacement_consolidated": True,
            "model_conditioned_input_to_ranked_midi_completed": True,
            "model_conditioned_input_to_ranked_wav_completed": True,
            "fallback_replacement_technical_path_ready": True,
            "fallback_replacement_ready": True,
            "listening_review_package_required": True,
            "human_review_required_now": False,
            "human_audio_preference_claimed": False,
            "midi_to_solo_musical_quality_claimed": False,
            "audio_rendered_quality_claimed": False,
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
            "reason": (
                "model-conditioned ranked MIDI and WAV technical path is consolidated; next boundary should "
                "package listening review without claiming preference"
            ),
        },
        "not_proven": [
            "human_audio_preference",
            "midi_to_solo_musical_quality",
            "audio_rendered_quality",
            "broad_trained_model_quality",
            "brad_style_adaptation",
        ],
        "next_recommended_issue": "Stage B MIDI-to-solo model-conditioned input path listening review package",
    }


def validate_replacement_consolidation_report(
    report: dict[str, Any],
    *,
    expected_boundary: str | None,
    expected_next_boundary: str | None,
    require_technical_replacement: bool,
    require_listening_review_package: bool,
    require_no_quality_claim: bool,
) -> dict[str, Any]:
    boundary = str(report.get("boundary") or "")
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    if expected_boundary and boundary != expected_boundary:
        raise StageBMidiToSoloModelConditionedInputPathReplacementConsolidationError(
            f"expected boundary {expected_boundary}, got {boundary}"
        )
    if expected_next_boundary and str(decision.get("next_boundary") or "") != expected_next_boundary:
        raise StageBMidiToSoloModelConditionedInputPathReplacementConsolidationError(
            "unexpected next boundary"
        )
    if require_technical_replacement:
        required = [
            "model_conditioned_input_path_replacement_consolidated",
            "model_conditioned_input_to_ranked_midi_completed",
            "model_conditioned_input_to_ranked_wav_completed",
            "fallback_replacement_technical_path_ready",
            "fallback_replacement_ready",
        ]
        missing = [name for name in required if not bool(readiness.get(name, False))]
        if missing:
            raise StageBMidiToSoloModelConditionedInputPathReplacementConsolidationError(
                f"missing technical replacement readiness: {missing}"
            )
    if require_listening_review_package and not bool(readiness.get("listening_review_package_required", False)):
        raise StageBMidiToSoloModelConditionedInputPathReplacementConsolidationError(
            "listening review package should be required"
        )
    if bool(decision.get("critical_user_input_required", True)):
        raise StageBMidiToSoloModelConditionedInputPathReplacementConsolidationError(
            "critical user input should not be required"
        )
    if require_no_quality_claim:
        _require_no_quality_claim(readiness, label="replacement consolidation readiness")
    candidate = _dict(report.get("candidate_export"))
    audio = _dict(report.get("audio_render"))
    return {
        "boundary": boundary,
        "next_boundary": str(decision.get("next_boundary") or ""),
        "model_conditioned_input_path_replacement_consolidated": bool(
            readiness.get("model_conditioned_input_path_replacement_consolidated", False)
        ),
        "model_conditioned_input_to_ranked_midi_completed": bool(
            readiness.get("model_conditioned_input_to_ranked_midi_completed", False)
        ),
        "model_conditioned_input_to_ranked_wav_completed": bool(
            readiness.get("model_conditioned_input_to_ranked_wav_completed", False)
        ),
        "fallback_replacement_technical_path_ready": bool(
            readiness.get("fallback_replacement_technical_path_ready", False)
        ),
        "fallback_replacement_ready": bool(readiness.get("fallback_replacement_ready", False)),
        "listening_review_package_required": bool(
            readiness.get("listening_review_package_required", False)
        ),
        "exported_candidate_count": _int(candidate.get("exported_candidate_count")),
        "rendered_audio_file_count": _int(audio.get("rendered_audio_file_count")),
        "wav_duration_min_seconds": _float(audio.get("wav_duration_min_seconds")),
        "wav_duration_max_seconds": _float(audio.get("wav_duration_max_seconds")),
        "human_review_required_now": bool(readiness.get("human_review_required_now", True)),
        "human_audio_preference_claimed": bool(readiness.get("human_audio_preference_claimed", True)),
        "midi_to_solo_musical_quality_claimed": bool(
            readiness.get("midi_to_solo_musical_quality_claimed", True)
        ),
        "critical_user_input_required": bool(decision.get("critical_user_input_required", True)),
        "next_recommended_issue": str(report.get("next_recommended_issue") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    readiness = report["readiness"]
    decision = report["decision"]
    candidate = report["candidate_export"]
    audio = report["audio_render"]
    lines = [
        "# Stage B MIDI-to-Solo Model-Conditioned Input Path Replacement Consolidation",
        "",
        "## Summary",
        "",
        f"- boundary: `{report['boundary']}`",
        f"- next boundary: `{decision['next_boundary']}`",
        f"- replacement consolidated: `{_bool_token(readiness['model_conditioned_input_path_replacement_consolidated'])}`",
        f"- input to ranked MIDI completed: `{_bool_token(readiness['model_conditioned_input_to_ranked_midi_completed'])}`",
        f"- input to ranked WAV completed: `{_bool_token(readiness['model_conditioned_input_to_ranked_wav_completed'])}`",
        f"- fallback replacement technical path ready: `{_bool_token(readiness['fallback_replacement_technical_path_ready'])}`",
        f"- listening review package required: `{_bool_token(readiness['listening_review_package_required'])}`",
        "",
        "## Evidence",
        "",
        f"- exported candidate count: `{candidate['exported_candidate_count']}`",
        f"- rendered audio file count: `{audio['rendered_audio_file_count']}`",
        f"- WAV duration range: `{audio['wav_duration_min_seconds']:.3f}s - {audio['wav_duration_max_seconds']:.3f}s`",
        f"- best note / unique pitch / max simultaneous: `{candidate['best_note_count']} / {candidate['best_unique_pitch_count']} / {candidate['best_max_simultaneous_notes']}`",
        "",
        "## Claim Boundary",
        "",
        f"- human/audio preference claimed: `{_bool_token(readiness['human_audio_preference_claimed'])}`",
        f"- MIDI-to-solo musical quality claimed: `{_bool_token(readiness['midi_to_solo_musical_quality_claimed'])}`",
        f"- audio rendered quality claimed: `{_bool_token(readiness['audio_rendered_quality_claimed'])}`",
        f"- broad trained-model quality claimed: `{_bool_token(readiness['broad_trained_model_quality_claimed'])}`",
        "",
        "## Next",
        "",
        f"- `{report['next_recommended_issue']}`",
    ]
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Consolidate model-conditioned input-path replacement evidence")
    parser.add_argument("--candidate_export_report", type=str, required=True)
    parser.add_argument("--audio_render_report", type=str, required=True)
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/stage_b_midi_to_solo_model_conditioned_input_path_replacement_consolidation",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--issue_number", type=int, default=628)
    parser.add_argument("--expected_count", type=int, default=3)
    parser.add_argument("--expected_boundary", type=str, default="")
    parser.add_argument("--expected_next_boundary", type=str, default="")
    parser.add_argument("--require_technical_replacement", action="store_true")
    parser.add_argument("--require_listening_review_package", action="store_true")
    parser.add_argument("--require_no_quality_claim", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    report = build_replacement_consolidation_report(
        candidate_export_report=read_json(Path(args.candidate_export_report)),
        audio_render_report=read_json(Path(args.audio_render_report)),
        output_dir=output_dir,
        issue_number=int(args.issue_number),
        expected_count=int(args.expected_count),
    )
    summary = validate_replacement_consolidation_report(
        report,
        expected_boundary=str(args.expected_boundary or ""),
        expected_next_boundary=str(args.expected_next_boundary or ""),
        require_technical_replacement=bool(args.require_technical_replacement),
        require_listening_review_package=bool(args.require_listening_review_package),
        require_no_quality_claim=bool(args.require_no_quality_claim),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "stage_b_midi_to_solo_model_conditioned_input_path_replacement_consolidation.json", report)
    write_json(
        output_dir / "stage_b_midi_to_solo_model_conditioned_input_path_replacement_consolidation_validation_summary.json",
        summary,
    )
    markdown = markdown_report(report)
    write_text(output_dir / "stage_b_midi_to_solo_model_conditioned_input_path_replacement_consolidation.md", markdown)
    if args.doc_path:
        write_text(Path(args.doc_path), markdown)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
