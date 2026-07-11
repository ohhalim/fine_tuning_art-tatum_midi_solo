"""Consolidate model-direct MIDI objective evidence and WAV render evidence."""

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


class StageBMidiToSoloModelDirectAudioEvidenceConsolidationError(ValueError):
    pass


OBJECTIVE_BOUNDARY = "stage_b_midi_to_solo_model_direct_monophonic_overlap_repair"
AUDIO_BOUNDARY = "stage_b_midi_to_solo_model_direct_audio_render_package"
BOUNDARY = "stage_b_midi_to_solo_model_direct_audio_evidence_consolidation"
NEXT_BOUNDARY = "stage_b_midi_to_solo_model_direct_phrase_quality_diagnostics"
SCHEMA_VERSION = "stage_b_midi_to_solo_model_direct_audio_evidence_consolidation_v1"


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


def file_exists(path: str) -> bool:
    return bool(path and Path(path).exists())


def build_model_direct_audio_evidence_report(
    *,
    objective_report: dict[str, Any],
    audio_render_report: dict[str, Any],
    output_dir: Path,
    issue_number: int,
) -> dict[str, Any]:
    objective_readiness = _dict(objective_report.get("readiness"))
    objective_generation = _dict(objective_report.get("repaired_generation_summary"))
    objective_result = _dict(objective_report.get("repair_result"))
    audio_boundary = _dict(audio_render_report.get("audio_render_boundary"))
    audio_decision = _dict(audio_render_report.get("decision"))
    rendered_files = _list(audio_render_report.get("rendered_audio_files"))
    midi_paths = [str(path) for path in _list(objective_generation.get("midi_paths"))]
    wav_paths = [str(_dict(_dict(item).get("wav_file")).get("path") or "") for item in rendered_files]
    durations = [_float(_dict(_dict(item).get("wav_file")).get("duration_seconds")) for item in rendered_files]
    sample_rates = sorted(
        {
            _int(_dict(_dict(item).get("wav_file")).get("sample_rate"))
            for item in rendered_files
            if _int(_dict(_dict(item).get("wav_file")).get("sample_rate")) > 0
        }
    )
    source_valid = (
        str(objective_report.get("boundary") or objective_readiness.get("boundary") or "") == OBJECTIVE_BOUNDARY
        and str(audio_boundary.get("boundary") or "") == AUDIO_BOUNDARY
        and str(audio_render_report.get("source_boundary") or "") == OBJECTIVE_BOUNDARY
        and bool(objective_readiness.get("direct_generation_review_gate_passed", False))
        and bool(audio_boundary.get("technical_wav_validation", False))
        and _int(objective_generation.get("valid_sample_count")) >= 3
        and _int(objective_generation.get("strict_valid_sample_count")) >= 3
        and _int(audio_boundary.get("rendered_audio_file_count")) >= 3
        and all(file_exists(path) for path in midi_paths[:3])
        and all(file_exists(path) for path in wav_paths[:3])
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "issue_number": int(issue_number),
        "boundary": BOUNDARY,
        "source_boundaries": {
            "objective": str(objective_report.get("boundary") or objective_readiness.get("boundary") or ""),
            "audio": str(audio_boundary.get("boundary") or ""),
        },
        "objective_evidence": {
            "sample_count": _int(objective_generation.get("sample_count")),
            "valid_sample_count": _int(objective_generation.get("valid_sample_count")),
            "strict_valid_sample_count": _int(objective_generation.get("strict_valid_sample_count")),
            "avg_postprocess_removal_ratio": _float(
                objective_generation.get("avg_postprocess_removal_ratio")
            ),
            "collapse_warning_sample_rate": _float(objective_generation.get("collapse_warning_sample_rate")),
            "previous_valid_sample_count": _int(objective_result.get("previous_valid_sample_count")),
            "previous_strict_valid_sample_count": _int(
                objective_result.get("previous_strict_valid_sample_count")
            ),
            "midi_paths": midi_paths,
        },
        "audio_evidence": {
            "rendered_audio_file_count": _int(audio_boundary.get("rendered_audio_file_count")),
            "technical_wav_validation": bool(audio_boundary.get("technical_wav_validation", False)),
            "sample_rates": sample_rates,
            "min_duration_seconds": min(durations) if durations else 0.0,
            "max_duration_seconds": max(durations) if durations else 0.0,
            "wav_paths": wav_paths,
        },
        "readiness": {
            "boundary": BOUNDARY,
            "model_direct_objective_gate_passed": bool(
                objective_readiness.get("direct_generation_review_gate_passed", False)
            ),
            "model_direct_audio_render_completed": bool(
                audio_boundary.get("render_attempted", False)
                and _int(audio_boundary.get("rendered_audio_file_count")) >= 3
            ),
            "model_direct_midi_to_wav_technical_path_completed": bool(source_valid),
            "model_direct_generation_quality_claimed": False,
            "midi_to_solo_musical_quality_claimed": False,
            "audio_rendered_quality_claimed": False,
            "human_audio_preference_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
        },
        "decision": {
            "current_boundary": BOUNDARY,
            "next_boundary": NEXT_BOUNDARY,
            "auto_progress_allowed": True,
            "critical_user_input_required": False,
            "reason": "model-direct MIDI objective gate and WAV technical render evidence consolidated",
            "source_audio_next_boundary": str(audio_decision.get("next_boundary") or ""),
        },
        "not_proven": [
            "model_direct_generation_quality",
            "midi_to_solo_musical_quality",
            "audio_rendered_quality",
            "human_audio_preference",
            "broad_trained_model_quality",
            "brad_style_adaptation",
        ],
        "next_recommended_issue": "Stage B MIDI-to-solo model-direct phrase quality diagnostics",
    }


def validate_model_direct_audio_evidence_report(
    report: dict[str, Any],
    *,
    expected_boundary: str | None,
    expected_next_boundary: str | None,
    require_technical_path: bool,
    require_no_quality_claim: bool,
    min_midi_count: int,
    min_wav_count: int,
) -> dict[str, Any]:
    boundary = str(report.get("boundary") or "")
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    objective = _dict(report.get("objective_evidence"))
    audio = _dict(report.get("audio_evidence"))
    if expected_boundary and boundary != expected_boundary:
        raise StageBMidiToSoloModelDirectAudioEvidenceConsolidationError(
            f"expected boundary {expected_boundary}, got {boundary}"
        )
    if expected_next_boundary and str(decision.get("next_boundary") or "") != expected_next_boundary:
        raise StageBMidiToSoloModelDirectAudioEvidenceConsolidationError("unexpected next boundary")
    if require_technical_path and not bool(
        readiness.get("model_direct_midi_to_wav_technical_path_completed", False)
    ):
        raise StageBMidiToSoloModelDirectAudioEvidenceConsolidationError("technical path must be complete")
    if _int(objective.get("strict_valid_sample_count")) < int(min_midi_count):
        raise StageBMidiToSoloModelDirectAudioEvidenceConsolidationError("strict-valid MIDI count below threshold")
    if _int(audio.get("rendered_audio_file_count")) < int(min_wav_count):
        raise StageBMidiToSoloModelDirectAudioEvidenceConsolidationError("rendered WAV count below threshold")
    if not bool(audio.get("technical_wav_validation", False)):
        raise StageBMidiToSoloModelDirectAudioEvidenceConsolidationError("technical WAV validation required")
    for path in _list(objective.get("midi_paths"))[: int(min_midi_count)] + _list(audio.get("wav_paths"))[
        : int(min_wav_count)
    ]:
        if not file_exists(str(path)):
            raise StageBMidiToSoloModelDirectAudioEvidenceConsolidationError(f"artifact missing: {path}")
    if bool(decision.get("critical_user_input_required", True)):
        raise StageBMidiToSoloModelDirectAudioEvidenceConsolidationError("critical user input should not be required")
    if require_no_quality_claim:
        blocked = [
            "model_direct_generation_quality_claimed",
            "midi_to_solo_musical_quality_claimed",
            "audio_rendered_quality_claimed",
            "human_audio_preference_claimed",
            "broad_trained_model_quality_claimed",
            "brad_style_adaptation_claimed",
        ]
        claimed = [name for name in blocked if bool(readiness.get(name, True))]
        if claimed:
            raise StageBMidiToSoloModelDirectAudioEvidenceConsolidationError(
                f"unexpected quality claim: {claimed}"
            )
    return {
        "boundary": boundary,
        "next_boundary": str(decision.get("next_boundary") or ""),
        "model_direct_objective_gate_passed": bool(
            readiness.get("model_direct_objective_gate_passed", False)
        ),
        "model_direct_audio_render_completed": bool(
            readiness.get("model_direct_audio_render_completed", False)
        ),
        "model_direct_midi_to_wav_technical_path_completed": bool(
            readiness.get("model_direct_midi_to_wav_technical_path_completed", False)
        ),
        "strict_valid_sample_count": _int(objective.get("strict_valid_sample_count")),
        "rendered_audio_file_count": _int(audio.get("rendered_audio_file_count")),
        "sample_rates": _list(audio.get("sample_rates")),
        "min_duration_seconds": _float(audio.get("min_duration_seconds")),
        "max_duration_seconds": _float(audio.get("max_duration_seconds")),
        "model_direct_generation_quality_claimed": bool(
            readiness.get("model_direct_generation_quality_claimed", True)
        ),
        "human_audio_preference_claimed": bool(readiness.get("human_audio_preference_claimed", True)),
        "critical_user_input_required": bool(decision.get("critical_user_input_required", True)),
        "next_recommended_issue": str(report.get("next_recommended_issue") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    readiness = report["readiness"]
    decision = report["decision"]
    objective = report["objective_evidence"]
    audio = report["audio_evidence"]
    lines = [
        "# Stage B MIDI-to-Solo Model-Direct Audio Evidence Consolidation",
        "",
        "## Summary",
        "",
        f"- boundary: `{report['boundary']}`",
        f"- next boundary: `{decision['next_boundary']}`",
        f"- model-direct objective gate passed: `{_bool_token(readiness['model_direct_objective_gate_passed'])}`",
        f"- model-direct audio render completed: `{_bool_token(readiness['model_direct_audio_render_completed'])}`",
        f"- model-direct MIDI-to-WAV technical path completed: `{_bool_token(readiness['model_direct_midi_to_wav_technical_path_completed'])}`",
        f"- model-direct generation quality claimed: `{_bool_token(readiness['model_direct_generation_quality_claimed'])}`",
        f"- human/audio preference claimed: `{_bool_token(readiness['human_audio_preference_claimed'])}`",
        "",
        "## Evidence",
        "",
        f"- strict valid sample count: `{objective['strict_valid_sample_count']}`",
        f"- avg postprocess removal ratio: `{objective['avg_postprocess_removal_ratio']}`",
        f"- collapse warning sample rate: `{objective['collapse_warning_sample_rate']}`",
        f"- rendered audio file count: `{audio['rendered_audio_file_count']}`",
        f"- sample rates: `{audio['sample_rates']}`",
        f"- duration range: `{float(audio['min_duration_seconds']):.3f}s` - `{float(audio['max_duration_seconds']):.3f}s`",
        "",
        "## Not Proven",
        "",
    ]
    for item in report.get("not_proven", []):
        lines.append(f"- `{item}`")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Consolidate model-direct MIDI and audio evidence")
    parser.add_argument("--objective_report", type=str, required=True)
    parser.add_argument("--audio_render_report", type=str, required=True)
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/stage_b_midi_to_solo_model_direct_audio_evidence_consolidation",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--issue_number", type=int, default=503)
    parser.add_argument("--expected_boundary", type=str, default="")
    parser.add_argument("--expected_next_boundary", type=str, default="")
    parser.add_argument("--min_midi_count", type=int, default=3)
    parser.add_argument("--min_wav_count", type=int, default=3)
    parser.add_argument("--require_technical_path", action="store_true")
    parser.add_argument("--require_no_quality_claim", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    report = build_model_direct_audio_evidence_report(
        objective_report=read_json(Path(args.objective_report)),
        audio_render_report=read_json(Path(args.audio_render_report)),
        output_dir=output_dir,
        issue_number=int(args.issue_number),
    )
    summary = validate_model_direct_audio_evidence_report(
        report,
        expected_boundary=str(args.expected_boundary or ""),
        expected_next_boundary=str(args.expected_next_boundary or ""),
        require_technical_path=bool(args.require_technical_path),
        require_no_quality_claim=bool(args.require_no_quality_claim),
        min_midi_count=int(args.min_midi_count),
        min_wav_count=int(args.min_wav_count),
    )
    write_json(output_dir / "stage_b_midi_to_solo_model_direct_audio_evidence_consolidation.json", report)
    write_json(
        output_dir / "stage_b_midi_to_solo_model_direct_audio_evidence_consolidation_validation_summary.json",
        summary,
    )
    markdown = markdown_report(report)
    write_text(output_dir / "stage_b_midi_to_solo_model_direct_audio_evidence_consolidation.md", markdown)
    if args.doc_path:
        write_text(Path(args.doc_path), markdown)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
