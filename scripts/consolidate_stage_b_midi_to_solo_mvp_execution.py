"""Consolidate the Stage B MIDI-to-solo MVP execution path evidence."""

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


class StageBMidiToSoloMvpExecutionConsolidationError(ValueError):
    pass


BOUNDARY = "stage_b_midi_to_solo_mvp_execution_consolidation"
NEXT_BOUNDARY = "stage_b_midi_to_solo_model_direct_generation_repair"
SCHEMA_VERSION = "stage_b_midi_to_solo_mvp_execution_consolidation_v1"


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise StageBMidiToSoloMvpExecutionConsolidationError(f"report missing: {path}")
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


def _bool_token(value: Any) -> str:
    return "true" if bool(value) else "false"


def file_exists(path: str) -> bool:
    return bool(path and Path(path).exists())


def build_execution_consolidation_report(
    *,
    contract_report: dict[str, Any],
    context_report: dict[str, Any],
    resource_probe: dict[str, Any],
    generation_probe: dict[str, Any],
    audio_render: dict[str, Any],
    output_dir: Path,
    issue_number: int,
) -> dict[str, Any]:
    contract_output = _dict(contract_report.get("output_contract"))
    contract_gate = _dict(contract_report.get("objective_gate"))
    context_summary = _dict(context_report.get("summary"))
    resource_readiness = _dict(resource_probe.get("readiness"))
    generation_readiness = _dict(generation_probe.get("readiness"))
    generation_summary = _dict(generation_probe.get("summary"))
    audio_boundary = _dict(audio_render.get("audio_render_boundary"))
    decision = _dict(audio_render.get("decision"))
    top_candidates = _list(generation_probe.get("top_candidates"))
    rendered_audio_files = _list(audio_render.get("rendered_audio_files"))

    midi_paths = [str(_dict(item).get("export_midi_path") or "") for item in top_candidates]
    wav_paths = [str(_dict(_dict(item).get("wav_file")).get("path") or "") for item in rendered_audio_files]
    technical_path_completed = (
        str(contract_report.get("boundary") or "") == "stage_b_midi_to_solo_mvp_input_contract"
        and str(context_report.get("boundary") or "") == "stage_b_midi_to_solo_context_extraction_mvp"
        and str(resource_probe.get("boundary") or "") == "stage_b_midi_to_solo_training_resource_probe"
        and str(generation_probe.get("boundary") or "") == "stage_b_midi_to_solo_conditioned_generation_probe"
        and str(audio_boundary.get("boundary") or "") == "stage_b_midi_to_solo_candidate_audio_render_package"
        and bool(resource_readiness.get("midi_to_solo_training_resource_ready", False))
        and bool(generation_readiness.get("ranked_midi_candidates_exported", False))
        and bool(audio_boundary.get("technical_wav_validation", False))
        and all(file_exists(path) for path in midi_paths)
        and all(file_exists(path) for path in wav_paths)
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "issue_number": int(issue_number),
        "boundary": BOUNDARY,
        "source_boundaries": {
            "contract": str(contract_report.get("boundary") or ""),
            "context": str(context_report.get("boundary") or ""),
            "resource": str(resource_probe.get("boundary") or ""),
            "generation": str(generation_probe.get("boundary") or ""),
            "audio": str(audio_boundary.get("boundary") or ""),
        },
        "contract": {
            "target_date": str(contract_report.get("target_date") or ""),
            "candidate_count": _int(contract_output.get("candidate_count")),
            "export_top_midi_count": _int(contract_output.get("export_top_midi_count")),
            "target_solo_bars": _int(contract_output.get("target_solo_bars")),
            "min_note_count": _int(contract_gate.get("min_note_count")),
            "min_unique_pitch_count": _int(contract_gate.get("min_unique_pitch_count")),
            "max_simultaneous_notes": _int(contract_gate.get("max_simultaneous_notes")),
        },
        "execution_path": {
            "context_event_count": _int(context_summary.get("context_event_count")),
            "resource_ready": bool(resource_readiness.get("midi_to_solo_training_resource_ready", False)),
            "generation_source": str(_dict(generation_probe.get("generation_config")).get("generation_source") or ""),
            "candidate_count": _int(generation_summary.get("candidate_count")),
            "exported_candidate_count": _int(generation_summary.get("exported_candidate_count")),
            "exported_qualified_candidate_count": _int(
                generation_summary.get("exported_qualified_candidate_count")
            ),
            "rendered_audio_file_count": _int(audio_boundary.get("rendered_audio_file_count")),
            "technical_wav_validation": bool(audio_boundary.get("technical_wav_validation", False)),
            "midi_paths": midi_paths,
            "wav_paths": wav_paths,
        },
        "readiness": {
            "boundary": BOUNDARY,
            "technical_execution_path_completed": bool(technical_path_completed),
            "input_to_ranked_midi_completed": bool(
                technical_path_completed and _int(generation_summary.get("exported_candidate_count")) >= 3
            ),
            "input_to_rendered_audio_completed": bool(
                technical_path_completed and _int(audio_boundary.get("rendered_audio_file_count")) >= 3
            ),
            "midi_to_solo_technical_mvp_completed": bool(technical_path_completed),
            "midi_to_solo_musical_quality_claimed": False,
            "model_checkpoint_direct_generation_quality_claimed": False,
            "human_audio_preference_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
        },
        "decision": {
            "current_boundary": BOUNDARY,
            "next_boundary": NEXT_BOUNDARY,
            "auto_progress_allowed": True,
            "critical_user_input_required": False,
            "reason": (
                "technical input-to-MIDI-to-WAV path is complete; next repair target is direct "
                "model-conditioned generation rather than quality claim"
            ),
            "source_audio_next_boundary": str(decision.get("next_boundary") or ""),
        },
        "not_proven": [
            "musical_quality",
            "human_audio_preference",
            "model_checkpoint_direct_8bar_generation_quality",
            "brad_style_adaptation",
            "production_ready_improviser",
        ],
        "next_recommended_issue": "Stage B MIDI-to-solo model-direct generation repair",
    }


def validate_execution_consolidation_report(
    report: dict[str, Any],
    *,
    expected_boundary: str | None,
    expected_next_boundary: str | None,
    require_technical_mvp: bool,
    require_no_quality_claim: bool,
    min_exported_candidates: int,
    min_rendered_wav_files: int,
) -> dict[str, Any]:
    boundary = str(report.get("boundary") or "")
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    path = _dict(report.get("execution_path"))
    if expected_boundary and boundary != expected_boundary:
        raise StageBMidiToSoloMvpExecutionConsolidationError(f"expected boundary {expected_boundary}, got {boundary}")
    if expected_next_boundary and str(decision.get("next_boundary") or "") != expected_next_boundary:
        raise StageBMidiToSoloMvpExecutionConsolidationError("unexpected next boundary")
    if require_technical_mvp and not bool(readiness.get("midi_to_solo_technical_mvp_completed", False)):
        raise StageBMidiToSoloMvpExecutionConsolidationError("technical MVP path must be complete")
    if _int(path.get("exported_candidate_count")) < int(min_exported_candidates):
        raise StageBMidiToSoloMvpExecutionConsolidationError("exported candidate count below threshold")
    if _int(path.get("rendered_audio_file_count")) < int(min_rendered_wav_files):
        raise StageBMidiToSoloMvpExecutionConsolidationError("rendered wav count below threshold")
    if not bool(path.get("technical_wav_validation", False)):
        raise StageBMidiToSoloMvpExecutionConsolidationError("technical wav validation required")
    for item in _list(path.get("midi_paths")) + _list(path.get("wav_paths")):
        if not file_exists(str(item)):
            raise StageBMidiToSoloMvpExecutionConsolidationError(f"artifact missing: {item}")
    if bool(decision.get("critical_user_input_required", True)):
        raise StageBMidiToSoloMvpExecutionConsolidationError("critical user input should not be required")
    if require_no_quality_claim:
        blocked = [
            "midi_to_solo_musical_quality_claimed",
            "model_checkpoint_direct_generation_quality_claimed",
            "human_audio_preference_claimed",
            "broad_trained_model_quality_claimed",
            "brad_style_adaptation_claimed",
        ]
        claimed = [name for name in blocked if bool(readiness.get(name, True))]
        if claimed:
            raise StageBMidiToSoloMvpExecutionConsolidationError(f"unexpected quality claim: {claimed}")
    return {
        "boundary": boundary,
        "next_boundary": str(decision.get("next_boundary") or ""),
        "technical_execution_path_completed": bool(readiness.get("technical_execution_path_completed", False)),
        "midi_to_solo_technical_mvp_completed": bool(
            readiness.get("midi_to_solo_technical_mvp_completed", False)
        ),
        "input_to_ranked_midi_completed": bool(readiness.get("input_to_ranked_midi_completed", False)),
        "input_to_rendered_audio_completed": bool(readiness.get("input_to_rendered_audio_completed", False)),
        "generation_source": str(path.get("generation_source") or ""),
        "exported_candidate_count": _int(path.get("exported_candidate_count")),
        "rendered_audio_file_count": _int(path.get("rendered_audio_file_count")),
        "technical_wav_validation": bool(path.get("technical_wav_validation", False)),
        "midi_to_solo_musical_quality_claimed": bool(
            readiness.get("midi_to_solo_musical_quality_claimed", True)
        ),
        "model_checkpoint_direct_generation_quality_claimed": bool(
            readiness.get("model_checkpoint_direct_generation_quality_claimed", True)
        ),
        "human_audio_preference_claimed": bool(readiness.get("human_audio_preference_claimed", True)),
        "critical_user_input_required": bool(decision.get("critical_user_input_required", True)),
        "next_recommended_issue": str(report.get("next_recommended_issue") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    readiness = report["readiness"]
    decision = report["decision"]
    path = report["execution_path"]
    lines = [
        "# Stage B MIDI-to-Solo MVP Execution Consolidation",
        "",
        "## Summary",
        "",
        f"- boundary: `{report['boundary']}`",
        f"- next boundary: `{decision['next_boundary']}`",
        f"- technical execution path completed: `{_bool_token(readiness['technical_execution_path_completed'])}`",
        f"- MIDI-to-solo technical MVP completed: `{_bool_token(readiness['midi_to_solo_technical_mvp_completed'])}`",
        f"- musical quality claimed: `{_bool_token(readiness['midi_to_solo_musical_quality_claimed'])}`",
        f"- human/audio preference claimed: `{_bool_token(readiness['human_audio_preference_claimed'])}`",
        "",
        "## Execution Path",
        "",
        f"- context event count: `{path['context_event_count']}`",
        f"- generation source: `{path['generation_source']}`",
        f"- exported candidate count: `{path['exported_candidate_count']}`",
        f"- rendered audio file count: `{path['rendered_audio_file_count']}`",
        f"- technical WAV validation: `{_bool_token(path['technical_wav_validation'])}`",
        "",
        "## MIDI Paths",
        "",
    ]
    for item in path["midi_paths"]:
        lines.append(f"- `{item}`")
    lines.extend(["", "## WAV Paths", ""])
    for item in path["wav_paths"]:
        lines.append(f"- `{item}`")
    lines.extend(["", "## Not Proven", ""])
    for item in report.get("not_proven", []):
        lines.append(f"- `{item}`")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Consolidate MIDI-to-solo MVP execution evidence")
    parser.add_argument("--contract_report", type=str, required=True)
    parser.add_argument("--context_report", type=str, required=True)
    parser.add_argument("--resource_probe", type=str, required=True)
    parser.add_argument("--generation_probe", type=str, required=True)
    parser.add_argument("--audio_render", type=str, required=True)
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/stage_b_midi_to_solo_mvp_execution_consolidation",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--issue_number", type=int, default=491)
    parser.add_argument("--expected_boundary", type=str, default="")
    parser.add_argument("--expected_next_boundary", type=str, default="")
    parser.add_argument("--require_technical_mvp", action="store_true")
    parser.add_argument("--require_no_quality_claim", action="store_true")
    parser.add_argument("--min_exported_candidates", type=int, default=3)
    parser.add_argument("--min_rendered_wav_files", type=int, default=3)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    report = build_execution_consolidation_report(
        contract_report=read_json(Path(args.contract_report)),
        context_report=read_json(Path(args.context_report)),
        resource_probe=read_json(Path(args.resource_probe)),
        generation_probe=read_json(Path(args.generation_probe)),
        audio_render=read_json(Path(args.audio_render)),
        output_dir=output_dir,
        issue_number=int(args.issue_number),
    )
    summary = validate_execution_consolidation_report(
        report,
        expected_boundary=str(args.expected_boundary or ""),
        expected_next_boundary=str(args.expected_next_boundary or ""),
        require_technical_mvp=bool(args.require_technical_mvp),
        require_no_quality_claim=bool(args.require_no_quality_claim),
        min_exported_candidates=int(args.min_exported_candidates),
        min_rendered_wav_files=int(args.min_rendered_wav_files),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "stage_b_midi_to_solo_mvp_execution_consolidation.json", report)
    write_json(output_dir / "stage_b_midi_to_solo_mvp_execution_consolidation_validation_summary.json", summary)
    markdown = markdown_report(report)
    write_text(output_dir / "stage_b_midi_to_solo_mvp_execution_consolidation.md", markdown)
    if args.doc_path:
        write_text(Path(args.doc_path), markdown)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
