"""Probe the Stage B MIDI-to-solo model-conditioned input path boundary."""

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
from scripts.decide_stage_b_midi_to_solo_model_conditioned_input_path_quality_alignment import (  # noqa: E402
    BOUNDARY as ALIGNMENT_BOUNDARY,
    NEXT_BOUNDARY as ALIGNMENT_NEXT_BOUNDARY,
    SELECTED_PROBE_TARGET,
)


class StageBMidiToSoloModelConditionedInputPathProbeError(ValueError):
    pass


BOUNDARY = "stage_b_midi_to_solo_model_conditioned_input_path_probe"
NEXT_BOUNDARY = "stage_b_midi_to_solo_model_conditioned_input_path_candidate_export"
SCHEMA_VERSION = "stage_b_midi_to_solo_model_conditioned_input_path_probe_v1"

FALLBACK_GENERATION_BOUNDARY = "stage_b_midi_to_solo_conditioned_generation_probe"
FALLBACK_AUDIO_BOUNDARY = "stage_b_midi_to_solo_candidate_audio_render_package"
MODEL_DIRECT_REPAIR_BOUNDARY = "stage_b_midi_to_solo_model_direct_monophonic_overlap_repair"
MODEL_DIRECT_AUDIO_BOUNDARY = "stage_b_midi_to_solo_model_direct_audio_render_package"

FALLBACK_SOURCE = "context_conditioned_fallback"
MODEL_CONDITIONED_SOURCE = "model_checkpoint_direct_constrained"

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


def _claim_names(container: dict[str, Any]) -> list[str]:
    return [name for name in QUALITY_CLAIM_KEYS if bool(container.get(name, False))]


def _require_no_quality_claim(
    container: dict[str, Any],
    *,
    label: str,
) -> None:
    claimed = _claim_names(container)
    if claimed:
        raise StageBMidiToSoloModelConditionedInputPathProbeError(
            f"unexpected quality claim in {label}: {claimed}"
        )


def validate_alignment_report(report: dict[str, Any]) -> dict[str, Any]:
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    alignment = _dict(report.get("alignment_decision"))
    if str(report.get("boundary") or readiness.get("boundary") or "") != ALIGNMENT_BOUNDARY:
        raise StageBMidiToSoloModelConditionedInputPathProbeError(
            "model-conditioned input path alignment boundary required"
        )
    if str(decision.get("next_boundary") or "") != ALIGNMENT_NEXT_BOUNDARY:
        raise StageBMidiToSoloModelConditionedInputPathProbeError(
            "alignment report must route to input path probe"
        )
    if str(readiness.get("selected_probe_target") or "") != SELECTED_PROBE_TARGET:
        raise StageBMidiToSoloModelConditionedInputPathProbeError("selected probe target mismatch")
    if not bool(readiness.get("fallback_replacement_probe_required", False)):
        raise StageBMidiToSoloModelConditionedInputPathProbeError(
            "fallback replacement probe should be required"
        )
    if bool(readiness.get("model_conditioned_input_path_aligned", True)):
        raise StageBMidiToSoloModelConditionedInputPathProbeError(
            "input path should not already be aligned"
        )
    if bool(decision.get("critical_user_input_required", True)):
        raise StageBMidiToSoloModelConditionedInputPathProbeError(
            "critical user input should not be required"
        )
    _require_no_quality_claim(readiness, label="alignment readiness")
    return {
        "boundary": ALIGNMENT_BOUNDARY,
        "selected_probe_target": str(readiness.get("selected_probe_target") or ""),
        "fallback_replacement_probe_required": True,
        "alignment_decision_probe_required": bool(
            alignment.get("fallback_replacement_probe_required", False)
        ),
        "human_review_required_now": bool(readiness.get("human_review_required_now", False)),
    }


def validate_fallback_generation(report: dict[str, Any], *, min_count: int) -> dict[str, Any]:
    readiness = _dict(report.get("readiness"))
    summary = _dict(report.get("summary"))
    config = _dict(report.get("generation_config"))
    top_candidates = [_dict(item) for item in _list(report.get("top_candidates"))]
    if str(report.get("boundary") or readiness.get("boundary") or "") != FALLBACK_GENERATION_BOUNDARY:
        raise StageBMidiToSoloModelConditionedInputPathProbeError("fallback generation boundary required")
    if str(config.get("generation_source") or "") != FALLBACK_SOURCE:
        raise StageBMidiToSoloModelConditionedInputPathProbeError("fallback source mismatch")
    if bool(config.get("model_checkpoint_generation_used", True)):
        raise StageBMidiToSoloModelConditionedInputPathProbeError(
            "fallback generation should not be marked as checkpoint generation"
        )
    if not bool(readiness.get("conditioned_generation_probe_completed", False)):
        raise StageBMidiToSoloModelConditionedInputPathProbeError(
            "fallback generation completion required"
        )
    if not bool(readiness.get("ranked_midi_candidates_exported", False)):
        raise StageBMidiToSoloModelConditionedInputPathProbeError(
            "fallback ranked MIDI export required"
        )
    if _int(summary.get("exported_candidate_count")) < int(min_count):
        raise StageBMidiToSoloModelConditionedInputPathProbeError(
            "fallback exported candidate count below minimum"
        )
    if _int(summary.get("exported_qualified_candidate_count")) < int(min_count):
        raise StageBMidiToSoloModelConditionedInputPathProbeError(
            "fallback exported qualified candidate count below minimum"
        )
    midi_paths = [str(item.get("export_midi_path") or "") for item in top_candidates[: int(min_count)]]
    if len(midi_paths) < int(min_count) or not all(_path_exists(path) for path in midi_paths):
        raise StageBMidiToSoloModelConditionedInputPathProbeError(
            "fallback exported MIDI artifacts required"
        )
    _require_no_quality_claim(readiness, label="fallback generation readiness")
    input_context = _dict(report.get("input_context"))
    return {
        "boundary": FALLBACK_GENERATION_BOUNDARY,
        "generation_source": FALLBACK_SOURCE,
        "model_checkpoint_generation_used": False,
        "candidate_count": _int(summary.get("candidate_count")),
        "qualified_candidate_count": _int(summary.get("qualified_candidate_count")),
        "exported_candidate_count": _int(summary.get("exported_candidate_count")),
        "exported_qualified_candidate_count": _int(summary.get("exported_qualified_candidate_count")),
        "best_note_count": _int(summary.get("best_note_count")),
        "best_unique_pitch_count": _int(summary.get("best_unique_pitch_count")),
        "best_max_simultaneous_notes": _int(summary.get("best_max_simultaneous_notes")),
        "midi_paths": midi_paths,
        "context_bars": _int(input_context.get("bars")),
        "context_bpm": _int(input_context.get("bpm")),
        "chord_progression": list(_list(input_context.get("chord_progression"))),
    }


def validate_audio_report(
    report: dict[str, Any],
    *,
    expected_boundary: str,
    expected_source_boundary: str,
    min_count: int,
    label: str,
) -> dict[str, Any]:
    boundary = _dict(report.get("audio_render_boundary"))
    files = [_dict(item) for item in _list(report.get("rendered_audio_files"))]
    if str(boundary.get("boundary") or "") != expected_boundary:
        raise StageBMidiToSoloModelConditionedInputPathProbeError(
            f"{label} audio render boundary required"
        )
    if str(report.get("source_boundary") or "") != expected_source_boundary:
        raise StageBMidiToSoloModelConditionedInputPathProbeError(
            f"{label} audio source boundary mismatch"
        )
    if _int(boundary.get("rendered_audio_file_count")) < int(min_count) or len(files) < int(min_count):
        raise StageBMidiToSoloModelConditionedInputPathProbeError(
            f"{label} rendered WAV count below minimum"
        )
    if not bool(boundary.get("technical_wav_validation", False)):
        raise StageBMidiToSoloModelConditionedInputPathProbeError(
            f"{label} technical WAV validation required"
        )
    wav_paths: list[str] = []
    durations: list[float] = []
    for item in files[: int(min_count)]:
        wav = _dict(item.get("wav_file"))
        wav_path = str(wav.get("path") or "")
        if not bool(wav.get("exists", False)) or not _path_exists(wav_path):
            raise StageBMidiToSoloModelConditionedInputPathProbeError(f"{label} WAV artifact required")
        if _int(wav.get("frame_count")) <= 0 or _int(wav.get("size_bytes")) <= 44:
            raise StageBMidiToSoloModelConditionedInputPathProbeError(f"{label} WAV metadata invalid")
        wav_paths.append(wav_path)
        durations.append(_float(wav.get("duration_seconds")))
    _require_no_quality_claim(boundary, label=f"{label} audio boundary")
    return {
        "boundary": expected_boundary,
        "source_boundary": expected_source_boundary,
        "rendered_audio_file_count": len(files),
        "technical_wav_validation": True,
        "duration_min_seconds": min(durations) if durations else 0.0,
        "duration_max_seconds": max(durations) if durations else 0.0,
        "wav_paths": wav_paths,
    }


def validate_model_direct_repair(report: dict[str, Any], *, min_count: int) -> dict[str, Any]:
    readiness = _dict(report.get("readiness"))
    repair_config = _dict(report.get("repair_config"))
    generation = _dict(report.get("repaired_generation_summary"))
    context = _dict(report.get("context_summary"))
    if str(report.get("boundary") or readiness.get("boundary") or "") != MODEL_DIRECT_REPAIR_BOUNDARY:
        raise StageBMidiToSoloModelConditionedInputPathProbeError(
            "model-direct repair boundary required"
        )
    if str(repair_config.get("generation_source") or "") != MODEL_CONDITIONED_SOURCE:
        raise StageBMidiToSoloModelConditionedInputPathProbeError(
            "model-conditioned source mismatch"
        )
    if not bool(readiness.get("monophonic_overlap_repair_completed", False)):
        raise StageBMidiToSoloModelConditionedInputPathProbeError(
            "model-direct overlap repair completion required"
        )
    if not bool(readiness.get("direct_generated_midi_written", False)):
        raise StageBMidiToSoloModelConditionedInputPathProbeError(
            "model-direct MIDI output required"
        )
    if not bool(readiness.get("direct_generation_review_gate_passed", False)):
        raise StageBMidiToSoloModelConditionedInputPathProbeError(
            "model-direct review gate should pass before input-path probe"
        )
    if _int(generation.get("strict_valid_sample_count")) < int(min_count):
        raise StageBMidiToSoloModelConditionedInputPathProbeError(
            "model-direct strict-valid sample count below minimum"
        )
    if _int(generation.get("min_postprocess_note_count")) < 24:
        raise StageBMidiToSoloModelConditionedInputPathProbeError(
            "model-direct postprocess note count below contract minimum"
        )
    if not bool(generation.get("all_midi_paths_exist", False)):
        raise StageBMidiToSoloModelConditionedInputPathProbeError(
            "model-direct MIDI paths must exist"
        )
    midi_paths = [str(path) for path in _list(generation.get("midi_paths"))[: int(min_count)]]
    if len(midi_paths) < int(min_count) or not all(_path_exists(path) for path in midi_paths):
        raise StageBMidiToSoloModelConditionedInputPathProbeError(
            "model-direct MIDI artifacts required"
        )
    _require_no_quality_claim(readiness, label="model-direct repair readiness")
    return {
        "boundary": MODEL_DIRECT_REPAIR_BOUNDARY,
        "generation_source": MODEL_CONDITIONED_SOURCE,
        "review_gate_passed": True,
        "sample_count": _int(generation.get("sample_count")),
        "strict_valid_sample_count": _int(generation.get("strict_valid_sample_count")),
        "grammar_gate_sample_count": _int(generation.get("grammar_gate_sample_count")),
        "min_postprocess_note_count": _int(generation.get("min_postprocess_note_count")),
        "max_postprocess_note_count": _int(generation.get("max_postprocess_note_count")),
        "avg_postprocess_removal_ratio": _float(generation.get("avg_postprocess_removal_ratio")),
        "avg_onset_coverage_ratio": _float(generation.get("avg_onset_coverage_ratio")),
        "avg_sustained_coverage_ratio": _float(generation.get("avg_sustained_coverage_ratio")),
        "midi_paths": midi_paths,
        "target_bars": _int(repair_config.get("target_bars")),
        "note_groups_per_bar": _int(repair_config.get("note_groups_per_bar")),
        "cap_duration_to_next_position": bool(repair_config.get("cap_duration_to_next_position", False)),
        "context_bars": _int(context.get("context_bars")),
        "context_bpm": _int(context.get("bpm")),
        "chord_progression": list(_list(context.get("chord_progression"))),
    }


def model_direct_ranked_export_contract_matched(report: dict[str, Any], *, min_count: int) -> bool:
    readiness = _dict(report.get("readiness"))
    top_candidates = [_dict(item) for item in _list(report.get("top_candidates"))]
    if not bool(readiness.get("ranked_midi_candidates_exported", False)):
        return False
    if len(top_candidates) < int(min_count):
        return False
    midi_paths = [str(item.get("export_midi_path") or "") for item in top_candidates[: int(min_count)]]
    return bool(midi_paths and all(_path_exists(path) for path in midi_paths))


def build_probe_report(
    *,
    alignment_report: dict[str, Any],
    fallback_generation_report: dict[str, Any],
    fallback_audio_report: dict[str, Any],
    model_direct_repair_report: dict[str, Any],
    model_direct_audio_report: dict[str, Any],
    output_dir: Path,
    issue_number: int,
    min_count: int,
) -> dict[str, Any]:
    alignment = validate_alignment_report(alignment_report)
    fallback_generation = validate_fallback_generation(
        fallback_generation_report,
        min_count=min_count,
    )
    fallback_audio = validate_audio_report(
        fallback_audio_report,
        expected_boundary=FALLBACK_AUDIO_BOUNDARY,
        expected_source_boundary=FALLBACK_GENERATION_BOUNDARY,
        min_count=min_count,
        label="fallback",
    )
    model_direct_repair = validate_model_direct_repair(
        model_direct_repair_report,
        min_count=min_count,
    )
    model_direct_audio = validate_audio_report(
        model_direct_audio_report,
        expected_boundary=MODEL_DIRECT_AUDIO_BOUNDARY,
        expected_source_boundary=MODEL_DIRECT_REPAIR_BOUNDARY,
        min_count=min_count,
        label="model-direct",
    )
    same_input_context = bool(
        fallback_generation["context_bars"] == model_direct_repair["context_bars"]
        and fallback_generation["context_bpm"] == model_direct_repair["context_bpm"]
        and fallback_generation["chord_progression"] == model_direct_repair["chord_progression"]
    )
    ranked_export_contract_matched = model_direct_ranked_export_contract_matched(
        model_direct_repair_report,
        min_count=min_count,
    )
    model_conditioned_candidate_source_available = bool(
        model_direct_repair["strict_valid_sample_count"] >= int(min_count)
        and model_direct_repair["review_gate_passed"]
    )
    model_conditioned_audio_technical_path_available = bool(
        model_direct_audio["rendered_audio_file_count"] >= int(min_count)
        and model_direct_audio["technical_wav_validation"]
    )
    fallback_replacement_ready = bool(
        same_input_context
        and model_conditioned_candidate_source_available
        and model_conditioned_audio_technical_path_available
        and ranked_export_contract_matched
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "issue_number": int(issue_number),
        "boundary": BOUNDARY,
        "source_boundaries": {
            "alignment": ALIGNMENT_BOUNDARY,
            "fallback_generation": FALLBACK_GENERATION_BOUNDARY,
            "fallback_audio": FALLBACK_AUDIO_BOUNDARY,
            "model_direct_repair": MODEL_DIRECT_REPAIR_BOUNDARY,
            "model_direct_audio": MODEL_DIRECT_AUDIO_BOUNDARY,
        },
        "alignment_source": alignment,
        "fallback_input_path": {
            **fallback_generation,
            "rendered_audio_file_count": fallback_audio["rendered_audio_file_count"],
            "technical_wav_validation": fallback_audio["technical_wav_validation"],
            "wav_duration_min_seconds": fallback_audio["duration_min_seconds"],
            "wav_duration_max_seconds": fallback_audio["duration_max_seconds"],
        },
        "model_conditioned_probe": {
            **model_direct_repair,
            "rendered_audio_file_count": model_direct_audio["rendered_audio_file_count"],
            "technical_wav_validation": model_direct_audio["technical_wav_validation"],
            "wav_duration_min_seconds": model_direct_audio["duration_min_seconds"],
            "wav_duration_max_seconds": model_direct_audio["duration_max_seconds"],
            "same_input_context_as_fallback": same_input_context,
            "ranked_input_path_export_contract_matched": ranked_export_contract_matched,
        },
        "replacement_decision": {
            "model_conditioned_candidate_source_available": model_conditioned_candidate_source_available,
            "model_conditioned_audio_technical_path_available": model_conditioned_audio_technical_path_available,
            "same_input_context_as_fallback": same_input_context,
            "ranked_input_path_export_contract_matched": ranked_export_contract_matched,
            "fallback_replacement_ready": fallback_replacement_ready,
            "candidate_export_required": not ranked_export_contract_matched,
            "selected_next_boundary": NEXT_BOUNDARY,
            "human_review_required_now": False,
        },
        "gap": {
            "current_input_to_wav_path_source": FALLBACK_SOURCE,
            "model_conditioned_source": MODEL_CONDITIONED_SOURCE,
            "missing_ranked_export_contract": not ranked_export_contract_matched,
            "missing_candidate_ranking_in_model_direct_path": not ranked_export_contract_matched,
            "separate_model_direct_audio_path": True,
        },
        "readiness": {
            "boundary": BOUNDARY,
            "model_conditioned_input_path_probe_completed": True,
            "model_conditioned_candidate_source_available": model_conditioned_candidate_source_available,
            "model_conditioned_audio_technical_path_available": model_conditioned_audio_technical_path_available,
            "same_input_context_as_fallback": same_input_context,
            "model_conditioned_ranked_input_path_contract_matched": ranked_export_contract_matched,
            "fallback_replacement_ready": fallback_replacement_ready,
            "candidate_export_required": not ranked_export_contract_matched,
            "human_review_required_now": False,
            "human_audio_preference_claimed": False,
            "midi_to_solo_musical_quality_claimed": False,
            "model_direct_generation_quality_claimed": False,
            "audio_rendered_quality_claimed": False,
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
                "model-conditioned strict MIDI and WAV technical evidence are available, but the "
                "current input-to-WAV contract still uses fallback-ranked candidates; next boundary "
                "should export model-conditioned candidates through the same ranked input path"
            ),
        },
        "not_proven": [
            "fallback_replacement_ready",
            "model_conditioned_ranked_input_path_export",
            "human_audio_preference",
            "midi_to_solo_musical_quality",
            "broad_trained_model_quality",
            "brad_style_adaptation",
        ],
        "next_recommended_issue": "Stage B MIDI-to-solo model-conditioned input path candidate export",
    }


def validate_probe_report(
    report: dict[str, Any],
    *,
    expected_boundary: str | None,
    expected_next_boundary: str | None,
    require_model_conditioned_evidence: bool,
    require_candidate_export: bool,
    require_replacement_not_ready: bool,
    require_no_quality_claim: bool,
) -> dict[str, Any]:
    boundary = str(report.get("boundary") or "")
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    replacement = _dict(report.get("replacement_decision"))
    if expected_boundary and boundary != expected_boundary:
        raise StageBMidiToSoloModelConditionedInputPathProbeError(
            f"expected boundary {expected_boundary}, got {boundary}"
        )
    if expected_next_boundary and str(decision.get("next_boundary") or "") != expected_next_boundary:
        raise StageBMidiToSoloModelConditionedInputPathProbeError("unexpected next boundary")
    if not bool(readiness.get("model_conditioned_input_path_probe_completed", False)):
        raise StageBMidiToSoloModelConditionedInputPathProbeError("probe completion required")
    if require_model_conditioned_evidence:
        required = [
            "model_conditioned_candidate_source_available",
            "model_conditioned_audio_technical_path_available",
            "same_input_context_as_fallback",
        ]
        missing = [name for name in required if not bool(readiness.get(name, False))]
        if missing:
            raise StageBMidiToSoloModelConditionedInputPathProbeError(
                f"missing model-conditioned evidence: {missing}"
            )
    if require_candidate_export and not bool(readiness.get("candidate_export_required", False)):
        raise StageBMidiToSoloModelConditionedInputPathProbeError("candidate export should be required")
    if require_replacement_not_ready and bool(readiness.get("fallback_replacement_ready", True)):
        raise StageBMidiToSoloModelConditionedInputPathProbeError(
            "fallback replacement should not be ready at this boundary"
        )
    if bool(decision.get("critical_user_input_required", True)):
        raise StageBMidiToSoloModelConditionedInputPathProbeError(
            "critical user input should not be required"
        )
    if require_no_quality_claim:
        _require_no_quality_claim(readiness, label="probe readiness")
    return {
        "boundary": boundary,
        "next_boundary": str(decision.get("next_boundary") or ""),
        "model_conditioned_candidate_source_available": bool(
            readiness.get("model_conditioned_candidate_source_available", False)
        ),
        "model_conditioned_audio_technical_path_available": bool(
            readiness.get("model_conditioned_audio_technical_path_available", False)
        ),
        "same_input_context_as_fallback": bool(readiness.get("same_input_context_as_fallback", False)),
        "model_conditioned_ranked_input_path_contract_matched": bool(
            readiness.get("model_conditioned_ranked_input_path_contract_matched", False)
        ),
        "fallback_replacement_ready": bool(readiness.get("fallback_replacement_ready", True)),
        "candidate_export_required": bool(readiness.get("candidate_export_required", False)),
        "human_review_required_now": bool(readiness.get("human_review_required_now", True)),
        "human_audio_preference_claimed": bool(readiness.get("human_audio_preference_claimed", True)),
        "midi_to_solo_musical_quality_claimed": bool(
            readiness.get("midi_to_solo_musical_quality_claimed", True)
        ),
        "critical_user_input_required": bool(decision.get("critical_user_input_required", True)),
        "missing_ranked_export_contract": bool(
            _dict(report.get("gap")).get("missing_ranked_export_contract", False)
        ),
        "selected_next_boundary": str(replacement.get("selected_next_boundary") or ""),
        "next_recommended_issue": str(report.get("next_recommended_issue") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    fallback = report["fallback_input_path"]
    model_probe = report["model_conditioned_probe"]
    readiness = report["readiness"]
    decision = report["decision"]
    lines = [
        "# Stage B MIDI-to-Solo Model-Conditioned Input Path Probe",
        "",
        "## Summary",
        "",
        f"- boundary: `{report['boundary']}`",
        f"- next boundary: `{decision['next_boundary']}`",
        f"- model-conditioned candidate source available: `{_bool_token(readiness['model_conditioned_candidate_source_available'])}`",
        f"- model-conditioned audio technical path available: `{_bool_token(readiness['model_conditioned_audio_technical_path_available'])}`",
        f"- ranked input-path export contract matched: `{_bool_token(readiness['model_conditioned_ranked_input_path_contract_matched'])}`",
        f"- fallback replacement ready: `{_bool_token(readiness['fallback_replacement_ready'])}`",
        f"- candidate export required: `{_bool_token(readiness['candidate_export_required'])}`",
        "",
        "## Fallback Input Path",
        "",
        f"- generation source: `{fallback['generation_source']}`",
        f"- exported candidate count: `{fallback['exported_candidate_count']}`",
        f"- exported qualified candidate count: `{fallback['exported_qualified_candidate_count']}`",
        f"- rendered WAV count: `{fallback['rendered_audio_file_count']}`",
        f"- technical WAV validation: `{_bool_token(fallback['technical_wav_validation'])}`",
        "",
        "## Model-Conditioned Probe",
        "",
        f"- generation source: `{model_probe['generation_source']}`",
        f"- strict-valid sample count: `{model_probe['strict_valid_sample_count']}`",
        f"- min postprocess note count: `{model_probe['min_postprocess_note_count']}`",
        f"- avg postprocess removal ratio: `{model_probe['avg_postprocess_removal_ratio']:.4f}`",
        f"- rendered WAV count: `{model_probe['rendered_audio_file_count']}`",
        f"- same input context as fallback: `{_bool_token(model_probe['same_input_context_as_fallback'])}`",
        "",
        "## Gap",
        "",
        "- current input-to-WAV path source: `context_conditioned_fallback`",
        "- model-conditioned source evidence: `model_checkpoint_direct_constrained`",
        "- missing ranked export contract: `true`",
        "- missing candidate ranking in model-direct path: `true`",
        "",
        "## Claim Boundary",
        "",
        f"- human/audio preference claimed: `{_bool_token(readiness['human_audio_preference_claimed'])}`",
        f"- MIDI-to-solo musical quality claimed: `{_bool_token(readiness['midi_to_solo_musical_quality_claimed'])}`",
        f"- model-direct generation quality claimed: `{_bool_token(readiness['model_direct_generation_quality_claimed'])}`",
        f"- broad trained model quality claimed: `{_bool_token(readiness['broad_trained_model_quality_claimed'])}`",
        "",
        "## Next",
        "",
        f"- `{report['next_recommended_issue']}`",
    ]
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Probe model-conditioned input-path replacement readiness")
    parser.add_argument("--alignment_report", type=str, required=True)
    parser.add_argument("--fallback_generation_report", type=str, required=True)
    parser.add_argument("--fallback_audio_report", type=str, required=True)
    parser.add_argument("--model_direct_repair_report", type=str, required=True)
    parser.add_argument("--model_direct_audio_report", type=str, required=True)
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/stage_b_midi_to_solo_model_conditioned_input_path_probe",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--issue_number", type=int, default=622)
    parser.add_argument("--min_count", type=int, default=3)
    parser.add_argument("--expected_boundary", type=str, default="")
    parser.add_argument("--expected_next_boundary", type=str, default="")
    parser.add_argument("--require_model_conditioned_evidence", action="store_true")
    parser.add_argument("--require_candidate_export", action="store_true")
    parser.add_argument("--require_replacement_not_ready", action="store_true")
    parser.add_argument("--require_no_quality_claim", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    report = build_probe_report(
        alignment_report=read_json(Path(args.alignment_report)),
        fallback_generation_report=read_json(Path(args.fallback_generation_report)),
        fallback_audio_report=read_json(Path(args.fallback_audio_report)),
        model_direct_repair_report=read_json(Path(args.model_direct_repair_report)),
        model_direct_audio_report=read_json(Path(args.model_direct_audio_report)),
        output_dir=output_dir,
        issue_number=int(args.issue_number),
        min_count=int(args.min_count),
    )
    summary = validate_probe_report(
        report,
        expected_boundary=str(args.expected_boundary or ""),
        expected_next_boundary=str(args.expected_next_boundary or ""),
        require_model_conditioned_evidence=bool(args.require_model_conditioned_evidence),
        require_candidate_export=bool(args.require_candidate_export),
        require_replacement_not_ready=bool(args.require_replacement_not_ready),
        require_no_quality_claim=bool(args.require_no_quality_claim),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "stage_b_midi_to_solo_model_conditioned_input_path_probe.json", report)
    write_json(
        output_dir / "stage_b_midi_to_solo_model_conditioned_input_path_probe_validation_summary.json",
        summary,
    )
    markdown = markdown_report(report)
    write_text(output_dir / "stage_b_midi_to_solo_model_conditioned_input_path_probe.md", markdown)
    if args.doc_path:
        write_text(Path(args.doc_path), markdown)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
