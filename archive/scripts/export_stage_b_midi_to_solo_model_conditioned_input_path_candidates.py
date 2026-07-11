"""Export model-conditioned MIDI-to-solo candidates through the ranked input path."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from scripts.assess_stage_b_generic_base_readiness import read_json, write_json, write_text  # noqa: E402
from scripts.probe_stage_b_midi_to_solo_model_conditioned_input_path import (  # noqa: E402
    BOUNDARY as PROBE_BOUNDARY,
    NEXT_BOUNDARY as PROBE_NEXT_BOUNDARY,
    MODEL_CONDITIONED_SOURCE,
    MODEL_DIRECT_REPAIR_BOUNDARY,
)


class StageBMidiToSoloModelConditionedInputPathCandidateExportError(ValueError):
    pass


BOUNDARY = "stage_b_midi_to_solo_model_conditioned_input_path_candidate_export"
NEXT_BOUNDARY = "stage_b_midi_to_solo_model_conditioned_input_path_audio_render_package"
SCHEMA_VERSION = "stage_b_midi_to_solo_model_conditioned_input_path_candidate_export_v1"

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
        raise StageBMidiToSoloModelConditionedInputPathCandidateExportError(
            f"unexpected quality claim in {label}: {claimed}"
        )


def validate_probe_report(report: dict[str, Any]) -> dict[str, Any]:
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    replacement = _dict(report.get("replacement_decision"))
    source = _dict(report.get("alignment_source"))
    if str(report.get("boundary") or readiness.get("boundary") or "") != PROBE_BOUNDARY:
        raise StageBMidiToSoloModelConditionedInputPathCandidateExportError(
            "model-conditioned input path probe boundary required"
        )
    if str(decision.get("next_boundary") or "") != PROBE_NEXT_BOUNDARY:
        raise StageBMidiToSoloModelConditionedInputPathCandidateExportError(
            "probe report must route to candidate export"
        )
    required_true = [
        "model_conditioned_input_path_probe_completed",
        "model_conditioned_candidate_source_available",
        "model_conditioned_audio_technical_path_available",
        "same_input_context_as_fallback",
        "candidate_export_required",
    ]
    missing = [name for name in required_true if not bool(readiness.get(name, False))]
    if missing:
        raise StageBMidiToSoloModelConditionedInputPathCandidateExportError(
            f"missing probe readiness: {missing}"
        )
    if bool(readiness.get("model_conditioned_ranked_input_path_contract_matched", True)):
        raise StageBMidiToSoloModelConditionedInputPathCandidateExportError(
            "candidate export should only run while ranked contract is missing"
        )
    if bool(readiness.get("fallback_replacement_ready", True)):
        raise StageBMidiToSoloModelConditionedInputPathCandidateExportError(
            "fallback replacement should not already be ready"
        )
    if str(replacement.get("selected_next_boundary") or "") != PROBE_NEXT_BOUNDARY:
        raise StageBMidiToSoloModelConditionedInputPathCandidateExportError(
            "probe replacement decision next boundary mismatch"
        )
    if bool(decision.get("critical_user_input_required", True)):
        raise StageBMidiToSoloModelConditionedInputPathCandidateExportError(
            "critical user input should not be required"
        )
    if not bool(source.get("phrase_bank_cli_technical_path_completed", False)):
        raise StageBMidiToSoloModelConditionedInputPathCandidateExportError(
            "phrase-bank CLI technical path completion required"
        )
    if _int(source.get("cli_candidate_count")) < 3:
        raise StageBMidiToSoloModelConditionedInputPathCandidateExportError("CLI candidate count below 3")
    if _int(source.get("cli_rendered_audio_file_count")) < 3:
        raise StageBMidiToSoloModelConditionedInputPathCandidateExportError("CLI rendered WAV count below 3")
    if _int(source.get("cli_input_context_bars")) <= 0:
        raise StageBMidiToSoloModelConditionedInputPathCandidateExportError("CLI input context bars required")
    if bool(source.get("cli_preference_fill_allowed", True)):
        raise StageBMidiToSoloModelConditionedInputPathCandidateExportError(
            "CLI preference fill should remain blocked"
        )
    _require_no_quality_claim(readiness, label="probe readiness")
    return {
        "boundary": PROBE_BOUNDARY,
        "model_conditioned_candidate_source_available": True,
        "model_conditioned_audio_technical_path_available": True,
        "same_input_context_as_fallback": True,
        "candidate_export_required": True,
        "phrase_bank_cli_technical_path_completed": True,
        "cli_candidate_count": _int(source.get("cli_candidate_count")),
        "cli_rendered_audio_file_count": _int(source.get("cli_rendered_audio_file_count")),
        "cli_input_context_bars": _int(source.get("cli_input_context_bars")),
        "cli_preference_fill_allowed": bool(source.get("cli_preference_fill_allowed", True)),
    }


def validate_model_direct_repair(report: dict[str, Any], *, min_count: int) -> dict[str, Any]:
    readiness = _dict(report.get("readiness"))
    repair_config = _dict(report.get("repair_config"))
    generation = _dict(report.get("repaired_generation_summary"))
    context = _dict(report.get("context_summary"))
    if str(report.get("boundary") or readiness.get("boundary") or "") != MODEL_DIRECT_REPAIR_BOUNDARY:
        raise StageBMidiToSoloModelConditionedInputPathCandidateExportError(
            "model-direct repair boundary required"
        )
    if str(repair_config.get("generation_source") or "") != MODEL_CONDITIONED_SOURCE:
        raise StageBMidiToSoloModelConditionedInputPathCandidateExportError(
            "model-conditioned source mismatch"
        )
    if not bool(readiness.get("direct_generation_review_gate_passed", False)):
        raise StageBMidiToSoloModelConditionedInputPathCandidateExportError(
            "model-direct review gate pass required"
        )
    if _int(generation.get("strict_valid_sample_count")) < int(min_count):
        raise StageBMidiToSoloModelConditionedInputPathCandidateExportError(
            "strict-valid sample count below minimum"
        )
    if _int(generation.get("min_postprocess_note_count")) < 24:
        raise StageBMidiToSoloModelConditionedInputPathCandidateExportError(
            "postprocess note count below contract minimum"
        )
    midi_paths = [str(path) for path in _list(generation.get("midi_paths"))[: int(min_count)]]
    if len(midi_paths) < int(min_count) or not all(_path_exists(path) for path in midi_paths):
        raise StageBMidiToSoloModelConditionedInputPathCandidateExportError(
            "model-direct MIDI artifacts required"
        )
    _require_no_quality_claim(readiness, label="model-direct repair readiness")
    return {
        "boundary": MODEL_DIRECT_REPAIR_BOUNDARY,
        "generation_source": MODEL_CONDITIONED_SOURCE,
        "sample_count": _int(generation.get("sample_count")),
        "strict_valid_sample_count": _int(generation.get("strict_valid_sample_count")),
        "grammar_gate_sample_count": _int(generation.get("grammar_gate_sample_count")),
        "min_postprocess_note_count": _int(generation.get("min_postprocess_note_count")),
        "max_postprocess_note_count": _int(generation.get("max_postprocess_note_count")),
        "midi_paths": midi_paths,
        "context_bars": _int(context.get("context_bars")),
        "context_bpm": _int(context.get("bpm")),
        "chord_progression": list(_list(context.get("chord_progression"))),
        "generation_report_path": str(report.get("generation_report_path") or ""),
    }


def validate_generation_report(report: dict[str, Any], *, min_count: int) -> list[dict[str, Any]]:
    if str(report.get("generation_mode") or "") != "constrained":
        raise StageBMidiToSoloModelConditionedInputPathCandidateExportError(
            "model-direct constrained generation report required"
        )
    if not bool(report.get("passed_grammar_gate", False)):
        raise StageBMidiToSoloModelConditionedInputPathCandidateExportError("grammar gate pass required")
    if not bool(report.get("passed_strict_review_gate", False)):
        raise StageBMidiToSoloModelConditionedInputPathCandidateExportError(
            "strict review gate pass required"
        )
    samples = [_dict(item) for item in _list(report.get("samples"))]
    qualified = [item for item in samples if bool(item.get("valid", False)) and bool(item.get("strict_valid", False))]
    if len(qualified) < int(min_count):
        raise StageBMidiToSoloModelConditionedInputPathCandidateExportError(
            "not enough strict-valid model-conditioned samples"
        )
    rows: list[dict[str, Any]] = []
    for sample in qualified[: int(min_count)]:
        midi_path = str(sample.get("midi_path") or "")
        metrics = _dict(sample.get("metrics"))
        collapse = _dict(sample.get("collapse"))
        phrase = _dict(sample.get("phrase_contour"))
        pitch_roles = _dict(sample.get("pitch_roles"))
        temporal = _dict(sample.get("temporal_coverage"))
        if not _path_exists(midi_path):
            raise StageBMidiToSoloModelConditionedInputPathCandidateExportError(
                f"model-conditioned MIDI missing: {midi_path}"
            )
        if _int(metrics.get("note_count")) < 24:
            raise StageBMidiToSoloModelConditionedInputPathCandidateExportError(
                "model-conditioned note count below contract minimum"
            )
        if _int(metrics.get("unique_pitch_count")) < 8:
            raise StageBMidiToSoloModelConditionedInputPathCandidateExportError(
                "model-conditioned unique pitch count below contract minimum"
            )
        if _int(metrics.get("max_simultaneous_notes")) > 1:
            raise StageBMidiToSoloModelConditionedInputPathCandidateExportError(
                "model-conditioned simultaneous note count above contract maximum"
            )
        score = (
            _float(metrics.get("dead_air_ratio")) * 40.0
            + (1.0 - _float(metrics.get("chord_tone_ratio"))) * 10.0
            + _float(collapse.get("repeated_pitch_ratio")) * 5.0
            + _float(collapse.get("postprocess_removal_ratio")) * 100.0
            + (1.0 - _float(temporal.get("position_span_ratio"))) * 5.0
            + _int(sample.get("sample_index")) * 0.001
        )
        rows.append(
            {
                "sample_index": _int(sample.get("sample_index")),
                "sample_seed": _int(sample.get("sample_seed")),
                "source_midi_path": midi_path,
                "generation_source": MODEL_CONDITIONED_SOURCE,
                "score": round(float(score), 6),
                "valid": True,
                "strict_valid": True,
                "grammar_gate_passed": bool(sample.get("grammar_gate_passed", False)),
                "contract_gate_passed": True,
                "note_count": _int(metrics.get("note_count")),
                "unique_pitch_count": _int(metrics.get("unique_pitch_count")),
                "unique_pitch_class_count": _int(metrics.get("unique_pitch_class_count")),
                "max_simultaneous_notes": _int(metrics.get("max_simultaneous_notes")),
                "dead_air_ratio": _float(metrics.get("dead_air_ratio")),
                "phrase_coverage_ratio": _float(metrics.get("phrase_coverage_ratio")),
                "chord_tone_ratio": _float(metrics.get("chord_tone_ratio")),
                "non_chord_tone_ratio": _float(pitch_roles.get("non_chord_tone_ratio")),
                "repeated_pitch_ratio": _float(collapse.get("repeated_pitch_ratio")),
                "postprocess_removal_ratio": _float(collapse.get("postprocess_removal_ratio")),
                "onset_coverage_ratio": _float(temporal.get("onset_coverage_ratio")),
                "sustained_coverage_ratio": _float(temporal.get("sustained_coverage_ratio")),
                "position_span_ratio": _float(temporal.get("position_span_ratio")),
                "direction_change_ratio": _float(phrase.get("direction_change_ratio")),
                "stepwise_motion_ratio": _float(phrase.get("stepwise_motion_ratio")),
                "leap_motion_ratio": _float(phrase.get("leap_motion_ratio")),
            }
        )
    return rows


def export_ranked_candidates(
    candidates: list[dict[str, Any]],
    *,
    output_dir: Path,
    export_top_midi_count: int,
) -> list[dict[str, Any]]:
    export_dir = output_dir / "midi"
    ranked = sorted(
        candidates,
        key=lambda row: (float(row["score"]), int(row["sample_index"])),
    )
    exported: list[dict[str, Any]] = []
    for rank, row in enumerate(ranked[: int(export_top_midi_count)], start=1):
        source = Path(str(row["source_midi_path"]))
        export_path = export_dir / f"rank_{rank:02d}_sample_{int(row['sample_index']):02d}.mid"
        export_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, export_path)
        exported.append(
            {
                **row,
                "rank": int(rank),
                "export_midi_path": str(export_path),
            }
        )
    return exported


def build_candidate_export_report(
    *,
    probe_report: dict[str, Any],
    model_direct_repair_report: dict[str, Any],
    model_direct_generation_report: dict[str, Any],
    output_dir: Path,
    issue_number: int,
    export_top_midi_count: int,
) -> dict[str, Any]:
    probe = validate_probe_report(probe_report)
    repair = validate_model_direct_repair(
        model_direct_repair_report,
        min_count=export_top_midi_count,
    )
    candidates = validate_generation_report(
        model_direct_generation_report,
        min_count=export_top_midi_count,
    )
    top_candidates = export_ranked_candidates(
        candidates,
        output_dir=output_dir,
        export_top_midi_count=export_top_midi_count,
    )
    export_ready = bool(len(top_candidates) >= int(export_top_midi_count))
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "issue_number": int(issue_number),
        "boundary": BOUNDARY,
        "source_boundaries": {
            "probe": PROBE_BOUNDARY,
            "model_direct_repair": MODEL_DIRECT_REPAIR_BOUNDARY,
        },
        "probe_source": probe,
        "model_direct_source": repair,
        "input_context": {
            "chord_progression": repair["chord_progression"],
            "bpm": repair["context_bpm"],
            "bars": repair["context_bars"],
        },
        "generation_config": {
            "generation_source": MODEL_CONDITIONED_SOURCE,
            "model_checkpoint_generation_used": True,
            "fallback_used": False,
            "export_top_midi_count": int(export_top_midi_count),
            "ranking_basis": "objective_proxy_metrics",
        },
        "objective_gate": {
            "min_note_count": 24,
            "min_unique_pitch_count": 8,
            "max_simultaneous_notes": 1,
        },
        "candidates": candidates,
        "top_candidates": top_candidates,
        "summary": {
            "candidate_count": int(len(candidates)),
            "qualified_candidate_count": int(len(candidates)),
            "exported_candidate_count": int(len(top_candidates)),
            "exported_qualified_candidate_count": int(len(top_candidates)),
            "best_score": float(top_candidates[0]["score"]) if top_candidates else None,
            "best_note_count": _int(top_candidates[0]["note_count"]) if top_candidates else 0,
            "best_unique_pitch_count": _int(top_candidates[0]["unique_pitch_count"]) if top_candidates else 0,
            "best_max_simultaneous_notes": _int(top_candidates[0]["max_simultaneous_notes"]) if top_candidates else 0,
            "best_chord_tone_ratio": _float(top_candidates[0]["chord_tone_ratio"]) if top_candidates else 0.0,
            "best_dead_air_ratio": _float(top_candidates[0]["dead_air_ratio"]) if top_candidates else 0.0,
            "best_position_span_ratio": _float(top_candidates[0]["position_span_ratio"]) if top_candidates else 0.0,
        },
        "readiness": {
            "boundary": BOUNDARY,
            "model_conditioned_input_path_candidate_export_completed": True,
            "ranked_midi_candidates_exported": export_ready,
            "model_conditioned_ranked_input_path_contract_matched": export_ready,
            "fallback_replacement_candidate_export_ready": export_ready,
            "fallback_replacement_ready": False,
            "candidate_audio_render_required": True,
            "human_review_required_now": False,
            "human_audio_preference_claimed": False,
            "midi_to_solo_musical_quality_claimed": False,
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
            "reason": (
                "model-conditioned strict MIDI candidates now match the ranked input-path export contract; "
                "next boundary should render exported ranked MIDI to WAV without claiming quality"
            ),
        },
        "not_proven": [
            "fallback_replacement_ready",
            "model_conditioned_audio_render_for_ranked_exports",
            "human_audio_preference",
            "midi_to_solo_musical_quality",
            "broad_trained_model_quality",
            "brad_style_adaptation",
        ],
        "next_recommended_issue": "Stage B MIDI-to-solo model-conditioned input path audio render package",
    }


def validate_candidate_export_report(
    report: dict[str, Any],
    *,
    expected_boundary: str | None,
    expected_next_boundary: str | None,
    min_exported_candidates: int,
    require_ranked_export_contract: bool,
    require_audio_render_required: bool,
    require_no_quality_claim: bool,
) -> dict[str, Any]:
    boundary = str(report.get("boundary") or "")
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    summary = _dict(report.get("summary"))
    top_candidates = [_dict(item) for item in _list(report.get("top_candidates"))]
    if expected_boundary and boundary != expected_boundary:
        raise StageBMidiToSoloModelConditionedInputPathCandidateExportError(
            f"expected boundary {expected_boundary}, got {boundary}"
        )
    if expected_next_boundary and str(decision.get("next_boundary") or "") != expected_next_boundary:
        raise StageBMidiToSoloModelConditionedInputPathCandidateExportError("unexpected next boundary")
    if _int(summary.get("exported_candidate_count")) < int(min_exported_candidates):
        raise StageBMidiToSoloModelConditionedInputPathCandidateExportError(
            "exported candidate count below threshold"
        )
    for row in top_candidates[: int(min_exported_candidates)]:
        if not _path_exists(str(row.get("export_midi_path") or "")):
            raise StageBMidiToSoloModelConditionedInputPathCandidateExportError(
                "exported MIDI path missing"
            )
        if str(row.get("generation_source") or "") != MODEL_CONDITIONED_SOURCE:
            raise StageBMidiToSoloModelConditionedInputPathCandidateExportError(
                "exported candidate source mismatch"
            )
        if not bool(row.get("contract_gate_passed", False)):
            raise StageBMidiToSoloModelConditionedInputPathCandidateExportError(
                "exported candidate contract gate failed"
            )
    if require_ranked_export_contract and not bool(
        readiness.get("model_conditioned_ranked_input_path_contract_matched", False)
    ):
        raise StageBMidiToSoloModelConditionedInputPathCandidateExportError(
            "ranked input-path export contract should be matched"
        )
    if require_audio_render_required and not bool(readiness.get("candidate_audio_render_required", False)):
        raise StageBMidiToSoloModelConditionedInputPathCandidateExportError(
            "candidate audio render should be required"
        )
    if bool(readiness.get("fallback_replacement_ready", True)):
        raise StageBMidiToSoloModelConditionedInputPathCandidateExportError(
            "fallback replacement should remain pending until ranked audio render"
        )
    if bool(decision.get("critical_user_input_required", True)):
        raise StageBMidiToSoloModelConditionedInputPathCandidateExportError(
            "critical user input should not be required"
        )
    if require_no_quality_claim:
        _require_no_quality_claim(readiness, label="candidate export readiness")
    return {
        "boundary": boundary,
        "next_boundary": str(decision.get("next_boundary") or ""),
        "model_conditioned_input_path_candidate_export_completed": bool(
            readiness.get("model_conditioned_input_path_candidate_export_completed", False)
        ),
        "ranked_midi_candidates_exported": bool(readiness.get("ranked_midi_candidates_exported", False)),
        "model_conditioned_ranked_input_path_contract_matched": bool(
            readiness.get("model_conditioned_ranked_input_path_contract_matched", False)
        ),
        "fallback_replacement_candidate_export_ready": bool(
            readiness.get("fallback_replacement_candidate_export_ready", False)
        ),
        "fallback_replacement_ready": bool(readiness.get("fallback_replacement_ready", True)),
        "candidate_audio_render_required": bool(readiness.get("candidate_audio_render_required", False)),
        "phrase_bank_cli_technical_path_completed": bool(
            _dict(report.get("probe_source")).get("phrase_bank_cli_technical_path_completed", False)
        ),
        "cli_candidate_count": _int(_dict(report.get("probe_source")).get("cli_candidate_count")),
        "cli_rendered_audio_file_count": _int(
            _dict(report.get("probe_source")).get("cli_rendered_audio_file_count")
        ),
        "cli_input_context_bars": _int(_dict(report.get("probe_source")).get("cli_input_context_bars")),
        "cli_preference_fill_allowed": bool(
            _dict(report.get("probe_source")).get("cli_preference_fill_allowed", True)
        ),
        "candidate_count": _int(summary.get("candidate_count")),
        "exported_candidate_count": _int(summary.get("exported_candidate_count")),
        "best_note_count": _int(summary.get("best_note_count")),
        "best_unique_pitch_count": _int(summary.get("best_unique_pitch_count")),
        "best_max_simultaneous_notes": _int(summary.get("best_max_simultaneous_notes")),
        "human_review_required_now": bool(readiness.get("human_review_required_now", True)),
        "human_audio_preference_claimed": bool(readiness.get("human_audio_preference_claimed", True)),
        "midi_to_solo_musical_quality_claimed": bool(
            readiness.get("midi_to_solo_musical_quality_claimed", True)
        ),
        "critical_user_input_required": bool(decision.get("critical_user_input_required", True)),
        "next_recommended_issue": str(report.get("next_recommended_issue") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    probe = report["probe_source"]
    summary = report["summary"]
    readiness = report["readiness"]
    decision = report["decision"]
    config = report["generation_config"]
    context = report["input_context"]
    lines = [
        "# Stage B MIDI-to-Solo Model-Conditioned Input Path Candidate Export",
        "",
        "## Summary",
        "",
        f"- boundary: `{report['boundary']}`",
        f"- next boundary: `{decision['next_boundary']}`",
        f"- generation source: `{config['generation_source']}`",
        f"- ranked MIDI candidates exported: `{_bool_token(readiness['ranked_midi_candidates_exported'])}`",
        f"- ranked input-path export contract matched: `{_bool_token(readiness['model_conditioned_ranked_input_path_contract_matched'])}`",
        f"- fallback replacement candidate export ready: `{_bool_token(readiness['fallback_replacement_candidate_export_ready'])}`",
        f"- fallback replacement ready: `{_bool_token(readiness['fallback_replacement_ready'])}`",
        f"- candidate audio render required: `{_bool_token(readiness['candidate_audio_render_required'])}`",
        "",
        "## Probe Source",
        "",
        f"- phrase-bank CLI technical path completed: `{_bool_token(probe['phrase_bank_cli_technical_path_completed'])}`",
        f"- CLI candidate / rendered WAV: `{probe['cli_candidate_count']}` / `{probe['cli_rendered_audio_file_count']}`",
        f"- CLI input context bars: `{probe['cli_input_context_bars']}`",
        f"- CLI preference fill allowed: `{_bool_token(probe['cli_preference_fill_allowed'])}`",
        "",
        "## Input Context",
        "",
        f"- chord progression: `{', '.join(context['chord_progression'])}`",
        f"- bars: `{context['bars']}`",
        f"- bpm: `{context['bpm']}`",
        "",
        "## Candidate Summary",
        "",
        f"- candidate count: `{summary['candidate_count']}`",
        f"- exported candidate count: `{summary['exported_candidate_count']}`",
        f"- exported qualified candidate count: `{summary['exported_qualified_candidate_count']}`",
        f"- best score: `{summary['best_score']}`",
        f"- best note count: `{summary['best_note_count']}`",
        f"- best unique pitch count: `{summary['best_unique_pitch_count']}`",
        f"- best max simultaneous notes: `{summary['best_max_simultaneous_notes']}`",
        f"- best chord-tone ratio: `{summary['best_chord_tone_ratio']}`",
        f"- best dead-air ratio: `{summary['best_dead_air_ratio']}`",
        "",
        "## Exported MIDI",
        "",
    ]
    for row in report["top_candidates"]:
        lines.append(
            f"- rank `{row['rank']}` sample `{row['sample_index']}` seed `{row['sample_seed']}`: "
            f"`{row['export_midi_path']}`, score `{row['score']}`, notes `{row['note_count']}`, "
            f"unique pitches `{row['unique_pitch_count']}`"
        )
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            f"- human/audio preference claimed: `{_bool_token(readiness['human_audio_preference_claimed'])}`",
            f"- MIDI-to-solo musical quality claimed: `{_bool_token(readiness['midi_to_solo_musical_quality_claimed'])}`",
            f"- model checkpoint generation quality claimed: `{_bool_token(readiness['model_checkpoint_generation_quality_claimed'])}`",
            f"- broad trained-model quality claimed: `{_bool_token(readiness['broad_trained_model_quality_claimed'])}`",
            "",
            "## Next",
            "",
            f"- `{report['next_recommended_issue']}`",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export model-conditioned candidates through ranked input path")
    parser.add_argument("--probe_report", type=str, required=True)
    parser.add_argument("--model_direct_repair_report", type=str, required=True)
    parser.add_argument("--model_direct_generation_report", type=str, required=True)
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/stage_b_midi_to_solo_model_conditioned_input_path_candidate_export",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--issue_number", type=int, default=624)
    parser.add_argument("--export_top_midi_count", type=int, default=3)
    parser.add_argument("--expected_boundary", type=str, default="")
    parser.add_argument("--expected_next_boundary", type=str, default="")
    parser.add_argument("--require_ranked_export_contract", action="store_true")
    parser.add_argument("--require_audio_render_required", action="store_true")
    parser.add_argument("--require_no_quality_claim", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    report = build_candidate_export_report(
        probe_report=read_json(Path(args.probe_report)),
        model_direct_repair_report=read_json(Path(args.model_direct_repair_report)),
        model_direct_generation_report=read_json(Path(args.model_direct_generation_report)),
        output_dir=output_dir,
        issue_number=int(args.issue_number),
        export_top_midi_count=int(args.export_top_midi_count),
    )
    summary = validate_candidate_export_report(
        report,
        expected_boundary=str(args.expected_boundary or ""),
        expected_next_boundary=str(args.expected_next_boundary or ""),
        min_exported_candidates=int(args.export_top_midi_count),
        require_ranked_export_contract=bool(args.require_ranked_export_contract),
        require_audio_render_required=bool(args.require_audio_render_required),
        require_no_quality_claim=bool(args.require_no_quality_claim),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "stage_b_midi_to_solo_model_conditioned_input_path_candidate_export.json", report)
    write_json(
        output_dir / "stage_b_midi_to_solo_model_conditioned_input_path_candidate_export_validation_summary.json",
        summary,
    )
    markdown = markdown_report(report)
    write_text(output_dir / "stage_b_midi_to_solo_model_conditioned_input_path_candidate_export.md", markdown)
    if args.doc_path:
        write_text(Path(args.doc_path), markdown)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
