"""Build a MIDI-to-solo phrase-bank retrieval baseline from input context."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import pretty_midi

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from inference.app.metrics import compute_midi_metrics  # noqa: E402
from inference.app.schemas import GenerationRequest  # noqa: E402
from scripts.assess_stage_b_generic_base_readiness import write_json, write_text  # noqa: E402
from scripts.run_stage_b_data_motif_generation_compare import (  # noqa: E402
    analyze_contour_landing_profile,
    build_stage_b_primer,
    candidate_sort_key,
    decode_stage_b_midi,
    generated_tokens_for_mode,
    postprocess_stage_b_midi,
    run_template_extraction,
    sample_report,
    write_overlap_free_solo_midi,
)
from scripts.run_stage_b_midi_to_solo_conditioned_generation_probe import (  # noqa: E402
    _bool_token,
    _dict,
    _float,
    _int,
    _list,
    derive_chord_progression,
    validate_context_source,
)


class StageBMidiToSoloPhraseBankRetrievalBaselineError(ValueError):
    pass


CONTEXT_BOUNDARY = "stage_b_midi_to_solo_context_extraction_mvp"
BOUNDARY = "stage_b_midi_to_solo_phrase_bank_retrieval_baseline"
NEXT_BOUNDARY = "stage_b_midi_to_solo_phrase_bank_audio_render_package"
SCHEMA_VERSION = "stage_b_midi_to_solo_phrase_bank_retrieval_baseline_v1"
DEFAULT_MODES = (
    "data_motif_rhythm_phrase_variation",
    "data_motif_contour_landing_repair",
    "data_motif_phrase_recovery",
)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise StageBMidiToSoloPhraseBankRetrievalBaselineError(f"report missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def parse_modes(raw: str) -> list[str]:
    modes = [item.strip() for item in str(raw or "").split(",") if item.strip()]
    return modes or list(DEFAULT_MODES)


def mode_label(mode: str) -> str:
    return str(mode).replace("_", "-")


def build_template_args(
    *,
    run_id: str,
    run_dir: Path,
    input_dir: Path,
    max_files: int,
    window_bars: int,
    window_stride_bars: int,
    min_window_target_notes: int,
    motif_length: int,
    max_bar_span: int,
    max_records: int,
    template_top_n: int,
) -> SimpleNamespace:
    return SimpleNamespace(
        run_id=run_id,
        output_root=run_dir / "phrase_bank_templates",
        input_dir=input_dir,
        max_files=int(max_files),
        window_bars=int(window_bars),
        window_stride_bars=int(window_stride_bars),
        min_window_target_notes=int(min_window_target_notes),
        motif_length=int(motif_length),
        max_bar_span=int(max_bar_span),
        max_records=int(max_records),
        template_top_n=int(template_top_n),
    )


def load_or_extract_template_report(
    *,
    output_dir: Path,
    run_id: str,
    input_dir: Path,
    max_files: int,
    window_bars: int,
    window_stride_bars: int,
    min_window_target_notes: int,
    motif_length: int,
    max_bar_span: int,
    max_records: int,
    template_top_n: int,
    template_report: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if template_report is not None:
        return template_report, {
            "skipped": True,
            "reason": "template_report provided by caller",
            "returncode": 0,
        }
    args = build_template_args(
        run_id=f"{run_id}_templates",
        run_dir=output_dir,
        input_dir=input_dir,
        max_files=max_files,
        window_bars=window_bars,
        window_stride_bars=window_stride_bars,
        min_window_target_notes=min_window_target_notes,
        motif_length=motif_length,
        max_bar_span=max_bar_span,
        max_records=max_records,
        template_top_n=template_top_n,
    )
    report_path, command = run_template_extraction(args, output_dir)
    if int(command.get("returncode", 1)) != 0:
        raise StageBMidiToSoloPhraseBankRetrievalBaselineError("phrase-bank template extraction failed")
    return read_json(report_path), command


def validate_template_report(report: dict[str, Any]) -> dict[str, Any]:
    summary = _dict(report.get("summary"))
    source_record_count = _int(summary.get("source_record_count"))
    motif_count = _int(summary.get("motif_count"))
    rhythm_count = _int(summary.get("unique_rhythm_template_count"))
    contour_count = _int(summary.get("unique_contour_template_count"))
    if source_record_count <= 0:
        raise StageBMidiToSoloPhraseBankRetrievalBaselineError("phrase-bank source records required")
    if motif_count <= 0:
        raise StageBMidiToSoloPhraseBankRetrievalBaselineError("phrase-bank motif rows required")
    if not _list(summary.get("top_rhythm_templates")):
        raise StageBMidiToSoloPhraseBankRetrievalBaselineError("rhythm templates required")
    if not _list(summary.get("top_contour_templates")):
        raise StageBMidiToSoloPhraseBankRetrievalBaselineError("contour templates required")
    return {
        "source_record_count": source_record_count,
        "motif_count": motif_count,
        "unique_rhythm_template_count": rhythm_count,
        "unique_contour_template_count": contour_count,
        "top_rhythm_template_support_ratio": _float(summary.get("top_rhythm_template_support_ratio")),
        "top_contour_template_support_ratio": _float(summary.get("top_contour_template_support_ratio")),
    }


def contract_reasons(
    row: dict[str, Any],
    *,
    min_note_count: int,
    min_unique_pitch_count: int,
    max_simultaneous_notes: int,
    min_phrase_coverage_ratio: float,
    max_dead_air_ratio: float,
) -> list[str]:
    metrics = _dict(row.get("metrics"))
    reasons: list[str] = []
    if _int(metrics.get("note_count")) < int(min_note_count):
        reasons.append(f"note_count {_int(metrics.get('note_count'))} < {int(min_note_count)}")
    if _int(metrics.get("unique_pitch_count")) < int(min_unique_pitch_count):
        reasons.append(
            f"unique_pitch_count {_int(metrics.get('unique_pitch_count'))} < {int(min_unique_pitch_count)}"
        )
    if _int(metrics.get("max_simultaneous_notes")) > int(max_simultaneous_notes):
        reasons.append(
            f"max_simultaneous_notes {_int(metrics.get('max_simultaneous_notes'))} > {int(max_simultaneous_notes)}"
        )
    phrase_coverage = _float(metrics.get("phrase_coverage_ratio"))
    if phrase_coverage < float(min_phrase_coverage_ratio):
        reasons.append(f"phrase_coverage_ratio {phrase_coverage:.3f} < {float(min_phrase_coverage_ratio):.3f}")
    dead_air = _float(metrics.get("dead_air_ratio"))
    if dead_air > float(max_dead_air_ratio):
        reasons.append(f"dead_air_ratio {dead_air:.3f} > {float(max_dead_air_ratio):.3f}")
    return reasons


def phrase_bank_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    metrics = _dict(row.get("metrics"))
    rhythm = _dict(row.get("rhythm_profile"))
    contour = _dict(row.get("contour_landing_profile"))
    base = candidate_sort_key(row)
    return (
        int(bool(row.get("contract_gate_passed"))),
        int(bool(row.get("strict_valid"))),
        int(bool(row.get("valid"))),
        int(bool(contour.get("final_landing_resolved"))),
        _float(metrics.get("phrase_coverage_ratio")),
        -_float(metrics.get("dead_air_ratio")),
        _int(metrics.get("note_count")),
        _int(metrics.get("unique_pitch_count")),
        _float(rhythm.get("ioi_diversity_ratio")),
        _float(rhythm.get("duration_diversity_ratio")),
        -_int(contour.get("max_abs_interval")),
        base,
    )


def write_generated_midi(
    *,
    tokens: Sequence[int],
    output_path: Path,
    bpm: int,
    simultaneous_limit: int,
) -> dict[str, Any]:
    midi = decode_stage_b_midi(tokens, tempo_bpm=bpm)
    postprocess_report = postprocess_stage_b_midi(midi, simultaneous_limit=int(simultaneous_limit))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    midi.write(str(output_path))
    return postprocess_report


def generate_phrase_bank_candidates(
    *,
    template_report: dict[str, Any],
    chord_progression: list[str],
    output_dir: Path,
    modes: Sequence[str],
    candidate_count: int,
    export_top_midi_count: int,
    seed_start: int,
    bpm: int,
    bars: int,
    density: str,
    energy: str,
    note_groups_per_bar: int,
    max_sequence: int,
    max_simultaneous_notes: int,
    min_note_count: int,
    min_unique_pitch_count: int,
    min_phrase_coverage_ratio: float,
    max_dead_air_ratio: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sample_dir = output_dir / "raw_midi"
    export_dir = output_dir / "midi"
    primer_tokens = build_stage_b_primer(chord_progression, bpm)
    request = GenerationRequest(
        bpm=int(bpm),
        chord_progression=chord_progression,
        bars=int(bars),
        density=density,
        energy=energy,
        seed=int(seed_start),
        temperature=0.9,
        top_k=8,
    )
    request.validate()

    rows: list[dict[str, Any]] = []
    selected_modes = list(modes) or list(DEFAULT_MODES)
    for index in range(max(1, int(candidate_count))):
        mode = selected_modes[index % len(selected_modes)]
        seed = int(seed_start) + index
        tokens = generated_tokens_for_mode(
            mode,
            primer_tokens=primer_tokens,
            chords=chord_progression,
            bars=int(bars),
            note_groups_per_bar=int(note_groups_per_bar),
            template_report=template_report,
            seed=seed,
        )[: int(max_sequence)]
        midi_path = sample_dir / mode / f"{mode_label(mode)}_seed_{seed}.mid"
        postprocess_report = write_generated_midi(
            tokens=tokens,
            output_path=midi_path,
            bpm=int(bpm),
            simultaneous_limit=int(max_simultaneous_notes),
        )
        row = sample_report(
            sample_index=index + 1,
            sample_seed=seed,
            tokens=tokens,
            primer_size=len(primer_tokens),
            target_length=int(max_sequence),
            midi_path=midi_path,
            request=request,
            postprocess_report=postprocess_report,
        )
        row["mode"] = mode
        row["generation_source"] = "phrase_bank_data_motif_retrieval"
        row["candidate_midi_path"] = str(midi_path)
        row["contour_landing_profile"] = analyze_contour_landing_profile(
            tokens,
            chords=chord_progression,
            primer_size=len(primer_tokens),
        )
        reasons = contract_reasons(
            row,
            min_note_count=min_note_count,
            min_unique_pitch_count=min_unique_pitch_count,
            max_simultaneous_notes=max_simultaneous_notes,
            min_phrase_coverage_ratio=min_phrase_coverage_ratio,
            max_dead_air_ratio=max_dead_air_ratio,
        )
        row["contract_gate_passed"] = not reasons
        row["contract_failure_reasons"] = reasons
        rows.append(row)

    ranked = sorted(rows, key=phrase_bank_sort_key, reverse=True)
    exportable = [row for row in ranked if bool(row.get("contract_gate_passed"))]
    top_source_rows = exportable[: int(export_top_midi_count)]
    if len(top_source_rows) < int(export_top_midi_count):
        top_source_rows = ranked[: int(export_top_midi_count)]

    top_rows: list[dict[str, Any]] = []
    for rank, row in enumerate(top_source_rows, start=1):
        source_path = Path(str(row["candidate_midi_path"]))
        export_path = export_dir / f"rank_{rank:02d}_{mode_label(str(row['mode']))}_seed_{int(row['sample_seed'])}.mid"
        if source_path.exists():
            write_overlap_free_solo_midi(source_path, export_path, bpm=bpm)
        else:
            export_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_path, export_path)
        exported_metrics = compute_midi_metrics(export_path, 0, fallback_used=True, request=request).to_dict()
        top = dict(row)
        top["rank"] = int(rank)
        top["export_midi_path"] = str(export_path)
        top["exported_metrics"] = exported_metrics
        top["export_contract_gate_passed"] = not contract_reasons(
            {"metrics": exported_metrics},
            min_note_count=min_note_count,
            min_unique_pitch_count=min_unique_pitch_count,
            max_simultaneous_notes=max_simultaneous_notes,
            min_phrase_coverage_ratio=min_phrase_coverage_ratio,
            max_dead_air_ratio=max_dead_air_ratio,
        )
        top_rows.append(top)
    return ranked, top_rows


def build_phrase_bank_retrieval_report(
    *,
    context_report: dict[str, Any],
    output_dir: Path,
    run_id: str,
    issue_number: int,
    template_report: dict[str, Any] | None,
    input_dir: Path,
    modes: Sequence[str],
    candidate_count: int,
    export_top_midi_count: int,
    seed_start: int,
    bpm: int,
    bars: int,
    density: str,
    energy: str,
    note_groups_per_bar: int,
    max_sequence: int,
    max_simultaneous_notes: int,
    min_note_count: int,
    min_unique_pitch_count: int,
    min_phrase_coverage_ratio: float,
    max_dead_air_ratio: float,
    max_files: int,
    window_bars: int,
    window_stride_bars: int,
    min_window_target_notes: int,
    motif_length: int,
    max_bar_span: int,
    max_records: int,
    template_top_n: int,
) -> dict[str, Any]:
    context_summary = validate_context_source(context_report)
    chord_progression = derive_chord_progression(context_report, target_bars=int(bars))
    loaded_template_report, template_command = load_or_extract_template_report(
        output_dir=output_dir,
        run_id=run_id,
        input_dir=input_dir,
        max_files=max_files,
        window_bars=window_bars,
        window_stride_bars=window_stride_bars,
        min_window_target_notes=min_window_target_notes,
        motif_length=motif_length,
        max_bar_span=max_bar_span,
        max_records=max_records,
        template_top_n=template_top_n,
        template_report=template_report,
    )
    template_summary = validate_template_report(loaded_template_report)
    candidates, top_candidates = generate_phrase_bank_candidates(
        template_report=loaded_template_report,
        chord_progression=chord_progression,
        output_dir=output_dir,
        modes=modes,
        candidate_count=candidate_count,
        export_top_midi_count=export_top_midi_count,
        seed_start=seed_start,
        bpm=bpm,
        bars=bars,
        density=density,
        energy=energy,
        note_groups_per_bar=note_groups_per_bar,
        max_sequence=max_sequence,
        max_simultaneous_notes=max_simultaneous_notes,
        min_note_count=min_note_count,
        min_unique_pitch_count=min_unique_pitch_count,
        min_phrase_coverage_ratio=min_phrase_coverage_ratio,
        max_dead_air_ratio=max_dead_air_ratio,
    )
    qualified = [row for row in candidates if bool(row.get("contract_gate_passed"))]
    exported_qualified = [row for row in top_candidates if bool(row.get("export_contract_gate_passed"))]
    export_ready = len(top_candidates) >= int(export_top_midi_count) and len(exported_qualified) >= int(export_top_midi_count)
    best = top_candidates[0] if top_candidates else {}
    best_metrics = _dict(best.get("exported_metrics")) or _dict(best.get("metrics"))
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "issue_number": int(issue_number),
        "boundary": BOUNDARY,
        "context_summary": context_summary,
        "template_summary": template_summary,
        "template_command": template_command,
        "input_context": {
            "chord_progression": chord_progression,
            "bpm": int(bpm),
            "bars": int(bars),
            "density": density,
            "energy": energy,
        },
        "generation_config": {
            "generation_source": "phrase_bank_data_motif_retrieval",
            "modes": list(modes),
            "candidate_count": int(candidate_count),
            "export_top_midi_count": int(export_top_midi_count),
            "seed_start": int(seed_start),
            "note_groups_per_bar": int(note_groups_per_bar),
            "max_sequence": int(max_sequence),
            "model_checkpoint_generation_used": False,
            "fallback_role": "quality_floor_candidate_path",
        },
        "objective_gate": {
            "min_note_count": int(min_note_count),
            "min_unique_pitch_count": int(min_unique_pitch_count),
            "max_simultaneous_notes": int(max_simultaneous_notes),
            "min_phrase_coverage_ratio": float(min_phrase_coverage_ratio),
            "max_dead_air_ratio": float(max_dead_air_ratio),
        },
        "candidates": candidates,
        "top_candidates": top_candidates,
        "summary": {
            "candidate_count": int(len(candidates)),
            "qualified_candidate_count": int(len(qualified)),
            "exported_candidate_count": int(len(top_candidates)),
            "exported_qualified_candidate_count": int(len(exported_qualified)),
            "best_note_count": _int(best_metrics.get("note_count")),
            "best_unique_pitch_count": _int(best_metrics.get("unique_pitch_count")),
            "best_max_simultaneous_notes": _int(best_metrics.get("max_simultaneous_notes")),
            "best_dead_air_ratio": _float(best_metrics.get("dead_air_ratio")),
            "best_phrase_coverage_ratio": _float(best_metrics.get("phrase_coverage_ratio")),
        },
        "readiness": {
            "boundary": BOUNDARY,
            "phrase_bank_template_extracted": True,
            "phrase_bank_retrieval_baseline_completed": True,
            "ranked_midi_candidates_exported": bool(export_ready),
            "midi_to_solo_mvp_claimed": False,
            "phrase_bank_musical_quality_claimed": False,
            "model_checkpoint_generation_quality_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
            "human_audio_preference_claimed": False,
        },
        "decision": {
            "current_boundary": BOUNDARY,
            "next_boundary": NEXT_BOUNDARY,
            "auto_progress_allowed": True,
            "critical_user_input_required": False,
            "reason": "phrase-bank retrieval baseline exported ranked MIDI candidates; audio render/listening review remains separate",
        },
        "not_proven": [
            "human_audio_preference",
            "phrase_bank_musical_quality",
            "model_checkpoint_direct_quality",
            "broad_trained_model_quality",
            "brad_style_adaptation",
        ],
        "next_recommended_issue": "Stage B MIDI-to-solo phrase-bank audio render package",
    }


def validate_phrase_bank_retrieval_report(
    report: dict[str, Any],
    *,
    expected_boundary: str | None,
    expected_next_boundary: str | None,
    require_exported_candidates: bool,
    require_no_final_claim: bool,
    min_exported_candidates: int,
    min_note_count: int,
    min_unique_pitch_count: int,
    max_simultaneous_notes: int,
) -> dict[str, Any]:
    boundary = str(report.get("boundary") or "")
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    summary = _dict(report.get("summary"))
    template_summary = _dict(report.get("template_summary"))
    top_candidates = _list(report.get("top_candidates"))
    if expected_boundary and boundary != expected_boundary:
        raise StageBMidiToSoloPhraseBankRetrievalBaselineError(f"expected boundary {expected_boundary}, got {boundary}")
    next_boundary = str(decision.get("next_boundary") or "")
    if expected_next_boundary and next_boundary != expected_next_boundary:
        raise StageBMidiToSoloPhraseBankRetrievalBaselineError(
            f"expected next boundary {expected_next_boundary}, got {next_boundary}"
        )
    if _int(template_summary.get("motif_count")) <= 0:
        raise StageBMidiToSoloPhraseBankRetrievalBaselineError("motif template evidence required")
    if require_exported_candidates and not bool(readiness.get("ranked_midi_candidates_exported", False)):
        raise StageBMidiToSoloPhraseBankRetrievalBaselineError("ranked MIDI candidates must be exported")
    if _int(summary.get("exported_candidate_count")) < int(min_exported_candidates):
        raise StageBMidiToSoloPhraseBankRetrievalBaselineError("exported candidate count below threshold")
    for row in top_candidates[: int(min_exported_candidates)]:
        candidate = _dict(row)
        if not Path(str(candidate.get("export_midi_path") or "")).exists():
            raise StageBMidiToSoloPhraseBankRetrievalBaselineError("exported MIDI path missing")
        if not bool(candidate.get("export_contract_gate_passed", False)):
            raise StageBMidiToSoloPhraseBankRetrievalBaselineError("top candidate export contract gate failed")
        metrics = _dict(candidate.get("exported_metrics"))
        if _int(metrics.get("note_count")) < int(min_note_count):
            raise StageBMidiToSoloPhraseBankRetrievalBaselineError("top candidate note count below threshold")
        if _int(metrics.get("unique_pitch_count")) < int(min_unique_pitch_count):
            raise StageBMidiToSoloPhraseBankRetrievalBaselineError("top candidate unique pitch count below threshold")
        if _int(metrics.get("max_simultaneous_notes")) > int(max_simultaneous_notes):
            raise StageBMidiToSoloPhraseBankRetrievalBaselineError("top candidate simultaneous note limit failed")
    if bool(decision.get("critical_user_input_required", True)):
        raise StageBMidiToSoloPhraseBankRetrievalBaselineError("critical user input should not be required")
    if require_no_final_claim:
        blocked = [
            "midi_to_solo_mvp_claimed",
            "phrase_bank_musical_quality_claimed",
            "model_checkpoint_generation_quality_claimed",
            "broad_trained_model_quality_claimed",
            "brad_style_adaptation_claimed",
            "human_audio_preference_claimed",
        ]
        claimed = [name for name in blocked if bool(readiness.get(name, True))]
        if claimed:
            raise StageBMidiToSoloPhraseBankRetrievalBaselineError(f"unexpected final claim: {claimed}")
    return {
        "boundary": boundary,
        "next_boundary": next_boundary,
        "phrase_bank_template_extracted": bool(readiness.get("phrase_bank_template_extracted", False)),
        "phrase_bank_retrieval_baseline_completed": bool(
            readiness.get("phrase_bank_retrieval_baseline_completed", False)
        ),
        "ranked_midi_candidates_exported": bool(readiness.get("ranked_midi_candidates_exported", False)),
        "source_record_count": _int(template_summary.get("source_record_count")),
        "motif_count": _int(template_summary.get("motif_count")),
        "candidate_count": _int(summary.get("candidate_count")),
        "qualified_candidate_count": _int(summary.get("qualified_candidate_count")),
        "exported_candidate_count": _int(summary.get("exported_candidate_count")),
        "exported_qualified_candidate_count": _int(summary.get("exported_qualified_candidate_count")),
        "best_note_count": _int(summary.get("best_note_count")),
        "best_unique_pitch_count": _int(summary.get("best_unique_pitch_count")),
        "best_max_simultaneous_notes": _int(summary.get("best_max_simultaneous_notes")),
        "best_dead_air_ratio": _float(summary.get("best_dead_air_ratio")),
        "best_phrase_coverage_ratio": _float(summary.get("best_phrase_coverage_ratio")),
        "midi_to_solo_mvp_claimed": bool(readiness.get("midi_to_solo_mvp_claimed", True)),
        "human_audio_preference_claimed": bool(readiness.get("human_audio_preference_claimed", True)),
        "critical_user_input_required": bool(decision.get("critical_user_input_required", True)),
        "next_recommended_issue": str(report.get("next_recommended_issue") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    summary = report["summary"]
    template = report["template_summary"]
    gate = report["objective_gate"]
    readiness = report["readiness"]
    decision = report["decision"]
    config = report["generation_config"]
    context = report["input_context"]
    lines = [
        "# Stage B MIDI-to-Solo Phrase-Bank Retrieval Baseline",
        "",
        "## Summary",
        "",
        f"- boundary: `{report['boundary']}`",
        f"- next boundary: `{decision['next_boundary']}`",
        f"- generation source: `{config['generation_source']}`",
        f"- ranked MIDI candidates exported: `{_bool_token(readiness['ranked_midi_candidates_exported'])}`",
        f"- MIDI-to-solo MVP claimed: `{_bool_token(readiness['midi_to_solo_mvp_claimed'])}`",
        f"- human/audio preference claimed: `{_bool_token(readiness['human_audio_preference_claimed'])}`",
        "",
        "## Input Context",
        "",
        f"- chord progression: `{', '.join(context['chord_progression'])}`",
        f"- bars: `{context['bars']}`",
        f"- bpm: `{context['bpm']}`",
        "",
        "## Phrase Bank",
        "",
        f"- source records: `{template['source_record_count']}`",
        f"- motif count: `{template['motif_count']}`",
        f"- unique rhythm templates: `{template['unique_rhythm_template_count']}`",
        f"- unique contour templates: `{template['unique_contour_template_count']}`",
        "",
        "## Objective Gate",
        "",
        f"- min note count: `{gate['min_note_count']}`",
        f"- min unique pitch count: `{gate['min_unique_pitch_count']}`",
        f"- max simultaneous notes: `{gate['max_simultaneous_notes']}`",
        f"- min phrase coverage ratio: `{gate['min_phrase_coverage_ratio']}`",
        f"- max dead-air ratio: `{gate['max_dead_air_ratio']}`",
        "",
        "## Candidate Summary",
        "",
        f"- candidate count: `{summary['candidate_count']}`",
        f"- qualified candidate count: `{summary['qualified_candidate_count']}`",
        f"- exported candidate count: `{summary['exported_candidate_count']}`",
        f"- exported qualified candidate count: `{summary['exported_qualified_candidate_count']}`",
        f"- best note count: `{summary['best_note_count']}`",
        f"- best unique pitch count: `{summary['best_unique_pitch_count']}`",
        f"- best max simultaneous notes: `{summary['best_max_simultaneous_notes']}`",
        f"- best dead-air ratio: `{summary['best_dead_air_ratio']}`",
        f"- best phrase coverage ratio: `{summary['best_phrase_coverage_ratio']}`",
        "",
        "## Exported MIDI",
        "",
    ]
    for row in report["top_candidates"]:
        metrics = row["exported_metrics"]
        lines.append(
            f"- rank `{row['rank']}` mode `{row['mode']}` seed `{row['sample_seed']}`: "
            f"`{row['export_midi_path']}`, notes `{metrics['note_count']}`, "
            f"unique pitches `{metrics['unique_pitch_count']}`, dead-air `{metrics['dead_air_ratio']}`"
        )
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            f"- phrase-bank musical quality claimed: `{_bool_token(readiness['phrase_bank_musical_quality_claimed'])}`",
            f"- model checkpoint generation quality claimed: `{_bool_token(readiness['model_checkpoint_generation_quality_claimed'])}`",
            f"- broad trained-model quality claimed: `{_bool_token(readiness['broad_trained_model_quality_claimed'])}`",
            f"- Brad style adaptation claimed: `{_bool_token(readiness['brad_style_adaptation_claimed'])}`",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MIDI-to-solo phrase-bank retrieval baseline")
    parser.add_argument("--context_report", type=str, required=True)
    parser.add_argument("--output_root", type=str, default="outputs/stage_b_midi_to_solo_phrase_bank_retrieval_baseline")
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--issue_number", type=int, default=632)
    parser.add_argument("--input_dir", type=str, default="./midi_dataset/midi/studio")
    parser.add_argument("--modes", type=str, default=",".join(DEFAULT_MODES))
    parser.add_argument("--candidate_count", type=int, default=9)
    parser.add_argument("--export_top_midi_count", type=int, default=3)
    parser.add_argument("--seed_start", type=int, default=632)
    parser.add_argument("--bpm", type=int, default=120)
    parser.add_argument("--bars", type=int, default=8)
    parser.add_argument("--density", type=str, default="medium")
    parser.add_argument("--energy", type=str, default="mid")
    parser.add_argument("--note_groups_per_bar", type=int, default=8)
    parser.add_argument("--max_sequence", type=int, default=384)
    parser.add_argument("--max_simultaneous_notes", type=int, default=1)
    parser.add_argument("--min_note_count", type=int, default=24)
    parser.add_argument("--min_unique_pitch_count", type=int, default=8)
    parser.add_argument("--min_phrase_coverage_ratio", type=float, default=0.75)
    parser.add_argument("--max_dead_air_ratio", type=float, default=0.65)
    parser.add_argument("--max_files", type=int, default=4)
    parser.add_argument("--window_bars", type=int, default=8)
    parser.add_argument("--window_stride_bars", type=int, default=4)
    parser.add_argument("--min_window_target_notes", type=int, default=16)
    parser.add_argument("--motif_length", type=int, default=4)
    parser.add_argument("--max_bar_span", type=int, default=2)
    parser.add_argument("--max_records", type=int, default=64)
    parser.add_argument("--template_top_n", type=int, default=32)
    parser.add_argument("--expected_boundary", type=str, default="")
    parser.add_argument("--expected_next_boundary", type=str, default="")
    parser.add_argument("--require_exported_candidates", action="store_true")
    parser.add_argument("--require_no_final_claim", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    report = build_phrase_bank_retrieval_report(
        context_report=read_json(Path(args.context_report)),
        output_dir=output_dir,
        run_id=run_id,
        issue_number=int(args.issue_number),
        template_report=None,
        input_dir=Path(args.input_dir),
        modes=parse_modes(args.modes),
        candidate_count=int(args.candidate_count),
        export_top_midi_count=int(args.export_top_midi_count),
        seed_start=int(args.seed_start),
        bpm=int(args.bpm),
        bars=int(args.bars),
        density=args.density,
        energy=args.energy,
        note_groups_per_bar=int(args.note_groups_per_bar),
        max_sequence=int(args.max_sequence),
        max_simultaneous_notes=int(args.max_simultaneous_notes),
        min_note_count=int(args.min_note_count),
        min_unique_pitch_count=int(args.min_unique_pitch_count),
        min_phrase_coverage_ratio=float(args.min_phrase_coverage_ratio),
        max_dead_air_ratio=float(args.max_dead_air_ratio),
        max_files=int(args.max_files),
        window_bars=int(args.window_bars),
        window_stride_bars=int(args.window_stride_bars),
        min_window_target_notes=int(args.min_window_target_notes),
        motif_length=int(args.motif_length),
        max_bar_span=int(args.max_bar_span),
        max_records=int(args.max_records),
        template_top_n=int(args.template_top_n),
    )
    summary = validate_phrase_bank_retrieval_report(
        report,
        expected_boundary=str(args.expected_boundary or ""),
        expected_next_boundary=str(args.expected_next_boundary or ""),
        require_exported_candidates=bool(args.require_exported_candidates),
        require_no_final_claim=bool(args.require_no_final_claim),
        min_exported_candidates=int(args.export_top_midi_count),
        min_note_count=int(args.min_note_count),
        min_unique_pitch_count=int(args.min_unique_pitch_count),
        max_simultaneous_notes=int(args.max_simultaneous_notes),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "stage_b_midi_to_solo_phrase_bank_retrieval_baseline.json", report)
    write_json(
        output_dir / "stage_b_midi_to_solo_phrase_bank_retrieval_baseline_validation_summary.json",
        summary,
    )
    markdown = markdown_report(report)
    write_text(output_dir / "stage_b_midi_to_solo_phrase_bank_retrieval_baseline.md", markdown)
    if args.doc_path:
        write_text(Path(args.doc_path), markdown)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
