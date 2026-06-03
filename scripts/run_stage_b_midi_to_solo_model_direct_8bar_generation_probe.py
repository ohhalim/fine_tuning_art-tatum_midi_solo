"""Run the Stage B MIDI-to-solo model-direct 8-bar generation probe."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from scripts.assess_stage_b_generic_base_readiness import write_json, write_text  # noqa: E402
from scripts.consolidate_stage_b_midi_to_solo_model_direct_sequence_budget_repair_smoke import (  # noqa: E402
    BOUNDARY as SEQUENCE_BUDGET_BOUNDARY,
)
from scripts.consolidate_stage_b_midi_to_solo_model_direct_sequence_budget_repair_smoke import (  # noqa: E402
    NEXT_BOUNDARY as BOUNDARY,
)
from scripts.extract_stage_b_midi_to_solo_context import BOUNDARY as CONTEXT_BOUNDARY  # noqa: E402
from scripts.run_stage_b_midi_to_solo_conditioned_generation_probe import derive_chord_progression  # noqa: E402


class StageBMidiToSoloModelDirect8BarGenerationProbeError(ValueError):
    pass


SCALE_SMOKE_BOUNDARY = "stage_b_generic_base_training_scale_smoke"
PASS_NEXT_BOUNDARY = "stage_b_midi_to_solo_model_direct_audio_render_package"
FAIL_NEXT_BOUNDARY = "stage_b_midi_to_solo_model_direct_monophonic_overlap_repair"
SCHEMA_VERSION = "stage_b_midi_to_solo_model_direct_8bar_generation_probe_v1"


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise StageBMidiToSoloModelDirect8BarGenerationProbeError(f"report missing: {path}")
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


def run_command(command: Sequence[str]) -> dict[str, Any]:
    completed = subprocess.run(
        list(command),
        cwd=str(ROOT_DIR),
        check=False,
        text=True,
        capture_output=True,
    )
    return {
        "cmd": list(command),
        "returncode": int(completed.returncode),
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
    }


def summarize_sequence_budget_repair(report: dict[str, Any]) -> dict[str, Any]:
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    result = _dict(report.get("repair_result"))
    boundary = str(report.get("boundary") or readiness.get("boundary") or "")
    if boundary != SEQUENCE_BUDGET_BOUNDARY:
        raise StageBMidiToSoloModelDirect8BarGenerationProbeError("sequence budget repair boundary required")
    if str(decision.get("next_boundary") or "") != BOUNDARY:
        raise StageBMidiToSoloModelDirect8BarGenerationProbeError("sequence budget repair must route to 8-bar probe")
    if not bool(readiness.get("model_direct_8bar_generation_probe_ready", False)):
        raise StageBMidiToSoloModelDirect8BarGenerationProbeError("direct 8-bar probe readiness required")
    return {
        "boundary": boundary,
        "previous_max_sequence": _int(result.get("previous_max_sequence")),
        "repaired_max_sequence": _int(result.get("repaired_max_sequence")),
        "minimum_contract_tokens": _int(result.get("minimum_contract_tokens")),
        "repaired_direct_note_capacity": _int(result.get("repaired_direct_note_capacity")),
        "target_min_note_count": _int(result.get("target_min_note_count")),
        "model_direct_8bar_generation_probe_ready": bool(
            readiness.get("model_direct_8bar_generation_probe_ready", False)
        ),
        "model_direct_generation_quality_claimed": bool(
            readiness.get("model_direct_generation_quality_claimed", True)
        ),
        "midi_to_solo_musical_quality_claimed": bool(
            readiness.get("midi_to_solo_musical_quality_claimed", True)
        ),
        "human_audio_preference_claimed": bool(readiness.get("human_audio_preference_claimed", True)),
        "broad_trained_model_quality_claimed": bool(
            readiness.get("broad_trained_model_quality_claimed", True)
        ),
        "brad_style_adaptation_claimed": bool(readiness.get("brad_style_adaptation_claimed", True)),
    }


def summarize_context(report: dict[str, Any], *, target_bars: int) -> dict[str, Any]:
    readiness = _dict(report.get("readiness"))
    summary = _dict(report.get("summary"))
    context = _dict(report.get("context"))
    boundary = str(report.get("boundary") or readiness.get("boundary") or "")
    if boundary != CONTEXT_BOUNDARY:
        raise StageBMidiToSoloModelDirect8BarGenerationProbeError("context extraction boundary required")
    if not bool(readiness.get("context_extraction_completed", False)):
        raise StageBMidiToSoloModelDirect8BarGenerationProbeError("context extraction completion required")
    chord_progression = derive_chord_progression(report, int(target_bars))
    bars = _list(context.get("bar_contexts"))
    tempo_values = [_float(_dict(item).get("tempo")) for item in bars if _float(_dict(item).get("tempo")) > 0]
    bpm = int(round(tempo_values[0])) if tempo_values else 120
    return {
        "boundary": boundary,
        "context_bars": _int(summary.get("context_bars")),
        "context_event_count": _int(summary.get("context_event_count")),
        "unknown_chord_bar_count": _int(summary.get("unknown_chord_bar_count")),
        "low_confidence_bar_count": _int(summary.get("low_confidence_bar_count")),
        "bpm": int(bpm),
        "chord_progression": chord_progression,
    }


def summarize_repaired_scale_smoke(report: dict[str, Any]) -> dict[str, Any]:
    readiness = _dict(report.get("readiness"))
    training_config = _dict(report.get("training_config"))
    training = _dict(report.get("training"))
    artifacts = _dict(report.get("artifacts"))
    boundary = str(readiness.get("boundary") or report.get("boundary") or "")
    if boundary != SCALE_SMOKE_BOUNDARY:
        raise StageBMidiToSoloModelDirect8BarGenerationProbeError("scale-smoke boundary required")
    if not bool(readiness.get("training_scale_smoke_passed", False)):
        raise StageBMidiToSoloModelDirect8BarGenerationProbeError("scale-smoke pass required")
    checkpoint_dir = Path(str(report.get("checkpoint_dir") or ""))
    if not (checkpoint_dir / "checkpoint_epoch1.pt").exists():
        raise StageBMidiToSoloModelDirect8BarGenerationProbeError("checkpoint_epoch1.pt required")
    return {
        "boundary": boundary,
        "checkpoint_dir": str(checkpoint_dir),
        "max_sequence": _int(training_config.get("max_sequence")),
        "best_validation_loss": _float(training.get("best_validation_loss")),
        "checkpoint_count": _int(artifacts.get("checkpoint_count")),
        "lora_weights_exists": bool(artifacts.get("lora_weights_exists", False)),
        "broad_trained_model_quality_claimed": bool(
            readiness.get("broad_trained_model_quality_claimed", True)
        ),
        "brad_style_adaptation_claimed": bool(readiness.get("brad_style_adaptation_claimed", True)),
    }


def build_generation_command(
    *,
    args: argparse.Namespace,
    checkpoint_dir: Path,
    generation_output_root: Path,
    generation_run_id: str,
    context_summary: dict[str, Any],
) -> list[str]:
    command = [
        sys.executable,
        "scripts/run_stage_b_generation_probe.py",
        "--output_root",
        str(generation_output_root),
        "--run_id",
        generation_run_id,
        "--checkpoint_dir",
        str(checkpoint_dir),
        "--skip_prepare",
        "--skip_train",
        "--issue_number",
        str(args.issue_number),
        "--max_sequence",
        str(args.max_sequence),
        "--num_samples",
        str(args.num_samples),
        "--seed",
        str(args.seed),
        "--bpm",
        str(context_summary["bpm"]),
        "--bars",
        str(args.target_bars),
        "--chords",
        ",".join(context_summary["chord_progression"]),
        "--generation_mode",
        "constrained",
        "--constrained_note_groups_per_bar",
        str(args.note_groups_per_bar),
        "--coverage_aware_positions",
        "--chord_aware_pitches",
        "--chord_pitch_mode",
        args.chord_pitch_mode,
        "--postprocess_overlap",
        "--max_simultaneous_notes",
        str(args.max_simultaneous_notes),
        "--min_valid_samples",
        str(args.min_valid_samples),
        "--min_strict_valid_samples",
        str(args.min_strict_valid_samples),
    ]
    if bool(getattr(args, "cap_duration_to_next_position", False)):
        command.append("--cap_duration_to_next_position")
    return command


def generation_sample_paths(generation_report: dict[str, Any]) -> list[str]:
    paths: list[str] = []
    for sample in _list(generation_report.get("samples")):
        path = str(_dict(sample).get("midi_path") or "")
        if path:
            paths.append(path)
    return paths


def summarize_generation(generation_report: dict[str, Any]) -> dict[str, Any]:
    summary = _dict(generation_report.get("summary"))
    samples = [_dict(item) for item in _list(generation_report.get("samples"))]
    note_group_counts = [_int(_dict(sample.get("grammar")).get("complete_note_groups")) for sample in samples]
    postprocess_after_counts = [
        _int(_dict(sample.get("postprocess")).get("after_note_count"))
        for sample in samples
        if sample.get("postprocess") is not None
    ]
    postprocess_before_counts = [
        _int(_dict(sample.get("postprocess")).get("before_note_count"))
        for sample in samples
        if sample.get("postprocess") is not None
    ]
    metric_note_counts = [_int(_dict(sample.get("metrics")).get("note_count")) for sample in samples]
    paths = generation_sample_paths(generation_report)
    return {
        "sample_count": _int(summary.get("sample_count")),
        "valid_sample_count": _int(summary.get("valid_sample_count")),
        "strict_valid_sample_count": _int(summary.get("strict_valid_sample_count")),
        "grammar_gate_sample_count": _int(summary.get("grammar_gate_sample_count")),
        "passed_generation_gate": bool(summary.get("passed_generation_gate", False)),
        "passed_grammar_gate": bool(summary.get("passed_grammar_gate", False)),
        "passed_strict_review_gate": bool(summary.get("passed_strict_review_gate", False)),
        "collapse_warning_sample_count": _int(summary.get("collapse_warning_sample_count")),
        "collapse_warning_sample_rate": _float(summary.get("collapse_warning_sample_rate")),
        "avg_postprocess_removal_ratio": _float(summary.get("avg_postprocess_removal_ratio")),
        "max_postprocess_removal_ratio": _float(summary.get("max_postprocess_removal_ratio")),
        "avg_onset_coverage_ratio": _float(summary.get("avg_onset_coverage_ratio")),
        "avg_sustained_coverage_ratio": _float(summary.get("avg_sustained_coverage_ratio")),
        "diagnostic_failure_reasons": _dict(summary.get("diagnostic_failure_reasons")),
        "strict_failure_reasons": _dict(summary.get("strict_failure_reasons")),
        "midi_paths": paths,
        "all_midi_paths_exist": all(Path(path).exists() for path in paths) if paths else False,
        "min_pre_postprocess_note_groups": min(note_group_counts) if note_group_counts else 0,
        "max_pre_postprocess_note_groups": max(note_group_counts) if note_group_counts else 0,
        "min_postprocess_note_count": min(postprocess_after_counts or metric_note_counts) if (postprocess_after_counts or metric_note_counts) else 0,
        "max_postprocess_note_count": max(postprocess_after_counts or metric_note_counts) if (postprocess_after_counts or metric_note_counts) else 0,
        "min_pre_postprocess_note_count": min(postprocess_before_counts) if postprocess_before_counts else 0,
        "max_pre_postprocess_note_count": max(postprocess_before_counts) if postprocess_before_counts else 0,
    }


def build_direct_8bar_generation_probe_report(
    *,
    sequence_budget_repair: dict[str, Any],
    context_report: dict[str, Any],
    repaired_training_scale_smoke: dict[str, Any],
    generation_result: dict[str, Any],
    generation_report: dict[str, Any],
    generation_report_path: Path,
    output_dir: Path,
    issue_number: int,
    target_bars: int,
    note_groups_per_bar: int,
) -> dict[str, Any]:
    sequence = summarize_sequence_budget_repair(sequence_budget_repair)
    context = summarize_context(context_report, target_bars=int(target_bars))
    scale = summarize_repaired_scale_smoke(repaired_training_scale_smoke)
    generation = summarize_generation(generation_report)
    generation_command_succeeded = _int(generation_result.get("returncode")) == 0
    direct_generated_midi_written = bool(
        generation_command_succeeded
        and generation["sample_count"] > 0
        and generation["all_midi_paths_exist"]
    )
    review_gate_passed = bool(generation["passed_generation_gate"] and generation["passed_strict_review_gate"])
    next_boundary = PASS_NEXT_BOUNDARY if review_gate_passed else FAIL_NEXT_BOUNDARY
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "issue_number": int(issue_number),
        "boundary": BOUNDARY,
        "source_boundaries": {
            "sequence_budget_repair": sequence["boundary"],
            "context": context["boundary"],
            "repaired_training_scale_smoke": scale["boundary"],
        },
        "sequence_budget_summary": sequence,
        "context_summary": context,
        "repaired_scale_smoke_summary": scale,
        "generation_config": {
            "generation_source": "model_checkpoint_direct_constrained",
            "target_bars": int(target_bars),
            "note_groups_per_bar": int(note_groups_per_bar),
            "target_min_note_count": int(sequence["target_min_note_count"]),
            "max_sequence": int(scale["max_sequence"]),
            "checkpoint_dir": scale["checkpoint_dir"],
        },
        "generation_report_path": str(generation_report_path),
        "generation_command": generation_result,
        "generation_summary": generation,
        "readiness": {
            "boundary": BOUNDARY,
            "model_direct_8bar_generation_probe_completed": bool(generation_command_succeeded),
            "direct_generated_midi_written": bool(direct_generated_midi_written),
            "direct_generation_grammar_gate_passed": bool(generation["passed_grammar_gate"]),
            "direct_generation_review_gate_passed": bool(review_gate_passed),
            "model_direct_generation_quality_claimed": False,
            "midi_to_solo_musical_quality_claimed": False,
            "human_audio_preference_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
        },
        "decision": {
            "current_boundary": BOUNDARY,
            "next_boundary": next_boundary,
            "auto_progress_allowed": True,
            "critical_user_input_required": False,
            "reason": (
                "direct checkpoint generation wrote 8-bar MIDI but review gate remains blocked by "
                "postprocess overlap/removal"
            )
            if not review_gate_passed
            else "direct checkpoint generation passed current review gate; audio render can be prepared",
        },
        "not_proven": [
            "model_checkpoint_direct_8bar_generation_quality",
            "human_audio_preference",
            "broad_trained_model_quality",
            "brad_style_adaptation",
        ],
        "next_recommended_issue": (
            "Stage B MIDI-to-solo model-direct audio render package"
            if review_gate_passed
            else "Stage B MIDI-to-solo model-direct monophonic overlap repair"
        ),
    }


def validate_direct_8bar_generation_probe_report(
    report: dict[str, Any],
    *,
    expected_boundary: str | None,
    expected_next_boundary: str | None,
    require_probe_completed: bool,
    require_generated_midi: bool,
    require_grammar_gate: bool,
    require_no_quality_claim: bool,
) -> dict[str, Any]:
    boundary = str(report.get("boundary") or "")
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    sequence = _dict(report.get("sequence_budget_summary"))
    scale = _dict(report.get("repaired_scale_smoke_summary"))
    generation = _dict(report.get("generation_summary"))
    command = _dict(report.get("generation_command"))
    if expected_boundary and boundary != expected_boundary:
        raise StageBMidiToSoloModelDirect8BarGenerationProbeError(
            f"expected boundary {expected_boundary}, got {boundary}"
        )
    if expected_next_boundary and str(decision.get("next_boundary") or "") != expected_next_boundary:
        raise StageBMidiToSoloModelDirect8BarGenerationProbeError("unexpected next boundary")
    if require_probe_completed and not bool(readiness.get("model_direct_8bar_generation_probe_completed", False)):
        raise StageBMidiToSoloModelDirect8BarGenerationProbeError("direct generation probe must complete")
    if _int(command.get("returncode")) != 0:
        raise StageBMidiToSoloModelDirect8BarGenerationProbeError("generation command must succeed")
    if require_generated_midi and not bool(readiness.get("direct_generated_midi_written", False)):
        raise StageBMidiToSoloModelDirect8BarGenerationProbeError("generated MIDI files required")
    if _int(generation.get("sample_count")) <= 0:
        raise StageBMidiToSoloModelDirect8BarGenerationProbeError("generated sample count required")
    for path in _list(generation.get("midi_paths")):
        if not Path(str(path)).exists():
            raise StageBMidiToSoloModelDirect8BarGenerationProbeError(f"generated MIDI missing: {path}")
    if require_grammar_gate and not bool(readiness.get("direct_generation_grammar_gate_passed", False)):
        raise StageBMidiToSoloModelDirect8BarGenerationProbeError("grammar gate should pass")
    if _int(generation.get("min_pre_postprocess_note_groups")) < _int(sequence.get("target_min_note_count")):
        raise StageBMidiToSoloModelDirect8BarGenerationProbeError("pre-postprocess note groups below target")
    if bool(decision.get("critical_user_input_required", True)):
        raise StageBMidiToSoloModelDirect8BarGenerationProbeError("critical user input should not be required")
    if require_no_quality_claim:
        blocked = [
            "model_direct_generation_quality_claimed",
            "midi_to_solo_musical_quality_claimed",
            "human_audio_preference_claimed",
            "broad_trained_model_quality_claimed",
            "brad_style_adaptation_claimed",
        ]
        claimed = [name for name in blocked if bool(readiness.get(name, True))]
        if claimed:
            raise StageBMidiToSoloModelDirect8BarGenerationProbeError(f"unexpected quality claim: {claimed}")
        upstream_claims = [
            bool(sequence.get("model_direct_generation_quality_claimed", True)),
            bool(sequence.get("midi_to_solo_musical_quality_claimed", True)),
            bool(sequence.get("human_audio_preference_claimed", True)),
            bool(sequence.get("broad_trained_model_quality_claimed", True)),
            bool(sequence.get("brad_style_adaptation_claimed", True)),
            bool(scale.get("broad_trained_model_quality_claimed", True)),
            bool(scale.get("brad_style_adaptation_claimed", True)),
        ]
        if any(upstream_claims):
            raise StageBMidiToSoloModelDirect8BarGenerationProbeError(
                "upstream quality claims must remain false"
            )
    return {
        "boundary": boundary,
        "next_boundary": str(decision.get("next_boundary") or ""),
        "generation_source": str(_dict(report.get("generation_config")).get("generation_source") or ""),
        "target_bars": _int(_dict(report.get("generation_config")).get("target_bars")),
        "note_groups_per_bar": _int(_dict(report.get("generation_config")).get("note_groups_per_bar")),
        "max_sequence": _int(_dict(report.get("generation_config")).get("max_sequence")),
        "sample_count": _int(generation.get("sample_count")),
        "grammar_gate_sample_count": _int(generation.get("grammar_gate_sample_count")),
        "valid_sample_count": _int(generation.get("valid_sample_count")),
        "strict_valid_sample_count": _int(generation.get("strict_valid_sample_count")),
        "direct_generated_midi_written": bool(readiness.get("direct_generated_midi_written", False)),
        "direct_generation_grammar_gate_passed": bool(readiness.get("direct_generation_grammar_gate_passed", False)),
        "direct_generation_review_gate_passed": bool(readiness.get("direct_generation_review_gate_passed", False)),
        "min_pre_postprocess_note_groups": _int(generation.get("min_pre_postprocess_note_groups")),
        "max_pre_postprocess_note_groups": _int(generation.get("max_pre_postprocess_note_groups")),
        "min_postprocess_note_count": _int(generation.get("min_postprocess_note_count")),
        "max_postprocess_note_count": _int(generation.get("max_postprocess_note_count")),
        "avg_postprocess_removal_ratio": _float(generation.get("avg_postprocess_removal_ratio")),
        "collapse_warning_sample_rate": _float(generation.get("collapse_warning_sample_rate")),
        "checkpoint_best_validation_loss": scale.get("best_validation_loss"),
        "model_direct_generation_quality_claimed": bool(
            readiness.get("model_direct_generation_quality_claimed", True)
        ),
        "critical_user_input_required": bool(decision.get("critical_user_input_required", True)),
        "next_recommended_issue": str(report.get("next_recommended_issue") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    readiness = report["readiness"]
    decision = report["decision"]
    config = report["generation_config"]
    generation = report["generation_summary"]
    context = report["context_summary"]
    lines = [
        "# Stage B MIDI-to-Solo Model-Direct 8-Bar Generation Probe",
        "",
        "## Summary",
        "",
        f"- boundary: `{report['boundary']}`",
        f"- next boundary: `{decision['next_boundary']}`",
        f"- generation source: `{config['generation_source']}`",
        f"- direct generated MIDI written: `{_bool_token(readiness['direct_generated_midi_written'])}`",
        f"- direct generation grammar gate passed: `{_bool_token(readiness['direct_generation_grammar_gate_passed'])}`",
        f"- direct generation review gate passed: `{_bool_token(readiness['direct_generation_review_gate_passed'])}`",
        f"- model-direct generation quality claimed: `{_bool_token(readiness['model_direct_generation_quality_claimed'])}`",
        "",
        "## Context",
        "",
        f"- target bars: `{config['target_bars']}`",
        f"- BPM: `{context['bpm']}`",
        f"- chord progression: `{', '.join(context['chord_progression'])}`",
        f"- low-confidence chord bars: `{context['low_confidence_bar_count']}`",
        "",
        "## Generation",
        "",
        f"- max sequence: `{config['max_sequence']}`",
        f"- note groups per bar: `{config['note_groups_per_bar']}`",
        f"- sample count: `{generation['sample_count']}`",
        f"- grammar gate sample count: `{generation['grammar_gate_sample_count']}`",
        f"- valid sample count: `{generation['valid_sample_count']}`",
        f"- strict valid sample count: `{generation['strict_valid_sample_count']}`",
        f"- min pre-postprocess note groups: `{generation['min_pre_postprocess_note_groups']}`",
        f"- min postprocess note count: `{generation['min_postprocess_note_count']}`",
        f"- max postprocess note count: `{generation['max_postprocess_note_count']}`",
        f"- avg postprocess removal ratio: `{generation['avg_postprocess_removal_ratio']}`",
        f"- collapse warning sample rate: `{generation['collapse_warning_sample_rate']}`",
        "",
        "## MIDI Paths",
        "",
    ]
    for path in generation["midi_paths"]:
        lines.append(f"- `{path}`")
    lines.extend(["", "## Not Proven", ""])
    for item in report.get("not_proven", []):
        lines.append(f"- `{item}`")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MIDI-to-solo model-direct 8-bar generation probe")
    parser.add_argument("--sequence_budget_repair", type=str, required=True)
    parser.add_argument("--context_report", type=str, required=True)
    parser.add_argument("--repaired_training_scale_smoke", type=str, required=True)
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/stage_b_midi_to_solo_model_direct_8bar_generation_probe",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--issue_number", type=int, default=497)
    parser.add_argument("--target_bars", type=int, default=8)
    parser.add_argument("--note_groups_per_bar", type=int, default=3)
    parser.add_argument("--num_samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=497)
    parser.add_argument("--max_sequence", type=int, default=160)
    parser.add_argument("--chord_pitch_mode", type=str, default="tones_tensions")
    parser.add_argument("--max_simultaneous_notes", type=int, default=1)
    parser.add_argument("--min_valid_samples", type=int, default=1)
    parser.add_argument("--min_strict_valid_samples", type=int, default=1)
    parser.add_argument("--expected_boundary", type=str, default="")
    parser.add_argument("--expected_next_boundary", type=str, default="")
    parser.add_argument("--require_probe_completed", action="store_true")
    parser.add_argument("--require_generated_midi", action="store_true")
    parser.add_argument("--require_grammar_gate", action="store_true")
    parser.add_argument("--require_no_quality_claim", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    sequence_budget_repair = read_json(Path(args.sequence_budget_repair))
    context_report = read_json(Path(args.context_report))
    repaired_training_scale_smoke = read_json(Path(args.repaired_training_scale_smoke))
    context_summary = summarize_context(context_report, target_bars=int(args.target_bars))
    scale_summary = summarize_repaired_scale_smoke(repaired_training_scale_smoke)
    generation_output_root = output_dir / "generation_probe"
    generation_run_id = "model_direct_8bar"
    generation_result = run_command(
        build_generation_command(
            args=args,
            checkpoint_dir=Path(scale_summary["checkpoint_dir"]),
            generation_output_root=generation_output_root,
            generation_run_id=generation_run_id,
            context_summary=context_summary,
        )
    )
    generation_report_path = generation_output_root / generation_run_id / "report.json"
    generation_report = read_json(generation_report_path) if generation_report_path.exists() else {}
    report = build_direct_8bar_generation_probe_report(
        sequence_budget_repair=sequence_budget_repair,
        context_report=context_report,
        repaired_training_scale_smoke=repaired_training_scale_smoke,
        generation_result=generation_result,
        generation_report=generation_report,
        generation_report_path=generation_report_path,
        output_dir=output_dir,
        issue_number=int(args.issue_number),
        target_bars=int(args.target_bars),
        note_groups_per_bar=int(args.note_groups_per_bar),
    )
    summary = validate_direct_8bar_generation_probe_report(
        report,
        expected_boundary=str(args.expected_boundary or ""),
        expected_next_boundary=str(args.expected_next_boundary or ""),
        require_probe_completed=bool(args.require_probe_completed),
        require_generated_midi=bool(args.require_generated_midi),
        require_grammar_gate=bool(args.require_grammar_gate),
        require_no_quality_claim=bool(args.require_no_quality_claim),
    )
    write_json(output_dir / "stage_b_midi_to_solo_model_direct_8bar_generation_probe.json", report)
    write_json(output_dir / "stage_b_midi_to_solo_model_direct_8bar_generation_probe_validation_summary.json", summary)
    markdown = markdown_report(report)
    write_text(output_dir / "stage_b_midi_to_solo_model_direct_8bar_generation_probe.md", markdown)
    if args.doc_path:
        write_text(Path(args.doc_path), markdown)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
