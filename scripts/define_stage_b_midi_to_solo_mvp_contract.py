"""Define the Stage B MIDI-to-solo MVP input/output contract."""

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


class StageBMidiToSoloMvpContractError(ValueError):
    pass


BOUNDARY = "stage_b_midi_to_solo_mvp_input_contract"
NEXT_BOUNDARY = "stage_b_midi_to_solo_context_extraction_mvp"
TARGET_DATE = "2026-06-11"


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _bool_token(value: Any) -> str:
    return "true" if bool(value) else "false"


def build_contract_report(*, output_dir: Path, issue_number: int) -> dict[str, Any]:
    return {
        "schema_version": "stage_b_midi_to_solo_mvp_contract_v1",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "issue_number": int(issue_number),
        "boundary": BOUNDARY,
        "target_date": TARGET_DATE,
        "goal": {
            "user_goal": "input_midi_to_jazz_solo_midi",
            "mvp_definition": (
                "given an input MIDI file, extract musical context and export ranked monophonic "
                "jazz-solo MIDI candidates"
            ),
            "success_mode": "hybrid_model_constrained_decoder_candidate_ranking",
            "neural_only_required": False,
        },
        "input_contract": {
            "required_input": "midi_file_path",
            "supported_extensions": [".mid", ".midi"],
            "accepted_input_roles": [
                "chord_track_or_markers",
                "bass_track",
                "piano_or_comping_track",
                "melody_or_lead_track",
                "unlabeled_polyphonic_midi",
            ],
            "context_source_priority": [
                "explicit_chord_markers_or_chord_track",
                "bass_track_plus_pitch_class_window_inference",
                "polyphonic_pitch_class_window_inference",
                "user_supplied_chord_csv_optional_future",
            ],
            "required_context_fields": [
                "bar_index",
                "position_index",
                "tempo",
                "chord_root",
                "chord_quality",
                "next_chord_root",
                "next_chord_quality",
                "bass_note",
            ],
            "minimum_context_bars": 4,
            "target_context_bars": 8,
            "chord_confidence_fallback": "low_confidence_context_allowed_but_ranked_lower",
        },
        "output_contract": {
            "output_root": "outputs/stage_b_midi_to_solo_mvp",
            "candidate_count": 32,
            "export_top_midi_count": 3,
            "target_solo_bars": 8,
            "output_files": [
                "candidate_manifest.json",
                "candidate_ranking.md",
                "top_1.mid",
                "top_2.mid",
                "top_3.mid",
            ],
            "optional_files": [
                "top_1.wav",
                "top_2.wav",
                "top_3.wav",
            ],
        },
        "generation_stack": {
            "primary_path": "generic_base_checkpoint_conditioned_generation",
            "conditioning": [
                "bar_position",
                "current_chord",
                "next_chord",
                "bass_note",
                "input_density_summary",
            ],
            "decoder_guards": [
                "monophonic_solo_line",
                "range_guard",
                "max_interval_guard",
                "long_note_guard",
                "dead_air_guard",
                "adjacent_repeat_guard",
                "chord_tone_or_tension_landing_bonus",
            ],
            "candidate_ranking": [
                "objective_gate_pass",
                "dead_air_score",
                "unique_pitch_score",
                "interval_score",
                "chord_fit_score",
                "phrase_coverage_score",
                "fallback_penalty",
            ],
            "fallback_path": "phrase_retrieval_data_motif_hybrid",
            "fallback_trigger": "zero_ranked_candidates_after_two_conditioned_generation_attempts",
        },
        "objective_gate": {
            "min_note_count": 24,
            "min_unique_pitch_count": 8,
            "max_dead_air_ratio": 0.5,
            "max_long_note_ratio": 0.5,
            "max_simultaneous_notes": 1,
            "max_interval_semitones": 12,
            "min_phrase_coverage_ratio": 0.75,
            "required_top_candidate_count": 1,
        },
        "run_plan": [
            {
                "date": "2026-06-03",
                "boundary": BOUNDARY,
                "deliverable": "input/output contract and validation harness",
            },
            {
                "date": "2026-06-04",
                "boundary": NEXT_BOUNDARY,
                "deliverable": "MIDI context extractor MVP",
            },
            {
                "date": "2026-06-05",
                "boundary": "stage_b_midi_to_solo_training_resource_probe",
                "deliverable": "near-full generic training resource probe",
            },
            {
                "date": "2026-06-06",
                "boundary": "stage_b_midi_to_solo_conditioned_generation_probe",
                "deliverable": "conditioned generation candidate set",
            },
            {
                "date": "2026-06-07",
                "boundary": "stage_b_midi_to_solo_constrained_decoder_ranking",
                "deliverable": "ranked MIDI candidates",
            },
            {
                "date": "2026-06-08",
                "boundary": "stage_b_midi_to_solo_retrieval_fallback",
                "deliverable": "phrase retrieval fallback if needed",
            },
            {
                "date": "2026-06-09",
                "boundary": "stage_b_midi_to_solo_cli_mvp",
                "deliverable": "generate_solo_from_midi CLI",
            },
            {
                "date": "2026-06-10",
                "boundary": "stage_b_midi_to_solo_audio_review_package",
                "deliverable": "MIDI/WAV review package",
            },
            {
                "date": TARGET_DATE,
                "boundary": "stage_b_midi_to_solo_final_package",
                "deliverable": "README, usage guide, result and limitation boundary",
            },
        ],
        "claim_boundary": {
            "midi_to_solo_mvp_claimed": False,
            "contract_defined": True,
            "full_training_completed": False,
            "brad_style_fine_tuning_completed": False,
            "human_audio_preference_claimed": False,
            "production_ready_improviser_claimed": False,
        },
        "decision": {
            "current_boundary": BOUNDARY,
            "next_boundary": NEXT_BOUNDARY,
            "auto_progress_allowed": True,
            "critical_user_input_required": False,
            "reason": (
                "MVP target is now input MIDI to ranked jazz-solo MIDI candidates; contract must be "
                "fixed before extractor, training, and inference work"
            ),
        },
        "references": [
            {
                "name": "MINGUS",
                "use": "jazz melodic line generation with chord, bass, position conditioning",
            },
            {
                "name": "REMI / Pop Music Transformer",
                "use": "bar, position, chord and tempo context for symbolic piano generation",
            },
            {
                "name": "Music Transformer",
                "use": "self-attention for continuation and motif-level structure",
            },
            {
                "name": "Chord-progression phrase retrieval",
                "use": "fallback path when neural generation produces no ranked candidate",
            },
        ],
        "next_recommended_issue": "Stage B MIDI-to-solo context extraction MVP",
    }


def validate_contract_report(
    report: dict[str, Any],
    *,
    expected_boundary: str | None,
    expected_next_boundary: str | None,
    require_fallback: bool,
    require_no_final_claim: bool,
) -> dict[str, Any]:
    input_contract = _dict(report.get("input_contract"))
    output_contract = _dict(report.get("output_contract"))
    generation_stack = _dict(report.get("generation_stack"))
    objective_gate = _dict(report.get("objective_gate"))
    decision = _dict(report.get("decision"))
    claim = _dict(report.get("claim_boundary"))

    boundary = str(report.get("boundary") or "")
    next_boundary = str(decision.get("next_boundary") or "")
    if expected_boundary and boundary != expected_boundary:
        raise StageBMidiToSoloMvpContractError(f"expected boundary {expected_boundary}, got {boundary}")
    if expected_next_boundary and next_boundary != expected_next_boundary:
        raise StageBMidiToSoloMvpContractError(
            f"expected next boundary {expected_next_boundary}, got {next_boundary}"
        )
    if ".mid" not in list(input_contract.get("supported_extensions") or []):
        raise StageBMidiToSoloMvpContractError("MIDI extension support required")
    if _int(input_contract.get("minimum_context_bars")) < 4:
        raise StageBMidiToSoloMvpContractError("minimum context bars must be at least 4")
    if _int(output_contract.get("export_top_midi_count")) < 3:
        raise StageBMidiToSoloMvpContractError("at least 3 exported MIDI candidates required")
    if _int(output_contract.get("candidate_count")) < 16:
        raise StageBMidiToSoloMvpContractError("candidate count must be at least 16")
    if require_fallback and not str(generation_stack.get("fallback_path") or ""):
        raise StageBMidiToSoloMvpContractError("fallback path required")
    if _int(objective_gate.get("max_simultaneous_notes")) != 1:
        raise StageBMidiToSoloMvpContractError("MVP solo output must be monophonic")
    if _int(objective_gate.get("required_top_candidate_count")) < 1:
        raise StageBMidiToSoloMvpContractError("at least one ranked candidate required")
    if bool(decision.get("critical_user_input_required", True)):
        raise StageBMidiToSoloMvpContractError("contract should not require critical user input")
    if require_no_final_claim:
        blocked = [
            "midi_to_solo_mvp_claimed",
            "full_training_completed",
            "brad_style_fine_tuning_completed",
            "human_audio_preference_claimed",
            "production_ready_improviser_claimed",
        ]
        claimed = [name for name in blocked if bool(claim.get(name, True))]
        if claimed:
            raise StageBMidiToSoloMvpContractError(f"unexpected final claim: {claimed}")
    return {
        "boundary": boundary,
        "next_boundary": next_boundary,
        "target_date": str(report.get("target_date") or ""),
        "candidate_count": _int(output_contract.get("candidate_count")),
        "export_top_midi_count": _int(output_contract.get("export_top_midi_count")),
        "target_solo_bars": _int(output_contract.get("target_solo_bars")),
        "min_note_count": _int(objective_gate.get("min_note_count")),
        "min_unique_pitch_count": _int(objective_gate.get("min_unique_pitch_count")),
        "max_simultaneous_notes": _int(objective_gate.get("max_simultaneous_notes")),
        "fallback_path": str(generation_stack.get("fallback_path") or ""),
        "midi_to_solo_mvp_claimed": bool(claim.get("midi_to_solo_mvp_claimed", True)),
        "brad_style_fine_tuning_completed": bool(
            claim.get("brad_style_fine_tuning_completed", True)
        ),
        "critical_user_input_required": bool(decision.get("critical_user_input_required", True)),
        "next_recommended_issue": str(report.get("next_recommended_issue") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    goal = report["goal"]
    input_contract = report["input_contract"]
    output_contract = report["output_contract"]
    generation_stack = report["generation_stack"]
    objective_gate = report["objective_gate"]
    decision = report["decision"]
    claim = report["claim_boundary"]
    lines = [
        "# Stage B MIDI-to-Solo MVP Input Contract",
        "",
        "## Summary",
        "",
        f"- boundary: `{report['boundary']}`",
        f"- next boundary: `{decision['next_boundary']}`",
        f"- target date: `{report['target_date']}`",
        f"- user goal: `{goal['user_goal']}`",
        f"- success mode: `{goal['success_mode']}`",
        f"- neural-only required: `{_bool_token(goal['neural_only_required'])}`",
        f"- MIDI-to-solo MVP claimed: `{_bool_token(claim['midi_to_solo_mvp_claimed'])}`",
        f"- Brad style fine-tuning completed: `{_bool_token(claim['brad_style_fine_tuning_completed'])}`",
        "",
        "## Input Contract",
        "",
        f"- required input: `{input_contract['required_input']}`",
        f"- supported extensions: `{input_contract['supported_extensions']}`",
        f"- minimum / target context bars: `{input_contract['minimum_context_bars']}` / `{input_contract['target_context_bars']}`",
        f"- chord confidence fallback: `{input_contract['chord_confidence_fallback']}`",
        "",
        "## Output Contract",
        "",
        f"- output root: `{output_contract['output_root']}`",
        f"- candidate count: `{output_contract['candidate_count']}`",
        f"- exported MIDI candidates: `{output_contract['export_top_midi_count']}`",
        f"- target solo bars: `{output_contract['target_solo_bars']}`",
        "",
        "## Generation Stack",
        "",
        f"- primary path: `{generation_stack['primary_path']}`",
        f"- fallback path: `{generation_stack['fallback_path']}`",
        f"- fallback trigger: `{generation_stack['fallback_trigger']}`",
        "",
        "## Objective Gate",
        "",
        f"- min note count: `{objective_gate['min_note_count']}`",
        f"- min unique pitch count: `{objective_gate['min_unique_pitch_count']}`",
        f"- max dead-air ratio: `{objective_gate['max_dead_air_ratio']}`",
        f"- max long-note ratio: `{objective_gate['max_long_note_ratio']}`",
        f"- max simultaneous notes: `{objective_gate['max_simultaneous_notes']}`",
        f"- max interval semitones: `{objective_gate['max_interval_semitones']}`",
        f"- min phrase coverage ratio: `{objective_gate['min_phrase_coverage_ratio']}`",
        "",
        "## Run Plan",
        "",
    ]
    for item in report["run_plan"]:
        lines.append(f"- `{item['date']}` `{item['boundary']}`: {item['deliverable']}")
    lines.extend(["", "## References", ""])
    for item in report["references"]:
        lines.append(f"- {item['name']}: {item['use']}")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Define Stage B MIDI-to-solo MVP contract")
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/stage_b_midi_to_solo_mvp_contract",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--issue_number", type=int, default=481)
    parser.add_argument("--expected_boundary", type=str, default="")
    parser.add_argument("--expected_next_boundary", type=str, default="")
    parser.add_argument("--require_fallback", action="store_true")
    parser.add_argument("--require_no_final_claim", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    report = build_contract_report(output_dir=output_dir, issue_number=int(args.issue_number))
    summary = validate_contract_report(
        report,
        expected_boundary=str(args.expected_boundary or ""),
        expected_next_boundary=str(args.expected_next_boundary or ""),
        require_fallback=bool(args.require_fallback),
        require_no_final_claim=bool(args.require_no_final_claim),
    )
    write_json(output_dir / "stage_b_midi_to_solo_mvp_contract.json", report)
    write_json(output_dir / "stage_b_midi_to_solo_mvp_contract_validation_summary.json", summary)
    markdown = markdown_report(report)
    write_text(output_dir / "stage_b_midi_to_solo_mvp_contract.md", markdown)
    if args.doc_path:
        write_text(Path(args.doc_path), markdown)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
