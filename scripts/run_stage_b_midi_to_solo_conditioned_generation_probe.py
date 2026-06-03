"""Generate ranked MIDI-to-solo candidates from extracted MIDI context."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pretty_midi

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from inference.app.fallback import generate_fallback_midi  # noqa: E402
from inference.app.generator import candidate_quality_score  # noqa: E402
from inference.app.metrics import compute_midi_metrics, validate_metrics  # noqa: E402
from inference.app.schemas import GenerationRequest  # noqa: E402
from scripts.assess_stage_b_generic_base_readiness import write_json, write_text  # noqa: E402


class StageBMidiToSoloConditionedGenerationProbeError(ValueError):
    pass


RESOURCE_BOUNDARY = "stage_b_midi_to_solo_training_resource_probe"
CONTEXT_BOUNDARY = "stage_b_midi_to_solo_context_extraction_mvp"
BOUNDARY = "stage_b_midi_to_solo_conditioned_generation_probe"
NEXT_BOUNDARY = "stage_b_midi_to_solo_candidate_audio_render_package"
SCHEMA_VERSION = "stage_b_midi_to_solo_conditioned_generation_probe_v1"

QUALITY_TO_SUFFIX = {
    "maj": "",
    "maj7": "maj7",
    "min": "m",
    "min7": "m7",
    "dom7": "7",
    "halfdim": "m7b5",
    "dim": "dim",
    "sus": "sus",
}


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise StageBMidiToSoloConditionedGenerationProbeError(f"report missing: {path}")
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


def chord_symbol(root: str, quality: str) -> str:
    if not root or root == "N":
        return "C"
    return f"{root}{QUALITY_TO_SUFFIX.get(quality, '')}"


def derive_chord_progression(context_report: dict[str, Any], target_bars: int) -> list[str]:
    context = _dict(context_report.get("context"))
    bars = _list(context.get("bar_contexts"))
    chords: list[str] = []
    previous = "C"
    for item in bars[: max(1, int(target_bars))]:
        bar = _dict(item)
        root = str(bar.get("chord_root") or "N")
        quality = str(bar.get("chord_quality") or "unknown")
        symbol = chord_symbol(root, quality)
        if symbol == "C" and root == "N":
            symbol = previous
        chords.append(symbol)
        previous = symbol
    while len(chords) < int(target_bars):
        chords.append(previous)
    return chords


def enforce_monophonic_midi(input_path: Path, output_path: Path, bpm: int) -> Path:
    source = pretty_midi.PrettyMIDI(str(input_path))
    notes: list[pretty_midi.Note] = []
    for instrument in source.instruments:
        if instrument.is_drum:
            continue
        notes.extend(instrument.notes)
    notes.sort(key=lambda note: (float(note.start), int(note.pitch), float(note.end)))

    cleaned: list[pretty_midi.Note] = []
    for note in notes:
        if float(note.end) <= float(note.start):
            continue
        current = pretty_midi.Note(
            velocity=int(note.velocity),
            pitch=int(note.pitch),
            start=float(note.start),
            end=float(note.end),
        )
        if cleaned and float(cleaned[-1].end) > float(current.start):
            cleaned[-1].end = max(float(cleaned[-1].start) + 0.03, min(float(cleaned[-1].end), float(current.start)))
            if float(cleaned[-1].end) > float(current.start):
                current.start = float(cleaned[-1].end)
        if float(current.end) <= float(current.start):
            current.end = float(current.start) + 0.03
        cleaned.append(current)

    midi = pretty_midi.PrettyMIDI(initial_tempo=float(bpm))
    piano = pretty_midi.Instrument(program=0, is_drum=False, name="midi_to_solo_conditioned_candidate")
    piano.notes = cleaned
    midi.instruments.append(piano)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    midi.write(str(output_path))
    return output_path


def validate_resource_probe(report: dict[str, Any]) -> dict[str, Any]:
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    if str(report.get("boundary") or readiness.get("boundary") or "") != RESOURCE_BOUNDARY:
        raise StageBMidiToSoloConditionedGenerationProbeError("training resource probe boundary required")
    if str(decision.get("next_boundary") or "") != BOUNDARY:
        raise StageBMidiToSoloConditionedGenerationProbeError("training resource probe must route to generation probe")
    if not bool(readiness.get("midi_to_solo_training_resource_ready", False)):
        raise StageBMidiToSoloConditionedGenerationProbeError("training resource readiness required")
    if bool(readiness.get("midi_to_solo_mvp_claimed", True)):
        raise StageBMidiToSoloConditionedGenerationProbeError("MIDI-to-solo MVP must not already be claimed")
    if bool(readiness.get("conditioned_generation_completed", True)):
        raise StageBMidiToSoloConditionedGenerationProbeError("conditioned generation must not already be completed")
    return {
        "resource_ready": True,
        "context_event_count": _int(_dict(report.get("context_resource")).get("context_event_count")),
        "full_tokenized_train_files": _int(_dict(report.get("full_window_resource")).get("tokenized_train_files")),
        "scale_checkpoint_count": _int(_dict(report.get("scale_smoke_resource")).get("checkpoint_count")),
    }


def validate_context_source(report: dict[str, Any]) -> dict[str, Any]:
    readiness = _dict(report.get("readiness"))
    summary = _dict(report.get("summary"))
    if str(report.get("boundary") or readiness.get("boundary") or "") != CONTEXT_BOUNDARY:
        raise StageBMidiToSoloConditionedGenerationProbeError("context extraction boundary required")
    if not bool(readiness.get("context_extraction_completed", False)):
        raise StageBMidiToSoloConditionedGenerationProbeError("context extraction completion required")
    if _int(summary.get("context_event_count")) <= 0:
        raise StageBMidiToSoloConditionedGenerationProbeError("context events required")
    return {
        "context_bars": _int(summary.get("context_bars")),
        "context_event_count": _int(summary.get("context_event_count")),
        "unknown_chord_bar_count": _int(summary.get("unknown_chord_bar_count")),
        "low_confidence_bar_count": _int(summary.get("low_confidence_bar_count")),
    }


def generate_conditioned_candidates(
    *,
    chord_progression: list[str],
    output_dir: Path,
    candidate_count: int,
    export_top_midi_count: int,
    seed_start: int,
    bpm: int,
    bars: int,
    density: str,
    energy: str,
    min_note_count: int,
    min_unique_pitch_count: int,
    max_simultaneous_notes: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    raw_dir = output_dir / "raw"
    export_dir = output_dir / "midi"
    candidate_rows: list[dict[str, Any]] = []
    for index in range(max(1, int(candidate_count))):
        seed = int(seed_start) + index
        raw_path = raw_dir / f"candidate_seed_{seed}.mid"
        mono_path = raw_dir / f"candidate_seed_{seed}_mono.mid"
        request = GenerationRequest(
            bpm=int(bpm),
            chord_progression=chord_progression,
            bars=int(bars),
            density=density,
            energy=energy,
            seed=seed,
            temperature=0.9,
            top_k=8,
        )
        request.validate()
        generate_fallback_midi(request, raw_path)
        enforce_monophonic_midi(raw_path, mono_path, bpm=int(bpm))
        metrics = compute_midi_metrics(mono_path, 0, fallback_used=True, request=request)
        app_valid, app_reason = validate_metrics(metrics, density, bars=int(bars))
        contract_reasons: list[str] = []
        if int(metrics.note_count) < int(min_note_count):
            contract_reasons.append(f"note_count {metrics.note_count} < {int(min_note_count)}")
        if int(metrics.unique_pitch_count or 0) < int(min_unique_pitch_count):
            contract_reasons.append(f"unique_pitch_count {metrics.unique_pitch_count} < {int(min_unique_pitch_count)}")
        if int(metrics.max_simultaneous_notes or 0) > int(max_simultaneous_notes):
            contract_reasons.append(
                f"max_simultaneous_notes {metrics.max_simultaneous_notes} > {int(max_simultaneous_notes)}"
            )
        score = candidate_quality_score(metrics, density)
        if contract_reasons:
            score += 100.0 + len(contract_reasons)
        if not app_valid:
            score += 50.0
        candidate_rows.append(
            {
                "seed": int(seed),
                "raw_midi_path": str(raw_path),
                "candidate_midi_path": str(mono_path),
                "generation_source": "context_conditioned_fallback",
                "score": round(float(score), 6),
                "app_valid": bool(app_valid),
                "app_failure_reason": app_reason,
                "contract_gate_passed": bool(not contract_reasons),
                "contract_failure_reasons": contract_reasons,
                "note_count": int(metrics.note_count),
                "unique_pitch_count": int(metrics.unique_pitch_count or 0),
                "unique_pitch_class_count": int(metrics.unique_pitch_class_count or 0),
                "max_simultaneous_notes": int(metrics.max_simultaneous_notes or 0),
                "dead_air_ratio": float(metrics.dead_air_ratio),
                "repetition_score": float(metrics.repetition_score),
                "chord_tone_ratio": _float(metrics.chord_tone_ratio),
                "phrase_coverage_ratio": _float(metrics.phrase_coverage_ratio),
            }
        )

    ranked = sorted(candidate_rows, key=lambda row: (float(row["score"]), int(row["seed"])))
    top_rows: list[dict[str, Any]] = []
    for rank, row in enumerate(ranked[: max(1, int(export_top_midi_count))], start=1):
        source = Path(str(row["candidate_midi_path"]))
        export_path = export_dir / f"rank_{rank:02d}_seed_{int(row['seed'])}.mid"
        export_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, export_path)
        top = dict(row)
        top["rank"] = int(rank)
        top["export_midi_path"] = str(export_path)
        top_rows.append(top)
    return ranked, top_rows


def build_conditioned_generation_report(
    *,
    context_report: dict[str, Any],
    resource_probe: dict[str, Any],
    output_dir: Path,
    issue_number: int,
    candidate_count: int,
    export_top_midi_count: int,
    seed_start: int,
    bpm: int,
    bars: int,
    density: str,
    energy: str,
    min_note_count: int,
    min_unique_pitch_count: int,
    max_simultaneous_notes: int,
) -> dict[str, Any]:
    resource_summary = validate_resource_probe(resource_probe)
    context_summary = validate_context_source(context_report)
    chord_progression = derive_chord_progression(context_report, target_bars=int(bars))
    candidates, top_candidates = generate_conditioned_candidates(
        chord_progression=chord_progression,
        output_dir=output_dir,
        candidate_count=candidate_count,
        export_top_midi_count=export_top_midi_count,
        seed_start=seed_start,
        bpm=bpm,
        bars=bars,
        density=density,
        energy=energy,
        min_note_count=min_note_count,
        min_unique_pitch_count=min_unique_pitch_count,
        max_simultaneous_notes=max_simultaneous_notes,
    )
    qualified = [row for row in candidates if bool(row["contract_gate_passed"]) and bool(row["app_valid"])]
    exported_qualified = [
        row for row in top_candidates if bool(row["contract_gate_passed"]) and bool(row["app_valid"])
    ]
    export_ready = len(top_candidates) >= int(export_top_midi_count) and len(exported_qualified) >= int(export_top_midi_count)
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "issue_number": int(issue_number),
        "boundary": BOUNDARY,
        "resource_summary": resource_summary,
        "context_summary": context_summary,
        "input_context": {
            "chord_progression": chord_progression,
            "bpm": int(bpm),
            "bars": int(bars),
            "density": density,
            "energy": energy,
        },
        "generation_config": {
            "candidate_count": int(candidate_count),
            "export_top_midi_count": int(export_top_midi_count),
            "seed_start": int(seed_start),
            "generation_source": "context_conditioned_fallback",
            "model_checkpoint_generation_used": False,
            "checkpoint_direct_generation_skip_reason": "scale-smoke max_sequence not yet sufficient for 8-bar 24-note contract",
        },
        "objective_gate": {
            "min_note_count": int(min_note_count),
            "min_unique_pitch_count": int(min_unique_pitch_count),
            "max_simultaneous_notes": int(max_simultaneous_notes),
        },
        "candidates": candidates,
        "top_candidates": top_candidates,
        "summary": {
            "candidate_count": int(len(candidates)),
            "qualified_candidate_count": int(len(qualified)),
            "exported_candidate_count": int(len(top_candidates)),
            "exported_qualified_candidate_count": int(len(exported_qualified)),
            "best_score": float(top_candidates[0]["score"]) if top_candidates else None,
            "best_note_count": int(top_candidates[0]["note_count"]) if top_candidates else 0,
            "best_unique_pitch_count": int(top_candidates[0]["unique_pitch_count"]) if top_candidates else 0,
            "best_max_simultaneous_notes": int(top_candidates[0]["max_simultaneous_notes"]) if top_candidates else 0,
            "best_chord_tone_ratio": float(top_candidates[0]["chord_tone_ratio"]) if top_candidates else 0.0,
        },
        "readiness": {
            "boundary": BOUNDARY,
            "conditioned_generation_probe_completed": True,
            "ranked_midi_candidates_exported": bool(export_ready),
            "midi_to_solo_mvp_claimed": False,
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
            "reason": "ranked context-conditioned MIDI candidates are exported; audio render/review remains separate",
        },
        "not_proven": [
            "model_checkpoint_direct_8bar_generation_quality",
            "midi_to_solo_mvp_completion",
            "human_audio_preference",
            "broad_trained_model_quality",
            "brad_style_adaptation",
        ],
        "next_recommended_issue": "Stage B MIDI-to-solo candidate audio render package",
    }


def validate_conditioned_generation_report(
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
    top_candidates = _list(report.get("top_candidates"))
    next_boundary = str(decision.get("next_boundary") or "")
    if expected_boundary and boundary != expected_boundary:
        raise StageBMidiToSoloConditionedGenerationProbeError(f"expected boundary {expected_boundary}, got {boundary}")
    if expected_next_boundary and next_boundary != expected_next_boundary:
        raise StageBMidiToSoloConditionedGenerationProbeError(
            f"expected next boundary {expected_next_boundary}, got {next_boundary}"
        )
    if require_exported_candidates and not bool(readiness.get("ranked_midi_candidates_exported", False)):
        raise StageBMidiToSoloConditionedGenerationProbeError("ranked MIDI candidates must be exported")
    if _int(summary.get("exported_candidate_count")) < int(min_exported_candidates):
        raise StageBMidiToSoloConditionedGenerationProbeError("exported candidate count below threshold")
    for row in top_candidates[: int(min_exported_candidates)]:
        candidate = _dict(row)
        if not Path(str(candidate.get("export_midi_path") or "")).exists():
            raise StageBMidiToSoloConditionedGenerationProbeError("exported MIDI path missing")
        if not bool(candidate.get("contract_gate_passed", False)):
            raise StageBMidiToSoloConditionedGenerationProbeError("top candidate contract gate failed")
        if _int(candidate.get("note_count")) < int(min_note_count):
            raise StageBMidiToSoloConditionedGenerationProbeError("top candidate note count below threshold")
        if _int(candidate.get("unique_pitch_count")) < int(min_unique_pitch_count):
            raise StageBMidiToSoloConditionedGenerationProbeError("top candidate unique pitch count below threshold")
        if _int(candidate.get("max_simultaneous_notes")) > int(max_simultaneous_notes):
            raise StageBMidiToSoloConditionedGenerationProbeError("top candidate simultaneous note limit failed")
    if bool(decision.get("critical_user_input_required", True)):
        raise StageBMidiToSoloConditionedGenerationProbeError("critical user input should not be required")
    if require_no_final_claim:
        blocked = [
            "midi_to_solo_mvp_claimed",
            "model_checkpoint_generation_quality_claimed",
            "broad_trained_model_quality_claimed",
            "brad_style_adaptation_claimed",
            "human_audio_preference_claimed",
        ]
        claimed = [name for name in blocked if bool(readiness.get(name, True))]
        if claimed:
            raise StageBMidiToSoloConditionedGenerationProbeError(f"unexpected final claim: {claimed}")
    return {
        "boundary": boundary,
        "next_boundary": next_boundary,
        "conditioned_generation_probe_completed": bool(
            readiness.get("conditioned_generation_probe_completed", False)
        ),
        "ranked_midi_candidates_exported": bool(readiness.get("ranked_midi_candidates_exported", False)),
        "candidate_count": _int(summary.get("candidate_count")),
        "qualified_candidate_count": _int(summary.get("qualified_candidate_count")),
        "exported_candidate_count": _int(summary.get("exported_candidate_count")),
        "exported_qualified_candidate_count": _int(summary.get("exported_qualified_candidate_count")),
        "best_score": summary.get("best_score"),
        "best_note_count": _int(summary.get("best_note_count")),
        "best_unique_pitch_count": _int(summary.get("best_unique_pitch_count")),
        "best_max_simultaneous_notes": _int(summary.get("best_max_simultaneous_notes")),
        "best_chord_tone_ratio": _float(summary.get("best_chord_tone_ratio")),
        "midi_to_solo_mvp_claimed": bool(readiness.get("midi_to_solo_mvp_claimed", True)),
        "human_audio_preference_claimed": bool(readiness.get("human_audio_preference_claimed", True)),
        "critical_user_input_required": bool(decision.get("critical_user_input_required", True)),
        "next_recommended_issue": str(report.get("next_recommended_issue") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    summary = report["summary"]
    readiness = report["readiness"]
    decision = report["decision"]
    config = report["generation_config"]
    gate = report["objective_gate"]
    context = report["input_context"]
    lines = [
        "# Stage B MIDI-to-Solo Conditioned Generation Probe",
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
        f"- density: `{context['density']}`",
        "",
        "## Objective Gate",
        "",
        f"- min note count: `{gate['min_note_count']}`",
        f"- min unique pitch count: `{gate['min_unique_pitch_count']}`",
        f"- max simultaneous notes: `{gate['max_simultaneous_notes']}`",
        "",
        "## Candidate Summary",
        "",
        f"- candidate count: `{summary['candidate_count']}`",
        f"- qualified candidate count: `{summary['qualified_candidate_count']}`",
        f"- exported candidate count: `{summary['exported_candidate_count']}`",
        f"- exported qualified candidate count: `{summary['exported_qualified_candidate_count']}`",
        f"- best score: `{summary['best_score']}`",
        f"- best note count: `{summary['best_note_count']}`",
        f"- best unique pitch count: `{summary['best_unique_pitch_count']}`",
        f"- best max simultaneous notes: `{summary['best_max_simultaneous_notes']}`",
        f"- best chord-tone ratio: `{summary['best_chord_tone_ratio']}`",
        "",
        "## Exported MIDI",
        "",
    ]
    for row in report["top_candidates"]:
        lines.append(
            f"- rank `{row['rank']}` seed `{row['seed']}`: `{row['export_midi_path']}`, "
            f"score `{row['score']}`, notes `{row['note_count']}`, unique pitches `{row['unique_pitch_count']}`"
        )
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            f"- model checkpoint generation quality claimed: `{_bool_token(readiness['model_checkpoint_generation_quality_claimed'])}`",
            f"- broad trained-model quality claimed: `{_bool_token(readiness['broad_trained_model_quality_claimed'])}`",
            f"- Brad style adaptation claimed: `{_bool_token(readiness['brad_style_adaptation_claimed'])}`",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MIDI-to-solo conditioned generation probe")
    parser.add_argument("--context_report", type=str, required=True)
    parser.add_argument("--resource_probe", type=str, required=True)
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/stage_b_midi_to_solo_conditioned_generation_probe",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--issue_number", type=int, default=487)
    parser.add_argument("--candidate_count", type=int, default=8)
    parser.add_argument("--export_top_midi_count", type=int, default=3)
    parser.add_argument("--seed_start", type=int, default=487)
    parser.add_argument("--bpm", type=int, default=120)
    parser.add_argument("--bars", type=int, default=8)
    parser.add_argument("--density", type=str, default="medium")
    parser.add_argument("--energy", type=str, default="mid")
    parser.add_argument("--min_note_count", type=int, default=24)
    parser.add_argument("--min_unique_pitch_count", type=int, default=8)
    parser.add_argument("--max_simultaneous_notes", type=int, default=1)
    parser.add_argument("--expected_boundary", type=str, default="")
    parser.add_argument("--expected_next_boundary", type=str, default="")
    parser.add_argument("--require_exported_candidates", action="store_true")
    parser.add_argument("--require_no_final_claim", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    report = build_conditioned_generation_report(
        context_report=read_json(Path(args.context_report)),
        resource_probe=read_json(Path(args.resource_probe)),
        output_dir=output_dir,
        issue_number=int(args.issue_number),
        candidate_count=int(args.candidate_count),
        export_top_midi_count=int(args.export_top_midi_count),
        seed_start=int(args.seed_start),
        bpm=int(args.bpm),
        bars=int(args.bars),
        density=args.density,
        energy=args.energy,
        min_note_count=int(args.min_note_count),
        min_unique_pitch_count=int(args.min_unique_pitch_count),
        max_simultaneous_notes=int(args.max_simultaneous_notes),
    )
    summary = validate_conditioned_generation_report(
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
    write_json(output_dir / "stage_b_midi_to_solo_conditioned_generation_probe.json", report)
    write_json(
        output_dir / "stage_b_midi_to_solo_conditioned_generation_probe_validation_summary.json",
        summary,
    )
    markdown = markdown_report(report)
    write_text(output_dir / "stage_b_midi_to_solo_conditioned_generation_probe.md", markdown)
    if args.doc_path:
        write_text(Path(args.doc_path), markdown)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
