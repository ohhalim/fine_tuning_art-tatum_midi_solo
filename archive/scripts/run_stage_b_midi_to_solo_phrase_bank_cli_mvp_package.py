"""Build a runnable phrase-bank MIDI-to-solo CLI MVP package."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from inference.app.schemas import GenerationRequest  # noqa: E402
from scripts.assess_stage_b_generic_base_readiness import write_json, write_text  # noqa: E402
from scripts.extract_stage_b_midi_to_solo_context import (  # noqa: E402
    BOUNDARY as CONTEXT_BOUNDARY,
    build_context_report,
    build_fixture_midi,
    markdown_report as context_markdown_report,
    validate_context_report,
)
from scripts.run_stage_b_midi_to_solo_phrase_bank_dead_air_density_repair_probe import (  # noqa: E402
    build_repair_candidate,
    parse_additions_per_bar,
)
from scripts.run_stage_b_midi_to_solo_phrase_bank_retrieval_baseline import (  # noqa: E402
    BOUNDARY as PHRASE_BANK_BOUNDARY,
    DEFAULT_MODES,
    build_phrase_bank_retrieval_report,
    markdown_report as phrase_bank_markdown_report,
    parse_modes,
    validate_phrase_bank_retrieval_report,
)


class StageBMidiToSoloPhraseBankCliMvpPackageError(ValueError):
    pass


BOUNDARY = "stage_b_midi_to_solo_phrase_bank_cli_mvp_package"
NEXT_BOUNDARY = "stage_b_midi_to_solo_phrase_bank_cli_user_input_smoke"
SCHEMA_VERSION = "stage_b_midi_to_solo_phrase_bank_cli_mvp_package_v1"

QUALITY_CLAIM_KEYS = [
    "human_audio_preference_claimed",
    "midi_to_solo_musical_quality_claimed",
    "musical_quality_claimed",
    "phrase_bank_musical_quality_claimed",
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


def _require_no_quality_claim(container: dict[str, Any], *, label: str) -> None:
    claimed = [name for name in QUALITY_CLAIM_KEYS if bool(container.get(name, False))]
    if claimed:
        raise StageBMidiToSoloPhraseBankCliMvpPackageError(
            f"unexpected quality claim in {label}: {claimed}"
        )


def source_candidates_from_phrase_bank_report(
    report: dict[str, Any],
    *,
    min_candidate_count: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in _list(report.get("top_candidates")):
        candidate = _dict(item)
        midi_path = str(candidate.get("export_midi_path") or "")
        if not midi_path:
            continue
        if not Path(midi_path).exists():
            raise StageBMidiToSoloPhraseBankCliMvpPackageError(f"exported MIDI missing: {midi_path}")
        rows.append(
            {
                "rank": _int(candidate.get("rank")),
                "sample_seed": _int(candidate.get("sample_seed")),
                "midi_path": midi_path,
                "objective_metrics": _dict(candidate.get("exported_metrics")),
            }
        )
    if len(rows) < int(min_candidate_count):
        raise StageBMidiToSoloPhraseBankCliMvpPackageError("not enough phrase-bank source candidates")
    return rows[: int(min_candidate_count)]


def request_from_phrase_bank_report(report: dict[str, Any]) -> GenerationRequest:
    context = _dict(report.get("input_context"))
    request = GenerationRequest(
        bpm=_int(context.get("bpm")) or 120,
        chord_progression=[str(item) for item in _list(context.get("chord_progression"))],
        bars=_int(context.get("bars")) or 8,
        density=str(context.get("density") or "medium"),
        energy=str(context.get("energy") or "mid"),
    )
    request.validate()
    return request


def build_cli_repaired_candidates(
    *,
    phrase_bank_report: dict[str, Any],
    output_dir: Path,
    candidate_count: int,
    additions_per_bar: Sequence[int],
    dead_air_threshold_sec: float,
    min_start_separation_sec: float,
    min_dead_air_gain: float,
    max_dead_air_ratio: float,
    min_unique_density_patterns: int,
    min_note_count_gain: int,
) -> list[dict[str, Any]]:
    request = request_from_phrase_bank_report(phrase_bank_report)
    source_candidates = source_candidates_from_phrase_bank_report(
        phrase_bank_report,
        min_candidate_count=int(candidate_count),
    )
    return [
        build_repair_candidate(
            source_candidate=item,
            output_dir=output_dir,
            request=request,
            additions_per_bar=additions_per_bar,
            dead_air_threshold_sec=float(dead_air_threshold_sec),
            min_start_separation_sec=float(min_start_separation_sec),
            min_dead_air_gain=float(min_dead_air_gain),
            max_dead_air_ratio=float(max_dead_air_ratio),
            min_unique_density_patterns=int(min_unique_density_patterns),
            min_note_count_gain=int(min_note_count_gain),
        )
        for item in source_candidates
    ]


def compact_candidate_rows(repaired_candidates: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in repaired_candidates:
        candidate = _dict(item)
        metrics = _dict(candidate.get("repaired_metrics"))
        gate = _dict(candidate.get("repair_gate"))
        rows.append(
            {
                "rank": _int(candidate.get("rank")),
                "sample_seed": _int(candidate.get("sample_seed")),
                "source_midi_path": str(candidate.get("source_midi_path") or ""),
                "repaired_midi_path": str(candidate.get("repaired_midi_path") or ""),
                "objective_supported": bool(gate.get("qualified", False)),
                "repair_flags": [str(flag) for flag in _list(gate.get("flags"))],
                "note_count": _int(metrics.get("note_count")),
                "unique_pitch_count": _int(metrics.get("unique_pitch_count")),
                "max_simultaneous_notes": _int(metrics.get("max_simultaneous_notes")),
                "dead_air_ratio": _float(metrics.get("dead_air_ratio")),
                "phrase_coverage_ratio": _float(metrics.get("phrase_coverage_ratio")),
                "dead_air_gain": _float(candidate.get("dead_air_gain")),
                "note_count_gain": _int(candidate.get("note_count_gain")),
            }
        )
    return rows


def build_cli_mvp_package_report(
    *,
    input_midi: Path,
    output_dir: Path,
    issue_number: int,
    context_summary: dict[str, Any],
    phrase_bank_summary: dict[str, Any],
    repaired_candidates: Sequence[dict[str, Any]],
    context_report_path: Path,
    phrase_bank_report_path: Path,
    cli_command: str,
) -> dict[str, Any]:
    candidate_rows = compact_candidate_rows(repaired_candidates)
    objective_supported_count = sum(1 for item in candidate_rows if bool(item["objective_supported"]))
    dead_air_values = [_float(item.get("dead_air_ratio")) for item in candidate_rows]
    package_ready = bool(candidate_rows) and objective_supported_count == len(candidate_rows)
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "issue_number": int(issue_number),
        "boundary": BOUNDARY,
        "source_boundaries": {
            "context": CONTEXT_BOUNDARY,
            "phrase_bank": PHRASE_BANK_BOUNDARY,
        },
        "input": {
            "midi_path": str(input_midi),
        },
        "reports": {
            "context_report": str(context_report_path),
            "phrase_bank_report": str(phrase_bank_report_path),
        },
        "cli": {
            "script": "scripts/run_stage_b_midi_to_solo_phrase_bank_cli_mvp_package.py",
            "command": cli_command,
            "output_contract": [
                "stage_b_midi_to_solo_phrase_bank_cli_mvp_package.json",
                "stage_b_midi_to_solo_phrase_bank_cli_mvp_package_validation_summary.json",
                "stage_b_midi_to_solo_phrase_bank_cli_mvp_package.md",
                "midi/rank_*.mid",
            ],
        },
        "objective_summary": {
            "candidate_count": len(candidate_rows),
            "objective_supported_candidate_count": objective_supported_count,
            "all_candidates_objective_supported": objective_supported_count == len(candidate_rows),
            "min_dead_air_ratio": min(dead_air_values) if dead_air_values else 0.0,
            "max_dead_air_ratio": max(dead_air_values) if dead_air_values else 0.0,
            "input_context_bars": _int(context_summary.get("context_bars")),
            "phrase_bank_exported_candidate_count": _int(phrase_bank_summary.get("exported_candidate_count")),
            "phrase_bank_exported_qualified_candidate_count": _int(
                phrase_bank_summary.get("exported_qualified_candidate_count")
            ),
            "cli_mvp_package_ready": bool(package_ready),
        },
        "candidate_manifest": candidate_rows,
        "readiness": {
            "boundary": BOUNDARY,
            "cli_mvp_package_completed": True,
            "input_midi_context_extracted": True,
            "ranked_repaired_midi_exported": bool(package_ready),
            "human_audio_preference_claimed": False,
            "midi_to_solo_musical_quality_claimed": False,
            "phrase_bank_musical_quality_claimed": False,
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
            "reason": "input MIDI to ranked repaired MIDI package path is runnable without preference or quality claim",
        },
        "not_proven": [
            "human_audio_preference",
            "midi_to_solo_musical_quality",
            "phrase_bank_musical_quality",
            "audio_rendered_quality",
            "broad_trained_model_quality",
            "brad_style_adaptation",
            "production_ready",
        ],
        "next_recommended_issue": "Stage B MIDI-to-solo phrase-bank CLI user-input smoke",
    }


def validate_cli_mvp_package_report(
    report: dict[str, Any],
    *,
    expected_boundary: str | None,
    expected_next_boundary: str | None,
    min_candidate_count: int,
    require_cli_ready: bool,
    require_no_quality_claim: bool,
) -> dict[str, Any]:
    boundary = str(report.get("boundary") or "")
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    summary = _dict(report.get("objective_summary"))
    candidates = [_dict(item) for item in _list(report.get("candidate_manifest"))]
    if expected_boundary and boundary != expected_boundary:
        raise StageBMidiToSoloPhraseBankCliMvpPackageError(
            f"expected boundary {expected_boundary}, got {boundary}"
        )
    if expected_next_boundary and str(decision.get("next_boundary") or "") != expected_next_boundary:
        raise StageBMidiToSoloPhraseBankCliMvpPackageError("unexpected next boundary")
    if _int(summary.get("candidate_count")) < int(min_candidate_count):
        raise StageBMidiToSoloPhraseBankCliMvpPackageError("candidate count below threshold")
    if require_cli_ready and not bool(readiness.get("cli_mvp_package_completed", False)):
        raise StageBMidiToSoloPhraseBankCliMvpPackageError("CLI package completion required")
    if require_cli_ready and not bool(readiness.get("ranked_repaired_midi_exported", False)):
        raise StageBMidiToSoloPhraseBankCliMvpPackageError("ranked repaired MIDI export required")
    if bool(decision.get("critical_user_input_required", True)):
        raise StageBMidiToSoloPhraseBankCliMvpPackageError("critical user input should not be required")
    for item in candidates[: int(min_candidate_count)]:
        midi_path = str(item.get("repaired_midi_path") or "")
        if not Path(midi_path).exists():
            raise StageBMidiToSoloPhraseBankCliMvpPackageError(f"repaired MIDI missing: {midi_path}")
        if not bool(item.get("objective_supported", False)):
            raise StageBMidiToSoloPhraseBankCliMvpPackageError("candidate objective gate failed")
        if _int(item.get("max_simultaneous_notes")) > 1:
            raise StageBMidiToSoloPhraseBankCliMvpPackageError("candidate is not monophonic")
    if require_no_quality_claim:
        _require_no_quality_claim(readiness, label="CLI package readiness")
    return {
        "boundary": boundary,
        "next_boundary": str(decision.get("next_boundary") or ""),
        "candidate_count": _int(summary.get("candidate_count")),
        "objective_supported_candidate_count": _int(summary.get("objective_supported_candidate_count")),
        "all_candidates_objective_supported": bool(summary.get("all_candidates_objective_supported", False)),
        "min_dead_air_ratio": _float(summary.get("min_dead_air_ratio")),
        "max_dead_air_ratio": _float(summary.get("max_dead_air_ratio")),
        "input_context_bars": _int(summary.get("input_context_bars")),
        "phrase_bank_exported_candidate_count": _int(summary.get("phrase_bank_exported_candidate_count")),
        "cli_mvp_package_ready": bool(summary.get("cli_mvp_package_ready", False)),
        "human_audio_preference_claimed": bool(readiness.get("human_audio_preference_claimed", True)),
        "midi_to_solo_musical_quality_claimed": bool(
            readiness.get("midi_to_solo_musical_quality_claimed", True)
        ),
        "critical_user_input_required": bool(decision.get("critical_user_input_required", True)),
        "next_recommended_issue": str(report.get("next_recommended_issue") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    summary = report["objective_summary"]
    readiness = report["readiness"]
    decision = report["decision"]
    lines = [
        "# Stage B MIDI-to-Solo Phrase-Bank CLI MVP Package",
        "",
        "## Summary",
        "",
        f"- boundary: `{report['boundary']}`",
        f"- next boundary: `{decision['next_boundary']}`",
        f"- input MIDI: `{report['input']['midi_path']}`",
        f"- CLI MVP package completed: `{_bool_token(readiness['cli_mvp_package_completed'])}`",
        f"- ranked repaired MIDI exported: `{_bool_token(readiness['ranked_repaired_midi_exported'])}`",
        f"- candidate count: `{summary['candidate_count']}`",
        f"- objective supported candidate count: `{summary['objective_supported_candidate_count']}`",
        f"- dead-air range: `{summary['min_dead_air_ratio']:.4f} - {summary['max_dead_air_ratio']:.4f}`",
        f"- human/audio preference claimed: `{_bool_token(readiness['human_audio_preference_claimed'])}`",
        f"- MIDI-to-solo musical quality claimed: `{_bool_token(readiness['midi_to_solo_musical_quality_claimed'])}`",
        "",
        "## Command",
        "",
        "```bash",
        report["cli"]["command"],
        "```",
        "",
        "## Candidate Manifest",
        "",
    ]
    for item in report["candidate_manifest"]:
        lines.extend(
            [
                f"### Rank {item['rank']}",
                "",
                f"- seed: `{item['sample_seed']}`",
                f"- objective supported: `{_bool_token(item['objective_supported'])}`",
                f"- notes / unique pitches / max simultaneous: `{item['note_count']} / {item['unique_pitch_count']} / {item['max_simultaneous_notes']}`",
                f"- dead-air / phrase coverage: `{item['dead_air_ratio']:.4f} / {item['phrase_coverage_ratio']:.4f}`",
                f"- repaired MIDI: `{item['repaired_midi_path']}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Claim Boundary",
            "",
        ]
    )
    for item in report["not_proven"]:
        lines.append(f"- `{item}`")
    lines.extend(["", "## Next", "", f"- `{report['next_recommended_issue']}`", ""])
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build runnable phrase-bank MIDI-to-solo CLI MVP package")
    parser.add_argument("--input_midi", type=str, default="")
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/stage_b_midi_to_solo_phrase_bank_cli_mvp_package",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--issue_number", type=int, default=652)
    parser.add_argument("--input_dir", type=str, default="./midi_dataset/midi/studio")
    parser.add_argument("--target_context_bars", type=int, default=8)
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
    parser.add_argument("--max_phrase_bank_dead_air_ratio", type=float, default=0.65)
    parser.add_argument("--max_files", type=int, default=4)
    parser.add_argument("--window_bars", type=int, default=8)
    parser.add_argument("--window_stride_bars", type=int, default=4)
    parser.add_argument("--min_window_target_notes", type=int, default=16)
    parser.add_argument("--motif_length", type=int, default=4)
    parser.add_argument("--max_bar_span", type=int, default=2)
    parser.add_argument("--max_records", type=int, default=64)
    parser.add_argument("--template_top_n", type=int, default=32)
    parser.add_argument("--additions_per_bar", default="3,5,2,6,3,5,2,6")
    parser.add_argument("--dead_air_threshold_sec", type=float, default=0.18)
    parser.add_argument("--min_start_separation_sec", type=float, default=0.04)
    parser.add_argument("--min_dead_air_gain", type=float, default=0.15)
    parser.add_argument("--max_repaired_dead_air_ratio", type=float, default=0.45)
    parser.add_argument("--min_unique_density_patterns", type=int, default=3)
    parser.add_argument("--min_note_count_gain", type=int, default=16)
    parser.add_argument("--expected_boundary", type=str, default="")
    parser.add_argument("--expected_next_boundary", type=str, default="")
    parser.add_argument("--min_candidate_count", type=int, default=3)
    parser.add_argument("--require_cli_ready", action="store_true")
    parser.add_argument("--require_no_quality_claim", action="store_true")
    return parser


def cli_command_for_report(input_midi: Path, run_id: str) -> str:
    return (
        ".venv/bin/python scripts/run_stage_b_midi_to_solo_phrase_bank_cli_mvp_package.py "
        f"--input_midi {input_midi} --run_id {run_id}"
    )


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    input_midi = Path(args.input_midi) if args.input_midi else build_fixture_midi(output_dir / "input" / "fixture.mid")

    context_dir = output_dir / "context"
    phrase_bank_dir = output_dir / "phrase_bank"
    context_report = build_context_report(
        midi_path=input_midi,
        output_dir=context_dir,
        target_context_bars=int(args.target_context_bars),
        issue_number=int(args.issue_number),
    )
    context_summary = validate_context_report(
        context_report,
        expected_boundary=CONTEXT_BOUNDARY,
        expected_next_boundary=None,
        min_context_bars=4,
        require_no_final_claim=True,
    )
    context_dir.mkdir(parents=True, exist_ok=True)
    context_report_path = context_dir / "stage_b_midi_to_solo_context_extraction.json"
    write_json(context_report_path, context_report)
    write_json(context_dir / "stage_b_midi_to_solo_context_extraction_validation_summary.json", context_summary)
    write_text(context_dir / "stage_b_midi_to_solo_context_extraction.md", context_markdown_report(context_report))

    phrase_bank_report = build_phrase_bank_retrieval_report(
        context_report=context_report,
        output_dir=phrase_bank_dir,
        run_id=f"{run_id}_phrase_bank",
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
        max_dead_air_ratio=float(args.max_phrase_bank_dead_air_ratio),
        max_files=int(args.max_files),
        window_bars=int(args.window_bars),
        window_stride_bars=int(args.window_stride_bars),
        min_window_target_notes=int(args.min_window_target_notes),
        motif_length=int(args.motif_length),
        max_bar_span=int(args.max_bar_span),
        max_records=int(args.max_records),
        template_top_n=int(args.template_top_n),
    )
    phrase_bank_summary = validate_phrase_bank_retrieval_report(
        phrase_bank_report,
        expected_boundary=PHRASE_BANK_BOUNDARY,
        expected_next_boundary=None,
        require_exported_candidates=True,
        require_no_final_claim=True,
        min_exported_candidates=int(args.export_top_midi_count),
        min_note_count=int(args.min_note_count),
        min_unique_pitch_count=int(args.min_unique_pitch_count),
        max_simultaneous_notes=int(args.max_simultaneous_notes),
    )
    phrase_bank_dir.mkdir(parents=True, exist_ok=True)
    phrase_bank_report_path = phrase_bank_dir / "stage_b_midi_to_solo_phrase_bank_retrieval_baseline.json"
    write_json(phrase_bank_report_path, phrase_bank_report)
    write_json(
        phrase_bank_dir / "stage_b_midi_to_solo_phrase_bank_retrieval_baseline_validation_summary.json",
        phrase_bank_summary,
    )
    write_text(
        phrase_bank_dir / "stage_b_midi_to_solo_phrase_bank_retrieval_baseline.md",
        phrase_bank_markdown_report(phrase_bank_report),
    )

    repaired_candidates = build_cli_repaired_candidates(
        phrase_bank_report=phrase_bank_report,
        output_dir=output_dir,
        candidate_count=int(args.min_candidate_count),
        additions_per_bar=parse_additions_per_bar(args.additions_per_bar),
        dead_air_threshold_sec=float(args.dead_air_threshold_sec),
        min_start_separation_sec=float(args.min_start_separation_sec),
        min_dead_air_gain=float(args.min_dead_air_gain),
        max_dead_air_ratio=float(args.max_repaired_dead_air_ratio),
        min_unique_density_patterns=int(args.min_unique_density_patterns),
        min_note_count_gain=int(args.min_note_count_gain),
    )
    report = build_cli_mvp_package_report(
        input_midi=input_midi,
        output_dir=output_dir,
        issue_number=int(args.issue_number),
        context_summary=context_summary,
        phrase_bank_summary=phrase_bank_summary,
        repaired_candidates=repaired_candidates,
        context_report_path=context_report_path,
        phrase_bank_report_path=phrase_bank_report_path,
        cli_command=cli_command_for_report(input_midi, run_id),
    )
    summary = validate_cli_mvp_package_report(
        report,
        expected_boundary=str(args.expected_boundary or ""),
        expected_next_boundary=str(args.expected_next_boundary or ""),
        min_candidate_count=int(args.min_candidate_count),
        require_cli_ready=bool(args.require_cli_ready),
        require_no_quality_claim=bool(args.require_no_quality_claim),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "stage_b_midi_to_solo_phrase_bank_cli_mvp_package.json", report)
    write_json(output_dir / "stage_b_midi_to_solo_phrase_bank_cli_mvp_package_validation_summary.json", summary)
    markdown = markdown_report(report)
    write_text(output_dir / "stage_b_midi_to_solo_phrase_bank_cli_mvp_package.md", markdown)
    if args.doc_path:
        write_text(Path(args.doc_path), markdown)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
