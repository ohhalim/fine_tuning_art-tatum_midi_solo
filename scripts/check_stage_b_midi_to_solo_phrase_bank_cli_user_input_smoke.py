"""Validate a phrase-bank CLI package built from an explicit input MIDI."""

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
from scripts.run_stage_b_midi_to_solo_phrase_bank_cli_mvp_package import (  # noqa: E402
    BOUNDARY as CLI_PACKAGE_BOUNDARY,
    NEXT_BOUNDARY as CLI_PACKAGE_NEXT_BOUNDARY,
)


class StageBMidiToSoloPhraseBankCliUserInputSmokeError(ValueError):
    pass


BOUNDARY = "stage_b_midi_to_solo_phrase_bank_cli_user_input_smoke"
NEXT_BOUNDARY = "stage_b_midi_to_solo_phrase_bank_cli_audio_render_smoke"
SCHEMA_VERSION = "stage_b_midi_to_solo_phrase_bank_cli_user_input_smoke_v1"

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
        raise StageBMidiToSoloPhraseBankCliUserInputSmokeError(
            f"unexpected quality claim in {label}: {claimed}"
        )


def resolve_path(path_text: str) -> Path:
    return Path(path_text).expanduser().resolve()


def is_fixture_auto_input(path_text: str) -> bool:
    normalized = str(path_text).replace("\\", "/")
    return normalized.endswith("/input/fixture.mid")


def validate_cli_package_source(
    report: dict[str, Any],
    *,
    expected_input_midi: Path | None,
    min_candidate_count: int,
    require_explicit_input: bool,
) -> dict[str, Any]:
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    summary = _dict(report.get("objective_summary"))
    input_info = _dict(report.get("input"))
    candidates = [_dict(item) for item in _list(report.get("candidate_manifest"))]
    if str(report.get("boundary") or "") != CLI_PACKAGE_BOUNDARY:
        raise StageBMidiToSoloPhraseBankCliUserInputSmokeError("CLI package boundary required")
    if str(decision.get("next_boundary") or "") != CLI_PACKAGE_NEXT_BOUNDARY:
        raise StageBMidiToSoloPhraseBankCliUserInputSmokeError("CLI package must route to user-input smoke")
    input_midi = str(input_info.get("midi_path") or "")
    if not input_midi:
        raise StageBMidiToSoloPhraseBankCliUserInputSmokeError("input MIDI path required")
    if not Path(input_midi).exists():
        raise StageBMidiToSoloPhraseBankCliUserInputSmokeError(f"input MIDI missing: {input_midi}")
    if expected_input_midi is not None and resolve_path(input_midi) != resolve_path(str(expected_input_midi)):
        raise StageBMidiToSoloPhraseBankCliUserInputSmokeError("unexpected input MIDI path")
    if require_explicit_input and is_fixture_auto_input(input_midi):
        raise StageBMidiToSoloPhraseBankCliUserInputSmokeError("explicit input path required")
    if not bool(readiness.get("cli_mvp_package_completed", False)):
        raise StageBMidiToSoloPhraseBankCliUserInputSmokeError("CLI package completion required")
    if not bool(readiness.get("ranked_repaired_midi_exported", False)):
        raise StageBMidiToSoloPhraseBankCliUserInputSmokeError("ranked repaired MIDI export required")
    if _int(summary.get("candidate_count")) < int(min_candidate_count):
        raise StageBMidiToSoloPhraseBankCliUserInputSmokeError("candidate count below threshold")
    if bool(decision.get("critical_user_input_required", True)):
        raise StageBMidiToSoloPhraseBankCliUserInputSmokeError("critical user input should not be required")
    _require_no_quality_claim(readiness, label="CLI package readiness")
    for item in candidates[: int(min_candidate_count)]:
        midi_path = str(item.get("repaired_midi_path") or "")
        if not Path(midi_path).exists():
            raise StageBMidiToSoloPhraseBankCliUserInputSmokeError(f"repaired MIDI missing: {midi_path}")
        if not bool(item.get("objective_supported", False)):
            raise StageBMidiToSoloPhraseBankCliUserInputSmokeError("candidate objective support required")
    return {
        "input_midi": input_midi,
        "explicit_input_used": not is_fixture_auto_input(input_midi),
        "candidate_count": _int(summary.get("candidate_count")),
        "objective_supported_candidate_count": _int(summary.get("objective_supported_candidate_count")),
        "all_candidates_objective_supported": bool(summary.get("all_candidates_objective_supported", False)),
        "min_dead_air_ratio": _float(summary.get("min_dead_air_ratio")),
        "max_dead_air_ratio": _float(summary.get("max_dead_air_ratio")),
        "input_context_bars": _int(summary.get("input_context_bars")),
        "phrase_bank_exported_candidate_count": _int(summary.get("phrase_bank_exported_candidate_count")),
        "candidate_manifest": candidates,
    }


def build_user_input_smoke_report(
    *,
    cli_package_report: dict[str, Any],
    package_report_path: Path,
    output_dir: Path,
    issue_number: int,
    expected_input_midi: Path | None,
    min_candidate_count: int,
    require_explicit_input: bool,
) -> dict[str, Any]:
    source = validate_cli_package_source(
        cli_package_report,
        expected_input_midi=expected_input_midi,
        min_candidate_count=int(min_candidate_count),
        require_explicit_input=bool(require_explicit_input),
    )
    candidates = source["candidate_manifest"][: int(min_candidate_count)]
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "issue_number": int(issue_number),
        "boundary": BOUNDARY,
        "source_boundary": CLI_PACKAGE_BOUNDARY,
        "source_report": str(package_report_path),
        "input": {
            "midi_path": source["input_midi"],
            "explicit_input_used": bool(source["explicit_input_used"]),
        },
        "objective_summary": {
            "candidate_count": _int(source["candidate_count"]),
            "objective_supported_candidate_count": _int(source["objective_supported_candidate_count"]),
            "all_candidates_objective_supported": bool(source["all_candidates_objective_supported"]),
            "min_dead_air_ratio": _float(source["min_dead_air_ratio"]),
            "max_dead_air_ratio": _float(source["max_dead_air_ratio"]),
            "input_context_bars": _int(source["input_context_bars"]),
            "phrase_bank_exported_candidate_count": _int(source["phrase_bank_exported_candidate_count"]),
            "repaired_midi_file_count": len(candidates),
        },
        "candidate_manifest": [
            {
                "rank": _int(item.get("rank")),
                "sample_seed": _int(item.get("sample_seed")),
                "repaired_midi_path": str(item.get("repaired_midi_path") or ""),
                "objective_supported": bool(item.get("objective_supported", False)),
                "note_count": _int(item.get("note_count")),
                "unique_pitch_count": _int(item.get("unique_pitch_count")),
                "max_simultaneous_notes": _int(item.get("max_simultaneous_notes")),
                "dead_air_ratio": _float(item.get("dead_air_ratio")),
                "phrase_coverage_ratio": _float(item.get("phrase_coverage_ratio")),
            }
            for item in candidates
        ],
        "readiness": {
            "boundary": BOUNDARY,
            "user_input_smoke_completed": True,
            "explicit_input_path_used": bool(source["explicit_input_used"]),
            "ranked_repaired_midi_exported": True,
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
            "reason": "explicit input MIDI path produced ranked repaired MIDI candidates without quality claim",
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
        "next_recommended_issue": "Stage B MIDI-to-solo phrase-bank CLI audio render smoke",
    }


def validate_user_input_smoke_report(
    report: dict[str, Any],
    *,
    expected_boundary: str | None,
    expected_next_boundary: str | None,
    min_candidate_count: int,
    require_explicit_input: bool,
    require_no_quality_claim: bool,
) -> dict[str, Any]:
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    summary = _dict(report.get("objective_summary"))
    candidates = [_dict(item) for item in _list(report.get("candidate_manifest"))]
    boundary = str(report.get("boundary") or "")
    if expected_boundary and boundary != expected_boundary:
        raise StageBMidiToSoloPhraseBankCliUserInputSmokeError(
            f"expected boundary {expected_boundary}, got {boundary}"
        )
    if expected_next_boundary and str(decision.get("next_boundary") or "") != expected_next_boundary:
        raise StageBMidiToSoloPhraseBankCliUserInputSmokeError("unexpected next boundary")
    if require_explicit_input and not bool(readiness.get("explicit_input_path_used", False)):
        raise StageBMidiToSoloPhraseBankCliUserInputSmokeError("explicit input smoke required")
    if _int(summary.get("candidate_count")) < int(min_candidate_count):
        raise StageBMidiToSoloPhraseBankCliUserInputSmokeError("candidate count below threshold")
    if _int(summary.get("repaired_midi_file_count")) < int(min_candidate_count):
        raise StageBMidiToSoloPhraseBankCliUserInputSmokeError("repaired MIDI file count below threshold")
    for item in candidates[: int(min_candidate_count)]:
        midi_path = str(item.get("repaired_midi_path") or "")
        if not Path(midi_path).exists():
            raise StageBMidiToSoloPhraseBankCliUserInputSmokeError(f"repaired MIDI missing: {midi_path}")
        if not bool(item.get("objective_supported", False)):
            raise StageBMidiToSoloPhraseBankCliUserInputSmokeError("candidate objective support required")
    if bool(decision.get("critical_user_input_required", True)):
        raise StageBMidiToSoloPhraseBankCliUserInputSmokeError("critical user input should not be required")
    if require_no_quality_claim:
        _require_no_quality_claim(readiness, label="user-input smoke readiness")
    return {
        "boundary": boundary,
        "next_boundary": str(decision.get("next_boundary") or ""),
        "input_midi": str(_dict(report.get("input")).get("midi_path") or ""),
        "explicit_input_used": bool(readiness.get("explicit_input_path_used", False)),
        "candidate_count": _int(summary.get("candidate_count")),
        "objective_supported_candidate_count": _int(summary.get("objective_supported_candidate_count")),
        "all_candidates_objective_supported": bool(summary.get("all_candidates_objective_supported", False)),
        "min_dead_air_ratio": _float(summary.get("min_dead_air_ratio")),
        "max_dead_air_ratio": _float(summary.get("max_dead_air_ratio")),
        "input_context_bars": _int(summary.get("input_context_bars")),
        "repaired_midi_file_count": _int(summary.get("repaired_midi_file_count")),
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
        "# Stage B MIDI-to-Solo Phrase-Bank CLI User-Input Smoke",
        "",
        "## Summary",
        "",
        f"- boundary: `{report['boundary']}`",
        f"- next boundary: `{decision['next_boundary']}`",
        f"- input MIDI: `{report['input']['midi_path']}`",
        f"- explicit input used: `{_bool_token(readiness['explicit_input_path_used'])}`",
        f"- candidate count: `{summary['candidate_count']}`",
        f"- objective supported candidate count: `{summary['objective_supported_candidate_count']}`",
        f"- repaired MIDI file count: `{summary['repaired_midi_file_count']}`",
        f"- input context bars: `{summary['input_context_bars']}`",
        f"- dead-air range: `{summary['min_dead_air_ratio']:.4f} - {summary['max_dead_air_ratio']:.4f}`",
        f"- human/audio preference claimed: `{_bool_token(readiness['human_audio_preference_claimed'])}`",
        f"- MIDI-to-solo musical quality claimed: `{_bool_token(readiness['midi_to_solo_musical_quality_claimed'])}`",
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
    lines.extend(["## Claim Boundary", ""])
    for item in report["not_proven"]:
        lines.append(f"- `{item}`")
    lines.extend(["", "## Next", "", f"- `{report['next_recommended_issue']}`", ""])
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate explicit input MIDI CLI smoke package")
    parser.add_argument("--cli_package_report", type=str, required=True)
    parser.add_argument("--expected_input_midi", type=str, default="")
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/stage_b_midi_to_solo_phrase_bank_cli_user_input_smoke",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--issue_number", type=int, default=654)
    parser.add_argument("--expected_boundary", type=str, default="")
    parser.add_argument("--expected_next_boundary", type=str, default="")
    parser.add_argument("--min_candidate_count", type=int, default=3)
    parser.add_argument("--require_explicit_input", action="store_true")
    parser.add_argument("--require_no_quality_claim", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    expected_input_midi = Path(args.expected_input_midi) if args.expected_input_midi else None
    package_report_path = Path(args.cli_package_report)
    report = build_user_input_smoke_report(
        cli_package_report=read_json(package_report_path),
        package_report_path=package_report_path,
        output_dir=output_dir,
        issue_number=int(args.issue_number),
        expected_input_midi=expected_input_midi,
        min_candidate_count=int(args.min_candidate_count),
        require_explicit_input=bool(args.require_explicit_input),
    )
    summary = validate_user_input_smoke_report(
        report,
        expected_boundary=str(args.expected_boundary or ""),
        expected_next_boundary=str(args.expected_next_boundary or ""),
        min_candidate_count=int(args.min_candidate_count),
        require_explicit_input=bool(args.require_explicit_input),
        require_no_quality_claim=bool(args.require_no_quality_claim),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "stage_b_midi_to_solo_phrase_bank_cli_user_input_smoke.json", report)
    write_json(
        output_dir / "stage_b_midi_to_solo_phrase_bank_cli_user_input_smoke_validation_summary.json",
        summary,
    )
    markdown = markdown_report(report)
    write_text(output_dir / "stage_b_midi_to_solo_phrase_bank_cli_user_input_smoke.md", markdown)
    if args.doc_path:
        write_text(Path(args.doc_path), markdown)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
