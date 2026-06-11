"""Build final review handoff for larger-sample Music Transformer solo candidates."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from scripts.assess_stage_b_generic_base_readiness import write_json, write_text  # noqa: E402
from scripts.build_music_transformer_solo_yield_listening_package import (  # noqa: E402
    SCHEMA_VERSION as LISTENING_PACKAGE_SCHEMA_VERSION,
)
from scripts.decide_music_transformer_solo_yield_objective_next import (  # noqa: E402
    SCHEMA_VERSION as OBJECTIVE_DECISION_SCHEMA_VERSION,
)
from scripts.render_stage_b_midi_to_solo_candidate_audio import sha256_file  # noqa: E402


SCHEMA_VERSION = "music_transformer_solo_yield_larger_sample_final_review_handoff_v1"
BOUNDARY = "music_transformer_solo_yield_larger_sample_final_review_handoff"
NEXT_BOUNDARY = "music_transformer_solo_yield_larger_sample_listening_review"
LISTENING_INPUT_SCHEMA_VERSION = "music_transformer_solo_yield_listening_input_v1"


class SoloYieldLargerSampleFinalHandoffError(ValueError):
    pass


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


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SoloYieldLargerSampleFinalHandoffError(f"json not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _require_no_quality_claim(report: dict[str, Any], *, label: str) -> None:
    readiness = _dict(report.get("readiness"))
    claimed = [
        key
        for key in (
            "audio_rendered_quality_claimed",
            "musical_quality_claimed",
            "artist_style_claimed",
            "production_ready_claimed",
        )
        if bool(readiness.get(key, False))
    ]
    if claimed:
        raise SoloYieldLargerSampleFinalHandoffError(
            f"unexpected quality claim in {label}: {claimed}"
        )


def _path(value: Any) -> Path:
    return Path(str(value or ""))


def _require_existing_file(path_value: Any, *, label: str) -> Path:
    path = _path(path_value)
    if not str(path) or not path.exists() or not path.is_file():
        raise SoloYieldLargerSampleFinalHandoffError(f"{label} missing: {path}")
    return path


def _require_checksum(path: Path, expected: str, *, label: str) -> None:
    actual = sha256_file(path)
    if str(expected or "") != actual:
        raise SoloYieldLargerSampleFinalHandoffError(
            f"{label} checksum mismatch: expected={expected} actual={actual}"
        )


def validate_listening_package(report: dict[str, Any]) -> list[dict[str, Any]]:
    if str(report.get("schema_version") or "") != LISTENING_PACKAGE_SCHEMA_VERSION:
        raise SoloYieldLargerSampleFinalHandoffError("listening package schema required")
    _require_no_quality_claim(report, label="listening_package")
    readiness = _dict(report.get("readiness"))
    if not bool(readiness.get("listening_review_package_ready", False)):
        raise SoloYieldLargerSampleFinalHandoffError("listening package ready flag required")
    candidates = [_dict(row) for row in _list(report.get("candidates"))]
    if not candidates:
        raise SoloYieldLargerSampleFinalHandoffError("listening candidates required")
    if _int(report.get("candidate_count")) != len(candidates):
        raise SoloYieldLargerSampleFinalHandoffError("candidate count mismatch")
    if _int(readiness.get("candidate_midi_files_copied")) != len(candidates):
        raise SoloYieldLargerSampleFinalHandoffError("candidate MIDI count mismatch")
    if _int(readiness.get("candidate_wav_files_copied")) != len(candidates):
        raise SoloYieldLargerSampleFinalHandoffError("candidate WAV count mismatch")
    template = _dict(report.get("review_input_template"))
    if str(template.get("schema_version") or "") != LISTENING_INPUT_SCHEMA_VERSION:
        raise SoloYieldLargerSampleFinalHandoffError("review input template schema required")
    if str(template.get("review_status") or "") != "pending":
        raise SoloYieldLargerSampleFinalHandoffError("pending review input template required")
    return candidates


def validate_objective_decision(
    report: dict[str, Any],
    *,
    expected_count: int,
) -> list[dict[str, Any]]:
    if str(report.get("schema_version") or "") != OBJECTIVE_DECISION_SCHEMA_VERSION:
        raise SoloYieldLargerSampleFinalHandoffError("objective decision schema required")
    _require_no_quality_claim(report, label="objective_decision")
    summary = _dict(report.get("objective_summary"))
    if _int(summary.get("candidate_count")) != int(expected_count):
        raise SoloYieldLargerSampleFinalHandoffError("objective candidate count mismatch")
    decision = _dict(report.get("decision"))
    if str(decision.get("next_boundary") or "") != BOUNDARY:
        raise SoloYieldLargerSampleFinalHandoffError("final handoff next boundary required")
    selected = [_dict(row) for row in _list(report.get("selected_objective_candidates"))]
    if not selected:
        raise SoloYieldLargerSampleFinalHandoffError("selected objective candidates required")
    return selected


def build_candidate_handoff(
    candidates: list[dict[str, Any]],
    selected_candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    selected_by_index = {
        _int(row.get("review_index")): index
        for index, row in enumerate(selected_candidates, start=1)
    }
    handoff_rows: list[dict[str, Any]] = []
    for row in candidates:
        review_index = _int(row.get("review_index"))
        midi_path = _require_existing_file(row.get("review_midi_path"), label="review MIDI")
        wav_path = _require_existing_file(row.get("review_wav_path"), label="review WAV")
        _require_checksum(midi_path, str(row.get("review_midi_sha256") or ""), label="MIDI")
        wav_file = _dict(row.get("review_wav_file"))
        _require_checksum(wav_path, str(wav_file.get("sha256") or ""), label="WAV")
        handoff_rows.append(
            {
                "review_index": review_index,
                "case_label": str(row.get("case_label") or ""),
                "chords": str(row.get("chords") or ""),
                "review_midi_path": str(midi_path),
                "review_wav_path": str(wav_path),
                "review_midi_sha256": str(row.get("review_midi_sha256") or ""),
                "review_wav_sha256": str(wav_file.get("sha256") or ""),
                "duration_seconds": _float(wav_file.get("duration_seconds")),
                "sample_rate": _int(wav_file.get("sample_rate")),
                "objective_rank": selected_by_index.get(review_index),
                "selected_by_objective": review_index in selected_by_index,
                "metrics": {
                    "score": _float(row.get("score")),
                    "note_count": _int(row.get("note_count")),
                    "unique_pitch_count": _int(row.get("unique_pitch_count")),
                    "dead_air_ratio": _float(row.get("dead_air_ratio")),
                    "direction_change_ratio": _float(row.get("direction_change_ratio")),
                    "syncopated_onset_ratio": _float(row.get("syncopated_onset_ratio")),
                    "chord_tone_ratio": _float(row.get("chord_tone_ratio")),
                    "tension_ratio": _float(row.get("tension_ratio")),
                },
            }
        )
    return handoff_rows


def _range(values: list[float]) -> dict[str, float]:
    return {
        "min": min(values, default=0.0),
        "max": max(values, default=0.0),
        "avg": float(mean(values)) if values else 0.0,
    }


def build_handoff_package(
    listening_package: dict[str, Any],
    objective_decision: dict[str, Any],
    *,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = validate_listening_package(listening_package)
    selected = validate_objective_decision(objective_decision, expected_count=len(candidates))
    handoff_rows = build_candidate_handoff(candidates, selected)
    source_sweep = _dict(listening_package.get("source_sweep"))
    scores = [_float(_dict(row.get("metrics")).get("score")) for row in handoff_rows]
    note_counts = [_int(_dict(row.get("metrics")).get("note_count")) for row in handoff_rows]
    dead_air = [_float(_dict(row.get("metrics")).get("dead_air_ratio")) for row in handoff_rows]
    aggregate = {
        "candidate_count": len(handoff_rows),
        "midi_count": len(handoff_rows),
        "wav_count": len(handoff_rows),
        "selected_objective_candidate_count": len(selected),
        "source_strict_valid_sample_count": _int(source_sweep.get("strict_valid_sample_count")),
        "source_sample_count": _int(source_sweep.get("sample_count")),
        "source_strict_yield_rate": _float(source_sweep.get("strict_yield_rate")),
        "score_range": _range(scores),
        "note_count_min": min(note_counts, default=0),
        "note_count_max": max(note_counts, default=0),
        "note_count_avg": float(mean(note_counts)) if note_counts else 0.0,
        "dead_air_range": _range(dead_air),
        "checksum_mismatch_count": 0,
        "missing_file_count": 0,
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "boundary": BOUNDARY,
        "source_reports": {
            "listening_package": {
                "schema_version": listening_package.get("schema_version"),
                "output_dir": listening_package.get("output_dir"),
                "candidate_count": _int(listening_package.get("candidate_count")),
            },
            "objective_decision": {
                "schema_version": objective_decision.get("schema_version"),
                "output_dir": objective_decision.get("output_dir"),
                "next_boundary": _dict(objective_decision.get("decision")).get("next_boundary"),
            },
        },
        "candidate_handoff": handoff_rows,
        "aggregate": aggregate,
        "readiness": {
            "final_review_handoff_ready": bool(handoff_rows),
            "validated_listening_input_present": False,
            "preference_fill_allowed": False,
            "audio_rendered_quality_claimed": False,
            "musical_quality_claimed": False,
            "artist_style_claimed": False,
            "production_ready_claimed": False,
        },
        "decision": {
            "current_boundary": BOUNDARY,
            "selected_next_target": "manual_listening_review_pending",
            "next_boundary": NEXT_BOUNDARY,
            "critical_user_input_required": False,
            "reason": "larger-sample MIDI/WAV review candidates are packaged; listening preference input remains pending",
        },
        "not_proven": [
            "human_audio_preference",
            "stable_jazz_solo_quality",
            "artist_level_long_solo_generation",
            "production_ready_improviser",
        ],
    }
    write_json(output_dir / "larger_sample_final_review_handoff.json", report)
    write_json(
        output_dir / "larger_sample_final_review_handoff_summary.json",
        validate_report(report, require_no_quality_claim=True),
    )
    write_text(output_dir / "larger_sample_final_review_handoff.md", markdown_report(report))
    return report


def validate_report(report: dict[str, Any], *, require_no_quality_claim: bool) -> dict[str, Any]:
    if str(report.get("schema_version") or "") != SCHEMA_VERSION:
        raise SoloYieldLargerSampleFinalHandoffError("schema version mismatch")
    if require_no_quality_claim:
        _require_no_quality_claim(report, label="handoff")
    readiness = _dict(report.get("readiness"))
    aggregate = _dict(report.get("aggregate"))
    decision = _dict(report.get("decision"))
    if not bool(readiness.get("final_review_handoff_ready", False)):
        raise SoloYieldLargerSampleFinalHandoffError("handoff ready flag required")
    if _int(aggregate.get("missing_file_count")) != 0:
        raise SoloYieldLargerSampleFinalHandoffError("missing file count must be zero")
    if _int(aggregate.get("checksum_mismatch_count")) != 0:
        raise SoloYieldLargerSampleFinalHandoffError("checksum mismatch count must be zero")
    return {
        "schema_version": str(report.get("schema_version") or ""),
        "boundary": str(report.get("boundary") or ""),
        "candidate_count": _int(aggregate.get("candidate_count")),
        "midi_count": _int(aggregate.get("midi_count")),
        "wav_count": _int(aggregate.get("wav_count")),
        "selected_objective_candidate_count": _int(
            aggregate.get("selected_objective_candidate_count")
        ),
        "source_strict_valid_sample_count": _int(
            aggregate.get("source_strict_valid_sample_count")
        ),
        "source_sample_count": _int(aggregate.get("source_sample_count")),
        "source_strict_yield_rate": _float(aggregate.get("source_strict_yield_rate")),
        "missing_file_count": _int(aggregate.get("missing_file_count")),
        "checksum_mismatch_count": _int(aggregate.get("checksum_mismatch_count")),
        "validated_listening_input_present": bool(
            readiness.get("validated_listening_input_present", True)
        ),
        "preference_fill_allowed": bool(readiness.get("preference_fill_allowed", True)),
        "selected_next_target": str(decision.get("selected_next_target") or ""),
        "next_boundary": str(decision.get("next_boundary") or ""),
        "musical_quality_claimed": bool(readiness.get("musical_quality_claimed", True)),
    }


def markdown_report(report: dict[str, Any]) -> str:
    aggregate = report["aggregate"]
    readiness = report["readiness"]
    decision = report["decision"]
    score_range = aggregate["score_range"]
    dead_air_range = aggregate["dead_air_range"]
    lines = [
        "# Music Transformer Solo Yield Larger Sample Final Review Handoff",
        "",
        "## Summary",
        "",
        f"- boundary: `{report['boundary']}`",
        f"- candidate count: `{aggregate['candidate_count']}`",
        f"- MIDI/WAV count: `{aggregate['midi_count']}` / `{aggregate['wav_count']}`",
        f"- selected objective candidates: `{aggregate['selected_objective_candidate_count']}`",
        f"- source strict yield: `{aggregate['source_strict_valid_sample_count']}` / `{aggregate['source_sample_count']}`",
        f"- source strict yield rate: `{float(aggregate['source_strict_yield_rate']):.4f}`",
        f"- score range: `{float(score_range['min']):.3f}` - `{float(score_range['max']):.3f}`",
        f"- note count range: `{aggregate['note_count_min']}` - `{aggregate['note_count_max']}`",
        f"- dead-air range: `{float(dead_air_range['min']):.4f}` - `{float(dead_air_range['max']):.4f}`",
        f"- missing file count: `{aggregate['missing_file_count']}`",
        f"- checksum mismatch count: `{aggregate['checksum_mismatch_count']}`",
        f"- validated listening input present: `{_bool_token(readiness['validated_listening_input_present'])}`",
        f"- preference fill allowed: `{_bool_token(readiness['preference_fill_allowed'])}`",
        f"- selected next target: `{decision['selected_next_target']}`",
        f"- next boundary: `{decision['next_boundary']}`",
        f"- musical quality claimed: `{_bool_token(readiness['musical_quality_claimed'])}`",
        "",
        "## Candidate Files",
        "",
    ]
    for row in report.get("candidate_handoff", []):
        metrics = row["metrics"]
        selected = "true" if row["selected_by_objective"] else "false"
        lines.extend(
            [
                f"- candidate `{row['review_index']}` / `{row['case_label']}`",
                f"  - MIDI: `{row['review_midi_path']}`",
                f"  - WAV: `{row['review_wav_path']}`",
                f"  - selected by objective: `{selected}`",
                f"  - score: `{float(metrics['score']):.3f}`",
                f"  - note count: `{metrics['note_count']}`",
                f"  - dead-air ratio: `{float(metrics['dead_air_ratio']):.4f}`",
            ]
        )
    lines.extend(["", "## Not Proven", ""])
    for item in report.get("not_proven", []):
        lines.append(f"- `{item}`")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build larger sample final review handoff")
    parser.add_argument(
        "--listening_package_report",
        type=str,
        default=(
            "outputs/music_transformer_finetune_mvp/solo_yield_listening_review/"
            "issue_1334_larger_sample_listening_package/listening_review_package.json"
        ),
    )
    parser.add_argument(
        "--objective_decision_report",
        type=str,
        default=(
            "outputs/music_transformer_finetune_mvp/solo_yield_objective_next_decision/"
            "issue_1338_larger_sample_objective_next/objective_next_decision.json"
        ),
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/music_transformer_finetune_mvp/solo_yield_larger_sample_final_handoff",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--require_no_quality_claim", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    report = build_handoff_package(
        read_json(Path(args.listening_package_report)),
        read_json(Path(args.objective_decision_report)),
        output_dir=output_dir,
    )
    summary = validate_report(report, require_no_quality_claim=bool(args.require_no_quality_claim))
    if args.doc_path:
        write_text(Path(args.doc_path), markdown_report(report))
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
