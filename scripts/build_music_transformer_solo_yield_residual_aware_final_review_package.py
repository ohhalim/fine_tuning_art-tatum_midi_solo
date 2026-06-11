"""Build a residual-aware final review package for solo-yield repair candidates."""

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
from scripts.render_stage_b_midi_to_solo_candidate_audio import sha256_file  # noqa: E402


SCHEMA_VERSION = "music_transformer_solo_yield_residual_aware_final_review_package_v1"
REVIEW_INPUT_SCHEMA_VERSION = "music_transformer_solo_yield_residual_aware_review_input_v1"


class SoloYieldResidualAwareFinalReviewPackageError(ValueError):
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
        raise SoloYieldResidualAwareFinalReviewPackageError(f"json not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _reject_quality_claim(report: dict[str, Any], *, label: str) -> None:
    readiness = _dict(report.get("readiness"))
    claimed = [
        key
        for key in ("musical_quality_claimed", "artist_style_claimed", "production_ready_claimed")
        if bool(readiness.get(key, False))
    ]
    if claimed:
        raise SoloYieldResidualAwareFinalReviewPackageError(
            f"unexpected quality claim in {label}: {claimed}"
        )


def _require_file(path_value: Any, *, label: str) -> Path:
    path = Path(str(path_value or ""))
    if not str(path) or not path.exists() or not path.is_file():
        raise SoloYieldResidualAwareFinalReviewPackageError(f"{label} missing: {path}")
    return path


def _require_checksum(path: Path, expected: str, *, label: str) -> None:
    actual = sha256_file(path)
    if str(expected or "") != actual:
        raise SoloYieldResidualAwareFinalReviewPackageError(
            f"{label} checksum mismatch: expected={expected} actual={actual}"
        )


def rendered_audio_by_index(source_package: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {
        _int(row.get("repair_index")): _dict(row)
        for row in _list(source_package.get("rendered_audio_files"))
    }


def rubric_labels_by_index(rubric_report: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {
        _int(row.get("review_index")): _dict(row)
        for row in _list(rubric_report.get("candidate_labels"))
    }


def build_candidate_rows(
    *,
    source_package: dict[str, Any],
    rubric_report: dict[str, Any],
) -> list[dict[str, Any]]:
    rendered_by_index = rendered_audio_by_index(source_package)
    labels_by_index = rubric_labels_by_index(rubric_report)
    rows: list[dict[str, Any]] = []
    for candidate in _list(source_package.get("selected_candidates")):
        item = _dict(candidate)
        review_index = _int(item.get("repair_index"))
        midi_path = _require_file(item.get("repair_midi_path"), label="candidate MIDI")
        _require_checksum(midi_path, str(item.get("repair_midi_sha256") or ""), label="MIDI")

        rendered = rendered_by_index.get(review_index, {})
        wav_file = _dict(rendered.get("wav_file"))
        wav_path = _require_file(wav_file.get("path"), label="candidate WAV")
        _require_checksum(wav_path, str(wav_file.get("sha256") or ""), label="WAV")
        labels = labels_by_index.get(review_index, {})
        rows.append(
            {
                "review_index": review_index,
                "case_label": str(item.get("case_label") or ""),
                "sample_index": _int(item.get("sample_index")),
                "sample_seed": _int(item.get("sample_seed")),
                "review_midi_path": str(midi_path),
                "review_wav_path": str(wav_path),
                "review_midi_sha256": str(item.get("repair_midi_sha256") or ""),
                "review_wav_sha256": str(wav_file.get("sha256") or ""),
                "duration_seconds": _float(wav_file.get("duration_seconds")),
                "sample_rate": _int(wav_file.get("sample_rate")),
                "metrics": {
                    "note_count": _int(item.get("note_count")),
                    "dead_air_ratio": _float(item.get("dead_air_ratio")),
                    "direction_change_ratio": _float(item.get("direction_change_ratio")),
                    "syncopated_onset_ratio": _float(item.get("syncopated_onset_ratio")),
                    "chord_tone_ratio": _float(item.get("chord_tone_ratio")),
                    "tension_ratio": _float(item.get("tension_ratio")),
                },
                "rubric_major_labels": _list(labels.get("major_labels")),
                "rubric_watch_labels": _list(labels.get("watch_labels")),
                "quality_proxy_pass": bool(labels.get("quality_proxy_pass", False)),
            }
        )
    return rows


def build_review_input_template(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": REVIEW_INPUT_SCHEMA_VERSION,
        "review_status": "pending",
        "overall_decision": "pending",
        "preferred_review_index": None,
        "reviewer_notes": "",
        "candidates": [
            {
                "review_index": _int(row.get("review_index")),
                "case_label": row.get("case_label"),
                "decision": "pending",
                "usable_as_jazz_solo_phrase": None,
                "primary_failure": None,
                "notes": "",
            }
            for row in candidates
        ],
        "allowed_primary_failures": [
            "not_solo_like",
            "too_mechanical",
            "too_sparse",
            "too_dense",
            "rhythm_not_swinging",
            "outside_harmony",
            "weak_phrase_shape",
            "audio_render_issue",
            "none",
        ],
    }


def _range(values: list[float]) -> dict[str, float]:
    return {
        "min": min(values, default=0.0),
        "max": max(values, default=0.0),
        "avg": float(mean(values)) if values else 0.0,
    }


def build_final_review_package(
    *,
    source_package: dict[str, Any],
    rubric_report: dict[str, Any],
    feasibility_decision: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    _reject_quality_claim(source_package, label="source_package")
    _reject_quality_claim(rubric_report, label="rubric_report")
    _reject_quality_claim(feasibility_decision, label="feasibility_decision")
    candidates = build_candidate_rows(
        source_package=source_package,
        rubric_report=rubric_report,
    )
    if not candidates:
        raise SoloYieldResidualAwareFinalReviewPackageError("candidate rows required")
    review_input_template = build_review_input_template(candidates)
    rubric_aggregate = _dict(rubric_report.get("aggregate"))
    feasibility = _dict(feasibility_decision.get("decision"))
    dead_air = [_float(_dict(row.get("metrics")).get("dead_air_ratio")) for row in candidates]
    direction = [_float(_dict(row.get("metrics")).get("direction_change_ratio")) for row in candidates]
    syncopation = [_float(_dict(row.get("metrics")).get("syncopated_onset_ratio")) for row in candidates]
    tension = [_float(_dict(row.get("metrics")).get("tension_ratio")) for row in candidates]
    aggregate = {
        "candidate_count": len(candidates),
        "midi_count": len(candidates),
        "wav_count": len(candidates),
        "quality_proxy_pass_count": _int(rubric_aggregate.get("quality_proxy_pass_count")),
        "quality_proxy_fail_count": _int(rubric_aggregate.get("quality_proxy_fail_count")),
        "major_label_counts": _dict(rubric_aggregate.get("major_label_counts")),
        "watch_label_counts": _dict(rubric_aggregate.get("watch_label_counts")),
        "dead_air_range": _range(dead_air),
        "direction_change_range": _range(direction),
        "syncopation_range": _range(syncopation),
        "tension_range": _range(tension),
        "checksum_mismatch_count": 0,
        "missing_file_count": 0,
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "source_reports": {
            "source_package": {
                "schema_version": source_package.get("schema_version"),
                "output_dir": source_package.get("output_dir"),
            },
            "rubric_report": {
                "schema_version": rubric_report.get("schema_version"),
                "output_dir": rubric_report.get("output_dir"),
            },
            "feasibility_decision": {
                "schema_version": feasibility_decision.get("schema_version"),
                "output_dir": feasibility_decision.get("output_dir"),
            },
        },
        "aggregate": aggregate,
        "candidate_handoff": candidates,
        "review_input_template": review_input_template,
        "residual_context": {
            "tension_repeat_feasible": bool(feasibility.get("tension_repeat_feasible", True)),
            "tension_repeat_blocked_cases": _list(feasibility.get("tension_repeat_blocked_cases")),
            "selected_repair_target_from_rubric": _dict(rubric_report.get("decision")).get(
                "selected_repair_target"
            ),
            "residual_major_labels": _dict(rubric_aggregate.get("major_label_counts")),
            "residual_watch_labels": _dict(rubric_aggregate.get("watch_label_counts")),
        },
        "readiness": {
            "residual_aware_final_review_package_ready": True,
            "candidate_midi_files_validated": len(candidates),
            "candidate_wav_files_validated": len(candidates),
            "review_input_template_written": True,
            "validated_listening_input_present": False,
            "musical_quality_claimed": False,
            "artist_style_claimed": False,
            "production_ready_claimed": False,
        },
        "decision": {
            "current_boundary": "music_transformer_solo_yield_residual_aware_final_review_package",
            "next_boundary": "music_transformer_solo_yield_residual_aware_listening_input_guard",
            "critical_user_input_required": False,
            "reason": "objective repair loop reached residual tension blocked by guard; package candidates for listening review without quality claim",
        },
        "not_proven": [
            "human_audio_preference",
            "stable_jazz_solo_quality",
            "residual_tension_resolved",
            "production_ready_improviser",
        ],
    }
    write_json(output_dir / "residual_aware_final_review_package.json", report)
    write_json(output_dir / "residual_aware_review_input_template.json", review_input_template)
    write_text(output_dir / "residual_aware_final_review_package.md", markdown_report(report))
    return report


def validate_final_review_package(
    report: dict[str, Any],
    *,
    min_candidate_count: int,
    require_residual_context: bool,
    require_no_quality_claim: bool,
) -> dict[str, Any]:
    if str(report.get("schema_version")) != SCHEMA_VERSION:
        raise SoloYieldResidualAwareFinalReviewPackageError("schema version mismatch")
    aggregate = _dict(report.get("aggregate"))
    readiness = _dict(report.get("readiness"))
    if _int(aggregate.get("candidate_count")) < int(min_candidate_count):
        raise SoloYieldResidualAwareFinalReviewPackageError("candidate count below requirement")
    if _int(aggregate.get("checksum_mismatch_count")):
        raise SoloYieldResidualAwareFinalReviewPackageError("checksum mismatch found")
    if _int(aggregate.get("missing_file_count")):
        raise SoloYieldResidualAwareFinalReviewPackageError("missing file found")
    residual = _dict(report.get("residual_context"))
    if require_residual_context and not residual:
        raise SoloYieldResidualAwareFinalReviewPackageError("residual context required")
    if require_no_quality_claim:
        claimed = [
            key
            for key in ("musical_quality_claimed", "artist_style_claimed", "production_ready_claimed")
            if bool(readiness.get(key, True))
        ]
        if claimed:
            raise SoloYieldResidualAwareFinalReviewPackageError(
                f"unexpected quality claim: {claimed}"
            )
    return {
        "schema_version": str(report.get("schema_version")),
        "candidate_count": _int(aggregate.get("candidate_count")),
        "midi_count": _int(aggregate.get("midi_count")),
        "wav_count": _int(aggregate.get("wav_count")),
        "quality_proxy_pass_count": _int(aggregate.get("quality_proxy_pass_count")),
        "quality_proxy_fail_count": _int(aggregate.get("quality_proxy_fail_count")),
        "major_label_counts": _dict(aggregate.get("major_label_counts")),
        "watch_label_counts": _dict(aggregate.get("watch_label_counts")),
        "tension_repeat_feasible": bool(residual.get("tension_repeat_feasible", True)),
        "validated_listening_input_present": bool(readiness.get("validated_listening_input_present", True)),
        "musical_quality_claimed": bool(readiness.get("musical_quality_claimed", True)),
        "next_boundary": str(_dict(report.get("decision")).get("next_boundary") or ""),
    }


def _format_counts(value: dict[str, Any]) -> str:
    if not value:
        return "none"
    return ", ".join(f"{key}={value[key]}" for key in sorted(value))


def markdown_report(report: dict[str, Any]) -> str:
    aggregate = report["aggregate"]
    residual = report["residual_context"]
    readiness = report["readiness"]
    decision = report["decision"]
    lines = [
        "# Music Transformer Solo Yield Residual-Aware Final Review Package",
        "",
        "## Summary",
        "",
        f"- candidate count: `{aggregate['candidate_count']}`",
        f"- MIDI/WAV: `{aggregate['midi_count']}` / `{aggregate['wav_count']}`",
        f"- quality proxy pass/fail: `{aggregate['quality_proxy_pass_count']}` / `{aggregate['quality_proxy_fail_count']}`",
        f"- residual major labels: `{_format_counts(aggregate['major_label_counts'])}`",
        f"- residual watch labels: `{_format_counts(aggregate['watch_label_counts'])}`",
        f"- tension repeat feasible: `{_bool_token(residual['tension_repeat_feasible'])}`",
        f"- tension repeat blocked cases: `{', '.join(residual['tension_repeat_blocked_cases']) or 'none'}`",
        f"- review input template written: `{_bool_token(readiness['review_input_template_written'])}`",
        f"- validated listening input present: `{_bool_token(readiness['validated_listening_input_present'])}`",
        f"- musical quality claimed: `{_bool_token(readiness['musical_quality_claimed'])}`",
        f"- next boundary: `{decision['next_boundary']}`",
        "",
        "## Candidate Handoff",
        "",
        "| idx | case | MIDI | WAV | major labels | watch labels |",
        "|---:|---|---|---|---|---|",
    ]
    for row in report.get("candidate_handoff", []):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["review_index"]),
                    f"`{row['case_label']}`",
                    f"`{row['review_midi_path']}`",
                    f"`{row['review_wav_path']}`",
                    f"`{','.join(row['rubric_major_labels']) or 'none'}`",
                    f"`{','.join(row['rubric_watch_labels']) or 'none'}`",
                ]
            )
            + " |"
        )
    lines.extend(["", "## Not Proven", ""])
    for item in report.get("not_proven", []):
        lines.append(f"- `{item}`")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build residual-aware final review package")
    parser.add_argument("--source_package", type=str, required=True)
    parser.add_argument("--rubric_report", type=str, required=True)
    parser.add_argument("--feasibility_decision", type=str, required=True)
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/music_transformer_finetune_mvp/solo_yield_residual_aware_final_review",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--min_candidate_count", type=int, default=8)
    parser.add_argument("--require_residual_context", action="store_true")
    parser.add_argument("--require_no_quality_claim", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    report = build_final_review_package(
        source_package=read_json(Path(args.source_package)),
        rubric_report=read_json(Path(args.rubric_report)),
        feasibility_decision=read_json(Path(args.feasibility_decision)),
        output_dir=output_dir,
    )
    summary = validate_final_review_package(
        report,
        min_candidate_count=int(args.min_candidate_count),
        require_residual_context=bool(args.require_residual_context),
        require_no_quality_claim=bool(args.require_no_quality_claim),
    )
    write_json(output_dir / "residual_aware_final_review_package_summary.json", summary)
    if args.doc_path:
        write_text(Path(args.doc_path), markdown_report(report))
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
