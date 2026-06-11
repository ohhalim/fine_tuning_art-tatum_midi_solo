"""Audit reproducibility of the interval-contour final review handoff package."""

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
from scripts.build_music_transformer_solo_yield_interval_contour_final_review_handoff import (  # noqa: E402
    SCHEMA_VERSION as HANDOFF_SCHEMA_VERSION,
    validate_report as validate_handoff_report,
)
from scripts.render_stage_b_midi_to_solo_candidate_audio import sha256_file, wav_meta  # noqa: E402


SCHEMA_VERSION = "music_transformer_solo_yield_interval_contour_handoff_reproducibility_audit_v1"
BOUNDARY = "music_transformer_solo_yield_interval_contour_handoff_reproducibility_audit"
NEXT_BOUNDARY = "music_transformer_solo_yield_sampling_repeatability_audit"


class SoloYieldIntervalContourHandoffReproducibilityAuditError(ValueError):
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
        raise SoloYieldIntervalContourHandoffReproducibilityAuditError(f"json not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _require_no_quality_claim(report: dict[str, Any]) -> None:
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
        raise SoloYieldIntervalContourHandoffReproducibilityAuditError(
            f"unexpected quality claim: {claimed}"
        )


def validate_source_handoff(
    handoff_report: dict[str, Any],
    *,
    expected_candidate_count: int,
    expected_residual_label_count: int,
) -> dict[str, Any]:
    if str(handoff_report.get("schema_version") or "") != HANDOFF_SCHEMA_VERSION:
        raise SoloYieldIntervalContourHandoffReproducibilityAuditError("handoff schema required")
    summary = validate_handoff_report(handoff_report, require_no_quality_claim=True)
    if _int(summary.get("candidate_count")) != int(expected_candidate_count):
        raise SoloYieldIntervalContourHandoffReproducibilityAuditError("candidate count mismatch")
    if _int(summary.get("midi_count")) != int(expected_candidate_count):
        raise SoloYieldIntervalContourHandoffReproducibilityAuditError("MIDI count mismatch")
    if _int(summary.get("wav_count")) != int(expected_candidate_count):
        raise SoloYieldIntervalContourHandoffReproducibilityAuditError("WAV count mismatch")
    if _int(summary.get("objective_residual_label_count")) != int(expected_residual_label_count):
        raise SoloYieldIntervalContourHandoffReproducibilityAuditError(
            "objective residual label count mismatch"
        )
    if not bool(summary.get("technical_wav_validation", False)):
        raise SoloYieldIntervalContourHandoffReproducibilityAuditError("technical WAV validation required")
    return summary


def audit_candidate_files(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    audit_rows: list[dict[str, Any]] = []
    for row in rows:
        review_index = _int(row.get("review_index"))
        midi_path = Path(str(row.get("review_midi_path") or ""))
        wav_path = Path(str(row.get("review_wav_path") or ""))
        midi_exists = midi_path.exists()
        wav_exists = wav_path.exists()
        expected_midi_sha = str(row.get("review_midi_sha256") or "")
        expected_wav_sha = str(row.get("review_wav_sha256") or "")
        actual_midi_sha = sha256_file(midi_path) if midi_exists else ""
        actual_wav_meta = wav_meta(wav_path) if wav_exists else {}
        actual_wav_sha = str(actual_wav_meta.get("sha256") or "")
        audit_rows.append(
            {
                "review_index": review_index,
                "case_label": str(row.get("case_label") or ""),
                "midi_path": str(midi_path),
                "wav_path": str(wav_path),
                "midi_exists": midi_exists,
                "wav_exists": wav_exists,
                "midi_sha256_matches": bool(expected_midi_sha)
                and expected_midi_sha == actual_midi_sha,
                "wav_sha256_matches": bool(expected_wav_sha) and expected_wav_sha == actual_wav_sha,
                "sample_rate": _int(actual_wav_meta.get("sample_rate")),
                "duration_seconds": _float(actual_wav_meta.get("duration_seconds")),
                "residual_label_count": len(_list(row.get("residual_labels"))),
            }
        )
    return audit_rows


def aggregate_candidate_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    missing_midi = sum(1 for row in rows if not bool(row.get("midi_exists", False)))
    missing_wav = sum(1 for row in rows if not bool(row.get("wav_exists", False)))
    midi_mismatch = sum(1 for row in rows if not bool(row.get("midi_sha256_matches", False)))
    wav_mismatch = sum(1 for row in rows if not bool(row.get("wav_sha256_matches", False)))
    residual_labels = sum(_int(row.get("residual_label_count")) for row in rows)
    durations = [_float(row.get("duration_seconds")) for row in rows]
    sample_rates = sorted({_int(row.get("sample_rate")) for row in rows})
    return {
        "candidate_count": len(rows),
        "missing_midi_count": missing_midi,
        "missing_wav_count": missing_wav,
        "midi_checksum_mismatch_count": midi_mismatch,
        "wav_checksum_mismatch_count": wav_mismatch,
        "objective_residual_label_count": residual_labels,
        "duration_min_seconds": min(durations, default=0.0),
        "duration_max_seconds": max(durations, default=0.0),
        "sample_rates": sample_rates,
        "reproducible_handoff": (
            bool(rows)
            and missing_midi == 0
            and missing_wav == 0
            and midi_mismatch == 0
            and wav_mismatch == 0
            and residual_labels == 0
        ),
    }


def build_reproducibility_audit(
    handoff_report: dict[str, Any],
    *,
    output_dir: Path,
    expected_candidate_count: int,
    expected_residual_label_count: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    source_summary = validate_source_handoff(
        handoff_report,
        expected_candidate_count=int(expected_candidate_count),
        expected_residual_label_count=int(expected_residual_label_count),
    )
    _require_no_quality_claim(handoff_report)
    candidate_rows = [_dict(row) for row in _list(handoff_report.get("candidate_handoff"))]
    if len(candidate_rows) != int(expected_candidate_count):
        raise SoloYieldIntervalContourHandoffReproducibilityAuditError("handoff row count mismatch")
    candidate_file_audit = audit_candidate_files(candidate_rows)
    aggregate = aggregate_candidate_audit(candidate_file_audit)
    if not bool(aggregate.get("reproducible_handoff", False)):
        raise SoloYieldIntervalContourHandoffReproducibilityAuditError(
            f"handoff reproducibility audit failed: {aggregate}"
        )
    report = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "boundary": BOUNDARY,
        "source_handoff": {
            "schema_version": handoff_report.get("schema_version"),
            "output_dir": handoff_report.get("output_dir"),
            "candidate_count": _int(source_summary.get("candidate_count")),
            "midi_count": _int(source_summary.get("midi_count")),
            "wav_count": _int(source_summary.get("wav_count")),
            "objective_residual_label_count": _int(
                source_summary.get("objective_residual_label_count")
            ),
            "technical_wav_validation": bool(
                source_summary.get("technical_wav_validation", False)
            ),
        },
        "candidate_file_audit": candidate_file_audit,
        "aggregate": aggregate,
        "readiness": {
            "reproducibility_audit_completed": True,
            "reproducible_handoff": bool(aggregate.get("reproducible_handoff", False)),
            "validated_listening_input_present": False,
            "preference_fill_allowed": False,
            "audio_rendered_quality_claimed": False,
            "musical_quality_claimed": False,
            "artist_style_claimed": False,
            "production_ready_claimed": False,
        },
        "decision": {
            "current_boundary": BOUNDARY,
            "selected_next_target": "sampling_repeatability_audit",
            "next_boundary": NEXT_BOUNDARY,
            "critical_user_input_required": False,
            "reason": "final handoff is reproducible; next objective-only check is broader sampling repeatability",
        },
        "not_proven": [
            "human_audio_preference",
            "stable_jazz_solo_quality",
            "artist_level_long_solo_generation",
            "production_ready_improviser",
        ],
    }
    write_json(output_dir / "interval_contour_handoff_reproducibility_audit.json", report)
    write_json(
        output_dir / "interval_contour_handoff_reproducibility_audit_summary.json",
        validate_report(report, require_no_quality_claim=True),
    )
    write_text(output_dir / "interval_contour_handoff_reproducibility_audit.md", markdown_report(report))
    return report


def validate_report(report: dict[str, Any], *, require_no_quality_claim: bool) -> dict[str, Any]:
    if str(report.get("schema_version") or "") != SCHEMA_VERSION:
        raise SoloYieldIntervalContourHandoffReproducibilityAuditError("schema version mismatch")
    readiness = _dict(report.get("readiness"))
    if require_no_quality_claim:
        _require_no_quality_claim(report)
    aggregate = _dict(report.get("aggregate"))
    decision = _dict(report.get("decision"))
    return {
        "schema_version": str(report.get("schema_version") or ""),
        "boundary": str(report.get("boundary") or ""),
        "candidate_count": _int(aggregate.get("candidate_count")),
        "missing_midi_count": _int(aggregate.get("missing_midi_count")),
        "missing_wav_count": _int(aggregate.get("missing_wav_count")),
        "midi_checksum_mismatch_count": _int(
            aggregate.get("midi_checksum_mismatch_count")
        ),
        "wav_checksum_mismatch_count": _int(aggregate.get("wav_checksum_mismatch_count")),
        "objective_residual_label_count": _int(aggregate.get("objective_residual_label_count")),
        "reproducible_handoff": bool(readiness.get("reproducible_handoff", False)),
        "selected_next_target": str(decision.get("selected_next_target") or ""),
        "next_boundary": str(decision.get("next_boundary") or ""),
        "musical_quality_claimed": bool(readiness.get("musical_quality_claimed", True)),
    }


def markdown_report(report: dict[str, Any]) -> str:
    aggregate = report["aggregate"]
    readiness = report["readiness"]
    decision = report["decision"]
    lines = [
        "# Music Transformer Solo Yield Interval Contour Handoff Reproducibility Audit",
        "",
        "## Summary",
        "",
        f"- boundary: `{report['boundary']}`",
        f"- candidate count: `{aggregate['candidate_count']}`",
        f"- missing MIDI/WAV: `{aggregate['missing_midi_count']}` / `{aggregate['missing_wav_count']}`",
        f"- checksum mismatch MIDI/WAV: `{aggregate['midi_checksum_mismatch_count']}` / `{aggregate['wav_checksum_mismatch_count']}`",
        f"- objective residual label count: `{aggregate['objective_residual_label_count']}`",
        f"- duration range seconds: `{float(aggregate['duration_min_seconds']):.4f}` - `{float(aggregate['duration_max_seconds']):.4f}`",
        f"- sample rates: `{','.join(str(rate) for rate in aggregate['sample_rates'])}`",
        f"- reproducible handoff: `{_bool_token(readiness['reproducible_handoff'])}`",
        f"- selected next target: `{decision['selected_next_target']}`",
        f"- next boundary: `{decision['next_boundary']}`",
        f"- musical quality claimed: `{_bool_token(readiness['musical_quality_claimed'])}`",
        "",
        "## Candidate File Audit",
        "",
    ]
    for row in report.get("candidate_file_audit", []):
        lines.extend(
            [
                f"- candidate `{row['review_index']}` / `{row['case_label']}`",
                f"  - MIDI exists/checksum: `{_bool_token(row['midi_exists'])}` / `{_bool_token(row['midi_sha256_matches'])}`",
                f"  - WAV exists/checksum: `{_bool_token(row['wav_exists'])}` / `{_bool_token(row['wav_sha256_matches'])}`",
                f"  - duration seconds: `{float(row['duration_seconds']):.4f}`",
                f"  - residual label count: `{row['residual_label_count']}`",
            ]
        )
    lines.extend(["", "## Not Proven", ""])
    for item in report.get("not_proven", []):
        lines.append(f"- `{item}`")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit interval contour final handoff reproducibility")
    parser.add_argument(
        "--handoff_report",
        type=str,
        default=(
            "outputs/music_transformer_finetune_mvp/solo_yield_interval_contour_final_handoff/"
            "issue_1314_interval_contour_final_handoff/interval_contour_final_review_handoff.json"
        ),
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/music_transformer_finetune_mvp/solo_yield_interval_contour_handoff_audit",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--expected_candidate_count", type=int, default=8)
    parser.add_argument("--expected_residual_label_count", type=int, default=0)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--require_no_quality_claim", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    report = build_reproducibility_audit(
        read_json(Path(args.handoff_report)),
        output_dir=output_dir,
        expected_candidate_count=int(args.expected_candidate_count),
        expected_residual_label_count=int(args.expected_residual_label_count),
    )
    summary = validate_report(report, require_no_quality_claim=bool(args.require_no_quality_claim))
    if args.doc_path:
        write_text(Path(args.doc_path), markdown_report(report))
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
