"""Audit final technical status for the Music Transformer solo-yield MVP."""

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


SCHEMA_VERSION = "music_transformer_solo_yield_final_status_audit_v1"


class SoloYieldFinalStatusAuditError(ValueError):
    pass


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


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
        raise SoloYieldFinalStatusAuditError(f"json not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _require_no_quality_claim(*reports: dict[str, Any]) -> None:
    claimed: list[str] = []
    for index, report in enumerate(reports, start=1):
        readiness = _dict(report.get("readiness"))
        for key in ("musical_quality_claimed", "artist_style_claimed", "production_ready_claimed"):
            if bool(readiness.get(key, False)):
                claimed.append(f"report_{index}:{key}")
    if claimed:
        raise SoloYieldFinalStatusAuditError(f"unexpected quality claim: {claimed}")


def build_audit_report(
    *,
    repair_sweep: dict[str, Any],
    repaired_package: dict[str, Any],
    repaired_guard: dict[str, Any],
    repaired_objective: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    _require_no_quality_claim(repair_sweep, repaired_package, repaired_guard, repaired_objective)

    sweep_aggregate = _dict(repair_sweep.get("aggregate"))
    sweep_readiness = _dict(repair_sweep.get("readiness"))
    package_readiness = _dict(repaired_package.get("readiness"))
    guard_readiness = _dict(repaired_guard.get("readiness"))
    guard_validation = _dict(repaired_guard.get("input_validation"))
    objective_readiness = _dict(repaired_objective.get("readiness"))
    objective_summary = _dict(repaired_objective.get("objective_summary"))

    repaired_candidate_count = _int(repaired_package.get("candidate_count"))
    midi_count = _int(package_readiness.get("candidate_midi_files_copied"))
    wav_count = _int(package_readiness.get("candidate_wav_files_copied"))
    selected_objective_count = _int(objective_readiness.get("selected_objective_candidate_count"))
    strict_count = _int(sweep_aggregate.get("strict_valid_sample_count"))
    sample_count = _int(sweep_aggregate.get("sample_count"))
    grammar_count = _int(sweep_aggregate.get("grammar_gate_sample_count"))

    technical_ready = bool(
        strict_count > 0
        and grammar_count == sample_count
        and repaired_candidate_count >= 1
        and midi_count == repaired_candidate_count
        and wav_count == repaired_candidate_count
        and selected_objective_count >= 1
        and not bool(guard_readiness.get("preference_fill_allowed", True))
    )

    report = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "source_reports": {
            "repair_sweep": repair_sweep.get("output_dir"),
            "repaired_package": repaired_package.get("output_dir"),
            "repaired_guard": repaired_guard.get("output_dir"),
            "repaired_objective": repaired_objective.get("output_dir"),
        },
        "final_status": {
            "technical_mvp_evidence_ready": technical_ready,
            "music_transformer_checkpoint_generation_used": bool(
                sweep_readiness.get("music_transformer_checkpoint_generation_used", False)
            ),
            "constrained_decoding_used": bool(sweep_readiness.get("constrained_decoding_used", False)),
            "case_count": _int(sweep_aggregate.get("case_count")),
            "sample_count": sample_count,
            "strict_valid_sample_count": strict_count,
            "grammar_gate_sample_count": grammar_count,
            "strict_yield_rate": _float(sweep_aggregate.get("strict_yield_rate")),
            "min_case_strict_yield_rate": _float(sweep_aggregate.get("min_case_strict_yield_rate")),
            "rendered_audio_file_count": _int(sweep_aggregate.get("rendered_audio_file_count")),
            "repaired_candidate_count": repaired_candidate_count,
            "candidate_midi_files_copied": midi_count,
            "candidate_wav_files_copied": wav_count,
            "selected_objective_candidate_count": selected_objective_count,
            "objective_dead_air_min": _float(objective_summary.get("dead_air_min")),
            "objective_dead_air_max": _float(objective_summary.get("dead_air_max")),
            "validated_listening_input_present": bool(
                guard_validation.get("validated_listening_input_present", False)
            ),
            "preference_fill_allowed": bool(guard_readiness.get("preference_fill_allowed", True)),
            "musical_quality_claimed": False,
            "artist_style_claimed": False,
            "production_ready_claimed": False,
            "raw_artifact_upload_required": False,
        },
        "readiness": {
            "final_status_audit_completed": True,
            "technical_mvp_evidence_ready": technical_ready,
            "human_listening_review_pending": not bool(
                guard_validation.get("validated_listening_input_present", False)
            ),
            "musical_quality_claimed": False,
            "artist_style_claimed": False,
            "production_ready_claimed": False,
        },
        "decision": {
            "current_boundary": "music_transformer_solo_yield_final_status_audit",
            "next_boundary": "music_transformer_solo_yield_readme_final_evidence_refresh",
            "critical_user_input_required": False,
            "reason": "technical evidence is ready; refresh README evidence without quality claim",
        },
        "not_proven": [
            "human_audio_preference",
            "stable_jazz_solo_quality",
            "artist_level_long_solo_generation",
            "production_ready_improviser",
        ],
    }
    write_json(output_dir / "final_status_audit.json", report)
    write_json(output_dir / "final_status_audit_summary.json", validate_audit_report(report))
    write_text(output_dir / "final_status_audit.md", markdown_report(report))
    return report


def validate_audit_report(report: dict[str, Any]) -> dict[str, Any]:
    final_status = _dict(report.get("final_status"))
    readiness = _dict(report.get("readiness"))
    if str(report.get("schema_version")) != SCHEMA_VERSION:
        raise SoloYieldFinalStatusAuditError("schema version mismatch")
    if not bool(readiness.get("final_status_audit_completed", False)):
        raise SoloYieldFinalStatusAuditError("final status audit completion required")
    claimed = [
        key
        for key in ("musical_quality_claimed", "artist_style_claimed", "production_ready_claimed")
        if bool(final_status.get(key, True)) or bool(readiness.get(key, True))
    ]
    if claimed:
        raise SoloYieldFinalStatusAuditError(f"unexpected quality claim: {claimed}")
    return {
        "schema_version": str(report.get("schema_version")),
        "technical_mvp_evidence_ready": bool(final_status.get("technical_mvp_evidence_ready", False)),
        "case_count": _int(final_status.get("case_count")),
        "sample_count": _int(final_status.get("sample_count")),
        "strict_valid_sample_count": _int(final_status.get("strict_valid_sample_count")),
        "grammar_gate_sample_count": _int(final_status.get("grammar_gate_sample_count")),
        "strict_yield_rate": _float(final_status.get("strict_yield_rate")),
        "min_case_strict_yield_rate": _float(final_status.get("min_case_strict_yield_rate")),
        "rendered_audio_file_count": _int(final_status.get("rendered_audio_file_count")),
        "repaired_candidate_count": _int(final_status.get("repaired_candidate_count")),
        "selected_objective_candidate_count": _int(final_status.get("selected_objective_candidate_count")),
        "validated_listening_input_present": bool(final_status.get("validated_listening_input_present", True)),
        "preference_fill_allowed": bool(final_status.get("preference_fill_allowed", True)),
        "musical_quality_claimed": bool(final_status.get("musical_quality_claimed", True)),
        "next_boundary": str(_dict(report.get("decision")).get("next_boundary") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    status = report["final_status"]
    decision = report["decision"]
    lines = [
        "# Music Transformer Solo Yield Final Status Audit",
        "",
        "## Summary",
        "",
        f"- technical MVP evidence ready: `{_bool_token(status['technical_mvp_evidence_ready'])}`",
        f"- checkpoint generation used: `{_bool_token(status['music_transformer_checkpoint_generation_used'])}`",
        f"- constrained decoding used: `{_bool_token(status['constrained_decoding_used'])}`",
        f"- case count: `{status['case_count']}`",
        f"- sample count: `{status['sample_count']}`",
        f"- strict yield: `{status['strict_valid_sample_count']}` / `{status['sample_count']}`",
        f"- grammar yield: `{status['grammar_gate_sample_count']}` / `{status['sample_count']}`",
        f"- strict yield rate: `{float(status['strict_yield_rate']):.4f}`",
        f"- min case strict yield rate: `{float(status['min_case_strict_yield_rate']):.4f}`",
        f"- rendered WAV files: `{status['rendered_audio_file_count']}`",
        f"- repaired candidate count: `{status['repaired_candidate_count']}`",
        f"- selected objective candidates: `{status['selected_objective_candidate_count']}`",
        f"- objective dead-air range: `{float(status['objective_dead_air_min']):.4f}` - `{float(status['objective_dead_air_max']):.4f}`",
        f"- validated listening input present: `{_bool_token(status['validated_listening_input_present'])}`",
        f"- preference fill allowed: `{_bool_token(status['preference_fill_allowed'])}`",
        f"- musical quality claimed: `{_bool_token(status['musical_quality_claimed'])}`",
        f"- raw artifact upload required: `{_bool_token(status['raw_artifact_upload_required'])}`",
        f"- next boundary: `{decision['next_boundary']}`",
        "",
        "## Not Proven",
        "",
    ]
    for item in report.get("not_proven", []):
        lines.append(f"- `{item}`")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit Music Transformer solo-yield final technical status")
    parser.add_argument(
        "--repair_sweep_report",
        type=str,
        default=(
            "outputs/music_transformer_finetune_mvp/solo_yield_sweep/"
            "issue_1248_4bar_dead_air_repair_n9_seq160/solo_yield_sweep_report.json"
        ),
    )
    parser.add_argument(
        "--repaired_package_report",
        type=str,
        default=(
            "outputs/music_transformer_finetune_mvp/solo_yield_listening_review/"
            "issue_1250_4bar_repaired_top8_listening_package/listening_review_package.json"
        ),
    )
    parser.add_argument(
        "--repaired_guard_report",
        type=str,
        default=(
            "outputs/music_transformer_finetune_mvp/solo_yield_listening_input_guard/"
            "issue_1252_4bar_repaired_input_guard/listening_input_guard.json"
        ),
    )
    parser.add_argument(
        "--repaired_objective_report",
        type=str,
        default=(
            "outputs/music_transformer_finetune_mvp/solo_yield_objective_next_decision/"
            "issue_1254_4bar_repaired_objective_next_decision/objective_next_decision.json"
        ),
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/music_transformer_finetune_mvp/solo_yield_final_status_audit",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    report = build_audit_report(
        repair_sweep=read_json(Path(args.repair_sweep_report)),
        repaired_package=read_json(Path(args.repaired_package_report)),
        repaired_guard=read_json(Path(args.repaired_guard_report)),
        repaired_objective=read_json(Path(args.repaired_objective_report)),
        output_dir=output_dir,
    )
    summary = validate_audit_report(report)
    if args.doc_path:
        write_text(Path(args.doc_path), markdown_report(report))
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
