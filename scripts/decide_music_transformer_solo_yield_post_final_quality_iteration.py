"""Decide the next quality-iteration target after solo-yield final status audit."""

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


SCHEMA_VERSION = "music_transformer_solo_yield_post_final_quality_iteration_decision_v1"


class SoloYieldPostFinalQualityIterationDecisionError(ValueError):
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
        raise SoloYieldPostFinalQualityIterationDecisionError(f"json not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _reject_quality_claim(*reports: dict[str, Any]) -> None:
    claimed: list[str] = []
    for index, report in enumerate(reports, start=1):
        for section_name in ("final_status", "readiness"):
            section = _dict(report.get(section_name))
            for key in ("musical_quality_claimed", "artist_style_claimed", "production_ready_claimed"):
                if bool(section.get(key, False)):
                    claimed.append(f"report_{index}:{section_name}:{key}")
    if claimed:
        raise SoloYieldPostFinalQualityIterationDecisionError(
            f"unexpected quality claim: {claimed}"
        )


def build_decision_report(
    *,
    final_status_audit: dict[str, Any],
    objective_decision: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    _reject_quality_claim(final_status_audit, objective_decision)

    final_status = _dict(final_status_audit.get("final_status"))
    final_readiness = _dict(final_status_audit.get("readiness"))
    objective_summary = _dict(objective_decision.get("objective_summary"))
    objective_readiness = _dict(objective_decision.get("readiness"))

    technical_ready = bool(final_status.get("technical_mvp_evidence_ready", False))
    final_status_completed = bool(final_readiness.get("final_status_audit_completed", False))
    reproducible_handoff = bool(final_status.get("reproducible_handoff", False))
    validated_listening_input_present = bool(
        final_status.get("validated_listening_input_present", False)
    )
    preference_fill_allowed = bool(final_status.get("preference_fill_allowed", True))
    selected_objective_count = _int(
        objective_readiness.get("selected_objective_candidate_count")
    )
    candidate_count = _int(objective_summary.get("candidate_count"))

    if not final_status_completed:
        raise SoloYieldPostFinalQualityIterationDecisionError(
            "final status audit completion required"
        )
    if not technical_ready:
        raise SoloYieldPostFinalQualityIterationDecisionError(
            "technical MVP evidence readiness required"
        )
    if not reproducible_handoff:
        raise SoloYieldPostFinalQualityIterationDecisionError(
            "reproducible final handoff required"
        )
    if selected_objective_count < 1:
        raise SoloYieldPostFinalQualityIterationDecisionError(
            "selected objective candidates required"
        )

    selected_target = (
        "human_listening_review_consolidation"
        if validated_listening_input_present and not preference_fill_allowed
        else "objective_candidate_quality_rubric"
    )
    next_boundary = (
        "music_transformer_solo_yield_listening_review_consolidation"
        if selected_target == "human_listening_review_consolidation"
        else "music_transformer_solo_yield_objective_quality_rubric_baseline"
    )
    reason = (
        "validated listening input exists, so consolidate preference evidence before more repairs"
        if selected_target == "human_listening_review_consolidation"
        else (
            "technical gates are clean and listening input is absent; label latest candidates "
            "with an objective quality rubric before further repair or training"
        )
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "source_reports": {
            "final_status_audit": final_status_audit.get("output_dir"),
            "objective_decision": objective_decision.get("output_dir"),
        },
        "source_summary": {
            "technical_mvp_evidence_ready": technical_ready,
            "strict_valid_sample_count": _int(final_status.get("strict_valid_sample_count")),
            "sample_count": _int(final_status.get("sample_count")),
            "grammar_gate_sample_count": _int(final_status.get("grammar_gate_sample_count")),
            "final_handoff_midi_count": _int(final_status.get("final_handoff_midi_count")),
            "final_handoff_wav_count": _int(final_status.get("final_handoff_wav_count")),
            "reproducible_handoff": reproducible_handoff,
            "validated_listening_input_present": validated_listening_input_present,
            "preference_fill_allowed": preference_fill_allowed,
            "candidate_count": candidate_count,
            "selected_objective_candidate_count": selected_objective_count,
            "score_min": _float(objective_summary.get("score_min")),
            "score_max": _float(objective_summary.get("score_max")),
            "note_count_avg": _float(objective_summary.get("note_count_avg")),
            "dead_air_min": _float(objective_summary.get("dead_air_min")),
            "dead_air_max": _float(objective_summary.get("dead_air_max")),
        },
        "decision": {
            "current_boundary": "music_transformer_solo_yield_post_final_quality_iteration_decision",
            "selected_next_target": selected_target,
            "next_boundary": next_boundary,
            "critical_user_input_required": False,
            "reason": reason,
        },
        "readiness": {
            "post_final_quality_iteration_decision_completed": True,
            "objective_quality_rubric_required": selected_target == "objective_candidate_quality_rubric",
            "human_listening_review_pending": not validated_listening_input_present,
            "musical_quality_claimed": False,
            "artist_style_claimed": False,
            "production_ready_claimed": False,
        },
        "not_proven": [
            "human_audio_preference",
            "stable_jazz_solo_quality",
            "objective_quality_rubric_result",
            "artist_level_long_solo_generation",
        ],
    }


def validate_decision_report(
    report: dict[str, Any],
    *,
    expected_target: str,
    require_no_quality_claim: bool,
) -> dict[str, Any]:
    if str(report.get("schema_version")) != SCHEMA_VERSION:
        raise SoloYieldPostFinalQualityIterationDecisionError("schema version mismatch")
    source = _dict(report.get("source_summary"))
    decision = _dict(report.get("decision"))
    readiness = _dict(report.get("readiness"))
    if str(decision.get("selected_next_target") or "") != expected_target:
        raise SoloYieldPostFinalQualityIterationDecisionError("selected target mismatch")
    if not bool(readiness.get("post_final_quality_iteration_decision_completed", False)):
        raise SoloYieldPostFinalQualityIterationDecisionError("decision completion required")
    if require_no_quality_claim:
        claimed = [
            key
            for key in ("musical_quality_claimed", "artist_style_claimed", "production_ready_claimed")
            if bool(readiness.get(key, True))
        ]
        if claimed:
            raise SoloYieldPostFinalQualityIterationDecisionError(
                f"unexpected quality claim: {claimed}"
            )
    return {
        "schema_version": str(report.get("schema_version")),
        "technical_mvp_evidence_ready": bool(source.get("technical_mvp_evidence_ready", False)),
        "strict_valid_sample_count": _int(source.get("strict_valid_sample_count")),
        "sample_count": _int(source.get("sample_count")),
        "reproducible_handoff": bool(source.get("reproducible_handoff", False)),
        "validated_listening_input_present": bool(
            source.get("validated_listening_input_present", False)
        ),
        "candidate_count": _int(source.get("candidate_count")),
        "selected_objective_candidate_count": _int(
            source.get("selected_objective_candidate_count")
        ),
        "selected_next_target": str(decision.get("selected_next_target") or ""),
        "next_boundary": str(decision.get("next_boundary") or ""),
        "critical_user_input_required": bool(decision.get("critical_user_input_required", True)),
        "musical_quality_claimed": bool(readiness.get("musical_quality_claimed", True)),
    }


def markdown_report(report: dict[str, Any]) -> str:
    source = report["source_summary"]
    decision = report["decision"]
    readiness = report["readiness"]
    lines = [
        "# Music Transformer Solo Yield Post-Final Quality Iteration Decision",
        "",
        "## Summary",
        "",
        f"- technical MVP evidence ready: `{_bool_token(source['technical_mvp_evidence_ready'])}`",
        f"- strict yield: `{source['strict_valid_sample_count']}` / `{source['sample_count']}`",
        f"- grammar yield: `{source['grammar_gate_sample_count']}` / `{source['sample_count']}`",
        f"- final handoff MIDI/WAV: `{source['final_handoff_midi_count']}` / `{source['final_handoff_wav_count']}`",
        f"- reproducible handoff: `{_bool_token(source['reproducible_handoff'])}`",
        f"- validated listening input present: `{_bool_token(source['validated_listening_input_present'])}`",
        f"- candidate count: `{source['candidate_count']}`",
        f"- selected objective candidates: `{source['selected_objective_candidate_count']}`",
        f"- score range: `{float(source['score_min']):.4f}` - `{float(source['score_max']):.4f}`",
        f"- note count avg: `{float(source['note_count_avg']):.4f}`",
        f"- dead-air range: `{float(source['dead_air_min']):.4f}` - `{float(source['dead_air_max']):.4f}`",
        f"- selected next target: `{decision['selected_next_target']}`",
        f"- next boundary: `{decision['next_boundary']}`",
        f"- critical user input required: `{_bool_token(decision['critical_user_input_required'])}`",
        f"- musical quality claimed: `{_bool_token(readiness['musical_quality_claimed'])}`",
        "",
        "## Decision Reason",
        "",
        f"- {decision['reason']}",
        "",
        "## Not Proven",
        "",
    ]
    for item in report.get("not_proven", []):
        lines.append(f"- `{item}`")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Decide post-final Music Transformer solo-yield quality iteration target"
    )
    parser.add_argument("--final_status_audit", type=str, required=True)
    parser.add_argument("--objective_decision", type=str, required=True)
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/music_transformer_finetune_mvp/solo_yield_post_final_quality_iteration_decision",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument(
        "--expected_target",
        type=str,
        default="objective_candidate_quality_rubric",
    )
    parser.add_argument("--require_no_quality_claim", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    report = build_decision_report(
        final_status_audit=read_json(Path(args.final_status_audit)),
        objective_decision=read_json(Path(args.objective_decision)),
        output_dir=output_dir,
    )
    summary = validate_decision_report(
        report,
        expected_target=str(args.expected_target),
        require_no_quality_claim=bool(args.require_no_quality_claim),
    )
    write_json(output_dir / "post_final_quality_iteration_decision.json", report)
    write_json(output_dir / "post_final_quality_iteration_decision_summary.json", summary)
    markdown = markdown_report(report)
    write_text(output_dir / "post_final_quality_iteration_decision.md", markdown)
    if args.doc_path:
        write_text(Path(args.doc_path), markdown)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
