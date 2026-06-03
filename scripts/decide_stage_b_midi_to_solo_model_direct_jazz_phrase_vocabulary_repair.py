"""Decide repair targets after model-direct songlike melody rejection analysis."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from scripts.analyze_stage_b_midi_to_solo_model_direct_songlike_rejection import (  # noqa: E402
    BOUNDARY as SOURCE_BOUNDARY,
)
from scripts.assess_stage_b_generic_base_readiness import write_json, write_text  # noqa: E402


class StageBMidiToSoloModelDirectJazzPhraseVocabularyRepairDecisionError(ValueError):
    pass


BOUNDARY = "stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_decision"
NEXT_BOUNDARY = "stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_probe"
SCHEMA_VERSION = "stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_decision_v1"


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise StageBMidiToSoloModelDirectJazzPhraseVocabularyRepairDecisionError(f"report missing: {path}")
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


def _bool_token(value: Any) -> str:
    return "true" if bool(value) else "false"


def validate_songlike_analysis(analysis_report: dict[str, Any]) -> dict[str, Any]:
    aggregate = _dict(analysis_report.get("aggregate"))
    readiness = _dict(analysis_report.get("readiness"))
    decision = _dict(analysis_report.get("decision"))
    if str(analysis_report.get("boundary") or "") != SOURCE_BOUNDARY:
        raise StageBMidiToSoloModelDirectJazzPhraseVocabularyRepairDecisionError(
            "songlike rejection analysis boundary required"
        )
    if str(decision.get("next_boundary") or "") != BOUNDARY:
        raise StageBMidiToSoloModelDirectJazzPhraseVocabularyRepairDecisionError(
            "songlike analysis must route to repair decision"
        )
    if not _list(aggregate.get("key_failure_signals")):
        raise StageBMidiToSoloModelDirectJazzPhraseVocabularyRepairDecisionError("key failure signals required")
    blocked_claims = [
        "human_audio_preference_claimed",
        "model_direct_candidate_keep_claimed",
        "model_direct_generation_quality_claimed",
        "midi_to_solo_musical_quality_claimed",
        "broad_trained_model_quality_claimed",
        "brad_style_adaptation_claimed",
    ]
    claimed = [name for name in blocked_claims if bool(readiness.get(name, False))]
    if claimed:
        raise StageBMidiToSoloModelDirectJazzPhraseVocabularyRepairDecisionError(
            f"unexpected upstream claim: {claimed}"
        )
    return {
        "candidate_count": _int(aggregate.get("candidate_count")),
        "uniform_bar_density_count": _int(aggregate.get("uniform_bar_density_count")),
        "four_notes_per_bar_template_count": _int(aggregate.get("four_notes_per_bar_template_count")),
        "duration_template_monotony_count": _int(aggregate.get("duration_template_monotony_count")),
        "ioi_template_monotony_count": _int(aggregate.get("ioi_template_monotony_count")),
        "safe_interval_cap_compression_count": _int(aggregate.get("safe_interval_cap_compression_count")),
        "four_bar_rhythm_cycle_repeated_count": _int(aggregate.get("four_bar_rhythm_cycle_repeated_count")),
        "shared_rhythm_signature_count": _int(aggregate.get("shared_rhythm_signature_count")),
        "max_abs_interval_max": _int(aggregate.get("max_abs_interval_max")),
        "key_failure_signals": _list(aggregate.get("key_failure_signals")),
    }


def repair_targets_from_evidence(evidence: dict[str, Any]) -> list[dict[str, Any]]:
    candidate_count = max(1, _int(evidence.get("candidate_count")))
    return [
        {
            "target_id": "break_uniform_bar_density",
            "current_value": _int(evidence.get("uniform_bar_density_count")),
            "target_value": "count <= 1",
            "repair_scope": "vary notes-per-bar density across the 8-bar phrase",
            "acceptance_signal": "uniform_bar_density_count <= 1",
        },
        {
            "target_id": "replace_shared_rhythm_template",
            "current_value": _int(evidence.get("shared_rhythm_signature_count")),
            "target_value": "count <= 1",
            "repair_scope": "sample or derive distinct rhythm signatures per candidate",
            "acceptance_signal": "shared_rhythm_signature_count <= 1",
        },
        {
            "target_id": "reduce_duration_ioi_monotony",
            "current_value": (
                f"{_int(evidence.get('duration_template_monotony_count'))}/"
                f"{_int(evidence.get('ioi_template_monotony_count'))}"
            ),
            "target_value": f"each count < {candidate_count}",
            "repair_scope": "expand duration and onset-position template choices",
            "acceptance_signal": "duration_template_monotony_count < candidate_count and ioi_template_monotony_count < candidate_count",
        },
        {
            "target_id": "restore_phrase_vocabulary",
            "current_value": "songlike_melody_not_soloing",
            "target_value": "phrase vocabulary candidate set",
            "repair_scope": "introduce data-derived motif cells, approach tones, enclosure-like movement, and call-response segmentation",
            "acceptance_signal": "phrase_vocabulary_source != fixed_compact_template_only",
        },
        {
            "target_id": "relax_interval_cap_tradeoff",
            "current_value": _int(evidence.get("max_abs_interval_max")),
            "target_value": "max interval <= 12 with controlled leap ratio",
            "repair_scope": "avoid wide-interval failure while allowing less songlike contour variety",
            "acceptance_signal": "max_abs_interval_max <= 12 and safe_interval_cap_compression_count < candidate_count",
        },
        {
            "target_id": "preserve_objective_guards",
            "current_value": "timing repair strict candidates available",
            "target_value": "no overlap, bounded dead-air, bounded interval",
            "repair_scope": "keep prior monophonic, timing, and technical MIDI/WAV guards",
            "acceptance_signal": "no quality claim without a later listening package",
        },
    ]


def build_jazz_phrase_vocabulary_repair_decision(
    analysis_report: dict[str, Any],
    *,
    output_dir: Path,
) -> dict[str, Any]:
    evidence = validate_songlike_analysis(analysis_report)
    repair_targets = repair_targets_from_evidence(evidence)
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "boundary": BOUNDARY,
        "source_boundary": SOURCE_BOUNDARY,
        "current_evidence": evidence,
        "repair_targets": repair_targets,
        "repair_probe_requirements": {
            "candidate_count": 3,
            "target_bars": 8,
            "require_distinct_rhythm_signatures": True,
            "require_density_variation": True,
            "require_phrase_vocabulary_source_recorded": True,
            "max_allowed_interval": 12,
            "require_no_overlap": True,
            "require_no_quality_claim": True,
            "requires_audio_render_after_probe": True,
        },
        "readiness": {
            "boundary": BOUNDARY,
            "repair_decision_completed": True,
            "midi_note_evidence_used": True,
            "single_user_review_input_used": True,
            "human_audio_preference_claimed": False,
            "model_direct_candidate_keep_claimed": False,
            "model_direct_generation_quality_claimed": False,
            "midi_to_solo_musical_quality_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
        },
        "decision": {
            "current_boundary": BOUNDARY,
            "next_boundary": NEXT_BOUNDARY,
            "auto_progress_allowed": True,
            "critical_user_input_required": False,
            "reason": "songlike rejection signals are actionable as rhythm-template and phrase-vocabulary repair targets",
        },
        "not_proven": [
            "repair_probe_improves_listening_quality",
            "jazz_solo_musical_quality",
            "human_audio_keep_preference",
            "model_direct_generation_quality",
            "broad_trained_model_quality",
            "brad_style_adaptation",
        ],
        "next_recommended_issue": "Stage B MIDI-to-solo model-direct jazz phrase vocabulary repair probe",
    }


def validate_jazz_phrase_vocabulary_repair_decision(
    report: dict[str, Any],
    *,
    expected_boundary: str | None,
    expected_next_boundary: str | None,
    min_repair_target_count: int,
    require_auto_progress_allowed: bool,
    require_no_quality_claim: bool,
) -> dict[str, Any]:
    boundary = str(report.get("boundary") or "")
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    repair_targets = _list(report.get("repair_targets"))
    if expected_boundary and boundary != expected_boundary:
        raise StageBMidiToSoloModelDirectJazzPhraseVocabularyRepairDecisionError(
            f"expected boundary {expected_boundary}, got {boundary}"
        )
    if expected_next_boundary and str(decision.get("next_boundary") or "") != expected_next_boundary:
        raise StageBMidiToSoloModelDirectJazzPhraseVocabularyRepairDecisionError("unexpected next boundary")
    if len(repair_targets) < int(min_repair_target_count):
        raise StageBMidiToSoloModelDirectJazzPhraseVocabularyRepairDecisionError("repair target count too low")
    if require_auto_progress_allowed and not bool(decision.get("auto_progress_allowed", False)):
        raise StageBMidiToSoloModelDirectJazzPhraseVocabularyRepairDecisionError("auto progress must be allowed")
    if require_no_quality_claim:
        blocked = [
            "human_audio_preference_claimed",
            "model_direct_candidate_keep_claimed",
            "model_direct_generation_quality_claimed",
            "midi_to_solo_musical_quality_claimed",
            "broad_trained_model_quality_claimed",
            "brad_style_adaptation_claimed",
        ]
        claimed = [name for name in blocked if bool(readiness.get(name, True))]
        if claimed:
            raise StageBMidiToSoloModelDirectJazzPhraseVocabularyRepairDecisionError(
                f"unexpected quality claim: {claimed}"
            )
    return {
        "boundary": boundary,
        "source_boundary": str(report.get("source_boundary") or ""),
        "repair_target_count": len(repair_targets),
        "repair_target_ids": [str(_dict(item).get("target_id") or "") for item in repair_targets],
        "require_distinct_rhythm_signatures": bool(
            _dict(report.get("repair_probe_requirements")).get("require_distinct_rhythm_signatures", False)
        ),
        "max_allowed_interval": _int(_dict(report.get("repair_probe_requirements")).get("max_allowed_interval")),
        "human_audio_preference_claimed": bool(readiness.get("human_audio_preference_claimed", True)),
        "midi_to_solo_musical_quality_claimed": bool(
            readiness.get("midi_to_solo_musical_quality_claimed", True)
        ),
        "auto_progress_allowed": bool(decision.get("auto_progress_allowed", False)),
        "critical_user_input_required": bool(decision.get("critical_user_input_required", True)),
        "next_boundary": str(decision.get("next_boundary") or ""),
        "next_recommended_issue": str(report.get("next_recommended_issue") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    evidence = report["current_evidence"]
    readiness = report["readiness"]
    decision = report["decision"]
    lines = [
        "# Stage B MIDI-to-Solo Model-Direct Jazz Phrase Vocabulary Repair Decision",
        "",
        "## Summary",
        "",
        f"- boundary: `{report['boundary']}`",
        f"- source boundary: `{report['source_boundary']}`",
        f"- next boundary: `{decision['next_boundary']}`",
        f"- candidate count: `{evidence['candidate_count']}`",
        f"- shared rhythm signature count: `{evidence['shared_rhythm_signature_count']}`",
        f"- uniform bar density count: `{evidence['uniform_bar_density_count']}`",
        f"- four-notes-per-bar template count: `{evidence['four_notes_per_bar_template_count']}`",
        f"- max abs interval max: `{evidence['max_abs_interval_max']}`",
        f"- repair target count: `{len(report.get('repair_targets', []))}`",
        f"- human/audio preference claimed: `{_bool_token(readiness['human_audio_preference_claimed'])}`",
        f"- MIDI-to-solo musical quality claimed: `{_bool_token(readiness['midi_to_solo_musical_quality_claimed'])}`",
        "",
        "## Repair Targets",
        "",
        "| target | current | target | acceptance signal |",
        "|---|---|---|---|",
    ]
    for item in report.get("repair_targets", []):
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{item['target_id']}`",
                    f"`{item['current_value']}`",
                    f"`{item['target_value']}`",
                    f"`{item['acceptance_signal']}`",
                ]
            )
            + " |"
        )
    lines.extend(["", "## Not Proven", ""])
    for item in report.get("not_proven", []):
        lines.append(f"- `{item}`")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Decide model-direct jazz phrase vocabulary repair targets")
    parser.add_argument("--songlike_rejection_analysis", type=str, required=True)
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_decision",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--expected_boundary", type=str, default="")
    parser.add_argument("--expected_next_boundary", type=str, default="")
    parser.add_argument("--min_repair_target_count", type=int, default=6)
    parser.add_argument("--require_auto_progress_allowed", action="store_true")
    parser.add_argument("--require_no_quality_claim", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    report = build_jazz_phrase_vocabulary_repair_decision(
        read_json(Path(args.songlike_rejection_analysis)),
        output_dir=output_dir,
    )
    summary = validate_jazz_phrase_vocabulary_repair_decision(
        report,
        expected_boundary=str(args.expected_boundary or ""),
        expected_next_boundary=str(args.expected_next_boundary or ""),
        min_repair_target_count=int(args.min_repair_target_count),
        require_auto_progress_allowed=bool(args.require_auto_progress_allowed),
        require_no_quality_claim=bool(args.require_no_quality_claim),
    )
    write_json(output_dir / "stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_decision.json", report)
    write_json(
        output_dir
        / "stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_decision_validation_summary.json",
        summary,
    )
    markdown = markdown_report(report)
    write_text(output_dir / "stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_decision.md", markdown)
    if args.doc_path:
        write_text(Path(args.doc_path), markdown)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
