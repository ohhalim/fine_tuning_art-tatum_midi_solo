"""Review failing cases from Music Transformer solo-yield sweep reports."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from scripts.assess_stage_b_generic_base_readiness import write_json, write_text  # noqa: E402


SCHEMA_VERSION = "music_transformer_solo_yield_failure_review_v1"


class SoloYieldFailureReviewError(ValueError):
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
        raise SoloYieldFailureReviewError(f"json not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def avg(values: Sequence[float]) -> float:
    return float(mean(values)) if values else 0.0


def compact_sample(row: dict[str, Any]) -> dict[str, Any]:
    metrics = _dict(row.get("metrics"))
    collapse = _dict(row.get("collapse"))
    postprocess = _dict(row.get("postprocess"))
    contour = _dict(row.get("phrase_contour"))
    rhythm = _dict(row.get("rhythm_profile"))
    pitch_roles = _dict(row.get("pitch_roles"))
    collapse_gate = _dict(row.get("collapse_gate"))
    return {
        "sample_index": _int(row.get("sample_index")),
        "sample_seed": _int(row.get("sample_seed")),
        "valid": bool(row.get("valid", False)),
        "strict_valid": bool(row.get("strict_valid", False)),
        "failure_reason": row.get("failure_reason"),
        "diagnostic_failure_reason": row.get("diagnostic_failure_reason"),
        "collapse_gate_failure_reasons": _list(collapse_gate.get("failure_reasons")),
        "note_count": _int(metrics.get("note_count")),
        "unique_pitch_count": _int(metrics.get("unique_pitch_count")),
        "dead_air_ratio": _float(metrics.get("dead_air_ratio")),
        "phrase_coverage_ratio": _float(metrics.get("phrase_coverage_ratio")),
        "chord_tone_ratio": _float(metrics.get("chord_tone_ratio")),
        "repetition_score": _float(metrics.get("repetition_score")),
        "postprocess_removed_note_count": _int(postprocess.get("removed_note_count")),
        "postprocess_removal_ratio": _float(collapse.get("postprocess_removal_ratio")),
        "before_max_simultaneous_notes": _int(postprocess.get("before_max_simultaneous_notes")),
        "after_max_simultaneous_notes": _int(postprocess.get("after_max_simultaneous_notes")),
        "collapse_warning": bool(collapse.get("collapse_warning", False)),
        "collapse_reasons": _list(collapse.get("collapse_reasons")),
        "repeated_position_pitch_pair_ratio": _float(collapse.get("repeated_position_pitch_pair_ratio")),
        "direction_change_ratio": _float(contour.get("direction_change_ratio")),
        "pitch_span": _int(contour.get("pitch_span")),
        "syncopated_onset_ratio": _float(rhythm.get("syncopated_onset_ratio")),
        "duration_diversity_ratio": _float(rhythm.get("duration_diversity_ratio")),
        "most_common_duration_ratio": _float(rhythm.get("most_common_duration_ratio")),
        "ioi_diversity_ratio": _float(rhythm.get("ioi_diversity_ratio")),
        "tension_ratio": _float(pitch_roles.get("tension_ratio")),
        "non_chord_tone_ratio": _float(pitch_roles.get("non_chord_tone_ratio")),
        "midi_path": str(row.get("midi_path") or ""),
    }


def review_probe_case(case: dict[str, Any]) -> dict[str, Any]:
    probe_report_path = Path(str(case.get("probe_report_path") or ""))
    probe_report = read_json(probe_report_path)
    summary = _dict(probe_report.get("summary"))
    samples = [compact_sample(_dict(row)) for row in _list(probe_report.get("samples"))]
    invalid_samples = [row for row in samples if not bool(row["strict_valid"])]
    strict_samples = [row for row in samples if bool(row["strict_valid"])]
    failure_reasons = Counter(str(row.get("failure_reason") or "none") for row in invalid_samples)
    diagnostic_reasons = Counter(str(row.get("diagnostic_failure_reason") or "none") for row in invalid_samples)
    collapse_gate_reasons = Counter(
        reason
        for row in invalid_samples
        for reason in _list(row.get("collapse_gate_failure_reasons"))
    )
    dead_air_fail_count = sum(
        1
        for row in invalid_samples
        if str(row.get("failure_reason") or "").startswith("dead-air ratio too high")
    )
    postprocess_overlap_removed_count = sum(
        1 for row in invalid_samples if _int(row.get("postprocess_removed_note_count")) > 0
    )
    grammar_excluded = _int(case.get("grammar_gate_sample_count")) == _int(case.get("sample_count"))
    collapse_excluded = _int(case.get("collapse_warning_sample_count")) == 0
    dominant_failure = "undetermined"
    repair_target = "inspect_failed_samples"
    if invalid_samples and dead_air_fail_count == len(invalid_samples):
        dominant_failure = "dead_air_threshold_miss"
        repair_target = "duration_fill_or_overlap_aftercare"
    elif collapse_gate_reasons:
        dominant_failure = "strict_collapse_gate_miss"
        repair_target = "collapse_gate_repair"
    elif failure_reasons:
        dominant_failure = "midi_review_gate_miss"
        repair_target = "midi_review_gate_repair"

    return {
        "label": str(case.get("label")),
        "chords": str(case.get("chords")),
        "seed": _int(case.get("seed")),
        "probe_report_path": str(probe_report_path),
        "sample_count": _int(case.get("sample_count")),
        "valid_sample_count": _int(case.get("valid_sample_count")),
        "strict_valid_sample_count": _int(case.get("strict_valid_sample_count")),
        "grammar_gate_sample_count": _int(case.get("grammar_gate_sample_count")),
        "collapse_warning_sample_count": _int(case.get("collapse_warning_sample_count")),
        "strict_yield_rate": _float(case.get("strict_yield_rate")),
        "summary_failure_reasons": _dict(summary.get("failure_reasons")),
        "summary_strict_failure_reasons": _dict(summary.get("strict_failure_reasons")),
        "failure_reason_counts": dict(sorted(failure_reasons.items())),
        "diagnostic_failure_reason_counts": dict(sorted(diagnostic_reasons.items())),
        "collapse_gate_failure_reason_counts": dict(sorted(collapse_gate_reasons.items())),
        "invalid_sample_count": len(invalid_samples),
        "dead_air_fail_count": dead_air_fail_count,
        "postprocess_overlap_removed_invalid_count": postprocess_overlap_removed_count,
        "grammar_gate_excluded_as_primary_cause": grammar_excluded,
        "collapse_warning_excluded_as_primary_cause": collapse_excluded,
        "invalid_avg_dead_air_ratio": avg([_float(row.get("dead_air_ratio")) for row in invalid_samples]),
        "strict_avg_dead_air_ratio": avg([_float(row.get("dead_air_ratio")) for row in strict_samples]),
        "invalid_avg_postprocess_removal_ratio": avg(
            [_float(row.get("postprocess_removal_ratio")) for row in invalid_samples]
        ),
        "strict_avg_postprocess_removal_ratio": avg(
            [_float(row.get("postprocess_removal_ratio")) for row in strict_samples]
        ),
        "invalid_avg_note_count": avg([_float(row.get("note_count")) for row in invalid_samples]),
        "strict_avg_note_count": avg([_float(row.get("note_count")) for row in strict_samples]),
        "dominant_failure": dominant_failure,
        "repair_target": repair_target,
        "samples": samples,
    }


def build_failure_review(
    sweep_report: dict[str, Any],
    *,
    output_dir: Path,
) -> dict[str, Any]:
    if output_dir.exists():
        raise SoloYieldFailureReviewError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    aggregate = _dict(sweep_report.get("aggregate"))
    failing_labels = {str(item.get("label")) for item in _list(aggregate.get("failing_cases"))}
    failing_cases = [
        _dict(case)
        for case in _list(sweep_report.get("cases"))
        if str(_dict(case).get("label")) in failing_labels
    ]
    case_reviews = [review_probe_case(case) for case in failing_cases]
    dominant_targets = Counter(str(case.get("repair_target")) for case in case_reviews)
    next_repair_target = dominant_targets.most_common(1)[0][0] if dominant_targets else "none"
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "source_sweep": {
            "schema_version": sweep_report.get("schema_version"),
            "output_dir": sweep_report.get("output_dir"),
            "case_count": _int(aggregate.get("case_count")),
            "sample_count": _int(aggregate.get("sample_count")),
            "strict_valid_sample_count": _int(aggregate.get("strict_valid_sample_count")),
            "strict_yield_rate": _float(aggregate.get("strict_yield_rate")),
            "min_case_strict_yield_rate": _float(aggregate.get("min_case_strict_yield_rate")),
        },
        "case_reviews": case_reviews,
        "aggregate": {
            "failing_case_count": len(case_reviews),
            "reviewed_invalid_sample_count": sum(_int(case.get("invalid_sample_count")) for case in case_reviews),
            "dead_air_fail_count": sum(_int(case.get("dead_air_fail_count")) for case in case_reviews),
            "grammar_primary_cause_excluded_count": sum(
                1 for case in case_reviews if bool(case.get("grammar_gate_excluded_as_primary_cause", False))
            ),
            "collapse_primary_cause_excluded_count": sum(
                1 for case in case_reviews if bool(case.get("collapse_warning_excluded_as_primary_cause", False))
            ),
            "repair_target_counts": dict(sorted(dominant_targets.items())),
        },
        "decision": {
            "current_boundary": "music_transformer_solo_yield_failure_case_review",
            "next_boundary": (
                "music_transformer_solo_yield_dead_air_repair_sweep"
                if next_repair_target == "duration_fill_or_overlap_aftercare"
                else "music_transformer_solo_yield_failure_repair_decision"
            ),
            "selected_repair_target": next_repair_target,
            "critical_user_input_required": False,
            "reason": (
                "failing samples missed MIDI review gate by dead-air threshold while grammar and collapse warnings were excluded"
                if next_repair_target == "duration_fill_or_overlap_aftercare"
                else "failure target requires additional review"
            ),
        },
        "readiness": {
            "failure_case_review_completed": True,
            "music_transformer_checkpoint_generation_used": True,
            "constrained_decoding_used": True,
            "musical_quality_claimed": False,
            "artist_style_claimed": False,
            "production_ready_claimed": False,
        },
        "not_proven": [
            "human_audio_preference",
            "stable_jazz_solo_quality",
            "repair_effectiveness",
            "artist_level_long_solo_generation",
        ],
    }


def validate_review(
    report: dict[str, Any],
    *,
    min_failing_cases: int,
    require_no_quality_claim: bool,
) -> dict[str, Any]:
    aggregate = _dict(report.get("aggregate"))
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    if _int(aggregate.get("failing_case_count")) < int(min_failing_cases):
        raise SoloYieldFailureReviewError("failing case count below requirement")
    if require_no_quality_claim:
        claimed = [
            key
            for key in ("musical_quality_claimed", "artist_style_claimed", "production_ready_claimed")
            if bool(readiness.get(key, True))
        ]
        if claimed:
            raise SoloYieldFailureReviewError(f"unexpected quality claim: {claimed}")
    return {
        "schema_version": str(report.get("schema_version")),
        "failing_case_count": _int(aggregate.get("failing_case_count")),
        "reviewed_invalid_sample_count": _int(aggregate.get("reviewed_invalid_sample_count")),
        "dead_air_fail_count": _int(aggregate.get("dead_air_fail_count")),
        "grammar_primary_cause_excluded_count": _int(aggregate.get("grammar_primary_cause_excluded_count")),
        "collapse_primary_cause_excluded_count": _int(aggregate.get("collapse_primary_cause_excluded_count")),
        "selected_repair_target": str(decision.get("selected_repair_target") or ""),
        "next_boundary": str(decision.get("next_boundary") or ""),
        "critical_user_input_required": bool(decision.get("critical_user_input_required", True)),
        "musical_quality_claimed": bool(readiness.get("musical_quality_claimed", True)),
    }


def markdown_report(report: dict[str, Any]) -> str:
    source = report["source_sweep"]
    aggregate = report["aggregate"]
    decision = report["decision"]
    readiness = report["readiness"]
    lines = [
        "# Stage B MIDI-to-Solo Yield Failure Case Review",
        "",
        "## Summary",
        "",
        f"- source strict yield: `{source['strict_valid_sample_count']}` / `{source['sample_count']}`",
        f"- min case strict yield rate: `{float(source['min_case_strict_yield_rate']):.4f}`",
        f"- failing case count: `{aggregate['failing_case_count']}`",
        f"- reviewed invalid samples: `{aggregate['reviewed_invalid_sample_count']}`",
        f"- dead-air failure count: `{aggregate['dead_air_fail_count']}`",
        f"- grammar primary cause excluded count: `{aggregate['grammar_primary_cause_excluded_count']}`",
        f"- collapse primary cause excluded count: `{aggregate['collapse_primary_cause_excluded_count']}`",
        f"- selected repair target: `{decision['selected_repair_target']}`",
        f"- next boundary: `{decision['next_boundary']}`",
        f"- musical quality claimed: `{_bool_token(readiness['musical_quality_claimed'])}`",
        "",
        "## Case Reviews",
        "",
    ]
    for case in report.get("case_reviews", []):
        lines.extend(
            [
                f"### {case['label']}",
                "",
                f"- chords: `{case['chords']}`",
                f"- strict: `{case['strict_valid_sample_count']}` / `{case['sample_count']}`",
                f"- dominant failure: `{case['dominant_failure']}`",
                f"- repair target: `{case['repair_target']}`",
                f"- invalid avg dead-air: `{float(case['invalid_avg_dead_air_ratio']):.4f}`",
                f"- strict avg dead-air: `{float(case['strict_avg_dead_air_ratio']):.4f}`",
                f"- invalid avg postprocess removal: `{float(case['invalid_avg_postprocess_removal_ratio']):.4f}`",
                f"- strict avg postprocess removal: `{float(case['strict_avg_postprocess_removal_ratio']):.4f}`",
                "",
                "| sample | strict | failure | notes | dead air | removed | removal ratio | duration common | MIDI |",
                "|---:|---|---|---:|---:|---:|---:|---:|---|",
            ]
        )
        for sample in case.get("samples", []):
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(sample["sample_index"]),
                        f"`{_bool_token(sample['strict_valid'])}`",
                        f"`{sample.get('failure_reason') or 'none'}`",
                        str(sample["note_count"]),
                        f"{float(sample['dead_air_ratio']):.4f}",
                        str(sample["postprocess_removed_note_count"]),
                        f"{float(sample['postprocess_removal_ratio']):.4f}",
                        f"{float(sample['most_common_duration_ratio']):.4f}",
                        f"`{sample['midi_path']}`",
                    ]
                )
                + " |"
            )
        lines.append("")
    lines.extend(["## Not Proven", ""])
    for item in report.get("not_proven", []):
        lines.append(f"- `{item}`")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Review solo-yield sweep failure cases")
    parser.add_argument("--sweep_report", type=str, required=True)
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/music_transformer_finetune_mvp/solo_yield_failure_review",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--min_failing_cases", type=int, default=1)
    parser.add_argument("--require_no_quality_claim", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    report = build_failure_review(read_json(Path(args.sweep_report)), output_dir=output_dir)
    summary = validate_review(
        report,
        min_failing_cases=int(args.min_failing_cases),
        require_no_quality_claim=bool(args.require_no_quality_claim),
    )
    write_json(output_dir / "solo_yield_failure_review.json", report)
    write_json(output_dir / "solo_yield_failure_review_summary.json", summary)
    markdown = markdown_report(report)
    write_text(output_dir / "solo_yield_failure_review.md", markdown)
    if args.doc_path:
        write_text(Path(args.doc_path), markdown)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
