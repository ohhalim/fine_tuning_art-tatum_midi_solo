"""Summarize policy-level repeatability for outside-soloing repair variants."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class StageBDurationCoverageOutsideSoloingRepairBroaderRepeatabilitySweepError(ValueError):
    pass


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def validate_inputs(next_decision: dict[str, Any], repair_sweep: dict[str, Any]) -> None:
    decision = _dict(next_decision.get("decision"))
    decision_claim = _dict(next_decision.get("claim_boundary"))
    repair_summary = _dict(repair_sweep.get("repair_summary"))
    repair_claim = _dict(repair_sweep.get("claim_boundary"))
    if str(decision.get("next_boundary") or "") != "outside_soloing_repair_broader_repeatability_sweep":
        raise StageBDurationCoverageOutsideSoloingRepairBroaderRepeatabilitySweepError(
            "broader repeatability next boundary required"
        )
    if not bool(decision.get("auto_progress_allowed", False)):
        raise StageBDurationCoverageOutsideSoloingRepairBroaderRepeatabilitySweepError("auto progress must be allowed")
    if bool(decision.get("critical_user_input_required", True)):
        raise StageBDurationCoverageOutsideSoloingRepairBroaderRepeatabilitySweepError(
            "critical user input must not be required"
        )
    if bool(decision_claim.get("broad_model_quality_claimed", True)):
        raise StageBDurationCoverageOutsideSoloingRepairBroaderRepeatabilitySweepError(
            "broad model quality must not be claimed"
        )
    if str(repair_summary.get("boundary") or "") != "outside_soloing_pitch_role_repair_candidates":
        raise StageBDurationCoverageOutsideSoloingRepairBroaderRepeatabilitySweepError(
            "outside-soloing repair sweep boundary required"
        )
    if bool(repair_claim.get("human_audio_preference_claimed", True)):
        raise StageBDurationCoverageOutsideSoloingRepairBroaderRepeatabilitySweepError(
            "human/audio preference must not be claimed"
        )
    if bool(repair_claim.get("broad_model_quality_claimed", True)):
        raise StageBDurationCoverageOutsideSoloingRepairBroaderRepeatabilitySweepError(
            "broad model quality must not be claimed"
        )


def variant_rows(repair_sweep: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for source in _list(repair_sweep.get("source_repair_results")):
        if not isinstance(source, dict):
            continue
        source_dead_air = float(source.get("source_selected_dead_air_ratio", 1.0) or 1.0)
        for variant in _list(source.get("variants")):
            if not isinstance(variant, dict):
                continue
            metrics = _dict(variant.get("metrics"))
            focused = _dict(variant.get("focused_solo_metrics"))
            pitch_role = _dict(variant.get("pitch_role_metrics"))
            gate = _dict(variant.get("outside_soloing_gate"))
            rows.append(
                {
                    "source_candidate_id": str(source.get("source_candidate_id") or ""),
                    "sample_seed": int(source.get("sample_seed", 0) or 0),
                    "repair_policy": str(variant.get("repair_policy") or ""),
                    "candidate_id": str(variant.get("candidate_id") or ""),
                    "qualified": bool(gate.get("qualified", False)),
                    "flags": list(gate.get("flags") or []),
                    "source_dead_air_ratio": source_dead_air,
                    "dead_air_ratio": float(metrics.get("dead_air_ratio", 1.0) or 1.0),
                    "chord_tone_ratio": float(metrics.get("chord_tone_ratio", 0.0) or 0.0),
                    "focused_max_interval": int(focused.get("focused_max_interval", 0) or 0),
                    "focused_unique_pitch_count": int(focused.get("focused_unique_pitch_count", 0) or 0),
                    "max_non_chord_tone_run": int(pitch_role.get("max_non_chord_tone_run", 0) or 0),
                }
            )
    return rows


def policy_summary(
    rows: list[dict[str, Any]],
    *,
    min_source_candidates_per_policy: int,
    min_chord_tone_ratio: float,
    max_non_chord_run: int,
    max_interval: int,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["repair_policy"])].append(row)
    summaries = []
    for policy, policy_rows in sorted(grouped.items()):
        source_seeds = sorted({int(row["sample_seed"]) for row in policy_rows})
        qualified_count = sum(1 for row in policy_rows if row["qualified"])
        dead_air_preserved_count = sum(
            1 for row in policy_rows if row["dead_air_ratio"] <= row["source_dead_air_ratio"] + 1e-9
        )
        chord_tone_pass_count = sum(1 for row in policy_rows if row["chord_tone_ratio"] >= float(min_chord_tone_ratio))
        non_chord_run_pass_count = sum(1 for row in policy_rows if row["max_non_chord_tone_run"] <= int(max_non_chord_run))
        interval_pass_count = sum(1 for row in policy_rows if row["focused_max_interval"] <= int(max_interval))
        repeated = (
            len(source_seeds) >= int(min_source_candidates_per_policy)
            and qualified_count >= int(min_source_candidates_per_policy)
            and dead_air_preserved_count >= int(min_source_candidates_per_policy)
            and chord_tone_pass_count >= int(min_source_candidates_per_policy)
            and non_chord_run_pass_count >= int(min_source_candidates_per_policy)
            and interval_pass_count >= int(min_source_candidates_per_policy)
        )
        summaries.append(
            {
                "repair_policy": policy,
                "source_candidate_count": int(len(source_seeds)),
                "sample_seeds": source_seeds,
                "variant_count": int(len(policy_rows)),
                "qualified_variant_count": int(qualified_count),
                "dead_air_preserved_variant_count": int(dead_air_preserved_count),
                "chord_tone_pass_variant_count": int(chord_tone_pass_count),
                "non_chord_run_pass_variant_count": int(non_chord_run_pass_count),
                "interval_pass_variant_count": int(interval_pass_count),
                "min_chord_tone_ratio": min(row["chord_tone_ratio"] for row in policy_rows),
                "max_non_chord_tone_run": max(row["max_non_chord_tone_run"] for row in policy_rows),
                "max_interval": max(row["focused_max_interval"] for row in policy_rows),
                "policy_repeatability_supported": bool(repeated),
            }
        )
    return summaries


def build_broader_repeatability_sweep_report(
    *,
    next_decision: dict[str, Any],
    repair_sweep: dict[str, Any],
    output_dir: Path,
    min_source_candidates: int,
    min_policy_repeatability_count: int,
    min_source_candidates_per_policy: int,
    min_chord_tone_ratio: float,
    max_non_chord_run: int,
    max_interval: int,
) -> dict[str, Any]:
    validate_inputs(next_decision, repair_sweep)
    rows = variant_rows(repair_sweep)
    if not rows:
        raise StageBDurationCoverageOutsideSoloingRepairBroaderRepeatabilitySweepError("no repair variants found")
    source_count = len({int(row["sample_seed"]) for row in rows})
    if source_count < int(min_source_candidates):
        raise StageBDurationCoverageOutsideSoloingRepairBroaderRepeatabilitySweepError(
            f"expected {int(min_source_candidates)} source candidates, got {source_count}"
        )
    policies = policy_summary(
        rows,
        min_source_candidates_per_policy=int(min_source_candidates_per_policy),
        min_chord_tone_ratio=float(min_chord_tone_ratio),
        max_non_chord_run=int(max_non_chord_run),
        max_interval=int(max_interval),
    )
    supported_policy_count = sum(1 for row in policies if row["policy_repeatability_supported"])
    qualified_variant_count = sum(1 for row in rows if row["qualified"])
    total_variant_count = len(rows)
    boundary = (
        "outside_soloing_repair_policy_repeatability_support"
        if supported_policy_count >= int(min_policy_repeatability_count)
        else "outside_soloing_repair_policy_repeatability_incomplete"
    )
    return {
        "schema_version": "stage_b_duration_coverage_fill_outside_soloing_repair_broader_repeatability_sweep_v1",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "source_schemas": {
            "next_decision": str(next_decision.get("schema_version") or ""),
            "repair_sweep": str(repair_sweep.get("schema_version") or ""),
        },
        "previous_boundary": str(_dict(next_decision.get("decision")).get("next_boundary") or ""),
        "thresholds": {
            "min_source_candidates": int(min_source_candidates),
            "min_policy_repeatability_count": int(min_policy_repeatability_count),
            "min_source_candidates_per_policy": int(min_source_candidates_per_policy),
            "min_chord_tone_ratio": float(min_chord_tone_ratio),
            "max_non_chord_run": int(max_non_chord_run),
            "max_interval": int(max_interval),
        },
        "policy_summaries": policies,
        "variant_rows": rows,
        "repeatability_summary": {
            "boundary": boundary,
            "source_candidate_count": int(source_count),
            "repair_policy_count": int(len(policies)),
            "supported_repair_policy_count": int(supported_policy_count),
            "total_variant_count": int(total_variant_count),
            "total_qualified_variant_count": int(qualified_variant_count),
            "selected_min_chord_tone_ratio": min(row["chord_tone_ratio"] for row in rows),
            "selected_max_non_chord_tone_run": max(row["max_non_chord_tone_run"] for row in rows),
            "selected_max_interval": max(row["focused_max_interval"] for row in rows),
            "human_audio_preference_claimed": False,
            "broad_model_quality_claimed": False,
        },
        "claim_boundary": {
            "boundary": boundary,
            "policy_repeatability_claimed": supported_policy_count >= int(min_policy_repeatability_count),
            "human_audio_preference_claimed": False,
            "multi_reviewer_preference_claimed": False,
            "broad_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
            "production_ready_improviser_claimed": False,
        },
        "not_proven": [
            "human_audio_preference",
            "multi_reviewer_preference",
            "broad_trained_model_quality",
            "brad_style_adaptation",
            "production_ready_improviser",
        ],
        "next_recommended_issue": (
            "Stage B margin-recovered phrase/vocabulary duration coverage fill outside-soloing repair "
            "repeatability consolidation"
        ),
    }


def validate_broader_repeatability_sweep(
    report: dict[str, Any],
    *,
    expected_boundary: str | None,
    min_source_candidates: int,
    min_policy_repeatability_count: int,
    require_no_preference_claim: bool,
    require_no_broad_quality_claim: bool,
) -> dict[str, Any]:
    summary = _dict(report.get("repeatability_summary"))
    claim = _dict(report.get("claim_boundary"))
    boundary = str(summary.get("boundary") or "")
    if expected_boundary and boundary != expected_boundary:
        raise StageBDurationCoverageOutsideSoloingRepairBroaderRepeatabilitySweepError(
            f"expected boundary {expected_boundary}, got {boundary}"
        )
    if int(summary.get("source_candidate_count", 0) or 0) < int(min_source_candidates):
        raise StageBDurationCoverageOutsideSoloingRepairBroaderRepeatabilitySweepError("not enough source candidates")
    if int(summary.get("supported_repair_policy_count", 0) or 0) < int(min_policy_repeatability_count):
        raise StageBDurationCoverageOutsideSoloingRepairBroaderRepeatabilitySweepError("not enough supported policies")
    if require_no_preference_claim and bool(claim.get("human_audio_preference_claimed", True)):
        raise StageBDurationCoverageOutsideSoloingRepairBroaderRepeatabilitySweepError(
            "human/audio preference must not be claimed"
        )
    if require_no_broad_quality_claim and bool(claim.get("broad_model_quality_claimed", True)):
        raise StageBDurationCoverageOutsideSoloingRepairBroaderRepeatabilitySweepError(
            "broad model quality must not be claimed"
        )
    return {
        "boundary": boundary,
        "source_candidate_count": int(summary.get("source_candidate_count", 0) or 0),
        "repair_policy_count": int(summary.get("repair_policy_count", 0) or 0),
        "supported_repair_policy_count": int(summary.get("supported_repair_policy_count", 0) or 0),
        "total_variant_count": int(summary.get("total_variant_count", 0) or 0),
        "total_qualified_variant_count": int(summary.get("total_qualified_variant_count", 0) or 0),
        "selected_min_chord_tone_ratio": float(summary.get("selected_min_chord_tone_ratio", 0.0) or 0.0),
        "selected_max_non_chord_tone_run": int(summary.get("selected_max_non_chord_tone_run", 0) or 0),
        "selected_max_interval": int(summary.get("selected_max_interval", 0) or 0),
        "human_audio_preference_claimed": bool(summary.get("human_audio_preference_claimed", True)),
        "broad_model_quality_claimed": bool(summary.get("broad_model_quality_claimed", True)),
        "next_recommended_issue": str(report.get("next_recommended_issue") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    summary = report["repeatability_summary"]
    lines = [
        "# Stage B Duration Coverage Fill Outside-Soloing Repair Broader Repeatability Sweep",
        "",
        f"- previous boundary: `{report['previous_boundary']}`",
        f"- boundary: `{summary['boundary']}`",
        f"- source candidates: `{summary['source_candidate_count']}`",
        f"- repair policies: `{summary['repair_policy_count']}`",
        f"- supported repair policies: `{summary['supported_repair_policy_count']}`",
        f"- total variants: `{summary['total_variant_count']}`",
        f"- qualified variants: `{summary['total_qualified_variant_count']}`",
        f"- selected min chord-tone ratio: `{summary['selected_min_chord_tone_ratio']:.3f}`",
        f"- selected max non-chord run: `{summary['selected_max_non_chord_tone_run']}`",
        f"- selected max interval: `{summary['selected_max_interval']}`",
        f"- human/audio preference claimed: `{summary['human_audio_preference_claimed']}`",
        f"- broad model quality claimed: `{summary['broad_model_quality_claimed']}`",
        "",
        "| policy | sources | qualified | chord-tone min | non-chord max | interval max | supported |",
        "|---|---:|---:|---:|---:|---:|:---:|",
    ]
    for policy in report["policy_summaries"]:
        lines.append(
            "| `{policy}` | {sources} | {qualified}/{variants} | {chord:.3f} | {non_chord} | "
            "{interval} | {supported} |".format(
                policy=policy["repair_policy"],
                sources=int(policy["source_candidate_count"]),
                qualified=int(policy["qualified_variant_count"]),
                variants=int(policy["variant_count"]),
                chord=float(policy["min_chord_tone_ratio"]),
                non_chord=int(policy["max_non_chord_tone_run"]),
                interval=int(policy["max_interval"]),
                supported=bool(policy["policy_repeatability_supported"]),
            )
        )
    lines.extend(["", "## Not Proven", ""])
    for item in report.get("not_proven", []):
        lines.append(f"- `{item}`")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize outside-soloing repair broader repeatability")
    parser.add_argument(
        "--next_decision",
        type=str,
        default="outputs/stage_b_duration_coverage_fill_outside_soloing_repair_next_decision/"
        "harness_stage_b_duration_coverage_fill_outside_soloing_repair_next_decision/"
        "stage_b_duration_coverage_fill_outside_soloing_repair_next_decision.json",
    )
    parser.add_argument(
        "--repair_sweep",
        type=str,
        default="outputs/stage_b_duration_coverage_fill_outside_soloing_repair_sweep/"
        "harness_stage_b_duration_coverage_fill_outside_soloing_repair_sweep/"
        "stage_b_duration_coverage_fill_outside_soloing_repair_sweep.json",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/stage_b_duration_coverage_fill_outside_soloing_repair_broader_repeatability_sweep",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--min_source_candidates", type=int, default=2)
    parser.add_argument("--min_policy_repeatability_count", type=int, default=3)
    parser.add_argument("--min_source_candidates_per_policy", type=int, default=2)
    parser.add_argument("--min_chord_tone_ratio", type=float, default=0.72)
    parser.add_argument("--max_non_chord_run", type=int, default=1)
    parser.add_argument("--max_interval", type=int, default=7)
    parser.add_argument("--expected_boundary", type=str, default="")
    parser.add_argument("--require_no_preference_claim", action="store_true")
    parser.add_argument("--require_no_broad_quality_claim", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    report = build_broader_repeatability_sweep_report(
        next_decision=read_json(Path(args.next_decision)),
        repair_sweep=read_json(Path(args.repair_sweep)),
        output_dir=output_dir,
        min_source_candidates=int(args.min_source_candidates),
        min_policy_repeatability_count=int(args.min_policy_repeatability_count),
        min_source_candidates_per_policy=int(args.min_source_candidates_per_policy),
        min_chord_tone_ratio=float(args.min_chord_tone_ratio),
        max_non_chord_run=int(args.max_non_chord_run),
        max_interval=int(args.max_interval),
    )
    summary = validate_broader_repeatability_sweep(
        report,
        expected_boundary=str(args.expected_boundary or ""),
        min_source_candidates=int(args.min_source_candidates),
        min_policy_repeatability_count=int(args.min_policy_repeatability_count),
        require_no_preference_claim=bool(args.require_no_preference_claim),
        require_no_broad_quality_claim=bool(args.require_no_broad_quality_claim),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "stage_b_duration_coverage_fill_outside_soloing_repair_broader_repeatability_sweep.json"
    markdown_path = output_dir / "stage_b_duration_coverage_fill_outside_soloing_repair_broader_repeatability_sweep.md"
    write_json(report_path, report)
    write_json(
        output_dir / "stage_b_duration_coverage_fill_outside_soloing_repair_broader_repeatability_sweep_validation_summary.json",
        summary,
    )
    markdown_path.write_text(markdown_report(report), encoding="utf-8")
    print(json.dumps({**summary, "report_path": str(report_path), "markdown_path": str(markdown_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
