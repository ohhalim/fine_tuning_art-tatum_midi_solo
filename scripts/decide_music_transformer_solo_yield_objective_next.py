"""Decide next Music Transformer solo-yield task from objective evidence."""

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


SCHEMA_VERSION = "music_transformer_solo_yield_objective_next_decision_v1"


class SoloYieldObjectiveDecisionError(ValueError):
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
        raise SoloYieldObjectiveDecisionError(f"json not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def rank_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        candidates,
        key=lambda row: (
            _float(row.get("score")),
            _int(row.get("note_count")),
            -_float(row.get("dead_air_ratio")),
            _float(row.get("direction_change_ratio")),
            _float(row.get("syncopated_onset_ratio")),
        ),
        reverse=True,
    )


def objective_summary(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [_float(row.get("score")) for row in candidates]
    note_counts = [_int(row.get("note_count")) for row in candidates]
    dead_air = [_float(row.get("dead_air_ratio")) for row in candidates]
    return {
        "candidate_count": len(candidates),
        "score_min": min(scores, default=0.0),
        "score_max": max(scores, default=0.0),
        "score_avg": float(mean(scores)) if scores else 0.0,
        "note_count_min": min(note_counts, default=0),
        "note_count_max": max(note_counts, default=0),
        "note_count_avg": float(mean(note_counts)) if note_counts else 0.0,
        "dead_air_min": min(dead_air, default=0.0),
        "dead_air_max": max(dead_air, default=0.0),
        "dead_air_avg": float(mean(dead_air)) if dead_air else 0.0,
    }


def build_decision(
    package_report: dict[str, Any],
    guard_report: dict[str, Any],
    *,
    output_dir: Path,
    top_n: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    guard_readiness = _dict(guard_report.get("readiness"))
    if bool(guard_readiness.get("preference_fill_allowed", True)):
        next_boundary = "music_transformer_solo_yield_listening_review_fill"
        reason = "validated listening input present"
    else:
        next_boundary = "music_transformer_solo_yield_larger_sample_repeatability_sweep"
        reason = "listening input pending; objective yield is clean, so increase sample count before quality claim"
    candidates = [_dict(row) for row in _list(package_report.get("candidates"))]
    ranked = rank_candidates(candidates)
    selected = ranked[: int(top_n)]
    report = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "source_package": {
            "schema_version": package_report.get("schema_version"),
            "output_dir": package_report.get("output_dir"),
            "candidate_count": _int(package_report.get("candidate_count")),
        },
        "source_guard": {
            "schema_version": guard_report.get("schema_version"),
            "output_dir": guard_report.get("output_dir"),
            "validated_listening_input_present": bool(
                _dict(guard_report.get("input_validation")).get("validated_listening_input_present", False)
            ),
            "preference_fill_allowed": bool(guard_readiness.get("preference_fill_allowed", False)),
        },
        "objective_summary": objective_summary(candidates),
        "selected_objective_candidates": selected,
        "readiness": {
            "objective_only_next_decision_completed": True,
            "validated_listening_input_present": bool(
                _dict(guard_report.get("input_validation")).get("validated_listening_input_present", False)
            ),
            "preference_fill_allowed": bool(guard_readiness.get("preference_fill_allowed", False)),
            "selected_objective_candidate_count": len(selected),
            "musical_quality_claimed": False,
            "artist_style_claimed": False,
            "production_ready_claimed": False,
        },
        "decision": {
            "current_boundary": "music_transformer_solo_yield_objective_only_next_decision",
            "next_boundary": next_boundary,
            "critical_user_input_required": False,
            "reason": reason,
        },
        "not_proven": [
            "human_audio_preference",
            "stable_jazz_solo_quality",
            "artist_level_long_solo_generation",
            "production_ready_improviser",
        ],
    }
    write_json(output_dir / "objective_next_decision.json", report)
    write_json(output_dir / "objective_next_decision_summary.json", validate_report(report, require_no_quality_claim=True))
    write_text(output_dir / "objective_next_decision.md", markdown_report(report))
    return report


def validate_report(report: dict[str, Any], *, require_no_quality_claim: bool) -> dict[str, Any]:
    readiness = _dict(report.get("readiness"))
    if require_no_quality_claim:
        claimed = [
            key
            for key in ("musical_quality_claimed", "artist_style_claimed", "production_ready_claimed")
            if bool(readiness.get(key, True))
        ]
        if claimed:
            raise SoloYieldObjectiveDecisionError(f"unexpected quality claim: {claimed}")
    return {
        "schema_version": str(report.get("schema_version")),
        "candidate_count": _int(_dict(report.get("objective_summary")).get("candidate_count")),
        "selected_objective_candidate_count": _int(
            readiness.get("selected_objective_candidate_count")
        ),
        "validated_listening_input_present": bool(readiness.get("validated_listening_input_present", True)),
        "preference_fill_allowed": bool(readiness.get("preference_fill_allowed", True)),
        "musical_quality_claimed": bool(readiness.get("musical_quality_claimed", True)),
        "next_boundary": str(_dict(report.get("decision")).get("next_boundary") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    summary = report["objective_summary"]
    readiness = report["readiness"]
    decision = report["decision"]
    lines = [
        "# Music Transformer Solo Yield Objective-Only Next Decision",
        "",
        "## Summary",
        "",
        f"- candidate count: `{summary['candidate_count']}`",
        f"- selected objective candidates: `{readiness['selected_objective_candidate_count']}`",
        f"- score range: `{float(summary['score_min']):.3f}` - `{float(summary['score_max']):.3f}`",
        f"- note count range: `{summary['note_count_min']}` - `{summary['note_count_max']}`",
        f"- dead-air range: `{float(summary['dead_air_min']):.4f}` - `{float(summary['dead_air_max']):.4f}`",
        f"- validated listening input present: `{_bool_token(readiness['validated_listening_input_present'])}`",
        f"- preference fill allowed: `{_bool_token(readiness['preference_fill_allowed'])}`",
        f"- musical quality claimed: `{_bool_token(readiness['musical_quality_claimed'])}`",
        f"- next boundary: `{decision['next_boundary']}`",
        "",
        "## Selected Objective Candidates",
        "",
        "| review | case | score | notes | dead air | WAV |",
        "|---:|---|---:|---:|---:|---|",
    ]
    for row in report.get("selected_objective_candidates", []):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["review_index"]),
                    f"`{row['case_label']}`",
                    f"{float(row['score']):.3f}",
                    str(row["note_count"]),
                    f"{float(row['dead_air_ratio']):.4f}",
                    f"`{row['review_wav_path']}`",
                ]
            )
            + " |"
        )
    lines.extend(["", "## Not Proven", ""])
    for item in report.get("not_proven", []):
        lines.append(f"- `{item}`")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Decide objective-only next solo-yield task")
    parser.add_argument("--package_report", type=str, required=True)
    parser.add_argument("--guard_report", type=str, required=True)
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/music_transformer_finetune_mvp/solo_yield_objective_next_decision",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--top_n", type=int, default=4)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--require_no_quality_claim", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    report = build_decision(
        read_json(Path(args.package_report)),
        read_json(Path(args.guard_report)),
        output_dir=output_dir,
        top_n=int(args.top_n),
    )
    summary = validate_report(report, require_no_quality_claim=bool(args.require_no_quality_claim))
    if args.doc_path:
        write_text(Path(args.doc_path), markdown_report(report))
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
