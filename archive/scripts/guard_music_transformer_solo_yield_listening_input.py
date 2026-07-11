"""Guard listening-review input before Music Transformer solo quality decisions."""

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


SCHEMA_VERSION = "music_transformer_solo_yield_listening_input_guard_v1"


class SoloYieldListeningInputGuardError(ValueError):
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


def _bool_token(value: Any) -> str:
    return "true" if bool(value) else "false"


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SoloYieldListeningInputGuardError(f"json not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def pending_candidate_fields(candidate: dict[str, Any]) -> list[str]:
    pending: list[str] = []
    if str(candidate.get("decision") or "pending") == "pending":
        pending.append("decision")
    if candidate.get("usable_as_jazz_solo_phrase") is None:
        pending.append("usable_as_jazz_solo_phrase")
    if candidate.get("primary_failure") is None:
        pending.append("primary_failure")
    return pending


def validate_input(input_report: dict[str, Any], candidate_count: int) -> dict[str, Any]:
    candidates = [_dict(row) for row in _list(input_report.get("candidates"))]
    pending_status = str(input_report.get("review_status") or "pending") != "reviewed"
    pending_overall = str(input_report.get("overall_decision") or "pending") == "pending"
    candidate_pending = {
        str(_int(row.get("review_index"))): pending_candidate_fields(row)
        for row in candidates
        if pending_candidate_fields(row)
    }
    candidate_count_matched = len(candidates) == int(candidate_count)
    validated = bool(
        candidate_count_matched
        and not pending_status
        and not pending_overall
        and not candidate_pending
    )
    return {
        "validated_listening_input_present": validated,
        "candidate_count_matched": candidate_count_matched,
        "input_candidate_count": len(candidates),
        "expected_candidate_count": int(candidate_count),
        "pending_status": pending_status,
        "pending_overall_decision": pending_overall,
        "pending_candidate_field_count": sum(len(fields) for fields in candidate_pending.values()),
        "pending_candidate_fields": candidate_pending,
    }


def build_guard_report(
    package_report: dict[str, Any],
    *,
    output_dir: Path,
    listening_input_path: Path | None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    default_input_path = Path(str(package_report.get("output_dir") or "")) / "listening_review_input_template.json"
    input_path = listening_input_path or default_input_path
    input_report = read_json(input_path)
    candidate_count = _int(package_report.get("candidate_count"))
    input_validation = validate_input(input_report, candidate_count)
    preference_fill_allowed = bool(input_validation["validated_listening_input_present"])
    report = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "source_package": {
            "schema_version": package_report.get("schema_version"),
            "output_dir": package_report.get("output_dir"),
            "candidate_count": candidate_count,
            "listening_review_package_ready": bool(
                _dict(package_report.get("readiness")).get("listening_review_package_ready", False)
            ),
        },
        "listening_input_path": str(input_path),
        "input_validation": input_validation,
        "readiness": {
            "listening_input_guard_completed": True,
            "validated_listening_input_present": bool(input_validation["validated_listening_input_present"]),
            "preference_fill_allowed": preference_fill_allowed,
            "objective_only_next_decision_required": not preference_fill_allowed,
            "musical_quality_claimed": False,
            "artist_style_claimed": False,
            "production_ready_claimed": False,
        },
        "decision": {
            "current_boundary": "music_transformer_solo_yield_listening_input_guard",
            "next_boundary": (
                "music_transformer_solo_yield_listening_review_fill"
                if preference_fill_allowed
                else "music_transformer_solo_yield_objective_only_next_decision"
            ),
            "critical_user_input_required": False,
            "reason": (
                "validated listening input present"
                if preference_fill_allowed
                else "listening input pending; route to objective-only next decision without quality claim"
            ),
        },
        "not_proven": [
            "human_audio_preference",
            "stable_jazz_solo_quality",
            "artist_level_long_solo_generation",
            "production_ready_improviser",
        ],
    }
    write_json(output_dir / "listening_input_guard.json", report)
    write_json(output_dir / "listening_input_guard_summary.json", validate_report(report, require_no_quality_claim=True))
    write_text(output_dir / "listening_input_guard.md", markdown_report(report))
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
            raise SoloYieldListeningInputGuardError(f"unexpected quality claim: {claimed}")
    validation = _dict(report.get("input_validation"))
    return {
        "schema_version": str(report.get("schema_version")),
        "validated_listening_input_present": bool(validation.get("validated_listening_input_present", False)),
        "preference_fill_allowed": bool(readiness.get("preference_fill_allowed", True)),
        "objective_only_next_decision_required": bool(
            readiness.get("objective_only_next_decision_required", False)
        ),
        "pending_status": bool(validation.get("pending_status", False)),
        "pending_overall_decision": bool(validation.get("pending_overall_decision", False)),
        "pending_candidate_field_count": _int(validation.get("pending_candidate_field_count")),
        "musical_quality_claimed": bool(readiness.get("musical_quality_claimed", True)),
        "next_boundary": str(_dict(report.get("decision")).get("next_boundary") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    validation = report["input_validation"]
    readiness = report["readiness"]
    decision = report["decision"]
    lines = [
        "# Music Transformer Solo Yield Listening Input Guard",
        "",
        "## Summary",
        "",
        f"- source candidate count: `{report['source_package']['candidate_count']}`",
        f"- listening input path: `{report['listening_input_path']}`",
        f"- validated listening input present: `{_bool_token(validation['validated_listening_input_present'])}`",
        f"- preference fill allowed: `{_bool_token(readiness['preference_fill_allowed'])}`",
        f"- objective-only next decision required: `{_bool_token(readiness['objective_only_next_decision_required'])}`",
        f"- pending status: `{_bool_token(validation['pending_status'])}`",
        f"- pending overall decision: `{_bool_token(validation['pending_overall_decision'])}`",
        f"- pending candidate field count: `{validation['pending_candidate_field_count']}`",
        f"- musical quality claimed: `{_bool_token(readiness['musical_quality_claimed'])}`",
        f"- next boundary: `{decision['next_boundary']}`",
        "",
        "## Pending Candidate Fields",
        "",
    ]
    pending = _dict(validation.get("pending_candidate_fields"))
    if pending:
        for review_index, fields in pending.items():
            lines.append(f"- review `{review_index}`: `{','.join(str(field) for field in fields)}`")
    else:
        lines.append("- `none`")
    lines.extend(["", "## Not Proven", ""])
    for item in report.get("not_proven", []):
        lines.append(f"- `{item}`")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Guard solo-yield listening input")
    parser.add_argument("--package_report", type=str, required=True)
    parser.add_argument("--listening_input", type=str, default="")
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/music_transformer_finetune_mvp/solo_yield_listening_input_guard",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--require_no_quality_claim", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    listening_input_path = Path(args.listening_input) if args.listening_input else None
    report = build_guard_report(
        read_json(Path(args.package_report)),
        output_dir=output_dir,
        listening_input_path=listening_input_path,
    )
    summary = validate_report(report, require_no_quality_claim=bool(args.require_no_quality_claim))
    if args.doc_path:
        write_text(Path(args.doc_path), markdown_report(report))
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
