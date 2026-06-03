"""Guard model-direct user listening review fill against missing review input."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from scripts.assess_stage_b_generic_base_readiness import write_json, write_text  # noqa: E402
from scripts.build_stage_b_midi_to_solo_model_direct_listening_review_package import (  # noqa: E402
    BOUNDARY as SOURCE_BOUNDARY,
)


class StageBMidiToSoloModelDirectUserListeningReviewInputGuardError(ValueError):
    pass


BOUNDARY = "stage_b_midi_to_solo_model_direct_user_listening_review_input_guard"
FILL_BOUNDARY = "stage_b_midi_to_solo_model_direct_user_listening_review_fill"
OBJECTIVE_NEXT_BOUNDARY = "stage_b_midi_to_solo_model_direct_objective_only_next_decision"
SCHEMA_VERSION = "stage_b_midi_to_solo_model_direct_user_listening_review_input_guard_v1"


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise StageBMidiToSoloModelDirectUserListeningReviewInputGuardError(f"report missing: {path}")
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


def _clean_cell(value: str) -> str:
    return value.strip().strip("`").strip()


def parse_review_input(markdown: str) -> dict[str, Any]:
    status: dict[str, str] = {}
    per_candidate_fields: dict[str, dict[str, str]] = {}
    current_rank: str | None = None
    candidate_table_decisions: dict[str, str] = {}
    for raw_line in markdown.splitlines():
        line = raw_line.strip()
        status_match = re.match(r"^- ([a-zA-Z0-9_]+): `?([^`]+)`?$", line)
        if status_match and current_rank is None:
            status[status_match.group(1)] = _clean_cell(status_match.group(2))
            continue
        rank_match = re.match(r"^### Rank ([0-9]+)$", line)
        if rank_match:
            current_rank = rank_match.group(1)
            per_candidate_fields.setdefault(current_rank, {})
            continue
        if status_match and current_rank is not None:
            per_candidate_fields.setdefault(current_rank, {})[status_match.group(1)] = _clean_cell(
                status_match.group(2)
            )
            continue
        if line.startswith("|") and not line.startswith("|---") and "decision" not in line:
            cells = [_clean_cell(cell) for cell in line.strip("|").split("|")]
            if cells and cells[0].isdigit() and len(cells) >= 8:
                candidate_table_decisions[cells[0]] = cells[7]
    required_status_fields = ["reviewer", "reviewed_at", "preferred_rank", "reject_all"]
    pending_status_fields = [
        field for field in required_status_fields if status.get(field, "pending").lower() == "pending"
    ]
    pending_candidate_decisions = [
        rank for rank, decision in sorted(candidate_table_decisions.items()) if decision.lower() == "pending"
    ]
    pending_candidate_fields: list[str] = []
    for rank, fields in sorted(per_candidate_fields.items()):
        for field in ("musical_acceptance", "issue_tags", "short_note"):
            if fields.get(field, "pending").lower() == "pending":
                pending_candidate_fields.append(f"rank_{rank}.{field}")
    validated = not pending_status_fields and not pending_candidate_decisions and not pending_candidate_fields
    return {
        "status": status,
        "candidate_table_decisions": candidate_table_decisions,
        "per_candidate_fields": per_candidate_fields,
        "pending_status_fields": pending_status_fields,
        "pending_candidate_decisions": pending_candidate_decisions,
        "pending_candidate_fields": pending_candidate_fields,
        "validated_review_input_present": bool(validated),
    }


def validate_source_package(report: dict[str, Any]) -> dict[str, Any]:
    boundary = _dict(report.get("listening_review_package_boundary"))
    decision = _dict(report.get("decision"))
    if str(boundary.get("boundary") or "") != SOURCE_BOUNDARY:
        raise StageBMidiToSoloModelDirectUserListeningReviewInputGuardError("listening review package boundary required")
    if str(decision.get("next_boundary") or "") != FILL_BOUNDARY:
        raise StageBMidiToSoloModelDirectUserListeningReviewInputGuardError("review package must route to fill boundary")
    if not bool(boundary.get("review_input_template_written", False)):
        raise StageBMidiToSoloModelDirectUserListeningReviewInputGuardError("review input template required")
    if bool(boundary.get("human_audio_preference_claimed", True)):
        raise StageBMidiToSoloModelDirectUserListeningReviewInputGuardError("preference must not be claimed upstream")
    review_input_path = Path(str(report.get("review_input_template_path") or ""))
    if not review_input_path.exists():
        raise StageBMidiToSoloModelDirectUserListeningReviewInputGuardError("review input template missing")
    return {
        "boundary": str(boundary.get("boundary") or ""),
        "candidate_count": _int(boundary.get("candidate_count")),
        "rendered_audio_file_count": _int(boundary.get("rendered_audio_file_count")),
        "review_input_template_path": str(review_input_path),
    }


def build_user_listening_review_input_guard_report(
    source_package: dict[str, Any],
    *,
    output_dir: Path,
) -> dict[str, Any]:
    source = validate_source_package(source_package)
    parsed = parse_review_input(Path(source["review_input_template_path"]).read_text(encoding="utf-8"))
    validated_input = bool(parsed["validated_review_input_present"])
    next_boundary = FILL_BOUNDARY if validated_input else OBJECTIVE_NEXT_BOUNDARY
    reason = (
        "validated listening review input present; route to user listening review fill"
        if validated_input
        else "listening review input pending; preference fill blocked"
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "boundary": BOUNDARY,
        "source_boundary": source["boundary"],
        "source_package_summary": source,
        "review_input_summary": parsed,
        "guard_result": {
            "review_input_template_present": True,
            "validated_review_input_present": bool(validated_input),
            "pending_status_field_count": len(_list(parsed.get("pending_status_fields"))),
            "pending_candidate_decision_count": len(_list(parsed.get("pending_candidate_decisions"))),
            "pending_candidate_field_count": len(_list(parsed.get("pending_candidate_fields"))),
            "preference_fill_allowed": bool(validated_input),
        },
        "readiness": {
            "boundary": BOUNDARY,
            "user_listening_review_input_guard_completed": True,
            "validated_review_input_present": bool(validated_input),
            "preference_fill_allowed": bool(validated_input),
            "listening_review_completed": False,
            "human_audio_preference_claimed": False,
            "model_direct_generation_quality_claimed": False,
            "midi_to_solo_musical_quality_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
        },
        "decision": {
            "current_boundary": BOUNDARY,
            "next_boundary": next_boundary,
            "auto_progress_allowed": True,
            "critical_user_input_required": False,
            "reason": reason,
        },
        "not_proven": [
            "listening_review_completed",
            "human_audio_preference",
            "model_direct_generation_quality",
            "midi_to_solo_musical_quality",
            "broad_trained_model_quality",
            "brad_style_adaptation",
        ],
        "next_recommended_issue": (
            "Stage B MIDI-to-solo model-direct user listening review fill"
            if validated_input
            else "Stage B MIDI-to-solo model-direct objective-only next decision"
        ),
    }


def validate_user_listening_review_input_guard_report(
    report: dict[str, Any],
    *,
    expected_boundary: str | None,
    expected_next_boundary: str | None,
    require_guard_completed: bool,
    require_pending_input: bool,
    require_no_quality_claim: bool,
) -> dict[str, Any]:
    boundary = str(report.get("boundary") or "")
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    guard = _dict(report.get("guard_result"))
    if expected_boundary and boundary != expected_boundary:
        raise StageBMidiToSoloModelDirectUserListeningReviewInputGuardError(
            f"expected boundary {expected_boundary}, got {boundary}"
        )
    if expected_next_boundary and str(decision.get("next_boundary") or "") != expected_next_boundary:
        raise StageBMidiToSoloModelDirectUserListeningReviewInputGuardError("unexpected next boundary")
    if require_guard_completed and not bool(readiness.get("user_listening_review_input_guard_completed", False)):
        raise StageBMidiToSoloModelDirectUserListeningReviewInputGuardError("guard completion required")
    if require_pending_input and bool(guard.get("validated_review_input_present", True)):
        raise StageBMidiToSoloModelDirectUserListeningReviewInputGuardError("review input should remain pending")
    if require_no_quality_claim:
        blocked = [
            "listening_review_completed",
            "human_audio_preference_claimed",
            "model_direct_generation_quality_claimed",
            "midi_to_solo_musical_quality_claimed",
            "broad_trained_model_quality_claimed",
            "brad_style_adaptation_claimed",
        ]
        claimed = [name for name in blocked if bool(readiness.get(name, True))]
        if claimed:
            raise StageBMidiToSoloModelDirectUserListeningReviewInputGuardError(f"unexpected claim: {claimed}")
    return {
        "boundary": boundary,
        "source_boundary": str(report.get("source_boundary") or ""),
        "validated_review_input_present": bool(guard.get("validated_review_input_present", True)),
        "preference_fill_allowed": bool(guard.get("preference_fill_allowed", True)),
        "pending_status_field_count": _int(guard.get("pending_status_field_count")),
        "pending_candidate_decision_count": _int(guard.get("pending_candidate_decision_count")),
        "pending_candidate_field_count": _int(guard.get("pending_candidate_field_count")),
        "human_audio_preference_claimed": bool(readiness.get("human_audio_preference_claimed", True)),
        "model_direct_generation_quality_claimed": bool(
            readiness.get("model_direct_generation_quality_claimed", True)
        ),
        "critical_user_input_required": bool(decision.get("critical_user_input_required", True)),
        "next_boundary": str(decision.get("next_boundary") or ""),
        "next_recommended_issue": str(report.get("next_recommended_issue") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    readiness = report["readiness"]
    decision = report["decision"]
    guard = report["guard_result"]
    lines = [
        "# Stage B MIDI-to-Solo Model-Direct User Listening Review Input Guard",
        "",
        "## Summary",
        "",
        f"- boundary: `{report['boundary']}`",
        f"- source boundary: `{report['source_boundary']}`",
        f"- next boundary: `{decision['next_boundary']}`",
        f"- validated review input present: `{_bool_token(guard['validated_review_input_present'])}`",
        f"- preference fill allowed: `{_bool_token(guard['preference_fill_allowed'])}`",
        f"- pending status field count: `{guard['pending_status_field_count']}`",
        f"- pending candidate decision count: `{guard['pending_candidate_decision_count']}`",
        f"- pending candidate field count: `{guard['pending_candidate_field_count']}`",
        f"- human/audio preference claimed: `{_bool_token(readiness['human_audio_preference_claimed'])}`",
        f"- model-direct generation quality claimed: `{_bool_token(readiness['model_direct_generation_quality_claimed'])}`",
        "",
        "## Not Proven",
        "",
    ]
    for item in report.get("not_proven", []):
        lines.append(f"- `{item}`")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Guard model-direct listening review fill input")
    parser.add_argument("--listening_review_package", type=str, required=True)
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/stage_b_midi_to_solo_model_direct_user_listening_review_input_guard",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--expected_boundary", type=str, default="")
    parser.add_argument("--expected_next_boundary", type=str, default="")
    parser.add_argument("--require_guard_completed", action="store_true")
    parser.add_argument("--require_pending_input", action="store_true")
    parser.add_argument("--require_no_quality_claim", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    report = build_user_listening_review_input_guard_report(
        read_json(Path(args.listening_review_package)),
        output_dir=output_dir,
    )
    summary = validate_user_listening_review_input_guard_report(
        report,
        expected_boundary=str(args.expected_boundary or ""),
        expected_next_boundary=str(args.expected_next_boundary or ""),
        require_guard_completed=bool(args.require_guard_completed),
        require_pending_input=bool(args.require_pending_input),
        require_no_quality_claim=bool(args.require_no_quality_claim),
    )
    write_json(output_dir / "stage_b_midi_to_solo_model_direct_user_listening_review_input_guard.json", report)
    write_json(
        output_dir / "stage_b_midi_to_solo_model_direct_user_listening_review_input_guard_validation_summary.json",
        summary,
    )
    markdown = markdown_report(report)
    write_text(output_dir / "stage_b_midi_to_solo_model_direct_user_listening_review_input_guard.md", markdown)
    if args.doc_path:
        write_text(Path(args.doc_path), markdown)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
