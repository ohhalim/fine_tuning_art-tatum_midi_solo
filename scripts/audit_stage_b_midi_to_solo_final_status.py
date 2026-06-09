"""Audit the final technical MVP status after README evidence refresh."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from scripts.assess_stage_b_generic_base_readiness import read_json, write_json, write_text  # noqa: E402
from scripts.build_stage_b_midi_to_solo_mvp_delivery_package import (  # noqa: E402
    BOUNDARY as DELIVERY_BOUNDARY,
    StageBMidiToSoloMvpDeliveryPackageError,
    validate_delivery_package_report,
)


class StageBMidiToSoloFinalStatusAuditError(ValueError):
    pass


BOUNDARY = "stage_b_midi_to_solo_final_status_audit"
NEXT_BOUNDARY = "stage_b_midi_to_solo_post_mvp_quality_iteration_plan"
SCHEMA_VERSION = "stage_b_midi_to_solo_final_status_audit_v1"

QUALITY_CLAIM_KEYS = [
    "human_audio_preference_claimed",
    "midi_to_solo_musical_quality_claimed",
    "musical_quality_claimed",
    "phrase_bank_musical_quality_claimed",
    "audio_rendered_quality_claimed",
    "model_checkpoint_generation_quality_claimed",
    "model_direct_generation_quality_claimed",
    "broad_trained_model_quality_claimed",
    "brad_style_adaptation_claimed",
    "production_ready_claimed",
]

REQUIRED_README_SNIPPETS = [
    "- latest evidence boundary: `stage_b_midi_to_solo_mvp_delivery_package`",
    "- MVP delivery package completed: `true`",
    "- runnable CLI ready: `true`",
    "- input to ranked MIDI ready: `true`",
    "- input to rendered WAV evidence ready: `true`",
    "- changed-ratio repair audio evidence ready: `true`",
    "- MVP delivery raw artifact upload required: `false`",
    "- human/audio preference claim: `false`",
    "- MIDI-to-solo musical quality claim: `false`",
    "- boundary: `stage_b_midi_to_solo_mvp_delivery_package`",
    "- next boundary: `stage_b_midi_to_solo_readme_final_evidence_refresh`",
]


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _bool_token(value: Any) -> str:
    return "true" if bool(value) else "false"


def _require_no_quality_claim(container: dict[str, Any], *, label: str) -> None:
    claimed = [name for name in QUALITY_CLAIM_KEYS if bool(container.get(name, False))]
    if claimed:
        raise StageBMidiToSoloFinalStatusAuditError(
            f"unexpected quality claim in {label}: {claimed}"
        )


def validate_delivery_report(report: dict[str, Any]) -> dict[str, Any]:
    if str(report.get("boundary") or "") != DELIVERY_BOUNDARY:
        raise StageBMidiToSoloFinalStatusAuditError("MVP delivery package boundary required")
    try:
        summary = validate_delivery_package_report(
            report,
            expected_boundary=DELIVERY_BOUNDARY,
            expected_next_boundary="stage_b_midi_to_solo_readme_final_evidence_refresh",
            require_delivery_completed=True,
            require_no_quality_claim=True,
        )
    except StageBMidiToSoloMvpDeliveryPackageError as exc:
        raise StageBMidiToSoloFinalStatusAuditError(str(exc)) from exc
    return summary


def validate_readme(readme_text: str) -> dict[str, Any]:
    missing = [snippet for snippet in REQUIRED_README_SNIPPETS if snippet not in readme_text]
    if missing:
        raise StageBMidiToSoloFinalStatusAuditError(f"README snippets missing: {missing}")
    return {
        "required_snippet_count": len(REQUIRED_README_SNIPPETS),
        "missing_required_snippet_count": 0,
        "readme_final_evidence_reflected": True,
    }


def build_final_status_audit_report(
    *,
    delivery_package: dict[str, Any],
    readme_text: str,
    output_dir: Path,
    issue_number: int,
) -> dict[str, Any]:
    delivery = validate_delivery_report(delivery_package)
    readme = validate_readme(readme_text)
    final_status = {
        "technical_mvp_complete": bool(delivery["mvp_delivery_package_completed"]),
        "technical_mvp_ready_for_local_review": bool(delivery["runnable_cli_ready"]),
        "readme_final_evidence_reflected": bool(readme["readme_final_evidence_reflected"]),
        "input_to_ranked_midi_ready": bool(delivery["input_to_ranked_midi_ready"]),
        "input_to_rendered_wav_evidence_ready": bool(delivery["input_to_rendered_wav_evidence_ready"]),
        "changed_ratio_repair_audio_evidence_ready": bool(
            delivery["changed_ratio_repair_audio_evidence_ready"]
        ),
        "cli_candidate_count": _int(delivery["cli_candidate_count"]),
        "changed_ratio_repair_wav_count": _int(delivery["changed_ratio_repair_wav_count"]),
        "listening_review_quality_gap_open": bool(delivery["listening_review_quality_gap_open"]),
        "raw_artifact_upload_required": bool(delivery["raw_artifact_upload_required"]),
        "human_audio_preference_claimed": bool(delivery["human_audio_preference_claimed"]),
        "midi_to_solo_musical_quality_claimed": bool(
            delivery["midi_to_solo_musical_quality_claimed"]
        ),
    }
    _require_no_quality_claim(final_status, label="final status")
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "issue_number": int(issue_number),
        "boundary": BOUNDARY,
        "source_boundary": DELIVERY_BOUNDARY,
        "final_status": final_status,
        "readme_audit": readme,
        "readiness": {
            "boundary": BOUNDARY,
            "final_status_audit_completed": True,
            "technical_mvp_complete": bool(final_status["technical_mvp_complete"]),
            "technical_mvp_ready_for_local_review": bool(
                final_status["technical_mvp_ready_for_local_review"]
            ),
            "readme_final_evidence_reflected": bool(
                final_status["readme_final_evidence_reflected"]
            ),
            "human_audio_preference_claimed": False,
            "midi_to_solo_musical_quality_claimed": False,
            "audio_rendered_quality_claimed": False,
            "model_checkpoint_generation_quality_claimed": False,
            "model_direct_generation_quality_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
            "production_ready_claimed": False,
        },
        "decision": {
            "current_boundary": BOUNDARY,
            "next_boundary": NEXT_BOUNDARY,
            "auto_progress_allowed": True,
            "critical_user_input_required": False,
            "reason": "technical MVP is complete and README-reflected; musical quality remains a post-MVP iteration target",
        },
        "not_proven": [
            "listening_review_completed",
            "human_audio_preference",
            "midi_to_solo_musical_quality",
            "audio_rendered_quality",
            "broad_trained_model_quality",
            "brad_style_adaptation",
            "production_ready_improviser",
        ],
        "next_recommended_issue": "Stage B MIDI-to-solo post-MVP musical quality iteration plan",
    }


def validate_final_status_audit_report(
    report: dict[str, Any],
    *,
    expected_boundary: str | None,
    expected_next_boundary: str | None,
    require_technical_mvp_complete: bool,
    require_readme_reflected: bool,
    require_no_quality_claim: bool,
) -> dict[str, Any]:
    boundary = str(report.get("boundary") or "")
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    final_status = _dict(report.get("final_status"))
    if expected_boundary and boundary != expected_boundary:
        raise StageBMidiToSoloFinalStatusAuditError(
            f"expected boundary {expected_boundary}, got {boundary}"
        )
    if expected_next_boundary and str(decision.get("next_boundary") or "") != expected_next_boundary:
        raise StageBMidiToSoloFinalStatusAuditError("unexpected next boundary")
    if not bool(readiness.get("final_status_audit_completed", False)):
        raise StageBMidiToSoloFinalStatusAuditError("final status audit completion required")
    if require_technical_mvp_complete and not bool(final_status.get("technical_mvp_complete", False)):
        raise StageBMidiToSoloFinalStatusAuditError("technical MVP completion required")
    if require_readme_reflected and not bool(final_status.get("readme_final_evidence_reflected", False)):
        raise StageBMidiToSoloFinalStatusAuditError("README final evidence reflection required")
    if bool(decision.get("critical_user_input_required", True)):
        raise StageBMidiToSoloFinalStatusAuditError("critical user input should not be required")
    if require_no_quality_claim:
        _require_no_quality_claim(readiness, label="final status readiness")
    return {
        "boundary": boundary,
        "next_boundary": str(decision.get("next_boundary") or ""),
        "final_status_audit_completed": bool(
            readiness.get("final_status_audit_completed", False)
        ),
        "technical_mvp_complete": bool(final_status.get("technical_mvp_complete", False)),
        "technical_mvp_ready_for_local_review": bool(
            final_status.get("technical_mvp_ready_for_local_review", False)
        ),
        "readme_final_evidence_reflected": bool(
            final_status.get("readme_final_evidence_reflected", False)
        ),
        "cli_candidate_count": _int(final_status.get("cli_candidate_count")),
        "changed_ratio_repair_wav_count": _int(
            final_status.get("changed_ratio_repair_wav_count")
        ),
        "listening_review_quality_gap_open": bool(
            final_status.get("listening_review_quality_gap_open", False)
        ),
        "raw_artifact_upload_required": bool(final_status.get("raw_artifact_upload_required", True)),
        "human_audio_preference_claimed": bool(
            readiness.get("human_audio_preference_claimed", True)
        ),
        "midi_to_solo_musical_quality_claimed": bool(
            readiness.get("midi_to_solo_musical_quality_claimed", True)
        ),
        "critical_user_input_required": bool(decision.get("critical_user_input_required", True)),
        "next_recommended_issue": str(report.get("next_recommended_issue") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    final_status = report["final_status"]
    readiness = report["readiness"]
    decision = report["decision"]
    lines = [
        "# Stage B MIDI-to-Solo Final Status Audit",
        "",
        "## Summary",
        "",
        f"- boundary: `{report['boundary']}`",
        f"- source boundary: `{report['source_boundary']}`",
        f"- next boundary: `{decision['next_boundary']}`",
        f"- technical MVP complete: `{_bool_token(final_status['technical_mvp_complete'])}`",
        f"- technical MVP ready for local review: `{_bool_token(final_status['technical_mvp_ready_for_local_review'])}`",
        f"- README final evidence reflected: `{_bool_token(final_status['readme_final_evidence_reflected'])}`",
        "",
        "## Evidence",
        "",
        f"- input to ranked MIDI ready: `{_bool_token(final_status['input_to_ranked_midi_ready'])}`",
        f"- input to rendered WAV evidence ready: `{_bool_token(final_status['input_to_rendered_wav_evidence_ready'])}`",
        f"- changed-ratio repair audio evidence ready: `{_bool_token(final_status['changed_ratio_repair_audio_evidence_ready'])}`",
        f"- CLI candidate count: `{final_status['cli_candidate_count']}`",
        f"- changed-ratio repair WAV count: `{final_status['changed_ratio_repair_wav_count']}`",
        f"- listening review quality gap open: `{_bool_token(final_status['listening_review_quality_gap_open'])}`",
        f"- raw artifact upload required: `{_bool_token(final_status['raw_artifact_upload_required'])}`",
        "",
        "## Claim Boundary",
        "",
        f"- human/audio preference claimed: `{_bool_token(readiness['human_audio_preference_claimed'])}`",
        f"- MIDI-to-solo musical quality claimed: `{_bool_token(readiness['midi_to_solo_musical_quality_claimed'])}`",
        f"- broad trained model quality claimed: `{_bool_token(readiness['broad_trained_model_quality_claimed'])}`",
        f"- Brad style adaptation claimed: `{_bool_token(readiness['brad_style_adaptation_claimed'])}`",
        "",
        "## Next",
        "",
        f"- `{report['next_recommended_issue']}`",
    ]
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit MIDI-to-solo final technical MVP status")
    parser.add_argument("--delivery_package", type=str, required=True)
    parser.add_argument("--readme_path", type=str, default="README.md")
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/stage_b_midi_to_solo_final_status_audit",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--issue_number", type=int, default=742)
    parser.add_argument("--expected_boundary", type=str, default="")
    parser.add_argument("--expected_next_boundary", type=str, default="")
    parser.add_argument("--require_technical_mvp_complete", action="store_true")
    parser.add_argument("--require_readme_reflected", action="store_true")
    parser.add_argument("--require_no_quality_claim", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    report = build_final_status_audit_report(
        delivery_package=read_json(Path(args.delivery_package)),
        readme_text=Path(args.readme_path).read_text(encoding="utf-8"),
        output_dir=output_dir,
        issue_number=int(args.issue_number),
    )
    summary = validate_final_status_audit_report(
        report,
        expected_boundary=str(args.expected_boundary or ""),
        expected_next_boundary=str(args.expected_next_boundary or ""),
        require_technical_mvp_complete=bool(args.require_technical_mvp_complete),
        require_readme_reflected=bool(args.require_readme_reflected),
        require_no_quality_claim=bool(args.require_no_quality_claim),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "stage_b_midi_to_solo_final_status_audit.json", report)
    write_json(output_dir / "stage_b_midi_to_solo_final_status_audit_validation_summary.json", summary)
    markdown = markdown_report(report)
    write_text(output_dir / "stage_b_midi_to_solo_final_status_audit.md", markdown)
    if args.doc_path:
        write_text(Path(args.doc_path), markdown)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
