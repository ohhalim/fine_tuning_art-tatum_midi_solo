"""Decide the next renderer-path boundary for Stage B local audio review."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class StageBRendererPathDecisionError(ValueError):
    pass


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def decision_for_status(status: str) -> dict[str, Any]:
    if status == "ready_for_local_render":
        return {
            "decision": "ready_for_local_audio_render_attempt",
            "critical_user_input_required": False,
            "blocked_reason": "",
            "next_recommended_issue": "Stage B margin-recovered phrase/vocabulary duration coverage fill local audio render attempt",
        }
    if status == "soundfont_missing":
        return {
            "decision": "soundfont_path_required_before_render_attempt",
            "critical_user_input_required": True,
            "blocked_reason": "renderer_available_but_soundfont_missing",
            "next_recommended_issue": "Stage B margin-recovered phrase/vocabulary duration coverage fill soundfont path handoff",
        }
    if status == "renderer_unavailable":
        return {
            "decision": "renderer_path_or_install_approval_required",
            "critical_user_input_required": True,
            "blocked_reason": "renderer_unavailable",
            "next_recommended_issue": "Stage B margin-recovered phrase/vocabulary duration coverage fill renderer dependency handoff",
        }
    raise StageBRendererPathDecisionError(f"unsupported tooling status: {status}")


def build_renderer_path_decision(
    tooling_report: dict[str, Any],
    *,
    output_dir: Path,
) -> dict[str, Any]:
    status = str(tooling_report.get("tooling_status") or "")
    boundary = tooling_report.get("setup_boundary") if isinstance(tooling_report.get("setup_boundary"), dict) else {}
    if bool(boundary.get("system_modified", True)):
        raise StageBRendererPathDecisionError("tooling report must not modify system")
    if bool(boundary.get("package_install_executed", True)):
        raise StageBRendererPathDecisionError("tooling report must not install packages")
    if bool(boundary.get("download_executed", True)):
        raise StageBRendererPathDecisionError("tooling report must not download files")
    if bool(boundary.get("audio_render_attempted", True)):
        raise StageBRendererPathDecisionError("tooling report must not attempt audio render")
    probe = tooling_report.get("renderer_probe") if isinstance(tooling_report.get("renderer_probe"), dict) else {}
    decision = decision_for_status(status)
    return {
        "schema_version": "stage_b_renderer_path_decision_v1",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "source_tooling_schema": str(tooling_report.get("schema_version") or ""),
        "tooling_status": status,
        "renderer_probe": {
            "selected_renderer_name": str(probe.get("selected_renderer_name") or ""),
            "fluidsynth_available": bool(probe.get("fluidsynth_path")),
            "timidity_available": bool(probe.get("timidity_path")),
            "soundfont_path": str(probe.get("soundfont_path") or ""),
            "soundfont_exists": bool(probe.get("soundfont_exists", False)),
        },
        "renderer_path_decision": decision,
        "allowed_paths": [
            {
                "path": "existing_fluidsynth_and_soundfont",
                "requires_user_action": True,
                "required_inputs": ["fluidsynth executable path or PATH availability", ".sf2/.sf3 soundfont path"],
                "auto_install": False,
            },
            {
                "path": "existing_timidity",
                "requires_user_action": True,
                "required_inputs": ["timidity executable path or PATH availability"],
                "auto_install": False,
            },
            {
                "path": "skip_audio_render_and_continue_midi_evidence_only",
                "requires_user_action": False,
                "required_inputs": [],
                "auto_install": False,
            },
        ],
        "not_executed": [
            "package_install",
            "external_download",
            "audio_render_attempt",
            "generated_audio_commit",
        ],
        "not_proven": [
            "audio_rendered_quality",
            "human_audio_preference",
            "broad_trained_model_quality",
            "brad_style_adaptation",
            "production_ready_improviser",
        ],
        "next_recommended_issue": decision["next_recommended_issue"],
    }


def validate_renderer_path_decision(
    report: dict[str, Any],
    *,
    expected_decision: str | None,
    require_no_execution: bool,
) -> dict[str, Any]:
    decision = (
        report.get("renderer_path_decision")
        if isinstance(report.get("renderer_path_decision"), dict)
        else {}
    )
    decision_name = str(decision.get("decision") or "")
    if expected_decision and decision_name != expected_decision:
        raise StageBRendererPathDecisionError(f"expected decision {expected_decision}, got {decision_name}")
    if require_no_execution:
        not_executed = set(str(item) for item in report.get("not_executed", []))
        required = {"package_install", "external_download", "audio_render_attempt", "generated_audio_commit"}
        missing = required - not_executed
        if missing:
            raise StageBRendererPathDecisionError(f"missing no-execution guards: {sorted(missing)}")
    return {
        "tooling_status": str(report.get("tooling_status") or ""),
        "decision": decision_name,
        "critical_user_input_required": bool(decision.get("critical_user_input_required", False)),
        "blocked_reason": str(decision.get("blocked_reason") or ""),
        "next_recommended_issue": str(report.get("next_recommended_issue") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    decision = report["renderer_path_decision"]
    lines = [
        "# Stage B Renderer Path Decision",
        "",
        f"- tooling status: `{report['tooling_status']}`",
        f"- decision: `{decision['decision']}`",
        f"- critical user input required: `{decision['critical_user_input_required']}`",
        f"- blocked reason: `{decision['blocked_reason']}`",
        "",
        "## Allowed Paths",
        "",
    ]
    for item in report.get("allowed_paths", []):
        lines.append(
            f"- `{item['path']}`: requires user action `{item['requires_user_action']}`, auto install `{item['auto_install']}`"
        )
    lines.extend(["", "## Not Executed", ""])
    for item in report.get("not_executed", []):
        lines.append(f"- `{item}`")
    lines.extend(["", "## Not Proven", ""])
    for item in report.get("not_proven", []):
        lines.append(f"- `{item}`")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Decide Stage B renderer path boundary")
    parser.add_argument("--tooling_report", type=str, required=True)
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/stage_b_renderer_path_decision",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--expected_decision", type=str, default="")
    parser.add_argument("--require_no_execution", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    report = build_renderer_path_decision(read_json(Path(args.tooling_report)), output_dir=output_dir)
    summary = validate_renderer_path_decision(
        report,
        expected_decision=str(args.expected_decision or ""),
        require_no_execution=bool(args.require_no_execution),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "stage_b_renderer_path_decision.json"
    markdown_path = output_dir / "stage_b_renderer_path_decision.md"
    write_json(report_path, report)
    write_json(output_dir / "stage_b_renderer_path_decision_validation_summary.json", summary)
    markdown_path.write_text(markdown_report(report), encoding="utf-8")
    print(json.dumps({**summary, "report_path": str(report_path), "markdown_path": str(markdown_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
