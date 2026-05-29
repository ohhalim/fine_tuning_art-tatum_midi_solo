"""Check local audio render tooling readiness without modifying the system."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_local_audio_render_package import (
    renderer_probe,
)


class StageBLocalAudioRenderToolingError(ValueError):
    pass


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def install_guidance(status: str) -> list[str]:
    if status == "ready_for_local_render":
        return []
    if status == "soundfont_missing":
        return [
            "provide an existing .sf2/.sf3 path with --soundfont",
            "verify the soundfont license before committing any derived audio",
        ]
    return [
        "install or provide a local renderer path outside this script",
        "supported renderer options: fluidsynth with soundfont, or timidity",
        "rerun readiness check before render attempt",
    ]


def build_tooling_report(
    *,
    output_dir: Path,
    requested_renderer: str = "",
    soundfont_path: str = "",
    renderer_paths: dict[str, str] | None = None,
) -> dict[str, Any]:
    probe = renderer_probe(
        requested_renderer=requested_renderer,
        soundfont_path=soundfont_path,
        renderer_paths=renderer_paths,
    )
    status = str(probe.get("status") or "")
    return {
        "schema_version": "stage_b_local_audio_render_tooling_readiness_v1",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "tooling_status": status,
        "renderer_probe": probe,
        "setup_boundary": {
            "system_modified": False,
            "package_install_executed": False,
            "download_executed": False,
            "generated_audio_created": False,
            "audio_render_attempted": False,
        },
        "install_guidance": install_guidance(status),
        "next_recommended_issue": "Stage B margin-recovered phrase/vocabulary duration coverage fill local audio render attempt"
        if status == "ready_for_local_render"
        else "Stage B margin-recovered phrase/vocabulary duration coverage fill renderer path decision",
    }


def validate_tooling_report(
    report: dict[str, Any],
    *,
    expected_status: str | None,
    require_no_system_modification: bool,
) -> dict[str, Any]:
    status = str(report.get("tooling_status") or "")
    if expected_status and status != expected_status:
        raise StageBLocalAudioRenderToolingError(f"expected status {expected_status}, got {status}")
    boundary = report.get("setup_boundary") if isinstance(report.get("setup_boundary"), dict) else {}
    if require_no_system_modification:
        changed = [
            bool(boundary.get("system_modified", True)),
            bool(boundary.get("package_install_executed", True)),
            bool(boundary.get("download_executed", True)),
            bool(boundary.get("generated_audio_created", True)),
            bool(boundary.get("audio_render_attempted", True)),
        ]
        if any(changed):
            raise StageBLocalAudioRenderToolingError("tooling check must not modify system or render audio")
    probe = report.get("renderer_probe") if isinstance(report.get("renderer_probe"), dict) else {}
    return {
        "tooling_status": status,
        "selected_renderer_name": str(probe.get("selected_renderer_name") or ""),
        "fluidsynth_available": bool(probe.get("fluidsynth_path")),
        "timidity_available": bool(probe.get("timidity_path")),
        "soundfont_exists": bool(probe.get("soundfont_exists", False)),
        "system_modified": bool(boundary.get("system_modified", True)),
        "audio_render_attempted": bool(boundary.get("audio_render_attempted", True)),
        "next_recommended_issue": str(report.get("next_recommended_issue") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    probe = report["renderer_probe"]
    boundary = report["setup_boundary"]
    lines = [
        "# Stage B Local Audio Render Tooling Readiness",
        "",
        f"- tooling status: `{report['tooling_status']}`",
        f"- selected renderer: `{probe['selected_renderer_name']}`",
        f"- fluidsynth path: `{probe['fluidsynth_path']}`",
        f"- timidity path: `{probe['timidity_path']}`",
        f"- soundfont path: `{probe['soundfont_path']}`",
        f"- soundfont exists: `{probe['soundfont_exists']}`",
        f"- system modified: `{boundary['system_modified']}`",
        f"- audio render attempted: `{boundary['audio_render_attempted']}`",
        "",
        "## Guidance",
        "",
    ]
    for item in report.get("install_guidance", []):
        lines.append(f"- {item}")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check Stage B local audio render tooling")
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/stage_b_local_audio_render_tooling",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--renderer", type=str, default="")
    parser.add_argument("--soundfont", type=str, default="")
    parser.add_argument("--expected_status", type=str, default="")
    parser.add_argument("--require_no_system_modification", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    report = build_tooling_report(
        output_dir=output_dir,
        requested_renderer=str(args.renderer or ""),
        soundfont_path=str(args.soundfont or ""),
    )
    summary = validate_tooling_report(
        report,
        expected_status=str(args.expected_status or ""),
        require_no_system_modification=bool(args.require_no_system_modification),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "stage_b_local_audio_render_tooling.json"
    markdown_path = output_dir / "stage_b_local_audio_render_tooling.md"
    write_json(report_path, report)
    write_json(output_dir / "stage_b_local_audio_render_tooling_validation_summary.json", summary)
    markdown_path.write_text(markdown_report(report), encoding="utf-8")
    print(json.dumps({**summary, "report_path": str(report_path), "markdown_path": str(markdown_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
