"""Build local audio render package metadata for duration coverage fill review."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class DurationCoverageFillLocalAudioRenderPackageError(ValueError):
    pass


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def file_meta(path: str, *, required: bool) -> dict[str, Any]:
    if not path:
        return {
            "path": "",
            "required": required,
            "exists": False,
            "size_bytes": 0,
        }
    target = Path(path)
    exists = target.exists()
    if required and not exists:
        raise DurationCoverageFillLocalAudioRenderPackageError(f"required file not found: {path}")
    return {
        "path": path,
        "required": required,
        "exists": exists,
        "size_bytes": int(target.stat().st_size) if exists else 0,
    }


def resolve_tool(name: str, renderer_paths: dict[str, str] | None) -> str:
    if renderer_paths and name in renderer_paths:
        return str(renderer_paths.get(name) or "")
    return shutil.which(name) or ""


def renderer_probe(
    *,
    requested_renderer: str,
    soundfont_path: str,
    renderer_paths: dict[str, str] | None = None,
) -> dict[str, Any]:
    fluidsynth_path = resolve_tool("fluidsynth", renderer_paths)
    timidity_path = resolve_tool("timidity", renderer_paths)
    selected = ""
    status = "renderer_unavailable"
    soundfont = Path(soundfont_path) if soundfont_path else None
    soundfont_exists = bool(soundfont and soundfont.exists())
    if requested_renderer:
        if requested_renderer == "fluidsynth":
            selected = fluidsynth_path
            status = "ready_for_local_render" if selected and soundfont_exists else "soundfont_missing"
        elif requested_renderer == "timidity":
            selected = timidity_path
            status = "ready_for_local_render" if selected else "renderer_unavailable"
        else:
            raise DurationCoverageFillLocalAudioRenderPackageError(f"unsupported renderer: {requested_renderer}")
    elif fluidsynth_path:
        selected = fluidsynth_path
        status = "ready_for_local_render" if soundfont_exists else "soundfont_missing"
    elif timidity_path:
        selected = timidity_path
        status = "ready_for_local_render"
    return {
        "requested_renderer": requested_renderer,
        "selected_renderer": selected,
        "selected_renderer_name": "fluidsynth" if selected == fluidsynth_path and fluidsynth_path else "timidity"
        if selected == timidity_path and timidity_path
        else "",
        "fluidsynth_path": fluidsynth_path,
        "timidity_path": timidity_path,
        "soundfont_path": str(soundfont_path or ""),
        "soundfont_exists": soundfont_exists,
        "status": status,
    }


def validate_external_boundary(external_boundary: dict[str, Any]) -> None:
    boundary = (
        external_boundary.get("external_review_boundary")
        if isinstance(external_boundary.get("external_review_boundary"), dict)
        else {}
    )
    if str(boundary.get("boundary") or "") != "external_human_audio_review_required_for_human_preference_claim":
        raise DurationCoverageFillLocalAudioRenderPackageError("unexpected external review boundary")
    if str(boundary.get("status") or "") != "pending_external_review_input":
        raise DurationCoverageFillLocalAudioRenderPackageError("external review input must remain pending")
    if bool(boundary.get("human_audio_preference_claimed", True)):
        raise DurationCoverageFillLocalAudioRenderPackageError("human/audio preference must not be claimed")


def validate_audio_review_package(audio_review_package: dict[str, Any]) -> None:
    boundary = (
        audio_review_package.get("package_boundary")
        if isinstance(audio_review_package.get("package_boundary"), dict)
        else {}
    )
    if str(boundary.get("status") or "") != "ready_for_external_review_input":
        raise DurationCoverageFillLocalAudioRenderPackageError("audio review package must be ready")
    if bool(boundary.get("preference_claimed", True)):
        raise DurationCoverageFillLocalAudioRenderPackageError("preference must not be claimed")
    items = audio_review_package.get("review_items")
    if not isinstance(items, list) or len(items) != 2:
        raise DurationCoverageFillLocalAudioRenderPackageError("audio review package must contain two review items")


def compact_review_item(item: dict[str, Any]) -> dict[str, Any]:
    role = str(item.get("role") or "")
    midi_file = item.get("midi_file") if isinstance(item.get("midi_file"), dict) else {}
    context_file = item.get("context_midi_file") if isinstance(item.get("context_midi_file"), dict) else {}
    return {
        "role": role,
        "candidate_id": str(item.get("candidate_id") or ""),
        "midi_file": file_meta(str(midi_file.get("path") or ""), required=bool(midi_file.get("required", False))),
        "context_midi_file": file_meta(
            str(context_file.get("path") or ""), required=bool(context_file.get("required", False))
        ),
        "metric_summary": dict(item.get("metric_summary") or {}),
    }


def render_command(probe: dict[str, Any], midi_path: str, output_wav_path: str) -> list[str]:
    renderer_name = str(probe.get("selected_renderer_name") or "")
    renderer = str(probe.get("selected_renderer") or "")
    if str(probe.get("status") or "") != "ready_for_local_render" or not renderer:
        return []
    if renderer_name == "fluidsynth":
        return [
            renderer,
            "-ni",
            str(probe.get("soundfont_path") or ""),
            midi_path,
            "-F",
            output_wav_path,
            "-r",
            "44100",
        ]
    if renderer_name == "timidity":
        return [renderer, midi_path, "-Ow", "-o", output_wav_path]
    return []


def build_local_audio_render_package(
    external_boundary: dict[str, Any],
    audio_review_package: dict[str, Any],
    *,
    output_dir: Path,
    requested_renderer: str = "",
    soundfont_path: str = "",
    renderer_paths: dict[str, str] | None = None,
) -> dict[str, Any]:
    validate_external_boundary(external_boundary)
    validate_audio_review_package(audio_review_package)
    candidate_id = str(audio_review_package.get("candidate_id") or "")
    probe = renderer_probe(
        requested_renderer=requested_renderer,
        soundfont_path=soundfont_path,
        renderer_paths=renderer_paths,
    )
    items = [compact_review_item(item) for item in audio_review_package["review_items"]]
    planned_outputs = []
    for item in items:
        output_wav_path = output_dir / "audio" / f"{item['role']}.wav"
        planned_outputs.append(
            {
                "role": item["role"],
                "candidate_id": item["candidate_id"],
                "source_midi_path": item["midi_file"]["path"],
                "planned_wav_path": str(output_wav_path),
                "planned_wav_exists": output_wav_path.exists(),
                "render_command": render_command(probe, item["midi_file"]["path"], str(output_wav_path)),
            }
        )
    return {
        "schema_version": "stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_local_audio_render_package_v1",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "source_external_boundary_schema": str(external_boundary.get("schema_version") or ""),
        "source_audio_review_package_schema": str(audio_review_package.get("schema_version") or ""),
        "candidate_id": candidate_id,
        "review_items": items,
        "renderer_probe": probe,
        "planned_audio_outputs": planned_outputs,
        "local_audio_render_boundary": {
            "status": str(probe["status"]),
            "render_attempted": False,
            "rendered_audio_file_count": 0,
            "audio_output_claimed": False,
            "audio_rendered_quality_claimed": False,
            "human_audio_preference_claimed": False,
        },
        "not_proven": [
            "audio_rendered_quality",
            "human_audio_preference",
            "broad_trained_model_quality",
            "brad_style_adaptation",
            "production_ready_improviser",
        ],
        "next_recommended_issue": "Stage B margin-recovered phrase/vocabulary duration coverage fill local audio render attempt"
        if str(probe["status"]) == "ready_for_local_render"
        else "Stage B margin-recovered phrase/vocabulary duration coverage fill local audio render tooling setup",
    }


def validate_local_audio_render_package(
    report: dict[str, Any],
    *,
    expected_candidate_id: str | None,
    expected_status: str | None,
    require_required_midi_exists: bool,
    require_no_audio_claim: bool,
) -> dict[str, Any]:
    candidate_id = str(report.get("candidate_id") or "")
    if expected_candidate_id and candidate_id != expected_candidate_id:
        raise DurationCoverageFillLocalAudioRenderPackageError(
            f"expected candidate {expected_candidate_id}, got {candidate_id}"
        )
    boundary = (
        report.get("local_audio_render_boundary")
        if isinstance(report.get("local_audio_render_boundary"), dict)
        else {}
    )
    status = str(boundary.get("status") or "")
    if expected_status and status != expected_status:
        raise DurationCoverageFillLocalAudioRenderPackageError(f"expected status {expected_status}, got {status}")
    if require_no_audio_claim:
        claimed = [
            bool(boundary.get("audio_output_claimed", True)),
            bool(boundary.get("audio_rendered_quality_claimed", True)),
            bool(boundary.get("human_audio_preference_claimed", True)),
        ]
        if any(claimed):
            raise DurationCoverageFillLocalAudioRenderPackageError("audio or human preference must not be claimed")
    if require_required_midi_exists:
        missing = []
        for item in report.get("review_items", []):
            for key in ("midi_file", "context_midi_file"):
                file_info = item.get(key) if isinstance(item.get(key), dict) else {}
                if bool(file_info.get("required", False)) and not bool(file_info.get("exists", False)):
                    missing.append(str(file_info.get("path") or key))
        if missing:
            raise DurationCoverageFillLocalAudioRenderPackageError(f"missing required MIDI files: {missing}")
    return {
        "candidate_id": candidate_id,
        "render_status": status,
        "selected_renderer_name": str(report.get("renderer_probe", {}).get("selected_renderer_name") or ""),
        "planned_audio_output_count": len(report.get("planned_audio_outputs", [])),
        "render_attempted": bool(boundary.get("render_attempted", True)),
        "audio_rendered_quality_claimed": bool(boundary.get("audio_rendered_quality_claimed", True)),
        "human_audio_preference_claimed": bool(boundary.get("human_audio_preference_claimed", True)),
        "next_recommended_issue": str(report.get("next_recommended_issue") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    boundary = report["local_audio_render_boundary"]
    probe = report["renderer_probe"]
    lines = [
        "# Stage B Margin-Recovered Phrase/Vocabulary Duration Coverage Fill Local Audio Render Package",
        "",
        f"- candidate: `{report['candidate_id']}`",
        f"- render status: `{boundary['status']}`",
        f"- selected renderer: `{probe['selected_renderer_name']}`",
        f"- soundfont exists: `{probe['soundfont_exists']}`",
        f"- render attempted: `{boundary['render_attempted']}`",
        f"- audio rendered quality claimed: `{boundary['audio_rendered_quality_claimed']}`",
        f"- human/audio preference claimed: `{boundary['human_audio_preference_claimed']}`",
        "",
        "| role | MIDI exists | planned WAV | command ready |",
        "|---|:---:|---|:---:|",
    ]
    for item in report.get("planned_audio_outputs", []):
        command_ready = bool(item.get("render_command"))
        lines.append(
            "| "
            + " | ".join(
                [
                    str(item.get("role") or ""),
                    str(Path(str(item.get("source_midi_path") or "")).exists()),
                    str(item.get("planned_wav_path") or ""),
                    str(command_ready),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Not Proven", ""])
    for item in report.get("not_proven", []):
        lines.append(f"- `{item}`")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build local audio render package metadata")
    parser.add_argument("--external_human_audio_boundary", type=str, required=True)
    parser.add_argument("--audio_review_package", type=str, required=True)
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_local_audio_render_package",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--renderer", type=str, default="")
    parser.add_argument("--soundfont", type=str, default="")
    parser.add_argument("--expected_candidate_id", type=str, default="")
    parser.add_argument("--expected_status", type=str, default="")
    parser.add_argument("--require_required_midi_exists", action="store_true")
    parser.add_argument("--require_no_audio_claim", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    report = build_local_audio_render_package(
        read_json(Path(args.external_human_audio_boundary)),
        read_json(Path(args.audio_review_package)),
        output_dir=output_dir,
        requested_renderer=str(args.renderer or ""),
        soundfont_path=str(args.soundfont or ""),
    )
    summary = validate_local_audio_render_package(
        report,
        expected_candidate_id=str(args.expected_candidate_id or ""),
        expected_status=str(args.expected_status or ""),
        require_required_midi_exists=bool(args.require_required_midi_exists),
        require_no_audio_claim=bool(args.require_no_audio_claim),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "duration_coverage_fill_local_audio_render_package.json"
    markdown_path = output_dir / "duration_coverage_fill_local_audio_render_package.md"
    write_json(report_path, report)
    write_json(output_dir / "duration_coverage_fill_local_audio_render_package_validation_summary.json", summary)
    markdown_path.write_text(markdown_report(report), encoding="utf-8")
    print(json.dumps({**summary, "report_path": str(report_path), "markdown_path": str(markdown_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
