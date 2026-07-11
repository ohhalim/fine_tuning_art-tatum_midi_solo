"""Build audio render package metadata for generic tiny checkpoint repair candidates."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from scripts.assess_stage_b_generic_base_readiness import read_json, write_json, write_text  # noqa: E402
from scripts.run_stage_b_generic_tiny_checkpoint_generation_probe import (  # noqa: E402
    _bool_token,
    _dict,
    _int,
)


class StageBGenericTinyCheckpointRepairAudioRenderPackageError(ValueError):
    pass


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
        raise StageBGenericTinyCheckpointRepairAudioRenderPackageError(f"required file not found: {path}")
    return {
        "path": path,
        "required": required,
        "exists": exists,
        "size_bytes": int(target.stat().st_size) if exists else 0,
    }


def resolve_tool(name: str, renderer_paths: dict[str, str] | None = None) -> str:
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
    soundfont = Path(soundfont_path).expanduser() if soundfont_path else None
    soundfont_exists = bool(soundfont and soundfont.exists())
    resolved_soundfont_path = str(soundfont) if soundfont else ""

    if requested_renderer:
        if requested_renderer == "fluidsynth":
            selected = fluidsynth_path
            if not selected:
                status = "renderer_unavailable"
            else:
                status = "ready_for_local_render" if soundfont_exists else "soundfont_missing"
        elif requested_renderer == "timidity":
            selected = timidity_path
            status = "ready_for_local_render" if selected else "renderer_unavailable"
        else:
            raise StageBGenericTinyCheckpointRepairAudioRenderPackageError(
                f"unsupported renderer: {requested_renderer}"
            )
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
        "soundfont_path": resolved_soundfont_path,
        "soundfont_exists": soundfont_exists,
        "status": status,
    }


def validate_listening_fill_report(report: dict[str, Any]) -> list[dict[str, Any]]:
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    fill = _dict(report.get("listening_fill"))
    if str(readiness.get("boundary") or "") != "stage_b_generic_tiny_checkpoint_repair_listening_fill":
        raise StageBGenericTinyCheckpointRepairAudioRenderPackageError("unexpected listening fill boundary")
    if bool(report.get("review_input_present", True)):
        raise StageBGenericTinyCheckpointRepairAudioRenderPackageError("review input must be absent for this package")
    if str(report.get("fill_status") or "") != "pending_review_input":
        raise StageBGenericTinyCheckpointRepairAudioRenderPackageError("fill status must remain pending")
    if str(fill.get("status") or "") != "pending_review_input":
        raise StageBGenericTinyCheckpointRepairAudioRenderPackageError("listening fill must remain pending")
    if bool(readiness.get("human_review_filled", True)):
        raise StageBGenericTinyCheckpointRepairAudioRenderPackageError("human review must not be marked filled")
    if bool(readiness.get("musical_quality_claimed", True)):
        raise StageBGenericTinyCheckpointRepairAudioRenderPackageError("musical quality must not be claimed")
    if bool(readiness.get("broad_trained_model_quality_claimed", True)):
        raise StageBGenericTinyCheckpointRepairAudioRenderPackageError("broad trained-model quality must not be claimed")
    if bool(readiness.get("brad_style_adaptation_claimed", True)):
        raise StageBGenericTinyCheckpointRepairAudioRenderPackageError("Brad style adaptation must not be claimed")
    if str(decision.get("next_boundary") or "") != "stage_b_generic_tiny_checkpoint_repair_audio_render_package":
        raise StageBGenericTinyCheckpointRepairAudioRenderPackageError("unexpected next boundary")
    reviews = fill.get("candidate_reviews")
    if not isinstance(reviews, list) or not reviews:
        raise StageBGenericTinyCheckpointRepairAudioRenderPackageError("candidate reviews required")
    return [dict(review) for review in reviews if isinstance(review, dict)]


def compact_review_item(review: dict[str, Any]) -> dict[str, Any]:
    return {
        "review_rank": _int(review.get("review_rank")),
        "sample_seed": _int(review.get("sample_seed")),
        "sample_index": _int(review.get("sample_index")),
        "midi_file": file_meta(str(review.get("midi_path") or ""), required=True),
        "listening_status": str(review.get("status") or ""),
        "keep_decision": str(review.get("keep_decision") or ""),
    }


def item_output_stem(item: dict[str, Any]) -> str:
    return f"rank_{item['review_rank']:02d}_seed_{item['sample_seed']}_sample_{item['sample_index']}"


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


def build_audio_render_package(
    listening_fill_report: dict[str, Any],
    *,
    output_dir: Path,
    requested_renderer: str = "",
    soundfont_path: str = "",
    renderer_paths: dict[str, str] | None = None,
) -> dict[str, Any]:
    reviews = validate_listening_fill_report(listening_fill_report)
    probe = renderer_probe(
        requested_renderer=requested_renderer,
        soundfont_path=soundfont_path,
        renderer_paths=renderer_paths,
    )
    review_items = [compact_review_item(review) for review in reviews]
    planned_outputs: list[dict[str, Any]] = []
    for item in review_items:
        output_stem = item_output_stem(item)
        output_wav_path = output_dir / "audio" / f"{output_stem}.wav"
        midi_path = str(item["midi_file"]["path"])
        planned_outputs.append(
            {
                "review_rank": item["review_rank"],
                "sample_seed": item["sample_seed"],
                "sample_index": item["sample_index"],
                "source_midi_path": midi_path,
                "planned_wav_path": str(output_wav_path),
                "planned_wav_exists": output_wav_path.exists(),
                "render_command": render_command(probe, midi_path, str(output_wav_path)),
            }
        )
    render_ready = str(probe["status"]) == "ready_for_local_render"
    return {
        "schema_version": "stage_b_generic_tiny_checkpoint_repair_audio_render_package_v1",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "source_listening_fill_schema": str(listening_fill_report.get("schema_version") or ""),
        "source_listening_fill_run_dir": str(listening_fill_report.get("run_dir") or ""),
        "review_items": review_items,
        "renderer_probe": probe,
        "planned_audio_outputs": planned_outputs,
        "local_audio_render_boundary": {
            "boundary": "stage_b_generic_tiny_checkpoint_repair_audio_render_package",
            "status": str(probe["status"]),
            "render_attempted": False,
            "planned_audio_output_count": len(planned_outputs),
            "rendered_audio_file_count": 0,
            "audio_output_claimed": False,
            "audio_rendered_quality_claimed": False,
            "human_audio_preference_claimed": False,
            "musical_quality_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
        },
        "decision": {
            "current_boundary": "stage_b_generic_tiny_checkpoint_repair_audio_render_package",
            "next_boundary": "stage_b_generic_tiny_checkpoint_repair_local_audio_render_attempt"
            if render_ready
            else "stage_b_generic_tiny_checkpoint_repair_audio_render_tooling_setup",
            "auto_progress_allowed": render_ready,
            "critical_user_input_required": not render_ready,
        },
        "not_proven": [
            "audio_output",
            "audio_rendered_quality",
            "human_audio_preference",
            "musical_quality",
            "broad_trained_model_quality",
            "brad_style_adaptation",
        ],
        "next_recommended_issue": "Stage B generic tiny checkpoint repair local audio render attempt"
        if render_ready
        else "Stage B generic tiny checkpoint repair audio render tooling setup",
    }


def validate_audio_render_package(
    report: dict[str, Any],
    *,
    expected_boundary: str | None,
    expected_status: str | None,
    min_planned_outputs: int,
    require_required_midi_exists: bool,
    require_no_audio_claim: bool,
) -> dict[str, Any]:
    boundary = _dict(report.get("local_audio_render_boundary"))
    status = str(boundary.get("status") or "")
    if expected_boundary and str(boundary.get("boundary") or "") != expected_boundary:
        raise StageBGenericTinyCheckpointRepairAudioRenderPackageError(
            f"expected boundary {expected_boundary}, got {boundary.get('boundary')}"
        )
    if expected_status and status != expected_status:
        raise StageBGenericTinyCheckpointRepairAudioRenderPackageError(f"expected status {expected_status}, got {status}")
    planned_outputs = report.get("planned_audio_outputs")
    if not isinstance(planned_outputs, list):
        planned_outputs = []
    if len(planned_outputs) < min_planned_outputs:
        raise StageBGenericTinyCheckpointRepairAudioRenderPackageError(
            f"planned output count below target: {len(planned_outputs)} < {min_planned_outputs}"
        )
    if require_required_midi_exists:
        missing = []
        for item in report.get("review_items", []):
            midi_file = _dict(item.get("midi_file"))
            if bool(midi_file.get("required", False)) and not bool(midi_file.get("exists", False)):
                missing.append(str(midi_file.get("path") or "midi_file"))
        if missing:
            raise StageBGenericTinyCheckpointRepairAudioRenderPackageError(f"missing required MIDI files: {missing}")
    if require_no_audio_claim:
        claimed = [
            bool(boundary.get("render_attempted", True)),
            bool(boundary.get("audio_output_claimed", True)),
            bool(boundary.get("audio_rendered_quality_claimed", True)),
            bool(boundary.get("human_audio_preference_claimed", True)),
            bool(boundary.get("musical_quality_claimed", True)),
            bool(boundary.get("broad_trained_model_quality_claimed", True)),
            bool(boundary.get("brad_style_adaptation_claimed", True)),
        ]
        if any(claimed):
            raise StageBGenericTinyCheckpointRepairAudioRenderPackageError("audio or quality claims must not be set")
    decision = _dict(report.get("decision"))
    return {
        "boundary": str(boundary.get("boundary") or ""),
        "render_status": status,
        "selected_renderer_name": str(_dict(report.get("renderer_probe")).get("selected_renderer_name") or ""),
        "soundfont_exists": bool(_dict(report.get("renderer_probe")).get("soundfont_exists", False)),
        "planned_audio_output_count": len(planned_outputs),
        "render_attempted": bool(boundary.get("render_attempted", True)),
        "audio_rendered_quality_claimed": bool(boundary.get("audio_rendered_quality_claimed", True)),
        "human_audio_preference_claimed": bool(boundary.get("human_audio_preference_claimed", True)),
        "auto_progress_allowed": bool(decision.get("auto_progress_allowed", False)),
        "critical_user_input_required": bool(decision.get("critical_user_input_required", True)),
        "next_boundary": str(decision.get("next_boundary") or ""),
        "next_recommended_issue": str(report.get("next_recommended_issue") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    boundary = report["local_audio_render_boundary"]
    probe = report["renderer_probe"]
    lines = [
        "# Stage B Generic Tiny Checkpoint Repair Audio Render Package",
        "",
        "## Summary",
        "",
        f"- boundary: `{boundary['boundary']}`",
        f"- render status: `{boundary['status']}`",
        f"- selected renderer: `{probe['selected_renderer_name']}`",
        f"- soundfont exists: `{_bool_token(probe['soundfont_exists'])}`",
        f"- planned audio outputs: `{boundary['planned_audio_output_count']}`",
        f"- render attempted: `{_bool_token(boundary['render_attempted'])}`",
        f"- audio rendered quality claimed: `{_bool_token(boundary['audio_rendered_quality_claimed'])}`",
        f"- human/audio preference claimed: `{_bool_token(boundary['human_audio_preference_claimed'])}`",
        f"- musical quality claimed: `{_bool_token(boundary['musical_quality_claimed'])}`",
        "",
        "## Planned Outputs",
        "",
        "| rank | seed | sample | MIDI exists | planned WAV | command ready |",
        "|---:|---:|---:|:---:|---|:---:|",
    ]
    for item in report.get("planned_audio_outputs", []):
        midi_exists = Path(str(item.get("source_midi_path") or "")).exists()
        command_ready = bool(item.get("render_command"))
        lines.append(
            "| "
            + " | ".join(
                [
                    str(item.get("review_rank") or ""),
                    str(item.get("sample_seed") or ""),
                    str(item.get("sample_index") or ""),
                    _bool_token(midi_exists),
                    str(item.get("planned_wav_path") or ""),
                    _bool_token(command_ready),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Not Proven", ""])
    for item in report.get("not_proven", []):
        lines.append(f"- `{item}`")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    default_soundfont = Path.home() / ".local/share/soundfonts/generaluser-gs/v1.471.sf2"
    parser = argparse.ArgumentParser(description="Build generic tiny checkpoint repair audio render package metadata")
    parser.add_argument(
        "--listening_fill_report",
        type=str,
        default="outputs/stage_b_generic_tiny_checkpoint_repair_listening_fill/"
        "harness_stage_b_generic_tiny_checkpoint_repair_listening_fill/"
        "stage_b_generic_tiny_checkpoint_repair_listening_fill.json",
    )
    parser.add_argument("--output_root", type=str, default="outputs/stage_b_generic_tiny_checkpoint_repair_audio_render_package")
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--renderer", type=str, default="")
    parser.add_argument("--soundfont", type=str, default=str(default_soundfont))
    parser.add_argument("--expected_boundary", type=str, default="")
    parser.add_argument("--expected_status", type=str, default="")
    parser.add_argument("--min_planned_outputs", type=int, default=5)
    parser.add_argument("--require_required_midi_exists", action="store_true")
    parser.add_argument("--require_no_audio_claim", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    listening_fill_report_path = Path(args.listening_fill_report)
    if not listening_fill_report_path.exists():
        raise StageBGenericTinyCheckpointRepairAudioRenderPackageError("listening fill report required")
    report = build_audio_render_package(
        read_json(listening_fill_report_path),
        output_dir=output_dir,
        requested_renderer=str(args.renderer or ""),
        soundfont_path=str(args.soundfont or ""),
    )
    summary = validate_audio_render_package(
        report,
        expected_boundary=str(args.expected_boundary or ""),
        expected_status=str(args.expected_status or ""),
        min_planned_outputs=int(args.min_planned_outputs),
        require_required_midi_exists=bool(args.require_required_midi_exists),
        require_no_audio_claim=bool(args.require_no_audio_claim),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "stage_b_generic_tiny_checkpoint_repair_audio_render_package.json", report)
    write_json(output_dir / "stage_b_generic_tiny_checkpoint_repair_audio_render_package_validation_summary.json", summary)
    markdown = markdown_report(report)
    write_text(output_dir / "stage_b_generic_tiny_checkpoint_repair_audio_render_package.md", markdown)
    if args.doc_path:
        write_text(Path(args.doc_path), markdown)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
