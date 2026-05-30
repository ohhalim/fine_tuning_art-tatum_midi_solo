"""Render generic tiny checkpoint repair MIDI candidates to local WAV files."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from scripts.assess_stage_b_generic_base_readiness import read_json, write_json, write_text  # noqa: E402
from scripts.run_stage_b_generic_tiny_checkpoint_generation_probe import (  # noqa: E402
    _bool_token,
    _dict,
    _int,
)


class StageBGenericTinyCheckpointRepairAudioRenderError(ValueError):
    pass


CommandRunner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def wav_meta(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise StageBGenericTinyCheckpointRepairAudioRenderError(f"WAV not found: {path}")
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        sample_rate = handle.getframerate()
        frame_count = handle.getnframes()
    return {
        "path": str(path),
        "exists": True,
        "size_bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
        "channels": int(channels),
        "sample_width_bytes": int(sample_width),
        "sample_rate": int(sample_rate),
        "frame_count": int(frame_count),
        "duration_seconds": float(frame_count / sample_rate) if sample_rate else 0.0,
    }


def item_output_stem(item: dict[str, Any]) -> str:
    return f"rank_{item['review_rank']:02d}_seed_{item['sample_seed']}_sample_{item['sample_index']}"


def required_review_items(local_audio_render_package: dict[str, Any]) -> list[dict[str, Any]]:
    boundary = _dict(local_audio_render_package.get("local_audio_render_boundary"))
    if str(boundary.get("boundary") or "") != "stage_b_generic_tiny_checkpoint_repair_audio_render_package":
        raise StageBGenericTinyCheckpointRepairAudioRenderError("unexpected audio render package boundary")
    if str(boundary.get("status") or "") != "ready_for_local_render":
        raise StageBGenericTinyCheckpointRepairAudioRenderError("audio render package is not ready for local render")
    if bool(boundary.get("render_attempted", True)):
        raise StageBGenericTinyCheckpointRepairAudioRenderError("source package must not already attempt rendering")
    if bool(boundary.get("audio_rendered_quality_claimed", True)):
        raise StageBGenericTinyCheckpointRepairAudioRenderError("source package must not claim audio quality")
    if bool(boundary.get("human_audio_preference_claimed", True)):
        raise StageBGenericTinyCheckpointRepairAudioRenderError("source package must not claim human preference")
    items = local_audio_render_package.get("review_items")
    if not isinstance(items, list) or not items:
        raise StageBGenericTinyCheckpointRepairAudioRenderError("review items required")
    compacted: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        midi_file = _dict(item.get("midi_file"))
        midi_path = str(midi_file.get("path") or "")
        if bool(midi_file.get("required", False)) and not Path(midi_path).exists():
            raise StageBGenericTinyCheckpointRepairAudioRenderError(f"required MIDI not found: {midi_path}")
        compacted.append(
            {
                "review_rank": _int(item.get("review_rank")),
                "sample_seed": _int(item.get("sample_seed")),
                "sample_index": _int(item.get("sample_index")),
                "midi_path": midi_path,
            }
        )
    if not compacted:
        raise StageBGenericTinyCheckpointRepairAudioRenderError("no renderable review items")
    return compacted


def build_render_plan(
    local_audio_render_package: dict[str, Any],
    *,
    output_dir: Path,
    renderer_path: str,
    soundfont_path: str,
    sample_rate: int,
) -> list[dict[str, Any]]:
    renderer = Path(renderer_path)
    soundfont = Path(soundfont_path).expanduser()
    if not renderer.exists():
        raise StageBGenericTinyCheckpointRepairAudioRenderError(f"renderer not found: {renderer}")
    if not soundfont.exists():
        raise StageBGenericTinyCheckpointRepairAudioRenderError(f"soundfont not found: {soundfont}")
    plan: list[dict[str, Any]] = []
    for item in required_review_items(local_audio_render_package):
        output_stem = item_output_stem(item)
        wav_path = output_dir / "audio" / f"{output_stem}.wav"
        plan.append(
            {
                "review_rank": item["review_rank"],
                "sample_seed": item["sample_seed"],
                "sample_index": item["sample_index"],
                "midi_path": item["midi_path"],
                "wav_path": str(wav_path),
                "command": [
                    str(renderer),
                    "-ni",
                    "-F",
                    str(wav_path),
                    "-r",
                    str(sample_rate),
                    str(soundfont),
                    item["midi_path"],
                ],
            }
        )
    return plan


def default_runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(list(command), check=False, text=True, capture_output=True)


def execute_render_plan(
    plan: list[dict[str, Any]],
    *,
    runner: CommandRunner = default_runner,
) -> list[dict[str, Any]]:
    rendered: list[dict[str, Any]] = []
    for item in plan:
        wav_path = Path(str(item["wav_path"]))
        wav_path.parent.mkdir(parents=True, exist_ok=True)
        completed = runner(item["command"])
        if completed.returncode != 0:
            raise StageBGenericTinyCheckpointRepairAudioRenderError(
                f"render failed for rank {item['review_rank']}: {completed.stderr or completed.stdout}"
            )
        rendered.append(
            {
                "review_rank": item["review_rank"],
                "sample_seed": item["sample_seed"],
                "sample_index": item["sample_index"],
                "source_midi_path": item["midi_path"],
                "wav_file": wav_meta(wav_path),
                "command": list(item["command"]),
                "stdout_tail": (completed.stdout or "")[-1000:],
                "stderr_tail": (completed.stderr or "")[-1000:],
            }
        )
    return rendered


def build_audio_render_report(
    local_audio_render_package: dict[str, Any],
    *,
    output_dir: Path,
    renderer_path: str,
    soundfont_path: str,
    sample_rate: int,
    runner: CommandRunner = default_runner,
) -> dict[str, Any]:
    if not renderer_path:
        renderer_path = str(_dict(local_audio_render_package.get("renderer_probe")).get("selected_renderer") or "")
    if not soundfont_path:
        soundfont_path = str(_dict(local_audio_render_package.get("renderer_probe")).get("soundfont_path") or "")
    plan = build_render_plan(
        local_audio_render_package,
        output_dir=output_dir,
        renderer_path=renderer_path,
        soundfont_path=soundfont_path,
        sample_rate=sample_rate,
    )
    rendered = execute_render_plan(plan, runner=runner)
    return {
        "schema_version": "stage_b_generic_tiny_checkpoint_repair_local_audio_render_attempt_v1",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "source_local_audio_render_package_schema": str(local_audio_render_package.get("schema_version") or ""),
        "renderer": {
            "name": "fluidsynth",
            "path": renderer_path,
            "version": "",
        },
        "soundfont": {
            "path": str(Path(soundfont_path).expanduser()),
            "sha256": sha256_file(Path(soundfont_path).expanduser()),
        },
        "rendered_audio_files": rendered,
        "audio_render_boundary": {
            "boundary": "stage_b_generic_tiny_checkpoint_repair_local_audio_render_attempt",
            "render_attempted": True,
            "rendered_audio_file_count": len(rendered),
            "audio_output_claimed": True,
            "technical_wav_validation": True,
            "audio_rendered_quality_claimed": False,
            "human_audio_preference_claimed": False,
            "musical_quality_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
        },
        "decision": {
            "current_boundary": "stage_b_generic_tiny_checkpoint_repair_local_audio_render_attempt",
            "next_boundary": "stage_b_generic_tiny_checkpoint_repair_user_listening_review_input",
            "auto_progress_allowed": False,
            "critical_user_input_required": True,
            "reason": "human listening input required before audio quality or preference claim",
        },
        "not_proven": [
            "audio_rendered_quality",
            "human_audio_preference",
            "musical_quality",
            "broad_trained_model_quality",
            "brad_style_adaptation",
        ],
        "next_recommended_issue": "Stage B generic tiny checkpoint repair user listening review input",
    }


def validate_audio_render_report(
    report: dict[str, Any],
    *,
    expected_boundary: str | None,
    expected_file_count: int,
    expected_sample_rate: int,
    require_no_quality_claim: bool,
) -> dict[str, Any]:
    boundary = _dict(report.get("audio_render_boundary"))
    if expected_boundary and str(boundary.get("boundary") or "") != expected_boundary:
        raise StageBGenericTinyCheckpointRepairAudioRenderError(
            f"expected boundary {expected_boundary}, got {boundary.get('boundary')}"
        )
    files = report.get("rendered_audio_files")
    if not isinstance(files, list) or len(files) != expected_file_count:
        raise StageBGenericTinyCheckpointRepairAudioRenderError(f"expected {expected_file_count} rendered files")
    for item in files:
        wav_file = _dict(item.get("wav_file"))
        if not bool(wav_file.get("exists", False)):
            raise StageBGenericTinyCheckpointRepairAudioRenderError(f"missing rendered wav for rank {item.get('review_rank')}")
        if int(wav_file.get("sample_rate", 0) or 0) != expected_sample_rate:
            raise StageBGenericTinyCheckpointRepairAudioRenderError(
                f"unexpected sample rate for rank {item.get('review_rank')}"
            )
        if int(wav_file.get("frame_count", 0) or 0) <= 0:
            raise StageBGenericTinyCheckpointRepairAudioRenderError(f"empty wav for rank {item.get('review_rank')}")
        if int(wav_file.get("size_bytes", 0) or 0) <= 44:
            raise StageBGenericTinyCheckpointRepairAudioRenderError(f"invalid wav size for rank {item.get('review_rank')}")
    if require_no_quality_claim:
        claimed = [
            bool(boundary.get("audio_rendered_quality_claimed", True)),
            bool(boundary.get("human_audio_preference_claimed", True)),
            bool(boundary.get("musical_quality_claimed", True)),
            bool(boundary.get("broad_trained_model_quality_claimed", True)),
            bool(boundary.get("brad_style_adaptation_claimed", True)),
        ]
        if any(claimed):
            raise StageBGenericTinyCheckpointRepairAudioRenderError("quality or preference claims must not be set")
    decision = _dict(report.get("decision"))
    return {
        "boundary": str(boundary.get("boundary") or ""),
        "render_attempted": bool(boundary.get("render_attempted", False)),
        "rendered_audio_file_count": int(boundary.get("rendered_audio_file_count", 0) or 0),
        "technical_wav_validation": bool(boundary.get("technical_wav_validation", False)),
        "audio_rendered_quality_claimed": bool(boundary.get("audio_rendered_quality_claimed", True)),
        "human_audio_preference_claimed": bool(boundary.get("human_audio_preference_claimed", True)),
        "musical_quality_claimed": bool(boundary.get("musical_quality_claimed", True)),
        "critical_user_input_required": bool(decision.get("critical_user_input_required", False)),
        "wav_paths": [str(_dict(item.get("wav_file")).get("path") or "") for item in files],
        "next_recommended_issue": str(report.get("next_recommended_issue") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    boundary = report["audio_render_boundary"]
    lines = [
        "# Stage B Generic Tiny Checkpoint Repair Local Audio Render Attempt",
        "",
        "## Summary",
        "",
        f"- boundary: `{boundary['boundary']}`",
        f"- render attempted: `{_bool_token(boundary['render_attempted'])}`",
        f"- rendered audio file count: `{boundary['rendered_audio_file_count']}`",
        f"- technical WAV validation: `{_bool_token(boundary['technical_wav_validation'])}`",
        f"- audio rendered quality claimed: `{_bool_token(boundary['audio_rendered_quality_claimed'])}`",
        f"- human/audio preference claimed: `{_bool_token(boundary['human_audio_preference_claimed'])}`",
        f"- musical quality claimed: `{_bool_token(boundary['musical_quality_claimed'])}`",
        "",
        "## Rendered Files",
        "",
        "| rank | seed | sample | wav path | duration | sample rate | size | sha256 |",
        "|---:|---:|---:|---|---:|---:|---:|---|",
    ]
    for item in report.get("rendered_audio_files", []):
        wav_file = item["wav_file"]
        lines.append(
            "| "
            + " | ".join(
                [
                    str(item["review_rank"]),
                    str(item["sample_seed"]),
                    str(item["sample_index"]),
                    str(wav_file["path"]),
                    f"{float(wav_file['duration_seconds']):.3f}",
                    str(wav_file["sample_rate"]),
                    str(wav_file["size_bytes"]),
                    str(wav_file["sha256"][:12]),
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
    parser = argparse.ArgumentParser(description="Render generic tiny checkpoint repair audio")
    parser.add_argument("--local_audio_render_package", type=str, required=True)
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/stage_b_generic_tiny_checkpoint_repair_local_audio_render_attempt",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--renderer", type=str, default=shutil.which("fluidsynth") or "")
    parser.add_argument("--soundfont", type=str, default=str(default_soundfont))
    parser.add_argument("--sample_rate", type=int, default=44100)
    parser.add_argument("--expected_boundary", type=str, default="")
    parser.add_argument("--expected_file_count", type=int, default=5)
    parser.add_argument("--require_no_quality_claim", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    report = build_audio_render_report(
        read_json(Path(args.local_audio_render_package)),
        output_dir=output_dir,
        renderer_path=str(args.renderer or ""),
        soundfont_path=str(args.soundfont or ""),
        sample_rate=int(args.sample_rate),
    )
    summary = validate_audio_render_report(
        report,
        expected_boundary=str(args.expected_boundary or ""),
        expected_file_count=int(args.expected_file_count),
        expected_sample_rate=int(args.sample_rate),
        require_no_quality_claim=bool(args.require_no_quality_claim),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "stage_b_generic_tiny_checkpoint_repair_local_audio_render_attempt.json", report)
    write_json(
        output_dir / "stage_b_generic_tiny_checkpoint_repair_local_audio_render_attempt_validation_summary.json",
        summary,
    )
    markdown = markdown_report(report)
    write_text(output_dir / "stage_b_generic_tiny_checkpoint_repair_local_audio_render_attempt.md", markdown)
    if args.doc_path:
        write_text(Path(args.doc_path), markdown)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
