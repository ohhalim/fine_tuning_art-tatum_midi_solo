"""Render duration/coverage fill review MIDI files to local WAV files."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence


class StageBDurationCoverageFillAudioRenderError(ValueError):
    pass


CommandRunner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def wav_meta(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise StageBDurationCoverageFillAudioRenderError(f"WAV not found: {path}")
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


def required_review_items(local_audio_render_package: dict[str, Any]) -> list[dict[str, Any]]:
    items = local_audio_render_package.get("review_items")
    if not isinstance(items, list) or len(items) != 2:
        raise StageBDurationCoverageFillAudioRenderError("local audio render package must contain two review items")
    compacted: list[dict[str, Any]] = []
    for item in items:
        midi_file = item.get("midi_file") if isinstance(item.get("midi_file"), dict) else {}
        midi_path = str(midi_file.get("path") or "")
        if bool(midi_file.get("required", False)) and not Path(midi_path).exists():
            raise StageBDurationCoverageFillAudioRenderError(f"required MIDI not found: {midi_path}")
        compacted.append(
            {
                "role": str(item.get("role") or ""),
                "candidate_id": str(item.get("candidate_id") or ""),
                "midi_path": midi_path,
            }
        )
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
    soundfont = Path(soundfont_path)
    if not renderer.exists():
        raise StageBDurationCoverageFillAudioRenderError(f"renderer not found: {renderer}")
    if not soundfont.exists():
        raise StageBDurationCoverageFillAudioRenderError(f"soundfont not found: {soundfont}")
    plan = []
    for item in required_review_items(local_audio_render_package):
        wav_path = output_dir / "audio" / f"{item['role']}.wav"
        plan.append(
            {
                "role": item["role"],
                "candidate_id": item["candidate_id"],
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
    rendered = []
    for item in plan:
        wav_path = Path(str(item["wav_path"]))
        wav_path.parent.mkdir(parents=True, exist_ok=True)
        completed = runner(item["command"])
        if completed.returncode != 0:
            raise StageBDurationCoverageFillAudioRenderError(
                f"render failed for {item['role']}: {completed.stderr or completed.stdout}"
            )
        meta = wav_meta(wav_path)
        rendered.append(
            {
                "role": item["role"],
                "candidate_id": item["candidate_id"],
                "source_midi_path": item["midi_path"],
                "wav_file": meta,
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
    plan = build_render_plan(
        local_audio_render_package,
        output_dir=output_dir,
        renderer_path=renderer_path,
        soundfont_path=soundfont_path,
        sample_rate=sample_rate,
    )
    rendered = execute_render_plan(plan, runner=runner)
    return {
        "schema_version": "stage_b_duration_coverage_fill_local_audio_render_attempt_v1",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "source_local_audio_render_package_schema": str(local_audio_render_package.get("schema_version") or ""),
        "candidate_id": str(local_audio_render_package.get("candidate_id") or ""),
        "renderer": {
            "name": "fluidsynth",
            "path": renderer_path,
            "version": "",
        },
        "soundfont": {
            "path": soundfont_path,
            "sha256": sha256_file(Path(soundfont_path)),
        },
        "rendered_audio_files": rendered,
        "audio_render_boundary": {
            "render_attempted": True,
            "rendered_audio_file_count": len(rendered),
            "audio_output_claimed": True,
            "technical_wav_validation": True,
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
        "next_recommended_issue": "Stage B margin-recovered phrase/vocabulary duration coverage fill user listening review fill",
    }


def validate_audio_render_report(
    report: dict[str, Any],
    *,
    expected_file_count: int,
    expected_sample_rate: int,
    require_no_quality_claim: bool,
) -> dict[str, Any]:
    files = report.get("rendered_audio_files")
    if not isinstance(files, list) or len(files) != expected_file_count:
        raise StageBDurationCoverageFillAudioRenderError(f"expected {expected_file_count} rendered files")
    for item in files:
        wav_file = item.get("wav_file") if isinstance(item.get("wav_file"), dict) else {}
        if not bool(wav_file.get("exists", False)):
            raise StageBDurationCoverageFillAudioRenderError(f"missing rendered wav for {item.get('role')}")
        if int(wav_file.get("sample_rate", 0) or 0) != expected_sample_rate:
            raise StageBDurationCoverageFillAudioRenderError(f"unexpected sample rate for {item.get('role')}")
        if int(wav_file.get("frame_count", 0) or 0) <= 0:
            raise StageBDurationCoverageFillAudioRenderError(f"empty wav for {item.get('role')}")
        if int(wav_file.get("size_bytes", 0) or 0) <= 44:
            raise StageBDurationCoverageFillAudioRenderError(f"invalid wav size for {item.get('role')}")
    boundary = report.get("audio_render_boundary") if isinstance(report.get("audio_render_boundary"), dict) else {}
    if require_no_quality_claim:
        if bool(boundary.get("audio_rendered_quality_claimed", True)):
            raise StageBDurationCoverageFillAudioRenderError("audio rendered quality must not be claimed")
        if bool(boundary.get("human_audio_preference_claimed", True)):
            raise StageBDurationCoverageFillAudioRenderError("human/audio preference must not be claimed")
    return {
        "candidate_id": str(report.get("candidate_id") or ""),
        "render_attempted": bool(boundary.get("render_attempted", False)),
        "rendered_audio_file_count": int(boundary.get("rendered_audio_file_count", 0) or 0),
        "technical_wav_validation": bool(boundary.get("technical_wav_validation", False)),
        "audio_rendered_quality_claimed": bool(boundary.get("audio_rendered_quality_claimed", True)),
        "human_audio_preference_claimed": bool(boundary.get("human_audio_preference_claimed", True)),
        "wav_paths": [str(item.get("wav_file", {}).get("path") or "") for item in files],
        "next_recommended_issue": str(report.get("next_recommended_issue") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    boundary = report["audio_render_boundary"]
    lines = [
        "# Stage B Duration Coverage Fill Local Audio Render Attempt",
        "",
        f"- candidate: `{report['candidate_id']}`",
        f"- render attempted: `{boundary['render_attempted']}`",
        f"- rendered audio file count: `{boundary['rendered_audio_file_count']}`",
        f"- technical WAV validation: `{boundary['technical_wav_validation']}`",
        f"- audio rendered quality claimed: `{boundary['audio_rendered_quality_claimed']}`",
        f"- human/audio preference claimed: `{boundary['human_audio_preference_claimed']}`",
        "",
        "| role | wav path | duration | sample rate | size | sha256 |",
        "|---|---|---:|---:|---:|---|",
    ]
    for item in report.get("rendered_audio_files", []):
        wav_file = item["wav_file"]
        lines.append(
            "| "
            + " | ".join(
                [
                    str(item["role"]),
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
    parser = argparse.ArgumentParser(description="Render Stage B duration/coverage fill audio")
    parser.add_argument("--local_audio_render_package", type=str, required=True)
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/stage_b_duration_coverage_fill_local_audio_render_attempt",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--renderer", type=str, default=shutil.which("fluidsynth") or "")
    parser.add_argument("--soundfont", type=str, required=True)
    parser.add_argument("--sample_rate", type=int, default=44100)
    parser.add_argument("--expected_file_count", type=int, default=2)
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
        expected_file_count=int(args.expected_file_count),
        expected_sample_rate=int(args.sample_rate),
        require_no_quality_claim=bool(args.require_no_quality_claim),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "stage_b_duration_coverage_fill_local_audio_render_attempt.json"
    markdown_path = output_dir / "stage_b_duration_coverage_fill_local_audio_render_attempt.md"
    write_json(report_path, report)
    write_json(output_dir / "stage_b_duration_coverage_fill_local_audio_render_attempt_validation_summary.json", summary)
    markdown_path.write_text(markdown_report(report), encoding="utf-8")
    print(json.dumps({**summary, "report_path": str(report_path), "markdown_path": str(markdown_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
