"""Render Stage B MIDI-to-solo model-direct candidates to WAV files."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from scripts.assess_stage_b_generic_base_readiness import read_json, write_json, write_text  # noqa: E402
from scripts.render_stage_b_midi_to_solo_candidate_audio import (  # noqa: E402
    default_soundfont_candidates,
    resolve_soundfont,
    sha256_file,
    wav_meta,
)


class StageBMidiToSoloModelDirectAudioRenderError(ValueError):
    pass


SOURCE_BOUNDARY = "stage_b_midi_to_solo_model_direct_monophonic_overlap_repair"
BOUNDARY = "stage_b_midi_to_solo_model_direct_audio_render_package"
NEXT_BOUNDARY = "stage_b_midi_to_solo_model_direct_audio_evidence_consolidation"
SCHEMA_VERSION = "stage_b_midi_to_solo_model_direct_audio_render_package_v1"
CommandRunner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]


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


def default_runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(list(command), check=False, text=True, capture_output=True)


def validate_source_report(report: dict[str, Any], expected_count: int) -> list[dict[str, Any]]:
    readiness = _dict(report.get("readiness"))
    generation = _dict(report.get("repaired_generation_summary"))
    if str(report.get("boundary") or readiness.get("boundary") or "") != SOURCE_BOUNDARY:
        raise StageBMidiToSoloModelDirectAudioRenderError("model-direct overlap repair boundary required")
    if not bool(readiness.get("direct_generation_review_gate_passed", False)):
        raise StageBMidiToSoloModelDirectAudioRenderError("direct generation review gate must pass")
    if bool(readiness.get("model_direct_generation_quality_claimed", True)):
        raise StageBMidiToSoloModelDirectAudioRenderError("model-direct quality must not be claimed")
    if bool(readiness.get("human_audio_preference_claimed", True)):
        raise StageBMidiToSoloModelDirectAudioRenderError("human/audio preference must not be claimed")
    if _int(generation.get("valid_sample_count")) < int(expected_count):
        raise StageBMidiToSoloModelDirectAudioRenderError("not enough valid direct MIDI samples")
    if _int(generation.get("strict_valid_sample_count")) < int(expected_count):
        raise StageBMidiToSoloModelDirectAudioRenderError("not enough strict-valid direct MIDI samples")
    paths = [str(path) for path in _list(generation.get("midi_paths"))[: int(expected_count)]]
    if len(paths) < int(expected_count):
        raise StageBMidiToSoloModelDirectAudioRenderError("not enough MIDI paths")
    candidates: list[dict[str, Any]] = []
    for index, midi_path_text in enumerate(paths, start=1):
        midi_path = Path(midi_path_text)
        if not midi_path.exists():
            raise StageBMidiToSoloModelDirectAudioRenderError(f"direct MIDI not found: {midi_path}")
        candidates.append(
            {
                "rank": index,
                "sample_index": index,
                "midi_path": str(midi_path),
                "source_valid": True,
                "source_strict_valid": True,
                "source_note_count": _int(generation.get("min_postprocess_note_count")),
            }
        )
    return candidates


def build_render_plan(
    candidates: list[dict[str, Any]],
    *,
    output_dir: Path,
    renderer_path: str,
    soundfont_path: str,
    sample_rate: int,
) -> list[dict[str, Any]]:
    renderer = Path(renderer_path)
    soundfont = Path(soundfont_path).expanduser()
    if not renderer.exists():
        raise StageBMidiToSoloModelDirectAudioRenderError(f"renderer not found: {renderer}")
    if not soundfont.exists():
        raise StageBMidiToSoloModelDirectAudioRenderError(f"soundfont not found: {soundfont}")
    plan: list[dict[str, Any]] = []
    for item in candidates:
        wav_path = output_dir / "audio" / f"model_direct_sample_{int(item['sample_index']):02d}.wav"
        plan.append(
            {
                **item,
                "wav_path": str(wav_path),
                "command": [
                    str(renderer),
                    "-ni",
                    "-F",
                    str(wav_path),
                    "-r",
                    str(sample_rate),
                    str(soundfont),
                    str(item["midi_path"]),
                ],
            }
        )
    return plan


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
            raise StageBMidiToSoloModelDirectAudioRenderError(
                f"render failed for sample {item['sample_index']}: {completed.stderr or completed.stdout}"
            )
        rendered.append(
            {
                "rank": item["rank"],
                "sample_index": item["sample_index"],
                "source_midi_path": item["midi_path"],
                "source_valid": item["source_valid"],
                "source_strict_valid": item["source_strict_valid"],
                "source_note_count": item["source_note_count"],
                "wav_file": wav_meta(wav_path),
                "command": list(item["command"]),
                "stdout_tail": (completed.stdout or "")[-1000:],
                "stderr_tail": (completed.stderr or "")[-1000:],
            }
        )
    return rendered


def build_audio_render_report(
    source_report: dict[str, Any],
    *,
    output_dir: Path,
    renderer_path: str,
    soundfont_path: str,
    sample_rate: int,
    expected_file_count: int,
    runner: CommandRunner = default_runner,
) -> dict[str, Any]:
    candidates = validate_source_report(source_report, expected_count=expected_file_count)
    resolved_renderer = renderer_path or shutil.which("fluidsynth") or ""
    resolved_soundfont = resolve_soundfont(soundfont_path)
    plan = build_render_plan(
        candidates,
        output_dir=output_dir,
        renderer_path=resolved_renderer,
        soundfont_path=resolved_soundfont,
        sample_rate=sample_rate,
    )
    rendered = execute_render_plan(plan, runner=runner)
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "source_boundary": SOURCE_BOUNDARY,
        "source_generation_summary": _dict(source_report.get("repaired_generation_summary")),
        "renderer": {
            "name": "fluidsynth",
            "path": resolved_renderer,
        },
        "soundfont": {
            "path": str(Path(resolved_soundfont).expanduser()),
            "sha256": sha256_file(Path(resolved_soundfont).expanduser()),
            "default_candidates_checked": [str(path) for path in default_soundfont_candidates()],
        },
        "rendered_audio_files": rendered,
        "audio_render_boundary": {
            "boundary": BOUNDARY,
            "render_attempted": True,
            "rendered_audio_file_count": int(len(rendered)),
            "technical_wav_validation": True,
            "audio_output_claimed": True,
            "audio_rendered_quality_claimed": False,
            "human_audio_preference_claimed": False,
            "model_direct_generation_quality_claimed": False,
            "musical_quality_claimed": False,
            "midi_to_solo_mvp_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
        },
        "decision": {
            "current_boundary": BOUNDARY,
            "next_boundary": NEXT_BOUNDARY,
            "auto_progress_allowed": True,
            "critical_user_input_required": False,
            "reason": "strict-valid model-direct MIDI rendered to WAV; quality and preference remain unclaimed",
        },
        "not_proven": [
            "audio_rendered_quality",
            "human_audio_preference",
            "model_direct_generation_quality",
            "midi_to_solo_mvp_completion",
            "broad_trained_model_quality",
            "brad_style_adaptation",
        ],
        "next_recommended_issue": "Stage B MIDI-to-solo model-direct audio evidence consolidation",
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
        raise StageBMidiToSoloModelDirectAudioRenderError(
            f"expected boundary {expected_boundary}, got {boundary.get('boundary')}"
        )
    files = _list(report.get("rendered_audio_files"))
    if len(files) != int(expected_file_count):
        raise StageBMidiToSoloModelDirectAudioRenderError(f"expected {expected_file_count} rendered files")
    for item in files:
        wav_file = _dict(_dict(item).get("wav_file"))
        if not bool(wav_file.get("exists", False)):
            raise StageBMidiToSoloModelDirectAudioRenderError("missing rendered WAV")
        if _int(wav_file.get("sample_rate")) != int(expected_sample_rate):
            raise StageBMidiToSoloModelDirectAudioRenderError("unexpected sample rate")
        if _int(wav_file.get("frame_count")) <= 0:
            raise StageBMidiToSoloModelDirectAudioRenderError("empty WAV")
        if _int(wav_file.get("size_bytes")) <= 44:
            raise StageBMidiToSoloModelDirectAudioRenderError("invalid WAV size")
    if require_no_quality_claim:
        blocked = [
            "audio_rendered_quality_claimed",
            "human_audio_preference_claimed",
            "model_direct_generation_quality_claimed",
            "musical_quality_claimed",
            "midi_to_solo_mvp_claimed",
            "broad_trained_model_quality_claimed",
            "brad_style_adaptation_claimed",
        ]
        claimed = [name for name in blocked if bool(boundary.get(name, True))]
        if claimed:
            raise StageBMidiToSoloModelDirectAudioRenderError(f"unexpected quality claim: {claimed}")
    decision = _dict(report.get("decision"))
    return {
        "boundary": str(boundary.get("boundary") or ""),
        "source_boundary": str(report.get("source_boundary") or ""),
        "render_attempted": bool(boundary.get("render_attempted", False)),
        "rendered_audio_file_count": _int(boundary.get("rendered_audio_file_count")),
        "technical_wav_validation": bool(boundary.get("technical_wav_validation", False)),
        "audio_rendered_quality_claimed": bool(boundary.get("audio_rendered_quality_claimed", True)),
        "human_audio_preference_claimed": bool(boundary.get("human_audio_preference_claimed", True)),
        "model_direct_generation_quality_claimed": bool(
            boundary.get("model_direct_generation_quality_claimed", True)
        ),
        "critical_user_input_required": bool(decision.get("critical_user_input_required", True)),
        "wav_paths": [str(_dict(_dict(item).get("wav_file")).get("path") or "") for item in files],
        "next_boundary": str(decision.get("next_boundary") or ""),
        "next_recommended_issue": str(report.get("next_recommended_issue") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    boundary = report["audio_render_boundary"]
    decision = report["decision"]
    lines = [
        "# Stage B MIDI-to-Solo Model-Direct Audio Render Package",
        "",
        "## Summary",
        "",
        f"- boundary: `{boundary['boundary']}`",
        f"- source boundary: `{report['source_boundary']}`",
        f"- next boundary: `{decision['next_boundary']}`",
        f"- render attempted: `{_bool_token(boundary['render_attempted'])}`",
        f"- rendered audio file count: `{boundary['rendered_audio_file_count']}`",
        f"- technical WAV validation: `{_bool_token(boundary['technical_wav_validation'])}`",
        f"- audio rendered quality claimed: `{_bool_token(boundary['audio_rendered_quality_claimed'])}`",
        f"- human/audio preference claimed: `{_bool_token(boundary['human_audio_preference_claimed'])}`",
        f"- model-direct generation quality claimed: `{_bool_token(boundary['model_direct_generation_quality_claimed'])}`",
        "",
        "## Rendered Files",
        "",
        "| rank | sample | wav path | duration | sample rate | size | sha256 |",
        "|---:|---:|---|---:|---:|---:|---|",
    ]
    for item in report.get("rendered_audio_files", []):
        wav_file = item["wav_file"]
        lines.append(
            "| "
            + " | ".join(
                [
                    str(item["rank"]),
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
    parser = argparse.ArgumentParser(description="Render MIDI-to-solo model-direct WAV files")
    parser.add_argument("--model_direct_overlap_repair", type=str, required=True)
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/stage_b_midi_to_solo_model_direct_audio_render_package",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--renderer", type=str, default=shutil.which("fluidsynth") or "")
    parser.add_argument("--soundfont", type=str, default="")
    parser.add_argument("--sample_rate", type=int, default=44100)
    parser.add_argument("--expected_boundary", type=str, default="")
    parser.add_argument("--expected_file_count", type=int, default=3)
    parser.add_argument("--require_no_quality_claim", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    report = build_audio_render_report(
        read_json(Path(args.model_direct_overlap_repair)),
        output_dir=output_dir,
        renderer_path=str(args.renderer or ""),
        soundfont_path=str(args.soundfont or ""),
        sample_rate=int(args.sample_rate),
        expected_file_count=int(args.expected_file_count),
    )
    summary = validate_audio_render_report(
        report,
        expected_boundary=str(args.expected_boundary or ""),
        expected_file_count=int(args.expected_file_count),
        expected_sample_rate=int(args.sample_rate),
        require_no_quality_claim=bool(args.require_no_quality_claim),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "stage_b_midi_to_solo_model_direct_audio_render_package.json", report)
    write_json(output_dir / "stage_b_midi_to_solo_model_direct_audio_render_package_validation_summary.json", summary)
    markdown = markdown_report(report)
    write_text(output_dir / "stage_b_midi_to_solo_model_direct_audio_render_package.md", markdown)
    if args.doc_path:
        write_text(Path(args.doc_path), markdown)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
