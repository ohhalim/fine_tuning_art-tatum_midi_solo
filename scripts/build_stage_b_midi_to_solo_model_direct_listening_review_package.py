"""Build a listening review package for timing-repaired model-direct MIDI candidates."""

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

from scripts.assess_stage_b_generic_base_readiness import write_json, write_text  # noqa: E402
from scripts.render_stage_b_midi_to_solo_candidate_audio import (  # noqa: E402
    default_soundfont_candidates,
    resolve_soundfont,
    sha256_file,
    wav_meta,
)
from scripts.run_stage_b_midi_to_solo_model_direct_timing_phrase_repair import (  # noqa: E402
    BOUNDARY as SOURCE_BOUNDARY,
)


class StageBMidiToSoloModelDirectListeningReviewPackageError(ValueError):
    pass


BOUNDARY = "stage_b_midi_to_solo_model_direct_listening_review_package"
NEXT_BOUNDARY = "stage_b_midi_to_solo_model_direct_user_listening_review_fill"
SCHEMA_VERSION = "stage_b_midi_to_solo_model_direct_listening_review_package_v1"
CommandRunner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise StageBMidiToSoloModelDirectListeningReviewPackageError(f"report missing: {path}")
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


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _bool_token(value: Any) -> str:
    return "true" if bool(value) else "false"


def default_runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(list(command), check=False, text=True, capture_output=True)


def validate_source_timing_repair(report: dict[str, Any], expected_count: int) -> list[dict[str, Any]]:
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    generation = _dict(report.get("generation_summary"))
    diagnostics = _dict(report.get("repaired_diagnostics_summary"))
    if str(report.get("boundary") or readiness.get("boundary") or "") != SOURCE_BOUNDARY:
        raise StageBMidiToSoloModelDirectListeningReviewPackageError("timing phrase repair boundary required")
    if str(decision.get("next_boundary") or "") != BOUNDARY:
        raise StageBMidiToSoloModelDirectListeningReviewPackageError("timing repair must route to review package")
    if not bool(readiness.get("timing_phrase_repair_passed", False)):
        raise StageBMidiToSoloModelDirectListeningReviewPackageError("timing phrase repair pass required")
    if not bool(generation.get("all_midi_paths_exist", False)):
        raise StageBMidiToSoloModelDirectListeningReviewPackageError("generated MIDI paths must exist")
    if bool(readiness.get("model_direct_generation_quality_claimed", True)):
        raise StageBMidiToSoloModelDirectListeningReviewPackageError("model-direct quality must not be claimed")
    if bool(readiness.get("human_audio_preference_claimed", True)):
        raise StageBMidiToSoloModelDirectListeningReviewPackageError("human/audio preference must not be claimed")
    paths = [str(path) for path in _list(generation.get("midi_paths"))[: int(expected_count)]]
    if len(paths) < int(expected_count):
        raise StageBMidiToSoloModelDirectListeningReviewPackageError("not enough timing-repaired MIDI paths")
    diagnostic_by_rank = {
        _int(_dict(item).get("rank")): _dict(item)
        for item in _list(diagnostics.get("candidate_diagnostics"))
        if _int(_dict(item).get("rank")) > 0
    }
    candidates: list[dict[str, Any]] = []
    for index, midi_path_text in enumerate(paths, start=1):
        midi_path = Path(midi_path_text)
        if not midi_path.exists():
            raise StageBMidiToSoloModelDirectListeningReviewPackageError(f"MIDI not found: {midi_path}")
        diagnostic = diagnostic_by_rank.get(index, {})
        candidates.append(
            {
                "rank": index,
                "sample_index": index,
                "source_midi_path": str(midi_path),
                "source_note_count": _int(diagnostic.get("note_count")),
                "source_unique_pitch_count": _int(diagnostic.get("unique_pitch_count")),
                "source_max_interval": _int(diagnostic.get("max_interval")),
                "source_dead_air_ratio": _float(diagnostic.get("dead_air_ratio")),
                "source_diagnostic_flags": _list(diagnostic.get("diagnostic_flags")),
            }
        )
    return candidates


def copy_midi_candidates(candidates: list[dict[str, Any]], *, output_dir: Path) -> list[dict[str, Any]]:
    copied: list[dict[str, Any]] = []
    midi_dir = output_dir / "midi"
    midi_dir.mkdir(parents=True, exist_ok=True)
    for item in candidates:
        target_path = midi_dir / f"timing_repair_rank_{int(item['rank']):02d}.mid"
        shutil.copy2(str(item["source_midi_path"]), target_path)
        copied.append(
            {
                **item,
                "package_midi_path": str(target_path),
                "package_midi_sha256": sha256_file(target_path),
            }
        )
    return copied


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
        raise StageBMidiToSoloModelDirectListeningReviewPackageError(f"renderer not found: {renderer}")
    if not soundfont.exists():
        raise StageBMidiToSoloModelDirectListeningReviewPackageError(f"soundfont not found: {soundfont}")
    plan: list[dict[str, Any]] = []
    for item in candidates:
        wav_path = output_dir / "audio" / f"timing_repair_rank_{int(item['rank']):02d}.wav"
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
                    str(item["package_midi_path"]),
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
            raise StageBMidiToSoloModelDirectListeningReviewPackageError(
                f"render failed for rank {item['rank']}: {completed.stderr or completed.stdout}"
            )
        rendered.append(
            {
                "rank": item["rank"],
                "sample_index": item["sample_index"],
                "source_midi_path": item["source_midi_path"],
                "package_midi_path": item["package_midi_path"],
                "package_midi_sha256": item["package_midi_sha256"],
                "source_note_count": item["source_note_count"],
                "source_unique_pitch_count": item["source_unique_pitch_count"],
                "source_max_interval": item["source_max_interval"],
                "source_dead_air_ratio": item["source_dead_air_ratio"],
                "source_diagnostic_flags": item["source_diagnostic_flags"],
                "wav_file": wav_meta(wav_path),
                "command": list(item["command"]),
                "stdout_tail": (completed.stdout or "")[-1000:],
                "stderr_tail": (completed.stderr or "")[-1000:],
            }
        )
    return rendered


def listening_review_template(rendered: list[dict[str, Any]]) -> str:
    lines = [
        "# Stage B MIDI-to-Solo Model-Direct Listening Review Input",
        "",
        "## Review Status",
        "",
        "- reviewer: `pending`",
        "- reviewed_at: `pending`",
        "- preferred_rank: `pending`",
        "- reject_all: `pending`",
        "- broad_model_quality_claim_allowed: `false`",
        "",
        "## Candidates",
        "",
        "| rank | midi | wav | note count | unique pitch | max interval | dead-air ratio | decision |",
        "|---:|---|---|---:|---:|---:|---:|---|",
    ]
    for item in rendered:
        wav_file = _dict(item.get("wav_file"))
        lines.append(
            "| "
            + " | ".join(
                [
                    str(item["rank"]),
                    str(item["package_midi_path"]),
                    str(wav_file.get("path") or ""),
                    str(item["source_note_count"]),
                    str(item["source_unique_pitch_count"]),
                    str(item["source_max_interval"]),
                    f"{float(item['source_dead_air_ratio']):.4f}",
                    "`pending`",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Per-Candidate Notes",
            "",
        ]
    )
    for item in rendered:
        lines.extend(
            [
                f"### Rank {int(item['rank'])}",
                "",
                "- musical_acceptance: `pending`",
                "- issue_tags: `pending`",
                "- short_note: `pending`",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def build_listening_review_package_report(
    source_report: dict[str, Any],
    *,
    output_dir: Path,
    renderer_path: str,
    soundfont_path: str,
    sample_rate: int,
    expected_file_count: int,
    runner: CommandRunner = default_runner,
) -> dict[str, Any]:
    candidates = validate_source_timing_repair(source_report, expected_count=expected_file_count)
    copied = copy_midi_candidates(candidates, output_dir=output_dir)
    resolved_renderer = renderer_path or shutil.which("fluidsynth") or ""
    resolved_soundfont = resolve_soundfont(soundfont_path)
    plan = build_render_plan(
        copied,
        output_dir=output_dir,
        renderer_path=resolved_renderer,
        soundfont_path=resolved_soundfont,
        sample_rate=int(sample_rate),
    )
    rendered = execute_render_plan(plan, runner=runner)
    review_input_path = output_dir / "review" / "listening_review_input.md"
    write_text(review_input_path, listening_review_template(rendered))
    boundary = {
        "boundary": BOUNDARY,
        "source_boundary": SOURCE_BOUNDARY,
        "candidate_count": int(len(copied)),
        "midi_file_count": int(len(copied)),
        "rendered_audio_file_count": int(len(rendered)),
        "technical_wav_validation": True,
        "review_input_template_written": True,
        "audio_output_claimed": True,
        "listening_review_completed": False,
        "audio_rendered_quality_claimed": False,
        "human_audio_preference_claimed": False,
        "model_direct_generation_quality_claimed": False,
        "midi_to_solo_musical_quality_claimed": False,
        "broad_trained_model_quality_claimed": False,
        "brad_style_adaptation_claimed": False,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "source_boundary": SOURCE_BOUNDARY,
        "source_timing_repair_result": _dict(source_report.get("repair_result")),
        "renderer": {
            "name": "fluidsynth",
            "path": resolved_renderer,
        },
        "soundfont": {
            "path": str(Path(resolved_soundfont).expanduser()),
            "sha256": sha256_file(Path(resolved_soundfont).expanduser()),
            "default_candidates_checked": [str(path) for path in default_soundfont_candidates()],
        },
        "packaged_candidates": copied,
        "rendered_audio_files": rendered,
        "review_input_template_path": str(review_input_path),
        "listening_review_package_boundary": boundary,
        "decision": {
            "current_boundary": BOUNDARY,
            "next_boundary": NEXT_BOUNDARY,
            "auto_progress_allowed": True,
            "critical_user_input_required": False,
            "reason": "timing-repaired MIDI candidates packaged with WAV files and pending review input",
        },
        "not_proven": [
            "listening_review_completed",
            "audio_rendered_quality",
            "human_audio_preference",
            "model_direct_generation_quality",
            "midi_to_solo_musical_quality",
            "broad_trained_model_quality",
            "brad_style_adaptation",
        ],
        "next_recommended_issue": "Stage B MIDI-to-solo model-direct user listening review fill",
    }


def validate_listening_review_package_report(
    report: dict[str, Any],
    *,
    expected_boundary: str | None,
    expected_file_count: int,
    expected_sample_rate: int,
    require_no_quality_claim: bool,
) -> dict[str, Any]:
    boundary = _dict(report.get("listening_review_package_boundary"))
    if expected_boundary and str(boundary.get("boundary") or "") != expected_boundary:
        raise StageBMidiToSoloModelDirectListeningReviewPackageError(
            f"expected boundary {expected_boundary}, got {boundary.get('boundary')}"
        )
    files = _list(report.get("rendered_audio_files"))
    candidates = _list(report.get("packaged_candidates"))
    if len(candidates) != int(expected_file_count):
        raise StageBMidiToSoloModelDirectListeningReviewPackageError("unexpected candidate count")
    if len(files) != int(expected_file_count):
        raise StageBMidiToSoloModelDirectListeningReviewPackageError("unexpected rendered file count")
    if not Path(str(report.get("review_input_template_path") or "")).exists():
        raise StageBMidiToSoloModelDirectListeningReviewPackageError("review input template missing")
    for item in files:
        wav_file = _dict(_dict(item).get("wav_file"))
        if not bool(wav_file.get("exists", False)):
            raise StageBMidiToSoloModelDirectListeningReviewPackageError("missing rendered WAV")
        if _int(wav_file.get("sample_rate")) != int(expected_sample_rate):
            raise StageBMidiToSoloModelDirectListeningReviewPackageError("unexpected sample rate")
        if _int(wav_file.get("frame_count")) <= 0:
            raise StageBMidiToSoloModelDirectListeningReviewPackageError("empty WAV")
        if _int(wav_file.get("size_bytes")) <= 44:
            raise StageBMidiToSoloModelDirectListeningReviewPackageError("invalid WAV size")
    if require_no_quality_claim:
        blocked = [
            "listening_review_completed",
            "audio_rendered_quality_claimed",
            "human_audio_preference_claimed",
            "model_direct_generation_quality_claimed",
            "midi_to_solo_musical_quality_claimed",
            "broad_trained_model_quality_claimed",
            "brad_style_adaptation_claimed",
        ]
        claimed = [name for name in blocked if bool(boundary.get(name, True))]
        if claimed:
            raise StageBMidiToSoloModelDirectListeningReviewPackageError(f"unexpected quality claim: {claimed}")
    decision = _dict(report.get("decision"))
    return {
        "boundary": str(boundary.get("boundary") or ""),
        "source_boundary": str(boundary.get("source_boundary") or ""),
        "candidate_count": _int(boundary.get("candidate_count")),
        "midi_file_count": _int(boundary.get("midi_file_count")),
        "rendered_audio_file_count": _int(boundary.get("rendered_audio_file_count")),
        "technical_wav_validation": bool(boundary.get("technical_wav_validation", False)),
        "review_input_template_written": bool(boundary.get("review_input_template_written", False)),
        "listening_review_completed": bool(boundary.get("listening_review_completed", True)),
        "human_audio_preference_claimed": bool(boundary.get("human_audio_preference_claimed", True)),
        "model_direct_generation_quality_claimed": bool(
            boundary.get("model_direct_generation_quality_claimed", True)
        ),
        "critical_user_input_required": bool(decision.get("critical_user_input_required", True)),
        "wav_paths": [str(_dict(_dict(item).get("wav_file")).get("path") or "") for item in files],
        "review_input_template_path": str(report.get("review_input_template_path") or ""),
        "next_boundary": str(decision.get("next_boundary") or ""),
        "next_recommended_issue": str(report.get("next_recommended_issue") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    boundary = report["listening_review_package_boundary"]
    decision = report["decision"]
    lines = [
        "# Stage B MIDI-to-Solo Model-Direct Listening Review Package",
        "",
        "## Summary",
        "",
        f"- boundary: `{boundary['boundary']}`",
        f"- source boundary: `{boundary['source_boundary']}`",
        f"- next boundary: `{decision['next_boundary']}`",
        f"- candidate count: `{boundary['candidate_count']}`",
        f"- rendered audio file count: `{boundary['rendered_audio_file_count']}`",
        f"- technical WAV validation: `{_bool_token(boundary['technical_wav_validation'])}`",
        f"- review input template written: `{_bool_token(boundary['review_input_template_written'])}`",
        f"- listening review completed: `{_bool_token(boundary['listening_review_completed'])}`",
        f"- human/audio preference claimed: `{_bool_token(boundary['human_audio_preference_claimed'])}`",
        f"- model-direct generation quality claimed: `{_bool_token(boundary['model_direct_generation_quality_claimed'])}`",
        "",
        "## Rendered Files",
        "",
        "| rank | midi path | wav path | duration | sample rate | size | sha256 |",
        "|---:|---|---|---:|---:|---:|---|",
    ]
    for item in report.get("rendered_audio_files", []):
        wav_file = item["wav_file"]
        lines.append(
            "| "
            + " | ".join(
                [
                    str(item["rank"]),
                    str(item["package_midi_path"]),
                    str(wav_file["path"]),
                    f"{float(wav_file['duration_seconds']):.3f}",
                    str(wav_file["sample_rate"]),
                    str(wav_file["size_bytes"]),
                    str(wav_file["sha256"][:12]),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Review Input", "", f"- template: `{report['review_input_template_path']}`", ""])
    lines.extend(["## Not Proven", ""])
    for item in report.get("not_proven", []):
        lines.append(f"- `{item}`")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build model-direct listening review package")
    parser.add_argument("--timing_phrase_repair", type=str, required=True)
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/stage_b_midi_to_solo_model_direct_listening_review_package",
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
    output_dir.mkdir(parents=True, exist_ok=True)
    report = build_listening_review_package_report(
        read_json(Path(args.timing_phrase_repair)),
        output_dir=output_dir,
        renderer_path=str(args.renderer or ""),
        soundfont_path=str(args.soundfont or ""),
        sample_rate=int(args.sample_rate),
        expected_file_count=int(args.expected_file_count),
    )
    summary = validate_listening_review_package_report(
        report,
        expected_boundary=str(args.expected_boundary or ""),
        expected_file_count=int(args.expected_file_count),
        expected_sample_rate=int(args.sample_rate),
        require_no_quality_claim=bool(args.require_no_quality_claim),
    )
    write_json(output_dir / "stage_b_midi_to_solo_model_direct_listening_review_package.json", report)
    write_json(
        output_dir / "stage_b_midi_to_solo_model_direct_listening_review_package_validation_summary.json",
        summary,
    )
    markdown = markdown_report(report)
    write_text(output_dir / "stage_b_midi_to_solo_model_direct_listening_review_package.md", markdown)
    if args.doc_path:
        write_text(Path(args.doc_path), markdown)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
