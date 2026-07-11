"""Build audio review package for controlled checkpoint temperature-guard candidates."""

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
from scripts.consolidate_stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repeatability_temperature_guard_repair import (  # noqa: E402
    BOUNDARY as SOURCE_BOUNDARY,
)
from scripts.render_stage_b_midi_to_solo_candidate_audio import (  # noqa: E402
    default_soundfont_candidates,
    resolve_soundfont,
    sha256_file,
    wav_meta,
)


class StageBMidiToSoloControlledScaleCheckpointTemperatureGuardAudioPackageError(
    ValueError
):
    pass


BOUNDARY = (
    "stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repeatability_"
    "temperature_guard_audio_review_package"
)
NEXT_BOUNDARY = (
    "stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repeatability_"
    "temperature_guard_listening_review"
)
SCHEMA_VERSION = (
    "stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repeatability_"
    "temperature_guard_audio_review_package_v1"
)
CommandRunner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _bool_token(value: Any) -> str:
    return "true" if bool(value) else "false"


def default_runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(list(command), check=False, text=True, capture_output=True)


def validate_consolidation_report(report: dict[str, Any], expected_seed_count: int) -> dict[str, Any]:
    evidence = _dict(report.get("evidence_summary"))
    result = _dict(report.get("consolidation_result"))
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    if str(report.get("boundary") or "") != SOURCE_BOUNDARY:
        raise StageBMidiToSoloControlledScaleCheckpointTemperatureGuardAudioPackageError(
            "temperature guard repair consolidation boundary required"
        )
    if str(decision.get("next_boundary") or "") != BOUNDARY:
        raise StageBMidiToSoloControlledScaleCheckpointTemperatureGuardAudioPackageError(
            "temperature guard consolidation must route to audio review package"
        )
    if not bool(result.get("objective_temperature_guard_support", False)):
        raise StageBMidiToSoloControlledScaleCheckpointTemperatureGuardAudioPackageError(
            "objective temperature guard support required"
        )
    if not bool(result.get("audio_review_package_required", False)):
        raise StageBMidiToSoloControlledScaleCheckpointTemperatureGuardAudioPackageError(
            "audio review package requirement required"
        )
    if _int(evidence.get("seed_count")) != int(expected_seed_count):
        raise StageBMidiToSoloControlledScaleCheckpointTemperatureGuardAudioPackageError(
            "unexpected seed count"
        )
    blocked = [
        "human_audio_preference_claimed",
        "midi_to_solo_musical_quality_claimed",
        "broad_trained_model_quality_claimed",
        "brad_style_adaptation_claimed",
    ]
    claimed = [name for name in blocked if bool(readiness.get(name, False))]
    if claimed:
        raise StageBMidiToSoloControlledScaleCheckpointTemperatureGuardAudioPackageError(
            f"unexpected upstream claim: {claimed}"
        )
    report_paths = [Path(str(path)) for path in _list(evidence.get("generation_report_paths"))]
    if len(report_paths) != int(expected_seed_count):
        raise StageBMidiToSoloControlledScaleCheckpointTemperatureGuardAudioPackageError(
            "generation report path count mismatch"
        )
    for path in report_paths:
        if not path.exists():
            raise StageBMidiToSoloControlledScaleCheckpointTemperatureGuardAudioPackageError(
                f"generation report missing: {path}"
            )
    return {
        "seed_count": _int(evidence.get("seed_count")),
        "sample_count": _int(evidence.get("sample_count")),
        "valid_sample_count": _int(evidence.get("valid_sample_count")),
        "strict_valid_sample_count": _int(evidence.get("strict_valid_sample_count")),
        "grammar_gate_sample_count": _int(evidence.get("grammar_gate_sample_count")),
        "source_temperature": _float(evidence.get("source_temperature")),
        "temperature": _float(evidence.get("temperature")),
        "top_k": _int(evidence.get("top_k")),
        "generation_report_paths": [str(path) for path in report_paths],
    }


def select_seed_candidate(report_path: Path) -> dict[str, Any]:
    report = read_json(report_path)
    samples = [_dict(sample) for sample in _list(report.get("samples"))]
    eligible = [
        sample
        for sample in samples
        if bool(sample.get("strict_valid", False))
        and bool(sample.get("grammar_gate_passed", False))
        and Path(str(sample.get("midi_path") or "")).exists()
    ]
    if not eligible:
        raise StageBMidiToSoloControlledScaleCheckpointTemperatureGuardAudioPackageError(
            f"no strict-valid sample in {report_path}"
        )
    return min(
        eligible,
        key=lambda sample: (
            _float(_dict(sample.get("metrics")).get("dead_air_ratio")),
            _float(_dict(sample.get("collapse")).get("postprocess_removal_ratio")),
            -_int(_dict(sample.get("metrics")).get("unique_pitch_count")),
            _int(sample.get("sample_index")),
        ),
    )


def candidate_rows(consolidation_summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rank, report_path_str in enumerate(consolidation_summary["generation_report_paths"], start=1):
        report_path = Path(str(report_path_str))
        sample = select_seed_candidate(report_path)
        midi_path = Path(str(sample.get("midi_path") or ""))
        metrics = _dict(sample.get("metrics"))
        collapse = _dict(sample.get("collapse"))
        rows.append(
            {
                "rank": rank,
                "seed": _int(sample.get("sample_seed")),
                "sample_index": _int(sample.get("sample_index")),
                "generation_report_path": str(report_path),
                "midi_path": str(midi_path),
                "midi_sha256": sha256_file(midi_path),
                "note_count": _int(metrics.get("note_count")),
                "unique_pitch_count": _int(metrics.get("unique_pitch_count")),
                "dead_air_ratio": _float(metrics.get("dead_air_ratio")),
                "postprocess_removal_ratio": _float(collapse.get("postprocess_removal_ratio")),
                "duration_sec": _float(metrics.get("duration_sec")),
            }
        )
    return rows


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
        raise StageBMidiToSoloControlledScaleCheckpointTemperatureGuardAudioPackageError(
            f"renderer not found: {renderer}"
        )
    if not soundfont.exists():
        raise StageBMidiToSoloControlledScaleCheckpointTemperatureGuardAudioPackageError(
            f"soundfont not found: {soundfont}"
        )
    plan: list[dict[str, Any]] = []
    for item in candidates:
        wav_path = (
            output_dir
            / "audio"
            / f"temperature_guard_rank_{int(item['rank']):02d}_seed_{int(item['seed'])}.wav"
        )
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
            raise StageBMidiToSoloControlledScaleCheckpointTemperatureGuardAudioPackageError(
                f"render failed for rank {item['rank']}: {completed.stderr or completed.stdout}"
            )
        rendered.append(
            {
                "rank": item["rank"],
                "seed": item["seed"],
                "sample_index": item["sample_index"],
                "midi_path": item["midi_path"],
                "midi_sha256": item["midi_sha256"],
                "note_count": item["note_count"],
                "unique_pitch_count": item["unique_pitch_count"],
                "dead_air_ratio": item["dead_air_ratio"],
                "postprocess_removal_ratio": item["postprocess_removal_ratio"],
                "wav_file": wav_meta(wav_path),
                "command": list(item["command"]),
                "stdout_tail": (completed.stdout or "")[-1000:],
                "stderr_tail": (completed.stderr or "")[-1000:],
            }
        )
    return rendered


def build_audio_package_report(
    consolidation_report: dict[str, Any],
    *,
    output_dir: Path,
    renderer_path: str,
    soundfont_path: str,
    sample_rate: int,
    expected_file_count: int,
    runner: CommandRunner = default_runner,
) -> dict[str, Any]:
    consolidation = validate_consolidation_report(
        consolidation_report,
        expected_seed_count=int(expected_file_count),
    )
    candidates = candidate_rows(consolidation)
    resolved_renderer = renderer_path or shutil.which("fluidsynth") or ""
    resolved_soundfont = resolve_soundfont(soundfont_path)
    plan = build_render_plan(
        candidates,
        output_dir=output_dir,
        renderer_path=resolved_renderer,
        soundfont_path=resolved_soundfont,
        sample_rate=int(sample_rate),
    )
    rendered = execute_render_plan(plan, runner=runner)
    durations = [_float(_dict(item.get("wav_file")).get("duration_seconds")) for item in rendered]
    boundary = {
        "boundary": BOUNDARY,
        "source_boundary": SOURCE_BOUNDARY,
        "candidate_count": len(candidates),
        "rendered_audio_file_count": len(rendered),
        "technical_wav_validation": True,
        "audio_output_claimed": True,
        "listening_review_completed": False,
        "audio_rendered_quality_claimed": False,
        "human_audio_preference_claimed": False,
        "midi_to_solo_musical_quality_claimed": False,
        "broad_trained_model_quality_claimed": False,
        "brad_style_adaptation_claimed": False,
        "duration_seconds_min": min(durations) if durations else 0.0,
        "duration_seconds_max": max(durations) if durations else 0.0,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
            "+00:00", "Z"
        ),
        "output_dir": str(output_dir),
        "source_boundary": SOURCE_BOUNDARY,
        "input_summary": consolidation,
        "renderer": {
            "name": "fluidsynth",
            "path": resolved_renderer,
        },
        "soundfont": {
            "path": str(Path(resolved_soundfont).expanduser()),
            "sha256": sha256_file(Path(resolved_soundfont).expanduser()),
            "default_candidates_checked": [str(path) for path in default_soundfont_candidates()],
        },
        "candidate_selection": candidates,
        "rendered_audio_files": rendered,
        "audio_package_boundary": boundary,
        "decision": {
            "current_boundary": BOUNDARY,
            "next_boundary": NEXT_BOUNDARY,
            "auto_progress_allowed": True,
            "critical_user_input_required": False,
            "reason": "objective-supported temperature guard MIDI candidates rendered to WAV for listening review",
        },
        "not_proven": [
            "listening_review_completed",
            "audio_rendered_quality",
            "human_audio_preference",
            "midi_to_solo_musical_quality",
            "broad_trained_model_quality",
            "brad_style_adaptation",
        ],
        "next_recommended_issue": (
            "Stage B MIDI-to-solo controlled checkpoint dead-air temperature guard listening review"
        ),
    }


def validate_audio_package_report(
    report: dict[str, Any],
    *,
    expected_boundary: str | None,
    expected_file_count: int,
    expected_sample_rate: int,
    require_no_quality_claim: bool,
) -> dict[str, Any]:
    boundary = _dict(report.get("audio_package_boundary"))
    decision = _dict(report.get("decision"))
    files = _list(report.get("rendered_audio_files"))
    if expected_boundary and str(boundary.get("boundary") or "") != expected_boundary:
        raise StageBMidiToSoloControlledScaleCheckpointTemperatureGuardAudioPackageError(
            "unexpected boundary"
        )
    if len(files) != int(expected_file_count):
        raise StageBMidiToSoloControlledScaleCheckpointTemperatureGuardAudioPackageError(
            "unexpected rendered file count"
        )
    for item in files:
        wav_file = _dict(_dict(item).get("wav_file"))
        if not bool(wav_file.get("exists", False)):
            raise StageBMidiToSoloControlledScaleCheckpointTemperatureGuardAudioPackageError(
                "missing rendered WAV"
            )
        if _int(wav_file.get("sample_rate")) != int(expected_sample_rate):
            raise StageBMidiToSoloControlledScaleCheckpointTemperatureGuardAudioPackageError(
                "unexpected sample rate"
            )
        if _int(wav_file.get("frame_count")) <= 0:
            raise StageBMidiToSoloControlledScaleCheckpointTemperatureGuardAudioPackageError(
                "empty WAV"
            )
        if _int(wav_file.get("size_bytes")) <= 44:
            raise StageBMidiToSoloControlledScaleCheckpointTemperatureGuardAudioPackageError(
                "invalid WAV size"
            )
    if require_no_quality_claim:
        blocked = [
            "listening_review_completed",
            "audio_rendered_quality_claimed",
            "human_audio_preference_claimed",
            "midi_to_solo_musical_quality_claimed",
            "broad_trained_model_quality_claimed",
            "brad_style_adaptation_claimed",
        ]
        claimed = [name for name in blocked if bool(boundary.get(name, True))]
        if claimed:
            raise StageBMidiToSoloControlledScaleCheckpointTemperatureGuardAudioPackageError(
                f"unexpected quality claim: {claimed}"
            )
    return {
        "boundary": str(boundary.get("boundary") or ""),
        "source_boundary": str(boundary.get("source_boundary") or ""),
        "candidate_count": _int(boundary.get("candidate_count")),
        "rendered_audio_file_count": _int(boundary.get("rendered_audio_file_count")),
        "technical_wav_validation": bool(boundary.get("technical_wav_validation", False)),
        "listening_review_completed": bool(boundary.get("listening_review_completed", True)),
        "human_audio_preference_claimed": bool(boundary.get("human_audio_preference_claimed", True)),
        "midi_to_solo_musical_quality_claimed": bool(
            boundary.get("midi_to_solo_musical_quality_claimed", True)
        ),
        "duration_seconds_min": _float(boundary.get("duration_seconds_min")),
        "duration_seconds_max": _float(boundary.get("duration_seconds_max")),
        "critical_user_input_required": bool(decision.get("critical_user_input_required", True)),
        "wav_paths": [str(_dict(_dict(item).get("wav_file")).get("path") or "") for item in files],
        "next_boundary": str(decision.get("next_boundary") or ""),
        "next_recommended_issue": str(report.get("next_recommended_issue") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    boundary = report["audio_package_boundary"]
    decision = report["decision"]
    lines = [
        "# Stage B MIDI-to-Solo Controlled Scale Checkpoint Dead-Air Repeatability Temperature Guard Audio Review Package",
        "",
        "## Summary",
        "",
        f"- boundary: `{boundary['boundary']}`",
        f"- source boundary: `{boundary['source_boundary']}`",
        f"- next boundary: `{decision['next_boundary']}`",
        f"- candidate count: `{boundary['candidate_count']}`",
        f"- rendered audio file count: `{boundary['rendered_audio_file_count']}`",
        f"- technical WAV validation: `{_bool_token(boundary['technical_wav_validation'])}`",
        f"- duration range: `{float(boundary['duration_seconds_min']):.3f}s-{float(boundary['duration_seconds_max']):.3f}s`",
        f"- listening review completed: `{_bool_token(boundary['listening_review_completed'])}`",
        f"- human/audio preference claimed: `{_bool_token(boundary['human_audio_preference_claimed'])}`",
        f"- MIDI-to-solo musical quality claimed: `{_bool_token(boundary['midi_to_solo_musical_quality_claimed'])}`",
        "",
        "## Rendered Files",
        "",
        "| rank | seed | sample | MIDI path | WAV path | duration | sample rate | size | sha256 |",
        "|---:|---:|---:|---|---|---:|---:|---:|---|",
    ]
    for item in report.get("rendered_audio_files", []):
        wav_file = item["wav_file"]
        lines.append(
            "| "
            + " | ".join(
                [
                    str(item["rank"]),
                    str(item["seed"]),
                    str(item["sample_index"]),
                    str(item["midi_path"]),
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
    parser = argparse.ArgumentParser(
        description="Build controlled temperature guard audio review package"
    )
    parser.add_argument("--consolidation_report", type=str, required=True)
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repeatability_temperature_guard_audio_review_package",
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
    report = build_audio_package_report(
        read_json(Path(args.consolidation_report)),
        output_dir=output_dir,
        renderer_path=str(args.renderer or ""),
        soundfont_path=str(args.soundfont or ""),
        sample_rate=int(args.sample_rate),
        expected_file_count=int(args.expected_file_count),
    )
    summary = validate_audio_package_report(
        report,
        expected_boundary=str(args.expected_boundary or ""),
        expected_file_count=int(args.expected_file_count),
        expected_sample_rate=int(args.sample_rate),
        require_no_quality_claim=bool(args.require_no_quality_claim),
    )
    write_json(
        output_dir
        / "stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repeatability_temperature_guard_audio_review_package.json",
        report,
    )
    write_json(
        output_dir
        / "stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repeatability_temperature_guard_audio_review_package_validation_summary.json",
        summary,
    )
    markdown = markdown_report(report)
    write_text(
        output_dir
        / "stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repeatability_temperature_guard_audio_review_package.md",
        markdown,
    )
    if args.doc_path:
        write_text(Path(args.doc_path), markdown)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
