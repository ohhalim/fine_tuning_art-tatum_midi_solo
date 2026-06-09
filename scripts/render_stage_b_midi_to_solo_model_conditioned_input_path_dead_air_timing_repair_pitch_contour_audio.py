"""Render pitch-contour repaired model-conditioned MIDI candidates to WAV."""

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
    resolve_soundfont,
    sha256_file,
    wav_meta,
)
from scripts.run_stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_pitch_contour_probe import (  # noqa: E402
    BOUNDARY as SOURCE_BOUNDARY,
    NEXT_BOUNDARY as SOURCE_NEXT_BOUNDARY,
)


class StageBMidiToSoloModelConditionedInputPathDeadAirTimingRepairPitchContourAudioError(
    ValueError
):
    pass


BOUNDARY = (
    "stage_b_midi_to_solo_model_conditioned_input_path_"
    "dead_air_timing_repair_pitch_contour_audio_package"
)
NEXT_BOUNDARY = (
    "stage_b_midi_to_solo_model_conditioned_input_path_"
    "dead_air_timing_repair_pitch_contour_listening_review_package"
)
SCHEMA_VERSION = (
    "stage_b_midi_to_solo_model_conditioned_input_path_"
    "dead_air_timing_repair_pitch_contour_audio_package_v1"
)
CommandRunner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]

QUALITY_CLAIM_KEYS = [
    "human_audio_preference_claimed",
    "midi_to_solo_musical_quality_claimed",
    "musical_quality_claimed",
    "audio_rendered_quality_claimed",
    "model_checkpoint_generation_quality_claimed",
    "model_direct_generation_quality_claimed",
    "broad_trained_model_quality_claimed",
    "brad_style_adaptation_claimed",
    "production_ready_claimed",
]


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


def _path_exists(path_text: str) -> bool:
    return bool(path_text and Path(path_text).exists())


def _require_no_quality_claim(container: dict[str, Any], *, label: str) -> None:
    claimed = [name for name in QUALITY_CLAIM_KEYS if bool(container.get(name, False))]
    if claimed:
        raise StageBMidiToSoloModelConditionedInputPathDeadAirTimingRepairPitchContourAudioError(
            f"unexpected quality claim in {label}: {claimed}"
        )


def validate_source_report(
    report: dict[str, Any],
    *,
    expected_count: int,
) -> list[dict[str, Any]]:
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    summary = _dict(report.get("summary"))
    if str(report.get("boundary") or readiness.get("boundary") or "") != SOURCE_BOUNDARY:
        raise StageBMidiToSoloModelConditionedInputPathDeadAirTimingRepairPitchContourAudioError(
            "pitch-contour repair probe boundary required"
        )
    if str(decision.get("next_boundary") or "") != SOURCE_NEXT_BOUNDARY:
        raise StageBMidiToSoloModelConditionedInputPathDeadAirTimingRepairPitchContourAudioError(
            "pitch-contour repair probe must route to audio package"
        )
    required_true = [
        "pitch_contour_repair_probe_completed",
        "repaired_ranked_midi_written",
        "pitch_contour_repair_passed",
        "pitch_contour_audio_render_required",
    ]
    missing = [name for name in required_true if not bool(readiness.get(name, False))]
    if missing:
        raise StageBMidiToSoloModelConditionedInputPathDeadAirTimingRepairPitchContourAudioError(
            f"missing pitch-contour probe readiness: {missing}"
        )
    if _int(summary.get("repaired_pass_count")) < int(expected_count):
        raise StageBMidiToSoloModelConditionedInputPathDeadAirTimingRepairPitchContourAudioError(
            "repaired pass count below expected count"
        )
    if _int(summary.get("repaired_max_interval")) > _int(summary.get("target_max_interval")):
        raise StageBMidiToSoloModelConditionedInputPathDeadAirTimingRepairPitchContourAudioError(
            "repaired max interval exceeds target"
        )
    if bool(decision.get("critical_user_input_required", True)):
        raise StageBMidiToSoloModelConditionedInputPathDeadAirTimingRepairPitchContourAudioError(
            "critical user input should not be required"
        )
    _require_no_quality_claim(readiness, label="pitch-contour probe readiness")
    rows = [_dict(item) for item in _list(report.get("candidate_repairs"))]
    if len(rows) < int(expected_count):
        raise StageBMidiToSoloModelConditionedInputPathDeadAirTimingRepairPitchContourAudioError(
            "candidate repair count below expected count"
        )
    compacted: list[dict[str, Any]] = []
    for row in rows[: int(expected_count)]:
        repaired_midi_path = str(row.get("repaired_midi_path") or "")
        if not _path_exists(repaired_midi_path):
            raise StageBMidiToSoloModelConditionedInputPathDeadAirTimingRepairPitchContourAudioError(
                f"pitch-contour repaired MIDI missing: {repaired_midi_path}"
            )
        if not bool(row.get("candidate_repair_passed", False)):
            raise StageBMidiToSoloModelConditionedInputPathDeadAirTimingRepairPitchContourAudioError(
                "candidate pitch-contour repair should pass before audio render"
            )
        source_metrics = _dict(row.get("source_metrics"))
        repaired_metrics = _dict(row.get("repaired_metrics"))
        pitch_stats = _dict(row.get("pitch_repair_stats"))
        compacted.append(
            {
                "rank": _int(row.get("rank")),
                "sample_index": _int(row.get("sample_index")),
                "sample_seed": _int(row.get("sample_seed")),
                "source_midi_path": str(row.get("source_midi_path") or ""),
                "repaired_midi_path": repaired_midi_path,
                "source_max_interval": _int(source_metrics.get("max_interval")),
                "repaired_max_interval": _int(repaired_metrics.get("max_interval")),
                "repaired_dead_air_ratio": _float(repaired_metrics.get("dead_air_ratio")),
                "repaired_note_count": _int(repaired_metrics.get("note_count")),
                "repaired_unique_pitch_count": _int(
                    repaired_metrics.get("unique_pitch_count")
                ),
                "repaired_max_simultaneous_notes": _int(
                    repaired_metrics.get("max_simultaneous_notes")
                ),
                "pitch_changed_ratio": _float(pitch_stats.get("pitch_changed_ratio")),
                "max_pitch_shift_abs": _int(pitch_stats.get("max_pitch_shift_abs")),
            }
        )
    return compacted


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
        raise StageBMidiToSoloModelConditionedInputPathDeadAirTimingRepairPitchContourAudioError(
            f"renderer not found: {renderer}"
        )
    if not soundfont.exists():
        raise StageBMidiToSoloModelConditionedInputPathDeadAirTimingRepairPitchContourAudioError(
            f"soundfont not found: {soundfont}"
        )
    plan: list[dict[str, Any]] = []
    for item in candidates:
        wav_path = output_dir / "audio" / (
            f"rank_{int(item['rank']):02d}_sample_{int(item['sample_index']):02d}_"
            "pitch_contour_repair.wav"
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
                    str(item["repaired_midi_path"]),
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
            raise StageBMidiToSoloModelConditionedInputPathDeadAirTimingRepairPitchContourAudioError(
                f"render failed for rank {item['rank']}: {completed.stderr or completed.stdout}"
            )
        rendered.append(
            {
                "rank": item["rank"],
                "sample_index": item["sample_index"],
                "sample_seed": item["sample_seed"],
                "source_midi_path": item["source_midi_path"],
                "repaired_midi_path": item["repaired_midi_path"],
                "source_max_interval": item["source_max_interval"],
                "repaired_max_interval": item["repaired_max_interval"],
                "repaired_dead_air_ratio": item["repaired_dead_air_ratio"],
                "repaired_note_count": item["repaired_note_count"],
                "repaired_unique_pitch_count": item["repaired_unique_pitch_count"],
                "repaired_max_simultaneous_notes": item["repaired_max_simultaneous_notes"],
                "pitch_changed_ratio": item["pitch_changed_ratio"],
                "max_pitch_shift_abs": item["max_pitch_shift_abs"],
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
        sample_rate=int(sample_rate),
    )
    rendered = execute_render_plan(plan, runner=runner)
    repaired_dead_air_values = [_float(item.get("repaired_dead_air_ratio")) for item in candidates]
    repaired_interval_values = [_int(item.get("repaired_max_interval")) for item in candidates]
    pitch_changed_ratios = [_float(item.get("pitch_changed_ratio")) for item in candidates]
    unique_pitch_values = [_int(item.get("repaired_unique_pitch_count")) for item in candidates]
    simultaneous_values = [_int(item.get("repaired_max_simultaneous_notes")) for item in candidates]
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
            "+00:00", "Z"
        ),
        "output_dir": str(output_dir),
        "source_boundary": SOURCE_BOUNDARY,
        "source_summary": _dict(source_report.get("summary")),
        "renderer": {
            "name": "fluidsynth",
            "path": resolved_renderer,
        },
        "soundfont": {
            "path": str(Path(resolved_soundfont).expanduser()),
            "sha256": sha256_file(Path(resolved_soundfont).expanduser()),
        },
        "rendered_audio_files": rendered,
        "summary": {
            "rendered_audio_file_count": int(len(rendered)),
            "technical_wav_validation": True,
            "repaired_candidate_count": int(len(candidates)),
            "repaired_dead_air_max": max(repaired_dead_air_values, default=0.0),
            "max_repaired_interval": max(repaired_interval_values, default=0),
            "min_repaired_unique_pitch_count": min(unique_pitch_values, default=0),
            "max_repaired_simultaneous_notes": max(simultaneous_values, default=0),
            "max_pitch_changed_ratio": max(pitch_changed_ratios, default=0.0),
            "audio_review_required": True,
        },
        "audio_render_boundary": {
            "boundary": BOUNDARY,
            "render_attempted": True,
            "rendered_audio_file_count": int(len(rendered)),
            "technical_wav_validation": True,
            "pitch_contour_audio_package_completed": True,
            "human_audio_preference_claimed": False,
            "audio_rendered_quality_claimed": False,
            "midi_to_solo_musical_quality_claimed": False,
            "model_checkpoint_generation_quality_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
            "production_ready_claimed": False,
        },
        "decision": {
            "current_boundary": BOUNDARY,
            "next_boundary": NEXT_BOUNDARY,
            "auto_progress_allowed": True,
            "critical_user_input_required": False,
            "reason": (
                "pitch-contour repaired MIDI rendered to WAV; prepare listening review package next"
            ),
        },
        "not_proven": [
            "audio_rendered_quality",
            "human_audio_preference",
            "midi_to_solo_musical_quality",
            "model_checkpoint_generation_quality",
            "broad_trained_model_quality",
            "brad_style_adaptation",
        ],
        "next_recommended_issue": (
            "Stage B MIDI-to-solo model-conditioned input path dead-air timing repair pitch contour listening review package"
        ),
    }


def validate_audio_render_report(
    report: dict[str, Any],
    *,
    expected_boundary: str | None,
    expected_next_boundary: str | None,
    expected_file_count: int,
    expected_sample_rate: int,
    require_audio_package_completed: bool,
    require_no_quality_claim: bool,
) -> dict[str, Any]:
    boundary = _dict(report.get("audio_render_boundary"))
    decision = _dict(report.get("decision"))
    summary = _dict(report.get("summary"))
    if expected_boundary and str(boundary.get("boundary") or "") != expected_boundary:
        raise StageBMidiToSoloModelConditionedInputPathDeadAirTimingRepairPitchContourAudioError(
            "unexpected boundary"
        )
    if expected_next_boundary and str(decision.get("next_boundary") or "") != expected_next_boundary:
        raise StageBMidiToSoloModelConditionedInputPathDeadAirTimingRepairPitchContourAudioError(
            "unexpected next boundary"
        )
    files = [_dict(item) for item in _list(report.get("rendered_audio_files"))]
    if len(files) != int(expected_file_count):
        raise StageBMidiToSoloModelConditionedInputPathDeadAirTimingRepairPitchContourAudioError(
            f"expected {expected_file_count} rendered files"
        )
    for item in files:
        wav_file = _dict(item.get("wav_file"))
        if not bool(wav_file.get("exists", False)) or not _path_exists(str(wav_file.get("path") or "")):
            raise StageBMidiToSoloModelConditionedInputPathDeadAirTimingRepairPitchContourAudioError(
                "missing rendered WAV"
            )
        if _int(wav_file.get("sample_rate")) != int(expected_sample_rate):
            raise StageBMidiToSoloModelConditionedInputPathDeadAirTimingRepairPitchContourAudioError(
                "unexpected sample rate"
            )
        if _int(wav_file.get("frame_count")) <= 0:
            raise StageBMidiToSoloModelConditionedInputPathDeadAirTimingRepairPitchContourAudioError(
                "empty WAV"
            )
        if _int(wav_file.get("size_bytes")) <= 44:
            raise StageBMidiToSoloModelConditionedInputPathDeadAirTimingRepairPitchContourAudioError(
                "invalid WAV size"
            )
    if require_audio_package_completed and not bool(
        boundary.get("pitch_contour_audio_package_completed", False)
    ):
        raise StageBMidiToSoloModelConditionedInputPathDeadAirTimingRepairPitchContourAudioError(
            "audio package completion required"
        )
    if bool(decision.get("critical_user_input_required", True)):
        raise StageBMidiToSoloModelConditionedInputPathDeadAirTimingRepairPitchContourAudioError(
            "critical user input should not be required"
        )
    if require_no_quality_claim:
        _require_no_quality_claim(boundary, label="audio render boundary")
    return {
        "boundary": str(boundary.get("boundary") or ""),
        "next_boundary": str(decision.get("next_boundary") or ""),
        "render_attempted": bool(boundary.get("render_attempted", False)),
        "rendered_audio_file_count": _int(boundary.get("rendered_audio_file_count")),
        "technical_wav_validation": bool(boundary.get("technical_wav_validation", False)),
        "pitch_contour_audio_package_completed": bool(
            boundary.get("pitch_contour_audio_package_completed", False)
        ),
        "repaired_candidate_count": _int(summary.get("repaired_candidate_count")),
        "repaired_dead_air_max": _float(summary.get("repaired_dead_air_max")),
        "max_repaired_interval": _int(summary.get("max_repaired_interval")),
        "min_repaired_unique_pitch_count": _int(
            summary.get("min_repaired_unique_pitch_count")
        ),
        "max_repaired_simultaneous_notes": _int(
            summary.get("max_repaired_simultaneous_notes")
        ),
        "max_pitch_changed_ratio": _float(summary.get("max_pitch_changed_ratio")),
        "audio_review_required": bool(summary.get("audio_review_required", False)),
        "audio_rendered_quality_claimed": bool(boundary.get("audio_rendered_quality_claimed", True)),
        "human_audio_preference_claimed": bool(boundary.get("human_audio_preference_claimed", True)),
        "midi_to_solo_musical_quality_claimed": bool(
            boundary.get("midi_to_solo_musical_quality_claimed", True)
        ),
        "critical_user_input_required": bool(decision.get("critical_user_input_required", True)),
        "wav_paths": [str(_dict(item.get("wav_file")).get("path") or "") for item in files],
        "next_recommended_issue": str(report.get("next_recommended_issue") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    boundary = report["audio_render_boundary"]
    decision = report["decision"]
    summary = report["summary"]
    lines = [
        "# Stage B MIDI-to-Solo Model-Conditioned Input Path Dead-Air Timing Repair Pitch Contour Audio Package",
        "",
        "## Summary",
        "",
        f"- boundary: `{boundary['boundary']}`",
        f"- next boundary: `{decision['next_boundary']}`",
        f"- render attempted: `{_bool_token(boundary['render_attempted'])}`",
        f"- rendered audio file count: `{boundary['rendered_audio_file_count']}`",
        f"- technical WAV validation: `{_bool_token(boundary['technical_wav_validation'])}`",
        f"- repaired dead-air max: `{summary['repaired_dead_air_max']:.4f}`",
        f"- max repaired interval: `{summary['max_repaired_interval']}`",
        f"- min repaired unique pitch count: `{summary['min_repaired_unique_pitch_count']}`",
        f"- max repaired simultaneous notes: `{summary['max_repaired_simultaneous_notes']}`",
        f"- max pitch changed ratio: `{summary['max_pitch_changed_ratio']:.4f}`",
        f"- audio review required: `{_bool_token(summary['audio_review_required'])}`",
        f"- audio rendered quality claimed: `{_bool_token(boundary['audio_rendered_quality_claimed'])}`",
        f"- human/audio preference claimed: `{_bool_token(boundary['human_audio_preference_claimed'])}`",
        "",
        "## Rendered Files",
        "",
    ]
    for item in report.get("rendered_audio_files", []):
        wav_file = item["wav_file"]
        lines.append(
            f"- rank `{item['rank']}` sample `{item['sample_index']}` seed `{item['sample_seed']}`: "
            f"`{wav_file['path']}`, duration `{float(wav_file['duration_seconds']):.3f}`, "
            f"sample rate `{wav_file['sample_rate']}`, max interval `{item['repaired_max_interval']}`, "
            f"pitch changed ratio `{float(item['pitch_changed_ratio']):.4f}`, "
            f"sha256 `{str(wav_file['sha256'])[:12]}`"
        )
    lines.extend(["", "## Claim Boundary", ""])
    for item in report.get("not_proven", []):
        lines.append(f"- `{item}`")
    lines.extend(["", "## Next", "", f"- `{report['next_recommended_issue']}`"])
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render model-conditioned pitch-contour repair WAV files"
    )
    parser.add_argument("--pitch_contour_probe_report", type=str, required=True)
    parser.add_argument(
        "--output_root",
        type=str,
        default=(
            "outputs/stage_b_midi_to_solo_model_conditioned_input_path_"
            "dead_air_timing_repair_pitch_contour_audio_package"
        ),
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--renderer", type=str, default=shutil.which("fluidsynth") or "")
    parser.add_argument("--soundfont", type=str, default="")
    parser.add_argument("--sample_rate", type=int, default=44100)
    parser.add_argument("--expected_boundary", type=str, default="")
    parser.add_argument("--expected_next_boundary", type=str, default="")
    parser.add_argument("--expected_file_count", type=int, default=3)
    parser.add_argument("--require_audio_package_completed", action="store_true")
    parser.add_argument("--require_no_quality_claim", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    report = build_audio_render_report(
        read_json(Path(args.pitch_contour_probe_report)),
        output_dir=output_dir,
        renderer_path=str(args.renderer or ""),
        soundfont_path=str(args.soundfont or ""),
        sample_rate=int(args.sample_rate),
        expected_file_count=int(args.expected_file_count),
    )
    summary = validate_audio_render_report(
        report,
        expected_boundary=str(args.expected_boundary or ""),
        expected_next_boundary=str(args.expected_next_boundary or ""),
        expected_file_count=int(args.expected_file_count),
        expected_sample_rate=int(args.sample_rate),
        require_audio_package_completed=bool(args.require_audio_package_completed),
        require_no_quality_claim=bool(args.require_no_quality_claim),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        output_dir
        / "stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_pitch_contour_audio_package.json",
        report,
    )
    write_json(
        output_dir
        / "stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_pitch_contour_audio_package_validation_summary.json",
        summary,
    )
    markdown = markdown_report(report)
    write_text(
        output_dir
        / "stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_pitch_contour_audio_package.md",
        markdown,
    )
    if args.doc_path:
        write_text(Path(args.doc_path), markdown)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
