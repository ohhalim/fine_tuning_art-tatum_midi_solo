"""Render chord-tone landing repair MIDI candidates to WAV files."""

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
from scripts.build_stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_context_pitch_role_bridge import (  # noqa: E402
    BRIDGE_REQUIRED_SOURCE_CONTEXT_KEYS,
    BRIDGE_SOURCE_CONTEXT_PRESERVED_KEYS,
)
from scripts.run_stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_tone_landing_repair_sweep import (  # noqa: E402
    BOUNDARY as SOURCE_BOUNDARY,
    NEXT_BOUNDARY as SOURCE_NEXT_BOUNDARY,
    SCHEMA_VERSION as SOURCE_SCHEMA_VERSION,
    StageBMidiToSoloChordToneLandingRepairSweepError,
    validate_repair_sweep_report,
)


class StageBMidiToSoloChordToneLandingRepairAudioError(ValueError):
    pass


BOUNDARY = (
    "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_tone_landing_repair_audio_package"
)
NEXT_BOUNDARY = (
    "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_tone_landing_repair_listening_review_package"
)
SCHEMA_VERSION = (
    "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_tone_landing_repair_audio_package_v4"
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
        raise StageBMidiToSoloChordToneLandingRepairAudioError(
            f"unexpected quality claim in {label}: {claimed}"
        )


def _source_context_fields(container: dict[str, Any], *, label: str) -> dict[str, Any]:
    for key in BRIDGE_REQUIRED_SOURCE_CONTEXT_KEYS:
        if key not in container:
            raise StageBMidiToSoloChordToneLandingRepairAudioError(
                f"{label} source-context field required: {key}"
            )
    for key in BRIDGE_SOURCE_CONTEXT_PRESERVED_KEYS:
        if not bool(container.get(key, False)):
            raise StageBMidiToSoloChordToneLandingRepairAudioError(
                f"{label} source context should be preserved: {key}"
            )
    return {key: container[key] for key in BRIDGE_REQUIRED_SOURCE_CONTEXT_KEYS}


def validate_source_report(
    report: dict[str, Any],
    *,
    expected_count: int,
) -> list[dict[str, Any]]:
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    if str(report.get("boundary") or readiness.get("boundary") or "") != SOURCE_BOUNDARY:
        raise StageBMidiToSoloChordToneLandingRepairAudioError(
            "chord-tone landing repair sweep boundary required"
        )
    if str(report.get("schema_version") or "") != SOURCE_SCHEMA_VERSION:
        raise StageBMidiToSoloChordToneLandingRepairAudioError(
            "repair sweep schema version must match current repair sweep report"
        )
    if str(decision.get("next_boundary") or "") != SOURCE_NEXT_BOUNDARY:
        raise StageBMidiToSoloChordToneLandingRepairAudioError(
            "repair sweep must route to audio package"
        )
    try:
        validate_repair_sweep_report(
            report,
            expected_boundary=SOURCE_BOUNDARY,
            expected_next_boundary=SOURCE_NEXT_BOUNDARY,
            require_repair_completed=True,
            require_target_supported=True,
            require_no_quality_claim=True,
        )
    except StageBMidiToSoloChordToneLandingRepairSweepError as exc:
        raise StageBMidiToSoloChordToneLandingRepairAudioError(str(exc)) from exc
    if bool(decision.get("critical_user_input_required", True)):
        raise StageBMidiToSoloChordToneLandingRepairAudioError(
            "critical user input should not be required"
        )
    _require_no_quality_claim(readiness, label="repair sweep readiness")

    rows = [_dict(item) for item in _list(report.get("candidate_repairs"))]
    if len(rows) < int(expected_count):
        raise StageBMidiToSoloChordToneLandingRepairAudioError(
            "candidate repair count below expected count"
        )
    compacted: list[dict[str, Any]] = []
    for index, row in enumerate(rows[: int(expected_count)], start=1):
        repaired_midi_path = str(row.get("repaired_midi_path") or "")
        if not _path_exists(repaired_midi_path):
            raise StageBMidiToSoloChordToneLandingRepairAudioError(
                f"repaired MIDI missing: {repaired_midi_path}"
            )
        before = _dict(row.get("before"))
        after = _dict(row.get("after"))
        repair = _dict(row.get("repair"))
        compacted.append(
            {
                "candidate_index": int(index),
                "rank": _int(row.get("rank")) or index,
                "source_midi_path": str(row.get("source_midi_path") or ""),
                "repaired_midi_path": repaired_midi_path,
                "changed_note_count": _int(repair.get("changed_note_count")),
                "before_bridge_flags": [str(flag) for flag in _list(before.get("bridge_flags"))],
                "after_bridge_flags": [str(flag) for flag in _list(after.get("bridge_flags"))],
                "before_chord_tone_ratio": _float(before.get("chord_tone_ratio")),
                "after_chord_tone_ratio": _float(after.get("chord_tone_ratio")),
                "before_strong_beat_chord_tone_ratio": _float(
                    before.get("strong_beat_chord_tone_ratio")
                ),
                "after_strong_beat_chord_tone_ratio": _float(
                    after.get("strong_beat_chord_tone_ratio")
                ),
                "before_cadence_landing_chord_tone": bool(
                    before.get("cadence_landing_chord_tone", False)
                ),
                "after_cadence_landing_chord_tone": bool(
                    after.get("cadence_landing_chord_tone", False)
                ),
                "before_cadence_landing_role": str(
                    before.get("cadence_landing_role") or ""
                ),
                "after_cadence_landing_role": str(
                    after.get("cadence_landing_role") or ""
                ),
                "before_max_non_chord_tone_run": _int(
                    before.get("max_non_chord_tone_run")
                ),
                "after_max_non_chord_tone_run": _int(
                    after.get("max_non_chord_tone_run")
                ),
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
        raise StageBMidiToSoloChordToneLandingRepairAudioError(
            f"renderer not found: {renderer}"
        )
    if not soundfont.exists():
        raise StageBMidiToSoloChordToneLandingRepairAudioError(
            f"soundfont not found: {soundfont}"
        )
    plan: list[dict[str, Any]] = []
    for item in candidates:
        wav_path = output_dir / "audio" / (
            f"candidate_{int(item['candidate_index']):02d}_rank_{int(item['rank']):02d}_"
            "chord_tone_landing_repair.wav"
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
            raise StageBMidiToSoloChordToneLandingRepairAudioError(
                f"render failed for candidate {item['candidate_index']}: "
                f"{completed.stderr or completed.stdout}"
            )
        rendered.append(
            {
                "candidate_index": item["candidate_index"],
                "rank": item["rank"],
                "source_midi_path": item["source_midi_path"],
                "repaired_midi_path": item["repaired_midi_path"],
                "changed_note_count": item["changed_note_count"],
                "before_bridge_flags": item["before_bridge_flags"],
                "after_bridge_flags": item["after_bridge_flags"],
                "before_chord_tone_ratio": item["before_chord_tone_ratio"],
                "after_chord_tone_ratio": item["after_chord_tone_ratio"],
                "before_strong_beat_chord_tone_ratio": item[
                    "before_strong_beat_chord_tone_ratio"
                ],
                "after_strong_beat_chord_tone_ratio": item[
                    "after_strong_beat_chord_tone_ratio"
                ],
                "before_cadence_landing_chord_tone": item[
                    "before_cadence_landing_chord_tone"
                ],
                "after_cadence_landing_chord_tone": item[
                    "after_cadence_landing_chord_tone"
                ],
                "before_cadence_landing_role": item["before_cadence_landing_role"],
                "after_cadence_landing_role": item["after_cadence_landing_role"],
                "before_max_non_chord_tone_run": item["before_max_non_chord_tone_run"],
                "after_max_non_chord_tone_run": item["after_max_non_chord_tone_run"],
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
    issue_number: int = 1046,
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
    aggregate = _dict(source_report.get("aggregate"))
    source_context = _source_context_fields(aggregate, label="repair sweep aggregate")
    durations = [
        _float(_dict(item.get("wav_file")).get("duration_seconds")) for item in rendered
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "issue_number": int(issue_number),
        "source_boundary": SOURCE_BOUNDARY,
        "source_schema_version": SOURCE_SCHEMA_VERSION,
        "source_summary": aggregate,
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
            "source_schema_version": SOURCE_SCHEMA_VERSION,
            "source_objective_schema_version": str(source_report.get("source_schema_version") or ""),
            "source_bridge_schema_version": str(source_report.get("bridge_schema_version") or ""),
            "technical_wav_validation": True,
            "sample_rate": int(sample_rate),
            "duration_min_seconds": min(durations, default=0.0),
            "duration_max_seconds": max(durations, default=0.0),
            "changed_note_total": _int(aggregate.get("changed_note_total")),
            "objective_outside_soloing_pitch_role_risk_count": _int(
                aggregate.get("objective_outside_soloing_pitch_role_risk_count")
            ),
            "weak_chord_tone_landing_risk_count_before": _int(
                aggregate.get("weak_chord_tone_landing_risk_count_before")
            ),
            "weak_chord_tone_landing_risk_count_after": _int(
                aggregate.get("weak_chord_tone_landing_risk_count_after")
            ),
            "weak_chord_tone_landing_risk_delta": _int(
                aggregate.get("weak_chord_tone_landing_risk_delta")
            ),
            "outside_soloing_pitch_role_risk_count_before": _int(
                aggregate.get("outside_soloing_pitch_role_risk_count_before")
            ),
            "outside_soloing_pitch_role_risk_count_after": _int(
                aggregate.get("outside_soloing_pitch_role_risk_count_after")
            ),
            "outside_soloing_pitch_role_risk_delta": _int(
                aggregate.get("outside_soloing_pitch_role_risk_delta")
            ),
            "outside_soloing_repair_targeted": bool(
                aggregate.get("outside_soloing_repair_targeted", True)
            ),
            "outside_soloing_residual_risk_preserved": bool(
                aggregate.get("outside_soloing_residual_risk_preserved", False)
            ),
            "final_landing_chord_tone_count_before": _int(
                aggregate.get("final_landing_chord_tone_count_before")
            ),
            "final_landing_chord_tone_count_after": _int(
                aggregate.get("final_landing_chord_tone_count_after")
            ),
            "target_supported": bool(aggregate.get("target_supported", False)),
            **source_context,
            "audio_review_required": True,
        },
        "audio_render_boundary": {
            "boundary": BOUNDARY,
            "render_attempted": True,
            "rendered_audio_file_count": int(len(rendered)),
            "technical_wav_validation": True,
            "songlike_melody_contour_phrase_rhythm_chord_tone_landing_repair_audio_package_completed": True,
            "human_audio_preference_claimed": False,
            "audio_rendered_quality_claimed": False,
            "midi_to_solo_musical_quality_claimed": False,
            "model_checkpoint_generation_quality_claimed": False,
            "model_direct_generation_quality_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
            "production_ready_claimed": False,
        },
        "decision": {
            "current_boundary": BOUNDARY,
            "next_boundary": NEXT_BOUNDARY,
            "auto_progress_allowed": True,
            "critical_user_input_required": False,
            "reason": "chord-tone landing repair MIDI rendered to WAV; prepare listening review package next",
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
            "Stage B MIDI-to-solo songlike melody contour phrase/rhythm chord-tone landing repair listening review package source-context refresh"
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
        raise StageBMidiToSoloChordToneLandingRepairAudioError("unexpected boundary")
    if expected_next_boundary and str(decision.get("next_boundary") or "") != expected_next_boundary:
        raise StageBMidiToSoloChordToneLandingRepairAudioError(
            "unexpected next boundary"
        )
    if str(report.get("source_schema_version") or "") != SOURCE_SCHEMA_VERSION:
        raise StageBMidiToSoloChordToneLandingRepairAudioError(
            "unexpected source schema version"
        )
    if str(summary.get("source_schema_version") or "") != SOURCE_SCHEMA_VERSION:
        raise StageBMidiToSoloChordToneLandingRepairAudioError(
            "summary source schema version mismatch"
        )
    files = [_dict(item) for item in _list(report.get("rendered_audio_files"))]
    if len(files) != int(expected_file_count):
        raise StageBMidiToSoloChordToneLandingRepairAudioError(
            f"expected {expected_file_count} rendered files"
        )
    for item in files:
        wav_file = _dict(item.get("wav_file"))
        if not bool(wav_file.get("exists", False)) or not _path_exists(
            str(wav_file.get("path") or "")
        ):
            raise StageBMidiToSoloChordToneLandingRepairAudioError("missing rendered WAV")
        if _int(wav_file.get("sample_rate")) != int(expected_sample_rate):
            raise StageBMidiToSoloChordToneLandingRepairAudioError(
                "unexpected sample rate"
            )
        if _int(wav_file.get("frame_count")) <= 0:
            raise StageBMidiToSoloChordToneLandingRepairAudioError("empty WAV")
        if _int(wav_file.get("size_bytes")) <= 44:
            raise StageBMidiToSoloChordToneLandingRepairAudioError("invalid WAV size")
    completed_key = (
        "songlike_melody_contour_phrase_rhythm_chord_tone_landing_repair_audio_package_completed"
    )
    if require_audio_package_completed and not bool(boundary.get(completed_key, False)):
        raise StageBMidiToSoloChordToneLandingRepairAudioError(
            "audio package completion required"
        )
    if bool(decision.get("critical_user_input_required", True)):
        raise StageBMidiToSoloChordToneLandingRepairAudioError(
            "critical user input should not be required"
        )
    if _int(summary.get("objective_outside_soloing_pitch_role_risk_count")) != _int(
        summary.get("outside_soloing_pitch_role_risk_count_before")
    ):
        raise StageBMidiToSoloChordToneLandingRepairAudioError(
            "outside-soloing objective and source counts must match"
        )
    if bool(summary.get("outside_soloing_repair_targeted", True)):
        raise StageBMidiToSoloChordToneLandingRepairAudioError(
            "outside-soloing repair target should remain false in audio package"
        )
    if not bool(summary.get("outside_soloing_residual_risk_preserved", False)):
        raise StageBMidiToSoloChordToneLandingRepairAudioError(
            "outside-soloing residual risk context must be preserved"
        )
    for key in BRIDGE_REQUIRED_SOURCE_CONTEXT_KEYS:
        if key not in summary:
            raise StageBMidiToSoloChordToneLandingRepairAudioError(
                f"audio package source-context field required: {key}"
            )
    for key in BRIDGE_SOURCE_CONTEXT_PRESERVED_KEYS:
        if not bool(summary.get(key, False)):
            raise StageBMidiToSoloChordToneLandingRepairAudioError(
                f"audio package source-context preserved field must be true: {key}"
            )
    if require_no_quality_claim:
        _require_no_quality_claim(boundary, label="audio render boundary")
    return {
        "boundary": str(boundary.get("boundary") or ""),
        "source_schema_version": str(summary.get("source_schema_version") or ""),
        "source_objective_schema_version": str(
            summary.get("source_objective_schema_version") or ""
        ),
        "source_bridge_schema_version": str(summary.get("source_bridge_schema_version") or ""),
        "next_boundary": str(decision.get("next_boundary") or ""),
        "render_attempted": bool(boundary.get("render_attempted", False)),
        "rendered_audio_file_count": _int(boundary.get("rendered_audio_file_count")),
        "technical_wav_validation": bool(boundary.get("technical_wav_validation", False)),
        "songlike_melody_contour_phrase_rhythm_chord_tone_landing_repair_audio_package_completed": bool(
            boundary.get(completed_key, False)
        ),
        "sample_rate": _int(summary.get("sample_rate")),
        "duration_min_seconds": _float(summary.get("duration_min_seconds")),
        "duration_max_seconds": _float(summary.get("duration_max_seconds")),
        "changed_note_total": _int(summary.get("changed_note_total")),
        "objective_outside_soloing_pitch_role_risk_count": _int(
            summary.get("objective_outside_soloing_pitch_role_risk_count")
        ),
        "weak_chord_tone_landing_risk_count_before": _int(
            summary.get("weak_chord_tone_landing_risk_count_before")
        ),
        "weak_chord_tone_landing_risk_count_after": _int(
            summary.get("weak_chord_tone_landing_risk_count_after")
        ),
        "weak_chord_tone_landing_risk_delta": _int(
            summary.get("weak_chord_tone_landing_risk_delta")
        ),
        "outside_soloing_pitch_role_risk_count_before": _int(
            summary.get("outside_soloing_pitch_role_risk_count_before")
        ),
        "outside_soloing_pitch_role_risk_count_after": _int(
            summary.get("outside_soloing_pitch_role_risk_count_after")
        ),
        "outside_soloing_pitch_role_risk_delta": _int(
            summary.get("outside_soloing_pitch_role_risk_delta")
        ),
        "outside_soloing_repair_targeted": bool(
            summary.get("outside_soloing_repair_targeted", True)
        ),
        "outside_soloing_residual_risk_preserved": bool(
            summary.get("outside_soloing_residual_risk_preserved", False)
        ),
        "final_landing_chord_tone_count_before": _int(
            summary.get("final_landing_chord_tone_count_before")
        ),
        "final_landing_chord_tone_count_after": _int(
            summary.get("final_landing_chord_tone_count_after")
        ),
        "target_supported": bool(summary.get("target_supported", False)),
        "audio_review_required": bool(summary.get("audio_review_required", False)),
        **{key: summary.get(key) for key in BRIDGE_REQUIRED_SOURCE_CONTEXT_KEYS},
        "audio_rendered_quality_claimed": bool(
            boundary.get("audio_rendered_quality_claimed", True)
        ),
        "human_audio_preference_claimed": bool(
            boundary.get("human_audio_preference_claimed", True)
        ),
        "midi_to_solo_musical_quality_claimed": bool(
            boundary.get("midi_to_solo_musical_quality_claimed", True)
        ),
        "critical_user_input_required": bool(
            decision.get("critical_user_input_required", True)
        ),
        "wav_paths": [str(_dict(item.get("wav_file")).get("path") or "") for item in files],
        "next_recommended_issue": str(report.get("next_recommended_issue") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    boundary = report["audio_render_boundary"]
    decision = report["decision"]
    summary = report["summary"]
    lines = [
        "# Stage B MIDI-to-Solo Chord-Tone Landing Repair Audio Package",
        "",
        "## Summary",
        "",
        f"- boundary: `{boundary['boundary']}`",
        f"- source schema version: `{report['source_schema_version']}`",
        f"- source objective schema version: `{summary['source_objective_schema_version']}`",
        f"- source bridge schema version: `{summary['source_bridge_schema_version']}`",
        f"- next boundary: `{decision['next_boundary']}`",
        f"- render attempted: `{_bool_token(boundary['render_attempted'])}`",
        f"- rendered audio file count: `{boundary['rendered_audio_file_count']}`",
        f"- technical WAV validation: `{_bool_token(boundary['technical_wav_validation'])}`",
        f"- duration range: `{summary['duration_min_seconds']:.3f}s-{summary['duration_max_seconds']:.3f}s`",
        f"- changed note total: `{summary['changed_note_total']}`",
        f"- objective outside-soloing pitch-role risk count: `{summary['objective_outside_soloing_pitch_role_risk_count']}`",
        f"- weak chord-tone landing risk count: `{summary['weak_chord_tone_landing_risk_count_before']} -> {summary['weak_chord_tone_landing_risk_count_after']}`",
        f"- outside-soloing pitch-role risk count: `{summary['outside_soloing_pitch_role_risk_count_before']} -> {summary['outside_soloing_pitch_role_risk_count_after']}`",
        f"- outside-soloing repair targeted: `{_bool_token(summary['outside_soloing_repair_targeted'])}`",
        f"- outside-soloing residual risk preserved: `{_bool_token(summary['outside_soloing_residual_risk_preserved'])}`",
        f"- follow-up objective source outside-soloing source pitch-role risk: `{summary['followup_objective_source_outside_soloing_source_pitch_role_risk_count_before']} -> {summary['followup_objective_source_outside_soloing_source_pitch_role_risk_count_after']}`",
        f"- follow-up objective source outside-soloing source pitch-role risk delta: `{summary['followup_objective_source_outside_soloing_source_pitch_role_risk_delta']}`",
        f"- follow-up objective source outside-soloing source context preserved: `{_bool_token(summary['followup_objective_source_outside_soloing_source_context_preserved'])}`",
        f"- follow-up objective source outside-soloing source targeted: `{_bool_token(summary['followup_objective_source_outside_soloing_source_targeted'])}`",
        f"- follow-up objective source outside-soloing source residual risk preserved: `{_bool_token(summary['followup_objective_source_outside_soloing_source_residual_risk_preserved'])}`",
        f"- follow-up objective source outside-soloing current repair pitch-role risk after/delta: `{summary['followup_objective_source_outside_soloing_current_pitch_role_risk_count_after']} / {summary['followup_objective_source_outside_soloing_current_pitch_role_risk_delta']}`",
        f"- follow-up repair sweep source outside-soloing source pitch-role risk: `{summary['followup_repair_sweep_source_outside_soloing_source_pitch_role_risk_count_before']} -> {summary['followup_repair_sweep_source_outside_soloing_source_pitch_role_risk_count_after']}`",
        f"- follow-up repair sweep source outside-soloing source pitch-role risk delta: `{summary['followup_repair_sweep_source_outside_soloing_source_pitch_role_risk_delta']}`",
        f"- follow-up repair sweep source outside-soloing source context preserved: `{_bool_token(summary['followup_repair_sweep_source_outside_soloing_source_context_preserved'])}`",
        f"- follow-up repair sweep source outside-soloing source targeted: `{_bool_token(summary['followup_repair_sweep_source_outside_soloing_source_targeted'])}`",
        f"- follow-up repair sweep source outside-soloing source residual risk preserved: `{_bool_token(summary['followup_repair_sweep_source_outside_soloing_source_residual_risk_preserved'])}`",
        f"- follow-up repair sweep source outside-soloing current repair pitch-role risk after/delta: `{summary['followup_repair_sweep_source_outside_soloing_current_pitch_role_risk_count_after']} / {summary['followup_repair_sweep_source_outside_soloing_current_pitch_role_risk_delta']}`",
        f"- bridge repair sweep source outside-soloing source pitch-role risk: `{summary['repair_sweep_source_outside_soloing_source_pitch_role_risk_count_before']} -> {summary['repair_sweep_source_outside_soloing_source_pitch_role_risk_count_after']}`",
        f"- bridge repair sweep source outside-soloing source pitch-role risk delta: `{summary['repair_sweep_source_outside_soloing_source_pitch_role_risk_delta']}`",
        f"- bridge repair sweep source outside-soloing source context preserved: `{_bool_token(summary['repair_sweep_source_outside_soloing_source_context_preserved'])}`",
        f"- bridge repair sweep source outside-soloing source targeted: `{_bool_token(summary['repair_sweep_source_outside_soloing_source_targeted'])}`",
        f"- bridge repair sweep source outside-soloing source residual risk preserved: `{_bool_token(summary['repair_sweep_source_outside_soloing_source_residual_risk_preserved'])}`",
        f"- bridge repair sweep source outside-soloing current repair pitch-role risk after/delta: `{summary['repair_sweep_source_outside_soloing_current_pitch_role_risk_count_after']} / {summary['repair_sweep_source_outside_soloing_current_pitch_role_risk_delta']}`",
        f"- final landing chord-tone count: `{summary['final_landing_chord_tone_count_before']} -> {summary['final_landing_chord_tone_count_after']}`",
        f"- audio review required: `{_bool_token(summary['audio_review_required'])}`",
        f"- audio rendered quality claimed: `{_bool_token(boundary['audio_rendered_quality_claimed'])}`",
        f"- human/audio preference claimed: `{_bool_token(boundary['human_audio_preference_claimed'])}`",
        f"- MIDI-to-solo musical quality claimed: `{_bool_token(boundary['midi_to_solo_musical_quality_claimed'])}`",
        "",
        "## Rendered Files",
        "",
    ]
    for item in report.get("rendered_audio_files", []):
        wav_file = item["wav_file"]
        lines.append(
            f"- candidate `{item['candidate_index']}` rank `{item['rank']}`: "
            f"`{wav_file['path']}`, duration `{float(wav_file['duration_seconds']):.3f}`, "
            f"sample rate `{wav_file['sample_rate']}`, changed `{item['changed_note_count']}`, "
            f"flags `{','.join(item['after_bridge_flags']) or 'none'}`, "
            f"sha256 `{str(wav_file['sha256'])[:12]}`"
        )
    lines.extend(["", "## Claim Boundary", ""])
    for item in report.get("not_proven", []):
        lines.append(f"- `{item}`")
    lines.extend(["", "## Next", "", f"- `{report['next_recommended_issue']}`"])
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render chord-tone landing repair MIDI candidates to WAV files"
    )
    parser.add_argument("--chord_tone_landing_repair_sweep_report", type=str, required=True)
    parser.add_argument(
        "--output_root",
        type=str,
        default=(
            "outputs/stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_"
            "chord_tone_landing_repair_audio_package"
        ),
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--issue_number", type=int, default=1130)
    parser.add_argument("--renderer", type=str, default=shutil.which("fluidsynth") or "")
    parser.add_argument("--soundfont", type=str, default="")
    parser.add_argument("--sample_rate", type=int, default=44100)
    parser.add_argument("--expected_boundary", type=str, default="")
    parser.add_argument("--expected_next_boundary", type=str, default="")
    parser.add_argument("--expected_file_count", type=int, default=6)
    parser.add_argument("--require_audio_package_completed", action="store_true")
    parser.add_argument("--require_no_quality_claim", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    report = build_audio_render_report(
        read_json(Path(args.chord_tone_landing_repair_sweep_report)),
        output_dir=output_dir,
        renderer_path=str(args.renderer or ""),
        soundfont_path=str(args.soundfont or ""),
        sample_rate=int(args.sample_rate),
        expected_file_count=int(args.expected_file_count),
        issue_number=int(args.issue_number),
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
        / (
            "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_"
            "chord_tone_landing_repair_audio_package.json"
        ),
        report,
    )
    write_json(
        output_dir
        / (
            "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_"
            "chord_tone_landing_repair_audio_package_validation_summary.json"
        ),
        summary,
    )
    markdown = markdown_report(report)
    write_text(
        output_dir
        / (
            "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_"
            "chord_tone_landing_repair_audio_package.md"
        ),
        markdown,
    )
    if args.doc_path:
        write_text(Path(args.doc_path), markdown)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
