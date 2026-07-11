"""Render songlike melody contour phrase/rhythm repair MIDI candidates to WAV files."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from scripts.assess_stage_b_generic_base_readiness import read_json, write_json, write_text  # noqa: E402
from scripts.audit_stage_b_midi_to_solo_final_status import (  # noqa: E402
    BRIDGE_REQUIRED_SOURCE_CONTEXT_KEYS,
    BRIDGE_SOURCE_CONTEXT_PRESERVED_KEYS,
    OUTSIDE_SOLOING_REPAIR_OBJECTIVE_SCHEMA_VERSION,
)
from scripts.render_stage_b_midi_to_solo_candidate_audio import (  # noqa: E402
    resolve_soundfont,
    sha256_file,
    wav_meta,
)
from scripts.run_stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_repair_sweep import (  # noqa: E402
    BOUNDARY as SOURCE_BOUNDARY,
    EXPECTED_SOURCE_SCHEMA_VERSIONS as SOURCE_EXPECTED_SCHEMA_VERSIONS,
    NEXT_BOUNDARY as SOURCE_NEXT_BOUNDARY,
    SCHEMA_VERSION as SOURCE_SCHEMA_VERSION,
    StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairSweepError,
    validate_songlike_melody_contour_phrase_rhythm_repair_sweep_report,
)


class StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairAudioError(ValueError):
    pass


BOUNDARY = "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_repair_audio_package"
NEXT_BOUNDARY = "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_repair_listening_review_package"
SCHEMA_VERSION = "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_repair_audio_package_v5"
CommandRunner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]
EXPECTED_SOURCE_SCHEMA_VERSIONS = {
    "songlike_melody_contour_phrase_rhythm_repair_sweep": SOURCE_SCHEMA_VERSION,
    **SOURCE_EXPECTED_SCHEMA_VERSIONS,
}

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
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairAudioError(
            f"unexpected quality claim in {label}: {claimed}"
        )


def _validate_source_schema_versions(
    source_schema_versions: dict[str, Any],
    *,
    label: str,
) -> dict[str, str]:
    normalized = {key: str(value) for key, value in source_schema_versions.items()}
    for key, expected in EXPECTED_SOURCE_SCHEMA_VERSIONS.items():
        if str(normalized.get(key) or "") != expected:
            raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairAudioError(
                f"{label} source schema version mismatch: {key}"
            )
    return normalized


def _source_context_fields(source: dict[str, Any]) -> dict[str, Any]:
    for key in BRIDGE_REQUIRED_SOURCE_CONTEXT_KEYS:
        objective_key = f"objective_{key}"
        if objective_key not in source or source[objective_key] is None:
            raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairAudioError(
                f"objective source-context field required: {objective_key}"
            )
        if key not in source or source[key] is None:
            raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairAudioError(
                f"source-context field required: {key}"
            )
    missing_preserved = []
    for key in BRIDGE_SOURCE_CONTEXT_PRESERVED_KEYS:
        objective_key = f"objective_{key}"
        if not bool(source.get(objective_key)):
            missing_preserved.append(objective_key)
        if not bool(source.get(key)):
            missing_preserved.append(key)
    if missing_preserved:
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairAudioError(
            f"source-context preserved field must be true: {missing_preserved}"
        )
    return {
        "objective_source_outside_soloing_repair_source_objective_pitch_role_risk_count": _int(
            source.get(
                "objective_source_outside_soloing_repair_source_objective_pitch_role_risk_count"
            )
        ),
        "objective_source_outside_soloing_repair_source_context_preserved": bool(
            source.get("objective_source_outside_soloing_repair_source_context_preserved", False)
        ),
        "objective_source_outside_soloing_repair_schema_context_preserved": bool(
            source.get("objective_source_outside_soloing_repair_schema_context_preserved", False)
        ),
        "objective_source_outside_soloing_repair_objective_schema_version": str(
            source.get("objective_source_outside_soloing_repair_objective_schema_version") or ""
        ),
        "objective_source_outside_soloing_repair_source_pitch_role_risk_count_before": _int(
            source.get(
                "objective_source_outside_soloing_repair_source_pitch_role_risk_count_before"
            )
        ),
        "objective_source_outside_soloing_repair_source_pitch_role_risk_count_after": _int(
            source.get(
                "objective_source_outside_soloing_repair_source_pitch_role_risk_count_after"
            )
        ),
        "objective_source_outside_soloing_repair_source_pitch_role_risk_delta": _int(
            source.get(
                "objective_source_outside_soloing_repair_source_pitch_role_risk_delta"
            )
        ),
        "objective_source_outside_soloing_repair_source_targeted": bool(
            source.get("objective_source_outside_soloing_repair_source_targeted", True)
        ),
        "objective_source_outside_soloing_repair_source_residual_risk_preserved": bool(
            source.get(
                "objective_source_outside_soloing_repair_source_residual_risk_preserved",
                False,
            )
        ),
        "objective_source_outside_soloing_repair_pitch_role_risk_count_after": _int(
            source.get("objective_source_outside_soloing_repair_pitch_role_risk_count_after")
        ),
        "objective_source_outside_soloing_repair_pitch_role_risk_delta": _int(
            source.get("objective_source_outside_soloing_repair_pitch_role_risk_delta")
        ),
        "source_outside_soloing_repair_source_objective_pitch_role_risk_count": _int(
            source.get("source_outside_soloing_repair_source_objective_pitch_role_risk_count")
        ),
        "source_outside_soloing_repair_source_context_preserved": bool(
            source.get("source_outside_soloing_repair_source_context_preserved", False)
        ),
        "source_outside_soloing_repair_schema_context_preserved": bool(
            source.get("source_outside_soloing_repair_schema_context_preserved", False)
        ),
        "source_outside_soloing_repair_objective_schema_version": str(
            source.get("source_outside_soloing_repair_objective_schema_version") or ""
        ),
        "source_outside_soloing_repair_source_pitch_role_risk_count_before": _int(
            source.get("source_outside_soloing_repair_source_pitch_role_risk_count_before")
        ),
        "source_outside_soloing_repair_source_pitch_role_risk_count_after": _int(
            source.get("source_outside_soloing_repair_source_pitch_role_risk_count_after")
        ),
        "source_outside_soloing_repair_source_pitch_role_risk_delta": _int(
            source.get("source_outside_soloing_repair_source_pitch_role_risk_delta")
        ),
        "source_outside_soloing_repair_source_targeted": bool(
            source.get("source_outside_soloing_repair_source_targeted", True)
        ),
        "source_outside_soloing_repair_source_residual_risk_preserved": bool(
            source.get("source_outside_soloing_repair_source_residual_risk_preserved", False)
        ),
        "source_outside_soloing_repair_pitch_role_risk_count_after": _int(
            source.get("source_outside_soloing_repair_pitch_role_risk_count_after")
        ),
        "source_outside_soloing_repair_pitch_role_risk_delta": _int(
            source.get("source_outside_soloing_repair_pitch_role_risk_delta")
        ),
        **{
            f"objective_{key}": source.get(f"objective_{key}")
            for key in BRIDGE_REQUIRED_SOURCE_CONTEXT_KEYS
        },
        **{key: source.get(key) for key in BRIDGE_REQUIRED_SOURCE_CONTEXT_KEYS},
    }


def _validate_source_context_group(context: dict[str, Any], *, base: str, label: str) -> None:
    objective_risk = _int(context.get(f"{base}_source_objective_pitch_role_risk_count"))
    source_before = _int(context.get(f"{base}_source_pitch_role_risk_count_before"))
    source_after = _int(context.get(f"{base}_source_pitch_role_risk_count_after"))
    source_delta = _int(context.get(f"{base}_source_pitch_role_risk_delta"))
    current_after = _int(context.get(f"{base}_pitch_role_risk_count_after"))
    current_delta = _int(context.get(f"{base}_pitch_role_risk_delta"))
    if not bool(context.get(f"{base}_source_context_preserved", False)):
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairAudioError(
            f"{label} source context preservation required"
        )
    if not bool(context.get(f"{base}_schema_context_preserved", False)):
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairAudioError(
            f"{label} schema context preservation required"
        )
    if (
        str(context.get(f"{base}_objective_schema_version") or "")
        != OUTSIDE_SOLOING_REPAIR_OBJECTIVE_SCHEMA_VERSION
    ):
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairAudioError(
            f"{label} objective schema version mismatch"
        )
    if objective_risk <= 0 or source_before <= 0:
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairAudioError(
            f"{label} source pitch-role risk context required"
        )
    if objective_risk != source_before:
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairAudioError(
            f"{label} objective/source risk mismatch"
        )
    if source_before - source_after != source_delta:
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairAudioError(
            f"{label} source pitch-role risk delta mismatch"
        )
    if source_delta <= 0:
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairAudioError(
            f"{label} positive source pitch-role risk delta required"
        )
    if bool(context.get(f"{base}_source_targeted", True)):
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairAudioError(
            f"{label} source-targeted flag should remain false"
        )
    if not bool(context.get(f"{base}_source_residual_risk_preserved", False)):
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairAudioError(
            f"{label} source residual risk preservation required"
        )
    if current_after != 0:
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairAudioError(
            f"{label} current repair residual pitch-role risk should be zero"
        )
    if current_delta <= 0:
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairAudioError(
            f"{label} positive current repair pitch-role risk delta required"
        )


def _validate_source_context(context: dict[str, Any], *, label: str) -> None:
    _validate_source_context_group(
        context,
        base="objective_source_outside_soloing_repair",
        label=f"objective {label}",
    )
    _validate_source_context_group(
        context,
        base="source_outside_soloing_repair",
        label=label,
    )


def validate_source_report(
    report: dict[str, Any],
    *,
    expected_count: int,
) -> dict[str, Any]:
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    aggregate = _dict(report.get("aggregate"))
    if str(report.get("boundary") or readiness.get("boundary") or "") != SOURCE_BOUNDARY:
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairAudioError(
            "phrase/rhythm repair sweep boundary required"
        )
    if str(decision.get("next_boundary") or "") != SOURCE_NEXT_BOUNDARY:
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairAudioError(
            "phrase/rhythm repair sweep must route to audio package"
        )
    try:
        source_summary = validate_songlike_melody_contour_phrase_rhythm_repair_sweep_report(
            report,
            expected_boundary=SOURCE_BOUNDARY,
            expected_next_boundary=SOURCE_NEXT_BOUNDARY,
            min_candidate_count=int(expected_count),
            require_sweep_completed=True,
            require_target_supported=True,
            require_phrase_rhythm_delta=True,
            require_no_quality_claim=True,
        )
    except StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairSweepError as exc:
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairAudioError(str(exc)) from exc
    source_schema_versions = _validate_source_schema_versions(
        {
            "songlike_melody_contour_phrase_rhythm_repair_sweep": (
                source_summary.get("schema_version")
            ),
            **_dict(report.get("source_schema_versions")),
        },
        label="phrase/rhythm repair audio package source",
    )
    if not bool(readiness.get("songlike_melody_contour_phrase_rhythm_repair_target_supported", False)):
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairAudioError(
            "phrase/rhythm repair support required"
        )
    if not bool(readiness.get("audio_package_ready", False)):
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairAudioError(
            "audio package readiness required"
        )
    if _int(aggregate.get("technical_regression_count")) != 0:
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairAudioError(
            "technical regression count should be zero"
        )
    source_context = _source_context_fields({**source_summary, **aggregate})
    _validate_source_context(source_context, label="phrase/rhythm repair source context")
    if bool(decision.get("critical_user_input_required", True)):
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairAudioError(
            "critical user input should not be required"
        )
    _require_no_quality_claim(readiness, label="phrase/rhythm repair readiness")

    rows = [_dict(item) for item in _list(report.get("candidate_repairs"))]
    if len(rows) < int(expected_count):
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairAudioError(
            "candidate repair count below expected count"
        )
    compacted: list[dict[str, Any]] = []
    for index, row in enumerate(rows[: int(expected_count)], start=1):
        repaired_midi_path = str(row.get("phrase_rhythm_repaired_midi_path") or "")
        if not _path_exists(repaired_midi_path):
            raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairAudioError(
                f"phrase/rhythm repair MIDI missing: {repaired_midi_path}"
            )
        repaired_labeling = _dict(row.get("phrase_rhythm_repaired_labeling"))
        repaired_metrics = _dict(repaired_labeling.get("metrics"))
        if "technical_gate_regression" in _list(repaired_labeling.get("failure_labels")):
            raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairAudioError(
                "technical gate regression label should not be rendered"
            )
        compacted.append(
            {
                "candidate_index": int(index),
                "source": str(row.get("source") or ""),
                "rank": _int(row.get("rank")) or index,
                "source_rank": _int(row.get("source_rank")),
                "source_midi_path": str(row.get("source_midi_path") or ""),
                "repaired_midi_path": repaired_midi_path,
                "source_failure_labels": _list(row.get("source_failure_labels")),
                "repaired_failure_labels": _list(
                    repaired_labeling.get("failure_labels")
                ),
                "density_pattern": _list(row.get("density_pattern")),
                "note_count": _int(repaired_metrics.get("note_count")),
                "repaired_unique_pitch_count": _int(
                    repaired_metrics.get("unique_pitch_count")
                ),
                "repaired_dead_air_ratio": _float(repaired_metrics.get("dead_air_ratio")),
                "repaired_max_interval": _int(repaired_metrics.get("max_abs_interval")),
                "repaired_max_simultaneous_notes": _int(
                    repaired_metrics.get("max_simultaneous_notes")
                ),
            }
        )
    return {
        "source_schema_versions": source_schema_versions,
        "source_sweep_schema_version": str(source_summary.get("schema_version") or ""),
        "candidates": compacted,
        **source_context,
    }


def _safe_name(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)
    return safe.strip("_") or "candidate"


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
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairAudioError(
            f"renderer not found: {renderer}"
        )
    if not soundfont.exists():
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairAudioError(
            f"soundfont not found: {soundfont}"
        )
    plan: list[dict[str, Any]] = []
    for item in candidates:
        name = _safe_name(str(item["source"]))
        wav_path = output_dir / "audio" / (
            f"candidate_{int(item['candidate_index']):02d}_{name}_rank_{int(item['rank']):02d}_"
            "phrase_rhythm_repair.wav"
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
            raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairAudioError(
                f"render failed for candidate {item['candidate_index']}: "
                f"{completed.stderr or completed.stdout}"
            )
        rendered.append(
            {
                "candidate_index": item["candidate_index"],
                "source": item["source"],
                "rank": item["rank"],
                "source_rank": item["source_rank"],
                "source_midi_path": item["source_midi_path"],
                "repaired_midi_path": item["repaired_midi_path"],
                "source_failure_labels": item["source_failure_labels"],
                "repaired_failure_labels": item["repaired_failure_labels"],
                "density_pattern": item["density_pattern"],
                "note_count": item["note_count"],
                "repaired_unique_pitch_count": item["repaired_unique_pitch_count"],
                "repaired_dead_air_ratio": item["repaired_dead_air_ratio"],
                "repaired_max_interval": item["repaired_max_interval"],
                "repaired_max_simultaneous_notes": item["repaired_max_simultaneous_notes"],
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
    issue_number: int = 1030,
    runner: CommandRunner = default_runner,
) -> dict[str, Any]:
    source = validate_source_report(source_report, expected_count=expected_file_count)
    candidates = [_dict(item) for item in _list(source.get("candidates"))]
    source_context = {
        key: value
        for key, value in source.items()
        if key.startswith("objective_source_outside_soloing_repair_")
        or key.startswith("source_outside_soloing_repair_")
        or key in BRIDGE_REQUIRED_SOURCE_CONTEXT_KEYS
        or key in {f"objective_{source_key}" for source_key in BRIDGE_REQUIRED_SOURCE_CONTEXT_KEYS}
    }
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
    failure_counts = Counter(
        label for item in rendered for label in _list(item.get("repaired_failure_labels"))
    )
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
        "source_schema_versions": source["source_schema_versions"],
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
            "technical_wav_validation": True,
            "sample_rate": int(sample_rate),
            "duration_min_seconds": min(durations, default=0.0),
            "duration_max_seconds": max(durations, default=0.0),
            "source_total_failure_label_count": _int(
                aggregate.get("source_total_failure_label_count")
            ),
            "repaired_total_failure_label_count": _int(
                aggregate.get("repaired_total_failure_label_count")
            ),
            "failure_label_delta": _int(aggregate.get("failure_label_delta")),
            "source_phrase_rhythm_failure_count": _int(
                aggregate.get("source_phrase_rhythm_failure_count")
            ),
            "repaired_phrase_rhythm_failure_count": _int(
                aggregate.get("repaired_phrase_rhythm_failure_count")
            ),
            "phrase_rhythm_failure_delta": _int(aggregate.get("phrase_rhythm_failure_delta")),
            "improved_candidate_count": _int(aggregate.get("improved_candidate_count")),
            "technical_regression_count": _int(aggregate.get("technical_regression_count")),
            "source_outside_soloing_repair_evidence_ready": bool(
                aggregate.get("source_outside_soloing_repair_evidence_ready", False)
            ),
            "source_outside_soloing_repair_pitch_role_risk_count_after": _int(
                aggregate.get("source_outside_soloing_repair_pitch_role_risk_count_after")
            ),
            **source_context,
            "source_outside_soloing_not_evaluable_count": _int(
                aggregate.get("source_outside_soloing_not_evaluable_count")
            ),
            "repaired_outside_soloing_not_evaluable_count": _int(
                aggregate.get("repaired_outside_soloing_not_evaluable_count")
            ),
            "repaired_not_evaluable_counts": _dict(
                aggregate.get("repaired_not_evaluable_counts")
            ),
            "remaining_failure_counts": dict(sorted(failure_counts.items())),
            "audio_review_required": True,
        },
        "audio_render_boundary": {
            "boundary": BOUNDARY,
            "render_attempted": True,
            "rendered_audio_file_count": int(len(rendered)),
            "technical_wav_validation": True,
            "songlike_melody_contour_phrase_rhythm_repair_audio_package_completed": True,
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
            "reason": "phrase/rhythm repair MIDI rendered to WAV; prepare listening review package next",
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
            "Stage B MIDI-to-solo songlike melody contour phrase/rhythm repair listening review package source-context refresh"
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
    source_schema_versions = _dict(report.get("source_schema_versions"))
    if str(report.get("schema_version") or "") != SCHEMA_VERSION:
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairAudioError(
            "phrase/rhythm repair audio package schema version mismatch"
        )
    _validate_source_schema_versions(
        source_schema_versions,
        label="phrase/rhythm repair audio package",
    )
    if expected_boundary and str(boundary.get("boundary") or "") != expected_boundary:
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairAudioError(
            "unexpected boundary"
        )
    if expected_next_boundary and str(decision.get("next_boundary") or "") != expected_next_boundary:
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairAudioError(
            "unexpected next boundary"
        )
    files = [_dict(item) for item in _list(report.get("rendered_audio_files"))]
    if len(files) != int(expected_file_count):
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairAudioError(
            f"expected {expected_file_count} rendered files"
        )
    for item in files:
        wav_file = _dict(item.get("wav_file"))
        if not bool(wav_file.get("exists", False)) or not _path_exists(
            str(wav_file.get("path") or "")
        ):
            raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairAudioError(
                "missing rendered WAV"
            )
        if _int(wav_file.get("sample_rate")) != int(expected_sample_rate):
            raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairAudioError(
                "unexpected sample rate"
            )
        if _int(wav_file.get("frame_count")) <= 0:
            raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairAudioError("empty WAV")
        if _int(wav_file.get("size_bytes")) <= 44:
            raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairAudioError(
                "invalid WAV size"
            )
    if require_audio_package_completed and not bool(
        boundary.get("songlike_melody_contour_phrase_rhythm_repair_audio_package_completed", False)
    ):
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairAudioError(
            "audio package completion required"
        )
    if _int(summary.get("technical_regression_count")) != 0:
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairAudioError(
            "technical regression count should be zero"
        )
    if bool(decision.get("critical_user_input_required", True)):
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairAudioError(
            "critical user input should not be required"
        )
    if require_no_quality_claim:
        _require_no_quality_claim(boundary, label="audio render boundary")
    return {
        "schema_version": str(report.get("schema_version") or ""),
        "source_songlike_melody_contour_phrase_rhythm_repair_sweep_schema_version": str(
            source_schema_versions.get("songlike_melody_contour_phrase_rhythm_repair_sweep")
            or ""
        ),
        "source_songlike_melody_contour_repair_followup_schema_version": str(
            source_schema_versions.get("songlike_melody_contour_repair_followup_decision")
            or ""
        ),
        "source_songlike_melody_contour_repair_objective_next_schema_version": str(
            source_schema_versions.get("songlike_melody_contour_repair_objective_next")
            or ""
        ),
        "source_songlike_melody_contour_repair_sweep_schema_version": str(
            source_schema_versions.get("songlike_melody_contour_repair_sweep") or ""
        ),
        "source_songlike_melody_contour_repair_listening_review_input_guard_schema_version": str(
            source_schema_versions.get(
                "songlike_melody_contour_repair_listening_review_input_guard"
            )
            or ""
        ),
        "source_songlike_melody_contour_repair_listening_review_package_schema_version": str(
            source_schema_versions.get(
                "songlike_melody_contour_repair_listening_review_package"
            )
            or ""
        ),
        "source_songlike_melody_contour_repair_audio_package_schema_version": str(
            source_schema_versions.get("songlike_melody_contour_repair_audio_package") or ""
        ),
        "boundary": str(boundary.get("boundary") or ""),
        "next_boundary": str(decision.get("next_boundary") or ""),
        "render_attempted": bool(boundary.get("render_attempted", False)),
        "rendered_audio_file_count": _int(boundary.get("rendered_audio_file_count")),
        "technical_wav_validation": bool(boundary.get("technical_wav_validation", False)),
        "songlike_melody_contour_phrase_rhythm_repair_audio_package_completed": bool(
            boundary.get("songlike_melody_contour_phrase_rhythm_repair_audio_package_completed", False)
        ),
        "sample_rate": _int(summary.get("sample_rate")),
        "duration_min_seconds": _float(summary.get("duration_min_seconds")),
        "duration_max_seconds": _float(summary.get("duration_max_seconds")),
        "source_total_failure_label_count": _int(
            summary.get("source_total_failure_label_count")
        ),
        "repaired_total_failure_label_count": _int(
            summary.get("repaired_total_failure_label_count")
        ),
        "failure_label_delta": _int(summary.get("failure_label_delta")),
        "source_phrase_rhythm_failure_count": _int(
            summary.get("source_phrase_rhythm_failure_count")
        ),
        "repaired_phrase_rhythm_failure_count": _int(
            summary.get("repaired_phrase_rhythm_failure_count")
        ),
        "phrase_rhythm_failure_delta": _int(summary.get("phrase_rhythm_failure_delta")),
        "improved_candidate_count": _int(summary.get("improved_candidate_count")),
        "technical_regression_count": _int(summary.get("technical_regression_count")),
        "source_outside_soloing_repair_evidence_ready": bool(
            summary.get("source_outside_soloing_repair_evidence_ready", False)
        ),
        "objective_source_outside_soloing_repair_source_objective_pitch_role_risk_count": _int(
            summary.get(
                "objective_source_outside_soloing_repair_source_objective_pitch_role_risk_count"
            )
        ),
        "objective_source_outside_soloing_repair_source_context_preserved": bool(
            summary.get("objective_source_outside_soloing_repair_source_context_preserved", False)
        ),
        "objective_source_outside_soloing_repair_schema_context_preserved": bool(
            summary.get("objective_source_outside_soloing_repair_schema_context_preserved", False)
        ),
        "objective_source_outside_soloing_repair_objective_schema_version": str(
            summary.get("objective_source_outside_soloing_repair_objective_schema_version") or ""
        ),
        "objective_source_outside_soloing_repair_source_pitch_role_risk_count_before": _int(
            summary.get(
                "objective_source_outside_soloing_repair_source_pitch_role_risk_count_before"
            )
        ),
        "objective_source_outside_soloing_repair_source_pitch_role_risk_count_after": _int(
            summary.get(
                "objective_source_outside_soloing_repair_source_pitch_role_risk_count_after"
            )
        ),
        "objective_source_outside_soloing_repair_source_pitch_role_risk_delta": _int(
            summary.get("objective_source_outside_soloing_repair_source_pitch_role_risk_delta")
        ),
        "objective_source_outside_soloing_repair_source_targeted": bool(
            summary.get("objective_source_outside_soloing_repair_source_targeted", True)
        ),
        "objective_source_outside_soloing_repair_source_residual_risk_preserved": bool(
            summary.get(
                "objective_source_outside_soloing_repair_source_residual_risk_preserved",
                False,
            )
        ),
        "objective_source_outside_soloing_repair_pitch_role_risk_count_after": _int(
            summary.get("objective_source_outside_soloing_repair_pitch_role_risk_count_after")
        ),
        "objective_source_outside_soloing_repair_pitch_role_risk_delta": _int(
            summary.get("objective_source_outside_soloing_repair_pitch_role_risk_delta")
        ),
        "source_outside_soloing_repair_pitch_role_risk_count_after": _int(
            summary.get("source_outside_soloing_repair_pitch_role_risk_count_after")
        ),
        "source_outside_soloing_repair_source_objective_pitch_role_risk_count": _int(
            summary.get("source_outside_soloing_repair_source_objective_pitch_role_risk_count")
        ),
        "source_outside_soloing_repair_source_context_preserved": bool(
            summary.get("source_outside_soloing_repair_source_context_preserved", False)
        ),
        "source_outside_soloing_repair_schema_context_preserved": bool(
            summary.get("source_outside_soloing_repair_schema_context_preserved", False)
        ),
        "source_outside_soloing_repair_objective_schema_version": str(
            summary.get("source_outside_soloing_repair_objective_schema_version") or ""
        ),
        "source_outside_soloing_repair_source_pitch_role_risk_count_before": _int(
            summary.get("source_outside_soloing_repair_source_pitch_role_risk_count_before")
        ),
        "source_outside_soloing_repair_source_pitch_role_risk_count_after": _int(
            summary.get("source_outside_soloing_repair_source_pitch_role_risk_count_after")
        ),
        "source_outside_soloing_repair_source_pitch_role_risk_delta": _int(
            summary.get("source_outside_soloing_repair_source_pitch_role_risk_delta")
        ),
        "source_outside_soloing_repair_source_targeted": bool(
            summary.get("source_outside_soloing_repair_source_targeted", True)
        ),
        "source_outside_soloing_repair_source_residual_risk_preserved": bool(
            summary.get("source_outside_soloing_repair_source_residual_risk_preserved", False)
        ),
        "source_outside_soloing_repair_pitch_role_risk_delta": _int(
            summary.get("source_outside_soloing_repair_pitch_role_risk_delta")
        ),
        **{
            f"objective_{key}": summary.get(f"objective_{key}")
            for key in BRIDGE_REQUIRED_SOURCE_CONTEXT_KEYS
        },
        **{key: summary.get(key) for key in BRIDGE_REQUIRED_SOURCE_CONTEXT_KEYS},
        "source_outside_soloing_not_evaluable_count": _int(
            summary.get("source_outside_soloing_not_evaluable_count")
        ),
        "repaired_outside_soloing_not_evaluable_count": _int(
            summary.get("repaired_outside_soloing_not_evaluable_count")
        ),
        "repaired_not_evaluable_counts": _dict(
            summary.get("repaired_not_evaluable_counts")
        ),
        "audio_review_required": bool(summary.get("audio_review_required", False)),
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
        "# Stage B MIDI-to-Solo Songlike Melody Contour Phrase/Rhythm Repair Audio Package",
        "",
        "## Summary",
        "",
        f"- boundary: `{boundary['boundary']}`",
        f"- schema version: `{report['schema_version']}`",
        f"- source phrase/rhythm repair sweep schema version: `{report['source_schema_versions']['songlike_melody_contour_phrase_rhythm_repair_sweep']}`",
        f"- source follow-up schema version: `{report['source_schema_versions']['songlike_melody_contour_repair_followup_decision']}`",
        f"- source objective next schema version: `{report['source_schema_versions']['songlike_melody_contour_repair_objective_next']}`",
        f"- source repair sweep schema version: `{report['source_schema_versions']['songlike_melody_contour_repair_sweep']}`",
        f"- next boundary: `{decision['next_boundary']}`",
        f"- render attempted: `{_bool_token(boundary['render_attempted'])}`",
        f"- rendered audio file count: `{boundary['rendered_audio_file_count']}`",
        f"- technical WAV validation: `{_bool_token(boundary['technical_wav_validation'])}`",
        f"- duration range: `{summary['duration_min_seconds']:.3f}s-{summary['duration_max_seconds']:.3f}s`",
        f"- failure labels: `{summary['source_total_failure_label_count']} -> {summary['repaired_total_failure_label_count']}`",
        f"- phrase/rhythm failure count: `{summary['source_phrase_rhythm_failure_count']} -> {summary['repaired_phrase_rhythm_failure_count']}`",
        f"- phrase/rhythm failure delta: `{summary['phrase_rhythm_failure_delta']}`",
        f"- improved candidate count: `{summary['improved_candidate_count']}`",
        f"- technical regression count: `{summary['technical_regression_count']}`",
        f"- source outside-soloing repair evidence ready: `{_bool_token(summary['source_outside_soloing_repair_evidence_ready'])}`",
        f"- objective source outside-soloing source context preserved: `{_bool_token(summary['objective_source_outside_soloing_repair_source_context_preserved'])}`",
        f"- objective source outside-soloing schema context preserved: `{_bool_token(summary['objective_source_outside_soloing_repair_schema_context_preserved'])}`",
        f"- objective source outside-soloing objective schema version: `{summary['objective_source_outside_soloing_repair_objective_schema_version']}`",
        f"- objective source outside-soloing source pitch-role risk before / after / delta: `{summary['objective_source_outside_soloing_repair_source_pitch_role_risk_count_before']}` / `{summary['objective_source_outside_soloing_repair_source_pitch_role_risk_count_after']}` / `{summary['objective_source_outside_soloing_repair_source_pitch_role_risk_delta']}`",
        f"- objective source outside-soloing current repair pitch-role risk after / delta: `{summary['objective_source_outside_soloing_repair_pitch_role_risk_count_after']}` / `{summary['objective_source_outside_soloing_repair_pitch_role_risk_delta']}`",
        f"- source outside-soloing source context preserved: `{_bool_token(summary['source_outside_soloing_repair_source_context_preserved'])}`",
        f"- source outside-soloing schema context preserved: `{_bool_token(summary['source_outside_soloing_repair_schema_context_preserved'])}`",
        f"- source outside-soloing objective schema version: `{summary['source_outside_soloing_repair_objective_schema_version']}`",
        f"- follow-up objective source outside-soloing source context preserved: `{_bool_token(summary['followup_objective_source_outside_soloing_source_context_preserved'])}`",
        f"- follow-up repair sweep source outside-soloing source context preserved: `{_bool_token(summary['followup_repair_sweep_source_outside_soloing_source_context_preserved'])}`",
        f"- bridge repair sweep source outside-soloing source context preserved: `{_bool_token(summary['repair_sweep_source_outside_soloing_source_context_preserved'])}`",
        f"- source outside-soloing source pitch-role risk before / after / delta: `{summary['source_outside_soloing_repair_source_pitch_role_risk_count_before']}` / `{summary['source_outside_soloing_repair_source_pitch_role_risk_count_after']}` / `{summary['source_outside_soloing_repair_source_pitch_role_risk_delta']}`",
        f"- source outside-soloing source repair targeted: `{_bool_token(summary['source_outside_soloing_repair_source_targeted'])}`",
        f"- source outside-soloing source residual risk preserved: `{_bool_token(summary['source_outside_soloing_repair_source_residual_risk_preserved'])}`",
        f"- source outside-soloing repair pitch-role risk after: `{summary['source_outside_soloing_repair_pitch_role_risk_count_after']}`",
        f"- source outside-soloing current repair pitch-role risk delta: `{summary['source_outside_soloing_repair_pitch_role_risk_delta']}`",
        f"- source outside-soloing not evaluable count: `{summary['source_outside_soloing_not_evaluable_count']}`",
        f"- repaired outside-soloing not evaluable count: `{summary['repaired_outside_soloing_not_evaluable_count']}`",
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
            f"sample rate `{wav_file['sample_rate']}`, failure labels "
            f"`{','.join(item['repaired_failure_labels']) or 'none'}`, "
            f"sha256 `{str(wav_file['sha256'])[:12]}`"
        )
    lines.extend(["", "## Repaired Not Evaluable Counts", ""])
    for label, count in summary["repaired_not_evaluable_counts"].items():
        lines.append(f"- `{label}`: `{count}`")
    lines.extend(["", "## Claim Boundary", ""])
    for item in report.get("not_proven", []):
        lines.append(f"- `{item}`")
    lines.extend(["", "## Next", "", f"- `{report['next_recommended_issue']}`"])
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render songlike melody contour phrase/rhythm repair MIDI candidates to WAV files"
    )
    parser.add_argument("--phrase_rhythm_repair_sweep_report", type=str, required=True)
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_repair_audio_package",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--issue_number", type=int, default=1198)
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
        read_json(Path(args.phrase_rhythm_repair_sweep_report)),
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
        / "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_repair_audio_package.json",
        report,
    )
    write_json(
        output_dir
        / "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_repair_audio_package_validation_summary.json",
        summary,
    )
    markdown = markdown_report(report)
    write_text(
        output_dir / "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_repair_audio_package.md",
        markdown,
    )
    if args.doc_path:
        write_text(Path(args.doc_path), markdown)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
