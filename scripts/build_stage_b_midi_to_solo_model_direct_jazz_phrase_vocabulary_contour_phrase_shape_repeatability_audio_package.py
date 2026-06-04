"""Build WAV package for contour phrase-shape repeatability MIDI candidates."""

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
from scripts.consolidate_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_repeatability import (  # noqa: E402
    BOUNDARY as SOURCE_BOUNDARY,
)
from scripts.render_stage_b_midi_to_solo_candidate_audio import (  # noqa: E402
    default_soundfont_candidates,
    resolve_soundfont,
    sha256_file,
    wav_meta,
)


class StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeRepeatabilityAudioPackageError(
    ValueError
):
    pass


BOUNDARY = (
    "stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_"
    "repeatability_audio_review_package"
)
NEXT_BOUNDARY = (
    "stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_"
    "repeatability_listening_review"
)
SCHEMA_VERSION = (
    "stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_"
    "repeatability_audio_package_v1"
)
CommandRunner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeRepeatabilityAudioPackageError(
            f"report missing: {path}"
        )
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


def validate_consolidation_report(consolidation_report: dict[str, Any], expected_count: int) -> list[dict[str, Any]]:
    if str(consolidation_report.get("boundary") or "") != SOURCE_BOUNDARY:
        raise StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeRepeatabilityAudioPackageError(
            "repeatability consolidation boundary required"
        )
    evidence = _dict(consolidation_report.get("evidence_summary"))
    result = _dict(consolidation_report.get("consolidation_result"))
    readiness = _dict(consolidation_report.get("readiness"))
    decision = _dict(consolidation_report.get("decision"))
    if str(decision.get("next_boundary") or "") != BOUNDARY:
        raise StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeRepeatabilityAudioPackageError(
            "repeatability consolidation must route to audio package"
        )
    if not bool(result.get("objective_repeatability_support", False)):
        raise StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeRepeatabilityAudioPackageError(
            "objective repeatability support required"
        )
    if not bool(result.get("audio_review_package_required", False)):
        raise StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeRepeatabilityAudioPackageError(
            "audio review package requirement required"
        )
    if _int(evidence.get("generated_midi_file_count")) != int(expected_count):
        raise StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeRepeatabilityAudioPackageError(
            "unexpected generated MIDI count"
        )
    blocked_claims = [
        "human_audio_preference_claimed",
        "model_direct_generation_quality_claimed",
        "midi_to_solo_musical_quality_claimed",
        "broad_trained_model_quality_claimed",
        "brad_style_adaptation_claimed",
    ]
    claimed = [name for name in blocked_claims if bool(readiness.get(name, False))]
    if claimed:
        raise StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeRepeatabilityAudioPackageError(
            f"unexpected upstream claim: {claimed}"
        )
    midi_paths = [Path(str(path)) for path in _list(evidence.get("generated_midi_paths"))]
    if len(midi_paths) != int(expected_count):
        raise StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeRepeatabilityAudioPackageError(
            "generated MIDI path count mismatch"
        )
    candidates: list[dict[str, Any]] = []
    for index, midi_path in enumerate(midi_paths, start=1):
        if not midi_path.exists():
            raise StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeRepeatabilityAudioPackageError(
                f"generated MIDI missing: {midi_path}"
            )
        candidates.append(
            {
                "rank": index,
                "midi_path": str(midi_path),
                "midi_sha256": sha256_file(midi_path),
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
        raise StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeRepeatabilityAudioPackageError(
            f"renderer not found: {renderer}"
        )
    if not soundfont.exists():
        raise StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeRepeatabilityAudioPackageError(
            f"soundfont not found: {soundfont}"
        )
    plan: list[dict[str, Any]] = []
    for item in candidates:
        wav_path = output_dir / "audio" / f"contour_phrase_shape_repeatability_seed_{int(item['rank']):02d}.wav"
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
            raise StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeRepeatabilityAudioPackageError(
                f"render failed for rank {item['rank']}: {completed.stderr or completed.stdout}"
            )
        rendered.append(
            {
                "rank": item["rank"],
                "midi_path": item["midi_path"],
                "midi_sha256": item["midi_sha256"],
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
    candidates = validate_consolidation_report(consolidation_report, expected_count=expected_file_count)
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
        "model_direct_generation_quality_claimed": False,
        "midi_to_solo_musical_quality_claimed": False,
        "broad_trained_model_quality_claimed": False,
        "brad_style_adaptation_claimed": False,
        "duration_seconds_min": min(durations) if durations else 0.0,
        "duration_seconds_max": max(durations) if durations else 0.0,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "source_boundary": SOURCE_BOUNDARY,
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
        "audio_package_boundary": boundary,
        "decision": {
            "current_boundary": BOUNDARY,
            "next_boundary": NEXT_BOUNDARY,
            "auto_progress_allowed": True,
            "critical_user_input_required": False,
            "reason": "objective-clean repeatability MIDI variants rendered to WAV for listening review",
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
        "next_recommended_issue": (
            "Stage B MIDI-to-solo model-direct jazz phrase vocabulary contour phrase-shape repeatability listening review"
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
        raise StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeRepeatabilityAudioPackageError(
            f"expected boundary {expected_boundary}, got {boundary.get('boundary')}"
        )
    if len(files) != int(expected_file_count):
        raise StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeRepeatabilityAudioPackageError(
            "unexpected rendered file count"
        )
    for item in files:
        wav_file = _dict(_dict(item).get("wav_file"))
        if not bool(wav_file.get("exists", False)):
            raise StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeRepeatabilityAudioPackageError(
                "missing rendered WAV"
            )
        if _int(wav_file.get("sample_rate")) != int(expected_sample_rate):
            raise StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeRepeatabilityAudioPackageError(
                "unexpected sample rate"
            )
        if _int(wav_file.get("frame_count")) <= 0:
            raise StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeRepeatabilityAudioPackageError(
                "empty WAV"
            )
        if _int(wav_file.get("size_bytes")) <= 44:
            raise StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeRepeatabilityAudioPackageError(
                "invalid WAV size"
            )
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
            raise StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeRepeatabilityAudioPackageError(
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
        "# Stage B MIDI-to-Solo Model-Direct Jazz Phrase Vocabulary Contour Phrase-Shape Repeatability Audio Package",
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
    parser = argparse.ArgumentParser(description="Build contour phrase-shape repeatability WAV package")
    parser.add_argument("--repeatability_consolidation", type=str, required=True)
    parser.add_argument(
        "--output_root",
        type=str,
        default=(
            "outputs/stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_"
            "contour_phrase_shape_repeatability_audio_package"
        ),
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--renderer", type=str, default=shutil.which("fluidsynth") or "")
    parser.add_argument("--soundfont", type=str, default="")
    parser.add_argument("--sample_rate", type=int, default=44100)
    parser.add_argument("--expected_boundary", type=str, default="")
    parser.add_argument("--expected_file_count", type=int, default=6)
    parser.add_argument("--require_no_quality_claim", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    report = build_audio_package_report(
        read_json(Path(args.repeatability_consolidation)),
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
        output_dir / "stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_repeatability_audio_package.json",
        report,
    )
    write_json(
        output_dir
        / "stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_repeatability_audio_package_validation_summary.json",
        summary,
    )
    markdown = markdown_report(report)
    write_text(
        output_dir / "stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_repeatability_audio_package.md",
        markdown,
    )
    if args.doc_path:
        write_text(Path(args.doc_path), markdown)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
