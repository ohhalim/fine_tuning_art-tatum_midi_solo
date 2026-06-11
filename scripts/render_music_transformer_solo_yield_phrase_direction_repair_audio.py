"""Render phrase-direction repaired solo-yield MIDI candidates to WAV."""

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
    resolve_soundfont,
    sha256_file,
    wav_meta,
)
from scripts.run_music_transformer_solo_yield_phrase_direction_repair_sweep import (  # noqa: E402
    SCHEMA_VERSION as PHRASE_DIRECTION_REPAIR_SCHEMA_VERSION,
    build_repair_sweep,
    read_json,
)


SCHEMA_VERSION = "music_transformer_solo_yield_phrase_direction_repair_audio_package_v1"
BOUNDARY = "music_transformer_solo_yield_phrase_direction_repair_audio_package"
NEXT_BOUNDARY = "music_transformer_solo_yield_phrase_direction_repair_listening_package"
CommandRunner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]


class SoloYieldPhraseDirectionRepairAudioPackageError(ValueError):
    pass


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


def _require_no_quality_claim(report: dict[str, Any]) -> None:
    readiness = _dict(report.get("readiness"))
    claimed = [
        key
        for key in (
            "audio_rendered_quality_claimed",
            "musical_quality_claimed",
            "artist_style_claimed",
            "production_ready_claimed",
        )
        if bool(readiness.get(key, False))
    ]
    if claimed:
        raise SoloYieldPhraseDirectionRepairAudioPackageError(f"unexpected quality claim: {claimed}")


def _validate_repair_sweep(report: dict[str, Any]) -> list[dict[str, Any]]:
    if str(report.get("schema_version") or "") != PHRASE_DIRECTION_REPAIR_SCHEMA_VERSION:
        raise SoloYieldPhraseDirectionRepairAudioPackageError("phrase direction repair schema required")
    _require_no_quality_claim(report)
    aggregate = _dict(report.get("aggregate"))
    if not bool(aggregate.get("target_supported", False)):
        raise SoloYieldPhraseDirectionRepairAudioPackageError("target support required")
    rows = [_dict(item) for item in _list(report.get("candidate_repairs"))]
    if not rows:
        raise SoloYieldPhraseDirectionRepairAudioPackageError("candidate repairs required")
    for row in rows:
        midi_path = Path(str(row.get("repaired_midi_path") or ""))
        if not midi_path.exists():
            raise SoloYieldPhraseDirectionRepairAudioPackageError(f"repaired MIDI missing: {midi_path}")
    return rows


def load_or_build_repair_sweep(
    *,
    repair_sweep_report_path: Path,
    output_dir: Path,
    source_repair_sweep_report_path: Path,
    objective_decision_report_path: Path,
) -> dict[str, Any]:
    if repair_sweep_report_path.exists():
        return read_json(repair_sweep_report_path)
    return build_repair_sweep(
        source_repair_sweep=read_json(source_repair_sweep_report_path),
        objective_decision=read_json(objective_decision_report_path),
        output_dir=output_dir / "source_phrase_direction_repair",
    )


def build_render_plan(
    repairs: list[dict[str, Any]],
    *,
    output_dir: Path,
    renderer_path: str,
    soundfont_path: str,
    sample_rate: int,
) -> list[dict[str, Any]]:
    renderer = Path(renderer_path)
    soundfont = Path(soundfont_path).expanduser()
    if not renderer.exists():
        raise SoloYieldPhraseDirectionRepairAudioPackageError(f"renderer not found: {renderer}")
    if not soundfont.exists():
        raise SoloYieldPhraseDirectionRepairAudioPackageError(f"soundfont not found: {soundfont}")
    plan: list[dict[str, Any]] = []
    for row in repairs:
        review_index = _int(row.get("review_index"))
        case_label = str(row.get("case_label") or "candidate")
        source_midi = Path(str(row.get("repaired_midi_path") or ""))
        wav_path = output_dir / "audio" / f"candidate_{review_index:02d}_{case_label}_phrase_direction_repair.wav"
        plan.append(
            {
                "review_index": review_index,
                "case_label": case_label,
                "repaired_midi_path": str(source_midi),
                "source_midi_path": str(row.get("source_midi_path") or ""),
                "wav_path": str(wav_path),
                "command": [
                    str(renderer),
                    "-ni",
                    "-F",
                    str(wav_path),
                    "-r",
                    str(int(sample_rate)),
                    str(soundfont),
                    str(source_midi),
                ],
            }
        )
    return plan


def default_runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(list(command), check=False, text=True, capture_output=True)


def execute_render_plan(plan: list[dict[str, Any]], *, runner: CommandRunner = default_runner) -> list[dict[str, Any]]:
    rendered: list[dict[str, Any]] = []
    for item in plan:
        wav_path = Path(str(item["wav_path"]))
        wav_path.parent.mkdir(parents=True, exist_ok=True)
        completed = runner(item["command"])
        if completed.returncode != 0:
            raise SoloYieldPhraseDirectionRepairAudioPackageError(
                f"render failed for candidate {item['review_index']}: {completed.stderr or completed.stdout}"
            )
        rendered.append(
            {
                "review_index": item["review_index"],
                "case_label": item["case_label"],
                "source_midi_path": item["source_midi_path"],
                "repaired_midi_path": item["repaired_midi_path"],
                "wav_file": wav_meta(wav_path),
                "command": list(item["command"]),
                "stdout_tail": (completed.stdout or "")[-1000:],
                "stderr_tail": (completed.stderr or "")[-1000:],
            }
        )
    return rendered


def build_audio_package(
    repair_sweep: dict[str, Any],
    *,
    output_dir: Path,
    renderer_path: str,
    soundfont_path: str,
    sample_rate: int,
    runner: CommandRunner = default_runner,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    repairs = _validate_repair_sweep(repair_sweep)
    resolved_renderer = renderer_path or shutil.which("fluidsynth") or ""
    resolved_soundfont = resolve_soundfont(soundfont_path)
    plan = build_render_plan(
        repairs,
        output_dir=output_dir,
        renderer_path=resolved_renderer,
        soundfont_path=resolved_soundfont,
        sample_rate=int(sample_rate),
    )
    rendered = execute_render_plan(plan, runner=runner)
    durations = [_float(_dict(item.get("wav_file")).get("duration_seconds")) for item in rendered]
    source_aggregate = _dict(repair_sweep.get("aggregate"))
    report = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "boundary": BOUNDARY,
        "source_repair_sweep": {
            "schema_version": repair_sweep.get("schema_version"),
            "output_dir": repair_sweep.get("output_dir"),
            "candidate_count": _int(source_aggregate.get("candidate_count")),
            "repaired_midi_count": _int(source_aggregate.get("repaired_midi_count")),
            "weak_direction_change_count_before": _int(
                source_aggregate.get("weak_direction_change_count_before")
            ),
            "weak_direction_change_count_after": _int(
                source_aggregate.get("weak_direction_change_count_after")
            ),
            "chord_tone_ratio_decrease_count": _int(
                source_aggregate.get("chord_tone_ratio_decrease_count")
            ),
            "final_landing_not_chord_tone_count_after": _int(
                source_aggregate.get("final_landing_not_chord_tone_count_after")
            ),
        },
        "renderer": {
            "name": "fluidsynth",
            "path": resolved_renderer,
        },
        "soundfont": {
            "path": str(Path(resolved_soundfont).expanduser()),
            "sha256": sha256_file(Path(resolved_soundfont).expanduser()),
        },
        "rendered_audio_files": rendered,
        "aggregate": {
            "rendered_wav_count": len(rendered),
            "technical_wav_validation": len(rendered) == len(repairs),
            "wav_duration_min_seconds": min(durations) if durations else 0.0,
            "wav_duration_max_seconds": max(durations) if durations else 0.0,
            "sample_rate": int(sample_rate),
        },
        "readiness": {
            "audio_package_completed": True,
            "technical_wav_validation": len(rendered) == len(repairs),
            "validated_listening_input_present": False,
            "audio_rendered_quality_claimed": False,
            "musical_quality_claimed": False,
            "artist_style_claimed": False,
            "production_ready_claimed": False,
        },
        "decision": {
            "current_boundary": BOUNDARY,
            "selected_next_target": "phrase_direction_repair_listening_package",
            "next_boundary": NEXT_BOUNDARY,
            "critical_user_input_required": False,
            "reason": "phrase direction repaired MIDI rendered to WAV for technical listening package",
        },
        "not_proven": [
            "audio_rendered_quality",
            "human_audio_preference",
            "stable_jazz_solo_quality",
            "artist_level_long_solo_generation",
            "production_ready_improviser",
        ],
    }
    write_json(output_dir / "phrase_direction_repair_audio_package.json", report)
    write_json(output_dir / "phrase_direction_repair_audio_package_summary.json", validate_report(report))
    write_text(output_dir / "phrase_direction_repair_audio_package.md", markdown_report(report))
    return report


def validate_report(report: dict[str, Any], *, min_wav_count: int = 1) -> dict[str, Any]:
    if str(report.get("schema_version") or "") != SCHEMA_VERSION:
        raise SoloYieldPhraseDirectionRepairAudioPackageError("schema version mismatch")
    readiness = _dict(report.get("readiness"))
    aggregate = _dict(report.get("aggregate"))
    rendered_count = _int(aggregate.get("rendered_wav_count"))
    if rendered_count < int(min_wav_count):
        raise SoloYieldPhraseDirectionRepairAudioPackageError("rendered WAV count below requirement")
    if not bool(readiness.get("audio_package_completed", False)):
        raise SoloYieldPhraseDirectionRepairAudioPackageError("audio package completion required")
    if not bool(aggregate.get("technical_wav_validation", False)):
        raise SoloYieldPhraseDirectionRepairAudioPackageError("technical WAV validation required")
    _require_no_quality_claim(report)
    decision = _dict(report.get("decision"))
    return {
        "schema_version": str(report.get("schema_version") or ""),
        "rendered_wav_count": rendered_count,
        "technical_wav_validation": bool(aggregate.get("technical_wav_validation", False)),
        "wav_duration_min_seconds": _float(aggregate.get("wav_duration_min_seconds")),
        "wav_duration_max_seconds": _float(aggregate.get("wav_duration_max_seconds")),
        "sample_rate": _int(aggregate.get("sample_rate")),
        "audio_rendered_quality_claimed": bool(readiness.get("audio_rendered_quality_claimed", True)),
        "musical_quality_claimed": bool(readiness.get("musical_quality_claimed", True)),
        "next_boundary": str(decision.get("next_boundary") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    source = report["source_repair_sweep"]
    aggregate = report["aggregate"]
    readiness = report["readiness"]
    decision = report["decision"]
    lines = [
        "# Music Transformer Solo Yield Phrase Direction Repair Audio Package",
        "",
        "## Summary",
        "",
        f"- source repaired MIDI count: `{source['repaired_midi_count']}`",
        f"- source weak direction-change: `{source['weak_direction_change_count_before']} -> {source['weak_direction_change_count_after']}`",
        f"- source chord-tone ratio decrease count: `{source['chord_tone_ratio_decrease_count']}`",
        f"- source final landing not chord-tone after: `{source['final_landing_not_chord_tone_count_after']}`",
        f"- rendered WAV count: `{aggregate['rendered_wav_count']}`",
        f"- technical WAV validation: `{_bool_token(aggregate['technical_wav_validation'])}`",
        f"- WAV duration range: `{float(aggregate['wav_duration_min_seconds']):.3f} - {float(aggregate['wav_duration_max_seconds']):.3f}`",
        f"- audio rendered quality claimed: `{_bool_token(readiness['audio_rendered_quality_claimed'])}`",
        f"- musical quality claimed: `{_bool_token(readiness['musical_quality_claimed'])}`",
        f"- next boundary: `{decision['next_boundary']}`",
        "",
        "## Rendered Files",
        "",
    ]
    for item in report.get("rendered_audio_files", []):
        wav = item["wav_file"]
        lines.extend(
            [
                f"- candidate `{item['review_index']}` / `{item['case_label']}`",
                f"  - MIDI: `{item['repaired_midi_path']}`",
                f"  - WAV: `{wav['path']}`",
                f"  - duration: `{float(wav['duration_seconds']):.3f}`",
                f"  - sample rate: `{wav['sample_rate']}`",
            ]
        )
    lines.extend(["", "## Not Proven", ""])
    for item in report.get("not_proven", []):
        lines.append(f"- `{item}`")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render phrase-direction repaired MIDI to WAV")
    parser.add_argument(
        "--repair_sweep_report",
        type=str,
        default=(
            "outputs/music_transformer_finetune_mvp/solo_yield_phrase_direction_repair/"
            "issue_1274_phrase_direction_repair_sweep/phrase_direction_repair_sweep.json"
        ),
    )
    parser.add_argument(
        "--source_repair_sweep_report",
        type=str,
        default=(
            "outputs/music_transformer_finetune_mvp/solo_yield_chord_tone_landing_repair/"
            "issue_1264_chord_tone_landing_repair/chord_tone_landing_repair_sweep.json"
        ),
    )
    parser.add_argument(
        "--objective_decision_report",
        type=str,
        default=(
            "outputs/music_transformer_finetune_mvp/solo_yield_chord_tone_landing_repair_objective_next/"
            "issue_1272_chord_tone_landing_objective_next/"
            "chord_tone_landing_objective_next_decision.json"
        ),
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/music_transformer_finetune_mvp/solo_yield_phrase_direction_repair_audio",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--renderer", type=str, default=shutil.which("fluidsynth") or "")
    parser.add_argument("--soundfont", type=str, default="")
    parser.add_argument("--sample_rate", type=int, default=44100)
    parser.add_argument("--min_wav_count", type=int, default=8)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    repair_sweep = load_or_build_repair_sweep(
        repair_sweep_report_path=Path(args.repair_sweep_report),
        output_dir=output_dir,
        source_repair_sweep_report_path=Path(args.source_repair_sweep_report),
        objective_decision_report_path=Path(args.objective_decision_report),
    )
    report = build_audio_package(
        repair_sweep,
        output_dir=output_dir,
        renderer_path=str(args.renderer or ""),
        soundfont_path=str(args.soundfont or ""),
        sample_rate=int(args.sample_rate),
    )
    summary = validate_report(report, min_wav_count=int(args.min_wav_count))
    write_json(output_dir / "phrase_direction_repair_audio_package_summary.json", summary)
    if args.doc_path:
        write_text(Path(args.doc_path), markdown_report(report))
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
