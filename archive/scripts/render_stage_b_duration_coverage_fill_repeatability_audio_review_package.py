"""Render repeatability source candidates for duration/coverage fill audio review."""

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

from scripts.render_stage_b_duration_coverage_fill_audio import (
    build_audio_render_report,
    validate_audio_render_report,
)


class StageBDurationCoverageRepeatabilityAudioReviewPackageError(ValueError):
    pass


CommandRunner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def review_items_from_repair(dead_air_gain_repair: dict[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for result in _list(dead_air_gain_repair.get("source_repeatability_results")):
        if not isinstance(result, dict):
            continue
        midi_path = str(result.get("selected_midi_path") or "")
        if not midi_path or not Path(midi_path).exists():
            raise StageBDurationCoverageRepeatabilityAudioReviewPackageError(
                f"selected MIDI not found: {midi_path}"
            )
        sample_seed = int(result.get("sample_seed", 0) or 0)
        items.append(
            {
                "role": f"repeatability_sample_seed_{sample_seed}_duration_fill",
                "candidate_id": str(result.get("selected_candidate_id") or ""),
                "source_candidate_id": str(result.get("source_candidate_id") or ""),
                "sample_seed": sample_seed,
                "midi_file": {
                    "path": midi_path,
                    "required": True,
                },
                "metrics": {
                    "baseline_dead_air_ratio": float(result.get("baseline_dead_air_ratio", 0.0) or 0.0),
                    "selected_dead_air_ratio": float(result.get("selected_dead_air_ratio", 0.0) or 0.0),
                    "dead_air_delta_from_baseline": float(result.get("dead_air_delta_from_baseline", 0.0) or 0.0),
                    "selected_focused_note_count": int(result.get("selected_focused_note_count", 0) or 0),
                    "selected_focused_unique_pitch_count": int(
                        result.get("selected_focused_unique_pitch_count", 0) or 0
                    ),
                    "selected_adjacent_pitch_repeats": int(
                        result.get("selected_adjacent_pitch_repeats", 0) or 0
                    ),
                    "selected_max_interval": int(result.get("selected_max_interval", 0) or 0),
                },
            }
        )
    if len(items) != 2:
        raise StageBDurationCoverageRepeatabilityAudioReviewPackageError(
            f"expected 2 repeatability review items, got {len(items)}"
        )
    return items


def build_local_audio_render_package(
    repeatability_consolidation: dict[str, Any],
    dead_air_gain_repair: dict[str, Any],
) -> dict[str, Any]:
    consolidation_boundary = _dict(repeatability_consolidation.get("consolidated_claim_boundary"))
    repair_summary = _dict(dead_air_gain_repair.get("repair_summary"))
    if str(consolidation_boundary.get("boundary") or "") != (
        "current_keep_and_distinct_source_dead_air_gain_midi_support"
    ):
        raise StageBDurationCoverageRepeatabilityAudioReviewPackageError("repeatability consolidation boundary required")
    if bool(consolidation_boundary.get("broad_model_quality_claimed", True)):
        raise StageBDurationCoverageRepeatabilityAudioReviewPackageError("broad model quality must not be claimed")
    if str(repair_summary.get("boundary") or "") != "qualified_gate_repeatability_with_dead_air_gain":
        raise StageBDurationCoverageRepeatabilityAudioReviewPackageError("dead-air gain repair boundary required")
    return {
        "schema_version": "stage_b_duration_coverage_fill_repeatability_audio_review_local_package_v1",
        "candidate_id": "duration_coverage_fill_repeatability_sources",
        "review_items": review_items_from_repair(dead_air_gain_repair),
    }


def build_repeatability_audio_review_package(
    *,
    repeatability_consolidation: dict[str, Any],
    dead_air_gain_repair: dict[str, Any],
    output_dir: Path,
    renderer_path: str,
    soundfont_path: str,
    sample_rate: int,
    runner: CommandRunner | None = None,
) -> dict[str, Any]:
    local_package = build_local_audio_render_package(repeatability_consolidation, dead_air_gain_repair)
    render_kwargs: dict[str, Any] = {}
    if runner is not None:
        render_kwargs["runner"] = runner
    render_report = build_audio_render_report(
        local_package,
        output_dir=output_dir,
        renderer_path=renderer_path,
        soundfont_path=soundfont_path,
        sample_rate=int(sample_rate),
        **render_kwargs,
    )
    render_summary = validate_audio_render_report(
        render_report,
        expected_file_count=2,
        expected_sample_rate=int(sample_rate),
        require_no_quality_claim=True,
    )
    rendered_by_role = {
        str(item.get("role") or ""): item for item in _list(render_report.get("rendered_audio_files"))
    }
    review_items = []
    for item in local_package["review_items"]:
        rendered = _dict(rendered_by_role.get(item["role"]))
        review_items.append(
            {
                "role": item["role"],
                "candidate_id": item["candidate_id"],
                "source_candidate_id": item["source_candidate_id"],
                "sample_seed": item["sample_seed"],
                "midi_file": item["midi_file"],
                "wav_file": _dict(rendered.get("wav_file")),
                "metrics": item["metrics"],
            }
        )
    return {
        "schema_version": "stage_b_duration_coverage_fill_repeatability_audio_review_package_v1",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "source_schemas": {
            "repeatability_consolidation": str(repeatability_consolidation.get("schema_version") or ""),
            "dead_air_gain_repair": str(dead_air_gain_repair.get("schema_version") or ""),
            "render_report": str(render_report.get("schema_version") or ""),
        },
        "candidate_id": "duration_coverage_fill_repeatability_sources",
        "review_items": review_items,
        "audio_render_summary": render_summary,
        "audio_review_boundary": {
            "status": "ready_for_user_listening_review",
            "render_attempted": True,
            "rendered_audio_file_count": int(render_summary["rendered_audio_file_count"]),
            "technical_wav_validation": True,
            "audio_rendered_quality_claimed": False,
            "human_audio_preference_claimed": False,
            "broad_model_quality_claimed": False,
        },
        "not_proven": [
            "audio_rendered_quality",
            "human_audio_preference",
            "multi_reviewer_preference",
            "broad_trained_model_quality",
            "brad_style_adaptation",
            "production_ready_improviser",
        ],
        "next_recommended_issue": (
            "Stage B margin-recovered phrase/vocabulary duration coverage fill repeatability user listening review fill"
        ),
    }


def validate_repeatability_audio_review_package(
    report: dict[str, Any],
    *,
    expected_file_count: int,
    expected_sample_rate: int,
    require_no_quality_claim: bool,
) -> dict[str, Any]:
    boundary = _dict(report.get("audio_review_boundary"))
    items = _list(report.get("review_items"))
    if len(items) != int(expected_file_count):
        raise StageBDurationCoverageRepeatabilityAudioReviewPackageError(
            f"expected {int(expected_file_count)} review items"
        )
    for item in items:
        wav_file = _dict(item.get("wav_file"))
        if not bool(wav_file.get("exists", False)):
            raise StageBDurationCoverageRepeatabilityAudioReviewPackageError(f"missing wav for {item.get('role')}")
        if int(wav_file.get("sample_rate", 0) or 0) != int(expected_sample_rate):
            raise StageBDurationCoverageRepeatabilityAudioReviewPackageError(
                f"unexpected sample rate for {item.get('role')}"
            )
    if require_no_quality_claim:
        blocked = [
            "audio_rendered_quality_claimed",
            "human_audio_preference_claimed",
            "broad_model_quality_claimed",
        ]
        claimed = [name for name in blocked if bool(boundary.get(name, True))]
        if claimed:
            raise StageBDurationCoverageRepeatabilityAudioReviewPackageError(f"unexpected audio claim: {claimed}")
    return {
        "candidate_id": str(report.get("candidate_id") or ""),
        "status": str(boundary.get("status") or ""),
        "render_attempted": bool(boundary.get("render_attempted", False)),
        "rendered_audio_file_count": int(boundary.get("rendered_audio_file_count", 0) or 0),
        "technical_wav_validation": bool(boundary.get("technical_wav_validation", False)),
        "audio_rendered_quality_claimed": bool(boundary.get("audio_rendered_quality_claimed", True)),
        "human_audio_preference_claimed": bool(boundary.get("human_audio_preference_claimed", True)),
        "wav_paths": [str(_dict(item.get("wav_file")).get("path") or "") for item in items],
        "next_recommended_issue": str(report.get("next_recommended_issue") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    boundary = report["audio_review_boundary"]
    lines = [
        "# Stage B Duration Coverage Fill Repeatability Audio Review Package",
        "",
        f"- candidate: `{report['candidate_id']}`",
        f"- status: `{boundary['status']}`",
        f"- rendered audio file count: `{boundary['rendered_audio_file_count']}`",
        f"- technical WAV validation: `{boundary['technical_wav_validation']}`",
        f"- audio rendered quality claimed: `{boundary['audio_rendered_quality_claimed']}`",
        f"- human/audio preference claimed: `{boundary['human_audio_preference_claimed']}`",
        "",
        "| role | sample seed | wav path | duration | sample rate | dead-air | unique |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for item in report.get("review_items", []):
        wav_file = item["wav_file"]
        metrics = item["metrics"]
        lines.append(
            "| {role} | {sample_seed} | `{wav}` | {duration:.3f} | {sample_rate} | "
            "{dead_air:.4f} | {unique} |".format(
                role=item["role"],
                sample_seed=int(item["sample_seed"]),
                wav=wav_file["path"],
                duration=float(wav_file["duration_seconds"]),
                sample_rate=int(wav_file["sample_rate"]),
                dead_air=float(metrics["selected_dead_air_ratio"]),
                unique=int(metrics["selected_focused_unique_pitch_count"]),
            )
        )
    lines.extend(["", "## Not Proven", ""])
    for item in report.get("not_proven", []):
        lines.append(f"- `{item}`")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render repeatability source candidates for audio review")
    parser.add_argument(
        "--repeatability_consolidation",
        type=str,
        default="outputs/stage_b_duration_coverage_fill_repeatability_consolidation/"
        "harness_stage_b_duration_coverage_fill_repeatability_consolidation/"
        "stage_b_duration_coverage_fill_repeatability_consolidation.json",
    )
    parser.add_argument(
        "--dead_air_gain_repair",
        type=str,
        default="outputs/stage_b_duration_coverage_fill_dead_air_gain_repeatability_repair/"
        "harness_stage_b_duration_coverage_fill_dead_air_gain_repeatability_repair/"
        "stage_b_duration_coverage_fill_dead_air_gain_repeatability_repair.json",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/stage_b_duration_coverage_fill_repeatability_audio_review_package",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--renderer_path", type=str, default=shutil.which("fluidsynth") or "/opt/homebrew/bin/fluidsynth")
    parser.add_argument(
        "--soundfont",
        type=str,
        default=str(Path.home() / ".local/share/soundfonts/generaluser-gs/v1.471.sf2"),
    )
    parser.add_argument("--sample_rate", type=int, default=44100)
    parser.add_argument("--expected_file_count", type=int, default=2)
    parser.add_argument("--require_no_quality_claim", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    report = build_repeatability_audio_review_package(
        repeatability_consolidation=read_json(Path(args.repeatability_consolidation)),
        dead_air_gain_repair=read_json(Path(args.dead_air_gain_repair)),
        output_dir=output_dir,
        renderer_path=str(args.renderer_path),
        soundfont_path=str(args.soundfont),
        sample_rate=int(args.sample_rate),
    )
    summary = validate_repeatability_audio_review_package(
        report,
        expected_file_count=int(args.expected_file_count),
        expected_sample_rate=int(args.sample_rate),
        require_no_quality_claim=bool(args.require_no_quality_claim),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "stage_b_duration_coverage_fill_repeatability_audio_review_package.json"
    markdown_path = output_dir / "stage_b_duration_coverage_fill_repeatability_audio_review_package.md"
    write_json(report_path, report)
    write_json(output_dir / "stage_b_duration_coverage_fill_repeatability_audio_review_package_validation_summary.json", summary)
    markdown_path.write_text(markdown_report(report), encoding="utf-8")
    print(json.dumps({**summary, "report_path": str(report_path), "markdown_path": str(markdown_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
