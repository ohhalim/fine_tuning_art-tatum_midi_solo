"""Build audio render package metadata for the phrase-continuation repair candidate."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from scripts.assess_stage_b_generic_base_readiness import read_json, write_json, write_text  # noqa: E402
from scripts.build_stage_b_generic_tiny_checkpoint_repair_audio_render_package import (  # noqa: E402
    file_meta,
    render_command,
    renderer_probe,
)
from scripts.run_stage_b_generic_tiny_checkpoint_generation_probe import (  # noqa: E402
    _bool_token,
    _dict,
    _float,
    _int,
)


class StageBGenericTinyCheckpointRepairPhraseContinuationAudioRenderPackageError(ValueError):
    pass


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def validate_sweep_report(report: dict[str, Any]) -> list[dict[str, Any]]:
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    phrase = _dict(report.get("phrase_continuation"))
    if str(readiness.get("boundary") or "") != "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_repair_sweep":
        raise StageBGenericTinyCheckpointRepairPhraseContinuationAudioRenderPackageError(
            "unexpected phrase-continuation sweep boundary"
        )
    if not bool(phrase.get("target_passed", False)):
        raise StageBGenericTinyCheckpointRepairPhraseContinuationAudioRenderPackageError(
            "phrase-continuation target must pass before audio package"
        )
    if _int(phrase.get("target_qualified_count")) < 1:
        raise StageBGenericTinyCheckpointRepairPhraseContinuationAudioRenderPackageError(
            "target-qualified candidate required"
        )
    if str(decision.get("next_boundary") or "") != (
        "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_audio_render_package"
    ):
        raise StageBGenericTinyCheckpointRepairPhraseContinuationAudioRenderPackageError("unexpected next boundary")
    claimed = [
        bool(readiness.get("musical_quality_claimed", True)),
        bool(readiness.get("broad_trained_model_quality_claimed", True)),
        bool(readiness.get("brad_style_adaptation_claimed", True)),
    ]
    if any(claimed):
        raise StageBGenericTinyCheckpointRepairPhraseContinuationAudioRenderPackageError(
            "quality claims must not be set"
        )
    ranked = [dict(row) for row in _list(phrase.get("ranked_candidates")) if isinstance(row, dict)]
    selected = [row for row in ranked if bool(row.get("target_qualified", False))]
    if not selected:
        raise StageBGenericTinyCheckpointRepairPhraseContinuationAudioRenderPackageError(
            "target-qualified candidate list empty"
        )
    return selected


def compact_candidate(candidate: dict[str, Any], *, review_rank: int) -> dict[str, Any]:
    return {
        "review_rank": int(review_rank),
        "sample_seed": _int(candidate.get("sample_seed")),
        "sample_index": _int(candidate.get("sample_index")),
        "midi_file": file_meta(str(candidate.get("midi_path") or ""), required=True),
        "target_qualified": bool(candidate.get("target_qualified", False)),
        "note_count": _int(candidate.get("note_count")),
        "phrase_coverage_ratio": _float(candidate.get("phrase_coverage_ratio")),
        "dead_air_ratio": _float(candidate.get("dead_air_ratio")),
        "tail_empty_steps": _int(candidate.get("tail_empty_steps")),
        "pitch_role_chord_tone_ratio": _float(candidate.get("pitch_role_chord_tone_ratio")),
        "postprocess_removal_ratio": _float(candidate.get("postprocess_removal_ratio")),
        "max_simultaneous_notes": _int(candidate.get("max_simultaneous_notes")),
    }


def item_output_stem(item: dict[str, Any]) -> str:
    return f"rank_{item['review_rank']:02d}_seed_{item['sample_seed']}_sample_{item['sample_index']}"


def build_audio_render_package(
    sweep_report: dict[str, Any],
    *,
    output_dir: Path,
    requested_renderer: str = "",
    soundfont_path: str = "",
    renderer_paths: dict[str, str] | None = None,
) -> dict[str, Any]:
    selected_candidates = validate_sweep_report(sweep_report)
    probe = renderer_probe(
        requested_renderer=requested_renderer,
        soundfont_path=soundfont_path,
        renderer_paths=renderer_paths,
    )
    review_items = [
        compact_candidate(candidate, review_rank=index)
        for index, candidate in enumerate(selected_candidates, start=1)
    ]
    planned_outputs: list[dict[str, Any]] = []
    for item in review_items:
        output_stem = item_output_stem(item)
        output_wav_path = output_dir / "audio" / f"{output_stem}.wav"
        midi_path = str(item["midi_file"]["path"])
        planned_outputs.append(
            {
                "review_rank": item["review_rank"],
                "sample_seed": item["sample_seed"],
                "sample_index": item["sample_index"],
                "source_midi_path": midi_path,
                "planned_wav_path": str(output_wav_path),
                "planned_wav_exists": output_wav_path.exists(),
                "render_command": render_command(probe, midi_path, str(output_wav_path)),
            }
        )
    render_ready = str(probe["status"]) == "ready_for_local_render"
    return {
        "schema_version": "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_audio_render_package_v1",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "source_sweep_schema": str(sweep_report.get("schema_version") or ""),
        "source_sweep_run_dir": str(sweep_report.get("run_dir") or ""),
        "review_items": review_items,
        "renderer_probe": probe,
        "planned_audio_outputs": planned_outputs,
        "local_audio_render_boundary": {
            "boundary": "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_audio_render_package",
            "status": str(probe["status"]),
            "render_attempted": False,
            "planned_audio_output_count": len(planned_outputs),
            "rendered_audio_file_count": 0,
            "audio_output_claimed": False,
            "audio_rendered_quality_claimed": False,
            "human_audio_preference_claimed": False,
            "musical_quality_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
        },
        "decision": {
            "current_boundary": "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_audio_render_package",
            "next_boundary": (
                "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_local_audio_render_attempt"
                if render_ready
                else "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_audio_render_tooling_setup"
            ),
            "auto_progress_allowed": render_ready,
            "critical_user_input_required": not render_ready,
        },
        "not_proven": [
            "audio_output",
            "audio_rendered_quality",
            "human_audio_preference",
            "musical_quality",
            "broad_trained_model_quality",
            "brad_style_adaptation",
        ],
        "next_recommended_issue": (
            "Stage B generic tiny checkpoint repair phrase continuation local audio render attempt"
            if render_ready
            else "Stage B generic tiny checkpoint repair phrase continuation audio render tooling setup"
        ),
    }


def validate_audio_render_package(
    report: dict[str, Any],
    *,
    expected_boundary: str | None,
    expected_status: str | None,
    min_planned_outputs: int,
    require_target_qualified: bool,
    require_required_midi_exists: bool,
    require_no_audio_claim: bool,
) -> dict[str, Any]:
    boundary = _dict(report.get("local_audio_render_boundary"))
    status = str(boundary.get("status") or "")
    if expected_boundary and str(boundary.get("boundary") or "") != expected_boundary:
        raise StageBGenericTinyCheckpointRepairPhraseContinuationAudioRenderPackageError(
            f"expected boundary {expected_boundary}, got {boundary.get('boundary')}"
        )
    if expected_status and status != expected_status:
        raise StageBGenericTinyCheckpointRepairPhraseContinuationAudioRenderPackageError(
            f"expected status {expected_status}, got {status}"
        )
    planned_outputs = report.get("planned_audio_outputs")
    if not isinstance(planned_outputs, list):
        planned_outputs = []
    if len(planned_outputs) < min_planned_outputs:
        raise StageBGenericTinyCheckpointRepairPhraseContinuationAudioRenderPackageError(
            f"planned output count below target: {len(planned_outputs)} < {min_planned_outputs}"
        )
    review_items = _list(report.get("review_items"))
    if require_target_qualified and not all(bool(_dict(item).get("target_qualified", False)) for item in review_items):
        raise StageBGenericTinyCheckpointRepairPhraseContinuationAudioRenderPackageError(
            "review items must be target-qualified"
        )
    if require_required_midi_exists:
        missing = []
        for item in review_items:
            midi_file = _dict(_dict(item).get("midi_file"))
            if bool(midi_file.get("required", False)) and not bool(midi_file.get("exists", False)):
                missing.append(str(midi_file.get("path") or "midi_file"))
        if missing:
            raise StageBGenericTinyCheckpointRepairPhraseContinuationAudioRenderPackageError(
                f"missing required MIDI files: {missing}"
            )
    if require_no_audio_claim:
        claimed = [
            bool(boundary.get("render_attempted", True)),
            bool(boundary.get("audio_output_claimed", True)),
            bool(boundary.get("audio_rendered_quality_claimed", True)),
            bool(boundary.get("human_audio_preference_claimed", True)),
            bool(boundary.get("musical_quality_claimed", True)),
            bool(boundary.get("broad_trained_model_quality_claimed", True)),
            bool(boundary.get("brad_style_adaptation_claimed", True)),
        ]
        if any(claimed):
            raise StageBGenericTinyCheckpointRepairPhraseContinuationAudioRenderPackageError(
                "audio or quality claims must not be set"
            )
    decision = _dict(report.get("decision"))
    return {
        "boundary": str(boundary.get("boundary") or ""),
        "render_status": status,
        "selected_renderer_name": str(_dict(report.get("renderer_probe")).get("selected_renderer_name") or ""),
        "soundfont_exists": bool(_dict(report.get("renderer_probe")).get("soundfont_exists", False)),
        "planned_audio_output_count": len(planned_outputs),
        "render_attempted": bool(boundary.get("render_attempted", True)),
        "audio_rendered_quality_claimed": bool(boundary.get("audio_rendered_quality_claimed", True)),
        "human_audio_preference_claimed": bool(boundary.get("human_audio_preference_claimed", True)),
        "auto_progress_allowed": bool(decision.get("auto_progress_allowed", False)),
        "critical_user_input_required": bool(decision.get("critical_user_input_required", True)),
        "next_boundary": str(decision.get("next_boundary") or ""),
        "next_recommended_issue": str(report.get("next_recommended_issue") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    boundary = report["local_audio_render_boundary"]
    probe = report["renderer_probe"]
    lines = [
        "# Stage B Generic Tiny Checkpoint Repair Phrase Continuation Audio Render Package",
        "",
        "## Summary",
        "",
        f"- boundary: `{boundary['boundary']}`",
        f"- render status: `{boundary['status']}`",
        f"- selected renderer: `{probe['selected_renderer_name']}`",
        f"- soundfont exists: `{_bool_token(probe['soundfont_exists'])}`",
        f"- planned audio outputs: `{boundary['planned_audio_output_count']}`",
        f"- render attempted: `{_bool_token(boundary['render_attempted'])}`",
        f"- audio rendered quality claimed: `{_bool_token(boundary['audio_rendered_quality_claimed'])}`",
        f"- human/audio preference claimed: `{_bool_token(boundary['human_audio_preference_claimed'])}`",
        f"- musical quality claimed: `{_bool_token(boundary['musical_quality_claimed'])}`",
        "",
        "## Selected Candidate",
        "",
        "| rank | seed | sample | notes | coverage | tail empty | chord role | MIDI exists | planned WAV | command ready |",
        "|---:|---:|---:|---:|---:|---:|---:|:---:|---|:---:|",
    ]
    review_by_key = {
        (_int(item.get("review_rank")), _int(item.get("sample_seed")), _int(item.get("sample_index"))): item
        for item in report.get("review_items", [])
        if isinstance(item, dict)
    }
    for item in report.get("planned_audio_outputs", []):
        key = (_int(item.get("review_rank")), _int(item.get("sample_seed")), _int(item.get("sample_index")))
        review_item = _dict(review_by_key.get(key))
        midi_exists = bool(_dict(review_item.get("midi_file")).get("exists", False))
        command_ready = bool(item.get("render_command"))
        lines.append(
            "| "
            + " | ".join(
                [
                    str(item.get("review_rank") or ""),
                    str(item.get("sample_seed") or ""),
                    str(item.get("sample_index") or ""),
                    str(review_item.get("note_count") or ""),
                    str(review_item.get("phrase_coverage_ratio") or ""),
                    str(review_item.get("tail_empty_steps") or ""),
                    str(review_item.get("pitch_role_chord_tone_ratio") or ""),
                    _bool_token(midi_exists),
                    str(item.get("planned_wav_path") or ""),
                    _bool_token(command_ready),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Not Proven", ""])
    for item in report.get("not_proven", []):
        lines.append(f"- `{item}`")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    default_soundfont = Path.home() / ".local/share/soundfonts/generaluser-gs/v1.471.sf2"
    parser = argparse.ArgumentParser(
        description="Build phrase-continuation repair audio render package metadata"
    )
    parser.add_argument(
        "--sweep_report",
        type=str,
        default="outputs/stage_b_generic_tiny_checkpoint_repair_phrase_continuation_sweep/"
        "harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_sweep/"
        "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_sweep.json",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/stage_b_generic_tiny_checkpoint_repair_phrase_continuation_audio_render_package",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--renderer", type=str, default="")
    parser.add_argument("--soundfont", type=str, default=str(default_soundfont))
    parser.add_argument("--expected_boundary", type=str, default="")
    parser.add_argument("--expected_status", type=str, default="")
    parser.add_argument("--min_planned_outputs", type=int, default=1)
    parser.add_argument("--require_target_qualified", action="store_true")
    parser.add_argument("--require_required_midi_exists", action="store_true")
    parser.add_argument("--require_no_audio_claim", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    sweep_report_path = Path(args.sweep_report)
    if not sweep_report_path.exists():
        raise StageBGenericTinyCheckpointRepairPhraseContinuationAudioRenderPackageError(
            "phrase-continuation sweep report required"
        )
    report = build_audio_render_package(
        read_json(sweep_report_path),
        output_dir=output_dir,
        requested_renderer=str(args.renderer or ""),
        soundfont_path=str(args.soundfont or ""),
    )
    summary = validate_audio_render_package(
        report,
        expected_boundary=str(args.expected_boundary or ""),
        expected_status=str(args.expected_status or ""),
        min_planned_outputs=int(args.min_planned_outputs),
        require_target_qualified=bool(args.require_target_qualified),
        require_required_midi_exists=bool(args.require_required_midi_exists),
        require_no_audio_claim=bool(args.require_no_audio_claim),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        output_dir / "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_audio_render_package.json",
        report,
    )
    write_json(
        output_dir / "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_audio_render_package_validation_summary.json",
        summary,
    )
    markdown = markdown_report(report)
    write_text(output_dir / "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_audio_render_package.md", markdown)
    if args.doc_path:
        write_text(Path(args.doc_path), markdown)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
