"""Build audio render package metadata for sparse phrase repair candidates."""

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
    renderer_probe,
)
from scripts.build_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_audio_render_package import (  # noqa: E402
    render_command,
)
from scripts.run_stage_b_generic_tiny_checkpoint_generation_probe import (  # noqa: E402
    _bool_token,
    _dict,
    _float,
    _int,
)


class StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardSparsePhraseAudioRenderPackageError(
    ValueError
):
    pass


BOUNDARY = (
    "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_audio_render_package"
)
SOURCE_BOUNDARY = (
    "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_repair_sweep"
)
SCHEMA_VERSION = (
    "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_audio_render_package_v1"
)


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def validate_sparse_phrase_sweep_report(report: dict[str, Any]) -> list[dict[str, Any]]:
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    repair = _dict(report.get("sparse_phrase_repair"))
    if str(readiness.get("boundary") or "") != SOURCE_BOUNDARY:
        raise StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardSparsePhraseAudioRenderPackageError(
            "unexpected sparse phrase repair sweep boundary"
        )
    if not bool(repair.get("target_passed", False)):
        raise StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardSparsePhraseAudioRenderPackageError(
            "sparse phrase repair target must pass before audio package"
        )
    if _int(repair.get("target_qualified_count")) < 1:
        raise StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardSparsePhraseAudioRenderPackageError(
            "target-qualified candidate required"
        )
    if str(decision.get("next_boundary") or "") != BOUNDARY:
        raise StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardSparsePhraseAudioRenderPackageError(
            "unexpected next boundary"
        )
    claimed = [
        bool(readiness.get("human_audio_preference_claimed", True)),
        bool(readiness.get("musical_quality_claimed", True)),
        bool(readiness.get("quality_cause_claimed", True)),
        bool(readiness.get("broad_trained_model_quality_claimed", True)),
        bool(readiness.get("brad_style_adaptation_claimed", True)),
    ]
    if any(claimed):
        raise StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardSparsePhraseAudioRenderPackageError(
            "quality claims must not be set"
        )
    ranked = [dict(row) for row in _list(repair.get("ranked_candidates")) if isinstance(row, dict)]
    selected = [row for row in ranked if bool(row.get("target_qualified", False))]
    if not selected:
        raise StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardSparsePhraseAudioRenderPackageError(
            "target-qualified candidate list empty"
        )
    return selected


def compact_candidate(candidate: dict[str, Any], *, review_rank: int) -> dict[str, Any]:
    audit = _dict(candidate.get("midi_note_audit"))
    sparse = _dict(candidate.get("sparse_phrase_metrics"))
    return {
        "review_rank": int(review_rank),
        "interval_cap": _int(candidate.get("interval_cap")),
        "sample_seed": _int(candidate.get("sample_seed")),
        "sample_index": _int(candidate.get("sample_index")),
        "midi_file": file_meta(str(candidate.get("midi_path") or ""), required=True),
        "target_qualified": bool(candidate.get("target_qualified", False)),
        "note_count": _int(candidate.get("note_count")),
        "phrase_coverage_ratio": _float(candidate.get("phrase_coverage_ratio")),
        "tail_empty_steps": _int(candidate.get("tail_empty_steps")),
        "postprocess_removal_ratio": _float(candidate.get("postprocess_removal_ratio")),
        "soft_failure_reasons": _list(candidate.get("soft_failure_reasons")),
        "sparse_phrase_metrics": {
            "gap_ratio_to_window": _float(sparse.get("gap_ratio_to_window")),
            "max_internal_gap_beats": _float(sparse.get("max_internal_gap_beats")),
            "adjacent_repeat_count": _int(sparse.get("adjacent_repeat_count")),
            "evidence_flags": _list(sparse.get("evidence_flags")),
        },
        "midi_note_audit": {
            "pitch_min": audit.get("pitch_min"),
            "pitch_max": audit.get("pitch_max"),
            "pitch_span": _int(audit.get("pitch_span")),
            "max_abs_interval": _int(audit.get("max_abs_interval")),
            "large_interval_ratio": _float(audit.get("large_interval_ratio")),
            "severe_interval_count": _int(audit.get("severe_interval_count")),
            "pitch_sequence": _list(audit.get("pitch_sequence")),
            "intervals": _list(audit.get("intervals")),
        },
    }


def item_output_stem(item: dict[str, Any]) -> str:
    return (
        f"rank_{item['review_rank']:02d}_cap_{item['interval_cap']}"
        f"_seed_{item['sample_seed']}_sample_{item['sample_index']}"
    )


def build_audio_render_package(
    sweep_report: dict[str, Any],
    *,
    output_dir: Path,
    requested_renderer: str = "",
    soundfont_path: str = "",
    renderer_paths: dict[str, str] | None = None,
    max_review_items: int = 3,
) -> dict[str, Any]:
    selected_candidates = validate_sparse_phrase_sweep_report(sweep_report)[: int(max_review_items)]
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
                "interval_cap": item["interval_cap"],
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
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "source_sweep_schema": str(sweep_report.get("schema_version") or ""),
        "source_sweep_run_dir": str(sweep_report.get("run_dir") or ""),
        "review_items": review_items,
        "renderer_probe": probe,
        "planned_audio_outputs": planned_outputs,
        "local_audio_render_boundary": {
            "boundary": BOUNDARY,
            "status": str(probe["status"]),
            "render_attempted": False,
            "planned_audio_output_count": len(planned_outputs),
            "rendered_audio_file_count": 0,
            "audio_output_claimed": False,
            "audio_rendered_quality_claimed": False,
            "human_audio_preference_claimed": False,
            "musical_quality_claimed": False,
            "quality_cause_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
        },
        "decision": {
            "current_boundary": BOUNDARY,
            "next_boundary": (
                "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_local_audio_render_attempt"
                if render_ready
                else "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_audio_render_tooling_setup"
            ),
            "auto_progress_allowed": render_ready,
            "critical_user_input_required": not render_ready,
        },
        "not_proven": [
            "audio_output",
            "audio_rendered_quality",
            "human_audio_preference",
            "musical_quality",
            "quality_root_cause",
            "broad_trained_model_quality",
            "brad_style_adaptation",
        ],
        "next_recommended_issue": (
            "Stage B generic tiny checkpoint repair phrase continuation range interval guard sparse phrase local audio render attempt"
            if render_ready
            else "Stage B generic tiny checkpoint repair phrase continuation range interval guard sparse phrase audio render tooling setup"
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
        raise StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardSparsePhraseAudioRenderPackageError(
            f"expected boundary {expected_boundary}, got {boundary.get('boundary')}"
        )
    if expected_status and status != expected_status:
        raise StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardSparsePhraseAudioRenderPackageError(
            f"expected status {expected_status}, got {status}"
        )
    planned_outputs = _list(report.get("planned_audio_outputs"))
    if len(planned_outputs) < min_planned_outputs:
        raise StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardSparsePhraseAudioRenderPackageError(
            f"planned output count below target: {len(planned_outputs)} < {min_planned_outputs}"
        )
    review_items = _list(report.get("review_items"))
    if require_target_qualified and not all(bool(_dict(item).get("target_qualified", False)) for item in review_items):
        raise StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardSparsePhraseAudioRenderPackageError(
            "review items must be target-qualified"
        )
    if require_required_midi_exists:
        missing = []
        for item in review_items:
            midi_file = _dict(_dict(item).get("midi_file"))
            if bool(midi_file.get("required", False)) and not bool(midi_file.get("exists", False)):
                missing.append(str(midi_file.get("path") or "midi_file"))
        if missing:
            raise StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardSparsePhraseAudioRenderPackageError(
                f"missing required MIDI files: {missing}"
            )
    if require_no_audio_claim:
        claimed = [
            bool(boundary.get("render_attempted", True)),
            bool(boundary.get("audio_output_claimed", True)),
            bool(boundary.get("audio_rendered_quality_claimed", True)),
            bool(boundary.get("human_audio_preference_claimed", True)),
            bool(boundary.get("musical_quality_claimed", True)),
            bool(boundary.get("quality_cause_claimed", True)),
            bool(boundary.get("broad_trained_model_quality_claimed", True)),
            bool(boundary.get("brad_style_adaptation_claimed", True)),
        ]
        if any(claimed):
            raise StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardSparsePhraseAudioRenderPackageError(
                "audio or quality claims must not be set"
            )
    decision = _dict(report.get("decision"))
    probe = _dict(report.get("renderer_probe"))
    return {
        "boundary": str(boundary.get("boundary") or ""),
        "render_status": status,
        "selected_renderer": str(probe.get("selected_renderer") or ""),
        "soundfont_exists": bool(probe.get("soundfont_exists", False)),
        "planned_audio_output_count": len(planned_outputs),
        "review_item_count": len(review_items),
        "audio_output_claimed": bool(boundary.get("audio_output_claimed", True)),
        "audio_rendered_quality_claimed": bool(boundary.get("audio_rendered_quality_claimed", True)),
        "human_audio_preference_claimed": bool(boundary.get("human_audio_preference_claimed", True)),
        "musical_quality_claimed": bool(boundary.get("musical_quality_claimed", True)),
        "auto_progress_allowed": bool(decision.get("auto_progress_allowed", False)),
        "critical_user_input_required": bool(decision.get("critical_user_input_required", True)),
        "next_boundary": str(decision.get("next_boundary") or ""),
        "next_recommended_issue": str(report.get("next_recommended_issue") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    boundary = report["local_audio_render_boundary"]
    decision = report["decision"]
    probe = report["renderer_probe"]
    lines = [
        "# Stage B Generic Tiny Checkpoint Repair Phrase Continuation Range Interval Guard Sparse Phrase Audio Render Package",
        "",
        "## Summary",
        "",
        f"- boundary: `{boundary['boundary']}`",
        f"- status: `{boundary['status']}`",
        f"- next boundary: `{decision['next_boundary']}`",
        f"- planned audio outputs: `{boundary['planned_audio_output_count']}`",
        f"- selected renderer: `{probe.get('selected_renderer')}`",
        f"- soundfont exists: `{_bool_token(bool(probe.get('soundfont_exists', False)))}`",
        f"- audio quality claimed: `{_bool_token(boundary['audio_rendered_quality_claimed'])}`",
        "",
        "## Review Items",
        "",
        "| rank | cap | seed | sample | notes | gap ratio | max gap | max interval | soft failures | midi | planned wav |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|",
    ]
    planned_by_rank = {
        _int(item.get("review_rank")): item
        for item in report.get("planned_audio_outputs", [])
        if isinstance(item, dict)
    }
    for item in report.get("review_items", []):
        sparse = _dict(item.get("sparse_phrase_metrics"))
        audit = _dict(item.get("midi_note_audit"))
        planned = _dict(planned_by_rank.get(_int(item.get("review_rank"))))
        lines.append(
            "| "
            + " | ".join(
                [
                    str(item.get("review_rank")),
                    str(item.get("interval_cap")),
                    str(item.get("sample_seed")),
                    str(item.get("sample_index")),
                    str(item.get("note_count")),
                    f"{_float(sparse.get('gap_ratio_to_window')):.4f}",
                    f"{_float(sparse.get('max_internal_gap_beats')):.4f}",
                    str(audit.get("max_abs_interval")),
                    ", ".join(_list(item.get("soft_failure_reasons"))) or "none",
                    str(_dict(item.get("midi_file")).get("path")),
                    str(planned.get("planned_wav_path") or ""),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Not Proven", ""])
    for item in report.get("not_proven", []):
        lines.append(f"- `{item}`")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build sparse phrase audio render package")
    parser.add_argument(
        "--sweep_report",
        type=str,
        default="outputs/stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_repair_sweep/"
        "harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_repair_sweep/"
        "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_repair_sweep.json",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=(
            "outputs/"
            "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_audio_render_package"
        ),
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--renderer", type=str, default="fluidsynth")
    parser.add_argument("--soundfont", type=str, default="")
    parser.add_argument("--max_review_items", type=int, default=3)
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
    report = build_audio_render_package(
        read_json(Path(args.sweep_report)),
        output_dir=output_dir,
        requested_renderer=str(args.renderer),
        soundfont_path=str(args.soundfont or ""),
        max_review_items=int(args.max_review_items),
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
        output_dir
        / "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_audio_render_package.json",
        report,
    )
    write_json(
        output_dir
        / "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_audio_render_package_validation_summary.json",
        summary,
    )
    markdown = markdown_report(report)
    write_text(
        output_dir
        / "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_audio_render_package.md",
        markdown,
    )
    if args.doc_path:
        write_text(Path(args.doc_path), markdown)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
