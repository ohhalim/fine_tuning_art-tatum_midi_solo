"""Build a tension/color balanced repair package from solo-yield probe samples."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from scripts.assess_stage_b_generic_base_readiness import write_json, write_text  # noqa: E402
from scripts.build_music_transformer_solo_yield_dead_air_density_repair_package import (  # noqa: E402
    _bool_token,
    _dict,
    _float,
    _int,
    _list,
    _reject_quality_claim,
    compact_probe_sample,
    copy_selected_midis,
    read_json,
    render_selected,
)


SCHEMA_VERSION = "music_transformer_solo_yield_tension_color_balance_repair_package_v1"


class SoloYieldTensionColorBalanceRepairPackageError(ValueError):
    pass


def avg(values: Sequence[float]) -> float:
    return float(mean(values)) if values else 0.0


def package_candidates(report: dict[str, Any]) -> list[Any]:
    candidates = _list(report.get("candidates"))
    if candidates:
        return candidates
    return _list(report.get("selected_candidates"))


def candidate_metrics(report: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "dead_air_ratio": _float(_dict(item).get("dead_air_ratio")),
            "direction_change_ratio": _float(_dict(item).get("direction_change_ratio")),
            "syncopated_onset_ratio": _float(_dict(item).get("syncopated_onset_ratio")),
            "tension_ratio": _float(_dict(item).get("tension_ratio")),
            "note_count": _int(_dict(item).get("note_count")),
        }
        for item in package_candidates(report)
    ]


def metric_summary(
    candidates: Sequence[dict[str, Any]],
    *,
    max_dead_air_ratio: float,
    min_direction_change_ratio: float,
    min_tension_ratio: float,
) -> dict[str, Any]:
    dead_air_values = [_float(item.get("dead_air_ratio")) for item in candidates]
    direction_values = [_float(item.get("direction_change_ratio")) for item in candidates]
    syncopation_values = [_float(item.get("syncopated_onset_ratio")) for item in candidates]
    tension_values = [_float(item.get("tension_ratio")) for item in candidates]
    note_counts = [_float(item.get("note_count")) for item in candidates]
    return {
        "candidate_count": len(candidates),
        "dead_air_avg": avg(dead_air_values),
        "dead_air_max": max(dead_air_values) if dead_air_values else 0.0,
        "dead_air_guard_violation_count": sum(
            1 for value in dead_air_values if value > float(max_dead_air_ratio)
        ),
        "direction_change_avg": avg(direction_values),
        "direction_low_count": sum(
            1 for value in direction_values if value < float(min_direction_change_ratio)
        ),
        "syncopation_avg": avg(syncopation_values),
        "tension_avg": avg(tension_values),
        "tension_low_count": sum(1 for value in tension_values if value < float(min_tension_ratio)),
        "note_count_avg": avg(note_counts),
    }


def tension_sort_key(candidate: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        -_float(candidate.get("tension_ratio")),
        -_float(candidate.get("direction_change_ratio")),
        _float(candidate.get("dead_air_ratio")),
        -_float(candidate.get("syncopated_onset_ratio")),
    )


def select_repair_candidates(
    sweep_report: dict[str, Any],
    *,
    selected_per_case: int,
    max_dead_air_ratio: float,
    min_direction_change_ratio: float,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for case in _list(sweep_report.get("cases")):
        case_dict = _dict(case)
        probe_report_path = Path(str(case_dict.get("probe_report_path") or ""))
        probe_report = read_json(probe_report_path)
        guarded_samples = [
            compact_probe_sample(case_dict, _dict(sample))
            for sample in _list(probe_report.get("samples"))
            if bool(_dict(sample).get("strict_valid", False))
        ]
        guarded_samples = [
            sample
            for sample in guarded_samples
            if Path(str(sample["source_midi_path"])).exists()
            and _float(sample.get("dead_air_ratio")) <= float(max_dead_air_ratio)
            and _float(sample.get("direction_change_ratio")) >= float(min_direction_change_ratio)
        ]
        if len(guarded_samples) < int(selected_per_case):
            raise SoloYieldTensionColorBalanceRepairPackageError(
                f"not enough guarded strict samples for case {case_dict.get('label')}"
            )
        selected.extend(sorted(guarded_samples, key=tension_sort_key)[: int(selected_per_case)])
    return selected


def build_repair_package(
    *,
    sweep_report: dict[str, Any],
    source_repair_package: dict[str, Any],
    output_dir: Path,
    selected_per_case: int,
    max_dead_air_ratio: float,
    min_direction_change_ratio: float,
    min_tension_ratio: float,
    renderer: str,
    soundfont: str,
    sample_rate: int,
    render_audio: bool = True,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    _reject_quality_claim(sweep_report, source_repair_package)
    selected = select_repair_candidates(
        sweep_report,
        selected_per_case=int(selected_per_case),
        max_dead_air_ratio=float(max_dead_air_ratio),
        min_direction_change_ratio=float(min_direction_change_ratio),
    )
    copied = copy_selected_midis(selected, output_dir=output_dir)
    rendered, render_setup = render_selected(
        copied,
        output_dir=output_dir,
        renderer=renderer,
        soundfont=soundfont,
        sample_rate=int(sample_rate),
        render_audio=bool(render_audio),
    )
    source_summary = metric_summary(
        candidate_metrics(source_repair_package),
        max_dead_air_ratio=float(max_dead_air_ratio),
        min_direction_change_ratio=float(min_direction_change_ratio),
        min_tension_ratio=float(min_tension_ratio),
    )
    repair_summary = metric_summary(
        copied,
        max_dead_air_ratio=float(max_dead_air_ratio),
        min_direction_change_ratio=float(min_direction_change_ratio),
        min_tension_ratio=float(min_tension_ratio),
    )
    tension_avg_delta = repair_summary["tension_avg"] - source_summary["tension_avg"]
    tension_low_count_delta = source_summary["tension_low_count"] - repair_summary["tension_low_count"]
    direction_change_avg_delta = (
        repair_summary["direction_change_avg"] - source_summary["direction_change_avg"]
    )
    dead_air_avg_delta = source_summary["dead_air_avg"] - repair_summary["dead_air_avg"]

    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "source_reports": {
            "sweep_report": sweep_report.get("output_dir"),
            "source_repair_package": source_repair_package.get("output_dir"),
        },
        "request": {
            "selected_per_case": int(selected_per_case),
            "max_dead_air_ratio": float(max_dead_air_ratio),
            "min_direction_change_ratio": float(min_direction_change_ratio),
            "min_tension_ratio": float(min_tension_ratio),
        },
        "source_summary": source_summary,
        "repair_summary": {
            **repair_summary,
            "candidate_midi_files_copied": len(copied),
            "candidate_wav_files_rendered": len(rendered),
            "tension_avg_delta": tension_avg_delta,
            "tension_low_count_delta": tension_low_count_delta,
            "direction_change_avg_delta": direction_change_avg_delta,
            "dead_air_avg_delta": dead_air_avg_delta,
        },
        "selected_candidates": copied,
        "render_setup": render_setup,
        "rendered_audio_files": rendered,
        "decision": {
            "current_boundary": "music_transformer_solo_yield_tension_color_balance_repair",
            "next_boundary": "music_transformer_solo_yield_tension_color_balance_repair_rubric_review",
            "critical_user_input_required": False,
            "reason": "phrase-direction repair removed weak direction labels but left low tension-color labels",
        },
        "readiness": {
            "tension_color_balance_repair_completed": True,
            "tension_avg_increased": tension_avg_delta > 0,
            "tension_low_count_reduced": tension_low_count_delta > 0,
            "dead_air_guard_preserved": repair_summary["dead_air_guard_violation_count"] == 0,
            "direction_guard_preserved": repair_summary["direction_low_count"] == 0,
            "technical_wav_render_completed": len(rendered) == len(copied),
            "musical_quality_claimed": False,
            "artist_style_claimed": False,
            "production_ready_claimed": False,
        },
        "not_proven": [
            "human_audio_preference",
            "stable_jazz_solo_quality",
            "tension_threshold_calibration",
            "dead_air_watch_resolved",
            "repair_listening_preference",
        ],
    }


def validate_repair_package(
    report: dict[str, Any],
    *,
    min_candidate_count: int,
    require_tension_avg_increased: bool,
    require_tension_low_count_reduced: bool,
    require_dead_air_guard: bool,
    require_direction_guard: bool,
    require_wav_rendered: bool,
    require_no_quality_claim: bool,
) -> dict[str, Any]:
    if str(report.get("schema_version")) != SCHEMA_VERSION:
        raise SoloYieldTensionColorBalanceRepairPackageError("schema version mismatch")
    repair_summary = _dict(report.get("repair_summary"))
    readiness = _dict(report.get("readiness"))
    if _int(repair_summary.get("candidate_count")) < int(min_candidate_count):
        raise SoloYieldTensionColorBalanceRepairPackageError("candidate count below requirement")
    if require_tension_avg_increased and not bool(readiness.get("tension_avg_increased", False)):
        raise SoloYieldTensionColorBalanceRepairPackageError("tension average increase required")
    if require_tension_low_count_reduced and not bool(
        readiness.get("tension_low_count_reduced", False)
    ):
        raise SoloYieldTensionColorBalanceRepairPackageError("tension low-count reduction required")
    if require_dead_air_guard and not bool(readiness.get("dead_air_guard_preserved", False)):
        raise SoloYieldTensionColorBalanceRepairPackageError("dead-air guard required")
    if require_direction_guard and not bool(readiness.get("direction_guard_preserved", False)):
        raise SoloYieldTensionColorBalanceRepairPackageError("direction guard required")
    if require_wav_rendered and not bool(readiness.get("technical_wav_render_completed", False)):
        raise SoloYieldTensionColorBalanceRepairPackageError("WAV render completion required")
    if require_no_quality_claim:
        claimed = [
            key
            for key in ("musical_quality_claimed", "artist_style_claimed", "production_ready_claimed")
            if bool(readiness.get(key, True))
        ]
        if claimed:
            raise SoloYieldTensionColorBalanceRepairPackageError(
                f"unexpected quality claim: {claimed}"
            )
    source_summary = _dict(report.get("source_summary"))
    return {
        "schema_version": str(report.get("schema_version")),
        "candidate_count": _int(repair_summary.get("candidate_count")),
        "candidate_midi_files_copied": _int(repair_summary.get("candidate_midi_files_copied")),
        "candidate_wav_files_rendered": _int(repair_summary.get("candidate_wav_files_rendered")),
        "source_tension_avg": _float(source_summary.get("tension_avg")),
        "repair_tension_avg": _float(repair_summary.get("tension_avg")),
        "tension_avg_delta": _float(repair_summary.get("tension_avg_delta")),
        "source_tension_low_count": _int(source_summary.get("tension_low_count")),
        "repair_tension_low_count": _int(repair_summary.get("tension_low_count")),
        "tension_low_count_delta": _int(repair_summary.get("tension_low_count_delta")),
        "source_direction_change_avg": _float(source_summary.get("direction_change_avg")),
        "repair_direction_change_avg": _float(repair_summary.get("direction_change_avg")),
        "direction_change_avg_delta": _float(repair_summary.get("direction_change_avg_delta")),
        "source_dead_air_avg": _float(source_summary.get("dead_air_avg")),
        "repair_dead_air_avg": _float(repair_summary.get("dead_air_avg")),
        "dead_air_avg_delta": _float(repair_summary.get("dead_air_avg_delta")),
        "repair_dead_air_guard_violation_count": _int(
            repair_summary.get("dead_air_guard_violation_count")
        ),
        "repair_direction_low_count": _int(repair_summary.get("direction_low_count")),
        "tension_avg_increased": bool(readiness.get("tension_avg_increased", False)),
        "tension_low_count_reduced": bool(readiness.get("tension_low_count_reduced", False)),
        "dead_air_guard_preserved": bool(readiness.get("dead_air_guard_preserved", False)),
        "direction_guard_preserved": bool(readiness.get("direction_guard_preserved", False)),
        "technical_wav_render_completed": bool(readiness.get("technical_wav_render_completed", False)),
        "next_boundary": str(_dict(report.get("decision")).get("next_boundary") or ""),
        "musical_quality_claimed": bool(readiness.get("musical_quality_claimed", True)),
    }


def markdown_report(report: dict[str, Any]) -> str:
    source = report["source_summary"]
    repair = report["repair_summary"]
    decision = report["decision"]
    readiness = report["readiness"]
    lines = [
        "# Music Transformer Solo Yield Tension Color Balance Repair Package",
        "",
        "## Summary",
        "",
        f"- candidate count: `{repair['candidate_count']}`",
        f"- MIDI/WAV: `{repair['candidate_midi_files_copied']}` / `{repair['candidate_wav_files_rendered']}`",
        f"- source tension avg: `{float(source['tension_avg']):.4f}`",
        f"- repair tension avg: `{float(repair['tension_avg']):.4f}`",
        f"- tension avg delta: `{float(repair['tension_avg_delta']):.4f}`",
        f"- source/repair tension low count: `{source['tension_low_count']}` / `{repair['tension_low_count']}`",
        f"- tension low count delta: `{repair['tension_low_count_delta']}`",
        f"- source direction avg: `{float(source['direction_change_avg']):.4f}`",
        f"- repair direction avg: `{float(repair['direction_change_avg']):.4f}`",
        f"- direction avg delta: `{float(repair['direction_change_avg_delta']):.4f}`",
        f"- repair direction low count: `{repair['direction_low_count']}`",
        f"- source dead-air avg/max: `{float(source['dead_air_avg']):.4f}` / `{float(source['dead_air_max']):.4f}`",
        f"- repair dead-air avg/max: `{float(repair['dead_air_avg']):.4f}` / `{float(repair['dead_air_max']):.4f}`",
        f"- dead-air avg delta: `{float(repair['dead_air_avg_delta']):.4f}`",
        f"- repair dead-air guard violation count: `{repair['dead_air_guard_violation_count']}`",
        f"- technical WAV render completed: `{_bool_token(readiness['technical_wav_render_completed'])}`",
        f"- musical quality claimed: `{_bool_token(readiness['musical_quality_claimed'])}`",
        f"- next boundary: `{decision['next_boundary']}`",
        "",
        "## Selected Candidates",
        "",
        "| idx | case | sample | notes | dead air | direction | syncopation | tension | MIDI | WAV |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    rendered_by_index = {
        int(item["repair_index"]): _dict(item.get("wav_file")).get("path", "")
        for item in report.get("rendered_audio_files", [])
    }
    for item in report.get("selected_candidates", []):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(item["repair_index"]),
                    f"`{item['case_label']}`",
                    str(item["sample_index"]),
                    str(item["note_count"]),
                    f"{float(item['dead_air_ratio']):.4f}",
                    f"{float(item['direction_change_ratio']):.4f}",
                    f"{float(item['syncopated_onset_ratio']):.4f}",
                    f"{float(item['tension_ratio']):.4f}",
                    f"`{item['repair_midi_path']}`",
                    f"`{rendered_by_index.get(int(item['repair_index']), '')}`",
                ]
            )
            + " |"
        )
    lines.extend(["", "## Not Proven", ""])
    for item in report.get("not_proven", []):
        lines.append(f"- `{item}`")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build tension/color balance repair package")
    parser.add_argument("--sweep_report", type=str, required=True)
    parser.add_argument("--source_repair_package", type=str, required=True)
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/music_transformer_finetune_mvp/solo_yield_tension_color_balance_repair",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--selected_per_case", type=int, default=2)
    parser.add_argument("--max_dead_air_ratio", type=float, default=0.68)
    parser.add_argument("--min_direction_change_ratio", type=float, default=0.50)
    parser.add_argument("--min_tension_ratio", type=float, default=0.20)
    parser.add_argument("--renderer", type=str, default="")
    parser.add_argument("--soundfont", type=str, default="")
    parser.add_argument("--sample_rate", type=int, default=44100)
    parser.add_argument("--skip_render", action="store_true")
    parser.add_argument("--min_candidate_count", type=int, default=8)
    parser.add_argument("--require_tension_avg_increased", action="store_true")
    parser.add_argument("--require_tension_low_count_reduced", action="store_true")
    parser.add_argument("--require_dead_air_guard", action="store_true")
    parser.add_argument("--require_direction_guard", action="store_true")
    parser.add_argument("--require_wav_rendered", action="store_true")
    parser.add_argument("--require_no_quality_claim", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    report = build_repair_package(
        sweep_report=read_json(Path(args.sweep_report)),
        source_repair_package=read_json(Path(args.source_repair_package)),
        output_dir=output_dir,
        selected_per_case=int(args.selected_per_case),
        max_dead_air_ratio=float(args.max_dead_air_ratio),
        min_direction_change_ratio=float(args.min_direction_change_ratio),
        min_tension_ratio=float(args.min_tension_ratio),
        renderer=str(args.renderer),
        soundfont=str(args.soundfont),
        sample_rate=int(args.sample_rate),
        render_audio=not bool(args.skip_render),
    )
    summary = validate_repair_package(
        report,
        min_candidate_count=int(args.min_candidate_count),
        require_tension_avg_increased=bool(args.require_tension_avg_increased),
        require_tension_low_count_reduced=bool(args.require_tension_low_count_reduced),
        require_dead_air_guard=bool(args.require_dead_air_guard),
        require_direction_guard=bool(args.require_direction_guard),
        require_wav_rendered=bool(args.require_wav_rendered),
        require_no_quality_claim=bool(args.require_no_quality_claim),
    )
    write_json(output_dir / "tension_color_balance_repair_package.json", report)
    write_json(output_dir / "tension_color_balance_repair_package_summary.json", summary)
    markdown = markdown_report(report)
    write_text(output_dir / "tension_color_balance_repair_package.md", markdown)
    if args.doc_path:
        write_text(Path(args.doc_path), markdown)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
