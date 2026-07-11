"""Build a dead-air/density balanced repair package from solo-yield probe samples."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from scripts.assess_stage_b_generic_base_readiness import write_json, write_text  # noqa: E402
from scripts.render_stage_b_midi_to_solo_candidate_audio import (  # noqa: E402
    resolve_soundfont,
    sha256_file,
    wav_meta,
)


SCHEMA_VERSION = "music_transformer_solo_yield_dead_air_density_repair_package_v1"


class SoloYieldDeadAirDensityRepairPackageError(ValueError):
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


def avg(values: Sequence[float]) -> float:
    return float(mean(values)) if values else 0.0


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SoloYieldDeadAirDensityRepairPackageError(f"json not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _reject_quality_claim(*reports: dict[str, Any]) -> None:
    claimed: list[str] = []
    for index, report in enumerate(reports, start=1):
        readiness = _dict(report.get("readiness"))
        for key in ("musical_quality_claimed", "artist_style_claimed", "production_ready_claimed"):
            if bool(readiness.get(key, False)):
                claimed.append(f"report_{index}:{key}")
    if claimed:
        raise SoloYieldDeadAirDensityRepairPackageError(f"unexpected quality claim: {claimed}")


def compact_probe_sample(case: dict[str, Any], sample: dict[str, Any]) -> dict[str, Any]:
    metrics = _dict(sample.get("metrics"))
    contour = _dict(sample.get("phrase_contour"))
    rhythm = _dict(sample.get("rhythm_profile"))
    pitch_roles = _dict(sample.get("pitch_roles"))
    return {
        "case_label": str(case.get("label") or ""),
        "chords": str(case.get("chords") or ""),
        "case_seed": _int(case.get("seed")),
        "sample_index": _int(sample.get("sample_index")),
        "sample_seed": _int(sample.get("sample_seed")),
        "source_midi_path": str(sample.get("midi_path") or ""),
        "strict_valid": bool(sample.get("strict_valid", False)),
        "valid": bool(sample.get("valid", False)),
        "grammar_gate_passed": bool(sample.get("grammar_gate_passed", False)),
        "note_count": _int(metrics.get("note_count")),
        "unique_pitch_count": _int(metrics.get("unique_pitch_count")),
        "dead_air_ratio": _float(metrics.get("dead_air_ratio")),
        "direction_change_ratio": _float(contour.get("direction_change_ratio")),
        "syncopated_onset_ratio": _float(rhythm.get("syncopated_onset_ratio")),
        "chord_tone_ratio": _float(pitch_roles.get("chord_tone_ratio")),
        "tension_ratio": _float(pitch_roles.get("tension_ratio")),
    }


def repair_sort_key(candidate: dict[str, Any]) -> tuple[float, int, int, float, float, float]:
    direction_penalty = 1 if _float(candidate.get("direction_change_ratio")) < 0.50 else 0
    tension_penalty = 1 if _float(candidate.get("tension_ratio")) < 0.20 else 0
    return (
        _float(candidate.get("dead_air_ratio")),
        direction_penalty,
        tension_penalty,
        -_float(candidate.get("direction_change_ratio")),
        -_float(candidate.get("tension_ratio")),
        -_float(candidate.get("syncopated_onset_ratio")),
    )


def select_repair_candidates(
    sweep_report: dict[str, Any],
    *,
    selected_per_case: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for case in _list(sweep_report.get("cases")):
        case_dict = _dict(case)
        probe_report_path = Path(str(case_dict.get("probe_report_path") or ""))
        probe_report = read_json(probe_report_path)
        strict_samples = [
            compact_probe_sample(case_dict, _dict(sample))
            for sample in _list(probe_report.get("samples"))
            if bool(_dict(sample).get("strict_valid", False))
        ]
        strict_samples = [
            sample for sample in strict_samples if Path(str(sample["source_midi_path"])).exists()
        ]
        if len(strict_samples) < int(selected_per_case):
            raise SoloYieldDeadAirDensityRepairPackageError(
                f"not enough strict samples for case {case_dict.get('label')}"
            )
        selected.extend(sorted(strict_samples, key=repair_sort_key)[: int(selected_per_case)])
    return selected


def source_candidate_metrics(listening_package: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "dead_air_ratio": _float(item.get("dead_air_ratio")),
            "direction_change_ratio": _float(item.get("direction_change_ratio")),
            "tension_ratio": _float(item.get("tension_ratio")),
            "note_count": _int(item.get("note_count")),
        }
        for item in _list(listening_package.get("candidates"))
    ]


def copy_selected_midis(selected: Sequence[dict[str, Any]], output_dir: Path) -> list[dict[str, Any]]:
    midi_dir = output_dir / "midi"
    midi_dir.mkdir(parents=True, exist_ok=True)
    copied: list[dict[str, Any]] = []
    for index, candidate in enumerate(selected, start=1):
        source = Path(str(candidate["source_midi_path"]))
        target = (
            midi_dir
            / f"candidate_{index:02d}_{candidate['case_label']}_sample_{int(candidate['sample_index']):02d}.mid"
        )
        shutil.copy2(source, target)
        copied.append(
            {
                **candidate,
                "repair_index": index,
                "repair_midi_path": str(target),
                "repair_midi_sha256": sha256_file(target),
            }
        )
    return copied


def render_selected(
    candidates: Sequence[dict[str, Any]],
    *,
    output_dir: Path,
    renderer: str,
    soundfont: str,
    sample_rate: int,
    render_audio: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    renderer_path = renderer or shutil.which("fluidsynth") or ""
    soundfont_path = resolve_soundfont(soundfont)
    setup = {
        "renderer_path": renderer_path,
        "soundfont_path": soundfont_path,
        "sample_rate": int(sample_rate),
        "render_attempted": bool(render_audio and renderer_path and soundfont_path),
    }
    if not setup["render_attempted"]:
        return [], setup

    renderer_file = Path(renderer_path)
    soundfont_file = Path(soundfont_path).expanduser()
    if not renderer_file.exists():
        raise SoloYieldDeadAirDensityRepairPackageError(f"renderer not found: {renderer_file}")
    if not soundfont_file.exists():
        raise SoloYieldDeadAirDensityRepairPackageError(f"soundfont not found: {soundfont_file}")
    setup["soundfont_sha256"] = sha256_file(soundfont_file)

    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    rendered: list[dict[str, Any]] = []
    for candidate in candidates:
        wav_path = (
            audio_dir
            / f"candidate_{int(candidate['repair_index']):02d}_{candidate['case_label']}_sample_{int(candidate['sample_index']):02d}.wav"
        )
        command = [
            str(renderer_file),
            "-ni",
            "-F",
            str(wav_path),
            "-r",
            str(sample_rate),
            str(soundfont_file),
            str(candidate["repair_midi_path"]),
        ]
        completed = subprocess.run(command, check=False, text=True, capture_output=True)
        if completed.returncode != 0:
            raise SoloYieldDeadAirDensityRepairPackageError(
                f"audio render failed for candidate {candidate['repair_index']}: "
                f"{completed.stderr or completed.stdout}"
            )
        rendered.append(
            {
                "repair_index": int(candidate["repair_index"]),
                "case_label": candidate["case_label"],
                "sample_index": int(candidate["sample_index"]),
                "source_midi_path": str(candidate["repair_midi_path"]),
                "wav_file": wav_meta(wav_path),
                "command": command,
                "stdout_tail": (completed.stdout or "")[-1000:],
                "stderr_tail": (completed.stderr or "")[-1000:],
            }
        )
    return rendered, setup


def metric_summary(candidates: Sequence[dict[str, Any]], *, major_dead_air_threshold: float) -> dict[str, Any]:
    dead_air_values = [_float(item.get("dead_air_ratio")) for item in candidates]
    direction_values = [_float(item.get("direction_change_ratio")) for item in candidates]
    tension_values = [_float(item.get("tension_ratio")) for item in candidates]
    note_counts = [_float(item.get("note_count")) for item in candidates]
    return {
        "candidate_count": len(candidates),
        "dead_air_avg": avg(dead_air_values),
        "dead_air_max": max(dead_air_values) if dead_air_values else 0.0,
        "dead_air_high_count": sum(1 for value in dead_air_values if value > major_dead_air_threshold),
        "direction_change_avg": avg(direction_values),
        "tension_avg": avg(tension_values),
        "note_count_avg": avg(note_counts),
    }


def build_repair_package(
    *,
    sweep_report: dict[str, Any],
    source_listening_package: dict[str, Any],
    output_dir: Path,
    selected_per_case: int,
    major_dead_air_threshold: float,
    renderer: str,
    soundfont: str,
    sample_rate: int,
    render_audio: bool = True,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    _reject_quality_claim(sweep_report, source_listening_package)
    selected = select_repair_candidates(sweep_report, selected_per_case=int(selected_per_case))
    copied = copy_selected_midis(selected, output_dir=output_dir)
    rendered, render_setup = render_selected(
        copied,
        output_dir=output_dir,
        renderer=renderer,
        soundfont=soundfont,
        sample_rate=int(sample_rate),
        render_audio=bool(render_audio),
    )
    source_metrics = source_candidate_metrics(source_listening_package)
    source_summary = metric_summary(
        source_metrics,
        major_dead_air_threshold=float(major_dead_air_threshold),
    )
    repair_summary = metric_summary(
        copied,
        major_dead_air_threshold=float(major_dead_air_threshold),
    )
    dead_air_avg_delta = source_summary["dead_air_avg"] - repair_summary["dead_air_avg"]
    dead_air_high_count_delta = (
        source_summary["dead_air_high_count"] - repair_summary["dead_air_high_count"]
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "source_reports": {
            "sweep_report": sweep_report.get("output_dir"),
            "source_listening_package": source_listening_package.get("output_dir"),
        },
        "request": {
            "selected_per_case": int(selected_per_case),
            "major_dead_air_threshold": float(major_dead_air_threshold),
        },
        "source_summary": source_summary,
        "repair_summary": {
            **repair_summary,
            "candidate_midi_files_copied": len(copied),
            "candidate_wav_files_rendered": len(rendered),
            "dead_air_avg_delta": dead_air_avg_delta,
            "dead_air_high_count_delta": dead_air_high_count_delta,
        },
        "selected_candidates": copied,
        "render_setup": render_setup,
        "rendered_audio_files": rendered,
        "decision": {
            "current_boundary": "music_transformer_solo_yield_dead_air_density_balance_repair",
            "next_boundary": "music_transformer_solo_yield_dead_air_density_balance_repair_rubric_review",
            "critical_user_input_required": False,
            "reason": "source strict-valid pool contains lower dead-air candidates; repackage before changing model or decoder",
        },
        "readiness": {
            "dead_air_density_repair_completed": True,
            "dead_air_avg_reduced": dead_air_avg_delta > 0,
            "dead_air_high_count_reduced": dead_air_high_count_delta > 0,
            "technical_wav_render_completed": len(rendered) == len(copied),
            "musical_quality_claimed": False,
            "artist_style_claimed": False,
            "production_ready_claimed": False,
        },
        "not_proven": [
            "human_audio_preference",
            "stable_jazz_solo_quality",
            "direction_tension_tradeoff_resolved",
            "repair_listening_preference",
        ],
    }


def validate_repair_package(
    report: dict[str, Any],
    *,
    min_candidate_count: int,
    require_dead_air_avg_reduced: bool,
    require_wav_rendered: bool,
    require_no_quality_claim: bool,
) -> dict[str, Any]:
    if str(report.get("schema_version")) != SCHEMA_VERSION:
        raise SoloYieldDeadAirDensityRepairPackageError("schema version mismatch")
    repair_summary = _dict(report.get("repair_summary"))
    readiness = _dict(report.get("readiness"))
    if _int(repair_summary.get("candidate_count")) < int(min_candidate_count):
        raise SoloYieldDeadAirDensityRepairPackageError("candidate count below requirement")
    if require_dead_air_avg_reduced and not bool(readiness.get("dead_air_avg_reduced", False)):
        raise SoloYieldDeadAirDensityRepairPackageError("dead-air average reduction required")
    if require_wav_rendered and not bool(readiness.get("technical_wav_render_completed", False)):
        raise SoloYieldDeadAirDensityRepairPackageError("WAV render completion required")
    if require_no_quality_claim:
        claimed = [
            key
            for key in ("musical_quality_claimed", "artist_style_claimed", "production_ready_claimed")
            if bool(readiness.get(key, True))
        ]
        if claimed:
            raise SoloYieldDeadAirDensityRepairPackageError(
                f"unexpected quality claim: {claimed}"
            )
    return {
        "schema_version": str(report.get("schema_version")),
        "candidate_count": _int(repair_summary.get("candidate_count")),
        "candidate_midi_files_copied": _int(repair_summary.get("candidate_midi_files_copied")),
        "candidate_wav_files_rendered": _int(repair_summary.get("candidate_wav_files_rendered")),
        "source_dead_air_avg": _float(_dict(report.get("source_summary")).get("dead_air_avg")),
        "repair_dead_air_avg": _float(repair_summary.get("dead_air_avg")),
        "dead_air_avg_delta": _float(repair_summary.get("dead_air_avg_delta")),
        "source_dead_air_high_count": _int(_dict(report.get("source_summary")).get("dead_air_high_count")),
        "repair_dead_air_high_count": _int(repair_summary.get("dead_air_high_count")),
        "dead_air_high_count_delta": _int(repair_summary.get("dead_air_high_count_delta")),
        "dead_air_avg_reduced": bool(readiness.get("dead_air_avg_reduced", False)),
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
        "# Music Transformer Solo Yield Dead-Air Density Repair Package",
        "",
        "## Summary",
        "",
        f"- candidate count: `{repair['candidate_count']}`",
        f"- MIDI/WAV: `{repair['candidate_midi_files_copied']}` / `{repair['candidate_wav_files_rendered']}`",
        f"- source dead-air avg/max: `{float(source['dead_air_avg']):.4f}` / `{float(source['dead_air_max']):.4f}`",
        f"- repair dead-air avg/max: `{float(repair['dead_air_avg']):.4f}` / `{float(repair['dead_air_max']):.4f}`",
        f"- dead-air avg delta: `{float(repair['dead_air_avg_delta']):.4f}`",
        f"- source/repair dead-air high count: `{source['dead_air_high_count']}` / `{repair['dead_air_high_count']}`",
        f"- dead-air high count delta: `{repair['dead_air_high_count_delta']}`",
        f"- source direction avg: `{float(source['direction_change_avg']):.4f}`",
        f"- repair direction avg: `{float(repair['direction_change_avg']):.4f}`",
        f"- source tension avg: `{float(source['tension_avg']):.4f}`",
        f"- repair tension avg: `{float(repair['tension_avg']):.4f}`",
        f"- technical WAV render completed: `{_bool_token(readiness['technical_wav_render_completed'])}`",
        f"- musical quality claimed: `{_bool_token(readiness['musical_quality_claimed'])}`",
        f"- next boundary: `{decision['next_boundary']}`",
        "",
        "## Selected Candidates",
        "",
        "| idx | case | sample | notes | dead air | direction | tension | MIDI | WAV |",
        "|---:|---|---:|---:|---:|---:|---:|---|---|",
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
    parser = argparse.ArgumentParser(description="Build dead-air/density repair package")
    parser.add_argument("--sweep_report", type=str, required=True)
    parser.add_argument("--source_listening_package", type=str, required=True)
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/music_transformer_finetune_mvp/solo_yield_dead_air_density_repair",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--selected_per_case", type=int, default=2)
    parser.add_argument("--major_dead_air_threshold", type=float, default=0.68)
    parser.add_argument("--renderer", type=str, default="")
    parser.add_argument("--soundfont", type=str, default="")
    parser.add_argument("--sample_rate", type=int, default=44100)
    parser.add_argument("--skip_render", action="store_true")
    parser.add_argument("--min_candidate_count", type=int, default=8)
    parser.add_argument("--require_dead_air_avg_reduced", action="store_true")
    parser.add_argument("--require_wav_rendered", action="store_true")
    parser.add_argument("--require_no_quality_claim", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    report = build_repair_package(
        sweep_report=read_json(Path(args.sweep_report)),
        source_listening_package=read_json(Path(args.source_listening_package)),
        output_dir=output_dir,
        selected_per_case=int(args.selected_per_case),
        major_dead_air_threshold=float(args.major_dead_air_threshold),
        renderer=str(args.renderer),
        soundfont=str(args.soundfont),
        sample_rate=int(args.sample_rate),
        render_audio=not bool(args.skip_render),
    )
    summary = validate_repair_package(
        report,
        min_candidate_count=int(args.min_candidate_count),
        require_dead_air_avg_reduced=bool(args.require_dead_air_avg_reduced),
        require_wav_rendered=bool(args.require_wav_rendered),
        require_no_quality_claim=bool(args.require_no_quality_claim),
    )
    write_json(output_dir / "dead_air_density_repair_package.json", report)
    write_json(output_dir / "dead_air_density_repair_package_summary.json", summary)
    markdown = markdown_report(report)
    write_text(output_dir / "dead_air_density_repair_package.md", markdown)
    if args.doc_path:
        write_text(Path(args.doc_path), markdown)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
